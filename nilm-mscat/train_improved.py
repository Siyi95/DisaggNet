#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的NILM训练脚本
解决功率预测不稳定、实现预训练-微调流程、增加数据增强
"""

import os
import sys
import yaml
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
from typing import Dict, Any, Optional

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from enhanced_datamodule import EnhancedAMPds2DataModule
from train_pretrain import MaskedReconstructionModel
from train_finetune import NILMModel

class ImprovedNILMModel(NILMModel):
    """改进的NILM模型，解决功率预测不稳定问题"""
    
    def __init__(self, *args, **kwargs):
        # 添加功率损失稳定化参数
        self.power_loss_smoothing = kwargs.pop('power_loss_smoothing', 0.1)
        self.gradient_clip_val = kwargs.pop('gradient_clip_val', 1.0)
        self.power_scale_factor = kwargs.pop('power_scale_factor', 0.01)  # 功率缩放因子
        
        super().__init__(*args, **kwargs)
        
        # 添加功率损失的指数移动平均
        self.register_buffer('power_loss_ema', torch.tensor(0.0))
        self.ema_initialized = False
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """改进的训练步骤，增加功率损失稳定化"""
        x = batch['x'].float()
        y_power = batch['y_power'].float()
        y_state = batch['y_state'].float()
        
        # 功率标准化（减少数值不稳定）
        y_power_scaled = y_power * self.power_scale_factor
        
        # 前向传播
        predictions = self.forward(x)
        
        # 缩放预测结果
        predictions_scaled = predictions.copy()
        predictions_scaled['power_pred'] = predictions['power_pred'] * self.power_scale_factor
        
        # 计算损失
        targets = {'power': y_power_scaled, 'state': y_state}
        losses = self.head.compute_loss(predictions_scaled, targets)
        
        # 功率损失平滑
        current_power_loss = losses['power_loss']
        if not self.ema_initialized:
            self.power_loss_ema = current_power_loss.detach()
            self.ema_initialized = True
        else:
            self.power_loss_ema = (1 - self.power_loss_smoothing) * self.power_loss_ema + \
                                 self.power_loss_smoothing * current_power_loss.detach()
        
        # 使用平滑后的损失进行梯度裁剪判断
        if current_power_loss > 2 * self.power_loss_ema:
            # 如果当前损失过大，进行额外的梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val * 0.5)
        
        # 记录损失（恢复原始尺度）
        self.log('train/total_loss', losses['total_loss'], prog_bar=True)
        self.log('train/power_loss', losses['power_loss'] / (self.power_scale_factor ** 2))
        self.log('train/event_loss', losses['event_loss'])
        self.log('train/power_loss_ema', self.power_loss_ema / (self.power_scale_factor ** 2))
        
        # 计算指标（使用原始尺度）
        predictions_original = predictions.copy()
        targets_original = {'power': y_power, 'state': y_state}
        metrics = self.compute_metrics(predictions_original, targets_original, 'train/')
        for name, value in metrics.items():
            self.log(name, value)
        
        return losses['total_loss']
    
    def configure_optimizers(self):
        """配置优化器，使用不同的学习率"""
        # 编码器使用较小的学习率
        encoder_params = list(self.encoder.parameters())
        head_params = list(self.head.parameters())
        
        optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': self.hparams.learning_rate * 0.1},  # 编码器学习率更小
            {'params': head_params, 'lr': self.hparams.learning_rate}
        ], weight_decay=self.hparams.weight_decay)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/total_loss',
                'frequency': 1
            }
        }

def create_pretrain_config():
    """创建预训练配置"""
    return {
        'data': {
            'data_path': 'data/AMPds2.h5',
            'window_length': 240,  # 更长的窗口用于预训练
            'step_size': 60,       # 更大的步长
            'batch_size': 16,
            'num_workers': 4,
            'augment': True,
            'min_samples': 50
        },
        'model': {
            'd_model': 192,
            'num_heads': 6,
            'local_layers': 4,
            'global_layers': 3,
            'window_size': 32,
            'dropout': 0.1,
            'mask_ratio': 0.25,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'warmup_steps': 1000,
            'max_steps': 20000
        },
        'training': {
            'max_epochs': 30,
            'patience': 8
        }
    }

def create_finetune_config():
    """创建微调配置"""
    return {
        'data': {
            'data_path': 'data/AMPds2.h5',
            'window_length': 128,
            'step_size': 16,
            'batch_size': 8,
            'num_workers': 2,
            'power_threshold': 10.0,
            'augment': True,
            'min_samples': 100,
            # 数据增强参数
            'noise_std': 0.02,
            'amplitude_range': [0.95, 1.05],
            'time_jitter': 2
        },
        'model': {
            'd_model': 192,
            'num_heads': 6,
            'local_layers': 4,
            'global_layers': 3,
            'window_size': 32,
            'dropout': 0.1,
            'regression_hidden_dim': 96,
            'event_hidden_dim': 48,
            'learning_rate': 5e-4,  # 微调时使用更小的学习率
            'weight_decay': 1e-5,
            'power_loss_weight': 2.0,  # 增加功率损失权重
            'event_loss_weight': 0.5,
            'use_crf': False,
            # 稳定化参数
            'power_loss_smoothing': 0.1,
            'gradient_clip_val': 1.0,
            'power_scale_factor': 0.01
        },
        'training': {
            'max_epochs': 50,
            'patience': 10,
            'val_check_interval': 1.0
        }
    }

def run_pretraining(output_dir: str, config: Dict[str, Any]):
    """运行预训练"""
    print("\n=== 开始预训练 ===")
    
    pretrain_dir = os.path.join(output_dir, 'pretrain')
    os.makedirs(pretrain_dir, exist_ok=True)
    
    # 创建数据模块
    datamodule = EnhancedAMPds2DataModule(**config['data'])
    datamodule.setup('fit')
    
    input_dim = datamodule.feature_dim
    print(f"预训练输入特征维度: {input_dim}")
    
    # 创建预训练模型
    model_config = config['model'].copy()
    model_config['input_dim'] = input_dim
    
    model = MaskedReconstructionModel(**model_config)
    
    # 回调
    callbacks = [
        ModelCheckpoint(
            dirpath=pretrain_dir,
            filename='pretrain-{epoch:02d}-{val/recon_loss:.3f}',
            monitor='val/recon_loss',
            mode='min',
            save_top_k=1,
            save_last=True
        ),
        EarlyStopping(
            monitor='val/recon_loss',
            patience=config['training']['patience'],
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # 日志记录器
    logger = TensorBoardLogger(
        save_dir=pretrain_dir,
        name='pretrain_logs'
    )
    
    # 训练器
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='cpu',
        devices=1,
        precision='32',
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=10
    )
    
    # 开始预训练
    trainer.fit(model, datamodule)
    
    # 返回最佳检查点路径
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"预训练完成，最佳模型: {best_ckpt}")
    
    return best_ckpt

def run_finetuning(output_dir: str, config: Dict[str, Any], pretrain_ckpt: Optional[str] = None):
    """运行微调"""
    print("\n=== 开始微调 ===")
    
    finetune_dir = os.path.join(output_dir, 'finetune')
    os.makedirs(finetune_dir, exist_ok=True)
    
    # 创建数据模块
    datamodule = EnhancedAMPds2DataModule(**config['data'])
    datamodule.setup('fit')
    
    input_dim = datamodule.feature_dim
    num_devices = datamodule.num_devices
    device_names = datamodule.device_columns
    
    print(f"微调输入特征维度: {input_dim}")
    print(f"设备数量: {num_devices}")
    print(f"设备名称: {device_names}")
    
    # 创建微调模型
    model_config = config['model'].copy()
    model_config.update({
        'input_dim': input_dim,
        'num_devices': num_devices,
        'device_names': device_names
    })
    
    model = ImprovedNILMModel(**model_config)
    
    # 加载预训练权重
    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        print(f"加载预训练权重: {pretrain_ckpt}")
        model.load_pretrained_weights(pretrain_ckpt, strict=False)
        
        # 冻结编码器的部分参数
        model.freeze_encoder(freeze_ratio=0.5)
    
    # 回调
    callbacks = [
        ModelCheckpoint(
            dirpath=finetune_dir,
            filename='finetune-{epoch:02d}-{val/total_loss:.3f}',
            monitor='val/total_loss',
            mode='min',
            save_top_k=1,
            save_last=True
        ),
        EarlyStopping(
            monitor='val/total_loss',
            patience=config['training']['patience'],
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # 日志记录器
    logger = TensorBoardLogger(
        save_dir=finetune_dir,
        name='finetune_logs'
    )
    
    # 训练器
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='cpu',
        devices=1,
        precision='32',
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=5,
        val_check_interval=config['training']['val_check_interval']
    )
    
    # 开始微调
    trainer.fit(model, datamodule)
    
    # 在第20个epoch后解冻编码器
    if trainer.current_epoch >= 20:
        print("解冻编码器参数...")
        model.unfreeze_encoder()
        
        # 继续训练几个epoch
        trainer.fit(model, datamodule, ckpt_path=trainer.checkpoint_callback.last_model_path)
    
    print(f"微调完成，最佳模型: {trainer.checkpoint_callback.best_model_path}")
    
    return trainer.checkpoint_callback.best_model_path

def main():
    parser = argparse.ArgumentParser(description='改进的NILM训练')
    parser.add_argument('--output_dir', type=str, default='./outputs/improved_training', help='输出目录')
    parser.add_argument('--skip_pretrain', action='store_true', help='跳过预训练')
    parser.add_argument('--pretrain_ckpt', type=str, help='预训练检查点路径')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    pl.seed_everything(42)
    
    print("=== 改进的NILM训练流程 ===")
    print(f"输出目录: {args.output_dir}")
    
    pretrain_ckpt = args.pretrain_ckpt
    
    # 预训练阶段
    if not args.skip_pretrain:
        pretrain_config = create_pretrain_config()
        pretrain_ckpt = run_pretraining(args.output_dir, pretrain_config)
        
        # 保存预训练配置
        with open(os.path.join(args.output_dir, 'pretrain_config.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(pretrain_config, f, default_flow_style=False, allow_unicode=True)
    
    # 微调阶段
    finetune_config = create_finetune_config()
    best_model = run_finetuning(args.output_dir, finetune_config, pretrain_ckpt)
    
    # 保存微调配置
    with open(os.path.join(args.output_dir, 'finetune_config.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(finetune_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n=== 训练完成 ===")
    print(f"最佳模型: {best_model}")
    print(f"配置文件保存在: {args.output_dir}")

if __name__ == '__main__':
    main()