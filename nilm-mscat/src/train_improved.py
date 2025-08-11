#!/usr/bin/env python3
"""
改进的NILM训练模块
解决功率损失不稳定问题，实现预训练+微调流程
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 导入现有模块
try:
    from train_pretrain import MaskedReconstructionModel
    from train_finetune import NILMModel
except ImportError:
    # 如果直接导入失败，尝试相对导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from train_pretrain import MaskedReconstructionModel
    from train_finetune import NILMModel

class ImprovedNILMModel(NILMModel):
    """改进的NILM模型，解决功率损失不稳定问题"""
    
    def __init__(self,
                 input_dim: int,
                 num_devices: int,
                 device_names: List[str],
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 power_loss_weight: float = 1.0,
                 event_loss_weight: float = 0.5,
                 use_crf: bool = False,
                 freeze_encoder: bool = True,
                 encoder_lr: float = 1e-4,
                 head_lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 # 新增稳定化参数
                 power_loss_smoothing: float = 0.1,
                 gradient_clip_val: float = 1.0,
                 power_scale_factor: float = 1000.0,
                 **kwargs):
        
        super().__init__(
            input_dim=input_dim,
            num_devices=num_devices,
            device_names=device_names,
            d_model=d_model,
            num_heads=nhead,
            num_layers=num_layers,
            dropout=dropout,
            regression_weight=power_loss_weight,
            classification_weight=event_loss_weight,
            use_crf=use_crf,
            **kwargs
        )
        
        # 稳定化参数
        self.power_loss_smoothing = power_loss_smoothing
        self.gradient_clip_val = gradient_clip_val
        self.power_scale_factor = power_scale_factor
        self.encoder_lr = encoder_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay
        
        # 损失平滑
        self.register_buffer('smoothed_power_loss', torch.tensor(0.0))
        self.register_buffer('loss_count', torch.tensor(0))
        
    def configure_optimizers(self):
        """配置优化器，使用分离的学习率"""
        # 分离编码器和头部参数
        encoder_params = list(self.encoder.parameters())
        head_params = list(self.head.parameters())
        
        # 创建参数组
        param_groups = [
            {'params': encoder_params, 'lr': self.encoder_lr, 'weight_decay': self.weight_decay},
            {'params': head_params, 'lr': self.head_lr, 'weight_decay': self.weight_decay}
        ]
        
        optimizer = torch.optim.AdamW(param_groups)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/total_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def training_step(self, batch, batch_idx):
        """训练步骤，包含损失稳定化"""
        x = batch['x']
        y_power = batch['y_power'] * self.power_scale_factor  # 功率缩放
        y_state = batch['y_state']
        
        # 前向传播
        predictions = self(x)
        power_pred = predictions['power_pred'] * self.power_scale_factor
        event_logits = predictions['event_logits']
        
        # 计算损失
        power_loss = F.l1_loss(power_pred, y_power)
        event_loss = F.binary_cross_entropy_with_logits(event_logits, y_state)
        
        # 损失平滑
        if self.loss_count == 0:
            self.smoothed_power_loss = power_loss.detach()
        else:
            self.smoothed_power_loss = (
                self.power_loss_smoothing * power_loss.detach() + 
                (1 - self.power_loss_smoothing) * self.smoothed_power_loss
            )
        self.loss_count += 1
        
        # 总损失
        total_loss = (
            self.hparams.regression_weight * self.smoothed_power_loss + 
            self.hparams.classification_weight * event_loss
        )
        
        # 记录指标
        self.log('train_power_loss', power_loss, prog_bar=True)
        self.log('train_event_loss', event_loss, prog_bar=True)
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('smoothed_power_loss', self.smoothed_power_loss, prog_bar=True)
        
        return total_loss
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        """配置梯度裁剪"""
        if self.gradient_clip_val > 0:
            self.clip_gradients(
                optimizer, 
                gradient_clip_val=self.gradient_clip_val, 
                gradient_clip_algorithm="norm"
            )

def create_pretrain_config(config):
    """创建预训练配置"""
    pretrain_config = config['pretrain'].copy()
    pretrain_config.update({
        'data_config': config['data'],
        'training_config': config['training'],
        'output_config': config['output']
    })
    return pretrain_config

def create_finetune_config(config):
    """创建微调配置"""
    finetune_config = config['finetune'].copy()
    finetune_config.update({
        'data_config': config['data'],
        'training_config': config['training'],
        'output_config': config['output']
    })
    return finetune_config

def run_pretraining(config, datamodule):
    """运行预训练"""
    print("开始预训练...")
    
    # 设置数据
    datamodule.setup('fit')
    
    # 创建模型
    model = MaskedReconstructionModel(
        input_dim=config['input_dim'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        mask_ratio=config['mask_ratio'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 设置回调和日志
    save_dir = Path(config['output_config']['save_dir']) / 'pretrain'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / 'checkpoints',
        filename='pretrain-{epoch:02d}-{val/total_loss:.3f}',
        monitor='val/total_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/total_loss',
        patience=config['patience'],
        mode='min'
    )
    
    logger = TensorBoardLogger(
        save_dir=config['output_config']['save_dir'],
        name=config['output_config']['experiment_name'],
        version='pretrain'
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator=config['training_config']['accelerator'],
        devices=config['training_config']['devices'],
        precision=config['training_config']['precision'],
        accumulate_grad_batches=config['training_config']['accumulate_grad_batches'],
        log_every_n_steps=config['training_config']['log_every_n_steps'],
        val_check_interval=config['training_config']['val_check_interval']
    )
    
    # 训练
    try:
        trainer.fit(model, datamodule)
        best_model_path = checkpoint_callback.best_model_path
        print(f"预训练完成，最佳模型: {best_model_path}")
        return best_model_path
    except Exception as e:
        print(f"预训练失败: {e}")
        return None

def run_finetuning(config, datamodule, pretrained_model_path=None):
    """运行微调"""
    print("开始微调...")
    
    # 设置数据
    datamodule.setup('fit')
    
    # 创建模型
    model = ImprovedNILMModel(
        input_dim=config['input_dim'],
        num_devices=config['num_devices'],
        device_names=config['device_names'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        power_loss_weight=config['power_loss_weight'],
        event_loss_weight=config['event_loss_weight'],
        use_crf=config['use_crf'],
        freeze_encoder=config['freeze_encoder'],
        encoder_lr=config['encoder_lr'],
        head_lr=config['head_lr'],
        weight_decay=config['weight_decay'],
        power_loss_smoothing=config['power_loss_smoothing'],
        gradient_clip_val=config['gradient_clip_val'],
        power_scale_factor=config['power_scale_factor']
    )
    
    # 加载预训练权重
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"加载预训练模型: {pretrained_model_path}")
        model.load_pretrained_weights(pretrained_model_path)
    
    # 设置回调和日志
    save_dir = Path(config['output_config']['save_dir']) / 'finetune'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / 'checkpoints',
        filename='finetune-{epoch:02d}-{val/total_loss:.3f}',
        monitor='val/total_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/total_loss',
        patience=config['patience'],
        mode='min'
    )
    
    logger = TensorBoardLogger(
        save_dir=config['output_config']['save_dir'],
        name=config['output_config']['experiment_name'],
        version='finetune'
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=config['max_epochs'],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        accelerator=config['training_config']['accelerator'],
        devices=config['training_config']['devices'],
        precision=config['training_config']['precision'],
        accumulate_grad_batches=config['training_config']['accumulate_grad_batches'],
        log_every_n_steps=config['training_config']['log_every_n_steps'],
        val_check_interval=config['training_config']['val_check_interval'],
        gradient_clip_val=config['gradient_clip_val']
    )
    
    # 训练
    try:
        trainer.fit(model, datamodule)
        best_model_path = checkpoint_callback.best_model_path
        print(f"微调完成，最佳模型: {best_model_path}")
        return best_model_path
    except Exception as e:
        print(f"微调失败: {e}")
        return None