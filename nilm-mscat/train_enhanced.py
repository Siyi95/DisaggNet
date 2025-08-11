#!/usr/bin/env python3
"""
增强版训练脚本
支持自适应设备检测、TensorBoard可视化和可解释性分析
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.datamodule import AMPds2DataModule
from src.train_pretrain import MaskedReconstructionModel
from src.train_finetune import NILMModel

def detect_device():
    """
    自适应检测可用的计算设备
    优先级: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        print(f"🚀 检测到CUDA设备: {device_name} (共{device_count}个GPU)")
        return 'gpu', device_count
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 检测到MPS设备 (Apple Silicon)")
        return 'mps', 1
    else:
        print("💻 使用CPU设备")
        return 'cpu', 1

class EnhancedVisualizationCallback(pl.Callback):
    """
    增强的可视化回调函数
    提供训练过程中的可解释性分析和可视化
    """
    
    def __init__(self, log_dir: str, visualize_every_n_epochs: int = 10):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.visualize_every_n_epochs = visualize_every_n_epochs
        self.vis_dir = self.log_dir / 'visualizations'
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """验证轮次结束时的可视化"""
        if trainer.current_epoch % self.visualize_every_n_epochs == 0:
            self._visualize_training_progress(trainer, pl_module)
            self._visualize_attention_weights(trainer, pl_module)
            self._visualize_feature_importance(trainer, pl_module)
    
    def _visualize_training_progress(self, trainer, pl_module):
        """可视化训练进度"""
        try:
            # 获取训练历史
            metrics = trainer.logged_metrics
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress - Epoch {trainer.current_epoch}', fontsize=16)
            
            # 损失曲线
            if 'train_loss' in metrics and 'val_loss' in metrics:
                axes[0, 0].plot(range(trainer.current_epoch + 1), 
                               [metrics.get('train_loss', 0)] * (trainer.current_epoch + 1), 
                               label='Train Loss', alpha=0.7)
                axes[0, 0].plot(range(trainer.current_epoch + 1), 
                               [metrics.get('val_loss', 0)] * (trainer.current_epoch + 1), 
                               label='Val Loss', alpha=0.7)
                axes[0, 0].set_title('Loss Curves')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # 学习率曲线
            if 'lr-AdamW' in metrics:
                axes[0, 1].plot(range(trainer.current_epoch + 1), 
                               [metrics.get('lr-AdamW', 0)] * (trainer.current_epoch + 1))
                axes[0, 1].set_title('Learning Rate')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('LR')
                axes[0, 1].grid(True, alpha=0.3)
            
            # MAE指标
            if 'val_MAE_total' in metrics:
                axes[1, 0].bar(['Total MAE'], [metrics.get('val_MAE_total', 0)])
                axes[1, 0].set_title('Validation MAE')
                axes[1, 0].set_ylabel('MAE')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 设备级别MAE
            device_maes = {k: v for k, v in metrics.items() if 'MAE_' in k and k != 'val_MAE_total'}
            if device_maes:
                device_names = [k.replace('val_MAE_', '') for k in device_maes.keys()]
                mae_values = list(device_maes.values())
                axes[1, 1].bar(device_names, mae_values)
                axes[1, 1].set_title('Device-level MAE')
                axes[1, 1].set_ylabel('MAE')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            save_path = self.vis_dir / f'training_progress_epoch_{trainer.current_epoch}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 记录到TensorBoard
            if hasattr(trainer.logger, 'experiment'):
                trainer.logger.experiment.add_figure(
                    'Training/Progress', fig, trainer.current_epoch
                )
                
        except Exception as e:
            print(f"⚠️ 训练进度可视化失败: {e}")
    
    def _visualize_attention_weights(self, trainer, pl_module):
        """可视化注意力权重"""
        try:
            if hasattr(pl_module, 'encoder') and hasattr(pl_module.encoder, 'local_branch'):
                # 获取一个验证批次
                val_dataloader = trainer.val_dataloaders[0] if trainer.val_dataloaders else None
                if val_dataloader is None:
                    return
                
                batch = next(iter(val_dataloader))
                x = batch['features'][:1]  # 取第一个样本
                timestamps = batch.get('timestamps', None)
                if timestamps is not None:
                    timestamps = timestamps[:1]
                
                # 前向传播获取注意力权重
                pl_module.eval()
                with torch.no_grad():
                    if timestamps is not None:
                        _ = pl_module.encoder(x, timestamps)
                    else:
                        _ = pl_module.encoder(x)
                
                # 这里可以添加具体的注意力权重可视化代码
                # 需要根据模型结构调整
                
        except Exception as e:
            print(f"⚠️ 注意力权重可视化失败: {e}")
    
    def _visualize_feature_importance(self, trainer, pl_module):
        """可视化特征重要性"""
        try:
            # 这里可以添加特征重要性分析
            # 例如梯度分析、SHAP值等
            pass
        except Exception as e:
            print(f"⚠️ 特征重要性可视化失败: {e}")

def create_enhanced_trainer(config: Dict[str, Any], 
                          output_dir: str,
                          accelerator: str,
                          devices: int) -> pl.Trainer:
    """
    创建增强的训练器
    """
    # TensorBoard日志记录器
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name='enhanced_logs',
        version=None,
        log_graph=True  # 记录计算图
    )
    
    # 回调函数
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            filename='best_model_{epoch:02d}_{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
            save_weights_only=False
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config.get('early_stopping_patience', 15),
            mode='min',
            verbose=True
        ),
        LearningRateMonitor(
            logging_interval='step',
            log_momentum=True
        ),
        EnhancedVisualizationCallback(
            log_dir=output_dir,
            visualize_every_n_epochs=config.get('visualize_every_n_epochs', 5)
        )
    ]
    
    # 训练器配置
    trainer_config = {
        'max_epochs': config.get('max_epochs', 100),
        'accelerator': accelerator,
        'devices': devices if accelerator != 'cpu' else 'auto',
        'precision': config.get('precision', '16-mixed'),
        'logger': logger,
        'callbacks': callbacks,
        'gradient_clip_val': config.get('gradient_clip_val', 1.0),
        'accumulate_grad_batches': config.get('accumulate_grad_batches', 1),
        'val_check_interval': config.get('val_check_interval', 1.0),
        'log_every_n_steps': config.get('log_every_n_steps', 50),
        'enable_progress_bar': True,
        'enable_model_summary': True
    }
    
    # 如果使用CPU，移除precision设置
    if accelerator == 'cpu':
        trainer_config.pop('precision', None)
    
    return pl.Trainer(**trainer_config)

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='增强版MS-CAT训练脚本')
    parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune'], 
                       required=True, help='训练模式')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data_path', type=str, help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--pretrain_ckpt', type=str, help='预训练模型检查点路径')
    parser.add_argument('--resume_from', type=str, help='从检查点恢复训练')
    parser.add_argument('--force_cpu', action='store_true', help='强制使用CPU')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    
    args = parser.parse_args()
    
    # 检测设备
    if args.force_cpu:
        accelerator, devices = 'cpu', 1
        print("💻 强制使用CPU设备")
    else:
        accelerator, devices = detect_device()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置中的参数
    if args.data_path:
        config['data']['data_path'] = args.data_path
    if args.epochs:
        config['max_epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['optimizer']['lr'] = args.learning_rate
    
    # 创建输出目录
    output_dir = os.path.join(args.output_dir, args.mode)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置随机种子
    pl.seed_everything(config.get('seed', 42))
    
    print(f"🎯 开始{args.mode}训练...")
    print(f"📁 输出目录: {output_dir}")
    print(f"⚙️ 设备配置: {accelerator} (设备数: {devices})")
    
    # 数据模块
    data_module = AMPds2DataModule(**config['data'])
    data_module.setup('fit')
    
    # 获取特征维度
    input_dim = data_module.get_feature_dim()
    print(f"📊 输入特征维度: {input_dim}")
    
    # 创建模型
    model_config = config['model']
    model_config['input_dim'] = input_dim
    
    if args.mode == 'pretrain':
        model = MaskedReconstructionModel(**model_config)
        print("🏗️ 创建预训练模型")
    else:  # finetune
        # 获取设备信息
        device_names = data_module.get_device_names()
        num_devices = len(device_names)
        model_config['num_devices'] = num_devices
        model_config['device_names'] = device_names
        
        model = NILMModel(**model_config)
        
        # 加载预训练权重
        if args.pretrain_ckpt:
            print(f"📥 加载预训练模型: {args.pretrain_ckpt}")
            pretrain_model = MaskedReconstructionModel.load_from_checkpoint(args.pretrain_ckpt)
            model.load_pretrained_encoder(pretrain_model.encoder)
        
        print(f"🏗️ 创建微调模型 (设备数: {num_devices})")
    
    # 创建训练器
    trainer = create_enhanced_trainer(config, output_dir, accelerator, devices)
    
    # 开始训练
    if args.resume_from:
        print(f"🔄 从检查点恢复训练: {args.resume_from}")
        trainer.fit(model, data_module, ckpt_path=args.resume_from)
    else:
        trainer.fit(model, data_module)
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, f'final_model_{args.mode}.ckpt')
    trainer.save_checkpoint(final_model_path)
    print(f"✅ 训练完成，模型保存至: {final_model_path}")
    
    # 打印TensorBoard启动命令
    print(f"\n📈 查看训练日志:")
    print(f"tensorboard --logdir {output_dir}")
    print(f"然后在浏览器中打开: http://localhost:6006")

if __name__ == '__main__':
    main()