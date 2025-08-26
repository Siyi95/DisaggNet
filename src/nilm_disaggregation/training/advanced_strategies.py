#!/usr/bin/env python3
"""
高级训练策略
包含学习率调度、早停、模型检查点、数据增强等技术
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, 
    CosineAnnealingWarmRestarts
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateMonitor,
    GradientAccumulationScheduler, StochasticWeightAveraging
)
import numpy as np
import random
import math
from typing import Dict, Any, Optional

class LightweightDataAugmentation(nn.Module):
    """轻量级数据增强模块
    
    基于最终方案，实现物理合理的轻量级增强
    """
    
    def __init__(self, 
                 jitter_std: float = 0.005,
                 scale_range: tuple = (0.98, 1.02),
                 time_mask_ratio: float = 0.05,
                 time_mask_max_length: int = 32):
        super().__init__()
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.time_mask_ratio = time_mask_ratio
        self.time_mask_max_length = time_mask_max_length
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """应用轻量级数据增强
        
        Args:
            x: [B, S, 1] 输入序列
            
        Returns:
            [B, S, 1] 增强后的序列
        """
        if not self.training:
            return x
            
        batch_size, seq_len, features = x.shape
        
        # 1. 轻微抖动（模拟传感器噪声）
        if self.jitter_std > 0:
            jitter = torch.randn_like(x) * self.jitter_std
            x = x + jitter
        
        # 2. 轻微缩放（模拟校准差异）
        if self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0:
            scale = torch.empty(batch_size, 1, 1, device=x.device).uniform_(
                self.scale_range[0], self.scale_range[1]
            )
            x = x * scale
        
        # 3. 随机时间遮挡（模拟短暂断电）
        if self.time_mask_ratio > 0 and torch.rand(1).item() < self.time_mask_ratio:
            for i in range(batch_size):
                mask_length = torch.randint(1, min(self.time_mask_max_length, seq_len // 10) + 1, (1,)).item()
                mask_start = torch.randint(0, seq_len - mask_length + 1, (1,)).item()
                
                # 使用前一个值填充（更符合物理直觉）
                if mask_start > 0:
                    x[i, mask_start:mask_start + mask_length] = x[i, mask_start - 1:mask_start]
                else:
                    x[i, mask_start:mask_start + mask_length] = 0
        
        return x

class WarmupScheduler:
    """学习率预热调度器"""
    
    def __init__(self, optimizer, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

class GradientClipping:
    """梯度裁剪"""
    
    def __init__(self, max_norm=1.0, norm_type=2):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, model):
        return torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                            self.max_norm, 
                                            self.norm_type)

class AdvancedNILMLightningModule(pl.LightningModule):
    """高级NILM Lightning模块，基于最终方案重构"""
    
    def __init__(self, model, loss_function, 
                 learning_rate: float = 1e-3,
                 weight_decay: float = 0.05,
                 warmup_ratio: float = 0.05,
                 max_epochs: int = 100,
                 gradient_clip_val: float = 1.0,
                 use_mixed_precision: bool = True,
                 data_augmentation: Optional[Any] = None,
                 **kwargs):
        super().__init__()
        
        self.model = model
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.use_mixed_precision = use_mixed_precision
        self.data_augmentation = data_augmentation
        
        # 保存超参数
        self.save_hyperparameters(ignore=['model', 'loss_function', 'data_augmentation'])
        
        # 验证指标存储
        self.validation_step_outputs = []
        self.training_step_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, power_targets, state_targets = batch
        
        # 数据增强（轻量级）
        if self.data_augmentation is not None and self.training:
            x = self.data_augmentation(x)
        
        # 前向传播
        predictions = self(x)
        
        # 组合目标
        y = {'power': power_targets, 'state': state_targets}
        
        # 计算损失
        loss_dict = self.loss_function(predictions, y)
        
        # 记录训练指标
        self.log('train_loss', loss_dict['total_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_power_loss', loss_dict['power_loss'], on_step=False, on_epoch=True)
        self.log('train_state_loss', loss_dict['state_loss'], on_step=False, on_epoch=True)
        self.log('train_corr_loss', loss_dict['corr_loss'], on_step=False, on_epoch=True)
        
        # 记录不确定性权重
        if 'uncertainties' in loss_dict:
            for unc_name, unc_value in loss_dict['uncertainties'].items():
                self.log(f'train_{unc_name}', unc_value, on_step=False, on_epoch=True)
        
        self.training_step_outputs.append({
            'loss': loss_dict['total_loss'],
            'power_loss': loss_dict['power_loss'],
            'state_loss': loss_dict['state_loss'],
            'corr_loss': loss_dict['corr_loss']
        })
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        x, power_targets, state_targets = batch
        
        # 前向传播（无数据增强）
        predictions = self(x)
        
        # 组合目标
        y = {'power': power_targets, 'state': state_targets}
        
        # 计算损失
        loss_dict = self.loss_function(predictions, y)
        
        # 记录损失
        self.log('val_loss', loss_dict['total_loss'], 
                on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_power_loss', loss_dict['power_loss'], on_epoch=True)
        self.log('val_state_loss', loss_dict['state_loss'], on_epoch=True)
        self.log('val_corr_loss', loss_dict['corr_loss'], on_epoch=True)
        
        # 计算额外指标
        metrics = self._compute_metrics(predictions, y)
        for key, value in metrics.items():
            self.log(f'val_{key}', value, on_epoch=True)
        
        self.validation_step_outputs.append(loss_dict['total_loss'])
        
        return loss_dict['total_loss']
    
    def test_step(self, batch, batch_idx):
        x, power_targets, state_targets = batch
        
        # 前向传播
        predictions = self(x)
        
        # 组合目标
        y = {'power': power_targets, 'state': state_targets}
        
        # 计算损失
        loss_dict = self.loss_function(predictions, y)
        
        # 记录损失
        self.log('test_loss', loss_dict['total_loss'])
        self.log('test_power_loss', loss_dict['power_loss'])
        self.log('test_state_loss', loss_dict['state_loss'])
        self.log('test_corr_loss', loss_dict['corr_loss'])
        
        # 计算额外指标
        metrics = self._compute_metrics(predictions, y)
        for key, value in metrics.items():
            self.log(f'test_{key}', value)
        
        return loss_dict['total_loss']
    
    def _compute_metrics(self, predictions, targets):
        """计算评估指标"""
        pred_power = predictions['power']
        target_power = targets['power']
        
        # MAE
        mae = torch.mean(torch.abs(pred_power - target_power))
        
        # RMSE
        rmse = torch.sqrt(torch.mean((pred_power - target_power) ** 2))
        
        # MAPE (避免除零)
        mape = torch.mean(torch.abs((pred_power - target_power) / (target_power + 1e-8))) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    
    def on_train_epoch_end(self):
        # 清空输出列表
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        # 清空输出列表
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器（基于最终方案）"""
        
        # AdamW优化器（推荐配置）
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine调度器 + Warmup
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Warmup阶段：线性增长
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine退火阶段
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def on_before_optimizer_step(self, optimizer):
        # 梯度裁剪
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_val)

def create_advanced_callbacks(monitor='val_loss', patience=15, 
                            save_top_k=3, mode='min'):
    """创建高级回调函数"""
    callbacks = []
    
    # 早停
    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # 模型检查点
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        save_top_k=save_top_k,
        mode=mode,
        save_last=True,
        filename='{epoch}-{val_loss:.4f}'
    )
    callbacks.append(checkpoint)
    
    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # 梯度累积调度器 - 已移除，避免与trainer的accumulate_grad_batches冲突
    # gradient_accumulation = GradientAccumulationScheduler(
    #     scheduling={0: 1, 10: 2, 20: 4}
    # )
    # callbacks.append(gradient_accumulation)
    
    # 随机权重平均
    swa = StochasticWeightAveraging(swa_lrs=1e-2)
    callbacks.append(swa)
    
    return callbacks

def create_advanced_trainer(max_epochs=50, gpus=None, precision=32,
                          accumulate_grad_batches=1, 
                          gradient_clip_val=1.0,
                          callbacks=None, **kwargs):
    """创建高级训练器"""
    if callbacks is None:
        callbacks = create_advanced_callbacks()
    
    trainer_args = {
        'max_epochs': max_epochs,
        'callbacks': callbacks,
        'accumulate_grad_batches': accumulate_grad_batches,
        'gradient_clip_val': gradient_clip_val,
        'precision': precision,
        'log_every_n_steps': 50,
        'check_val_every_n_epoch': 1,
        'enable_progress_bar': True,
        'enable_model_summary': True
    }
    
    # 添加GPU支持
    if gpus is not None:
        trainer_args['devices'] = gpus
        trainer_args['accelerator'] = 'gpu'
    
    # 添加其他参数
    trainer_args.update(kwargs)
    
    return pl.Trainer(**trainer_args)

if __name__ == '__main__':
    # 测试数据增强
    print("测试数据增强...")
    aug = LightweightDataAugmentation()
    test_data = torch.randn(4, 128, 1)
    augmented_data = aug(test_data)
    print(f"原始数据形状: {test_data.shape}")
    print(f"增强数据形状: {augmented_data.shape}")
    
    # 测试回调函数
    print("\n创建高级回调函数...")
    callbacks = create_advanced_callbacks()
    print(f"创建了 {len(callbacks)} 个回调函数")
    
    print("\n高级训练策略模块创建完成!")