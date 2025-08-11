#!/usr/bin/env python3
"""
掩蔽预训练脚本
实现MS-CAT的自监督预训练，通过掩蔽重构任务提升表示能力
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from typing import Dict, Any, Optional, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datamodule import AMPds2DataModule
from src.models.mscat import MSCAT
from src.models.heads import ReconstructionHead

class MaskedReconstructionModel(pl.LightningModule):
    """掩蔽重构预训练模型"""
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 192,
                 num_heads: int = 6,
                 local_layers: int = 4,
                 global_layers: int = 3,
                 window_size: int = 32,
                 sparsity_factor: int = 4,
                 conv_kernel_size: int = 7,
                 fusion_type: str = 'weighted_sum',
                 use_time_features: bool = True,
                 learnable_pos: bool = False,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 mask_ratio: float = 0.25,
                 mask_strategy: str = 'random',
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 warmup_steps: int = 1000,
                 max_steps: int = 50000,
                 **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        # MS-CAT编码器
        self.encoder = MSCAT(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            local_layers=local_layers,
            global_layers=global_layers,
            window_size=window_size,
            sparsity_factor=sparsity_factor,
            conv_kernel_size=conv_kernel_size,
            fusion_type=fusion_type,
            use_time_features=use_time_features,
            learnable_pos=learnable_pos,
            dropout=dropout,
            max_len=max_len
        )
        
        # 重构头
        self.reconstruction_head = ReconstructionHead(
            d_model=d_model,
            output_dim=input_dim,
            dropout=dropout
        )
        
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy
        
        # 用于记录训练指标
        self.train_losses = []
        self.val_losses = []
        
    def create_mask(self, 
                   batch_size: int, 
                   seq_len: int, 
                   device: torch.device) -> torch.Tensor:
        """
        创建掩蔽模式
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            device: 设备
        Returns:
            [batch_size, seq_len] 掩蔽矩阵（True表示被掩蔽）
        """
        if self.mask_strategy == 'random':
            # 随机掩蔽
            mask = torch.rand(batch_size, seq_len, device=device) < self.mask_ratio
            
        elif self.mask_strategy == 'block':
            # 块状掩蔽
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            
            for b in range(batch_size):
                # 随机选择掩蔽块的数量和位置
                num_blocks = max(1, int(self.mask_ratio * seq_len / 10))  # 平均每块10个时间步
                
                for _ in range(num_blocks):
                    block_size = torch.randint(5, 15, (1,)).item()  # 块大小5-15
                    start_pos = torch.randint(0, max(1, seq_len - block_size), (1,)).item()
                    end_pos = min(start_pos + block_size, seq_len)
                    
                    mask[b, start_pos:end_pos] = True
                    
        elif self.mask_strategy == 'channel':
            # 通道掩蔽（在forward中处理）
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            
        else:
            # 默认随机掩蔽
            mask = torch.rand(batch_size, seq_len, device=device) < self.mask_ratio
        
        # 确保至少有一些位置被掩蔽
        if not mask.any():
            # 随机掩蔽一些位置
            num_mask = max(1, int(self.mask_ratio * seq_len))
            for b in range(batch_size):
                indices = torch.randperm(seq_len)[:num_mask]
                mask[b, indices] = True
        
        return mask
    
    def apply_mask(self, 
                  x: torch.Tensor, 
                  mask: torch.Tensor) -> torch.Tensor:
        """
        应用掩蔽到输入
        
        Args:
            x: [batch_size, seq_len, input_dim] 输入特征
            mask: [batch_size, seq_len] 掩蔽矩阵
        Returns:
            [batch_size, seq_len, input_dim] 掩蔽后的特征
        """
        masked_x = x.clone()
        
        if self.mask_strategy == 'channel':
            # 通道级掩蔽：随机掩蔽某些特征通道
            batch_size, seq_len, input_dim = x.shape
            
            for b in range(batch_size):
                # 随机选择要掩蔽的通道
                num_mask_channels = max(1, int(self.mask_ratio * input_dim))
                mask_channels = torch.randperm(input_dim)[:num_mask_channels]
                
                # 掩蔽选定的通道
                masked_x[b, :, mask_channels] = 0
                
                # 更新时间掩码（标记有通道被掩蔽的时间步）
                mask[b, :] = True
        else:
            # 时间步掩蔽：将掩蔽位置设为0或噪声
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            
            if torch.rand(1).item() < 0.5:
                # 50%概率设为0
                masked_x[mask_expanded] = 0
            else:
                # 50%概率添加噪声
                noise = torch.randn_like(x) * 0.1
                masked_x[mask_expanded] = noise[mask_expanded]
        
        return masked_x
    
    def forward(self, 
               x: torch.Tensor, 
               timestamps: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, input_dim] 输入特征
            timestamps: [batch_size, seq_len] 时间戳（可选）
        Returns:
            预测结果字典
        """
        batch_size, seq_len, input_dim = x.shape
        
        # 创建掩蔽
        mask = self.create_mask(batch_size, seq_len, x.device)
        
        # 应用掩蔽
        masked_x = self.apply_mask(x, mask)
        
        # 编码
        encoded = self.encoder(masked_x, timestamps)
        
        # 重构
        reconstructed = self.reconstruction_head(encoded)
        
        return {
            'reconstructed': reconstructed,
            'mask': mask,
            'masked_input': masked_x,
            'encoded': encoded
        }
    
    def compute_loss(self, 
                    predictions: Dict[str, torch.Tensor],
                    targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算重构损失
        
        Args:
            predictions: 模型预测结果
            targets: [batch_size, seq_len, input_dim] 原始输入
        Returns:
            损失字典
        """
        reconstructed = predictions['reconstructed']
        mask = predictions['mask']
        
        # 计算重构损失（仅在掩蔽位置）
        reconstruction_loss = self.reconstruction_head.compute_loss(
            reconstructed, targets, mask
        )
        
        # 可选：添加一致性正则化
        consistency_loss = 0.0
        if hasattr(self, 'consistency_weight') and self.consistency_weight > 0:
            # 对未掩蔽位置，重构应该接近原始输入
            unmask = ~mask
            if unmask.any():
                unmask_expanded = unmask.unsqueeze(-1).expand_as(targets)
                unmask_recon_loss = F.mse_loss(
                    reconstructed[unmask_expanded], 
                    targets[unmask_expanded]
                )
                consistency_loss = self.consistency_weight * unmask_recon_loss
        
        total_loss = reconstruction_loss + consistency_loss
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'consistency_loss': consistency_loss
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        x = batch['x']  # [batch_size, seq_len, input_dim]
        
        # 前向传播
        predictions = self.forward(x)
        
        # 计算损失
        losses = self.compute_loss(predictions, x)
        
        # 记录指标
        self.log('train/total_loss', losses['total_loss'], prog_bar=True)
        self.log('train/reconstruction_loss', losses['reconstruction_loss'])
        self.log('train/consistency_loss', losses['consistency_loss'])
        
        # 记录掩蔽统计
        mask_ratio_actual = predictions['mask'].float().mean()
        self.log('train/mask_ratio', mask_ratio_actual)
        
        self.train_losses.append(losses['total_loss'].item())
        
        return losses['total_loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """验证步骤"""
        x = batch['x']
        
        # 前向传播
        predictions = self.forward(x)
        
        # 计算损失
        losses = self.compute_loss(predictions, x)
        
        # 记录指标
        self.log('val/total_loss', losses['total_loss'], prog_bar=True)
        self.log('val/reconstruction_loss', losses['reconstruction_loss'])
        self.log('val/consistency_loss', losses['consistency_loss'])
        
        self.val_losses.append(losses['total_loss'].item())
        
        return losses['total_loss']
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # 学习率调度器
        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                # 线性预热
                return step / self.hparams.warmup_steps
            else:
                # 余弦退火
                progress = (step - self.hparams.warmup_steps) / (self.hparams.max_steps - self.hparams.warmup_steps)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def on_train_epoch_end(self):
        """训练轮次结束时的回调"""
        if self.train_losses:
            avg_train_loss = np.mean(self.train_losses)
            self.log('train/epoch_avg_loss', avg_train_loss)
            self.train_losses.clear()
    
    def on_validation_epoch_end(self):
        """验证轮次结束时的回调"""
        if self.val_losses:
            avg_val_loss = np.mean(self.val_losses)
            self.log('val/epoch_avg_loss', avg_val_loss)
            self.val_losses.clear()

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='MS-CAT掩蔽预训练')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data_path', type=str, help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./outputs/pretrain', help='输出目录')
    parser.add_argument('--resume_from', type=str, help='从检查点恢复训练')
    parser.add_argument('--gpus', type=int, default=1, help='GPU数量')
    parser.add_argument('--precision', type=str, default='16-mixed', help='训练精度')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置中的参数
    if args.data_path:
        config['data']['data_path'] = args.data_path
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    pl.seed_everything(config.get('seed', 42))
    
    # 数据模块
    data_module = AMPds2DataModule(**config['data'])
    data_module.setup('fit')
    
    # 获取特征维度
    input_dim = data_module.get_feature_dim()
    print(f"输入特征维度: {input_dim}")
    
    # 模型
    model_config = config['model']
    model_config['input_dim'] = input_dim
    model = MaskedReconstructionModel(**model_config)
    
    # 日志记录器
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='pretrain_logs',
        version=None
    )
    
    # 回调函数
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, 'checkpoints'),
            filename='mscat_pretrain_{epoch:02d}_{val/total_loss:.4f}',
            monitor='val/total_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor='val/total_loss',
            patience=config.get('early_stopping_patience', 10),
            mode='min'
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # 训练器
    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', 100),
        max_steps=config['model'].get('max_steps', 50000),
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 'auto',
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        val_check_interval=config.get('val_check_interval', 1.0),
        log_every_n_steps=config.get('log_every_n_steps', 50)
    )
    
    # 开始训练
    if args.resume_from:
        print(f"从检查点恢复训练: {args.resume_from}")
        trainer.fit(model, data_module, ckpt_path=args.resume_from)
    else:
        print("开始预训练...")
        trainer.fit(model, data_module)
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'mscat_pretrain_final.ckpt')
    trainer.save_checkpoint(final_model_path)
    print(f"预训练完成，模型保存至: {final_model_path}")

if __name__ == '__main__':
    main()