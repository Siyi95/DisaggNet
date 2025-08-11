#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的NILM训练脚本
解决数据不足和训练稳定性问题
"""

import os
import sys
import yaml
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import argparse

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from enhanced_datamodule import EnhancedAMPds2DataModule
from train_finetune import NILMModel

def create_simple_config():
    """创建简化的训练配置"""
    return {
        'data': {
            'data_path': 'data/AMPds2.h5',
            'window_length': 128,  # 减小窗口长度
            'step_size': 16,       # 减小步长，增加重叠
            'batch_size': 8,       # 减小批次大小
            'num_workers': 2,
            'power_threshold': 10.0,
            'augment': True,
            'min_samples': 100     # 最小样本数
        },
        'model': {
            'd_model': 128,        # 减小模型大小
            'num_heads': 4,
            'num_layers': 3,
            'dropout': 0.1,
            'tcn_channels': [32, 64, 128],
            'kernel_size': 3,
            'regression_hidden_dim': 64,
            'event_hidden_dim': 32,
            'learning_rate': 0.001,
            'encoder_lr': 0.0005,
            'weight_decay': 1e-5,
            'power_loss_weight': 1.0,
            'event_loss_weight': 0.5,
            'use_crf': False       # 暂时禁用CRF
        },
        'training': {
            'max_epochs': 50,
            'patience': 10,
            'val_check_interval': 1.0
        }
    }

def main():
    parser = argparse.ArgumentParser(description='简化NILM训练')
    parser.add_argument('--output_dir', type=str, default='./outputs/simple_training', help='输出目录')
    parser.add_argument('--gpus', type=int, default=0, help='GPU数量')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    pl.seed_everything(42)
    
    # 创建配置
    config = create_simple_config()
    
    print("=== 简化NILM训练 ===")
    print(f"输出目录: {args.output_dir}")
    print(f"配置: {config}")
    
    # 创建数据模块
    print("\n创建数据模块...")
    datamodule = EnhancedAMPds2DataModule(**config['data'])
    datamodule.setup('fit')
    
    # 检查数据
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")
    
    if len(train_loader) == 0:
        print("错误: 没有训练数据！")
        return
    
    # 获取特征信息
    input_dim = datamodule.feature_dim
    num_devices = datamodule.num_devices
    device_names = datamodule.device_columns
    
    print(f"输入特征维度: {input_dim}")
    print(f"设备数量: {num_devices}")
    print(f"设备名称: {device_names}")
    
    # 创建模型
    print("\n创建模型...")
    model_config = config['model'].copy()
    model_config.update({
        'input_dim': input_dim,
        'num_devices': num_devices,
        'device_names': device_names
    })
    
    model = NILMModel(**model_config)
    
    # 创建回调
    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename='best-{epoch:02d}-{val/total_loss:.3f}',
            monitor='val/total_loss',
            mode='min',
            save_top_k=1,
            save_last=True
        ),
        EarlyStopping(
            monitor='val/total_loss',
            patience=config['training']['patience'],
            mode='min'
        )
    ]
    
    # 创建日志记录器
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='simple_training'
    )
    
    # 创建训练器
    print("\n创建训练器...")
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='cpu',
        devices=1,
        precision='32',
        logger=logger,
        callbacks=callbacks,
        val_check_interval=config['training']['val_check_interval'],
        log_every_n_steps=1,  # 每步都记录
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # 开始训练
    print("\n开始训练...")
    try:
        trainer.fit(model, datamodule)
        print("\n训练完成！")
        
        # 保存最终模型
        final_model_path = os.path.join(args.output_dir, 'final_model.ckpt')
        trainer.save_checkpoint(final_model_path)
        print(f"最终模型保存至: {final_model_path}")
        
        # 保存配置
        config_path = os.path.join(args.output_dir, 'config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"配置保存至: {config_path}")
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()