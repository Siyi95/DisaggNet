#!/usr/bin/env python3
"""
完整的NILM训练脚本
包含预训练、微调和数据增强功能
解决功率损失不稳定问题
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import argparse
import yaml
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_improved import (
    ImprovedNILMModel, 
    create_pretrain_config, 
    create_finetune_config,
    run_pretraining,
    run_finetuning
)
from enhanced_datamodule import EnhancedAMPds2DataModule

def create_complete_config():
    """创建完整的训练配置"""
    config = {
        # 数据配置
        'data': {
            'data_path': '/home/yu/Workspace/DisaggNet/nilm-mscat/data/AMPds2.h5',
            'window_length': 128,
            'step_size': 32,
            'batch_size': 16,
            'num_workers': 4,
            'power_threshold': 10.0,
            'augment': True,
            'min_samples': 200
        },
        
        # 预训练配置
        'pretrain': {
            'input_dim': 38,  # 实际特征维度
            'd_model': 192,
            'nhead': 6,
            'num_layers': 6,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'mask_ratio': 0.15,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'max_epochs': 30,
            'patience': 10
        },
        
        # 微调配置
        'finetune': {
            'input_dim': 38,
            'num_devices': 10,
            'device_names': [
                'device_01', 'device_02', 'device_03', 'device_04', 'device_05',
                'device_06', 'device_07', 'device_08', 'device_09', 'device_10'
            ],
            'd_model': 192,
            'nhead': 6,
            'num_layers': 6,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'power_loss_weight': 1.0,
            'event_loss_weight': 0.5,
            'use_crf': False,
            'freeze_encoder': True,
            'encoder_lr': 1e-4,
            'head_lr': 1e-3,
            'weight_decay': 1e-4,
            'max_epochs': 50,
            'patience': 15,
            # 功率损失稳定化参数
            'power_loss_smoothing': 0.1,
            'gradient_clip_val': 1.0,
            'power_scale_factor': 1000.0
        },
        
        # 训练配置
        'training': {
            'accelerator': 'cpu',  # 使用CPU避免CUDA兼容性问题
            'devices': 1,
            'precision': 32,  # CPU使用32位精度
            'accumulate_grad_batches': 2,
            'log_every_n_steps': 50,
            'val_check_interval': 0.25
        },
        
        # 输出配置
        'output': {
            'save_dir': './outputs',
            'experiment_name': 'nilm_complete_training',
            'save_top_k': 3
        }
    }
    
    return config

def setup_data_module(config):
    """设置数据模块"""
    data_config = config['data']
    
    # 检查数据文件是否存在
    if not os.path.exists(data_config['data_path']):
        print(f"警告: 数据文件 {data_config['data_path']} 不存在")
        print("请确保数据文件路径正确")
        return None
    
    datamodule = EnhancedAMPds2DataModule(
        data_path=data_config['data_path'],
        window_length=data_config['window_length'],
        step_size=data_config['step_size'],
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        power_threshold=data_config['power_threshold'],
        augment=data_config['augment'],
        min_samples=data_config['min_samples']
    )
    
    return datamodule

def setup_callbacks_and_logger(config, stage='pretrain'):
    """设置回调函数和日志记录器"""
    output_config = config['output']
    stage_config = config[stage]
    
    # 创建输出目录
    save_dir = Path(output_config['save_dir']) / stage
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir / 'checkpoints',
        filename=f'{stage}-{{epoch:02d}}-{{val_loss:.3f}}',
        monitor='val_loss',
        mode='min',
        save_top_k=output_config['save_top_k'],
        save_last=True,
        verbose=True
    )
    
    # 早停回调
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=stage_config['patience'],
        mode='min',
        verbose=True
    )
    
    # 日志记录器
    logger = TensorBoardLogger(
        save_dir=output_config['save_dir'],
        name=output_config['experiment_name'],
        version=stage
    )
    
    return [checkpoint_callback, early_stop_callback], logger

def run_complete_training(config):
    """运行完整的训练流程"""
    print("=" * 60)
    print("开始完整的NILM训练流程")
    print("=" * 60)
    
    # 设置数据模块
    print("\n1. 设置数据模块...")
    datamodule = setup_data_module(config)
    if datamodule is None:
        return None, None
    
    # 预训练阶段
    print("\n2. 开始预训练阶段...")
    pretrain_config = create_pretrain_config(config)
    pretrained_model = run_pretraining(pretrain_config, datamodule)
    
    if pretrained_model is None:
        print("预训练失败，退出")
        return None, None
    
    # 微调阶段
    print("\n3. 开始微调阶段...")
    finetune_config = create_finetune_config(config)
    finetuned_model = run_finetuning(
        finetune_config, 
        datamodule, 
        pretrained_model_path=pretrained_model
    )
    
    if finetuned_model is None:
        print("微调失败，退出")
        return pretrained_model, None
    
    print("\n=" * 60)
    print("训练完成！")
    print(f"预训练模型: {pretrained_model}")
    print(f"微调模型: {finetuned_model}")
    print("=" * 60)
    
    return pretrained_model, finetuned_model

def save_config(config, save_path):
    """保存配置文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"配置已保存到: {save_path}")

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='完整的NILM训练脚本')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--save-config', type=str, help='保存配置文件路径')
    parser.add_argument('--data-path', type=str, help='数据文件路径')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='输出目录')
    parser.add_argument('--pretrain-only', action='store_true', help='仅运行预训练')
    parser.add_argument('--finetune-only', action='store_true', help='仅运行微调')
    parser.add_argument('--pretrained-model', type=str, help='预训练模型路径（用于仅微调模式）')
    
    args = parser.parse_args()
    
    # 加载或创建配置
    if args.config:
        config = load_config(args.config)
        print(f"已加载配置文件: {args.config}")
    else:
        config = create_complete_config()
        print("使用默认配置")
    
    # 更新配置
    if args.data_path:
        config['data']['data_path'] = args.data_path
    if args.output_dir:
        config['output']['save_dir'] = args.output_dir
    
    # 保存配置
    if args.save_config:
        save_config(config, args.save_config)
    
    # 设置随机种子
    pl.seed_everything(42)
    
    # 运行训练
    if args.pretrain_only:
        print("仅运行预训练")
        datamodule = setup_data_module(config)
        if datamodule:
            pretrain_config = create_pretrain_config(config)
            run_pretraining(pretrain_config, datamodule)
    elif args.finetune_only:
        print("仅运行微调")
        if not args.pretrained_model:
            print("错误: 微调模式需要指定预训练模型路径")
            return
        datamodule = setup_data_module(config)
        if datamodule:
            finetune_config = create_finetune_config(config)
            run_finetuning(finetune_config, datamodule, args.pretrained_model)
    else:
        # 运行完整训练流程
        run_complete_training(config)

if __name__ == '__main__':
    main()