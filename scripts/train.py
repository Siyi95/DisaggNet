#!/usr/bin/env python3
"""训练脚本"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from src.nilm_disaggregation.data import NILMDataModule
from src.nilm_disaggregation.training import EnhancedTransformerNILMModule
from src.nilm_disaggregation.utils import load_config, get_default_config


def setup_callbacks(config):
    """设置训练回调函数"""
    callbacks = []
    
    # 模型检查点回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.get('checkpoint.save_dir', 'outputs/checkpoints'),
        filename=config.get('checkpoint.filename', 'enhanced_transformer_nilm-{epoch:02d}-{val_loss:.2f}'),
        monitor=config.get('checkpoint.monitor', 'val_loss'),
        mode=config.get('checkpoint.mode', 'min'),
        save_top_k=config.get('checkpoint.save_top_k', 3),
        save_last=config.get('checkpoint.save_last', True),
        every_n_epochs=config.get('checkpoint.every_n_epochs', 1),
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # 早停回调
    early_stopping_callback = EarlyStopping(
        monitor=config.get('checkpoint.monitor', 'val_loss'),
        patience=config.get('training.patience', 10),
        min_delta=config.get('training.min_delta', 1e-4),
        mode=config.get('checkpoint.mode', 'min'),
        verbose=True
    )
    callbacks.append(early_stopping_callback)
    
    return callbacks


def setup_logger(config):
    """设置日志记录器"""
    logger = TensorBoardLogger(
        save_dir=config.get('logging.save_dir', 'outputs/logs'),
        name=config.get('logging.name', 'enhanced_transformer_nilm'),
        version=config.get('logging.version')
    )
    return logger


def setup_trainer(config, callbacks, logger):
    """设置训练器"""
    trainer = pl.Trainer(
        max_epochs=config.get('training.max_epochs', 100),
        accelerator=config.get('device.accelerator', 'auto'),
        devices=config.get('device.devices', 'auto'),
        strategy=config.get('device.strategy', 'auto'),
        precision=config.get('training.precision', 32),
        gradient_clip_val=config.get('training.gradient_clip_val', 1.0),
        accumulate_grad_batches=config.get('training.accumulate_grad_batches', 1),
        val_check_interval=config.get('validation.val_check_interval', 1.0),
        check_val_every_n_epoch=config.get('validation.check_val_every_n_epoch', 1),
        log_every_n_steps=config.get('logging.log_every_n_steps', 50),
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    return trainer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练增强版Transformer NILM模型')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--data_path', type=str, help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--resume_from_checkpoint', type=str, help='从检查点恢复训练')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--fast_dev_run', action='store_true', help='快速开发运行模式')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # 命令行参数覆盖配置
    if args.data_path:
        config.set('data.data_path', args.data_path)
    if args.seed:
        config.set('seed', args.seed)
    
    # 设置随机种子
    seed = config.get('seed', 42)
    pl.seed_everything(seed)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_save_path = output_dir / f'config_{timestamp}.yaml'
    config.save_config(config_save_path)
    print(f"配置文件已保存到: {config_save_path}")
    
    # 创建数据模块
    data_module = NILMDataModule(
        data_path=config.get('data.data_path'),
        sequence_length=config.get('data.sequence_length', 512),
        batch_size=config.get('data.batch_size', 32),
        num_workers=config.get('data.num_workers', 4),
        train_ratio=config.get('data.train_ratio', 0.8),
        max_samples=config.get('data.max_samples', 50000)
    )
    
    # 创建模型模块
    model_params = config.get('model', {})
    loss_params = config.get('loss', {})
    learning_rate = config.get('training.learning_rate', 1e-4)
    appliances = config.get('data.appliances', ['fridge', 'washer_dryer', 'microwave', 'dishwasher'])
    
    model_module = EnhancedTransformerNILMModule(
        model_params=model_params,
        loss_params=loss_params,
        learning_rate=learning_rate,
        appliances=appliances
    )
    
    # 设置回调函数和日志记录器
    callbacks = setup_callbacks(config)
    logger = setup_logger(config)
    
    # 设置训练器
    trainer_kwargs = {
        'config': config,
        'callbacks': callbacks,
        'logger': logger
    }
    
    if args.fast_dev_run:
        trainer_kwargs['config'].set('training.max_epochs', 1)
        trainer_kwargs['config'].set('data.max_samples', 1000)
        print("快速开发运行模式已启用")
    
    trainer = setup_trainer(**trainer_kwargs)
    
    # 开始训练
    print("开始训练...")
    print(f"设备: {trainer.accelerator}")
    print(f"最大训练轮数: {config.get('training.max_epochs', 100)}")
    print(f"批次大小: {config.get('data.batch_size', 32)}")
    print(f"学习率: {learning_rate}")
    
    try:
        trainer.fit(
            model_module,
            datamodule=data_module,
            ckpt_path=args.resume_from_checkpoint
        )
        
        # 训练完成后进行测试
        print("\n开始测试...")
        trainer.test(model_module, datamodule=data_module)
        
        # 保存最终模型
        final_model_path = output_dir / f'final_model_{timestamp}.ckpt'
        trainer.save_checkpoint(final_model_path)
        print(f"最终模型已保存到: {final_model_path}")
        
        print("训练完成！")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    main()