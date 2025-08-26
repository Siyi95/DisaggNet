#!/usr/bin/env python3
"""训练脚本"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import argparse
from datetime import datetime
import json
from pathlib import Path
import logging
import warnings
from typing import Dict, Any, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.nilm_disaggregation.data.datamodule import NILMDataModule
from src.nilm_disaggregation.training.advanced_strategies import AdvancedNILMLightningModule
from src.nilm_disaggregation.training.optimization import HyperparameterOptimizer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 忽略一些警告
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*The dataloader.*")


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


def get_default_config() -> Dict[str, Any]:
    """获取基于最终方案的默认配置"""
    return {
        'model': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 3,
            'd_ff': 512,  # d_model * 4
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'stochastic_depth': 0.1,
            'conv_channels': 128,
        },
        'training': {
            'learning_rate': 1e-3,
            'weight_decay': 0.05,
            'warmup_ratio': 0.05,
            'batch_size': 16,
            'max_epochs': 100,
            'gradient_clip_val': 1.0,
            'patience': 20,
            'mixed_precision': False,
            'output_dir': 'outputs/training'
        },
        'data': {
            'data_dir': 'data',
            'sequence_length': 512,
            'num_workers': 0,
            'jitter_std': 0.005,
            'scale_range': (0.98, 1.02),
            'time_mask_ratio': 0.05
        }
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            return json.load(f)
        else:
            import yaml
            return yaml.safe_load(f)


def train_model(config_path: str = None, data_dir: str = None, output_dir: str = None, 
                use_wandb: bool = False, wandb_project: str = "nilm-final") -> tuple:
    """基于最终方案训练模型"""
    
    # 加载配置
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
        logger.info(f"从 {config_path} 加载配置")
    else:
        config = get_default_config()
        logger.info("使用默认配置")
    
    # 覆盖配置
    if data_dir:
        config['data']['data_dir'] = data_dir
    if output_dir:
        config['training']['output_dir'] = output_dir
    
    # 创建输出目录
    output_dir = config['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"开始基于最终方案的模型训练...")
    logger.info(f"数据目录: {config['data']['data_dir']}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"序列长度: {config['data']['sequence_length']}")
    logger.info(f"模型维度: {config['model']['d_model']}")
    
    # 创建数据模块
    data_module = NILMDataModule(
        data_path=config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        sequence_length=config['data']['sequence_length'],
        num_workers=config['data']['num_workers']
    )
    
    # 创建模型和损失函数
    from src.nilm_disaggregation.models.enhanced_model_architecture import EnhancedTransformerNILMModel, UncertaintyWeightedLoss
    
    # 创建基础模型
    base_model = EnhancedTransformerNILMModel(
        input_dim=1,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        window_size=64,
        num_appliances=4
    )
    
    # 创建损失函数
    loss_function = UncertaintyWeightedLoss(num_appliances=4)
    
    # 创建Lightning模块
    model = AdvancedNILMLightningModule(
        model=base_model,
        loss_function=loss_function,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_ratio=config['training']['warmup_ratio'],
        gradient_clip_val=config['training']['gradient_clip_val']
    )
    
    # 创建回调函数
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, 'checkpoints'),
            filename='best_model_{epoch:02d}_{val_loss:.4f}',
        monitor='val_loss',
            mode='min',
            save_top_k=3,
            verbose=True,
            save_last=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config['training']['patience'],
            mode='min',
            verbose=True,
            min_delta=0.001
        ),
        LearningRateMonitor(logging_interval='step')
    ]
    
    # 创建日志记录器
    loggers = []
    
    # TensorBoard日志
    tb_logger = TensorBoardLogger(
        save_dir=output_dir,
        name='tensorboard_logs',
        version=datetime.now().strftime('%Y%m%d_%H%M%S')
    )
    loggers.append(tb_logger)
    
    # WandB日志（可选）
    if use_wandb:
        try:
            wandb_logger = WandbLogger(
                project=wandb_project,
                name=f"nilm_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                save_dir=output_dir,
                config=config
            )
            loggers.append(wandb_logger)
            logger.info("启用WandB日志记录")
        except ImportError:
            logger.warning("WandB未安装，跳过WandB日志记录")
    
    # 创建训练器（基于最终方案配置）
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        callbacks=callbacks,
        logger=loggers,
        accelerator='cpu',
        devices=1,
        precision='16-mixed' if config['training']['mixed_precision'] else 32,
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=1,
        val_check_interval=0.5,  # 每半个epoch验证一次
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # 允许非确定性操作以提高性能
        benchmark=True  # 优化CUDA性能
    )
    
    # 训练模型
    logger.info("开始训练...")
    trainer.fit(model, data_module)
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, 'final_model.ckpt')
    trainer.save_checkpoint(final_model_path)
    
    # 保存训练统计信息
    training_stats = {
        'best_val_metric': float(trainer.callback_metrics.get('val_loss', float('inf'))),
        'total_epochs': trainer.current_epoch + 1,
        'best_model_path': trainer.checkpoint_callback.best_model_path,
        'final_model_path': final_model_path,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    stats_file = os.path.join(output_dir, 'training_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    logger.info(f"\n训练完成!")
    logger.info(f"最佳验证指标: {training_stats['best_val_metric']:.4f}")
    logger.info(f"训练轮数: {training_stats['total_epochs']}")
    logger.info(f"最佳模型: {trainer.checkpoint_callback.best_model_path}")
    logger.info(f"最终模型: {final_model_path}")
    logger.info(f"训练统计: {stats_file}")
    
    return trainer, model


def main(args=None):
    """主函数"""
    if args is None:
        parser = argparse.ArgumentParser(description='训练增强版Transformer NILM模型')
        parser.add_argument('--config', type=str, help='配置文件路径')
        parser.add_argument('--data_dir', type=str, help='数据目录路径')
        parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
        parser.add_argument('--use_wandb', action='store_true', help='使用WandB日志记录')
        parser.add_argument('--wandb_project', type=str, default='nilm-final', help='WandB项目名称')
        parser.add_argument('--seed', type=int, default=42, help='随机种子')
        
        args = parser.parse_args()
    
    # 设置随机种子
    pl.seed_everything(args.seed)
    
    try:
        trainer, model = train_model(
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            use_wandb=getattr(args, 'use_wandb', False),
            wandb_project=getattr(args, 'wandb_project', 'nilm-final')
        )
        
        logger.info("训练完成！")
        
    except KeyboardInterrupt:
        logger.info("\n训练被用户中断")
    except Exception as e:
        logger.error(f"\n训练过程中发生错误: {e}")
        raise


if __name__ == '__main__':
    main()