#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NILM-MSCAT Optuna超参数优化训练脚本
使用Optuna进行自动超参数搜索，确保模型充分训练
"""

import os
import sys
import argparse
import yaml
import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
from typing import Dict, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from enhanced_datamodule import EnhancedAMPds2DataModule
from train_finetune import NILMModel

class EnhancedOptunaTrainer:
    """增强的Optuna超参数优化训练器"""
    
    def __init__(self, base_config_path: str, output_dir: str = "./outputs/optuna"):
        self.base_config_path = base_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载基础配置
        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)
            
        # 确保使用CPU训练（避免CUDA问题）
        self.base_config['trainer']['accelerator'] = 'cpu'
        self.base_config['trainer']['devices'] = 1
        self.base_config['trainer']['precision'] = '32'
        
        # 增加数据量和训练轮数
        self.base_config['data']['window_length'] = 256  # 减小窗口长度
        self.base_config['data']['step_size'] = 64       # 减小步长，增加重叠
        self.base_config['data']['batch_size'] = 16      # 减小batch size
        self.base_config['trainer']['max_epochs'] = 50   # 适中的训练轮数
        
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """建议超参数"""
        # 数据相关超参数
        window_length = trial.suggest_categorical('window_length', [128, 256, 512, 1024])
        step_size = trial.suggest_categorical('step_size', [16, 32, 64, 128])
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
        min_samples = trial.suggest_int('min_samples', 50, 500)
        
        # 模型架构超参数
        d_model = trial.suggest_categorical('d_model', [128, 256, 512])
        num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
        num_layers = trial.suggest_int('num_layers', 3, 8)
        dropout = trial.suggest_float('dropout', 0.05, 0.3)
        
        # TCN参数
        tcn_base = trial.suggest_categorical('tcn_base', [32, 64, 128])
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 7])
        
        # 任务头参数
        regression_hidden_dim = trial.suggest_categorical('regression_hidden_dim', [64, 128, 256])
        event_hidden_dim = trial.suggest_categorical('event_hidden_dim', [32, 64, 128])
        
        # 损失权重
        power_loss_weight = trial.suggest_float('power_loss_weight', 0.5, 3.0)
        event_loss_weight = trial.suggest_float('event_loss_weight', 0.1, 2.0)
        
        # Focal Loss参数
        focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.5)
        focal_gamma = trial.suggest_float('focal_gamma', 1.0, 3.0)
        
        # 优化器参数
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        encoder_lr_factor = trial.suggest_float('encoder_lr_factor', 0.1, 1.0)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # 训练相关超参数
        max_epochs = trial.suggest_int('max_epochs', 30, 200)
        patience = trial.suggest_int('patience', 10, 30)
        
        # 数据增强
        augment = trial.suggest_categorical('augment', [True, False])
        
        # CRF相关
        use_crf = trial.suggest_categorical('use_crf', [True, False])
        
        return {
            'window_length': window_length,
            'step_size': step_size,
            'batch_size': batch_size,
            'min_samples': min_samples,
            'd_model': d_model,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout': dropout,
            'tcn_base': tcn_base,
            'tcn_channels': [tcn_base, tcn_base*2, tcn_base*4],
            'kernel_size': kernel_size,
            'regression_hidden_dim': regression_hidden_dim,
            'event_hidden_dim': event_hidden_dim,
            'power_loss_weight': power_loss_weight,
            'event_loss_weight': event_loss_weight,
            'focal_alpha': focal_alpha,
            'focal_gamma': focal_gamma,
            'learning_rate': learning_rate,
            'encoder_lr_factor': encoder_lr_factor,
            'weight_decay': weight_decay,
            'max_epochs': max_epochs,
            'patience': patience,
            'augment': augment,
            'use_crf': use_crf
        }
        
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna目标函数"""
        try:
            # 获取超参数
            hyperparams = self.suggest_hyperparameters(trial)
            
            print(f"\n=== Trial {trial.number} ===")
            print(f"超参数: {hyperparams}")
            
            # 创建增强数据模块
            datamodule = EnhancedAMPds2DataModule(
                data_path=self.base_config['data']['data_path'],
                window_length=hyperparams['window_length'],
                step_size=hyperparams['step_size'],
                batch_size=hyperparams['batch_size'],
                num_workers=self.base_config['data']['num_workers'],
                channels=self.base_config['data'].get('channels', None),
                device_columns=self.base_config['data'].get('device_columns', None),
                power_threshold=self.base_config['data']['power_threshold'],
                augment=hyperparams['augment'],
                min_samples=hyperparams['min_samples']
            )
            
            # 设置数据
            datamodule.setup('fit')
            
            # 检查训练数据是否足够
            train_loader = datamodule.train_dataloader()
            val_loader = datamodule.val_dataloader()
            
            print(f"训练批次数: {len(train_loader)}")
            print(f"验证批次数: {len(val_loader)}")
            
            if len(train_loader) < 5:
                print(f"Trial {trial.number}: 训练数据不足，跳过")
                return float('inf')
                
            # 更新配置
            config = self.base_config.copy()
            config['model']['d_model'] = hyperparams['d_model']
            config['model']['num_heads'] = hyperparams['num_heads']
            config['model']['num_layers'] = hyperparams['num_layers']
            config['model']['dropout'] = hyperparams['dropout']
            config['model']['tcn_channels'] = hyperparams['tcn_channels']
            config['model']['kernel_size'] = hyperparams['kernel_size']
            config['model']['regression_hidden_dim'] = hyperparams['regression_hidden_dim']
            config['model']['event_hidden_dim'] = hyperparams['event_hidden_dim']
            config['model']['power_loss_weight'] = hyperparams['power_loss_weight']
            config['model']['event_loss_weight'] = hyperparams['event_loss_weight']
            config['model']['focal_alpha'] = hyperparams['focal_alpha']
            config['model']['focal_gamma'] = hyperparams['focal_gamma']
            config['model']['use_crf'] = hyperparams['use_crf']
            
            # 创建模型
            model_config = config['model'].copy()
            model_config.update({
                'input_dim': datamodule.feature_dim,
                'num_devices': datamodule.num_devices,
                'device_names': datamodule.device_columns,
                'learning_rate': hyperparams['learning_rate'],
                'encoder_lr': hyperparams['learning_rate'] * hyperparams['encoder_lr_factor'],
                'weight_decay': hyperparams['weight_decay']
            })
            
            model = NILMModel(**model_config)
            
            # 创建回调
            callbacks = [
                EarlyStopping(
                    monitor='val_total_loss',
                    patience=hyperparams['patience'],
                    mode='min',
                    verbose=False,
                    min_delta=0.001
                ),
                ModelCheckpoint(
                    dirpath=self.output_dir / f"trial_{trial.number}",
                    filename='best_model',
                    monitor='val_total_loss',
                    mode='min',
                    save_top_k=1,
                    verbose=False
                )
            ]
            
            # 创建logger
            logger = TensorBoardLogger(
                save_dir=self.output_dir,
                name=f"trial_{trial.number}",
                version=None
            )
            
            # 创建训练器
            trainer = pl.Trainer(
                max_epochs=hyperparams['max_epochs'],
                accelerator='cpu',  # 强制使用CPU
                devices=1,
                callbacks=callbacks,
                logger=logger,
                enable_progress_bar=False,
                enable_model_summary=False,
                check_val_every_n_epoch=5,  # 每5个epoch验证一次
                log_every_n_steps=10,
                enable_checkpointing=True
            )
            
            # 训练模型
            trainer.fit(model, datamodule)
            
            # 获取最佳验证损失
            if trainer.callback_metrics:
                best_val_loss = trainer.callback_metrics.get('val_total_loss', float('inf'))
            else:
                # 如果没有验证指标，使用训练损失
                best_val_loss = trainer.logged_metrics.get('train_total_loss', float('inf'))
            
            # 确保返回有效数值
            if torch.is_tensor(best_val_loss):
                best_val_loss = float(best_val_loss.item())
            elif not isinstance(best_val_loss, (int, float)):
                best_val_loss = float('inf')
                
            print(f"Trial {trial.number}: val_loss = {best_val_loss:.4f}")
            
            # 清理内存
            del model, trainer, datamodule
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return best_val_loss
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return float('inf')
            
    def optimize(self, n_trials: int = 50, timeout: int = 3600):
        """执行超参数优化"""
        # 创建study
        study = optuna.create_study(
            direction='minimize',
            study_name='nilm_mscat_optimization',
            storage=f'sqlite:///{self.output_dir}/optuna.db',
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        print(f"开始Optuna超参数优化，目标: {n_trials} trials, 超时: {timeout}秒")
        print(f"结果保存至: {self.output_dir}")
        print(f"数据库路径: {self.output_dir}/optuna.db")
        
        # 执行优化
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # 输出结果
        print("\n=== 优化完成 ===")
        print(f"完成试验数: {len(study.trials)}")
        print(f"最佳试验: {study.best_trial.number}")
        print(f"最佳验证损失: {study.best_value:.6f}")
        print("\n最佳超参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
            
        # 保存最佳配置
        best_config = self.base_config.copy()
        best_params = study.best_params
        
        # 更新最佳配置
        best_config['data']['window_length'] = best_params['window_length']
        best_config['data']['step_size'] = best_params['step_size']
        best_config['data']['batch_size'] = best_params['batch_size']
        best_config['model']['d_model'] = best_params['d_model']
        best_config['model']['num_heads'] = best_params['num_heads']
        best_config['model']['num_layers'] = best_params['num_layers']
        best_config['model']['dropout'] = best_params['dropout']
        best_config['model']['tcn_channels'] = [best_params['tcn_base'], best_params['tcn_base']*2, best_params['tcn_base']*4]
        best_config['model']['kernel_size'] = best_params['kernel_size']
        best_config['model']['regression_hidden_dim'] = best_params['regression_hidden_dim']
        best_config['model']['event_hidden_dim'] = best_params['event_hidden_dim']
        best_config['model']['power_loss_weight'] = best_params['power_loss_weight']
        best_config['model']['event_loss_weight'] = best_params['event_loss_weight']
        best_config['model']['focal_alpha'] = best_params['focal_alpha']
        best_config['model']['focal_gamma'] = best_params['focal_gamma']
        best_config['model']['use_crf'] = best_params['use_crf']
        best_config['optimizer']['lr'] = best_params['learning_rate']
        best_config['optimizer']['weight_decay'] = best_params['weight_decay']
        best_config['trainer']['max_epochs'] = best_params['max_epochs']
        
        best_config_path = self.output_dir / 'best_config.yaml'
        with open(best_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(best_config, f, default_flow_style=False, allow_unicode=True)
        print(f"\n最佳配置已保存至: {best_config_path}")
        
        # 保存优化历史
        trials_df = study.trials_dataframe()
        trials_df.to_csv(self.output_dir / "optimization_history.csv", index=False)
        print(f"优化历史已保存至: {self.output_dir}/optimization_history.csv")
        
        return study
        
def train_with_best_config(best_config_path: str):
    """使用最佳配置进行完整训练"""
    print(f"\n=== 使用最佳配置进行完整训练 ===")
    
    with open(best_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建数据模块
    datamodule = EnhancedAMPds2DataModule(
        data_path=config['data']['data_path'],
        window_length=config['data']['window_length'],
        step_size=config['data']['step_size'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        channels=config['data'].get('channels', None),
        device_columns=config['data'].get('device_columns', None),
        power_threshold=config['data']['power_threshold'],
        augment=True,
        min_samples=200  # 更多训练样本
    )
    
    datamodule.setup('fit')
    
    # 创建模型
    model_config = config['model'].copy()
    model_config.update({
        'input_dim': datamodule.feature_dim,
        'num_devices': datamodule.num_devices,
        'device_names': datamodule.device_columns
    })
    
    model = NILMModel(**model_config)
    
    # 创建回调
    callbacks = [
        EarlyStopping(
            monitor='val_total_loss',
            patience=20,
            mode='min',
            verbose=True,
            min_delta=0.001
        ),
        ModelCheckpoint(
            dirpath="./outputs/best_model",
            filename='final_best_model',
            monitor='val_total_loss',
            mode='min',
            save_top_k=1,
            verbose=True
        )
    ]
    
    # 创建logger
    logger = TensorBoardLogger(
        save_dir="./outputs",
        name="best_model_training",
        version=None
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=config['trainer']['max_epochs'],
        accelerator='cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=5
    )
    
    # 训练模型
    trainer.fit(model, datamodule)
    
    print("\n=== 最佳模型训练完成 ===")
    print(f"最佳模型保存至: ./outputs/best_model/final_best_model.ckpt")
    print(f"训练日志: ./outputs/best_model_training")

def main():
    parser = argparse.ArgumentParser(description='NILM-MSCAT Optuna超参数优化')
    parser.add_argument('--config', type=str, default='configs/quick_start.yaml',
                       help='基础配置文件路径')
    parser.add_argument('--output', type=str, default='./outputs/optuna',
                       help='输出目录')
    parser.add_argument('--trials', type=int, default=20,
                       help='优化试验次数')
    parser.add_argument('--timeout', type=int, default=7200,
                       help='优化超时时间（秒）')
    parser.add_argument('--train-best', action='store_true', help='使用最佳配置训练')
    
    args = parser.parse_args()
    
    if args.train_best:
        best_config_path = "./outputs/optuna/best_config.yaml"
        if os.path.exists(best_config_path):
            train_with_best_config(best_config_path)
        else:
            print(f"最佳配置文件不存在: {best_config_path}")
            print("请先运行优化: python train_optuna.py")
    else:
        # 创建优化器
        optimizer = EnhancedOptunaTrainer(
            base_config_path=args.config,
            output_dir=args.output
        )
        
        # 执行优化
        study = optimizer.optimize(
            n_trials=args.trials,
            timeout=args.timeout
        )
        
        print("\n=== 优化完成，可以使用以下命令: ===")
        print("1. 查看优化结果:")
        print(f"   optuna-dashboard sqlite:///{args.output}/optuna.db")
        print("2. 使用最佳配置训练:")
        print("   python train_optuna.py --train-best")
    
if __name__ == '__main__':
    main()