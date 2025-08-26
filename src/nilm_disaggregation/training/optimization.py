#!/usr/bin/env python3
"""
统一的超参数优化模块
整合Optuna优化、损失权重优化等功能
"""

import os
import sys
import json
import optuna
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, Callable
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import pickle
import math

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.nilm_disaggregation.data.datamodule import NILMDataModule
from src.nilm_disaggregation.training.lightning_module import EnhancedTransformerNILMModule
from src.nilm_disaggregation.data.csv_data_loader import AMPds2CSVDataset
from src.nilm_disaggregation.models.enhanced_model_architecture import (
    EnhancedTransformerNILMModel, UncertaintyWeightedLoss
)

class HyperparameterOptimizer:
    """基于最终方案的超参数优化器"""
    
    def __init__(self, 
                 model_class,
                 loss_class,
                 data_module,
                 data_dir: str = None,
                 output_dir: str = 'outputs/optimization',
                 study_name: str = "nilm_final_optimization",
                 storage: Optional[str] = None,
                 direction: str = "minimize",
                 n_trials: int = 100,
                 timeout: Optional[int] = None):
        
        self.model_class = model_class
        self.loss_class = loss_class
        self.data_module = data_module
        self.data_dir = data_dir or 'data'
        self.output_dir = output_dir
        self.study_name = study_name
        self.storage = storage
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置高级pruner和sampler
        self.pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=15,
            interval_steps=1
        )
        
        self.sampler = optuna.samplers.TPESampler(
            n_startup_trials=15,
            n_ei_candidates=32,
            multivariate=True,
            seed=42
        )
        
        # 创建study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=self.direction,
            pruner=self.pruner,
            sampler=self.sampler,
            load_if_exists=True
        )
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """基于最终方案建议超参数配置"""
        
        # 核心架构参数（基于最终方案推荐起点）
        d_model = trial.suggest_categorical('d_model', [192, 256, 320, 384])
        n_heads = trial.suggest_categorical('n_heads', [6, 8, 10, 12])
        n_layers = trial.suggest_int('n_layers', 3, 6)
        
        # FFN扩张比（固定为4倍，符合最终方案）
        d_ff = d_model * 4
        
        # Dropout配置
        dropout = trial.suggest_float('dropout', 0.05, 0.15)
        attention_dropout = trial.suggest_float('attention_dropout', 0.05, 0.15)
        stochastic_depth = trial.suggest_float('stochastic_depth', 0.05, 0.15)
        
        # 训练参数（基于最终方案）
        learning_rate = trial.suggest_float('learning_rate', 5e-4, 2e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 0.03, 0.08)
        warmup_ratio = trial.suggest_float('warmup_ratio', 0.03, 0.08)
        
        # 批次大小
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 24, 32])
        
        # 序列长度（离线任务，支持长序列）
        sequence_length = trial.suggest_categorical('sequence_length', [2048, 3072, 4096, 6144])
        
        # Conv前端参数
        conv_channels = trial.suggest_categorical('conv_channels', [96, 128, 160, 192])
        
        # 正则化
        gradient_clip_val = trial.suggest_float('gradient_clip_val', 0.8, 1.5)
        
        # 数据增强强度
        jitter_std = trial.suggest_float('jitter_std', 0.002, 0.008)
        scale_range_width = trial.suggest_float('scale_range_width', 0.01, 0.04)
        time_mask_ratio = trial.suggest_float('time_mask_ratio', 0.02, 0.08)
        
        return {
            # 模型架构
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'd_ff': d_ff,
            'dropout': dropout,
            'attention_dropout': attention_dropout,
            'stochastic_depth': stochastic_depth,
            'conv_channels': conv_channels,
            
            # 训练配置
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'warmup_ratio': warmup_ratio,
            'batch_size': batch_size,
            'gradient_clip_val': gradient_clip_val,
            
            # 数据配置
            'sequence_length': sequence_length,
            
            # 数据增强
            'jitter_std': jitter_std,
            'scale_range': (1.0 - scale_range_width/2, 1.0 + scale_range_width/2),
            'time_mask_ratio': time_mask_ratio,
        }
        
    def objective(self, trial):
        """Optuna目标函数"""
        
        # 基于最终方案的超参数搜索空间
        trial_params = self.suggest_hyperparameters(trial)
        
        try:
            # 创建数据模块
            if self.data_module is None:
                data_module = NILMDataModule(
                    data_dir=self.data_dir,
                    batch_size=trial_params['batch_size'],
                    sequence_length=trial_params['sequence_length'],
                    num_workers=2
                )
            else:
                data_module = self.data_module
                # 更新批次大小和序列长度
                data_module.batch_size = trial_params['batch_size']
                data_module.sequence_length = trial_params['sequence_length']
            
            # 创建模型（基于最终方案）
            model = EnhancedTransformerNILMModule(
                input_dim=1,
                num_appliances=10,
                d_model=trial_params['d_model'],
                n_heads=trial_params['n_heads'],
                n_layers=trial_params['n_layers'],
                d_ff=trial_params['d_ff'],
                dropout=trial_params['dropout'],
                attention_dropout=trial_params['attention_dropout'],
                stochastic_depth=trial_params['stochastic_depth'],
                conv_channels=trial_params['conv_channels'],
                learning_rate=trial_params['learning_rate'],
                weight_decay=trial_params['weight_decay'],
                warmup_ratio=trial_params['warmup_ratio'],
                gradient_clip_val=trial_params['gradient_clip_val'],
                jitter_std=trial_params['jitter_std'],
                scale_range=trial_params['scale_range'],
                time_mask_ratio=trial_params['time_mask_ratio']
            )
            
            # 设置回调
            callbacks = [
                EarlyStopping(
                    monitor='val_combined_metric',
                    patience=15,
                    mode='min',
                    verbose=False,
                    min_delta=0.001
                ),
                ModelCheckpoint(
                    dirpath=f"{self.output_dir}/trial_{trial.number}",
                    filename='best_model',
                    monitor='val_combined_metric',
                    mode='min',
                    save_top_k=1
                )
            ]
            
            # 创建训练器（基于最终方案配置）
            trainer = pl.Trainer(
                max_epochs=80,
                callbacks=callbacks,
                accelerator='auto',
                devices=1,
                precision='16-mixed',  # 混合精度训练
                gradient_clip_val=trial_params['gradient_clip_val'],
                accumulate_grad_batches=1,
                val_check_interval=0.5,  # 每半个epoch验证一次
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False
            )
            
            # 训练模型
            trainer.fit(model, data_module)
            
            # 获取最佳验证指标
            best_metric = trainer.callback_metrics.get('val_combined_metric', float('inf'))
            if isinstance(best_metric, torch.Tensor):
                best_metric = best_metric.item()
            
            # 报告中间结果用于pruning
            for epoch in range(trainer.current_epoch + 1):
                if f'val_combined_metric_epoch_{epoch}' in trainer.logged_metrics:
                    trial.report(trainer.logged_metrics[f'val_combined_metric_epoch_{epoch}'].item(), epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            
            return best_metric
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {e}")
            return float('inf')
    
    def optimize(self, n_trials: int = None, study_name: str = None) -> Tuple[Dict, optuna.Study]:
        """执行超参数优化"""
        
        if n_trials is None:
            n_trials = self.n_trials
            
        if study_name is None:
            study_name = f"nilm_final_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"开始基于最终方案的超参数优化，目标试验次数: {n_trials}")
        print(f"使用高级pruner和sampler进行优化")
        
        # 执行优化
        self.study.optimize(
            self.objective, 
            n_trials=n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # 保存结果
        results = {
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'study_name': study_name,
            'timestamp': datetime.now().isoformat(),
            'optimization_direction': self.direction,
            'pruner': str(self.pruner),
            'sampler': str(self.sampler)
        }
        
        # 保存到文件
        results_file = os.path.join(self.output_dir, f'{study_name}_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # 保存试验历史
        trials_df = self.study.trials_dataframe()
        trials_file = os.path.join(self.output_dir, f'{study_name}_trials.csv')
        trials_df.to_csv(trials_file, index=False)
        
        # 保存study对象
        study_file = os.path.join(self.output_dir, f'{study_name}_study.pkl')
        with open(study_file, 'wb') as f:
            pickle.dump(self.study, f)
        
        print(f"\n优化完成!")
        print(f"最佳参数: {self.study.best_params}")
        print(f"最佳验证指标: {self.study.best_value:.4f}")
        print(f"完成试验数: {len(self.study.trials)}")
        print(f"结果已保存到: {results_file}")
        
        return results, self.study

class LossWeightOptimizer:
    """损失函数权重优化器"""
    
    def __init__(self, data_dir: str, max_samples: int = 2000):
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.sequence_length = 128
        self.batch_size = 16
        
        # 创建数据集
        self.train_dataset = AMPds2CSVDataset(
            data_dir=data_dir,
            sequence_length=self.sequence_length,
            max_samples=max_samples,
            train=True
        )
        
        self.val_dataset = AMPds2CSVDataset(
            data_dir=data_dir,
            sequence_length=self.sequence_length,
            max_samples=max_samples // 4,
            train=False
        )
    
    def create_model_with_weights(self, power_weight: float, state_weight: float, corr_weight: float):
        """创建带有指定权重的模型"""
        
        class WeightedNILMModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.model = EnhancedTransformerNILMModel(
                    input_dim=1,
                    d_model=128,
                    n_heads=8,
                    n_layers=3,
                    d_ff=256,
                    dropout=0.1,
                    window_size=64,
                    num_appliances=10
                )
                self.loss_fn = UncertaintyWeightedLoss(
                    num_appliances=10
                )
                
            def forward(self, x):
                return self.model(x)
                
            def training_step(self, batch, batch_idx):
                x, targets = batch
                outputs = self(x)
                loss_dict = self.loss_fn(outputs, targets)
                return loss_dict['total_loss']
                
            def validation_step(self, batch, batch_idx):
                x, targets = batch
                outputs = self(x)
                loss_dict = self.loss_fn(outputs, targets)
                self.log('val_loss', loss_dict['total_loss'])
                return loss_dict['total_loss']
                
            def configure_optimizers(self):
                return torch.optim.AdamW(self.parameters(), lr=1e-3)
        
        return WeightedNILMModel()
    
    def evaluate_weights(self, power_weight: float, state_weight: float, corr_weight: float, max_epochs: int = 5) -> float:
        """评估特定权重组合的性能"""
        
        # 创建数据加载器
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=False
        )
        
        # 创建模型
        model = self.create_model_with_weights(power_weight, state_weight, corr_weight)
        
        # 设置早停
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=3,
            mode='min'
        )
        
        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stop],
            accelerator='auto',
            devices=1,
            precision='16-mixed',
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False
        )
        
        try:
            # 训练模型
            trainer.fit(model, train_loader, val_loader)
            
            # 获取最终验证损失
            val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
            
            return float(val_loss)
            
        except Exception as e:
            print(f"训练失败: {e}")
            return float('inf')
    
    def optimize_weights(self, n_trials: int = 20) -> Tuple[Dict, float, optuna.Study]:
        """使用Optuna优化权重"""
        print("\n=== 开始Optuna权重优化 ===")
        
        def objective(trial):
            # 定义搜索空间
            power_weight = trial.suggest_float('power_weight', 0.1, 3.0)
            state_weight = trial.suggest_float('state_weight', 0.1, 3.0)
            corr_weight = trial.suggest_float('corr_weight', 0.01, 2.0)
            
            # 评估权重组合
            val_loss = self.evaluate_weights(power_weight, state_weight, corr_weight)
            
            return val_loss
        
        # 创建研究
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_loss = study.best_value
        
        return best_params, best_loss, study

def run_hyperparameter_optimization(data_dir: str, n_trials: int = 50, output_dir: str = 'outputs/optimization'):
    """运行超参数优化"""
    optimizer = HyperparameterOptimizer(data_dir, output_dir)
    results, study = optimizer.optimize(n_trials)
    return results, study

def run_loss_weight_optimization(data_dir: str, n_trials: int = 20, max_samples: int = 2000):
    """运行损失权重优化"""
    optimizer = LossWeightOptimizer(data_dir, max_samples)
    best_params, best_loss, study = optimizer.optimize_weights(n_trials)
    
    print(f"\n=== 损失权重优化结果 ===")
    print(f"最佳权重: {best_params}")
    print(f"最佳验证损失: {best_loss:.4f}")
    
    return best_params, best_loss, study

if __name__ == '__main__':
    data_dir = '/Users/siyili/Workspace/DisaggNet/Dataset/dataverse_files'
    
    print("选择优化类型:")
    print("1. 超参数优化")
    print("2. 损失权重优化")
    
    choice = input("请输入选择 (1 或 2): ")
    
    if choice == '1':
        results, study = run_hyperparameter_optimization(data_dir, n_trials=30)
    elif choice == '2':
        best_params, best_loss, study = run_loss_weight_optimization(data_dir, n_trials=15)
    else:
        print("无效选择")