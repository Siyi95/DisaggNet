#!/usr/bin/env python3
"""Optuna超参数优化脚本"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import tempfile
import shutil

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from src.nilm_disaggregation.data import NILMDataModule
from src.nilm_disaggregation.training import EnhancedTransformerNILMModule
from src.nilm_disaggregation.utils import load_config, get_default_config


class OptunaPruningCallback(PyTorchLightningPruningCallback):
    """Optuna剪枝回调"""
    
    def on_validation_end(self, trainer, pl_module):
        # 获取验证损失
        logs = trainer.callback_metrics
        current_score = logs.get('val_loss')
        
        if current_score is None:
            return
        
        # 报告中间结果
        self.trial.report(current_score, step=trainer.current_epoch)
        
        # 检查是否应该剪枝
        if self.trial.should_prune():
            message = f"Trial was pruned at epoch {trainer.current_epoch}."
            raise optuna.exceptions.TrialPruned(message)


def objective(trial, config, data_module, output_dir, device):
    """Optuna优化目标函数"""
    
    # 建议超参数
    suggested_params = {
        # 模型参数
        'model.d_model': trial.suggest_categorical('d_model', [128, 256, 512]),
        'model.nhead': trial.suggest_categorical('nhead', [4, 8, 16]),
        'model.num_layers': trial.suggest_int('num_layers', 2, 8),
        'model.dim_feedforward': trial.suggest_categorical('dim_feedforward', [512, 1024, 2048]),
        'model.dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'model.lstm_hidden_size': trial.suggest_categorical('lstm_hidden_size', [128, 256, 512]),
        'model.lstm_num_layers': trial.suggest_int('lstm_num_layers', 1, 3),
        
        # 训练参数
        'training.learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'training.weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'data.batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
        
        # 损失函数参数
        'loss.power_weight': trial.suggest_float('power_weight', 0.5, 2.0),
        'loss.state_weight': trial.suggest_float('state_weight', 0.5, 2.0),
        'loss.correlation_weight': trial.suggest_float('correlation_weight', 0.0, 1.0),
    }
    
    # 更新配置
    trial_config = config.copy()
    for key, value in suggested_params.items():
        trial_config.set(key, value)
    
    # 确保nhead能被d_model整除
    d_model = trial_config.get('model.d_model')
    nhead = trial_config.get('model.nhead')
    if d_model % nhead != 0:
        # 调整nhead为d_model的因子
        possible_nheads = [i for i in [4, 8, 16] if d_model % i == 0]
        if possible_nheads:
            trial_config.set('model.nhead', possible_nheads[0])
        else:
            trial_config.set('model.nhead', 4)
            trial_config.set('model.d_model', 256)  # 使用默认值
    
    # 更新数据模块的批次大小
    data_module.batch_size = trial_config.get('data.batch_size')
    
    # 创建模型
    model = EnhancedTransformerNILMModule(
        model_params=trial_config.get('model', {}),
        loss_params=trial_config.get('loss', {}),
        learning_rate=trial_config.get('training.learning_rate'),
        weight_decay=trial_config.get('training.weight_decay', 1e-4),
        appliances=trial_config.get('data.appliances', ['fridge', 'washer_dryer', 'microwave', 'dishwasher'])
    )
    
    # 创建临时目录用于此次试验
    trial_dir = output_dir / f'trial_{trial.number}'
    trial_dir.mkdir(exist_ok=True)
    
    # 设置回调函数
    callbacks = [
        ModelCheckpoint(
            dirpath=trial_dir,
            filename='best-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=False
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=trial_config.get('training.early_stopping_patience', 10),
            mode='min',
            verbose=True
        ),
        OptunaPruningCallback(trial, monitor='val_loss')
    ]
    
    # 设置日志记录器
    logger = TensorBoardLogger(
        save_dir=trial_dir,
        name='optuna_logs',
        version=f'trial_{trial.number}'
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=trial_config.get('training.max_epochs', 50),
        accelerator='gpu' if device.type == 'cuda' else 'cpu',
        devices=1 if device.type == 'cuda' else 'auto',
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=False,  # 减少输出
        enable_model_summary=False,
        deterministic=True,
        check_val_every_n_epoch=1
    )
    
    try:
        # 训练模型
        trainer.fit(model, data_module)
        
        # 获取最佳验证损失
        best_score = trainer.callback_metrics.get('val_loss')
        
        if best_score is None:
            # 如果没有验证损失，使用训练损失
            best_score = trainer.callback_metrics.get('train_loss', float('inf'))
        
        # 清理临时文件（保留最佳模型）
        if trial_dir.exists():
            # 只保留最佳检查点
            checkpoint_files = list(trial_dir.glob('*.ckpt'))
            if len(checkpoint_files) > 1:
                # 找到最佳检查点
                best_checkpoint = None
                best_val_loss = float('inf')
                for ckpt_file in checkpoint_files:
                    if 'best-' in ckpt_file.name:
                        best_checkpoint = ckpt_file
                        break
                
                # 删除其他检查点
                for ckpt_file in checkpoint_files:
                    if ckpt_file != best_checkpoint:
                        ckpt_file.unlink(missing_ok=True)
        
        return float(best_score)
        
    except optuna.exceptions.TrialPruned:
        # 试验被剪枝
        raise
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return float('inf')
    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Optuna超参数优化')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--data_path', type=str, required=True, help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='outputs/optimization', help='输出目录')
    parser.add_argument('--n_trials', type=int, default=100, help='试验次数')
    parser.add_argument('--timeout', type=int, help='优化超时时间（秒）')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    parser.add_argument('--study_name', type=str, help='研究名称')
    parser.add_argument('--storage', type=str, help='存储URL（用于分布式优化）')
    parser.add_argument('--pruner', type=str, default='median', 
                       choices=['median', 'percentile', 'hyperband'], help='剪枝器类型')
    parser.add_argument('--sampler', type=str, default='tpe', 
                       choices=['tpe', 'random', 'cmaes'], help='采样器类型')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # 命令行参数覆盖配置
    config.set('data.data_path', args.data_path)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    print(f"数据路径: {args.data_path}")
    print(f"输出目录: {output_dir}")
    print(f"试验次数: {args.n_trials}")
    
    # 设置随机种子
    pl.seed_everything(config.get('seed', 42))
    
    # 创建数据模块
    data_module = NILMDataModule(
        data_path=config.get('data.data_path'),
        sequence_length=config.get('data.sequence_length', 512),
        batch_size=config.get('data.batch_size', 32),
        num_workers=config.get('data.num_workers', 4),
        train_ratio=config.get('data.train_ratio', 0.8),
        max_samples=config.get('data.max_samples', 50000)
    )
    
    # 准备数据
    data_module.setup('fit')
    
    # 设置剪枝器
    if args.pruner == 'median':
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    elif args.pruner == 'percentile':
        pruner = optuna.pruners.PercentilePruner(25.0, n_startup_trials=5, n_warmup_steps=10)
    elif args.pruner == 'hyperband':
        pruner = optuna.pruners.HyperbandPruner(min_resource=5, max_resource=50)
    else:
        pruner = optuna.pruners.MedianPruner()
    
    # 设置采样器
    if args.sampler == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=config.get('seed', 42))
    elif args.sampler == 'random':
        sampler = optuna.samplers.RandomSampler(seed=config.get('seed', 42))
    elif args.sampler == 'cmaes':
        sampler = optuna.samplers.CmaEsSampler(seed=config.get('seed', 42))
    else:
        sampler = optuna.samplers.TPESampler()
    
    # 创建研究
    study_name = args.study_name or f"nilm_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if args.storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=args.storage,
            direction='minimize',
            pruner=pruner,
            sampler=sampler,
            load_if_exists=True
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            pruner=pruner,
            sampler=sampler
        )
    
    print(f"开始优化研究: {study_name}")
    
    # 运行优化
    try:
        study.optimize(
            lambda trial: objective(trial, config, data_module, output_dir, device),
            n_trials=args.n_trials,
            timeout=args.timeout,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n优化被用户中断")
    
    # 保存结果
    print("\n优化完成！")
    print(f"最佳试验: {study.best_trial.number}")
    print(f"最佳值: {study.best_value:.6f}")
    print("最佳参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 保存最佳参数到配置文件
    best_config = config.copy()
    for key, value in study.best_params.items():
        if key == 'learning_rate':
            best_config.set('training.learning_rate', value)
        elif key == 'weight_decay':
            best_config.set('training.weight_decay', value)
        elif key == 'batch_size':
            best_config.set('data.batch_size', value)
        elif key.startswith('power_weight') or key.startswith('state_weight') or key.startswith('correlation_weight'):
            best_config.set(f'loss.{key}', value)
        else:
            best_config.set(f'model.{key}', value)
    
    # 保存最佳配置
    best_config_path = output_dir / f'best_config_{study_name}.yaml'
    best_config.save(best_config_path)
    print(f"\n最佳配置已保存到: {best_config_path}")
    
    # 保存研究结果
    study_results = {
        'study_name': study_name,
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'datetime': datetime.now().isoformat()
    }
    
    results_path = output_dir / f'optimization_results_{study_name}.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(study_results, f, indent=2, ensure_ascii=False)
    print(f"优化结果已保存到: {results_path}")
    
    # 生成优化历史图表
    try:
        import matplotlib.pyplot as plt
        
        # 优化历史
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 目标值历史
        values = [trial.value for trial in study.trials if trial.value is not None]
        ax1.plot(values)
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Optimization History')
        ax1.grid(True)
        
        # 参数重要性
        if len(study.trials) > 10:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())
            importances = list(importance.values())
            
            ax2.barh(params, importances)
            ax2.set_xlabel('Importance')
            ax2.set_title('Parameter Importance')
        
        plt.tight_layout()
        plot_path = output_dir / f'optimization_plot_{study_name}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"优化图表已保存到: {plot_path}")
        
    except ImportError:
        print("matplotlib未安装，跳过图表生成")
    except Exception as e:
        print(f"生成图表时出错: {e}")
    
    print("\n优化完成！")


if __name__ == '__main__':
    main()