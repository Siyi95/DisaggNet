#!/usr/bin/env python3
"""
简单训练调试脚本
用于找出训练立即失败的根本原因
"""

import os
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# 添加项目路径
sys.path.insert(0, '/Users/siyili/Workspace/DisaggNet')
sys.path.insert(0, '/Users/siyili/Workspace/DisaggNet/src')

from src.nilm_disaggregation.data.datamodule import NILMDataModule
from src.nilm_disaggregation.models.enhanced_transformer import EnhancedTransformerNILM
from src.nilm_disaggregation.training.lightning_module import EnhancedTransformerNILMModule
from src.nilm_disaggregation.utils.losses import CombinedLoss

def main():
    print("开始简单训练调试...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 固定超参数
    params = {
        # 模型参数
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 2,
        'd_ff': 256,
        'dropout': 0.2,
        'window_size': 32,
        
        # 训练参数
        'learning_rate': 1e-4,
        'batch_size': 16,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        
        # 损失权重
        'power_weight': 1.0,
        'state_weight': 0.5,
        'correlation_weight': 0.3,
        
        # 数据参数
        'sequence_length': 64
    }
    
    print(f"使用参数: {params}")
    
    try:
        # 创建数据模块
        print("\n创建数据模块...")
        data_module = NILMDataModule(
            data_path='/Users/siyili/Workspace/DisaggNet/Dataset/dataverse_files/AMPds2.h5',
            sequence_length=params['sequence_length'],
            batch_size=params['batch_size'],
            num_workers=0,
            max_samples=1000  # 减少样本数
        )
        
        # 准备数据
        print("准备数据...")
        data_module.prepare_data()
        data_module.setup('fit')
        
        # 检查数据集
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        
        print(f"训练样本数: {len(data_module.train_dataset)}")
        print(f"验证样本数: {len(data_module.val_dataset)}")
        
        if len(data_module.train_dataset) == 0:
            print("错误: 训练数据集为空！")
            return
            
        if len(data_module.val_dataset) == 0:
            print("错误: 验证数据集为空！")
            return
        
        # 获取一个批次数据进行测试
        print("\n测试数据加载...")
        for batch in train_loader:
            x, y_power, y_state = batch
            print(f"输入形状: {x.shape}")
            print(f"功率输出形状: {y_power.shape}")
            print(f"状态输出形状: {y_state.shape}")
            print(f"输入数据范围: [{x.min():.4f}, {x.max():.4f}]")
            print(f"功率数据范围: [{y_power.min():.4f}, {y_power.max():.4f}]")
            print(f"状态数据范围: [{y_state.min():.4f}, {y_state.max():.4f}]")
            break
        
        # 创建模型
        print("\n创建模型...")
        model = EnhancedTransformerNILM(
            input_dim=x.shape[-1],
            d_model=params['d_model'],
            n_heads=params['n_heads'],
            n_layers=params['n_layers'],
            d_ff=params['d_ff'],
            dropout=params['dropout'],
            window_size=params['window_size'],
            num_appliances=y_power.shape[-1]
        )
        
        # 测试模型前向传播
        print("\n测试模型前向传播...")
        model.eval()
        with torch.no_grad():
            power_pred, state_pred = model(x)
            print(f"预测功率形状: {power_pred.shape}")
            print(f"预测状态形状: {state_pred.shape}")
            print(f"预测功率范围: [{power_pred.min():.4f}, {power_pred.max():.4f}]")
            print(f"预测状态范围: [{state_pred.min():.4f}, {state_pred.max():.4f}]")
        
        # 创建损失函数
        print("\n创建损失函数...")
        loss_fn = CombinedLoss(
            power_weight=params['power_weight'],
            state_weight=params['state_weight'],
            correlation_weight=params['correlation_weight']
        )
        
        # 测试损失计算
        print("\n测试损失计算...")
        model.train()
        power_pred, state_pred = model(x)
        loss_result = loss_fn(power_pred, state_pred, y_power, y_state)
        
        if isinstance(loss_result, tuple):
            total_loss, power_loss, state_loss, corr_loss = loss_result
            print(f"总损失: {total_loss.item():.6f}")
            print(f"功率损失: {power_loss.item():.6f}")
            print(f"状态损失: {state_loss.item():.6f}")
            print(f"相关性损失: {corr_loss.item():.6f}")
            loss = total_loss
        else:
            loss = loss_result
            print(f"损失值: {loss.item():.6f}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("错误: 损失值为NaN或Inf！")
            return
        
        # 测试反向传播
        print("\n测试反向传播...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'])
        optimizer.zero_grad()
        loss.backward()
        
        # 检查梯度
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        print(f"梯度范数: {total_norm:.6f}")
        
        optimizer.step()
        print("反向传播成功！")
        
        # 创建Lightning模块
        print("\n创建Lightning模块...")
        model_params = {
            'input_dim': x.shape[-1],
            'd_model': params['d_model'],
            'n_heads': params['n_heads'],
            'n_layers': params['n_layers'],
            'd_ff': params['d_ff'],
            'dropout': params['dropout'],
            'window_size': params['window_size'],
            'num_appliances': y_power.shape[-1]
        }
        loss_params = {
            'power_weight': params['power_weight'],
            'state_weight': params['state_weight'],
            'correlation_weight': params['correlation_weight']
        }
        lightning_module = EnhancedTransformerNILMModule(
            model_params=model_params,
            loss_params=loss_params,
            learning_rate=params['learning_rate']
        )
        
        # 创建训练器
        print("\n创建训练器...")
        trainer = pl.Trainer(
            max_epochs=1,  # 只训练1个epoch
            limit_train_batches=2,  # 只训练2个批次
            limit_val_batches=1,   # 只验证1个批次
            accelerator='auto',
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=False
        )
        
        # 开始训练
        print("\n开始训练...")
        trainer.fit(lightning_module, data_module)
        
        print("\n训练成功完成！")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()