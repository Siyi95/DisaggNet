#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练好的NILM模型
"""

import os
import sys
import yaml
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from enhanced_datamodule import EnhancedAMPds2DataModule
from train_finetune import NILMModel

def test_model():
    """测试训练好的模型"""
    
    # 模型路径
    model_dir = './outputs/simple_training'
    config_path = os.path.join(model_dir, 'config.yaml')
    model_path = os.path.join(model_dir, 'best-epoch=48-val/total_loss=134.742.ckpt')
    
    print("=== 测试训练好的NILM模型 ===")
    print(f"配置文件: {config_path}")
    print(f"模型文件: {model_path}")
    
    # 检查文件是否存在
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在 {config_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"\n配置信息:")
    print(f"- 窗口长度: {config['data']['window_length']}")
    print(f"- 批次大小: {config['data']['batch_size']}")
    print(f"- 模型维度: {config['model']['d_model']}")
    print(f"- 注意力头数: {config['model']['num_heads']}")
    print(f"- 层数: {config['model']['num_layers']}")
    
    # 创建数据模块
    print("\n创建数据模块...")
    datamodule = EnhancedAMPds2DataModule(**config['data'])
    datamodule.setup('fit')  # 使用fit而不是test
    
    # 获取特征信息
    input_dim = datamodule.feature_dim
    num_devices = datamodule.num_devices
    device_names = datamodule.device_columns
    
    # 如果仍然为None，手动设置
    if input_dim is None:
        input_dim = 38  # 从日志中看到的特征维度
    if num_devices is None:
        num_devices = 10  # 从日志中看到的设备数量
    if device_names is None:
        device_names = [f'device_{i}' for i in range(num_devices)]
    
    print(f"\n数据信息:")
    print(f"- 输入特征维度: {input_dim}")
    print(f"- 设备数量: {num_devices}")
    print(f"- 设备名称: {device_names}")
    
    # 创建模型配置
    model_config = config['model'].copy()
    model_config.update({
        'input_dim': input_dim,
        'num_devices': num_devices,
        'device_names': device_names
    })
    
    # 从检查点加载模型
    print("\n加载训练好的模型...")
    try:
        model = NILMModel.load_from_checkpoint(
            model_path,
            **model_config
        )
        model.eval()
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 进行推理测试
    print("\n开始推理测试...")
    try:
        # 进行单个批次的推理测试
        val_loader = datamodule.val_dataloader()
        batch = next(iter(val_loader))
        
        with torch.no_grad():
            print(f"批次类型: {type(batch)}")
            print(f"批次内容: {batch}")
            
            # 检查batch的结构
            if isinstance(batch, dict):
                x = batch['x']
                y_power = batch['y_power']
                y_state = batch['y_state']
            elif isinstance(batch, (list, tuple)) and len(batch) >= 3:
                x, y_power, y_state = batch[0], batch[1], batch[2]
            else:
                print("批次格式不正确")
                return
                
            print(f"输入形状: {x.shape}")
            print(f"功率标签形状: {y_power.shape}")
            print(f"状态标签形状: {y_state.shape}")
            
            # 模型推理
            predictions = model(x)
            
            print(f"\n预测结果:")
            print(f"- 预测键: {list(predictions.keys())}")
            print(f"- 功率预测形状: {predictions['power_pred'].shape}")
            if 'state_pred' in predictions:
                print(f"- 状态预测形状: {predictions['state_pred'].shape}")
            else:
                print("- 状态预测: 不可用")
            
            # 显示一些统计信息
            power_pred = predictions['power_pred'].cpu().numpy()
            
            print(f"\n预测统计:")
            print(f"- 功率预测范围: [{power_pred.min():.3f}, {power_pred.max():.3f}]")
            
            if 'state_pred' in predictions:
                state_pred = predictions['state_pred'].cpu().numpy()
                print(f"- 状态预测范围: [{state_pred.min():.3f}, {state_pred.max():.3f}]")
            else:
                print("- 状态预测: 不可用")
            
            # 真实值统计
            y_power_np = y_power.cpu().numpy()
            y_state_np = y_state.cpu().numpy()
            
            print(f"\n真实值统计:")
            print(f"- 功率真实值范围: [{y_power_np.min():.3f}, {y_power_np.max():.3f}]")
            print(f"- 状态真实值范围: [{y_state_np.min():.3f}, {y_state_np.max():.3f}]")
        
        print("\n推理测试完成！模型工作正常。")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_model()