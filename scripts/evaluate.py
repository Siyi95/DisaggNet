#!/usr/bin/env python3
"""评估脚本"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytorch_lightning as pl
import numpy as np

from src.nilm_disaggregation.data import NILMDataModule
from src.nilm_disaggregation.training import EnhancedTransformerNILMModule
from src.nilm_disaggregation.utils import (
    load_config, get_default_config, evaluate_model, 
    create_visualizations, calculate_metrics
)


def load_model_from_checkpoint(checkpoint_path, config=None):
    """从检查点加载模型"""
    if config:
        model_params = config.get('model', {})
        loss_params = config.get('loss', {})
        learning_rate = config.get('training.learning_rate', 1e-4)
        appliances = config.get('data.appliances', ['fridge', 'washer_dryer', 'microwave', 'dishwasher'])
        
        model = EnhancedTransformerNILMModule(
            model_params=model_params,
            loss_params=loss_params,
            learning_rate=learning_rate,
            appliances=appliances
        )
        
        # 加载检查点权重
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # 直接从检查点加载
        model = EnhancedTransformerNILMModule.load_from_checkpoint(checkpoint_path)
    
    return model


def evaluate_on_test_set(model, data_module, device, appliances):
    """在测试集上评估模型"""
    model.eval()
    model.to(device)
    
    # 获取测试数据加载器
    test_loader = data_module.test_dataloader()
    
    all_power_preds = []
    all_power_trues = []
    all_state_preds = []
    all_state_trues = []
    
    print("正在评估模型...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i % 10 == 0:
                print(f"处理批次 {i+1}/{len(test_loader)}")
            
            x, power_true, state_true = batch
            x = x.to(device)
            
            power_pred, state_pred = model(x)
            
            all_power_preds.append(power_pred.cpu())
            all_power_trues.append(power_true)
            all_state_preds.append(state_pred.cpu())
            all_state_trues.append(state_true)
    
    # 合并所有预测结果
    power_preds = torch.cat(all_power_preds).numpy()
    power_trues = torch.cat(all_power_trues).numpy()
    state_preds = torch.cat(all_state_preds).numpy()
    state_trues = torch.cat(all_state_trues).numpy()
    
    # 计算评估指标
    results = {}
    for i, appliance in enumerate(appliances):
        pred_power = power_preds[:, i]
        true_power = power_trues[:, i]
        
        metrics = calculate_metrics(true_power, pred_power)
        results[appliance] = metrics
    
    # 计算平均指标
    avg_mae = np.mean([results[app]['mae'] for app in appliances])
    avg_rmse = np.mean([results[app]['rmse'] for app in appliances])
    avg_r2 = np.mean([results[app]['r2'] for app in appliances])
    avg_correlation = np.mean([results[app]['correlation'] for app in appliances])
    
    results['average'] = {
        'mae': float(avg_mae),
        'rmse': float(avg_rmse),
        'r2': float(avg_r2),
        'correlation': float(avg_correlation)
    }
    
    return results, power_preds, power_trues, state_preds, state_trues


def print_results(results, appliances):
    """打印评估结果"""
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    
    # 打印每个设备的指标
    for appliance in appliances:
        if appliance in results:
            metrics = results[appliance]
            print(f"\n{appliance}:")
            print(f"  MAE:         {metrics['mae']:.4f}")
            print(f"  RMSE:        {metrics['rmse']:.4f}")
            print(f"  R²:          {metrics['r2']:.4f}")
            print(f"  相关系数:     {metrics['correlation']:.4f}")
    
    # 打印平均指标
    if 'average' in results:
        avg_metrics = results['average']
        print(f"\n平均指标:")
        print(f"  平均 MAE:    {avg_metrics['mae']:.4f}")
        print(f"  平均 RMSE:   {avg_metrics['rmse']:.4f}")
        print(f"  平均 R²:     {avg_metrics['r2']:.4f}")
        print(f"  平均相关系数: {avg_metrics['correlation']:.4f}")
    
    print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估增强版Transformer NILM模型')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--data_path', type=str, help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation', help='输出目录')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--no_visualization', action='store_true', help='不生成可视化图表')
    
    args = parser.parse_args()
    
    # 检查检查点文件是否存在
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
    
    # 命令行参数覆盖配置
    if args.data_path:
        config.set('data.data_path', args.data_path)
    if args.batch_size:
        config.set('data.batch_size', args.batch_size)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"从检查点加载模型: {checkpoint_path}")
    try:
        model = load_model_from_checkpoint(checkpoint_path, config)
        appliances = model.appliances
        print(f"模型加载成功，设备列表: {appliances}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
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
    data_module.setup('test')
    
    # 评估模型
    results, power_preds, power_trues, state_preds, state_trues = evaluate_on_test_set(
        model, data_module, device, appliances
    )
    
    # 打印结果
    print_results(results, appliances)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存评估指标
    results_file = output_dir / f'evaluation_results_{timestamp}.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果已保存到: {results_file}")
    
    # 保存预测结果
    predictions_file = output_dir / f'predictions_{timestamp}.npz'
    np.savez(
        predictions_file,
        power_preds=power_preds,
        power_trues=power_trues,
        state_preds=state_preds,
        state_trues=state_trues,
        appliances=appliances
    )
    print(f"预测结果已保存到: {predictions_file}")
    
    # 生成可视化图表
    if not args.no_visualization:
        print("\n生成可视化图表...")
        viz_dir = output_dir / f'visualizations_{timestamp}'
        viz_dir.mkdir(exist_ok=True)
        
        try:
            create_visualizations(
                results, power_preds, power_trues, appliances, str(viz_dir)
            )
            print(f"可视化图表已保存到: {viz_dir}")
        except Exception as e:
            print(f"生成可视化图表时出错: {e}")
    
    print("\n评估完成！")


if __name__ == '__main__':
    main()