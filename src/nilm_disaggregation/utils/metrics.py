"""评估指标模块"""

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, test_loader, appliances, device):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        appliances: 设备名称列表
        device: 计算设备
        
    Returns:
        results: 评估结果字典
        power_preds: 功率预测值
        power_trues: 功率真实值
        state_preds: 状态预测值
        state_trues: 状态真实值
    """
    model.eval()
    all_power_preds = []
    all_power_trues = []
    all_state_preds = []
    all_state_trues = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, power_true, state_true = batch
            x = x.to(device)
            
            power_pred, state_pred = model(x)
            
            all_power_preds.append(power_pred.cpu())
            all_power_trues.append(power_true)
            all_state_preds.append(state_pred.cpu())
            all_state_trues.append(state_true)
    
    power_preds = torch.cat(all_power_preds).numpy()
    power_trues = torch.cat(all_power_trues).numpy()
    state_preds = torch.cat(all_state_preds).numpy()
    state_trues = torch.cat(all_state_trues).numpy()
    
    results = {}
    
    for i, appliance in enumerate(appliances):
        pred_power = power_preds[:, i]
        true_power = power_trues[:, i]
        
        mae = mean_absolute_error(true_power, pred_power)
        rmse = np.sqrt(mean_squared_error(true_power, pred_power))
        r2 = r2_score(true_power, pred_power)
        correlation = np.corrcoef(pred_power, true_power)[0, 1]
        
        if np.isnan(correlation):
            correlation = 0.0
        
        results[appliance] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'correlation': float(correlation)
        }
    
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


def calculate_metrics(y_true, y_pred):
    """
    计算单个设备的评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        metrics: 指标字典
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    correlation = np.corrcoef(y_pred, y_true)[0, 1]
    
    if np.isnan(correlation):
        correlation = 0.0
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'correlation': float(correlation)
    }