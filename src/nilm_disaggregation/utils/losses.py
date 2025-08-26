"""损失函数模块"""

import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """组合损失函数，包含功率预测损失、状态预测损失和相关性损失"""
    
    def __init__(self, power_weight=1.0, state_weight=0.5, correlation_weight=0.3):
        super().__init__()
        self.power_weight = power_weight
        self.state_weight = state_weight
        self.correlation_weight = correlation_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, power_pred, state_pred, power_true, state_true):
        # 功率预测损失
        power_loss = self.mse_loss(power_pred, power_true)
        
        # 状态预测损失
        state_loss = self.bce_loss(state_pred, state_true)
        
        # 相关性损失 - 使用更稳定的计算方式
        correlation_loss = torch.tensor(0.0, device=power_pred.device, requires_grad=True)
        
        for i in range(power_pred.size(1)):
            pred_i = power_pred[:, i]
            true_i = power_true[:, i]
            
            # 使用torch.var和torch.cov来计算相关系数，更稳定
            pred_mean = torch.mean(pred_i)
            true_mean = torch.mean(true_i)
            
            pred_centered = pred_i - pred_mean
            true_centered = true_i - true_mean
            
            # 添加小的epsilon避免除零
            pred_std = torch.sqrt(torch.mean(pred_centered**2) + 1e-8)
            true_std = torch.sqrt(torch.mean(true_centered**2) + 1e-8)
            
            # 计算相关系数
            covariance = torch.mean(pred_centered * true_centered)
            correlation = covariance / (pred_std * true_std + 1e-8)
            
            # 确保相关系数在[-1, 1]范围内
            correlation = torch.clamp(correlation, -1.0, 1.0)
            correlation_loss = correlation_loss + (1 - correlation)
        
        correlation_loss = correlation_loss / power_pred.size(1)
        
        total_loss = (self.power_weight * power_loss + 
                     self.state_weight * state_loss + 
                     self.correlation_weight * correlation_loss)
        
        return total_loss, power_loss, state_loss, correlation_loss