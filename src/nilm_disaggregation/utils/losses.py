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
        
        # 相关性损失
        correlation_loss = 0
        for i in range(power_pred.size(1)):
            pred_i = power_pred[:, i]
            true_i = power_true[:, i]
            
            pred_mean = torch.mean(pred_i)
            true_mean = torch.mean(true_i)
            
            pred_centered = pred_i - pred_mean
            true_centered = true_i - true_mean
            
            numerator = torch.sum(pred_centered * true_centered)
            denominator = torch.sqrt(torch.sum(pred_centered**2) * torch.sum(true_centered**2))
            
            correlation = numerator / (denominator + 1e-8)
            correlation_loss += (1 - correlation)
        
        correlation_loss /= power_pred.size(1)
        
        total_loss = (self.power_weight * power_loss + 
                     self.state_weight * state_loss + 
                     self.correlation_weight * correlation_loss)
        
        return total_loss, power_loss, state_loss, correlation_loss