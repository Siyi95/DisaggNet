import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np

class RegressionHead(nn.Module):
    """功率回归头"""
    
    def __init__(self,
                 d_model: int,
                 num_devices: int,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        
        self.d_model = d_model
        self.num_devices = num_devices
        
        if hidden_dim is None:
            hidden_dim = d_model // 2
        
        # 激活函数选择
        if activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'swish':
            act_fn = nn.SiLU()
        else:
            act_fn = nn.ReLU()
        
        # 回归网络
        self.regression_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_devices)
        )
        
        # 输出激活（确保非负功率）
        self.output_activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, num_devices] 预测的功率值
        """
        power_pred = self.regression_net(x)
        power_pred = self.output_activation(power_pred)
        
        return power_pred

class EventDetectionHead(nn.Module):
    """事件检测头（状态分类）"""
    
    def __init__(self,
                 d_model: int,
                 num_devices: int,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 use_focal_loss: bool = True,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()
        
        self.d_model = d_model
        self.num_devices = num_devices
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        if hidden_dim is None:
            hidden_dim = d_model // 2
        
        # 事件检测网络
        self.event_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_devices)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            Dict containing:
                - 'logits': [batch_size, seq_len, num_devices] 原始logits
                - 'probs': [batch_size, seq_len, num_devices] 概率值
        """
        logits = self.event_net(x)
        probs = torch.sigmoid(logits)
        
        return {
            'logits': logits,
            'probs': probs
        }
    
    def compute_loss(self, 
                    logits: torch.Tensor, 
                    targets: torch.Tensor,
                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算事件检测损失
        
        Args:
            logits: [batch_size, seq_len, num_devices]
            targets: [batch_size, seq_len, num_devices]
            mask: [batch_size, seq_len] 可选掩码
        Returns:
            损失值
        """
        if self.use_focal_loss:
            loss = self._focal_loss(logits, targets, mask)
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, reduction='none'
            )
            
            if mask is not None:
                mask = mask.unsqueeze(-1).expand_as(loss)
                loss = loss * mask
            
            loss = loss.mean()
        
        return loss
    
    def _focal_loss(self, 
                   logits: torch.Tensor, 
                   targets: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Focal Loss实现"""
        # 计算BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        # 计算概率
        probs = torch.sigmoid(logits)
        
        # 计算pt
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        # 计算alpha权重
        alpha_t = torch.where(targets == 1, self.focal_alpha, 1 - self.focal_alpha)
        
        # Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * bce_loss
        
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(focal_loss)
            focal_loss = focal_loss * mask
        
        return focal_loss.mean()

class MultiTaskHead(nn.Module):
    """多任务头部，整合回归和事件检测"""
    
    def __init__(self,
                 d_model: int,
                 num_devices: int,
                 regression_hidden_dim: Optional[int] = None,
                 event_hidden_dim: Optional[int] = None,
                 dropout: float = 0.1,
                 use_focal_loss: bool = True,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 power_loss_weight: float = 1.0,
                 event_loss_weight: float = 0.5):
        super().__init__()
        
        self.power_loss_weight = power_loss_weight
        self.event_loss_weight = event_loss_weight
        
        # 回归头
        self.regression_head = RegressionHead(
            d_model=d_model,
            num_devices=num_devices,
            hidden_dim=regression_hidden_dim,
            dropout=dropout
        )
        
        # 事件检测头
        self.event_head = EventDetectionHead(
            d_model=d_model,
            num_devices=num_devices,
            hidden_dim=event_hidden_dim,
            dropout=dropout,
            use_focal_loss=use_focal_loss,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            Dict containing:
                - 'power_pred': [batch_size, seq_len, num_devices] 功率预测
                - 'event_logits': [batch_size, seq_len, num_devices] 事件logits
                - 'event_probs': [batch_size, seq_len, num_devices] 事件概率
        """
        # 功率回归
        power_pred = self.regression_head(x)
        
        # 事件检测
        event_outputs = self.event_head(x)
        
        return {
            'power_pred': power_pred,
            'event_logits': event_outputs['logits'],
            'event_probs': event_outputs['probs']
        }
    
    def compute_loss(self,
                    predictions: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor],
                    mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算多任务损失
        
        Args:
            predictions: 模型预测结果
            targets: 真实标签，包含 'power' 和 'state'
            mask: 可选掩码
        Returns:
            损失字典
        """
        losses = {}
        
        # 功率回归损失（MAE）
        power_loss = F.l1_loss(
            predictions['power_pred'], 
            targets['power'], 
            reduction='none'
        )
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(power_loss)
            power_loss = power_loss * mask_expanded
        
        losses['power_loss'] = power_loss.mean()
        
        # 事件检测损失
        event_loss = self.event_head.compute_loss(
            predictions['event_logits'],
            targets['state'],
            mask
        )
        losses['event_loss'] = event_loss
        
        # 总损失
        total_loss = (self.power_loss_weight * losses['power_loss'] + 
                     self.event_loss_weight * losses['event_loss'])
        losses['total_loss'] = total_loss
        
        return losses

class ReconstructionHead(nn.Module):
    """重构头部，用于掩蔽预训练"""
    
    def __init__(self,
                 d_model: int,
                 output_dim: int,
                 hidden_dim: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = d_model
        
        self.reconstruction_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, output_dim] 重构的特征
        """
        return self.reconstruction_net(x)
    
    def compute_loss(self,
                    predictions: torch.Tensor,
                    targets: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
        """
        计算重构损失（仅在掩蔽位置）
        
        Args:
            predictions: [batch_size, seq_len, output_dim]
            targets: [batch_size, seq_len, output_dim]
            mask: [batch_size, seq_len] 掩蔽位置（True表示被掩蔽）
        Returns:
            重构损失
        """
        # 只计算掩蔽位置的损失
        loss = F.mse_loss(predictions, targets, reduction='none')
        
        # 应用掩码
        mask_expanded = mask.unsqueeze(-1).expand_as(loss)
        masked_loss = loss * mask_expanded
        
        # 计算平均损失（只在掩蔽位置）
        num_masked = mask_expanded.sum()
        if num_masked > 0:
            return masked_loss.sum() / num_masked
        else:
            return torch.tensor(0.0, device=loss.device)

class AdaptiveHead(nn.Module):
    """自适应头部，可根据任务动态调整"""
    
    def __init__(self,
                 d_model: int,
                 num_devices: int,
                 task_embedding_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_devices = num_devices
        self.task_embedding_dim = task_embedding_dim
        
        # 任务嵌入
        self.task_embeddings = nn.Embedding(3, task_embedding_dim)  # 3个任务：回归、分类、重构
        
        # 任务特定的变换
        self.task_transform = nn.Linear(d_model + task_embedding_dim, d_model)
        
        # 输出层
        self.output_layers = nn.ModuleDict({
            'regression': nn.Linear(d_model, num_devices),
            'classification': nn.Linear(d_model, num_devices),
            'reconstruction': nn.Linear(d_model, d_model)
        })
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor, 
                task_id: int) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            task_id: 任务ID (0: 回归, 1: 分类, 2: 重构)
        Returns:
            任务特定的输出
        """
        batch_size, seq_len, d_model = x.shape
        
        # 获取任务嵌入
        task_emb = self.task_embeddings(torch.tensor(task_id, device=x.device))
        task_emb = task_emb.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        # 融合任务嵌入
        x_with_task = torch.cat([x, task_emb], dim=-1)
        x_transformed = self.task_transform(x_with_task)
        x_transformed = self.dropout(x_transformed)
        
        # 根据任务选择输出层
        if task_id == 0:  # 回归
            output = self.output_layers['regression'](x_transformed)
            output = F.relu(output)  # 确保非负
        elif task_id == 1:  # 分类
            output = self.output_layers['classification'](x_transformed)
        else:  # 重构
            output = self.output_layers['reconstruction'](x_transformed)
        
        return output