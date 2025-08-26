"""增强版Transformer NILM模型"""

import torch
import torch.nn as nn
from typing import Tuple

from .components import (
    EnhancedMultiScaleConvBlock,
    ChannelAttention,
    EnhancedTransformerBlock,
    PositionalEncoding
)


class EnhancedTransformerNILM(nn.Module):
    """增强版Transformer NILM模型"""
    
    def __init__(
        self, 
        input_dim: int = 1, 
        d_model: int = 256, 
        n_heads: int = 8, 
        n_layers: int = 6,
        d_ff: int = 1024, 
        dropout: float = 0.1, 
        num_appliances: int = 4, 
        window_size: int = 64
    ):
        """
        初始化增强版Transformer NILM模型
        
        Args:
            input_dim: 输入维度
            d_model: 模型维度
            n_heads: 注意力头数
            n_layers: Transformer层数
            d_ff: 前馈网络维度
            dropout: Dropout率
            num_appliances: 设备数量
            window_size: 窗口大小
        """
        super().__init__()
        self.d_model = d_model
        self.num_appliances = num_appliances
        
        # 输入嵌入
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # 多尺度卷积特征提取
        self.multi_scale_conv = EnhancedMultiScaleConvBlock(d_model, d_model)
        self.channel_attention = ChannelAttention(d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            EnhancedTransformerBlock(d_model, n_heads, d_ff, dropout, window_size)
            for _ in range(n_layers)
        ])
        
        # 双向LSTM层
        self.bi_lstm = nn.LSTM(
            d_model, d_model // 2, batch_first=True, 
            bidirectional=True, dropout=dropout if dropout > 0 else 0
        )
        
        # 时间注意力机制
        self.temporal_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 输出头
        self.power_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1)
            ) for _ in range(num_appliances)
        ])
        
        self.state_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_appliances)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [B, L, input_dim]
            
        Returns:
            power_pred: 功率预测 [B, num_appliances]
            state_pred: 状态预测 [B, num_appliances]
        """
        # 输入嵌入
        x = self.input_embedding(x)  # [B, L, d_model]
        
        # 多尺度卷积特征提取
        conv_input = x.transpose(1, 2)  # [B, d_model, L]
        conv_features = self.multi_scale_conv(conv_input)
        conv_features = self.channel_attention(conv_features)
        conv_features = conv_features.transpose(1, 2)  # [B, L, d_model]
        
        # 位置编码
        x = self.pos_encoding(conv_features)
        x = self.dropout(x)
        
        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 双向LSTM
        lstm_out, _ = self.bi_lstm(x)
        
        # 时间注意力
        attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        
        # 特征融合
        fused_features = torch.cat([lstm_out, attn_out], dim=-1)
        final_features = self.feature_fusion(fused_features)
        
        # 全局平均池化
        pooled_features = torch.mean(final_features, dim=1)  # [B, d_model]
        
        # 多任务输出
        power_outputs = [head(pooled_features) for head in self.power_heads]
        state_outputs = [head(pooled_features) for head in self.state_heads]
        
        power_pred = torch.cat(power_outputs, dim=1)  # [B, num_appliances]
        state_pred = torch.cat(state_outputs, dim=1)  # [B, num_appliances]
        
        # 确保状态预测在[0,1]范围内
        state_pred = torch.clamp(state_pred, 0.0, 1.0)
        
        return power_pred, state_pred
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            包含模型参数信息的字典
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'd_model': self.d_model,
            'num_appliances': self.num_appliances,
            'transformer_layers': len(self.transformer_layers)
        }