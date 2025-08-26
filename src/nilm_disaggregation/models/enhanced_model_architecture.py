#!/usr/bin/env python3
"""
增强的模型架构
改进Transformer NILM模型的设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ALiBiPositionalBias(nn.Module):
    """ALiBi (Attention with Linear Biases) 位置编码
    
    相比传统位置编码，ALiBi更适合长序列外推，训练稳定性更好
    """
    
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        
        # 计算每个头的斜率
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        
    def _get_slopes(self, num_heads: int) -> torch.Tensor:
        """计算ALiBi斜率"""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            slopes = get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            slopes.extend(get_slopes_power_of_2(2*closest_power_of_2)[0:num_heads-closest_power_of_2])
        
        return torch.tensor(slopes, dtype=torch.float32)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """生成ALiBi偏置矩阵
        
        Returns:
            bias: [num_heads, seq_len, seq_len]
        """
        # 创建距离矩阵
        positions = torch.arange(seq_len, device=self.slopes.device)
        distances = positions[:, None] - positions[None, :]
        
        # 应用斜率
        bias = distances[None, :, :] * self.slopes[:, None, None]
        
        return bias

class ConvStemSE(nn.Module):
    """Conv/TCN + SE 前端模块
    
    三路Conv1D(k=3/5/7) + 小TCN段(dilation=1,2) + SE + 映射到d_model
    """
    
    def __init__(self, d_model: int = 256, conv_channels: int = 128):
        super().__init__()
        
        # 三路不同kernel size的卷积
        self.conv_branches = nn.ModuleList()
        kernel_sizes = [3, 5, 7]
        
        for k in kernel_sizes:
            branch = nn.Sequential(
                # 初始卷积
                nn.Conv1d(1, conv_channels, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(conv_channels),
                nn.GELU(),
                
                # 小TCN段：两层膨胀卷积
                nn.Conv1d(conv_channels, conv_channels, kernel_size=3, 
                         padding=1, dilation=1),
                nn.BatchNorm1d(conv_channels),
                nn.GELU(),
                
                nn.Conv1d(conv_channels, conv_channels, kernel_size=3, 
                         padding=2, dilation=2),
                nn.BatchNorm1d(conv_channels),
                nn.GELU(),
            )
            self.conv_branches.append(branch)
        
        # SE模块
        total_channels = conv_channels * len(kernel_sizes)
        self.se = SqueezeExcitation(total_channels)
        
        # 1x1卷积映射到d_model
        self.projection = nn.Conv1d(total_channels, d_model, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: [B, S, 1] 输入序列
            
        Returns:
            [B, S, d_model] 特征表示
        """
        # 转换为卷积格式 [B, 1, S]
        x = x.transpose(1, 2)
        
        # 三路卷积
        branch_outputs = []
        for branch in self.conv_branches:
            out = branch(x)
            branch_outputs.append(out)
        
        # 拼接特征
        x = torch.cat(branch_outputs, dim=1)  # [B, total_channels, S]
        
        # SE注意力
        x = self.se(x)
        
        # 投影到d_model
        x = self.projection(x)  # [B, d_model, S]
        
        # 转回序列格式 [B, S, d_model]
        x = x.transpose(1, 2)
        
        return x


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation模块"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: [B, C, S] 输入特征
            
        Returns:
            [B, C, S] 重标定后的特征
        """
        # 全局平均池化
        b, c, s = x.size()
        y = self.global_pool(x).view(b, c)  # [B, C]
        
        # FC层计算通道权重
        y = self.fc(y).view(b, c, 1)  # [B, C, 1]
        
        # 重标定
        return x * y.expand_as(x)

class PreLNTransformerLayer(nn.Module):
    """Pre-LN Transformer层
    
    使用Pre-LN结构提升训练稳定性，特别适合深层网络
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, 
                 dropout: float = 0.1, attention_dropout: float = 0.1,
                 stochastic_depth: float = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.stochastic_depth = stochastic_depth
        
        # Pre-LN结构
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attention_dropout, batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播
        
        Args:
            x: [B, S, d_model] 输入序列
            attn_bias: [n_heads, S, S] ALiBi偏置矩阵
            
        Returns:
            [B, S, d_model] 输出序列
        """
        # Stochastic Depth
        if self.training and self.stochastic_depth > 0:
            if torch.rand(1).item() < self.stochastic_depth:
                return x
        
        # Pre-LN + 自注意力
        norm_x = self.norm1(x)
        
        # 如果有ALiBi偏置，需要手动实现注意力
        if attn_bias is not None:
            attn_out = self._attention_with_bias(norm_x, attn_bias)
        else:
            attn_out, _ = self.self_attn(norm_x, norm_x, norm_x)
        
        x = x + self.dropout(attn_out)
        
        # Pre-LN + FFN
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        
        return x
    
    def _attention_with_bias(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """带ALiBi偏置的注意力计算"""
        B, S, d = x.size()
        
        # 计算Q, K, V
        qkv = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
        qkv = qkv.view(B, S, 3, self.n_heads, d // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, S, d_k]
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d // self.n_heads)
        
        # 添加ALiBi偏置
        scores = scores + bias.unsqueeze(0)  # [B, n_heads, S, S]
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.self_attn.dropout, training=self.training)
        
        # 应用注意力
        out = torch.matmul(attn_weights, v)  # [B, n_heads, S, d_k]
        out = out.transpose(1, 2).contiguous().view(B, S, d)  # [B, S, d]
        
        # 输出投影
        out = F.linear(out, self.self_attn.out_proj.weight, self.self_attn.out_proj.bias)
        
        return out


class PreLNTransformer(nn.Module):
    """Pre-LN Transformer主干网络"""
    
    def __init__(self, d_model: int = 256, n_layers: int = 4, n_heads: int = 8,
                 d_ff: int = 1024, dropout: float = 0.1, attention_dropout: float = 0.1,
                 stochastic_depth: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        
        # ALiBi位置偏置
        self.alibi = ALiBiPositionalBias(n_heads)
        
        # Transformer层
        self.layers = nn.ModuleList([
            PreLNTransformerLayer(
                d_model=d_model,
                n_heads=n_heads, 
                d_ff=d_ff,
                dropout=dropout,
                attention_dropout=attention_dropout,
                stochastic_depth=stochastic_depth * (i / n_layers)  # 线性增长
            )
            for i in range(n_layers)
        ])
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: [B, S, d_model] 输入序列
            
        Returns:
            [B, S, d_model] 输出序列
        """
        seq_len = x.size(1)
        
        # 生成ALiBi偏置
        attn_bias = self.alibi(seq_len)  # [n_heads, S, S]
        
        # 通过Transformer层
        for layer in self.layers:
            x = layer(x, attn_bias)
        
        # 最终归一化
        x = self.final_norm(x)
        
        return x

class GatedFusion(nn.Module):
    """门控融合机制
    
    融合Conv特征和Transformer特征
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        
        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # 特征变换
        self.conv_transform = nn.Linear(d_model, d_model)
        self.transformer_transform = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, d_model)
        
    def forward(self, conv_features: torch.Tensor, 
                transformer_features: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            conv_features: [B, S, d_model] Conv特征
            transformer_features: [B, S, d_model] Transformer特征
            
        Returns:
            [B, S, d_model] 融合后的特征
        """
        # 特征变换
        conv_feat = self.conv_transform(conv_features)
        trans_feat = self.transformer_transform(transformer_features)
        
        # 计算门控权重
        concat_feat = torch.cat([conv_feat, trans_feat], dim=-1)
        gate_weight = self.gate(concat_feat)  # [B, S, d_model]
        
        # 门控融合
        fused = gate_weight * conv_feat + (1 - gate_weight) * trans_feat
        
        # 输出投影
        output = self.output_proj(fused)
        
        return output


class ImprovedPositionalEncoding(nn.Module):
    """改进的位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class EnhancedTransformerNILMModel(nn.Module):
    """增强的Transformer NILM模型"""
    
    def __init__(self, input_dim, d_model=256, n_heads=8, n_layers=6, 
                 d_ff=1024, dropout=0.1, window_size=64, num_appliances=5):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.window_size = window_size
        self.num_appliances = num_appliances
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Conv/TCN + SE 前端特征提取
        self.conv_stem = ConvStemSE(d_model)
        
        # 位置编码
        self.pos_encoding = ImprovedPositionalEncoding(d_model, dropout=dropout)
        
        # Pre-LN Transformer主干
        self.transformer = PreLNTransformer(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # 门控融合模块
        self.gated_fusion = GatedFusion(d_model)
        
        # 任务头
        self.heads = TaskHeads(d_model, num_appliances)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.size()
        
        # Conv/TCN + SE 前端特征提取（使用原始输入）
        conv_features = self.conv_stem(x)
        
        # 输入嵌入用于Transformer分支
        embedded_x = self.input_embedding(x)  # [batch_size, seq_len, d_model]
        
        # 位置编码
        transformer_input = self.pos_encoding(embedded_x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer处理
        transformer_features = self.transformer(transformer_input)
        
        # 门控融合Conv和Transformer特征
        x = self.gated_fusion(conv_features, transformer_features)
        
        # 任务头输出
        outputs = self.heads(x)
        
        return outputs

class TaskHeads(nn.Module):
    """多任务头模块"""
    
    def __init__(self, d_model: int, num_appliances: int):
        super().__init__()
        
        # 功率回归头
        self.power_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_appliances)
        )
        
        # 状态分类头
        self.state_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_appliances),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            x: [B, S, d_model] 输入特征
            
        Returns:
            Dict包含power和state预测
        """
        power_output = self.power_head(x)  # [B, S, num_appliances]
        state_output = self.state_head(x)  # [B, S, num_appliances]
        
        # 只取最后一个时间步的预测
        power_output = power_output[:, -1, :]  # [B, num_appliances]
        state_output = state_output[:, -1, :]  # [B, num_appliances]
        
        return {
            'power': power_output,
            'state': state_output
        }


class UncertaintyWeightedLoss(nn.Module):
    """不确定性加权损失函数
    
    基于"Multi-Task Learning Using Uncertainty to Weigh Losses"论文
    自动学习任务权重，平衡多任务学习
    """
    
    def __init__(self, num_appliances: int):
        super().__init__()
        
        # 可学习的不确定性参数（对数方差）
        self.log_var_power = nn.Parameter(torch.zeros(1))
        self.log_var_state = nn.Parameter(torch.zeros(1))
        self.log_var_corr = nn.Parameter(torch.zeros(1))
        
        # 基础损失函数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算不确定性加权损失
        
        Args:
            predictions: 模型预测结果
            targets: 真实标签
            
        Returns:
            损失字典
        """
        pred_power = predictions['power']
        pred_state = predictions['state']
        target_power = targets['power']
        target_state = targets['state']
        
        # 计算基础损失
        power_loss = self.mse_loss(pred_power, target_power)
        state_loss = self.bce_loss(pred_state, target_state)
        corr_loss = self._correlation_loss(pred_power, target_power)
        
        # 不确定性加权
        # L = 1/(2*σ²) * loss + log(σ)
        precision_power = torch.exp(-self.log_var_power)
        precision_state = torch.exp(-self.log_var_state)
        precision_corr = torch.exp(-self.log_var_corr)
        
        weighted_power_loss = precision_power * power_loss + self.log_var_power
        weighted_state_loss = precision_state * state_loss + self.log_var_state
        weighted_corr_loss = precision_corr * corr_loss + self.log_var_corr
        
        # 总损失
        total_loss = weighted_power_loss + weighted_state_loss + weighted_corr_loss
        total_loss = total_loss.squeeze()  # 确保是标量
        
        return {
            'total_loss': total_loss,
            'power_loss': power_loss,
            'state_loss': state_loss,
            'corr_loss': corr_loss,
            'weighted_power_loss': weighted_power_loss,
            'weighted_state_loss': weighted_state_loss,
            'weighted_corr_loss': weighted_corr_loss,
            'uncertainties': {
                'power_uncertainty': torch.exp(self.log_var_power),
                'state_uncertainty': torch.exp(self.log_var_state),
                'corr_uncertainty': torch.exp(self.log_var_corr)
            }
        }
    
    def _correlation_loss(self, pred_power: torch.Tensor, 
                         target_power: torch.Tensor) -> torch.Tensor:
        """计算相关性损失"""
        batch_size, num_appliances = pred_power.shape
        
        corr_losses = []
        for i in range(num_appliances):
            pred_i = pred_power[:, i]  # [B]
            target_i = target_power[:, i]  # [B]
            
            # 计算皮尔逊相关系数
            pred_mean = torch.mean(pred_i)
            target_mean = torch.mean(target_i)
            
            pred_centered = pred_i - pred_mean
            target_centered = target_i - target_mean
            
            numerator = torch.sum(pred_centered * target_centered)
            denominator = torch.sqrt(torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2))
            
            correlation = numerator / (denominator + 1e-8)
            corr_loss = 1 - correlation  # 1 - 相关系数作为损失
            corr_losses.append(corr_loss)
        
        return torch.mean(torch.stack(corr_losses))

def create_enhanced_model(input_dim=1, num_appliances=5, **kwargs):
    """创建增强的模型"""
    model_params = {
        'input_dim': input_dim,
        'd_model': kwargs.get('d_model', 256),
        'n_heads': kwargs.get('n_heads', 8),
        'n_layers': kwargs.get('n_layers', 6),
        'd_ff': kwargs.get('d_ff', 1024),
        'dropout': kwargs.get('dropout', 0.1),
        'window_size': kwargs.get('window_size', 64),
        'num_appliances': num_appliances
    }
    
    model = EnhancedTransformerNILMModel(**model_params)
    loss_fn = UncertaintyWeightedLoss(num_appliances)
    
    return model, loss_fn

if __name__ == '__main__':
    # 测试模型
    model, loss_fn = create_enhanced_model()
    
    # 创建测试数据
    batch_size, seq_len, input_dim = 4, 128, 1
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # 前向传播
    outputs = model(x)
    
    print(f"模型输出形状:")
    print(f"  功率: {outputs['power'].shape}")
    print(f"  状态: {outputs['state'].shape}")
    
    # 测试损失函数
    targets = {
        'power': torch.randn_like(outputs['power']),
        'state': torch.randint(0, 2, outputs['state'].shape).float()
    }
    
    loss_dict = loss_fn(outputs, targets)
    print(f"\n损失:")
    print(f"  总损失: {loss_dict['total_loss'].item():.4f}")
    print(f"  功率损失: {loss_dict['power_loss'].item():.4f}")
    print(f"  状态损失: {loss_dict['state_loss'].item():.4f}")
    print(f"  相关性损失: {loss_dict['corr_loss'].item():.4f}")