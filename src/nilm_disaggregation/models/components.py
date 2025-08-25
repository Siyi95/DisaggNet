"""模型组件模块"""

import torch
import torch.nn as nn
import numpy as np
from typing import List


class EnhancedMultiScaleConvBlock(nn.Module):
    """增强多尺度卷积块"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: List[int] = [3, 5, 7]):
        super().__init__()
        # 确保每个卷积的输出通道数能被整除
        channels_per_conv = out_channels // len(kernel_sizes)
        remaining_channels = out_channels - channels_per_conv * (len(kernel_sizes) - 1)
        
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            # 最后一个卷积处理剩余的通道
            out_ch = remaining_channels if i == len(kernel_sizes) - 1 else channels_per_conv
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels, out_ch, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(out_ch),
                nn.GELU()
            ))
        
        self.fusion = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_outputs = [conv(x) for conv in self.convs]
        fused = torch.cat(conv_outputs, dim=1)
        output = self.fusion(fused)
        return self.dropout(output)


class ChannelAttention(nn.Module):
    """通道注意力机制"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * self.sigmoid(out).view(b, c, 1)


class EnhancedLocalWindowAttention(nn.Module):
    """增强局部窗口注意力机制"""
    
    def __init__(self, d_model: int, n_heads: int = 8, window_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        # 相对位置编码
        self.relative_position_bias = nn.Parameter(
            torch.zeros(2 * window_size - 1, n_heads)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # 如果序列长度小于窗口大小，使用全局注意力
        if L <= self.window_size:
            return self._global_attention(x)
        
        # 局部窗口注意力
        return self._windowed_attention(x)
    
    def _global_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)
    
    def _windowed_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # 分割成窗口
        num_windows = L // self.window_size
        x_windowed = x[:, :num_windows * self.window_size].view(
            B, num_windows, self.window_size, D
        )
        
        # 对每个窗口应用注意力
        outputs = []
        for i in range(num_windows):
            window = x_windowed[:, i]  # [B, window_size, D]
            window_out = self._global_attention(window)
            outputs.append(window_out)
        
        # 处理剩余部分
        if L % self.window_size != 0:
            remainder = x[:, num_windows * self.window_size:]
            remainder_out = self._global_attention(remainder)
            outputs.append(remainder_out)
        
        return torch.cat(outputs, dim=1)


class EnhancedTransformerBlock(nn.Module):
    """增强Transformer块"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, window_size: int = 64):
        super().__init__()
        self.attention = EnhancedLocalWindowAttention(d_model, n_heads, window_size, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # CNN分支
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(d_model, d_ff, 1),
            nn.BatchNorm1d(d_ff),
            nn.GELU(),
            nn.Conv1d(d_ff, d_model, 1),
            nn.Dropout(dropout)
        )
        
        # LSTM分支
        self.lstm_branch = nn.LSTM(
            d_model, d_model // 2, batch_first=True, 
            bidirectional=True, dropout=dropout if dropout > 0 else 0
        )
        
        # 分支融合
        self.branch_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多头注意力 + 残差连接
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        
        # CNN分支
        cnn_input = x.transpose(1, 2)  # [B, D, L]
        cnn_out = self.cnn_branch(cnn_input).transpose(1, 2)  # [B, L, D]
        
        # LSTM分支
        lstm_out, _ = self.lstm_branch(x)
        
        # 分支融合
        fused = torch.cat([cnn_out, lstm_out], dim=-1)
        fused_out = self.branch_fusion(fused)
        x = self.norm2(x + fused_out)
        
        # 前馈网络 + 残差连接
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        return x


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1), :].transpose(0, 1)