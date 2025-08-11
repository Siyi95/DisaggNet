import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict

class ChannelMixer(nn.Module):
    """通道嵌入模块，包含SE注意力和时间特征融合"""
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int,
                 reduction_ratio: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 1x1卷积进行通道映射
        self.channel_projection = nn.Conv1d(input_dim, d_model, kernel_size=1)
        
        # SE注意力模块
        self.se_attention = SEBlock(d_model, reduction_ratio)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            [batch_size, seq_len, d_model]
        """
        # 转换为卷积格式 [batch, channels, seq_len]
        x = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        
        # 通道投影
        x = self.channel_projection(x)  # [batch, d_model, seq_len]
        
        # SE注意力
        x = self.se_attention(x)
        
        # 转回原格式
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]
        
        # 层归一化和dropout
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        return x

class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力块"""
    
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        
        reduced_channels = max(1, channels // reduction_ratio)
        
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len]
        Returns:
            [batch, channels, seq_len]
        """
        batch_size, channels, seq_len = x.size()
        
        # Squeeze: 全局平均池化
        squeeze = self.squeeze(x).view(batch_size, channels)  # [batch, channels]
        
        # Excitation: 生成通道权重
        excitation = self.excitation(squeeze).view(batch_size, channels, 1)  # [batch, channels, 1]
        
        # 应用注意力权重
        return x * excitation

class PositionalEncoding(nn.Module):
    """位置编码模块，支持可学习和正余弦编码"""
    
    def __init__(self, 
                 d_model: int, 
                 max_len: int = 5000,
                 learnable: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.learnable = learnable
        self.dropout = nn.Dropout(dropout)
        
        if learnable:
            # 可学习位置编码
            self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        else:
            # 正余弦位置编码
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                               (-math.log(10000.0) / d_model))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        
        if self.learnable:
            pos_enc = self.pos_embedding[:, :seq_len, :]
        else:
            pos_enc = self.pe[:, :seq_len, :]
            
        x = x + pos_enc
        return self.dropout(x)

class TimeFeatureEmbedding(nn.Module):
    """时间特征嵌入模块"""
    
    def __init__(self, d_model: int):
        super().__init__()
        
        self.d_model = d_model
        
        # 小时嵌入 (0-23)
        self.hour_embedding = nn.Embedding(24, d_model // 4)
        
        # 星期嵌入 (0-6)
        self.weekday_embedding = nn.Embedding(7, d_model // 4)
        
        # 月份嵌入 (0-11)
        self.month_embedding = nn.Embedding(12, d_model // 4)
        
        # 季节嵌入 (0-3)
        self.season_embedding = nn.Embedding(4, d_model // 4)
        
        # 融合层
        self.fusion = nn.Linear(d_model, d_model)
        
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: [batch_size, seq_len] 时间戳（分钟级）
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = timestamps.shape
        
        # 提取时间特征
        hours = (timestamps // 60) % 24  # 小时
        weekdays = (timestamps // (60 * 24)) % 7  # 星期
        months = (timestamps // (60 * 24 * 30)) % 12  # 月份（近似）
        seasons = months // 3  # 季节
        
        # 嵌入
        hour_emb = self.hour_embedding(hours.long())  # [batch, seq_len, d_model//4]
        weekday_emb = self.weekday_embedding(weekdays.long())
        month_emb = self.month_embedding(months.long())
        season_emb = self.season_embedding(seasons.long())
        
        # 拼接时间特征
        time_emb = torch.cat([hour_emb, weekday_emb, month_emb, season_emb], dim=-1)
        
        # 融合
        time_emb = self.fusion(time_emb)
        
        return time_emb

class FeatureExtractor(nn.Module):
    """完整的特征提取器，整合所有特征处理模块"""
    
    def __init__(self,
                 input_dim: int,
                 d_model: int,
                 max_len: int = 5000,
                 use_time_features: bool = True,
                 learnable_pos: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_time_features = use_time_features
        
        # 通道混合器
        self.channel_mixer = ChannelMixer(input_dim, d_model, dropout=dropout)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(
            d_model, max_len, learnable=learnable_pos, dropout=dropout
        )
        
        # 时间特征嵌入（可选）
        if use_time_features:
            self.time_embedding = TimeFeatureEmbedding(d_model)
            self.time_fusion = nn.Linear(d_model * 2, d_model)
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, 
                x: torch.Tensor, 
                timestamps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim] 输入特征
            timestamps: [batch_size, seq_len] 时间戳（可选）
        Returns:
            [batch_size, seq_len, d_model] 提取的特征
        """
        # 通道嵌入
        x = self.channel_mixer(x)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # 时间特征融合
        if self.use_time_features and timestamps is not None:
            time_emb = self.time_embedding(timestamps)
            x = torch.cat([x, time_emb], dim=-1)
            x = self.time_fusion(x)
        
        # 最终归一化
        x = self.final_norm(x)
        
        return x
    
    def extract_single_sample(self, 
                             channels: Dict[str, float], 
                             timestamp: float) -> np.ndarray:
        """
        从单个样本提取特征（用于在线推理）
        
        Args:
            channels: 通道数据字典 {channel_name: value}
            timestamp: 时间戳
        Returns:
            特征向量
        """
        import numpy as np
        
        # 构建基础特征向量
        features = []
        
        # 基础通道（按预定义顺序）
        base_channels = ['P_total', 'Q_total', 'S_total', 'I', 'V', 'PF']
        for channel in base_channels:
            features.append(channels.get(channel, 0.0))
        
        # 派生特征（简化版，用于在线处理）
        # 注意：完整的派生特征需要历史数据，这里提供基础实现
        
        # 一阶差分（需要历史数据，这里设为0）
        features.append(0.0)  # delta_P
        
        # 时间特征
        from datetime import datetime
        dt = datetime.fromtimestamp(timestamp)
        
        # 小时特征（0-23）
        hour = dt.hour / 23.0
        features.append(hour)
        
        # 星期特征（0-6）
        weekday = dt.weekday() / 6.0
        features.append(weekday)
        
        # 月份特征（1-12）
        month = (dt.month - 1) / 11.0
        features.append(month)
        
        # 季节特征（0-3）
        season = ((dt.month - 1) // 3) / 3.0
        features.append(season)
        
        return np.array(features, dtype=np.float32)

class MultiScaleConv1D(nn.Module):
    """多尺度1D卷积模块"""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: list = [3, 5, 7, 9],
                 dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels // len(kernel_sizes), 
                         kernel_size=kernel_size, 
                         padding=(kernel_size - 1) // 2 * dilation,
                         dilation=dilation),
                nn.BatchNorm1d(out_channels // len(kernel_sizes)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            self.convs.append(conv)
            
        self.fusion = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len]
        Returns:
            [batch, out_channels, seq_len]
        """
        conv_outputs = []
        
        for conv in self.convs:
            conv_outputs.append(conv(x))
            
        # 拼接多尺度特征
        x = torch.cat(conv_outputs, dim=1)
        
        # 融合
        x = self.fusion(x)
        
        return x

class DepthwiseConv1D(nn.Module):
    """深度可分离1D卷积"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 5,
                 dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        
        padding = (kernel_size - 1) // 2 * dilation
        
        # 深度卷积
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, 
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels
        )
        
        # 点卷积
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        # 归一化和激活
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, seq_len]
        Returns:
            [batch, out_channels, seq_len]
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x