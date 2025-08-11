import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
import numpy as np

from ..features import FeatureExtractor, DepthwiseConv1D, MultiScaleConv1D

class LocalWindowAttention(nn.Module):
    """局部窗口注意力模块"""
    
    def __init__(self,
                 d_model: int,
                 num_heads: int = 8,
                 window_size: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 相对位置编码
        self.relative_position_bias = nn.Parameter(
            torch.zeros(2 * window_size - 1, num_heads)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] 可选掩码
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 如果序列长度小于窗口大小，使用全局注意力
        if seq_len <= self.window_size:
            return self._global_attention(x, mask)
        
        # 分窗口处理
        num_windows = (seq_len + self.window_size - 1) // self.window_size
        pad_len = num_windows * self.window_size - seq_len
        
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=False)
        
        # 重塑为窗口格式
        x_windows = x.view(batch_size, num_windows, self.window_size, d_model)
        
        # 对每个窗口应用注意力
        output_windows = []
        for i in range(num_windows):
            window = x_windows[:, i]  # [batch_size, window_size, d_model]
            window_mask = mask[:, i*self.window_size:(i+1)*self.window_size] if mask is not None else None
            
            window_output = self._window_attention(window, window_mask)
            output_windows.append(window_output)
        
        # 合并窗口
        output = torch.stack(output_windows, dim=1)  # [batch_size, num_windows, window_size, d_model]
        output = output.view(batch_size, -1, d_model)
        
        # 移除填充
        if pad_len > 0:
            output = output[:, :seq_len]
        
        return output
    
    def _window_attention(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """窗口内注意力计算"""
        batch_size, window_size, d_model = x.shape
        
        # QKV计算
        qkv = self.qkv(x).reshape(batch_size, window_size, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, window_size, head_dim]
        
        # 注意力分数
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # 相对位置偏置
        relative_position_bias = self._get_relative_position_bias(window_size)
        attn = attn + relative_position_bias
        
        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, window_size]
            attn = attn.masked_fill(~mask, float('-inf'))
        
        # Softmax和dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力权重
        out = torch.matmul(attn, v)  # [batch_size, num_heads, window_size, head_dim]
        out = out.transpose(1, 2).reshape(batch_size, window_size, d_model)
        
        # 输出投影
        out = self.proj(out)
        
        return out
    
    def _global_attention(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """全局注意力（用于短序列）"""
        batch_size, seq_len, d_model = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        out = self.proj(out)
        
        return out
    
    def _get_relative_position_bias(self, window_size: int) -> torch.Tensor:
        """获取相对位置偏置"""
        coords = torch.arange(window_size, device=self.relative_position_bias.device)
        relative_coords = coords[:, None] - coords[None, :]  # [window_size, window_size]
        relative_coords += window_size - 1  # 偏移到正数
        
        relative_position_bias = self.relative_position_bias[relative_coords]  # [window_size, window_size, num_heads]
        return relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, window_size, window_size]

class SparseGlobalAttention(nn.Module):
    """稀疏全局注意力模块"""
    
    def __init__(self,
                 d_model: int,
                 num_heads: int = 8,
                 sparsity_factor: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.sparsity_factor = sparsity_factor
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] 可选掩码
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # 降采样以减少计算量
        if seq_len > self.sparsity_factor * 32:
            # 使用步长采样
            stride = max(1, seq_len // (self.sparsity_factor * 32))
            sparse_indices = torch.arange(0, seq_len, stride, device=x.device)
            
            x_sparse = x[:, sparse_indices]  # [batch_size, sparse_len, d_model]
            
            # 计算稀疏注意力
            sparse_output = self._compute_attention(x_sparse)
            
            # 插值回原始长度
            output = F.interpolate(
                sparse_output.transpose(1, 2),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            # 直接计算全局注意力
            output = self._compute_attention(x, mask)
        
        return output
    
    def _compute_attention(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算注意力"""
        batch_size, seq_len, d_model = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        out = self.proj(out)
        
        return out

class LocalBranch(nn.Module):
    """局部分支：局部注意力 + 深度卷积"""
    
    def __init__(self,
                 d_model: int,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 window_size: int = 32,
                 conv_kernel_size: int = 7,
                 dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attention': LocalWindowAttention(d_model, num_heads, window_size, dropout),
                'conv': DepthwiseConv1D(d_model, d_model, conv_kernel_size, dropout=dropout),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout)
                )
            })
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] 可选掩码
        Returns:
            [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            # 局部注意力
            attn_out = layer['attention'](x, mask)
            x = layer['norm1'](x + attn_out)
            
            # 深度卷积（需要转换维度）
            x_conv = x.transpose(1, 2)  # [batch, d_model, seq_len]
            conv_out = layer['conv'](x_conv)
            conv_out = conv_out.transpose(1, 2)  # [batch, seq_len, d_model]
            x = layer['norm2'](x + conv_out)
            
            # FFN
            ffn_out = layer['ffn'](x)
            x = x + ffn_out
        
        return x

class GlobalBranch(nn.Module):
    """全局分支：稀疏全局注意力"""
    
    def __init__(self,
                 d_model: int,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 sparsity_factor: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attention': SparseGlobalAttention(d_model, num_heads, sparsity_factor, dropout),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                    nn.Dropout(dropout)
                )
            })
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] 可选掩码
        Returns:
            [batch_size, seq_len, d_model]
        """
        for layer in self.layers:
            # 稀疏全局注意力
            attn_out = layer['attention'](x, mask)
            x = layer['norm1'](x + attn_out)
            
            # FFN
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        
        return x

class BranchFusion(nn.Module):
    """分支融合模块"""
    
    def __init__(self, d_model: int, fusion_type: str = 'weighted_sum'):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == 'weighted_sum':
            self.weight_local = nn.Parameter(torch.tensor(0.5))
            self.weight_global = nn.Parameter(torch.tensor(0.5))
        elif fusion_type == 'concat':
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        elif fusion_type == 'attention':
            self.attention_weights = nn.Linear(d_model * 2, 2)
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, local_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_features: [batch_size, seq_len, d_model]
            global_features: [batch_size, seq_len, d_model]
        Returns:
            [batch_size, seq_len, d_model]
        """
        if self.fusion_type == 'weighted_sum':
            # 加权求和
            weights = F.softmax(torch.stack([self.weight_local, self.weight_global]), dim=0)
            fused = weights[0] * local_features + weights[1] * global_features
        
        elif self.fusion_type == 'concat':
            # 拼接后投影
            concat_features = torch.cat([local_features, global_features], dim=-1)
            fused = self.fusion_proj(concat_features)
        
        elif self.fusion_type == 'attention':
            # 注意力加权
            concat_features = torch.cat([local_features, global_features], dim=-1)
            attention_weights = F.softmax(self.attention_weights(concat_features), dim=-1)
            
            fused = (attention_weights[:, :, 0:1] * local_features + 
                    attention_weights[:, :, 1:2] * global_features)
        
        else:
            # 默认简单相加
            fused = local_features + global_features
        
        return self.norm(fused)

class MSCAT(nn.Module):
    """Multi-Scale Channel-Aware Transformer主模型"""
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 192,
                 num_heads: int = 6,
                 local_layers: int = 4,
                 global_layers: int = 3,
                 window_size: int = 32,
                 sparsity_factor: int = 4,
                 conv_kernel_size: int = 7,
                 fusion_type: str = 'weighted_sum',
                 use_time_features: bool = True,
                 learnable_pos: bool = False,
                 dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            d_model=d_model,
            max_len=max_len,
            use_time_features=use_time_features,
            learnable_pos=learnable_pos,
            dropout=dropout
        )
        
        # 局部分支
        self.local_branch = LocalBranch(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=local_layers,
            window_size=window_size,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout
        )
        
        # 全局分支
        self.global_branch = GlobalBranch(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=global_layers,
            sparsity_factor=sparsity_factor,
            dropout=dropout
        )
        
        # 分支融合
        self.branch_fusion = BranchFusion(d_model, fusion_type)
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, 
                x: torch.Tensor, 
                timestamps: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim] 输入特征
            timestamps: [batch_size, seq_len] 时间戳（可选）
            mask: [batch_size, seq_len] 掩码（可选）
        Returns:
            [batch_size, seq_len, d_model] 编码后的特征
        """
        # 特征提取
        x = self.feature_extractor(x, timestamps)
        
        # 双分支处理
        local_features = self.local_branch(x, mask)
        global_features = self.global_branch(x, mask)
        
        # 分支融合
        fused_features = self.branch_fusion(local_features, global_features)
        
        # 最终归一化
        output = self.final_norm(fused_features)
        
        return output
    
    def get_attention_maps(self, 
                          x: torch.Tensor, 
                          timestamps: Optional[torch.Tensor] = None,
                          mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """获取注意力图用于可视化"""
        # 这里可以添加注意力图提取逻辑
        # 为简化，暂时返回空字典
        return {}