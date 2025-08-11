import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from collections import deque

class CausalConv1d(nn.Module):
    """因果1D卷积，确保只使用过去信息"""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            groups=groups,
            bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_channels, seq_len]
        Returns:
            [batch_size, out_channels, seq_len]
        """
        # 应用卷积
        out = self.conv(x)
        
        # 移除未来信息（右侧填充）
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        
        return out

class TemporalBlock(nn.Module):
    """TCN的基本时间块"""
    
    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 kernel_size: int,
                 dilation: int,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        
        # 第一个因果卷积
        self.conv1 = CausalConv1d(
            in_channels=n_inputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            dilation=dilation
        )
        
        # 第二个因果卷积
        self.conv2 = CausalConv1d(
            in_channels=n_outputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            dilation=dilation
        )
        
        # 归一化层
        self.norm1 = nn.BatchNorm1d(n_outputs)
        self.norm2 = nn.BatchNorm1d(n_outputs)
        
        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'glu':
            # GLU需要输出通道数为偶数
            assert n_outputs % 2 == 0, "GLU requires even number of output channels"
            self.activation = nn.GLU(dim=1)
            # 调整第二个卷积的输出通道数
            self.conv2 = CausalConv1d(
                in_channels=n_outputs,
                out_channels=n_outputs * 2,  # GLU会减半
                kernel_size=kernel_size,
                dilation=dilation
            )
        else:
            self.activation = nn.ReLU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接的投影层（如果输入输出维度不同）
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        # 权重初始化
        self.init_weights()
        
    def init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, n_inputs, seq_len]
        Returns:
            [batch_size, n_outputs, seq_len]
        """
        # 保存输入用于残差连接
        residual = x
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # 残差连接
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        # 确保维度匹配
        if residual.size() != out.size():
            # 如果序列长度不匹配，截断或填充
            min_len = min(residual.size(2), out.size(2))
            residual = residual[:, :, :min_len]
            out = out[:, :, :min_len]
        
        return out + residual

class CausalTCN(nn.Module):
    """因果时间卷积网络"""
    
    def __init__(self,
                 input_size: int,
                 num_channels: List[int],
                 kernel_size: int = 3,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        super().__init__()
        
        self.input_size = input_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                n_inputs=in_channels,
                n_outputs=out_channels,
                kernel_size=kernel_size,
                dilation=dilation_size,
                dropout=dropout,
                activation=activation
            ))
        
        self.network = nn.Sequential(*layers)
        
        # 计算感受野
        self.receptive_field = self._calculate_receptive_field()
        
    def _calculate_receptive_field(self) -> int:
        """计算感受野大小"""
        receptive_field = 1
        for i in range(len(self.num_channels)):
            dilation = 2 ** i
            receptive_field += (self.kernel_size - 1) * dilation
        return receptive_field
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_size, seq_len]
        Returns:
            [batch_size, num_channels[-1], seq_len]
        """
        return self.network(x)

class OnlineEventDetector(nn.Module):
    """在线事件检测器"""
    
    def __init__(self,
                 input_dim: int,
                 num_devices: int,
                 hidden_channels: List[int] = None,
                 kernel_size: int = 3,
                 dropout: float = 0.1,
                 activation: str = 'relu',
                 output_activation: str = 'sigmoid',
                 min_duration: int = 5):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_devices = num_devices
        self.min_duration = min_duration
        
        # 默认隐藏层配置
        if hidden_channels is None:
            hidden_channels = [64, 64, 32, 32, 16, 16]
        
        self.hidden_channels = hidden_channels
        
        # TCN骨干网络
        self.tcn = CausalTCN(
            input_size=input_dim,
            num_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            activation=activation
        )
        
        # 输出头
        self.output_head = nn.Sequential(
            nn.Conv1d(hidden_channels[-1], hidden_channels[-1] // 2, 1),
            nn.BatchNorm1d(hidden_channels[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels[-1] // 2, num_devices, 1)
        )
        
        # 输出激活函数
        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.output_activation = nn.Softmax(dim=1)
        else:
            self.output_activation = nn.Identity()
        
        # 状态缓冲区（用于最小持续时间过滤）
        self.state_buffers = [deque(maxlen=min_duration) for _ in range(num_devices)]
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, input_dim] 或 [batch_size, input_dim, seq_len]
        Returns:
            Dict containing:
                - 'logits': [batch_size, num_devices, seq_len]
                - 'probs': [batch_size, num_devices, seq_len]
        """
        # 确保输入格式正确
        if x.dim() == 3 and x.size(1) != self.input_dim:
            x = x.transpose(1, 2)  # [batch, seq_len, input_dim] -> [batch, input_dim, seq_len]
        
        # TCN前向传播
        tcn_out = self.tcn(x)  # [batch, hidden_channels[-1], seq_len]
        
        # 输出头
        logits = self.output_head(tcn_out)  # [batch, num_devices, seq_len]
        probs = self.output_activation(logits)
        
        return {
            'logits': logits,
            'probs': probs
        }
    
    def predict_online(self, 
                      x: torch.Tensor,
                      apply_duration_filter: bool = True) -> Dict[str, torch.Tensor]:
        """
        在线预测（单时间步）
        
        Args:
            x: [batch_size, input_dim] 单时间步输入
            apply_duration_filter: 是否应用持续时间过滤
        Returns:
            预测结果
        """
        # 添加时间维度
        x = x.unsqueeze(-1)  # [batch_size, input_dim, 1]
        
        # 前向传播
        with torch.no_grad():
            outputs = self.forward(x)
            
        # 提取最后时间步的结果
        logits = outputs['logits'][:, :, -1]  # [batch_size, num_devices]
        probs = outputs['probs'][:, :, -1]    # [batch_size, num_devices]
        
        # 阈值化得到状态
        states = (probs > 0.5).float()
        
        # 应用持续时间过滤（如果启用）
        if apply_duration_filter:
            states = self._apply_duration_filter(states)
        
        return {
            'logits': logits,
            'probs': probs,
            'states': states
        }
    
    def _apply_duration_filter(self, states: torch.Tensor) -> torch.Tensor:
        """
        应用最小持续时间过滤
        
        Args:
            states: [batch_size, num_devices] 当前状态
        Returns:
            [batch_size, num_devices] 过滤后的状态
        """
        batch_size, num_devices = states.shape
        filtered_states = states.clone()
        
        for batch_idx in range(batch_size):
            for device_idx in range(num_devices):
                current_state = states[batch_idx, device_idx].item()
                
                # 更新缓冲区
                self.state_buffers[device_idx].append(current_state)
                
                # 如果缓冲区未满，保持当前状态
                if len(self.state_buffers[device_idx]) < self.min_duration:
                    continue
                
                # 检查缓冲区中的状态一致性
                buffer_states = list(self.state_buffers[device_idx])
                
                # 如果所有状态都相同，则确认状态变化
                if all(s == current_state for s in buffer_states):
                    filtered_states[batch_idx, device_idx] = current_state
                else:
                    # 否则保持之前的状态
                    if len(buffer_states) > 1:
                        filtered_states[batch_idx, device_idx] = buffer_states[0]
        
        return filtered_states
    
    def reset_buffers(self):
        """重置状态缓冲区"""
        for buffer in self.state_buffers:
            buffer.clear()
    
    def get_receptive_field(self) -> int:
        """获取感受野大小"""
        return self.tcn.receptive_field

class OnlineBuffer:
    """在线推理的滚动缓冲区"""
    
    def __init__(self, 
                 buffer_size: int,
                 feature_dim: int,
                 device: str = 'cpu'):
        self.buffer_size = buffer_size
        self.feature_dim = feature_dim
        self.device = device
        
        # 初始化缓冲区
        self.buffer = torch.zeros(buffer_size, feature_dim, device=device)
        self.current_pos = 0
        self.is_full = False
        
    def append(self, features: torch.Tensor):
        """
        添加新特征到缓冲区
        
        Args:
            features: [feature_dim] 新特征
        """
        self.buffer[self.current_pos] = features.to(self.device)
        self.current_pos = (self.current_pos + 1) % self.buffer_size
        
        if self.current_pos == 0:
            self.is_full = True
    
    def get_sequence(self) -> torch.Tensor:
        """
        获取当前序列（按时间顺序）
        
        Returns:
            [seq_len, feature_dim] 当前序列
        """
        if not self.is_full:
            # 缓冲区未满，返回已有数据
            return self.buffer[:self.current_pos]
        else:
            # 缓冲区已满，按正确顺序返回
            return torch.cat([
                self.buffer[self.current_pos:],
                self.buffer[:self.current_pos]
            ], dim=0)
    
    def is_ready(self, min_length: int) -> bool:
        """检查是否有足够的数据进行预测"""
        current_length = self.buffer_size if self.is_full else self.current_pos
        return current_length >= min_length
    
    def reset(self):
        """重置缓冲区"""
        self.buffer.zero_()
        self.current_pos = 0
        self.is_full = False

class KnowledgeDistillationLoss(nn.Module):
    """知识蒸馏损失"""
    
    def __init__(self, 
                 temperature: float = 4.0,
                 alpha: float = 0.3,
                 hard_loss_fn: nn.Module = None):
        super().__init__()
        
        self.temperature = temperature
        self.alpha = alpha
        
        if hard_loss_fn is None:
            self.hard_loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.hard_loss_fn = hard_loss_fn
    
    def forward(self,
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            student_logits: [batch_size, num_devices, seq_len] 学生模型logits
            teacher_logits: [batch_size, num_devices, seq_len] 教师模型logits
            targets: [batch_size, num_devices, seq_len] 真实标签
        Returns:
            损失字典
        """
        # 硬标签损失
        hard_loss = self.hard_loss_fn(student_logits, targets)
        
        # 软标签损失（知识蒸馏）
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        soft_loss = F.kl_div(
            student_soft, teacher_soft, 
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 总损失
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        return {
            'total_loss': total_loss,
            'hard_loss': hard_loss,
            'soft_loss': soft_loss
        }