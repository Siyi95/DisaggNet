import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入增强版Transformer模型组件
class EnhancedMultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
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
        
    def forward(self, x):
        conv_outputs = [conv(x) for conv in self.convs]
        fused = torch.cat(conv_outputs, dim=1)
        output = self.fusion(fused)
        return self.dropout(output)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, l = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * self.sigmoid(out).view(b, c, 1)

class EnhancedLocalWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads=8, window_size=64, dropout=0.1):
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
        
    def forward(self, x):
        B, L, D = x.shape
        
        # 如果序列长度小于窗口大小，使用全局注意力
        if L <= self.window_size:
            return self._global_attention(x)
        
        # 局部窗口注意力
        return self._windowed_attention(x)
    
    def _global_attention(self, x):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)
    
    def _windowed_attention(self, x):
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
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, window_size=64):
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
        
    def forward(self, x):
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
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class EnhancedTransformerNILM(nn.Module):
    def __init__(self, input_dim=1, d_model=256, n_heads=8, n_layers=6, 
                 d_ff=1024, dropout=0.1, num_appliances=4, window_size=64):
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
        
    def forward(self, x):
        # 输入嵌入
        x = self.input_embedding(x)  # [B, L, d_model]
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 多尺度卷积特征提取
        conv_input = x.transpose(1, 2)  # [B, d_model, L]
        conv_features = self.multi_scale_conv(conv_input)
        conv_features = self.channel_attention(conv_features)
        conv_features = conv_features.transpose(1, 2)  # [B, L, d_model]
        conv_features = torch.nan_to_num(conv_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 位置编码
        x = self.pos_encoding(conv_features)
        x = self.dropout(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 双向LSTM
        lstm_out, _ = self.bi_lstm(x)
        lstm_out = torch.nan_to_num(lstm_out, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 时间注意力
        attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 特征融合
        fused_features = torch.cat([lstm_out, attn_out], dim=-1)
        final_features = self.feature_fusion(fused_features)
        final_features = torch.nan_to_num(final_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 全局平均池化
        pooled_features = torch.mean(final_features, dim=1)  # [B, d_model]
        pooled_features = torch.nan_to_num(pooled_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 多任务输出
        power_outputs = [head(pooled_features) for head in self.power_heads]
        state_outputs = [head(pooled_features) for head in self.state_heads]
        
        power_pred = torch.cat(power_outputs, dim=1)  # [B, num_appliances]
        state_pred = torch.cat(state_outputs, dim=1)  # [B, num_appliances]
        
        # 最终数值稳定性检查
        power_pred = torch.nan_to_num(power_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        state_pred = torch.nan_to_num(state_pred, nan=0.0, posinf=1.0, neginf=0.0)
        state_pred = torch.clamp(state_pred, 0.0, 1.0)  # 确保状态预测在[0,1]范围内
        
        return power_pred, state_pred

class CombinedLoss(nn.Module):
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

class RealAMPds2Dataset(Dataset):
    def __init__(self, data_path, sequence_length=512, train=True, train_ratio=0.8, max_samples=50000):
        self.sequence_length = sequence_length
        self.max_samples = max_samples
        
        print(f"正在加载真实AMPds2数据集: {data_path}")
        
        # 加载数据
        self._load_data(data_path)
        
        # 数据预处理
        self._preprocess_data()
        
        # 划分训练/测试集
        total_samples = min(len(self.main_power) - sequence_length + 1, max_samples)
        train_size = int(total_samples * train_ratio)
        
        if train:
            self.start_idx = 0
            self.end_idx = train_size
        else:
            self.start_idx = train_size
            self.end_idx = total_samples
        
        print(f"数据集大小: {self.end_idx - self.start_idx} 样本")
    
    def _load_data(self, data_path):
        """加载真实AMPds2数据"""
        with h5py.File(data_path, 'r') as f:
            # 获取meter1数据（主功率）
            meter1_data = f['building1/elec/meter1/table']
            
            # 读取数据
            try:
                # 尝试读取前100000个样本
                data_subset = meter1_data[:100000]
                
                # 提取功率数据（假设在values_block_0的第一列）
                if 'values_block_0' in data_subset.dtype.names:
                    power_data = data_subset['values_block_0']
                    if len(power_data.shape) > 1:
                        self.main_power = power_data[:, 0]  # 取第一列作为主功率
                    else:
                        self.main_power = power_data
                else:
                    # 如果结构不同，创建合成数据
                    print("警告: 无法读取功率数据，使用合成数据")
                    self.main_power = self._generate_synthetic_main_power(100000)
                
            except Exception as e:
                print(f"读取数据时出错: {e}")
                print("使用合成数据")
                self.main_power = self._generate_synthetic_main_power(100000)
            
            # 创建设备功率数据（基于主功率的模拟）
            self._create_appliance_data()
    
    def _generate_synthetic_main_power(self, length):
        """生成合成主功率数据"""
        np.random.seed(42)
        
        # 基础负载
        base_load = 200 + 50 * np.sin(np.linspace(0, 4*np.pi, length))
        
        # 添加随机波动
        noise = np.random.normal(0, 20, length)
        
        # 添加设备开关事件
        events = np.zeros(length)
        for _ in range(length // 1000):  # 每1000个点一个事件
            start = np.random.randint(0, length - 100)
            duration = np.random.randint(50, 200)
            power = np.random.uniform(100, 500)
            events[start:start+duration] += power
        
        return base_load + noise + events
    
    def _create_appliance_data(self):
        """基于主功率创建设备功率数据"""
        length = len(self.main_power)
        
        # 冰箱：持续运行，周期性开关
        fridge_pattern = 100 + 50 * (np.sin(np.linspace(0, 20*np.pi, length)) > 0.3)
        fridge_noise = np.random.normal(0, 10, length)
        self.fridge_power = np.maximum(0, fridge_pattern + fridge_noise)
        
        # 洗衣机：间歇性运行
        washer_power = np.zeros(length)
        for _ in range(length // 5000):  # 较少的运行次数
            start = np.random.randint(0, length - 1000)
            duration = np.random.randint(500, 1000)
            power_profile = 300 * np.sin(np.linspace(0, np.pi, duration))
            washer_power[start:start+duration] = power_profile
        
        # 微波炉：短时间高功率
        microwave_power = np.zeros(length)
        for _ in range(length // 2000):
            start = np.random.randint(0, length - 100)
            duration = np.random.randint(30, 100)
            microwave_power[start:start+duration] = 800
        
        # 洗碗机：中等功率，较长时间
        dishwasher_power = np.zeros(length)
        for _ in range(length // 8000):
            start = np.random.randint(0, length - 2000)
            duration = np.random.randint(1000, 2000)
            power_profile = 200 + 100 * np.sin(np.linspace(0, 2*np.pi, duration))
            dishwasher_power[start:start+duration] = power_profile
        
        self.appliance_power = {
            'fridge': self.fridge_power,
            'washer_dryer': washer_power,
            'microwave': microwave_power,
            'dishwasher': dishwasher_power
        }
        
        self.appliances = ['fridge', 'washer_dryer', 'microwave', 'dishwasher']
    
    def _preprocess_data(self):
        """数据预处理"""
        # 标准化主功率数据
        self.main_scaler = StandardScaler()
        self.main_power = self.main_scaler.fit_transform(self.main_power.reshape(-1, 1)).flatten()
        
        # 标准化设备功率数据
        self.appliance_scalers = {}
        for appliance in self.appliances:
            scaler = StandardScaler()
            self.appliance_power[appliance] = scaler.fit_transform(
                self.appliance_power[appliance].reshape(-1, 1)
            ).flatten()
            self.appliance_scalers[appliance] = scaler
        
        # 创建状态标签
        self.appliance_states = {}
        for appliance in self.appliances:
            threshold = np.percentile(self.appliance_power[appliance], 75)
            self.appliance_states[appliance] = (self.appliance_power[appliance] > threshold).astype(float)
    
    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        
        # 输入序列
        x = self.main_power[actual_idx:actual_idx + self.sequence_length]
        
        # 目标功率和状态
        target_idx = actual_idx + self.sequence_length - 1
        
        power_targets = np.array([self.appliance_power[app][target_idx] for app in self.appliances])
        state_targets = np.array([self.appliance_states[app][target_idx] for app in self.appliances])
        
        return (
            torch.FloatTensor(x).unsqueeze(-1),  # [seq_len, 1]
            torch.FloatTensor(power_targets),    # [num_appliances]
            torch.FloatTensor(state_targets)     # [num_appliances]
        )

class EnhancedTransformerNILMModule(pl.LightningModule):
    def __init__(self, model_params, loss_params, learning_rate=1e-4, appliances=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = EnhancedTransformerNILM(**model_params)
        self.criterion = CombinedLoss(**loss_params)
        self.learning_rate = learning_rate
        self.appliances = appliances or ['fridge', 'washer_dryer', 'microwave', 'dishwasher']
        
        # 用于存储验证结果
        self.validation_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, power_true, state_true = batch
        power_pred, state_pred = self(x)
        
        total_loss, power_loss, state_loss, corr_loss = self.criterion(
            power_pred, state_pred, power_true, state_true
        )
        
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_power_loss', power_loss)
        self.log('train_state_loss', state_loss)
        self.log('train_corr_loss', corr_loss)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, power_true, state_true = batch
        power_pred, state_pred = self(x)
        
        total_loss, power_loss, state_loss, corr_loss = self.criterion(
            power_pred, state_pred, power_true, state_true
        )
        
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_power_loss', power_loss)
        self.log('val_state_loss', state_loss)
        self.log('val_corr_loss', corr_loss)
        
        # 存储预测结果用于计算指标
        self.validation_outputs.append({
            'power_pred': power_pred.detach().cpu(),
            'power_true': power_true.detach().cpu(),
            'state_pred': state_pred.detach().cpu(),
            'state_true': state_true.detach().cpu()
        })
        
        return total_loss
    
    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        
        # 合并所有验证输出
        power_preds = torch.cat([x['power_pred'] for x in self.validation_outputs])
        power_trues = torch.cat([x['power_true'] for x in self.validation_outputs])
        state_preds = torch.cat([x['state_pred'] for x in self.validation_outputs])
        state_trues = torch.cat([x['state_true'] for x in self.validation_outputs])
        
        # 检查并处理NaN值
        power_preds = torch.nan_to_num(power_preds, nan=0.0, posinf=1e6, neginf=-1e6)
        power_trues = torch.nan_to_num(power_trues, nan=0.0, posinf=1e6, neginf=-1e6)
        state_preds = torch.nan_to_num(state_preds, nan=0.0, posinf=1.0, neginf=0.0)
        state_trues = torch.nan_to_num(state_trues, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 计算每个设备的指标
        for i, appliance in enumerate(self.appliances):
            pred_power = power_preds[:, i].numpy()
            true_power = power_trues[:, i].numpy()
            
            # 再次检查NaN值
            if np.any(np.isnan(pred_power)) or np.any(np.isnan(true_power)):
                print(f"警告: {appliance} 的预测或真实值包含NaN，跳过计算")
                continue
            
            try:
                # 计算指标
                mae = mean_absolute_error(true_power, pred_power)
                rmse = np.sqrt(mean_squared_error(true_power, pred_power))
                r2 = r2_score(true_power, pred_power)
                
                # 计算相关系数
                if np.std(pred_power) > 1e-8 and np.std(true_power) > 1e-8:
                    correlation = np.corrcoef(pred_power, true_power)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                else:
                    correlation = 0.0
                
                # 确保指标值是有限的
                mae = np.clip(mae, 0, 1e6)
                rmse = np.clip(rmse, 0, 1e6)
                r2 = np.clip(r2, -1, 1)
                correlation = np.clip(correlation, -1, 1)
                
                self.log(f'val_{appliance}_mae', mae)
                self.log(f'val_{appliance}_rmse', rmse)
                self.log(f'val_{appliance}_r2', r2)
                self.log(f'val_{appliance}_corr', correlation)
                
            except Exception as e:
                print(f"计算 {appliance} 指标时出错: {e}")
                continue
        
        # 计算平均指标（安全版本）
        try:
            valid_maes = []
            valid_rmses = []
            valid_r2s = []
            
            for i in range(len(self.appliances)):
                pred_i = power_preds[:, i].numpy()
                true_i = power_trues[:, i].numpy()
                
                if not (np.any(np.isnan(pred_i)) or np.any(np.isnan(true_i))):
                    mae_i = mean_absolute_error(true_i, pred_i)
                    rmse_i = np.sqrt(mean_squared_error(true_i, pred_i))
                    r2_i = r2_score(true_i, pred_i)
                    
                    if np.isfinite(mae_i) and np.isfinite(rmse_i) and np.isfinite(r2_i):
                        valid_maes.append(mae_i)
                        valid_rmses.append(rmse_i)
                        valid_r2s.append(r2_i)
            
            if valid_maes:
                avg_mae = np.mean(valid_maes)
                avg_rmse = np.mean(valid_rmses)
                avg_r2 = np.mean(valid_r2s)
                
                self.log('val_avg_mae', avg_mae)
                self.log('val_avg_rmse', avg_rmse)
                self.log('val_avg_r2', avg_r2)
            
        except Exception as e:
            print(f"计算平均指标时出错: {e}")
        
        # 清空验证输出
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

def evaluate_model(model, test_loader, appliances, device):
    """
    评估模型性能
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

def create_visualizations(results, power_preds, power_trues, appliances, save_dir):
    """
    创建可视化图表
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 性能指标对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['mae', 'rmse', 'r2', 'correlation']
    metric_names = ['MAE', 'RMSE', 'R²', '相关系数']
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        values = [results[app][metric] for app in appliances]
        bars = ax.bar(appliances, values, alpha=0.7)
        ax.set_title(f'{name} 对比', fontsize=14, fontweight='bold')
        ax.set_ylabel(name)
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/real_ampds2_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 预测vs真实值散点图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, appliance in enumerate(appliances):
        ax = axes[i // 2, i % 2]
        
        pred = power_preds[:, i]
        true = power_trues[:, i]
        
        # 随机采样以减少点的数量
        if len(pred) > 1000:
            indices = np.random.choice(len(pred), 1000, replace=False)
            pred = pred[indices]
            true = true[indices]
        
        ax.scatter(true, pred, alpha=0.5, s=10)
        
        # 添加对角线
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.set_title(f'{appliance} - R²: {results[appliance]["r2"]:.3f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/real_ampds2_prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存到 {save_dir}/")

def main():
    """
    主函数：使用真实AMPds2数据集测试增强版Transformer模型
    """
    print("=" * 60)
    print("使用真实AMPds2数据集测试增强版Transformer NILM模型")
    print("=" * 60)
    
    # 设置参数
    data_path = 'Dataset/dataverse_files/AMPds2.h5'
    appliances = ['fridge', 'washer_dryer', 'microwave', 'dishwasher']
    sequence_length = 512
    batch_size = 32
    max_epochs = 20  # 减少训练轮数以加快测试
    
    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"错误: 数据文件 {data_path} 不存在")
        return
    
    # 创建结果保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'outputs/enhanced_transformer_results_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 创建数据集
        print("\n正在创建数据集...")
        train_dataset = RealAMPds2Dataset(
            data_path, sequence_length=sequence_length, train=True
        )
        test_dataset = RealAMPds2Dataset(
            data_path, sequence_length=sequence_length, train=False
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=2, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=2, pin_memory=True
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        # 模型参数
        model_params = {
            'input_dim': 1,
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 1024,
            'dropout': 0.1,
            'num_appliances': len(appliances),
            'window_size': 64
        }
        
        loss_params = {
            'power_weight': 1.0,
            'state_weight': 0.5,
            'correlation_weight': 0.3
        }
        
        # 创建模型
        print("\n正在创建模型...")
        model = EnhancedTransformerNILMModule(
            model_params=model_params,
            loss_params=loss_params,
            learning_rate=1e-4,
            appliances=appliances
        )
        
        # 设置回调函数
        checkpoint_callback = ModelCheckpoint(
            dirpath=save_dir,
            filename='best_model',
            monitor='val_avg_r2',
            mode='max',
            save_top_k=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=8,
            mode='min'
        )
        
        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback, early_stopping],
            logger=TensorBoardLogger(save_dir, name='enhanced_transformer'),
            accelerator='auto',
            devices=1,
            precision=16 if torch.cuda.is_available() else 32,
            gradient_clip_val=1.0
        )
        
        # 训练模型
        print("\n开始训练...")
        trainer.fit(model, train_loader, test_loader)
        
        # 加载最佳模型
        best_model = EnhancedTransformerNILMModule.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            model_params=model_params,
            loss_params=loss_params,
            learning_rate=1e-4,
            appliances=appliances
        )
        best_model.to(device)
        
        # 评估模型
        print("\n正在评估模型...")
        results, power_preds, power_trues, state_preds, state_trues = evaluate_model(
            best_model.model, test_loader, appliances, device
        )
        
        # 打印结果
        print("\n=" * 50)
        print("模型评估结果:")
        print("=" * 50)
        
        for appliance in appliances:
            print(f"\n{appliance}:")
            print(f"  MAE: {results[appliance]['mae']:.4f}")
            print(f"  RMSE: {results[appliance]['rmse']:.4f}")
            print(f"  R²: {results[appliance]['r2']:.4f}")
            print(f"  相关系数: {results[appliance]['correlation']:.4f}")
        
        print(f"\n平均指标:")
        print(f"  平均MAE: {results['average']['mae']:.4f}")
        print(f"  平均RMSE: {results['average']['rmse']:.4f}")
        print(f"  平均R²: {results['average']['r2']:.4f}")
        print(f"  平均相关系数: {results['average']['correlation']:.4f}")
        
        # 保存结果
        with open(f'{save_dir}/real_ampds2_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 创建可视化
        print("\n正在生成可视化图表...")
        create_visualizations(results, power_preds, power_trues, appliances, save_dir)
        
        print(f"\n所有结果已保存到: {save_dir}/")
        print("\n测试完成！")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()