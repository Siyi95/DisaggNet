import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback
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

# 导入增强版Transformer模型组件（复制自之前的脚本）
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

class OptimizedAMPds2Dataset(Dataset):
    """优化的AMPds2数据集，用于超参数优化"""
    def __init__(self, sequence_length=512, train=True, train_ratio=0.8, max_samples=20000):
        self.sequence_length = sequence_length
        self.max_samples = max_samples
        
        print(f"正在创建优化数据集 (max_samples={max_samples})")
        
        # 生成合成数据（用于快速超参数优化）
        self._generate_synthetic_data()
        
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
    
    def _generate_synthetic_data(self):
        """生成合成数据用于快速优化"""
        np.random.seed(42)
        length = self.max_samples + self.sequence_length
        
        # 基础负载
        base_load = 200 + 100 * np.sin(np.linspace(0, 8*np.pi, length))
        
        # 添加随机波动
        noise = np.random.normal(0, 30, length)
        
        # 添加设备开关事件
        events = np.zeros(length)
        for _ in range(length // 500):  # 更频繁的事件
            start = np.random.randint(0, length - 100)
            duration = np.random.randint(50, 200)
            power = np.random.uniform(100, 800)
            events[start:start+duration] += power
        
        self.main_power = base_load + noise + events
        
        # 创建设备功率数据
        self._create_appliance_data(length)
    
    def _create_appliance_data(self, length):
        """创建设备功率数据"""
        # 冰箱：持续运行，周期性开关
        fridge_pattern = 120 + 80 * (np.sin(np.linspace(0, 30*np.pi, length)) > 0.2)
        fridge_noise = np.random.normal(0, 15, length)
        self.fridge_power = np.maximum(0, fridge_pattern + fridge_noise)
        
        # 洗衣机：间歇性运行
        washer_power = np.zeros(length)
        for _ in range(length // 3000):
            start = np.random.randint(0, length - 800)
            duration = np.random.randint(400, 800)
            power_profile = 400 * np.sin(np.linspace(0, np.pi, duration))
            washer_power[start:start+duration] = power_profile
        
        # 微波炉：短时间高功率
        microwave_power = np.zeros(length)
        for _ in range(length // 1000):
            start = np.random.randint(0, length - 80)
            duration = np.random.randint(20, 80)
            microwave_power[start:start+duration] = 1000
        
        # 洗碗机：中等功率，较长时间
        dishwasher_power = np.zeros(length)
        for _ in range(length // 5000):
            start = np.random.randint(0, length - 1500)
            duration = np.random.randint(800, 1500)
            power_profile = 250 + 150 * np.sin(np.linspace(0, 3*np.pi, duration))
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
            threshold = np.percentile(self.appliance_power[appliance], 70)
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

class OptimizedTransformerNILMModule(pl.LightningModule):
    def __init__(self, trial_params, appliances=None):
        super().__init__()
        self.save_hyperparameters()
        
        # 从trial_params中提取参数
        model_params = {
            'input_dim': 1,
            'd_model': trial_params['d_model'],
            'n_heads': trial_params['n_heads'],
            'n_layers': trial_params['n_layers'],
            'd_ff': trial_params['d_ff'],
            'dropout': trial_params['dropout'],
            'num_appliances': 4,
            'window_size': trial_params['window_size']
        }
        
        loss_params = {
            'power_weight': trial_params['power_weight'],
            'state_weight': trial_params['state_weight'],
            'correlation_weight': trial_params['correlation_weight']
        }
        
        self.model = EnhancedTransformerNILM(**model_params)
        self.criterion = CombinedLoss(**loss_params)
        self.learning_rate = trial_params['learning_rate']
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
        
        # 检查并处理NaN值
        power_preds = torch.nan_to_num(power_preds, nan=0.0, posinf=1e6, neginf=-1e6)
        power_trues = torch.nan_to_num(power_trues, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 计算平均R²作为优化目标
        try:
            valid_r2s = []
            for i in range(len(self.appliances)):
                pred_i = power_preds[:, i].numpy()
                true_i = power_trues[:, i].numpy()
                
                if not (np.any(np.isnan(pred_i)) or np.any(np.isnan(true_i))):
                    r2_i = r2_score(true_i, pred_i)
                    if np.isfinite(r2_i):
                        valid_r2s.append(r2_i)
            
            if valid_r2s:
                avg_r2 = np.mean(valid_r2s)
                self.log('val_avg_r2', avg_r2)
            else:
                self.log('val_avg_r2', 0.0)
                
        except Exception as e:
            print(f"计算R²时出错: {e}")
            self.log('val_avg_r2', 0.0)
        
        # 清空验证输出
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

def objective(trial):
    """
    Optuna优化目标函数
    """
    # 定义超参数搜索空间
    trial_params = {
        # 模型架构参数
        'd_model': trial.suggest_categorical('d_model', [128, 256, 512]),
        'n_heads': trial.suggest_categorical('n_heads', [4, 8, 16]),
        'n_layers': trial.suggest_int('n_layers', 3, 8),
        'd_ff': trial.suggest_categorical('d_ff', [512, 1024, 2048]),
        'window_size': trial.suggest_categorical('window_size', [32, 64, 128]),
        
        # 正则化参数
        'dropout': trial.suggest_float('dropout', 0.05, 0.3),
        
        # 损失函数权重
        'power_weight': trial.suggest_float('power_weight', 0.5, 2.0),
        'state_weight': trial.suggest_float('state_weight', 0.1, 1.0),
        'correlation_weight': trial.suggest_float('correlation_weight', 0.1, 0.5),
        
        # 优化器参数
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
    }
    
    # 确保n_heads能被d_model整除
    if trial_params['d_model'] % trial_params['n_heads'] != 0:
        # 调整n_heads使其能被d_model整除
        valid_heads = [h for h in [4, 8, 16] if trial_params['d_model'] % h == 0]
        if valid_heads:
            trial_params['n_heads'] = trial.suggest_categorical(f'n_heads_adjusted_{trial_params["d_model"]}', valid_heads)
        else:
            trial_params['n_heads'] = 4  # 默认值
    
    print(f"\n试验 {trial.number}: {trial_params}")
    
    # 创建数据集
    train_dataset = OptimizedAMPds2Dataset(
        sequence_length=256,  # 减少序列长度以加快训练
        train=True,
        max_samples=10000  # 减少样本数以加快训练
    )
    val_dataset = OptimizedAMPds2Dataset(
        sequence_length=256,
        train=False,
        max_samples=10000
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, 
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, 
        num_workers=2, pin_memory=True
    )
    
    # 创建模型
    model = OptimizedTransformerNILMModule(trial_params)
    
    # 设置回调函数
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )
    
    # 创建训练器（暂时不使用pruning callback以避免兼容性问题）
    trainer = pl.Trainer(
        max_epochs=15,  # 减少训练轮数
        callbacks=[early_stopping],
        logger=False,  # 禁用日志以加快训练
        enable_checkpointing=False,
        accelerator='auto',
        devices=1,
        precision=16 if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        enable_progress_bar=False  # 禁用进度条
    )
    
    try:
        # 训练模型
        trainer.fit(model, train_loader, val_loader)
        
        # 获取最佳验证R²
        if trainer.callback_metrics:
            val_r2 = trainer.callback_metrics.get('val_avg_r2', 0.0)
            if isinstance(val_r2, torch.Tensor):
                val_r2 = val_r2.item()
        else:
            val_r2 = 0.0
        
        print(f"试验 {trial.number} 完成，验证R²: {val_r2:.4f}")
        return val_r2
        
    except Exception as e:
        print(f"试验 {trial.number} 失败: {str(e)}")
        return 0.0

def run_hyperparameter_optimization():
    """
    运行超参数优化
    """
    print("=" * 60)
    print("增强版Transformer NILM模型超参数优化")
    print("=" * 60)
    
    # 创建结果保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'outputs/optuna_optimization_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建Optuna研究
    study = optuna.create_study(
        direction='maximize',  # 最大化R²
        study_name='enhanced_transformer_nilm_optimization',
        storage=f'sqlite:///{save_dir}/optuna_study.db',
        load_if_exists=True
    )
    
    print(f"开始超参数优化，目标：最大化验证R²")
    print(f"结果将保存到: {save_dir}/")
    
    try:
        # 运行优化
        study.optimize(objective, n_trials=50, timeout=7200)  # 2小时超时
        
        # 打印最佳结果
        print("\n=" * 50)
        print("优化完成！")
        print("=" * 50)
        
        print(f"\n最佳试验: {study.best_trial.number}")
        print(f"最佳验证R²: {study.best_value:.4f}")
        print("\n最佳超参数:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # 保存结果
        results = {
            'best_trial': study.best_trial.number,
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials)
        }
        
        with open(f'{save_dir}/optimization_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 创建优化历史图
        create_optimization_plots(study, save_dir)
        
        print(f"\n所有结果已保存到: {save_dir}/")
        
        return study.best_params
        
    except Exception as e:
        print(f"\n优化过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_optimization_plots(study, save_dir):
    """
    创建优化过程的可视化图表
    """
    try:
        import optuna.visualization as vis
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 优化历史
        fig = vis.plot_optimization_history(study)
        fig.write_image(f'{save_dir}/optimization_history.png')
        
        # 2. 参数重要性
        fig = vis.plot_param_importances(study)
        fig.write_image(f'{save_dir}/param_importances.png')
        
        # 3. 参数关系
        if len(study.best_params) >= 2:
            param_names = list(study.best_params.keys())[:2]
            fig = vis.plot_contour(study, params=param_names)
            fig.write_image(f'{save_dir}/param_contour.png')
        
        print(f"优化可视化图表已保存到 {save_dir}/")
        
    except Exception as e:
        print(f"创建可视化图表时出错: {e}")
        # 创建简单的matplotlib图表作为备选
        create_simple_plots(study, save_dir)

def create_simple_plots(study, save_dir):
    """
    创建简单的matplotlib图表
    """
    try:
        # 提取试验数据
        trial_numbers = [trial.number for trial in study.trials]
        trial_values = [trial.value if trial.value is not None else 0 for trial in study.trials]
        
        # 优化历史图
        plt.figure(figsize=(10, 6))
        plt.plot(trial_numbers, trial_values, 'b-', alpha=0.7)
        plt.scatter(trial_numbers, trial_values, c='red', alpha=0.5)
        plt.xlabel('试验编号')
        plt.ylabel('验证R²')
        plt.title('超参数优化历史')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/simple_optimization_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 最佳值累积图
        best_values = []
        current_best = float('-inf')
        for value in trial_values:
            if value > current_best:
                current_best = value
            best_values.append(current_best)
        
        plt.figure(figsize=(10, 6))
        plt.plot(trial_numbers, best_values, 'g-', linewidth=2)
        plt.xlabel('试验编号')
        plt.ylabel('最佳验证R²')
        plt.title('最佳性能演进')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/best_value_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"简单可视化图表已保存到 {save_dir}/")
        
    except Exception as e:
        print(f"创建简单图表时出错: {e}")

if __name__ == "__main__":
    # 运行超参数优化
    best_params = run_hyperparameter_optimization()
    
    if best_params:
        print("\n超参数优化成功完成！")
        print("可以使用最佳参数重新训练完整模型。")
    else:
        print("\n超参数优化失败。")