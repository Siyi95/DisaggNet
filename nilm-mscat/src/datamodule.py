import os
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.preprocessing import RobustScaler
from typing import Optional, Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class AMPds2Dataset(Dataset):
    """AMPds2数据集，支持多通道特征和滑窗切片"""
    
    def __init__(self, 
                 data_path: str,
                 window_length: int = 120,
                 step_size: int = 60,
                 channels: List[str] = None,
                 device_columns: List[str] = None,
                 power_threshold: float = 10.0,
                 split: str = 'train',
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 scaler: Optional[RobustScaler] = None,
                 augment: bool = False):
        
        self.data_path = data_path
        self.window_length = window_length
        self.step_size = step_size
        self.power_threshold = power_threshold
        self.split = split
        self.augment = augment
        
        # 默认通道配置
        if channels is None:
            self.channels = ['P_total', 'Q_total', 'S_total', 'I', 'V', 'PF']
        else:
            self.channels = channels
            
        # 默认设备列表（AMPds2主要设备，根据实际meter数量调整）
        if device_columns is None:
            # AMPds2数据集有11个meter，meter_01是主电表，meter_02到meter_11是设备
            self.device_columns = [
                'device_01', 'device_02', 'device_03', 'device_04', 'device_05',
                'device_06', 'device_07', 'device_08', 'device_09', 'device_10'
            ]
        else:
            self.device_columns = device_columns
            
        self.scaler = scaler
        self.load_data()
        self.create_windows()
        
    def load_data(self):
        """加载AMPds2数据"""
        print(f"Loading data from {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            # 加载电力数据 - AMPds2格式
            electricity_data = {}
            
            # 加载主电表数据 (meter_01通常是主电表)
            if 'electricity' in f and 'meter_01' in f['electricity']:
                main_meter = f['electricity']['meter_01']
                if 'P' in main_meter:
                    electricity_data['P_total'] = main_meter['P'][:]
                if 'Q' in main_meter:
                    electricity_data['Q_total'] = main_meter['Q'][:]
                if 'S' in main_meter:
                    electricity_data['S_total'] = main_meter['S'][:]
                if 'I' in main_meter:
                    electricity_data['I'] = main_meter['I'][:]
                if 'V' in main_meter:
                    electricity_data['V'] = main_meter['V'][:]
                if 'PF' in main_meter:
                    electricity_data['PF'] = main_meter['PF'][:]
            
            # 加载各个设备的功率数据
            for i, device in enumerate(self.device_columns):
                meter_key = f'meter_{i+2:02d}'  # 从meter_02开始
                if 'electricity' in f and meter_key in f['electricity']:
                    device_meter = f['electricity'][meter_key]
                    if 'P' in device_meter:
                        electricity_data[device] = device_meter['P'][:]
                        
            # 创建DataFrame
            df = pd.DataFrame(electricity_data)
            
            # 生成时间索引（假设1分钟采样）
            df.index = pd.date_range(start='2012-04-01', periods=len(df), freq='1min')
            
        # 生成派生特征
        self.create_features(df)
        
        # 数据分割
        self.split_data(df)
        
    def create_features(self, df: pd.DataFrame):
        """创建多通道特征"""
        print("Creating multi-channel features...")
        
        # 基础特征
        features = {'P_total': df['P_total'].values}
        
        # 添加其他可用的基础通道（如果数据中存在）
        for ch in ['Q_total', 'S_total', 'I', 'V', 'PF']:
            if ch in df.columns:
                features[ch] = df[ch].values
            else:
                # 如果不存在，生成模拟数据或设为零
                features[ch] = np.zeros_like(features['P_total'])
                
        # 派生特征
        # 1. 一阶差分
        features['delta_P'] = np.diff(features['P_total'], prepend=features['P_total'][0])
        
        # 2. 滑窗统计特征（3-10分钟窗口）
        for window_min in [3, 5, 10]:
            window_size = window_min  # 已经是分钟级数据
            
            # 滑窗均值
            features[f'P_mean_{window_min}min'] = pd.Series(features['P_total']).rolling(
                window=window_size, min_periods=1).mean().values
            
            # 滑窗方差
            features[f'P_var_{window_min}min'] = pd.Series(features['P_total']).rolling(
                window=window_size, min_periods=1).var().fillna(0).values
            
            # 窗口能量
            features[f'P_energy_{window_min}min'] = pd.Series(features['P_total']).rolling(
                window=window_size, min_periods=1).sum().values
                
        # 3. 简易频域特征（低频/全频能量比）
        # 使用简单的移动平均作为低频近似
        low_freq = pd.Series(features['P_total']).rolling(window=10, min_periods=1).mean().values
        total_energy = np.abs(features['P_total']) + 1e-8
        features['freq_ratio'] = np.abs(low_freq) / total_energy
        
        # 4. 时间特征
        time_features = self.create_time_features(df.index)
        features.update(time_features)
        
        # 转换为numpy数组并堆叠
        self.feature_names = list(features.keys())
        self.X = np.stack([features[name] for name in self.feature_names], axis=1)
        
        # 设备功率标签
        device_data = df[self.device_columns].values
        self.Y_power = device_data
        
        # 状态标签（阈值化）
        self.Y_state = (device_data > self.power_threshold).astype(np.float32)
        
        print(f"Feature shape: {self.X.shape}")
        print(f"Power labels shape: {self.Y_power.shape}")
        print(f"State labels shape: {self.Y_state.shape}")
        
    def create_time_features(self, time_index):
        """创建时间特征"""
        features = {}
        
        # 小时 one-hot (0-23)
        hour = time_index.hour
        for h in range(24):
            features[f'hour_{h}'] = (hour == h).astype(np.float32)
            
        # 星期 one-hot (0-6)
        weekday = time_index.weekday
        for w in range(7):
            features[f'weekday_{w}'] = (weekday == w).astype(np.float32)
            
        return features
        
    def split_data(self, df: pd.DataFrame):
        """数据分割"""
        n_samples = len(df)
        train_end = int(n_samples * 0.7)
        val_end = int(n_samples * 0.85)
        
        if self.split == 'train':
            self.X = self.X[:train_end]
            self.Y_power = self.Y_power[:train_end]
            self.Y_state = self.Y_state[:train_end]
        elif self.split == 'val':
            self.X = self.X[train_end:val_end]
            self.Y_power = self.Y_power[train_end:val_end]
            self.Y_state = self.Y_state[train_end:val_end]
        else:  # test
            self.X = self.X[val_end:]
            self.Y_power = self.Y_power[val_end:]
            self.Y_state = self.Y_state[val_end:]
            
    def create_windows(self):
        """创建滑窗样本"""
        print(f"Creating windows with length={self.window_length}, step={self.step_size}")
        
        n_samples = len(self.X)
        windows_X, windows_Y_power, windows_Y_state = [], [], []
        
        for i in range(0, n_samples - self.window_length + 1, self.step_size):
            end_idx = i + self.window_length
            
            windows_X.append(self.X[i:end_idx])
            windows_Y_power.append(self.Y_power[i:end_idx])
            windows_Y_state.append(self.Y_state[i:end_idx])
            
        self.windows_X = np.array(windows_X)
        self.windows_Y_power = np.array(windows_Y_power)
        self.windows_Y_state = np.array(windows_Y_state)
        
        print(f"Created {len(self.windows_X)} windows")
        
    def normalize_features(self, scaler=None):
        """特征归一化"""
        if scaler is None:
            self.scaler = RobustScaler()
            # 重塑为 (n_samples * window_length, n_features)
            X_reshaped = self.windows_X.reshape(-1, self.windows_X.shape[-1])
            self.scaler.fit(X_reshaped)
        else:
            self.scaler = scaler
            
        # 归一化
        X_reshaped = self.windows_X.reshape(-1, self.windows_X.shape[-1])
        X_normalized = self.scaler.transform(X_reshaped)
        self.windows_X = X_normalized.reshape(self.windows_X.shape)
        
        return self.scaler
        
    def augment_data(self, x, y_power, y_state):
        """数据增强"""
        if not self.augment or self.split != 'train':
            return x, y_power, y_state
            
        # 加性噪声
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.02, x.shape)
            x = x + noise
            
        # 幅度缩放
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale
            y_power = y_power * scale
            
        # 通道dropout
        if np.random.random() < 0.1:
            n_channels = x.shape[-1]
            drop_channels = np.random.choice(n_channels, size=max(1, n_channels//10), replace=False)
            x[:, drop_channels] = 0
            
        return x, y_power, y_state
        
    def __len__(self):
        return len(self.windows_X)
        
    def __getitem__(self, idx):
        x = self.windows_X[idx].astype(np.float32)
        y_power = self.windows_Y_power[idx].astype(np.float32)
        y_state = self.windows_Y_state[idx].astype(np.float32)
        
        # 数据增强
        x, y_power, y_state = self.augment_data(x, y_power, y_state)
        
        return {
            'x': torch.from_numpy(x),
            'y_power': torch.from_numpy(y_power),
            'y_state': torch.from_numpy(y_state)
        }

class AMPds2DataModule(pl.LightningDataModule):
    """AMPds2 Lightning数据模块"""
    
    def __init__(self,
                 data_path: str = 'data/AMPds2.h5',
                 window_length: int = 120,
                 step_size: int = 60,
                 batch_size: int = 32,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 channels: List[str] = None,
                 device_columns: List[str] = None,
                 power_threshold: float = 10.0):
        
        super().__init__()
        self.data_path = data_path
        self.window_length = window_length
        self.step_size = step_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.channels = channels
        self.device_columns = device_columns
        self.power_threshold = power_threshold
        self.scaler = None
        
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        if stage == 'fit' or stage is None:
            # 训练集
            self.train_dataset = AMPds2Dataset(
                data_path=self.data_path,
                window_length=self.window_length,
                step_size=self.step_size,
                channels=self.channels,
                device_columns=self.device_columns,
                power_threshold=self.power_threshold,
                split='train',
                augment=True
            )
            
            # 拟合归一化器
            self.scaler = self.train_dataset.normalize_features()
            
            # 验证集
            self.val_dataset = AMPds2Dataset(
                data_path=self.data_path,
                window_length=self.window_length,
                step_size=self.step_size,
                channels=self.channels,
                device_columns=self.device_columns,
                power_threshold=self.power_threshold,
                split='val',
                augment=False
            )
            self.val_dataset.normalize_features(self.scaler)
            
        if stage == 'test' or stage is None:
            # 测试集
            self.test_dataset = AMPds2Dataset(
                data_path=self.data_path,
                window_length=self.window_length,
                step_size=self.step_size,
                channels=self.channels,
                device_columns=self.device_columns,
                power_threshold=self.power_threshold,
                split='test',
                augment=False
            )
            if self.scaler is not None:
                self.test_dataset.normalize_features(self.scaler)
                
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_device_names(self):
        """获取设备名称列表"""
        if self.device_columns is None:
            # 如果device_columns为None，返回默认设备列表
            return [
                'device_01', 'device_02', 'device_03', 'device_04', 'device_05',
                'device_06', 'device_07', 'device_08', 'device_09', 'device_10'
            ]
        return self.device_columns
    
    def get_input_dim(self):
        """获取输入特征维度"""
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            return self.train_dataset.windows_X.shape[-1]
        return len(self.channels) if self.channels else 6  # 默认6个通道
    
    def get_num_devices(self):
        """获取设备数量"""
        return len(self.device_columns)
        
    def get_feature_dim(self):
        """获取特征维度"""
        if hasattr(self, 'train_dataset'):
            return self.train_dataset.windows_X.shape[-1]
        return None
        
    def get_num_devices(self):
        """获取设备数量"""
        if hasattr(self, 'train_dataset'):
            return len(self.train_dataset.device_columns)
        return None