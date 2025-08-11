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

# 导入数据增强模块
from data_augmentation import NILMDataAugmentation, DevicePatternLibrary

class EnhancedAMPds2Dataset(Dataset):
    """增强的AMPds2数据集，支持更多数据增强和更好的窗口生成"""
    
    def __init__(self, 
                 data_path: str,
                 window_length: int = 256,
                 step_size: int = 64,
                 channels: List[str] = None,
                 device_columns: List[str] = None,
                 power_threshold: float = 10.0,
                 split: str = 'train',
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 scaler: Optional[RobustScaler] = None,
                 augment: bool = False,
                 min_samples: int = 100):  # 最小样本数
        
        self.data_path = data_path
        self.window_length = window_length
        self.step_size = step_size
        self.power_threshold = power_threshold
        self.split = split
        self.augment = augment
        self.min_samples = min_samples
        
        # 默认通道配置
        if channels is None:
            self.channels = ['P_total', 'Q_total', 'S_total', 'I', 'V', 'PF']
        else:
            self.channels = channels
            
        # 默认设备列表
        if device_columns is None:
            self.device_columns = [
                'device_01', 'device_02', 'device_03', 'device_04', 'device_05',
                'device_06', 'device_07', 'device_08', 'device_09', 'device_10'
            ]
        else:
            self.device_columns = device_columns
            
        self.scaler = scaler
        
        # 初始化数据增强器
        if self.augment:
            self.data_augmenter = NILMDataAugmentation(
                noise_std=0.01,
                amplitude_range=(0.8, 1.2),
                time_jitter=2,
                channel_dropout=0.1,
                synthetic_overlay_prob=0.3,
                frequency_shift_range=(0.98, 1.02)
            )
            # 初始化设备模式库
            self.pattern_library = DevicePatternLibrary()
        else:
            self.data_augmenter = None
            self.pattern_library = None
            
        self.load_data()
        self.create_windows()
        
    def load_data(self):
        """加载AMPds2数据"""
        print(f"Loading data from {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            # 加载电力数据
            electricity_data = {}
            
            # 加载主电表数据
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
                meter_key = f'meter_{i+2:02d}'
                if 'electricity' in f and meter_key in f['electricity']:
                    device_meter = f['electricity'][meter_key]
                    if 'P' in device_meter:
                        electricity_data[device] = device_meter['P'][:]
                        
            # 创建DataFrame
            df = pd.DataFrame(electricity_data)
            
            # 生成时间索引
            df.index = pd.date_range(start='2012-04-01', periods=len(df), freq='1min')
            
        # 生成派生特征
        self.create_features(df)
        
        # 数据分割
        self.split_data(df)
        
    def create_features(self, df: pd.DataFrame):
        """创建多通道特征"""
        print("Creating enhanced multi-channel features...")
        
        # 基础特征
        features = {'P_total': df['P_total'].values}
        
        # 添加其他基础通道
        for ch in ['Q_total', 'S_total', 'I', 'V', 'PF']:
            if ch in df.columns:
                features[ch] = df[ch].values
            else:
                features[ch] = np.zeros_like(features['P_total'])
                
        # 派生特征
        # 1. 多阶差分
        features['delta_P'] = np.diff(features['P_total'], prepend=features['P_total'][0])
        features['delta2_P'] = np.diff(features['delta_P'], prepend=features['delta_P'][0])
        
        # 2. 多尺度滑窗统计特征
        for window_min in [3, 5, 10, 15, 30]:
            window_size = window_min
            
            # 滑窗统计
            p_series = pd.Series(features['P_total'])
            features[f'P_mean_{window_min}min'] = p_series.rolling(
                window=window_size, min_periods=1).mean().values
            features[f'P_std_{window_min}min'] = p_series.rolling(
                window=window_size, min_periods=1).std().fillna(0).values
            features[f'P_max_{window_min}min'] = p_series.rolling(
                window=window_size, min_periods=1).max().values
            features[f'P_min_{window_min}min'] = p_series.rolling(
                window=window_size, min_periods=1).min().values
                
        # 3. 频域特征
        # 使用不同窗口的移动平均作为频域近似
        for freq_window in [5, 15, 60]:
            low_freq = pd.Series(features['P_total']).rolling(
                window=freq_window, min_periods=1).mean().values
            total_energy = np.abs(features['P_total']) + 1e-8
            features[f'freq_ratio_{freq_window}'] = np.abs(low_freq) / total_energy
        
        # 4. 时间特征
        time_features = self.create_time_features(df.index)
        features.update(time_features)
        
        # 转换为numpy数组，确保float32类型
        self.feature_names = list(features.keys())
        self.X = np.stack([features[name] for name in self.feature_names], axis=1).astype(np.float32)
        
        # 设备功率标签
        device_data = df[self.device_columns].values.astype(np.float32)
        self.Y_power = device_data
        
        # 状态标签
        self.Y_state = (device_data > self.power_threshold).astype(np.float32)
        
        print(f"Feature shape: {self.X.shape}")
        print(f"Power labels shape: {self.Y_power.shape}")
        print(f"State labels shape: {self.Y_state.shape}")
        
    def create_time_features(self, time_index):
        """创建时间特征"""
        features = {}
        
        # 小时特征（正弦余弦编码）
        hour = time_index.hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # 星期特征
        weekday = time_index.weekday
        features['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
        features['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
        
        # 月份特征
        month = time_index.month
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        # 工作日/周末
        features['is_weekend'] = (weekday >= 5).astype(np.float32)
        
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
        """创建滑窗样本，确保足够的训练数据"""
        print(f"Creating windows with length={self.window_length}, step={self.step_size}")
        
        n_samples = len(self.X)
        windows_X, windows_Y_power, windows_Y_state = [], [], []
        
        # 基础窗口
        for i in range(0, n_samples - self.window_length + 1, self.step_size):
            end_idx = i + self.window_length
            
            windows_X.append(self.X[i:end_idx])
            windows_Y_power.append(self.Y_power[i:end_idx])
            windows_Y_state.append(self.Y_state[i:end_idx])
        
        # 如果训练数据不足，增加重叠窗口
        if self.split == 'train' and len(windows_X) < self.min_samples:
            print(f"训练样本不足({len(windows_X)} < {self.min_samples})，增加重叠窗口")
            
            # 使用更小的步长创建更多窗口
            small_step = max(1, self.step_size // 4)
            for i in range(0, n_samples - self.window_length + 1, small_step):
                end_idx = i + self.window_length
                
                windows_X.append(self.X[i:end_idx])
                windows_Y_power.append(self.Y_power[i:end_idx])
                windows_Y_state.append(self.Y_state[i:end_idx])
                
                if len(windows_X) >= self.min_samples:
                    break
        
        self.windows_X = np.array(windows_X)
        self.windows_Y_power = np.array(windows_Y_power)
        self.windows_Y_state = np.array(windows_Y_state)
        
        print(f"Created {len(self.windows_X)} windows")
        
    def normalize_features(self, scaler=None):
        """特征归一化"""
        if scaler is None:
            self.scaler = RobustScaler()
            X_reshaped = self.windows_X.reshape(-1, self.windows_X.shape[-1])
            self.scaler.fit(X_reshaped)
        else:
            self.scaler = scaler
            
        X_reshaped = self.windows_X.reshape(-1, self.windows_X.shape[-1])
        X_normalized = self.scaler.transform(X_reshaped)
        self.windows_X = X_normalized.reshape(self.windows_X.shape)
        
        return self.scaler
        
    def augment_data(self, x, y_power, y_state):
        """增强的数据增强"""
        if not self.augment or self.split != 'train' or self.data_augmenter is None:
            return x, y_power, y_state
            
        # 使用新的数据增强器
        # 随机选择增强方法
        aug_prob = np.random.random()
        
        if aug_prob < 0.2:  # 高斯噪声
            x = self.data_augmenter.add_gaussian_noise(x)
        elif aug_prob < 0.4:  # 幅度缩放
            x = self.data_augmenter.amplitude_scaling(x)
            y_power = y_power * (x.mean() / (x.mean() + 1e-8))  # 相应调整功率
        elif aug_prob < 0.5:  # 时间抖动
            x, y_power, y_state = self.data_augmenter.time_jitter(x, y_power, y_state)
        elif aug_prob < 0.6:  # 通道丢弃
            x = self.data_augmenter.channel_dropout(x)
        elif aug_prob < 0.7:  # 频率偏移
            x = self.data_augmenter.frequency_shift(x)
        
        return x, y_power, y_state
        
    def __len__(self):
        return len(self.windows_X)
        
    def __getitem__(self, idx):
        x = self.windows_X[idx].astype(np.float32)
        y_power = self.windows_Y_power[idx].astype(np.float32)
        y_state = self.windows_Y_state[idx].astype(np.float32)
        
        # 数据增强
        x, y_power, y_state = self.augment_data(x, y_power, y_state)
        
        # 确保数据类型为float32
        x = x.astype(np.float32)
        y_power = y_power.astype(np.float32)
        y_state = y_state.astype(np.float32)
        
        return {
            'x': torch.from_numpy(x),
            'y_power': torch.from_numpy(y_power),
            'y_state': torch.from_numpy(y_state)
        }

class EnhancedAMPds2DataModule(pl.LightningDataModule):
    """增强的AMPds2数据模块"""
    
    def __init__(self,
                 data_path: str,
                 window_length: int = 256,
                 step_size: int = 64,
                 batch_size: int = 16,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 channels: List[str] = None,
                 device_columns: List[str] = None,
                 power_threshold: float = 10.0,
                 augment: bool = True,
                 min_samples: int = 100):
        
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
        self.augment = augment
        self.min_samples = min_samples
        self.scaler = None
        
    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        if stage == 'fit' or stage is None:
            # 训练集
            self.train_dataset = EnhancedAMPds2Dataset(
                data_path=self.data_path,
                window_length=self.window_length,
                step_size=self.step_size,
                channels=self.channels,
                device_columns=self.device_columns,
                power_threshold=self.power_threshold,
                split='train',
                augment=self.augment,
                min_samples=self.min_samples
            )
            
            # 拟合归一化器
            self.scaler = self.train_dataset.normalize_features()
            
            # 验证集
            self.val_dataset = EnhancedAMPds2Dataset(
                data_path=self.data_path,
                window_length=self.window_length,
                step_size=self.step_size,
                channels=self.channels,
                device_columns=self.device_columns,
                power_threshold=self.power_threshold,
                split='val',
                augment=False,
                min_samples=10  # 验证集不需要太多样本
            )
            self.val_dataset.normalize_features(self.scaler)
            
        if stage == 'test' or stage is None:
            # 测试集
            self.test_dataset = EnhancedAMPds2Dataset(
                data_path=self.data_path,
                window_length=self.window_length,
                step_size=self.step_size,
                channels=self.channels,
                device_columns=self.device_columns,
                power_threshold=self.power_threshold,
                split='test',
                augment=False,
                min_samples=10
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
    
    @property
    def feature_dim(self):
        """获取特征维度"""
        if hasattr(self, 'train_dataset'):
            return self.train_dataset.windows_X.shape[-1]
        return None
        
    @property
    def num_devices(self):
        """获取设备数量"""
        if hasattr(self, 'train_dataset'):
            return len(self.train_dataset.device_columns)
        return None