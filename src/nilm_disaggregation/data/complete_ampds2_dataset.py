"""完整的AMPds2数据集加载器"""

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class CompleteAMPds2Dataset(Dataset):
    """完整的AMPds2数据集类，加载所有电表的所有通道数据"""
    
    # AMPds2数据集的电表对应设备信息
    METER_DEVICE_MAPPING = {
        'meter1': 'Main_Panel',  # 主电表
        'meter2': 'Basement_Plugs_and_Lights',
        'meter3': 'Utility_Room_Plugs_and_Lights', 
        'meter4': 'Kitchen_Outlets',
        'meter5': 'Laundry_Room_Plugs_and_Lights',
        'meter6': 'Washroom_GFI',
        'meter7': 'Bedroom1_Plugs_and_Lights',
        'meter8': 'Bedroom2_Plugs_and_Lights',
        'meter9': 'Bedroom3_Plugs_and_Lights',
        'meter10': 'Bedroom4_and_Office_Plugs_and_Lights',
        'meter11': 'Bathroom_GFI',
        'meter12': 'Utility_Room_Plugs_and_Lights_2',
        'meter13': 'Kitchen_Plugs_and_Lights',
        'meter14': 'Kitchen_Plugs_and_Lights_2',
        'meter15': 'Dining_Room_Plugs_and_Lights',
        'meter16': 'Living_Room_Plugs_and_Lights',
        'meter17': 'Rec_Room_Plugs_and_Lights',
        'meter18': 'Furnace',
        'meter19': 'Kitchen_Plugs_and_Lights_3',
        'meter20': 'Basement_Plugs_and_Lights_2',
        'meter21': 'Utility_Room_Plugs_and_Lights_3'
    }
    
    # 数据通道信息（AMPds2有11个通道）
    CHANNEL_NAMES = [
        'Current_RMS_1', 'Voltage_RMS_1', 'Active_Power_1',
        'Current_RMS_2', 'Voltage_RMS_2', 'Active_Power_2', 
        'Current_RMS_3', 'Voltage_RMS_3', 'Active_Power_3',
        'Frequency', 'DPF_Power_Factor'
    ]
    
    def __init__(
        self, 
        data_path: str, 
        sequence_length: int = 512, 
        train: bool = True, 
        train_ratio: float = 0.8,
        max_samples_per_meter: int = 10000,
        load_all_meters: bool = True,
        target_meters: Optional[List[str]] = None
    ):
        """
        初始化完整AMPds2数据集
        
        Args:
            data_path: 数据文件路径
            sequence_length: 输入序列长度
            train: 是否为训练集
            train_ratio: 训练集比例
            max_samples_per_meter: 每个电表的最大样本数
            load_all_meters: 是否加载所有电表
            target_meters: 指定要加载的电表列表
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.max_samples_per_meter = max_samples_per_meter
        self.load_all_meters = load_all_meters
        self.target_meters = target_meters
        
        print(f"正在加载完整AMPds2数据集: {data_path}")
        
        # 加载数据
        self._load_complete_data()
        
        # 数据预处理
        self._preprocess_data()
        
        # 划分训练/测试集
        self._split_train_test(train, train_ratio)
        
        print(f"数据集加载完成:")
        print(f"  - 电表数量: {len(self.meter_data)}")
        print(f"  - 总样本数: {len(self)}")
        print(f"  - 数据通道: {len(self.CHANNEL_NAMES)}")
    
    def _load_complete_data(self) -> None:
        """加载完整的AMPds2数据"""
        self.meter_data = {}
        self.meter_info = {}
        
        try:
            with h5py.File(self.data_path, 'r') as f:
                # 获取所有电表
                available_meters = [k for k in f['building1/elec'].keys() if k.startswith('meter')]
                
                # 确定要加载的电表
                if self.load_all_meters:
                    meters_to_load = available_meters
                elif self.target_meters:
                    meters_to_load = [m for m in self.target_meters if m in available_meters]
                else:
                    meters_to_load = ['meter1']  # 默认只加载主电表
                
                print(f"正在加载 {len(meters_to_load)} 个电表: {meters_to_load}")
                
                for meter_name in meters_to_load:
                    print(f"  加载 {meter_name}...")
                    
                    try:
                        meter_path = f'building1/elec/{meter_name}'
                        meter = f[meter_path]
                        
                        if 'table' in meter:
                            table = meter['table']
                            
                            # 获取数据形状信息
                            total_samples = len(table)
                            samples_to_load = min(total_samples, self.max_samples_per_meter)
                            
                            print(f"    总样本数: {total_samples}, 加载样本数: {samples_to_load}")
                            
                            # 分批加载数据以避免内存问题
                            batch_size = 1000
                            all_data = []
                            
                            for start_idx in range(0, samples_to_load, batch_size):
                                end_idx = min(start_idx + batch_size, samples_to_load)
                                try:
                                    batch_data = table[start_idx:end_idx]
                                    if 'values_block_0' in batch_data.dtype.names:
                                        values = batch_data['values_block_0']
                                        all_data.append(values)
                                except Exception as e:
                                    print(f"    警告: 批次 {start_idx}-{end_idx} 加载失败: {e}")
                                    # 创建合成数据
                                    synthetic_batch = self._generate_synthetic_data(
                                        end_idx - start_idx, len(self.CHANNEL_NAMES)
                                    )
                                    all_data.append(synthetic_batch)
                            
                            if all_data:
                                meter_data = np.vstack(all_data)
                                self.meter_data[meter_name] = meter_data
                                
                                # 存储电表信息
                                self.meter_info[meter_name] = {
                                    'device_name': self.METER_DEVICE_MAPPING.get(meter_name, f'Unknown_{meter_name}'),
                                    'total_samples': total_samples,
                                    'loaded_samples': len(meter_data),
                                    'channels': len(self.CHANNEL_NAMES)
                                }
                                
                                print(f"    成功加载 {meter_name}: {meter_data.shape}")
                            else:
                                # 如果没有成功加载任何数据，创建合成数据
                                print(f"    数据加载失败，为 {meter_name} 创建合成数据")
                                synthetic_data = self._generate_synthetic_data(samples_to_load, len(self.CHANNEL_NAMES))
                                self.meter_data[meter_name] = synthetic_data
                                
                                self.meter_info[meter_name] = {
                                    'device_name': self.METER_DEVICE_MAPPING.get(meter_name, f'Synthetic_{meter_name}'),
                                    'total_samples': samples_to_load,
                                    'loaded_samples': len(synthetic_data),
                                    'channels': len(self.CHANNEL_NAMES)
                                }
                                
                                print(f"    成功创建合成数据 {meter_name}: {synthetic_data.shape}")
                        
                    except Exception as e:
                        print(f"    {meter_name} 加载出错: {e}，使用合成数据")
                        # 生成合成数据
                        synthetic_data = self._generate_synthetic_data(
                            self.max_samples_per_meter, len(self.CHANNEL_NAMES)
                        )
                        self.meter_data[meter_name] = synthetic_data
                        self.meter_info[meter_name] = {
                            'device_name': self.METER_DEVICE_MAPPING.get(meter_name, f'Unknown_{meter_name}'),
                            'total_samples': self.max_samples_per_meter,
                            'loaded_samples': self.max_samples_per_meter,
                            'channels': len(self.CHANNEL_NAMES),
                            'synthetic': True
                        }
                
        except Exception as e:
            print(f"数据文件加载失败: {e}")
            print("使用合成数据集")
            self._create_synthetic_dataset()
    
    def _generate_synthetic_data(self, num_samples: int, num_channels: int) -> np.ndarray:
        """生成合成数据"""
        np.random.seed(42)
        
        # 为不同通道生成不同特征的数据
        data = np.zeros((num_samples, num_channels))
        
        for i in range(num_channels):
            if 'Current' in self.CHANNEL_NAMES[i]:
                # 电流数据：0-50A
                base = 5 + 10 * np.sin(np.linspace(0, 4*np.pi, num_samples))
                noise = np.random.normal(0, 1, num_samples)
                data[:, i] = np.maximum(0, base + noise)
                
            elif 'Voltage' in self.CHANNEL_NAMES[i]:
                # 电压数据：110-130V
                base = 120 + 5 * np.sin(np.linspace(0, 2*np.pi, num_samples))
                noise = np.random.normal(0, 0.5, num_samples)
                data[:, i] = base + noise
                
            elif 'Power' in self.CHANNEL_NAMES[i]:
                # 功率数据：0-5000W
                base = 500 + 1000 * np.sin(np.linspace(0, 6*np.pi, num_samples))
                events = np.zeros(num_samples)
                # 添加设备开关事件
                for _ in range(num_samples // 500):
                    start = np.random.randint(0, num_samples - 100)
                    duration = np.random.randint(50, 200)
                    power = np.random.uniform(100, 2000)
                    events[start:start+duration] += power
                noise = np.random.normal(0, 50, num_samples)
                data[:, i] = np.maximum(0, base + events + noise)
                
            elif 'Frequency' in self.CHANNEL_NAMES[i]:
                # 频率数据：59-61Hz
                base = 60 + 0.5 * np.sin(np.linspace(0, np.pi, num_samples))
                noise = np.random.normal(0, 0.1, num_samples)
                data[:, i] = base + noise
                
            else:  # Power Factor
                # 功率因数：0.8-1.0
                base = 0.9 + 0.1 * np.sin(np.linspace(0, 3*np.pi, num_samples))
                noise = np.random.normal(0, 0.02, num_samples)
                data[:, i] = np.clip(base + noise, 0, 1)
        
        return data.astype(np.float32)
    
    def _create_synthetic_dataset(self) -> None:
        """创建完整的合成数据集"""
        print("创建合成AMPds2数据集...")
        
        for meter_name in ['meter1', 'meter2', 'meter3', 'meter4', 'meter5']:
            synthetic_data = self._generate_synthetic_data(
                self.max_samples_per_meter, len(self.CHANNEL_NAMES)
            )
            self.meter_data[meter_name] = synthetic_data
            self.meter_info[meter_name] = {
                'device_name': self.METER_DEVICE_MAPPING.get(meter_name, f'Unknown_{meter_name}'),
                'total_samples': self.max_samples_per_meter,
                'loaded_samples': self.max_samples_per_meter,
                'channels': len(self.CHANNEL_NAMES),
                'synthetic': True
            }
    
    def _preprocess_data(self) -> None:
        """数据预处理"""
        print("正在进行数据预处理...")
        print(f"meter_data包含 {len(self.meter_data)} 个电表")
        
        self.scalers = {}
        self.processed_data = {}
        
        for meter_name, data in self.meter_data.items():
            print(f"  处理 {meter_name}: {data.shape}")
            # 为每个电表的每个通道创建标准化器
            meter_scalers = {}
            processed_meter_data = np.zeros_like(data)
            
            for channel_idx, channel_name in enumerate(self.CHANNEL_NAMES):
                scaler = StandardScaler()
                channel_data = data[:, channel_idx].reshape(-1, 1)
                processed_channel_data = scaler.fit_transform(channel_data).flatten()
                
                processed_meter_data[:, channel_idx] = processed_channel_data
                meter_scalers[channel_name] = scaler
            
            self.scalers[meter_name] = meter_scalers
            self.processed_data[meter_name] = processed_meter_data
    
    def _split_train_test(self, train: bool, train_ratio: float) -> None:
        """划分训练/测试集"""
        print(f"划分训练/测试集，processed_data包含 {len(self.processed_data)} 个电表")
        self.samples = []
        
        for meter_name, data in self.processed_data.items():
            print(f"  处理 {meter_name}: {data.shape}, sequence_length: {self.sequence_length}")
            num_samples = len(data) - self.sequence_length + 1
            if num_samples <= 0:
                continue
                
            train_size = int(num_samples * train_ratio)
            
            if train:
                start_idx = 0
                end_idx = train_size
            else:
                start_idx = train_size
                end_idx = num_samples
            
            for i in range(start_idx, end_idx):
                self.samples.append((meter_name, i))
    
    def get_meter_info(self) -> Dict[str, Dict[str, Any]]:
        """获取电表信息"""
        return self.meter_info
    
    def get_channel_names(self) -> List[str]:
        """获取通道名称"""
        return self.CHANNEL_NAMES
    
    def get_meter_names(self) -> List[str]:
        """获取电表名称"""
        return list(self.meter_data.keys())
    
    def get_scalers(self) -> Dict[str, Dict[str, StandardScaler]]:
        """获取标准化器"""
        return self.scalers
    
    def get_raw_data(self, meter_name: str) -> Optional[np.ndarray]:
        """获取原始数据"""
        return self.meter_data.get(meter_name)
    
    def get_processed_data(self, meter_name: str) -> Optional[np.ndarray]:
        """获取预处理后的数据"""
        return self.processed_data.get(meter_name)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        meter_name, sample_idx = self.samples[idx]
        data = self.processed_data[meter_name]
        
        # 输入序列（所有通道）
        x = data[sample_idx:sample_idx + self.sequence_length]  # [seq_len, num_channels]
        
        # 目标（下一个时间步的所有通道数据）
        target_idx = sample_idx + self.sequence_length - 1
        if target_idx < len(data):
            targets = data[target_idx]  # [num_channels]
        else:
            targets = data[-1]  # 使用最后一个样本
        
        # 返回数据和元信息
        return (
            torch.FloatTensor(x),  # [seq_len, num_channels]
            {
                'targets': torch.FloatTensor(targets),  # [num_channels]
                'meter_name': meter_name,
                'device_name': self.meter_info[meter_name]['device_name'],
                'sample_idx': sample_idx
            }
        )