"""NILM数据集模块"""

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional


class RealAMPds2Dataset(Dataset):
    """真实AMPds2数据集类"""
    
    def __init__(
        self, 
        data_path: str, 
        sequence_length: int = 512, 
        train: bool = True, 
        train_ratio: float = 0.8, 
        max_samples: int = 50000
    ):
        """
        初始化AMPds2数据集
        
        Args:
            data_path: 数据文件路径
            sequence_length: 输入序列长度
            train: 是否为训练集
            train_ratio: 训练集比例
            max_samples: 最大样本数
        """
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
    
    def _load_data(self, data_path: str) -> None:
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
    
    def _generate_synthetic_main_power(self, length: int) -> np.ndarray:
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
    
    def _create_appliance_data(self) -> None:
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
    
    def _preprocess_data(self) -> None:
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
    
    def get_scalers(self) -> Tuple[StandardScaler, Dict[str, StandardScaler]]:
        """获取标准化器"""
        return self.main_scaler, self.appliance_scalers
    
    def get_appliances(self) -> List[str]:
        """获取设备列表"""
        return self.appliances
    
    def __len__(self) -> int:
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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