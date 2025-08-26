
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import logging

class ImprovedAMPds2Dataset(Dataset):
    """改进的AMPds2数据集类，解决HDF5加载问题"""
    
    def __init__(self, data_path, sequence_length=128, window_size=64, max_samples=None):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.max_samples = max_samples
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 尝试加载数据
        self.data_loaded = self._load_data()
        
        if not self.data_loaded:
            self.logger.warning("无法加载真实数据，使用合成数据")
            self._create_synthetic_data()
    
    def _load_data(self):
        """尝试多种方法加载真实数据"""
        if not os.path.exists(self.data_path):
            self.logger.error(f"数据文件不存在: {self.data_path}")
            return False
        
        # 尝试不同的HDF5打开方式
        open_methods = [
            {'mode': 'r'},
            {'mode': 'r', 'libver': 'latest'},
            {'mode': 'r', 'locking': False},
            {'mode': 'r', 'swmr': True}
        ]
        
        for i, method in enumerate(open_methods):
            try:
                self.logger.info(f"尝试方法 {i+1}: {method}")
                with h5py.File(self.data_path, **method) as f:
                    self._extract_data_from_hdf5(f)
                    self.logger.info("成功加载真实数据")
                    return True
            except Exception as e:
                self.logger.warning(f"方法 {i+1} 失败: {str(e)}")
                continue
        
        return False
    
    def _extract_data_from_hdf5(self, f):
        """从HDF5文件中提取数据"""
        # 主功率数据
        main_power_key = 'WHE'  # 整屋电力消耗
        if main_power_key in f:
            self.main_power = f[main_power_key][:self.max_samples] if self.max_samples else f[main_power_key][:]
        else:
            raise ValueError(f"找不到主功率数据键: {main_power_key}")
        
        # 电器功率数据
        appliance_keys = ['B1E', 'B2E', 'BME', 'CDE', 'CWE', 'DNE', 'DWE', 'EBE', 'EQE', 'FGE']
        self.appliance_data = {}
        
        for key in appliance_keys:
            if key in f:
                data = f[key][:self.max_samples] if self.max_samples else f[key][:]
                self.appliance_data[key] = data
        
        if not self.appliance_data:
            raise ValueError("找不到任何电器数据")
        
        self.logger.info(f"加载了 {len(self.appliance_data)} 个电器的数据")
        self.logger.info(f"主功率数据形状: {self.main_power.shape}")
    
    def _create_synthetic_data(self):
        """创建合成数据作为备选"""
        num_samples = self.max_samples or 10000
        
        # 创建合成主功率数据
        self.main_power = np.random.normal(2000, 500, num_samples).astype(np.float32)
        self.main_power = np.maximum(self.main_power, 0)  # 确保非负
        
        # 创建合成电器数据
        appliance_names = ['B1E', 'B2E', 'BME', 'CDE', 'CWE']
        self.appliance_data = {}
        
        for name in appliance_names:
            # 每个电器有不同的功率特征
            base_power = np.random.uniform(50, 300)
            appliance_power = np.random.exponential(base_power, num_samples).astype(np.float32)
            self.appliance_data[name] = appliance_power
        
        self.logger.info(f"创建了合成数据: {num_samples} 个样本")
    
    def __len__(self):
        return max(0, len(self.main_power) - self.sequence_length + 1)
    
    def __getitem__(self, idx):
        # 获取输入序列
        input_seq = self.main_power[idx:idx + self.sequence_length]
        
        # 获取目标电器功率
        target_powers = []
        for appliance_data in self.appliance_data.values():
            target_power = appliance_data[idx:idx + self.sequence_length]
            target_powers.append(target_power)
        
        # 转换为张量
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(-1)  # [seq_len, 1]
        target_tensor = torch.FloatTensor(np.array(target_powers).T)  # [seq_len, num_appliances]
        
        return input_tensor, target_tensor
