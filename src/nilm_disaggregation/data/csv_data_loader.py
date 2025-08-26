#!/usr/bin/env python3
"""
CSV数据加载器 - 处理AMPds2 CSV格式数据
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple, Optional

class AMPds2CSVDataset(Dataset):
    """AMPds2 CSV数据集加载器"""
    
    def __init__(self, 
                 data_dir: str,
                 sequence_length: int = 128,
                 max_samples: int = 50000,
                 train: bool = True,
                 train_ratio: float = 0.8):
        """
        初始化CSV数据集
        
        Args:
            data_dir: 数据目录路径
            sequence_length: 序列长度
            max_samples: 最大样本数
            train: 是否为训练集
            train_ratio: 训练集比例
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.max_samples = max_samples
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 加载数据
        self._load_csv_data()
        
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
        
        self.logger.info(f"数据集大小: {self.end_idx - self.start_idx} 样本")
    
    def _load_csv_data(self):
        """加载CSV数据文件"""
        self.logger.info(f"正在加载CSV数据: {self.data_dir}")
        
        # 查找可用的电力数据文件
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        electricity_files = [f for f in csv_files if 'Electricity' in f or any(code in f for code in ['WHE', 'B1E', 'B2E', 'BME', 'CDE', 'CWE', 'DNE', 'DWE', 'EBE', 'EQE', 'FGE'])]
        
        self.logger.info(f"找到 {len(electricity_files)} 个电力数据文件")
        
        if not electricity_files:
            self.logger.warning("未找到电力数据文件，使用合成数据")
            self._create_synthetic_data()
            return
        
        # 加载主电力数据（整屋消耗）
        main_power_file = None
        for file in electricity_files:
            if 'WHE' in file or 'Electricity_WHE' in file:
                main_power_file = file
                break
        
        if main_power_file:
            self.logger.info(f"加载主电力数据: {main_power_file}")
            main_df = pd.read_csv(os.path.join(self.data_dir, main_power_file))
            # 假设功率数据在第二列（第一列通常是时间戳）
            if len(main_df.columns) >= 2:
                self.main_power = main_df.iloc[:self.max_samples, 1].values.astype(np.float32)
            else:
                self.main_power = main_df.iloc[:self.max_samples, 0].values.astype(np.float32)
        else:
            self.logger.warning("未找到主电力数据文件，使用第一个可用文件")
            first_file = electricity_files[0]
            df = pd.read_csv(os.path.join(self.data_dir, first_file))
            self.main_power = df.iloc[:self.max_samples, 1].values.astype(np.float32)
        
        # 加载电器数据
        self.appliance_data = {}
        appliance_codes = ['B1E', 'B2E', 'BME', 'CDE', 'CWE', 'DNE', 'DWE', 'EBE', 'EQE', 'FGE']
        
        for code in appliance_codes:
            appliance_file = None
            for file in electricity_files:
                if code in file:
                    appliance_file = file
                    break
            
            if appliance_file:
                try:
                    df = pd.read_csv(os.path.join(self.data_dir, appliance_file))
                    if len(df.columns) >= 2:
                        data = df.iloc[:self.max_samples, 1].values.astype(np.float32)
                    else:
                        data = df.iloc[:self.max_samples, 0].values.astype(np.float32)
                    self.appliance_data[code] = data
                    self.logger.info(f"加载电器数据 {code}: {len(data)} 样本")
                except Exception as e:
                    self.logger.warning(f"加载电器数据 {code} 失败: {e}")
        
        # 如果没有找到足够的电器数据，创建合成数据
        if len(self.appliance_data) < 3:
            self.logger.warning("电器数据不足，补充合成数据")
            self._supplement_appliance_data()
    
    def _supplement_appliance_data(self):
        """补充电器数据"""
        target_appliances = 5
        current_count = len(self.appliance_data)
        
        for i in range(current_count, target_appliances):
            appliance_name = f"Appliance_{i+1}"
            # 基于主功率创建合成电器数据
            ratio = np.random.uniform(0.1, 0.3)  # 电器功率占总功率的比例
            noise = np.random.normal(0, 0.1, len(self.main_power))
            appliance_power = self.main_power * ratio + noise
            appliance_power = np.maximum(appliance_power, 0)  # 确保非负
            self.appliance_data[appliance_name] = appliance_power.astype(np.float32)
    
    def _create_synthetic_data(self):
        """创建完全合成的数据"""
        num_samples = self.max_samples or 10000
        
        # 创建合成主功率数据
        self.main_power = np.random.normal(2000, 500, num_samples).astype(np.float32)
        self.main_power = np.maximum(self.main_power, 0)
        
        # 创建合成电器数据
        self.appliance_data = {}
        for i in range(5):
            appliance_name = f"Synthetic_Appliance_{i+1}"
            base_power = np.random.uniform(100, 400)
            appliance_power = np.random.exponential(base_power, num_samples).astype(np.float32)
            self.appliance_data[appliance_name] = appliance_power
        
        self.logger.info(f"创建了完全合成的数据: {num_samples} 个样本")
    
    def _preprocess_data(self):
        """数据预处理"""
        # 处理异常值
        self.main_power = np.nan_to_num(self.main_power, nan=0.0, posinf=0.0, neginf=0.0)
        
        for name, data in self.appliance_data.items():
            self.appliance_data[name] = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 确保所有数据长度一致
        min_length = min(len(self.main_power), 
                        min(len(data) for data in self.appliance_data.values()))
        
        self.main_power = self.main_power[:min_length]
        for name in self.appliance_data:
            self.appliance_data[name] = self.appliance_data[name][:min_length]
        
        self.logger.info(f"预处理完成，数据长度: {len(self.main_power)}")
        self.logger.info(f"电器数量: {len(self.appliance_data)}")
    
    def __len__(self):
        return self.end_idx - self.start_idx
    
    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        
        # 获取输入序列（主功率）
        input_seq = self.main_power[actual_idx:actual_idx + self.sequence_length]
        
        # 获取目标序列（电器功率和状态）
        appliance_powers = []
        appliance_states = []
        
        for appliance_data in self.appliance_data.values():
            power_seq = appliance_data[actual_idx:actual_idx + self.sequence_length]
            appliance_powers.append(power_seq)
            
            # 简单的状态检测：功率大于阈值则认为设备开启
            threshold = np.mean(appliance_data) * 0.1
            state_seq = (power_seq > threshold).astype(np.float32)
            appliance_states.append(state_seq)
        
        # 转换为张量
        input_tensor = torch.FloatTensor(input_seq).unsqueeze(-1)  # [seq_len, 1]
        power_tensor = torch.FloatTensor(np.array(appliance_powers).T)  # [seq_len, num_appliances]
        state_tensor = torch.FloatTensor(np.array(appliance_states).T)  # [seq_len, num_appliances]
        
        # 返回格式与其他数据集保持一致
        targets = {
            'power': power_tensor,
            'state': state_tensor
        }
        
        return input_tensor, targets

def test_csv_dataset():
    """测试CSV数据集加载"""
    data_dir = '/Users/siyili/Workspace/DisaggNet/Dataset/dataverse_files'
    
    print("=== 测试CSV数据集加载 ===")
    
    # 创建数据集
    dataset = AMPds2CSVDataset(
        data_dir=data_dir,
        sequence_length=128,
        max_samples=10000,
        train=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 测试数据访问
    sample_input, sample_targets = dataset[0]
    print(f"输入形状: {sample_input.shape}")
    print(f"功率目标形状: {sample_targets['power'].shape}")
    print(f"状态目标形状: {sample_targets['state'].shape}")
    
    # 测试DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch_idx, (batch_input, batch_targets) in enumerate(dataloader):
        print(f"批次 {batch_idx}:")
        print(f"  输入形状: {batch_input.shape}")
        print(f"  功率目标形状: {batch_targets['power'].shape}")
        print(f"  状态目标形状: {batch_targets['state'].shape}")
        break
    
    print("CSV数据集测试完成!")

if __name__ == '__main__':
    test_csv_dataset()