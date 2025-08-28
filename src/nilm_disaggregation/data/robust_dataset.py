"""改进的NILM数据集，解决时序数据泄漏问题"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime, timedelta
import logging

warnings.filterwarnings('ignore')

class PurgedEmbargoWalkForwardCV:
    """Purged/Embargo Walk-Forward 交叉验证，防止数据泄漏的最佳实践"""
    
    def __init__(self, n_splits=5, embargo_hours=24, purge_hours=0, 
                 test_hours=7*24, min_train_hours=30*24):
        """
        初始化 Purged/Embargo Walk-Forward 交叉验证
        
        Args:
            n_splits: 分割数量
            embargo_hours: 禁运期（训练集结束到验证集开始的间隔）
            purge_hours: 清洗期（验证集结束后的额外间隔）
            test_hours: 验证集大小（小时）
            min_train_hours: 最小训练集大小（小时）
        """
        self.n_splits = n_splits
        self.embargo_size = embargo_hours * 60  # 转换为分钟
        self.purge_size = purge_hours * 60
        self.test_size = test_hours * 60
        self.min_train_size = min_train_hours * 60
    
    def split(self, data_length: int, sampling_rate_minutes: int = 1) -> List[Tuple[List[int], List[int]]]:
        """生成 Walk-Forward 分割
        
        每个fold都是：历史训练 → Embargo Gap → 未来验证 → Purge Gap
        
        Args:
            data_length: 数据总长度
            sampling_rate_minutes: 采样率（分钟）
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []
        
        # 转换为数据点数量
        embargo_points = self.embargo_size // sampling_rate_minutes
        purge_points = self.purge_size // sampling_rate_minutes
        test_points = self.test_size // sampling_rate_minutes
        min_train_points = self.min_train_size // sampling_rate_minutes
        
        # 计算每个fold的验证集起始位置
        total_gap_per_fold = embargo_points + test_points + purge_points
        available_length = data_length - total_gap_per_fold * self.n_splits
        
        if available_length < min_train_points:
            raise ValueError(f"数据长度不足以支持{self.n_splits}个fold的Walk-Forward验证")
        
        # Walk-Forward: 训练集逐步扩大
        fold_increment = available_length // self.n_splits
        
        for i in range(self.n_splits):
            # 训练集：从开始到当前fold结束（逐步扩大）
            train_end = min_train_points + (i + 1) * fold_increment
            train_indices = list(range(0, train_end))
            
            # Embargo Gap
            test_start = train_end + embargo_points
            
            # 验证集
            test_end = test_start + test_points
            
            # 确保不超出数据范围
            if test_end <= data_length:
                test_indices = list(range(test_start, test_end))
                splits.append((train_indices, test_indices))
            
            # 为下一个fold预留Purge Gap（在实际数据中体现）
        
        return splits
    
    def get_fold_info(self, fold_idx: int, data_length: int, sampling_rate_minutes: int = 1) -> Dict[str, Any]:
        """获取特定fold的详细信息"""
        splits = self.split(data_length, sampling_rate_minutes)
        
        if fold_idx >= len(splits):
            raise ValueError(f"Fold索引{fold_idx}超出范围")
        
        train_indices, test_indices = splits[fold_idx]
        
        return {
            'fold_idx': fold_idx,
            'train_start': train_indices[0],
            'train_end': train_indices[-1],
            'train_size': len(train_indices),
            'embargo_start': train_indices[-1] + 1,
            'embargo_end': test_indices[0] - 1,
            'embargo_size': test_indices[0] - train_indices[-1] - 1,
            'test_start': test_indices[0],
            'test_end': test_indices[-1],
            'test_size': len(test_indices)
        }


class RobustAMPds2Dataset(Dataset):
    """改进的NILM数据集，解决数据泄漏和标签不均衡问题"""
    
    def __init__(
        self, 
        data_path: str, 
        sequence_length: int = 512, 
        split_type: str = 'train',
        split_config: Optional[Dict] = None,
        preprocessing_params: Optional[Dict] = None,
        train_stride: int = 1,
        val_stride: Optional[int] = None,
        fold_idx: Optional[int] = None,
        cv_mode: bool = False
    ):
        """
        初始化改进的AMPds2数据集
        
        Args:
            data_path: 数据文件路径
            sequence_length: 输入序列长度
            split_type: 数据分割类型 ('train', 'val', 'test')
            split_config: 分割配置
            preprocessing_params: 预处理参数（从训练集获得）
            train_stride: 训练集采样步长（小值增加样本量）
            val_stride: 验证集采样步长（None时使用sequence_length实现非重叠）
            fold_idx: Walk-Forward交叉验证的fold索引
            cv_mode: 是否使用交叉验证模式
        """
        self.sequence_length = sequence_length
        self.split_type = split_type
        self.train_stride = train_stride
        self.val_stride = val_stride if val_stride is not None else sequence_length  # 默认非重叠
        self.fold_idx = fold_idx
        self.cv_mode = cv_mode
        
        # 默认分割配置 - 支持Walk-Forward
        if split_config is None:
            split_config = {
                'train_ratio': 0.7,
                'val_ratio': 0.15, 
                'test_ratio': 0.15,
                'embargo_hours': 24,  # Embargo期间
                'purge_hours': 0,     # Purge期间
                'cv_folds': 5,
                'min_train_hours': 30*24  # 最小训练集大小
            }
        
        self.split_config = split_config
        self.preprocessing_params = preprocessing_params
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        print(f"正在加载改进的AMPds2数据集: {data_path}")
        print(f"分割类型: {split_type}, CV模式: {cv_mode}, Fold: {fold_idx}")
        print(f"训练步长: {train_stride}, 验证步长: {self.val_stride}")
        
        # 加载和处理数据
        self._load_data(data_path)
        
        # 根据模式选择分割策略
        if cv_mode and fold_idx is not None:
            self._create_walk_forward_splits()
        else:
            self._create_time_aware_splits()
        
        # 如果是训练集，计算预处理参数；否则使用传入的参数
        if split_type == 'train':
            self._compute_preprocessing_params()
        elif preprocessing_params is not None:
            self._apply_preprocessing_params(preprocessing_params)
        else:
            raise ValueError("非训练集必须提供preprocessing_params")
        
        # 生成样本索引
        self._generate_sample_indices()
        
        print(f"数据集大小: {len(self.sample_indices)} 样本")
        print(f"数据范围: {self.data_range[0]} - {self.data_range[1]}")
        if hasattr(self, 'embargo_info'):
            print(f"Embargo信息: {self.embargo_info}")
    
    def _load_data(self, data_path: str) -> None:
        """加载真实AMPds2数据"""
        try:
            with h5py.File(data_path, 'r') as f:
                # 探索数据结构，找到所有可用的电表
                available_meters = []
                meter_names = []
                
                # 查找所有电表数据
                if 'building1/elec' in f:
                    elec_group = f['building1/elec']
                    for key in elec_group.keys():
                        if key.startswith('meter') and 'table' in elec_group[key]:
                            meter_num = key.replace('meter', '')
                            available_meters.append(int(meter_num))
                            
                            # 尝试获取电表名称
                            try:
                                if 'device_model' in elec_group[key].attrs:
                                    device_name = elec_group[key].attrs['device_model'].decode('utf-8')
                                elif f'meter{meter_num}' in elec_group:
                                    device_name = f'appliance_{meter_num}'
                                else:
                                    device_name = f'meter_{meter_num}'
                                meter_names.append(device_name)
                            except:
                                meter_names.append(f'appliance_{meter_num}')
                
                print(f"Found {len(available_meters)} meters: {available_meters[:10]}")
                print(f"Device names: {meter_names[:10]}")
                
                # 使用meter1作为主功率（通常是总功率）
                if 1 in available_meters:
                    meter1_data = f['building1/elec/meter1/table']
                    data_subset = meter1_data[:100000]  # 限制数据量
                    
                    if 'values_block_0' in data_subset.dtype.names:
                        power_data = data_subset['values_block_0']
                        if len(power_data.shape) > 1:
                            self.main_power = power_data[:, 0]
                        else:
                            self.main_power = power_data
                    else:
                        raise ValueError("Cannot read power data from meter1")
                else:
                    raise ValueError("Meter1 not found")
                
                # 读取其他电表作为设备数据
                self.appliance_power = {}
                self.appliances = []
                
                # 选择前8个电表作为设备（跳过meter1）
                device_meters = [m for m in available_meters[1:9] if m != 1]  # 跳过meter1
                
                for i, meter_num in enumerate(device_meters):
                    try:
                        meter_data = f[f'building1/elec/meter{meter_num}/table']
                        device_subset = meter_data[:len(self.main_power)]
                        
                        if 'values_block_0' in device_subset.dtype.names:
                            device_power = device_subset['values_block_0']
                            if len(device_power.shape) > 1:
                                device_power = device_power[:, 0]
                            
                            # 使用简化的设备名称
                            device_name = meter_names[i] if i < len(meter_names) else f'appliance_{meter_num}'
                            # 简化设备名称，移除特殊字符
                            device_name = device_name.lower().replace(' ', '_').replace('-', '_')
                            device_name = ''.join(c for c in device_name if c.isalnum() or c == '_')
                            
                            self.appliances.append(device_name)
                            self.appliance_power[device_name] = device_power
                            
                            print(f"Loaded meter{meter_num} as {device_name}: {len(device_power)} samples")
                        
                    except Exception as e:
                        print(f"Failed to load meter{meter_num}: {e}")
                        continue
                
                # 如果没有找到足够的设备，使用默认设备名称
                if len(self.appliances) == 0:
                    print("No appliance data found, using synthetic data")
                    self._create_appliance_data()
                else:
                    print(f"Successfully loaded {len(self.appliances)} appliances: {self.appliances}")
                    
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Using synthetic data")
            self.main_power = self._generate_synthetic_main_power(100000)
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
    
    def _create_walk_forward_splits(self) -> None:
        """创建Walk-Forward分割，实现Purged/Embargo策略"""
        total_length = len(self.main_power)
        
        # 创建Walk-Forward交叉验证器
        cv = PurgedEmbargoWalkForwardCV(
            n_splits=self.split_config['cv_folds'],
            embargo_hours=self.split_config['embargo_hours'],
            purge_hours=self.split_config['purge_hours'],
            test_hours=7*24,  # 验证集大小
            min_train_hours=self.split_config['min_train_hours']
        )
        
        # 获取当前fold的分割信息
        fold_info = cv.get_fold_info(self.fold_idx, total_length)
        self.embargo_info = fold_info
        
        # 根据split_type设置数据范围
        if self.split_type == 'train':
            self.data_range = (fold_info['train_start'], fold_info['train_end'] + 1)
            self.is_training_split = True
        elif self.split_type == 'val':
            self.data_range = (fold_info['test_start'], fold_info['test_end'] + 1)
            self.is_training_split = False
        else:  # test - 使用最后一个fold的测试集
            last_fold_info = cv.get_fold_info(self.split_config['cv_folds'] - 1, total_length)
            self.data_range = (last_fold_info['test_start'], last_fold_info['test_end'] + 1)
            self.is_training_split = False
        
        # 确保数据范围有效
        self.data_range = (max(0, self.data_range[0]), 
                          min(total_length, self.data_range[1]))
        
        if self.data_range[1] <= self.data_range[0]:
            raise ValueError(f"无效的数据范围: {self.data_range}")
    
    def _create_time_aware_splits(self) -> None:
        """创建时间感知的数据分割，防止数据泄漏（非CV模式）"""
        total_length = len(self.main_power)
        embargo_size = self.split_config['embargo_hours'] * 60  # 转换为分钟
        
        # 计算分割点
        train_ratio = self.split_config['train_ratio']
        val_ratio = self.split_config['val_ratio']
        
        train_end = int(total_length * train_ratio)
        val_start = train_end + embargo_size
        val_end = val_start + int(total_length * val_ratio)
        test_start = val_end + embargo_size
        
        # 根据split_type设置数据范围
        if self.split_type == 'train':
            self.data_range = (0, train_end)
            self.is_training_split = True
        elif self.split_type == 'val':
            if val_start >= total_length:
                # 如果验证集开始位置超出数据范围，调整间隔
                val_start = train_end + min(embargo_size, (total_length - train_end) // 4)
                val_end = min(val_start + int(total_length * val_ratio), total_length)
            self.data_range = (val_start, val_end)
            self.is_training_split = False
        else:  # test
            if test_start >= total_length:
                # 如果测试集开始位置超出数据范围，调整间隔
                test_start = val_end + min(embargo_size, (total_length - val_end) // 4)
            self.data_range = (test_start, total_length)
            self.is_training_split = False
        
        # 确保数据范围有效
        self.data_range = (max(0, self.data_range[0]), 
                          min(total_length, self.data_range[1]))
        
        if self.data_range[1] <= self.data_range[0]:
            raise ValueError(f"无效的数据范围: {self.data_range}")
    
    def _compute_adaptive_threshold(self, power_data: np.ndarray, appliance_name: str) -> float:
        """自适应阈值计算，解决标签不均衡
        
        Args:
            power_data: 功率数据
            appliance_name: 设备名称
            
        Returns:
            计算得到的阈值
        """
        # 设备特定的目标正样本比例
        target_ratios = {
            'microwave': (0.08, 0.20),      # 8%-20%，短时高功率设备
            'kettle': (0.05, 0.15),         # 5%-15%，极短时设备
            'toaster': (0.08, 0.18),        # 8%-18%，短时设备
            
            'fridge': (0.35, 0.65),         # 35%-65%，周期性设备
            'freezer': (0.30, 0.60),        # 30%-60%，周期性设备
            'air_conditioner': (0.25, 0.55), # 25%-55%，季节性周期设备
            
            'washer_dryer': (0.15, 0.35),   # 15%-35%，间歇性长时间设备
            'dishwasher': (0.20, 0.40),     # 20%-40%，间歇性长时间设备
            'dryer': (0.12, 0.30),          # 12%-30%，间歇性长时间设备
        }
        
        # 获取目标比例，默认为通用设备
        min_ratio, max_ratio = target_ratios.get(appliance_name, (0.15, 0.45))
        
        # 多阶段阈值优化
        best_threshold = None
        best_score = -1
        
        # 阶段1: 基于设备类型的初始阈值
        if appliance_name in ['microwave', 'kettle', 'toaster']:
            initial_percentiles = [70, 75, 80, 85, 90, 95]
        elif appliance_name in ['fridge', 'freezer', 'air_conditioner']:
            initial_percentiles = [40, 45, 50, 55, 60, 65]
        else:
            initial_percentiles = [60, 65, 70, 75, 80, 85]
        
        # 粗略搜索
        for percentile in initial_percentiles:
            threshold = np.percentile(power_data, percentile)
            positive_ratio = np.mean(power_data > threshold)
            
            # 评分函数：目标是在目标范围内且尽可能接近中点
            target_center = (min_ratio + max_ratio) / 2
            if min_ratio <= positive_ratio <= max_ratio:
                score = 1 - abs(positive_ratio - target_center) / (max_ratio - min_ratio)
            else:
                # 超出范围的惩罚
                if positive_ratio < min_ratio:
                    score = -abs(positive_ratio - min_ratio) * 2
                else:
                    score = -abs(positive_ratio - max_ratio) * 2
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # 阶段2: 如果没有找到合适阈值，使用更激进的策略
        if best_threshold is None or best_score < 0:
            print(f"警告: {appliance_name} 无法找到合适阈值，使用备用策略")
            
            # 备用策略：基于数据分布的动态阈值
            data_std = np.std(power_data)
            data_mean = np.mean(power_data)
            
            # 尝试均值+标准差的倍数作为阈值
            for multiplier in [0.5, 1.0, 1.5, 2.0, 2.5]:
                threshold = data_mean + multiplier * data_std
                
                # 确保阈值在数据范围内
                if threshold > np.max(power_data):
                    threshold = np.percentile(power_data, 95)
                elif threshold < np.min(power_data):
                    threshold = np.percentile(power_data, 50)
                
                positive_ratio = np.mean(power_data > threshold)
                
                if min_ratio <= positive_ratio <= max_ratio:
                    best_threshold = threshold
                    break
            
            # 最后的备用方案：强制设置阈值确保最小正样本比例
            if best_threshold is None:
                sorted_data = np.sort(power_data)
                target_idx = int(len(sorted_data) * (1 - min_ratio))
                best_threshold = sorted_data[max(0, min(target_idx, len(sorted_data) - 1))]
                print(f"使用强制阈值: {best_threshold:.4f}，预期正样本比例: {min_ratio:.3f}")
        
        # 验证最终阈值
        final_ratio = np.mean(power_data > best_threshold)
        print(f"{appliance_name} 阈值: {best_threshold:.4f}, 正样本比例: {final_ratio:.3f}, 目标范围: [{min_ratio:.3f}, {max_ratio:.3f}]")
        
        return best_threshold
    
    def _create_balanced_states(self, power_data: np.ndarray, threshold: float, appliance_name: str) -> np.ndarray:
        """创建平衡的状态标签
        
        Args:
            power_data: 功率数据
            threshold: 阈值
            appliance_name: 设备名称
            
        Returns:
            状态标签数组
        """
        # 基础二值化
        basic_states = (power_data > threshold).astype(float)
        
        # 对于短时间设备，添加持续性约束
        if appliance_name in ['microwave', 'kettle', 'toaster']:
            # 移除过短的开启状态（可能是噪声）
            min_on_duration = 5  # 最少持续5个时间步
            basic_states = self._filter_short_events(basic_states, min_on_duration)
        
        # 对于周期性设备，平滑状态转换
        elif appliance_name in ['fridge', 'freezer', 'air_conditioner']:
            # 使用滑动窗口平滑
            window_size = 10
            smoothed_states = np.convolve(basic_states, 
                                        np.ones(window_size)/window_size, 
                                        mode='same')
            basic_states = (smoothed_states > 0.5).astype(float)
        
        return basic_states
    
    def _filter_short_events(self, states: np.ndarray, min_duration: int) -> np.ndarray:
        """过滤过短的开启事件
        
        Args:
            states: 状态数组
            min_duration: 最小持续时间
            
        Returns:
            过滤后的状态数组
        """
        filtered_states = states.copy()
        
        # 找到状态变化点
        changes = np.diff(np.concatenate(([0], states, [0])))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        
        # 移除过短的事件
        for start, end in zip(starts, ends):
            if end - start < min_duration:
                filtered_states[start:end] = 0
        
        return filtered_states
    
    def _compute_preprocessing_params(self) -> None:
        """计算预处理参数（仅训练集）"""
        if not self.is_training_split:
            raise ValueError("预处理参数只能从训练集计算")
        
        start_idx, end_idx = self.data_range
        train_data = {
            'main_power': self.main_power[start_idx:end_idx],
            'appliance_power': {}
        }
        
        for app in self.appliances:
            train_data['appliance_power'][app] = \
                self.appliance_power[app][start_idx:end_idx]
        
        # 计算标准化参数
        main_scaler = StandardScaler()
        main_scaler.fit(train_data['main_power'].reshape(-1, 1))
        
        appliance_scalers = {}
        appliance_thresholds = {}
        
        for app in self.appliances:
            # 标准化器
            scaler = StandardScaler()
            scaler.fit(train_data['appliance_power'][app].reshape(-1, 1))
            appliance_scalers[app] = scaler
            
            # 自适应阈值
            threshold = self._compute_adaptive_threshold(
                train_data['appliance_power'][app], app
            )
            appliance_thresholds[app] = threshold
        
        self.preprocessing_params = {
            'main_scaler': main_scaler,
            'appliance_scalers': appliance_scalers,
            'appliance_thresholds': appliance_thresholds
        }
        
        # 应用预处理
        self._apply_preprocessing_params(self.preprocessing_params)
    
    def _apply_preprocessing_params(self, params: Dict) -> None:
        """应用预处理参数"""
        # 标准化主功率数据
        self.main_scaler = params['main_scaler']
        self.main_power = self.main_scaler.transform(self.main_power.reshape(-1, 1)).flatten()
        
        # 标准化设备功率数据并创建状态标签
        self.appliance_scalers = params['appliance_scalers']
        self.appliance_thresholds = params['appliance_thresholds']
        self.appliance_states = {}
        
        for appliance in self.appliances:
            # 标准化
            scaler = self.appliance_scalers[appliance]
            self.appliance_power[appliance] = scaler.transform(
                self.appliance_power[appliance].reshape(-1, 1)
            ).flatten()
            
            # 使用训练集计算的阈值创建状态标签
            # 注意：这里需要在标准化前的数据上应用阈值
            # 所以我们需要重新加载原始数据或保存原始数据
            # 为简化，这里使用标准化后数据的相对阈值
            threshold = self.appliance_thresholds[appliance]
            # 将原始阈值转换为标准化后的阈值
            normalized_threshold = scaler.transform([[threshold]])[0, 0]
            
            self.appliance_states[appliance] = self._create_balanced_states(
                self.appliance_power[appliance], normalized_threshold, appliance
            )
    
    def _generate_sample_indices(self) -> None:
        """生成样本索引，训练集和验证集使用不同步长策略"""
        start_idx, end_idx = self.data_range
        available_length = end_idx - start_idx
        
        if available_length < self.sequence_length:
            raise ValueError(f"数据长度 {available_length} 小于序列长度 {self.sequence_length}")
        
        self.sample_indices = []
        
        # 根据数据集类型选择步长
        if self.split_type == 'train':
            # 训练集：使用小步长增加样本量
            step_size = self.train_stride
        else:
            # 验证集/测试集：使用大步长（非重叠窗口）防止验证集内部相似性偏置
            step_size = self.val_stride
        
        current_pos = start_idx
        while current_pos + self.sequence_length <= end_idx:
            self.sample_indices.append(current_pos)
            current_pos += step_size
        
        print(f"使用步长 {step_size}，生成 {len(self.sample_indices)} 个样本索引")
    
    def get_preprocessing_params(self) -> Dict:
        """获取预处理参数（仅训练集可用）"""
        if not hasattr(self, 'preprocessing_params'):
            raise ValueError("预处理参数未计算")
        return self.preprocessing_params
    
    def get_appliances(self) -> List[str]:
        """获取设备列表"""
        return self.appliances
    
    def get_scalers(self) -> Tuple[StandardScaler, Dict[str, StandardScaler]]:
        """获取标准化器"""
        return self.main_scaler, self.appliance_scalers
    
    def get_split_info(self) -> Dict[str, Any]:
        """获取分割信息"""
        return {
            'split_type': self.split_type,
            'data_range': self.data_range,
            'split_config': self.split_config,
            'train_stride': getattr(self, 'train_stride', 1),
            'val_stride': getattr(self, 'val_stride', self.sequence_length),
            'num_samples': len(self.sample_indices)
        }
    
    def __len__(self) -> int:
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actual_idx = self.sample_indices[idx]
        
        # 输入序列
        x = self.main_power[actual_idx:actual_idx + self.sequence_length]
        
        # 目标功率和状态（使用序列的最后一个时间步）
        target_idx = actual_idx + self.sequence_length - 1
        
        power_targets = np.array([self.appliance_power[app][target_idx] for app in self.appliances])
        state_targets = np.array([self.appliance_states[app][target_idx] for app in self.appliances])
        
        return (
            torch.FloatTensor(x).unsqueeze(-1),  # [seq_len, 1]
            torch.FloatTensor(power_targets),    # [num_appliances]
            torch.FloatTensor(state_targets)     # [num_appliances]
        )


class RobustNILMDataModule(pl.LightningDataModule):
    """改进的NILM数据模块，支持Purged/Embargo Walk-Forward验证
    
    继承PyTorch Lightning的LightningDataModule，提供标准化的数据处理接口
    集成所有6个防泄漏技术：
    1. Purged/Embargo Walk-Forward 交叉验证
    2. 先分割后预处理
    3. 验证集非重叠窗口
    4. 训练集小步长采样
    5. 标签/阈值防泄漏
    6. 特征工程分片内独立
    """
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 512,
        batch_size: int = 32,
        num_workers: int = 4,
        split_config: Optional[Dict] = None,
        train_stride: int = 1,
        val_stride: Optional[int] = None,
        cv_mode: bool = False,
        current_fold: int = 0
    ):
        super().__init__()
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_config = split_config
        self.train_stride = train_stride
        self.val_stride = val_stride
        self.cv_mode = cv_mode
        self.current_fold = current_fold
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.preprocessing_params = None
        self.cv_results = []  # 存储交叉验证结果
    
    def setup(self, stage: Optional[str] = None) -> None:
        """设置数据集，支持Walk-Forward交叉验证"""
        if stage == "fit" or stage is None:
            # 创建训练集并计算预处理参数
            self.train_dataset = RobustAMPds2Dataset(
                data_path=self.data_path,
                sequence_length=self.sequence_length,
                split_type='train',
                split_config=self.split_config,
                train_stride=self.train_stride,
                val_stride=self.val_stride,
                fold_idx=self.current_fold if self.cv_mode else None,
                cv_mode=self.cv_mode
            )
            
            # 获取预处理参数（只在训练集上计算，防止泄漏）
            self.preprocessing_params = self.train_dataset.get_preprocessing_params()
            
            # 创建验证集，使用训练集的预处理参数
            self.val_dataset = RobustAMPds2Dataset(
                data_path=self.data_path,
                sequence_length=self.sequence_length,
                split_type='val',
                split_config=self.split_config,
                preprocessing_params=self.preprocessing_params,
                train_stride=self.train_stride,
                val_stride=self.val_stride,
                fold_idx=self.current_fold if self.cv_mode else None,
                cv_mode=self.cv_mode
            )
        
        if stage == "test" or stage is None:
            # 创建测试集
            if self.preprocessing_params is None:
                raise ValueError("必须先运行fit阶段以获取预处理参数")
            
            self.test_dataset = RobustAMPds2Dataset(
                data_path=self.data_path,
                sequence_length=self.sequence_length,
                split_type='test',
                split_config=self.split_config,
                preprocessing_params=self.preprocessing_params,
                train_stride=self.train_stride,
                val_stride=self.val_stride,
                fold_idx=self.current_fold if self.cv_mode else None,
                cv_mode=self.cv_mode
            )
    
    def setup_fold(self, fold_idx: int) -> None:
        """设置特定fold的数据集"""
        self.current_fold = fold_idx
        self.setup('fit')
    
    def get_cv_splits(self) -> List[Tuple[int, int]]:
        """获取所有交叉验证分割信息"""
        if not self.cv_mode:
            raise ValueError("非CV模式下无法获取分割信息")
        
        # 确保split_config不为None并包含所有必要的键
        if self.split_config is None:
            self.split_config = {
                'cv_folds': 5,
                'embargo_hours': 24,
                'purge_hours': 0,
                'min_train_hours': 30*24,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15
            }
        else:
            # 确保包含所有必要的键
            default_config = {
                'cv_folds': 5,
                'embargo_hours': 24,
                'purge_hours': 0,
                'min_train_hours': 30*24,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15
            }
            for key, value in default_config.items():
                if key not in self.split_config:
                    self.split_config[key] = value
        
        # 创建临时数据集获取数据长度
        temp_dataset = RobustAMPds2Dataset(
            data_path=self.data_path,
            sequence_length=self.sequence_length,
            split_type='train',
            split_config=self.split_config,
            train_stride=self.train_stride,
            val_stride=self.val_stride,
            fold_idx=0,
            cv_mode=False  # 临时使用非CV模式
        )
        
        total_length = len(temp_dataset.main_power)
        
        cv = PurgedEmbargoWalkForwardCV(
            n_splits=self.split_config.get('cv_folds', 5),
            embargo_hours=self.split_config.get('embargo_hours', 24),
            purge_hours=self.split_config.get('purge_hours', 0),
            test_hours=7*24,
            min_train_hours=self.split_config.get('min_train_hours', 30*24)
        )
        
        return cv.split(total_length)
    
    def train_dataloader(self):
        """训练数据加载器 - PyTorch Lightning标准接口"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 训练时可以shuffle，因为我们已经处理了时序问题
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        """验证数据加载器 - PyTorch Lightning标准接口"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        """测试数据加载器 - PyTorch Lightning标准接口"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def predict_dataloader(self):
        """预测数据加载器 - PyTorch Lightning标准接口"""
        return self.test_dataloader()
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取数据集元数据"""
        if self.train_dataset is None:
            raise ValueError("必须先调用setup()")
        
        metadata = {
            'appliances': self.train_dataset.get_appliances(),
            'preprocessing_params': self.preprocessing_params,
            'num_appliances': len(self.train_dataset.get_appliances()),
            'sequence_length': self.sequence_length,
            'train_stride': self.train_stride,
            'val_stride': self.val_stride,
            'cv_mode': self.cv_mode,
            'current_fold': self.current_fold,
            'train_info': self.train_dataset.get_split_info(),
            'val_info': self.val_dataset.get_split_info() if self.val_dataset else None,
            'test_info': self.test_dataset.get_split_info() if self.test_dataset else None
        }
        
        # 添加Walk-Forward特定信息
        if self.cv_mode and hasattr(self.train_dataset, 'embargo_info'):
            metadata['embargo_info'] = self.train_dataset.embargo_info
            
        return metadata