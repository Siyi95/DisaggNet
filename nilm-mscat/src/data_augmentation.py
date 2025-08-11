#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强模块
解决NILM训练中数据不足的问题
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import random
from scipy import signal
from scipy.interpolate import interp1d

class NILMDataAugmentation:
    """NILM数据增强类"""
    
    def __init__(self,
                 noise_std: float = 0.02,
                 amplitude_range: Tuple[float, float] = (0.95, 1.05),
                 time_jitter: int = 2,
                 channel_dropout: float = 0.1,
                 synthetic_overlay_prob: float = 0.3,
                 frequency_shift_range: Tuple[float, float] = (0.98, 1.02)):
        """
        Args:
            noise_std: 加性噪声标准差
            amplitude_range: 幅度缩放范围
            time_jitter: 时间抖动范围（样本点）
            channel_dropout: 通道dropout概率
            synthetic_overlay_prob: 合成叠加概率
            frequency_shift_range: 频率偏移范围
        """
        self.noise_std = noise_std
        self.amplitude_range = amplitude_range
        self.time_jitter_range = time_jitter
        self.channel_dropout_prob = channel_dropout
        self.synthetic_overlay_prob = synthetic_overlay_prob
        self.frequency_shift_range = frequency_shift_range
    
    def add_gaussian_noise(self, x: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_std, x.shape)
        return x + noise
    
    def amplitude_scaling(self, x: np.ndarray) -> np.ndarray:
        """幅度缩放"""
        scale = np.random.uniform(*self.amplitude_range)
        return x * scale
    
    def time_jitter(self, x: np.ndarray, y_power: np.ndarray, y_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """时间抖动"""
        if self.time_jitter_range <= 0:
            return x, y_power, y_state
        
        seq_len = x.shape[0]
        jitter = np.random.randint(-self.time_jitter_range, self.time_jitter_range + 1)
        
        if jitter == 0:
            return x, y_power, y_state
        
        # 创建新的索引
        if jitter > 0:
            # 向右偏移，前面补零
            new_x = np.zeros_like(x)
            new_y_power = np.zeros_like(y_power)
            new_y_state = np.zeros_like(y_state)
            
            new_x[jitter:] = x[:-jitter]
            new_y_power[jitter:] = y_power[:-jitter]
            new_y_state[jitter:] = y_state[:-jitter]
        else:
            # 向左偏移，后面补零
            jitter = abs(jitter)
            new_x = np.zeros_like(x)
            new_y_power = np.zeros_like(y_power)
            new_y_state = np.zeros_like(y_state)
            
            new_x[:-jitter] = x[jitter:]
            new_y_power[:-jitter] = y_power[jitter:]
            new_y_state[:-jitter] = y_state[jitter:]
        
        return new_x, new_y_power, new_y_state
    
    def channel_dropout(self, x: np.ndarray) -> np.ndarray:
        """通道dropout"""
        if self.channel_dropout_prob <= 0:
            return x
        
        num_channels = x.shape[-1]
        dropout_mask = np.random.random(num_channels) > self.channel_dropout_prob
        
        x_aug = x.copy()
        x_aug[:, ~dropout_mask] = 0
        
        return x_aug
    
    def frequency_shift(self, x: np.ndarray) -> np.ndarray:
        """频率域偏移（模拟电网频率变化）"""
        if x.shape[0] < 10:  # 序列太短，跳过
            return x
        
        shift_factor = np.random.uniform(*self.frequency_shift_range)
        
        # 对每个通道进行频率偏移
        x_shifted = np.zeros_like(x)
        for i in range(x.shape[1]):
            # 使用插值实现频率偏移
            original_indices = np.arange(len(x))
            new_indices = original_indices * shift_factor
            
            # 确保新索引在有效范围内
            valid_mask = new_indices < len(x)
            if valid_mask.sum() > 1:
                interp_func = interp1d(original_indices, x[:, i], 
                                     kind='linear', fill_value='extrapolate')
                x_shifted[valid_mask, i] = interp_func(new_indices[valid_mask])
            else:
                x_shifted[:, i] = x[:, i]
        
        return x_shifted
    
    def synthetic_overlay(self, x: np.ndarray, y_power: np.ndarray, y_state: np.ndarray,
                         device_library: Optional[Dict[str, List[np.ndarray]]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """合成设备叠加"""
        if device_library is None or np.random.random() > self.synthetic_overlay_prob:
            return x, y_power, y_state
        
        # 随机选择一个设备模式进行叠加
        device_names = list(device_library.keys())
        if not device_names:
            return x, y_power, y_state
        
        selected_device = np.random.choice(device_names)
        device_patterns = device_library[selected_device]
        
        if not device_patterns:
            return x, y_power, y_state
        
        # 随机选择一个模式
        pattern = np.random.choice(device_patterns)
        
        # 确保模式长度不超过序列长度
        if len(pattern) > len(x):
            pattern = pattern[:len(x)]
        
        # 随机选择叠加位置
        max_start = len(x) - len(pattern)
        if max_start <= 0:
            return x, y_power, y_state
        
        start_pos = np.random.randint(0, max_start + 1)
        end_pos = start_pos + len(pattern)
        
        # 叠加到总功率
        x_aug = x.copy()
        y_power_aug = y_power.copy()
        y_state_aug = y_state.copy()
        
        # 假设第一个通道是总功率
        x_aug[start_pos:end_pos, 0] += pattern
        
        # 更新对应设备的功率和状态（如果知道设备索引）
        # 这里简化处理，实际应用中需要根据具体的设备映射
        
        return x_aug, y_power_aug, y_state_aug
    
    def mixup(self, x1: np.ndarray, y1_power: np.ndarray, y1_state: np.ndarray,
              x2: np.ndarray, y2_power: np.ndarray, y2_state: np.ndarray,
              alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mixup数据增强"""
        lam = np.random.beta(alpha, alpha)
        
        # 确保两个样本长度相同
        min_len = min(len(x1), len(x2))
        x1, y1_power, y1_state = x1[:min_len], y1_power[:min_len], y1_state[:min_len]
        x2, y2_power, y2_state = x2[:min_len], y2_power[:min_len], y2_state[:min_len]
        
        # 混合
        x_mixed = lam * x1 + (1 - lam) * x2
        y_power_mixed = lam * y1_power + (1 - lam) * y2_power
        y_state_mixed = lam * y1_state + (1 - lam) * y2_state
        
        return x_mixed, y_power_mixed, y_state_mixed
    
    def cutmix(self, x1: np.ndarray, y1_power: np.ndarray, y1_state: np.ndarray,
               x2: np.ndarray, y2_power: np.ndarray, y2_state: np.ndarray,
               alpha: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CutMix数据增强"""
        lam = np.random.beta(alpha, alpha)
        
        # 确保两个样本长度相同
        min_len = min(len(x1), len(x2))
        x1, y1_power, y1_state = x1[:min_len], y1_power[:min_len], y1_state[:min_len]
        x2, y2_power, y2_state = x2[:min_len], y2_power[:min_len], y2_state[:min_len]
        
        # 计算切割区域
        cut_len = int(min_len * (1 - lam))
        cut_start = np.random.randint(0, min_len - cut_len + 1)
        cut_end = cut_start + cut_len
        
        # 复制第一个样本
        x_mixed = x1.copy()
        y_power_mixed = y1_power.copy()
        y_state_mixed = y1_state.copy()
        
        # 替换切割区域
        x_mixed[cut_start:cut_end] = x2[cut_start:cut_end]
        y_power_mixed[cut_start:cut_end] = y2[cut_start:cut_end]
        y_state_mixed[cut_start:cut_end] = y2_state[cut_start:cut_end]
        
        return x_mixed, y_power_mixed, y_state_mixed
    
    def augment_batch(self, batch: Dict[str, np.ndarray], 
                     device_library: Optional[Dict[str, List[np.ndarray]]] = None) -> Dict[str, np.ndarray]:
        """批量数据增强"""
        x = batch['x']
        y_power = batch['y_power']
        y_state = batch['y_state']
        
        batch_size = x.shape[0]
        augmented_x = []
        augmented_y_power = []
        augmented_y_state = []
        
        for i in range(batch_size):
            x_sample = x[i]
            y_power_sample = y_power[i]
            y_state_sample = y_state[i]
            
            # 随机选择增强方法
            aug_methods = []
            
            # 基础增强（总是应用）
            if np.random.random() < 0.8:
                x_sample = self.add_gaussian_noise(x_sample)
            
            if np.random.random() < 0.6:
                x_sample = self.amplitude_scaling(x_sample)
            
            if np.random.random() < 0.4:
                x_sample, y_power_sample, y_state_sample = self.time_jitter(
                    x_sample, y_power_sample, y_state_sample
                )
            
            if np.random.random() < 0.3:
                x_sample = self.channel_dropout(x_sample)
            
            if np.random.random() < 0.2:
                x_sample = self.frequency_shift(x_sample)
            
            # 合成叠加
            if device_library and np.random.random() < self.synthetic_overlay_prob:
                x_sample, y_power_sample, y_state_sample = self.synthetic_overlay(
                    x_sample, y_power_sample, y_state_sample, device_library
                )
            
            augmented_x.append(x_sample)
            augmented_y_power.append(y_power_sample)
            augmented_y_state.append(y_state_sample)
        
        # Mixup/CutMix（批次级别）
        if batch_size > 1 and np.random.random() < 0.2:
            # 随机配对进行mixup或cutmix
            indices = np.random.permutation(batch_size)
            for i in range(0, batch_size - 1, 2):
                idx1, idx2 = indices[i], indices[i + 1]
                
                if np.random.random() < 0.5:  # Mixup
                    x_mixed, y_power_mixed, y_state_mixed = self.mixup(
                        augmented_x[idx1], augmented_y_power[idx1], augmented_y_state[idx1],
                        augmented_x[idx2], augmented_y_power[idx2], augmented_y_state[idx2]
                    )
                else:  # CutMix
                    x_mixed, y_power_mixed, y_state_mixed = self.cutmix(
                        augmented_x[idx1], augmented_y_power[idx1], augmented_y_state[idx1],
                        augmented_x[idx2], augmented_y_power[idx2], augmented_y_state[idx2]
                    )
                
                # 随机替换其中一个样本
                if np.random.random() < 0.5:
                    augmented_x[idx1] = x_mixed
                    augmented_y_power[idx1] = y_power_mixed
                    augmented_y_state[idx1] = y_state_mixed
                else:
                    augmented_x[idx2] = x_mixed
                    augmented_y_power[idx2] = y_power_mixed
                    augmented_y_state[idx2] = y_state_mixed
        
        return {
            'x': np.stack(augmented_x),
            'y_power': np.stack(augmented_y_power),
            'y_state': np.stack(augmented_y_state)
        }

class DevicePatternLibrary:
    """设备模式库，用于合成数据增强"""
    
    def __init__(self):
        self.patterns = {}
    
    def extract_patterns_from_data(self, data: Dict[str, np.ndarray], 
                                  device_names: List[str],
                                  min_duration: int = 10,
                                  power_threshold: float = 10.0) -> Dict[str, List[np.ndarray]]:
        """从真实数据中提取设备模式"""
        x = data['x']
        y_power = data['y_power']
        y_state = data['y_state']
        
        patterns = {device: [] for device in device_names}
        
        # 对每个设备提取开启模式
        for device_idx, device_name in enumerate(device_names):
            if device_idx >= y_power.shape[-1]:
                continue
            
            device_power = y_power[:, :, device_idx].flatten()
            device_state = y_state[:, :, device_idx].flatten()
            
            # 找到设备开启的连续段
            on_segments = self._find_on_segments(device_state, min_duration)
            
            for start, end in on_segments:
                if end - start >= min_duration:
                    # 提取对应的总功率变化模式
                    total_power_before = x[max(0, start-5):start, 0].mean() if start > 5 else 0
                    total_power_during = x[start:end, 0]
                    
                    # 计算功率增量
                    power_increment = total_power_during - total_power_before
                    
                    # 过滤掉功率变化太小的模式
                    if power_increment.max() > power_threshold:
                        patterns[device_name].append(power_increment)
        
        return patterns
    
    def _find_on_segments(self, state_sequence: np.ndarray, min_duration: int) -> List[Tuple[int, int]]:
        """找到状态序列中的开启段"""
        segments = []
        start = None
        
        for i, state in enumerate(state_sequence):
            if state > 0.5 and start is None:
                start = i
            elif state <= 0.5 and start is not None:
                if i - start >= min_duration:
                    segments.append((start, i))
                start = None
        
        # 处理序列末尾的开启段
        if start is not None and len(state_sequence) - start >= min_duration:
            segments.append((start, len(state_sequence)))
        
        return segments
    
    def generate_synthetic_patterns(self, device_names: List[str]) -> Dict[str, List[np.ndarray]]:
        """生成合成设备模式"""
        patterns = {}
        
        for device_name in device_names:
            device_patterns = []
            
            # 为每个设备生成几种典型模式
            # 1. 阶跃模式（如电灯）
            step_pattern = self._generate_step_pattern()
            device_patterns.append(step_pattern)
            
            # 2. 渐变模式（如加热器）
            ramp_pattern = self._generate_ramp_pattern()
            device_patterns.append(ramp_pattern)
            
            # 3. 周期模式（如洗衣机）
            if 'wash' in device_name.lower() or 'dry' in device_name.lower():
                cycle_pattern = self._generate_cycle_pattern()
                device_patterns.append(cycle_pattern)
            
            # 4. 脉冲模式（如微波炉）
            if 'micro' in device_name.lower():
                pulse_pattern = self._generate_pulse_pattern()
                device_patterns.append(pulse_pattern)
            
            patterns[device_name] = device_patterns
        
        return patterns
    
    def _generate_step_pattern(self, duration: int = 20, power: float = 100.0) -> np.ndarray:
        """生成阶跃模式"""
        pattern = np.zeros(duration)
        pattern[2:] = power + np.random.normal(0, power * 0.05, duration - 2)
        return pattern
    
    def _generate_ramp_pattern(self, duration: int = 30, max_power: float = 150.0) -> np.ndarray:
        """生成渐变模式"""
        pattern = np.linspace(0, max_power, duration)
        pattern += np.random.normal(0, max_power * 0.03, duration)
        return pattern
    
    def _generate_cycle_pattern(self, duration: int = 60, base_power: float = 50.0) -> np.ndarray:
        """生成周期模式"""
        t = np.linspace(0, 4 * np.pi, duration)
        pattern = base_power * (1 + 0.5 * np.sin(t) + 0.3 * np.sin(3 * t))
        pattern += np.random.normal(0, base_power * 0.05, duration)
        return np.maximum(pattern, 0)
    
    def _generate_pulse_pattern(self, duration: int = 15, pulse_power: float = 200.0) -> np.ndarray:
        """生成脉冲模式"""
        pattern = np.zeros(duration)
        pulse_start = duration // 4
        pulse_end = 3 * duration // 4
        pattern[pulse_start:pulse_end] = pulse_power
        pattern += np.random.normal(0, pulse_power * 0.02, duration)
        return pattern