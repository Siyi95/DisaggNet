import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings

try:
    import pycrfsuite
    PYCRFSUITE_AVAILABLE = True
except ImportError:
    PYCRFSUITE_AVAILABLE = False
    warnings.warn("pycrfsuite not available, using simplified CRF implementation")

class SimpleCRF:
    """简化的CRF实现，用于二分类状态序列平滑"""
    
    def __init__(self, 
                 num_states: int = 2,
                 transition_cost: float = 1.0,
                 min_duration: int = 5):
        """
        Args:
            num_states: 状态数量（通常为2：off/on）
            transition_cost: 状态转移代价
            min_duration: 最小持续时间（分钟）
        """
        self.num_states = num_states
        self.transition_cost = transition_cost
        self.min_duration = min_duration
        
        # 初始化转移矩阵（鼓励状态持久）
        self.transition_matrix = np.full((num_states, num_states), transition_cost)
        np.fill_diagonal(self.transition_matrix, 0.0)  # 保持同状态代价为0
        
    def viterbi_decode(self, 
                      emission_scores: np.ndarray,
                      device_idx: int = 0) -> np.ndarray:
        """
        维特比解码
        
        Args:
            emission_scores: [seq_len, num_states] 发射分数
            device_idx: 设备索引（用于调试）
        Returns:
            [seq_len] 最优状态序列
        """
        seq_len, num_states = emission_scores.shape
        
        # 动态规划表
        dp = np.zeros((seq_len, num_states))
        path = np.zeros((seq_len, num_states), dtype=int)
        
        # 初始化
        dp[0] = emission_scores[0]
        
        # 前向传播
        for t in range(1, seq_len):
            for curr_state in range(num_states):
                # 计算从所有前一状态转移到当前状态的分数
                transition_scores = dp[t-1] - self.transition_matrix[:, curr_state]
                
                # 选择最佳前一状态
                best_prev_state = np.argmax(transition_scores)
                dp[t, curr_state] = transition_scores[best_prev_state] + emission_scores[t, curr_state]
                path[t, curr_state] = best_prev_state
        
        # 回溯找到最优路径
        states = np.zeros(seq_len, dtype=int)
        states[-1] = np.argmax(dp[-1])
        
        for t in range(seq_len - 2, -1, -1):
            states[t] = path[t + 1, states[t + 1]]
        
        return states
    
    def apply_min_duration_filter(self, states: np.ndarray) -> np.ndarray:
        """
        应用最小持续时间过滤
        
        Args:
            states: [seq_len] 状态序列
        Returns:
            [seq_len] 过滤后的状态序列
        """
        if len(states) == 0:
            return states
            
        filtered_states = states.copy()
        
        # 找到状态变化点
        change_points = [0]
        for i in range(1, len(states)):
            if states[i] != states[i-1]:
                change_points.append(i)
        change_points.append(len(states))
        
        # 检查每个段的持续时间
        for i in range(len(change_points) - 1):
            start = change_points[i]
            end = change_points[i + 1]
            duration = end - start
            
            # 如果持续时间太短，则修正
            if duration < self.min_duration:
                # 选择与前一段相同的状态（如果存在）
                if i > 0:
                    prev_start = change_points[i-1]
                    prev_state = filtered_states[prev_start]
                    filtered_states[start:end] = prev_state
                # 或者选择与后一段相同的状态
                elif i < len(change_points) - 2:
                    next_start = change_points[i+1]
                    if next_start < len(filtered_states):
                        next_state = filtered_states[next_start]
                        filtered_states[start:end] = next_state
        
        return filtered_states

class CRFPostProcessor:
    """CRF后处理器，整合多种平滑方法"""
    
    def __init__(self,
                 num_devices: int,
                 power_threshold: float = 10.0,
                 min_on_duration: int = 5,
                 min_off_duration: int = 3,
                 transition_cost: float = 1.0,
                 use_pycrfsuite: bool = True,
                 temperature: float = 1.0):
        """
        Args:
            num_devices: 设备数量
            power_threshold: 功率阈值（W）
            min_on_duration: 最小开启持续时间（分钟）
            min_off_duration: 最小关闭持续时间（分钟）
            transition_cost: 状态转移代价
            use_pycrfsuite: 是否使用pycrfsuite
            temperature: 温度参数（用于logits转概率）
        """
        self.num_devices = num_devices
        self.power_threshold = power_threshold
        self.min_on_duration = min_on_duration
        self.min_off_duration = min_off_duration
        self.temperature = temperature
        
        # 选择CRF实现
        if use_pycrfsuite and PYCRFSUITE_AVAILABLE:
            self.use_pycrfsuite = True
            self.crf_models = []
            for _ in range(num_devices):
                self.crf_models.append(None)  # 延迟初始化
        else:
            self.use_pycrfsuite = False
            self.simple_crfs = []
            for _ in range(num_devices):
                crf = SimpleCRF(
                    num_states=2,
                    transition_cost=transition_cost,
                    min_duration=min_on_duration
                )
                self.simple_crfs.append(crf)
    
    def process_predictions(self,
                          power_predictions: np.ndarray,
                          event_logits: Optional[np.ndarray] = None,
                          event_probs: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        处理模型预测，返回平滑后的结果
        
        Args:
            power_predictions: [seq_len, num_devices] 功率预测
            event_logits: [seq_len, num_devices] 事件logits（可选）
            event_probs: [seq_len, num_devices] 事件概率（可选）
        Returns:
            处理后的结果字典
        """
        seq_len, num_devices = power_predictions.shape
        
        # 基于功率阈值的初始状态
        power_states = (power_predictions > self.power_threshold).astype(int)
        
        # 如果有事件概率，结合使用
        if event_probs is not None:
            prob_states = (event_probs > 0.5).astype(int)
            # 加权结合
            combined_states = (power_states + prob_states) >= 1
        else:
            combined_states = power_states
        
        # 为每个设备应用CRF平滑
        smoothed_states = np.zeros_like(combined_states)
        
        for device_idx in range(num_devices):
            if event_logits is not None:
                # 使用logits作为发射分数
                emission_scores = self._prepare_emission_scores(
                    event_logits[:, device_idx],
                    power_predictions[:, device_idx]
                )
            else:
                # 基于功率预测生成发射分数
                emission_scores = self._power_to_emission_scores(
                    power_predictions[:, device_idx]
                )
            
            # 应用CRF解码
            if self.use_pycrfsuite:
                device_states = self._pycrfsuite_decode(
                    emission_scores, device_idx
                )
            else:
                device_states = self.simple_crfs[device_idx].viterbi_decode(
                    emission_scores, device_idx
                )
            
            # 应用最小持续时间过滤
            device_states = self._apply_duration_constraints(
                device_states, device_idx
            )
            
            smoothed_states[:, device_idx] = device_states
        
        # 基于平滑状态调整功率预测
        adjusted_power = self._adjust_power_predictions(
            power_predictions, smoothed_states
        )
        
        return {
            'power_predictions': adjusted_power,
            'state_predictions': smoothed_states,
            'raw_power': power_predictions,
            'raw_states': combined_states
        }
    
    def _prepare_emission_scores(self,
                               event_logits: np.ndarray,
                               power_values: np.ndarray) -> np.ndarray:
        """
        准备发射分数矩阵
        
        Args:
            event_logits: [seq_len] 事件logits
            power_values: [seq_len] 功率值
        Returns:
            [seq_len, 2] 发射分数（off, on）
        """
        seq_len = len(event_logits)
        emission_scores = np.zeros((seq_len, 2))
        
        # 将logits转换为概率
        probs = 1 / (1 + np.exp(-event_logits / self.temperature))
        
        # 发射分数：off状态得分，on状态得分
        emission_scores[:, 0] = np.log(1 - probs + 1e-8)  # off状态
        emission_scores[:, 1] = np.log(probs + 1e-8)      # on状态
        
        # 结合功率信息
        power_factor = np.tanh(power_values / self.power_threshold)
        emission_scores[:, 1] += power_factor  # 功率越高，on状态得分越高
        
        return emission_scores
    
    def _power_to_emission_scores(self, power_values: np.ndarray) -> np.ndarray:
        """
        基于功率值生成发射分数
        
        Args:
            power_values: [seq_len] 功率值
        Returns:
            [seq_len, 2] 发射分数
        """
        seq_len = len(power_values)
        emission_scores = np.zeros((seq_len, 2))
        
        # 基于功率阈值的概率
        normalized_power = power_values / (self.power_threshold + 1e-8)
        on_prob = 1 / (1 + np.exp(-5 * (normalized_power - 1)))  # sigmoid
        
        emission_scores[:, 0] = np.log(1 - on_prob + 1e-8)  # off
        emission_scores[:, 1] = np.log(on_prob + 1e-8)      # on
        
        return emission_scores
    
    def _pycrfsuite_decode(self,
                          emission_scores: np.ndarray,
                          device_idx: int) -> np.ndarray:
        """
        使用pycrfsuite进行解码（如果可用）
        
        Args:
            emission_scores: [seq_len, 2] 发射分数
            device_idx: 设备索引
        Returns:
            [seq_len] 状态序列
        """
        # 这里简化实现，实际使用时需要训练CRF模型
        # 暂时回退到简单CRF
        simple_crf = SimpleCRF()
        return simple_crf.viterbi_decode(emission_scores, device_idx)
    
    def _apply_duration_constraints(self,
                                  states: np.ndarray,
                                  device_idx: int) -> np.ndarray:
        """
        应用持续时间约束
        
        Args:
            states: [seq_len] 状态序列
            device_idx: 设备索引
        Returns:
            [seq_len] 约束后的状态序列
        """
        if len(states) == 0:
            return states
            
        filtered_states = states.copy()
        
        # 应用最小开启时间
        filtered_states = self._filter_short_segments(
            filtered_states, target_state=1, min_duration=self.min_on_duration
        )
        
        # 应用最小关闭时间
        filtered_states = self._filter_short_segments(
            filtered_states, target_state=0, min_duration=self.min_off_duration
        )
        
        return filtered_states
    
    def _filter_short_segments(self,
                             states: np.ndarray,
                             target_state: int,
                             min_duration: int) -> np.ndarray:
        """
        过滤过短的状态段
        
        Args:
            states: [seq_len] 状态序列
            target_state: 目标状态（0或1）
            min_duration: 最小持续时间
        Returns:
            [seq_len] 过滤后的状态序列
        """
        filtered_states = states.copy()
        
        i = 0
        while i < len(filtered_states):
            if filtered_states[i] == target_state:
                # 找到目标状态段的结束
                j = i
                while j < len(filtered_states) and filtered_states[j] == target_state:
                    j += 1
                
                # 检查段长度
                segment_length = j - i
                if segment_length < min_duration:
                    # 段太短，需要修正
                    # 选择与相邻段相同的状态
                    replacement_state = 1 - target_state
                    filtered_states[i:j] = replacement_state
                
                i = j
            else:
                i += 1
        
        return filtered_states
    
    def _adjust_power_predictions(self,
                                power_predictions: np.ndarray,
                                state_predictions: np.ndarray) -> np.ndarray:
        """
        基于状态预测调整功率预测
        
        Args:
            power_predictions: [seq_len, num_devices] 原始功率预测
            state_predictions: [seq_len, num_devices] 状态预测
        Returns:
            [seq_len, num_devices] 调整后的功率预测
        """
        adjusted_power = power_predictions.copy()
        
        # 当状态为off时，将功率设为0或很小的值
        off_mask = (state_predictions == 0)
        adjusted_power[off_mask] = np.minimum(
            adjusted_power[off_mask], 
            self.power_threshold * 0.1
        )
        
        # 当状态为on但功率很小时，设置最小功率
        on_mask = (state_predictions == 1)
        low_power_mask = (adjusted_power < self.power_threshold * 0.5)
        combined_mask = on_mask & low_power_mask
        
        adjusted_power[combined_mask] = np.maximum(
            adjusted_power[combined_mask],
            self.power_threshold * 0.5
        )
        
        return adjusted_power
    
    def get_statistics(self, 
                      raw_states: np.ndarray,
                      smoothed_states: np.ndarray) -> Dict[str, float]:
        """
        获取平滑统计信息
        
        Args:
            raw_states: [seq_len, num_devices] 原始状态
            smoothed_states: [seq_len, num_devices] 平滑状态
        Returns:
            统计信息字典
        """
        stats = {}
        
        # 计算状态变化次数
        raw_changes = np.sum(np.diff(raw_states, axis=0) != 0)
        smoothed_changes = np.sum(np.diff(smoothed_states, axis=0) != 0)
        
        stats['raw_state_changes'] = int(raw_changes)
        stats['smoothed_state_changes'] = int(smoothed_changes)
        stats['change_reduction_ratio'] = 1 - (smoothed_changes / (raw_changes + 1e-8))
        
        # 计算平均段长度
        avg_segment_lengths = []
        for device_idx in range(smoothed_states.shape[1]):
            device_states = smoothed_states[:, device_idx]
            segment_lengths = self._get_segment_lengths(device_states)
            if len(segment_lengths) > 0:
                avg_segment_lengths.append(np.mean(segment_lengths))
        
        if avg_segment_lengths:
            stats['avg_segment_length'] = np.mean(avg_segment_lengths)
        else:
            stats['avg_segment_length'] = 0.0
        
        return stats
    
    def _get_segment_lengths(self, states: np.ndarray) -> List[int]:
        """获取状态段长度列表"""
        if len(states) == 0:
            return []
        
        lengths = []
        current_length = 1
        
        for i in range(1, len(states)):
            if states[i] == states[i-1]:
                current_length += 1
            else:
                lengths.append(current_length)
                current_length = 1
        
        lengths.append(current_length)
        return lengths