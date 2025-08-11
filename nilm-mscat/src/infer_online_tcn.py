#!/usr/bin/env python3
"""
在线TCN检测脚本
使用轻量级因果TCN模型进行实时设备启停检测
"""

import os
import sys
import argparse
import yaml
import time
import json
import threading
from queue import Queue, Empty
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.tcn_online import OnlineEventDetector, OnlineBuffer
from src.features import FeatureExtractor

class RealTimeDataSimulator:
    """实时数据模拟器（用于测试）"""
    
    def __init__(self, 
                 data_path: str,
                 feature_extractor: FeatureExtractor,
                 start_idx: int = 0,
                 speed_factor: float = 1.0):
        """
        初始化数据模拟器
        
        Args:
            data_path: 数据文件路径
            feature_extractor: 特征提取器
            start_idx: 开始索引
            speed_factor: 速度因子（>1表示加速）
        """
        self.data_path = data_path
        self.feature_extractor = feature_extractor
        self.start_idx = start_idx
        self.speed_factor = speed_factor
        
        # 加载数据
        self._load_data()
        
        # 状态
        self.current_idx = start_idx
        self.is_running = False
        
    def _load_data(self):
        """加载数据文件"""
        import h5py
        
        print(f"加载数据: {self.data_path}")
        with h5py.File(self.data_path, 'r') as f:
            # 加载原始数据
            self.timestamps = f['timestamps'][:]
            self.power_data = f['power'][:]
            
            # 如果有其他通道数据
            self.other_channels = {}
            for key in f.keys():
                if key not in ['timestamps', 'power'] and 'power' not in key.lower():
                    self.other_channels[key] = f[key][:]
        
        print(f"数据加载完成，长度: {len(self.timestamps)}")
    
    def get_next_sample(self) -> Optional[Dict[str, Any]]:
        """
        获取下一个数据样本
        
        Returns:
            数据样本字典，如果没有更多数据则返回None
        """
        if self.current_idx >= len(self.timestamps):
            return None
        
        # 获取当前时间点的数据
        timestamp = self.timestamps[self.current_idx]
        power = self.power_data[self.current_idx]
        
        # 构建多通道输入
        channels = {'P_total': power}
        for key, data in self.other_channels.items():
            if self.current_idx < len(data):
                channels[key] = data[self.current_idx]
        
        # 提取特征
        features = self.feature_extractor.extract_single_sample(
            channels, timestamp
        )
        
        sample = {
            'timestamp': timestamp,
            'features': features,
            'raw_power': power
        }
        
        self.current_idx += 1
        return sample
    
    def start_streaming(self, callback: Callable[[Dict[str, Any]], None]):
        """
        开始流式传输数据
        
        Args:
            callback: 数据回调函数
        """
        self.is_running = True
        
        def stream_worker():
            while self.is_running:
                sample = self.get_next_sample()
                if sample is None:
                    print("数据流结束")
                    break
                
                # 调用回调函数
                callback(sample)
                
                # 控制速度
                time.sleep(60.0 / self.speed_factor)  # 每分钟一个数据点
        
        # 在单独线程中运行
        self.stream_thread = threading.Thread(target=stream_worker)
        self.stream_thread.start()
    
    def stop_streaming(self):
        """停止流式传输"""
        self.is_running = False
        if hasattr(self, 'stream_thread'):
            self.stream_thread.join()

class OnlineDetectionSystem:
    """在线检测系统"""
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 device: str = 'auto',
                 buffer_size: int = 120,
                 detection_threshold: float = 0.5,
                 min_duration: int = 3,
                 event_callback: Optional[Callable] = None):
        """
        初始化在线检测系统
        
        Args:
            model_path: TCN模型路径
            config_path: 配置文件路径
            device: 计算设备
            buffer_size: 缓冲区大小（分钟）
            detection_threshold: 检测阈值
            min_duration: 最小持续时间（分钟）
            event_callback: 事件回调函数
        """
        self.device = self._setup_device(device)
        self.buffer_size = buffer_size
        self.detection_threshold = detection_threshold
        self.min_duration = min_duration
        self.event_callback = event_callback
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化特征提取器
        feature_config = self.config.get('features', {})
        self.feature_extractor = FeatureExtractor(**feature_config)
        
        # 加载模型
        print(f"加载TCN模型: {model_path}")
        self.detector = OnlineEventDetector.load_from_checkpoint(
            model_path, map_location=self.device
        )
        self.detector.eval()
        self.detector.to(self.device)
        
        # 获取模型信息
        self.input_dim = self.detector.hparams.input_dim
        self.num_devices = self.detector.hparams.num_devices
        self.device_names = getattr(self.detector, 'device_names', 
                                   [f'Device_{i}' for i in range(self.num_devices)])
        
        # 初始化缓冲区
        self.buffer = OnlineBuffer(
            buffer_size=buffer_size,
            feature_dim=self.input_dim
        )
        
        # 状态跟踪
        self.current_states = np.zeros(self.num_devices, dtype=bool)
        self.state_durations = np.zeros(self.num_devices, dtype=int)
        self.last_change_times = [None] * self.num_devices
        
        # 统计信息
        self.stats = {
            'total_samples': 0,
            'total_events': 0,
            'device_events': {name: 0 for name in self.device_names},
            'start_time': datetime.now()
        }
        
        # 日志设置
        self._setup_logging()
        
        print(f"在线检测系统初始化完成")
        print(f"支持设备: {self.device_names}")
        print(f"缓冲区大小: {buffer_size} 分钟")
    
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        return torch.device(device)
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('online_detection.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个数据样本
        
        Args:
            sample: 数据样本
        Returns:
            检测结果
        """
        timestamp = sample['timestamp']
        features = sample['features']
        
        # 添加到缓冲区
        self.buffer.add_sample(features, timestamp)
        self.stats['total_samples'] += 1
        
        # 检查缓冲区是否准备好
        if not self.buffer.is_ready():
            return {
                'timestamp': timestamp,
                'ready': False,
                'message': f'缓冲区填充中 ({self.buffer.current_size}/{self.buffer_size})'
            }
        
        # 获取缓冲区数据
        buffer_data = self.buffer.get_buffer()  # [buffer_size, feature_dim]
        
        # 转换为tensor并添加batch维度
        x = torch.from_numpy(buffer_data).float().unsqueeze(0).to(self.device)  # [1, seq_len, feature_dim]
        
        # 在线预测
        with torch.no_grad():
            predictions = self.detector.predict_online(x)
        
        # 提取最新时刻的预测
        latest_probs = predictions['state_prob'][0, -1, :].cpu().numpy()  # [num_devices]
        latest_states = (latest_probs > self.detection_threshold).astype(bool)
        
        # 检测状态变化
        events = self._detect_state_changes(latest_states, timestamp)
        
        # 更新状态
        self._update_states(latest_states, timestamp)
        
        result = {
            'timestamp': timestamp,
            'ready': True,
            'probabilities': latest_probs.tolist(),
            'states': latest_states.tolist(),
            'events': events,
            'current_states': self.current_states.tolist(),
            'state_durations': self.state_durations.tolist()
        }
        
        # 触发事件回调
        if events and self.event_callback:
            self.event_callback(result)
        
        return result
    
    def _detect_state_changes(self, new_states: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        """
        检测状态变化事件
        
        Args:
            new_states: 新的状态数组
            timestamp: 时间戳
        Returns:
            事件列表
        """
        events = []
        
        for i, (old_state, new_state) in enumerate(zip(self.current_states, new_states)):
            if old_state != new_state:
                # 检查最小持续时间
                if self.state_durations[i] >= self.min_duration:
                    event = {
                        'device_id': i,
                        'device_name': self.device_names[i],
                        'event_type': 'turn_on' if new_state else 'turn_off',
                        'timestamp': timestamp,
                        'datetime': datetime.fromtimestamp(timestamp).isoformat(),
                        'duration': self.state_durations[i],
                        'probability': float(new_states[i])
                    }
                    events.append(event)
                    
                    # 更新统计
                    self.stats['total_events'] += 1
                    self.stats['device_events'][self.device_names[i]] += 1
                    
                    # 记录日志
                    self.logger.info(
                        f"设备 {self.device_names[i]} {event['event_type']} "
                        f"(持续 {self.state_durations[i]} 分钟)"
                    )
        
        return events
    
    def _update_states(self, new_states: np.ndarray, timestamp: float):
        """
        更新设备状态
        
        Args:
            new_states: 新的状态数组
            timestamp: 时间戳
        """
        for i, (old_state, new_state) in enumerate(zip(self.current_states, new_states)):
            if old_state == new_state:
                # 状态未变化，增加持续时间
                self.state_durations[i] += 1
            else:
                # 状态变化，重置持续时间
                self.state_durations[i] = 1
                self.last_change_times[i] = timestamp
        
        # 更新当前状态
        self.current_states = new_states.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        current_time = datetime.now()
        runtime = current_time - self.stats['start_time']
        
        stats = self.stats.copy()
        stats.update({
            'runtime_seconds': runtime.total_seconds(),
            'runtime_str': str(runtime),
            'samples_per_minute': self.stats['total_samples'] / max(runtime.total_seconds() / 60, 1),
            'current_states': {
                self.device_names[i]: bool(state) 
                for i, state in enumerate(self.current_states)
            },
            'state_durations': {
                self.device_names[i]: int(duration) 
                for i, duration in enumerate(self.state_durations)
            }
        })
        
        return stats
    
    def save_statistics(self, output_path: str):
        """
        保存统计信息
        
        Args:
            output_path: 输出文件路径
        """
        stats = self.get_statistics()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"统计信息已保存到: {output_path}")

def event_handler(result: Dict[str, Any]):
    """事件处理函数示例"""
    events = result.get('events', [])
    
    for event in events:
        print(f"\n🔔 设备事件检测:")
        print(f"   设备: {event['device_name']}")
        print(f"   事件: {event['event_type']}")
        print(f"   时间: {event['datetime']}")
        print(f"   持续: {event['duration']} 分钟")
        print(f"   概率: {event['probability']:.3f}")

def main():
    parser = argparse.ArgumentParser(description='在线TCN检测')
    parser.add_argument('--ckpt', type=str, required=True, help='TCN模型检查点路径')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data_path', type=str, help='测试数据路径（用于模拟）')
    parser.add_argument('--output_dir', type=str, default='./outputs/online', help='输出目录')
    parser.add_argument('--buffer_size', type=int, default=120, help='缓冲区大小（分钟）')
    parser.add_argument('--threshold', type=float, default=0.5, help='检测阈值')
    parser.add_argument('--min_duration', type=int, default=3, help='最小持续时间（分钟）')
    parser.add_argument('--speed_factor', type=float, default=60.0, help='模拟速度因子')
    parser.add_argument('--max_samples', type=int, default=1000, help='最大处理样本数')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    parser.add_argument('--quiet', action='store_true', help='静默模式')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建在线检测系统
    detection_system = OnlineDetectionSystem(
        model_path=args.ckpt,
        config_path=args.config,
        device=args.device,
        buffer_size=args.buffer_size,
        detection_threshold=args.threshold,
        min_duration=args.min_duration,
        event_callback=event_handler if not args.quiet else None
    )
    
    if args.data_path:
        # 使用数据文件进行模拟
        print(f"开始模拟在线检测，数据源: {args.data_path}")
        
        # 创建数据模拟器
        simulator = RealTimeDataSimulator(
            data_path=args.data_path,
            feature_extractor=detection_system.feature_extractor,
            speed_factor=args.speed_factor
        )
        
        # 处理计数器
        sample_count = 0
        results = []
        
        def data_callback(sample: Dict[str, Any]):
            nonlocal sample_count
            
            # 处理样本
            result = detection_system.process_sample(sample)
            results.append(result)
            
            sample_count += 1
            
            # 显示进度
            if not args.quiet and sample_count % 10 == 0:
                stats = detection_system.get_statistics()
                print(f"\r处理样本: {sample_count}/{args.max_samples}, "
                      f"事件: {stats['total_events']}, "
                      f"速度: {stats['samples_per_minute']:.1f} 样本/分钟", end='')
            
            # 检查是否达到最大样本数
            if sample_count >= args.max_samples:
                simulator.stop_streaming()
        
        # 开始流式处理
        try:
            simulator.start_streaming(data_callback)
            
            # 等待处理完成
            while simulator.is_running and sample_count < args.max_samples:
                time.sleep(1)
            
        except KeyboardInterrupt:
            print("\n用户中断，正在停止...")
            simulator.stop_streaming()
        
        # 保存结果
        results_path = os.path.join(args.output_dir, 'detection_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存统计信息
        stats_path = os.path.join(args.output_dir, 'statistics.json')
        detection_system.save_statistics(stats_path)
        
        # 显示最终统计
        final_stats = detection_system.get_statistics()
        print(f"\n\n📊 检测完成统计:")
        print(f"   总样本数: {final_stats['total_samples']}")
        print(f"   总事件数: {final_stats['total_events']}")
        print(f"   运行时间: {final_stats['runtime_str']}")
        print(f"   处理速度: {final_stats['samples_per_minute']:.1f} 样本/分钟")
        print(f"\n📁 结果保存到: {args.output_dir}")
        
    else:
        # 实时模式（等待外部数据输入）
        print("进入实时检测模式，等待数据输入...")
        print("按 Ctrl+C 退出")
        
        try:
            while True:
                # 在实际应用中，这里应该从传感器或数据流中获取数据
                # 这里只是一个示例循环
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n检测系统已停止")
            
            # 保存统计信息
            stats_path = os.path.join(args.output_dir, 'statistics.json')
            detection_system.save_statistics(stats_path)

if __name__ == '__main__':
    main()