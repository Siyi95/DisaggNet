#!/usr/bin/env python3
"""
离线推理脚本
使用训练好的NILM模型进行批量负荷分解和结果可视化
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import h5py
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datamodule import AMPds2DataModule
from src.train_finetune import NILMModel
from src.models.crf import CRFPostProcessor

class OfflineInference:
    """离线推理器"""
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 device: str = 'auto'):
        """
        初始化推理器
        
        Args:
            model_path: 模型检查点路径
            config_path: 配置文件路径
            device: 计算设备
        """
        self.device = self._setup_device(device)
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = NILMModel.load_from_checkpoint(
            model_path, map_location=self.device
        )
        self.model.eval()
        self.model.to(self.device)
        
        # 设备信息
        self.device_names = self.model.device_names
        self.num_devices = self.model.num_devices
        
        print(f"模型加载完成，设备: {self.device}")
        print(f"支持的设备: {self.device_names}")
    
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        return torch.device(device)
    
    def predict_batch(self, 
                     x: torch.Tensor,
                     timestamps: Optional[torch.Tensor] = None) -> Dict[str, np.ndarray]:
        """
        批量预测
        
        Args:
            x: [batch_size, seq_len, input_dim] 输入特征
            timestamps: [batch_size, seq_len] 时间戳（可选）
        Returns:
            预测结果字典
        """
        with torch.no_grad():
            x = x.to(self.device)
            if timestamps is not None:
                timestamps = timestamps.to(self.device)
            
            # 前向传播
            predictions = self.model(x, timestamps)
            
            # 转换为numpy
            results = {}
            for key, value in predictions.items():
                if isinstance(value, torch.Tensor):
                    results[key] = value.cpu().numpy()
                else:
                    results[key] = value
        
        return results
    
    def predict_sequence(self, 
                        data_path: str,
                        start_time: Optional[str] = None,
                        end_time: Optional[str] = None,
                        window_size: int = 240,
                        overlap: int = 120) -> Dict[str, Any]:
        """
        预测时间序列
        
        Args:
            data_path: 数据文件路径
            start_time: 开始时间 (YYYY-MM-DD HH:MM:SS)
            end_time: 结束时间 (YYYY-MM-DD HH:MM:SS)
            window_size: 窗口大小（分钟）
            overlap: 重叠大小（分钟）
        Returns:
            预测结果和元数据
        """
        print(f"开始预测序列: {data_path}")
        
        # 创建数据模块
        data_config = self.config['data'].copy()
        data_config['data_path'] = data_path
        data_config['window_size'] = window_size
        data_config['overlap'] = overlap
        
        data_module = AMPds2DataModule(**data_config)
        data_module.setup('predict')
        
        # 获取数据加载器
        dataloader = data_module.predict_dataloader()
        
        # 收集所有预测结果
        all_predictions = []
        all_timestamps = []
        all_indices = []
        
        print("开始批量预测...")
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="预测进度")):
            x = batch['x']
            timestamps = batch.get('timestamps', None)
            indices = batch.get('indices', None)
            
            # 预测
            predictions = self.predict_batch(x, timestamps)
            
            all_predictions.append(predictions)
            if timestamps is not None:
                all_timestamps.append(timestamps.numpy())
            if indices is not None:
                all_indices.append(indices.numpy())
        
        # 合并结果
        merged_predictions = self._merge_predictions(
            all_predictions, all_timestamps, all_indices, overlap
        )
        
        # 添加元数据
        merged_predictions['metadata'] = {
            'device_names': self.device_names,
            'num_devices': self.num_devices,
            'window_size': window_size,
            'overlap': overlap,
            'prediction_time': datetime.now().isoformat()
        }
        
        print("预测完成")
        return merged_predictions
    
    def _merge_predictions(self, 
                          predictions_list: List[Dict[str, np.ndarray]],
                          timestamps_list: List[np.ndarray],
                          indices_list: List[np.ndarray],
                          overlap: int) -> Dict[str, np.ndarray]:
        """
        合并重叠窗口的预测结果
        
        Args:
            predictions_list: 预测结果列表
            timestamps_list: 时间戳列表
            indices_list: 索引列表
            overlap: 重叠大小
        Returns:
            合并后的预测结果
        """
        if not predictions_list:
            return {}
        
        # 获取第一个预测的键
        keys = predictions_list[0].keys()
        merged = {}
        
        for key in keys:
            if key in ['power', 'state_prob', 'state']:
                # 合并序列预测
                sequences = [pred[key] for pred in predictions_list]
                merged_seq = self._merge_sequences(sequences, overlap)
                merged[key] = merged_seq
        
        # 合并时间戳
        if timestamps_list:
            merged_timestamps = self._merge_sequences(timestamps_list, overlap)
            merged['timestamps'] = merged_timestamps
        
        return merged
    
    def _merge_sequences(self, 
                        sequences: List[np.ndarray], 
                        overlap: int) -> np.ndarray:
        """
        合并重叠序列
        
        Args:
            sequences: 序列列表 [batch_size, seq_len, ...]
            overlap: 重叠大小
        Returns:
            合并后的序列
        """
        if not sequences:
            return np.array([])
        
        if len(sequences) == 1:
            return sequences[0].squeeze(0)  # 移除batch维度
        
        # 初始化结果
        first_seq = sequences[0].squeeze(0)  # [seq_len, ...]
        result = first_seq.copy()
        
        # 逐个合并后续序列
        for seq in sequences[1:]:
            seq = seq.squeeze(0)  # [seq_len, ...]
            
            if overlap > 0 and len(result) >= overlap:
                # 重叠区域取平均
                overlap_start = len(result) - overlap
                result[overlap_start:] = (result[overlap_start:] + seq[:overlap]) / 2
                
                # 添加非重叠部分
                if len(seq) > overlap:
                    result = np.concatenate([result, seq[overlap:]], axis=0)
            else:
                # 直接拼接
                result = np.concatenate([result, seq], axis=0)
        
        return result
    
    def save_results(self, 
                    predictions: Dict[str, Any], 
                    output_path: str,
                    format: str = 'hdf5'):
        """
        保存预测结果
        
        Args:
            predictions: 预测结果
            output_path: 输出路径
            format: 保存格式 ('hdf5', 'csv', 'npz')
        """
        print(f"保存结果到: {output_path}")
        
        if format == 'hdf5':
            self._save_hdf5(predictions, output_path)
        elif format == 'csv':
            self._save_csv(predictions, output_path)
        elif format == 'npz':
            self._save_npz(predictions, output_path)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def _save_hdf5(self, predictions: Dict[str, Any], output_path: str):
        """保存为HDF5格式"""
        with h5py.File(output_path, 'w') as f:
            # 保存预测数据
            for key, value in predictions.items():
                if key != 'metadata' and isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression='gzip')
            
            # 保存元数据
            if 'metadata' in predictions:
                metadata_group = f.create_group('metadata')
                for key, value in predictions['metadata'].items():
                    if isinstance(value, (str, int, float)):
                        metadata_group.attrs[key] = value
                    elif isinstance(value, list):
                        metadata_group.attrs[key] = str(value)
    
    def _save_csv(self, predictions: Dict[str, Any], output_path: str):
        """保存为CSV格式"""
        # 创建DataFrame
        data = {}
        
        # 添加时间戳
        if 'timestamps' in predictions:
            timestamps = predictions['timestamps']
            if timestamps.ndim > 1:
                timestamps = timestamps.flatten()
            data['timestamp'] = pd.to_datetime(timestamps, unit='s')
        else:
            # 生成默认时间戳
            seq_len = len(predictions['power']) if 'power' in predictions else 1000
            data['timestamp'] = pd.date_range(
                start='2023-01-01', periods=seq_len, freq='1min'
            )
        
        # 添加功率预测
        if 'power' in predictions:
            power = predictions['power']  # [seq_len, num_devices]
            for i, device_name in enumerate(self.device_names):
                if i < power.shape[1]:
                    data[f'power_{device_name}'] = power[:, i]
        
        # 添加状态预测
        if 'state' in predictions:
            state = predictions['state']  # [seq_len, num_devices]
            for i, device_name in enumerate(self.device_names):
                if i < state.shape[1]:
                    data[f'state_{device_name}'] = state[:, i]
        
        # 保存
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    
    def _save_npz(self, predictions: Dict[str, Any], output_path: str):
        """保存为NPZ格式"""
        save_dict = {}
        
        for key, value in predictions.items():
            if key != 'metadata' and isinstance(value, np.ndarray):
                save_dict[key] = value
        
        np.savez_compressed(output_path, **save_dict)
    
    def visualize_results(self, 
                         predictions: Dict[str, Any],
                         output_dir: str,
                         days_to_plot: int = 7,
                         start_idx: int = 0):
        """
        可视化预测结果
        
        Args:
            predictions: 预测结果
            output_dir: 输出目录
            days_to_plot: 绘制天数
            start_idx: 开始索引
        """
        print(f"生成可视化图表，保存到: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 准备数据
        power = predictions.get('power', None)
        state = predictions.get('state', None)
        timestamps = predictions.get('timestamps', None)
        
        if power is None:
            print("没有功率预测数据，跳过可视化")
            return
        
        # 计算绘制范围
        seq_len = len(power)
        points_per_day = 24 * 60  # 每天1440个点（每分钟一个）
        end_idx = min(start_idx + days_to_plot * points_per_day, seq_len)
        
        # 截取数据
        power_plot = power[start_idx:end_idx]
        if state is not None:
            state_plot = state[start_idx:end_idx]
        else:
            state_plot = None
        
        # 生成时间轴
        if timestamps is not None:
            time_axis = pd.to_datetime(timestamps[start_idx:end_idx], unit='s')
        else:
            time_axis = pd.date_range(
                start='2023-01-01', periods=len(power_plot), freq='1min'
            )
        
        # 1. 总功率图
        self._plot_total_power(power_plot, time_axis, output_dir)
        
        # 2. 各设备功率图
        self._plot_device_power(power_plot, state_plot, time_axis, output_dir)
        
        # 3. 状态热图
        if state_plot is not None:
            self._plot_state_heatmap(state_plot, time_axis, output_dir)
        
        # 4. 每日功率分布
        self._plot_daily_distribution(power_plot, time_axis, output_dir)
        
        print("可视化完成")
    
    def _plot_total_power(self, power: np.ndarray, time_axis: pd.DatetimeIndex, output_dir: str):
        """绘制总功率图"""
        plt.figure(figsize=(15, 6))
        
        # 计算总功率
        total_power = power.sum(axis=1)
        
        plt.plot(time_axis, total_power, linewidth=1, alpha=0.8, label='预测总功率')
        
        plt.title('总功率预测', fontsize=14, fontweight='bold')
        plt.xlabel('时间')
        plt.ylabel('功率 (W)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 格式化x轴
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'total_power.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_device_power(self, power: np.ndarray, state: Optional[np.ndarray], 
                          time_axis: pd.DatetimeIndex, output_dir: str):
        """绘制各设备功率图"""
        num_devices = min(len(self.device_names), power.shape[1])
        
        # 计算子图布局
        cols = 2
        rows = (num_devices + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_devices):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            device_name = self.device_names[i]
            device_power = power[:, i]
            
            # 绘制功率曲线
            ax.plot(time_axis, device_power, linewidth=1, alpha=0.8, 
                   label=f'{device_name} 功率')
            
            # 如果有状态信息，添加状态背景
            if state is not None and i < state.shape[1]:
                device_state = state[:, i]
                on_mask = device_state > 0.5
                
                if on_mask.any():
                    ax.fill_between(time_axis, 0, device_power.max(), 
                                   where=on_mask, alpha=0.2, color='green',
                                   label='开启状态')
            
            ax.set_title(f'{device_name} 功率预测')
            ax.set_ylabel('功率 (W)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 格式化x轴
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
        
        # 隐藏多余的子图
        for i in range(num_devices, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'device_power.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_state_heatmap(self, state: np.ndarray, time_axis: pd.DatetimeIndex, output_dir: str):
        """绘制状态热图"""
        plt.figure(figsize=(15, 8))
        
        # 转置状态矩阵以便绘制
        state_T = state.T  # [num_devices, seq_len]
        
        # 创建热图
        im = plt.imshow(state_T, aspect='auto', cmap='RdYlGn', 
                       interpolation='nearest', vmin=0, vmax=1)
        
        # 设置y轴标签
        plt.yticks(range(len(self.device_names)), self.device_names)
        
        # 设置x轴
        num_ticks = min(10, len(time_axis))
        tick_indices = np.linspace(0, len(time_axis)-1, num_ticks, dtype=int)
        plt.xticks(tick_indices, [time_axis[i].strftime('%m-%d %H:%M') for i in tick_indices], 
                  rotation=45)
        
        plt.title('设备状态热图', fontsize=14, fontweight='bold')
        plt.xlabel('时间')
        plt.ylabel('设备')
        
        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label('开启概率')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'state_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_daily_distribution(self, power: np.ndarray, time_axis: pd.DatetimeIndex, output_dir: str):
        """绘制每日功率分布"""
        # 创建DataFrame
        df = pd.DataFrame({
            'timestamp': time_axis,
            'total_power': power.sum(axis=1)
        })
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date
        
        # 按小时统计平均功率
        hourly_avg = df.groupby('hour')['total_power'].mean()
        
        plt.figure(figsize=(12, 6))
        
        # 绘制每小时平均功率
        plt.subplot(1, 2, 1)
        plt.bar(hourly_avg.index, hourly_avg.values, alpha=0.7)
        plt.title('每小时平均功率')
        plt.xlabel('小时')
        plt.ylabel('平均功率 (W)')
        plt.grid(True, alpha=0.3)
        
        # 绘制功率分布直方图
        plt.subplot(1, 2, 2)
        plt.hist(df['total_power'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('总功率分布')
        plt.xlabel('功率 (W)')
        plt.ylabel('频次')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'daily_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='NILM离线推理')
    parser.add_argument('--ckpt', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--data_path', type=str, required=True, help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./outputs/inference', help='输出目录')
    parser.add_argument('--start_time', type=str, help='开始时间 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end_time', type=str, help='结束时间 (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--days', type=int, default=7, help='预测天数')
    parser.add_argument('--format', type=str, default='hdf5', choices=['hdf5', 'csv', 'npz'], help='输出格式')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图表')
    parser.add_argument('--device', type=str, default='auto', help='计算设备')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建推理器
    inferencer = OfflineInference(
        model_path=args.ckpt,
        config_path=args.config,
        device=args.device
    )
    
    # 执行预测
    predictions = inferencer.predict_sequence(
        data_path=args.data_path,
        start_time=args.start_time,
        end_time=args.end_time
    )
    
    # 保存结果
    output_file = os.path.join(args.output_dir, f'predictions.{args.format}')
    inferencer.save_results(predictions, output_file, args.format)
    
    # 生成可视化
    if args.visualize:
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        inferencer.visualize_results(predictions, viz_dir, days_to_plot=args.days)
    
    print(f"推理完成，结果保存到: {args.output_dir}")

if __name__ == '__main__':
    main()