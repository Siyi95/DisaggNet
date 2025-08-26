#!/usr/bin/env python3
"""完整AMPds2数据集可视化脚本"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.nilm_disaggregation.data.complete_ampds2_dataset import CompleteAMPds2Dataset
from src.nilm_disaggregation.utils.font_config import setup_chinese_fonts

# 设置绘图样式和中文字体
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
setup_chinese_fonts()
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12


def create_output_directory(base_dir: str = "outputs/ampds2_visualization") -> Path:
    """创建输出目录"""
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (output_dir / "meter_overview").mkdir(exist_ok=True)
    (output_dir / "channel_analysis").mkdir(exist_ok=True)
    (output_dir / "time_series").mkdir(exist_ok=True)
    (output_dir / "statistics").mkdir(exist_ok=True)
    (output_dir / "correlation").mkdir(exist_ok=True)
    
    return output_dir


def plot_dataset_overview(dataset: CompleteAMPds2Dataset, output_dir: Path) -> None:
    """绘制数据集概览"""
    print("绘制数据集概览...")
    
    meter_info = dataset.get_meter_info()
    meter_names = dataset.get_meter_names()
    
    # 1. 电表信息概览
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('AMPds2 数据集概览', fontsize=16, fontweight='bold')
    
    # 电表数量和样本数
    meter_counts = [info['loaded_samples'] for info in meter_info.values()]
    device_names = [info['device_name'] for info in meter_info.values()]
    
    # 样本数分布
    axes[0, 0].bar(range(len(meter_names)), meter_counts, color='skyblue')
    axes[0, 0].set_title('各电表样本数量')
    axes[0, 0].set_xlabel('电表')
    axes[0, 0].set_ylabel('样本数')
    axes[0, 0].set_xticks(range(len(meter_names)))
    axes[0, 0].set_xticklabels(meter_names, rotation=45)
    
    # 添加数值标签
    for i, count in enumerate(meter_counts):
        axes[0, 0].text(i, count + max(meter_counts) * 0.01, str(count), 
                        ha='center', va='bottom', fontsize=10)
    
    # 设备类型分布
    device_types = {}
    for device_name in device_names:
        category = device_name.split('_')[0] if '_' in device_name else device_name
        device_types[category] = device_types.get(category, 0) + 1
    
    axes[0, 1].pie(device_types.values(), labels=device_types.keys(), autopct='%1.1f%%')
    axes[0, 1].set_title('设备类型分布')
    
    # 数据通道信息
    channel_names = dataset.get_channel_names()
    channel_categories = {
        'Current': len([c for c in channel_names if 'Current' in c]),
        'Voltage': len([c for c in channel_names if 'Voltage' in c]),
        'Power': len([c for c in channel_names if 'Power' in c]),
        'Other': len([c for c in channel_names if not any(x in c for x in ['Current', 'Voltage', 'Power'])])
    }
    
    axes[1, 0].bar(channel_categories.keys(), channel_categories.values(), color='lightgreen')
    axes[1, 0].set_title('数据通道类型分布')
    axes[1, 0].set_ylabel('通道数量')
    
    # 数据集统计信息
    total_samples = sum(meter_counts)
    total_meters = len(meter_names)
    total_channels = len(channel_names)
    
    stats_text = f"""
    数据集统计信息:
    
    • 总电表数: {total_meters}
    • 总样本数: {total_samples:,}
    • 数据通道数: {total_channels}
    • 平均样本/电表: {total_samples//total_meters:,}
    
    通道详情:
    {chr(10).join([f'  • {name}' for name in channel_names])}
    """
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis('off')
    axes[1, 1].set_title('数据集详细信息')
    
    plt.tight_layout()
    plt.savefig(output_dir / "meter_overview" / "dataset_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"数据集概览已保存到: {output_dir / 'meter_overview' / 'dataset_overview.png'}")


def plot_meter_time_series(dataset: CompleteAMPds2Dataset, output_dir: Path, 
                          max_samples: int = 2000) -> None:
    """绘制各电表的时间序列数据"""
    print("绘制电表时间序列数据...")
    
    meter_names = dataset.get_meter_names()
    channel_names = dataset.get_channel_names()
    meter_info = dataset.get_meter_info()
    
    for meter_name in meter_names:
        print(f"  处理 {meter_name}...")
        
        # 获取原始数据
        raw_data = dataset.get_raw_data(meter_name)
        if raw_data is None:
            continue
            
        # 限制样本数以提高绘图性能
        data_to_plot = raw_data[:max_samples] if len(raw_data) > max_samples else raw_data
        
        # 创建时间轴
        time_axis = np.arange(len(data_to_plot))
        
        # 绘制所有通道
        fig, axes = plt.subplots(len(channel_names), 1, figsize=(20, 3*len(channel_names)))
        if len(channel_names) == 1:
            axes = [axes]
            
        device_name = meter_info[meter_name]['device_name']
        fig.suptitle(f'{meter_name} - {device_name} 时间序列数据', fontsize=16, fontweight='bold')
        
        for i, channel_name in enumerate(channel_names):
            channel_data = data_to_plot[:, i]
            
            axes[i].plot(time_axis, channel_data, linewidth=0.8, alpha=0.8)
            axes[i].set_title(f'{channel_name}')
            axes[i].set_ylabel('数值')
            axes[i].grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)
            
            stats_text = f'均值: {mean_val:.2f}, 标准差: {std_val:.2f}\n最小值: {min_val:.2f}, 最大值: {max_val:.2f}'
            axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[-1].set_xlabel('时间步')
        
        plt.tight_layout()
        plt.savefig(output_dir / "time_series" / f"{meter_name}_time_series.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"时间序列图已保存到: {output_dir / 'time_series'}")


def plot_channel_analysis(dataset: CompleteAMPds2Dataset, output_dir: Path) -> None:
    """绘制通道分析图"""
    print("绘制通道分析图...")
    
    meter_names = dataset.get_meter_names()
    channel_names = dataset.get_channel_names()
    
    # 为每个通道创建分析图
    for channel_idx, channel_name in enumerate(channel_names):
        print(f"  分析通道: {channel_name}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(f'{channel_name} 通道分析', fontsize=16, fontweight='bold')
        
        # 收集所有电表的该通道数据
        all_channel_data = []
        meter_channel_data = {}
        
        for meter_name in meter_names:
            raw_data = dataset.get_raw_data(meter_name)
            if raw_data is not None:
                channel_data = raw_data[:, channel_idx]
                all_channel_data.extend(channel_data)
                meter_channel_data[meter_name] = channel_data
        
        if not all_channel_data:
            continue
            
        all_channel_data = np.array(all_channel_data)
        
        # 1. 数值分布直方图
        axes[0, 0].hist(all_channel_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title(f'{channel_name} 数值分布')
        axes[0, 0].set_xlabel('数值')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 各电表的箱线图
        box_data = [meter_channel_data[meter] for meter in meter_names if meter in meter_channel_data]
        box_labels = [meter for meter in meter_names if meter in meter_channel_data]
        
        axes[0, 1].boxplot(box_data, labels=box_labels)
        axes[0, 1].set_title(f'{channel_name} 各电表分布')
        axes[0, 1].set_xlabel('电表')
        axes[0, 1].set_ylabel('数值')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 统计摘要
        stats = {
            '均值': np.mean(all_channel_data),
            '中位数': np.median(all_channel_data),
            '标准差': np.std(all_channel_data),
            '最小值': np.min(all_channel_data),
            '最大值': np.max(all_channel_data),
            '25%分位数': np.percentile(all_channel_data, 25),
            '75%分位数': np.percentile(all_channel_data, 75)
        }
        
        stats_text = '\n'.join([f'{k}: {v:.4f}' for k, v in stats.items()])
        axes[1, 0].text(0.1, 0.9, stats_text, transform=axes[1, 0].transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')
        axes[1, 0].set_title('统计摘要')
        
        # 4. 各电表均值比较
        meter_means = [np.mean(meter_channel_data[meter]) for meter in box_labels]
        bars = axes[1, 1].bar(range(len(box_labels)), meter_means, color='lightcoral')
        axes[1, 1].set_title(f'{channel_name} 各电表均值')
        axes[1, 1].set_xlabel('电表')
        axes[1, 1].set_ylabel('均值')
        axes[1, 1].set_xticks(range(len(box_labels)))
        axes[1, 1].set_xticklabels(box_labels, rotation=45)
        
        # 添加数值标签
        for i, (bar, mean_val) in enumerate(zip(bars, meter_means)):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(meter_means) * 0.01,
                           f'{mean_val:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / "channel_analysis" / f"{channel_name}_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"通道分析图已保存到: {output_dir / 'channel_analysis'}")


def plot_correlation_analysis(dataset: CompleteAMPds2Dataset, output_dir: Path) -> None:
    """绘制相关性分析图"""
    print("绘制相关性分析图...")
    
    meter_names = dataset.get_meter_names()
    channel_names = dataset.get_channel_names()
    
    # 为每个电表创建通道间相关性矩阵
    for meter_name in meter_names:
        print(f"  分析 {meter_name} 相关性...")
        
        raw_data = dataset.get_raw_data(meter_name)
        if raw_data is None:
            continue
            
        # 计算相关性矩阵
        correlation_matrix = np.corrcoef(raw_data.T)
        
        # 绘制相关性热图
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        
        # 设置标签
        ax.set_xticks(range(len(channel_names)))
        ax.set_yticks(range(len(channel_names)))
        ax.set_xticklabels(channel_names, rotation=45, ha='right')
        ax.set_yticklabels(channel_names)
        
        # 添加数值标签
        for i in range(len(channel_names)):
            for j in range(len(channel_names)):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white")
        
        ax.set_title(f'{meter_name} 通道间相关性矩阵')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('相关系数', rotation=270, labelpad=20)
        
        plt.tight_layout()
        plt.savefig(output_dir / "correlation" / f"{meter_name}_correlation.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"相关性分析图已保存到: {output_dir / 'correlation'}")


def generate_statistics_report(dataset: CompleteAMPds2Dataset, output_dir: Path) -> None:
    """生成统计报告"""
    print("生成统计报告...")
    
    meter_info = dataset.get_meter_info()
    meter_names = dataset.get_meter_names()
    channel_names = dataset.get_channel_names()
    
    # 创建详细统计报告
    report_lines = []
    report_lines.append("# AMPds2 数据集统计报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n" + "="*80 + "\n")
    
    # 数据集概览
    report_lines.append("## 数据集概览")
    report_lines.append(f"- 总电表数: {len(meter_names)}")
    report_lines.append(f"- 数据通道数: {len(channel_names)}")
    report_lines.append(f"- 总样本数: {sum(info['loaded_samples'] for info in meter_info.values()):,}")
    report_lines.append("")
    
    # 通道信息
    report_lines.append("## 数据通道")
    for i, channel_name in enumerate(channel_names, 1):
        report_lines.append(f"{i:2d}. {channel_name}")
    report_lines.append("")
    
    # 电表详细信息
    report_lines.append("## 电表详细信息")
    report_lines.append("| 电表 | 设备名称 | 样本数 | 是否合成 |")
    report_lines.append("|------|----------|--------|----------|")
    
    for meter_name in sorted(meter_names):
        info = meter_info[meter_name]
        is_synthetic = "是" if info.get('synthetic', False) else "否"
        report_lines.append(f"| {meter_name} | {info['device_name']} | {info['loaded_samples']:,} | {is_synthetic} |")
    
    report_lines.append("")
    
    # 数据统计
    report_lines.append("## 数据统计")
    
    for meter_name in meter_names:
        raw_data = dataset.get_raw_data(meter_name)
        if raw_data is None:
            continue
            
        report_lines.append(f"\n### {meter_name} - {meter_info[meter_name]['device_name']}")
        report_lines.append("| 通道 | 均值 | 标准差 | 最小值 | 最大值 |")
        report_lines.append("|------|------|--------|--------|--------|")
        
        for i, channel_name in enumerate(channel_names):
            channel_data = raw_data[:, i]
            mean_val = np.mean(channel_data)
            std_val = np.std(channel_data)
            min_val = np.min(channel_data)
            max_val = np.max(channel_data)
            
            report_lines.append(f"| {channel_name} | {mean_val:.4f} | {std_val:.4f} | {min_val:.4f} | {max_val:.4f} |")
    
    # 保存报告
    report_content = "\n".join(report_lines)
    
    with open(output_dir / "statistics" / "dataset_report.md", 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # 同时保存为文本文件
    with open(output_dir / "statistics" / "dataset_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"统计报告已保存到: {output_dir / 'statistics'}")


def main():
    parser = argparse.ArgumentParser(description='完整AMPds2数据集可视化')
    parser.add_argument('--data_path', type=str, 
                       default='/Users/siyili/Workspace/DisaggNet/Dataset/dataverse_files/AMPds2.h5',
                       help='AMPds2数据文件路径')
    parser.add_argument('--output_dir', type=str, 
                       default='outputs/ampds2_visualization',
                       help='输出目录')
    parser.add_argument('--max_samples_per_meter', type=int, default=5000,
                       help='每个电表的最大样本数')
    parser.add_argument('--max_plot_samples', type=int, default=2000,
                       help='绘图时的最大样本数')
    parser.add_argument('--load_all_meters', action='store_true', default=True,
                       help='是否加载所有电表')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("AMPds2 完整数据集可视化")
    print("=" * 80)
    
    # 创建输出目录
    output_dir = create_output_directory(args.output_dir)
    print(f"输出目录: {output_dir}")
    
    # 加载数据集
    print("\n正在加载数据集...")
    dataset = CompleteAMPds2Dataset(
        data_path=args.data_path,
        max_samples_per_meter=args.max_samples_per_meter,
        load_all_meters=args.load_all_meters
    )
    
    print(f"\n数据集加载完成!")
    print(f"- 电表数量: {len(dataset.get_meter_names())}")
    print(f"- 数据通道: {len(dataset.get_channel_names())}")
    print(f"- 总样本数: {len(dataset)}")
    
    # 生成可视化
    print("\n开始生成可视化...")
    
    try:
        # 1. 数据集概览
        plot_dataset_overview(dataset, output_dir)
        
        # 2. 时间序列图
        plot_meter_time_series(dataset, output_dir, args.max_plot_samples)
        
        # 3. 通道分析
        plot_channel_analysis(dataset, output_dir)
        
        # 4. 相关性分析
        plot_correlation_analysis(dataset, output_dir)
        
        # 5. 统计报告
        generate_statistics_report(dataset, output_dir)
        
        print("\n" + "="*80)
        print("可视化完成!")
        print(f"所有结果已保存到: {output_dir}")
        print("\n生成的文件:")
        print(f"  - 数据集概览: {output_dir / 'meter_overview'}")
        print(f"  - 时间序列图: {output_dir / 'time_series'}")
        print(f"  - 通道分析: {output_dir / 'channel_analysis'}")
        print(f"  - 相关性分析: {output_dir / 'correlation'}")
        print(f"  - 统计报告: {output_dir / 'statistics'}")
        print("="*80)
        
    except Exception as e:
        print(f"\n错误: 可视化过程中出现问题: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())