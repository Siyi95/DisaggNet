#!/usr/bin/env python3
"""完整AMPds2数据集使用演示"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入字体配置模块
from src.nilm_disaggregation.utils.font_config import setup_chinese_fonts

from src.nilm_disaggregation.data.complete_ampds2_dataset import CompleteAMPds2Dataset


class SimpleNILMModel(nn.Module):
    """简单的NILM模型用于演示"""
    
    def __init__(self, input_channels=11, hidden_size=128, output_channels=11):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.output_layer = nn.Linear(hidden_size, output_channels)
        
    def forward(self, x):
        # x: [batch_size, seq_len, input_channels]
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # 预测下一个时间步的所有通道
        predictions = self.output_layer(last_output)  # [batch_size, output_channels]
        
        return predictions


def create_demo_dataset():
    """创建演示数据集"""
    print("创建演示数据集...")
    
    data_path = '/Users/siyili/Workspace/DisaggNet/Dataset/dataverse_files/AMPds2.h5'
    
    # 创建训练数据集
    train_dataset = CompleteAMPds2Dataset(
        data_path=data_path,
        sequence_length=64,
        train=True,
        train_ratio=0.8,
        max_samples_per_meter=2000,
        load_all_meters=False,
        target_meters=['meter1', 'meter2', 'meter3', 'meter4', 'meter5']
    )
    
    # 创建测试数据集
    test_dataset = CompleteAMPds2Dataset(
        data_path=data_path,
        sequence_length=64,
        train=False,
        train_ratio=0.8,
        max_samples_per_meter=2000,
        load_all_meters=False,
        target_meters=['meter1', 'meter2', 'meter3', 'meter4', 'meter5']
    )
    
    print(f"✓ 数据集创建完成")
    print(f"  - 训练集: {len(train_dataset)} 样本")
    print(f"  - 测试集: {len(test_dataset)} 样本")
    print(f"  - 电表数量: {len(train_dataset.get_meter_names())}")
    print(f"  - 通道数量: {len(train_dataset.get_channel_names())}")
    
    return train_dataset, test_dataset


def train_simple_model(train_dataset, test_dataset, epochs=5):
    """训练简单模型"""
    print(f"\n开始训练简单NILM模型 ({epochs} epochs)...")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNILMModel(input_channels=11, hidden_size=64, output_channels=11)
    model.to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练历史
    train_losses = []
    test_losses = []
    
    print(f"使用设备: {device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_x, batch_info in train_loader:
            batch_x = batch_x.to(device)
            targets = batch_info['targets'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(batch_x)
            loss = criterion(predictions, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_info in test_loader:
                batch_x = batch_x.to(device)
                targets = batch_info['targets'].to(device)
                
                predictions = model(batch_x)
                loss = criterion(predictions, targets)
                
                test_loss += loss.item()
                test_batches += 1
        
        avg_test_loss = test_loss / test_batches
        test_losses.append(avg_test_loss)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.6f}, Test Loss = {avg_test_loss:.6f}")
    
    print("✓ 训练完成")
    
    return model, train_losses, test_losses


def visualize_results(model, test_dataset, train_losses, test_losses):
    """可视化结果"""
    print("\n生成可视化结果...")
    
    # 设置中文字体支持
    setup_chinese_fonts()
    
    # 创建输出目录
    output_dir = Path("outputs/demo_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 绘制训练损失曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', color='blue')
    plt.plot(test_losses, label='测试损失', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('训练过程损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 绘制预测示例
    plt.subplot(1, 2, 2)
    
    # 获取一个测试样本
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        sample_x, sample_info = test_dataset[0]
        sample_x = sample_x.unsqueeze(0).to(device)  # 添加batch维度
        targets = sample_info['targets'].cpu().numpy()
        
        predictions = model(sample_x).cpu().numpy()[0]
    
    # 绘制预测vs真实值
    channel_names = test_dataset.get_channel_names()
    x_pos = np.arange(len(channel_names))
    
    width = 0.35
    plt.bar(x_pos - width/2, targets, width, label='真实值', alpha=0.7)
    plt.bar(x_pos + width/2, predictions, width, label='预测值', alpha=0.7)
    
    plt.xlabel('通道')
    plt.ylabel('标准化数值')
    plt.title('预测结果示例')
    plt.xticks(x_pos, [name.split('_')[0] for name in channel_names], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_results.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 绘制各电表的预测性能
    plt.figure(figsize=(15, 10))
    
    meter_names = test_dataset.get_meter_names()
    meter_info = test_dataset.get_meter_info()
    
    # 为每个电表计算预测误差
    meter_errors = {}
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    with torch.no_grad():
        for batch_x, batch_info in test_loader:
            batch_x = batch_x.to(device)
            targets = batch_info['targets'].cpu().numpy()[0]
            meter_name = batch_info['meter_name'][0]
            
            predictions = model(batch_x).cpu().numpy()[0]
            error = np.mean(np.abs(predictions - targets))
            
            if meter_name not in meter_errors:
                meter_errors[meter_name] = []
            meter_errors[meter_name].append(error)
    
    # 计算每个电表的平均误差
    avg_errors = {meter: np.mean(errors) for meter, errors in meter_errors.items()}
    
    # 绘制误差条形图
    meters = list(avg_errors.keys())
    errors = list(avg_errors.values())
    device_names = [meter_info[meter]['device_name'].replace('_', ' ') for meter in meters]
    
    plt.bar(range(len(meters)), errors, color='lightcoral')
    plt.xlabel('电表/设备')
    plt.ylabel('平均绝对误差')
    plt.title('各电表预测性能')
    plt.xticks(range(len(meters)), [f"{meter}\n{device_names[i][:20]}" for i, meter in enumerate(meters)], 
               rotation=45, ha='right')
    
    # 添加数值标签
    for i, error in enumerate(errors):
        plt.text(i, error + max(errors) * 0.01, f'{error:.4f}', 
                ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "meter_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 可视化结果已保存到: {output_dir}")
    
    return avg_errors


def generate_summary_report(train_dataset, test_dataset, avg_errors, output_dir="outputs/demo_results"):
    """生成总结报告"""
    print("\n生成总结报告...")
    
    output_dir = Path(output_dir)
    
    # 收集信息
    meter_info = train_dataset.get_meter_info()
    channel_names = train_dataset.get_channel_names()
    
    # 创建报告
    report_lines = []
    report_lines.append("# 完整AMPds2数据集使用演示报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n" + "="*80 + "\n")
    
    # 数据集信息
    report_lines.append("## 数据集信息")
    report_lines.append(f"- 训练样本数: {len(train_dataset):,}")
    report_lines.append(f"- 测试样本数: {len(test_dataset):,}")
    report_lines.append(f"- 电表数量: {len(train_dataset.get_meter_names())}")
    report_lines.append(f"- 数据通道数: {len(channel_names)}")
    report_lines.append(f"- 序列长度: {train_dataset.sequence_length}")
    report_lines.append("")
    
    # 电表信息
    report_lines.append("## 加载的电表")
    report_lines.append("| 电表 | 设备名称 | 训练样本 |")
    report_lines.append("|------|----------|----------|")
    
    for meter_name in train_dataset.get_meter_names():
        info = meter_info[meter_name]
        device_name = info['device_name'].replace('_', ' ')
        samples = info['loaded_samples']
        report_lines.append(f"| {meter_name} | {device_name} | {samples:,} |")
    
    report_lines.append("")
    
    # 数据通道
    report_lines.append("## 数据通道")
    for i, channel_name in enumerate(channel_names, 1):
        report_lines.append(f"{i:2d}. {channel_name}")
    report_lines.append("")
    
    # 模型性能
    report_lines.append("## 模型性能")
    report_lines.append("| 电表 | 设备名称 | 平均绝对误差 |")
    report_lines.append("|------|----------|--------------|")
    
    for meter_name, error in avg_errors.items():
        device_name = meter_info[meter_name]['device_name'].replace('_', ' ')
        report_lines.append(f"| {meter_name} | {device_name} | {error:.6f} |")
    
    report_lines.append("")
    
    # 总结
    best_meter = min(avg_errors.items(), key=lambda x: x[1])
    worst_meter = max(avg_errors.items(), key=lambda x: x[1])
    avg_error = np.mean(list(avg_errors.values()))
    
    report_lines.append("## 总结")
    report_lines.append(f"- 平均误差: {avg_error:.6f}")
    report_lines.append(f"- 最佳预测电表: {best_meter[0]} (误差: {best_meter[1]:.6f})")
    report_lines.append(f"- 最差预测电表: {worst_meter[0]} (误差: {worst_meter[1]:.6f})")
    report_lines.append("")
    report_lines.append("## 说明")
    report_lines.append("- 这是一个简单的演示模型，仅用于展示数据集的使用方法")
    report_lines.append("- 实际应用中应使用更复杂的模型架构和更多的训练数据")
    report_lines.append("- 数据已经过标准化处理，误差值为标准化后的数值")
    
    # 保存报告
    report_content = "\n".join(report_lines)
    
    with open(output_dir / "demo_report.md", 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✓ 总结报告已保存到: {output_dir / 'demo_report.md'}")


def main():
    """主函数"""
    print("=" * 80)
    print("完整AMPds2数据集使用演示")
    print("=" * 80)
    
    try:
        # 1. 创建数据集
        train_dataset, test_dataset = create_demo_dataset()
        
        # 2. 训练模型
        model, train_losses, test_losses = train_simple_model(train_dataset, test_dataset, epochs=3)
        
        # 3. 可视化结果
        avg_errors = visualize_results(model, test_dataset, train_losses, test_losses)
        
        # 4. 生成报告
        generate_summary_report(train_dataset, test_dataset, avg_errors)
        
        print("\n" + "=" * 80)
        print("演示完成! ✓")
        print("完整AMPds2数据集可以正常用于NILM模型训练")
        print("查看 outputs/demo_results/ 目录获取详细结果")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())