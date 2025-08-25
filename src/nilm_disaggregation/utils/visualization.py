"""可视化模块"""

import matplotlib.pyplot as plt
import numpy as np
import os


def create_visualizations(results, power_preds, power_trues, appliances, save_dir):
    """
    创建可视化图表
    
    Args:
        results: 评估结果字典
        power_preds: 功率预测值
        power_trues: 功率真实值
        appliances: 设备名称列表
        save_dir: 保存目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 性能指标对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['mae', 'rmse', 'r2', 'correlation']
    metric_names = ['MAE', 'RMSE', 'R²', '相关系数']
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        values = [results[app][metric] for app in appliances]
        bars = ax.bar(appliances, values, alpha=0.7)
        ax.set_title(f'{name} 对比', fontsize=14, fontweight='bold')
        ax.set_ylabel(name)
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 预测vs真实值散点图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, appliance in enumerate(appliances):
        ax = axes[i // 2, i % 2]
        
        pred = power_preds[:, i]
        true = power_trues[:, i]
        
        # 随机采样以减少点的数量
        if len(pred) > 1000:
            indices = np.random.choice(len(pred), 1000, replace=False)
            pred = pred[indices]
            true = true[indices]
        
        ax.scatter(true, pred, alpha=0.5, s=10)
        
        # 添加对角线
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.set_title(f'{appliance} - R²: {results[appliance]["r2"]:.3f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存到 {save_dir}/")


def plot_training_curves(train_losses, val_losses, save_dir):
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', alpha=0.8)
    plt.plot(val_losses, label='验证损失', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存到 {save_dir}/training_curves.png")


def plot_power_disaggregation(main_power, appliance_powers, appliance_names, save_dir, sample_length=1000):
    """
    绘制功率分解结果
    
    Args:
        main_power: 主功率序列
        appliance_powers: 设备功率序列字典
        appliance_names: 设备名称列表
        save_dir: 保存目录
        sample_length: 采样长度
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 采样数据以减少绘图时间
    if len(main_power) > sample_length:
        indices = np.linspace(0, len(main_power)-1, sample_length, dtype=int)
        main_power = main_power[indices]
        appliance_powers = {name: power[indices] for name, power in appliance_powers.items()}
    
    fig, axes = plt.subplots(len(appliance_names) + 1, 1, figsize=(15, 3 * (len(appliance_names) + 1)))
    
    # 绘制主功率
    axes[0].plot(main_power, label='主功率', color='black', linewidth=1)
    axes[0].set_title('主功率')
    axes[0].set_ylabel('功率 (W)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 绘制各设备功率
    for i, appliance in enumerate(appliance_names):
        axes[i+1].plot(appliance_powers[appliance], label=f'{appliance}功率', linewidth=1)
        axes[i+1].set_title(f'{appliance}功率分解')
        axes[i+1].set_ylabel('功率 (W)')
        axes[i+1].grid(True, alpha=0.3)
        axes[i+1].legend()
    
    axes[-1].set_xlabel('时间步')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/power_disaggregation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"功率分解图已保存到 {save_dir}/power_disaggregation.png")