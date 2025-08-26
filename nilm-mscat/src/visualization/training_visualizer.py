"""训练过程可视化工具"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from collections import defaultdict, deque
import torch
import torch.nn as nn
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, 
                 save_dir: Optional[Path] = None,
                 max_history: int = 1000,
                 update_frequency: int = 10):
        """
        初始化可视化器
        
        Args:
            save_dir: 保存目录
            max_history: 最大历史记录数
            update_frequency: 更新频率（每N个batch更新一次）
        """
        self.save_dir = Path(save_dir) if save_dir else Path("outputs/visualization")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_history = max_history
        self.update_frequency = update_frequency
        
        # 历史记录
        self.train_losses = deque(maxlen=max_history)
        self.val_losses = deque(maxlen=max_history)
        self.train_metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.val_metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.learning_rates = deque(maxlen=max_history)
        self.gradient_norms = deque(maxlen=max_history)
        self.epochs = deque(maxlen=max_history)
        self.batch_times = deque(maxlen=max_history)
        
        # 权重和梯度历史
        self.weight_history = defaultdict(lambda: deque(maxlen=100))
        self.gradient_history = defaultdict(lambda: deque(maxlen=100))
        
        # 预测样本历史
        self.prediction_samples = deque(maxlen=50)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.step_count = 0
        
    def log_training_step(self, 
                         epoch: int,
                         step: int,
                         train_loss: float,
                         metrics: Dict[str, float],
                         learning_rate: float,
                         batch_time: float):
        """记录训练步骤"""
        self.step_count += 1
        
        if self.step_count % self.update_frequency == 0:
            self.epochs.append(epoch + step / 1000)  # 近似的epoch位置
            self.train_losses.append(train_loss)
            self.learning_rates.append(learning_rate)
            self.batch_times.append(batch_time)
            
            for metric_name, value in metrics.items():
                self.train_metrics[metric_name].append(value)
    
    def log_validation_step(self, 
                          epoch: int,
                          val_loss: float,
                          metrics: Dict[str, float]):
        """记录验证步骤"""
        self.val_losses.append(val_loss)
        
        for metric_name, value in metrics.items():
            self.val_metrics[metric_name].append(value)
    
    def log_model_weights(self, model: nn.Module, epoch: int):
        """记录模型权重统计"""
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # 权重统计
                weight_stats = {
                    'mean': float(param.data.mean()),
                    'std': float(param.data.std()),
                    'min': float(param.data.min()),
                    'max': float(param.data.max()),
                    'norm': float(param.data.norm())
                }
                self.weight_history[name].append((epoch, weight_stats))
                
                # 梯度统计
                grad_stats = {
                    'mean': float(param.grad.mean()),
                    'std': float(param.grad.std()),
                    'min': float(param.grad.min()),
                    'max': float(param.grad.max()),
                    'norm': float(param.grad.norm())
                }
                self.gradient_history[name].append((epoch, grad_stats))
    
    def log_gradient_norm(self, grad_norm: float):
        """记录梯度范数"""
        self.gradient_norms.append(grad_norm)
    
    def log_prediction_sample(self, 
                            y_true: torch.Tensor,
                            y_pred: torch.Tensor,
                            device_names: List[str],
                            epoch: int):
        """记录预测样本"""
        # 转换为numpy并取第一个样本
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        sample = {
            'epoch': epoch,
            'y_true': y_true[0] if len(y_true.shape) > 1 else y_true,
            'y_pred': y_pred[0] if len(y_pred.shape) > 1 else y_pred,
            'device_names': device_names,
            'timestamp': datetime.now().isoformat()
        }
        
        self.prediction_samples.append(sample)
    
    def plot_training_curves(self, save: bool = True) -> Optional[plt.Figure]:
        """绘制训练曲线"""
        if not self.train_losses:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        epochs_array = np.array(self.epochs)
        train_losses_array = np.array(self.train_losses)
        
        axes[0, 0].plot(epochs_array, train_losses_array, label='训练损失', alpha=0.7)
        if self.val_losses:
            # 验证损失通常比训练损失少，需要插值或者只在有验证数据的点绘制
            val_epochs = epochs_array[::len(epochs_array)//len(self.val_losses)] if len(self.val_losses) > 0 else []
            if len(val_epochs) == len(self.val_losses):
                axes[0, 0].plot(val_epochs, list(self.val_losses), label='验证损失', alpha=0.7)
        
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 学习率曲线
        if self.learning_rates:
            axes[0, 1].plot(epochs_array, list(self.learning_rates), color='orange', alpha=0.7)
            axes[0, 1].set_title('学习率变化')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('学习率')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 梯度范数
        if self.gradient_norms:
            axes[1, 0].plot(epochs_array, list(self.gradient_norms), color='red', alpha=0.7)
            axes[1, 0].set_title('梯度范数')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('梯度范数')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 批次时间
        if self.batch_times:
            axes[1, 1].plot(epochs_array, list(self.batch_times), color='green', alpha=0.7)
            axes[1, 1].set_title('批次处理时间')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('时间 (秒)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            return None
        else:
            return fig
    
    def plot_metrics_evolution(self, save: bool = True) -> Optional[plt.Figure]:
        """绘制指标演化"""
        if not self.train_metrics:
            return None
        
        metric_names = list(self.train_metrics.keys())
        n_metrics = len(metric_names)
        
        if n_metrics == 0:
            return None
        
        # 计算子图布局
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        epochs_array = np.array(self.epochs)
        
        for i, metric_name in enumerate(metric_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 训练指标
            train_values = list(self.train_metrics[metric_name])
            if train_values:
                ax.plot(epochs_array[:len(train_values)], train_values, 
                       label=f'训练 {metric_name}', alpha=0.7)
            
            # 验证指标
            if metric_name in self.val_metrics:
                val_values = list(self.val_metrics[metric_name])
                if val_values:
                    val_epochs = epochs_array[::len(epochs_array)//len(val_values)] if len(val_values) > 0 else []
                    if len(val_epochs) == len(val_values):
                        ax.plot(val_epochs, val_values, 
                               label=f'验证 {metric_name}', alpha=0.7)
            
            ax.set_title(f'{metric_name} 演化')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'metrics_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
            return None
        else:
            return fig
    
    def plot_weight_distributions(self, save: bool = True) -> Optional[plt.Figure]:
        """绘制权重分布"""
        if not self.weight_history:
            return None
        
        # 选择几个主要层进行可视化
        layer_names = list(self.weight_history.keys())[:6]  # 最多显示6个层
        
        if not layer_names:
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, layer_name in enumerate(layer_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            history = self.weight_history[layer_name]
            
            if not history:
                continue
            
            epochs = [item[0] for item in history]
            norms = [item[1]['norm'] for item in history]
            means = [item[1]['mean'] for item in history]
            stds = [item[1]['std'] for item in history]
            
            # 绘制权重范数
            ax2 = ax.twinx()
            
            line1 = ax.plot(epochs, means, 'b-', alpha=0.7, label='均值')
            line2 = ax.plot(epochs, stds, 'g-', alpha=0.7, label='标准差')
            line3 = ax2.plot(epochs, norms, 'r-', alpha=0.7, label='范数')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('权重统计', color='b')
            ax2.set_ylabel('权重范数', color='r')
            ax.set_title(f'{layer_name.split(".")[-1]} 权重演化')
            
            # 合并图例
            lines = line1 + line2 + line3
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(layer_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'weight_distributions.png', dpi=300, bbox_inches='tight')
            plt.close()
            return None
        else:
            return fig
    
    def plot_gradient_flow(self, save: bool = True) -> Optional[plt.Figure]:
        """绘制梯度流"""
        if not self.gradient_history:
            return None
        
        # 选择几个主要层
        layer_names = list(self.gradient_history.keys())[:6]
        
        if not layer_names:
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, layer_name in enumerate(layer_names):
            if i >= len(axes):
                break
                
            ax = axes[i]
            history = self.gradient_history[layer_name]
            
            if not history:
                continue
            
            epochs = [item[0] for item in history]
            norms = [item[1]['norm'] for item in history]
            means = [abs(item[1]['mean']) for item in history]  # 取绝对值
            
            ax.semilogy(epochs, norms, 'r-', alpha=0.7, label='梯度范数')
            ax.semilogy(epochs, means, 'b-', alpha=0.7, label='梯度均值(绝对值)')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('梯度大小 (log scale)')
            ax.set_title(f'{layer_name.split(".")[-1]} 梯度流')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(layer_names), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'gradient_flow.png', dpi=300, bbox_inches='tight')
            plt.close()
            return None
        else:
            return fig
    
    def plot_prediction_samples(self, save: bool = True) -> Optional[plt.Figure]:
        """绘制预测样本"""
        if not self.prediction_samples:
            return None
        
        # 取最近的几个样本
        recent_samples = list(self.prediction_samples)[-4:]  # 最多显示4个样本
        
        fig, axes = plt.subplots(len(recent_samples), 1, figsize=(15, 4*len(recent_samples)))
        if len(recent_samples) == 1:
            axes = [axes]
        
        for i, sample in enumerate(recent_samples):
            ax = axes[i]
            
            y_true = sample['y_true']
            y_pred = sample['y_pred']
            device_names = sample['device_names']
            epoch = sample['epoch']
            
            # 确保是1D数组
            if len(y_true.shape) > 1:
                y_true = y_true.flatten()
            if len(y_pred.shape) > 1:
                y_pred = y_pred.flatten()
            
            x = np.arange(len(y_true))
            
            ax.plot(x, y_true, 'b-', alpha=0.7, label='真实值', linewidth=2)
            ax.plot(x, y_pred, 'r--', alpha=0.7, label='预测值', linewidth=2)
            
            ax.set_title(f'Epoch {epoch} 预测样本对比')
            ax.set_xlabel('时间步')
            ax.set_ylabel('功率 (W)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 计算并显示MAE
            mae = np.mean(np.abs(y_true - y_pred))
            ax.text(0.02, 0.98, f'MAE: {mae:.2f}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'prediction_samples.png', dpi=300, bbox_inches='tight')
            plt.close()
            return None
        else:
            return fig
    
    def create_interactive_dashboard(self):
        """创建交互式仪表板"""
        try:
            # 创建子图
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('训练损失', '学习率', '梯度范数', '批次时间', '指标演化', '权重统计'),
                specs=[[{"secondary_y": True}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": True}, {"secondary_y": False}]]
            )
            
            epochs_array = list(self.epochs) if self.epochs else []
            
            # 训练损失
            if self.train_losses:
                fig.add_trace(
                    go.Scatter(x=epochs_array, y=list(self.train_losses), 
                             name='训练损失', line=dict(color='blue')),
                    row=1, col=1
                )
            
            if self.val_losses:
                val_epochs = epochs_array[::len(epochs_array)//len(self.val_losses)] if len(self.val_losses) > 0 else []
                if len(val_epochs) == len(self.val_losses):
                    fig.add_trace(
                        go.Scatter(x=val_epochs, y=list(self.val_losses), 
                                 name='验证损失', line=dict(color='red')),
                        row=1, col=1
                    )
            
            # 学习率
            if self.learning_rates:
                fig.add_trace(
                    go.Scatter(x=epochs_array, y=list(self.learning_rates), 
                             name='学习率', line=dict(color='orange')),
                    row=1, col=2
                )
            
            # 梯度范数
            if self.gradient_norms:
                fig.add_trace(
                    go.Scatter(x=epochs_array, y=list(self.gradient_norms), 
                             name='梯度范数', line=dict(color='green')),
                    row=2, col=1
                )
            
            # 批次时间
            if self.batch_times:
                fig.add_trace(
                    go.Scatter(x=epochs_array, y=list(self.batch_times), 
                             name='批次时间', line=dict(color='purple')),
                    row=2, col=2
                )
            
            # 更新布局
            fig.update_layout(
                height=900,
                title_text="训练过程实时监控仪表板",
                showlegend=True
            )
            
            # 设置y轴为对数刻度（学习率和梯度范数）
            fig.update_yaxes(type="log", row=1, col=2)
            fig.update_yaxes(type="log", row=2, col=1)
            
            # 保存为HTML
            fig.write_html(str(self.save_dir / 'training_dashboard.html'))
            
            logger.info(f"交互式仪表板已保存到: {self.save_dir / 'training_dashboard.html'}")
            
        except Exception as e:
            logger.warning(f"创建交互式仪表板失败: {e}")
    
    def save_training_history(self):
        """保存训练历史到JSON文件"""
        history = {
            'epochs': list(self.epochs),
            'train_losses': list(self.train_losses),
            'val_losses': list(self.val_losses),
            'learning_rates': list(self.learning_rates),
            'gradient_norms': list(self.gradient_norms),
            'batch_times': list(self.batch_times),
            'train_metrics': {k: list(v) for k, v in self.train_metrics.items()},
            'val_metrics': {k: list(v) for k, v in self.val_metrics.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练历史已保存到: {history_path}")
    
    def load_training_history(self, history_path: Path):
        """从JSON文件加载训练历史"""
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            self.epochs = deque(history.get('epochs', []), maxlen=self.max_history)
            self.train_losses = deque(history.get('train_losses', []), maxlen=self.max_history)
            self.val_losses = deque(history.get('val_losses', []), maxlen=self.max_history)
            self.learning_rates = deque(history.get('learning_rates', []), maxlen=self.max_history)
            self.gradient_norms = deque(history.get('gradient_norms', []), maxlen=self.max_history)
            self.batch_times = deque(history.get('batch_times', []), maxlen=self.max_history)
            
            for metric_name, values in history.get('train_metrics', {}).items():
                self.train_metrics[metric_name] = deque(values, maxlen=self.max_history)
            
            for metric_name, values in history.get('val_metrics', {}).items():
                self.val_metrics[metric_name] = deque(values, maxlen=self.max_history)
            
            logger.info(f"训练历史已从 {history_path} 加载")
            
        except Exception as e:
            logger.error(f"加载训练历史失败: {e}")
    
    def generate_all_plots(self):
        """生成所有可视化图表"""
        logger.info("生成所有训练可视化图表...")
        
        self.plot_training_curves(save=True)
        self.plot_metrics_evolution(save=True)
        self.plot_weight_distributions(save=True)
        self.plot_gradient_flow(save=True)
        self.plot_prediction_samples(save=True)
        self.create_interactive_dashboard()
        self.save_training_history()
        
        logger.info(f"所有可视化图表已保存到: {self.save_dir}")
    
    def reset(self):
        """重置所有历史记录"""
        self.train_losses.clear()
        self.val_losses.clear()
        self.train_metrics.clear()
        self.val_metrics.clear()
        self.learning_rates.clear()
        self.gradient_norms.clear()
        self.epochs.clear()
        self.batch_times.clear()
        self.weight_history.clear()
        self.gradient_history.clear()
        self.prediction_samples.clear()
        self.step_count = 0
        
        logger.info("训练可视化器已重置")

class RealTimeMonitor:
    """实时监控器"""
    
    def __init__(self, visualizer: TrainingVisualizer, update_interval: int = 100):
        """
        初始化实时监控器
        
        Args:
            visualizer: 训练可视化器
            update_interval: 更新间隔（步数）
        """
        self.visualizer = visualizer
        self.update_interval = update_interval
        self.step_count = 0
    
    def update(self, force: bool = False):
        """更新可视化"""
        self.step_count += 1
        
        if force or self.step_count % self.update_interval == 0:
            self.visualizer.plot_training_curves(save=True)
            self.visualizer.plot_metrics_evolution(save=True)
            self.visualizer.create_interactive_dashboard()

def create_training_visualizer(save_dir: Optional[Path] = None,
                             max_history: int = 1000,
                             update_frequency: int = 10) -> TrainingVisualizer:
    """创建训练可视化器的便捷函数"""
    return TrainingVisualizer(save_dir, max_history, update_frequency)

def create_real_time_monitor(visualizer: TrainingVisualizer,
                           update_interval: int = 100) -> RealTimeMonitor:
    """创建实时监控器的便捷函数"""
    return RealTimeMonitor(visualizer, update_interval)