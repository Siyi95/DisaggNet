import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# 添加项目根目录到Python路径
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入字体配置模块
from src.nilm_disaggregation.utils.font_config import setup_chinese_fonts

# 设置绘图样式和中文字体支持
setup_chinese_fonts()

def create_detailed_network_architecture():
    """
    创建详细的增强版Transformer NILM网络结构图，标注所有不同的网络结构组件
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 24)
    ax.axis('off')
    
    # 定义颜色方案
    colors = {
        'input': '#E8F4FD',
        'cnn': '#FFE6CC',
        'transformer': '#E6F3FF',
        'lstm': '#F0E6FF',
        'attention': '#FFE6E6',
        'fusion': '#E6FFE6',
        'output': '#FFF0E6',
        'connection': '#666666'
    }
    
    # 绘制模块的辅助函数
    def draw_detailed_module(x, y, width, height, text, color, text_size=10, subtext=None):
        # 主模块框
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(box)
        
        # 主标题
        ax.text(x + width/2, y + height*0.7, text, 
                ha='center', va='center', fontsize=text_size, fontweight='bold')
        
        # 子标题（如果有）
        if subtext:
            ax.text(x + width/2, y + height*0.3, subtext, 
                    ha='center', va='center', fontsize=text_size-2, style='italic')
    
    def draw_connection_arrow(start_x, start_y, end_x, end_y, text=None):
        arrow = ConnectionPatch(
            (start_x, start_y), (end_x, end_y),
            "data", "data",
            arrowstyle="->",
            shrinkA=5, shrinkB=5,
            mutation_scale=20,
            fc=colors['connection'],
            ec=colors['connection'],
            linewidth=2
        )
        ax.add_patch(arrow)
        
        if text:
            mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
            ax.text(mid_x + 0.3, mid_y, text, fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # 1. 输入层
    draw_detailed_module(4, 22, 2, 1, "输入层", colors['input'], 12, "时间序列数据\n[batch, seq_len, 1]")
    
    # 2. 输入嵌入层
    draw_detailed_module(4, 20.5, 2, 1, "输入嵌入层", colors['input'], 11, "Linear Projection\n[batch, seq_len, d_model]")
    
    # 3. 多尺度卷积特征提取
    draw_detailed_module(1, 18.5, 3, 1.5, "多尺度卷积块", colors['cnn'], 11, "CNN特征提取\n3个并行卷积分支\nkernel_size: 3,5,7")
    draw_detailed_module(6, 18.5, 3, 1.5, "通道注意力机制", colors['attention'], 11, "Channel Attention\nSE-Net结构\nGlobal Avg Pool + FC")
    
    # 4. 位置编码
    draw_detailed_module(4, 16.5, 2, 1, "位置编码", colors['transformer'], 11, "Positional Encoding\nSinusoidal Encoding")
    
    # 5. Transformer层（核心部分）
    # 5.1 多头注意力
    draw_detailed_module(0.5, 14, 2.5, 1.5, "多头注意力", colors['transformer'], 10, "Multi-Head Attention\n局部窗口注意力\n相对位置编码")
    
    # 5.2 CNN分支
    draw_detailed_module(3.5, 14, 2, 1.5, "CNN分支", colors['cnn'], 10, "1D Convolution\nBatch Norm\nGELU Activation")
    
    # 5.3 LSTM分支
    draw_detailed_module(6, 14, 2.5, 1.5, "LSTM分支", colors['lstm'], 10, "Bidirectional LSTM\n双向序列建模\nDropout")
    
    # 5.4 分支融合
    draw_detailed_module(3.5, 12, 2, 1, "分支融合层", colors['fusion'], 10, "Feature Fusion\nWeighted Sum")
    
    # 5.5 残差连接和层归一化
    draw_detailed_module(1, 10.5, 3, 1, "残差连接", colors['transformer'], 10, "Residual Connection\n+ Layer Normalization")
    draw_detailed_module(6, 10.5, 3, 1, "前馈网络", colors['transformer'], 10, "Feed Forward\nLinear + GELU + Linear")
    
    # 6. Transformer块堆叠指示
    draw_detailed_module(4, 9, 2, 0.8, "× N层", colors['transformer'], 10, "Transformer Blocks\nStacked N times")
    
    # 7. 双向LSTM层
    draw_detailed_module(3, 7.5, 4, 1.2, "双向LSTM层", colors['lstm'], 11, "Bidirectional LSTM\n序列建模和时间依赖")
    
    # 8. 时间注意力机制
    draw_detailed_module(3, 6, 4, 1, "时间注意力机制", colors['attention'], 11, "Temporal Attention\n时间维度注意力权重")
    
    # 9. 特征融合层
    draw_detailed_module(3.5, 4.5, 3, 1, "特征融合层", colors['fusion'], 11, "Feature Fusion\nCNN + Transformer + LSTM")
    
    # 10. 输出头（多任务）
    draw_detailed_module(1, 2.5, 2.5, 1.5, "功率预测头", colors['output'], 10, "Power Prediction\nLinear + Dropout\n回归任务")
    draw_detailed_module(6, 2.5, 2.5, 1.5, "状态预测头", colors['output'], 10, "State Prediction\nLinear + Sigmoid\n分类任务")
    
    # 11. 最终输出
    draw_detailed_module(3.5, 0.5, 3, 1, "最终输出", colors['output'], 11, "Power & State\n功率值 + 开关状态")
    
    # 绘制连接箭头
    # 主要数据流
    draw_connection_arrow(5, 22, 5, 21.5)
    draw_connection_arrow(5, 20.5, 5, 20)
    
    # 多尺度卷积和注意力
    draw_connection_arrow(5, 20, 2.5, 19.5)
    draw_connection_arrow(5, 20, 7.5, 19.5)
    draw_connection_arrow(2.5, 18.5, 4.5, 17.5)
    draw_connection_arrow(7.5, 18.5, 5.5, 17.5)
    
    # 位置编码
    draw_connection_arrow(5, 16.5, 5, 15.5)
    
    # Transformer内部连接
    draw_connection_arrow(5, 15.5, 1.75, 15.5)
    draw_connection_arrow(5, 15.5, 4.5, 15.5)
    draw_connection_arrow(5, 15.5, 7.25, 15.5)
    
    draw_connection_arrow(1.75, 14, 4.5, 13)
    draw_connection_arrow(4.5, 14, 4.5, 13)
    draw_connection_arrow(7.25, 14, 4.5, 13)
    
    draw_connection_arrow(4.5, 12, 2.5, 11.5)
    draw_connection_arrow(4.5, 12, 7.5, 11.5)
    
    draw_connection_arrow(2.5, 10.5, 5, 9.8)
    draw_connection_arrow(7.5, 10.5, 5, 9.8)
    
    # 后续层连接
    draw_connection_arrow(5, 9, 5, 8.7)
    draw_connection_arrow(5, 7.5, 5, 7)
    draw_connection_arrow(5, 6, 5, 5.5)
    
    # 输出分支
    draw_connection_arrow(4.5, 4.5, 2.25, 4)
    draw_connection_arrow(5.5, 4.5, 7.25, 4)
    
    draw_connection_arrow(2.25, 2.5, 4.5, 1.5)
    draw_connection_arrow(7.25, 2.5, 5.5, 1.5)
    
    # 添加网络结构类型标注
    ax.text(0.5, 23, "增强版Transformer NILM网络架构", fontsize=16, fontweight='bold')
    
    # 添加图例
    legend_elements = [
        patches.Patch(color=colors['input'], label='输入/嵌入层'),
        patches.Patch(color=colors['cnn'], label='CNN组件'),
        patches.Patch(color=colors['transformer'], label='Transformer组件'),
        patches.Patch(color=colors['lstm'], label='LSTM组件'),
        patches.Patch(color=colors['attention'], label='注意力机制'),
        patches.Patch(color=colors['fusion'], label='特征融合'),
        patches.Patch(color=colors['output'], label='输出层')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # 添加网络结构特点说明
    structure_text = (
        "网络结构特点:\n"
        "• 混合架构: CNN + Transformer + LSTM\n"
        "• 多尺度特征提取: 并行卷积分支\n"
        "• 注意力机制: 通道注意力 + 时间注意力\n"
        "• 残差连接: 深度网络稳定训练\n"
        "• 多任务学习: 功率预测 + 状态分类\n"
        "• 端到端优化: 联合损失函数"
    )
    
    ax.text(0.2, 1, structure_text, fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8),
            verticalalignment='bottom')
    
    plt.tight_layout()
    return fig

def create_component_breakdown():
    """
    创建网络组件分解图，详细展示每个组件的内部结构
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Transformer Block详细结构
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_title("Transformer Block 内部结构", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Transformer组件
    transformer_components = [
        (2, 8.5, 6, 1, "输入: [batch, seq_len, d_model]", '#E6F3FF'),
        (1, 7, 3.5, 1, "Multi-Head\nAttention", '#FFE6E6'),
        (5.5, 7, 3.5, 1, "Add & Norm", '#E6FFE6'),
        (2, 5.5, 6, 1, "Feed Forward Network", '#FFE6CC'),
        (2, 4, 6, 1, "Add & Norm", '#E6FFE6'),
        (2, 2.5, 6, 1, "输出: [batch, seq_len, d_model]", '#E6F3FF')
    ]
    
    for x, y, w, h, text, color in transformer_components:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black')
        ax1.add_patch(box)
        ax1.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10)
    
    # 2. CNN分支详细结构
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title("多尺度CNN分支结构", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    cnn_components = [
        (2, 8.5, 6, 1, "输入特征", '#E6F3FF'),
        (0.5, 6.5, 2.5, 1.5, "Conv1D\nkernel=3", '#FFE6CC'),
        (3.75, 6.5, 2.5, 1.5, "Conv1D\nkernel=5", '#FFE6CC'),
        (7, 6.5, 2.5, 1.5, "Conv1D\nkernel=7", '#FFE6CC'),
        (2, 4.5, 6, 1, "Concatenate", '#E6FFE6'),
        (2, 3, 6, 1, "Batch Norm + GELU", '#FFE6CC'),
        (2, 1.5, 6, 1, "输出特征", '#E6F3FF')
    ]
    
    for x, y, w, h, text, color in cnn_components:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black')
        ax2.add_patch(box)
        ax2.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10)
    
    # 3. LSTM分支详细结构
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.set_title("双向LSTM分支结构", fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    lstm_components = [
        (2, 8.5, 6, 1, "输入序列", '#E6F3FF'),
        (1, 6.5, 3.5, 1.5, "Forward\nLSTM", '#F0E6FF'),
        (5.5, 6.5, 3.5, 1.5, "Backward\nLSTM", '#F0E6FF'),
        (2, 4.5, 6, 1, "Concatenate", '#E6FFE6'),
        (2, 3, 6, 1, "Dropout", '#F0E6FF'),
        (2, 1.5, 6, 1, "输出特征", '#E6F3FF')
    ]
    
    for x, y, w, h, text, color in lstm_components:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black')
        ax3.add_patch(box)
        ax3.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10)
    
    # 4. 注意力机制详细结构
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.set_title("注意力机制结构", fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    attention_components = [
        (2, 8.5, 6, 1, "输入特征", '#E6F3FF'),
        (0.5, 6.5, 2.5, 1.5, "Query\nProjection", '#FFE6E6'),
        (3.75, 6.5, 2.5, 1.5, "Key\nProjection", '#FFE6E6'),
        (7, 6.5, 2.5, 1.5, "Value\nProjection", '#FFE6E6'),
        (2, 4.5, 6, 1, "Scaled Dot-Product Attention", '#FFE6E6'),
        (2, 3, 6, 1, "Multi-Head Concat", '#E6FFE6'),
        (2, 1.5, 6, 1, "输出特征", '#E6F3FF')
    ]
    
    for x, y, w, h, text, color in attention_components:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black')
        ax4.add_patch(box)
        ax4.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def main():
    """
    主函数：生成详细的网络结构图
    """
    print("正在生成详细的网络结构图...")
    
    # 生成主要网络架构图
    fig1 = create_detailed_network_architecture()
    fig1.savefig('detailed_network_architecture.png', dpi=300, bbox_inches='tight')
    fig1.savefig('detailed_network_architecture.pdf', bbox_inches='tight')
    print("✓ 详细网络架构图已保存: detailed_network_architecture.png/pdf")
    
    # 生成组件分解图
    fig2 = create_component_breakdown()
    fig2.savefig('network_components_breakdown.png', dpi=300, bbox_inches='tight')
    fig2.savefig('network_components_breakdown.pdf', bbox_inches='tight')
    print("✓ 网络组件分解图已保存: network_components_breakdown.png/pdf")
    
    plt.show()
    print("\n网络结构图生成完成！")
    print("\n生成的文件:")
    print("- detailed_network_architecture.png: 详细网络架构图")
    print("- detailed_network_architecture.pdf: 详细网络架构图(PDF)")
    print("- network_components_breakdown.png: 网络组件分解图")
    print("- network_components_breakdown.pdf: 网络组件分解图(PDF)")

if __name__ == "__main__":
    main()