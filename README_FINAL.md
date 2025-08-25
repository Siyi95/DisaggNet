# Enhanced Transformer NILM Model

## 项目简介

这是一个增强版的Transformer NILM（非侵入式负荷监测）模型，结合了Transformer、CNN、LSTM的优势，具有多尺度特征提取和注意力机制。

## 核心特性

- **多尺度卷积特征提取**：使用不同卷积核尺寸并行处理
- **Transformer架构**：局部窗口注意力机制，提高计算效率
- **双向LSTM**：捕获时序依赖关系
- **通道注意力机制**：自适应特征权重调整
- **多任务学习**：同时预测功率和设备状态
- **数值稳定性**：全面的NaN处理和梯度裁剪

## 项目结构

```
DisaggNet/
├── enhanced_transformer_nilm_model.py  # 核心增强模型（主文件）
├── outputs/                            # 统一输出目录
│   ├── optuna_hyperparameter_optimization.py  # 超参数优化
│   ├── draw_detailed_network_architecture.py  # 网络架构可视化
│   ├── detailed_network_architecture.png      # 主网络架构图
│   ├── detailed_network_architecture.pdf      # 主网络架构图（PDF）
│   ├── network_components_breakdown.png       # 组件分解图
│   └── network_components_breakdown.pdf       # 组件分解图（PDF）
├── Dataset/                            # 数据集目录
│   └── dataverse_files/
│       └── AMPds2.h5                  # AMPds2数据集
├── LICENSE                             # 许可证
└── README_FINAL.md                     # 项目说明（本文件）
```

## 快速开始

### 1. 训练模型

```bash
python enhanced_transformer_nilm_model.py
```

### 2. 超参数优化

```bash
python outputs/optuna_hyperparameter_optimization.py
```

### 3. 生成网络架构图

```bash
python outputs/draw_detailed_network_architecture.py
```

## 模型架构

增强版Transformer NILM模型包含以下核心组件：

1. **输入嵌入层**：将原始功率信号转换为高维特征
2. **多尺度卷积块**：并行处理不同尺度的特征
3. **通道注意力**：自适应调整特征通道权重
4. **位置编码**：为序列添加位置信息
5. **增强Transformer块**：
   - 局部窗口多头注意力
   - CNN分支（卷积特征提取）
   - LSTM分支（时序建模）
   - 分支融合层
   - 前馈网络
6. **双向LSTM**：捕获长期时序依赖
7. **时间注意力**：关注重要时间步
8. **多任务输出头**：功率预测 + 状态预测

## 技术特点

- **数值稳定性**：全面的NaN值处理和数值裁剪
- **高效训练**：梯度裁剪和学习率调度
- **多任务学习**：联合优化功率和状态预测
- **自动化优化**：集成Optuna超参数搜索
- **可视化支持**：详细的网络架构图生成

## 依赖环境

- Python 3.8+
- PyTorch 1.12+
- PyTorch Lightning
- Optuna
- NumPy
- Pandas
- Matplotlib
- Seaborn
- h5py
- scikit-learn

## 输出说明

所有训练结果、模型检查点、可视化图表和优化结果都将保存在 `outputs/` 目录下，便于统一管理和查看。