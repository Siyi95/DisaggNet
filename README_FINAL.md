# Enhanced Transformer NILM Model

## 项目简介

这是一个增强版的Transformer NILM（非侵入式负荷监测）模型，结合了Transformer、CNN、LSTM的优势，具有多尺度特征提取和注意力机制。**注意：本模型专注于模型架构创新，如需防泄漏数据处理，请使用项目中的 `robust_dataset.py`。**

## 核心特性

- **多尺度卷积特征提取**：使用不同卷积核尺寸并行处理
- **Transformer架构**：局部窗口注意力机制，提高计算效率
- **双向LSTM**：捕获时序依赖关系
- **通道注意力机制**：自适应特征权重调整
- **多任务学习**：同时预测功率和设备状态
- **数值稳定性**：全面的NaN处理和梯度裁剪
- **Optuna超参数优化**：自动化参数搜索

## 项目结构

```
DisaggNet/
├── enhanced_transformer_nilm_model.py  # 🎯 核心增强模型（主文件）
├── src/nilm_disaggregation/data/
│   └── robust_dataset.py              # 🔒 防泄漏数据处理（推荐配合使用）
├── Dataset/                            # 📊 数据集目录
│   └── dataverse_files/
│       └── AMPds2.h5                  # AMPds2数据集（23个电表）
├── outputs/                            # 📈 输出目录
│   ├── model_checkpoints/              # 模型检查点
│   ├── training_logs/                  # 训练日志
│   └── visualization/                  # 可视化结果
├── LICENSE                             # 📄 许可证
└── README_FINAL.md                     # 📖 项目说明（本文件）
```

## 快速开始

### 1. 基础模型训练

```bash
# 直接运行增强Transformer模型
python enhanced_transformer_nilm_model.py
```

### 2. 推荐：结合防泄漏数据处理

```python
# 在代码中结合使用防泄漏数据处理
from src.nilm_disaggregation.data.robust_dataset import RobustNILMDataModule

# 创建防泄漏数据模块
data_module = RobustNILMDataModule(
    data_path='Dataset/dataverse_files/AMPds2.h5',
    sequence_length=64,
    batch_size=32,
    cv_mode=True  # 启用Walk-Forward交叉验证
)

# 然后使用enhanced_transformer_nilm_model.py中的模型进行训练
```

### 3. 查看演示效果

```bash
# 运行防泄漏技术演示（推荐先运行）
python demo_robust_dataset.py

# 查看真实数据可视化
python test_real_ampds2_data.py
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

### 模型训练输出
- `outputs/`: 模型检查点、训练日志、损失曲线等
- 自动保存最佳模型和训练历史

### 推荐配合使用的可视化输出
- `outputs/robust_dataset_demo/`: 防泄漏技术演示结果
- `outputs/real_ampds2_test/`: 真实AMPds2数据测试和可视化结果
  - 包含23个电表的详细功率分析图表
  - Walk-Forward交叉验证可视化
  - 个别电表功率时序分析

## 与防泄漏数据处理的集成

本增强Transformer模型可以与项目中的防泄漏数据处理技术完美结合：

1. **数据处理**: 使用 `robust_dataset.py` 进行防泄漏数据预处理
2. **模型训练**: 使用 `enhanced_transformer_nilm_model.py` 进行模型训练
3. **结果验证**: 通过Walk-Forward交叉验证确保模型真实性能

这种组合提供了业界最先进的NILM解决方案，既有创新的模型架构，又有严格的数据泄漏防护。