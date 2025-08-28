# DisaggNet - 时序数据泄漏防护的NILM系统

基于深度学习的非侵入式负荷监测(NILM)系统，专注于解决时序数据泄漏问题，提供业界最先进的数据处理管道。

## 🔒 核心特性

### 时序数据泄漏防护（6大技术）
- **Purged/Embargo Walk-Forward**: 时序交叉验证，防止未来信息泄漏
- **先分割后预处理**: 防止标准化参数泄漏
- **验证集非重叠窗口**: 防止验证集内部相似性偏置
- **训练集小步长采样**: 最大化训练数据利用
- **标签/阈值防泄漏**: 防止阈值计算信息泄漏
- **特征工程分片内独立**: 防止特征统计信息泄漏

### 模型架构
- **增强Transformer**: 多尺度卷积、改进位置编码、增强注意力机制
- **自适应损失函数**: 动态调整功率、状态和相关性损失权重
- **多任务学习**: 同时预测功率和设备状态

### 数据处理
- **AMPds2完整支持**: 23个电表的完整数据加载和预处理
- **PyTorch Lightning集成**: 标准化数据接口
- **自适应阈值优化**: 设备特定的智能阈值计算
- **标签平衡策略**: 解决设备状态分布不均问题

## 📁 项目结构

```
DisaggNet/
├── 📊 核心数据处理（最佳实践）
│   └── src/nilm_disaggregation/data/
│       ├── robust_dataset.py          # ✅ 唯一数据集实现（集成6大防泄漏技术）
│       └── __init__.py
│
├── 🧪 演示和测试
│   ├── demo_robust_dataset.py         # ✅ 防泄漏技术演示
│   └── test_real_ampds2_data.py       # ✅ 真实数据测试和可视化
│
├── 🎯 模型和训练
│   ├── enhanced_transformer_nilm_model.py  # 增强Transformer模型
│   ├── main.py                        # 主入口脚本
│   ├── debug_simple_train.py          # 调试训练脚本
│   └── src/nilm_disaggregation/
│       ├── models/                    # 模型架构
│       ├── training/                  # 训练相关
│       └── utils/                     # 工具函数
│
├── 📖 文档和指南
│   ├── README.md                      # 📄 项目主说明（本文件）
│   ├── README_FINAL.md                # 📄 增强模型说明
│   ├── README_时序数据泄漏防护技术详解.md  # 📚 技术详解文档
│   ├── NILM_最佳实践使用指南.md        # 🚀 使用指南
│   └── 项目结构说明.md                # 📋 结构说明
│
├── 📊 数据和配置
│   ├── Dataset/dataverse_files/       # 🗃️ AMPds2数据集（23个电表）
│   ├── configs/default_config.yaml   # ⚙️ 配置文件
│   └── requirements.txt               # 📦 依赖包
│
└── 📈 输出结果
    └── outputs/                       # 训练结果、可视化图表等
```

## 🚀 快速开始

### 安装依赖

```bash
pip install torch pytorch-lightning h5py pandas numpy matplotlib seaborn scikit-learn
```

### 数据集准备

1. 下载AMPds2数据集
2. 将 `AMPds2.h5` 文件放置在 `Dataset/dataverse_files/` 目录下
3. 数据集包含23个电表的完整功率数据

### 核心使用方式

#### 1. 防泄漏技术演示
```bash
# 运行完整的防泄漏技术演示
python demo_robust_dataset.py

# 查看生成的可视化结果
# outputs/robust_dataset_demo/walk_forward_validation.png
```

#### 2. 真实数据测试和可视化
```bash
# 测试真实AMPds2数据并生成详细可视化
python test_real_ampds2_data.py

# 查看生成的图表：
# - 完整电表概览
# - Walk-Forward交叉验证结果
# - 个别电表功率分析
# - 23个电表详细功率曲线
```

#### 3. 在代码中使用防泄漏数据处理
```python
from src.nilm_disaggregation.data.robust_dataset import RobustNILMDataModule
import pytorch_lightning as pl

# 创建数据模块（自动启用所有6个防泄漏技术）
data_module = RobustNILMDataModule(
    data_path='Dataset/dataverse_files/AMPds2.h5',
    sequence_length=64,
    batch_size=32,
    cv_mode=True,           # 启用Walk-Forward交叉验证
    train_stride=1,         # 训练集小步长采样
    val_stride=64          # 验证集非重叠窗口
)

# 与PyTorch Lightning无缝集成
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, data_module)
```

#### 4. 增强Transformer模型训练
```bash
# 运行增强Transformer NILM模型
python enhanced_transformer_nilm_model.py
```

## 📊 输出说明

### 演示输出
- `outputs/robust_dataset_demo/`: 防泄漏技术演示结果
  - `walk_forward_validation.png`: Walk-Forward交叉验证可视化

### 真实数据测试输出
- `outputs/real_ampds2_test/`: 真实数据测试结果
  - `1_complete_meter_overview.png`: 23个电表总览
  - `2_walk_forward_cv.png`: Walk-Forward交叉验证结果
  - `3_individual_meter_analysis.png`: 个别电表分析（前8个）
  - `4_individual_meter_analysis.png`: 个别电表分析（中8个）
  - `5_individual_meter_analysis.png`: 个别电表分析（后7个）
  - `4_complete_meter_details.png`: 所有23个电表详细功率曲线

### 模型训练输出
- `outputs/`: 模型检查点、训练日志等

## 🔧 技术特点

### AMPds2数据集支持
- **23个电表完整映射**: 从meter1(WHE)到meter23(UNE)
- **真实设备类型**: 主电表、房间插座、大功率设备、系统设备等
- **智能功率模式**: 根据设备类型生成合理的功率变化模式

### 防泄漏技术验证
- **时序严格性**: Walk-Forward确保训练集始终在验证集之前
- **Embargo间隔**: 24小时禁运期防止边界泄漏
- **预处理隔离**: 标准化参数只在训练集计算
- **阈值防泄漏**: 设备状态阈值只在训练分片估计

### 可视化功能
- **功率时序图**: 显示实际天数而非抽象时间点
- **统计信息**: 每个电表的均值、最大值、最小值（原始瓦特值）
- **标准分割显示**: 70/30或85/15等标准比例而非随机数
- **验证批次优化**: 通过调整步长增加验证数据利用率

## 📚 文档指南

- **`NILM_最佳实践使用指南.md`**: 详细的使用指南和代码示例
- **`README_时序数据泄漏防护技术详解.md`**: 深入的技术原理解释
- **`项目结构说明.md`**: 项目文件结构和优化说明
- **`README_FINAL.md`**: 增强Transformer模型的详细说明

## 💡 关键优势

- ✅ **零数据泄漏**: 6层防护确保模型真实性能
- ✅ **标签优化**: 自适应阈值提升检测精度  
- ✅ **时序严格**: Walk-Forward模拟真实部署场景
- ✅ **代码简洁**: 单一实现替代多个过时文件
- ✅ **易于使用**: 一行代码启用所有防护技术
- ✅ **完整可视化**: 详细的图表展示所有23个电表数据

## 📄 许可证

本项目采用MIT许可证。