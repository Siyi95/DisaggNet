# DisaggNet - 非侵入式负荷监测系统

基于深度学习的非侵入式负荷监测(NILM)系统，使用增强的Transformer架构进行电器负荷分解。

## 核心特性

### 模型架构
- **增强Transformer**: 多尺度卷积、改进位置编码、增强注意力机制
- **自适应损失函数**: 动态调整功率、状态和相关性损失权重
- **多头预测**: 同时预测功率和设备状态

### 训练策略
- **高级优化器**: AdamW、SGD等多种优化器支持
- **学习率调度**: 余弦退火、步长衰减等策略
- **数据增强**: 噪声添加、时间偏移、缩放等技术
- **梯度裁剪**: 防止梯度爆炸
- **早停机制**: 防止过拟合

### 超参数优化
- **Optuna集成**: 高效的贝叶斯优化
- **多维搜索**: 模型架构、训练参数、损失权重等
- **剪枝策略**: 自动终止表现差的试验

### 数据处理
- **AMPds2支持**: 完整的AMPds2数据集加载和预处理
- **CSV格式**: 支持CSV格式数据加载
- **序列化处理**: 时间序列数据的滑动窗口处理
- **数据增强**: 多种数据增强技术

## 项目结构

```
DisaggNet/
├── main.py                    # 主入口脚本
├── src/
│   └── nilm_disaggregation/
│       ├── data/              # 数据处理模块
│       │   ├── complete_ampds2_dataset.py    # 完整AMPds2数据集
│       │   ├── csv_data_loader.py           # CSV数据加载器
│       │   ├── datamodule.py                # PyTorch Lightning数据模块
│       │   ├── dataset.py                   # 基础数据集
│       │   ├── demo_complete_dataset_usage.py # 数据集使用演示
│       │   ├── explore_ampds2_structure.py   # 数据结构探索
│       │   └── improved_dataset.py          # 改进的数据集
│       ├── models/            # 模型架构
│       │   ├── components.py                # 模型组件
│       │   ├── enhanced_model_architecture.py # 增强模型架构
│       │   └── enhanced_transformer.py      # 增强Transformer
│       ├── training/          # 训练相关
│       │   ├── advanced_strategies.py       # 高级训练策略
│       │   ├── evaluate.py                  # 模型评估
│       │   ├── lightning_module.py          # Lightning模块
│       │   ├── optimization.py              # 超参数优化
│       │   └── train.py                     # 训练脚本
│       └── utils/             # 工具函数
│           ├── config.py                    # 配置管理
│           ├── font_config.py               # 字体配置
│           ├── losses.py                    # 损失函数
│           ├── metrics.py                   # 评估指标
│           ├── test_chinese_fonts.py        # 中文字体测试
│           ├── visualization.py             # 可视化工具
│           └── visualize_complete_ampds2.py # AMPds2数据可视化
├── Dataset/                   # 数据集目录
└── outputs/                   # 输出目录
```

## 安装依赖

```bash
pip install torch pytorch-lightning optuna h5py pandas numpy matplotlib seaborn scikit-learn
```

## 数据集准备

1. 下载AMPds2数据集
2. 将数据文件放置在 `Dataset/dataverse_files/` 目录下
3. 确保数据文件格式正确（HDF5或CSV格式）

## 主要功能

### 1. 模型训练
```bash
# 基础训练
python main.py train --data-dir ./Dataset/dataverse_files

# 自定义参数训练
python main.py train --data-dir ./Dataset/dataverse_files --epochs 100 --batch-size 64 --learning-rate 1e-3
```

### 2. 超参数优化
```bash
# 超参数优化
python main.py optimize --data-dir ./Dataset/dataverse_files --trials 50 --type hyperparams

# 损失权重优化
python main.py optimize --data-dir ./Dataset/dataverse_files --trials 20 --type loss_weights
```

### 3. 模型评估
```bash
python main.py evaluate --checkpoint ./outputs/model.ckpt --data-dir ./Dataset/dataverse_files
```

### 4. 数据可视化
```bash
python main.py visualize --data-dir ./Dataset/dataverse_files
```

### 5. 演示运行
```bash
python main.py demo --data-dir ./Dataset/dataverse_files
```

### 6. 数据探索
```bash
python main.py explore --data-dir ./Dataset/dataverse_files
```

## 输出说明

- `outputs/training/`: 训练输出（模型检查点、日志等）
- `outputs/optimization/`: 优化结果（最佳参数、试验历史等）
- `outputs/evaluation/`: 评估结果（指标报告、可视化图表等）
- `outputs/visualization/`: 数据可视化图表
- `outputs/demo/`: 演示运行结果

## 配置文件

项目支持YAML格式的配置文件，可以通过 `--config` 参数指定：

```yaml
model:
  d_model: 256
  n_heads: 8
  n_layers: 4
  dropout: 0.1

training:
  learning_rate: 1e-4
  batch_size: 32
  epochs: 50

data:
  sequence_length: 128
  appliances: ['fridge', 'washer_dryer', 'microwave', 'dishwasher']
```

## 注意事项

1. **内存使用**: 大型数据集可能需要大量内存，建议使用GPU训练
2. **数据格式**: 确保数据格式与加载器兼容
3. **路径设置**: 使用绝对路径避免路径问题
4. **依赖版本**: 确保PyTorch和相关库版本兼容



## 许可证

本项目采用MIT许可证。