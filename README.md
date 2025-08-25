# DisaggNet - 增强版Transformer NILM模型

一个基于增强版Transformer架构的非侵入式负荷监测（NILM）深度学习框架，用于家庭电器功耗分解。

## 项目特性

- **增强版Transformer架构**：集成多尺度卷积、通道注意力、位置编码和双向LSTM
- **模块化设计**：基于PyTorch Lightning的清晰代码结构
- **多任务学习**：同时预测功率和设备状态
- **自动超参数优化**：集成Optuna进行自动调优
- **丰富的可视化**：训练曲线、性能指标和功率分解图表
- **灵活的配置系统**：基于YAML的配置管理

## 项目结构

```
DisaggNet/
├── src/
│   └── nilm_disaggregation/
│       ├── data/                 # 数据模块
│       │   ├── __init__.py
│       │   ├── dataset.py        # 数据集定义
│       │   └── datamodule.py     # PyTorch Lightning数据模块
│       ├── models/               # 模型模块
│       │   ├── __init__.py
│       │   ├── components.py     # 模型组件
│       │   └── enhanced_transformer.py  # 主模型
│       ├── training/             # 训练模块
│       │   ├── __init__.py
│       │   └── lightning_module.py  # PyTorch Lightning模块
│       └── utils/                # 工具模块
│           ├── __init__.py
│           ├── config.py         # 配置管理
│           ├── losses.py         # 损失函数
│           ├── metrics.py        # 评估指标
│           └── visualization.py  # 可视化工具
├── scripts/                      # 脚本目录
│   ├── train.py                 # 训练脚本
│   ├── evaluate.py              # 评估脚本
│   └── optimize.py              # 超参数优化脚本
├── configs/                      # 配置文件
│   └── default_config.yaml      # 默认配置
├── requirements.txt              # 依赖包
└── README.md                    # 项目说明
```

## 安装

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (可选，用于GPU加速)

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd DisaggNet

# 创建虚拟环境（推荐）
conda create -n nilm python=3.9
conda activate nilm

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 准备数据

数据应该是HDF5格式，包含以下结构：
- `main_power`: 主功率数据 (N, 1)
- `appliance_power`: 设备功率数据 (N, num_appliances)
- `appliance_status`: 设备状态数据 (N, num_appliances)

### 2. 配置设置

编辑 `configs/default_config.yaml` 文件，设置数据路径和模型参数：

```yaml
data:
  data_path: "path/to/your/data.h5"
  appliances: ["fridge", "washer_dryer", "microwave", "dishwasher"]
  sequence_length: 512
  batch_size: 32

model:
  d_model: 256
  nhead: 8
  num_layers: 6
  dropout: 0.1

training:
  max_epochs: 100
  learning_rate: 0.001
```

### 3. 训练模型

```bash
# 使用默认配置训练
python scripts/train.py --data_path path/to/your/data.h5

# 使用自定义配置
python scripts/train.py --config configs/custom_config.yaml --data_path path/to/your/data.h5

# 从检查点恢复训练
python scripts/train.py --data_path path/to/your/data.h5 --resume_from_checkpoint path/to/checkpoint.ckpt
```

### 4. 评估模型

```bash
# 评估训练好的模型
python scripts/evaluate.py --checkpoint path/to/best_model.ckpt --data_path path/to/your/data.h5

# 不生成可视化图表
python scripts/evaluate.py --checkpoint path/to/best_model.ckpt --data_path path/to/your/data.h5 --no_visualization
```

### 5. 超参数优化

```bash
# 运行Optuna优化
python scripts/optimize.py --data_path path/to/your/data.h5 --n_trials 100

# 使用自定义配置和更多试验
python scripts/optimize.py --config configs/optimization_config.yaml --data_path path/to/your/data.h5 --n_trials 200 --timeout 7200
```

## 模型架构

### 增强版Transformer NILM模型

模型采用多层次的架构设计：

1. **多尺度卷积特征提取**：捕获不同时间尺度的特征
2. **通道注意力机制**：增强重要特征通道
3. **位置编码**：为序列数据添加位置信息
4. **Transformer编码器**：捕获长距离依赖关系
5. **双向LSTM**：处理序列的前后文信息
6. **时间注意力**：关注重要的时间步
7. **特征融合**：结合多层次特征
8. **多任务输出**：同时预测功率和状态

### 损失函数

组合损失函数包括：
- **功率损失**：均方误差（MSE）
- **状态损失**：二元交叉熵（BCE）
- **相关性损失**：增强预测与真实值的相关性

## 使用示例

### 基本训练

```python
from src.nilm_disaggregation.data import NILMDataModule
from src.nilm_disaggregation.training import EnhancedTransformerNILMModule
from src.nilm_disaggregation.utils import load_config
import pytorch_lightning as pl

# 加载配置
config = load_config('configs/default_config.yaml')

# 创建数据模块
data_module = NILMDataModule(
    data_path=config.get('data.data_path'),
    sequence_length=config.get('data.sequence_length'),
    batch_size=config.get('data.batch_size')
)

# 创建模型
model = EnhancedTransformerNILMModule(
    model_params=config.get('model'),
    loss_params=config.get('loss'),
    learning_rate=config.get('training.learning_rate')
)

# 创建训练器
trainer = pl.Trainer(
    max_epochs=config.get('training.max_epochs'),
    accelerator='gpu' if torch.cuda.is_available() else 'cpu'
)

# 训练模型
trainer.fit(model, data_module)
```

## 性能指标

模型评估使用以下指标：

- **MAE (Mean Absolute Error)**：平均绝对误差
- **RMSE (Root Mean Square Error)**：均方根误差
- **R² (Coefficient of Determination)**：决定系数
- **Correlation**：皮尔逊相关系数
- **F1-Score**：设备状态分类的F1分数

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小批次大小
   - 减少序列长度
   - 使用梯度累积

2. **训练不收敛**
   - 调整学习率
   - 检查数据质量
   - 增加正则化

3. **数据加载错误**
   - 检查数据文件格式
   - 验证数据路径
   - 确认设备列表匹配

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证。