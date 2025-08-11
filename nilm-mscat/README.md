# NILM MS-CAT: Multi-Scale Channel-Aware Transformer for Non-Intrusive Load Monitoring

基于 **MS-CAT + 掩蔽预训练 + CRF** 的非侵入式负荷监测系统，集成**因果 TCN 在线启停检测**模块。

## 🏗️ 项目架构

```
nilm-mscat/
├── data/
│   └── AMPds2.h5              # AMPds2 数据集
├── src/
│   ├── datamodule.py          # 数据加载与预处理
│   ├── features.py            # 特征提取模块
│   ├── models/
│   │   ├── mscat.py          # MS-CAT 主模型
│   │   ├── heads.py          # 多任务输出头
│   │   ├── crf.py            # CRF 后处理
│   │   └── tcn_online.py     # 在线 TCN 检测
│   ├── train_pretrain.py      # 掩蔽预训练脚本
│   ├── train_finetune.py      # 监督微调脚本
│   ├── infer_offline.py       # 离线推理脚本
│   └── infer_online_tcn.py    # 在线检测脚本
├── configs/
│   ├── pretrain.yaml         # 预训练配置
│   ├── finetune.yaml         # 微调配置
│   └── online.yaml           # 在线检测配置
├── requirements.txt           # 依赖管理
└── README.md                 # 项目说明
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
conda create -n nilm-mscat python=3.11
conda activate nilm-mscat

# 安装 PyTorch (CUDA 12.8)
pip install --pre torch torchvision torchaudio -i https://download.pytorch.org/whl/nightly/cu128

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 数据准备

将 AMPds2 数据集放置在 `data/AMPds2.h5`：

```bash
# 下载 AMPds2 数据集
# 请从官方网站下载并转换为 HDF5 格式
# 数据结构应包含：
# - /electricity/meter_01 (总功率)
# - /electricity/meter_02-20 (各设备功率)
# - 时间戳索引
```

### 3. 训练流程

#### 步骤 1: 掩蔽预训练

```bash
python src/train_pretrain.py --config configs/pretrain.yaml
```

预训练目标：
- 对输入序列的 15-30% 时间步进行掩蔽
- 使用 MS-CAT 编码器重建被掩蔽的功率值
- 提升模型的表示学习能力

#### 步骤 2: 监督微调

```bash
python src/train_finetune.py --config configs/finetune.yaml --ckpt outputs/pretrain/mscat_pretrain.ckpt
```

微调目标：
- 加载预训练权重
- 多任务学习：功率回归 + 状态分类
- CRF 后处理进行时序平滑

#### 步骤 3: TCN 在线检测训练

```bash
python src/train_finetune.py --config configs/online.yaml --mode tcn
```

在线检测目标：
- 训练轻量级因果 TCN 模型
- 知识蒸馏：从 MS-CAT 学习软标签
- 实时启停检测能力

### 4. 推理与评估

#### 离线批量推理

```bash
python src/infer_offline.py --ckpt outputs/finetune/best.ckpt --days 7 --visualize
```

#### 在线实时检测

```bash
python src/infer_online_tcn.py --ckpt outputs/tcn/tcn_best.ckpt --buffer_size 120 --threshold 0.5
```

## 🧠 模型架构

### MS-CAT (Multi-Scale Channel-Aware Transformer)

```
输入: [batch, seq_len, channels]
  ↓
特征提取器 (ChannelMixer + 位置编码 + 时间特征)
  ↓
双分支架构:
├── Local Branch (局部窗口注意力 + Depthwise Conv)
└── Global Branch (稀疏全局注意力)
  ↓
分支融合 (加权和/拼接)
  ↓
多任务头:
├── 回归头 → 功率预测 [seq_len, n_devices]
└── 分类头 → 状态预测 [seq_len, n_devices]
  ↓
CRF 后处理 → 时序平滑
```

### 关键特性

1. **多通道特征**：
   - 基础：P_total, Q_total, S_total, I, V, PF
   - 派生：ΔP, 滑窗统计, 频域特征, 时间特征

2. **双分支注意力**：
   - Local：捕获短期启停模式
   - Global：建模长期周期性

3. **时序后处理**：
   - 最小持续时间约束
   - CRF/Viterbi 解码
   - 状态平滑

4. **在线检测**：
   - 因果 TCN 架构
   - 知识蒸馏训练
   - 实时推理能力

## 📊 评估指标

### 回归指标
- **MAE/RMSE/SAE**：每设备功率预测误差
- **总功率重建误差**：∑设备预测 vs 真实总功率

### 分类指标
- **Precision/Recall/F1**：每设备启停检测
- **事件检测准确率**：状态变化点检测

### 在线检测指标
- **延迟**：检测到状态变化的时间延迟
- **误报率**：虚假启停事件比例
- **漏检率**：遗漏真实事件比例

## ⚙️ 配置说明

### 关键超参数

```yaml
# 模型架构
d_model: 192              # 特征维度
num_heads: 6              # 注意力头数
local_layers: 4           # 局部分支层数
global_layers: 3          # 全局分支层数
dropout: 0.1              # Dropout 率

# 数据处理
window_size: 120          # 滑窗长度（分钟）
step_size: 60             # 滑窗步长（分钟）
batch_size: 32            # 批大小

# 训练参数
learning_rate: 1e-3       # 初始学习率
weight_decay: 1e-4        # 权重衰减
max_epochs: 100           # 最大训练轮数

# 损失权重
regression_weight: 1.0    # 回归损失权重
classification_weight: 0.5 # 分类损失权重

# CRF 参数
min_on_duration: 5        # 最小开启时长（分钟）
min_off_duration: 3       # 最小关闭时长（分钟）
power_threshold: 10       # 功率阈值（瓦特）
```

## 🔧 扩展功能

### 1. 自定义数据集

继承 `AMPds2Dataset` 类并重写数据加载方法：

```python
class CustomDataset(AMPds2Dataset):
    def load_data(self):
        # 实现自定义数据加载逻辑
        pass
```

### 2. 新增设备类型

在配置文件中添加设备信息：

```yaml
devices:
  - name: "washing_machine"
    meter_id: "meter_02"
    power_threshold: 50
  - name: "dishwasher"
    meter_id: "meter_03"
    power_threshold: 30
```

### 3. 在线数据源集成

支持多种实时数据源：

```python
# MQTT 数据源
from src.data_sources import MQTTDataSource
source = MQTTDataSource(broker="localhost", topic="power/data")

# 串口数据源
from src.data_sources import SerialDataSource
source = SerialDataSource(port="/dev/ttyUSB0", baudrate=9600)
```

## 📈 性能优化

### 1. 模型压缩
- 知识蒸馏：大模型 → 小模型
- 量化：FP32 → INT8
- 剪枝：移除冗余参数

### 2. 推理加速
- TensorRT 优化
- ONNX 导出
- 批处理推理

### 3. 内存优化
- 梯度检查点
- 混合精度训练
- 数据流水线

## 🐛 故障排除

### 常见问题

1. **CUDA 内存不足**
   ```bash
   # 减小批大小或序列长度
   batch_size: 16
   window_size: 60
   ```

2. **训练不收敛**
   ```bash
   # 调整学习率和权重衰减
   learning_rate: 5e-4
   weight_decay: 1e-5
   ```

3. **数据加载慢**
   ```bash
   # 增加数据加载进程数
   num_workers: 16
   pin_memory: true
   ```

## 📚 参考文献

1. Transformer 架构："Attention Is All You Need"
2. NILM 综述："Non-intrusive load monitoring approaches for disaggregated energy sensing"
3. AMPds2 数据集："AMPds2: The Almanac of Minutely Power dataset"
4. CRF 序列标注："Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data"

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系

如有问题，请联系：[your-email@example.com]