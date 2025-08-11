# NILM训练系统使用指南

本项目提供了一个完整的NILM（非侵入式负荷监测）训练系统，解决了以下问题：

1. **功率损失不稳定问题** - 通过改进的损失函数和训练策略
2. **预训练+微调流程** - 实现了完整的自监督预训练和监督微调
3. **数据增强** - 丰富的数据增强技术解决数据不足问题

## 文件结构

```
src/
├── train_complete.py          # 完整训练脚本（推荐使用）
├── train_improved.py          # 改进的训练模块
├── data_augmentation.py       # 数据增强模块
├── enhanced_datamodule.py     # 增强的数据模块
├── train_pretrain.py          # 原始预训练脚本
├── train_finetune.py          # 原始微调脚本
└── train_simple.py            # 简单训练脚本

configs/
└── complete_training.yaml     # 完整训练配置文件
```

## 快速开始

### 1. 使用默认配置运行完整训练

```bash
cd src
python train_complete.py
```

### 2. 使用自定义配置文件

```bash
python train_complete.py --config ../configs/complete_training.yaml
```

### 3. 指定数据路径和输出目录

```bash
python train_complete.py \
    --data-path /path/to/your/AMPds2.h5 \
    --output-dir ./my_outputs
```

### 4. 仅运行预训练

```bash
python train_complete.py --pretrain-only
```

### 5. 仅运行微调（需要预训练模型）

```bash
python train_complete.py \
    --finetune-only \
    --pretrained-model ./outputs/pretrain/checkpoints/pretrain-epoch=29-val_loss=0.123.ckpt
```

## 主要改进

### 1. 功率损失稳定化

- **损失平滑**: 使用指数移动平均平滑功率损失
- **梯度裁剪**: 防止梯度爆炸
- **功率缩放**: 将功率值缩放到合适范围
- **分离学习率**: 编码器和头部使用不同学习率

### 2. 数据增强技术

- **高斯噪声**: 添加适量噪声提高鲁棒性
- **幅度缩放**: 模拟不同功率水平
- **时间抖动**: 增加时间序列的多样性
- **通道丢弃**: 提高对传感器故障的鲁棒性
- **频率偏移**: 模拟频率变化
- **合成叠加**: 使用设备模式库生成合成数据
- **Mixup/CutMix**: 先进的数据增强技术

### 3. 预训练+微调流程

- **自监督预训练**: 使用掩码重建任务学习通用特征
- **监督微调**: 在预训练基础上进行任务特定训练
- **渐进解冻**: 逐步解冻编码器参数

## 配置说明

### 数据配置

```yaml
data:
  data_path: '/path/to/AMPds2.h5'  # 数据文件路径
  window_length: 128              # 时间窗口长度
  step_size: 32                   # 滑动步长
  batch_size: 16                  # 批次大小
  augment: true                   # 是否启用数据增强
  min_samples: 200                # 最小样本数
```

### 预训练配置

```yaml
pretrain:
  d_model: 128                    # 模型维度
  mask_ratio: 0.15                # 掩码比例
  learning_rate: 0.001            # 学习率
  max_epochs: 30                  # 最大训练轮数
```

### 微调配置

```yaml
finetune:
  power_loss_weight: 1.0          # 功率损失权重
  event_loss_weight: 0.5          # 事件损失权重
  freeze_encoder: true            # 是否冻结编码器
  encoder_lr: 0.0001              # 编码器学习率
  head_lr: 0.001                  # 头部学习率
  power_loss_smoothing: 0.1       # 功率损失平滑系数
  gradient_clip_val: 1.0          # 梯度裁剪值
  power_scale_factor: 1000.0      # 功率缩放因子
```

## 监控训练过程

### 使用TensorBoard

```bash
tensorboard --logdir ./outputs
```

然后在浏览器中访问 `http://localhost:6006`

### 关键指标

- **预训练阶段**:
  - `train_loss`: 训练损失
  - `val_loss`: 验证损失
  - `reconstruction_loss`: 重建损失

- **微调阶段**:
  - `power_mae`: 功率平均绝对误差
  - `power_rmse`: 功率均方根误差
  - `event_f1`: 事件检测F1分数
  - `total_loss`: 总损失

## 故障排除

### 1. 数据文件不存在

确保AMPds2.h5文件路径正确，或使用 `--data-path` 参数指定正确路径。

### 2. 内存不足

- 减小 `batch_size`
- 减小 `window_length`
- 减少 `num_workers`

### 3. 功率损失仍然不稳定

- 增加 `power_loss_smoothing` 值
- 减小 `head_lr` 学习率
- 增加 `gradient_clip_val` 值

### 4. 训练速度慢

- 增加 `batch_size`（如果内存允许）
- 使用GPU训练
- 增加 `num_workers`

## 高级用法

### 自定义数据增强

修改 `data_augmentation.py` 中的参数或添加新的增强方法。

### 自定义模型架构

修改 `train_improved.py` 中的 `ImprovedNILMModel` 类。

### 自定义损失函数

在 `heads.py` 中修改损失计算逻辑。

## 性能优化建议

1. **数据预处理**: 预先计算并缓存特征
2. **混合精度**: 使用16位精度训练
3. **梯度累积**: 在小批次上累积梯度
4. **学习率调度**: 使用学习率调度器
5. **模型剪枝**: 移除不重要的连接

## 实验记录

建议记录以下信息：
- 配置参数
- 训练时间
- 最佳验证指标
- 模型大小
- 推理速度

这样可以帮助优化和复现实验结果。