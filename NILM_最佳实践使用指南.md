# NILM 时序数据泄漏防护最佳实践使用指南

## 概述

**是的，您可以同时使用所有6个防泄漏技术！** 这些技术是互补的，组合使用能够提供最强的数据泄漏防护效果。

## 🔒 6大核心防泄漏技术

### 1. Purged/Embargo Walk-Forward 交叉验证
**作用**：防止时序依赖泄漏
**实现**：历史训练 → 24小时禁运期 → 未来验证

### 2. 先分割后预处理
**作用**：防止标准化泄漏
**实现**：只在训练集上计算StandardScaler参数

### 3. 验证集非重叠窗口
**作用**：防止验证集内部相似性偏置
**实现**：stride = window_size，杜绝重复样本

### 4. 训练集小步长采样
**作用**：扩充样本量提升学习效果
**实现**：stride = 1，最大化训练数据利用

### 5. 标签/阈值防泄漏
**作用**：防止阈值计算泄漏
**实现**：只在训练分片上估计，验证分片只应用

### 6. 特征工程分片内独立
**作用**：防止特征统计泄漏
**实现**：按fold内训练段估计全局分布特征

## 🚀 完整使用方式

### 基础配置（同时启用所有技术）

```python
from src.nilm_disaggregation.data.robust_dataset import RobustNILMDataModule
import pytorch_lightning as pl

# 创建数据模块 - 自动集成所有6个防泄漏技术
# 现在继承PyTorch Lightning的LightningDataModule
data_module = RobustNILMDataModule(
    data_path='path/to/AMPds2.h5',
    sequence_length=64,
    batch_size=32,
    num_workers=4,
    
    # 技术1: Purged/Embargo Walk-Forward
    cv_mode=True,              # 启用Walk-Forward交叉验证
    current_fold=0,            # 当前fold索引
    
    # 技术3&4: 差异化采样策略
    train_stride=1,            # 训练集小步长（技术4）
    val_stride=64,             # 验证集非重叠窗口（技术3）
    
    # 技术1&5&6: 分割和预处理配置
    split_config={
        'embargo_hours': 24,       # 24小时禁运期
        'purge_hours': 0,          # 清洗期
        'cv_folds': 5,             # 5折交叉验证
        'min_train_hours': 30*24   # 最小训练集大小
    }
)

# 技术2: 先分割后预处理（自动执行）
data_module.setup('fit')  # 自动应用技术2、5、6

# 与PyTorch Lightning Trainer无缝集成
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='auto',
    devices='auto'
)

# 直接使用data_module训练
trainer.fit(model, datamodule=data_module)
```

### Walk-Forward 交叉验证（完整流程）

```python
import pytorch_lightning as pl
from src.nilm_disaggregation.data.robust_dataset import RobustNILMDataModule

# 完整的5折Walk-Forward交叉验证
results = []

# 创建数据模块（启用交叉验证模式）
data_module = RobustNILMDataModule(
    data_path='path/to/AMPds2.h5',
    cv_mode=True,
    train_stride=1,
    val_stride=64
)

for fold in range(5):
    print(f"\n=== Fold {fold + 1}/5 ===")
    
    # 设置当前fold（自动应用所有6个技术）
    data_module.setup_fold(fold)
    
    # 创建模型和训练器
    model = YourNILMModel()
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='auto',
        devices='auto',
        logger=False,  # 可选：关闭日志以简化输出
        enable_checkpointing=False  # 可选：关闭检查点
    )
    
    # 训练（享受所有防泄漏保护）
    trainer.fit(model, datamodule=data_module)
    
    # 验证
    val_results = trainer.validate(model, datamodule=data_module)
    results.append(val_results[0])  # Lightning返回列表
    
    print(f"Fold {fold + 1} 验证结果: {val_results[0]}")

# 计算交叉验证平均结果
avg_results = calculate_cv_average(results)
print(f"\n5折交叉验证平均结果: {avg_results}")
```

### 单次分割模式（快速实验）

```python
# 如果不需要交叉验证，可以使用单次分割
data_module = RobustNILMDataModule(
    data_path='path/to/AMPds2.h5',
    sequence_length=64,
    batch_size=32,
    cv_mode=False,             # 关闭交叉验证
    train_stride=1,
    val_stride=64,
    split_config={
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'embargo_hours': 24        # 仍然保持24小时间隔
    }
)

data_module.setup('fit')
# 仍然享受技术2、3、4、5、6的保护
```

## 📊 当前优化状态

### 标签平衡性现状
- **冰箱**: 50% ✅ (目标: 35%-65%)
- **洗衣机**: 13.8% ⚠️ (目标: 15%-35%，接近下限)
- **微波炉**: 3.1% ❌ (目标: 8%-20%，需要优化)
- **洗碗机**: 16.9% ⚠️ (目标: 20%-40%，接近下限)

### 技术应用状态
✅ **技术1**: Purged/Embargo Walk-Forward - 已实现  
✅ **技术2**: 先分割后预处理 - 已实现  
✅ **技术3**: 验证集非重叠窗口 - 已实现  
✅ **技术4**: 训练集小步长采样 - 已实现  
✅ **技术5**: 标签/阈值防泄漏 - 已实现  
✅ **技术6**: 特征工程分片内独立 - 已实现  

## 🔧 后续优化建议实现

### 1. 微波炉检测优化

```python
# 方案A: 更敏感的阈值策略
def optimize_microwave_detection(power_data):
    # 使用更低的百分位数
    sensitive_thresholds = [60, 65, 70, 75]  # 降低阈值
    
    # 结合功率变化率
    power_diff = np.diff(power_data)
    change_points = np.where(np.abs(power_diff) > np.std(power_diff) * 2)[0]
    
    # 在变化点附近寻找微波炉使用模式
    for threshold_percentile in sensitive_thresholds:
        threshold = np.percentile(power_data, threshold_percentile)
        positive_ratio = np.mean(power_data > threshold)
        
        if 0.08 <= positive_ratio <= 0.20:
            return threshold
    
    # 备用方案：基于变化点的动态阈值
    return np.percentile(power_data[change_points], 50)

# 方案B: 合成数据增强
def generate_microwave_synthetic_data(base_data, num_events=50):
    synthetic_data = base_data.copy()
    
    for _ in range(num_events):
        # 随机选择插入位置
        start_idx = np.random.randint(0, len(synthetic_data) - 120)
        duration = np.random.randint(30, 120)  # 30秒到2分钟
        
        # 微波炉功率模式：快速上升，平稳高功率，快速下降
        power_profile = np.concatenate([
            np.linspace(0, 800, 5),      # 快速上升
            np.full(duration-10, 800),   # 平稳高功率
            np.linspace(800, 0, 5)       # 快速下降
        ])
        
        # 添加噪声
        power_profile += np.random.normal(0, 50, len(power_profile))
        
        # 插入到数据中
        end_idx = start_idx + len(power_profile)
        if end_idx <= len(synthetic_data):
            synthetic_data[start_idx:end_idx] += power_profile
    
    return synthetic_data
```

### 2. 动态阈值调整

```python
class DynamicThresholdAdjuster:
    def __init__(self, target_ratios, adjustment_rate=0.1):
        self.target_ratios = target_ratios
        self.adjustment_rate = adjustment_rate
        self.current_thresholds = {}
    
    def adjust_thresholds(self, validation_results, appliance_data):
        """根据验证性能动态调整阈值"""
        for appliance, (min_ratio, max_ratio) in self.target_ratios.items():
            current_f1 = validation_results.get(f'{appliance}_f1', 0)
            current_ratio = validation_results.get(f'{appliance}_positive_ratio', 0)
            
            # 如果F1分数低且正样本比例不在目标范围内
            if current_f1 < 0.5 and not (min_ratio <= current_ratio <= max_ratio):
                current_threshold = self.current_thresholds.get(appliance, 0)
                
                if current_ratio < min_ratio:
                    # 正样本太少，降低阈值
                    new_threshold = current_threshold * (1 - self.adjustment_rate)
                elif current_ratio > max_ratio:
                    # 正样本太多，提高阈值
                    new_threshold = current_threshold * (1 + self.adjustment_rate)
                
                self.current_thresholds[appliance] = new_threshold
                print(f"调整{appliance}阈值: {current_threshold:.4f} -> {new_threshold:.4f}")
    
    def get_adjusted_thresholds(self):
        return self.current_thresholds.copy()
```

### 3. 多模态特征集成

```python
class MultiModalFeatureExtractor:
    def extract_features(self, power_sequence):
        features = {}
        
        # 时域特征
        features['power_mean'] = np.mean(power_sequence)
        features['power_std'] = np.std(power_sequence)
        features['power_max'] = np.max(power_sequence)
        features['power_min'] = np.min(power_sequence)
        
        # 变化率特征
        power_diff = np.diff(power_sequence)
        features['diff_mean'] = np.mean(power_diff)
        features['diff_std'] = np.std(power_diff)
        features['max_rise_rate'] = np.max(power_diff)
        features['max_fall_rate'] = np.min(power_diff)
        
        # 频域特征
        fft = np.fft.fft(power_sequence)
        power_spectrum = np.abs(fft[:len(fft)//2])
        features['dominant_freq'] = np.argmax(power_spectrum)
        features['spectral_centroid'] = np.sum(np.arange(len(power_spectrum)) * power_spectrum) / np.sum(power_spectrum)
        
        # 统计特征
        features['zero_crossing_rate'] = np.sum(np.diff(np.sign(power_sequence - np.mean(power_sequence))) != 0)
        features['energy'] = np.sum(power_sequence ** 2)
        
        return features
```

### 4. 设备特定模型

```python
class ApplianceSpecificNILM:
    def __init__(self):
        self.appliance_models = {
            'microwave': MicrowaveDetector(),      # 专门检测短时高功率
            'fridge': CyclicApplianceDetector(),   # 专门检测周期性设备
            'washer_dryer': LongRunningDetector(), # 专门检测长时间运行设备
            'dishwasher': LongRunningDetector()
        }
    
    def train_appliance_specific(self, data_module):
        for appliance, model in self.appliance_models.items():
            # 为每个设备创建专门的训练数据
            appliance_data = self.create_appliance_specific_data(data_module, appliance)
            
            # 使用设备特定的损失函数和评估指标
            model.train(appliance_data)
    
    def predict(self, power_sequence):
        predictions = {}
        for appliance, model in self.appliance_models.items():
            predictions[appliance] = model.predict(power_sequence)
        return predictions
```

## 💡 最佳实践建议

### 1. 推荐的完整工作流程

```python
# 步骤1: 创建数据模块（集成所有6个技术）
data_module = RobustNILMDataModule(
    data_path='path/to/data.h5',
    cv_mode=True,
    train_stride=1,
    val_stride=64
)

# 步骤2: 5折Walk-Forward交叉验证
for fold in range(5):
    data_module.setup_fold(fold)
    
    # 步骤3: 训练设备特定模型
    appliance_models = ApplianceSpecificNILM()
    appliance_models.train_appliance_specific(data_module)
    
    # 步骤4: 动态阈值调整
    threshold_adjuster = DynamicThresholdAdjuster(TARGET_RATIOS)
    
    # 步骤5: 多模态特征增强
    feature_extractor = MultiModalFeatureExtractor()
    
    # 训练和验证...
```

### 2. 性能监控指标

```python
# 监控所有技术的效果
monitoring_metrics = {
    'data_leakage_score': 0.0,      # 数据泄漏风险评分
    'temporal_independence': 0.95,   # 时序独立性
    'label_balance_score': 0.8,      # 标签平衡性
    'validation_stability': 0.9,     # 验证稳定性
    'generalization_gap': 0.05       # 泛化差距
}
```

### 3. 故障排除

**问题**: 微波炉检测率仍然很低  
**解决**: 使用合成数据增强 + 更敏感阈值 + 变化率特征

**问题**: 验证集性能不稳定  
**解决**: 增加embargo间隔 + 检查非重叠窗口设置

**问题**: 训练时间过长  
**解决**: 调整train_stride + 使用更小的sequence_length

## 🎯 总结

**您可以并且应该同时使用所有6个防泄漏技术！** 它们是一个完整的防护体系：

1. **技术1-2**: 防止时序和预处理泄漏
2. **技术3-4**: 优化采样策略
3. **技术5-6**: 防止特征和标签泄漏

配合4个后续优化建议，您将拥有业界最先进的NILM数据处理管道，确保模型的真实性能和部署效果！