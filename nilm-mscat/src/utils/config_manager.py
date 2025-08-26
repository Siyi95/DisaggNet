"""统一配置管理系统"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """数据配置"""
    dataset_name: str = "AMPds2"
    data_path: str = "data/AMPds2"
    sequence_length: int = 512
    stride: int = 256
    target_devices: list = field(default_factory=lambda: ["B1E", "BME", "CDE", "DWE", "EBE"])
    sampling_rate: str = "1min"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    normalize: bool = True
    feature_engineering: dict = field(default_factory=dict)

@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "UnifiedNILMLightningModule"
    architecture: dict = field(default_factory=dict)
    pretrain: dict = field(default_factory=dict)

@dataclass
class LossConfig:
    """损失函数配置"""
    power_loss_weight: float = 1.0
    event_loss_weight: float = 0.5
    energy_conservation_weight: float = 0.3
    temporal_consistency_weight: float = 0.2
    power_loss: dict = field(default_factory=dict)
    event_loss: dict = field(default_factory=dict)
    energy_conservation: dict = field(default_factory=dict)
    temporal_consistency: dict = field(default_factory=dict)

@dataclass
class TrainingConfig:
    """训练配置"""
    pretrain: dict = field(default_factory=dict)
    finetune: dict = field(default_factory=dict)
    optimizer: dict = field(default_factory=dict)
    scheduler: dict = field(default_factory=dict)

@dataclass
class CRFConfig:
    """CRF后处理配置"""
    enable: bool = True
    power_threshold: float = 10.0
    min_on_duration: int = 5
    min_off_duration: int = 3
    transition_cost: float = 1.0
    use_pycrfsuite: bool = False
    temperature: float = 1.0
    adaptive_threshold: bool = True
    confidence_weight: float = 0.3
    temporal_consistency_weight: float = 0.2

@dataclass
class EvaluationConfig:
    """评估配置"""
    metrics: list = field(default_factory=lambda: ["mae", "mse", "rmse", "f1"])
    threshold: float = 10.0
    save_predictions: bool = True
    save_visualizations: bool = True

@dataclass
class VisualizationConfig:
    """可视化配置"""
    tensorboard: dict = field(default_factory=dict)
    plots: dict = field(default_factory=dict)
    monitoring: dict = field(default_factory=dict)

@dataclass
class CheckpointConfig:
    """检查点配置"""
    save_dir: str = "outputs/checkpoints"
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 3
    save_last: bool = True
    filename: str = "{epoch:02d}-{val_loss:.4f}"

@dataclass
class HardwareConfig:
    """硬件配置"""
    accelerator: str = "auto"
    devices: str = "auto"
    precision: int = 16
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str = "unified_nilm_training"
    version: str = "v1.0"
    tags: list = field(default_factory=lambda: ["mscat", "lightning", "nilm"])
    notes: str = ""

@dataclass
class UnifiedConfig:
    """统一配置类"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    crf: CRFConfig = field(default_factory=CRFConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    seed: int = 42
    debug: dict = field(default_factory=dict)
    logging: dict = field(default_factory=dict)

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[DictConfig] = None
        self._unified_config: Optional[UnifiedConfig] = None
        
    def load_config(self, 
                   config_path: Optional[Union[str, Path]] = None,
                   overrides: Optional[Dict[str, Any]] = None) -> DictConfig:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            overrides: 配置覆盖字典
        Returns:
            加载的配置
        """
        if config_path:
            self.config_path = Path(config_path)
        
        if not self.config_path or not self.config_path.exists():
            logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
            self.config = OmegaConf.structured(UnifiedConfig())
        else:
            try:
                # 使用OmegaConf加载YAML配置
                self.config = OmegaConf.load(self.config_path)
                logger.info(f"成功加载配置文件: {self.config_path}")
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
                self.config = OmegaConf.structured(UnifiedConfig())
        
        # 应用覆盖配置
        if overrides:
            override_config = OmegaConf.create(overrides)
            self.config = OmegaConf.merge(self.config, override_config)
            logger.info(f"应用配置覆盖: {list(overrides.keys())}")
        
        # 验证配置
        self._validate_config()
        
        return self.config
    
    def get_config(self) -> DictConfig:
        """
        获取当前配置
        
        Returns:
            当前配置
        """
        if self.config is None:
            raise ValueError("配置未加载，请先调用load_config()")
        return self.config
    
    def get_unified_config(self) -> UnifiedConfig:
        """
        获取结构化的统一配置
        
        Returns:
            统一配置对象
        """
        if self._unified_config is None:
            if self.config is None:
                raise ValueError("配置未加载，请先调用load_config()")
            
            # 转换为结构化配置
            self._unified_config = OmegaConf.to_object(self.config)
        
        return self._unified_config
    
    def save_config(self, save_path: Union[str, Path], config: Optional[DictConfig] = None):
        """
        保存配置到文件
        
        Args:
            save_path: 保存路径
            config: 要保存的配置（默认使用当前配置）
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_to_save = config or self.config
        if config_to_save is None:
            raise ValueError("没有可保存的配置")
        
        try:
            OmegaConf.save(config_to_save, save_path)
            logger.info(f"配置已保存到: {save_path}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise
    
    def update_config(self, updates: Dict[str, Any]):
        """
        更新配置
        
        Args:
            updates: 更新字典
        """
        if self.config is None:
            raise ValueError("配置未加载，请先调用load_config()")
        
        update_config = OmegaConf.create(updates)
        self.config = OmegaConf.merge(self.config, update_config)
        
        # 重新验证配置
        self._validate_config()
        
        # 清除缓存的统一配置
        self._unified_config = None
        
        logger.info(f"配置已更新: {list(updates.keys())}")
    
    def get_stage_config(self, stage: str) -> Dict[str, Any]:
        """
        获取特定训练阶段的配置
        
        Args:
            stage: 训练阶段 ('pretrain', 'finetune')
        Returns:
            阶段配置字典
        """
        if self.config is None:
            raise ValueError("配置未加载，请先调用load_config()")
        
        if stage not in ['pretrain', 'finetune']:
            raise ValueError(f"不支持的训练阶段: {stage}")
        
        # 获取基础配置
        base_config = OmegaConf.to_container(self.config, resolve=True)
        
        # 获取阶段特定配置
        stage_config = base_config['training'].get(stage, {})
        
        # 合并配置
        merged_config = {
            'data': base_config['data'],
            'model': base_config['model'],
            'loss': base_config['loss'],
            'training': stage_config,
            'hardware': base_config['hardware'],
            'checkpoint': base_config['checkpoint'],
            'visualization': base_config['visualization'],
            'experiment': base_config['experiment'],
            'seed': base_config['seed']
        }
        
        # 添加阶段特定的模型配置
        if stage == 'pretrain' and 'pretrain' in base_config['model']:
            merged_config['model']['pretrain'] = base_config['model']['pretrain']
        
        return merged_config
    
    def _validate_config(self):
        """
        验证配置的有效性
        """
        if self.config is None:
            return
        
        try:
            # 验证数据配置
            if 'data' in self.config:
                data_config = self.config.data
                if data_config.train_ratio + data_config.val_ratio + data_config.test_ratio != 1.0:
                    logger.warning("数据分割比例之和不等于1.0")
                
                if data_config.sequence_length <= 0:
                    raise ValueError("sequence_length必须大于0")
                
                if data_config.stride <= 0:
                    raise ValueError("stride必须大于0")
            
            # 验证训练配置
            if 'training' in self.config:
                training_config = self.config.training
                for stage in ['pretrain', 'finetune']:
                    if stage in training_config:
                        stage_config = training_config[stage]
                        if 'learning_rate' in stage_config and stage_config.learning_rate <= 0:
                            raise ValueError(f"{stage}阶段的learning_rate必须大于0")
                        
                        if 'batch_size' in stage_config and stage_config.batch_size <= 0:
                            raise ValueError(f"{stage}阶段的batch_size必须大于0")
            
            # 验证损失配置
            if 'loss' in self.config:
                loss_config = self.config.loss
                weights = [
                    loss_config.get('power_loss_weight', 0),
                    loss_config.get('event_loss_weight', 0),
                    loss_config.get('energy_conservation_weight', 0),
                    loss_config.get('temporal_consistency_weight', 0)
                ]
                if all(w <= 0 for w in weights):
                    raise ValueError("至少需要一个损失权重大于0")
            
            logger.info("配置验证通过")
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            raise
    
    @staticmethod
    def create_default_config() -> DictConfig:
        """
        创建默认配置
        
        Returns:
            默认配置
        """
        return OmegaConf.structured(UnifiedConfig())
    
    @staticmethod
    def merge_configs(*configs: DictConfig) -> DictConfig:
        """
        合并多个配置
        
        Args:
            *configs: 要合并的配置列表
        Returns:
            合并后的配置
        """
        if not configs:
            return OmegaConf.create({})
        
        merged = configs[0]
        for config in configs[1:]:
            merged = OmegaConf.merge(merged, config)
        
        return merged
    
    def print_config(self, resolve: bool = True):
        """
        打印当前配置
        
        Args:
            resolve: 是否解析配置中的变量
        """
        if self.config is None:
            print("配置未加载")
            return
        
        print("=" * 50)
        print("当前配置:")
        print("=" * 50)
        print(OmegaConf.to_yaml(self.config, resolve=resolve))
        print("=" * 50)

# 全局配置管理器实例
config_manager = ConfigManager()

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    return config_manager

def load_config(config_path: Union[str, Path], 
               overrides: Optional[Dict[str, Any]] = None) -> DictConfig:
    """便捷函数：加载配置"""
    return config_manager.load_config(config_path, overrides)

def get_config() -> DictConfig:
    """便捷函数：获取当前配置"""
    return config_manager.get_config()

def get_stage_config(stage: str) -> Dict[str, Any]:
    """便捷函数：获取阶段配置"""
    return config_manager.get_stage_config(stage)