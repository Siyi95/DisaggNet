"""配置管理模块"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Union


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: Union[str, Path] = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径
        """
        self.config = {}
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: Union[str, Path]):
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    
    def save_config(self, config_path: Union[str, Path]):
        """
        保存配置文件
        
        Args:
            config_path: 配置文件路径
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点分隔的嵌套键（如 'model.d_model'）
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键，支持点分隔的嵌套键（如 'model.d_model'）
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, other_config: Union[Dict, 'Config']):
        """
        更新配置
        
        Args:
            other_config: 其他配置字典或Config对象
        """
        if isinstance(other_config, Config):
            other_config = other_config.config
        
        self._deep_update(self.config, other_config)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """
        深度更新字典
        
        Args:
            base_dict: 基础字典
            update_dict: 更新字典
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        return self.config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """支持字典式设置"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """支持 in 操作符"""
        return self.get(key) is not None
    
    def __repr__(self) -> str:
        return f"Config({self.config})"


def load_config(config_path: Union[str, Path]) -> Config:
    """
    加载配置文件的便捷函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Config对象
    """
    return Config(config_path)


def get_default_config() -> Config:
    """
    获取默认配置
    
    Returns:
        默认Config对象
    """
    # 查找默认配置文件
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent.parent
    default_config_path = project_root / "configs" / "default_config.yaml"
    
    if default_config_path.exists():
        return Config(default_config_path)
    else:
        # 如果找不到默认配置文件，返回空配置
        return Config()


def merge_configs(*configs: Union[Config, Dict, str, Path]) -> Config:
    """
    合并多个配置
    
    Args:
        *configs: 配置对象、字典或配置文件路径
        
    Returns:
        合并后的Config对象
    """
    merged_config = Config()
    
    for config in configs:
        if isinstance(config, (str, Path)):
            config = Config(config)
        elif isinstance(config, dict):
            temp_config = Config()
            temp_config.config = config
            config = temp_config
        
        merged_config.update(config)
    
    return merged_config