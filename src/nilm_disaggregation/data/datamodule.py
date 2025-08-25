"""PyTorch Lightning数据模块"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any

from .dataset import RealAMPds2Dataset


class NILMDataModule(pl.LightningDataModule):
    """NILM数据模块"""
    
    def __init__(
        self,
        data_path: str,
        sequence_length: int = 512,
        batch_size: int = 32,
        num_workers: int = 4,
        train_ratio: float = 0.8,
        max_samples: int = 50000,
        **kwargs
    ):
        """
        初始化数据模块
        
        Args:
            data_path: 数据文件路径
            sequence_length: 输入序列长度
            batch_size: 批次大小
            num_workers: 数据加载器工作进程数
            train_ratio: 训练集比例
            max_samples: 最大样本数
        """
        super().__init__()
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.max_samples = max_samples
        
        # 保存超参数
        self.save_hyperparameters()
        
        # 数据集占位符
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # 元数据
        self.appliances = None
        self.main_scaler = None
        self.appliance_scalers = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        设置数据集
        
        Args:
            stage: 训练阶段 ('fit', 'validate', 'test', 'predict')
        """
        if stage == "fit" or stage is None:
            # 创建训练集
            self.train_dataset = RealAMPds2Dataset(
                data_path=self.data_path,
                sequence_length=self.sequence_length,
                train=True,
                train_ratio=self.train_ratio,
                max_samples=self.max_samples
            )
            
            # 创建验证集
            self.val_dataset = RealAMPds2Dataset(
                data_path=self.data_path,
                sequence_length=self.sequence_length,
                train=False,
                train_ratio=self.train_ratio,
                max_samples=self.max_samples
            )
            
            # 获取元数据
            self.appliances = self.train_dataset.get_appliances()
            self.main_scaler, self.appliance_scalers = self.train_dataset.get_scalers()
        
        if stage == "test" or stage is None:
            # 测试集使用验证集的设置
            if self.val_dataset is None:
                self.val_dataset = RealAMPds2Dataset(
                    data_path=self.data_path,
                    sequence_length=self.sequence_length,
                    train=False,
                    train_ratio=self.train_ratio,
                    max_samples=self.max_samples
                )
            self.test_dataset = self.val_dataset
    
    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        """验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        """测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def predict_dataloader(self) -> DataLoader:
        """预测数据加载器"""
        return self.test_dataloader()
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        获取数据集元数据
        
        Returns:
            包含设备列表和标准化器的字典
        """
        return {
            'appliances': self.appliances,
            'main_scaler': self.main_scaler,
            'appliance_scalers': self.appliance_scalers,
            'num_appliances': len(self.appliances) if self.appliances else 0,
            'sequence_length': self.sequence_length
        }