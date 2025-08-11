#!/usr/bin/env python3
"""
监督微调脚本
基于预训练的MS-CAT模型进行负荷分解的监督学习
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datamodule import AMPds2DataModule
from src.models.mscat import MSCAT
from src.models.heads import MultiTaskHead
from src.models.crf import CRFPostProcessor
from src.train_pretrain import MaskedReconstructionModel

class NILMModel(pl.LightningModule):
    """NILM负荷分解模型"""
    
    def __init__(self,
                 input_dim: int,
                 num_devices: int,
                 device_names: List[str],
                 d_model: int = 192,
                 num_heads: int = 6,
                 local_layers: int = 4,
                 global_layers: int = 3,
                 window_size: int = 32,
                 sparsity_factor: int = 4,
                 conv_kernel_size: int = 7,
                 fusion_type: str = 'weighted_sum',
                 use_time_features: bool = True,
                 learnable_pos: bool = False,
                 dropout: float = 0.1,
                 max_len: int = 5000,
                 # 损失权重
                 regression_weight: float = 1.0,
                 classification_weight: float = 0.5,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 # 优化器参数
                 learning_rate: float = 1e-3,
                 min_learning_rate: float = 1e-5,
                 weight_decay: float = 1e-4,
                 warmup_epochs: int = 5,
                 max_epochs: int = 100,
                 # CRF后处理
                 use_crf: bool = True,
                 min_on_duration: int = 5,
                 min_off_duration: int = 3,
                 power_threshold: float = 10.0,
                 # 冻结策略
                 freeze_encoder_epochs: int = 0,
                 freeze_ratio: float = 0.5,
                 **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        # MS-CAT编码器
        self.encoder = MSCAT(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            local_layers=local_layers,
            global_layers=global_layers,
            window_size=window_size,
            sparsity_factor=sparsity_factor,
            conv_kernel_size=conv_kernel_size,
            fusion_type=fusion_type,
            use_time_features=use_time_features,
            learnable_pos=learnable_pos,
            dropout=dropout,
            max_len=max_len
        )
        
        # 多任务头
        self.head = MultiTaskHead(
            d_model=d_model,
            num_devices=num_devices,
            power_loss_weight=regression_weight,
            event_loss_weight=classification_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            dropout=dropout
        )
        
        # CRF后处理器
        if use_crf:
            self.crf_processor = CRFPostProcessor(
                num_devices=num_devices,
                min_on_duration=min_on_duration,
                min_off_duration=min_off_duration,
                power_threshold=power_threshold
            )
        else:
            self.crf_processor = None
        
        self.device_names = device_names
        self.num_devices = num_devices
        
        # 用于记录训练指标
        self.train_metrics = []
        self.val_metrics = []
        
        # 冻结参数标志
        self._frozen = False
        
    def load_pretrained_weights(self, pretrain_ckpt_path: str, strict: bool = False):
        """
        加载预训练权重
        
        Args:
            pretrain_ckpt_path: 预训练检查点路径
            strict: 是否严格匹配参数名
        """
        print(f"加载预训练权重: {pretrain_ckpt_path}")
        
        # 加载预训练模型
        pretrain_model = MaskedReconstructionModel.load_from_checkpoint(
            pretrain_ckpt_path, map_location='cpu'
        )
        
        # 提取编码器权重
        encoder_state_dict = {}
        for name, param in pretrain_model.encoder.state_dict().items():
            encoder_state_dict[name] = param
        
        # 加载到当前编码器
        missing_keys, unexpected_keys = self.encoder.load_state_dict(
            encoder_state_dict, strict=strict
        )
        
        if missing_keys:
            print(f"缺失的键: {missing_keys}")
        if unexpected_keys:
            print(f"意外的键: {unexpected_keys}")
        
        print("预训练权重加载完成")
    
    def freeze_encoder(self, freeze_ratio: float = 0.5):
        """
        冻结编码器的部分参数
        
        Args:
            freeze_ratio: 冻结比例（从前往后）
        """
        if freeze_ratio <= 0:
            return
        
        # 获取所有编码器参数
        encoder_params = list(self.encoder.parameters())
        num_freeze = int(len(encoder_params) * freeze_ratio)
        
        # 冻结前面的参数
        for i, param in enumerate(encoder_params):
            if i < num_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        self._frozen = True
        print(f"冻结了编码器的前 {freeze_ratio*100:.1f}% 参数")
    
    def unfreeze_encoder(self):
        """解冻编码器所有参数"""
        for param in self.encoder.parameters():
            param.requires_grad = True
        
        self._frozen = False
        print("解冻编码器所有参数")
    
    def forward(self, 
               x: torch.Tensor, 
               timestamps: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_len, input_dim] 输入特征
            timestamps: [batch_size, seq_len] 时间戳（可选）
        Returns:
            预测结果字典
        """
        # 编码
        encoded = self.encoder(x, timestamps)
        
        # 多任务预测
        predictions = self.head(encoded)
        
        return predictions
    
    def compute_metrics(self, 
                       predictions: Dict[str, torch.Tensor],
                       targets: Dict[str, torch.Tensor],
                       prefix: str = '') -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            predictions: 模型预测结果
            targets: 真实标签
            prefix: 指标前缀
        Returns:
            指标字典
        """
        metrics = {}
        
        # 回归指标
        if 'power' in predictions and 'power' in targets:
            pred_power = predictions['power'].detach().cpu().numpy()
            true_power = targets['power'].detach().cpu().numpy()
            
            # 总体指标
            mae_total = mean_absolute_error(true_power.flatten(), pred_power.flatten())
            rmse_total = np.sqrt(mean_squared_error(true_power.flatten(), pred_power.flatten()))
            
            metrics[f'{prefix}MAE_total'] = mae_total
            metrics[f'{prefix}RMSE_total'] = rmse_total
            
            # 每设备指标
            for i, device_name in enumerate(self.device_names):
                if i < pred_power.shape[-1]:
                    mae_device = mean_absolute_error(true_power[:, :, i].flatten(), 
                                                   pred_power[:, :, i].flatten())
                    metrics[f'{prefix}MAE_{device_name}'] = mae_device
            
            # SAE (Signal Aggregate Error)
            pred_total = pred_power.sum(axis=-1)  # [batch, seq_len]
            true_total = true_power.sum(axis=-1)
            sae = np.abs(pred_total - true_total).mean()
            metrics[f'{prefix}SAE'] = sae
        
        # 分类指标
        if 'state_prob' in predictions and 'state' in targets:
            pred_state_prob = predictions['state_prob'].detach().cpu().numpy()
            true_state = targets['state'].detach().cpu().numpy()
            
            # 阈值化预测
            pred_state = (pred_state_prob > 0.5).astype(int)
            
            # 总体指标
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_state.flatten(), pred_state.flatten(), average='macro', zero_division=0
            )
            
            metrics[f'{prefix}Precision'] = precision
            metrics[f'{prefix}Recall'] = recall
            metrics[f'{prefix}F1'] = f1
            
            # 每设备指标
            for i, device_name in enumerate(self.device_names):
                if i < pred_state.shape[-1]:
                    device_precision, device_recall, device_f1, _ = precision_recall_fscore_support(
                        true_state[:, :, i].flatten(), 
                        pred_state[:, :, i].flatten(), 
                        average='binary', zero_division=0
                    )
                    metrics[f'{prefix}F1_{device_name}'] = device_f1
        
        return metrics
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """训练步骤"""
        x = batch['x'].float()  # [batch_size, seq_len, input_dim] - 确保float32类型
        y_power = batch['y_power'].float()  # [batch_size, seq_len, num_devices]
        y_state = batch['y_state'].float()  # [batch_size, seq_len, num_devices]
        
        # 前向传播
        predictions = self.forward(x)
        
        # 计算损失
        targets = {'power': y_power, 'state': y_state}
        losses = self.head.compute_loss(predictions, targets)
        
        # 记录损失
        self.log('train/total_loss', losses['total_loss'], prog_bar=True)
        self.log('train/power_loss', losses['power_loss'])
        self.log('train/event_loss', losses['event_loss'])
        
        # 计算指标
        metrics = self.compute_metrics(predictions, targets, 'train/')
        for name, value in metrics.items():
            self.log(name, value)
        
        self.train_metrics.append({
            'loss': losses['total_loss'].item(),
            **{k: v for k, v in metrics.items() if 'train/' in k}
        })
        
        return losses['total_loss']
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """验证步骤"""
        x = batch['x'].float()
        y_power = batch['y_power'].float()
        y_state = batch['y_state'].float()
        
        # 前向传播
        predictions = self.forward(x)
        
        # CRF后处理（如果启用）
        if self.crf_processor is not None:
            # 转换为numpy数组进行CRF处理
            power_pred_np = predictions['power_pred'].detach().cpu().numpy()
            event_logits_np = predictions['event_logits'].detach().cpu().numpy()
            
            # 处理每个批次样本
            processed_results = []
            for i in range(power_pred_np.shape[0]):
                result = self.crf_processor.process_predictions(
                    power_pred_np[i], 
                    event_logits=event_logits_np[i]
                )
                processed_results.append(result)
            
            # 重新组装结果
            processed_power = torch.tensor(
                np.stack([r['power_predictions'] for r in processed_results]), 
                device=predictions['power_pred'].device
            )
            processed_states = torch.tensor(
                np.stack([r['state_predictions'] for r in processed_results]), 
                device=predictions['event_logits'].device
            )
            
            predictions['power_pred'] = processed_power
            predictions['event_probs'] = torch.sigmoid(torch.tensor(
                np.stack([r['state_predictions'] for r in processed_results]), 
                device=predictions['event_probs'].device
            ))
        
        # 计算损失
        targets = {'power': y_power, 'state': y_state}
        losses = self.head.compute_loss(predictions, targets)
        
        # 记录损失
        self.log('val/total_loss', losses['total_loss'], prog_bar=True)
        self.log('val/power_loss', losses['power_loss'])
        self.log('val/event_loss', losses['event_loss'])
        
        # 计算指标
        metrics = self.compute_metrics(predictions, targets, 'val/')
        for name, value in metrics.items():
            self.log(name, value)
        
        self.val_metrics.append({
            'loss': losses['total_loss'].item(),
            **{k: v for k, v in metrics.items() if 'val/' in k}
        })
        
        return losses['total_loss']
    
    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 分组参数：编码器 vs 头部
        encoder_params = list(self.encoder.parameters())
        head_params = list(self.head.parameters())
        
        # 如果编码器被冻结，只优化头部
        if self._frozen:
            trainable_encoder_params = [p for p in encoder_params if p.requires_grad]
            params = [
                {'params': trainable_encoder_params, 'lr': self.hparams.learning_rate * 0.1},
                {'params': head_params, 'lr': self.hparams.learning_rate}
            ]
        else:
            params = [
                {'params': encoder_params, 'lr': self.hparams.learning_rate * 0.1},
                {'params': head_params, 'lr': self.hparams.learning_rate}
            ]
        
        optimizer = torch.optim.AdamW(
            params,
            weight_decay=self.hparams.weight_decay
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.min_learning_rate
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def on_train_epoch_end(self):
        """训练轮次结束时的回调"""
        # 检查是否需要解冻编码器
        if (self._frozen and 
            self.current_epoch >= self.hparams.freeze_encoder_epochs):
            self.unfreeze_encoder()
        
        # 清理指标
        if self.train_metrics:
            self.train_metrics.clear()
    
    def on_validation_epoch_end(self):
        """验证轮次结束时的回调"""
        if self.val_metrics:
            self.val_metrics.clear()
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """预测步骤"""
        x = batch['x']
        
        # 前向传播
        predictions = self.forward(x)
        
        # CRF后处理
        if self.crf_processor is not None:
            predictions = self.crf_processor.process_batch(
                predictions, apply_duration_filter=True
            )
        
        return predictions

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='NILM监督微调')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--ckpt', type=str, help='预训练检查点路径')
    parser.add_argument('--data_path', type=str, help='数据文件路径')
    parser.add_argument('--output_dir', type=str, default='./outputs/finetune', help='输出目录')
    parser.add_argument('--resume_from', type=str, help='从检查点恢复训练')
    parser.add_argument('--gpus', type=int, default=1, help='GPU数量')
    parser.add_argument('--precision', type=str, default='16-mixed', help='训练精度')
    parser.add_argument('--test_only', action='store_true', help='仅进行测试')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖配置中的参数
    if args.data_path:
        config['data']['data_path'] = args.data_path
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    pl.seed_everything(config.get('seed', 42))
    
    # 数据模块
    data_module = AMPds2DataModule(**config['data'])
    data_module.setup('fit')
    
    # 获取特征和设备信息
    input_dim = data_module.get_feature_dim()
    num_devices = data_module.get_num_devices()
    device_names = data_module.get_device_names()
    
    print(f"输入特征维度: {input_dim}")
    print(f"设备数量: {num_devices}")
    print(f"设备名称: {device_names}")
    
    # 模型
    model_config = config['model']
    model_config.update({
        'input_dim': input_dim,
        'num_devices': num_devices,
        'device_names': device_names
    })
    model = NILMModel(**model_config)
    
    # 加载预训练权重
    if args.ckpt:
        model.load_pretrained_weights(args.ckpt, strict=False)
        
        # 应用冻结策略
        if model.hparams.freeze_encoder_epochs > 0:
            model.freeze_encoder(model.hparams.freeze_ratio)
    
    # 如果只是测试
    if args.test_only:
        if not args.resume_from:
            raise ValueError("测试模式需要提供 --resume_from 参数")
        
        # 加载训练好的模型
        model = NILMModel.load_from_checkpoint(args.resume_from)
        
        # 创建训练器
        trainer = pl.Trainer(
            accelerator='gpu' if args.gpus > 0 else 'cpu',
            devices=args.gpus if args.gpus > 0 else 'auto',
            precision=args.precision
        )
        
        # 测试
        data_module.setup('test')
        test_results = trainer.test(model, data_module)
        print("测试结果:", test_results)
        return
    
    # 日志记录器
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='finetune_logs',
        version=None
    )
    
    # 回调函数
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, 'checkpoints'),
            filename='nilm_best_{epoch:02d}_{val/MAE_total:.4f}',
            monitor='val/MAE_total',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        EarlyStopping(
            monitor='val/MAE_total',
            patience=config.get('early_stopping_patience', 15),
            mode='min'
        ),
        LearningRateMonitor(logging_interval='epoch')
    ]
    
    # 训练器
    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', 100),
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 'auto',
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config.get('accumulate_grad_batches', 1),
        val_check_interval=config.get('val_check_interval', 1.0),
        log_every_n_steps=config.get('log_every_n_steps', 50)
    )
    
    # 开始训练
    if args.resume_from:
        print(f"从检查点恢复训练: {args.resume_from}")
        trainer.fit(model, data_module, ckpt_path=args.resume_from)
    else:
        print("开始微调训练...")
        trainer.fit(model, data_module)
    
    # 测试最佳模型
    print("开始测试...")
    data_module.setup('test')
    test_results = trainer.test(model, data_module, ckpt_path='best')
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'nilm_final.ckpt')
    trainer.save_checkpoint(final_model_path)
    print(f"微调完成，模型保存至: {final_model_path}")
    print("测试结果:", test_results)

if __name__ == '__main__':
    main()