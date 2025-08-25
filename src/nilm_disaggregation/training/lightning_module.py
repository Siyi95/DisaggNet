"""PyTorch Lightning模块"""

import torch
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..models import EnhancedTransformerNILM
from ..utils import CombinedLoss


class EnhancedTransformerNILMModule(pl.LightningModule):
    """增强版Transformer NILM PyTorch Lightning模块"""
    
    def __init__(self, model_params, loss_params, learning_rate=1e-4, appliances=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = EnhancedTransformerNILM(**model_params)
        self.criterion = CombinedLoss(**loss_params)
        self.learning_rate = learning_rate
        self.appliances = appliances or ['fridge', 'washer_dryer', 'microwave', 'dishwasher']
        
        # 用于存储验证结果
        self.validation_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, power_true, state_true = batch
        power_pred, state_pred = self(x)
        
        total_loss, power_loss, state_loss, corr_loss = self.criterion(
            power_pred, state_pred, power_true, state_true
        )
        
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_power_loss', power_loss)
        self.log('train_state_loss', state_loss)
        self.log('train_corr_loss', corr_loss)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, power_true, state_true = batch
        power_pred, state_pred = self(x)
        
        total_loss, power_loss, state_loss, corr_loss = self.criterion(
            power_pred, state_pred, power_true, state_true
        )
        
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_power_loss', power_loss)
        self.log('val_state_loss', state_loss)
        self.log('val_corr_loss', corr_loss)
        
        # 存储预测结果用于计算指标
        self.validation_outputs.append({
            'power_pred': power_pred.detach().cpu(),
            'power_true': power_true.detach().cpu(),
            'state_pred': state_pred.detach().cpu(),
            'state_true': state_true.detach().cpu()
        })
        
        return total_loss
    
    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        
        # 合并所有验证输出
        power_preds = torch.cat([x['power_pred'] for x in self.validation_outputs])
        power_trues = torch.cat([x['power_true'] for x in self.validation_outputs])
        state_preds = torch.cat([x['state_pred'] for x in self.validation_outputs])
        state_trues = torch.cat([x['state_true'] for x in self.validation_outputs])
        
        # 检查并处理NaN值
        power_preds = torch.nan_to_num(power_preds, nan=0.0, posinf=1e6, neginf=-1e6)
        power_trues = torch.nan_to_num(power_trues, nan=0.0, posinf=1e6, neginf=-1e6)
        state_preds = torch.nan_to_num(state_preds, nan=0.0, posinf=1.0, neginf=0.0)
        state_trues = torch.nan_to_num(state_trues, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 计算每个设备的指标
        for i, appliance in enumerate(self.appliances):
            pred_power = power_preds[:, i].numpy()
            true_power = power_trues[:, i].numpy()
            
            # 再次检查NaN值
            if np.any(np.isnan(pred_power)) or np.any(np.isnan(true_power)):
                print(f"警告: {appliance} 的预测或真实值包含NaN，跳过计算")
                continue
            
            try:
                # 计算指标
                mae = mean_absolute_error(true_power, pred_power)
                rmse = np.sqrt(mean_squared_error(true_power, pred_power))
                r2 = r2_score(true_power, pred_power)
                
                # 计算相关系数
                if np.std(pred_power) > 1e-8 and np.std(true_power) > 1e-8:
                    correlation = np.corrcoef(pred_power, true_power)[0, 1]
                    if np.isnan(correlation):
                        correlation = 0.0
                else:
                    correlation = 0.0
                
                # 确保指标值是有限的
                mae = np.clip(mae, 0, 1e6)
                rmse = np.clip(rmse, 0, 1e6)
                r2 = np.clip(r2, -1, 1)
                correlation = np.clip(correlation, -1, 1)
                
                self.log(f'val_{appliance}_mae', mae)
                self.log(f'val_{appliance}_rmse', rmse)
                self.log(f'val_{appliance}_r2', r2)
                self.log(f'val_{appliance}_corr', correlation)
                
            except Exception as e:
                print(f"计算 {appliance} 指标时出错: {e}")
                continue
        
        # 计算平均指标（安全版本）
        try:
            valid_maes = []
            valid_rmses = []
            valid_r2s = []
            
            for i in range(len(self.appliances)):
                pred_i = power_preds[:, i].numpy()
                true_i = power_trues[:, i].numpy()
                
                if not (np.any(np.isnan(pred_i)) or np.any(np.isnan(true_i))):
                    mae_i = mean_absolute_error(true_i, pred_i)
                    rmse_i = np.sqrt(mean_squared_error(true_i, pred_i))
                    r2_i = r2_score(true_i, pred_i)
                    
                    if np.isfinite(mae_i) and np.isfinite(rmse_i) and np.isfinite(r2_i):
                        valid_maes.append(mae_i)
                        valid_rmses.append(rmse_i)
                        valid_r2s.append(r2_i)
            
            if valid_maes:
                avg_mae = np.mean(valid_maes)
                avg_rmse = np.mean(valid_rmses)
                avg_r2 = np.mean(valid_r2s)
                
                self.log('val_avg_mae', avg_mae)
                self.log('val_avg_rmse', avg_rmse)
                self.log('val_avg_r2', avg_r2)
            
        except Exception as e:
            print(f"计算平均指标时出错: {e}")
        
        # 清空验证输出
        self.validation_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        x, power_true, state_true = batch
        power_pred, state_pred = self(x)
        
        total_loss, power_loss, state_loss, corr_loss = self.criterion(
            power_pred, state_pred, power_true, state_true
        )
        
        self.log('test_loss', total_loss)
        self.log('test_power_loss', power_loss)
        self.log('test_state_loss', state_loss)
        self.log('test_corr_loss', corr_loss)
        
        return {
            'test_loss': total_loss,
            'power_pred': power_pred.detach().cpu(),
            'power_true': power_true.detach().cpu(),
            'state_pred': state_pred.detach().cpu(),
            'state_true': state_true.detach().cpu()
        }
    
    def predict_step(self, batch, batch_idx):
        """预测步骤"""
        x, power_true, state_true = batch
        power_pred, state_pred = self(x)
        
        return {
            'power_pred': power_pred.detach().cpu(),
            'state_pred': state_pred.detach().cpu(),
            'power_true': power_true.detach().cpu(),
            'state_true': state_true.detach().cpu()
        }
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }