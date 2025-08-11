#!/usr/bin/env python3
"""
系统功能测试脚本
用于验证 NILM MS-CAT 系统的基本功能
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加 src 目录到路径
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """测试所有模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 数据模块
        from datamodule import AMPds2Dataset, AMPds2DataModule
        print("✅ 数据模块导入成功")
        
        # 特征模块
        from features import FeatureExtractor, ChannelMixer, PositionalEncoding
        print("✅ 特征模块导入成功")
        
        # 模型模块
        from models.mscat import MSCAT, LocalBranch, GlobalBranch
        from models.heads import MultiTaskHead, RegressionHead, EventDetectionHead
        from models.crf import CRFPostProcessor, SimpleCRF
        from models.tcn_online import CausalTCN, OnlineEventDetector
        print("✅ 模型模块导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n🏗️ 测试模型创建...")
    
    try:
        # 导入必要的模块
        from features import FeatureExtractor
        from models.mscat import MSCAT
        from models.heads import MultiTaskHead
        from models.tcn_online import CausalTCN
        
        # 测试特征提取器
        feature_extractor = FeatureExtractor(
            input_dim=11,  # 6基础 + 1差分 + 4时间特征
            d_model=192,
            max_len=240,
            use_time_features=True
        )
        print("✅ 特征提取器创建成功")
        
        # 测试 MS-CAT 模型
        mscat = MSCAT(
            input_dim=11,
            d_model=192,
            local_layers=2,
            global_layers=2,
            num_heads=6,
            window_size=32,
            dropout=0.1
        )
        print("✅ MS-CAT 模型创建成功")
        
        # 测试多任务头
        multi_head = MultiTaskHead(
            d_model=192,
            num_devices=5,
            power_loss_weight=1.0,
            event_loss_weight=0.5
        )
        print("✅ 多任务头创建成功")
        
        # 测试 TCN 模型
        tcn = CausalTCN(
            input_size=11,
            num_channels=[64, 64, 64, 64],
            kernel_size=3,
            dropout=0.1
        )
        print("✅ TCN 模型创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

def test_forward_pass():
    """测试前向传播"""
    print("\n⚡ 测试前向传播...")
    
    try:
        # 导入必要的模块
        from features import FeatureExtractor
        from models.mscat import MSCAT
        from models.heads import MultiTaskHead
        from models.tcn_online import CausalTCN
        
        # 创建测试数据
        batch_size = 4
        seq_len = 120
        input_dim = 11
        num_devices = 5
        
        # 模拟输入数据
        x = torch.randn(batch_size, seq_len, input_dim)
        timestamps = torch.randint(0, 1000000, (batch_size, seq_len)).float()
        
        # 测试特征提取
        feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            d_model=192,
            max_len=240,
            use_time_features=True
        )
        
        features = feature_extractor(x, timestamps)
        print(f"✅ 特征提取输出形状: {features.shape}")
        
        # 测试 MS-CAT (直接使用原始输入)
        mscat = MSCAT(
            input_dim=input_dim,
            d_model=192,
            local_layers=2,
            global_layers=2,
            num_heads=6,
            window_size=32,
            dropout=0.1
        )
        
        encoded = mscat(x, timestamps)  # 使用原始输入x而不是features
        print(f"✅ MS-CAT 编码输出形状: {encoded.shape}")
        
        # 测试多任务头
        multi_head = MultiTaskHead(
            d_model=192,
            num_devices=num_devices,
            power_loss_weight=1.0,
            event_loss_weight=0.5
        )
        
        multi_output = multi_head(encoded)
        power_pred = multi_output['power_pred']
        event_logits = multi_output['event_logits']
        print(f"✅ 功率预测形状: {power_pred.shape}")
        print(f"✅ 事件检测形状: {event_logits.shape}")
        
        # 测试 TCN
        tcn = CausalTCN(
            input_size=input_dim,
            num_channels=[64, 64, 64, 64],
            kernel_size=3,
            dropout=0.1
        )
        
        # TCN 需要 [batch, channels, seq_len] 格式
        x_tcn = x.transpose(1, 2)
        tcn_output = tcn(x_tcn)
        print(f"✅ TCN 输出形状: {tcn_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_sample_extraction():
    """测试单样本特征提取"""
    print("\n🔧 测试单样本特征提取...")
    
    try:
        # 导入必要的模块
        from features import FeatureExtractor
        
        feature_extractor = FeatureExtractor(
            input_dim=11,
            d_model=192,
            max_len=240,
            use_time_features=True
        )
        
        # 模拟单个样本数据
        channels = {
            'P_total': 1500.0,
            'Q_total': 300.0,
            'S_total': 1530.0,
            'I': 6.5,
            'V': 235.0,
            'PF': 0.98
        }
        
        timestamp = 1640995200.0  # 2022-01-01 00:00:00
        
        features = feature_extractor.extract_single_sample(channels, timestamp)
        print(f"✅ 单样本特征提取成功，特征维度: {features.shape}")
        print(f"   特征值: {features[:5]}...")  # 显示前5个特征
        
        return True
        
    except Exception as e:
        print(f"❌ 单样本特征提取失败: {e}")
        return False

def test_config_loading():
    """测试配置文件加载"""
    print("\n📋 测试配置文件...")
    
    try:
        import yaml
        
        config_files = [
            "configs/pretrain.yaml",
            "configs/finetune.yaml", 
            "configs/online.yaml"
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"✅ {config_file} 加载成功")
            else:
                print(f"⚠️ {config_file} 不存在")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 NILM MS-CAT 系统测试")
    print("=" * 50)
    
    # 检查 PyTorch 和 CUDA
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备数量: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.get_device_name()}")
    print()
    
    # 运行测试
    tests = [
        test_imports,
        test_model_creation,
        test_forward_pass,
        test_single_sample_extraction,
        test_config_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统准备就绪。")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查错误信息。")
        return 1

if __name__ == "__main__":
    exit(main())