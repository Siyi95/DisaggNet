#!/usr/bin/env python3
"""
训练启动脚本
提供简化的训练启动接口，支持预训练和微调
"""

import os
import sys
import argparse
import yaml
import subprocess
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_enhanced import detect_device

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def update_config_for_device(config: Dict[str, Any], device_type: str) -> Dict[str, Any]:
    """
    根据设备类型更新配置
    """
    if device_type == 'cuda':
        config['trainer']['accelerator'] = 'gpu'
        config['trainer']['devices'] = 'auto'
        config['trainer']['precision'] = '16-mixed'  # 使用混合精度加速
    elif device_type == 'mps':
        config['trainer']['accelerator'] = 'mps'
        config['trainer']['devices'] = 1
        config['trainer']['precision'] = '32'  # MPS暂不支持16位精度
    else:
        config['trainer']['accelerator'] = 'cpu'
        config['trainer']['devices'] = 'auto'
        config['trainer']['precision'] = '32'
    
    return config

def run_pretraining(config_path: str, data_path: str, output_dir: str, 
                   resume_from: str = None, **kwargs):
    """
    运行预训练
    """
    print("🚀 开始预训练...")
    
    # 检测设备
    device_type, device_info = detect_device()
    print(f"🔧 检测到设备: {device_type} - {device_info}")
    
    # 构建命令
    cmd = [
        sys.executable, 'train_enhanced.py',
        '--config', config_path,
        '--data_path', data_path,
        '--output_dir', output_dir,
        '--mode', 'pretrain'
    ]
    
    if resume_from:
        cmd.extend(['--resume_from', resume_from])
    
    # 添加其他参数
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f'--{key}', str(value)])
    
    # 运行命令
    try:
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
        print("✅ 预训练完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 预训练失败: {e}")
        return False

def run_finetuning(config_path: str, data_path: str, output_dir: str,
                  pretrained_model: str = None, resume_from: str = None, **kwargs):
    """
    运行微调
    """
    print("🎯 开始微调...")
    
    # 检测设备
    device_type, device_info = detect_device()
    print(f"🔧 检测到设备: {device_type} - {device_info}")
    
    # 构建命令
    cmd = [
        sys.executable, 'train_enhanced.py',
        '--config', config_path,
        '--data_path', data_path,
        '--output_dir', output_dir,
        '--mode', 'finetune'
    ]
    
    if pretrained_model:
        cmd.extend(['--pretrained_model', pretrained_model])
    
    if resume_from:
        cmd.extend(['--resume_from', resume_from])
    
    # 添加其他参数
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f'--{key}', str(value)])
    
    # 运行命令
    try:
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
        print("✅ 微调完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 微调失败: {e}")
        return False

def run_analysis(model_path: str, config_path: str, data_path: str, output_dir: str):
    """
    运行结果分析
    """
    print("📊 开始结果分析...")
    
    cmd = [
        sys.executable, 'analyze_results.py',
        '--model_path', model_path,
        '--config_path', config_path,
        '--output_dir', output_dir
    ]
    
    if data_path:
        cmd.extend(['--data_path', data_path])
    
    try:
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
        print("✅ 结果分析完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 结果分析失败: {e}")
        return False

def print_usage_examples():
    """
    打印使用示例
    """
    print("""
🔥 NILM-MSCAT 训练系统使用指南

📋 基本用法:

1. 预训练:
   python start_training.py pretrain --config configs/pretrain.yaml --data_path /path/to/data

2. 微调:
   python start_training.py finetune --config configs/finetune.yaml --data_path /path/to/data --pretrained_model /path/to/pretrained.ckpt

3. 结果分析:
   python start_training.py analyze --model_path /path/to/model.ckpt --config configs/finetune.yaml --data_path /path/to/data

4. 完整流程:
   python start_training.py full --config configs/finetune.yaml --data_path /path/to/data

🔧 高级选项:
   --output_dir: 输出目录 (默认: ./outputs)
   --resume_from: 从检查点恢复训练
   --epochs: 训练轮数
   --batch_size: 批次大小
   --learning_rate: 学习率

💡 提示:
   - 系统会自动检测并使用最佳设备 (CUDA/MPS/CPU)
   - 训练过程会自动保存到 TensorBoard
   - 结果分析包含可视化图表和交互式仪表板
""")

def main():
    parser = argparse.ArgumentParser(description='NILM-MSCAT 训练系统')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 预训练命令
    pretrain_parser = subparsers.add_parser('pretrain', help='运行预训练')
    pretrain_parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    pretrain_parser.add_argument('--data_path', type=str, required=True, help='数据路径')
    pretrain_parser.add_argument('--output_dir', type=str, default='./outputs/pretrain', help='输出目录')
    pretrain_parser.add_argument('--resume_from', type=str, help='恢复训练的检查点')
    pretrain_parser.add_argument('--epochs', type=int, help='训练轮数')
    pretrain_parser.add_argument('--batch_size', type=int, help='批次大小')
    pretrain_parser.add_argument('--learning_rate', type=float, help='学习率')
    
    # 微调命令
    finetune_parser = subparsers.add_parser('finetune', help='运行微调')
    finetune_parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    finetune_parser.add_argument('--data_path', type=str, required=True, help='数据路径')
    finetune_parser.add_argument('--output_dir', type=str, default='./outputs/finetune', help='输出目录')
    finetune_parser.add_argument('--pretrained_model', type=str, help='预训练模型路径')
    finetune_parser.add_argument('--resume_from', type=str, help='恢复训练的检查点')
    finetune_parser.add_argument('--epochs', type=int, help='训练轮数')
    finetune_parser.add_argument('--batch_size', type=int, help='批次大小')
    finetune_parser.add_argument('--learning_rate', type=float, help='学习率')
    
    # 分析命令
    analyze_parser = subparsers.add_parser('analyze', help='分析训练结果')
    analyze_parser.add_argument('--model_path', type=str, required=True, help='模型检查点路径')
    analyze_parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    analyze_parser.add_argument('--data_path', type=str, help='数据路径')
    analyze_parser.add_argument('--output_dir', type=str, default='./analysis_results', help='分析结果输出目录')
    
    # 完整流程命令
    full_parser = subparsers.add_parser('full', help='运行完整训练流程 (预训练 + 微调 + 分析)')
    full_parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    full_parser.add_argument('--data_path', type=str, required=True, help='数据路径')
    full_parser.add_argument('--output_dir', type=str, default='./outputs', help='输出目录')
    full_parser.add_argument('--epochs', type=int, help='训练轮数')
    full_parser.add_argument('--batch_size', type=int, help='批次大小')
    full_parser.add_argument('--learning_rate', type=float, help='学习率')
    
    # 帮助命令
    help_parser = subparsers.add_parser('help', help='显示使用示例')
    
    args = parser.parse_args()
    
    if args.command is None or args.command == 'help':
        print_usage_examples()
        return
    
    # 创建输出目录
    if hasattr(args, 'output_dir'):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 执行相应命令
    if args.command == 'pretrain':
        kwargs = {k: v for k, v in vars(args).items() 
                 if k not in ['command', 'config', 'data_path', 'output_dir', 'resume_from']}
        success = run_pretraining(
            config_path=args.config,
            data_path=args.data_path,
            output_dir=args.output_dir,
            resume_from=args.resume_from,
            **kwargs
        )
        
    elif args.command == 'finetune':
        kwargs = {k: v for k, v in vars(args).items() 
                 if k not in ['command', 'config', 'data_path', 'output_dir', 'pretrained_model', 'resume_from']}
        success = run_finetuning(
            config_path=args.config,
            data_path=args.data_path,
            output_dir=args.output_dir,
            pretrained_model=args.pretrained_model,
            resume_from=args.resume_from,
            **kwargs
        )
        
    elif args.command == 'analyze':
        success = run_analysis(
            model_path=args.model_path,
            config_path=args.config,
            data_path=args.data_path,
            output_dir=args.output_dir
        )
        
    elif args.command == 'full':
        print("🔄 开始完整训练流程...")
        
        # 1. 预训练
        pretrain_dir = os.path.join(args.output_dir, 'pretrain')
        kwargs = {k: v for k, v in vars(args).items() 
                 if k not in ['command', 'config', 'data_path', 'output_dir']}
        
        success = run_pretraining(
            config_path=args.config,
            data_path=args.data_path,
            output_dir=pretrain_dir,
            **kwargs
        )
        
        if not success:
            print("❌ 预训练失败，停止流程")
            return
        
        # 2. 查找预训练模型
        pretrained_model = None
        pretrain_path = Path(pretrain_dir)
        if pretrain_path.exists():
            ckpt_files = list(pretrain_path.glob('**/*.ckpt'))
            if ckpt_files:
                pretrained_model = str(ckpt_files[-1])  # 使用最新的检查点
                print(f"📁 找到预训练模型: {pretrained_model}")
        
        # 3. 微调
        finetune_dir = os.path.join(args.output_dir, 'finetune')
        success = run_finetuning(
            config_path=args.config,
            data_path=args.data_path,
            output_dir=finetune_dir,
            pretrained_model=pretrained_model,
            **kwargs
        )
        
        if not success:
            print("❌ 微调失败，停止流程")
            return
        
        # 4. 查找微调模型
        finetuned_model = None
        finetune_path = Path(finetune_dir)
        if finetune_path.exists():
            ckpt_files = list(finetune_path.glob('**/*.ckpt'))
            if ckpt_files:
                finetuned_model = str(ckpt_files[-1])  # 使用最新的检查点
                print(f"📁 找到微调模型: {finetuned_model}")
        
        # 5. 结果分析
        if finetuned_model:
            analysis_dir = os.path.join(args.output_dir, 'analysis')
            success = run_analysis(
                model_path=finetuned_model,
                config_path=args.config,
                data_path=args.data_path,
                output_dir=analysis_dir
            )
            
            if success:
                print(f"\n🎉 完整训练流程成功完成!")
                print(f"📁 输出目录: {args.output_dir}")
                print(f"🤖 预训练模型: {pretrained_model}")
                print(f"🎯 微调模型: {finetuned_model}")
                print(f"📊 分析结果: {analysis_dir}")
                print(f"📈 TensorBoard日志: {args.output_dir}/*/tensorboard_logs")
                print(f"🌐 交互式仪表板: {analysis_dir}/interactive_dashboard.html")
        else:
            print("❌ 未找到微调模型，无法进行分析")
    
    else:
        print(f"❌ 未知命令: {args.command}")
        print_usage_examples()

if __name__ == '__main__':
    main()