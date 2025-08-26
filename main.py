#!/usr/bin/env python3
"""
DisaggNet 主入口脚本
提供统一的命令行接口访问所有功能
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(
        description='DisaggNet - 非侵入式负荷监测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py train --config config.yaml                    # 训练模型
  python main.py optimize --data-dir ./data --trials 50       # 超参数优化
  python main.py evaluate --checkpoint model.ckpt             # 评估模型
  python main.py visualize --data-dir ./data                  # 数据可视化
  python main.py demo --data-dir ./data                       # 运行演示
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', type=str, help='配置文件路径')
    train_parser.add_argument('--data-dir', type=str, required=True, help='数据目录')
    train_parser.add_argument('--output-dir', type=str, default='outputs/training', help='输出目录')
    train_parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    train_parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    train_parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    train_parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 优化命令
    optimize_parser = subparsers.add_parser('optimize', help='超参数优化')
    optimize_parser.add_argument('--data-dir', type=str, required=True, help='数据目录')
    optimize_parser.add_argument('--trials', type=int, default=50, help='优化试验次数')
    optimize_parser.add_argument('--output-dir', type=str, default='outputs/optimization', help='输出目录')
    optimize_parser.add_argument('--type', choices=['hyperparams', 'loss_weights'], default='hyperparams', help='优化类型')
    
    # 评估命令
    eval_parser = subparsers.add_parser('evaluate', help='评估模型')
    eval_parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    eval_parser.add_argument('--data-dir', type=str, required=True, help='数据目录')
    eval_parser.add_argument('--config', type=str, help='配置文件路径')
    eval_parser.add_argument('--output-dir', type=str, default='outputs/evaluation', help='输出目录')
    
    # 可视化命令
    viz_parser = subparsers.add_parser('visualize', help='数据可视化')
    viz_parser.add_argument('--data-dir', type=str, required=True, help='数据目录')
    viz_parser.add_argument('--output-dir', type=str, default='outputs/visualization', help='输出目录')
    viz_parser.add_argument('--max-samples', type=int, default=2000, help='最大样本数')
    
    # 演示命令
    demo_parser = subparsers.add_parser('demo', help='运行演示')
    demo_parser.add_argument('--data-dir', type=str, required=True, help='数据目录')
    demo_parser.add_argument('--output-dir', type=str, default='outputs/demo', help='输出目录')
    
    # 数据探索命令
    explore_parser = subparsers.add_parser('explore', help='探索数据结构')
    explore_parser.add_argument('--data-dir', type=str, required=True, help='数据目录')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # 执行相应命令
    if args.command == 'train':
        from src.nilm_disaggregation.training.train import main as train_main
        print("开始基于最终方案的模型训练...")
        try:
            train_main(args)
            print("基于最终方案的模型训练完成！")
        except Exception as e:
            print(f"训练过程中发生错误: {e}")
            raise
        
    elif args.command == 'optimize':
        from src.nilm_disaggregation.training.optimization import (
            run_hyperparameter_optimization, run_loss_weight_optimization
        )
        
        if args.type == 'hyperparams':
            print(f"开始基于最终方案的超参数优化，试验次数: {args.trials}")
            try:
                from src.nilm_disaggregation.training.advanced_strategies import AdvancedNILMLightningModule
                from src.nilm_disaggregation.models.enhanced_model_architecture import UncertaintyWeightedLoss
                from src.nilm_disaggregation.data.datamodule import NILMDataModule
                
                results, study = run_hyperparameter_optimization(
                    data_dir=args.data_dir,
                    n_trials=args.trials,
                    output_dir=args.output_dir
                )
                print(f"基于最终方案的超参数优化完成！")
                print(f"最佳参数: {results['best_params']}")
                print(f"最佳验证指标: {results.get('best_value', 'N/A')}")
            except Exception as e:
                print(f"超参数优化过程中发生错误: {e}")
                raise
            
        elif args.type == 'loss_weights':
            print(f"开始损失权重优化，试验次数: {args.trials}")
            best_params, best_loss, study = run_loss_weight_optimization(
                data_dir=args.data_dir,
                n_trials=args.trials
            )
            print(f"优化完成，最佳权重: {best_params}")
            
    elif args.command == 'evaluate':
        from src.nilm_disaggregation.training.evaluate import main as eval_main
        eval_main(args)
        
    elif args.command == 'visualize':
        from src.nilm_disaggregation.utils.visualize_complete_ampds2 import main as viz_main
        viz_main(args)
        
    elif args.command == 'demo':
        from src.nilm_disaggregation.data.demo_complete_dataset_usage import main as demo_main
        demo_main(args)
        
    elif args.command == 'explore':
        from src.nilm_disaggregation.data.explore_ampds2_structure import explore_ampds2_structure
        explore_ampds2_structure()
        
if __name__ == '__main__':
    main()