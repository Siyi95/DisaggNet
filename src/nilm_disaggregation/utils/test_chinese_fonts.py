#!/usr/bin/env python3
"""测试Mac上中文字体显示效果"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

def test_chinese_fonts():
    """测试不同中文字体的显示效果"""
    
    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = []
    
    # 常见的Mac中文字体
    mac_chinese_fonts = [
        'PingFang SC',
        'Arial Unicode MS', 
        'Hiragino Sans GB',
        'STHeiti',
        'SimHei',
        'Microsoft YaHei',
        'DejaVu Sans'
    ]
    
    # 检查哪些字体可用
    for font in mac_chinese_fonts:
        if font in available_fonts:
            chinese_fonts.append(font)
    
    print("系统可用的中文字体:")
    for font in chinese_fonts:
        print(f"  - {font}")
    
    # 创建测试图表
    fig, axes = plt.subplots(len(chinese_fonts), 1, figsize=(12, 2*len(chinese_fonts)))
    if len(chinese_fonts) == 1:
        axes = [axes]
    
    test_text = "中文字体测试 - 电表数据可视化 - 功率预测结果"
    
    for i, font in enumerate(chinese_fonts):
        # 设置字体
        plt.rcParams['font.sans-serif'] = [font]
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建简单的测试图
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        axes[i].plot(x, y, label='正弦波')
        axes[i].set_title(f'{test_text} (字体: {font})', fontsize=14)
        axes[i].set_xlabel('时间 (秒)')
        axes[i].set_ylabel('幅值')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存测试结果
    output_dir = Path("outputs/font_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "chinese_font_test.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n字体测试图已保存到: {output_dir / 'chinese_font_test.png'}")
    
    # 创建最佳字体配置的演示图
    create_optimized_demo(output_dir, chinese_fonts)
    
    return chinese_fonts

def create_optimized_demo(output_dir, available_fonts):
    """创建使用最佳字体配置的演示图"""
    
    # 设置最佳字体配置（优先使用Mac原生字体）
    best_fonts = ['PingFang SC', 'Arial Unicode MS', 'Hiragino Sans GB', 'SimHei', 'DejaVu Sans']
    plt.rcParams['font.sans-serif'] = best_fonts
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建综合演示图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 线图
    x = np.linspace(0, 24, 100)
    y1 = 1000 + 200 * np.sin(2 * np.pi * x / 24) + 50 * np.random.randn(100)
    y2 = 500 + 100 * np.sin(2 * np.pi * x / 12) + 30 * np.random.randn(100)
    
    axes[0, 0].plot(x, y1, label='主电表功率', linewidth=2)
    axes[0, 0].plot(x, y2, label='设备功率', linewidth=2)
    axes[0, 0].set_title('24小时功率消耗趋势', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('时间 (小时)')
    axes[0, 0].set_ylabel('功率 (瓦特)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 柱状图
    devices = ['冰箱', '洗衣机', '微波炉', '洗碗机', '空调']
    power_values = [150, 800, 1200, 600, 2000]
    
    bars = axes[0, 1].bar(devices, power_values, color=['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink'])
    axes[0, 1].set_title('各设备平均功率消耗', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('功率 (瓦特)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars, power_values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                        f'{value}W', ha='center', va='bottom', fontweight='bold')
    
    # 3. 散点图
    np.random.seed(42)
    true_values = np.random.normal(500, 100, 100)
    predicted_values = true_values + np.random.normal(0, 50, 100)
    
    axes[1, 0].scatter(true_values, predicted_values, alpha=0.6, s=30)
    min_val, max_val = min(true_values.min(), predicted_values.min()), max(true_values.max(), predicted_values.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
    axes[1, 0].set_title('预测值 vs 真实值', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('真实功率 (瓦特)')
    axes[1, 0].set_ylabel('预测功率 (瓦特)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 饼图
    energy_sources = ['照明', '制冷', '加热', '电子设备', '其他']
    energy_percentages = [15, 30, 25, 20, 10]
    colors = ['gold', 'lightblue', 'lightcoral', 'lightgreen', 'plum']
    
    wedges, texts, autotexts = axes[1, 1].pie(energy_percentages, labels=energy_sources, 
                                             colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('家庭能耗分布', fontsize=14, fontweight='bold')
    
    # 设置饼图文字样式
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "optimized_chinese_demo.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"优化后的中文演示图已保存到: {output_dir / 'optimized_chinese_demo.png'}")
    
    # 生成字体配置报告
    create_font_report(output_dir, available_fonts)

def create_font_report(output_dir, available_fonts):
    """生成字体配置报告"""
    
    report_lines = []
    report_lines.append("# Mac中文字体配置报告")
    report_lines.append(f"生成时间: {plt.datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n" + "="*60 + "\n")
    
    report_lines.append("## 系统可用的中文字体")
    for i, font in enumerate(available_fonts, 1):
        report_lines.append(f"{i}. {font}")
    
    report_lines.append("\n## 推荐的字体配置")
    report_lines.append("```python")
    report_lines.append("# 设置中文字体支持（Mac优化）")
    report_lines.append("plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'Hiragino Sans GB', 'SimHei', 'DejaVu Sans']")
    report_lines.append("plt.rcParams['axes.unicode_minus'] = False")
    report_lines.append("```")
    
    report_lines.append("\n## 字体说明")
    report_lines.append("- **PingFang SC**: macOS系统默认中文字体，显示效果最佳")
    report_lines.append("- **Arial Unicode MS**: 支持多种语言的Unicode字体")
    report_lines.append("- **Hiragino Sans GB**: 日系字体，在Mac上显示效果良好")
    report_lines.append("- **SimHei**: 黑体，Windows系统常用")
    report_lines.append("- **DejaVu Sans**: 开源字体，作为备选")
    
    report_lines.append("\n## 使用建议")
    report_lines.append("1. 优先使用PingFang SC，这是macOS的原生中文字体")
    report_lines.append("2. 设置字体列表时，将最佳字体放在前面")
    report_lines.append("3. 设置 `axes.unicode_minus = False` 避免负号显示问题")
    report_lines.append("4. 如果仍有显示问题，可以尝试安装额外的中文字体")
    
    report_content = "\n".join(report_lines)
    
    with open(output_dir / "font_config_report.md", 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"字体配置报告已保存到: {output_dir / 'font_config_report.md'}")

def main():
    """主函数"""
    print("=" * 60)
    print("Mac中文字体测试")
    print("=" * 60)
    
    try:
        available_fonts = test_chinese_fonts()
        
        print("\n" + "=" * 60)
        print("测试完成! ✓")
        print(f"发现 {len(available_fonts)} 个可用的中文字体")
        print("查看 outputs/font_test/ 目录获取详细结果")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # 修复datetime导入问题
    from datetime import datetime
    plt.datetime = type('datetime', (), {'datetime': datetime})()
    
    exit(main())