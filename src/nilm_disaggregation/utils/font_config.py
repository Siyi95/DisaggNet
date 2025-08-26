"""字体配置模块 - 为Mac系统优化中文字体显示"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings


def setup_chinese_fonts():
    """
    设置中文字体支持（Mac优化）
    
    这个函数会自动检测系统可用的中文字体，并设置最佳的字体配置。
    优先使用Mac原生字体，确保中文字符能够正确显示。
    """
    
    # Mac系统推荐的中文字体列表（按优先级排序）
    mac_chinese_fonts = [
        'PingFang SC',          # macOS默认中文字体
        'Arial Unicode MS',     # Unicode字体，支持多语言
        'Hiragino Sans GB',     # 日系字体，Mac上效果好
        'STHeiti',              # 黑体
        'SimHei',               # 简体中文黑体
        'Microsoft YaHei',      # 微软雅黑
        'DejaVu Sans'           # 开源字体备选
    ]
    
    # 检测系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    usable_fonts = []
    
    for font in mac_chinese_fonts:
        if font in available_fonts:
            usable_fonts.append(font)
    
    # 如果没有找到任何中文字体，使用默认配置
    if not usable_fonts:
        usable_fonts = ['sans-serif']
        warnings.warn(
            "未找到推荐的中文字体，可能会出现中文显示问题。"
            "建议安装 PingFang SC 或 Arial Unicode MS 字体。",
            UserWarning
        )
    
    # 设置matplotlib字体配置
    plt.rcParams['font.sans-serif'] = usable_fonts
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 清除字体缓存，确保新设置生效
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams['font.sans-serif'] = usable_fonts
    plt.rcParams['axes.unicode_minus'] = False
    
    return usable_fonts


def get_available_chinese_fonts():
    """
    获取系统可用的中文字体列表
    
    Returns:
        list: 可用的中文字体名称列表
    """
    mac_chinese_fonts = [
        'PingFang SC',
        'Arial Unicode MS',
        'Hiragino Sans GB',
        'STHeiti',
        'SimHei',
        'Microsoft YaHei',
        'DejaVu Sans'
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    return [font for font in mac_chinese_fonts if font in available_fonts]


def test_font_display(test_text="中文字体测试"):
    """
    测试中文字体显示效果
    
    Args:
        test_text (str): 测试文本
        
    Returns:
        bool: 是否能正确显示中文
    """
    try:
        # 设置字体
        setup_chinese_fonts()
        
        # 创建测试图
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, test_text, ha='center', va='center', fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # 检查是否有字体警告
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            plt.tight_layout()
            
            # 如果有字体相关警告，说明显示可能有问题
            font_warnings = [warning for warning in w 
                           if 'font' in str(warning.message).lower() or 
                              'glyph' in str(warning.message).lower()]
            
            plt.close(fig)
            return len(font_warnings) == 0
            
    except Exception:
        return False


def print_font_info():
    """
    打印字体配置信息
    """
    available_fonts = get_available_chinese_fonts()
    current_fonts = plt.rcParams['font.sans-serif']
    
    print("=" * 50)
    print("字体配置信息")
    print("=" * 50)
    print(f"系统可用中文字体: {len(available_fonts)} 个")
    for i, font in enumerate(available_fonts, 1):
        print(f"  {i}. {font}")
    
    print(f"\n当前matplotlib字体配置:")
    for i, font in enumerate(current_fonts, 1):
        print(f"  {i}. {font}")
    
    print(f"\n负号显示设置: {not plt.rcParams['axes.unicode_minus']}")
    
    # 测试显示效果
    display_ok = test_font_display()
    print(f"中文显示测试: {'✓ 正常' if display_ok else '✗ 可能有问题'}")
    print("=" * 50)


# 自动设置字体（导入时执行）
if __name__ != '__main__':
    setup_chinese_fonts()


if __name__ == '__main__':
    # 如果直接运行此脚本，显示字体信息
    setup_chinese_fonts()
    print_font_info()