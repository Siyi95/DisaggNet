#!/usr/bin/env python3
"""
基础系统测试脚本（不依赖PyTorch）
用于验证项目结构和配置文件
"""

import os
import sys
import yaml
from pathlib import Path

def test_project_structure():
    """测试项目结构"""
    print("🔍 测试项目结构...")
    
    required_files = [
        "src/datamodule.py",
        "src/features.py",
        "src/models/mscat.py",
        "src/models/heads.py",
        "src/models/crf.py",
        "src/models/tcn_online.py",
        "src/train_pretrain.py",
        "src/train_finetune.py",
        "src/infer_offline.py",
        "src/infer_online_tcn.py",
        "configs/pretrain.yaml",
        "configs/finetune.yaml",
        "configs/online.yaml",
        "requirements.txt",
        "README.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (缺失)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ 缺失 {len(missing_files)} 个文件")
        return False
    else:
        print("\n✅ 所有必需文件都存在")
        return True

def test_config_files():
    """测试配置文件"""
    print("\n📋 测试配置文件...")
    
    config_files = {
        "configs/pretrain.yaml": ["data", "model", "training", "optimizer"],
        "configs/finetune.yaml": ["data", "model", "training", "optimizer", "loss"],
        "configs/online.yaml": ["data", "tcn_model", "online_detection", "training"]
    }
    
    all_valid = True
    
    for config_file, required_sections in config_files.items():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print(f"✅ {config_file} 加载成功")
            
            # 检查必需的配置节
            missing_sections = []
            for section in required_sections:
                if section not in config:
                    missing_sections.append(section)
            
            if missing_sections:
                print(f"   ⚠️ 缺失配置节: {missing_sections}")
                all_valid = False
            else:
                print(f"   ✅ 包含所有必需配置节")
                
        except FileNotFoundError:
            print(f"❌ {config_file} 文件不存在")
            all_valid = False
        except yaml.YAMLError as e:
            print(f"❌ {config_file} YAML 格式错误: {e}")
            all_valid = False
        except Exception as e:
            print(f"❌ {config_file} 加载失败: {e}")
            all_valid = False
    
    return all_valid

def test_python_syntax():
    """测试Python文件语法"""
    print("\n🐍 测试Python文件语法...")
    
    python_files = [
        "src/datamodule.py",
        "src/features.py",
        "src/models/mscat.py",
        "src/models/heads.py",
        "src/models/crf.py",
        "src/models/tcn_online.py",
        "src/train_pretrain.py",
        "src/train_finetune.py",
        "src/infer_offline.py",
        "src/infer_online_tcn.py"
    ]
    
    all_valid = True
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                source = f.read()
            
            # 编译检查语法
            compile(source, py_file, 'exec')
            print(f"✅ {py_file} 语法正确")
            
        except FileNotFoundError:
            print(f"❌ {py_file} 文件不存在")
            all_valid = False
        except SyntaxError as e:
            print(f"❌ {py_file} 语法错误: {e}")
            all_valid = False
        except Exception as e:
            print(f"❌ {py_file} 检查失败: {e}")
            all_valid = False
    
    return all_valid

def test_requirements():
    """测试requirements.txt"""
    print("\n📦 测试依赖文件...")
    
    try:
        with open("requirements.txt", 'r', encoding='utf-8') as f:
            requirements = f.read().strip().split('\n')
        
        # 过滤掉注释和空行
        packages = []
        for line in requirements:
            line = line.strip()
            if line and not line.startswith('#'):
                packages.append(line)
        
        print(f"✅ requirements.txt 包含 {len(packages)} 个包")
        
        # 检查关键包
        key_packages = ['torch', 'pytorch-lightning', 'numpy', 'pandas', 'h5py']
        missing_packages = []
        
        for pkg in key_packages:
            found = any(pkg in req for req in packages)
            if found:
                print(f"   ✅ {pkg}")
            else:
                print(f"   ❌ {pkg} (缺失)")
                missing_packages.append(pkg)
        
        if missing_packages:
            print(f"   ⚠️ 缺失关键包: {missing_packages}")
            return False
        
        return True
        
    except FileNotFoundError:
        print("❌ requirements.txt 文件不存在")
        return False
    except Exception as e:
        print(f"❌ requirements.txt 检查失败: {e}")
        return False

def test_readme():
    """测试README文件"""
    print("\n📖 测试README文件...")
    
    try:
        with open("README.md", 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键章节
        required_sections = [
            "项目架构",
            "快速开始",
            "模型架构",
            "配置说明"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"⚠️ README 缺失章节: {missing_sections}")
        else:
            print("✅ README 包含所有关键章节")
        
        print(f"✅ README.md 长度: {len(content)} 字符")
        return True
        
    except FileNotFoundError:
        print("❌ README.md 文件不存在")
        return False
    except Exception as e:
        print(f"❌ README.md 检查失败: {e}")
        return False

def check_data_directory():
    """检查数据目录"""
    print("\n💾 检查数据目录...")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"⚠️ {data_dir} 目录不存在，将创建")
        os.makedirs(data_dir, exist_ok=True)
        print(f"✅ 已创建 {data_dir} 目录")
    else:
        print(f"✅ {data_dir} 目录存在")
    
    # 检查是否有数据文件
    ampds2_file = "data/AMPds2.h5"
    if os.path.exists(ampds2_file):
        file_size = os.path.getsize(ampds2_file) / (1024 * 1024)  # MB
        print(f"✅ {ampds2_file} 存在 ({file_size:.1f} MB)")
    else:
        print(f"⚠️ {ampds2_file} 不存在")
        print("   提示: 可以运行 'python generate_sample_data.py' 生成测试数据")
    
    return True

def main():
    """主测试函数"""
    print("🚀 NILM MS-CAT 基础系统测试")
    print("=" * 50)
    
    # 显示Python版本
    print(f"Python 版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    print()
    
    # 运行测试
    tests = [
        ("项目结构", test_project_structure),
        ("配置文件", test_config_files),
        ("Python语法", test_python_syntax),
        ("依赖文件", test_requirements),
        ("README文档", test_readme),
        ("数据目录", check_data_directory)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 基础测试全部通过！")
        print("\n📝 下一步:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 生成测试数据: python generate_sample_data.py")
        print("3. 运行完整测试: python test_system.py")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查错误信息。")
        return 1

if __name__ == "__main__":
    exit(main())