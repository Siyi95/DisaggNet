#!/usr/bin/env python3
"""
生成模拟 AMPds2 数据用于测试
"""

import h5py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path

def generate_device_pattern(hours, base_power, on_prob=0.3, min_duration=10, max_duration=120):
    """
    生成单个设备的功率模式
    
    Args:
        hours: 总时长（小时）
        base_power: 基础功率（瓦特）
        on_prob: 开启概率
        min_duration: 最小持续时间（分钟）
        max_duration: 最大持续时间（分钟）
    
    Returns:
        功率序列（每分钟一个点）
    """
    minutes = hours * 60
    power = np.zeros(minutes)
    
    i = 0
    while i < minutes:
        if np.random.random() < on_prob:
            # 设备开启
            duration = np.random.randint(min_duration, max_duration + 1)
            duration = min(duration, minutes - i)
            
            # 添加一些功率变化
            device_power = base_power * (0.8 + 0.4 * np.random.random())
            noise = np.random.normal(0, base_power * 0.05, duration)
            
            power[i:i+duration] = device_power + noise
            i += duration
        else:
            # 设备关闭，跳过一段时间
            off_duration = np.random.randint(min_duration, max_duration + 1)
            off_duration = min(off_duration, minutes - i)
            i += off_duration
    
    # 确保功率非负
    power = np.maximum(power, 0)
    
    return power

def generate_ampds2_data(output_path="data/AMPds2.h5", days=30):
    """
    生成模拟的 AMPds2 数据集
    
    Args:
        output_path: 输出文件路径
        days: 生成数据的天数
    """
    print(f"🔧 生成 {days} 天的模拟 AMPds2 数据...")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 时间参数
    hours = days * 24
    minutes = hours * 60
    
    # 生成时间戳（每分钟一个点）
    start_time = datetime(2022, 1, 1)
    timestamps = [start_time + timedelta(minutes=i) for i in range(minutes)]
    timestamp_index = pd.DatetimeIndex(timestamps)
    
    # 设备配置
    devices = {
        'meter_02': {'name': 'washing_machine', 'base_power': 500, 'on_prob': 0.1},
        'meter_03': {'name': 'dishwasher', 'base_power': 800, 'on_prob': 0.08},
        'meter_04': {'name': 'dryer', 'base_power': 1200, 'on_prob': 0.06},
        'meter_05': {'name': 'fridge', 'base_power': 150, 'on_prob': 0.4},
        'meter_06': {'name': 'microwave', 'base_power': 1000, 'on_prob': 0.05},
        'meter_07': {'name': 'oven', 'base_power': 2000, 'on_prob': 0.03},
        'meter_08': {'name': 'air_conditioner', 'base_power': 1500, 'on_prob': 0.2},
        'meter_09': {'name': 'water_heater', 'base_power': 3000, 'on_prob': 0.15},
        'meter_10': {'name': 'lighting', 'base_power': 200, 'on_prob': 0.3},
        'meter_11': {'name': 'tv', 'base_power': 120, 'on_prob': 0.25},
    }
    
    # 生成设备功率数据
    device_powers = {}
    for meter_id, config in devices.items():
        print(f"  生成 {config['name']} 数据...")
        power = generate_device_pattern(
            hours=hours,
            base_power=config['base_power'],
            on_prob=config['on_prob']
        )
        device_powers[meter_id] = power
    
    # 计算总功率
    total_power = sum(device_powers.values())
    
    # 添加基础负载（常开设备）
    base_load = 100 + 50 * np.random.random(minutes)  # 100-150W 基础负载
    total_power += base_load
    
    # 生成其他电气参数
    voltage = 230 + 10 * np.random.normal(0, 1, minutes)  # 230V ± 10V
    current = total_power / voltage  # I = P / V
    
    # 功率因数（0.85-0.98）
    power_factor = 0.85 + 0.13 * np.random.random(minutes)
    
    # 无功功率和视在功率
    reactive_power = total_power * np.tan(np.arccos(power_factor))
    apparent_power = total_power / power_factor
    
    # 添加噪声
    noise_level = 0.02
    total_power += np.random.normal(0, np.mean(total_power) * noise_level, minutes)
    reactive_power += np.random.normal(0, np.mean(reactive_power) * noise_level, minutes)
    voltage += np.random.normal(0, 2, minutes)
    current += np.random.normal(0, np.mean(current) * noise_level, minutes)
    
    # 确保数值合理
    total_power = np.maximum(total_power, 0)
    reactive_power = np.maximum(reactive_power, 0)
    voltage = np.clip(voltage, 200, 250)
    current = np.maximum(current, 0)
    power_factor = np.clip(power_factor, 0.5, 1.0)
    
    # 创建 HDF5 文件
    print("💾 保存到 HDF5 文件...")
    
    with h5py.File(output_path, 'w') as f:
        # 创建电力数据组
        electricity_group = f.create_group('electricity')
        
        # 总功率表（meter_01）
        meter_01 = electricity_group.create_group('meter_01')
        meter_01.create_dataset('P', data=total_power)
        meter_01.create_dataset('Q', data=reactive_power)
        meter_01.create_dataset('S', data=apparent_power)
        meter_01.create_dataset('I', data=current)
        meter_01.create_dataset('V', data=voltage)
        meter_01.create_dataset('PF', data=power_factor)
        
        # 设备功率表
        for meter_id, power in device_powers.items():
            meter_group = electricity_group.create_group(meter_id)
            meter_group.create_dataset('P', data=power)
            
            # 为设备生成简化的其他参数
            device_current = power / voltage
            device_pf = 0.9 + 0.1 * np.random.random(minutes)
            device_q = power * np.tan(np.arccos(device_pf))
            device_s = power / device_pf
            
            meter_group.create_dataset('Q', data=device_q)
            meter_group.create_dataset('S', data=device_s)
            meter_group.create_dataset('I', data=device_current)
            meter_group.create_dataset('V', data=voltage)  # 共享电压
            meter_group.create_dataset('PF', data=device_pf)
        
        # 保存时间戳信息
        time_group = f.create_group('timestamps')
        
        # 将时间戳转换为字符串格式
        timestamp_strings = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps]
        time_group.create_dataset('datetime', data=timestamp_strings)
        
        # Unix 时间戳
        unix_timestamps = [ts.timestamp() for ts in timestamps]
        time_group.create_dataset('unix', data=unix_timestamps)
        
        # 元数据
        metadata_group = f.create_group('metadata')
        metadata_group.attrs['description'] = 'Simulated AMPds2 dataset for NILM testing'
        metadata_group.attrs['sampling_rate'] = '1 minute'
        metadata_group.attrs['duration_days'] = days
        metadata_group.attrs['num_devices'] = len(devices)
        metadata_group.attrs['generated_by'] = 'NILM MS-CAT test script'
        
        # 设备信息
        device_info = metadata_group.create_group('devices')
        for meter_id, config in devices.items():
            device_group = device_info.create_group(meter_id)
            device_group.attrs['name'] = config['name']
            device_group.attrs['base_power'] = config['base_power']
            device_group.attrs['on_probability'] = config['on_prob']
    
    print(f"✅ 模拟数据生成完成！")
    print(f"   文件路径: {output_path}")
    print(f"   数据点数: {minutes:,} (每分钟)")
    print(f"   设备数量: {len(devices)}")
    print(f"   总功率范围: {total_power.min():.1f} - {total_power.max():.1f} W")
    print(f"   平均功率: {total_power.mean():.1f} W")

def verify_data(file_path="data/AMPds2.h5"):
    """
    验证生成的数据
    """
    print(f"\n🔍 验证数据文件: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print("📁 文件结构:")
            
            def print_structure(name, obj):
                indent = "  " * (name.count('/') - 1)
                if isinstance(obj, h5py.Group):
                    print(f"{indent}📂 {name.split('/')[-1]}/")
                else:
                    shape = obj.shape if hasattr(obj, 'shape') else 'scalar'
                    print(f"{indent}📄 {name.split('/')[-1]} {shape}")
            
            f.visititems(print_structure)
            
            # 检查数据完整性
            print("\n📊 数据统计:")
            
            # 总功率
            total_power = f['electricity/meter_01/P'][:]
            print(f"   总功率: {len(total_power)} 个数据点")
            print(f"   范围: {total_power.min():.1f} - {total_power.max():.1f} W")
            print(f"   平均: {total_power.mean():.1f} W")
            
            # 设备数量
            device_count = len([k for k in f['electricity'].keys() if k.startswith('meter_') and k != 'meter_01'])
            print(f"   设备数量: {device_count}")
            
            # 时间戳
            if 'timestamps' in f:
                timestamps = f['timestamps/unix'][:]
                print(f"   时间范围: {len(timestamps)} 个时间点")
                start_time = datetime.fromtimestamp(timestamps[0])
                end_time = datetime.fromtimestamp(timestamps[-1])
                print(f"   开始时间: {start_time}")
                print(f"   结束时间: {end_time}")
        
        print("✅ 数据验证通过！")
        return True
        
    except Exception as e:
        print(f"❌ 数据验证失败: {e}")
        return False

def main():
    """
    主函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='生成模拟 AMPds2 数据')
    parser.add_argument('--days', type=int, default=7, help='生成数据的天数')
    parser.add_argument('--output', type=str, default='data/AMPds2.h5', help='输出文件路径')
    parser.add_argument('--verify-only', action='store_true', help='仅验证现有数据')
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_data(args.output)
    else:
        generate_ampds2_data(args.output, args.days)
        verify_data(args.output)

if __name__ == "__main__":
    main()