#!/usr/bin/env python3
"""
探索AMPds2数据集结构
"""

import h5py
import numpy as np

def explore_ampds2_structure():
    """探索AMPds2数据集的详细结构"""
    data_path = '/Users/siyili/Workspace/DisaggNet/Dataset/dataverse_files/AMPds2.h5'
    
    with h5py.File(data_path, 'r') as f:
        print("=== AMPds2数据集结构探索 ===")
        
        # 探索building1
        building1 = f['building1']
        print(f"building1 keys: {list(building1.keys())}")
        
        # 探索elec
        elec = building1['elec']
        print(f"elec keys: {list(elec.keys())}")
        
        # 探索第一个电表
        meter1 = elec['meter1']
        print(f"\nmeter1 keys: {list(meter1.keys())}")
        print(f"meter1 type: {type(meter1)}")
        
        # 检查每个键的内容
        for key in meter1.keys():
            item = meter1[key]
            print(f"\nmeter1['{key}']:")
            print(f"  类型: {type(item)}")
            if hasattr(item, 'shape'):
                print(f"  形状: {item.shape}")
            if hasattr(item, 'dtype'):
                print(f"  数据类型: {item.dtype}")
            if hasattr(item, 'keys'):
                print(f"  子键: {list(item.keys())}")
        
        # 如果有power数据，查看其结构
        if 'power' in meter1:
            power_data = meter1['power']
            print(f"\npower数据详情:")
            print(f"  形状: {power_data.shape}")
            print(f"  数据类型: {power_data.dtype}")
            print(f"  前10个值: {power_data[:10]}")
        
        # 探索几个不同的电表
        print("\n=== 探索不同电表 ===")
        for meter_name in ['meter1', 'meter2', 'meter3']:
            if meter_name in elec:
                meter = elec[meter_name]
                print(f"\n{meter_name}:")
                print(f"  键: {list(meter.keys())}")
                
                # 检查是否有功率数据
                for data_key in ['power', 'energy', 'voltage', 'current']:
                    if data_key in meter:
                        data = meter[data_key]
                        print(f"  {data_key}: 形状={data.shape}, 类型={data.dtype}")
                        if len(data.shape) > 1:
                            print(f"    通道数: {data.shape[1]}")
        
        # 检查cache
        if 'cache' in elec:
            cache = elec['cache']
            print(f"\ncache:")
            print(f"  类型: {type(cache)}")
            if hasattr(cache, 'keys'):
                print(f"  键: {list(cache.keys())}")

if __name__ == '__main__':
    explore_ampds2_structure()