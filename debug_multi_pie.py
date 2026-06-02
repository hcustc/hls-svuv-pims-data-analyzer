#!/usr/bin/env python3
"""调试多段PIE合并功能"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.pie_analysis import merge_pie_segments, build_pie_curves


def create_test_data():
    """创建测试数据"""
    print("创建测试数据...")
    
    # 创建第一个段（7-8.1 eV）
    data1 = []
    for energy in np.arange(7.0, 8.2, 0.1):
        data1.append({
            'energy': round(energy, 2),  # 刚好两位小数
            'mz_rounded': 100,
            'species': 'test',
            'normalized_intensity': 1.0 + (energy - 7.0) * 0.1
        })
    df1 = pd.DataFrame(data1)
    df1['mz_rounded'] = 100
    df1['species'] = 'test1'
    df1['raw_area'] = 1000.0 + (df1['energy'] - 7.0) * 100
    df1['normalized_intensity'] = df1['raw_area'] / 1000.0
    df1['photon_normalized_intensity'] = df1['normalized_intensity']
    df1['mz'] = 100.0
    df1['file_count'] = 1
    df1['io'] = 1.0
    df1['light_source'] = 'io'
    df1['left_bound'] = 0
    df1['right_bound'] = 10
    
    # 创建第二个段（8.0-11.0 eV）
    # 能量在8.0到8.1时是8.00 + 0.005（这样舍入到两位小数后刚好是8.01）
    data2 = []
    for energy in np.arange(8.0, 11.1, 0.1):
        # 对于8.0和8.1，我们使用完全相同的舍入值，确保重叠
        if energy < 8.11:
            actual_energy = energy  # 完全一样，确保重叠
        else:
            actual_energy = energy + 0.005  # 其他点有小偏移
        data2.append({
            'energy': round(actual_energy, 2),
            'mz_rounded': 100,
            'species': 'test2',
            'raw_area': 1000.0 + (energy - 8.0) * 100 + 50,
            'normalized_intensity': (1000.0 + (energy - 8.0) * 100 + 50) / 1000.0,
            'photon_normalized_intensity': (1000.0 + (energy - 8.0) * 100 + 50) / 1000.0,
            'mz': 100.0,
            'file_count': 1,
            'io': 1.0,
            'light_source': 'io',
            'left_bound': 0,
            'right_bound': 10
        })
    df2 = pd.DataFrame(data2)
    df2['mz_rounded'] = 100
    
    print(f"  第一段能量示例: {df1['energy'].iloc[:5].tolist()}")
    print(f"  第二段能量示例: {df2['energy'].iloc[:5].tolist()}")
    
    return df1, df2


def test_merge_logic():
    """测试合并逻辑"""
    print("\n=== 测试合并逻辑 ===")
    
    df1, df2 = create_test_data()
    
    # 确保列顺序一致
    columns = ['energy', 'file_count', 'io', 'light_source', 'mz', 'mz_rounded', 
               'species', 'photon_normalized_intensity', 'normalized_intensity', 
               'raw_area', 'left_bound', 'right_bound']
    
    df1 = df1[columns]
    df2 = df2[columns]
    
    # 模拟merge_pie_segments的输入格式
    analysis_dfs = [df1, df2]
    
    print(f"第一段数据点: {len(df1)}")
    print(f"第二段数据点: {len(df2)}")
    
    # 测试合并
    merged_df = merge_pie_segments(analysis_dfs, merge_method='low_energy_dominant')
    print(f"合并后数据点: {len(merged_df)}")
    
    print("\n合并后的能量点:")
    print(merged_df[['energy', 'normalized_intensity']].to_string())
    
    # 测试曲线构建
    curves = build_pie_curves(merged_df)
    print(f"\n构建了 {len(curves)} 条曲线")
    for mz, curve in curves.items():
        print(f"  m/z {mz}: {len(curve['energies'])} 个点")
        print(f"  能量列表: {[round(e, 2) for e in curve['energies']]}")
    
    return merged_df


if __name__ == '__main__':
    print("多段PIE合并调试工具")
    print("=" * 50)
    
    test_merge_logic()
    
    print("\n" + "=" * 50)
    print("调试完成")
