
import sys
sys.path.insert(0, '.')

from core.pie_analysis import analyze_pie_folder, analyze_multiple_pie_folders
from core.calibration import Calibration

# 测试文件夹路径
folders = [
    r'C:\Users\Administrator\Desktop\研究生\邻氯苯酚\2025.03\30torr\邻氯苯酚氧化PIE/O2: C6H5Cl=1/PIE',
    r'C:\Users\Administrator\Desktop\研究生\邻氯苯酚\2025.03\30torr\邻氯苯酚氧化PIE/O2: C6H5Cl=1/PIE-2'
]

print("=== 测试多段PIE数据合并 ===")
print(f"文件夹数量: {len(folders)}")
print()

# 分别分析每个文件夹
for i, folder in enumerate(folders):
    print(f"--- 文件夹 {i+1}: {folder.split('/')[-1]} ---")
    try:
        df = analyze_pie_folder(folder, calibration=Calibration())
        if not df.empty:
            print(f"  数据行数: {len(df)}")
            print(f"  能量范围: {df['energy'].min():.2f} - {df['energy'].max():.2f} eV")
            print(f"  质量数数量: {df['mz_rounded'].nunique()}")
        else:
            print("  无数据！")
    except Exception as e:
        print(f"  错误: {e}")
    print()

# 测试合并
print("=== 合并分析 ===")
try:
    merged_df = analyze_multiple_pie_folders(folders, calibration=Calibration())
    if not merged_df.empty:
        print(f"合并后数据行数: {len(merged_df)}")
        print(f"合并后能量范围: {merged_df['energy'].min():.2f} - {merged_df['energy'].max():.2f} eV")
        print(f"合并后质量数数量: {merged_df['mz_rounded'].nunique()}")
        print()
        print("能量点分布:")
        energy_counts = merged_df['energy'].value_counts().sort_index()
        for energy, count in energy_counts.items():
            print(f"  {energy:.2f} eV: {count} 个质量数")
    else:
        print("合并后无数据！")
except Exception as e:
    print(f"合并错误: {e}")
