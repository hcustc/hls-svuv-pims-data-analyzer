
import sys
sys.path.insert(0, '.')

print("正在导入模块...")
from core.pie_analysis import analyze_pie_folder
from core.calibration import Calibration

print("导入成功！")

# 先测试单个文件夹
folder1 = r'C:\Users\Administrator\Desktop\研究生\邻氯苯酚\2025.03\30torr\邻氯苯酚氧化PIE\O2：C6H5Cl=1\PIE-2'
print(f"\n分析文件夹: {folder1}")

try:
    df1 = analyze_pie_folder(folder1, calibration=Calibration())
    print(f"成功！数据行数: {len(df1)}")
    if not df1.empty:
        print(f"能量范围: {df1['energy'].min():.4f} - {df1['energy'].max():.4f} eV")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

print("\n---\n")

folder2 = r'C:\Users\Administrator\Desktop\研究生\邻氯苯酚\2025.03\30torr\邻氯苯酚氧化PIE\O2：C6H5Cl=1\PIE'
print(f"分析文件夹: {folder2}")

try:
    df2 = analyze_pie_folder(folder2, calibration=Calibration())
    print(f"成功！数据行数: {len(df2)}")
    if not df2.empty:
        print(f"能量范围: {df2['energy'].min():.4f} - {df2['energy'].max():.4f} eV")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
