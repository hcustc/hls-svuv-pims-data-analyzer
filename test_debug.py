
import sys
import os
sys.path.insert(0, '.')

print("="*60)
print("开始测试")
print("="*60)

try:
    from core.pie_analysis import analyze_pie_folder
    from core.calibration import Calibration
    print("导入成功")
except Exception as e:
    print(f"导入错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

folders = [
    r'C:\Users\Administrator\Desktop\研究生\邻氯苯酚\2025.03\30torr\邻氯苯酚氧化PIE\O2：C6H5Cl=1\PIE-2',
    r'C:\Users\Administrator\Desktop\研究生\邻氯苯酚\2025.03\30torr\邻氯苯酚氧化PIE\O2：C6H5Cl=1\PIE',
]

analysis_dfs = []

for i, folder in enumerate(folders):
    print(f"\n--- 文件夹 {i+1} ---")
    print(f"路径: {folder}")
    
    if not os.path.exists(folder):
        print(f"文件夹不存在！")
        continue
        
    try:
        df = analyze_pie_folder(folder, calibration=Calibration())
        print(f"分析成功！数据行数: {len(df)}")
        if not df.empty:
            print(f"质量数: {sorted(df['mz_rounded'].unique())}")
            print(f"能量范围: {df['energy'].min():.4f} - {df['energy'].max():.4f} eV")
            print(f"能量点: {sorted(df['energy'].unique())}")
        analysis_dfs.append(df)
    except Exception as e:
        print(f"分析错误: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("开始合并")
print("="*60)

try:
    from core.pie_analysis import merge_pie_segments
    merged_df = merge_pie_segments(analysis_dfs)
    print(f"合并成功！总行数: {len(merged_df)}")
    if not merged_df.empty:
        print(f"合并后能量范围: {merged_df['energy'].min():.4f} - {merged_df['energy'].max():.4f} eV")
except Exception as e:
    print(f"合并错误: {e}")
    import traceback
    traceback.print_exc()
