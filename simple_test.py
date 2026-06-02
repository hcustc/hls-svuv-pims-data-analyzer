
import sys
sys.path.insert(0, '.')

print("测试开始...")

try:
    from core.pie_analysis import analyze_pie_folder
    from core.calibration import Calibration
    
    folder1 = r'C:\Users\Administrator\Desktop\研究生\邻氯苯酚\2025.03\30torr\邻氯苯酚氧化PIE\O2：C6H5Cl=1\PIE-2'
    
    print(f"分析文件夹1: {folder1.split('/')[-1]}")
    df1 = analyze_pie_folder(folder1, calibration=Calibration())
    
    print(f"成功！行数: {len(df1)}")
    print(f"能量范围: {df1['energy'].min():.4f} - {df1['energy'].max():.4f}")
    
    folder2 = r'C:\Users\Administrator\Desktop\研究生\邻氯苯酚\2025.03\30torr\邻氯苯酚氧化PIE\O2：C6H5Cl=1\PIE'
    
    print(f"\n分析文件夹2: {folder2.split('/')[-1]}")
    df2 = analyze_pie_folder(folder2, calibration=Calibration())
    
    print(f"成功！行数: {len(df2)}")
    print(f"能量范围: {df2['energy'].min():.4f} - {df2['energy'].max():.4f}")
    
    from core.pie_analysis import merge_pie_segments
    
    print("\n合并...")
    merged = merge_pie_segments([df1, df2])
    print(f"合并后: {len(merged)} 行")
    print(f"能量范围: {merged['energy'].min():.4f} - {merged['energy'].max():.4f}")
    
    from core.pie_analysis import build_pie_curves
    curves = build_pie_curves(merged)
    print(f"曲线数: {len(curves)}")
    
    if curves:
        mz = sorted(curves.keys())[0]
        c = curves[mz]
        print(f"\n第一个曲线 m/z {mz}:")
        print(f"  能量点数: {len(c['energies'])}")
        print(f"  能量: {sorted(c['energies'])[:5]}...")
    
    print("\n测试成功！")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
