
import sys
sys.path.insert(0, '.')

from core.pie_analysis import analyze_pie_folder, analyze_multiple_pie_folders, build_pie_curves, identify_species_for_mz_with_curve
from core.calibration import Calibration
from core.elements import get_all_elements_from_database, filter_species_by_elements

print("="*70)
print("测试1：数据合并")
print("="*70)

folders = [
    r'C:\Users\Administrator\Desktop\研究生\邻氯苯酚\2025.03\30torr\邻氯苯酚氧化PIE\O2：C6H5Cl=1\PIE-2',  # 7.0-8.1 eV
    r'C:\Users\Administrator\Desktop\研究生\邻氯苯酚\2025.03\30torr\邻氯苯酚氧化PIE\O2：C6H5Cl=1\PIE',   # 8.0-11.0 eV
]

print(f"\n添加了 {len(folders)} 个文件夹")

# 测试合并
print("\n执行合并分析...")
try:
    merged_df = analyze_multiple_pie_folders(folders, calibration=Calibration())
    print(f"✓ 合并成功！")
    print(f"  - 总行数: {len(merged_df)}")
    print(f"  - 质量数数量: {merged_df['mz_rounded'].nunique()}")
    print(f"  - 能量范围: {merged_df['energy'].min():.4f} - {merged_df['energy'].max():.4f} eV")
    
    # 检查能量点分布
    print(f"\n  能量点分布:")
    energy_points = sorted(merged_df['energy'].unique())
    print(f"  总共 {len(energy_points)} 个能量点:")
    for e in energy_points:
        count = len(merged_df[merged_df['energy'] == e])
        print(f"    {e:.4f} eV: {count} 个质量数")
    
    # 构建曲线
    curves = build_pie_curves(merged_df)
    print(f"\n  构建曲线: {len(curves)} 条")
    
    # 显示第一个曲线的能量范围
    if curves:
        first_mz = sorted(curves.keys())[0]
        curve = curves[first_mz]
        print(f"  第一个曲线 (m/z {first_mz}):")
        print(f"    能量点数: {len(curve['energies'])}")
        print(f"    能量范围: {min(curve['energies']):.4f} - {max(curve['energies']):.4f} eV")
        
except Exception as e:
    print(f"✗ 合并失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("测试2：物种数据库")
print("="*70)

try:
    # 加载数据库
    from core.pie_analysis import load_species_database_sqlite
    db, mz_idx = load_species_database_sqlite('database/species_database.sqlite')
    print(f"✓ 加载物种数据库成功")
    print(f"  - 总物种数: {len(db)}")
    print(f"  - m/z 索引数: {len(mz_idx)}")
    
    # 获取所有元素
    all_elements = get_all_elements_from_database(db)
    print(f"\n  数据库中的元素: {', '.join(sorted(all_elements))}")
    
    # 测试C、H、O、F、Cl元素的筛选
    test_elements = {'C', 'H', 'O', 'F', 'Cl'}
    filtered = filter_species_by_elements(db, test_elements)
    print(f"\n  按元素筛选 {test_elements}:")
    print(f"    筛选后物种数: {len(filtered)}")
    
    if filtered:
        print(f"\n  筛选后的物种示例（前10个）:")
        for i in range(min(10, len(filtered))):
            species = filtered[i]
            formula = species.get('formula', '')
            elements = species.get('elements', '')
            print(f"    - {species['species']} (m/z {species['mz']}, formula: {formula}, elements: {elements})")
    
except Exception as e:
    print(f"✗ 数据库测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("测试3：物种识别测试")
print("="*70)

try:
    # 选择一个有数据的m/z进行测试
    if curves:
        test_mz = sorted(curves.keys())[0]  # 使用第一个m/z
        curve = curves[test_mz]
        
        energies = curve['energies']
        intensities = curve['intensities']
        
        print(f"\n测试 m/z {test_mz}:")
        print(f"  能量点数: {len(energies)}")
        print(f"  能量范围: {min(energies):.4f} - {max(energies):.4f} eV")
        
        # 使用筛选后的数据库进行识别
        fit_model = identify_species_for_mz_with_curve(
            filtered, test_mz, energies, intensities
        )
        
        if fit_model and fit_model.get('species'):
            print(f"\n  ✓ 识别到 {len(fit_model['species'])} 个物种:")
            for i, species in enumerate(fit_model['species'][:5]):
                print(f"    {i+1}. {species.get('species', 'Unknown')} (IE: {species.get('ie', 'N/A')}, R²: {species.get('r_squared', 0):.4f})")
        else:
            print(f"\n  ✗ 未识别到任何物种")
            print(f"    可能原因:")
            print(f"      1. 数据库中不包含 m/z {test_mz} 的物种")
            print(f"      2. 该物质的PIE曲线与数据库中的物种不匹配")
            
            # 检查数据库中是否有这个m/z
            if test_mz in mz_idx:
                print(f"      3. 数据库中有该m/z，但被元素筛选过滤掉了")
                possible_species = [db[i] for i in mz_idx[test_mz]]
                print(f"         数据库中m/z {test_mz}的物种:")
                for sp in possible_species[:5]:
                    print(f"           - {sp['species']} (elements: {sp.get('elements', 'N/A')})")
    
except Exception as e:
    print(f"✗ 物种识别测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("测试完成")
print("="*70)
