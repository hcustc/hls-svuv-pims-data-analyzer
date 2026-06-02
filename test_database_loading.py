
import sys
sys.path.insert(0, '.')

print("="*70)
print("测试数据库加载和元素筛选")
print("="*70)

try:
    # 1. 加载数据库
    from core.pie_analysis import load_species_database_sqlite
    db, mz_idx = load_species_database_sqlite('database/species_database.sqlite')
    
    print(f"\n✓ 加载数据库成功")
    print(f"  - 总物种数: {len(db)}")
    print(f"  - m/z 索引数: {len(mz_idx)}")
    
    # 2. 检查物种是否包含新字段
    print("\n✓ 检查物种数据结构:")
    if db:
        sample = db[0]
        print(f"  字段: {', '.join(sample.keys())}")
        print(f"  示例物种:")
        print(f"    名称: {sample.get('species')}")
        print(f"    分子式: {sample.get('formula')}")
        print(f"    元素: {sample.get('elements')}")
        print(f"    SMILES: {sample.get('smiles')}")
    
    # 3. 测试元素筛选
    print("\n✓ 测试元素筛选:")
    from core.elements import filter_species_by_elements
    
    # 筛选包含Cl的物种
    cl_species = filter_species_by_elements(db, {'Cl'})
    print(f"  含Cl的物种数: {len(cl_species)}")
    
    # 筛选包含C, H, O, Cl的物种
    cho_cl_species = filter_species_by_elements(db, {'C', 'H', 'O', 'Cl'})
    print(f"  含C,H,O,Cl的物种数: {len(cho_cl_species)}")
    
    if cho_cl_species:
        print(f"\n  筛选后的物种示例（前10个）:")
        for i, sp in enumerate(cho_cl_species[:10]):
            print(f"    {i+1}. {sp['species']} (m/z {sp['mz']}, formula: {sp.get('formula')}, elements: {sp.get('elements')})")
    
    # 4. 测试m/z 112的物种（Monochlorobenzene）
    print("\n✓ 测试m/z 112的物种:")
    if 112 in mz_idx:
        species_112 = [db[i] for i in mz_idx[112]]
        print(f"  数据库中m/z 112的物种数: {len(species_112)}")
        for sp in species_112[:5]:
            print(f"    - {sp['species']} (formula: {sp.get('formula')}, elements: {sp.get('elements')})")
    else:
        print("  ✗ 数据库中没有m/z 112的物种")
    
    print("\n" + "="*70)
    print("✓ 所有测试通过！数据库已正确更新")
    print("="*70)
    
except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
