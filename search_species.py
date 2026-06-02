import sqlite3

db_path = r"c:\Users\Administrator\Desktop\测试\hls-svuv-pims-data-analyzer\database\species_database.sqlite"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 搜索 C6H5ClO
print("=== 搜索 C6H5ClO ===\n")

# 1. 搜索名称中包含 C6H5ClO 的物种
cursor.execute("""
    SELECT id, mz, name, formula, elements, smiles 
    FROM species 
    WHERE name LIKE '%C6H5ClO%' OR name LIKE '%ClO%'
""")
results = cursor.fetchall()

if results:
    print(f"找到 {len(results)} 个相关物种:\n")
    for row in results:
        print(f"ID: {row[0]}")
        print(f"  m/z: {row[1]}")
        print(f"  名称: {row[2]}")
        print(f"  分子式: {row[3]}")
        print(f"  元素: {row[4]}")
        print(f"  SMILES: {row[5]}")
        print()
else:
    print("直接在名称中未找到 C6H5ClO\n")

# 2. 搜索分子式为 C6H5ClO 的物种
cursor.execute("""
    SELECT id, mz, name, formula, elements, smiles 
    FROM species 
    WHERE formula = 'C6H5ClO' OR formula LIKE '%C6H5ClO%'
""")
results = cursor.fetchall()

if results:
    print(f"分子式匹配: {len(results)} 个\n")
    for row in results:
        print(f"ID: {row[0]}, m/z={row[1]}, name={row[2]}, formula={row[3]}")

# 3. 列出所有含 Cl 的物种
print(f"\n=== 含 Cl 元素的物种 (前20个) ===\n")
cursor.execute("""
    SELECT id, mz, name, formula, elements, smiles 
    FROM species 
    WHERE elements LIKE '%Cl%' OR name LIKE '%chloro%' OR name LIKE '%Chlor%'
    LIMIT 20
""")
results = cursor.fetchall()

if results:
    print(f"找到 {len(results)} 个含Cl物种:\n")
    for row in results:
        print(f"ID: {row[0]}, m/z={row[1]}, name={row[2]}, formula={row[3]}, elements={row[4]}")
else:
    print("未找到含Cl物种\n")

# 4. 搜索含氯苯酚类的物种
print(f"\n=== 搜索苯酚类物种 ===\n")
cursor.execute("""
    SELECT id, mz, name, formula, elements, smiles 
    FROM species 
    WHERE name LIKE '%phenol%' OR name LIKE '%Phenol%' OR name LIKE '%C6H5%'
    LIMIT 20
""")
results = cursor.fetchall()

if results:
    print(f"找到 {len(results)} 个相关物种:\n")
    for row in results:
        print(f"ID: {row[0]}, m/z={row[1]}, name={row[2]}, formula={row[3]}, elements={row[4]}")
else:
    print("未找到苯酚类物种\n")

conn.close()
