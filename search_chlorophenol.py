import sqlite3

db_path = r"c:\Users\Administrator\Desktop\测试\hls-svuv-pims-data-analyzer\database\species_database.sqlite"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=== 搜索氯苯酚相关物种 ===\n")

# 搜索所有含氯苯酚的物种
cursor.execute("""
    SELECT id, mz, name, formula, elements, smiles 
    FROM species 
    WHERE mz = 128 OR name LIKE '%chlorophenol%' OR name LIKE '%Chlorophenol%'
""")
results = cursor.fetchall()

print(f"m/z=128 的物种列表:\n")
for row in results:
    print(f"ID: {row[0]}")
    print(f"  m/z: {row[1]}")
    print(f"  名称: {row[2]}")
    print(f"  分子式: {row[3]}")
    print(f"  元素: {row[4]}")
    print(f"  SMILES: {row[5]}")
    print()

# 检查其他可能的异构体
print("\n=== 搜索邻/间/对氯苯酚 ===\n")
names_to_search = [
    'o-Chlorophenol', 'ortho-Chlorophenol', '2-Chlorophenol',
    'm-Chlorophenol', 'meta-Chlorophenol', '3-Chlorophenol',
    'p-Chlorophenol', 'para-Chlorophenol', '4-Chlorophenol',
    '邻氯苯酚', '间氯苯酚', '对氯苯酚'
]

for name in names_to_search:
    cursor.execute("SELECT id, mz, name, formula FROM species WHERE name LIKE ?", (f'%{name}%',))
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            print(f"找到: {row[2]} - m/z={row[1]}, formula={row[3]}")

print("\n=== m/z=128 的所有物种 ===\n")
cursor.execute("SELECT id, name, formula, ionization_energy FROM species WHERE mz = 128")
rows = cursor.fetchall()
if rows:
    for row in rows:
        print(f"ID={row[0]}: {row[1]} (IE={row[3]} eV), formula={row[2]}")

conn.close()
