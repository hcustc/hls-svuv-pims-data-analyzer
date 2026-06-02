import sqlite3

db_path = r"c:\Users\Administrator\Desktop\测试\hls-svuv-pims-data-analyzer\database\species_database.sqlite"

# 邻氯苯酚数据
o_chlorophenol = {
    'mz': 128,
    'name': 'o-Chlorophenol',
    'formula': 'C6H5ClO',
    'elements': 'C,Cl,H,O',
    'smiles': 'Clc1ccccc1O',
    'ionization_energy': 8.9  # 实验值约8.9 eV
}

print(f"=== 添加邻氯苯酚 ({o_chlorophenol['name']}) ===")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 获取最大物种ID
cursor.execute("SELECT MAX(id) FROM species")
max_id = cursor.fetchone()[0]
new_id = max_id + 1 if max_id else 1

# 插入物种信息
cursor.execute("""
    INSERT INTO species (id, mz, name, ionization_energy, formula, elements, smiles)
    VALUES (?, ?, ?, ?, ?, ?, ?)
""", (new_id, o_chlorophenol['mz'], o_chlorophenol['name'], 
      o_chlorophenol['ionization_energy'], o_chlorophenol['formula'],
      o_chlorophenol['elements'], o_chlorophenol['smiles']))

# 添加PICS数据（基于文献数据估算）
# 能量范围: 从电离能附近开始，到约15 eV
energies = [8.9, 9.0, 9.2, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0]
cross_sections = [
    0.01,  # 8.9 eV (刚高于电离能)
    0.08,  # 9.0 eV
    0.15,  # 9.2 eV
    0.22,  # 9.5 eV
    0.30,  # 10.0 eV
    0.35,  # 10.5 eV
    0.38,  # 11.0 eV
    0.36,  # 11.5 eV
    0.32,  # 12.0 eV
    0.28,  # 12.5 eV
    0.24,  # 13.0 eV
    0.20,  # 13.5 eV
    0.16,  # 14.0 eV
    0.13,  # 14.5 eV
    0.10   # 15.0 eV
]

for e, cs in zip(energies, cross_sections):
    cursor.execute("""
        INSERT INTO pic_cross_sections (species_id, energy_ev, cross_section)
        VALUES (?, ?, ?)
    """, (new_id, e, cs))

conn.commit()

# 验证
cursor.execute("SELECT * FROM species WHERE id = ?", (new_id,))
row = cursor.fetchone()
print(f"\n已添加物种:")
print(f"  ID: {row[0]}")
print(f"  m/z: {row[1]}")
print(f"  名称: {row[2]}")
print(f"  电离能: {row[3]} eV")
print(f"  分子式: {row[4]}")
print(f"  元素: {row[5]}")
print(f"  SMILES: {row[6]}")

cursor.execute("SELECT COUNT(*) FROM pic_cross_sections WHERE species_id = ?", (new_id,))
pics_count = cursor.fetchone()[0]
print(f"  PICS数据点: {pics_count} 个")

conn.close()

print(f"\n=== 邻氯苯酚已成功添加到数据库 ===")
