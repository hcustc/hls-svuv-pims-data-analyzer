#!/usr/bin/env python3
"""将 species_database.pkl 转换为 SQLite 数据库"""

import pickle
import sqlite3
import numpy as np
from pathlib import Path
import shutil
import time

# 路径配置
pkl_path = r"E:\质谱峰自动处理小程序\species_database.pkl"
output_path = r"c:\Users\Administrator\Desktop\测试\hls-svuv-pims-data-analyzer\database\species_database.sqlite"

print(f"加载 pkl 文件: {pkl_path}")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

database = data['database']
print(f"加载了 {len(database)} 个物种")

# 备份旧数据库
backup_path = output_path + f".backup.{int(time.time())}"
if Path(output_path).exists():
    print(f"备份旧数据库到: {backup_path}")
    shutil.copy(output_path, backup_path)
    
    # 尝试删除旧数据库
    try:
        Path(output_path).unlink()
        print(f"删除旧数据库成功")
    except PermissionError:
        print(f"警告：无法删除旧数据库，尝试重命名...")
        output_path = output_path + f".new.{int(time.time())}"

# 创建新数据库
print(f"创建新数据库: {output_path}")
conn = sqlite3.connect(output_path)
cursor = conn.cursor()

# 创建表结构
cursor.execute("""
CREATE TABLE IF NOT EXISTS species (
    id INTEGER PRIMARY KEY,
    mz INTEGER NOT NULL,
    name TEXT NOT NULL,
    ionization_energy REAL,
    formula TEXT,
    elements TEXT,
    smiles TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS pic_cross_sections (
    id INTEGER PRIMARY KEY,
    species_id INTEGER NOT NULL,
    energy_ev REAL NOT NULL,
    cross_section REAL NOT NULL,
    FOREIGN KEY (species_id) REFERENCES species(id) ON DELETE CASCADE
)
""")

cursor.execute("CREATE INDEX IF NOT EXISTS idx_species_mz ON species(mz)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_pics_species_energy ON pic_cross_sections(species_id, energy_ev)")

# 插入数据
print("插入物种数据...")
for species_id, species in enumerate(database, start=1):
    mz = int(species.get('mz', 0))
    name = str(species.get('species', ''))
    ie = species.get('ie')
    smiles = species.get('smiles', '')
    
    # 处理 smiles 字段（可能是列表）
    if isinstance(smiles, list):
        smiles = ','.join(str(s) for s in smiles)
    elif smiles is None:
        smiles = ''
    
    cursor.execute(
        "INSERT INTO species (id, mz, name, ionization_energy, formula, elements, smiles) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (species_id, mz, name, ie, None, None, str(smiles) if smiles else None)
    )
    
    # 插入 PICS 数据
    energies = species.get('energies', [])
    cross_sections = species.get('cross_sections', [])
    
    if isinstance(energies, np.ndarray):
        energies = energies.tolist()
    if isinstance(cross_sections, np.ndarray):
        cross_sections = cross_sections.tolist()
    
    for energy, cross_section in zip(energies, cross_sections):
        cursor.execute(
            "INSERT INTO pic_cross_sections (species_id, energy_ev, cross_section) VALUES (?, ?, ?)",
            (species_id, float(energy), float(cross_section))
        )

conn.commit()

# 验证
cursor.execute("SELECT COUNT(*) FROM species")
species_count = cursor.fetchone()[0]
cursor.execute("SELECT COUNT(*) FROM pic_cross_sections")
pics_count = cursor.fetchone()[0]

print(f"\n=== 转换完成 ===")
print(f"物种数量: {species_count}")
print(f"PICS 数据点数量: {pics_count}")

# 显示一些示例
print(f"\n示例物种:")
cursor.execute("SELECT id, mz, name, smiles FROM species LIMIT 5")
for row in cursor.fetchall():
    print(f"  ID={row[0]}, m/z={row[1]}, name={row[2]}, smiles={row[3]}")

conn.close()
print(f"\n数据库已保存到: {output_path}")

# 如果创建了新的文件，提示用户替换
if ".new." in output_path:
    print(f"\n注意：新数据库已创建在: {output_path}")
    print(f"请手动替换原始数据库文件: c:\\Users\\Administrator\\Desktop\\测试\\hls-svuv-pims-data-analyzer\\database\\species_database.sqlite")
