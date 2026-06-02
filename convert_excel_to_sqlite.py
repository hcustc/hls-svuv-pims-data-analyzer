#!/usr/bin/env python3
"""将桌面上的物种数据库Excel转换为SQLite格式"""

import pandas as pd
import sqlite3
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import math

# 路径配置
excel_path = r"C:\Users\Administrator\Desktop\物种数据库.xlsx"
output_path = r"c:\Users\Administrator\Desktop\测试\hls-svuv-pims-data-analyzer\database\species_database.sqlite"

print(f"=== 读取Excel文件 ===")
print(f"源文件: {excel_path}")

# 读取Excel
df = pd.read_excel(excel_path, sheet_name='Sheet1', header=None)
print(f"Excel尺寸: {df.shape[0]} 行 x {df.shape[1]} 列")

# 找到能量值的起始列
energy_start_col = None
energy_values = []

for col_idx in range(df.shape[1]):
    val = df.iloc[0, col_idx]
    if isinstance(val, (int, float)) and 5 < val < 30:
        energy_start_col = col_idx
        # 收集所有能量值
        for row_idx in range(df.shape[0]):
            e_val = df.iloc[row_idx, col_idx]
            if isinstance(e_val, (int, float)) and e_val > 0:
                energy_values.append(e_val)
        break

if energy_start_col is None:
    energy_start_col = 6
    energy_values = [df.iloc[0, i] for i in range(energy_start_col, min(energy_start_col + 100, df.shape[1])) 
                     if isinstance(df.iloc[0, i], (int, float))]

print(f"能量值起始列: {energy_start_col}")
print(f"能量值数量: {len(energy_values)}")

# 解析物种数据
species_list = []
current_species = None

for row_idx in range(df.shape[0]):
    row = df.iloc[row_idx]
    
    # 尝试获取物种信息
    species_name = None
    mz = None
    ie = None
    
    # 从多个可能的位置提取信息
    for col_idx in range(min(6, df.shape[1])):
        val = row.iloc[col_idx]
        
        if col_idx == 0:  # ID列
            if isinstance(val, int):
                if current_species is None or current_species.get('id') != val:
                    if current_species:
                        species_list.append(current_species)
                    current_species = {'id': val, 'rows': []}
        
        elif col_idx == 1:  # m/z 列
            if isinstance(val, (int, float)) and not pd.isna(val):
                mz = int(val)
                if current_species and current_species.get('mz') is None:
                    current_species['mz'] = mz
        
        elif col_idx == 2:  # 物种名称列
            if isinstance(val, str) and val != 'H' and len(val) > 0 and val.strip():
                species_name = val
                if current_species and current_species.get('name') is None:
                    current_species['name'] = species_name
        
        elif col_idx == 3:  # 另一个ID
            if isinstance(val, (int, float)) and not pd.isna(val):
                if current_species and current_species.get('group_id') is None:
                    current_species['group_id'] = int(val)
        
        elif col_idx == 4:  # 电离能列
            if isinstance(val, (int, float)) and not pd.isna(val) and 5 < val < 25:
                ie = float(val)
                if current_species and current_species.get('ie') is None:
                    current_species['ie'] = ie
    
    # 如果找到了有效的m/z和物种名，并且当前行有数据
    if current_species and current_species.get('mz') and current_species.get('name'):
        # 检查是否有PICS数据（能量值后面的列）
        has_pics = False
        cross_sections = []
        
        for col_idx in range(energy_start_col, df.shape[1]):
            val = row.iloc[col_idx]
            if isinstance(val, (int, float)) and not pd.isna(val):
                cross_sections.append(float(val))
                has_pics = True
        
        if has_pics and len(cross_sections) > 0:
            current_species['energies'] = energy_values[:len(cross_sections)]
            current_species['cross_sections'] = cross_sections

# 添加最后一个物种
if current_species:
    species_list.append(current_species)

print(f"\n=== 解析完成 ===")
print(f"找到 {len(species_list)} 个物种")

# 创建SQLite数据库
print(f"\n=== 创建数据库 ===")

# 备份旧数据库
backup_path = output_path + ".backup"
if Path(output_path).exists():
    try:
        Path(output_path).rename(backup_path)
        print(f"旧数据库已备份到: {backup_path}")
    except:
        pass

# 创建新数据库
conn = sqlite3.connect(output_path)
cursor = conn.cursor()

# 创建表
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
species_id = 1
for sp in species_list:
    if not sp.get('mz') or not sp.get('name'):
        continue
    
    cursor.execute(
        "INSERT INTO species (id, mz, name, ionization_energy, formula, elements, smiles) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (species_id, sp['mz'], sp['name'], sp.get('ie'), None, None, None)
    )
    
    energies = sp.get('energies', [])
    cross_sections = sp.get('cross_sections', [])
    
    for energy, cross_section in zip(energies, cross_sections):
        cursor.execute(
            "INSERT INTO pic_cross_sections (species_id, energy_ev, cross_section) VALUES (?, ?, ?)",
            (species_id, float(energy), float(cross_section))
        )
    
    species_id += 1

conn.commit()

# 验证
cursor.execute("SELECT COUNT(*) FROM species")
count = cursor.fetchone()[0]
print(f"插入 {count} 个物种")

cursor.execute("SELECT COUNT(*) FROM pic_cross_sections")
pics_count = cursor.fetchone()[0]
print(f"插入 {pics_count} 个PICS数据点")

# 显示示例
print(f"\n示例物种:")
cursor.execute("SELECT id, mz, name, ionization_energy FROM species LIMIT 10")
for row in cursor.fetchall():
    print(f"  ID={row[0]}, m/z={row[1]}, name={row[2]}, IE={row[3]}")

conn.close()

print(f"\n=== 转换完成 ===")
print(f"数据库已保存到: {output_path}")
print(f"\n注意: 此数据库需要补充SMILES和分子式数据")
print(f"建议使用RDKit或其他工具根据物种名称自动生成或手动添加")
