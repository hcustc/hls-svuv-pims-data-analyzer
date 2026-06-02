#!/usr/bin/env python3
"""简化版：将桌面物种数据库Excel转换为SQLite"""

import pandas as pd
import sqlite3
import numpy as np
from pathlib import Path

# 路径配置
excel_path = r"C:\Users\Administrator\Desktop\物种数据库.xlsx"
output_path = r"c:\Users\Administrator\Desktop\测试\hls-svuv-pims-data-analyzer\database\species_database.sqlite"

print(f"读取Excel: {excel_path}")
df = pd.read_excel(excel_path, sheet_name='Sheet1', header=None)
print(f"尺寸: {df.shape}")

# 创建数据库
conn = sqlite3.connect(output_path)
cursor = conn.cursor()

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
    cross_section REAL NOT NULL
)
""")

cursor.execute("CREATE INDEX idx_mz ON species(mz)")

# 简单方法：遍历每一行
species_counter = 1

for row_idx in range(df.shape[0]):
    row = df.iloc[row_idx]
    
    # 跳过空行
    if pd.isna(row.iloc[2]):
        continue
    
    # 获取物种名
    name = row.iloc[2]
    if not isinstance(name, str) or len(name.strip()) == 0:
        continue
    
    # 获取m/z
    mz_val = row.iloc[1] if not pd.isna(row.iloc[1]) else None
    if mz_val is None:
        continue
    
    try:
        mz = int(mz_val)
    except:
        continue
    
    # 获取电离能
    ie = None
    if not pd.isna(row.iloc[4]) and isinstance(row.iloc[4], (int, float)):
        ie = float(row.iloc[4])
    
    # 找能量值行（Energy(eV)标记的行）
    if row_idx + 1 < df.shape[0]:
        next_row = df.iloc[row_idx + 1]
        if isinstance(next_row.iloc[5], str) and 'Energy' in str(next_row.iloc[5]):
            # 下一行是能量值
            energy_row = next_row
            
            # 再下一行是截面数据
            if row_idx + 2 < df.shape[0]:
                data_row = df.iloc[row_idx + 2]
                
                # 提取能量和截面
                energies = []
                cross_sections = []
                
                for col_idx in range(6, df.shape[1]):
                    e_val = energy_row.iloc[col_idx]
                    c_val = data_row.iloc[col_idx]
                    
                    if isinstance(e_val, (int, float)) and not pd.isna(e_val) and e_val > 0:
                        energies.append(float(e_val))
                        
                        if isinstance(c_val, (int, float)) and not pd.isna(c_val):
                            cross_sections.append(float(c_val))
                        else:
                            cross_sections.append(0.0)
                
                # 插入物种
                cursor.execute(
                    "INSERT INTO species (id, mz, name, ionization_energy, formula, elements, smiles) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (species_counter, mz, name, ie, None, None, None)
                )
                
                # 插入PICS数据
                for e, c in zip(energies, cross_sections):
                    cursor.execute(
                        "INSERT INTO pic_cross_sections (species_id, energy_ev, cross_section) VALUES (?, ?, ?)",
                        (species_counter, e, c)
                    )
                
                species_counter += 1

conn.commit()

# 验证
cursor.execute("SELECT COUNT(*) FROM species")
print(f"物种总数: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM pic_cross_sections")
print(f"PICS数据点: {cursor.fetchone()[0]}")

print(f"\n示例物种:")
cursor.execute("SELECT id, mz, name FROM species LIMIT 20")
for row in cursor.fetchall():
    print(f"  {row[0]}: m/z={row[1]}, {row[2]}")

conn.close()
print(f"\n完成! 数据库: {output_path}")
