import pandas as pd
import sqlite3
import numpy as np
from core.elements import parse_formula, get_elements_from_formula

# 读取参考Excel数据库
print("正在读取参考Excel数据库...")
ref_file = r"D:\TRAE test\物种数据库.xlsx"
df_ref = pd.read_excel(ref_file, header=None)

# 解析参考数据库
print("正在解析参考数据库...")
ref_species_data = {}

i = 0
n_rows, n_cols = df_ref.shape

while i < n_rows - 1:
    try:
        col5_val = df_ref.iloc[i, 5] if pd.notna(df_ref.iloc[i, 5]) else ""
        if str(col5_val).strip() == "Energy(eV)":
            header_row = i
            data_row = i + 1
            
            # 读取基本信息
            mz_raw = df_ref.iloc[header_row, 0]
            if pd.isna(mz_raw):
                i += 1
                continue
            try:
                mz = int(round(float(str(mz_raw).strip())))
            except ValueError:
                i += 1
                continue
            
            species_symbol = str(df_ref.iloc[header_row, 2]).strip() if pd.notna(df_ref.iloc[header_row, 2]) else ""
            
            ie_raw = df_ref.iloc[header_row, 4]
            ie = None
            if pd.notna(ie_raw):
                try:
                    ie = float(str(ie_raw).strip())
                except ValueError:
                    pass
            
            # 读取物种名称
            species_name = str(df_ref.iloc[data_row, 2]).strip() if pd.notna(df_ref.iloc[data_row, 2]) else ""
            if not species_name or species_name == "nan":
                species_name = species_symbol
            
            # 用species_symbol作为分子式
            formula = species_symbol
            
            # 解析元素
            elements = get_elements_from_formula(formula) if formula else set()
            
            key = (mz, species_name, ie if ie else 0)
            ref_species_data[key] = {
                'formula': formula,
                'elements': ','.join(sorted(elements)) if elements else '',
                'smiles': ''  # SMILES暂时留空，后续可补充
            }
            
            i += 2
        else:
            i += 1
            
    except Exception as e:
        print(f"处理第{i}行时出错: {e}")
        i += 1
        continue

print(f"解析到 {len(ref_species_data)} 个物种")

# 连接现有数据库
db_path = r"C:\Users\Administrator\Desktop\测试\hls-svuv-pims-data-analyzer\database\species_database.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 添加新列
print("\n正在添加新列...")
try:
    cursor.execute("ALTER TABLE species ADD COLUMN formula TEXT")
    print("已添加formula列")
except sqlite3.OperationalError:
    print("formula列已存在")

try:
    cursor.execute("ALTER TABLE species ADD COLUMN elements TEXT")
    print("已添加elements列")
except sqlite3.OperationalError:
    print("elements列已存在")

try:
    cursor.execute("ALTER TABLE species ADD COLUMN smiles TEXT")
    print("已添加smiles列")
except sqlite3.OperationalError:
    print("smiles列已存在")

# 读取现有数据
df_old = pd.read_sql("SELECT * FROM species", conn)
print(f"\n现有数据库有 {len(df_old)} 条记录")

# 更新数据
updated_count = 0
for idx, row in df_old.iterrows():
    mz = row['mz']
    name = row['name']
    ie = row['ionization_energy'] if pd.notna(row['ionization_energy']) else 0
    
    key = (mz, name, ie)
    
    # 尝试匹配
    formula = None
    elements = None
    
    if key in ref_species_data:
        data = ref_species_data[key]
        formula = data['formula']
        elements = data['elements']
    else:
        # 尝试模糊匹配
        for (ref_mz, ref_name, ref_ie), data in ref_species_data.items():
            if ref_mz == mz and ref_name.lower() == name.lower():
                formula = data['formula']
                elements = data['elements']
                break
    
    # 如果还是找不到，尝试从name中解析
    if not formula:
        formula = name
        elements = ','.join(sorted(get_elements_from_formula(formula))) if formula else ''
    
    # 更新数据库
    cursor.execute("""
        UPDATE species 
        SET formula = ?, elements = ?
        WHERE id = ?
    """, (formula, elements, row['id']))
    updated_count += 1

conn.commit()
print(f"\n已更新 {updated_count} 条记录")

# 显示更新结果
df_new = pd.read_sql("SELECT * FROM species", conn)
print("\n=== 更新后的前20条记录 ===")
print(df_new[['id', 'mz', 'name', 'formula', 'elements', 'ionization_energy']].head(20))

conn.close()
print("\n数据库更新完成！")
