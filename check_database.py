
import sqlite3
import pandas as pd

db_path = r'C:\Users\Administrator\Desktop\测试\hls-svuv-pims-data-analyzer\database\species_database.sqlite'

print("="*70)
print("检查数据库结构")
print("="*70)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查看表结构
print("\n1. 查看species表结构:")
cursor.execute("PRAGMA table_info(species);")
columns = cursor.fetchall()
for col in columns:
    print(f"   - {col[1]} ({col[2]})")

# 查看数据样本
print("\n2. 查看前20条数据:")
df = pd.read_sql("SELECT mz, name, ionization_energy, formula, elements, smiles FROM species LIMIT 20", conn)
print(df.to_string())

# 统计信息
print("\n3. 数据统计:")
total = pd.read_sql("SELECT COUNT(*) as count FROM species", conn)
print(f"   总记录数: {total['count'].iloc[0]}")

with_formula = pd.read_sql("SELECT COUNT(*) as count FROM species WHERE formula IS NOT NULL AND formula != ''", conn)
print(f"   有分子式的记录: {with_formula['count'].iloc[0]}")

with_elements = pd.read_sql("SELECT COUNT(*) as count FROM species WHERE elements IS NOT NULL AND elements != ''", conn)
print(f"   有元素信息的记录: {with_elements['count'].iloc[0]}")

with_smiles = pd.read_sql("SELECT COUNT(*) as count FROM species WHERE smiles IS NOT NULL AND smiles != ''", conn)
print(f"   有SMILES的记录: {with_smiles['count'].iloc[0]}")

# 查看一些有Cl元素的物种
print("\n4. 含有Cl元素的物种示例:")
df_cl = pd.read_sql("SELECT * FROM species WHERE elements LIKE '%Cl%' LIMIT 10", conn)
if not df_cl.empty:
    print(df_cl[['mz', 'name', 'formula', 'elements']].to_string())
else:
    print("   没有找到含Cl元素的物种！")

# 查看一些有F元素的物种
print("\n5. 含有F元素的物种示例:")
df_f = pd.read_sql("SELECT * FROM species WHERE elements LIKE '%F%' LIMIT 10", conn)
if not df_f.empty:
    print(df_f[['mz', 'name', 'formula', 'elements']].to_string())
else:
    print("   没有找到含F元素的物种！")

conn.close()

print("\n" + "="*70)
print("检查完成")
print("="*70)
