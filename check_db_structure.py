import sqlite3
import pandas as pd
from pathlib import Path

db_path = Path(r"C:\Users\Administrator\Desktop\测试\hls-svuv-pims-data-analyzer\database\species_database.sqlite")
conn = sqlite3.connect(db_path)

# 查看表结构
print("=== 数据库表 ===")
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print(tables)

print("\n=== species 表结构 ===")
cursor.execute("PRAGMA table_info(species);")
cols = cursor.fetchall()
for col in cols:
    print(col)

print("\n=== 前10条数据 ===")
df = pd.read_sql("SELECT * FROM species LIMIT 10", conn)
print(df)

print("\n=== pic_cross_sections 表结构 ===")
cursor.execute("PRAGMA table_info(pic_cross_sections);")
cols = cursor.fetchall()
for col in cols:
    print(col)

conn.close()
