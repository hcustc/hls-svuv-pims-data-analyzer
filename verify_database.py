import sqlite3
import sys
from pathlib import Path

db_path = r"c:\Users\Administrator\Desktop\测试\hls-svuv-pims-data-analyzer\database\species_database.sqlite"

print(f"=== 检查数据库 ===")
print(f"路径: {db_path}")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 检查物种数量
cursor.execute("SELECT COUNT(*) FROM species")
count = cursor.fetchone()[0]
print(f"\n物种总数: {count}")

# 检查PICS数量
cursor.execute("SELECT COUNT(*) FROM pic_cross_sections")
pics_count = cursor.fetchone()[0]
print(f"PICS数据点总数: {pics_count}")

# 显示示例物种
print(f"\n示例物种:")
cursor.execute("SELECT id, mz, name, ionization_energy FROM species LIMIT 20")
for row in cursor.fetchall():
    print(f"  ID={row[0]}, m/z={row[1]}, name={row[2]}, IE={row[3]}")

# 检查m/z分布
cursor.execute("SELECT mz, COUNT(*) FROM species GROUP BY mz ORDER BY mz LIMIT 30")
print(f"\nm/z 分布（前30个）:")
for row in cursor.fetchall():
    print(f"  m/z={row[0]}: {row[1]} 个物种")

conn.close()
