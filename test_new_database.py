import sqlite3
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

db_path = r"c:\Users\Administrator\Desktop\测试\hls-svuv-pims-data-analyzer\database\species_database.sqlite"

print(f"=== 测试数据库加载 ===")
print(f"数据库路径: {db_path}")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 检查物种数量
cursor.execute("SELECT COUNT(*) FROM species")
species_count = cursor.fetchone()[0]
print(f"物种总数: {species_count}")

# 检查 PICS 数据点数量
cursor.execute("SELECT COUNT(*) FROM pic_cross_sections")
pics_count = cursor.fetchone()[0]
print(f"PICS 数据点总数: {pics_count}")

# 检查有 smiles 的物种数量
cursor.execute("SELECT COUNT(*) FROM species WHERE smiles IS NOT NULL AND smiles != ''")
with_smiles = cursor.fetchone()[0]
print(f"有 smiles 的物种: {with_smiles}")

# 显示一些示例
print(f"\n示例物种:")
cursor.execute("SELECT mz, name, ionization_energy, smiles FROM species LIMIT 10")
for row in cursor.fetchall():
    print(f"  m/z={row[0]}, name={row[1]}, IE={row[2]}, SMILES={row[3]}")

# 测试加载到应用程序的数据结构
print(f"\n=== 测试应用程序数据结构加载 ===")
from core.pie_analysis import load_species_database

try:
    database, mz_index = load_species_database(db_path)
    print(f"成功加载 {len(database)} 个物种")
    print(f"m/z 索引数量: {len(mz_index)}")
    
    # 检查第一个物种
    if database:
        first = database[0]
        print(f"\n第一个物种:")
        print(f"  m/z: {first['mz']}")
        print(f"  species: {first['species']}")
        print(f"  ie: {first.get('ie')}")
        print(f"  smiles: {first.get('smiles')}")
        print(f"  energies 数量: {len(first.get('energies', []))}")
        print(f"  cross_sections 数量: {len(first.get('cross_sections', []))}")
    
    print(f"\n✅ 数据库格式正确，可以被应用程序使用！")
    
except Exception as e:
    print(f"\n❌ 加载失败: {e}")
    import traceback
    traceback.print_exc()

conn.close()
