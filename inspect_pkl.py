import pickle
import sys
import numpy
from pathlib import Path

pkl_path = r"E:\质谱峰自动处理小程序\species_database.pkl"

print(f"加载 pkl 文件: {pkl_path}")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print(f"\n=== PKL 文件内容 ===")
print(f"数据库总数: {data.get('total_species', 'N/A')}")

database = data['database']
print(f"\ndatabase 列表长度: {len(database)}")
print(f"database 类型: {type(database)}")

if len(database) > 0:
    # 检查第一个物种的结构
    first_species = database[0]
    print(f"\n第一个物种类型: {type(first_species)}")
    if isinstance(first_species, dict):
        print(f"第一个物种的键: {list(first_species.keys())}")
        print(f"\n第一个物种的示例数据:")
        for key in first_species.keys():
            val = first_species[key]
            if isinstance(val, (list, tuple, numpy.ndarray)):
                print(f"  {key}: {type(val).__name__}, 长度={len(val)}")
                if len(val) > 0:
                    print(f"    示例值: {val[:3]}")
            else:
                print(f"  {key}: {val}")
    
    # 检查几个物种的 mz 值
    print(f"\n前10个物种的 m/z 值:")
    for i, species in enumerate(database[:10]):
        if isinstance(species, dict):
            print(f"  {i+1}. m/z={species.get('mz', 'N/A')}, species={species.get('species', 'N/A')}")
            
    # 检查是否有 formula, elements, smiles 字段
    print(f"\n检查关键字段:")
    has_formula = sum(1 for s in database if isinstance(s, dict) and 'formula' in s and s['formula'])
    has_elements = sum(1 for s in database if isinstance(s, dict) and 'elements' in s and s['elements'])
    has_smiles = sum(1 for s in database if isinstance(s, dict) and 'smiles' in s and s['smiles'])
    print(f"  有 formula 的物种: {has_formula}")
    print(f"  有 elements 的物种: {has_elements}")
    print(f"  有 smiles 的物种: {has_smiles}")
