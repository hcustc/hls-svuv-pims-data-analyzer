
import os
import re

def extract_energy_from_file(filepath):
    """从文件中提取能量值"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'Energy:' in line:
                    match = re.search(r'Energy:([\d.]+)\s*eV', line)
                    if match:
                        return float(match.group(1))
    except:
        return None
    return None

# PIE文件夹
pie_folder = r'C:\Users\Administrator\Desktop\研究生\邻氯苯酚\2025.03\30torr\邻氯苯酚氧化PIE\O2：C6H5Cl=1\PIE'
pie2_folder = r'C:\Users\Administrator\Desktop\研究生\邻氯苯酚\2025.03\30torr\邻氯苯酚氧化PIE\O2：C6H5Cl=1\PIE-2'

print("=== PIE文件夹能量范围 ===")
energies_pie = []
for filename in os.listdir(pie_folder):
    if filename.endswith('.txt'):
        filepath = os.path.join(pie_folder, filename)
        energy = extract_energy_from_file(filepath)
        if energy:
            energies_pie.append(energy)

if energies_pie:
    print(f"文件数量: {len(energies_pie)}")
    print(f"能量范围: {min(energies_pie):.2f} - {max(energies_pie):.2f} eV")
    print(f"能量点: {sorted(set(energies_pie))}")
else:
    print("未找到能量值")

print("\n=== PIE-2文件夹能量范围 ===")
energies_pie2 = []
for filename in os.listdir(pie2_folder):
    if filename.endswith('.txt'):
        filepath = os.path.join(pie2_folder, filename)
        energy = extract_energy_from_file(filepath)
        if energy:
            energies_pie2.append(energy)

if energies_pie2:
    print(f"文件数量: {len(energies_pie2)}")
    print(f"能量范围: {min(energies_pie2):.2f} - {max(energies_pie2):.2f} eV")
    print(f"能量点: {sorted(set(energies_pie2))}")
else:
    print("未找到能量值")

# 检查重叠
print("\n=== 能量重叠情况 ===")
pie_set = set(energies_pie)
pie2_set = set(energies_pie2)
overlap = pie_set & pie2_set
print(f"PIE能量点: {len(pie_set)}")
print(f"PIE-2能量点: {len(pie2_set)}")
print(f"重叠能量点: {sorted(list(overlap))}")
