#!/usr/bin/env python3
"""
物种数据库查询和导出工具

功能：
1. 搜索物种
2. 查看物种详细信息
3. 数据导出（CSV）
4. 统计分析
"""

import sqlite3
import csv
from pathlib import Path
from typing import List, Dict, Any


class SpeciesDatabaseQuery:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.conn = None
        self.cursor = None
    
    def connect(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
    
    def close(self):
        if self.conn:
            self.conn.close()
    
    def search_by_mz(self, mz: int) -> List[Dict[str, Any]]:
        """按分子量搜索物种"""
        self.cursor.execute("""
            SELECT id, mz, name, ionization_energy FROM species 
            WHERE mz = ? 
            ORDER BY name
        """, (mz,))
        return [dict(row) for row in self.cursor.fetchall()]
    
    def search_by_name(self, name: str) -> List[Dict[str, Any]]:
        """按物种名搜索（模糊匹配）"""
        self.cursor.execute("""
            SELECT id, mz, name, ionization_energy FROM species 
            WHERE name LIKE ? 
            ORDER BY mz, name
        """, (f"%{name}%",))
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_species_data(self, species_id: int) -> Dict[str, Any]:
        """获取物种的完整数据"""
        self.cursor.execute("""
            SELECT id, mz, name, ionization_energy FROM species WHERE id = ?
        """, (species_id,))
        row = self.cursor.fetchone()
        if not row:
            return None
        
        species = dict(row)
        
        # 获取截面数据
        self.cursor.execute("""
            SELECT energy_ev, cross_section FROM pic_cross_sections 
            WHERE species_id = ? 
            ORDER BY energy_ev
        """, (species_id,))
        species["cross_sections"] = [dict(row) for row in self.cursor.fetchall()]
        
        return species
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        stats = {}
        
        # 基本统计
        self.cursor.execute("SELECT COUNT(*) FROM species")
        stats["total_species"] = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM pic_cross_sections")
        stats["total_cross_sections"] = self.cursor.fetchone()[0]
        
        # m/z分布
        self.cursor.execute("""
            SELECT MIN(mz), MAX(mz), AVG(mz) FROM species
        """)
        min_mz, max_mz, avg_mz = self.cursor.fetchone()
        stats["mz_range"] = {"min": min_mz, "max": max_mz, "avg": avg_mz}
        
        # 能量分布
        self.cursor.execute("""
            SELECT MIN(energy_ev), MAX(energy_ev), AVG(energy_ev) FROM pic_cross_sections
        """)
        min_e, max_e, avg_e = self.cursor.fetchone()
        stats["energy_range"] = {"min": min_e, "max": max_e, "avg": avg_e}
        
        # 截面分布
        self.cursor.execute("""
            SELECT MIN(cross_section), MAX(cross_section), AVG(cross_section) 
            FROM pic_cross_sections
        """)
        min_cs, max_cs, avg_cs = self.cursor.fetchone()
        stats["cross_section_range"] = {"min": min_cs, "max": max_cs, "avg": avg_cs}
        
        return stats
    
    def get_top_species(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取数据点最多的物种"""
        self.cursor.execute("""
            SELECT s.id, s.mz, s.name, COUNT(pc.id) as point_count
            FROM species s
            LEFT JOIN pic_cross_sections pc ON s.id = pc.species_id
            GROUP BY s.id
            ORDER BY point_count DESC
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in self.cursor.fetchall()]
    
    def export_to_csv(self, output_dir: str = "."):
        """导出数据为CSV文件"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 导出species
        species_file = output_dir / "species.csv"
        with open(species_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'mz', 'name', 'ionization_energy'])
            
            self.cursor.execute("SELECT id, mz, name, ionization_energy FROM species ORDER BY mz, name")
            writer.writerows(self.cursor.fetchall())
        
        print(f"✓ 已导出 species.csv ({species_file})")
        
        # 导出cross_sections
        cs_file = output_dir / "cross_sections.csv"
        with open(cs_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['species_id', 'mz', 'name', 'energy_ev', 'cross_section'])
            
            self.cursor.execute("""
                SELECT pc.species_id, s.mz, s.name, pc.energy_ev, pc.cross_section
                FROM pic_cross_sections pc
                JOIN species s ON pc.species_id = s.id
                ORDER BY s.mz, s.name, pc.energy_ev
            """)
            writer.writerows(self.cursor.fetchall())
        
        print(f"✓ 已导出 cross_sections.csv ({cs_file})")
        
        # 导出统计
        stats_file = output_dir / "statistics.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            stats = self.get_statistics()
            f.write("Species Database Statistics\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total species: {stats['total_species']}\n")
            f.write(f"Total cross sections: {stats['total_cross_sections']}\n\n")
            f.write("m/z range:\n")
            f.write(f"  Min: {stats['mz_range']['min']}\n")
            f.write(f"  Max: {stats['mz_range']['max']}\n")
            f.write(f"  Avg: {stats['mz_range']['avg']:.2f}\n\n")
            f.write("Energy range (eV):\n")
            f.write(f"  Min: {stats['energy_range']['min']:.2f}\n")
            f.write(f"  Max: {stats['energy_range']['max']:.2f}\n")
            f.write(f"  Avg: {stats['energy_range']['avg']:.2f}\n\n")
            f.write("Cross section range (Mb):\n")
            f.write(f"  Min: {stats['cross_section_range']['min']:.2e}\n")
            f.write(f"  Max: {stats['cross_section_range']['max']:.2e}\n")
            f.write(f"  Avg: {stats['cross_section_range']['avg']:.2e}\n")
        
        print(f"✓ 已导出 statistics.txt ({stats_file})")


def main():
    """交互式查询程序"""
    db_path = "/Users/huangchen/VscodeProject/BL03U_MSTool_clean_repo/BL03U_MassSpectrumTool/database/species_database.sqlite"
    
    query = SpeciesDatabaseQuery(db_path)
    query.connect()
    
    try:
        print("\n" + "="*60)
        print("物种数据库查询工具")
        print("="*60 + "\n")
        
        while True:
            print("\n【主菜单】")
            print("1. 按m/z搜索物种")
            print("2. 按名称搜索物种")
            print("3. 查看物种详细信息")
            print("4. 数据库统计信息")
            print("5. 显示数据量最多的物种（Top 10）")
            print("6. 导出所有数据为CSV")
            print("7. 退出")
            
            choice = input("\n请选择操作 (1-7): ").strip()
            
            if choice == "1":
                mz = input("请输入m/z值: ").strip()
                if mz.isdigit():
                    results = query.search_by_mz(int(mz))
                    if results:
                        print(f"\n找到 {len(results)} 条记录:")
                        for r in results:
                            ie_str = f"{r['ionization_energy']:.2f}" if r['ionization_energy'] else "N/A"
                            print(f"  ID={r['id']:3d}, m/z={r['mz']:3d}, {r['name']:30s}, IE={ie_str} eV")
                    else:
                        print("未找到记录")
            
            elif choice == "2":
                name = input("请输入物种名称（支持模糊搜索）: ").strip()
                if name:
                    results = query.search_by_name(name)
                    if results:
                        print(f"\n找到 {len(results)} 条记录:")
                        for r in results:
                            ie_str = f"{r['ionization_energy']:.2f}" if r['ionization_energy'] else "N/A"
                            print(f"  ID={r['id']:3d}, m/z={r['mz']:3d}, {r['name']:30s}, IE={ie_str} eV")
                    else:
                        print("未找到记录")
            
            elif choice == "3":
                sp_id = input("请输入物种ID: ").strip()
                if sp_id.isdigit():
                    data = query.get_species_data(int(sp_id))
                    if data:
                        print(f"\n【物种信息】")
                        print(f"ID: {data['id']}")
                        print(f"m/z: {data['mz']}")
                        print(f"名称: {data['name']}")
                        ie_str = f"{data['ionization_energy']:.2f}" if data['ionization_energy'] else "N/A"
                        print(f"离子化能: {ie_str} eV")
                        print(f"\n【截面数据】({len(data['cross_sections'])}个数据点)")
                        print(f"{'能量(eV)':>12} {'截面(Mb)':>12}")
                        print("-" * 26)
                        for cs in data['cross_sections'][:10]:
                            print(f"{cs['energy_ev']:12.2f} {cs['cross_section']:12.4e}")
                        if len(data['cross_sections']) > 10:
                            print(f"... ({len(data['cross_sections']) - 10} more)")
                    else:
                        print("物种不存在")
            
            elif choice == "4":
                stats = query.get_statistics()
                print(f"\n【数据库统计】")
                print(f"物种总数: {stats['total_species']:,}")
                print(f"截面数据点: {stats['total_cross_sections']:,}")
                print(f"\nm/z范围: {stats['mz_range']['min']} - {stats['mz_range']['max']} (平均: {stats['mz_range']['avg']:.1f})")
                print(f"能量范围: {stats['energy_range']['min']:.2f} - {stats['energy_range']['max']:.2f} eV (平均: {stats['energy_range']['avg']:.2f})")
                print(f"截面范围: {stats['cross_section_range']['min']:.2e} - {stats['cross_section_range']['max']:.2e} Mb")
            
            elif choice == "5":
                top = query.get_top_species(10)
                print(f"\n【数据量最多的物种】")
                print(f"{'排名':<5} {'m/z':<8} {'物种名':<30} {'数据点':<8}")
                print("-" * 50)
                for i, item in enumerate(top, 1):
                    print(f"{i:<5} {item['mz']:<8} {item['name']:<30} {item['point_count']:<8}")
            
            elif choice == "6":
                output_dir = "./export"
                query.export_to_csv(output_dir)
                print(f"\n✓ 数据已导出到 {output_dir}")
            
            elif choice == "7":
                print("\n再见！")
                break
            
            else:
                print("无效选择，请重试")
    
    finally:
        query.close()


if __name__ == "__main__":
    main()
