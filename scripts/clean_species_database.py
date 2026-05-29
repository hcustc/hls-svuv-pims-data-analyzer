#!/usr/bin/env python3
"""
Species Database Cleaning Script

功能：
1. 检测并修复重复物种记录
2. 清除负截面异常值
3. 处理NULL离子化能
4. 生成清洗报告
"""

import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Any


class SpeciesDatabaseCleaner:
    def __init__(self, db_path: str, backup: bool = True):
        self.db_path = Path(db_path)
        self.conn = None
        self.cursor = None
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "backup_created": False,
            "duplicates_removed": 0,
            "negative_cross_section_removed": 0,
            "null_ie_records": 0,
            "issues_found": [],
            "warnings": [],
        }
        
        if backup:
            self._create_backup()
    
    def _create_backup(self):
        """创建数据库备份"""
        backup_path = self.db_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sqlite")
        shutil.copy2(self.db_path, backup_path)
        self.report["backup_created"] = True
        self.report["backup_path"] = str(backup_path)
        print(f"✓ 备份已创建: {backup_path}")
    
    def connect(self):
        """连接数据库"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        print(f"✓ 已连接数据库: {self.db_path}")
    
    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()
    
    def analyze_data(self) -> Dict[str, Any]:
        """分析数据质量"""
        print("\n" + "="*60)
        print("【数据质量分析】")
        print("="*60)
        
        stats = {}
        
        # 统计基本数据
        self.cursor.execute("SELECT COUNT(*) FROM species")
        species_count = self.cursor.fetchone()[0]
        stats["species_total"] = species_count
        
        self.cursor.execute("SELECT COUNT(*) FROM pic_cross_sections")
        cross_section_count = self.cursor.fetchone()[0]
        stats["cross_section_total"] = cross_section_count
        
        print(f"物种记录: {species_count:,}")
        print(f"截面数据点: {cross_section_count:,}")
        
        # 检查NULL值
        self.cursor.execute("""
            SELECT 
                SUM(CASE WHEN ionization_energy IS NULL THEN 1 ELSE 0 END) as null_ie
            FROM species
        """)
        null_ie = self.cursor.fetchone()[0] or 0
        stats["null_ie"] = null_ie
        self.report["null_ie_records"] = null_ie
        print(f"\n⚠ 缺失离子化能(NULL): {null_ie}")
        if null_ie > 0:
            self.report["issues_found"].append(f"species表有{null_ie}条离子化能为NULL")
        
        # 检查重复
        self.cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT mz, name, COUNT(*) as cnt FROM species 
                GROUP BY mz, name HAVING cnt > 1
            )
        """)
        dup_species = self.cursor.fetchone()[0] or 0
        stats["duplicate_species_groups"] = dup_species
        
        self.cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT mz, name FROM species 
                GROUP BY mz, name HAVING COUNT(*) > 1
            ) t1 
            JOIN species s ON s.mz = t1.mz AND s.name = t1.name
        """)
        dup_count = self.cursor.fetchone()[0] or 0
        stats["duplicate_species_records"] = dup_count
        print(f"⚠ 重复物种记录: {dup_count} (共{dup_species}个物种有重复)")
        if dup_count > 0:
            self.report["issues_found"].append(f"发现{dup_count}条重复物种记录")
        
        # 检查负截面
        self.cursor.execute("""
            SELECT COUNT(*) FROM pic_cross_sections WHERE cross_section < 0
        """)
        negative_cs = self.cursor.fetchone()[0] or 0
        stats["negative_cross_section"] = negative_cs
        print(f"✗ 负截面值: {negative_cs}")
        if negative_cs > 0:
            self.report["issues_found"].append(f"发现{negative_cs}条负截面值")
        
        # 检查孤立cross_section
        self.cursor.execute("""
            SELECT COUNT(*) FROM pic_cross_sections 
            WHERE species_id NOT IN (SELECT id FROM species)
        """)
        orphan_cs = self.cursor.fetchone()[0] or 0
        stats["orphan_cross_sections"] = orphan_cs
        if orphan_cs > 0:
            print(f"✗ 孤立截面记录: {orphan_cs}")
            self.report["warnings"].append(f"发现{orphan_cs}条孤立的截面记录")
        
        # 检查异常能量范围
        self.cursor.execute("""
            SELECT 
                MIN(energy_ev) as min_energy,
                MAX(energy_ev) as max_energy,
                AVG(energy_ev) as avg_energy
            FROM pic_cross_sections
        """)
        min_e, max_e, avg_e = self.cursor.fetchone()
        stats["energy_range"] = (min_e, max_e, avg_e)
        print(f"\n能量范围: {min_e:.2f} - {max_e:.2f} eV (平均: {avg_e:.2f})")
        
        self.cursor.execute("""
            SELECT 
                MIN(cross_section) as min_cs,
                MAX(cross_section) as max_cs,
                AVG(cross_section) as avg_cs
            FROM pic_cross_sections WHERE cross_section >= 0
        """)
        min_cs, max_cs, avg_cs = self.cursor.fetchone()
        stats["cross_section_range"] = (min_cs, max_cs, avg_cs)
        print(f"截面范围: {min_cs:.2e} - {max_cs:.2e} (平均: {avg_cs:.2e})")
        
        return stats
    
    def remove_negative_cross_sections(self) -> int:
        """移除负截面值"""
        print("\n" + "="*60)
        print("【清洗1: 移除负截面值】")
        print("="*60)
        
        self.cursor.execute("""
            SELECT COUNT(*) FROM pic_cross_sections WHERE cross_section < 0
        """)
        count = self.cursor.fetchone()[0] or 0
        
        if count > 0:
            self.cursor.execute("""
                DELETE FROM pic_cross_sections WHERE cross_section < 0
            """)
            self.conn.commit()
            self.report["negative_cross_section_removed"] = count
            print(f"✓ 已删除 {count} 条负截面记录")
        else:
            print("✓ 无负截面值需要清洗")
        
        return count
    
    def handle_duplicate_species(self, strategy: str = "keep_first") -> int:
        """
        处理重复物种记录
        
        策略:
        - keep_first: 保留第一条，删除其他（推荐用于相同数据）
        - keep_best: 保留离子化能不为NULL的版本
        - merge: 合并多条记录为一条（保留最多cross_section数据的版本）
        """
        print("\n" + "="*60)
        print(f"【清洗2: 处理重复物种记录 (策略: {strategy})】")
        print("="*60)
        
        if strategy == "keep_first":
            # 获取重复的species_id
            self.cursor.execute("""
                SELECT id, mz, name FROM species
                WHERE (mz, name) IN (
                    SELECT mz, name FROM species 
                    GROUP BY mz, name HAVING COUNT(*) > 1
                )
                ORDER BY mz, name, id
            """)
            rows = self.cursor.fetchall()
            
            to_delete = []
            current_group = None
            
            for row_id, mz, name in rows:
                group_key = (mz, name)
                if group_key != current_group:
                    current_group = group_key
                else:
                    to_delete.append(row_id)
            
            if to_delete:
                # 先处理外键关系
                for sp_id in to_delete:
                    self.cursor.execute("""
                        DELETE FROM pic_cross_sections WHERE species_id = ?
                    """, (sp_id,))
                
                # 删除重复的species记录
                for sp_id in to_delete:
                    self.cursor.execute("DELETE FROM species WHERE id = ?", (sp_id,))
                
                self.conn.commit()
                self.report["duplicates_removed"] = len(to_delete)
                print(f"✓ 已删除 {len(to_delete)} 条重复物种记录（及其{sum(1 for _ in to_delete)}个关联截面数据）")
            else:
                print("✓ 无重复物种记录")
        
        elif strategy == "keep_best":
            # 保留有离子化能的版本
            self.cursor.execute("""
                SELECT mz, name FROM species
                GROUP BY mz, name HAVING COUNT(*) > 1
            """)
            dup_groups = self.cursor.fetchall()
            
            total_deleted = 0
            for mz, name in dup_groups:
                self.cursor.execute("""
                    SELECT id FROM species 
                    WHERE mz = ? AND name = ?
                    ORDER BY ionization_energy DESC, id ASC
                """, (mz, name))
                ids = [row[0] for row in self.cursor.fetchall()]
                
                # 保留第一条（最优的），删除其他
                for sp_id in ids[1:]:
                    self.cursor.execute("DELETE FROM pic_cross_sections WHERE species_id = ?", (sp_id,))
                    self.cursor.execute("DELETE FROM species WHERE id = ?", (sp_id,))
                    total_deleted += 1
            
            self.conn.commit()
            self.report["duplicates_removed"] = total_deleted
            print(f"✓ 已删除 {total_deleted} 条重复物种记录")
        
        return self.report["duplicates_removed"]
    
    def fill_missing_ionization_energy(self, fill_value: float = None) -> int:
        """
        处理缺失的离子化能
        
        选项:
        - fill_value: 用固定值填充
        - None: 显示需要手动处理的记录
        """
        print("\n" + "="*60)
        print("【清洗3: 处理缺失的离子化能】")
        print("="*60)
        
        self.cursor.execute("""
            SELECT id, mz, name FROM species WHERE ionization_energy IS NULL
        """)
        missing = self.cursor.fetchall()
        
        if not missing:
            print("✓ 无缺失的离子化能记录")
            return 0
        
        print(f"\n找到 {len(missing)} 条缺失离子化能的记录:")
        for sp_id, mz, name in missing:
            print(f"  ID={sp_id}, m/z={mz}, 名称={name}")
        
        if fill_value is not None:
            self.cursor.execute("""
                UPDATE species SET ionization_energy = ? 
                WHERE ionization_energy IS NULL
            """, (fill_value,))
            self.conn.commit()
            print(f"\n✓ 已用 {fill_value} eV 填充 {len(missing)} 条记录")
        else:
            print("\n⚠ 建议手动补充这些数据或参考权威数据库（如NIST）")
        
        return len(missing)
    
    def validate_after_cleaning(self) -> bool:
        """清洗后验证"""
        print("\n" + "="*60)
        print("【清洗后验证】")
        print("="*60)
        
        # 检查referential integrity
        self.cursor.execute("""
            SELECT COUNT(*) FROM pic_cross_sections 
            WHERE species_id NOT IN (SELECT id FROM species)
        """)
        orphan = self.cursor.fetchone()[0] or 0
        
        if orphan > 0:
            print(f"✗ 发现 {orphan} 条孤立的截面记录")
            return False
        
        print("✓ 外键关系完整")
        
        # 检查负截面
        self.cursor.execute("SELECT COUNT(*) FROM pic_cross_sections WHERE cross_section < 0")
        negative = self.cursor.fetchone()[0] or 0
        
        if negative > 0:
            print(f"✗ 仍有 {negative} 条负截面值")
            return False
        
        print("✓ 无负截面值")
        
        # 最终统计
        self.cursor.execute("SELECT COUNT(*) FROM species")
        species = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM pic_cross_sections")
        cross_sections = self.cursor.fetchone()[0]
        
        print(f"\n✓ 清洗完成!")
        print(f"  最终物种记录: {species:,}")
        print(f"  最终截面数据点: {cross_sections:,}")
        
        return True
    
    def generate_report(self) -> str:
        """生成清洗报告"""
        report_lines = [
            "\n" + "="*60,
            "【数据清洗报告】",
            "="*60,
            f"时间: {self.report['timestamp']}",
            f"数据库: {self.db_path}",
            f"备份: {self.report.get('backup_path', 'N/A')}",
            "",
            "【检测到的问题】:",
        ]
        
        if self.report["issues_found"]:
            for issue in self.report["issues_found"]:
                report_lines.append(f"  - {issue}")
        else:
            report_lines.append("  无重大问题")
        
        report_lines.extend([
            "",
            "【执行的清洗操作】:",
            f"  - 移除负截面值: {self.report['negative_cross_section_removed']} 条",
            f"  - 移除重复物种: {self.report['duplicates_removed']} 条",
            f"  - 处理缺失离子化能: {self.report['null_ie_records']} 条",
        ])
        
        if self.report["warnings"]:
            report_lines.append("\n【警告】:")
            for warning in self.report["warnings"]:
                report_lines.append(f"  ⚠ {warning}")
        
        report_lines.append("\n" + "="*60)
        
        return "\n".join(report_lines)


def main():
    """主清洗流程"""
    db_path = "/Users/huangchen/VscodeProject/BL03U_MSTool_clean_repo/BL03U_MassSpectrumTool/database/species_database.sqlite"
    
    cleaner = SpeciesDatabaseCleaner(db_path, backup=True)
    
    try:
        cleaner.connect()
        
        # 分析
        cleaner.analyze_data()
        
        # 清洗
        cleaner.remove_negative_cross_sections()
        cleaner.handle_duplicate_species(strategy="keep_best")
        cleaner.fill_missing_ionization_energy(fill_value=None)  # 改为keep以手动处理
        
        # 验证
        cleaner.validate_after_cleaning()
        
        # 报告
        report = cleaner.generate_report()
        print(report)
        
        # 保存报告
        report_path = Path(db_path).parent / f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path.write_text(report)
        print(f"\n✓ 报告已保存: {report_path}")
    
    finally:
        cleaner.close()


if __name__ == "__main__":
    main()
