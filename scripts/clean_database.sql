/**
 * SQLite3 物种数据库清洗脚本
 * 
 * 用法:
 *   sqlite3 species_database.sqlite < clean_database.sql
 * 
 * 或者逐条执行SQL语句
 */

-- ============================================================
-- 【第一步】备份原始数据（可选）
-- ============================================================
-- 如果需要先备份，在shell中执行：
-- cp species_database.sqlite species_database.backup_$(date +%Y%m%d_%H%M%S).sqlite

-- ============================================================
-- 【第二步】分析数据质量
-- ============================================================
-- 查看原始统计
SELECT 'BEFORE CLEANING:' as status;
SELECT COUNT(*) as species_total FROM species;
SELECT COUNT(*) as cross_section_total FROM pic_cross_sections;

-- 检查NULL值
SELECT COUNT(*) as null_ionization_energy FROM species WHERE ionization_energy IS NULL;

-- 检查重复数据
SELECT COUNT(*) as duplicate_groups 
FROM (
    SELECT mz, name FROM species 
    GROUP BY mz, name HAVING COUNT(*) > 1
);

SELECT COUNT(*) as duplicate_records FROM (
    SELECT mz, name FROM species 
    GROUP BY mz, name HAVING COUNT(*) > 1
) t1 
JOIN species s ON s.mz = t1.mz AND s.name = t1.name;

-- 检查负截面
SELECT COUNT(*) as negative_cross_sections FROM pic_cross_sections WHERE cross_section < 0;

-- ============================================================
-- 【第三步】清洗第1步：删除负截面值
-- ============================================================
DELETE FROM pic_cross_sections WHERE cross_section < 0;
SELECT 'Deleted negative cross sections' as status;

-- ============================================================
-- 【第四步】清洗第2步：处理重复物种
-- ============================================================
-- 策略1：保留每个(mz, name)组合中ID最小的记录（即保留最先创建的）
-- 首先标记要删除的ID
CREATE TEMPORARY TABLE duplicate_to_delete AS
SELECT id FROM species
WHERE (mz, name) IN (
    SELECT mz, name FROM species 
    GROUP BY mz, name HAVING COUNT(*) > 1
)
AND id NOT IN (
    SELECT MIN(id) FROM species 
    GROUP BY mz, name
);

-- 删除关联的cross_section记录
DELETE FROM pic_cross_sections 
WHERE species_id IN (SELECT id FROM duplicate_to_delete);

-- 删除重复的species记录
DELETE FROM species 
WHERE id IN (SELECT id FROM duplicate_to_delete);

-- 清理临时表
DROP TABLE duplicate_to_delete;

SELECT 'Deleted duplicate species records' as status;

-- ============================================================
-- 【第五步】验证数据完整性
-- ============================================================
-- 验证外键关系
SELECT COUNT(*) as orphan_cross_sections 
FROM pic_cross_sections 
WHERE species_id NOT IN (SELECT id FROM species);

-- 验证无负截面
SELECT COUNT(*) as remaining_negative 
FROM pic_cross_sections WHERE cross_section < 0;

-- ============================================================
-- 【第六步】清洗后统计
-- ============================================================
SELECT 'AFTER CLEANING:' as status;
SELECT COUNT(*) as species_total FROM species;
SELECT COUNT(*) as cross_section_total FROM pic_cross_sections;

-- 显示仍需手动处理的记录
SELECT 'Records with NULL ionization_energy:' as note;
SELECT id, mz, name FROM species WHERE ionization_energy IS NULL;

-- ============================================================
-- 【附加：数据质量报告】
-- ============================================================

-- 能量分布
SELECT 'Energy statistics:' as note;
SELECT 
    MIN(energy_ev) as min_energy,
    MAX(energy_ev) as max_energy,
    AVG(energy_ev) as avg_energy,
    COUNT(*) as total_points
FROM pic_cross_sections;

-- 截面分布
SELECT 'Cross section statistics:' as note;
SELECT 
    MIN(cross_section) as min_cs,
    MAX(cross_section) as max_cs,
    AVG(cross_section) as avg_cs,
    COUNT(*) as total_points
FROM pic_cross_sections;

-- 每个物种的数据点数
SELECT 'Points per species (top 10):' as note;
SELECT 
    s.mz, 
    s.name, 
    COUNT(pc.id) as point_count
FROM species s
LEFT JOIN pic_cross_sections pc ON s.id = pc.species_id
GROUP BY s.id
ORDER BY point_count DESC
LIMIT 10;

-- ============================================================
-- 【可选：重建索引优化查询性能】
-- ============================================================
REINDEX idx_species_mz;
REINDEX idx_pics_species_energy;
VACUUM;
