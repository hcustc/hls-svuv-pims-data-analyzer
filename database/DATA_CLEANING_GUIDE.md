# Species Database 数据清洗指南

## 📋 数据库概述

### 表结构

```
species (物种表)
├── id: INTEGER PRIMARY KEY
├── mz: INTEGER NOT NULL (分子量)
├── name: TEXT NOT NULL (物种名称)
└── ionization_energy: REAL (可为NULL) (离子化能 eV)

pic_cross_sections (截面表)
├── id: INTEGER PRIMARY KEY
├── species_id: INTEGER FOREIGN KEY
├── energy_ev: REAL NOT NULL (光子能量 eV)
└── cross_section: REAL NOT NULL (截面 Mb)
```

### 数据规模

| 指标 | 清洗前 | 清洗后 | 变化 |
|------|--------|--------|------|
| 物种记录 | 587 | 394 | -193 (-32.9%) |
| 截面数据点 | 46,015 | 24,457 | -21,558 (-46.9%) |

---

## 🔍 检测到的数据问题

### 1. **负截面值（49条）** ⚠️ 高优先级
- **问题**: 物理上不可能的负截面值
- **原因**: 数据导入错误或符号转换错误
- **解决**: 直接删除所有负值
- **影响**: 清洗前后对比
  ```
  DELETE FROM pic_cross_sections WHERE cross_section < 0;
  -- 删除49条记录
  ```

### 2. **重复物种记录（284条）** ⚠️ 高优先级
- **问题**: 同一分子量(mz)和物种名(name)出现多条记录
- **范例**:
  ```
  m/z=2, name=Hydrogen, 出现2次
  m/z=4, name=Helium, 出现6次
  m/z=28, name=Nitrogen, 出现3次
  m/z=78, name=Benzene, 出现6次
  ```
- **原因**: 数据导入过程中重复或测量条件不同
- **解决**: 保留每组最优的记录（有效的离子化能）
- **影响**: 删除193条重复记录（覆盖约21,558个截面点）

### 3. **缺失离子化能（3条）** ⚠️ 中优先级
- **问题**: 3条物种记录的ionization_energy为NULL
- **受影响的记录**:
  ```
  ID=79, m/z=30, 名称=Hydroxymethylene
  ID=165, m/z=45, 名称=2-Hydroxyethyl radical
  ID=260, m/z=60, 名称=乙酸
  ```
- **解决**: 需要手动查阅权威数据库（如NIST）补充数据

### 4. **能量异常（1条）** ⚠️ 低优先级
- **问题**: 能量范围极大（1.5 - 62000 eV）
- **异常点**: energy_ev=62000是否有效需人工判断
- **建议**: 检查单位和测量条件

---

## 🛠️ 清洗方法

### 方法1️⃣：Python脚本自动清洗（推荐）

**优点**：
- 自动备份原始数据
- 详细的清洗报告
- 可自定义清洗策略
- 清洗后自动验证

**操作步骤**：

```bash
# 进入项目目录
cd <repository-root>

# 运行清洗脚本
python scripts/clean_species_database.py
```

**脚本会执行**：
1. ✅ 自动备份原始数据库（带时间戳）
2. ✅ 分析数据质量问题
3. ✅ 删除负截面值（49条）
4. ✅ 删除重复物种记录（193条）
5. ✅ 列出缺失离子化能的记录
6. ✅ 验证外键完整性
7. ✅ 生成详细报告

**输出示例**：
```
✓ 备份已创建: species_database.backup_20260516_002533.sqlite
✓ 已删除 49 条负截面记录
✓ 已删除 193 条重复物种记录
✓ 清洗完成!
  最终物种记录: 394
  最终截面数据点: 24,457
✓ 报告已保存: database/cleaning_report_20260516_002533.txt
```

---

### 方法2️⃣：SQL脚本手动清洗

**优点**：
- 完全控制每一步
- 易于审计和验证

**操作步骤**：

```bash
# 先备份
cp database/species_database.sqlite database/species_database.backup_$(date +%Y%m%d_%H%M%S).sqlite

# 运行SQL脚本
sqlite3 database/species_database.sqlite < scripts/clean_database.sql
```

**或者逐步运行**：

```bash
# 进入SQLite交互模式
sqlite3 database/species_database.sqlite

# 复制粘贴 clean_database.sql 中的SQL语句分步执行
```

---

### 方法3️⃣：交互式Python清洗

**适用场景**：需要自定义清洗策略

```python
from scripts.clean_species_database import SpeciesDatabaseCleaner

# 初始化清洗器（自动备份）
cleaner = SpeciesDatabaseCleaner(
    db_path="database/species_database.sqlite",
    backup=True
)

try:
    cleaner.connect()
    
    # 分析
    stats = cleaner.analyze_data()
    
    # 清洗选项1：删除负截面
    cleaner.remove_negative_cross_sections()
    
    # 清洗选项2：处理重复（3种策略）
    # "keep_first" - 保留第一条
    # "keep_best" - 保留有离子化能的版本
    # "merge" - 合并多条（保留最多数据的版本）
    cleaner.handle_duplicate_species(strategy="keep_best")
    
    # 清洗选项3：处理缺失离子化能
    # None - 只显示，不修改
    # 9.5 - 用9.5 eV填充
    cleaner.fill_missing_ionization_energy(fill_value=None)
    
    # 验证
    cleaner.validate_after_cleaning()
    
    # 生成报告
    print(cleaner.generate_report())

finally:
    cleaner.close()
```

---

## 📊 清洗前后对比

### 统计数据

```
【清洗前】
- 物种记录: 587
- 截面数据点: 46,015
- 重复物种: 91组（284条记录）
- 负截面: 49条
- 缺失离子化能: 3条

【清洗后】
- 物种记录: 394 (-193, -32.9%)
- 截面数据点: 24,457 (-21,558, -46.9%)
- 重复物种: 0组
- 负截面: 0条
- 缺失离子化能: 3条（需手动补充）
```

### 数据质量提升

| 指标 | 改善 |
|------|------|
| 数据完整性 | ✅ 外键关系完整（0条孤立记录） |
| 数据有效性 | ✅ 无负截面值 |
| 数据唯一性 | ✅ 消除重复 |
| 数据准确性 | ⚠️ 需手动补充3条缺失离子化能 |

---

## ✋ 手动处理缺失离子化能

需要查阅权威数据库补充的3条记录：

| ID | m/z | 名称 | 离子化能(eV) | 数据来源 |
|----|-----|------|-------------|---------|
| 79 | 30 | Hydroxymethylene | ? | NIST / CRC Handbook |
| 165 | 45 | 2-Hydroxyethyl radical | ? | NIST / CRC Handbook |
| 260 | 60 | 乙酸 | ? | NIST / CRC Handbook |

**补充方法**（以ID=79为例）：

```sql
-- 方法1：直接更新
UPDATE species SET ionization_energy = 10.5 WHERE id = 79;

-- 方法2：使用脚本
cleaner.fill_missing_ionization_energy(fill_value=10.5)
```

---

## 🔄 恢复原始数据

如果清洗出现问题，可以恢复备份：

```bash
# 查看备份文件
ls -lh database/species_database.backup_*.sqlite

# 恢复备份
cp database/species_database.backup_20260516_002533.sqlite database/species_database.sqlite

# 或删除清洗后的数据库，重新运行清洗
rm database/species_database.sqlite
cp database/species_database.backup_20260516_002533.sqlite database/species_database.sqlite
```

---

## 📈 清洗后数据特征

### 能量分布
```
最小: 1.50 eV
最大: 1000.00 eV
平均: 58.15 eV
```

### 截面分布
```
最小: 0.00 Mb
最大: 1430.00 Mb
平均: 20.70 Mb
```

### 物种类型分布（前10）
```
1. m/z=2, Hydrogen: 66个数据点
2. m/z=4, Helium: 58个数据点
3. m/z=12, Carbon: 52个数据点
...
```

---

## ⚙️ 高级清洗选项

### 自定义SQL查询示例

**查找数据异常**：
```sql
-- 查找能量间距过大的数据点
SELECT species_id, energy_ev, 
       LAG(energy_ev) OVER (PARTITION BY species_id ORDER BY energy_ev) as prev_energy,
       energy_ev - LAG(energy_ev) OVER (PARTITION BY species_id ORDER BY energy_ev) as gap
FROM pic_cross_sections
WHERE gap > 100
ORDER BY gap DESC;

-- 查找截面变化异常的数据
SELECT species_id, 
       cross_section,
       LAG(cross_section) OVER (PARTITION BY species_id ORDER BY energy_ev) as prev_cs,
       ABS(cross_section - LAG(cross_section) OVER (PARTITION BY species_id ORDER BY energy_ev)) as change
FROM pic_cross_sections
WHERE change > 1000
ORDER BY change DESC;
```

### 导出清洗后的数据

```bash
# 导出为CSV
sqlite3 database/species_database.sqlite \
  ".mode csv" \
  ".output export_species.csv" \
  "SELECT * FROM species;" \
  ".output export_cross_sections.csv" \
  "SELECT pc.*, s.name, s.mz FROM pic_cross_sections pc JOIN species s ON pc.species_id = s.id;"
```

---

## 📝 清洗报告位置

- **数据库备份**: `database/species_database.backup_*.sqlite`
- **清洗报告**: `database/cleaning_report_*.txt`
- **SQL脚本**: `scripts/clean_database.sql`
- **Python脚本**: `scripts/clean_species_database.py`

---

## ❓ FAQ

**Q: 清洗后是否影响现有代码？**
A: 否。数据库结构保持不变，只是记录更清洁。现有查询代码无需修改。

**Q: 能否只清洗负截面，保留重复数据？**
A: 可以。在Python脚本中注释掉`handle_duplicate_species()`调用即可。

**Q: 清洗是否可逆？**
A: 是的。每次运行都会自动创建备份。使用`cp`命令可恢复。

**Q: 如何验证清洗结果？**
A: 脚本自动运行验证。查看"清洗后验证"部分的输出确认无问题。

---

## 🚀 建议后续步骤

1. ✅ 运行清洗脚本（Python或SQL）
2. ✅ 查看清洗报告
3. ✅ 手动补充3条缺失离子化能
4. ✅ 使用`VACUUM`重建索引优化性能
5. ✅ 定期备份清洗后的数据库

---

**最后更新**: 2026-05-16  
**版本**: 1.0
