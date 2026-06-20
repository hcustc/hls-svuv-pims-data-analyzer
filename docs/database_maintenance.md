# SQLite 数据库维护

本项目使用本地 SQLite PICS 数据库进行物种查询和 PIE/PICS 拟合。运行时工作库默认位于：

```text
database/species_database.sqlite
```

默认路径由 `config/app.yaml` 中的 `database.pics` 配置。

可审查的默认数据源随包放在：

```text
src/bl03u_masstool/resources/pics/schema.sql
src/bl03u_masstool/resources/pics/species_seed.csv
```

## Schema 版本管理

数据库通过 SQLite 的 `PRAGMA user_version` 进行版本跟踪。当前版本号定义在：

```python
# src/bl03u_masstool/core/db_migration.py
SCHEMA_VERSION: int = 1
```

每次应用启动时，`species_database_path()` 会自动调用 `ensure_database_up_to_date()`：

- 数据库不存在 → 从 seed 重建，并写入版本号
- `user_version == SCHEMA_VERSION` → 无操作
- `user_version < SCHEMA_VERSION` → 按序执行所有待迁移的 SQL
- `user_version > SCHEMA_VERSION` → 抛出 `RuntimeError`（app 版本过旧）

### 添加新 migration

1. 在 `db_migration.py` 中递增 `SCHEMA_VERSION`
2. 在 `_MIGRATIONS` 字典中添加对应的 SQL：

```python
SCHEMA_VERSION: int = 2

_MIGRATIONS: dict[int, str] = {
    1: "",  # 初始 schema
    2: "ALTER TABLE species ADD COLUMN source TEXT;",
}
```

3. 同步更新 `resources/pics/schema.sql`（作为新建数据库时的完整 schema）

## 表结构

数据库包含两张表：

```sql
CREATE TABLE species (
    id INTEGER PRIMARY KEY,
    mz INTEGER NOT NULL,
    name TEXT NOT NULL,
    ionization_energy REAL,
    formula TEXT,
    elements TEXT,
    smiles TEXT
);

CREATE TABLE pic_cross_sections (
    id INTEGER PRIMARY KEY,
    species_id INTEGER NOT NULL,
    energy_ev REAL NOT NULL,
    cross_section REAL NOT NULL,
    FOREIGN KEY (species_id) REFERENCES species(id) ON DELETE CASCADE
);

CREATE INDEX idx_species_mz ON species(mz);
CREATE INDEX idx_pics_species_energy ON pic_cross_sections(species_id, energy_ev);
```

表结构的权威版本是 `resources/pics/schema.sql`；默认数据的权威版本是 `resources/pics/species_seed.csv`。

## 上传物种模板

用户可通过 Excel 模板批量上传物种数据到数据库：

```text
data/examples/pics_template.xlsx
```

模板列格式与 seed CSV 保持一致，通过 Web 界面上传后后端执行写入。支持三种写入模式：

- `upsert`：替换匹配的物种记录（按 mz + name 匹配）
- `append`：把上传记录追加为新物种条目
- `overwrite_all`：覆盖整个 PICS 库，需要显式确认

## 重建数据库

如果工作库不存在，程序会从 seed 自动生成。也可以显式重建：

```bash
bl03u-build-species-db --output database/species_database.sqlite
# 或
python -m bl03u_masstool.scripts.build_species_database --output database/species_database.sqlite
```

## 更新默认数据

默认数据库变更应优先修改 `species_seed.csv` 或生成该 CSV 的上游清洗流程，再重建 SQLite。这样
Git diff 可读，也方便代码审查。SQLite 文件应视为运行工作库或发布产物，而不是唯一数据源。

## 查询

交互式查询和 CSV 导出可使用：

```bash
python -m bl03u_masstool.scripts.query_species_database
```

常用直接检查：

```bash
sqlite3 database/species_database.sqlite "SELECT COUNT(*) FROM species;"
sqlite3 database/species_database.sqlite "SELECT COUNT(*) FROM pic_cross_sections;"
sqlite3 database/species_database.sqlite "PRAGMA user_version;"
sqlite3 database/species_database.sqlite "PRAGMA integrity_check;"
sqlite3 database/species_database.sqlite "PRAGMA foreign_key_check;"
```

## 备份与恢复

生成的备份文件不要提交到 Git（已由 `.gitignore` 忽略）。

手动维护前先备份：

```bash
cp database/species_database.sqlite database/species_database.backup_$(date +%Y%m%d_%H%M%S).sqlite
```

恢复备份：

```bash
cp database/species_database.backup_YYYYMMDD_HHMMSS.sqlite database/species_database.sqlite
```

服务器部署时，应把 `database/` 放在持久化存储中，并在应用容器外做定期备份。

## 压缩整理

大量删除或整库替换后，可以压缩数据库：

```bash
sqlite3 database/species_database.sqlite "VACUUM;"
sqlite3 database/species_database.sqlite "PRAGMA integrity_check;"
```
