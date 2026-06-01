# SQLite 数据库维护

本项目使用本地 SQLite PICS 数据库进行物种查询和 PIE/PICS 拟合。默认维护库为：

```text
database/species_database.sqlite
```

默认路径由 `config/app.yaml` 中的 `database.pics` 配置。

## 表结构

数据库包含两张表：

```sql
CREATE TABLE species (
    id INTEGER PRIMARY KEY,
    mz INTEGER NOT NULL,
    name TEXT NOT NULL,
    ionization_energy REAL
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

`species` 保存候选物种元数据，`pic_cross_sections` 保存每个物种的 PICS 曲线点。

## 查询

交互式查询和 CSV 导出可使用：

```bash
python scripts/query_species_database.py
```

常用直接检查：

```bash
sqlite3 database/species_database.sqlite "SELECT COUNT(*) FROM species;"
sqlite3 database/species_database.sqlite "SELECT COUNT(*) FROM pic_cross_sections;"
sqlite3 database/species_database.sqlite "PRAGMA integrity_check;"
sqlite3 database/species_database.sqlite "PRAGMA foreign_key_check;"
```

## 更新

通过 Web 上传维护 PICS 数据时，优先使用内置上传流程：

- 临时工作库：为当前浏览器会话创建隔离库，不修改服务器维护库。
- 服务器维护库：写入共享维护数据库，需要 `BL03U_ADMIN_TOKEN`。

写入服务器维护库前，后端会在 `database/backups/` 下创建时间戳备份。支持的写入模式：

- `upsert`：替换匹配的物种记录。
- `append`：把上传记录追加为新物种条目。
- `overwrite_all`：覆盖整个 PICS 库，需要显式确认。

如果通过脚本或命令行手动维护，先备份：

```bash
cp database/species_database.sqlite database/species_database.backup_$(date +%Y%m%d_%H%M%S).sqlite
```

只有在明确需要规范化维护库数据时，才运行清洗脚本：

```bash
python scripts/clean_species_database.py
```

## 备份与恢复

生成的备份文件不要提交到 Git；这些路径已由 `.gitignore` 忽略。

恢复备份：

```bash
cp database/species_database.backup_YYYYMMDD_HHMMSS.sqlite database/species_database.sqlite
```

服务器部署时，应把 `database/` 放在持久化存储中，并在应用容器外做定期备份。

## 压缩整理

大量删除或整库替换后，可以压缩数据库：

```bash
sqlite3 database/species_database.sqlite "VACUUM;"
```

压缩后再做完整性检查：

```bash
sqlite3 database/species_database.sqlite "PRAGMA integrity_check;"
sqlite3 database/species_database.sqlite "PRAGMA foreign_key_check;"
```
