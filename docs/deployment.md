# BL03U MassSpectrumTool 部署说明

本文档面向实验站内网或受控公网部署。公网开放前应先确认数据目录、管理员 token、上传限制和备份策略。

如果你准备使用 Docker / 阿里云轻量应用服务器（容器环境），请优先查看 [Docker 部署方案](docker_deployment.md)。

## 环境

项目统一使用 Python 3.12（见仓库根目录 `.python-version`）和 `requirements*.lock` 锁定依赖。推荐安装方式：

```bash
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps -e .
python -m pytest
```

若必须使用 conda，可只让 conda 提供 Python 运行时，再由 pip 按锁文件同步依赖：

```bash
conda create -n bl03u python=3.12
conda activate bl03u
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

`requirements.txt` 和 `requirements-dev.txt` 是薄入口，真实版本锁定在 `requirements.lock` 和 `requirements-dev.lock`；CI/Docker/发布检查均使用这些锁文件。


## 启动 Web 服务

```bash
uvicorn bl03u_masstool.api.app:app --host 127.0.0.1 --port 8000
```

浏览器访问：

```text
http://127.0.0.1:8000/
```

如果部署到服务器反向代理后面，建议只让反向代理暴露 HTTPS，`uvicorn` 继续监听内网地址。

## 关键环境变量

```bash
export BL03U_ADMIN_TOKEN='change-this-token'
export BL03U_MAX_UPLOAD_MB=20
export BL03U_PICS_LIBRARY_TTL_HOURS=24
export BL03U_ALLOWED_DATA_ROOTS='/data/bl03u:/data/shared'
export BL03U_PIE_JOB_TTL_HOURS=6
export BL03U_MAX_PIE_JOBS=100
```

说明：

- `BL03U_ADMIN_TOKEN`: 写入服务器维护 PICS 库时必须提供。
- `BL03U_ALLOWED_DATA_ROOTS`: 允许 API 读取的服务端数据目录；项目根目录默认允许。
- `BL03U_MAX_UPLOAD_MB`: 上传文件大小限制。
- `BL03U_PICS_LIBRARY_TTL_HOURS`: 临时 PICS 工作库保留时间。
- `BL03U_PIE_JOB_TTL_HOURS`: 已完成/失败 PIE job 的内存保留时间。
- `BL03U_MAX_PIE_JOBS`: 最多保留的 PIE job 数量。

## 数据与备份

- 维护库默认路径是 `database/species_database.sqlite`。
- 默认 PICS seed 随包位于 `resources/pics/`；维护库缺失时可由 seed 自动生成或用
  `python -m bl03u_masstool.scripts.build_species_database` 显式重建。
- 管理员上传服务器维护库前，后端会在数据库目录的 `backups/` 下创建备份。
- 临时 PICS 工作库保存在系统临时目录，只用于当前会话，不作为共享维护数据。
- 服务器部署时建议把维护库和备份目录放在持久化磁盘中，并定期离线备份。

## 发布前检查

完整且持续维护的检查规则见 [`pre_commit_checklist.md`](pre_commit_checklist.md)。发布前至少执行：

```bash
git diff --check
python -m compileall -q src tests main.py
python -m pytest
```

Web UI 变更还应手动打开 `http://127.0.0.1:8000/`，检查 PIE/PICS 查询、临时 PICS 上传、服务器库 token 拒绝和成功路径。
