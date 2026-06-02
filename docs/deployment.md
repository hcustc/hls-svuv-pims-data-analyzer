# BL03U MassSpectrumTool 部署说明

本文档面向实验站内网或受控公网部署。公网开放前应先确认数据目录、管理员 token、上传限制和备份策略。

如果你准备使用 Docker / 阿里云轻量应用服务器（容器环境），请优先查看 [Docker 部署方案](docs/docker_deployment.md)。

## 环境

推荐使用已有 conda 环境：

```bash
conda activate pyqt_env
python -m pytest tests/test_core_smoke.py tests/test_mole_fraction.py tests/test_api_server.py tests/test_cli.py
```

若需要新建运行/部署环境：

```bash
conda create -n bl03u python=3.9
conda activate bl03u
pip install -r requirements.txt
```

若需要新建开发/测试环境，`requirements-dev.txt` 已包含运行依赖：

```bash
conda create -n bl03u-dev python=3.9
conda activate bl03u-dev
pip install -r requirements-dev.txt
```

## 启动 Web 服务

```bash
conda activate pyqt_env
uvicorn api.server:app --host 127.0.0.1 --port 8000
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
- 管理员上传服务器维护库前，后端会在数据库目录的 `backups/` 下创建备份。
- 临时 PICS 工作库保存在系统临时目录，只用于当前会话，不作为共享维护数据。
- 服务器部署时建议把维护库和备份目录放在持久化磁盘中，并定期离线备份。

## 发布前检查

```bash
git diff --check
env PYTHONPYCACHEPREFIX=/private/tmp/bl03u_pycache python3 -m compileall -q core api tests scripts
conda run -n pyqt_env python -m pytest tests/test_core_smoke.py tests/test_mole_fraction.py tests/test_api_server.py tests/test_cli.py
```

Web UI 变更还应手动打开 `http://127.0.0.1:8000/`，检查 PIE/PICS 查询、临时 PICS 上传、服务器库 token 拒绝和成功路径。
