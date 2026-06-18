# 工程成熟度状态

本文是当前工程化状态的入口；`docs/completion/` 中的完成总结保留为算法专项历史记录，不代表整个项目已经生产就绪。

## Python 与依赖

- Python 版本统一为 3.12 系列：`.python-version` 与 `pyproject.toml` 保持一致。
- 人工维护依赖入口是 `pyproject.toml`。
- 可重复安装使用锁文件：`requirements.lock`（运行依赖）和 `requirements-dev.lock`（测试/打包依赖）。
- 变更依赖后，在已同步的虚拟环境中运行：

```bash
python scripts/lock_from_venv.py
```

## CI

- `.github/workflows/ci.yml` 在 push、pull request 和手动触发时安装 `requirements-dev.lock` 并运行 `python -m pytest`。
- `.github/workflows/desktop-build.yml` 在打包前也先运行测试，避免只产出未验证的桌面构建。

## PyQt 拆分

- 主谱图工作台仍在 `frontends/pyqt_app/spectrum/workbench.py`。
- 项目管理、顶层工具页导航和共享参数页已经拆到 `frontends/pyqt_app/spectrum/workspace_pages.py`。
- 谱图坐标轴和峰编辑对话框分别拆到 `axis.py` 与 `peak_dialog.py`。

## 日志与异常

- `bl03u_masstool.logging_config.configure_logging()` 统一日志格式。
- 桌面入口安装全局异常 hook；API 中间件记录未处理请求异常；PyQt `WorkerThread` 会记录后台任务 traceback。
- 可用环境变量 `BL03U_LOG_LEVEL` 和 `BL03U_LOG_FILE` 调整日志级别和文件输出。

## 历史完成文档

旧的阶段性文档用于追溯某次寻峰算法专项优化的结果。涉及“100% 完成”“生产就绪”的文字应按当时专项上下文理解；当前项目发布状态以 README、部署文档、CI 和本文为准。
