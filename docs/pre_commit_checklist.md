# 提交前检查清单

本文是仓库提交、推送和创建 Pull Request 前的统一检查入口。CI 工作流或开发流程变化时，必须同步更新本文，避免文档命令与 `.github/workflows/*.yml` 脱节。

## 使用原则

- 每次提交前执行“必检项”。命令必须全部以退出码 `0` 结束。
- 改动涉及 GUI、依赖、打包或科学计算时，再执行对应的“条件检查”。
- 修复缺陷时必须增加能够在修复前失败、修复后通过的回归测试。
- 不用“本机全量测试通过”替代 CI 等价命令；PR 的 Linux CI 当前明确执行 `python -m pytest -m "not gui"`。
- 发现新的 CI 失败模式后，在本文“CI 失败记录”中补充根因和预防命令。

## 首次准备

使用仓库指定的 Python 3.12，并在虚拟环境中安装锁定依赖：

```bash
python --version
uv pip install -r requirements-dev.lock
uv pip install --no-deps -e .
uv pip check
```

`python --version` 必须显示 `3.12.x`。不要使用系统 Python 或未安装锁文件的临时环境作为最终验证结果。

## 每次提交前必检

在仓库根目录执行：

```bash
git status --short
git diff --check
git diff --cached --check
python -m compileall -q src tests main.py
python -m pytest -m "not gui"
```

逐项确认：

- `git status --short` 中没有缓存、导出结果、临时图片、数据库、构建目录或其他意外文件。
- `git diff --check` 和 `git diff --cached --check` 均无空白错误。
- 新增测试文件已被 Git 跟踪，不处于遗漏的 `??` 状态。
- 删除或重命名的文件是有意操作，并已检查引用是否同步更新。
- 非 GUI 测试通过；这是 `.github/workflows/ci.yml` 中 PR `Test` job 的本地等价测试命令。

建议在提交前同时审阅改动范围：

```bash
git diff --stat
git diff --name-status
git diff --cached --stat
git diff --cached --name-status
```

## 条件检查

### 修改 PyQt/GUI

至少运行受影响的 GUI 测试文件。提交较大界面改动前运行全部 GUI 测试：

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -m gui
python -m pytest
```

Windows PowerShell 使用：

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest -m gui
python -m pytest
```

异步 Worker、Qt 定时器和 `WA_DeleteOnClose` 相关测试应等待状态或信号，不使用刚好略大于定时器间隔的固定睡眠。

所有直接导入 PyQt6 的测试模块还必须：

- 声明 `pytestmark = pytest.mark.gui`；
- 在 `try/except ImportError` 中导入实际 Qt 子模块；
- Qt 原生库不可用时调用 `pytest.skip(..., allow_module_level=True)`。

仅调用 `pytest.importorskip("PyQt6")` 不足以覆盖 `libEGL.so.1` 等 Qt 原生动态库缺失，因为顶层包可能导入成功，而 `QtWidgets` 随后才失败。

### 修改依赖或锁文件

改动 `pyproject.toml`、`requirements*.lock` 或 `uv.lock` 时，必须在干净的 Python 3.12 环境验证安装，防止本机已有依赖掩盖缺包或平台兼容问题：

```bash
python scripts/lock_from_venv.py
uv lock --check
uv pip install -r requirements-dev.lock
uv pip install --no-deps -e .
uv pip check
python -m pytest -m "not gui"
```

然后审阅依赖文件差异，确认没有意外的平台专属依赖或无关版本升级：

```bash
git diff -- pyproject.toml requirements.lock requirements-dev.lock uv.lock
git diff --cached -- pyproject.toml requirements.lock requirements-dev.lock uv.lock
```

只有依赖声明发生变化时才重新生成锁文件。`scripts/lock_from_venv.py` 基于当前虚拟环境生成结果，因此运行前必须确认虚拟环境完整且 Python 版本正确。

### 修改科学计算或数据模型

除受影响模块测试外，必须运行全量测试：

```bash
python -m pytest
```

审查时至少确认：

- 缺失或无效实验元数据不会被静默替换成看似有效的数值。
- 归一化、缩放、定标和同位素校正的单位、误差传播及拒绝条件保持明确。
- SQLite 曲线版本使用 `dataset_group` 识别逻辑结果，使用唯一 `dataset_key` 识别具体版本。
- 旧项目数据库和配置仍有覆盖明确的兼容测试。
- 临时数据不要求创建项目，也不会写入项目配置或工厂资源。
- 项目异步任务完成时会检查原项目是否仍处于活动状态。

### 修改桌面打包或资源

执行 Desktop workflow 中的编译和打包命令；至少在目标平台完成一次：

```bash
python -m pytest
python -m compileall -q src main.py
pyinstaller --noconfirm --clean packaging/BL03U_MassSpectrumTool.spec
```

确认新增资源已写入 `pyproject.toml` 的 package-data 或 PyInstaller spec，并检查构建产物能够启动。

## 提交完成标准

- 必检项全部通过。
- 条件检查与改动类型匹配并全部通过。
- 缺陷修复具备回归测试，测试名称能说明失败场景。
- 没有意外文件、调试输出、硬编码本机路径或凭据。
- 文档与实际 CI 命令一致。
- 推送后 GitHub Actions 全部通过；失败时先分析日志，不盲目重复运行。

## CI 失败处理

1. 记录 workflow、job、run 链接和失败步骤。
2. 获取日志中的第一个实际异常；后续级联错误通常不是根因。
3. 在干净的 Python 3.12 环境运行该步骤的原始命令。
4. 修复后增加或调整回归检查。
5. 在下表记录根因和以后必须执行的预防检查。

## CI 失败记录

| 日期 | Workflow / Job | 根因 | 本地复现与预防检查 |
|---|---|---|---|
| 2026-07-31 | [CI / Test，run 30620179410，job 91122644316](https://github.com/hcustc/hls-svuv-pims-data-analyzer/actions/runs/30620179410/job/91122644316) | Linux runner 缺少 `libEGL.so.1`；两个 GUI 测试在 marker 过滤前直接导入 `QtWidgets`，导致 pytest 收集失败 | 为两个文件补齐 GUI marker 和模块级 ImportError 跳过；新增静态契约测试，保证所有 PyQt 测试都具备安全收集保护 |

新增记录时不要只填写“测试失败”。必须写明首个根因、修复提交以及加入清单的具体预防命令。
