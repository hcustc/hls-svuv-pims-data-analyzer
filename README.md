# BL03U MassSpectrumTool

基于 PyQt6 的同步辐射质谱数据分析工具，用于处理合肥光源（NSRL）BL03U 燃烧光束线站的光电离质谱实验数据。

## 功能

- **谱图浏览与标定**：TOF → m/z 二次标定，自动/手动寻峰与高斯拟合
- **PIE 曲线分析**：多能量扫描文件夹的 PIE 曲线生成与 PICS 拟合
- **PICS 计算**：绝对光电离截面的比值法计算
- **变温扫描分析**：单光子能量下的温度扫描数据分析
- **摩尔分数计算**：物种浓度定量分析
- **同位素分布**：分子式解析与理论同位素谱模拟
- **NIST Webbook 检索**：在线获取参考光电离截面数据

## 环境要求

- Python 3.12（见 `.python-version`）
- `requirements.lock` / `requirements-dev.lock` 锁定运行、测试与打包依赖

## 安装

```bash
# 创建 Python 3.12 虚拟环境
uv venv --python 3.12
source .venv/bin/activate

# 安装开发、测试、打包依赖
uv pip install -r requirements-dev.lock
uv pip install --no-deps -e .

# 或仅安装运行依赖
uv pip install -r requirements.lock
uv pip install --no-deps -e .
```

## 运行

### 桌面 GUI

```bash
python main.py              # 启动完整 GUI
bl03u-gui                   # 安装后启动完整 GUI
python main.py --pics       # 直接打开 PICS 计算工具
python main.py --tools      # 直接打开核心工具集
python main.py --help       # 查看帮助
```

### Web 服务

```bash
uvicorn bl03u_masstool.api.app:app --reload
# 浏览器打开 http://127.0.0.1:8000/
```

### 命令行批处理

```bash
bl03u pie <文件夹> --output result.csv
bl03u temperature <文件夹> --output result.csv
bl03u formula 'C6H6' --isotopes
bl03u --help
```

## 配置

默认配置随包放在 `src/bl03u_masstool/resources/config/`。开发态或用户覆盖配置位于 `config/`
目录（YAML 格式）；如果工作副本不存在，程序会回退到内置默认资源。

| 文件 | 说明 |
|------|------|
| `app.yaml` | 主配置，指向用户数据库路径、输出目录等 |
| `calibration.yaml` | TOF → m/z 标定系数 |
| `calibration_points.yaml` | 标定参考点 |
| `peak_detection.yaml` | 寻峰算法参数 |
| `peak_integration.yaml` | 预定义峰积分区间 |
| `normalization.yaml` | 信号归一化参数 |
| `mole_fraction.yaml` | 摩尔分数计算参数 |

## 运行测试

```bash
python -m pytest
```

## 工程状态

当前 Python/依赖锁定、CI、PyQt 拆分和日志策略见 [`docs/engineering_maturity.md`](docs/engineering_maturity.md)。`docs/archive/completion/` 下的文件是历史里程碑记录，不作为当前全项目完成状态声明。

## 默认 PICS 数据库

默认 PICS 数据源为可审查的 seed 文件：

- `src/bl03u_masstool/resources/pics/schema.sql`
- `src/bl03u_masstool/resources/pics/species_seed.csv`

SQLite 工作库可由 seed 生成：

```bash
python -m bl03u_masstool.scripts.build_species_database --output database/species_database.sqlite
```

## Docker 部署

```bash
docker compose up -d --build
```

## 打包桌面版

使用 PyInstaller 打包为独立可执行文件，配置文件位于 `packaging/` 目录。

## 项目结构

```
├── main.py                # 桌面 GUI 入口
├── pyproject.toml         # Python 包、依赖与工具配置
├── src/bl03u_masstool/
│   ├── core/              # 核心算法（无 Qt 依赖）
│   ├── frontends/         # PyQt6 桌面界面与 Web 静态资源
│   ├── api/               # FastAPI 服务端
│   ├── resources/         # 内置默认配置、PICS seed/schema
│   └── scripts/           # CLI 工具与数据库脚本
├── config/                # 开发/用户覆盖 YAML 配置
├── database/              # 开发/用户 SQLite 工作库（可由 seed 生成）
├── output/                # 运行产物，仅保留 .gitkeep
├── tests/                 # pytest 测试（unit/ 与 integration/ 分层）
├── docs/                  # 详细文档
└── packaging/             # PyInstaller 打包配置
```
