# BL03U MassSpectrumTool

基于 PyQt6 的同步辐射质谱数据分析工具，用于处理上海光源（SSRF）BL03U 线站的光电离质谱实验数据。

## 功能

- **谱图浏览与标定**：TOF → m/z 二次标定，自动/手动寻峰与高斯拟合
- **PIE 曲线分析**：多能量扫描文件夹的 PIE 曲线生成与 PICS 拟合
- **PICS 计算**：绝对光电离截面的比值法计算
- **变温扫描分析**：单光子能量下的温度扫描数据分析
- **摩尔分数计算**：物种浓度定量分析
- **同位素分布**：分子式解析与理论同位素谱模拟
- **NIST Webbook 检索**：在线获取参考光电离截面数据

## 环境要求

- Python 3.12+
- uv（推荐）或 pip

## 安装

```bash
# 创建虚拟环境
uv venv --python 3.12
source .venv/bin/activate

# 安装依赖
uv pip install -r requirements-dev.txt    # 含测试/打包工具
# 或仅生产依赖
uv pip install -r requirements.txt
```

## 运行

### 桌面 GUI

```bash
python main.py              # 启动完整 GUI
python main.py --pics       # 直接打开 PICS 计算工具
python main.py --tools      # 直接打开核心工具集
python main.py --help       # 查看帮助
```

### Web 服务

```bash
uvicorn api.server:app --reload
# 浏览器打开 http://127.0.0.1:8000/
```

### 命令行批处理

```bash
python scripts/bl03u_cli.py pie <文件夹> --output result.csv
python scripts/bl03u_cli.py temperature <文件夹> --output result.csv
python scripts/bl03u_cli.py formula 'C6H6' --isotopes
python scripts/bl03u_cli.py --help
```

## 配置

配置文件位于 `config/` 目录（YAML 格式）：

| 文件 | 说明 |
|------|------|
| `app.yaml` | 主配置，指向数据库路径、输出目录等 |
| `calibration.yaml` | TOF → m/z 标定系数 |
| `calibration_points.yaml` | 标定参考点 |
| `peak_detection.yaml` | 寻峰算法参数 |
| `peak_integration.yaml` | 预定义峰积分区间 |
| `normalization.yaml` | 信号归一化参数 |
| `mole_fraction.yaml` | 摩尔分数计算参数 |

## 运行测试

```bash
pytest tests/
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
├── core/                  # 核心算法（无 Qt 依赖）
├── frontends/
│   ├── pyqt_app/          # PyQt6 桌面界面
│   └── web_app/           # Web 前端（静态 SPA）
├── api/                   # FastAPI 服务端
├── scripts/               # CLI 工具与数据库脚本
├── config/                # YAML 配置文件
├── database/              # SQLite 物种数据库
├── tests/                 # pytest 测试
├── docs/                  # 详细文档
└── packaging/             # PyInstaller 打包配置
```
