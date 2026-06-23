# 项目生命周期管理设计

本文记录 PyQt 项目管理页从“参数配置页”升级为“项目生命周期管理工具”的边界与实现策略。

## 职责边界

项目管理页负责：

- 初始化标准项目目录结构。
- 将原始谱图、温度扫描目录、PIE 扫描目录、样品信息和手动卡峰文件导入项目目录。
- 跟踪项目阶段状态，并提示下一步操作。
- 汇总展示项目目录内文件和已登记的外部分析产物。
- 创建版本快照和导出完整项目备份。

各分析工具页负责：

- 执行对应算法和交互分析。
- 导出结果文件。
- 在导出成功后调用项目产物登记接口，更新项目状态。

这样可以保持算法页面聚焦于分析逻辑，项目管理页作为统一状态源和文件中枢。

## 标准目录结构

```text
Project_<name>/
├── config/
├── raw_data/
├── analysis/
│   ├── calibration/
│   ├── spectrum/
│   ├── temperature_scan/
│   ├── pie/
│   ├── mole_fraction/
│   └── pics/
└── versions/
```

`raw_data/` 集中保存所有原始输入；`analysis/` 保存标定、谱图处理和各分析模块产物；`versions/` 用于项目 zip 快照。导出项目时也会包含已登记但不在项目目录内的外部结果文件。

## 阶段判断

阶段状态由 `core.project_lifecycle` 统一计算：

- 项目初始化：项目身份信息和项目目录存在。
- 数据导入：任一原始数据源或样品信息已登记，或 `raw_data/` 内已有文件。
- 标定：存在定标点或 `analysis/calibration/` 内已有文件。
- 寻峰/高斯拟合：存在手动卡峰文件或 `analysis/spectrum/` 内已有文件。
- 温度扫描：温度扫描结果已登记或 `analysis/temperature_scan/` 内已有文件。
- PIE 拟合：PIE 鉴定结果已登记或 `analysis/pie/` 内已有文件。
- 摩尔分数：摩尔分数结果已登记或 `analysis/mole_fraction/` 内已有文件。

已登记路径如果缺失，会显示为“需检查”，避免把丢失文件误判为健康完成状态。

## 当前实现入口

- 核心服务：`src/bl03u_masstool/core/project_lifecycle.py`
- 项目配置：`src/bl03u_masstool/core/project_settings.py`
- PyQt 项目页：`src/bl03u_masstool/frontends/pyqt_app/spectrum/workspace_pages.py`
- 产物登记：`src/bl03u_masstool/frontends/pyqt_app/project_artifacts.py`
