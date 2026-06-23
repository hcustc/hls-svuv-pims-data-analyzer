# BL03U 质谱分析工作流 / 数据流优化示意图

## 参数持久化与项目自动加载

### 项目配置文件结构

每个项目在初始化时自动创建 `config/` 目录，其中包含 `project.yaml` 配置文件。

**文件位置**: `project_root/config/project.yaml`

**内容**:
- 项目身份: 名称、系统、描述、输出目录
- 数据源路径: 原始谱图、和谱、温度扫描、PIE扫描位置
- 标定参数: Kr膨胀系数标定文件夹
- 归一化参数: 光源、光强归一化、质量歧视、膨胀系数
- 寻峰参数: 算法选择与17个算法特定参数
- 函数参数: 摩尔分数预设、光电离模式、PIE选项

### 参数持久化流程

```
保存项目 → config/project.yaml 创建/更新
    ↓
关闭项目
    ↓
重新打开项目 → load_project_settings()
    ├─ 自动检测 config/project.yaml
    ├─ 告诉 ProjectSettingsManager 使用项目路径
    ├─ 加载所有参数到 ProjectSettings
    └─ 立即同步到 UI 和参数 widgets
    ↓
所有参数自动恢复，无需手动重新设置 ✓
```

### 用户体验改进

✅ **新建项目按钮**: 绿色突出显示的 "➕ 新建项目" 按钮，一键清空表单
✅ **自动参数加载**: 打开项目时自动恢复所有参数
✅ **自动路径保存**: 数据源路径编辑完成后自动保存（无需手动点保存）
✅ **参数独立页面**: 通用参数和寻峰参数各占一个顶级标签页
✅ **项目隔离**: 每个项目有独立的配置文件，不会相互干扰

## 项目级总览

```mermaid
flowchart LR
    %% BL03U MassSpectrumTool workflow/dataflow overview

    subgraph L0["项目与配置层"]
        PM["项目生命周期管理\nproject_lifecycle.py\n- 初始化项目目录\n- 导入/登记数据源\n- 阶段状态与下一步提示"]
        RAW["原始数据源\n- 单谱 / 累计谱目录\n- PIE 扫描目录\n- 温度扫描目录\n- 样品信息\n- 手动卡峰文件"]
        CFG["参数配置 YAML\n- 标定 calibration\n- 寻峰 peak_detection\n- 归一化 normalization\n- 峰积分 peak_integration\n- 摩尔分数 mole_fraction"]
        DB["PICS SQLite 数据库\nspecies_database.sqlite\nspecies + pic_cross_sections"]
    end

    subgraph L1["公共预处理与峰信息层"]
        IO["谱图读取\nspectrum_io.read_spectrum"]
        CAL["TOF -> m/z 标定\ncalibration.py"]
        PEAK["寻峰 / 卡峰 / 高斯拟合\nprominence | legacy | cwt\nmanual peak ranges"]
        AREA["峰面积积分与归一化\nbaseline / gaussian area\nIO 光强归一化\n空白扣除 / 质量歧视"]
        PEAKART["谱图分析产物\n峰列表、峰边界、拟合参数\nanalysis/spectrum/"]
    end

    subgraph L2["分支分析层"]
        PIE0["PIE 扫描分析\nanalyze_pie_folder\n按 photon energy 分组\n平均谱图 + blank 扣除"]
        PIE1["m/z 级 PIE 曲线\nbuild_pie_curves\nenergy -> intensity"]
        PIE2["PIE 物种拟合\nquery species by m/z\nPICS 插值 -> NNLS/手动系数\nR² / RMSE / 残差"]
        PIE3["物种鉴别结果\n同一 m/z 候选物种贡献\ncomponent intensities\nanalysis/pie/"]

        TEMP0["温度扫描分析\nanalyze_temperature_folder\n按 temperature 聚合\nsum/max/manual 参考谱"]
        TEMP1["m/z 温度曲线\nbuild_temperature_curves\ntemperature -> area"]
        TEMP2["温度响应分类\nformation / consumption\nintermediate / unclassified"]
        TEMP3["温度扫描结果\nanalysis/temperature_scan/"]

        MF0["摩尔分数计算\ncompute_all_mole_fractions"]
        MF1["定量浓度曲线\nX_parent(T), X_product(T)\n质量歧视 + 膨胀系数 + PICS"]
        MF2["机理讨论输入\n物种随温度定量变化\nanalysis/mole_fraction/"]

        PICS0["PICS 截面计算\npics_calculator.py\nNO 比值法 / 多能量平均"]
        PICS1["新/修订截面数据\ncross_section(E)\n可用于数据库维护"]
    end

    subgraph L3["输出与项目闭环"]
        ART["分析产物登记\nproject_artifacts.py\n写回 project.yaml"]
        OUT["输出文件\nExcel / CSV / 图像 / manifest / report\noutput/exports + output/images"]
        REPORT["项目备份与版本快照\nversions/"]
        STATUS["项目阶段状态刷新\n数据导入 -> 标定 -> 谱图分析\n-> 温度扫描 / PIE 拟合\n-> 摩尔分数"]
    end

    PM --> RAW
    PM --> CFG
    PM --> DB
    RAW --> IO
    CFG --> CAL
    CFG --> PEAK
    CFG --> AREA
    IO --> CAL --> PEAK --> AREA --> PEAKART

    AREA --> PIE0 --> PIE1 --> PIE2 --> PIE3
    DB --> PIE2
    CFG --> PIE0

    AREA --> TEMP0 --> TEMP1 --> TEMP2 --> TEMP3
    CFG --> TEMP0

    TEMP1 --> MF0
    PIE3 --> MF0
    DB --> MF0
    CFG --> MF0
    MF0 --> MF1 --> MF2

    TEMP1 --> PICS0
    MF1 --> PICS0
    CFG --> PICS0
    PICS0 --> PICS1
    PICS1 -.->|审核后更新| DB

    PEAKART --> ART
    PIE3 --> ART
    TEMP3 --> ART
    MF2 --> ART
    PICS1 --> ART
    ART --> OUT --> REPORT
    ART --> STATUS --> PM
```

## PIE 拟合内部数据流

```mermaid
flowchart TB
    A["选择 m/z\nPIESpeciesFitDialog.on_mz_selected"] --> B["读取当前 PIE 曲线\nenergies + normalized_intensity"]
    B --> C["按 m/z 查询候选物种\nPICS SQLite: species + cross_sections"]
    C --> D["合并自动候选与锁定/手动物种\nFittingControlWidget.populate_unified_species_table"]
    D --> E["用户配置\n启用/禁用物种\n系数模式: fit / manual / locked_fit\n手动系数 / 锁定候选"]
    E --> F["重建拟合\n_rebuild_manual_fit"]
    F --> G["PICS 曲线插值到实验能量网格"]
    G --> H["设计矩阵 X\n每列 = 一个候选物种 PICS(E)"]
    H --> I["非负最小二乘或手动系数\ny_exp ~= X * coef"]
    I --> J["拟合质量\nR², RMSE, MAE, residuals"]
    I --> K["物种贡献\ncoefficient, contribution_percent\ncomponent_intensities"]
    J --> L["ResultDisplayWidget\n曲线数据 + 物种贡献明细"]
    K --> L
    L --> M["导出 PIE 结果\nExcel / 证据对象 / 项目产物登记"]
```

## 相比原图的优化点

- 将“项目管理”拆成项目生命周期、原始数据源、参数配置和 PICS 数据库，避免把控制层与数据源混在同一个框里。
- 增加公共预处理层：谱图读取、TOF 到 m/z 标定、寻峰/卡峰/高斯拟合、峰面积积分和归一化，这是 PIE 与温度扫描共同依赖的数据基础。
- 明确 PIE 拟合不是单纯“物种鉴别”，而是 `m/z 曲线 -> PICS 数据库候选 -> 插值设计矩阵 -> NNLS/手动系数 -> 贡献与拟合质量`。
- 明确温度扫描输出不仅是“m/z 随温度变化”，还包含 IO/Kr 校正、膨胀系数归一化和生成/消耗/中间体分类。
- 将摩尔分数计算放在 PIE 鉴别结果、温度曲线、PICS 数据库和配置参数的交汇处，突出它是定量机理分析的下游模块。
- 增加产物登记闭环：各分析模块导出后写回 `project.yaml`，项目页刷新阶段状态，并可创建项目备份。
