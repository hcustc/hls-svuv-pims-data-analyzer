# 当前功能实现评估范围

状态：已确认
确认日期：2026-08-03

## 目标

评估当前软件是否能够可靠地辅助 Scientific Analyst 处理 SR-PIMS Data，生成可检查、可复现、可追溯的分析结果与 Project Evidence。

软件和 Agent 可以执行数据处理、生成曲线、拟合 Candidate、QC Signal 和证据，但不自动形成 Scientific Interpretation。

领域术语以仓库根目录的 `CONTEXT.md` 为准，关键长期决策以 `docs/adr/` 为准。

## 功能与用户流程

### P0：科学基础与数据完整性

1. 创建、打开、保存和关闭 Project。
2. 导入 Raw Dataset，保留来源、稳定身份和内容指纹，不原地修改。
3. 查看单谱和累加谱。
4. 建立并检查 TOF/m/z 定标。
5. 生成、编辑、确认和激活 Peak Set Candidate。
6. 执行峰积分、光强归一化、质量响应等公共预处理。
7. 缺少会改变科学含义的信息时停止正式分析，不使用静默默认值。
8. 将结果绑定到 Raw Dataset、Analysis Baseline、完整参数和软件版本。
9. 依赖变化时保留旧 Analysis Version、显式取消其 Current Result 状态，并通过重算创建新版本。
10. 对明确支持的 Legacy Project 执行只读打开、迁移预览和非破坏另存。

P0 未通过时，下游功能即使能够运行，也只能标记为未验证。

### P1：可独立成立的 Curve Result

#### PIE Curve

1. 导入能量扫描数据。
2. 使用已确认的定标、Peak Set 和处理参数。
3. 生成并检查 m/z 通道的 PIE Curve。
4. 展示参数、关键中间结果与 QC Signal。
5. 由 Scientific Analyst 审阅并保存为独立 Curve Result。
6. 生成 Project Evidence 与 Derived Artifact。

#### Temperature Response Curve

1. 导入温度扫描数据。
2. 使用已确认的定标、Peak Set 和处理参数。
3. 生成并检查指定光子能量和 m/z 通道的 Temperature Response Curve。
4. 展示参数、关键中间结果与 QC Signal。
5. 由 Scientific Analyst 审阅并保存为独立 Curve Result。
6. 生成 Project Evidence 与 Derived Artifact。

Curve Result 不依赖后续物种解释、同位素校正或定量计算。下游无法进行时，应记录原因，但不能使已成立的 Curve Result 降级。

### 下游分析

- PICS Candidate 查询与拟合。
- 物种选择和拟合结果审阅。
- 同位素贡献校正。
- 温度响应分类。
- 摩尔分数计算。
- 绝对 PICS 计算。

本轮检查这些功能的实现、公式、单位、边界条件、错误处理和合成数据行为。由于目前没有定量 Reference Dataset，定量模块最多可评为“已实现、证据不足”，不能宣称科学验证通过，也不阻塞 Curve Result。

### 当前 Agent 接口

CLI 和 API 当前作为实验性集成接口，只评估：

- 已存在命令和端点的稳定性与基本安全性。
- 与桌面端重叠功能是否调用同一科学核心。
- 相同输入、配置和软件版本下，结果是否在规定容差内等价。
- 结构化输出是否足以供 Agent 读取和追踪。

完整 Project 作用域、完整 Project Evidence 契约、最小权限写入和审计属于后续路线图。缺失项记录为路线图差距，不判定为当前功能缺陷。

### P2：支撑能力

- 本地 PICS 数据库、查询和拟合。
- PICS Record 的来源、单位、能量范围、版本和确认记录。
- 分子式解析与理论同位素分布。
- NIST WebBook 查询与 External Reference 快照。
- Excel、CSV、图片、JSON 和 Markdown 导出。
- Windows 与 macOS 桌面打包。
- CLI/API 在 Windows、macOS 和 Linux 上的当前运行能力。

### 暂停维护能力

Web UI、线上 PICS 库部署和服务器管理员上传不参与当前功能通过判定。本轮只检查：

- 是否造成安全暴露。
- 文档是否错误暗示当前受支持。
- 是否增加核心模块维护负担。

本地 PICS 数据、查询与拟合核心，以及面向 Agent 的现有 CLI/API 仍在评估范围内。

## 统一验收原则

### 辅助分析而非自动结论

- 软件和 Agent 生成 Candidate、QC Signal 和 Project Evidence。
- Scientific Analyst 选择处理方法、确认 Analysis Baseline、审阅结果并形成 Scientific Interpretation。
- 自动处理不得自行宣称科学通过或形成科学结论。

### 可检查、可回退、显式确认

- 影响 Curve Result 的自动处理必须展示输入、参数、关键中间结果和 QC Signal。
- 自动寻峰、拟合和 Agent 建议只能创建 Candidate 或新版本。
- Candidate 不得静默覆盖 Analysis Baseline 或 Current Result。

### 科学错误默认关闭

- 缺失或无效信息只要可能改变科学含义，就必须停止正式结果生成。
- 错误应说明未满足的条件和修复方式。
- 不影响结果可审阅性的情况可以产生 QC Signal，但必须进入 Project Evidence 并对 Scientific Analyst 可见。

### 数据、配置和结果血缘

- Raw Dataset 不可变，转换结果写为独立 Derived Artifact。
- 每个 Analysis Version 绑定 Raw Dataset、Analysis Baseline、完整参数、软件身份和 Project Evidence。
- 项目内 Analysis Version 是权威记录；修改导出文件不会隐式改变 Project。
- 回导修改后的文件必须创建新的 Candidate 或 Analysis Version。

### 可重复与跨调用面等价

- 相同数据、基准、参数和软件版本应产生可重复结果。
- 确定性算法固定随机状态；不可避免的随机性必须记录。
- 跨平台和跨调用面比较采用方法规定的科学容差，不要求逐位相同。

### 长任务运行体验

- 数据导入、读取、寻峰、PIE/温扫处理和导出不得冻结桌面界面。
- 长任务展示阶段和进度，支持安全取消。
- 失败或取消后 Project 与 Raw Dataset 保持完整。
- 操作可从最后一个已确认状态重新执行。

## 支持的平台与输入

### 平台

- 桌面端：Windows、macOS。
- 当前 CLI/API：Windows、macOS、Linux。

### SR-PIMS 原始谱格式

- TXT。
- ASC。
- 888。

三种格式都使用 Representative Dataset 验证。只有当解析后的坐标、强度与元数据语义符合各自格式时，才能判定支持。

### PICS 表格导入格式

- CSV。
- TSV。
- TXT。
- XLSX。
- XLS。

## 验证证据

### 当前基线

- 2026-08-03 运行现有测试套件：659 passed，耗时 47.70 秒。
- 该结果证明当前自动化回归没有失败，但不等于科学验证通过。

### Representative Dataset

本地约 38 MB 的真实 SR-PIMS 数据包含：

- 389 个 TXT 文件。
- 22 个 ASC 文件。
- 22 个 888 文件。
- PIE 和温度扫描目录。

这些数据用于真实工作流回归、格式兼容和 Scientific Analyst 曲线审阅。当前存储目录名称是历史路径，不代表数据类型名称。

### Reference Dataset

科学验证应组合使用：

- 具有已知答案的合成数据。
- 由 Scientific Analyst 审核的代表性真实数据参考结果。
- 每项参考结果对应的明确容差。

当前没有定量 Reference Dataset，因此不能对摩尔分数、绝对 PICS 等能力给出科学验证通过结论。

### Project Evidence 目标契约

完整 Project Evidence 应绑定：

- 可识别的输入数据和内容身份。
- 已确认的分析配置与 Analysis Baseline。
- 软件版本和运行环境。
- 结构化中间结果与最终结果。
- QC Signal。
- Scientific Analyst 确认记录。

当前实现与该完整契约之间的差距，在 Agent 接口范围内记为路线图能力；对于当前桌面 Project 工作流，则作为可追溯性评估依据。

## PICS 与外部数据

- 正式 PICS Record 应包含来源引用、单位、能量范围、版本、导入记录和 Scientific Analyst 确认；有条件时记录不确定度。
- 缺少来源元数据的 PICS 数据只能用于探索性 Candidate 拟合，并产生来源不足的 QC Signal。
- NIST 等在线查询只提供 Candidate。
- 选用在线值时，应保存包含来源记录、查询条件和检索时间的 External Reference；后续在线变化不得静默改变既有 Analysis Version。

## 评估结论状态

每项功能使用以下状态之一：

- **已验证**：实现存在，并有足够证据支持本轮结论。
- **已实现、证据不足**：功能能运行，但缺少 Reference Dataset 或验收证据。
- **部分实现或行为不一致**：当前承诺、调用面或模块之间存在不一致。
- **路线图能力**：已确认未来需要，但不作为当前功能缺陷。
- **暂停维护**：代码可能存在，但当前不受支持。

## 交付物与边界

本轮交付一份带证据的功能实现评估报告。每个发现应包含：

- 功能与用户流程位置。
- 当前状态。
- 代码和测试证据。
- 可复现步骤或缺失证据。
- 对科学处理、数据完整性或人员工作流的影响。
- 建议优先级与后续动作。

评估阶段不修改实现，也不直接创建 GitHub Issues。Scientific Analyst 审阅报告后，再把确认的问题转换为 issue。
