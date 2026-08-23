# 质谱软件 Agent 化：相关工作、边界与 BL03U 改造建议

- 检索与代码审阅日期：2026-08-23
- 范围：质谱专用 Agent、代谢组/蛋白质组 Agent、可迁移的科学 Agent 与自主实验室、现有软件封装协议
- 证据规则：优先论文原文、官方规范和源代码仓库；把论文直接证明、本文判断和待验证事项分开

## 结论先行

**当前 BL03U MassSpectrumTool 可以改造成 Agent，但合适的目标不是让大模型替代质谱算法或 Scientific Analyst，而是增加一个“规划—调用—检查—留痕”的受约束分析层。**

现有软件已经具备较好的 Agent 工具底座：质谱算法集中在无 Qt 依赖的 `core`，桌面端、CLI 和 FastAPI 可以复用同一科学核心；PIE API 已有异步 `job_id`、进度、结果、拟合、候选和证据接口；CLI 能导出 manifest、evidence 和报告。项目 ADR 也已明确“Agent 只能生成 Candidate、QC Signal 和 Project Evidence，最终确认属于 Scientific Analyst”。因此不需要重写科学算法，优先把现有确定性能力包装成强类型工具，并增加 Project 作用域、持久状态、权限、验证器和审计即可。

但它**尚不是生产可用的科学 Agent**。当前 API 请求没有 Project/Analysis Version 身份；PIE job 保存在进程内存并按 TTL 清理；未见取消/恢复接口；多数 API 未做按工具授权；CORS 仍为 `*`；manifest 尚缺输入哈希、软件包版本、Project/基线/分析版本与完整谱系。现有功能评估文档也把“完整 Project 作用域、Project Evidence 契约、最小权限写入和审计”列为路线图差距。故当前成熟度适合先做**只读/探索型分析 Agent**，再进入 Project 内候选生成，不宜直接控制仪器或自动确认科学结论。

## 1. 什么才算 Agent

本文用可检查的行为定义，而不按论文标题中的 “agent” 字样判断。

| 类型 | 必要行为 | 本文是否称为 Agent |
| --- | --- | --- |
| 预测模型/算法 | 给定输入产生预测，如峰识别、谱图嵌入、结构排序 | 否，是 Agent 可调用的工具 |
| 聊天或 RAG 助手 | 回答说明书、文献或数据库问题，但不实际改变分析状态 | 否 |
| 固定自动化流水线 | 按预先写定的步骤顺序运行，结果不会改变下一步策略 | 否；可作为执行底座 |
| 分析 Agent | 保存任务/项目状态；按目标动态选择强类型工具；观察输出或错误后修正；运行独立验证；保留可重放轨迹 | 是 |
| 多 Agent 系统 | 多个角色分工，但仍须满足实际工具执行、状态和反馈闭环 | 可能是；“多人聊天”本身不足 |
| 闭环实验/仪器系统 | 依据测量反馈选择下一次实验或仪器参数，并有物理安全边界 | 是闭环自治系统；不一定是 LLM Agent |

这一划分直接影响 BL03U 的验收：加一个聊天框不构成 Agent；自然语言转成固定 CLI 命令也只是助手。至少要能把目标分解为工具调用，读取结构化结果和 QC，因失败/冲突而改变后续动作，并生成绑定数据、参数、软件与确认状态的证据轨迹。

## 2. 质谱与组学领域的直接相关工作

### 2.1 已较明确满足 Agent 定义

| 工作与来源状态 | Agent 性证据 | 结果与限制 | 对 BL03U 的直接启示 |
| --- | --- | --- | --- |
| **MSAgent**，bioRxiv 预印本，2026-04-24；[论文/DOI](https://doi.org/10.64898/2026.04.22.720103) | LLM 动态编排 MSToolbox 中 50 余个质谱和化学信息学工具，执行意图规划、跨资源证据综合和可追踪报告；不是固定脚本。 | 在 CASMI 2016/2022、CANOPUS 和 LLM 测试上报告排序、结构相似性和置信校准改进。仍是预印本，公开实现、跨实验室复现和运行安全尚未充分证明。 | 最同构的总体架构是“已有工具箱 + 规划层 + 证据层”，而非另训一个端到端谱图聊天模型。 |
| **GNPS2 Agentic AI for drug-metabolite elucidation**，bioRxiv 预印本，2026-06-26；[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13320852/)，[DOI](https://doi.org/10.64898/2026.06.23.734138) | Claude Code 通过 MCP 动态调用 26 个 GNPS2/化学工具，包括 ModiFinder、FIDDLE、ICEBERG、MassQL、MASST、SyGMa 和 RDKit；用质量误差、加合物一致性与 LogP—保留时间方向做程序化验证，失败后重选加合物、位点或结构。 | 478 个 analog pairs 中处理了优先子集；两类候选由合成标准品及 LC-MS/MS/RT 确认，另有 repository-scale 发现。作者明确承认总体错误率未知、欠明确提示会进入死路、11 个候选因 token 限制未处理、部分异构体未决；系统运行于隔离 VM，论文未解决网络安全。这是**有实验确认的 proof-of-principle，而非可放任自治的结构鉴定器**。论文给出的代码仓库链接在本次检索时返回 404，公开实现可用性尚未确认。 | 最值得复制的是“每个 LLM 候选都通过独立数值工具检查；失败触发修正；最终保持候选等级并接受标准品/专家确认”。MCP 只做工具互操作，科学约束仍在工具服务器。 |
| **MetaboT**，Journal of Cheminformatics 同行评审、online-first，2026-07-24；[论文/DOI](https://doi.org/10.1186/s13321-026-01257-8)，[arXiv](https://arxiv.org/abs/2510.01724)，[代码](https://github.com/HolobiomicsLab/MetaboT) | Entry、Validator、Supervisor、实体解析、SPARQL Runner 和 Interpreter 分工；实际调用 Wikidata、ChEMBL、NPClassifier 与 GNPS，查询失败后修正。 | 49 个计分问题上报告 83.67%，但任务是已处理代谢组知识图谱的自然语言查询，不读取原始谱图，也不执行定量分析；评测小且 schema 迁移需要修改提示/解析器。 | 可借鉴“自然语言解释层—实体解析—可执行查询—结果解释”，但不能让 KG 问答替代校准、峰识别、PIE/PICS 拟合。 |
| **Proteomics Lab Agent**，Molecular Systems Biology 同行评审论文，2025-12-15 online / 2026 issue；[论文/DOI](https://doi.org/10.1038/s44320-025-00179-1)，[代码](https://github.com/MannLabs/proteomics_lab_agent) | 主 Agent 编排 Protocol、Lab Note、Knowledge、Instrument 与 QC Memory；通过 MCP 读取质谱仪状态，并把历史 QC 与专家评价持久化。 | 主要验证协议视频、错误识别和实验室知识传递；仪器 Agent 偏监视而非自主控制。论文报告细粒度空间/快速动作识别困难并偶有幻觉，且只在作者实验室验证。 | 初期仪器能力应只读；把阈值 QC、历史基线与专家确认存入独立记忆，而不是依赖语言模型“目测正常”。 |
| **AstroAgents**，ICLR 2025 Agentic AI for Science workshop；[arXiv](https://arxiv.org/abs/2503.23170)，[项目与代码](https://astroagents.github.io/) | Data Analyst、Planner、三名 Scientist、Accumulator、Literature Reviewer 与 Critic 共八角色，批评反馈进入下一轮假设修正。 | 输入实际是已经鉴定的 48 个化合物、RT/m/z/样本出现表和用户提供文献，而非原始谱图。八个陨石和十个土壤样本的百余假设仅由一名专家评 plausibility/novelty；没有外部实验确认。 | 可借鉴“数据—文献—假设—批评”后处理，但不能把 LLM 对表格的解释当作谱图分析证据；多角色也不自动提高科学可靠性。 |
| **PROTEUS**，arXiv 预印本，2024；[论文](https://arxiv.org/abs/2411.03743) | 以层级目标和工作流管理工具执行，迭代更新分析并生成蛋白质组假设。 | 12 个蛋白质组数据集、191 个假设；评估包含 LLM 评分和专家复核。主要处理下游定量/表达数据而非厂商原始谱或仪器采集；专家指出部分生物学细节不足，长上下文限制工作流规模。 | Project 内可用显式“目标—工作流—结果—假设”图，但假设必须指回确定性数值结果和分析版本。 |

### 2.2 其他近期系统：真 Agent、边界系统与待复核声明

| 工作 | 本文判定 | 原因 |
| --- | --- | --- |
| **MS4MS**，bioRxiv 预印本，2025；[论文](https://www.biorxiv.org/content/10.64898/2025.12.02.691830v2) | **可能是真 Agent，但公开证据尚不足** | 摘要声称多 Agent 端到端完成 LC-MS/MS 小分子鉴定并区分异构体；尚需核查工具定义、运行轨迹、失败修正、代码和外部复现，不能仅据名称接受。 |
| **AIMe**，bioRxiv 预印本，2026；[论文](https://www.biorxiv.org/content/10.64898/2026.08.05.743095v1.full) | **神经符号多模块系统，非本文所指的通用工具 Agent** | 三个“agent”分别预测碎裂、构建/索引 8 亿余预测谱和检索映射，具有目标化模块与验证，但没有自然语言目标驱动的动态工具规划；论文自己把加入语言模型与多模态推理列为未来方向。它更适合作为 Agent 的结构检索工具。 |
| **PACE-SIMS**，arXiv 预印本，2026-08-12；[论文](https://arxiv.org/abs/2608.12277) | **真正的 checkpoint-gated 仪器 Agent** | 研究者给出问题与 QC、批准计划；Agent 执行 SIMS 表征，在检查点判断质量并选择纠正、重试或升级给人。盲法随机双极性研究运行 8.1 h、35 次测量，报告三次未预先脚本化的纠正。仍是很新的预印本，使用 ToF-SIMS 而非 SR-PIMS，跨仪器复现、安全边界和长期可靠性尚未建立。 |
| **AILA / AFM agents**，Nature Communications 同行评审，2025；[论文](https://www.nature.com/articles/s41467-025-64105-7) | **真实仪器操作 Agent 的严格反例/基准** | AFM Handler 与 Data Handler 执行原子力显微镜任务，并用 AFMBench 评测；多 Agent 优于单 Agent，但通用/领域 QA 表现不能预测实验室操作，观察到指令偏离、“sleepwalking”和显著提示敏感性。它不是质谱工作，却直接说明离线会答题不等于能安全控制仪器。 |

### 2.3 名称相近但不应算 Agent 的工作

| 工作 | 实际能力 | 为什么不是本文定义的 Agent |
| --- | --- | --- |
| **LLM4MS**，Communications Chemistry 同行评审，2025；[论文/DOI](https://doi.org/10.1038/s42004-025-01708-7) | 把 EI 谱转为文本表示并学习谱图嵌入，用于化合物检索。 | 单模型推断，没有任务状态、工具选择、结果反馈和验证循环；可成为检索工具。 |
| **CLAW-MRM**，Analytical Chemistry 同行评审，2025；[论文/DOI](https://doi.org/10.1021/acs.analchem.4c05039) | 自动完成 MRM 脂质注释、统计与数据解析；LLM 自然语言界面生成分析 JSON。 | 证据显示的是固定工作流与自然语言入口，没有动态 plan–act–observe–repair。 |
| **MassQL Chatbot**，MassQL 论文为 Nature Methods 同行评审，2025；[论文](https://www.nature.com/articles/s41592-025-02660-z) | 帮助用户编写和排错 MassQL 查询。 | 聊天式查询助手；若没有跨工具状态与验证器，不是完整分析 Agent。 |
| **MassSpecGym**，NeurIPS 2024 benchmark；[论文](https://arxiv.org/abs/2410.23326)，[代码](https://github.com/pluskal-lab/MassSpecGym) | MS/MS de novo、检索和谱图模拟的标准数据集、split 与指标。 | 是评测基础设施，不是 Agent；其价值在于提醒本项目必须建立独立测试集和 OOD/结构泄漏控制。 |
| SIRIUS、CANOPUS、FIDDLE、ICEBERG、ModiFinder、谱库检索、峰识别/拟合模型 | 各自完成窄任务。 | 它们是工具。只有上层根据目标选择、执行、读取并因结果改变下一步时，才构成 Agent。 |

## 3. 可迁移的科学 Agent 与自主仪器工作

| 工作与来源状态 | 系统类型与证据 | 限制 | 可迁移设计 |
| --- | --- | --- | --- |
| **ChemCrow**，Nature Machine Intelligence 同行评审，2024；[论文/DOI](https://doi.org/10.1038/s42256-024-00832-8)，[代码](https://github.com/ur-whitelab/chemcrow-public) | GPT-4 规划并调用 18 个化学工具，参与合成和发色团筛选；反应执行需要用户许可。是真正的工具使用化学 Agent。 | 不是质谱系统；性能被工具质量和覆盖范围约束。 | 把计算交给窄、强类型的领域工具；有副作用的执行必须审批。 |
| **Coscientist**，Nature 同行评审，2023；[论文/DOI](https://doi.org/10.1038/s41586-023-06792-0)，[简化代码](https://github.com/gomesgroup/coscientist) | Planner 调用网页/文档搜索、代码和实验机器人；错误 API 名称可通过查文档修正，能根据实验结果继续决策。 | proof-of-concept；仍有人移动孔板，完整代码因双重用途安全未公开。 | “分析 Agent”与“物理控制 Agent”应是两个权限等级；仪器控制需要审批、联锁、受限参数域和不可抵赖日志。 |
| **AutonoMS**，JASMS 同行评审，2024；[论文/DOI](https://doi.org/10.1021/jasms.3c00396)，[代码](https://github.com/gkreder/autonoms) | 打通 IM-MS 采集、原始数据处理和代谢组分析的端到端自动化。 | 试验 specification 仍由人/上游生成；本身更像自动化底座，不是动态 LLM Agent。无厂商 API 时依赖 GUI 自动化。 | 若未来控制 BL03U 仪器，先建立稳定 acquisition job abstraction；GUI 自动点击只能作脆弱兼容层。 |
| **MUSCLE**，Bioinformatics 同行评审，2014/2015；[论文/DOI](https://doi.org/10.1093/bioinformatics/btu740) | 遗传算法依据每次 LC-MS/MS 结果选择下一组参数，约 48 h/200 次分析；在两个厂商平台展示速度和灵敏度改进。是真闭环优化，但不是 LLM Agent。 | 目标和步骤受限；可能收敛到局部最优，通用峰检测简单；GUI visual scripting 脆弱。 | 闭环系统应有明确目标函数、可暂停/停止、参数边界与人审；“自主”不意味着能给开放式科学结论。 |
| **Autonomous chemical research with LLMs / Coscientist** 与其他自主实验室 | 测量反馈进入下一轮实验规划，构成物理闭环。 | 安全与可恢复性要求远高于离线分析；论文演示不能直接证明跨仪器部署成熟。 | BL03U 先完成离线分析 Agent；仪器写操作作为独立项目和独立威胁模型。 |

## 4. 当前 BL03U 软件的 Agent 化基础与缺口

### 4.1 已有基础（代码直接证据）

1. `src/bl03u_masstool/core/` 已把校准、峰识别、PIE、PICS、温度扫描、同位素、数据库与 Project 生命周期等科学能力从 PyQt 界面分离，适合作为 deterministic tools。
2. FastAPI 已暴露谱图读取/峰识别/求和、同位素、PICS 查询与受控上传、PIE start/progress/result/artifacts/curve/candidates/fit/export；FastAPI 自带 OpenAPI schema，可作为第一层机器接口。
3. PIE job 使用显式 `job_id`；artifact 接口返回 manifest 和 m/z 级 evidence；拟合后重建证据对象。CLI 的 `pie`、`temperature`、`formula` 也可输出结构化文件。
4. 路径访问有 allowed roots，上传有大小限制，服务器 PICS 写入有 admin token；这些是最小安全基础。
5. 项目语义已经比多数原型完整：
   - [ADR-0001](../adr/0001-agent-assistance-with-scientific-analyst-control.md)：Agent 执行/重放/检查确定性步骤，Scientific Analyst 掌握校准、峰集激活、物种、数据库变更和解释；
   - [ADR-0002](../adr/0002-require-project-scope-for-agent-analysis.md)：正式分析必须绑定 Project 和 committed config；无项目调用只能是 provisional Exploratory Run；
   - [ADR-0005](../adr/0005-scientific-equivalence-across-call-surfaces.md)：桌面、CLI、API 必须数值等价；
   - [ADR-0007](../adr/0007-automation-produces-reviewable-candidates.md)：自动化产物是可审阅 Candidate，不能覆盖 Analysis Baseline；
   - [ADR-0013](../adr/0013-long-running-analysis-is-observable-cancellable-and-recoverable.md)：长任务应可观测、取消与恢复；
   - [ADR-0015](../adr/0015-snapshot-external-scientific-references.md)：外部参考必须记录来源、查询、时间和确认快照。

### 4.2 进入生产 Agent 前的缺口（代码/文档判断）

| 缺口 | 当前证据 | 必须补什么 |
| --- | --- | --- |
| Project 作用域 | `PieStartPayload` 没有 project_id、config snapshot 或 analysis_version；当前评估文档也将完整 Project scope 列为路线图。 | 每次正式工具调用必须携带或解析 Project/Analysis Version；无 Project 时强制标 `exploratory=true`。 |
| 持久任务状态 | PIE job 是进程内字典，完成/错误后按 TTL 清理；进程退出即失效。 | 持久 job store、幂等键、取消 token、恢复点、明确终态与 artifact URI。 |
| 完整证据谱系 | manifest 有输入路径、参数、Python/platform 和摘要，但没有输入哈希、应用/核心版本、Project、基线、配置哈希、外部参考快照和确认记录。 | 对齐 Project Evidence 契约；每次 Agent step 写入 tool name/schema version/input digest/output digest/QC/actor/approval。 |
| 权限与最小副作用 | PICS 服务器写有 admin token，但大多数读/算接口未按身份授权；CORS 为 `*`。 | 收紧 origin；按 read/compute-candidate/confirm/admin/instrument-control 分权；确认权不可授予普通 Agent。 |
| 科学等价性 | ADR 规定三入口等价，但规则不等于测试证据。 | 用固定 synthetic + expert real datasets 比较 desktop/CLI/API/MCP 的数值容差和 evidence 语义。 |
| 验证器 | 现有 evidence 有点数、R²、候选数等 QC，是好起点，但不覆盖输入适用性、校准漂移、结果稳定性、外部证据冲突。 | 把 QC 变成独立、确定性的 verifier tools；Agent 不能自行改写验证阈值。 |
| Agent 运行审计 | 目前没有 agent_run、step、prompt/model、tool call、approval、retry、termination 的持久记录。 | 新建运行账本，但把自然语言 reasoning 与科学证据分开；最终报告引用可重放 tool outputs。 |

## 5. 推荐架构：ProjectAnalysis 领域接口为核心，OpenAPI 与 MCP 为薄适配层

```text
用户 / Scientific Analyst
        │ 目标、澄清、确认/拒绝
        ▼
Agent Host
  session + planner + budget + trace
        │
        ▼
Policy / Approval / Scientific Verifier
  read | compute-candidate | confirm-human-only | admin | instrument-control
        │
        ▼
MCP server（Agent adapter）/ FastAPI / CLI / Desktop
        │ 结构化 request/result；不实现第二套科学算法
        ▼
ProjectAnalysis application service（深模块）
        │
        ▼
bl03u_masstool.core
        │
        ▼
Project / Analysis Version / Project Evidence / Candidate / QC Signal
```

关键原则：

- **一个深的应用接口、一个科学核心**：先定义高内聚的 `ProjectAnalysis` 请求/结果契约；MCP、FastAPI、CLI 和 Desktop 都是薄 adapter，不复制计算逻辑。不要让 Agent 直接拼接几十个低层函数或依赖 GUI 状态。
- **显式状态**：MCP 核心请求是无状态的；Project ID、Analysis Version、job_id、baseline/config snapshot 必须作为显式 handle/参数，不藏在对话里。
- **候选与确认分离**：Agent 能创建新 Candidate/Analysis Version，但 `activate_baseline`、`confirm_species`、`accept_external_reference`、永久数据库写入应由人类身份完成。
- **外部数据不直写**：NIST/PICS/GNPS 等查询返回带来源和查询时间的 External Reference Candidate，经快照与确认后才进入 Project Evidence。
- **Agent 不直连文件系统、SQLite 或 GUI**：只允许经过路径白名单、schema 校验和审计的领域工具。
- **先单 Agent 后多 Agent**：当前问题不需要复制 AstroAgents 的八角色。一个 planner 加独立 deterministic verifiers 更容易评测；只有工具域和权限确实需要隔离时再引入多 Agent。

## 6. 协议与实现选择

### MCP（推荐作为 Agent 外部接口）

[Model Context Protocol 2026-07-28 specification](https://modelcontextprotocol.io/specification/2026-07-28/architecture) 定义 host/client/server 结构和 JSON-RPC 能力协商；server 可暴露 [tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)、[resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources) 与 prompts。它适合把 `ProjectAnalysis` 的少量高价值操作包装成 Agent 工具，并由 host 执行授权、同意与隔离。

对 BL03U 的映射：

- resources：Project 摘要、Raw Dataset 元数据、committed config、Analysis Versions、Evidence、QC、外部参考快照；
- tools：读取谱图、生成候选峰、运行 PIE/温度分析、获取 job、拟合候选、查询 PICS/NIST、生成证据包；
- elicitation/approval：补齐目标 m/z、解释 provisional 状态、让分析员确认高影响操作；
- opaque handle：project_id、analysis_version_id、job_id、candidate_id，且每次调用重新授权。

MCP **不是科学正确性协议**。它不会自动提供 Project Evidence、数值等价、权限或验证，必须由 BL03U server/host 强制执行。

### OpenAPI（保留为内部规范和非 Agent 自动化入口）

[OpenAPI 3.1.1](https://spec.openapis.org/oas/v3.1.1.html) 是语言无关的 HTTP 接口描述，可表达 operation、请求/响应 schema 和错误。FastAPI 已能生成 OpenAPI，因此推荐先完善现有 schema、错误模型和权限，再自动/半自动生成 MCP adapter。OpenAPI 本身不是 Agent 协议，也不管理会话规划与人类审批。

### Agent runtime/SDK（可替换的宿主层）

可选 runtime 只负责模型循环、session、tool routing、预算、审批与 tracing；科学规则不能只写在 prompt。以 [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) 为例，它支持工具、session、guardrail、handoff 和 tracing，但这是实现框架而不是跨厂商协议。无论选哪种 runtime，都应通过相同 MCP/OpenAPI conformance tests，避免把软件锁死在某个模型厂商。

## 7. 最小工具目录与权限

| 权限 | 初始工具 | 输出约束 |
| --- | --- | --- |
| read | `get_project_context`、`list_analysis_versions`、`inspect_dataset`、`get_job_status`、`get_evidence` | 不修改项目；包含 ID、版本与数据摘要 |
| compute-candidate | `read_spectrum`、`detect_peaks_candidate`、`run_pie_analysis`、`run_temperature_analysis`、`fit_pics_candidate`、`calculate_isotope` | 创建 Exploratory Run 或新 Candidate；不得覆盖 baseline |
| external-reference | `search_pics`、`query_nist` | 返回来源、查询、时间、原始结果摘要和待确认状态 |
| report | `build_evidence_report`、`compare_versions`、`explain_qc` | 只引用已登记 evidence；区分数据事实、软件 QC、Agent 推断和未决项 |
| human-only confirm | 激活峰集/基线、确认物种、接受外部参考、设为当前 Curve Result | 不作为普通 Agent 自主工具；通过明确界面与人类身份完成 |
| admin/instrument | 永久数据库维护、文件删除、仪器参数/采集控制 | 第一阶段不暴露；未来单独威胁模型、物理联锁与审批 |

示例 Agent 闭环：

1. 用户要求“比较 m/z 56 在 10.2–11.0 eV 的 PIE 候选物种”。
2. Agent 获取 Project、committed config 和当前 Analysis Version；若没有 Project，明确创建 provisional Exploratory Run。
3. 调用 PIE analysis，轮询 job；读取曲线 evidence 与 QC。
4. 若点数不足、校准状态缺失或 API 返回不适用，停止并请求数据/人类选择，不擅自放宽阈值。
5. 查询 PICS/NIST，冻结来源快照；运行候选拟合。
6. verifier 检查 R²、残差、候选贡献、参数边界、版本一致性和重复性；不通过则更换候选组合或说明冲突。
7. 输出 Candidate、支持/反对证据、QC、未决问题与重放 ID；不写成“已鉴定”。
8. Scientific Analyst 审阅后确认、修改或拒绝。

## 8. 分阶段实施与验收

### Phase 0：先完成可调用契约

- 为所有工具定义稳定 Pydantic/OpenAPI schema、错误码、单位、数据版本和副作用等级。
- 增加 Project/Analysis Version/Exploratory Run 作用域。
- 建立 desktop–CLI–API 的科学等价性测试和 reference datasets。
- 补齐 evidence manifest、数据哈希、软件版本、外部参考快照。

### Phase 1：只读 + 探索型 Agent

- MCP resources 暴露 Project 与证据；tools 只允许 read 与 provisional compute。
- Agent 能解释 QC、生成候选分析计划、运行 PIE/温度/同位素、生成报告。
- 不允许永久数据库写、baseline 激活、科学确认和仪器控制。

### Phase 2：Project 内候选 Agent

- durable jobs、取消/恢复、幂等、审计、按角色授权。
- Agent 创建新 Candidate/Analysis Version，独立 verifier 运行后交给人审。
- 引入模型/提示版本、token/时间预算、最大工具步数、失败关闭策略。

### Phase 3：受控科学分析 Agent

- 根据真实用户任务构建 benchmark：计划正确性、工具选择、数值正确性、证据完整性、失败识别、人工介入率、成本和可重放性。
- 对比固定脚本、聊天助手、无 verifier Agent、完整 Agent，证明“Agent 化”是否真的带来收益。
- 只有离线分析达到门槛后，才评估只读仪器监视；写仪器参数/闭环采集属于单独项目。

### 必须报告的评测指标

| 维度 | 指标示例 |
| --- | --- |
| 科学正确性 | 与核心函数/专家基线的数值容差、峰/曲线/拟合结果一致性 |
| 证据完整性 | input/config/software/intermediate/final/QC/confirmation 字段覆盖率 |
| Agent 能力 | 任务成功率、无效工具调用率、错误后恢复率、平均步骤/成本 |
| 安全性 | 越权调用拒绝率、路径逃逸/提示注入/恶意文件测试、敏感操作审批命中率 |
| 可重现性 | 相同 Project 与 committed config 的重放成功率、跨入口等价性 |
| 人机协同 | analyst 修改/拒绝率、介入位置、错误置信校准、节省时间 |

## 9. 决策建议

建议立项，但把项目名称和验收目标写成：

> **BL03U Scientific Analysis Agent：基于现有科学核心和 Project Evidence 的受约束分析协作层。**

第一版成功标准不是“能聊天”或“自动跑完”，而是：

1. 能把自然语言目标转换为可审计、可取消的强类型工具调用；
2. 每个输出绑定 Project/Analysis Version、输入、参数、软件、QC 和外部参考快照；
3. 能因数值 QC/错误而修正或安全停止；
4. 任何物种、峰集、基线和最终 Curve Result 都保持 Candidate 状态，直到 Scientific Analyst 明确确认；
5. 在 reference datasets 上与现有 desktop/CLI/API 数值等价，并优于固定脚本或聊天助手基线。

文献已证明“质谱工具箱 + LLM 规划 + 程序化验证 + 人工/实验确认”是可行方向，GNPS2 Agent 与 MSAgent 尤其直接；文献同样证明错误、提示死路、token 限制、单实验室复现和安全问题仍然真实存在。因此，**可改造，且现有代码结构适合改造；应从工具与证据工程开始，而不是从更换模型或增加多 Agent 角色开始。**

## 主要来源

### 质谱/组学 Agent

- Li et al. *MSAgent: An Evidence Grounded Agentic Framework for LLM-driven Scientific Exploration in Mass Spectrometry-based Metabolomics*. bioRxiv, 2026. [DOI](https://doi.org/10.64898/2026.04.22.720103).
- Wang et al. *Agentic AI for Structural Elucidation and Discovery of Drug Metabolites from Mass Spectrometry Data*. bioRxiv, 2026. [全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13320852/); [DOI](https://doi.org/10.64898/2026.06.23.734138). 论文列出的代码链接在本次检索时不可访问。
- Bekbergenova et al. *MetaboT: an LLM-based multi-agent framework for interactive analysis of mass spectrometry metabolomics knowledge graphs*. Journal of Cheminformatics, online-first 2026. [DOI](https://doi.org/10.1186/s13321-026-01257-8); [arXiv](https://arxiv.org/abs/2510.01724); [代码](https://github.com/HolobiomicsLab/MetaboT).
- Skowronek et al. *Multimodal AI agents for capturing and sharing proteomics laboratory practice*. Molecular Systems Biology, 2026 issue. [DOI](https://doi.org/10.1038/s44320-025-00179-1); [代码](https://github.com/MannLabs/proteomics_lab_agent).
- Saeedi et al. *AstroAgents: A Multi-Agent AI for Hypothesis Generation from Mass Spectrometry Data*. ICLR Workshop, 2025. [arXiv](https://arxiv.org/abs/2503.23170); [项目](https://astroagents.github.io/).
- Ding et al. *Automating Exploratory Proteomics Research via Language Models*. 2024. [arXiv](https://arxiv.org/abs/2411.03743).
- Ievlev et al. *PACE-SIMS: Checkpoint-Gated Autonomous SIMS Characterization with AI-Agent Quality Control*. arXiv preprint, 2026. [arXiv](https://arxiv.org/abs/2608.12277).

### 边界与评测

- Xu et al. *A large language model for deriving spectral embeddings for accurate compound identification in mass spectrometry*. Communications Chemistry, 2025. [DOI](https://doi.org/10.1038/s42004-025-01708-7).
- Beveridge et al. *CLAW-MRM*. Analytical Chemistry, 2025. [DOI](https://doi.org/10.1021/acs.analchem.4c05039).
- Bushuiev et al. *MassSpecGym*. NeurIPS, 2024. [arXiv](https://arxiv.org/abs/2410.23326); [代码](https://github.com/pluskal-lab/MassSpecGym).

### 科学 Agent、自动实验与协议

- Bran et al. *Augmenting large language models with chemistry tools*. Nature Machine Intelligence, 2024. [DOI](https://doi.org/10.1038/s42256-024-00832-8); [代码](https://github.com/ur-whitelab/chemcrow-public).
- Boiko et al. *Autonomous chemical research with large language models*. Nature, 2023. [DOI](https://doi.org/10.1038/s41586-023-06792-0); [代码](https://github.com/gomesgroup/coscientist).
- Krishnan et al. *Evaluating large language model agents for automation of atomic force microscopy*. Nature Communications 16, 9104 (2025). [论文](https://www.nature.com/articles/s41467-025-64105-7).
- Kreder et al. *AutonoMS: Automated Ion Mobility Metabolomic Fingerprinting*. JASMS, 2024. [DOI](https://doi.org/10.1021/jasms.3c00396); [代码](https://github.com/gkreder/autonoms).
- Bradbury et al. *MUSCLE: automated multi-objective evolutionary optimization of targeted LC-MS/MS analysis*. Bioinformatics, 2015. [DOI](https://doi.org/10.1093/bioinformatics/btu740).
- Model Context Protocol. [2026-07-28 architecture](https://modelcontextprotocol.io/specification/2026-07-28/architecture), [tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools), [resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources).
- OpenAPI Initiative. [OpenAPI Specification 3.1.1](https://spec.openapis.org/oas/v3.1.1.html).
