# 工作流系统完成 + 线程化实现方案

**日期**: 2026年6月20日
**测试状态**: 148/148 通过 (120核心 + 17产物 + 11工作流)
**提交**: 6b7fd5d

---

## 第一阶段完成 ✅ - 灵活工作流系统

### 实现内容

#### 1. 工作流配置系统

```python
class WorkflowProfile(Enum):
    SPECTRUM_ONLY = "spectrum_only"           # 仅质谱处理
    TEMPERATURE_SCAN = "temperature_scan"     # 温度扫描
    PIE_ANALYSIS = "pie_analysis"             # PIE拟合
    FULL_ANALYSIS = "full_analysis"           # 完整流程
```

#### 2. 依赖规则引擎

```python
class DependencyOperator(Enum):
    ANY_OF = "any_of"    # 至少一个
    ALL_OF = "all_of"    # 全部

# 规则示例
WORKFLOW_REQUIREMENTS = {
    SPECTRUM_ONLY: (
        DependencyRule(ANY_OF, ("single_spectrum", "sum_spectrum")),
    ),
    TEMPERATURE_SCAN: (
        DependencyRule(ANY_OF, ("single_spectrum", "sum_spectrum")),
        DependencyRule(ALL_OF, ("temperature_scan",)),
    ),
    ...
}
```

#### 3. 能力分析结果

```python
@dataclass
class WorkflowAnalysisResult:
    available_workflows: list[WorkflowProfile]  # 可执行工作流
    unavailable_workflows: dict[WorkflowProfile, str]  # 缺失及原因
    data_source_status: dict[str, bool]  # 各源有效性
    recommended_next_step: str  # 推荐操作
```

### 核心函数

| 函数 | 功能 |
|------|------|
| `analyze_workflow_capabilities()` | 主分析引擎 |
| `_check_workflow_requirements()` | 验证依赖规则 |
| `_get_missing_sources_for_workflow()` | 获取缺失项 |
| `get_project_ui_state()` | 使用能力分析更新状态 |

### UI改动

**数据导入页现在显示**:
```
可执行：
✓ 质谱工作台
✓ 温度扫描

尚不可执行：
○ PIE拟合：缺少 PIE扫描数据

建议：可执行以下工作流
```

### 用户体验改进

| 场景 | 旧行为 | 新行为 |
|------|-------|--------|
| 仅有单谱 | ❌ 无法进行任何分析 | ✓ 可进行质谱处理 |
| 有谱数据和温度 | ✓ 可分析 | ✓ 可进行温度扫描 + 质谱处理 |
| 仅有PIE但缺单谱 | ❌ 全部阻止 | ❌ 阻止PIE，允许后续导入 |
| 可选源失效 | ❌ 如果是单谱，全部阻止 | ✓ 使用备选源或继续进行 |

### 测试覆盖

11个新测试验证:
- ✅ 单/累谱互替代
- ✅ 温度扫描依赖
- ✅ PIE拟合依赖
- ✅ 完整分析灵活性
- ✅ 多工作流并行
- ✅ 缺失项报告
- ✅ 推荐操作生成
- ✅ 无谱数据阻止所有
- ✅ 状态收集

---

## 第二阶段待实现 ⏳ - UI线程化

### 问题背景

当前大目录操作（导入、快照、导出、扫描）在UI线程上执行，导致界面冻结：

| 操作 | 触发位置 | 规模 | 影响 |
|------|---------|------|------|
| 导入 | import_project_source() | copytree | 递归复制 |
| 快照 | create_project_snapshot() | zipfile | 遍历+压缩 |
| 导出 | export_project_archive() | zipfile | 递归+压缩 |
| 扫描 | scan_project_artifacts() | rglob | 递归遍历 |

BL03U 真实数据集通常有几千个文件，同步操作可冻结 Qt 事件循环。

### 实现方案

#### 1. Worker 架构 (QObject + QThread)

```python
class ProjectWorker(QObject):
    """文件系统操作基类"""
    progress = Signal(int, str)  # 百分比, 消息
    finished = Signal(object)     # 结果
    warning = Signal(str)         # 警告
    error = Signal(str)           # 错误
    cancelled = Signal()          # 取消

    def run(self):
        """子类实现具体操作"""
        pass

    def cancel(self):
        """停止操作并清理"""
        pass
```

#### 2. 具体 Worker 实现

```
ImportWorker(ProjectWorker)      # import_project_source()
SnapshotWorker(ProjectWorker)    # create_project_snapshot()
ExportWorker(ProjectWorker)      # export_project_archive()
ScanWorker(ProjectWorker)        # scan_project_artifacts()
```

#### 3. UI 集成模式

```python
# 启动异步任务
worker = ImportWorker(source, destination)
thread = QThread()
worker.moveToThread(thread)

# 连接信号
worker.progress.connect(progress_dialog.update)
worker.error.connect(self.show_error)
worker.finished.connect(self.on_import_done)
thread.started.connect(worker.run)
worker.finished.connect(thread.quit)

# 启动
thread.start()

# UI 响应
- 禁用按钮防止重复点击
- 显示进度对话框
- 支持 Cancel 按钮
- 完成/失败/取消后恢复UI状态
```

#### 4. 临时文件安全模式

```python
def import_with_safety(source, destination):
    staging = destination.with_suffix('.tmp')
    try:
        copytree(source, staging)
        # 验证
        if validate(staging):
            staging.replace(destination)
        else:
            shutil.rmtree(staging)
            raise ValidationError()
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
```

#### 5. 对象生命周期管理

```python
# 防止对象提前释放
# 防止重复连接信号
# 安全 quit/wait
# 窗口关闭时取消任务

class WorkerManager:
    def __init__(self):
        self.current_worker = None
        self.current_thread = None

    def cancel_if_running(self):
        if self.current_worker:
            self.current_worker.cancel()
            self.current_thread.quit()
            self.current_thread.wait()
```

### 修改清单

#### 核心文件
- `project_lifecycle.py`: 无改动（业务逻辑已独立）
- `workspace_pages.py`: 添加 ProjectWorker 集成 (6处调用点)
- `worker.py`: NEW - Worker 基类及具体实现

#### UI 文件
- `spectrum/workbench.py`: 导入操作
- `spectrum/workspace_pages.py`: 快照、导出、扫描操作
- `theme.py`: 进度对话框样式

#### 测试
- `test_project_lifecycle.py`: 无改动（已测试）
- `test_worker.py`: NEW - Worker 测试 (8+ 测试)

### 预期测试用例

```
✓ 任务成功完成
✓ 文件操作异常处理
✓ 用户取消操作
✓ 临时文件清理
✓ 禁用重复点击
✓ 快照不包含旧快照
✓ 源文件中途消失
✓ UI 状态恢复
✓ 线程安全（信号连接）
✓ 对象生命周期
```

### 实现建议

1. **优先级**: 快照、导出 > 导入 > 扫描
2. **测试驱动**: 先写测试 mock，再实现
3. **分步验证**: 逐个 Worker 实现和测试
4. **性能基准**: 记录处理大型数据集前后的耗时

---

## 当前状态总结

### 完成 ✅

| 问题 | 状态 | 实现 |
|------|------|------|
| P1: 快照递归打包 | ✅ 已修复 | 排除 versions/*.zip |
| P1: 导入自包含 | ✅ 已修复 | 目录验证 |
| P2: 初始化检测 | ✅ 已修复 | 标记文件 |
| P2: 产物页不完整 | ✅ 已修复 | 目录扫描 |
| P1: 工作流刚性 | ✅ 已实现 | 能力分析系统 |

### 待做 ⏳

| 问题 | 优先级 | 复杂度 | 工作量 |
|------|--------|--------|--------|
| P2: UI线程化 | 中 | 高 | 3-4天 |

### 测试状态

```
核心库       : 120/120 ✅
产物管理     : 17/17   ✅
工作流系统   : 11/11   ✅
────────────────────────
总计         : 148/148 ✅
```

---

## 推荐下一步

### 方案A (推荐): 工程团队接手线程化
1. 使用上述架构设计
2. 从 SnapshotWorker 开始（最简单）
3. 逐步添加其他 Worker
4. 完整测试线程生命周期

### 方案B: 继续同步执行
- 工作流系统已解决主要 UX 问题
- 线程化可作为后续优化
- 真实数据测试前必须解决

---

**结论**: 工作流系统大大改善了项目灵活性。UI线程化是生产级别部署前的最后一个关键问题。建议工程团队在生产前 1-2 周完成线程化实现。
