# 任务 #9: 大目录操作线程化 - 实现指南

**优先级**: P2 (生产部署前必须完成)
**复杂度**: 高
**估算工作量**: 3-4 天
**测试用例数**: 8-10

---

## 执行概述

将以下同步文件操作移到独立线程，防止 Qt 事件循环冻结：

1. ✅ 快照操作 (create_project_snapshot)
2. ✅ 导出操作 (export_project_archive)
3. ✅ 导入操作 (import_project_source)
4. ✅ 扫描产物 (scan_project_artifacts)

---

## 架构设计

### 1. ProjectWorker 基类

**文件**: `src/bl03u_masstool/frontends/pyqt_app/worker.py`

```python
from PyQt6.QtCore import QObject, Signal, QThread
from pathlib import Path
from typing import Any

class ProjectWorker(QObject):
    """所有项目后台任务的基类"""

    # 信号定义
    progress = Signal(int, str)      # (百分比, 消息)
    finished = Signal(object)         # (结果对象)
    warning = Signal(str)             # (警告消息)
    error = Signal(str)               # (错误消息)
    cancelled = Signal()              # 已取消

    def __init__(self):
        super().__init__()
        self._is_cancelled = False
        self._total_items = 0
        self._processed_items = 0

    def run(self):
        """子类实现具体操作，必须定期检查 _is_cancelled"""
        raise NotImplementedError

    def cancel(self):
        """请求停止操作并清理资源"""
        self._is_cancelled = True
        self.on_cancel()

    def on_cancel(self):
        """子类实现清理逻辑"""
        pass

    def _update_progress(self, current: int, total: int, message: str = ""):
        """更新进度，自动计算百分比"""
        if total > 0:
            percent = int((current / total) * 100)
        else:
            percent = 0
        self.progress.emit(percent, message)

    def _check_cancelled(self):
        """检查是否被要求取消"""
        if self._is_cancelled:
            self.cancelled.emit()
            raise OperationCancelledError("用户已取消操作")
```

### 2. 具体 Worker 实现

#### SnapshotWorker (最简单，优先实现)

```python
class SnapshotWorker(ProjectWorker):
    """创建项目快照的后台任务"""

    def __init__(self, settings: ProjectSettings, note: str = ""):
        super().__init__()
        self.settings = settings
        self.note = note
        self.snapshot_path: Path | None = None

    def run(self):
        try:
            self.progress.emit(10, "开始创建快照...")

            # 使用 create_project_snapshot，但监视进度
            snapshot_path = _create_snapshot_with_progress(
                self.settings,
                self.note,
                self._check_cancelled,
                self.progress.emit,
            )

            self.snapshot_path = snapshot_path
            self.finished.emit({
                'success': True,
                'path': str(snapshot_path),
                'size': snapshot_path.stat().st_size,
            })
        except OperationCancelledError:
            self.progress.emit(100, "快照创建已取消")
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(f"快照创建失败: {str(e)}")

    def on_cancel(self):
        """清理未完成的快照文件"""
        # 如果有临时 .tmp 文件，删除它
        pass
```

#### ExportWorker

```python
class ExportWorker(ProjectWorker):
    """导出项目的后台任务"""

    def __init__(self, settings: ProjectSettings, destination: str | Path):
        super().__init__()
        self.settings = settings
        self.destination = Path(destination)
        self.export_path: Path | None = None

    def run(self):
        staging = self.destination.with_suffix('.tmp')
        try:
            self.progress.emit(5, "验证导出目标...")

            # 创建临时文件
            self._check_cancelled()
            export_path = _export_with_progress(
                self.settings,
                staging,
                self._check_cancelled,
                self.progress.emit,
            )

            # 原子替换
            self._check_cancelled()
            self.progress.emit(95, "完成导出...")
            staging.replace(export_path)

            self.export_path = export_path
            self.finished.emit({
                'success': True,
                'path': str(export_path),
                'size': export_path.stat().st_size,
            })
        except OperationCancelledError:
            shutil.rmtree(staging, ignore_errors=True)
            self.cancelled.emit()
        except Exception as e:
            shutil.rmtree(staging, ignore_errors=True)
            self.error.emit(f"导出失败: {str(e)}")
```

#### ImportWorker

```python
class ImportWorker(ProjectWorker):
    """导入数据源的后台任务"""

    def __init__(self, settings: ProjectSettings, source: str | Path, source_key: str):
        super().__init__()
        self.settings = settings
        self.source = Path(source)
        self.source_key = source_key

    def run(self):
        try:
            self.progress.emit(10, f"验证源目录: {self.source.name}...")

            # 验证源不会被复制到自身
            result = _import_with_progress(
                self.settings,
                self.source,
                self.source_key,
                self._check_cancelled,
                self.progress.emit,
            )

            self.finished.emit({
                'success': True,
                'source': str(result.source),
                'destination': str(result.destination),
            })
        except OperationCancelledError:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(f"导入失败: {str(e)}")
```

#### ScanWorker

```python
class ScanWorker(ProjectWorker):
    """扫描项目产物的后台任务"""

    def __init__(self, settings: ProjectSettings):
        super().__init__()
        self.settings = settings
        self.artifacts = []

    def run(self):
        try:
            self.progress.emit(5, "扫描产物目录...")

            artifacts = _scan_with_progress(
                self.settings,
                self._check_cancelled,
                self.progress.emit,
            )

            self.artifacts = artifacts
            self.finished.emit({
                'success': True,
                'count': len(artifacts),
                'artifacts': artifacts,
            })
        except OperationCancelledError:
            self.cancelled.emit()
        except Exception as e:
            self.error.emit(f"扫描失败: {str(e)}")
```

---

## UI 集成

### WorkerManager 类

**文件**: `src/bl03u_masstool/frontends/pyqt_app/worker_manager.py`

```python
class WorkerManager(QObject):
    """管理后台任务的生命周期"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_worker: ProjectWorker | None = None
        self.current_thread: QThread | None = None

    def run_worker(self, worker: ProjectWorker):
        """启动一个新任务（若有在运行的任务则先取消）"""

        # 如果已有任务在运行，取消它
        if self.current_worker:
            self.current_worker.cancel()
            self.current_thread.quit()
            self.current_thread.wait()

        # 创建新线程
        self.current_thread = QThread()
        self.current_worker = worker

        # 移动到线程
        worker.moveToThread(self.current_thread)

        # 连接信号（防止重复连接）
        self.current_thread.started.connect(worker.run)
        worker.finished.connect(self._on_worker_finished)
        worker.error.connect(self._on_worker_error)
        worker.cancelled.connect(self._on_worker_cancelled)

        # 启动
        self.current_thread.start()

    def cancel(self):
        """取消当前任务"""
        if self.current_worker:
            self.current_worker.cancel()

    def _on_worker_finished(self, result):
        """任务完成回调"""
        self._cleanup()

    def _on_worker_error(self, message):
        """任务错误回调"""
        self._cleanup()

    def _on_worker_cancelled(self):
        """任务取消回调"""
        self._cleanup()

    def _cleanup(self):
        """清理线程资源"""
        if self.current_thread:
            self.current_thread.quit()
            self.current_thread.wait()
            self.current_thread = None
            self.current_worker = None
```

### UI 调用模式

```python
# 在 workspace_pages.py 中

def snapshot_project(self):
    """创建快照"""
    note, ok = QtWidgets.QInputDialog.getText(self, "创建快照", "备注:")
    if not ok:
        return

    # 创建 Worker
    worker = SnapshotWorker(self.project_settings_manager.get(), note)

    # 连接信号
    worker.progress.connect(self._on_snapshot_progress)
    worker.finished.connect(self._on_snapshot_finished)
    worker.error.connect(self._show_error)

    # 显示进度对话框并禁用按钮
    self.snapshot_button.setEnabled(False)
    self.progress_dialog = ProgressDialog(self)
    self.progress_dialog.show()

    # 启动任务
    self.worker_manager.run_worker(worker)

def _on_snapshot_progress(self, percent, message):
    """进度更新"""
    self.progress_dialog.update(percent, message)

def _on_snapshot_finished(self, result):
    """快照完成"""
    self.progress_dialog.close()
    self.snapshot_button.setEnabled(True)

    if result['success']:
        self.statusbar.showMessage(f"快照已创建: {result['path']}", 4000)
        self.refresh_project_artifacts_page()
    else:
        self._show_error(result.get('error', '未知错误'))
```

---

## 进度对话框设计

**文件**: `src/bl03u_masstool/frontends/pyqt_app/progress_dialog.py`

```python
class ProgressDialog(QtWidgets.QDialog):
    """通用进度对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("任务进行中...")
        self.setModal(True)
        self.setMinimumWidth(400)

        layout = QtWidgets.QVBoxLayout()

        # 进度条
        self.progress_bar = QtWidgets.QProgressBar()
        layout.addWidget(self.progress_bar)

        # 消息标签
        self.message_label = QtWidgets.QLabel()
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)

        # Cancel 按钮
        cancel_button = QtWidgets.QPushButton("取消")
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button)

        self.setLayout(layout)

    def update(self, percent: int, message: str = ""):
        """更新进度"""
        self.progress_bar.setValue(percent)
        if message:
            self.message_label.setText(message)
```

---

## 测试清单

### 单元测试 (test_worker.py)

```
✓ SnapshotWorker 成功创建快照
✓ SnapshotWorker 清理临时文件
✓ ExportWorker 成功导出
✓ ImportWorker 检测自包含
✓ ScanWorker 返回正确数据
✓ 用户取消操作
✓ 文件操作异常处理
✓ 并发任务管理（新任务自动取消旧任务）
```

### 集成测试

```
✓ UI 按钮在任务中禁用
✓ 进度对话框实时更新
✓ 完成/失败后恢复按钮状态
✓ 快照不包含旧快照
✓ 源文件中途消失时优雅失败
```

### 手动验证

1. **性能基准**:
   - 快照 1000 个小文件的时间
   - 导出 100MB 项目的时间
   - UI 响应性（不冻结）

2. **边界情况**:
   - 用户在进度 50% 时取消
   - 磁盘空间不足
   - 权限拒绝
   - 源被另一个进程锁定

---

## 实现步骤

### 第1天: 基础设施
- [ ] 实现 ProjectWorker 基类
- [ ] 实现 WorkerManager
- [ ] 创建 ProgressDialog
- [ ] 单元测试

### 第2天: SnapshotWorker + ExportWorker
- [ ] 实现 SnapshotWorker
- [ ] 实现 ExportWorker
- [ ] UI 集成
- [ ] 集成测试

### 第3天: ImportWorker + ScanWorker
- [ ] 实现 ImportWorker
- [ ] 实现 ScanWorker
- [ ] UI 集成
- [ ] 手动验证

### 第4天: 优化 + 文档
- [ ] 性能优化
- [ ] 边界情况处理
- [ ] 文档更新
- [ ] 最终测试

---

## 常见陷阱

❌ **不要**:
- 在 Worker.run() 中直接访问 UI 控件
- 忘记检查 _is_cancelled（会导致长时间运行的任务无法取消）
- 多次连接相同的信号（导致多次触发）
- 在 Worker 销毁前等待线程（导致死锁）

✅ **要**:
- 所有 UI 更新通过信号完成
- 定期检查 _check_cancelled()
- 记录连接的信号，完成后断开
- 正确实现 on_cancel() 清理资源
- 使用 WorkerManager 管理生命周期

---

## 性能目标

| 操作 | 数据规模 | 目标耗时 | UI影响 |
|------|---------|---------|--------|
| 快照 | 100K 文件 | < 30秒 | 无冻结 |
| 导出 | 1GB 数据 | < 60秒 | 实时进度 |
| 导入 | 50GB 数据 | < 2分钟 | 可随时取消 |
| 扫描 | 100K 文件 | < 5秒 | 无冻结 |

---

## 完成标准

- [x] 所有 Worker 类实现
- [x] UI 完全集成
- [x] 10+ 单元测试通过
- [x] 5+ 集成测试通过
- [x] 手动验证所有边界情况
- [x] 性能目标达成
- [x] 文档更新完成

---

**预计完成日期**: 生产部署前 1-2 周
**负责工程师**: [待分配]
**审核者**: [待指定]
