from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from PyQt6 import QtCore, QtWidgets

from bl03u_masstool.core.curve_database import (
    invalidate_curve_datasets_for_peak_source,
    project_curve_database_path,
)
from bl03u_masstool.core.peak_sets import (
    resolve_active_peak_file,
    verify_peak_set,
)
from bl03u_masstool.core.project_peak_generation import generate_project_peak_set
from bl03u_masstool.core.project_settings import ProjectSettings, ProjectSettingsManager
from bl03u_masstool.frontends.pyqt_app.workers import WorkerThread


logger = logging.getLogger(__name__)


def prepare_project_peak_set(
    owner: QtCore.QObject,
    settings: ProjectSettings,
    *,
    retry: Callable[[], None],
    set_busy: Callable[[bool, str], None],
    show_error: Callable[[str], None],
) -> bool:
    """Return True when a valid project peak set is ready for analysis.

    Missing or invalid project peak sets trigger an explicit user-approved,
    background generation from the project's accumulated spectrum. On success
    the saved project settings are reloaded and ``retry`` is queued.
    """
    project_dir = Path(settings.output_dir).expanduser().resolve()
    invalid_reason = ""
    try:
        resolved = resolve_active_peak_file(
            project_dir,
            active_peak_set_id=settings.active_peak_set_id,
            configured_peak_file=settings.manual_peak_file,
        )
        if resolved is not None:
            return True
    except (FileNotFoundError, ValueError) as exc:
        invalid_reason = str(exc)

    running = getattr(owner, "_project_peak_worker", None)
    if running is not None and running.isRunning():
        return False

    source_folder = Path(settings.sum_spectrum_folder).expanduser()
    if not source_folder.is_absolute():
        source_folder = project_dir / source_folder
    source_message = (
        f"\n\n当前卡峰集不可用：{invalid_reason}" if invalid_reason else ""
    )
    reply = QtWidgets.QMessageBox.question(
        owner,
        "项目缺少有效卡峰集",
        "项目模式的PIE和温度曲线必须使用已保存并校验通过的项目卡峰集。"
        f"{source_message}\n\n"
        "是否从项目累计质谱自动生成一个新卡峰集并继续？",
        QtWidgets.QMessageBox.StandardButton.Yes
        | QtWidgets.QMessageBox.StandardButton.Cancel,
        QtWidgets.QMessageBox.StandardButton.Cancel,
    )
    if reply != QtWidgets.QMessageBox.StandardButton.Yes:
        return False
    if not source_folder.is_dir():
        show_error(
            "项目累计谱目录不存在。请先在质谱工作台登记累计谱，"
            "或在项目管理中导入卡峰文件。"
        )
        return False

    set_busy(True, "正在从项目累计谱生成卡峰集…")
    worker = WorkerThread(
        lambda: generate_project_peak_set(
            settings,
            progress_callback=worker.report_progress,
        ),
        owner,
    )
    owner._project_peak_worker = worker

    def on_success(record) -> None:
        try:
            approved_path = verify_peak_set(project_dir, record)
            settings.active_peak_set_id = record.peak_set_id
            settings.manual_peak_file = str(approved_path)
            settings.temp_peak_source = "manual"
            manager = ProjectSettingsManager()
            if not manager.has_project_path():
                raise RuntimeError("当前项目配置尚未绑定到项目目录，不能激活卡峰集。")
            manager_root = manager.get_project_config_path().parent.parent.resolve()
            if manager_root != project_dir:
                raise RuntimeError(
                    "分析页面所属项目与当前项目管理器不一致，"
                    "已停止激活卡峰集，请重新打开项目后重试。"
                )
            manager.set(settings)
            manager.save()
            saved = manager.reload()
            verified = resolve_active_peak_file(
                project_dir,
                active_peak_set_id=saved.active_peak_set_id,
                configured_peak_file=saved.manual_peak_file,
            )
            if verified is None:
                raise RuntimeError("卡峰集写入成功，但项目配置读回校验失败。")
            invalidate_curve_datasets_for_peak_source(
                project_curve_database_path(saved),
                manual_peak_file=verified,
                active_peak_set_id=record.peak_set_id,
                peak_set_sha256=record.sha256,
            )
            if hasattr(owner, "project_settings"):
                owner.project_settings = saved
            set_busy(False, "项目卡峰集已生成")
            QtCore.QTimer.singleShot(0, retry)
        except Exception as exc:
            logger.exception("Failed to activate generated project peak set")
            set_busy(False, "项目卡峰集激活失败")
            show_error(str(exc))

    def on_failure(message: str) -> None:
        set_busy(False, "项目卡峰集生成失败")
        show_error(message)

    worker.finished_with_result.connect(on_success)
    worker.failed.connect(on_failure)
    worker.finished.connect(lambda: setattr(owner, "_project_peak_worker", None))
    worker.start()
    return False
