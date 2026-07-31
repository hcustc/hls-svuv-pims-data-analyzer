"""Scoped analysis-parameter editors for project and temporary data."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from bl03u_masstool.core.peak_ranges import load_peak_ranges
from bl03u_masstool.core.project_settings import (
    ProjectSettings,
    derive_project_compatibility_fields,
    load_factory_project_settings,
)
from bl03u_masstool.frontends.pyqt_app.normalization.widget import (
    CommonParametersWidget,
    FunctionDefaultsWidget,
)

ANALYSIS_PARAMETER_FIELDS: dict[str, tuple[str, ...]] = {
    "temperature": (
        "temperature_photon_normalize",
        "temperature_kr_correct",
        "temp_integration_method",
        "temp_curve_class_change_threshold",
        "temp_curve_class_peak_fraction",
        "temp_replicate_mode",
    ),
    "pie": (
        "pie_photon_mode",
        "pie_time_normalize",
        "pie_energy_decimals",
        "pie_recursive",
        "pie_integration_method",
        "pie_merge_method",
        "pie_replicate_mode",
        "selected_elements",
    ),
}

TEMPORARY_SHARED_PARAMETER_FIELDS: tuple[str, ...] = (
    "cal_a",
    "cal_b",
    "cal_c",
    "light_source",
    "kr_calibration_folder",
    "kr_calibration_peak_file",
    "kr_mz",
    "expansion_factors",
)


def copy_analysis_settings_fields(
    source: ProjectSettings,
    target: ProjectSettings,
    page: str,
) -> ProjectSettings:
    """Copy only the parameters owned by one analysis page."""
    try:
        fields = ANALYSIS_PARAMETER_FIELDS[page]
    except KeyError as exc:
        raise ValueError(f"unknown analysis settings page: {page}") from exc
    for field_name in fields:
        setattr(target, field_name, deepcopy(getattr(source, field_name)))
    normalized = derive_project_compatibility_fields(target)
    derived_fields = (
        ("temp_prefer_gaussian",)
        if page == "temperature"
        else ("pie_prefer_gaussian", "pie_multi_folder_mode")
    )
    for field_name in (*fields, *derived_fields):
        setattr(target, field_name, deepcopy(getattr(normalized, field_name)))
    return target


def build_temporary_settings(
    source: str,
    project_settings: ProjectSettings | None = None,
) -> ProjectSettings:
    """Return an isolated settings snapshot initialized from ``source``."""
    if source == "project" and project_settings is not None:
        return deepcopy(project_settings)
    if source == "factory":
        return deepcopy(load_factory_project_settings())
    raise ValueError("temporary settings source must be 'factory' or 'project'")


class TemporarySharedParametersDialog(QtWidgets.QDialog):
    """Edit shared processing resources for one temporary analysis session."""

    def __init__(
        self,
        settings: ProjectSettings,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._settings = deepcopy(settings)
        self.setWindowTitle("编辑通用参数")
        self.resize(980, 700)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        notice = QtWidgets.QLabel(
            "这些参数只用于本次临时数据分析；无需创建项目，也不会写入项目配置。"
        )
        notice.setWordWrap(True)
        notice.setObjectName("HintLabel")
        layout.addWidget(notice)

        self.common_widget = CommonParametersWidget(
            self._settings.to_normalization_settings(),
            self._settings.to_calibration(),
            self,
            show_actions=False,
        )
        self.common_widget.set_project_settings(self._settings)
        layout.addWidget(self.common_widget, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText(
            "应用到本次分析"
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText(
            "取消"
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self) -> None:
        self.common_widget.apply_to_settings(self._settings)
        super().accept()

    def settings(self) -> ProjectSettings:
        return deepcopy(self._settings)


class AnalysisSettingsDialog(QtWidgets.QDialog):
    """Edit one analysis page against an isolated ProjectSettings snapshot."""

    shared_parameters_requested = QtCore.pyqtSignal()

    def __init__(
        self,
        settings: ProjectSettings,
        parent=None,
        *,
        scope: str,
        page: str,
        pie_multi_segment: bool | None = None,
    ) -> None:
        super().__init__(parent)
        if scope not in {"project", "temporary"}:
            raise ValueError("analysis settings scope must be 'project' or 'temporary'")
        if page not in ANALYSIS_PARAMETER_FIELDS:
            raise ValueError("analysis settings page must be 'temperature' or 'pie'")

        self.scope = scope
        self.page = page
        self._pie_multi_segment = pie_multi_segment
        self._original_settings = deepcopy(settings)
        self._settings = deepcopy(settings)
        page_label = "温度扫描" if page == "temperature" else "PIE"
        scope_label = "项目参数" if scope == "project" else "临时参数"
        self.setWindowTitle(f"编辑{page_label}{scope_label}")
        self.resize(780, 500)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        notice = QtWidgets.QLabel(
            "保存后将立即写入当前项目，并同步到相关功能页。"
            if scope == "project"
            else "修改只用于本次分析；无需创建项目，也不会写回项目配置。"
        )
        notice.setWordWrap(True)
        notice.setObjectName("HintLabel")
        layout.addWidget(notice)

        self.function_widget = FunctionDefaultsWidget(
            self,
            show_actions=False,
            visible_pages=(page,),
        )
        self.function_widget.set_project_settings(self._settings)
        if page == "pie" and pie_multi_segment is not None:
            self.function_widget.set_pie_multi_segment_state(pie_multi_segment)
        layout.addWidget(self.function_widget, 1)

        if scope == "temporary" and page == "temperature":
            self.function_widget.temp_peak_source_label.setText("本次峰来源")
            self.function_widget.temp_peak_source_hint.setText(
                "临时卡峰文件（下方可导入；未选择时自动寻峰）"
            )
            self.function_widget.temp_peak_source_hint.setToolTip(
                "导入仅用于本次临时分析，不会复制到项目或写回项目配置"
            )

            temporary_peak_group = QtWidgets.QGroupBox("临时卡峰数据")
            temporary_peak_layout = QtWidgets.QGridLayout(temporary_peak_group)
            temporary_peak_layout.setContentsMargins(10, 8, 10, 8)
            temporary_peak_layout.setHorizontalSpacing(8)
            temporary_peak_layout.setVerticalSpacing(6)

            self.temporary_peak_file_edit = QtWidgets.QLineEdit(
                str(self._settings.manual_peak_file or "").strip()
            )
            self.temporary_peak_file_edit.setReadOnly(True)
            self.temporary_peak_file_edit.setClearButtonEnabled(False)
            self.temporary_peak_file_edit.setPlaceholderText(
                "未导入卡峰文件，将使用自动寻峰"
            )
            self.temporary_peak_file_edit.setToolTip(
                "支持 YAML、CSV 和 Excel 卡峰文件；路径只保存在本次分析会话中"
            )
            self.import_temporary_peak_button = QtWidgets.QPushButton(
                "导入卡峰文件…"
            )
            self.import_temporary_peak_button.setObjectName("BrowseButton")
            self.import_temporary_peak_button.clicked.connect(
                self._select_temporary_peak_file
            )
            self.clear_temporary_peak_button = QtWidgets.QPushButton("使用自动寻峰")
            self.clear_temporary_peak_button.clicked.connect(
                self._clear_temporary_peak_file
            )
            self.temporary_peak_status_label = QtWidgets.QLabel()
            self.temporary_peak_status_label.setObjectName("HintLabel")
            self.temporary_peak_status_label.setWordWrap(True)

            temporary_peak_layout.addWidget(
                QtWidgets.QLabel("卡峰文件"),
                0,
                0,
            )
            temporary_peak_layout.addWidget(self.temporary_peak_file_edit, 0, 1)
            temporary_peak_layout.addWidget(
                self.import_temporary_peak_button,
                0,
                2,
            )
            temporary_peak_layout.addWidget(
                self.clear_temporary_peak_button,
                0,
                3,
            )
            temporary_peak_layout.addWidget(
                self.temporary_peak_status_label,
                1,
                1,
                1,
                3,
            )
            temporary_peak_layout.setColumnStretch(1, 1)
            layout.addWidget(temporary_peak_group)
            self._refresh_temporary_peak_status()

        self.shared_parameters_group = QtWidgets.QGroupBox(
            "通用参数（项目）"
            if scope == "project"
            else "通用参数"
        )
        shared_layout = QtWidgets.QHBoxLayout(self.shared_parameters_group)
        shared_layout.setContentsMargins(10, 8, 10, 8)
        shared_layout.setSpacing(12)
        self.shared_parameters_label = QtWidgets.QLabel(
            self._shared_parameters_summary(self._settings)
        )
        self.shared_parameters_label.setWordWrap(True)
        self.shared_parameters_label.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        shared_layout.addWidget(self.shared_parameters_label, 1)
        self.open_shared_parameters_button = QtWidgets.QPushButton(
            "前往项目管理修改"
            if scope == "project"
            else "编辑通用参数…"
        )
        self.open_shared_parameters_button.setObjectName("BrowseButton")
        if scope == "project":
            self.open_shared_parameters_button.clicked.connect(
                self._request_shared_parameters
            )
        else:
            self.open_shared_parameters_button.clicked.connect(
                self._edit_temporary_shared_parameters
            )
        shared_layout.addWidget(self.open_shared_parameters_button)
        layout.addWidget(self.shared_parameters_group)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText(
            "保存并应用到项目"
            if scope == "project"
            else "应用到本次分析"
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText(
            "取消"
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _shared_parameters_summary(self, settings: ProjectSettings) -> str:
        light_source = "IO" if settings.light_source == "io" else "Beam Current"
        peak_set = str(settings.active_peak_set_id or "").strip()
        peak_text = peak_set if peak_set else "未配置"
        kr_text = (
            f"可用（{len(settings.expansion_factors)} 组）"
            if settings.expansion_factors
            else "不可用"
        )
        summary = (
            f"质量定标 A/B/C：{settings.cal_a:.4g} / "
            f"{settings.cal_b:.4g} / {settings.cal_c:.4g}　"
            f"光强来源：{light_source}　"
        )
        if self.scope == "project":
            summary += f"卡峰集：{peak_text}　"
        return summary + f"Kr：{kr_text}"

    def _request_shared_parameters(self) -> None:
        self.reject()
        self.shared_parameters_requested.emit()

    def _edit_temporary_shared_parameters(self) -> None:
        pending_settings = deepcopy(self._settings)
        self.function_widget.apply_to_settings(pending_settings)
        dialog = TemporarySharedParametersDialog(
            pending_settings,
            self,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        self._settings = dialog.settings()
        self.function_widget.set_project_settings(self._settings)
        if self.page == "pie" and self._pie_multi_segment is not None:
            self.function_widget.set_pie_multi_segment_state(
                self._pie_multi_segment
            )
        self.shared_parameters_label.setText(
            self._shared_parameters_summary(self._settings)
        )

    def _temporary_peak_start_dir(self) -> str:
        current = Path(self.temporary_peak_file_edit.text().strip()).expanduser()
        if current.is_file():
            return str(current.parent)
        if current.is_dir():
            return str(current)
        return str(Path.home())

    def _validate_temporary_peak_file(self, path_value: str) -> tuple[str, int]:
        path = Path(path_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"卡峰文件不存在：{path}")
        ranges = load_peak_ranges(
            path,
            calibration=self._settings.to_calibration(),
        )
        if not ranges:
            raise ValueError("卡峰文件中没有可用的卡峰范围")
        return str(path.resolve()), len(ranges)

    def _select_temporary_peak_file(self) -> None:
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "导入本次分析使用的卡峰文件",
            self._temporary_peak_start_dir(),
            "卡峰文件 (*.yaml *.yml *.csv *.xlsx *.xls);;所有文件 (*)",
        )
        if not path:
            return
        try:
            resolved_path, peak_count = self._validate_temporary_peak_file(path)
        except (OSError, TypeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "卡峰文件不可用",
                str(exc),
            )
            return
        self.temporary_peak_file_edit.setText(resolved_path)
        self._refresh_temporary_peak_status(peak_count=peak_count)

    def _clear_temporary_peak_file(self) -> None:
        self.temporary_peak_file_edit.clear()
        self._refresh_temporary_peak_status()

    def _refresh_temporary_peak_status(self, *, peak_count: int | None = None) -> None:
        path_value = self.temporary_peak_file_edit.text().strip()
        if not path_value:
            self.temporary_peak_status_label.setText(
                "未导入卡峰文件：生成曲线时将使用自动寻峰。"
            )
            self.clear_temporary_peak_button.setEnabled(False)
            return
        path = Path(path_value).expanduser()
        suffix = f" · {peak_count} 个卡峰" if peak_count is not None else ""
        self.temporary_peak_status_label.setText(
            f"本次分析将使用：{path.name}{suffix}"
        )
        self.temporary_peak_status_label.setToolTip(str(path))
        self.clear_temporary_peak_button.setEnabled(True)

    def accept(self) -> None:
        edited = deepcopy(self._settings)
        self.function_widget.apply_to_settings(edited)
        if self.scope == "temporary" and self.page == "temperature":
            path_value = self.temporary_peak_file_edit.text().strip()
            if path_value:
                try:
                    resolved_path, _peak_count = self._validate_temporary_peak_file(
                        path_value
                    )
                except (OSError, TypeError, ValueError) as exc:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "卡峰文件不可用",
                        str(exc),
                    )
                    return
                edited.manual_peak_file = resolved_path
                if resolved_path != str(
                    self._original_settings.manual_peak_file or ""
                ).strip():
                    edited.active_peak_set_id = ""
                edited.temp_peak_source = "manual"
            else:
                edited.manual_peak_file = ""
                edited.active_peak_set_id = ""
                edited.temp_peak_source = "auto"

        scoped_settings = copy_analysis_settings_fields(
            edited,
            deepcopy(self._original_settings),
            self.page,
        )
        if self.scope == "temporary":
            for field_name in TEMPORARY_SHARED_PARAMETER_FIELDS:
                setattr(
                    scoped_settings,
                    field_name,
                    deepcopy(getattr(edited, field_name)),
                )
        if self.scope == "temporary" and self.page == "temperature":
            scoped_settings.manual_peak_file = edited.manual_peak_file
            scoped_settings.active_peak_set_id = edited.active_peak_set_id
            scoped_settings.temp_peak_source = edited.temp_peak_source
        self._settings = scoped_settings
        super().accept()

    def settings(self) -> ProjectSettings:
        return deepcopy(self._settings)


class TemporaryAnalysisSettingsDialog(AnalysisSettingsDialog):
    """Compatibility wrapper for older temporary-parameter call sites."""

    def __init__(
        self,
        settings: ProjectSettings,
        parent=None,
        *,
        initial_tab: str | None = None,
        visible_function_pages: tuple[str, ...] | None = None,
        initial_function_page: str | None = None,
    ) -> None:
        page = (
            initial_function_page
            if initial_function_page in ANALYSIS_PARAMETER_FIELDS
            else None
        )
        if page is not None:
            super().__init__(
                settings,
                parent,
                scope="temporary",
                page=page,
            )
            return

        QtWidgets.QDialog.__init__(self, parent)
        self._settings = deepcopy(settings)
        self.setWindowTitle("编辑临时数据参数")
        self.resize(1040, 720)

        layout = QtWidgets.QVBoxLayout(self)
        notice = QtWidgets.QLabel(
            "这里编辑的是临时数据的独立参数副本；无需创建项目，"
            "修改只用于本次分析，不会写回项目配置。"
        )
        notice.setWordWrap(True)
        notice.setObjectName("HintLabel")
        layout.addWidget(notice)

        self.tabs = QtWidgets.QTabWidget()
        normalization = self._settings.to_normalization_settings()
        self.common_widget = CommonParametersWidget(
            normalization,
            self._settings.to_calibration(),
            self,
            show_actions=False,
        )
        self.common_widget.set_project_settings(self._settings)
        self.function_widget = FunctionDefaultsWidget(
            self,
            show_actions=False,
            visible_pages=visible_function_pages,
        )
        self.function_widget.set_project_settings(self._settings)
        self.tabs.addTab(self.common_widget, "通用处理参数")
        self.tabs.addTab(self.function_widget, "功能参数")
        layout.addWidget(self.tabs, 1)

        if initial_tab == "function":
            self.tabs.setCurrentIndex(1)
        if initial_function_page:
            self.function_widget.set_current_page(initial_function_page)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("应用到本次分析")
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self) -> None:
        if hasattr(self, "_original_settings"):
            super().accept()
            return
        self.common_widget.apply_to_settings(self._settings)
        self.function_widget.apply_to_settings(self._settings)
        QtWidgets.QDialog.accept(self)

    def settings(self) -> ProjectSettings:
        return deepcopy(self._settings)
