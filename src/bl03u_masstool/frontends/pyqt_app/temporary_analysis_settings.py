"""Session-only settings for analyses run against temporary data sources."""

from __future__ import annotations

from copy import deepcopy

from PyQt6 import QtWidgets

from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.core.normalization import NormalizationSettings
from bl03u_masstool.core.project_settings import ProjectSettings, migrate_from_legacy_configs
from bl03u_masstool.frontends.pyqt_app.normalization.widget import (
    CommonParametersWidget,
    FunctionDefaultsWidget,
)


def build_temporary_settings(
    source: str,
    project_settings: ProjectSettings | None = None,
    *,
    runtime_calibration: Calibration | None = None,
    runtime_normalization: NormalizationSettings | None = None,
) -> ProjectSettings:
    """Return an isolated settings snapshot initialized from ``source``."""
    if source == "project" and project_settings is not None:
        return deepcopy(project_settings)
    if source == "global":
        settings = migrate_from_legacy_configs()
        if runtime_calibration is not None:
            settings.cal_a = runtime_calibration.a
            settings.cal_b = runtime_calibration.b
            settings.cal_c = runtime_calibration.c
        if runtime_normalization is not None:
            settings.light_source = runtime_normalization.light_source
            settings.temperature_photon_normalize = runtime_normalization.temperature_photon_normalize
            settings.temperature_kr_correct = runtime_normalization.temperature_kr_correct
            settings.pie_photon_mode = runtime_normalization.pie_photon_mode
            settings.mass_discrimination = runtime_normalization.mass_discrimination
            settings.kr_calibration_folder = runtime_normalization.kr_calibration_folder
            settings.kr_calibration_peak_file = runtime_normalization.kr_calibration_peak_file
            settings.kr_mz = runtime_normalization.kr_mz
            settings.expansion_factors = dict(runtime_normalization.expansion_factors)
            settings.selected_elements = list(runtime_normalization.selected_elements)
        return settings
    return ProjectSettings()


class TemporaryAnalysisSettingsDialog(QtWidgets.QDialog):
    """Edit a ProjectSettings snapshot without persisting any changes."""

    def __init__(
        self,
        settings: ProjectSettings,
        parent=None,
        *,
        initial_tab: str | None = None,
        visible_function_pages: tuple[str, ...] | None = None,
        initial_function_page: str | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = deepcopy(settings)
        self.setWindowTitle("编辑临时数据参数")
        self.resize(1040, 720)

        layout = QtWidgets.QVBoxLayout(self)
        notice = QtWidgets.QLabel(
            "临时数据默认使用默认参数。这里的修改只用于本次分析，不会写回项目配置。"
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
            persist_changes=False,
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
        self.common_widget.apply_to_settings(self._settings)
        self.function_widget.apply_to_settings(self._settings)
        super().accept()

    def settings(self) -> ProjectSettings:
        return deepcopy(self._settings)
