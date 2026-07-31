from __future__ import annotations

from copy import deepcopy
import re

from PyQt6 import QtCore, QtGui, QtWidgets

from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.core.config import (
    PeakDetectionConfig,
    resolve_species_database_path,
)



class AutoSelectDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """QDoubleSpinBox that auto-selects all text when focused"""

    _FLOAT_PATTERN = re.compile(
        r"(?<![A-Za-z0-9_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])"
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lineEdit().installEventFilter(self)

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.selectAll()

    def eventFilter(self, watched, event):
        if (
            watched is self.lineEdit()
            and event.type() == QtCore.QEvent.Type.KeyPress
            and self._is_paste_event(event)
        ):
            self._paste_clipboard_text()
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self._is_paste_event(event):
            self._paste_clipboard_text()
            event.accept()
            return
        super().keyPressEvent(event)

    def _is_paste_event(self, event: QtGui.QKeyEvent) -> bool:
        modifiers = event.modifiers()
        return event.matches(QtGui.QKeySequence.StandardKey.Paste) or (
            event.key() == QtCore.Qt.Key.Key_V
            and bool(
                modifiers
                & (
                    QtCore.Qt.KeyboardModifier.ControlModifier
                    | QtCore.Qt.KeyboardModifier.MetaModifier
                )
            )
        )

    def _paste_clipboard_text(self) -> None:
        value = self._number_from_clipboard()
        if value is not None:
            self.setValue(value)
            self.selectAll()
            return
        self.lineEdit().paste()
        self.interpretText()

    def _number_from_clipboard(self) -> float | None:
        text = QtWidgets.QApplication.clipboard().text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            match = self._FLOAT_PATTERN.search(text)
            if match:
                return float(match.group(0))
        return None
from bl03u_masstool.core.elements import COMMON_ELEMENTS
from bl03u_masstool.core.normalization import (
    NormalizationSettings,
)
from bl03u_masstool.core.project_settings import (
    ProjectSettings,
    load_factory_project_settings,
)
from bl03u_masstool.core.temperature_scan import (
    compute_kr_expansion_factors,
)
from bl03u_masstool.core.mole_fraction import (
    MASS_DISCRIMINATION_PRESETS,
)
from bl03u_masstool.frontends.pyqt_app.workers import WorkerThread

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - only used when optional plotting is unavailable
    pg = None

from bl03u_masstool.frontends.pyqt_app.common.widgets import DataFrameTableMixin, combo_set_data


class CommonParametersWidget(QtWidgets.QWidget, DataFrameTableMixin):
    settings_saved = QtCore.pyqtSignal()

    def __init__(
        self,
        settings: NormalizationSettings,
        calibration: Calibration,
        parent=None,
        *,
        show_actions: bool = True,
        persist_changes: bool | None = None,
        show_calibration: bool = True,
        show_kr_expansion: bool = True,
        show_mass_response: bool = False,
        show_element_filter: bool = False,
        embedded: bool = False,
    ):
        super().__init__(parent)
        self.settings = settings
        self.calibration = calibration
        self.project_settings: ProjectSettings | None = None
        self.worker: WorkerThread | None = None
        self.show_actions = show_actions
        # Kept as an accepted keyword for callers from earlier releases. All
        # persistence is now handled by explicit project/session transactions.
        self.show_calibration = show_calibration
        self.show_kr_expansion = show_kr_expansion
        self.show_mass_response = show_mass_response
        self.show_element_filter = show_element_filter
        self.embedded = embedded
        self._kr_dependency_snapshot: tuple[object, ...] | None = None
        self._kr_factor_snapshot: dict = {}

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(
            0 if embedded else 8,
            0 if embedded else 8,
            0 if embedded else 8,
            0 if embedded or not show_actions else 8,
        )
        root.setSpacing(8)

        content_widget = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)

        top_grid = QtWidgets.QGridLayout()
        top_grid.setContentsMargins(0, 0, 0, 0)
        top_grid.setHorizontalSpacing(8)
        top_grid.setVerticalSpacing(8)
        normalization_section = self._build_normalization_section()
        if show_calibration:
            top_grid.addWidget(normalization_section, 0, 0)
            top_grid.addWidget(self._build_calibration_section(), 0, 1)
        else:
            top_grid.addWidget(normalization_section, 0, 0, 1, 2)
        if show_mass_response:
            top_grid.addWidget(self._build_mass_discrimination_section(), 1, 0)
        if show_element_filter:
            top_grid.addWidget(self._build_element_filter_section(), 1, 1)
        top_grid.setColumnStretch(0, 2)
        top_grid.setColumnStretch(1, 3)
        content_layout.addLayout(top_grid)
        if show_kr_expansion:
            content_layout.addWidget(self._build_kr_expansion_section())
        if not embedded:
            content_layout.addStretch(1)

        if embedded:
            # Embedded project cards are already hosted by a page-sized layout.
            # Avoid a nested scroll area: it carries an expanding viewport and
            # can vertically centre short content when adjacent pages are moved.
            root.addWidget(content_widget)
        else:
            # Standalone dialogs retain their own scroll area for compact
            # windows and the optional calibration/Kr sections.
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("QScrollArea { border: none; }")
            scroll.setWidget(content_widget)
            root.addWidget(scroll, 1)

        # 底部工具栏 - 统一操作栏
        self.action_bar = QtWidgets.QWidget()
        self.action_bar.setObjectName("ProjectActionBar")
        bl = QtWidgets.QHBoxLayout(self.action_bar)
        bl.setContentsMargins(8, 4, 8, 4)
        bl.setSpacing(8)

        self.status_label = QtWidgets.QLabel("已保存")
        self.status_label.setObjectName("ProjectStatus")
        self.status_label.setMaximumWidth(300)

        self.reset_button = QtWidgets.QPushButton("恢复已保存值")
        self.reset_button.setFixedHeight(30)
        self.reset_button.clicked.connect(self.load_from_settings)

        self.save_button = QtWidgets.QPushButton("💾  保存通用参数")
        self.save_button.setFixedHeight(30)
        self.save_button.clicked.connect(self.save_settings)

        bl.addWidget(self.status_label, 1)
        bl.addWidget(self.reset_button)
        bl.addWidget(self.save_button)
        self.action_bar.setVisible(show_actions)
        root.addWidget(self.action_bar)

        self.load_from_settings()

    # ── Section 1: 归一化参数 ────────────────────────────────────
    def _build_normalization_section(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("归一化参数")
        layout = QtWidgets.QGridLayout(group)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        self.light_source_combo = QtWidgets.QComboBox()
        self.light_source_combo.addItem("IO 光电流", "io")
        self.light_source_combo.addItem("Beam Current 储存环束流", "beam_current")

        row = 0
        layout.addWidget(QtWidgets.QLabel("光强来源:"), row, 0)
        layout.addWidget(self.light_source_combo, row, 1)

        layout.setColumnStretch(1, 1)
        return group

    # ── Section 2: 质量数定标 ────────────────────────────────────
    def _build_calibration_section(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("质量数定标  m/z = A·x² + B·x + C")
        layout = QtWidgets.QGridLayout(group)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        self.calibration_a_edit = AutoSelectDoubleSpinBox()
        self.calibration_b_edit = AutoSelectDoubleSpinBox()
        self.calibration_c_edit = AutoSelectDoubleSpinBox()

        for edit in (self.calibration_a_edit, self.calibration_b_edit, self.calibration_c_edit):
            edit.setRange(-1_000_000, 1_000_000)
            # Preserve fitted/project calibration coefficients when they make a
            # round trip through the project page.  Twelve fixed decimals leave
            # only about six significant digits for the quadratic coefficient.
            edit.setDecimals(18)
            edit.setSingleStep(0.000000001)
            edit.setMinimumWidth(128)
            edit.setMaximumWidth(180)
            # 隐藏上下箭头，允许直接修改数值
            edit.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)

        # 失焦只更新此编辑组件持有的草稿，不触发 Manager 或文件写入。
        self.calibration_a_edit.editingFinished.connect(self._sync_calibration_to_project_settings)
        self.calibration_b_edit.editingFinished.connect(self._sync_calibration_to_project_settings)
        self.calibration_c_edit.editingFinished.connect(self._sync_calibration_to_project_settings)

        layout.addWidget(QtWidgets.QLabel("A（二次项）"), 0, 0)
        layout.addWidget(QtWidgets.QLabel("B（一次项）"), 0, 1)
        layout.addWidget(QtWidgets.QLabel("C（常数项）"), 0, 2)
        layout.addWidget(self.calibration_a_edit, 1, 0)
        layout.addWidget(self.calibration_b_edit, 1, 1)
        layout.addWidget(self.calibration_c_edit, 1, 2)

        for col in range(3):
            layout.setColumnStretch(col, 1)
        return group

    # ── Section 3: 摩尔分数质量响应校正 ────────────────────────────
    def _build_mass_discrimination_section(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("摩尔分数质量响应校正")
        layout = QtWidgets.QGridLayout(group)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        # 实验条件预设
        self.mf_md_preset_combo = QtWidgets.QComboBox()
        for name in MASS_DISCRIMINATION_PRESETS:
            self.mf_md_preset_combo.addItem(name, name)
        self.mf_md_preset_combo.currentTextChanged.connect(self._on_mf_md_preset_changed)

        # 指数n
        self.mf_mass_disc_exponent_edit = QtWidgets.QDoubleSpinBox()
        self.mf_mass_disc_exponent_edit.setRange(0.0, 2.0)
        self.mf_mass_disc_exponent_edit.setDecimals(5)
        self.mf_mass_disc_exponent_edit.setMaximumWidth(120)

        # 公式标签
        formula_label = QtWidgets.QLabel("响应因子公式: Dᵢ = (MW / 30)ⁿ")
        formula_label.setObjectName("FormulaLabel")

        row = 0
        layout.addWidget(QtWidgets.QLabel("实验条件预设:"), row, 0)
        layout.addWidget(self.mf_md_preset_combo, row, 1)

        row += 1
        layout.addWidget(QtWidgets.QLabel("指数 n:"), row, 0)
        layout.addWidget(self.mf_mass_disc_exponent_edit, row, 1)

        row += 1
        layout.addWidget(formula_label, row, 0, 1, 2)

        layout.setColumnStretch(1, 1)
        return group

    # ── Section 4: Kr 膨胀校正 ────────────────────────────────────
    def _build_kr_expansion_section(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Kr 膨胀校正")
        group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # 扫描文件夹
        kr_form = QtWidgets.QGridLayout()
        kr_form.setHorizontalSpacing(8)
        kr_form.setVerticalSpacing(6)
        self.kr_folder_edit = QtWidgets.QLineEdit()
        self.kr_folder_edit.setPlaceholderText("包含温度扫描光谱文件的目录 (*.txt)")
        self.kr_folder_button = QtWidgets.QPushButton("浏览…")
        self.kr_folder_button.setFixedWidth(72)
        self.kr_folder_button.clicked.connect(self.select_kr_folder)
        kr_form.addWidget(QtWidgets.QLabel("扫描文件夹:"), 0, 0)
        kr_form.addWidget(self.kr_folder_edit, 0, 1)
        kr_form.addWidget(self.kr_folder_button, 0, 2)

        # 卡峰模式
        peak_mode_layout = QtWidgets.QHBoxLayout()
        peak_mode_layout.setContentsMargins(0, 0, 0, 0)
        peak_mode_layout.setSpacing(12)
        self.kr_peak_mode_auto_radio = QtWidgets.QRadioButton("自动卡峰")
        self.kr_peak_mode_manual_radio = QtWidgets.QRadioButton("手动卡峰文件")
        self.kr_peak_mode_auto_radio.setChecked(True)
        self.kr_peak_mode_auto_radio.toggled.connect(self.on_kr_peak_mode_changed)
        peak_mode_layout.addWidget(self.kr_peak_mode_auto_radio, 0)
        peak_mode_layout.addWidget(self.kr_peak_mode_manual_radio, 0)
        peak_mode_layout.addStretch(1)
        kr_form.addWidget(QtWidgets.QLabel("卡峰方式:"), 1, 0)
        kr_form.addLayout(peak_mode_layout, 1, 1, 1, 2)

        # Kr质量数选择
        kr_mz_layout = QtWidgets.QHBoxLayout()
        kr_mz_layout.setContentsMargins(0, 0, 0, 0)
        kr_mz_layout.setSpacing(8)
        self.kr_mz_combo = QtWidgets.QComboBox()
        # Kr同位素丰度: 78(0.355%), 80(2.286%), 82(11.593%), 83(11.500%), 84(56.987%), 86(17.279%)
        self.kr_mz_combo.addItems(["78", "80", "82", "83", "84", "86"])
        self.kr_mz_combo.setCurrentText("84")
        self.kr_mz_combo.setMaximumWidth(80)
        kr_mz_layout.addWidget(QtWidgets.QLabel("Kr质量数:"), 0)
        kr_mz_layout.addWidget(self.kr_mz_combo, 0)
        kr_mz_layout.addStretch(1)
        kr_form.addWidget(QtWidgets.QLabel("质量数:"), 2, 0)
        kr_form.addLayout(kr_mz_layout, 2, 1, 1, 2)

        # 卡峰文件
        self.kr_peak_file_edit = QtWidgets.QLineEdit()
        self.kr_peak_file_edit.setPlaceholderText("可选：卡峰结果文件 (peak_ranges-*.csv, *.yaml, *.yml)")
        self.kr_peak_file_edit.setEnabled(False)
        self.kr_peak_file_button = QtWidgets.QPushButton("选择…")
        self.kr_peak_file_button.setFixedWidth(72)
        self.kr_peak_file_button.setEnabled(False)
        self.kr_peak_file_button.clicked.connect(self.select_kr_peak_file)
        kr_form.addWidget(QtWidgets.QLabel("卡峰文件:"), 3, 0)
        kr_form.addWidget(self.kr_peak_file_edit, 3, 1)
        kr_form.addWidget(self.kr_peak_file_button, 3, 2)

        # 计算按钮和能量选择
        compute_layout = QtWidgets.QHBoxLayout()
        compute_layout.setContentsMargins(0, 0, 0, 0)
        compute_layout.setSpacing(8)
        self.compute_kr_button = QtWidgets.QPushButton("▶  计算 λ(T)")
        self.compute_kr_button.setFixedWidth(160)
        self.compute_kr_button.clicked.connect(self.compute_kr_factors)
        self.energy_combo = QtWidgets.QComboBox()
        self.energy_combo.setMinimumWidth(160)
        self.energy_combo.setMaximumWidth(240)
        self.energy_combo.setVisible(False)
        self.energy_combo.currentIndexChanged.connect(self.on_energy_selected)
        compute_layout.addWidget(self.compute_kr_button, 0)
        compute_layout.addWidget(QtWidgets.QLabel("能量选择:"), 0)
        compute_layout.addWidget(self.energy_combo, 0)
        compute_layout.addStretch(1)
        kr_form.addWidget(QtWidgets.QLabel("计算:"), 4, 0)
        kr_form.addLayout(compute_layout, 4, 1, 1, 2)
        kr_form.setColumnStretch(1, 1)
        layout.addLayout(kr_form)

        self.factor_table = QtWidgets.QTableWidget()
        self.factor_table.setColumnCount(3)
        self.factor_table.setHorizontalHeaderLabels(["温度 (°C)", "Kr 信号积分", "λ(T)"])
        self.factor_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.factor_table.setAlternatingRowColors(True)
        self.factor_table.setMinimumHeight(180)
        layout.addWidget(self.factor_table, 1)

        return group

    # ── Section 5: 元素筛选 ────────────────────────────────────
    def _build_element_filter_section(self) -> QtWidgets.QGroupBox:
        from bl03u_masstool.core.elements import COMMON_ELEMENTS

        group = QtWidgets.QGroupBox("元素筛选（影响PIE物种匹配范围）")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # 元素复选框（紧凑布局）
        elements_layout = QtWidgets.QGridLayout()
        elements_layout.setHorizontalSpacing(6)
        elements_layout.setVerticalSpacing(4)

        self.element_checks = {}
        for i, elem in enumerate(COMMON_ELEMENTS):
            row = i // 6
            col = i % 6
            check = QtWidgets.QCheckBox(elem)
            check.setMaximumWidth(60)
            self.element_checks[elem] = check
            elements_layout.addWidget(check, row, col)

        layout.addLayout(elements_layout)
        return group

    def load_from_settings(self) -> None:
        # 当有 ProjectSettings 时，优先从 ProjectSettings 加载项目特定参数
        # 这确保项目的参数设置在项目关闭/重新打开时被保留
        if self.project_settings:
            combo_set_data(self.light_source_combo, self.project_settings.light_source)
            if self.show_kr_expansion:
                self.kr_folder_edit.setText(self.project_settings.kr_calibration_folder)
                self.kr_peak_file_edit.setText(self.project_settings.kr_calibration_peak_file)
                self.kr_mz_combo.setCurrentText(str(self.project_settings.kr_mz))
            self.settings.expansion_factors = dict(self.project_settings.expansion_factors)
            self.settings.selected_elements = list(self.project_settings.selected_elements)
            if self.show_kr_expansion:
                use_manual_peak = bool(self.project_settings.kr_calibration_peak_file.strip())
                self.kr_peak_mode_manual_radio.setChecked(use_manual_peak)
                self.kr_peak_mode_auto_radio.setChecked(not use_manual_peak)
        else:
            # 无项目时，从本地 NormalizationSettings 加载
            combo_set_data(self.light_source_combo, self.settings.light_source)
            if self.show_kr_expansion:
                self.kr_folder_edit.setText(self.settings.kr_calibration_folder)
                self.kr_peak_file_edit.setText(self.settings.kr_calibration_peak_file)
                self.kr_mz_combo.setCurrentText(str(self.settings.kr_mz))
                use_manual_peak = bool(self.settings.kr_calibration_peak_file.strip())
                self.kr_peak_mode_manual_radio.setChecked(use_manual_peak)
                self.kr_peak_mode_auto_radio.setChecked(not use_manual_peak)

        if self.show_kr_expansion:
            self.on_kr_peak_mode_changed()

        # 加载校准参数；项目模式优先使用 project.yaml 中的定标值
        if self.show_calibration:
            if self.project_settings:
                self.calibration_a_edit.setValue(self.project_settings.cal_a)
                self.calibration_b_edit.setValue(self.project_settings.cal_b)
                self.calibration_c_edit.setValue(self.project_settings.cal_c)
            else:
                calibration = load_factory_project_settings().to_calibration()
                self.calibration_a_edit.setValue(calibration.a)
                self.calibration_b_edit.setValue(calibration.b)
                self.calibration_c_edit.setValue(calibration.c)

        # 模块专属参数仅供旧兼容入口显示；项目管理已移入对应模块页。
        if self.show_mass_response and self.project_settings:
            # 恢复预设选择
            if self.project_settings.mf_md_preset:
                if (
                    self.mf_md_preset_combo.findData(
                        self.project_settings.mf_md_preset
                    )
                    < 0
                ):
                    self.mf_md_preset_combo.addItem(
                        self.project_settings.mf_md_preset,
                        self.project_settings.mf_md_preset,
                    )
                combo_set_data(self.mf_md_preset_combo, self.project_settings.mf_md_preset)
            self.mf_mass_disc_exponent_edit.setValue(self.project_settings.mf_mass_disc_exponent)
        if self.show_kr_expansion:
            self._reset_kr_factor_display()
        if self.show_element_filter:
            selected = (
                self.project_settings.selected_elements
                if self.project_settings
                else self.settings.selected_elements
            )
            for elem, chk in self.element_checks.items():
                chk.setChecked(not selected or elem in selected)
        self._remember_kr_factor_state()

    def set_project_settings(self, project_settings: ProjectSettings) -> None:
        """Set the project settings reference for parameter synchronization."""
        self.project_settings = project_settings
        if project_settings:
            self.load_from_settings()

    def _kr_dependency_signature(
        self,
        target: ProjectSettings | None = None,
    ) -> tuple[object, ...]:
        settings_source = target or self.project_settings
        calibration_values: tuple[object, ...]
        if self.show_calibration:
            calibration_values = (
                self.calibration_a_edit.value(),
                self.calibration_b_edit.value(),
                self.calibration_c_edit.value(),
            )
        elif settings_source is not None:
            calibration_values = (
                settings_source.cal_a,
                settings_source.cal_b,
                settings_source.cal_c,
            )
        else:
            calibration_values = (
                self.calibration.a,
                self.calibration.b,
                self.calibration.c,
            )

        kr_values: tuple[object, ...]
        if self.show_kr_expansion:
            manual_peak_file = (
                self.kr_peak_file_edit.text().strip()
                if self.kr_peak_mode_manual_radio.isChecked()
                else ""
            )
            kr_values = (
                self.kr_folder_edit.text().strip(),
                manual_peak_file,
                int(self.kr_mz_combo.currentText()),
            )
        elif settings_source is not None:
            kr_values = (
                settings_source.kr_calibration_folder,
                settings_source.kr_calibration_peak_file,
                settings_source.kr_mz,
            )
        else:
            kr_values = (
                self.settings.kr_calibration_folder,
                self.settings.kr_calibration_peak_file,
                self.settings.kr_mz,
            )

        return (
            str(self.light_source_combo.currentData() or ""),
            *calibration_values,
            *kr_values,
        )

    def _remember_kr_factor_state(self) -> None:
        self._kr_dependency_snapshot = self._kr_dependency_signature()
        self._kr_factor_snapshot = deepcopy(self.settings.expansion_factors)

    def _reset_kr_factor_display(self) -> None:
        self._kr_display_factors = deepcopy(self.settings.expansion_factors)
        self._kr_signal_data = {}
        for attribute_name in ("_kr_result_df", "_energy_group_map"):
            if hasattr(self, attribute_name):
                delattr(self, attribute_name)
        self.energy_combo.blockSignals(True)
        self.energy_combo.clear()
        self.energy_combo.blockSignals(False)
        self.energy_combo.setVisible(False)
        self.refresh_factor_table()

    def _invalidate_stale_kr_factors(
        self,
        target: ProjectSettings | None = None,
    ) -> bool:
        inputs_changed = (
            self._kr_dependency_snapshot is not None
            and self._kr_dependency_signature(target)
            != self._kr_dependency_snapshot
        )
        factors_recomputed = (
            self.settings.expansion_factors != self._kr_factor_snapshot
        )
        if not inputs_changed or factors_recomputed:
            return False

        had_factors = bool(self.settings.expansion_factors)
        self.settings.expansion_factors = {}
        if self.show_kr_expansion:
            self._reset_kr_factor_display()
        if had_factors:
            self.status_label.setText("Kr 输入已变化，请重新计算 λ(T)")
        return had_factors

    def apply_to_settings(
        self,
        project_settings: ProjectSettings | None = None,
    ) -> bool:
        target = project_settings or self.project_settings
        if project_settings is not None:
            self.project_settings = project_settings
        kr_factors_invalidated = self._invalidate_stale_kr_factors(target)
        self.settings.light_source = self.light_source_combo.currentData()
        self.settings.mass_discrimination = 1.0
        if self.show_kr_expansion:
            self.settings.kr_calibration_folder = self.kr_folder_edit.text().strip()
            self.settings.kr_mz = int(self.kr_mz_combo.currentText())
            if self.kr_peak_mode_manual_radio.isChecked():
                self.settings.kr_calibration_peak_file = self.kr_peak_file_edit.text().strip()
            else:
                self.settings.kr_calibration_peak_file = ""
        if self.show_calibration:
            self.calibration = Calibration(
                a=self.calibration_a_edit.value(),
                b=self.calibration_b_edit.value(),
                c=self.calibration_c_edit.value(),
            )
        # 同步所有项目特定参数到 ProjectSettings
        if target:
            target.light_source = self.settings.light_source
            if self.show_kr_expansion:
                target.kr_calibration_folder = self.settings.kr_calibration_folder
                target.kr_mz = int(self.kr_mz_combo.currentText())
                if self.kr_peak_mode_manual_radio.isChecked():
                    target.kr_calibration_peak_file = self.kr_peak_file_edit.text().strip()
                else:
                    target.kr_calibration_peak_file = ""
            target.expansion_factors = dict(self.settings.expansion_factors)
            if self.show_calibration:
                target.cal_a = self.calibration.a
                target.cal_b = self.calibration.b
                target.cal_c = self.calibration.c
            if self.show_mass_response:
                target.mf_md_preset = self.mf_md_preset_combo.currentText()
                target.mf_mass_disc_exponent = (
                    self.mf_mass_disc_exponent_edit.value()
                )
        if self.show_element_filter:
            self.settings.selected_elements = [
                elem for elem, chk in self.element_checks.items() if chk.isChecked()
            ]
            if target:
                target.selected_elements = list(self.settings.selected_elements)
        self._remember_kr_factor_state()
        return kr_factors_invalidated

    def _sync_calibration_to_project_settings(self) -> None:
        """Update only the in-memory editor draft.

        Persistence belongs to the project-page transaction.  Losing focus in
        this widget must never write a machine-global or project configuration.
        """
        calibration = Calibration(
            a=self.calibration_a_edit.value(),
            b=self.calibration_b_edit.value(),
            c=self.calibration_c_edit.value(),
        )
        self.calibration = calibration
        if self.project_settings is not None:
            self.project_settings.cal_a = calibration.a
            self.project_settings.cal_b = calibration.b
            self.project_settings.cal_c = calibration.c

    def _set_all_elements(self, checked: bool):
        for chk in self.element_checks.values():
            chk.setChecked(checked)

    def on_kr_peak_mode_changed(self):
        """卡峰模式切换时，启用/禁用手动卡峰文件选择"""
        use_manual = self.kr_peak_mode_manual_radio.isChecked()
        self.kr_peak_file_edit.setEnabled(use_manual)
        self.kr_peak_file_button.setEnabled(use_manual)
        # 自动模式时清空卡峰文件路径
        if not use_manual:
            self.kr_peak_file_edit.clear()

    def select_kr_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "选择Kr定标文件夹")
        if folder:
            self.kr_folder_edit.setText(folder)

    def select_kr_peak_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择Kr手动卡峰文件",
            "",
            "Peak Files (*.yaml *.yml *.csv *.xlsx *.xls);;All Files (*)",
        )
        if path:
            self.kr_peak_file_edit.setText(path)

    def save_settings(self):
        kr_factors_invalidated = self.apply_to_settings()
        if not kr_factors_invalidated:
            self.status_label.setText("已更新当前草稿，尚未保存到项目")
        self.settings_saved.emit()

    def compute_kr_factors(self):
        self.apply_to_settings()
        folder = self.settings.kr_calibration_folder
        if not folder:
            QtWidgets.QMessageBox.warning(self, "提示", "请先选择Kr定标文件夹")
            return

        # 验证文件夹存在且包含光谱文件
        from pathlib import Path
        folder_path = Path(folder)
        if not folder_path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "错误",
                f"指定的文件夹不存在：\n{folder}"
            )
            return

        spectrum_files = list(folder_path.rglob("*.txt")) + list(folder_path.rglob("*.asc"))
        if not spectrum_files:
            QtWidgets.QMessageBox.warning(
                self,
                "错误",
                f"文件夹中没有找到光谱文件 (*.txt, *.asc)：\n{folder}"
            )
            return

        peak_config = (
            self.project_settings.to_peak_detection_config()
            if self.project_settings
            else load_factory_project_settings().to_peak_detection_config()
        )
        self.set_busy(True, "正在计算Kr膨胀系数...")
        self.worker = WorkerThread(
            lambda: compute_kr_expansion_factors(
                folder,
                calibration=self.calibration,
                kr_mz=self.settings.kr_mz,
                manual_peak_path=self.settings.kr_calibration_peak_file or None,
                light_source=self.settings.light_source,
                threshold_end=peak_config.threshold_end,
                min_intensity=peak_config.min_intensity,
                detection_min_idx=peak_config.detection_min_idx,
                nearby_peak_window=peak_config.nearby_peak_window,
                duplicate_window=peak_config.duplicate_window,
                weak_tail_early_window=peak_config.weak_tail_early_window,
                weak_tail_late_window=peak_config.weak_tail_late_window,
                weak_tail_ratio=peak_config.weak_tail_ratio,
                gaussian_window_max=peak_config.gaussian_window_max,
                gaussian_boundary_scale=peak_config.gaussian_boundary_scale,
                boundary_padding=peak_config.boundary_padding,
            ),
            self,
        )
        self.worker.finished_with_result.connect(self.on_kr_factors_ready)
        self.worker.failed.connect(self.on_kr_factors_failed)
        self.worker.finished.connect(lambda: self.set_busy(False, "就绪"))
        self.worker.start()

    def set_busy(self, busy: bool, message: str) -> None:
        self.status_label.setText(message)
        for widget in (
            self.compute_kr_button,
            self.save_button,
            self.kr_folder_button,
            self.kr_peak_file_button,
            self.kr_peak_mode_auto_radio,
            self.kr_peak_mode_manual_radio,
        ):
            widget.setDisabled(busy)

    def on_kr_factors_ready(self, result: object) -> None:
        df = result
        import pandas as pd
        from bl03u_masstool.core.mole_fraction import parse_expansion_factors_from_result
        from bl03u_masstool.core.temperature_scan import group_energies_by_tolerance

        # 保存原始结果用于能量选择
        self._kr_result_df = df

        # 检查是否为多能量数据
        is_multi_energy = "photon_energy" in df.columns and df["photon_energy"].nunique() > 1

        if is_multi_energy:
            # 多能量：计算平均结果
            avg_data = []
            for temp in sorted(df["temperature"].unique()):
                temp_data = df[df["temperature"] == temp]
                avg_lambda = float(temp_data["expansion_lambda"].mean())
                avg_signal = float(temp_data["kr_signal"].mean())
                avg_data.append({
                    "temperature": float(temp),
                    "kr_signal": avg_signal,
                    "expansion_lambda": avg_lambda,
                })
            avg_df = pd.DataFrame(avg_data)
            self.settings.expansion_factors = parse_expansion_factors_from_result(df)
            avg_display_factors = parse_expansion_factors_from_result(avg_df)
            self._kr_display_factors = dict(avg_display_factors)

            # 按误差容忍度将能量值分组
            all_energies = sorted(df["photon_energy"].unique().tolist())
            energy_groups = group_energies_by_tolerance(all_energies, tolerance=0.01)

            # 填充能量下拉列表（按能量升序）
            self.energy_combo.blockSignals(True)
            self.energy_combo.clear()
            self.energy_combo.addItem("平均结果", None)

            for center_energy in sorted(energy_groups.keys()):
                self.energy_combo.addItem(f"E={center_energy:.4f} eV", center_energy)

            self.energy_combo.blockSignals(False)
            self.energy_combo.setVisible(True)

            # 保存能量组映射用于后续查询
            self._energy_group_map = energy_groups

            # 显示平均结果
            self._kr_signal_data = {
                float(row["temperature"]): float(row["kr_signal"])
                for _, row in avg_df.iterrows()
            }
            msg = f"已计算 {len(energy_groups)} 个能量点 × {len(df)//len(all_energies)} 个温度点，"
            msg += f"显示 {len(avg_display_factors)} 个温度点的平均膨胀系数"
        else:
            # 单能量
            self.settings.expansion_factors = parse_expansion_factors_from_result(df)
            self._kr_display_factors = dict(self.settings.expansion_factors)
            self._kr_signal_data = {
                float(row["temperature"]): float(row["kr_signal"])
                for _, row in df.iterrows()
            }
            self.energy_combo.setVisible(False)
            msg = f"已计算 {len(self.settings.expansion_factors)} 个温度点的Kr膨胀系数"

        self.refresh_factor_table()
        if self.project_settings:
            self.project_settings.expansion_factors = dict(self.settings.expansion_factors)
        self._remember_kr_factor_state()
        self.status_label.setText(msg)

    def on_energy_selected(self, index: int) -> None:
        """能量选择改变时更新表格"""
        if not hasattr(self, '_kr_result_df'):
            return

        selected_center_energy = self.energy_combo.currentData()
        df = self._kr_result_df

        if selected_center_energy is None:
            # 显示平均结果
            result_data = []
            for temp in sorted(df["temperature"].unique()):
                temp_data = df[df["temperature"] == temp]
                avg_lambda = float(temp_data["expansion_lambda"].mean())
                avg_signal = float(temp_data["kr_signal"].mean())
                result_data.append({
                    "temperature": float(temp),
                    "kr_signal": avg_signal,
                    "expansion_lambda": avg_lambda,
                })
        else:
            # 显示特定能量组的结果
            # 获取该组包含的所有能量值
            group_energies = self._energy_group_map.get(selected_center_energy, [selected_center_energy])

            # 过滤数据，仅获取该组中的能量
            group_data = df[df["photon_energy"].isin(group_energies)]
            result_data = []
            for temp in sorted(group_data["temperature"].unique()):
                temp_data = group_data[group_data["temperature"] == temp]
                avg_lambda = float(temp_data["expansion_lambda"].mean())
                avg_signal = float(temp_data["kr_signal"].mean())
                result_data.append({
                    "temperature": float(temp),
                    "kr_signal": avg_signal,
                    "expansion_lambda": avg_lambda,
                })

        # 更新 Kr 信号数据用于显示
        self._kr_signal_data = {
            float(row["temperature"]): float(row["kr_signal"])
            for row in result_data
        }
        self._kr_display_factors = {
            float(row["temperature"]): float(row["expansion_lambda"])
            for row in result_data
        }

        self.refresh_factor_table()


    def on_kr_factors_failed(self, message: str) -> None:
        # 提供更详细的错误诊断
        error_msg = message
        if "all Kr calibration signals are zero" in error_msg or "No valid Kr signals found" in error_msg:
            error_msg += (
                "\n\n📋 可能的原因：\n"
                f"  1. Kr质量数设置错误（当前：{self.settings.kr_mz}）\n"
                f"  2. 光强来源设置错误（当前：{self.settings.light_source}）\n"
                f"  3. Kr定标文件夹路径错误（当前：{self.settings.kr_calibration_folder}）\n"
                f"  4. 光谱数据质量差或没有Kr信号\n\n"
                "💡 尝试：\n"
                f"  • 检查\"Kr质量数\"是否正确（常用：84, 86）\n"
                "  • 检查\"光强来源\"设置是否与光谱文件匹配\n"
                "  • 确认文件夹路径指向包含光谱文件的目录\n"
                "  • 使用包含足够Kr信号的能量点"
            )
        QtWidgets.QMessageBox.critical(self, "错误", error_msg)

    def refresh_factor_table(self) -> None:
        kr_signal = getattr(self, "_kr_signal_data", {})
        display_factors = getattr(self, "_kr_display_factors", self.settings.expansion_factors)
        self.factor_table.setRowCount(0)

        # 检测膨胀系数格式（单能量或多能量）
        if not display_factors:
            return

        first_key = next(iter(display_factors.keys()), None)
        if first_key is None:
            return

        is_multi_energy = isinstance(display_factors[first_key], dict)

        if is_multi_energy:
            # 多能量格式：{能量: {温度: 膨胀系数}}
            for energy in sorted(display_factors.keys()):
                temp_factors = display_factors[energy]
                for temp in sorted(temp_factors.keys()):
                    row = self.factor_table.rowCount()
                    self.factor_table.insertRow(row)
                    self.factor_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{int(temp)}"))
                    signal = kr_signal.get(temp, None)
                    sig_text = f"{signal:.4f}" if signal is not None else "-"
                    self.factor_table.setItem(row, 1, QtWidgets.QTableWidgetItem(sig_text))
                    lam = temp_factors[temp]
                    # 显示能量信息
                    energy_text = f"{lam:.6f} @{energy:.2f}eV"
                    self.factor_table.setItem(row, 2, QtWidgets.QTableWidgetItem(energy_text))
        else:
            # 单能量格式：{温度: 膨胀系数}
            for temp in sorted(display_factors.keys()):
                row = self.factor_table.rowCount()
                self.factor_table.insertRow(row)
                self.factor_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{int(temp)}"))
                signal = kr_signal.get(temp, None)
                sig_text = f"{signal:.4f}" if signal is not None else "-"
                self.factor_table.setItem(row, 1, QtWidgets.QTableWidgetItem(sig_text))
                lam = display_factors[temp]
                self.factor_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{lam:.6f}"))


    def _toggle_md_preview(self, checked: bool) -> None:
        if checked:
            self._preview_mf_mass_discrimination()
        else:
            self.mf_md_preview_table.setVisible(False)

    def _on_mf_md_preset_changed(self, text: str) -> None:
        if text in MASS_DISCRIMINATION_PRESETS:
            self.mf_mass_disc_exponent_edit.setValue(MASS_DISCRIMINATION_PRESETS[text])

    def _preview_mf_mass_discrimination(self) -> None:
        from bl03u_masstool.core.mole_fraction import calc_mass_discrimination
        from bl03u_masstool.core.pie_analysis import load_species_database
        db_path = resolve_species_database_path(
            self.project_settings.pics_database_path
            if self.project_settings is not None
            else None
        )
        if not db_path.exists():
            QtWidgets.QMessageBox.warning(self, "提示", "物种数据库未找到，请先在PIE鉴定中导入PICS")
            return
        database, _ = load_species_database(str(db_path))
        seen: set[tuple] = set()
        species_list = []
        for spec in database:
            key = (spec.get("species", spec.get("name", "")), spec["mz"])
            if key not in seen:
                seen.add(key)
                species_list.append(spec)
        n = self.mf_mass_disc_exponent_edit.value()
        self.mf_md_preview_table.setRowCount(0)
        for spec in species_list[:30]:
            row = self.mf_md_preview_table.rowCount()
            self.mf_md_preview_table.insertRow(row)
            name = spec.get("species", spec.get("name", ""))
            self.mf_md_preview_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.mf_md_preview_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(spec["mz"])))
            D_i = calc_mass_discrimination(spec["mz"], n)
            self.mf_md_preview_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{D_i:.6f}"))
        self.mf_md_preview_table.setVisible(True)


class PeakDetectionWidget(QtWidgets.QWidget):
    settings_saved = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.peak_detection = load_factory_project_settings().to_peak_detection_config()
        self.project_settings: ProjectSettings | None = None

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 单一面板 - 寻峰参数
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        peak_group = QtWidgets.QGroupBox("自动寻峰参数")
        peak_layout = QtWidgets.QGridLayout(peak_group)
        peak_layout.setHorizontalSpacing(8)
        peak_layout.setVerticalSpacing(8)

        self.peak_algorithm_combo = QtWidgets.QComboBox()
        self.peak_algorithm_combo.addItem("自适应高召回（实验）", "adaptive")
        self.peak_algorithm_combo.addItem("Ensemble 融合检测（推荐）", "ensemble")
        self.peak_algorithm_combo.addItem("Prominence", "prominence")
        self.peak_algorithm_combo.addItem("传统局部极大", "legacy")
        self.peak_algorithm_combo.addItem("CWT 小波", "cwt")
        self.peak_algorithm_combo.setToolTip("主工作台自动寻峰使用的算法")
        self.peak_detection_min_idx_edit = QtWidgets.QSpinBox()
        self.peak_detection_min_idx_edit.setRange(0, 10_000_000)
        self.peak_detection_min_idx_edit.setToolTip("自动寻峰从该数据点之后开始，避免文件头或低TOF噪声参与寻峰")
        self.peak_threshold_end_edit = QtWidgets.QDoubleSpinBox()
        self.peak_threshold_end_edit.setRange(0, 1_000_000)
        self.peak_threshold_end_edit.setDecimals(3)
        self.peak_threshold_end_edit.setToolTip("向峰两侧扩展边界时，低于该强度即认为到达峰结束")
        self.peak_min_intensity_edit = QtWidgets.QDoubleSpinBox()
        self.peak_min_intensity_edit.setRange(0, 1_000_000)
        self.peak_min_intensity_edit.setDecimals(3)
        self.peak_min_intensity_edit.setToolTip("低于该强度的局部极大值不会作为候选峰")
        self.peak_nearby_window_edit = QtWidgets.QSpinBox()
        self.peak_nearby_window_edit.setRange(0, 100_000)
        self.peak_nearby_window_edit.setToolTip("该半宽范围内若已有更高峰，则当前候选峰会被抑制")
        self.peak_duplicate_window_edit = QtWidgets.QSpinBox()
        self.peak_duplicate_window_edit.setRange(0, 100_000)
        self.peak_duplicate_window_edit.setToolTip("接受一个峰后，该半宽范围内的候选点会被视为同一峰")
        self.peak_boundary_padding_edit = QtWidgets.QSpinBox()
        self.peak_boundary_padding_edit.setRange(0, 100_000)
        self.peak_boundary_padding_edit.setToolTip("最终卡峰边界向左右额外扩展的数据点数")
        self.peak_weak_tail_ratio_edit = QtWidgets.QDoubleSpinBox()
        self.peak_weak_tail_ratio_edit.setRange(0.001, 1_000_000)
        self.peak_weak_tail_ratio_edit.setDecimals(3)
        self.peak_weak_tail_ratio_edit.setToolTip("弱肩峰过滤倍率；值越大，越容易保留主峰后的弱峰")
        self.peak_gaussian_window_max_edit = QtWidgets.QSpinBox()
        self.peak_gaussian_window_max_edit.setRange(3, 100_000)
        self.peak_gaussian_window_max_edit.setToolTip("自动高斯拟合时允许使用的最大半窗口")
        self.peak_gaussian_boundary_scale_edit = QtWidgets.QDoubleSpinBox()
        self.peak_gaussian_boundary_scale_edit.setRange(0.1, 100)
        self.peak_gaussian_boundary_scale_edit.setDecimals(3)
        self.peak_gaussian_boundary_scale_edit.setToolTip("高斯拟合成功后，以该倍数 FWHM 重新估计卡峰边界")
        self.peak_prominence_ratio_edit = QtWidgets.QDoubleSpinBox()
        self.peak_prominence_ratio_edit.setRange(0, 1)
        self.peak_prominence_ratio_edit.setDecimals(6)
        self.peak_prominence_ratio_edit.setSingleStep(0.001)
        self.peak_prominence_ratio_edit.setToolTip("Prominence阈值占基线校正后最高峰的比例；越大越保守")
        self.peak_smoothing_window_edit = QtWidgets.QSpinBox()
        self.peak_smoothing_window_edit.setRange(1, 100_001)
        self.peak_smoothing_window_edit.setToolTip("Savitzky-Golay平滑窗口，偶数会自动调为奇数")
        self.peak_baseline_window_edit = QtWidgets.QSpinBox()
        self.peak_baseline_window_edit.setRange(3, 1_000_001)
        self.peak_baseline_window_edit.setToolTip("滚动百分位基线窗口，通常应大于峰宽")
        self.peak_baseline_percentile_edit = QtWidgets.QDoubleSpinBox()
        self.peak_baseline_percentile_edit.setRange(0, 100)
        self.peak_baseline_percentile_edit.setDecimals(2)
        self.peak_baseline_percentile_edit.setToolTip("滚动基线使用的百分位数")
        self.peak_min_peak_width_edit = QtWidgets.QSpinBox()
        self.peak_min_peak_width_edit.setRange(1, 1_000_000)
        self.peak_min_peak_width_edit.setToolTip("Prominence/CWT允许的最小峰宽")
        self.peak_max_peak_width_edit = QtWidgets.QSpinBox()
        self.peak_max_peak_width_edit.setRange(1, 1_000_000)
        self.peak_max_peak_width_edit.setToolTip("Prominence/CWT允许的最大峰宽")

        peak_layout.addWidget(QtWidgets.QLabel("寻峰算法"), 0, 0)
        peak_layout.addWidget(self.peak_algorithm_combo, 0, 1)
        peak_layout.addWidget(QtWidgets.QLabel("寻峰起始索引"), 0, 2)
        peak_layout.addWidget(self.peak_detection_min_idx_edit, 0, 3)
        peak_layout.addWidget(QtWidgets.QLabel("最小峰强度"), 0, 4)
        peak_layout.addWidget(self.peak_min_intensity_edit, 0, 5)
        peak_layout.addWidget(QtWidgets.QLabel("峰结束阈值"), 1, 0)
        peak_layout.addWidget(self.peak_threshold_end_edit, 1, 1)
        peak_layout.addWidget(QtWidgets.QLabel("Prominence比例"), 1, 2)
        peak_layout.addWidget(self.peak_prominence_ratio_edit, 1, 3)
        peak_layout.addWidget(QtWidgets.QLabel("平滑窗口"), 1, 4)
        peak_layout.addWidget(self.peak_smoothing_window_edit, 1, 5)
        peak_layout.addWidget(QtWidgets.QLabel("邻峰抑制半宽"), 2, 0)
        peak_layout.addWidget(self.peak_nearby_window_edit, 2, 1)
        peak_layout.addWidget(QtWidgets.QLabel("重复峰排除半宽"), 2, 2)
        peak_layout.addWidget(self.peak_duplicate_window_edit, 2, 3)
        peak_layout.addWidget(QtWidgets.QLabel("边界外扩点数"), 2, 4)
        peak_layout.addWidget(self.peak_boundary_padding_edit, 2, 5)
        peak_layout.addWidget(QtWidgets.QLabel("弱肩峰过滤倍率"), 3, 0)
        peak_layout.addWidget(self.peak_weak_tail_ratio_edit, 3, 1)
        peak_layout.addWidget(QtWidgets.QLabel("高斯窗口上限"), 3, 2)
        peak_layout.addWidget(self.peak_gaussian_window_max_edit, 3, 3)
        peak_layout.addWidget(QtWidgets.QLabel("高斯边界倍数"), 3, 4)
        peak_layout.addWidget(self.peak_gaussian_boundary_scale_edit, 3, 5)
        peak_layout.addWidget(QtWidgets.QLabel("基线窗口"), 4, 0)
        peak_layout.addWidget(self.peak_baseline_window_edit, 4, 1)
        peak_layout.addWidget(QtWidgets.QLabel("基线百分位"), 4, 2)
        peak_layout.addWidget(self.peak_baseline_percentile_edit, 4, 3)
        peak_layout.addWidget(QtWidgets.QLabel("峰宽范围"), 4, 4)
        width_grp = QtWidgets.QWidget()
        width_lay = QtWidgets.QHBoxLayout(width_grp)
        width_lay.setContentsMargins(0, 0, 0, 0)
        width_lay.setSpacing(4)
        width_lay.addWidget(self.peak_min_peak_width_edit)
        width_lay.addWidget(QtWidgets.QLabel("~"))
        width_lay.addWidget(self.peak_max_peak_width_edit)
        peak_layout.addWidget(width_grp, 4, 5)
        for col in (1, 3, 5):
            peak_layout.setColumnStretch(col, 1)
        layout.addWidget(peak_group)
        layout.addStretch()

        root.addWidget(widget, 1)

        # 底部工具栏
        bottom_bar = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(bottom_bar)
        bl.setContentsMargins(8, 4, 8, 4)
        bl.setSpacing(8)
        self.save_button = QtWidgets.QPushButton("💾  保存寻峰参数")
        self.save_button.setFixedHeight(30)
        self.save_button.clicked.connect(self.save_settings)
        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setObjectName("ProjectStatus")
        bl.addWidget(self.save_button)
        bl.addWidget(self.status_label, 1)
        root.addWidget(bottom_bar)

        self.load_from_settings()

    def load_from_settings(self) -> None:
        # 优先从ProjectSettings读取，否则从本地配置读取
        if self.project_settings:
            config = self.project_settings.to_peak_detection_config()
        else:
            config = self.peak_detection

        combo_set_data(self.peak_algorithm_combo, config.algorithm)
        self.peak_detection_min_idx_edit.setValue(config.detection_min_idx)
        self.peak_threshold_end_edit.setValue(config.threshold_end)
        self.peak_min_intensity_edit.setValue(config.min_intensity)
        self.peak_nearby_window_edit.setValue(config.nearby_peak_window)
        self.peak_duplicate_window_edit.setValue(config.duplicate_window)
        self.peak_boundary_padding_edit.setValue(config.boundary_padding)
        self.peak_weak_tail_ratio_edit.setValue(config.weak_tail_ratio)
        self.peak_gaussian_window_max_edit.setValue(config.gaussian_window_max)
        self.peak_gaussian_boundary_scale_edit.setValue(config.gaussian_boundary_scale)
        self.peak_prominence_ratio_edit.setValue(config.prominence_ratio)
        self.peak_smoothing_window_edit.setValue(config.smoothing_window)
        self.peak_baseline_window_edit.setValue(config.baseline_window)
        self.peak_baseline_percentile_edit.setValue(config.baseline_percentile)
        self.peak_min_peak_width_edit.setValue(config.min_peak_width)
        self.peak_max_peak_width_edit.setValue(config.max_peak_width)

    def set_project_settings(self, project_settings: ProjectSettings) -> None:
        """Set the project settings reference for parameter synchronization."""
        self.project_settings = project_settings
        # 立即加载ProjectSettings中的参数
        if project_settings:
            self.load_from_settings()

    def apply_to_settings(self, project_settings: ProjectSettings | None = None) -> None:
        target = project_settings or self.project_settings
        if project_settings is not None:
            self.project_settings = project_settings
        self.peak_detection = PeakDetectionConfig(
            algorithm=str(self.peak_algorithm_combo.currentData()),
            detection_min_idx=self.peak_detection_min_idx_edit.value(),
            threshold_end=self.peak_threshold_end_edit.value(),
            min_intensity=self.peak_min_intensity_edit.value(),
            nearby_peak_window=self.peak_nearby_window_edit.value(),
            duplicate_window=self.peak_duplicate_window_edit.value(),
            weak_tail_early_window=self.peak_detection.weak_tail_early_window,
            weak_tail_late_window=self.peak_detection.weak_tail_late_window,
            weak_tail_ratio=self.peak_weak_tail_ratio_edit.value(),
            gaussian_window_max=self.peak_gaussian_window_max_edit.value(),
            gaussian_boundary_scale=self.peak_gaussian_boundary_scale_edit.value(),
            boundary_padding=self.peak_boundary_padding_edit.value(),
            prominence_ratio=self.peak_prominence_ratio_edit.value(),
            smoothing_window=self.peak_smoothing_window_edit.value(),
            smoothing_poly_order=self.peak_detection.smoothing_poly_order,
            baseline_window=self.peak_baseline_window_edit.value(),
            baseline_percentile=self.peak_baseline_percentile_edit.value(),
            min_peak_width=self.peak_min_peak_width_edit.value(),
            max_peak_width=max(self.peak_min_peak_width_edit.value(), self.peak_max_peak_width_edit.value()),
        )
        # 保存到ProjectSettings
        if target:
            target.peak_algorithm = self.peak_detection.algorithm
            target.detection_min_idx = self.peak_detection.detection_min_idx
            target.threshold_end = self.peak_detection.threshold_end
            target.min_intensity = self.peak_detection.min_intensity
            target.nearby_peak_window = self.peak_detection.nearby_peak_window
            target.duplicate_window = self.peak_detection.duplicate_window
            target.weak_tail_ratio = self.peak_detection.weak_tail_ratio
            target.gaussian_window_max = self.peak_detection.gaussian_window_max
            target.gaussian_boundary_scale = self.peak_detection.gaussian_boundary_scale
            target.boundary_padding = self.peak_detection.boundary_padding
            target.prominence_ratio = self.peak_detection.prominence_ratio
            target.smoothing_window = self.peak_detection.smoothing_window
            target.baseline_window = self.peak_detection.baseline_window
            target.baseline_percentile = self.peak_detection.baseline_percentile
            target.min_peak_width = self.peak_detection.min_peak_width
            target.max_peak_width = self.peak_detection.max_peak_width

    def save_settings(self):
        self.apply_to_settings()
        self.status_label.setText("已更新当前草稿，尚未保存到项目")
        self.settings_saved.emit()


class CommonParametersDialog(QtWidgets.QDialog):
    def __init__(self, settings: NormalizationSettings, calibration: Calibration, parent=None):
        super().__init__(parent)
        self.setWindowTitle("通用参数设置")
        self.resize(1280, 760)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.settings_widget = CommonParametersWidget(settings, calibration, self)
        layout.addWidget(self.settings_widget)

class FunctionDefaultsWidget(QtWidgets.QWidget):
    """集成寻峰参数和功能默认参数的统一 Widget。

    包含内容：
    - 寻峰参数（自动寻峰算法、边界检测等）
    - PIE 分析、温度扫描、PICS 计算、摩尔分数等功能的默认参数
    """
    settings_saved = QtCore.pyqtSignal()
    navigate_requested = QtCore.pyqtSignal(str)

    _PAGE_LABELS = {
        "spectrum": "质谱工作台",
        "temperature": "温度扫描",
        "pie": "PIE 拟合",
        "pics": "PICS 计算",
        "mole_fraction": "摩尔分数",
    }

    def __init__(
        self,
        parent=None,
        *,
        show_actions: bool = True,
        visible_pages: tuple[str, ...] | None = None,
    ):
        super().__init__(parent)
        self.peak_detection = load_factory_project_settings().to_peak_detection_config()
        self.project_settings: ProjectSettings | None = None
        visible_page_set = set(visible_pages) if visible_pages is not None else None
        self._visible_pages: list[str] = []
        self._visible_page_labels: list[str] = []

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        action_row = QtWidgets.QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.addStretch(1)
        self.open_function_page_button = QtWidgets.QPushButton(self)
        self.open_function_page_button.setObjectName("BrowseButton")
        self.open_function_page_button.setToolTip("打开当前默认值对应的功能页面")
        self.open_function_page_button.clicked.connect(self._request_current_function_page)
        self.open_function_page_button.setVisible(show_actions)
        action_row.addWidget(self.open_function_page_button)
        if show_actions:
            root.addLayout(action_row)

        # 使用标签页组织功能参数
        self.tabs = QtWidgets.QTabWidget()
        tab_specs = (
            (
                "spectrum",
                "寻峰与积分",
                self._build_peak_detection_tab(),
            ),
            ("temperature", "温度扫描", self._build_temperature_scan_tab()),
            ("pie", "PIE 分析", self._build_pie_fitting_tab()),
            ("pics", "PICS 计算", self._build_pics_tab()),
            ("mole_fraction", "摩尔分数", self._build_mole_fraction_tab()),
        )
        for page_name, tab_label, tab_widget in tab_specs:
            if visible_page_set is not None and page_name not in visible_page_set:
                tab_widget.setParent(self)
                tab_widget.hide()
                continue
            self.tabs.addTab(tab_widget, tab_label)
            self._visible_pages.append(page_name)
            self._visible_page_labels.append(self._PAGE_LABELS[page_name])
        self.tabs.currentChanged.connect(self._refresh_function_page_button)
        root.addWidget(self.tabs, 1)

        # 底部工具栏
        bottom_bar = QtWidgets.QWidget()
        bl = QtWidgets.QHBoxLayout(bottom_bar)
        bl.setContentsMargins(8, 4, 8, 4)
        bl.setSpacing(8)
        self.save_button = QtWidgets.QPushButton("应用到当前草稿")
        self.save_button.setObjectName("PrimaryButton")
        self.save_button.setFixedHeight(30)
        self.save_button.clicked.connect(self.save_settings)
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setObjectName("ProjectStatus")
        bl.addWidget(self.save_button)
        bl.addWidget(self.status_label, 1)
        bottom_bar.setVisible(show_actions)
        root.addWidget(bottom_bar)

        self._refresh_function_page_button(0)
        self.load_from_settings()

    def _refresh_function_page_button(self, index: int) -> None:
        if not 0 <= index < len(self._visible_page_labels):
            return
        self.open_function_page_button.setText(f"打开{self._visible_page_labels[index]}页")

    def _request_current_function_page(self) -> None:
        index = self.tabs.currentIndex()
        if 0 <= index < len(self._visible_pages):
            self.navigate_requested.emit(self._visible_pages[index])

    def set_current_page(self, page_name: str) -> None:
        if page_name in self._visible_pages:
            self.tabs.setCurrentIndex(self._visible_pages.index(page_name))

    def _build_peak_detection_tab(self) -> QtWidgets.QWidget:
        """寻峰与积分参数标签页"""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        peak_group = QtWidgets.QGroupBox("自动寻峰参数")
        peak_layout = QtWidgets.QGridLayout(peak_group)
        peak_layout.setHorizontalSpacing(8)
        peak_layout.setVerticalSpacing(8)

        self.peak_algorithm_combo = QtWidgets.QComboBox()
        self.peak_algorithm_combo.addItem("自适应高召回（实验）", "adaptive")
        self.peak_algorithm_combo.addItem("Ensemble 融合检测（推荐）", "ensemble")
        self.peak_algorithm_combo.addItem("Prominence", "prominence")
        self.peak_algorithm_combo.addItem("传统局部极大", "legacy")
        self.peak_algorithm_combo.addItem("CWT 小波", "cwt")
        self.peak_algorithm_combo.setToolTip("主工作台自动寻峰使用的算法")

        self.peak_detection_min_idx_edit = QtWidgets.QSpinBox()
        self.peak_detection_min_idx_edit.setRange(0, 10_000_000)
        self.peak_detection_min_idx_edit.setToolTip("自动寻峰从该数据点之后开始，避免文件头或低TOF噪声参与寻峰")

        self.peak_threshold_end_edit = QtWidgets.QDoubleSpinBox()
        self.peak_threshold_end_edit.setRange(0, 1_000_000)
        self.peak_threshold_end_edit.setDecimals(3)
        self.peak_threshold_end_edit.setToolTip("向峰两侧扩展边界时，低于该强度即认为到达峰结束")

        self.peak_min_intensity_edit = QtWidgets.QDoubleSpinBox()
        self.peak_min_intensity_edit.setRange(0, 1_000_000)
        self.peak_min_intensity_edit.setDecimals(3)
        self.peak_min_intensity_edit.setToolTip("低于该强度的局部极大值不会作为候选峰")

        self.peak_nearby_window_edit = QtWidgets.QSpinBox()
        self.peak_nearby_window_edit.setRange(0, 100_000)
        self.peak_nearby_window_edit.setToolTip("该半宽范围内若已有更高峰，则当前候选峰会被抑制")

        self.peak_duplicate_window_edit = QtWidgets.QSpinBox()
        self.peak_duplicate_window_edit.setRange(0, 100_000)
        self.peak_duplicate_window_edit.setToolTip("接受一个峰后，该半宽范围内的候选点会被视为同一峰")

        self.peak_boundary_padding_edit = QtWidgets.QSpinBox()
        self.peak_boundary_padding_edit.setRange(0, 100_000)
        self.peak_boundary_padding_edit.setToolTip("最终卡峰边界向左右额外扩展的数据点数")

        self.peak_weak_tail_ratio_edit = QtWidgets.QDoubleSpinBox()
        self.peak_weak_tail_ratio_edit.setRange(0.001, 1_000_000)
        self.peak_weak_tail_ratio_edit.setDecimals(3)
        self.peak_weak_tail_ratio_edit.setToolTip("弱肩峰过滤倍率；值越大，越容易保留主峰后的弱峰")

        self.peak_gaussian_window_max_edit = QtWidgets.QSpinBox()
        self.peak_gaussian_window_max_edit.setRange(3, 100_000)
        self.peak_gaussian_window_max_edit.setToolTip("自动高斯拟合时允许使用的最大半窗口")

        self.peak_gaussian_boundary_scale_edit = QtWidgets.QDoubleSpinBox()
        self.peak_gaussian_boundary_scale_edit.setRange(0.1, 100)
        self.peak_gaussian_boundary_scale_edit.setDecimals(3)
        self.peak_gaussian_boundary_scale_edit.setToolTip("高斯拟合成功后，以该倍数 FWHM 重新估计卡峰边界")

        row = 0
        peak_layout.addWidget(QtWidgets.QLabel("寻峰算法"), row, 0)
        peak_layout.addWidget(self.peak_algorithm_combo, row, 1)
        peak_layout.addWidget(QtWidgets.QLabel("起始 TOF 索引"), row, 2)
        peak_layout.addWidget(self.peak_detection_min_idx_edit, row, 3)

        row += 1
        peak_layout.addWidget(QtWidgets.QLabel("峰结束阈值"), row, 0)
        peak_layout.addWidget(self.peak_threshold_end_edit, row, 1)
        peak_layout.addWidget(QtWidgets.QLabel("最小峰强度"), row, 2)
        peak_layout.addWidget(self.peak_min_intensity_edit, row, 3)

        row += 1
        peak_layout.addWidget(QtWidgets.QLabel("邻近峰抑制窗口"), row, 0)
        peak_layout.addWidget(self.peak_nearby_window_edit, row, 1)
        peak_layout.addWidget(QtWidgets.QLabel("同峰合并窗口"), row, 2)
        peak_layout.addWidget(self.peak_duplicate_window_edit, row, 3)

        row += 1
        peak_layout.addWidget(QtWidgets.QLabel("边界扩展点数"), row, 0)
        peak_layout.addWidget(self.peak_boundary_padding_edit, row, 1)
        peak_layout.addWidget(QtWidgets.QLabel("弱肩峰保留倍率"), row, 2)
        peak_layout.addWidget(self.peak_weak_tail_ratio_edit, row, 3)

        row += 1
        peak_layout.addWidget(QtWidgets.QLabel("高斯最大半窗"), row, 0)
        peak_layout.addWidget(self.peak_gaussian_window_max_edit, row, 1)
        peak_layout.addWidget(QtWidgets.QLabel("高斯边界倍数"), row, 2)
        peak_layout.addWidget(self.peak_gaussian_boundary_scale_edit, row, 3)

        peak_layout.setColumnStretch(1, 1)
        peak_layout.setColumnStretch(3, 1)
        layout.addWidget(peak_group)

        layout.addStretch(1)

        return widget

    def _build_temperature_scan_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        group = QtWidgets.QGroupBox("温度扫描默认参数")
        form = QtWidgets.QGridLayout(group)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)

        self.temp_peak_source_combo = QtWidgets.QComboBox()
        self.temp_peak_source_combo.addItem("自动寻峰", "auto")
        self.temp_peak_source_combo.addItem("手动卡峰文件", "manual")
        self.temp_peak_source_combo.setVisible(False)

        self.temp_reference_mode_combo = QtWidgets.QComboBox()
        self.temp_reference_mode_combo.addItem("Sum 谱参考", "sum")
        self.temp_reference_mode_combo.addItem("独立参考", "individual")
        self.temp_reference_mode_combo.setVisible(False)

        self.temp_integration_method_combo = QtWidgets.QComboBox()
        self.temp_integration_method_combo.addItem("范围累加", "sum_counts")
        self.temp_integration_method_combo.addItem("扣基线积分", "baseline")
        self.temp_integration_method_combo.addItem("高斯", "gaussian")
        self.temp_integration_method_combo.setToolTip(
            "温度扫描原始积分信号的默认计算方式；"
            "高斯拟合不可用时回退范围累加，并在结果中记录实际方式"
        )

        self.temperature_photon_check = QtWidgets.QCheckBox("光强归一化")
        self.temperature_photon_check.setToolTip(
            "使用项目共享参数中配置的光强来源归一化温度扫描信号"
        )
        self.temperature_kr_check = QtWidgets.QCheckBox("Kr 校正")
        self.temperature_kr_check.setToolTip(
            "使用项目共享参数中已有的 Kr 膨胀系数 λ(T) 校正温度扫描信号"
        )

        self.temp_curve_class_change_threshold_edit = QtWidgets.QDoubleSpinBox()
        self.temp_curve_class_change_threshold_edit.setRange(0.0, 1.0)
        self.temp_curve_class_change_threshold_edit.setDecimals(3)
        self.temp_curve_class_change_threshold_edit.setSingleStep(0.01)
        self.temp_curve_class_change_threshold_edit.setToolTip("判定生成/消耗曲线所需的相对变化阈值")

        self.temp_curve_class_peak_fraction_edit = QtWidgets.QDoubleSpinBox()
        self.temp_curve_class_peak_fraction_edit.setRange(0.0, 1.0)
        self.temp_curve_class_peak_fraction_edit.setDecimals(3)
        self.temp_curve_class_peak_fraction_edit.setSingleStep(0.01)
        self.temp_curve_class_peak_fraction_edit.setToolTip("端点信号占最大值的比例阈值")
        self.temp_replicate_mode_combo = QtWidgets.QComboBox()
        self.temp_replicate_mode_combo.addItem("不合并重复采集", "off")
        self.temp_replicate_mode_combo.addItem("平均重复采集", "mean")
        self.temp_replicate_mode_combo.addItem("累加重复采集", "sum")
        self.temp_replicate_mode_combo.setToolTip("仅在确认同一条件多次采集时选择平均或累加")

        form.addWidget(self.temperature_photon_check, 0, 0)
        form.addWidget(self.temperature_kr_check, 0, 1)
        self.temp_peak_source_label = QtWidgets.QLabel("项目峰来源")
        self.temp_peak_source_hint = QtWidgets.QLabel("当前项目卡峰集（PIE与温扫共用）")
        self.temp_peak_source_hint.setToolTip("在项目管理中管理卡峰版本")
        form.addWidget(self.temp_peak_source_label, 1, 0)
        form.addWidget(self.temp_peak_source_hint, 1, 1, 1, 3)
        form.addWidget(QtWidgets.QLabel("积分方式"), 1, 4)
        form.addWidget(self.temp_integration_method_combo, 1, 5)
        form.addWidget(QtWidgets.QLabel("分类变化阈值"), 2, 0)
        form.addWidget(self.temp_curve_class_change_threshold_edit, 2, 1)
        form.addWidget(QtWidgets.QLabel("端点峰值比例"), 2, 2)
        form.addWidget(self.temp_curve_class_peak_fraction_edit, 2, 3)
        form.addWidget(QtWidgets.QLabel("重复采集处理"), 3, 0)
        form.addWidget(self.temp_replicate_mode_combo, 3, 1)
        for col in (1, 3, 5):
            form.setColumnStretch(col, 1)

        layout.addWidget(group)
        layout.addStretch(1)
        return widget

    def _build_pie_element_filter_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("物种匹配元素范围")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        hint = QtWidgets.QLabel(
            "仅影响 PIE 物种数据库匹配和同位素候选范围。"
        )
        hint.setObjectName("HintLabel")
        layout.addWidget(hint)

        elements_layout = QtWidgets.QGridLayout()
        elements_layout.setHorizontalSpacing(6)
        elements_layout.setVerticalSpacing(4)
        self.element_checks: dict[str, QtWidgets.QCheckBox] = {}
        for index, element in enumerate(COMMON_ELEMENTS):
            checkbox = QtWidgets.QCheckBox(element)
            checkbox.setMaximumWidth(60)
            self.element_checks[element] = checkbox
            elements_layout.addWidget(
                checkbox,
                index // 6,
                index % 6,
            )
        layout.addLayout(elements_layout)
        return group

    def _build_pie_fitting_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        group = QtWidgets.QGroupBox("PIE 分析参数")
        form = QtWidgets.QGridLayout(group)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)

        self.pie_energy_decimals_edit = QtWidgets.QSpinBox()
        self.pie_energy_decimals_edit.setRange(0, 6)
        self.pie_energy_decimals_edit.setToolTip("光子能量分组时保留的小数位数")

        self.pie_photon_mode_combo = QtWidgets.QComboBox()
        self.pie_photon_mode_combo.addItem("关闭光强归一", "off")
        self.pie_photon_mode_combo.addItem("按 IO 归一", "none")
        self.pie_photon_mode_combo.addItem("按 IO 及参考值归一", "first")
        self.pie_photon_mode_combo.setToolTip(
            "选择 PIE 曲线生成时的光强归一化方式；光强来源由项目共享参数维护"
        )

        self.pie_time_normalize_check = QtWidgets.QCheckBox(
            "按每个原始谱的扫描时间归一化"
        )
        self.pie_time_normalize_check.setToolTip(
            "先换算为 counts/(nA·s) 再合并重复谱"
        )

        self.pie_recursive_check = QtWidgets.QCheckBox("递归扩展拟合")
        self.pie_recursive_check.setToolTip("先拟合高能段，再逐步扩展到低能区")

        self.pie_integration_method_combo = QtWidgets.QComboBox()
        self.pie_integration_method_combo.addItem("范围累加", "sum_counts")
        self.pie_integration_method_combo.addItem("扣基线积分", "baseline")
        self.pie_integration_method_combo.addItem("高斯", "gaussian")
        self.pie_integration_method_combo.setToolTip(
            "PIE 原始积分信号的默认计算方式；"
            "高斯拟合不可用时回退范围累加，并在结果中记录实际方式"
        )

        self.pie_merge_method_combo = QtWidgets.QComboBox()
        self.pie_merge_method_combo.addItem("低能段为主", "low_energy_dominant")
        self.pie_merge_method_combo.addItem("第一组为主", "first_segment_dominant")
        self.pie_merge_method_combo.addItem("简单拼接", "mean")
        self.pie_merge_method_combo.setToolTip("项目登记多个 PIE 能段目录时使用的合并方式")
        self.pie_replicate_mode_combo = QtWidgets.QComboBox()
        self.pie_replicate_mode_combo.addItem("不合并重复采集", "off")
        self.pie_replicate_mode_combo.addItem("平均重复采集", "mean")
        self.pie_replicate_mode_combo.addItem("累加重复采集", "sum")
        self.pie_replicate_mode_combo.setToolTip("仅在确认同一条件多次采集时选择平均或累加")

        form.addWidget(QtWidgets.QLabel("光强归一化"), 0, 0)
        form.addWidget(self.pie_photon_mode_combo, 0, 1)
        form.addWidget(self.pie_time_normalize_check, 0, 2, 1, 3)
        form.addWidget(QtWidgets.QLabel("能量分组小数位"), 1, 0)
        form.addWidget(self.pie_energy_decimals_edit, 1, 1)
        form.addWidget(self.pie_recursive_check, 1, 2)
        form.addWidget(QtWidgets.QLabel("积分方式"), 1, 3)
        form.addWidget(self.pie_integration_method_combo, 1, 4)
        form.addWidget(QtWidgets.QLabel("能段合并方式"), 2, 0)
        form.addWidget(self.pie_merge_method_combo, 2, 1)
        self.pie_merge_method_hint = QtWidgets.QLabel("仅在存在多个能段时生效")
        self.pie_merge_method_hint.setObjectName("HintLabel")
        form.addWidget(self.pie_merge_method_hint, 2, 2, 1, 3)
        form.addWidget(QtWidgets.QLabel("重复采集处理"), 3, 0)
        form.addWidget(self.pie_replicate_mode_combo, 3, 1)
        for col in (1, 3):
            form.setColumnStretch(col, 1)

        layout.addWidget(group)
        layout.addWidget(self._build_pie_element_filter_group())
        layout.addStretch(1)
        return widget

    def _build_pics_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        group = QtWidgets.QGroupBox("PICS 计算默认参数")
        form = QtWidgets.QGridLayout(group)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)

        self.pics_no_mz_edit = QtWidgets.QSpinBox()
        self.pics_no_mz_edit.setRange(1, 1000)
        self.pics_no_mz_edit.setToolTip("参考物种 NO 的质量数")

        self.pics_no_formula_edit = QtWidgets.QLineEdit()
        self.pics_no_formula_edit.setToolTip("参考物种分子式，用于匹配截面数据库")

        self.pics_no_mf_edit = QtWidgets.QDoubleSpinBox()
        self.pics_no_mf_edit.setRange(0.0, 1.0)
        self.pics_no_mf_edit.setDecimals(6)
        self.pics_no_mf_edit.setSingleStep(0.001)
        self.pics_no_mf_edit.setToolTip("参考物种 NO 的摩尔分数")

        self.pics_new_species_mf_edit = QtWidgets.QDoubleSpinBox()
        self.pics_new_species_mf_edit.setRange(0.0, 1.0)
        self.pics_new_species_mf_edit.setDecimals(6)
        self.pics_new_species_mf_edit.setSingleStep(0.001)
        self.pics_new_species_mf_edit.setToolTip("新物种默认摩尔分数")

        self.pics_mass_disc_exponent_edit = QtWidgets.QDoubleSpinBox()
        self.pics_mass_disc_exponent_edit.setRange(0.0, 2.0)
        self.pics_mass_disc_exponent_edit.setDecimals(5)
        self.pics_mass_disc_exponent_edit.setSingleStep(0.001)
        self.pics_mass_disc_exponent_edit.setToolTip(
            "PICS 计算独立使用的质量响应指数，不再读取摩尔分数参数"
        )

        form.addWidget(QtWidgets.QLabel("NO m/z"), 0, 0)
        form.addWidget(self.pics_no_mz_edit, 0, 1)
        form.addWidget(QtWidgets.QLabel("NO 分子式"), 0, 2)
        form.addWidget(self.pics_no_formula_edit, 0, 3)
        form.addWidget(QtWidgets.QLabel("NO 摩尔分数"), 1, 0)
        form.addWidget(self.pics_no_mf_edit, 1, 1)
        form.addWidget(QtWidgets.QLabel("新物种摩尔分数"), 1, 2)
        form.addWidget(self.pics_new_species_mf_edit, 1, 3)
        form.addWidget(QtWidgets.QLabel("质量响应指数 n"), 2, 0)
        form.addWidget(self.pics_mass_disc_exponent_edit, 2, 1)
        for col in (1, 3):
            form.setColumnStretch(col, 1)

        layout.addWidget(group)
        layout.addStretch(1)
        return widget

    def _build_mole_fraction_tab(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        mass_response_group = QtWidgets.QGroupBox("质量响应校正")
        mass_response_layout = QtWidgets.QGridLayout(mass_response_group)
        mass_response_layout.setHorizontalSpacing(8)
        mass_response_layout.setVerticalSpacing(8)

        self.mf_md_preset_combo = QtWidgets.QComboBox()
        for name in MASS_DISCRIMINATION_PRESETS:
            self.mf_md_preset_combo.addItem(name, name)
        self.mf_md_preset_combo.currentTextChanged.connect(
            self._on_mf_md_preset_changed
        )

        self.mf_mass_disc_exponent_edit = QtWidgets.QDoubleSpinBox()
        self.mf_mass_disc_exponent_edit.setRange(0.0, 2.0)
        self.mf_mass_disc_exponent_edit.setDecimals(5)
        self.mf_mass_disc_exponent_edit.setToolTip(
            "质量响应因子 Dᵢ = (MW / 30)ⁿ 中使用的指数"
        )
        mass_response_layout.addWidget(
            QtWidgets.QLabel("实验条件预设"),
            0,
            0,
        )
        mass_response_layout.addWidget(self.mf_md_preset_combo, 0, 1)
        mass_response_layout.addWidget(QtWidgets.QLabel("指数 n"), 1, 0)
        mass_response_layout.addWidget(
            self.mf_mass_disc_exponent_edit,
            1,
            1,
        )
        formula_label = QtWidgets.QLabel("响应因子公式：Dᵢ = (MW / 30)ⁿ")
        formula_label.setObjectName("FormulaLabel")
        mass_response_layout.addWidget(formula_label, 2, 0, 1, 2)
        mass_response_layout.setColumnStretch(1, 1)

        group = QtWidgets.QGroupBox("摩尔分数默认参数")
        form = QtWidgets.QGridLayout(group)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(8)

        self.mf_parent_mz_edit = QtWidgets.QDoubleSpinBox()
        self.mf_parent_mz_edit.setRange(0.0, 1000.0)
        self.mf_parent_mz_edit.setDecimals(6)
        self.mf_parent_mz_edit.setSingleStep(0.001)
        self.mf_parent_mz_edit.setSpecialValueText("未设置")
        self.mf_parent_mz_edit.setToolTip("母体物种对应的精确峰 m/z")

        self.mf_parent_initial_mf_edit = QtWidgets.QDoubleSpinBox()
        self.mf_parent_initial_mf_edit.setRange(0.0, 1.0)
        self.mf_parent_initial_mf_edit.setDecimals(6)
        self.mf_parent_initial_mf_edit.setSingleStep(0.001)
        self.mf_parent_initial_mf_edit.setToolTip("母体物种在参考温度处的初始摩尔分数")

        self.mf_photon_energy_edit = QtWidgets.QDoubleSpinBox()
        self.mf_photon_energy_edit.setRange(0.0, 100.0)
        self.mf_photon_energy_edit.setDecimals(4)
        self.mf_photon_energy_edit.setSingleStep(0.1)
        self.mf_photon_energy_edit.setToolTip("默认光子能量")

        self.mf_reference_temperature_edit = QtWidgets.QSpinBox()
        self.mf_reference_temperature_edit.setRange(0, 2000)
        self.mf_reference_temperature_edit.setSpecialValueText("未设置")
        self.mf_reference_temperature_edit.setToolTip("母体初始摩尔分数对应的参考温度")

        form.addWidget(QtWidgets.QLabel("母体 m/z"), 0, 0)
        form.addWidget(self.mf_parent_mz_edit, 0, 1)
        form.addWidget(QtWidgets.QLabel("母体初始摩尔分数"), 0, 2)
        form.addWidget(self.mf_parent_initial_mf_edit, 0, 3)
        form.addWidget(QtWidgets.QLabel("光子能量 (eV)"), 1, 0)
        form.addWidget(self.mf_photon_energy_edit, 1, 1)
        form.addWidget(QtWidgets.QLabel("参考温度"), 1, 2)
        form.addWidget(self.mf_reference_temperature_edit, 1, 3)
        for col in (1, 3):
            form.setColumnStretch(col, 1)

        layout.addWidget(mass_response_group)
        layout.addWidget(group)
        layout.addStretch(1)
        return widget

    def _on_mf_md_preset_changed(self, text: str) -> None:
        if text in MASS_DISCRIMINATION_PRESETS:
            self.mf_mass_disc_exponent_edit.setValue(
                MASS_DISCRIMINATION_PRESETS[text]
            )

    def set_project_settings(self, project_settings: ProjectSettings) -> None:
        """设置项目级配置引用"""
        self.project_settings = project_settings
        self.peak_detection = project_settings.to_peak_detection_config()
        self.load_from_settings()

    def load_from_settings(self) -> None:
        """从配置文件加载参数"""
        if self.project_settings:
            self.peak_detection = self.project_settings.to_peak_detection_config()
        ps = self.project_settings or ProjectSettings()

        combo_set_data(self.peak_algorithm_combo, self.peak_detection.algorithm)
        self.peak_detection_min_idx_edit.setValue(self.peak_detection.detection_min_idx)
        self.peak_threshold_end_edit.setValue(self.peak_detection.threshold_end)
        self.peak_min_intensity_edit.setValue(self.peak_detection.min_intensity)
        self.peak_nearby_window_edit.setValue(self.peak_detection.nearby_peak_window)
        self.peak_duplicate_window_edit.setValue(self.peak_detection.duplicate_window)
        self.peak_boundary_padding_edit.setValue(self.peak_detection.boundary_padding)
        self.peak_weak_tail_ratio_edit.setValue(self.peak_detection.weak_tail_ratio)
        self.peak_gaussian_window_max_edit.setValue(self.peak_detection.gaussian_window_max)
        self.peak_gaussian_boundary_scale_edit.setValue(self.peak_detection.gaussian_boundary_scale)
        combo_set_data(self.temp_peak_source_combo, ps.temp_peak_source)
        combo_set_data(self.temp_reference_mode_combo, ps.temp_reference_mode)
        self.temperature_photon_check.setChecked(
            bool(ps.temperature_photon_normalize)
        )
        kr_available = bool(ps.expansion_factors)
        self.temperature_kr_check.setEnabled(kr_available)
        self.temperature_kr_check.setChecked(
            bool(ps.temperature_kr_correct) and kr_available
        )
        self.temperature_kr_check.setText(
            "Kr 校正" if kr_available else "Kr 校正（缺少系数）"
        )
        combo_set_data(
            self.temp_integration_method_combo,
            ps.temp_integration_method,
        )
        self.temp_curve_class_change_threshold_edit.setValue(ps.temp_curve_class_change_threshold)
        self.temp_curve_class_peak_fraction_edit.setValue(ps.temp_curve_class_peak_fraction)
        combo_set_data(self.temp_replicate_mode_combo, ps.temp_replicate_mode)
        combo_set_data(self.pie_photon_mode_combo, ps.pie_photon_mode)
        self.pie_time_normalize_check.setChecked(bool(ps.pie_time_normalize))
        self.pie_energy_decimals_edit.setValue(ps.pie_energy_decimals)
        self.pie_recursive_check.setChecked(ps.pie_recursive)
        combo_set_data(
            self.pie_integration_method_combo,
            ps.pie_integration_method,
        )
        combo_set_data(self.pie_merge_method_combo, ps.pie_merge_method)
        self.set_pie_multi_segment_state(
            len(ps.effective_pie_scan_folders()) > 1
        )
        combo_set_data(self.pie_replicate_mode_combo, ps.pie_replicate_mode)
        selected_elements = set(ps.selected_elements)
        for element, checkbox in self.element_checks.items():
            checkbox.setChecked(
                not selected_elements or element in selected_elements
            )
        self.pics_no_mz_edit.setValue(ps.pics_no_mz)
        self.pics_no_formula_edit.setText(ps.pics_no_formula)
        self.pics_no_mf_edit.setValue(ps.pics_no_mf)
        self.pics_new_species_mf_edit.setValue(ps.pics_new_species_mf)
        self.pics_mass_disc_exponent_edit.setValue(
            ps.pics_mass_disc_exponent
        )
        self.mf_parent_mz_edit.setValue(ps.mf_parent_mz)
        self.mf_parent_initial_mf_edit.setValue(ps.mf_parent_initial_mf)
        self.mf_photon_energy_edit.setValue(ps.mf_photon_energy)
        self.mf_reference_temperature_edit.setValue(
            0 if ps.mf_reference_temperature is None else int(ps.mf_reference_temperature)
        )
        if ps.mf_md_preset:
            if self.mf_md_preset_combo.findData(ps.mf_md_preset) < 0:
                self.mf_md_preset_combo.addItem(
                    ps.mf_md_preset,
                    ps.mf_md_preset,
                )
            combo_set_data(self.mf_md_preset_combo, ps.mf_md_preset)
        self.mf_mass_disc_exponent_edit.setValue(
            ps.mf_mass_disc_exponent
        )

    def apply_to_settings(self, project_settings: ProjectSettings | None = None) -> None:
        """Copy current function defaults into ProjectSettings."""
        target = project_settings or self.project_settings
        if project_settings is not None:
            self.project_settings = project_settings
        self.peak_detection = PeakDetectionConfig(
            algorithm=str(self.peak_algorithm_combo.currentData()),
            detection_min_idx=self.peak_detection_min_idx_edit.value(),
            threshold_end=self.peak_threshold_end_edit.value(),
            min_intensity=self.peak_min_intensity_edit.value(),
            nearby_peak_window=self.peak_nearby_window_edit.value(),
            duplicate_window=self.peak_duplicate_window_edit.value(),
            weak_tail_early_window=self.peak_detection.weak_tail_early_window,
            weak_tail_late_window=self.peak_detection.weak_tail_late_window,
            weak_tail_ratio=self.peak_weak_tail_ratio_edit.value(),
            gaussian_window_max=self.peak_gaussian_window_max_edit.value(),
            gaussian_boundary_scale=self.peak_gaussian_boundary_scale_edit.value(),
            boundary_padding=self.peak_boundary_padding_edit.value(),
            prominence_ratio=self.peak_detection.prominence_ratio,
            smoothing_window=self.peak_detection.smoothing_window,
            smoothing_poly_order=self.peak_detection.smoothing_poly_order,
            baseline_window=self.peak_detection.baseline_window,
            baseline_percentile=self.peak_detection.baseline_percentile,
            min_peak_width=self.peak_detection.min_peak_width,
            max_peak_width=self.peak_detection.max_peak_width,
            cwt_snr_threshold=self.peak_detection.cwt_snr_threshold,
            cwt_wavelet_max_width=self.peak_detection.cwt_wavelet_max_width,
            weak_tail_cutoff_idx=self.peak_detection.weak_tail_cutoff_idx,
            vote_threshold=self.peak_detection.vote_threshold,
            min_intensity_for_single_vote=self.peak_detection.min_intensity_for_single_vote,
            mz_tolerance=self.peak_detection.mz_tolerance,
        )
        if target is None:
            return

        target.peak_algorithm = self.peak_detection.algorithm
        target.detection_min_idx = self.peak_detection.detection_min_idx
        target.threshold_end = self.peak_detection.threshold_end
        target.min_intensity = self.peak_detection.min_intensity
        target.nearby_peak_window = self.peak_detection.nearby_peak_window
        target.duplicate_window = self.peak_detection.duplicate_window
        target.weak_tail_early_window = self.peak_detection.weak_tail_early_window
        target.weak_tail_late_window = self.peak_detection.weak_tail_late_window
        target.weak_tail_ratio = self.peak_detection.weak_tail_ratio
        target.gaussian_window_max = self.peak_detection.gaussian_window_max
        target.gaussian_boundary_scale = self.peak_detection.gaussian_boundary_scale
        target.boundary_padding = self.peak_detection.boundary_padding
        target.prominence_ratio = self.peak_detection.prominence_ratio
        target.smoothing_window = self.peak_detection.smoothing_window
        target.smoothing_poly_order = self.peak_detection.smoothing_poly_order
        target.baseline_window = self.peak_detection.baseline_window
        target.baseline_percentile = self.peak_detection.baseline_percentile
        target.min_peak_width = self.peak_detection.min_peak_width
        target.max_peak_width = self.peak_detection.max_peak_width
        target.cwt_snr_threshold = self.peak_detection.cwt_snr_threshold
        target.cwt_wavelet_max_width = self.peak_detection.cwt_wavelet_max_width
        target.weak_tail_cutoff_idx = self.peak_detection.weak_tail_cutoff_idx
        target.vote_threshold = self.peak_detection.vote_threshold
        target.min_intensity_for_single_vote = self.peak_detection.min_intensity_for_single_vote
        target.mz_tolerance = self.peak_detection.mz_tolerance
        target.temp_reference_mode = str(self.temp_reference_mode_combo.currentData())
        target.temperature_photon_normalize = self.temperature_photon_check.isChecked()
        target.temperature_kr_correct = (
            self.temperature_kr_check.isChecked()
            and self.temperature_kr_check.isEnabled()
        )
        target.temp_integration_method = str(
            self.temp_integration_method_combo.currentData()
        )
        target.temp_prefer_gaussian = (
            target.temp_integration_method == "gaussian"
        )
        target.temp_curve_class_change_threshold = self.temp_curve_class_change_threshold_edit.value()
        target.temp_curve_class_peak_fraction = self.temp_curve_class_peak_fraction_edit.value()
        target.temp_replicate_mode = str(self.temp_replicate_mode_combo.currentData())
        target.pie_photon_mode = str(self.pie_photon_mode_combo.currentData())
        target.pie_time_normalize = self.pie_time_normalize_check.isChecked()
        target.pie_energy_decimals = self.pie_energy_decimals_edit.value()
        target.pie_recursive = self.pie_recursive_check.isChecked()
        target.pie_integration_method = str(
            self.pie_integration_method_combo.currentData()
        )
        target.pie_prefer_gaussian = (
            target.pie_integration_method == "gaussian"
        )
        target.pie_multi_folder_mode = len(target.effective_pie_scan_folders()) > 1
        target.pie_merge_method = str(self.pie_merge_method_combo.currentData())
        target.pie_replicate_mode = str(self.pie_replicate_mode_combo.currentData())
        target.selected_elements = [
            element
            for element, checkbox in self.element_checks.items()
            if checkbox.isChecked()
        ]
        target.pics_no_mz = self.pics_no_mz_edit.value()
        target.pics_no_formula = self.pics_no_formula_edit.text().strip() or "NO"
        target.pics_no_mf = self.pics_no_mf_edit.value()
        target.pics_new_species_mf = self.pics_new_species_mf_edit.value()
        target.pics_mass_disc_exponent = (
            self.pics_mass_disc_exponent_edit.value()
        )
        target.mf_parent_mz = self.mf_parent_mz_edit.value()
        target.mf_parent_initial_mf = self.mf_parent_initial_mf_edit.value()
        target.mf_photon_energy = self.mf_photon_energy_edit.value()
        target.mf_md_preset = str(
            self.mf_md_preset_combo.currentData()
            or self.mf_md_preset_combo.currentText()
        )
        target.mf_mass_disc_exponent = (
            self.mf_mass_disc_exponent_edit.value()
        )
        reference_temperature = self.mf_reference_temperature_edit.value()
        target.mf_reference_temperature = None if reference_temperature == 0 else float(reference_temperature)

    def set_pie_multi_segment_state(self, enabled: bool) -> None:
        """Enable PIE merge controls only when more than one segment is active."""
        enabled = bool(enabled)
        self.pie_merge_method_combo.setEnabled(enabled)
        self.pie_merge_method_hint.setText(
            "用于当前多能段数据"
            if enabled
            else "仅在存在多个能段时生效"
        )

    def save_settings(self) -> None:
        """Apply controls to the in-memory draft and notify the owner."""
        self.apply_to_settings()
        self.status_label.setText("已更新当前草稿，尚未保存到项目")
        self.settings_saved.emit()
