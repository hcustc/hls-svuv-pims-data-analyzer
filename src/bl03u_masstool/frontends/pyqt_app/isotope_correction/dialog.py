from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtWidgets

from bl03u_masstool.core.curve_database import (
    CurveDataset,
    list_curve_datasets_read_only,
    list_species_assignments,
    load_curve_dataset_read_only,
    project_curve_database_path,
)
from bl03u_masstool.core.isotope import formula_nominal_mass
from bl03u_masstool.core.isotope_correction import (
    CORRECTION_MODES,
    IsotopeCorrectionResult,
    IsotopeHypothesis,
    apply_isotope_correction,
    infer_curve_columns,
    nominal_isotope_pattern,
    prepare_curve_matrix,
)
from bl03u_masstool.core.output_paths import ensure_output_dir
from bl03u_masstool.core.project_settings import ProjectSettings
from bl03u_masstool.frontends.pyqt_app.common.plot_spec import (
    CurveSeries,
    ScientificPlotSpec,
    SeriesRole,
)
from bl03u_masstool.frontends.pyqt_app.common.static_plot import StaticCurvePlot
from bl03u_masstool.frontends.pyqt_app.common.widgets import DataFrameTableMixin
from bl03u_masstool.frontends.pyqt_app.project_artifacts import record_project_artifact
from bl03u_masstool.frontends.pyqt_app.workers import WorkerThread


class IsotopeCorrectionDialog(QtWidgets.QWidget, DataFrameTableMixin):
    PROJECT_RESULT_GROUPS = {
        "temperature": "temperature:project",
        "pie": "pie:project",
    }
    """Formula-driven isotope contribution post-processing for curve results."""

    ENERGY_GROUP_DECIMALS = 2
    SOURCE_LABELS = {
        "temperature": "温度扫描结果",
        "pie": "PIE结果",
    }
    SOURCE_ARTIFACT_FIELDS = {
        "temperature": "temperature_scan_result_file",
        "pie": "pie_curve_result_file",
    }
    SIGNAL_LABELS = {
        "area": "分析信号（area，推荐）",
        "normalized_area": "归一化面积（normalized_area）",
        "photon_normalized_area": "光强归一化面积",
        "raw_area": "原始积分面积",
        "merged_intensity": "跨能段合并强度（推荐）",
        "io_time_normalized_intensity": "IO及扫描时间归一强度",
        "normalized_intensity": "归一化强度（推荐）",
        "photon_normalized_intensity": "光强归一化强度",
        "intensity": "信号强度",
        "强度": "信号强度",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("IsotopeCorrectionPage")
        self.project_settings: ProjectSettings | None = None
        self.project_scope_active = False
        self.input_df = pd.DataFrame()
        self._database_assignments = pd.DataFrame()
        self.curve_matrix = pd.DataFrame()
        self.correction_result: IsotopeCorrectionResult | None = None
        self.hypotheses: list[IsotopeHypothesis] = []
        self.axis_column = ""
        self.mass_column = ""
        self.source_path = ""
        self.loaded_database_path = ""
        self._loaded_project_dataset_id: int | None = None
        self._loaded_project_source_type = ""
        self._project_load_worker: WorkerThread | None = None
        self._project_load_request_id = 0
        self._project_datasets: dict[int, CurveDataset] = {}
        self._peak_track_selection: dict[int, object] = {}
        self._build_ui()
        self._update_project_source_button()
        self._update_mode_hint()
        self._update_action_state()

    def _build_ui(self) -> None:
        page_layout = QtWidgets.QVBoxLayout(self)
        page_layout.setContentsMargins(10, 8, 10, 8)
        page_layout.setSpacing(0)
        self.page_splitter = QtWidgets.QSplitter(
            QtCore.Qt.Orientation.Vertical,
            self,
        )
        self.page_splitter.setChildrenCollapsible(False)
        self.page_splitter.setHandleWidth(5)
        self.page_splitter.setToolTip("可上下拖动分隔线，调整设置区与曲线图区的高度")

        controls_scroll = QtWidgets.QScrollArea(self.page_splitter)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        controls_panel = QtWidgets.QWidget(controls_scroll)
        root = QtWidgets.QVBoxLayout(controls_panel)
        root.setContentsMargins(0, 0, 0, 4)
        root.setSpacing(7)

        header = QtWidgets.QWidget(self)
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        title_column = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("同位素贡献校正", header)
        title.setObjectName("ProjectTitle")
        subtitle = QtWidgets.QLabel(
            "按用户指定的分子式，对PIE或温度曲线计算同位素贡献与剩余信号；"
            "当前按单电荷、名义质量通道处理，不进行物种鉴定或自动生成结论。",
            header,
        )
        subtitle.setObjectName("ProjectHint")
        subtitle.setWordWrap(True)
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        header_layout.addLayout(title_column, 1)
        header.hide()

        source_group = QtWidgets.QGroupBox("1. 选择数据", self)
        source_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        source_layout = QtWidgets.QGridLayout(source_group)
        source_layout.setHorizontalSpacing(8)
        source_layout.setVerticalSpacing(7)
        self.source_type_combo = QtWidgets.QComboBox(source_group)
        self.source_type_combo.addItem("温度扫描结果", "temperature")
        self.source_type_combo.addItem("PIE结果", "pie")
        self.source_type_combo.currentIndexChanged.connect(self._on_source_type_changed)
        self.source_path_edit = QtWidgets.QLineEdit(source_group)
        self.source_path_edit.setReadOnly(True)
        self.source_path_edit.setPlaceholderText("项目 SQLite 或手动选择的兼容结果文件")
        self.source_path_edit.hide()
        self.project_dataset_combo = QtWidgets.QComboBox(source_group)
        self.project_dataset_combo.setMinimumContentsLength(36)
        self.project_dataset_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.project_dataset_combo.currentIndexChanged.connect(
            self._on_project_dataset_changed
        )
        self.project_dataset_combo.hide()
        self.refresh_source_button = QtWidgets.QPushButton("刷新", source_group)
        self.refresh_source_button.setObjectName("BrowseButton")
        self.refresh_source_button.setToolTip("重新查询项目 SQLite 和上游分析状态")
        self.refresh_source_button.clicked.connect(self.refresh_project_sources)
        self.project_source_button = QtWidgets.QPushButton("加载项目结果", source_group)
        self.project_source_button.setObjectName("BrowseButton")
        self.project_source_button.clicked.connect(self.load_project_result)
        self.browse_source_button = QtWidgets.QPushButton("导入文件…", source_group)
        self.browse_source_button.setObjectName("BrowseButton")
        self.browse_source_button.setToolTip(
            "仅用于导入旧版 CSV/Excel 或临时结果；项目分析优先使用 SQLite"
        )
        self.browse_source_button.clicked.connect(self.browse_result_file)

        self.signal_column_combo = QtWidgets.QComboBox(source_group)
        self.signal_column_combo.setEnabled(False)
        self.aggregation_combo = QtWidgets.QComboBox(source_group)
        self.aggregation_combo.addItem("同一坐标与质量取平均", "mean")
        self.aggregation_combo.addItem("同一坐标与质量求和", "sum")
        self.aggregation_combo.hide()
        self.energy_filter_label = QtWidgets.QLabel("能量", source_group)
        self.energy_filter_combo = QtWidgets.QComboBox(source_group)
        self.energy_filter_combo.setEnabled(False)
        self.energy_filter_combo.addItem("结果中无光子能量列", None)
        self.mz_min_spin = QtWidgets.QSpinBox(source_group)
        self.mz_min_spin.setRange(1, 100_000)
        self.mz_min_spin.setValue(1)
        self.mz_min_spin.hide()
        self.mz_max_spin = QtWidgets.QSpinBox(source_group)
        self.mz_max_spin.setRange(1, 100_000)
        self.mz_max_spin.setValue(500)
        self.mz_max_spin.hide()
        self.data_status_label = QtWidgets.QLabel("尚未载入结果文件", source_group)
        self.data_status_label.setObjectName("InlineStatusLabel")
        self.data_status_label.setWordWrap(False)

        source_layout.addWidget(QtWidgets.QLabel("曲线"), 0, 0)
        source_layout.addWidget(self.source_type_combo, 0, 1)
        source_layout.addWidget(QtWidgets.QLabel("项目结果"), 0, 2)
        source_layout.addWidget(self.project_source_button, 0, 3, 1, 3)
        source_layout.addWidget(self.refresh_source_button, 0, 6)
        source_layout.addWidget(self.browse_source_button, 0, 7)
        source_layout.addWidget(QtWidgets.QLabel("信号"), 1, 0)
        source_layout.addWidget(self.signal_column_combo, 1, 1)
        source_layout.addWidget(self.energy_filter_label, 1, 2)
        source_layout.addWidget(self.energy_filter_combo, 1, 3)
        source_layout.addWidget(self.data_status_label, 1, 4, 1, 4)
        source_layout.setColumnStretch(3, 1)
        root.addWidget(source_group)

        hypothesis_group = QtWidgets.QGroupBox("2. 设置母峰与分子式", self)
        hypothesis_group.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        hypothesis_layout = QtWidgets.QVBoxLayout(hypothesis_group)
        hypothesis_entry = QtWidgets.QGridLayout()
        hypothesis_entry.setHorizontalSpacing(8)
        hypothesis_entry.setVerticalSpacing(7)
        self.formula_edit = QtWidgets.QLineEdit(hypothesis_group)
        self.formula_edit.setPlaceholderText("例如 C11H8Cl2O")
        self.formula_edit.returnPressed.connect(self.add_hypothesis)
        self.formula_edit.textChanged.connect(self._sync_parent_mz_from_formula)
        self.parent_mz_combo = QtWidgets.QComboBox(hypothesis_group)
        self.parent_mz_combo.setEditable(True)
        self.parent_mz_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.parent_mz_combo.setMinimumContentsLength(8)
        self.parent_mz_combo.setMinimumWidth(125)
        self.parent_mz_combo.setMaximumWidth(170)
        self.parent_mz_combo.lineEdit().setPlaceholderText("选择 m/z")
        self.parent_mz_combo.setToolTip(
            "列出当前 SQLite 数据集中实际存在的名义质量通道；可输入 m/z 搜索"
        )
        parent_completer = self.parent_mz_combo.completer()
        if parent_completer is not None:
            parent_completer.setCaseSensitivity(
                QtCore.Qt.CaseSensitivity.CaseInsensitive
            )
            parent_completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        self.parent_mz_combo.currentIndexChanged.connect(
            self._on_parent_mz_changed
        )
        self.parent_mz_combo.addItem("未载入数据", None)
        self.parent_mz_combo.setEnabled(False)
        self.parent_mz_spin = QtWidgets.QSpinBox(hypothesis_group)
        self.parent_mz_spin.setRange(1, 100_000)
        self.parent_mz_spin.setValue(226)
        self.parent_mz_spin.setReadOnly(True)
        self.parent_mz_spin.setButtonSymbols(
            QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
        )
        self.parent_mz_spin.setToolTip("按单电荷分子式自动计算名义质量")
        self.parent_mz_spin.hide()
        self.fraction_spin = QtWidgets.QDoubleSpinBox(hypothesis_group)
        self.fraction_spin.setRange(0.0, 100.0)
        self.fraction_spin.setDecimals(3)
        self.fraction_spin.setSingleStep(5.0)
        self.fraction_spin.setValue(100.0)
        self.fraction_spin.setSuffix(" %")
        self.add_hypothesis_button = QtWidgets.QPushButton("添加分子式", hypothesis_group)
        self.add_hypothesis_button.setObjectName("PrimaryButton")
        self.add_hypothesis_button.clicked.connect(self.add_hypothesis)
        self.remove_hypothesis_button = QtWidgets.QPushButton("移除所选", hypothesis_group)
        self.remove_hypothesis_button.setObjectName("BrowseButton")
        self.remove_hypothesis_button.clicked.connect(self.remove_selected_hypotheses)
        hypothesis_entry.addWidget(QtWidgets.QLabel("处理母峰 m/z"), 0, 0)
        hypothesis_entry.addWidget(self.parent_mz_combo, 0, 1)
        hypothesis_entry.addWidget(QtWidgets.QLabel("分子式"), 0, 2)
        hypothesis_entry.addWidget(self.formula_edit, 0, 3, 1, 2)
        hypothesis_entry.addWidget(QtWidgets.QLabel("母峰中所占比例"), 0, 5)
        hypothesis_entry.addWidget(self.fraction_spin, 0, 6)
        hypothesis_entry.addWidget(self.add_hypothesis_button, 0, 7)
        self.database_candidate_combo = QtWidgets.QComboBox(hypothesis_group)
        self.database_candidate_combo.addItem("先选择处理母峰，再查看对应 PIE 候选", None)
        self.database_candidate_combo.setEnabled(False)
        self.database_candidate_combo.currentIndexChanged.connect(
            self._apply_database_candidate
        )
        hypothesis_entry.addWidget(QtWidgets.QLabel("该质量数的 PIE 候选"), 1, 0)
        hypothesis_entry.addWidget(self.database_candidate_combo, 1, 1, 1, 6)
        hypothesis_entry.addWidget(self.remove_hypothesis_button, 1, 7)
        hypothesis_entry.setColumnStretch(3, 1)
        hypothesis_layout.addLayout(hypothesis_entry)

        self.hypothesis_table = QtWidgets.QTableWidget(0, 8, hypothesis_group)
        self.hypothesis_table.setHorizontalHeaderLabels(
            ["编号", "母峰 m/z", "分子式", "母峰比例", "M+1", "M+2", "M+3", "M+4"]
        )
        self.hypothesis_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.hypothesis_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.hypothesis_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.hypothesis_table.setAlternatingRowColors(True)
        self.hypothesis_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        self.hypothesis_table.horizontalHeader().setSectionResizeMode(
            2,
            QtWidgets.QHeaderView.ResizeMode.Stretch,
        )
        self.hypothesis_table.setMinimumHeight(54)
        self.hypothesis_table.setMaximumHeight(82)
        hypothesis_layout.addWidget(self.hypothesis_table)

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("计算方式"))
        self.mode_combo = QtWidgets.QComboBox(hypothesis_group)
        self.mode_combo.addItem("按母峰比例", "manual_fraction")
        self.mode_combo.addItem("最大非负贡献", "max_compatible")
        self.mode_combo.setItemData(
            0,
            "按母峰实测曲线乘以设定比例计算同位素贡献",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.mode_combo.setItemData(
            1,
            "逐坐标求不产生负剩余的最大理论贡献",
            QtCore.Qt.ItemDataRole.ToolTipRole,
        )
        self.mode_combo.currentIndexChanged.connect(self._update_mode_hint)
        mode_row.addWidget(self.mode_combo)
        self.mode_hint_label = QtWidgets.QLabel("", hypothesis_group)
        self.mode_hint_label.setObjectName("HintLabel")
        self.mode_hint_label.setWordWrap(True)
        self.mode_hint_label.hide()
        mode_row.addStretch(1)
        self.calculate_button = QtWidgets.QPushButton("计算并显示曲线", hypothesis_group)
        self.calculate_button.setObjectName("PrimaryButton")
        self.calculate_button.clicked.connect(self.calculate_correction)
        mode_row.addWidget(self.calculate_button)
        hypothesis_layout.addLayout(mode_row)
        root.addWidget(hypothesis_group)

        result_group = QtWidgets.QGroupBox("3. 结果", self)
        result_layout = QtWidgets.QVBoxLayout(result_group)
        result_toolbar = QtWidgets.QHBoxLayout()
        result_toolbar.addWidget(QtWidgets.QLabel("显示质量通道"))
        self.result_mz_combo = QtWidgets.QComboBox(result_group)
        self.result_mz_combo.setEnabled(False)
        self.result_mz_combo.currentIndexChanged.connect(self.update_result_plot)
        result_toolbar.addWidget(self.result_mz_combo)
        self.result_status_label = QtWidgets.QLabel("等待计算", result_group)
        self.result_status_label.setObjectName("InlineStatusLabel")
        result_toolbar.addWidget(self.result_status_label, 1)
        self.export_button = QtWidgets.QPushButton("导出后处理结果", result_group)
        self.export_button.setObjectName("BrowseButton")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_result)
        result_toolbar.addWidget(self.export_button)
        result_layout.addLayout(result_toolbar)

        self.result_tabs = QtWidgets.QTabWidget(result_group)
        self.plot_widget = StaticCurvePlot("温度", "信号", min_height=300, parent=self.result_tabs)
        self.result_tabs.addTab(self.plot_widget, "曲线")
        self.curve_table_widget = QtWidgets.QTableWidget(self.result_tabs)
        self.result_tabs.addTab(self.curve_table_widget, "数据")
        self.pattern_table_widget = QtWidgets.QTableWidget(self.result_tabs)
        self.result_tabs.addTab(self.pattern_table_widget, "同位素比例")
        self.source_table_widget = QtWidgets.QTableWidget(self.result_tabs)
        self.result_tabs.addTab(self.source_table_widget, "母峰来源")
        self.diagnostic_table_widget = QtWidgets.QTableWidget(self.result_tabs)
        self.result_tabs.addTab(self.diagnostic_table_widget, "数值诊断")
        self.sensitivity_table_widget = QtWidgets.QTableWidget(self.result_tabs)
        self.result_tabs.addTab(self.sensitivity_table_widget, "系数灵敏度")
        result_layout.addWidget(self.result_tabs, 1)
        controls_scroll.setWidget(controls_panel)
        self.page_splitter.addWidget(controls_scroll)
        self.page_splitter.addWidget(result_group)
        self.page_splitter.setStretchFactor(0, 0)
        self.page_splitter.setStretchFactor(1, 1)
        self.page_splitter.setSizes([330, 650])
        page_layout.addWidget(self.page_splitter, 1)

    @staticmethod
    def _set_status(label: QtWidgets.QLabel, text: str, status: str = "") -> None:
        label.setText(text)
        label.setProperty("status", status)
        label.style().unpolish(label)
        label.style().polish(label)

    def set_project_settings(
        self,
        project_settings: ProjectSettings | None,
        *,
        activate_project_scope: bool | None = None,
    ) -> None:
        self.project_settings = project_settings
        if activate_project_scope is not None:
            self.project_scope_active = bool(activate_project_scope)
        self._update_project_source_button()

    def _source_type(self) -> str:
        return str(self.source_type_combo.currentData())

    def _on_source_type_changed(self, _index: int) -> None:
        self._project_load_request_id += 1
        if not self.input_df.empty:
            self.clear_loaded_data()
        self._update_project_source_button()
        if self.project_scope_active:
            self.ensure_project_source_loaded()

    def _on_project_dataset_changed(self, _index: int) -> None:
        self._update_project_source_action()

    def refresh_project_sources(self) -> None:
        current = self._update_project_source_button()
        if current is not None and (self.input_df.empty or self.loaded_database_path):
            self._start_project_dataset_load(current)

    def _project_curve_selection(
        self,
    ) -> tuple[Path | None, list[CurveDataset], CurveDataset | None]:
        source_type = self._source_type()
        database_path = (
            project_curve_database_path(self.project_settings)
            if self.project_settings is not None
            else None
        )
        all_datasets = (
            [
                dataset
                for dataset in list_curve_datasets_read_only(
                    database_path,
                    curve_type=source_type,
                    include_stale=True,
                )
                if dataset.dataset_group == self.PROJECT_RESULT_GROUPS[source_type]
                and dataset.validity_status == "valid"
            ]
            if database_path is not None and database_path.is_file()
            else []
        )
        current = all_datasets[0] if all_datasets else None
        return database_path, all_datasets, current

    def _populate_project_dataset_combo(
        self,
        datasets: list[CurveDataset],
        current: CurveDataset | None,
    ) -> None:
        selected_id = self.project_dataset_combo.currentData()
        if selected_id is None:
            selected_id = self._loaded_project_dataset_id
        available_ids = {dataset.dataset_id for dataset in datasets}
        if selected_id not in available_ids:
            selected_id = current.dataset_id if current is not None else None
        if selected_id is None and datasets:
            selected_id = datasets[0].dataset_id

        self._project_datasets = {
            dataset.dataset_id: dataset for dataset in datasets
        }
        previous = self.project_dataset_combo.blockSignals(True)
        self.project_dataset_combo.clear()
        for dataset in datasets:
            self.project_dataset_combo.addItem(
                f"{dataset.name} · "
                f"{dataset.channel_count} 峰 / {dataset.point_count} 点",
                dataset.dataset_id,
            )
        if selected_id is not None:
            index = self.project_dataset_combo.findData(selected_id)
            if index >= 0:
                self.project_dataset_combo.setCurrentIndex(index)
        self.project_dataset_combo.setEnabled(bool(datasets))
        self.project_dataset_combo.blockSignals(previous)

    def _selected_project_dataset(self) -> CurveDataset | None:
        dataset_id = self.project_dataset_combo.currentData()
        try:
            return self._project_datasets.get(int(dataset_id))
        except (TypeError, ValueError):
            return None

    def _update_project_source_action(self) -> None:
        selected = self._selected_project_dataset()
        if selected is None:
            return
        loaded = (
            self._loaded_project_dataset_id == selected.dataset_id
            and self._loaded_project_source_type == self._source_type()
            and bool(self.loaded_database_path)
            and not self.input_df.empty
        )
        suffix = {
            "valid": "",
            "stale": "（过期）",
            "archived": "（归档）",
        }.get(selected.validity_status, f"（{selected.validity_status}）")
        self.project_source_button.setEnabled(not loaded)
        self.project_source_button.setVisible(not loaded)
        self.project_source_button.setText(
            f"当前使用项目结果{suffix}"
            if loaded
            else f"加载项目结果{suffix}"
        )
        details = [
            f"状态={selected.validity_status}",
            f"精确质量通道={selected.channel_count}",
            f"曲线点={selected.point_count}",
        ]
        if selected.stale_reason:
            details.append(f"原因={selected.stale_reason}")
        self.project_source_button.setToolTip("\n".join(details))

    def _update_project_source_button(self) -> CurveDataset | None:
        source_type = self._source_type()
        try:
            database_path, all_datasets, current = self._project_curve_selection()
        except Exception as exc:
            self.project_source_button.setEnabled(False)
            self.project_source_button.setToolTip(str(exc))
            self.project_source_button.setText("SQLite读取失败")
            self.project_dataset_combo.clear()
            self.project_dataset_combo.setEnabled(False)
            if self.input_df.empty:
                self._set_status(
                    self.data_status_label,
                    f"项目 SQLite 读取失败：{exc}",
                    "error",
                )
            return None
        self._populate_project_dataset_combo(all_datasets, current)
        if database_path is not None and database_path.is_file() and self.input_df.empty:
            self.source_path_edit.setText(f"{database_path} · 尚未加载数据集")
        if current is not None:
            self._update_project_source_action()
            if self.input_df.empty:
                self._set_status(
                    self.data_status_label,
                    "正式来源：项目 SQLite；"
                    f"{current.channel_count} 个精确质量通道，"
                    f"{current.point_count} 个曲线点。",
                    "success",
                )
            return current
        self.project_dataset_combo.clear()
        self.project_dataset_combo.addItem("项目 SQLite 中没有该类型的有效结果", None)
        self.project_dataset_combo.setEnabled(False)
        field_name = self.SOURCE_ARTIFACT_FIELDS[source_type]
        path = (
            str(getattr(self.project_settings, field_name, "") or "")
            if self.project_settings
            else ""
        )
        self.project_source_button.setEnabled(bool(path))
        if path:
            self.project_source_button.setToolTip(path)
            self.project_source_button.setText(
                f"使用项目{self.SOURCE_LABELS[source_type]}"
            )
        else:
            self.project_source_button.setToolTip("当前项目未登记对应的分析结果")
            self.project_source_button.setText("项目无对应结果")
            if self.input_df.empty:
                path_text = str(database_path) if database_path is not None else "未配置"
                self._set_status(
                    self.data_status_label,
                    f"项目 SQLite：{path_text}；"
                    f"没有有效的{self.SOURCE_LABELS[source_type]}。",
                    "warning",
                )
        return None

    def ensure_project_source_loaded(self) -> bool:
        """Refresh and lazily load the current project SQLite dataset on page entry."""
        current = self._update_project_source_button()
        if not self.project_scope_active or current is None:
            return False
        if (
            self._loaded_project_dataset_id == current.dataset_id
            and self._loaded_project_source_type == self._source_type()
            and not self.input_df.empty
        ):
            return True
        # A manually selected/exported file remains the user's explicit source.
        if not self.input_df.empty and not self.loaded_database_path:
            return True
        return self._start_project_dataset_load(current)

    def _start_project_dataset_load(
        self,
        dataset: CurveDataset,
        *,
        show_message: bool = False,
    ) -> bool:
        database_path = (
            project_curve_database_path(self.project_settings)
            if self.project_settings is not None
            else None
        )
        if database_path is None or not database_path.is_file():
            return False
        worker = self._project_load_worker
        if worker is not None and worker.isRunning():
            return True
        selected_index = self.project_dataset_combo.findData(dataset.dataset_id)
        if selected_index >= 0:
            previous = self.project_dataset_combo.blockSignals(True)
            self.project_dataset_combo.setCurrentIndex(selected_index)
            self.project_dataset_combo.blockSignals(previous)
        self._project_load_request_id += 1
        request_id = self._project_load_request_id
        source_type = self._source_type()
        source_label = f"{database_path} · 项目 SQLite · {dataset.name}"
        self.project_source_button.setEnabled(False)
        self.project_source_button.setText("正在读取项目结果…")
        self.source_type_combo.setEnabled(False)
        self.project_dataset_combo.setEnabled(False)
        self.refresh_source_button.setEnabled(False)
        self.browse_source_button.setEnabled(False)
        self._set_status(
            self.data_status_label,
            f"正在从项目 curve_data.sqlite 读取{self.SOURCE_LABELS[source_type]}…",
            "busy",
        )
        worker = WorkerThread(
            lambda: load_curve_dataset_read_only(database_path, dataset.dataset_id),
            self,
        )
        self._project_load_worker = worker
        worker.finished_with_result.connect(
            lambda data, expected_request=request_id: self._finish_project_dataset_load(
                data,
                expected_request=expected_request,
                database_path=database_path,
                dataset=dataset,
                source_type=source_type,
                source_label=source_label,
                show_message=show_message,
            )
        )
        worker.failed.connect(
            lambda message, expected_request=request_id: self._fail_project_dataset_load(
                message,
                expected_request=expected_request,
                show_message=show_message,
            )
        )
        worker.start()
        return True

    def _finish_project_dataset_load(
        self,
        data: pd.DataFrame,
        *,
        expected_request: int,
        database_path: Path,
        dataset: CurveDataset,
        source_type: str,
        source_label: str,
        show_message: bool,
    ) -> None:
        if (
            expected_request != self._project_load_request_id
            or source_type != self._source_type()
        ):
            return
        try:
            self.loaded_database_path = str(database_path)
            self._loaded_project_dataset_id = dataset.dataset_id
            self._loaded_project_source_type = source_type
            self.load_dataframe(
                data,
                source_type=source_type,
                source_label=source_label,
            )
            self._populate_database_candidates(database_path)
            self.source_type_combo.setEnabled(True)
            self.project_dataset_combo.setEnabled(True)
            self.refresh_source_button.setEnabled(True)
            self.browse_source_button.setEnabled(True)
            self._update_project_source_button()
            self.source_path_edit.setText(
                f"项目 SQLite · {self.SOURCE_LABELS[source_type]} · "
                f"{dataset.name}"
            )
            self.source_path_edit.setToolTip(source_label)
        except Exception as exc:
            self._fail_project_dataset_load(
                str(exc),
                expected_request=expected_request,
                show_message=show_message,
            )

    def _fail_project_dataset_load(
        self,
        message: str,
        *,
        expected_request: int,
        show_message: bool,
    ) -> None:
        if expected_request != self._project_load_request_id:
            return
        self.loaded_database_path = ""
        self._loaded_project_dataset_id = None
        self._loaded_project_source_type = ""
        self.source_type_combo.setEnabled(True)
        self.project_dataset_combo.setEnabled(bool(self._project_datasets))
        self.refresh_source_button.setEnabled(True)
        self.browse_source_button.setEnabled(True)
        self._set_status(self.data_status_label, f"SQLite载入失败：{message}", "error")
        self._update_project_source_button()
        if show_message:
            QtWidgets.QMessageBox.critical(self, "SQLite载入失败", message)

    def load_project_result(self) -> None:
        source_type = self._source_type()
        try:
            _database_path, all_datasets, current = self._project_curve_selection()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "SQLite读取失败", str(exc))
            return
        selected = self._selected_project_dataset() or current
        if selected is not None:
            self._start_project_dataset_load(selected, show_message=True)
            return
        field_name = self.SOURCE_ARTIFACT_FIELDS[source_type]
        path = str(getattr(self.project_settings, field_name, "") or "") if self.project_settings else ""
        if not path:
            QtWidgets.QMessageBox.warning(self, "提示", "当前项目未登记对应的分析结果文件")
            return
        if not Path(path).is_file():
            QtWidgets.QMessageBox.warning(self, "提示", f"项目登记的结果文件不存在：\n{path}")
            return
        self.load_result_file(path)

    def browse_result_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            f"选择{self.SOURCE_LABELS[self._source_type()]}",
            str(Path(self.source_path).parent) if self.source_path else "",
            "Result Tables (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)",
        )
        if path:
            self.load_result_file(path)

    def load_result_file(self, path: str | Path, *, show_message: bool = True) -> None:
        path = Path(path)
        try:
            if path.suffix.lower() in {".xlsx", ".xls"}:
                data = pd.read_excel(path)
            else:
                data = pd.read_csv(path)
            self.loaded_database_path = ""
            self._loaded_project_dataset_id = None
            self._loaded_project_source_type = ""
            self._clear_database_candidates()
            self.load_dataframe(data, source_type=self._source_type(), source_label=str(path))
        except Exception as exc:
            self._set_status(self.data_status_label, f"载入失败：{exc}", "error")
            if show_message:
                QtWidgets.QMessageBox.critical(self, "载入失败", str(exc))

    def load_dataframe(
        self,
        data: pd.DataFrame,
        *,
        source_type: str | None = None,
        source_label: str = "",
    ) -> None:
        if source_type is not None:
            for index in range(self.source_type_combo.count()):
                if self.source_type_combo.itemData(index) == source_type:
                    self.source_type_combo.blockSignals(True)
                    self.source_type_combo.setCurrentIndex(index)
                    self.source_type_combo.blockSignals(False)
                    break
        source_type = self._source_type()
        inferred = infer_curve_columns(data, source_type)
        self._peak_track_selection = {}
        self.input_df = data.copy()
        self.axis_column = str(inferred["axis_column"])
        self.mass_column = str(inferred["mass_column"])
        self.source_path = source_label
        self.source_path_edit.setText(source_label)
        self.source_path_edit.setToolTip(source_label)
        self.signal_column_combo.clear()
        for column in inferred["signal_columns"]:
            column_name = str(column)
            self.signal_column_combo.addItem(
                self.SIGNAL_LABELS.get(column_name, column_name),
                column_name,
            )
        self.signal_column_combo.setEnabled(True)
        self._populate_energy_filter()

        masses = pd.to_numeric(self.input_df[self.mass_column], errors="coerce").dropna()
        if masses.empty:
            raise ValueError("结果表的质量列没有有效数值")
        minimum = int(np.rint(masses).min())
        maximum = int(np.rint(masses).max())
        self.mz_min_spin.setValue(minimum)
        self.mz_max_spin.setValue(maximum)
        self._populate_parent_mz_combo()
        energy_note = ""
        if source_type == "temperature" and self.energy_filter_combo.currentData() is not None:
            energy_note = f" · {self.energy_filter_combo.count()} 个能量"
        channel_count = int(np.rint(masses).nunique())
        self._set_status(
            self.data_status_label,
            f"已载入 · {channel_count} 个质量通道{energy_note}",
            "success",
        )
        self.data_status_label.setToolTip(
            f"{len(self.input_df):,} 行；质量范围 {minimum}–{maximum}；"
            f"坐标列={self.axis_column}；质量列={self.mass_column}；"
            f"校正信号={self.signal_column_combo.currentData()}"
        )
        self._update_action_state()
        self._clear_results()

    def _populate_energy_filter(self) -> None:
        previous_energy = self.energy_filter_combo.currentData()
        self.energy_filter_combo.blockSignals(True)
        self.energy_filter_combo.clear()
        if self._source_type() != "temperature":
            self.energy_filter_combo.addItem("PIE数据不适用", None)
            self.energy_filter_combo.setEnabled(False)
            self.energy_filter_label.setEnabled(False)
        elif "photon_energy" not in self.input_df:
            self.energy_filter_combo.addItem("结果中无光子能量列", None)
            self.energy_filter_combo.setEnabled(False)
            self.energy_filter_label.setEnabled(True)
        else:
            energy_values = pd.to_numeric(
                self.input_df["photon_energy"],
                errors="coerce",
            )
            rounded = energy_values.round(self.ENERGY_GROUP_DECIMALS)
            energy_groups = sorted(float(value) for value in rounded.dropna().unique())
            if energy_groups:
                for energy in energy_groups:
                    group_rows = self.input_df.loc[rounded == energy]
                    coordinate_count = pd.to_numeric(
                        group_rows[self.axis_column],
                        errors="coerce",
                    ).nunique()
                    self.energy_filter_combo.addItem(
                        f"{energy:.2f} eV · {coordinate_count} 个温度点",
                        energy,
                    )
                self.energy_filter_combo.setEnabled(True)
                if previous_energy is not None:
                    nearest_index = min(
                        range(self.energy_filter_combo.count()),
                        key=lambda index: abs(
                            float(self.energy_filter_combo.itemData(index))
                            - float(previous_energy)
                        ),
                    )
                    self.energy_filter_combo.setCurrentIndex(nearest_index)
            else:
                self.energy_filter_combo.addItem("结果中无有效光子能量", None)
                self.energy_filter_combo.setEnabled(False)
            self.energy_filter_label.setEnabled(True)
        self.energy_filter_combo.blockSignals(False)

    def _selected_input_data(self) -> pd.DataFrame:
        if self._source_type() != "temperature" or "photon_energy" not in self.input_df:
            return self.input_df
        selected_energy = self.energy_filter_combo.currentData()
        if selected_energy is None:
            return self.input_df
        energies = pd.to_numeric(
            self.input_df["photon_energy"],
            errors="coerce",
        ).round(self.ENERGY_GROUP_DECIMALS)
        return self.input_df.loc[
            energies == round(float(selected_energy), self.ENERGY_GROUP_DECIMALS)
        ].copy()

    def clear_loaded_data(self) -> None:
        self.input_df = pd.DataFrame()
        self.curve_matrix = pd.DataFrame()
        self.source_path = ""
        self.loaded_database_path = ""
        self._loaded_project_dataset_id = None
        self._loaded_project_source_type = ""
        self._peak_track_selection = {}
        self.source_path_edit.clear()
        self.signal_column_combo.clear()
        self.signal_column_combo.setEnabled(False)
        self.energy_filter_combo.clear()
        self.energy_filter_combo.addItem("结果中无光子能量列", None)
        self.energy_filter_combo.setEnabled(False)
        self._clear_parent_mz_combo()
        self._clear_database_candidates()
        self._set_status(self.data_status_label, "尚未载入结果文件")
        self._update_action_state()
        self._clear_results()

    def _clear_database_candidates(self) -> None:
        self._database_assignments = pd.DataFrame()
        self._refresh_database_candidate_combo()

    def _clear_parent_mz_combo(self) -> None:
        self.parent_mz_combo.blockSignals(True)
        self.parent_mz_combo.clear()
        self.parent_mz_combo.addItem("未载入数据", None)
        self.parent_mz_combo.setCurrentIndex(0)
        self.parent_mz_combo.setEnabled(False)
        self.parent_mz_combo.blockSignals(False)

    def _populate_parent_mz_combo(self) -> None:
        previous_mass = self.parent_mz_combo.currentData()
        masses = pd.to_numeric(
            self.input_df[self.mass_column],
            errors="coerce",
        )
        nominal = np.rint(masses)
        exact_values = (
            pd.to_numeric(self.input_df["mz"], errors="coerce")
            if "mz" in self.input_df
            else masses
        )
        available = pd.DataFrame(
            {"nominal_mz": nominal, "exact_mz": exact_values}
        ).dropna()
        counts = (
            available.groupby("nominal_mz", sort=True)["exact_mz"]
            .nunique()
            .astype(int)
        )

        self.parent_mz_combo.blockSignals(True)
        self.parent_mz_combo.clear()
        self.parent_mz_combo.addItem("选择 m/z", None)
        for nominal_mz in counts.index:
            nominal_value = int(nominal_mz)
            self.parent_mz_combo.addItem(
                f"m/z {nominal_value}",
                nominal_value,
            )
        selected_index = (
            self.parent_mz_combo.findData(int(previous_mass))
            if previous_mass is not None
            else 0
        )
        self.parent_mz_combo.setCurrentIndex(
            selected_index if selected_index >= 0 else 0
        )
        self.parent_mz_combo.setEnabled(self.parent_mz_combo.count() > 1)
        self.parent_mz_combo.blockSignals(False)
        self._on_parent_mz_changed(self.parent_mz_combo.currentIndex())

    def _selected_parent_mz(self) -> int | None:
        value = self.parent_mz_combo.currentData()
        return None if value is None else int(value)

    def _select_parent_mz(self, nominal_mz: int) -> bool:
        index = self.parent_mz_combo.findData(int(nominal_mz))
        if index < 0:
            return False
        self.parent_mz_combo.setCurrentIndex(index)
        return True

    def _on_parent_mz_changed(self, _index: int) -> None:
        parent_mz = self._selected_parent_mz()
        if parent_mz is not None:
            self.parent_mz_spin.setValue(parent_mz)
            if not self.hypotheses:
                self.mz_min_spin.setValue(parent_mz)
                self.mz_max_spin.setValue(parent_mz + 4)
        if hasattr(self, "database_candidate_combo"):
            self._refresh_database_candidate_combo()

    def _refresh_database_candidate_combo(self) -> None:
        self.database_candidate_combo.blockSignals(True)
        self.database_candidate_combo.clear()
        parent_mz = self._selected_parent_mz()
        if parent_mz is None:
            self.database_candidate_combo.addItem(
                "先选择处理母峰，再查看对应 PIE 候选",
                None,
            )
            self.database_candidate_combo.setEnabled(False)
            self.database_candidate_combo.blockSignals(False)
            return

        assignments = self._database_assignments
        if not assignments.empty and "nominal_mz" in assignments:
            nominal_values = pd.to_numeric(
                assignments["nominal_mz"],
                errors="coerce",
            )
            assignments = assignments.loc[nominal_values == parent_mz]

        candidates_added = 0
        self.database_candidate_combo.addItem(
            f"选择 m/z {parent_mz} 的候选以填入分子式（不会自动加入或确认）",
            None,
        )
        for _, row in assignments.iterrows():
            formula = str(row.get("formula") or "").strip()
            if not formula:
                continue
            exact_mz = float(row["exact_mz"])
            status = (
                "已确认"
                if row.get("assignment_status") == "confirmed"
                else "拟合候选"
            )
            name = str(row.get("species_name") or "").strip()
            self.database_candidate_combo.addItem(
                f"m/z {exact_mz:.6f} · {formula} · "
                f"{name or '未命名'} · {status}",
                {
                    "formula": formula,
                    "nominal_mz": parent_mz,
                },
            )
            candidates_added += 1
        if candidates_added == 0:
            self.database_candidate_combo.setItemText(
                0,
                f"m/z {parent_mz} 暂无带分子式的 PIE 候选",
            )
        self.database_candidate_combo.setEnabled(candidates_added > 0)
        self.database_candidate_combo.blockSignals(False)

    def _populate_database_candidates(self, database_path: str | Path) -> None:
        pie_datasets = [
            dataset
            for dataset in list_curve_datasets_read_only(
                database_path,
                curve_type="pie",
                include_stale=True,
            )
            if dataset.dataset_group == self.PROJECT_RESULT_GROUPS["pie"]
            and dataset.validity_status == "valid"
        ]
        assignments = (
            list_species_assignments(
                database_path,
                dataset_id=pie_datasets[0].dataset_id,
            )
            if pie_datasets
            else pd.DataFrame()
        )
        self._database_assignments = assignments.copy()
        self._refresh_database_candidate_combo()

    def _apply_database_candidate(self, _index: int) -> None:
        candidate = self.database_candidate_combo.currentData()
        if not candidate:
            return
        if int(candidate["nominal_mz"]) != self._selected_parent_mz():
            return
        self.formula_edit.setText(str(candidate["formula"]))

    def _sync_parent_mz_from_formula(self, formula: str) -> None:
        try:
            nominal_mass = formula_nominal_mass(str(formula).strip())
        except Exception:
            return
        if self._selected_parent_mz() is None:
            self._select_parent_mz(nominal_mass)

    def add_hypothesis(self) -> None:
        formula = self.formula_edit.text().strip()
        try:
            nominal_mass = formula_nominal_mass(formula)
            parent_mz = self._selected_parent_mz()
            if parent_mz is None and self._select_parent_mz(nominal_mass):
                parent_mz = self._selected_parent_mz()
            if parent_mz is None:
                raise ValueError("请先从下拉框选择要处理的母峰质量数")
            if nominal_mass != parent_mz:
                raise ValueError(
                    f"分子式 {formula} 的单电荷名义质量为 {nominal_mass}，"
                    f"与所选母峰 m/z {parent_mz} 不一致"
                )
            hypothesis = IsotopeHypothesis(
                formula=formula,
                parent_mz=parent_mz,
                fraction=self.fraction_spin.value() / 100.0,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "无法添加分子式", str(exc))
            return
        if any(
            existing.formula == hypothesis.formula
            and existing.parent_mz == hypothesis.parent_mz
            for existing in self.hypotheses
        ):
            QtWidgets.QMessageBox.information(self, "提示", "相同分子式和母峰已经存在")
            return
        self.hypotheses.append(hypothesis)
        self.mz_min_spin.setValue(
            min(item.parent_mz for item in self.hypotheses)
        )
        self.mz_max_spin.setValue(
            max(item.parent_mz + 4 for item in self.hypotheses)
        )
        self.formula_edit.clear()
        self._refresh_hypothesis_table()
        self._update_action_state()
        self._clear_results()

    def remove_selected_hypotheses(self) -> None:
        selected_rows = sorted(
            {index.row() for index in self.hypothesis_table.selectionModel().selectedRows()},
            reverse=True,
        )
        for row in selected_rows:
            if 0 <= row < len(self.hypotheses):
                self.hypotheses.pop(row)
        self._refresh_hypothesis_table()
        self._update_action_state()
        self._clear_results()

    def _refresh_hypothesis_table(self) -> None:
        self.hypothesis_table.setRowCount(len(self.hypotheses))
        for row, hypothesis in enumerate(self.hypotheses):
            pattern = nominal_isotope_pattern(hypothesis.formula, max_shift=4)
            values = [
                f"H{row + 1}",
                str(hypothesis.parent_mz),
                hypothesis.formula,
                f"{hypothesis.fraction * 100:.3f} %",
                f"{pattern.get(1, 0.0):.6g}",
                f"{pattern.get(2, 0.0):.6g}",
                f"{pattern.get(3, 0.0):.6g}",
                f"{pattern.get(4, 0.0):.6g}",
            ]
            for column, value in enumerate(values):
                self.hypothesis_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))

    def _update_mode_hint(self, _index: int | None = None) -> None:
        mode = str(self.mode_combo.currentData())
        manual = mode == "manual_fraction"
        self.fraction_spin.setEnabled(manual)
        if manual:
            text = "每个分子式使用其母峰实测曲线 × 用户比例；多个假设会同时扣除，结果允许出现负剩余。"
        else:
            text = "逐坐标点求最大理论贡献，并约束所有贡献与剩余均非负；表中的母峰比例不参与计算。"
        self.mode_hint_label.setText(text)

    def _update_action_state(self) -> None:
        self.calculate_button.setEnabled(not self.input_df.empty and bool(self.hypotheses))

    def calculate_correction(self) -> None:
        if self.input_df.empty:
            QtWidgets.QMessageBox.warning(self, "提示", "请先载入PIE或温度扫描结果")
            return
        if not self.hypotheses:
            QtWidgets.QMessageBox.warning(self, "提示", "请至少添加一个分子式假设")
            return
        if self.mz_min_spin.value() > self.mz_max_spin.value():
            QtWidgets.QMessageBox.warning(self, "提示", "质量范围的起点不能大于终点")
            return
        try:
            selected_input = self._selected_input_data()
            peak_track_selection = self._resolve_peak_track_collisions(
                selected_input
            )
            self.curve_matrix = prepare_curve_matrix(
                selected_input,
                axis_column=self.axis_column,
                mass_column=self.mass_column,
                intensity_column=str(self.signal_column_combo.currentData()),
                mz_min=self.mz_min_spin.value(),
                mz_max=self.mz_max_spin.value(),
                aggregation=str(self.aggregation_combo.currentData()),
                peak_track_selection=peak_track_selection,
            )
            incomplete_columns = [
                int(column)
                for column in self.curve_matrix.columns
                if self.curve_matrix[column].isna().any()
            ]
            if incomplete_columns:
                raise ValueError(
                    "以下质量通道在部分坐标点缺少信号："
                    + "、".join(str(value) for value in incomplete_columns)
                )
            self.correction_result = apply_isotope_correction(
                self.curve_matrix,
                self.hypotheses,
                mode=str(self.mode_combo.currentData()),
            )
        except Exception as exc:
            self._set_status(self.result_status_label, f"计算失败：{exc}", "error")
            QtWidgets.QMessageBox.warning(self, "计算失败", str(exc))
            return
        self._show_results()

    def _resolve_peak_track_collisions(
        self,
        data: pd.DataFrame,
    ) -> dict[int, object]:
        if "peak_track" not in data.columns or self.mass_column not in data.columns:
            return {}
        nominal = np.rint(
            pd.to_numeric(data[self.mass_column], errors="coerce")
        )
        working = pd.DataFrame(
            {
                "nominal_mz": nominal,
                "peak_track": data["peak_track"],
            }
        ).dropna()
        selection = dict(self._peak_track_selection)
        for mass, group in working.groupby("nominal_mz", sort=True):
            tracks = list(dict.fromkeys(group["peak_track"].tolist()))
            if len(tracks) <= 1:
                continue
            nominal_mz = int(mass)
            if nominal_mz in selection and selection[nominal_mz] in tracks:
                continue
            labels = [str(value) for value in tracks]
            chosen, accepted = QtWidgets.QInputDialog.getItem(
                self,
                "选择精确峰轨道",
                f"m/z {nominal_mz} 存在 {len(tracks)} 条精确峰轨道，"
                "同位素矩阵不能自动平均。请选择要使用的 peak_track：",
                labels,
                0,
                False,
            )
            if not accepted:
                raise ValueError(
                    f"m/z {nominal_mz} 尚未选择具体 peak_track，计算已停止"
                )
            selected_index = labels.index(str(chosen))
            selection[nominal_mz] = tracks[selected_index]
        self._peak_track_selection = selection
        return selection

    def _show_results(self) -> None:
        result = self.correction_result
        if result is None:
            return
        self.result_mz_combo.blockSignals(True)
        self.result_mz_combo.clear()
        for mass in sorted(result.curve_table["mz"].unique()):
            self.result_mz_combo.addItem(f"m/z {int(mass)}", int(mass))
        self.result_mz_combo.blockSignals(False)
        self.result_mz_combo.setEnabled(self.result_mz_combo.count() > 0)

        curve_display = result.curve_table.copy()
        for column in (
            "observed",
            "isotope_contribution",
            "residual",
            "isotope_contribution_percent",
            "residual_percent",
        ):
            curve_display[column] = curve_display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.8g}"
            )
        curve_display = curve_display.rename(
            columns={
                result.axis_name: "温度" if self._source_type() == "temperature" else "光子能量",
                "mz": "m/z",
                "observed": "实测信号",
                "isotope_contribution": "计算贡献",
                "residual": "扣除后信号",
                "isotope_contribution_percent": "计算贡献占比(%)",
                "residual_percent": "扣除后占比(%)",
                "mode": "计算模式",
            }
        )
        self.set_dataframe(self.curve_table_widget, curve_display)

        pattern_display = result.pattern_table.copy()
        if not pattern_display.empty:
            pattern_display["fraction"] = pattern_display["fraction"].map(lambda value: f"{float(value):.6g}")
            pattern_display["isotope_ratio"] = pattern_display["isotope_ratio"].map(
                lambda value: f"{float(value):.8g}"
            )
        pattern_display = pattern_display.rename(
            columns={
                "hypothesis": "编号",
                "label": "标签",
                "formula": "分子式",
                "parent_mz": "母峰m/z",
                "fraction": "母峰比例",
                "mz": "贡献通道m/z",
                "mass_shift": "质量偏移",
                "isotope_ratio": "理论比例",
            }
        )
        self.set_dataframe(self.pattern_table_widget, pattern_display)

        source_display = result.source_table.copy()
        if not source_display.empty:
            source_display["source_amplitude"] = source_display["source_amplitude"].map(
                lambda value: f"{float(value):.8g}"
            )
        source_display = source_display.rename(
            columns={
                result.axis_name: "温度" if self._source_type() == "temperature" else "光子能量",
                "hypothesis": "编号",
                "label": "标签",
                "formula": "分子式",
                "parent_mz": "母峰m/z",
                "fraction": "母峰比例",
                "source_amplitude": "来源曲线强度",
                "mode": "计算模式",
            }
        )
        self.set_dataframe(self.source_table_widget, source_display)

        diagnostic_display = result.diagnostic_table.copy()
        self.set_dataframe(self.diagnostic_table_widget, diagnostic_display)
        sensitivity_display = result.sensitivity_table.copy()
        self.set_dataframe(self.sensitivity_table_widget, sensitivity_display)

        rank_text = f"理论贡献矩阵秩 {result.pattern_rank}/{len(self.hypotheses)}"
        if np.isfinite(result.pattern_condition_number):
            rank_text += f"，条件数 {result.pattern_condition_number:.4g}"
        else:
            rank_text += "，条件数 ∞"
        self._set_status(
            self.result_status_label,
            f"已计算 · {self.curve_matrix.shape[0]} 个坐标点 · "
            f"{self.curve_matrix.shape[1]} 个质量通道",
            "success",
        )
        self.result_status_label.setToolTip(
            f"{CORRECTION_MODES[result.mode]}；{rank_text}\n"
            f"{result.scientific_note}"
        )
        self.export_button.setEnabled(True)
        self.update_result_plot()

    def update_result_plot(self, _index: int | None = None) -> None:
        result = self.correction_result
        mass = self.result_mz_combo.currentData()
        if result is None or mass is None:
            self.plot_widget.show_empty("尚无可显示的后处理曲线")
            return
        rows = result.curve_table[result.curve_table["mz"] == int(mass)].sort_values(result.axis_name)
        if rows.empty:
            self.plot_widget.show_empty("所选质量通道没有数据")
            return
        source_type = self._source_type()
        xlabel = "温度" if source_type == "temperature" else "光子能量 (eV)"
        ylabel = str(self.signal_column_combo.currentData() or "信号")
        x_values = rows[result.axis_name].to_numpy(dtype=float)
        spec = ScientificPlotSpec(
            title=f"m/z {int(mass)} 同位素贡献后处理",
            xlabel=xlabel,
            ylabel=ylabel,
            series=(
                CurveSeries(
                    key="observed",
                    role=SeriesRole.MEASUREMENT,
                    x=x_values,
                    y=rows["observed"].to_numpy(dtype=float),
                    label="实测",
                ),
                CurveSeries(
                    key="isotope_contribution",
                    role=SeriesRole.COMPONENT,
                    x=x_values,
                    y=rows["isotope_contribution"].to_numpy(dtype=float),
                    label="计算的同位素贡献",
                ),
                CurveSeries(
                    key="residual",
                    role=SeriesRole.RESIDUAL,
                    x=x_values,
                    y=rows["residual"].to_numpy(dtype=float),
                    label="扣除后信号",
                ),
            ),
            show_legend=True,
            show_zero_line=True,
        )
        self.plot_widget.render_spec(spec)

    def _clear_results(self) -> None:
        self.curve_matrix = pd.DataFrame()
        self.correction_result = None
        self.result_mz_combo.clear()
        self.result_mz_combo.setEnabled(False)
        self.export_button.setEnabled(False)
        for table in (
            self.curve_table_widget,
            self.pattern_table_widget,
            self.source_table_widget,
            self.diagnostic_table_widget,
            self.sensitivity_table_widget,
        ):
            table.clearContents()
            table.setRowCount(0)
        self.result_status_label.setText("等待计算")
        self.plot_widget.show_empty("载入结果并指定分子式后开始计算")

    def export_result(self) -> None:
        result = self.correction_result
        if result is None:
            QtWidgets.QMessageBox.warning(self, "提示", "没有可导出的后处理结果")
            return
        default_name = (
            "temperature_isotope_corrected.xlsx"
            if self._source_type() == "temperature"
            else "pie_isotope_corrected.xlsx"
        )
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出同位素贡献后处理结果",
            str(ensure_output_dir("exports", "isotope_correction") / default_name),
            "Excel Files (*.xlsx);;CSV Files (*.csv)",
        )
        if not path:
            return
        output_path = Path(path)
        parameter_table = self._export_parameter_table()
        try:
            if output_path.suffix.lower() == ".csv" or "CSV" in selected_filter:
                if output_path.suffix.lower() != ".csv":
                    output_path = output_path.with_suffix(".csv")
                result.curve_table.to_csv(output_path, index=False, encoding="utf-8-sig")
                result.pattern_table.to_csv(
                    output_path.with_name(f"{output_path.stem}_patterns.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )
                result.source_table.to_csv(
                    output_path.with_name(f"{output_path.stem}_sources.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )
                result.component_table.to_csv(
                    output_path.with_name(f"{output_path.stem}_components.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )
                result.diagnostic_table.to_csv(
                    output_path.with_name(f"{output_path.stem}_diagnostics.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )
                result.sensitivity_table.to_csv(
                    output_path.with_name(f"{output_path.stem}_sensitivity.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )
                parameter_table.to_csv(
                    output_path.with_name(f"{output_path.stem}_parameters.csv"),
                    index=False,
                    encoding="utf-8-sig",
                )
            else:
                if output_path.suffix.lower() != ".xlsx":
                    output_path = output_path.with_suffix(".xlsx")
                with pd.ExcelWriter(output_path) as writer:
                    result.curve_table.to_excel(writer, sheet_name="corrected_curves", index=False)
                    result.pattern_table.to_excel(writer, sheet_name="isotope_patterns", index=False)
                    result.source_table.to_excel(writer, sheet_name="source_curves", index=False)
                    result.component_table.to_excel(writer, sheet_name="components", index=False)
                    result.diagnostic_table.to_excel(
                        writer,
                        sheet_name="numerical_diagnostics",
                        index=False,
                    )
                    result.sensitivity_table.to_excel(
                        writer,
                        sheet_name="coefficient_sensitivity",
                        index=False,
                    )
                    parameter_table.to_excel(writer, sheet_name="parameters", index=False)
            if self.project_scope_active:
                record_project_artifact(
                    self,
                    "isotope_correction_result_file",
                    output_path,
                    message="同位素贡献校正结果已登记到项目管理",
                )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "导出失败", str(exc))
            return
        QtWidgets.QMessageBox.information(self, "导出完成", f"结果已导出：\n{output_path}")

    def _export_parameter_table(self) -> pd.DataFrame:
        result = self.correction_result
        selected_energy = self.energy_filter_combo.currentData()
        rows = [
            ("数据类型", self._source_type()),
            ("输入文件", self.source_path),
            ("坐标列", self.axis_column),
            ("质量列", self.mass_column),
            ("信号列", str(self.signal_column_combo.currentData() or "")),
            ("重复数据聚合", str(self.aggregation_combo.currentData() or "")),
            ("质量范围起点", self.mz_min_spin.value()),
            ("质量范围终点", self.mz_max_spin.value()),
            ("温扫光子能量", "" if selected_energy is None else float(selected_energy)),
            ("计算模式", "" if result is None else result.mode),
            ("理论贡献矩阵秩", "" if result is None else result.pattern_rank),
            (
                "理论贡献矩阵条件数",
                "" if result is None else result.pattern_condition_number,
            ),
            (
                "科学说明",
                "" if result is None else result.scientific_note,
            ),
        ]
        return pd.DataFrame(rows, columns=["parameter", "value"])


__all__ = ["IsotopeCorrectionDialog"]
