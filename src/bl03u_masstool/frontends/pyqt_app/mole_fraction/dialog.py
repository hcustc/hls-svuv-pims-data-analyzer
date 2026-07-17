from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets

from bl03u_masstool.core.calibration import Calibration, tof_to_mz
from bl03u_masstool.core.config import (
    PeakDetectionConfig,
    load_calibration_config,
    load_peak_detection_config,
    save_calibration_config,
    save_peak_detection_config,
    resolve_species_database_path,
    species_database_path,
)
from bl03u_masstool.core.isotope import (
    calculate_isotope_distribution,
    formula_monoisotopic_mass,
    formula_nominal_mass,
    generate_formula_candidates,
    parse_element_count_ranges,
    parse_formula,
)
from bl03u_masstool.core.nist_webbook import default_nist_webbook_client
from bl03u_masstool.core.output_paths import ensure_output_dir
from bl03u_masstool.core.peak_ranges import load_peak_ranges
from bl03u_masstool.core.pie_analysis import analyze_pie_folder, build_pie_curves, identify_species_for_mz_with_curve, load_species_database, analyze_multiple_pie_folders, merge_pie_segments
from bl03u_masstool.core.pics_calculator import calc_pics_single_energy
from bl03u_masstool.core.elements import get_all_elements_from_database, filter_species_by_elements, COMMON_ELEMENTS, parse_formula as parse_formula_elements, get_elements_from_formula
from bl03u_masstool.core.normalization import NormalizationSettings, load_normalization_settings, save_normalization_settings
from bl03u_masstool.core.project_settings import ProjectSettings
from bl03u_masstool.core.temperature_scan import (
    TEMPERATURE_CURVE_CLASS_LABELS,
    analyze_temperature_folder,
    build_temperature_curves,
)
from bl03u_masstool.core.mole_fraction import (
    MASS_DISCRIMINATION_PRESETS,
    MoleFractionSettings,
    _interpolate_cross_section,
    calc_isomeric_separation,
    calc_mass_discrimination,
    calc_parent_mole_fraction,
    compute_all_mole_fractions,
    extract_signal_from_temperature_curves,
    get_expansion_coefficient_for_energy,
    load_mole_fraction_settings,
    save_mole_fraction_settings,
    select_calc_energy,
    separate_coexisting_species_signals,
)
from bl03u_masstool.frontends.pyqt_app.workers import WorkerThread
from bl03u_masstool.frontends.pyqt_app.project_artifacts import record_project_artifact

from bl03u_masstool.frontends.pyqt_app.common.widgets import DataFrameTableMixin
from bl03u_masstool.frontends.pyqt_app.common.static_plot import StaticCurvePlot

class MoleFractionDialog(QtWidgets.QWidget, DataFrameTableMixin):
    def __init__(self, calibration: Calibration, normalization_settings, parent=None):
        super().__init__(parent)
        self.calibration = calibration
        self.normalization_settings = normalization_settings
        self.settings = load_mole_fraction_settings()
        self.project_settings: ProjectSettings | None = None
        self.database: list[dict] = []
        self.mz_index: dict[int, list[int]] = {}
        self.expansion_coefficients: dict[float, float] = {}
        self.parent_mf_results: dict[float, float] = {}
        self.parent_mf_by_energy: dict[float, dict[float, float]] = {}
        self.parent_signal_by_energy: dict[float, dict[float, float]] = {}
        self.parent_config_by_energy: dict[float, dict] = {}
        self.energy_parent_config: dict[float, dict] = {}
        self.product_mf_results: dict[str, dict[float, float]] = {}
        self.isomeric_results: dict[int, dict[str, dict[float, float]]] = {}
        self.temperature_scan_data: dict[float, dict] = {}
        self.pie_species_data: list[dict] = []
        self.available_energies: list[float] = []
        self.all_species_mf: dict[tuple, dict[float, float]] = {}
        self.peak_ranges: dict[int, tuple[int, int]] = {}
        self._loaded_database_path: str | None = None
        self._project_data_restore_key: tuple | None = None
        self._restoring_project_data = False
        self._setting_parent_mz = False
        self._parent_mz_origin = "saved" if self.settings.parent_mz > 0 else "unset"
        self._parent_mz_confirmed = False
        self._init_ui()
        self._auto_load_database()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # 摘要栏
        self.summary_bar = QtWidgets.QWidget()
        summary_layout = QtWidgets.QHBoxLayout(self.summary_bar)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(6)
        self.summary_project_label = QtWidgets.QLabel("项目: ---")
        self.summary_system_label = QtWidgets.QLabel("体系: ---")
        self.summary_data_label = QtWidgets.QLabel("数据源: ---")
        self.summary_project_label.setObjectName("ReadoutValue")
        self.summary_system_label.setObjectName("ReadoutValue")
        self.summary_data_label.setObjectName("ReadoutValue")
        self.summary_project_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary_system_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.summary_data_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        summary_layout.addWidget(self.summary_project_label)
        summary_layout.addWidget(self.summary_system_label)
        summary_layout.addWidget(self.summary_data_label)
        self.summary_open_project_btn = QtWidgets.QPushButton("项目管理")
        self.summary_open_project_btn.setObjectName("WorkflowButton")
        self.summary_open_project_btn.clicked.connect(self._open_project_settings)
        summary_layout.addWidget(self.summary_open_project_btn)
        self.status_label = QtWidgets.QLabel("就绪")
        self.status_label.setObjectName("ProjectStatus")
        summary_layout.addWidget(self.status_label)
        summary_layout.addStretch()
        layout.addWidget(self.summary_bar)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._create_data_tab(), "1. 数据加载")
        self.tabs.addTab(self._create_parent_tab(), "2. 母体摩尔分数")
        self.tabs.addTab(self._create_energy_parent_tab(), "3. 低能参考物种")
        self.tabs.addTab(self._create_auto_mf_tab(), "4. 自动计算摩尔分数")
        self.tabs.addTab(self._create_results_tab(), "5. 结果汇总")
        layout.addWidget(self.tabs)

    def _auto_load_database(self):
        try:
            db_path = species_database_path()
            if db_path.exists():
                self.database, self.mz_index = load_species_database(str(db_path))
                self._loaded_database_path = str(db_path)
                if hasattr(self, "_update_parent_species_list"):
                    self._update_parent_species_list()
                if hasattr(self, "energy_parent_table"):
                    self._refresh_energy_parent_table()
        except Exception:
            pass

    @property
    def _mass_disc_exponent(self) -> float:
        """统一读取质量响应指数：优先使用项目设置，回退到 legacy 设置。"""
        if self.project_settings is not None:
            return self.project_settings.mf_mass_disc_exponent
        return self.settings.mass_disc_exponent

    def _expansion_coefficients_for_energy(self, energy: float | None) -> dict[float, float]:
        """Return the Kr expansion curve for one calculation energy."""
        factors = self.expansion_coefficients or {}
        if not factors:
            return {}
        first_value = next(iter(factors.values()), None)
        if not isinstance(first_value, dict):
            return {float(temp): float(value) for temp, value in factors.items()}

        energy_groups: list[tuple[float, dict]] = []
        for raw_energy, raw_curve in factors.items():
            if not isinstance(raw_curve, dict):
                continue
            try:
                energy_groups.append((float(raw_energy), raw_curve))
            except (TypeError, ValueError):
                continue
        if not energy_groups:
            return {}
        _, selected_curve = (
            energy_groups[0]
            if energy is None
            else min(energy_groups, key=lambda item: abs(item[0] - float(energy)))
        )
        return {float(temp): float(value) for temp, value in selected_curve.items()}

    def _expansion_coefficient(self, temperature: float, energy: float | None) -> float:
        return get_expansion_coefficient_for_energy(
            float(temperature),
            None if energy is None else float(energy),
            self.expansion_coefficients,
        )

    def set_project_settings(self, ps: ProjectSettings) -> None:
        """Apply ProjectSettings defaults to MoleFractionDialog controls."""
        self.project_settings = ps
        self._load_project_database_from_settings()
        # 卡峰范围由项目管理统一维护，在此自动加载
        self._load_peak_ranges_from_project()
        if hasattr(self, "spin_parent_mz"):
            self._setting_parent_mz = True
            self.spin_parent_mz.setValue(ps.mf_parent_mz)
            self._setting_parent_mz = False
            self._parent_mz_origin = "project" if ps.mf_parent_mz > 0 else "unset"
            self._parent_mz_confirmed = False
        if hasattr(self, "spin_parent_mf0"):
            self.spin_parent_mf0.setValue(ps.mf_parent_initial_mf)
        if hasattr(self, "spin_parent_energy"):
            self.spin_parent_energy.setValue(ps.mf_photon_energy)
        if hasattr(self, "spin_parent_t0") and ps.mf_reference_temperature is not None:
            self.spin_parent_t0.setValue(int(ps.mf_reference_temperature))
        # 膨胀系数从项目全局设置同步（通用参数中计算的）
        if ps.expansion_factors:
            self.expansion_coefficients = dict(ps.expansion_factors)
        # 更新摘要栏
        if hasattr(self, "summary_project_label"):
            project_name = ps.project_name or "---"
            system = ps.system or "---"
            self.summary_project_label.setText(f"项目: {project_name}")
            self.summary_system_label.setText(f"体系: {system}")
            artifacts = []
            if ps.temperature_scan_result_file:
                artifacts.append("温度结果")
            if ps.pie_identification_result_file:
                artifacts.append("PIE结果")
            if ps.mole_fraction_result_file:
                artifacts.append("摩尔分数结果")
            data_flow = ", ".join(artifacts) if artifacts else (ps.temperature_scan_folder or ps.pie_scan_folder or "---")
            self.summary_data_label.setText(f"数据流: {data_flow}")
        if hasattr(self, "btn_load_project_ts_folder"):
            has_ts_folder = bool(ps.temperature_scan_folder)
            self.btn_load_project_ts_folder.setEnabled(has_ts_folder)
        if hasattr(self, "btn_load_project_ts_result"):
            has_ts_result = bool(ps.temperature_scan_result_file)
            self.btn_load_project_ts_result.setEnabled(has_ts_result)
            self.lbl_ts_project_artifact.setText(
                Path(ps.temperature_scan_result_file).name if has_ts_result else "项目未登记温度结果"
            )
        if hasattr(self, "btn_load_project_pie_result"):
            has_pie_result = bool(ps.pie_identification_result_file)
            self.btn_load_project_pie_result.setEnabled(has_pie_result)
            self.lbl_pie_project_artifact.setText(
                Path(ps.pie_identification_result_file).name if has_pie_result else "项目未登记PIE结果"
            )
        self._restore_project_managed_data(ps)
        self._refresh_parent_selection_state()

    def _path_restore_signature(self, value: str | Path | None) -> tuple:
        path_text = str(value or "").strip()
        if not path_text:
            return ("", False, None, None, None)
        path = Path(path_text)
        try:
            stat = path.stat()
        except OSError:
            return (path_text, False, None, None, None)
        kind = "dir" if path.is_dir() else "file" if path.is_file() else "other"
        size = stat.st_size if path.is_file() else None
        return (path_text, True, kind, stat.st_mtime_ns, size)

    def _project_data_restore_signature(self, ps: ProjectSettings) -> tuple:
        return (
            self._path_restore_signature(ps.temperature_scan_result_file),
            self._path_restore_signature(ps.temperature_scan_folder),
            self._path_restore_signature(ps.pie_identification_result_file),
            self._path_restore_signature(ps.manual_peak_file),
        )

    def _restore_project_managed_data(self, ps: ProjectSettings) -> None:
        """Restore project-owned calculation inputs without prompting the user."""
        if self._restoring_project_data:
            return
        restore_key = self._project_data_restore_signature(ps)
        if restore_key == self._project_data_restore_key:
            return

        restored: list[str] = []
        self._restoring_project_data = True
        try:
            ts_message = self._restore_project_temperature_data(ps)
            if ts_message:
                restored.append(ts_message)
            pie_message = self._restore_project_pie_data(ps)
            if pie_message:
                restored.append(pie_message)
            if self._clear_stale_legacy_parent_mz():
                restored.append("母体 m/z 待设置")
        finally:
            self._project_data_restore_key = restore_key
            self._restoring_project_data = False

        if restored and hasattr(self, "status_label"):
            self.status_label.setText("已从项目自动恢复: " + "；".join(restored))
        self._refresh_parent_selection_state()

    def _loaded_data_has_mz(self, mz: int) -> bool:
        if mz <= 0:
            return False
        for species in self.pie_species_data:
            try:
                species_mz = int(species.get("mz"))
            except (TypeError, ValueError):
                species_mz = -1
            if species_mz == mz:
                return True
        for energy_data in self.temperature_scan_data.values():
            for info in energy_data.values():
                precomputed = info.get("precomputed_signals", {}) or {}
                if mz in precomputed:
                    return True
                for peak in info.get("peaks_info", []) or []:
                    try:
                        peak_mz = int(peak.get("mz_rounded"))
                    except (TypeError, ValueError):
                        peak_mz = -1
                    if peak_mz == mz:
                        return True
        return False

    def _clear_stale_legacy_parent_mz(self) -> bool:
        if not hasattr(self, "spin_parent_mz"):
            return False
        current_mz = int(self.spin_parent_mz.value())
        if current_mz != 128:
            return False
        if self._loaded_data_has_mz(current_mz):
            return False
        if not self.temperature_scan_data and not self.pie_species_data:
            return False
        self._setting_parent_mz = True
        self.spin_parent_mz.setValue(0)
        self._setting_parent_mz = False
        self._parent_mz_origin = "unset"
        self._parent_mz_confirmed = False
        return True

    def _restore_project_temperature_data(self, ps: ProjectSettings) -> str | None:
        result_path_text = str(ps.temperature_scan_result_file or "").strip()
        if result_path_text:
            result_path = Path(result_path_text)
            if result_path.exists() and result_path.is_file():
                self._load_temperature_result_file(result_path, show_message=False, register_artifact=False)
                if self.available_energies:
                    return f"温度扫描结果 {len(self.available_energies)} 个能量"
            return None

        folder_text = str(ps.temperature_scan_folder or "").strip()
        if not folder_text:
            return None
        root = Path(folder_text)
        if not root.exists() or not root.is_dir():
            return None
        folders = self._discover_energy_folders(root)
        if not folders:
            return None
        # Raw-folder analysis is intentionally user-triggered. Parsing every
        # spectrum while opening a project blocks the GUI and duplicates the
        # Temperature page cache workflow.
        if hasattr(self, "lbl_ts_folder"):
            self.lbl_ts_folder.setText(f"项目原始目录已就绪：{root.name}（点击“项目原始目录”加载）")
        return None

    def _restore_project_pie_data(self, ps: ProjectSettings) -> str | None:
        file_text = str(ps.pie_identification_result_file or "").strip()
        if not file_text:
            return None
        file_path = Path(file_text)
        if not file_path.exists() or not file_path.is_file():
            return None
        self._load_pie_results(file_path, show_message=False, register_artifact=False)
        if self.pie_species_data:
            return f"PIE结果 {len(self.pie_species_data)} 条"
        return None

    def _load_project_database_from_settings(self, *, force: bool = False) -> None:
        """Rebuild the database from defaults plus the current project override."""
        ps = self.project_settings
        db_value = getattr(ps, "pics_database_path", "") if ps is not None else ""
        db_path = resolve_species_database_path(db_value)
        resolved = str(db_path)
        if not force and self._loaded_database_path == resolved:
            return
        try:
            default_path = species_database_path()
            default_database, _ = load_species_database(str(default_path))
            self.database = []
            self.mz_index = {}
            self._merge_species_database(default_database)
            if db_path != default_path:
                project_database, _ = load_species_database(resolved)
                self._merge_species_database(project_database)
            self._loaded_database_path = resolved
            if hasattr(self, "_update_parent_species_list"):
                self._update_parent_species_list()
            if hasattr(self, "energy_parent_table"):
                self._refresh_energy_parent_table()
            if hasattr(self, "status_label"):
                scope = "项目" if db_path != default_path else "默认"
                self.status_label.setText(f"已加载{scope}PICS数据库: {db_path.name}")
        except Exception as exc:
            if hasattr(self, "status_label"):
                self.status_label.setText(f"PICS数据库加载失败: {exc}")

    def refresh_database(self) -> None:
        self._load_project_database_from_settings(force=True)

    def _merge_species_database(self, records: list[dict]) -> None:
        """Merge project PICS records over the bundled database without losing fallbacks."""
        merged: list[dict] = []
        positions: dict[tuple[int, str], int] = {}
        for record in list(self.database) + list(records):
            try:
                mz = int(record.get("mz"))
            except Exception:
                continue
            name = str(record.get("species") or record.get("name") or "").strip()
            if not name:
                continue
            key = (mz, name)
            if key in positions:
                merged[positions[key]] = record
            else:
                positions[key] = len(merged)
                merged.append(record)
        mz_index: dict[int, list[int]] = {}
        for idx, record in enumerate(merged):
            try:
                mz = int(record.get("mz"))
            except Exception:
                continue
            mz_index.setdefault(mz, []).append(idx)
        self.database = merged
        self.mz_index = mz_index

    def _open_project_settings(self):
        """跳转到项目管理页面。"""
        win = self.window()
        if hasattr(win, "switch_workspace_page"):
            win.switch_workspace_page("project")

    def set_busy(self, busy: bool, message: str) -> None:
        """Disable UI during long-running computation."""
        self.status_label.setText(message)
        self.tabs.setDisabled(busy)

    def _create_data_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # 上下排布数据加载卡片
        cards_container = QtWidgets.QWidget()
        cards_layout = QtWidgets.QVBoxLayout(cards_container)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(8)

        # ========== 温度扫描卡片 ==========
        ts_card = QtWidgets.QGroupBox("温度扫描数据")
        ts_card_layout = QtWidgets.QVBoxLayout(ts_card)
        ts_card_layout.setContentsMargins(10, 6, 10, 6)
        ts_card_layout.setSpacing(6)

        # 单行：数据载入 + 能量选择 + 状态
        ts_row = QtWidgets.QHBoxLayout()
        ts_row.setSpacing(6)
        self.btn_load_project_ts_folder = QtWidgets.QPushButton("项目原始目录")
        self.btn_load_project_ts_folder.setObjectName("PrimaryButton")
        self.btn_load_project_ts_folder.setToolTip("从项目管理配置的温度扫描根目录自动载入各能量子文件夹")
        self.btn_load_project_ts_folder.clicked.connect(self._load_project_temperature_scan_folder)
        self.btn_load_project_ts_folder.setEnabled(False)
        ts_row.addWidget(self.btn_load_project_ts_folder)
        self.btn_load_project_ts_result = QtWidgets.QPushButton("项目结果")
        self.btn_load_project_ts_result.setObjectName("BrowseButton")
        self.btn_load_project_ts_result.setToolTip("读取项目管理中登记的温度扫描结果文件")
        self.btn_load_project_ts_result.clicked.connect(self._load_project_temperature_result)
        self.btn_load_project_ts_result.setEnabled(False)
        ts_row.addWidget(self.btn_load_project_ts_result)
        btn_load_ts_result = QtWidgets.QPushButton("选择结果文件")
        btn_load_ts_result.setObjectName("BrowseButton")
        btn_load_ts_result.setToolTip("选择温度扫描导出的xlsx/csv结果文件")
        btn_load_ts_result.clicked.connect(self._load_temperature_result_file)
        ts_row.addWidget(btn_load_ts_result)
        btn_add_folder = QtWidgets.QPushButton("添加能量文件夹")
        btn_add_folder.setObjectName("BrowseButton")
        btn_add_folder.setToolTip("备用入口：添加包含温度扫描txt文件的能量文件夹并重新分析")
        btn_add_folder.clicked.connect(self._add_energy_folder)
        ts_row.addWidget(btn_add_folder)
        btn_clear = QtWidgets.QPushButton("清空")
        btn_clear.setObjectName("WarningButton")
        btn_clear.setToolTip("清空所有已加载的温度扫描数据")
        btn_clear.clicked.connect(self._clear_temperature_scan)
        ts_row.addWidget(btn_clear)
        ts_row.addSpacing(12)
        ts_row.addWidget(QtWidgets.QLabel("能量:"))
        self.combo_energy_select = QtWidgets.QComboBox()
        self.combo_energy_select.addItem("全部能量")
        self.combo_energy_select.setMinimumWidth(120)
        ts_row.addWidget(self.combo_energy_select)
        self.lbl_energy_count = QtWidgets.QLabel("")
        self.lbl_energy_count.setObjectName("InlineStatusLabel")
        ts_row.addWidget(self.lbl_energy_count)
        self.btn_remove_energy = QtWidgets.QPushButton("移除选中能量")
        self.btn_remove_energy.clicked.connect(self._remove_selected_energy)
        self.btn_remove_energy.setEnabled(False)
        self.combo_energy_select.currentTextChanged.connect(self._on_energy_select)
        ts_row.addWidget(self.btn_remove_energy)
        ts_row.addSpacing(12)
        self.lbl_ts_folder = QtWidgets.QLabel("未选择文件夹")
        self.lbl_ts_folder.setObjectName("ProjectHint")
        ts_row.addWidget(self.lbl_ts_folder, 1)
        self.lbl_ts_project_artifact = QtWidgets.QLabel("项目未登记温度结果")
        self.lbl_ts_project_artifact.setObjectName("ProjectHint")
        ts_row.addWidget(self.lbl_ts_project_artifact)
        ts_card_layout.addLayout(ts_row)

        # 可展开数据表格（默认隐藏）
        self.ts_table_group = QtWidgets.QGroupBox("数据详情")
        self.ts_table_group.setCheckable(True)
        self.ts_table_group.setChecked(False)
        ts_table_layout = QtWidgets.QVBoxLayout(self.ts_table_group)
        ts_table_layout.setContentsMargins(4, 4, 4, 4)
        self.ts_data_table = QtWidgets.QTableWidget()
        self.ts_data_table.setColumnCount(6)
        self.ts_data_table.setHorizontalHeaderLabels(["能量(eV)", "温度(°C)", "文件名", "IO(nA)", "重复次数", "检测到的质量数"])
        self.ts_data_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.ts_data_table.setMaximumHeight(200)
        ts_table_layout.addWidget(self.ts_data_table)
        ts_card_layout.addWidget(self.ts_table_group)

        cards_layout.addWidget(ts_card)

        # ========== PIE鉴定结果卡片 ==========
        pie_card = QtWidgets.QGroupBox("PIE鉴定结果")
        pie_card_layout = QtWidgets.QVBoxLayout(pie_card)
        pie_card_layout.setContentsMargins(10, 6, 10, 6)
        pie_card_layout.setSpacing(6)

        pie_row = QtWidgets.QHBoxLayout()
        pie_row.setSpacing(6)
        self.btn_load_project_pie_result = QtWidgets.QPushButton("从项目载入")
        self.btn_load_project_pie_result.setObjectName("PrimaryButton")
        self.btn_load_project_pie_result.setToolTip("读取项目管理中登记的PIE鉴定结果文件")
        self.btn_load_project_pie_result.clicked.connect(self._load_project_pie_results)
        self.btn_load_project_pie_result.setEnabled(False)
        pie_row.addWidget(self.btn_load_project_pie_result)
        btn_load_pie = QtWidgets.QPushButton("加载PIE鉴定结果")
        btn_load_pie.setObjectName("BrowseButton")
        btn_load_pie.clicked.connect(self._load_pie_results)
        pie_row.addWidget(btn_load_pie)
        pie_row.addSpacing(12)
        self.lbl_pie_status = QtWidgets.QLabel("未加载")
        self.lbl_pie_status.setObjectName("ProjectHint")
        pie_row.addWidget(self.lbl_pie_status, 1)
        self.lbl_pie_project_artifact = QtWidgets.QLabel("项目未登记PIE结果")
        self.lbl_pie_project_artifact.setObjectName("ProjectHint")
        pie_row.addWidget(self.lbl_pie_project_artifact)
        pie_card_layout.addLayout(pie_row)

        # 可展开物种表格（默认隐藏）
        self.pie_table_group = QtWidgets.QGroupBox("物种列表")
        self.pie_table_group.setCheckable(True)
        self.pie_table_group.setChecked(False)
        pie_table_layout = QtWidgets.QVBoxLayout(self.pie_table_group)
        pie_table_layout.setContentsMargins(4, 4, 4, 4)
        self.pie_species_table = QtWidgets.QTableWidget()
        self.pie_species_table.setColumnCount(5)
        self.pie_species_table.setHorizontalHeaderLabels(["质量数", "物种名称", "电离能(eV)", "贡献比例(%)", "R²"])
        self.pie_species_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.pie_species_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.pie_species_table.setMaximumHeight(200)
        pie_table_layout.addWidget(self.pie_species_table)
        pie_card_layout.addWidget(self.pie_table_group)

        cards_layout.addWidget(pie_card)
        layout.addWidget(cards_container)
        layout.addStretch()
        return widget

    def _create_energy_parent_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        top_layout = QtWidgets.QHBoxLayout()
        btn_refresh = QtWidgets.QPushButton("刷新可用能量")
        btn_refresh.setObjectName("BrowseButton")
        btn_refresh.setToolTip("根据已加载温度扫描数据刷新可设置替代参考物种的能量行")
        btn_refresh.clicked.connect(self._refresh_energy_parent_table)
        top_layout.addWidget(btn_refresh)

        btn_reset = QtWidgets.QPushButton("清除替代参考")
        btn_reset.setObjectName("WarningButton")
        btn_reset.clicked.connect(self._reset_energy_parent_config)
        top_layout.addWidget(btn_reset)

        btn_apply = QtWidgets.QPushButton("应用替代参考")
        btn_apply.setObjectName("PrimaryButton")
        btn_apply.clicked.connect(self._apply_energy_parent_config)
        top_layout.addWidget(btn_apply)

        self.lbl_energy_parent_status = QtWidgets.QLabel("未加载温度扫描能量")
        self.lbl_energy_parent_status.setObjectName("InlineStatusLabel")
        top_layout.addWidget(self.lbl_energy_parent_status, 1)
        layout.addLayout(top_layout)

        self.energy_parent_table = QtWidgets.QTableWidget()
        self.energy_parent_table.setColumnCount(4)
        self.energy_parent_table.setHorizontalHeaderLabels(["计算能量(eV)", "参考物种 m/z", "参考物种", "状态"])
        self.energy_parent_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.energy_parent_table, 1)

        return widget

    def _create_parent_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        cfg_group = QtWidgets.QGroupBox("母体参数设置")
        cfg_layout = QtWidgets.QGridLayout(cfg_group)
        cfg_layout.setHorizontalSpacing(10)
        cfg_layout.setVerticalSpacing(8)

        self.parent_selection_banner = QtWidgets.QFrame(cfg_group)
        self.parent_selection_banner.setObjectName("ParentSelectionBanner")
        banner_layout = QtWidgets.QVBoxLayout(self.parent_selection_banner)
        banner_layout.setContentsMargins(10, 7, 10, 7)
        banner_layout.setSpacing(2)
        banner_title = QtWidgets.QLabel("先确认母体分子离子", self.parent_selection_banner)
        banner_title.setObjectName("ProjectBoundaryTitle")
        banner_text = QtWidgets.QLabel(
            "母体 m/z 应对应反应物的分子离子峰。项目中的数值只是上次保存的预设，不是软件通用默认值。",
            self.parent_selection_banner,
        )
        banner_text.setObjectName("ProjectBoundaryText")
        banner_text.setWordWrap(True)
        banner_layout.addWidget(banner_title)
        banner_layout.addWidget(banner_text)
        cfg_layout.addWidget(self.parent_selection_banner, 0, 0, 1, 6)

        cfg_layout.addWidget(QtWidgets.QLabel("母体分子离子 m/z:"), 1, 0)
        self.spin_parent_mz = QtWidgets.QSpinBox()
        self.spin_parent_mz.setToolTip("母体反应物的分子离子质量数；0 表示尚未设置")
        self.spin_parent_mz.setRange(0, 500)
        self.spin_parent_mz.setSpecialValueText("未设置")
        self.spin_parent_mz.setValue(self.settings.parent_mz)
        self.spin_parent_mz.valueChanged.connect(self._on_parent_mz_changed)
        cfg_layout.addWidget(self.spin_parent_mz, 1, 1)

        self.btn_select_parent_mz = QtWidgets.QPushButton("从数据选择")
        self.btn_select_parent_mz.setObjectName("BrowseButton")
        self.btn_select_parent_mz.setToolTip("列出温度扫描或 PIE 结果中实际出现的 m/z")
        self.btn_select_parent_mz.clicked.connect(self._select_parent_mz_from_data)
        cfg_layout.addWidget(self.btn_select_parent_mz, 1, 2)

        self.btn_confirm_parent_mz = QtWidgets.QPushButton("确认母体")
        self.btn_confirm_parent_mz.setObjectName("PrimaryButton")
        self.btn_confirm_parent_mz.clicked.connect(self._confirm_parent_mz)
        cfg_layout.addWidget(self.btn_confirm_parent_mz, 1, 3)

        self.lbl_parent_mz_status = QtWidgets.QLabel("")
        self.lbl_parent_mz_status.setObjectName("ParentSelectionStatus")
        self.lbl_parent_mz_status.setWordWrap(True)
        cfg_layout.addWidget(self.lbl_parent_mz_status, 1, 4, 1, 2)

        cfg_layout.addWidget(QtWidgets.QLabel("母体物种:"), 2, 0)
        self.combo_parent_species = QtWidgets.QComboBox()
        self.combo_parent_species.setToolTip("在当前母体质量数下选择具体参考母体物种")
        self.combo_parent_species.currentIndexChanged.connect(self._on_parent_species_changed)
        cfg_layout.addWidget(self.combo_parent_species, 2, 1, 1, 2)
        self.lbl_parent_info = QtWidgets.QLabel("")
        self.lbl_parent_info.setObjectName("ProjectHint")
        cfg_layout.addWidget(self.lbl_parent_info, 2, 3, 1, 3)

        cfg_layout.addWidget(QtWidgets.QLabel("参考温度 T₀ (°C):"), 3, 0)
        self.spin_parent_t0 = QtWidgets.QSpinBox()
        self.spin_parent_t0.setToolTip("选定一个参考温度点，用于计算母体摩尔分数的基准")
        self.spin_parent_t0.setRange(0, 2000)
        self.spin_parent_t0.setValue(int(self.settings.reference_temperature or 550))
        cfg_layout.addWidget(self.spin_parent_t0, 3, 1)
        cfg_layout.addWidget(QtWidgets.QLabel("初始摩尔分数 X(T₀):"), 3, 2)
        self.spin_parent_mf0 = QtWidgets.QDoubleSpinBox()
        self.spin_parent_mf0.setToolTip("母体物种在参考温度T₀处的摩尔分数（已知或假设值）")
        self.spin_parent_mf0.setRange(0.0, 1.0)
        self.spin_parent_mf0.setDecimals(6)
        self.spin_parent_mf0.setValue(self.settings.parent_initial_mf)
        self.spin_parent_mf0.setSingleStep(0.0001)
        cfg_layout.addWidget(self.spin_parent_mf0, 3, 3)
        cfg_layout.addWidget(QtWidgets.QLabel("光子能量 E (eV):"), 3, 4)
        self.spin_parent_energy = QtWidgets.QDoubleSpinBox()
        self.spin_parent_energy.setToolTip("实验使用的VUV光子能量 (eV)，用于查找物种在此能量下的光电离截面")
        self.spin_parent_energy.setRange(0.0, 30.0)
        self.spin_parent_energy.setDecimals(2)
        self.spin_parent_energy.setValue(self.settings.photon_energy)
        self.spin_parent_energy.setSingleStep(0.5)
        cfg_layout.addWidget(self.spin_parent_energy, 3, 5)
        for column in (1, 3, 5):
            cfg_layout.setColumnStretch(column, 1)
        layout.addWidget(cfg_group)

        action_row = QtWidgets.QHBoxLayout()
        self.lbl_parent_calc_hint = QtWidgets.QLabel("请先确认母体 m/z")
        self.lbl_parent_calc_hint.setObjectName("ProjectHint")
        action_row.addWidget(self.lbl_parent_calc_hint, 1)
        self.btn_calc_parent = QtWidgets.QPushButton("计算母体摩尔分数")
        self.btn_calc_parent.setObjectName("PrimaryButton")
        self.btn_calc_parent.clicked.connect(self._calc_parent_mole_fraction)
        action_row.addWidget(self.btn_calc_parent)
        layout.addLayout(action_row)

        self.parent_result_table = QtWidgets.QTableWidget()
        self.parent_result_table.setColumnCount(4)
        self.parent_result_table.setHorizontalHeaderLabels(["温度(°C)", "信号 S(T,E)", "λ(T)", "摩尔分数 X(T)"])
        self.parent_result_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.parent_result_table, 1)

        self._update_parent_species_list()
        self._refresh_parent_selection_state()
        return widget

    def _select_parent_mz_from_data(self) -> None:
        values = self._detected_parent_mz_values()
        if not values:
            QtWidgets.QMessageBox.information(self, "尚无候选", "请先在“数据加载”中载入温度扫描或 PIE 结果。")
            return
        labels = [str(value) for value in values]
        current = int(self.spin_parent_mz.value())
        initial = values.index(current) if current in values else 0
        selected, accepted = QtWidgets.QInputDialog.getItem(
            self,
            "选择母体分子离子",
            "数据中出现的 m/z：",
            labels,
            initial,
            False,
        )
        if not accepted or not selected:
            return
        self.spin_parent_mz.setValue(int(selected))

    def _detected_parent_mz_values(self) -> list[int]:
        values: set[int] = set()
        for species in self.pie_species_data:
            try:
                mz = int(species.get("mz"))
            except (TypeError, ValueError):
                continue
            if mz > 0:
                values.add(mz)
        for energy_data in self.temperature_scan_data.values():
            for info in energy_data.values():
                for mz in (info.get("precomputed_signals", {}) or {}).keys():
                    try:
                        value = int(mz)
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        values.add(value)
                for peak in info.get("peaks_info", []) or []:
                    try:
                        value = int(peak.get("mz_rounded"))
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        values.add(value)
        return sorted(values)

    def _confirm_parent_mz(self) -> None:
        mz = int(self.spin_parent_mz.value())
        if mz <= 0:
            QtWidgets.QMessageBox.warning(self, "尚未设置", "请先输入母体分子离子 m/z，或从已加载数据中选择。")
            return
        has_loaded_data = bool(self.temperature_scan_data or self.pie_species_data)
        if has_loaded_data and not self._loaded_data_has_mz(mz):
            QtWidgets.QMessageBox.warning(self, "数据中未找到", f"已加载数据中没有检测到 m/z {mz}，请检查母体设置。")
            return
        self._parent_mz_confirmed = True
        self._refresh_parent_selection_state()

    def _refresh_parent_selection_state(self) -> None:
        if not hasattr(self, "spin_parent_mz") or not hasattr(self, "lbl_parent_mz_status"):
            return
        mz = int(self.spin_parent_mz.value())
        has_loaded_data = bool(self.temperature_scan_data or self.pie_species_data)
        exists_in_data = self._loaded_data_has_mz(mz) if has_loaded_data and mz > 0 else False
        origin_label = {
            "project": "项目预设",
            "manual": "本页设置",
            "saved": "历史设置",
            "unset": "未设置",
        }.get(self._parent_mz_origin, "本页设置")

        if mz <= 0:
            state = "pending"
            text = "未设置：请从数据选择或手动输入"
        elif has_loaded_data and not exists_in_data:
            state = "warning"
            text = f"{origin_label} m/z {mz}：当前数据中未找到"
            self._parent_mz_confirmed = False
        elif not has_loaded_data:
            state = "pending"
            text = f"{origin_label} m/z {mz}：加载数据后验证"
        elif self._parent_mz_confirmed:
            state = "complete"
            text = f"已确认 m/z {mz}，数据中存在"
        else:
            state = "active"
            text = f"{origin_label} m/z {mz}，数据中存在，请确认"

        self.lbl_parent_mz_status.setText(text)
        self.lbl_parent_mz_status.setProperty("selectionState", state)
        self.lbl_parent_mz_status.style().unpolish(self.lbl_parent_mz_status)
        self.lbl_parent_mz_status.style().polish(self.lbl_parent_mz_status)
        self.btn_select_parent_mz.setEnabled(bool(self._detected_parent_mz_values()))
        self.btn_confirm_parent_mz.setEnabled(mz > 0 and (not has_loaded_data or exists_in_data))
        can_calculate = mz > 0 and self._parent_mz_confirmed and has_loaded_data and exists_in_data
        self.btn_calc_parent.setEnabled(can_calculate)
        if can_calculate:
            calc_hint = "母体已确认，可按当前参考温度和初始摩尔分数计算"
        elif has_loaded_data and exists_in_data:
            calc_hint = "数据已加载，请确认母体分子离子 m/z"
        elif has_loaded_data:
            calc_hint = "当前母体 m/z 不在已加载数据中，请重新选择"
        else:
            calc_hint = "请先加载数据，再选择母体分子离子 m/z"
        self.lbl_parent_calc_hint.setText(calc_hint)

    def _available_parent_mz_values(self) -> list[int]:
        mz_values: set[int] = set()
        if hasattr(self, "spin_parent_mz"):
            current_mz = int(self.spin_parent_mz.value())
            if current_mz > 0:
                mz_values.add(current_mz)

        for energy_data in self.temperature_scan_data.values():
            for info in energy_data.values():
                for peak in info.get("peaks_info", []) or []:
                    mz = peak.get("mz_rounded")
                    if mz:
                        mz_values.add(int(mz))

        for species in self.pie_species_data:
            mz = species.get("mz")
            if mz:
                mz_values.add(int(mz))

        for config in self.energy_parent_config.values():
            mz = config.get("mz")
            if mz:
                mz_values.add(int(mz))

        return sorted(mz_values)

    def _species_options_for_mz(self, mz: int) -> list[tuple[str, float | None]]:
        seen: set[str] = set()
        options: list[tuple[str, float | None]] = []

        for species in self.pie_species_data:
            if species.get("mz") != mz:
                continue
            name = species.get("species") or species.get("name") or ""
            if not name or name in seen:
                continue
            seen.add(name)
            options.append((name, species.get("ie")))

        for species in self.database:
            if species.get("mz") != mz:
                continue
            name = species.get("species") or species.get("name") or ""
            if not name or name in seen:
                continue
            seen.add(name)
            options.append((name, species.get("ie")))

        options.sort(key=lambda item: (item[1] if item[1] is not None else float("inf"), item[0]))
        return options

    def _populate_parent_species_combo(self, combo: QtWidgets.QComboBox, mz: int, include_auto: bool = True) -> None:
        combo.blockSignals(True)
        combo.clear()
        if include_auto:
            combo.addItem("自动检测", None)
        for species_name, ie in self._species_options_for_mz(mz):
            ie_text = f" (IE={ie:.3f} eV)" if ie is not None else ""
            combo.addItem(f"{species_name}{ie_text}", species_name)
        combo.blockSignals(False)

    def _set_combo_current_data(self, combo: QtWidgets.QComboBox, value) -> None:
        for idx in range(combo.count()):
            if combo.itemData(idx) == value:
                combo.setCurrentIndex(idx)
                return
        if value is None:
            combo.setCurrentIndex(0)

    def _selected_parent_species_name(self) -> str | None:
        if not hasattr(self, "combo_parent_species"):
            return None
        value = self.combo_parent_species.currentData()
        return str(value) if value else None

    def _parent_result_label(self, mz: int, species_name: str | None, energy: float | None = None) -> str:
        base_name = species_name or "母体"
        if energy is None:
            return f"{base_name}(母体,m/z={mz})" if species_name else f"母体(m/z={mz})"
        return f"{base_name}(母体,m/z={mz},E={energy:.2f}eV)"

    def _on_parent_mz_changed(self):
        self.settings.parent_mz = self.spin_parent_mz.value()
        if not self._setting_parent_mz:
            self._parent_mz_origin = "manual" if self.spin_parent_mz.value() > 0 else "unset"
            self._parent_mz_confirmed = False
        self._update_parent_species_list()
        self._refresh_energy_parent_table()
        self._refresh_parent_selection_state()

    def _on_parent_species_changed(self):
        selected = self._selected_parent_species_name()
        mz = self.spin_parent_mz.value()
        if mz <= 0:
            self.lbl_parent_info.setText("请在项目管理或本页设置母体 m/z")
            self._refresh_energy_parent_table()
            return
        options = self._species_options_for_mz(mz)
        if selected:
            ie = next((item_ie for name, item_ie in options if name == selected), None)
            ie_text = f"，IE={ie:.3f} eV" if ie is not None else ""
            self.lbl_parent_info.setText(f"当前母体: {selected}{ie_text}")
        elif options:
            self.lbl_parent_info.setText(f"自动检测：m/z {mz} 下有 {len(options)} 个候选物种")
        else:
            self.lbl_parent_info.setText("当前 m/z 未匹配到候选物种")
        self._refresh_energy_parent_table()

    def _update_parent_species_list(self):
        if not hasattr(self, "combo_parent_species"):
            return
        previous = self._selected_parent_species_name()
        mz = self.spin_parent_mz.value()
        self._populate_parent_species_combo(self.combo_parent_species, mz, include_auto=True)
        self._set_combo_current_data(self.combo_parent_species, previous)
        self._on_parent_species_changed()
        self._refresh_parent_selection_state()

    def _refresh_energy_parent_table(self):
        if not hasattr(self, "energy_parent_table"):
            return

        self.energy_parent_table.setRowCount(0)
        if not self.available_energies:
            self.lbl_energy_parent_status.setText("未加载温度扫描能量")
            return

        mz_values = self._available_parent_mz_values()
        default_mz = self.spin_parent_mz.value() if hasattr(self, "spin_parent_mz") else self.settings.parent_mz
        default_species = self._selected_parent_species_name()

        for energy in sorted(self.available_energies):
            row = self.energy_parent_table.rowCount()
            self.energy_parent_table.insertRow(row)

            energy_item = QtWidgets.QTableWidgetItem(f"{energy:.2f}")
            energy_item.setFlags(energy_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.energy_parent_table.setItem(row, 0, energy_item)

            cfg = self.energy_parent_config.get(energy, {})
            mz = int(cfg.get("mz") or default_mz)

            mz_combo = QtWidgets.QComboBox()
            for value in mz_values:
                mz_combo.addItem(str(value), value)
            self._set_combo_current_data(mz_combo, mz)
            self.energy_parent_table.setCellWidget(row, 1, mz_combo)

            species_combo = QtWidgets.QComboBox()
            self._populate_parent_species_combo(species_combo, mz, include_auto=True)
            self._set_combo_current_data(species_combo, cfg.get("species_name", default_species))
            self.energy_parent_table.setCellWidget(row, 2, species_combo)

            status_text = "替代参考" if cfg else "继承主母体"
            status_item = QtWidgets.QTableWidgetItem(status_text)
            status_item.setFlags(status_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.energy_parent_table.setItem(row, 3, status_item)

            mz_combo.currentIndexChanged.connect(
                lambda _, r=row, e=energy: self._on_mz_changed_for_parent_config(r, e)
            )
            species_combo.currentIndexChanged.connect(
                lambda _, e=energy, mz_cb=mz_combo, species_cb=species_combo: self._on_species_changed_for_parent_config(
                    e, mz_cb, species_cb
                )
            )

        self.lbl_energy_parent_status.setText(
            f"已加载 {len(self.available_energies)} 个能量行，"
            f"{len(self.energy_parent_config)} 个替代参考"
        )

    def _on_mz_changed_for_parent_config(self, row: int, energy: float):
        mz_combo = self.energy_parent_table.cellWidget(row, 1)
        species_combo = self.energy_parent_table.cellWidget(row, 2)
        if mz_combo is None or species_combo is None:
            return
        mz = mz_combo.currentData()
        if mz is None:
            return
        previous_species = species_combo.currentData()
        self._populate_parent_species_combo(species_combo, int(mz), include_auto=True)
        self._set_combo_current_data(species_combo, previous_species)
        self._on_species_changed_for_parent_config(energy, mz_combo, species_combo)

    def _on_species_changed_for_parent_config(
        self,
        energy: float,
        mz_combo: QtWidgets.QComboBox,
        species_combo: QtWidgets.QComboBox,
    ):
        mz = mz_combo.currentData()
        if mz is None:
            return
        species_name = species_combo.currentData()
        cfg = {
            "mz": int(mz),
            "species_name": str(species_name) if species_name else None,
        }
        if self._is_default_reference_config(cfg):
            self.energy_parent_config.pop(energy, None)
        else:
            self.energy_parent_config[energy] = cfg
        self.parent_mf_by_energy = {}
        self.parent_signal_by_energy = {}
        self.parent_config_by_energy = {}

    def _reset_energy_parent_config(self):
        self.energy_parent_config = {}
        self.parent_mf_by_energy = {}
        self.parent_signal_by_energy = {}
        self.parent_config_by_energy = {}
        self._refresh_energy_parent_table()
        self.lbl_energy_parent_status.setText("已清除替代参考，自动计算将继承主母体摩尔分数")

    def _apply_energy_parent_config(self):
        new_config: dict[float, dict] = {}
        if hasattr(self, "energy_parent_table"):
            for row in range(self.energy_parent_table.rowCount()):
                energy_item = self.energy_parent_table.item(row, 0)
                mz_combo = self.energy_parent_table.cellWidget(row, 1)
                species_combo = self.energy_parent_table.cellWidget(row, 2)
                if not energy_item or not mz_combo or not species_combo:
                    continue
                energy = float(energy_item.text())
                mz = mz_combo.currentData()
                if mz is None:
                    continue
                species_name = species_combo.currentData()
                cfg = {
                    "mz": int(mz),
                    "species_name": str(species_name) if species_name else None,
                }
                if not self._is_default_reference_config(cfg):
                    new_config[energy] = cfg

        self.energy_parent_config = new_config
        config_count = len(self.energy_parent_config)
        if self.parent_mf_results:
            self._recalculate_parent_mf_by_energy()
        self._refresh_results_view()
        self._refresh_energy_parent_table()
        self.lbl_energy_parent_status.setText(f"已应用 {config_count} 个能量的替代参考")
        QtWidgets.QMessageBox.information(self, "成功", f"已应用 {config_count} 个能量的替代参考")

    def _get_parent_config_for_energy(self, energy: float) -> dict:
        if energy in self.energy_parent_config:
            cfg = self.energy_parent_config[energy]
            if cfg.get("mz"):
                return {
                    "mz": int(cfg["mz"]),
                    "species_name": cfg.get("species_name"),
                }
        return {
            "mz": int(self.spin_parent_mz.value()),
            "species_name": self._selected_parent_species_name(),
        }

    def _is_default_reference_config(self, cfg: dict) -> bool:
        default_mz = int(self.spin_parent_mz.value()) if hasattr(self, "spin_parent_mz") else self.settings.parent_mz
        default_species = self._selected_parent_species_name()
        cfg_species = cfg.get("species_name")
        return int(cfg.get("mz") or default_mz) == default_mz and cfg_species == default_species

    def _species_ie(self, mz: int, species_name: str | None) -> float | None:
        for name, ie in self._species_options_for_mz(mz):
            if species_name is None or name == species_name:
                return ie
        record = self._find_species_record(mz, species_name)
        if record is None:
            return None
        return record.get("ie")

    def _resolved_reference_species_name(self, mz: int, species_name: str | None) -> str | None:
        if species_name:
            return species_name
        options = self._species_options_for_mz(mz)
        if options:
            return options[0][0]
        record = self._find_species_record(mz)
        if record is None:
            return None
        return record.get("species") or record.get("name")

    def _mf_value_at_reference_temperature(self, mf: dict[float, float]) -> float:
        if not mf:
            return 0.0
        T0 = float(self.spin_parent_t0.value())
        if T0 in mf:
            return float(mf[T0])
        closest_temp = min(mf.keys(), key=lambda temp: abs(float(temp) - T0))
        return float(mf[closest_temp])

    def _create_auto_mf_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(8)

        ctrl_bar = QtWidgets.QWidget()
        ctrl_bar.setObjectName("ControlBar")
        ctrl_layout = QtWidgets.QHBoxLayout(ctrl_bar)
        ctrl_layout.setContentsMargins(10, 8, 10, 8)
        ctrl_layout.setSpacing(8)
        btn_calc_auto = QtWidgets.QPushButton("开始计算")
        btn_calc_auto.setObjectName("PrimaryButton")
        btn_calc_auto.clicked.connect(self._calculate_auto_mf)
        ctrl_layout.addWidget(btn_calc_auto)
        ctrl_layout.addWidget(QtWidgets.QLabel("曲线显示:"))
        self.combo_auto_plot_scope = QtWidgets.QComboBox()
        self.combo_auto_plot_scope.addItem("仅产物曲线", "products")
        self.combo_auto_plot_scope.addItem("全部曲线", "all")
        self.combo_auto_plot_scope.addItem("仅母体参考", "parents")
        self.combo_auto_plot_scope.addItem("选中结果", "selected")
        self.combo_auto_plot_scope.currentIndexChanged.connect(self._plot_all_auto_mf)
        ctrl_layout.addWidget(self.combo_auto_plot_scope)
        self.lbl_auto_status = QtWidgets.QLabel("未计算")
        self.lbl_auto_status.setObjectName("ProjectHint")
        ctrl_layout.addWidget(self.lbl_auto_status, 1)
        layout.addWidget(ctrl_bar)

        auto_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        auto_splitter.setChildrenCollapsible(False)

        result_group = QtWidgets.QGroupBox("计算结果")
        result_layout = QtWidgets.QVBoxLayout(result_group)
        result_layout.setContentsMargins(10, 8, 10, 10)
        self.auto_mf_table = QtWidgets.QTableWidget()
        self.auto_mf_table.setColumnCount(6)
        self.auto_mf_table.setHorizontalHeaderLabels(["类型", "质量数", "物种名称", "电离能(eV)", "光子能量(eV)", "状态"])
        self.auto_mf_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.auto_mf_table.itemSelectionChanged.connect(self._on_auto_mf_selection_changed)
        self.auto_mf_table.setMinimumHeight(170)
        result_layout.addWidget(self.auto_mf_table, 1)
        auto_splitter.addWidget(result_group)

        detail_tabs = QtWidgets.QTabWidget()
        plot_panel = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(8, 8, 8, 8)
        self.auto_mf_plot_widget = StaticCurvePlot("温度 (°C)", "摩尔分数", min_height=250)
        plot_layout.addWidget(self.auto_mf_plot_widget)
        detail_tabs.addTab(plot_panel, "摩尔分数-温度曲线")

        warning_panel = QtWidgets.QWidget()
        warning_layout = QtWidgets.QVBoxLayout(warning_panel)
        warning_layout.setContentsMargins(8, 8, 8, 8)
        self.txt_warnings = QtWidgets.QTextEdit()
        self.txt_warnings.setReadOnly(True)
        warning_layout.addWidget(self.txt_warnings)
        detail_tabs.addTab(warning_panel, "警告信息")

        auto_splitter.addWidget(detail_tabs)
        auto_splitter.setStretchFactor(0, 1)
        auto_splitter.setStretchFactor(1, 2)
        auto_splitter.setSizes([240, 430])
        layout.addWidget(auto_splitter, 1)

        return widget

    def _create_results_tab(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setSpacing(8)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_export = QtWidgets.QPushButton("导出结果 (Excel/CSV)")
        btn_export.setObjectName("ExportButton")
        btn_export.clicked.connect(self._export_results)
        btn_layout.addWidget(btn_export)
        btn_export_plot = QtWidgets.QPushButton("导出图表 (PNG/PDF)")
        btn_export_plot.setObjectName("ExportButton")
        btn_export_plot.clicked.connect(self._export_plot)
        btn_layout.addWidget(btn_export_plot)
        btn_refresh_results = QtWidgets.QPushButton("刷新结果")
        btn_refresh_results.setObjectName("BrowseButton")
        btn_refresh_results.clicked.connect(self._refresh_results_view)
        btn_layout.addWidget(btn_refresh_results)
        self.lbl_results_status = QtWidgets.QLabel("暂无结果")
        self.lbl_results_status.setObjectName("ProjectHint")
        btn_layout.addWidget(self.lbl_results_status, 1)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        results_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        results_splitter.setChildrenCollapsible(False)

        table_group = QtWidgets.QGroupBox("结果表格")
        table_layout = QtWidgets.QVBoxLayout(table_group)
        table_layout.setContentsMargins(10, 8, 10, 10)
        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.results_table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.results_table.setWordWrap(False)
        self.results_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.results_table.setMinimumHeight(180)
        table_layout.addWidget(self.results_table, 1)
        results_splitter.addWidget(table_group)

        plot_group = QtWidgets.QGroupBox("摩尔分数-温度曲线")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(10, 8, 10, 10)
        plot_body = QtWidgets.QWidget()
        plot_body_layout = QtWidgets.QHBoxLayout(plot_body)
        plot_body_layout.setContentsMargins(0, 0, 0, 0)
        plot_body_layout.setSpacing(8)
        self.mf_plot_widget = StaticCurvePlot("温度 (°C)", "摩尔分数", min_height=300)
        plot_body_layout.addWidget(self.mf_plot_widget, 1)
        series_group = QtWidgets.QGroupBox("曲线列表")
        series_layout = QtWidgets.QVBoxLayout(series_group)
        series_layout.setContentsMargins(8, 8, 8, 8)
        self.mf_series_list = QtWidgets.QListWidget()
        self.mf_series_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.mf_series_list.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.mf_series_list.setMinimumWidth(240)
        self.mf_series_list.setMaximumWidth(420)
        series_layout.addWidget(self.mf_series_list)
        plot_body_layout.addWidget(series_group)
        plot_layout.addWidget(plot_body, 1)
        results_splitter.addWidget(plot_group)
        results_splitter.setStretchFactor(0, 1)
        results_splitter.setStretchFactor(1, 2)
        results_splitter.setSizes([240, 380])
        layout.addWidget(results_splitter, 1)

        return widget

    def _apply_mz_calibration(self, raw_index):
        return self.calibration.a * raw_index ** 2 + self.calibration.b * raw_index + self.calibration.c

    def _load_peak_ranges_from_project(self) -> None:
        """从项目管理的 manual_peak_file 加载卡峰范围。

        项目管理是卡峰配置的唯一来源；摩尔分数对话框不再提供独立的卡峰导入UI。
        加载失败时静默回退到自动寻峰。
        """
        ps = self.project_settings
        manual_path = getattr(ps, "manual_peak_file", "") if ps is not None else ""
        if not manual_path or not Path(manual_path).exists():
            self.peak_ranges = {}
            return
        try:
            ranges = load_peak_ranges(manual_path, calibration=self.calibration)
            peak_ranges: dict[int, tuple[int, int]] = {}
            for item in ranges:
                mz = int(round(item.mz))
                left = int(item.left_bound)
                right = int(item.right_bound)
                if left > right:
                    left, right = right, left
                peak_ranges[mz] = (left, right)
            self.peak_ranges = peak_ranges
        except Exception:
            self.peak_ranges = {}

    def _recompute_peak_info(self) -> int:
        peak_ranges = self.peak_ranges or None
        total_peaks = 0
        for energy, temp_data in self.temperature_scan_data.items():
            if not temp_data:
                continue

            best_peak_count = -1
            best_peaks_info = []
            for temp in sorted(temp_data.keys()):
                t_data = temp_data[temp].get("avg_data", [])
                t_peaks = self._detect_and_integrate_peaks(t_data, peak_ranges)
                t_count = len(t_peaks)
                if t_count > best_peak_count:
                    best_peak_count = t_count
                    best_peaks_info = t_peaks

            for info in temp_data.values():
                info["peaks_info"] = best_peaks_info
                info.pop("matched_species", None)
                total_peaks += len(best_peaks_info)

        self._update_ts_table_with_species()
        self._refresh_ts_table()
        return total_peaks

    def _load_database(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择PICS截面数据库文件", "",
            "SQLite Files (*.sqlite *.db);;Pickle Files (*.pkl);;Excel Files (*.xlsx);;All Files (*)"
        )
        if not file_path:
            return
        try:
            self.database, self.mz_index = load_species_database(file_path)
            self._loaded_database_path = str(Path(file_path))
            self._update_parent_species_list()
            self._refresh_energy_parent_table()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"加载数据库失败: {e}")

    def _project_artifact_path(self, field_name: str) -> str:
        ps = self.project_settings
        return str(getattr(ps, field_name, "") or "") if ps is not None else ""

    def _read_table_file(self, file_path: str | Path) -> pd.DataFrame:
        path = Path(file_path)
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        return pd.read_csv(path, encoding="utf-8-sig")

    def _find_df_column(self, df: pd.DataFrame, candidates: list[str]) -> str | None:
        normalized = {str(col).strip().lower(): col for col in df.columns}
        for candidate in candidates:
            key = candidate.strip().lower()
            if key in normalized:
                return normalized[key]
        return None

    def _load_project_temperature_result(self):
        file_path = self._project_artifact_path("temperature_scan_result_file")
        if not file_path:
            QtWidgets.QMessageBox.warning(self, "提示", "项目管理中尚未登记温度扫描结果文件")
            return
        if not Path(file_path).exists():
            QtWidgets.QMessageBox.warning(self, "提示", f"项目登记的温度扫描结果文件不存在:\n{file_path}")
            return
        self._load_temperature_result_file(file_path, show_message=False, register_artifact=False)

    def _load_project_temperature_scan_folder(self):
        folder = self._project_artifact_path("temperature_scan_folder")
        if not folder:
            QtWidgets.QMessageBox.warning(self, "提示", "项目管理中尚未配置温度扫描数据文件夹")
            return
        root = Path(folder)
        if not root.exists() or not root.is_dir():
            QtWidgets.QMessageBox.warning(self, "提示", f"项目配置的温度扫描数据文件夹不存在:\n{folder}")
            return

        folders = self._discover_energy_folders(root)
        if not folders:
            QtWidgets.QMessageBox.warning(
                self,
                "提示",
                "未在项目温度扫描目录中找到可识别的能量子文件夹。\n"
                "请确保子文件夹名称包含能量信息，例如 8.0eV、9.5 eV。",
            )
            return

        self._load_peak_ranges_from_project()
        self._load_energy_folders(
            folders,
            source_label=f"项目原始目录: {root.name}",
            show_message=True,
            confirm_overwrite=False,
            replace_all=True,
        )

    def _load_temperature_result_file(
        self,
        file_path: str | Path | None = None,
        *,
        show_message: bool = True,
        register_artifact: bool = True,
    ):
        if isinstance(file_path, bool):
            file_path = None
        if file_path is None:
            start_path = self._project_artifact_path("temperature_scan_result_file")
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "选择温度扫描结果文件",
                str(Path(start_path).parent) if start_path else "",
                "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)",
            )
        if not file_path:
            return

        try:
            df = self._read_table_file(file_path)
            temp_col = self._find_df_column(df, ["temperature", "温度", "温度(°C)", "温度(C)"])
            mz_col = self._find_df_column(df, ["mz_rounded", "mz", "m/z", "质量数"])
            signal_col = self._find_df_column(
                df,
                ["area", "normalized_area", "最终强度", "photon_normalized_area", "IO归一化", "raw_area", "原始积分"],
            )
            if temp_col is None or mz_col is None or signal_col is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "提示",
                    "温度扫描结果缺少必需列。需要 temperature/温度、mz/mz_rounded/质量数、area/normalized_area。",
                )
                return

            energy_col = self._find_df_column(df, ["photon_energy", "energy", "能量(eV)", "光子能量"])
            file_col = self._find_df_column(df, ["file", "filename", "文件", "文件名"])
            io_col = self._find_df_column(df, ["io", "IO(nA)", "光强"])
            left_col = self._find_df_column(df, ["left_bound", "left_idx", "起始通道"])
            right_col = self._find_df_column(df, ["right_bound", "right_idx", "结束通道"])
            default_energy = (
                self.project_settings.mf_photon_energy
                if self.project_settings is not None
                else self.settings.photon_energy
            )

            loaded: dict[float, dict] = {}
            skipped = 0
            for _, row in df.iterrows():
                try:
                    temperature = float(row[temp_col])
                    mz_raw = float(row[mz_col])
                    mz = int(round(mz_raw))
                    signal = float(row[signal_col])
                except Exception:
                    skipped += 1
                    continue
                if not np.isfinite(temperature) or not np.isfinite(mz_raw) or not np.isfinite(signal):
                    skipped += 1
                    continue

                energy = default_energy
                if energy_col is not None and pd.notna(row.get(energy_col)):
                    try:
                        candidate_energy = float(row[energy_col])
                        if np.isfinite(candidate_energy) and candidate_energy > 0:
                            energy = candidate_energy
                    except Exception:
                        pass

                energy_data = loaded.setdefault(float(energy), {})
                info = energy_data.setdefault(
                    float(temperature),
                    {
                        "repeats": [],
                        "avg_data": [],
                        "avg_io": 1.0,
                        "filenames": Path(file_path).name,
                        "repeat_count": 1,
                        "peaks_info": [],
                        "precomputed_signals": {},
                        "_peak_map": {},
                        "_filenames": set(),
                        "_io_values": [],
                    },
                )
                if file_col is not None and pd.notna(row.get(file_col)):
                    info["_filenames"].add(str(row[file_col]))
                if io_col is not None and pd.notna(row.get(io_col)):
                    try:
                        io_value = float(row[io_col])
                        if np.isfinite(io_value):
                            info["_io_values"].append(io_value)
                    except Exception:
                        pass

                info["precomputed_signals"][mz] = float(info["precomputed_signals"].get(mz, 0.0)) + signal
                peak_map = info["_peak_map"]
                if mz not in peak_map:
                    left_idx = 0
                    right_idx = 0
                    if left_col is not None and pd.notna(row.get(left_col)):
                        left_idx = int(float(row[left_col]))
                    if right_col is not None and pd.notna(row.get(right_col)):
                        right_idx = int(float(row[right_col]))
                    peak_map[mz] = {
                        "index": 0,
                        "mz_raw": mz_raw,
                        "mz_rounded": mz,
                        "left_idx": left_idx,
                        "right_idx": right_idx,
                        "integral": 0.0,
                        "overlapped": False,
                    }
                peak_map[mz]["integral"] += signal

            if not loaded:
                QtWidgets.QMessageBox.warning(self, "提示", "未从温度扫描结果中读取到有效数据")
                return

            for energy_data in loaded.values():
                for info in energy_data.values():
                    filenames = sorted(info.pop("_filenames"))
                    if filenames:
                        info["filenames"] = ", ".join(filenames)
                        info["repeat_count"] = len(filenames)
                    io_values = info.pop("_io_values")
                    if io_values:
                        info["avg_io"] = float(np.mean(io_values))
                    peak_map = info.pop("_peak_map")
                    info["peaks_info"] = sorted(peak_map.values(), key=lambda item: item["mz_rounded"])

            self.temperature_scan_data = loaded
            self.available_energies = sorted(self.temperature_scan_data.keys())
            self.combo_energy_select.clear()
            self.combo_energy_select.addItem("全部能量")
            for energy in self.available_energies:
                self.combo_energy_select.addItem(f"{energy:.2f} eV")
            self.lbl_energy_count.setText(f"共 {len(self.available_energies)} 个能量点")
            self.lbl_ts_folder.setText(f"已加载温度结果: {Path(file_path).name}")
            self.lbl_ts_folder.setStyleSheet("color: #6495ed;")
            self.parent_mf_by_energy = {}
            self.parent_signal_by_energy = {}
            self.parent_config_by_energy = {}
            self._update_ts_table_with_species()
            self._refresh_ts_table()
            self._refresh_energy_parent_table()
            if register_artifact:
                record_project_artifact(
                    self,
                    "temperature_scan_result_file",
                    file_path,
                    message="温度扫描结果已登记到项目管理",
                )
            self.status_label.setText(f"已加载温度扫描结果: {len(df) - skipped} 行")
            if show_message:
                QtWidgets.QMessageBox.information(
                    self,
                    "成功",
                    f"加载了 {len(self.available_energies)} 个能量点的温度扫描结果",
                )
        except Exception as e:
            import traceback
            QtWidgets.QMessageBox.critical(self, "错误", f"加载温度扫描结果失败: {e}\n{traceback.format_exc()}")

    def _load_project_pie_results(self):
        file_path = self._project_artifact_path("pie_identification_result_file")
        if not file_path:
            QtWidgets.QMessageBox.warning(self, "提示", "项目管理中尚未登记PIE鉴定结果文件")
            return
        if not Path(file_path).exists():
            QtWidgets.QMessageBox.warning(self, "提示", f"项目登记的PIE鉴定结果文件不存在:\n{file_path}")
            return
        self._load_pie_results(file_path, show_message=False, register_artifact=False)

    def _load_pie_results(
        self,
        file_path: str | Path | None = None,
        *,
        show_message: bool = True,
        register_artifact: bool = True,
    ):
        if isinstance(file_path, bool):
            file_path = None
        if file_path is None:
            start_path = self._project_artifact_path("pie_identification_result_file")
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "选择PIE鉴定结果文件",
                str(Path(start_path).parent) if start_path else "",
                "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)",
            )
        if not file_path:
            return
        try:
            df = self._read_table_file(file_path)

            required_cols = ["质量数", "物种名称"]
            for col in required_cols:
                if col not in df.columns:
                    QtWidgets.QMessageBox.warning(self, "提示", f"文件中缺少必需列: {col}\n当前列: {', '.join(df.columns.tolist())}")
                    return

            self.pie_species_data = []
            self.pie_species_table.setRowCount(0)

            for _, row in df.iterrows():
                mz = int(row["质量数"]) if pd.notna(row["质量数"]) else None
                species_name = str(row["物种名称"]) if pd.notna(row["物种名称"]) else ""
                ie = float(row["电离能(eV)"]) if "电离能(eV)" in df.columns and pd.notna(row.get("电离能(eV)")) else None
                contribution = float(row["贡献比例(%)"]) if "贡献比例(%)" in df.columns and pd.notna(row.get("贡献比例(%)")) else None
                r_squared = float(row["R²"]) if "R²" in df.columns and pd.notna(row.get("R²")) else None

                if mz is None or not species_name:
                    continue

                self.pie_species_data.append({
                    "mz": mz,
                    "species": species_name,
                    "ie": ie,
                    "contribution": contribution,
                    "r_squared": r_squared,
                })

                table_row = self.pie_species_table.rowCount()
                self.pie_species_table.insertRow(table_row)
                self.pie_species_table.setItem(table_row, 0, QtWidgets.QTableWidgetItem(str(mz)))
                self.pie_species_table.setItem(table_row, 1, QtWidgets.QTableWidgetItem(species_name))
                self.pie_species_table.setItem(table_row, 2, QtWidgets.QTableWidgetItem(f"{ie:.2f}" if ie is not None else "N/A"))
                self.pie_species_table.setItem(table_row, 3, QtWidgets.QTableWidgetItem(f"{contribution:.1f}" if contribution is not None else "N/A"))
                self.pie_species_table.setItem(table_row, 4, QtWidgets.QTableWidgetItem(f"{r_squared:.4f}" if r_squared is not None else "N/A"))

            unique_mz = len(set(d["mz"] for d in self.pie_species_data))
            unique_species = len(set(d["species"] for d in self.pie_species_data))
            self.lbl_pie_status.setText(f"已加载 {unique_mz} 个质量数, {unique_species} 个物种")
            self.lbl_pie_status.setStyleSheet("color: #6495ed;")
            # Auto-expand PIE table group if data is present
            if hasattr(self, "pie_table_group"):
                self.pie_table_group.setChecked(self.pie_species_table.rowCount() > 0)

            self._update_ts_table_with_species()
            self._update_parent_species_list()
            self._refresh_energy_parent_table()
            if register_artifact:
                record_project_artifact(
                    self,
                    "pie_identification_result_file",
                    file_path,
                    message="PIE鉴定结果已登记到项目管理",
                )
            self.status_label.setText(f"已加载PIE鉴定结果: {unique_mz} 个质量数")

            if show_message:
                QtWidgets.QMessageBox.information(self, "成功", f"加载了 {len(self.pie_species_data)} 条鉴定结果\n{unique_mz} 个质量数, {unique_species} 个物种")
        except Exception as e:
            import traceback
            QtWidgets.QMessageBox.critical(self, "错误", f"加载PIE鉴定结果失败: {e}\n{traceback.format_exc()}")

    def _update_ts_table_with_species(self):
        if not self.pie_species_data:
            return
        species_by_mz: dict[int, list[str]] = {}
        for d in self.pie_species_data:
            mz = d["mz"]
            if mz not in species_by_mz:
                species_by_mz[mz] = []
            species_by_mz[mz].append(d["species"])

        for energy in self.available_energies:
            if energy not in self.temperature_scan_data:
                continue
            for temp, info in self.temperature_scan_data[energy].items():
                peaks_info = info.get("peaks_info", [])
                matched_species = []
                for peak in peaks_info:
                    if not peak.get("overlapped", False):
                        mz = peak["mz_rounded"]
                        if mz in species_by_mz:
                            matched_species.extend(species_by_mz[mz])
                info["matched_species"] = matched_species

    def _energy_from_folder_name(self, folder: str | Path) -> float | None:
        import re

        match = re.search(r"([\d.]+)\s*[eE][vV]", Path(folder).name)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _discover_energy_folders(self, root: Path) -> list[Path]:
        folders: list[Path] = []
        candidates = [root] + sorted(path for path in root.rglob("*") if path.is_dir())
        seen: set[Path] = set()
        for folder in candidates:
            if self._energy_from_folder_name(folder) is None:
                continue
            if not any(folder.glob("*.txt")):
                continue
            resolved = folder.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            folders.append(folder)
        return folders

    def _refresh_loaded_temperature_scan_state(self, source_label: str) -> None:
        self.available_energies = sorted(self.temperature_scan_data.keys())
        self.combo_energy_select.clear()
        self.combo_energy_select.addItem("全部能量")
        for e in self.available_energies:
            self.combo_energy_select.addItem(f"{e:.2f} eV")
        self.lbl_energy_count.setText(f"共 {len(self.available_energies)} 个能量点")
        self.parent_mf_by_energy = {}
        self.parent_signal_by_energy = {}
        self.parent_config_by_energy = {}
        self._update_ts_table_with_species()
        self._refresh_ts_table()
        self._refresh_energy_parent_table()
        self.lbl_ts_folder.setText(f"{source_label}，已加载 {len(self.available_energies)} 个能量点")
        self.lbl_ts_folder.setStyleSheet("color: #6495ed;")

    def _load_energy_folders(
        self,
        folders: list[str | Path],
        *,
        source_label: str = "温度扫描文件夹",
        show_message: bool = True,
        confirm_overwrite: bool = True,
        replace_all: bool = False,
    ) -> int:
        import re

        if replace_all:
            self.temperature_scan_data = {}

        added_count = 0
        skipped_count = 0
        added_energies: set[float] = set()

        for folder_value in folders:
            folder = Path(folder_value)
            try:
                energy = self._energy_from_folder_name(folder)
                folder_name = folder.name
                if energy is None:
                    if show_message:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "提示",
                            f"无法从文件夹名 '{folder_name}' 中识别能量值，请确保文件夹名包含能量信息（如 8.0eV）",
                        )
                    skipped_count += 1
                    continue
                if energy in self.temperature_scan_data and confirm_overwrite:
                    if QtWidgets.QMessageBox.question(self, "确认", f"能量 {energy:.2f} eV 已存在，是否覆盖？") != QtWidgets.QMessageBox.StandardButton.Yes:
                        skipped_count += 1
                        continue
                files = [f for f in folder.iterdir() if f.suffix.lower() == ".txt"]
                if not files:
                    if show_message:
                        QtWidgets.QMessageBox.warning(self, "提示", f"文件夹 '{folder_name}' 中没有txt文件")
                    skipped_count += 1
                    continue

                raw_data: dict[float, list] = {energy: []}

                for file_path in sorted(files):
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()

                    temp = None
                    for line in lines[:15]:
                        temp_match = re.search(r"Temperature[:\s]*([\d.]+)\s*C", line, re.IGNORECASE)
                        if not temp_match:
                            temp_match = re.search(r"Temp[:\s]*([\d.]+)", line, re.IGNORECASE)
                        if not temp_match:
                            temp_match = re.search(r"(\d+)\s*°?C", line)
                        if temp_match:
                            temp = int(float(temp_match.group(1)))
                            break
                    if temp is None:
                        temp_match = re.search(r"_(\d+)C?_", file_path.name)
                        if not temp_match:
                            temp_match = re.search(r"-(\d+)C?\.", file_path.name)
                        if not temp_match:
                            temp_match = re.search(r"(\d{2,3})C", file_path.name, re.IGNORECASE)
                        if not temp_match:
                            temp_match = re.search(r"(\d{2,3})_", file_path.name)
                        if temp_match:
                            temp = int(temp_match.group(1))
                        else:
                            try:
                                fallback_match = re.search(r"(\d+)", file_path.name)
                                temp = int(fallback_match.group(1)) if fallback_match else None
                            except Exception:
                                temp = None
                    if temp is None:
                        continue

                    io = None
                    for line in lines[:10]:
                        io_match = re.search(r"IO[:\s]*([\d.]+)", line, re.IGNORECASE)
                        if io_match:
                            io = float(io_match.group(1))
                            break
                    if io is None:
                        io = 100.0

                    data = []
                    start_line = min(10, len(lines))
                    for line in lines[start_line:]:
                        try:
                            data.append(float(line.strip()))
                        except ValueError:
                            pass
                    if len(data) < 100:
                        continue

                    raw_data[energy].append({
                        "file_path": str(file_path),
                        "filename": file_path.name,
                        "io": io,
                        "data": data,
                        "temp": temp,
                    })

                temp_groups: dict[int, list] = {}
                for entry in raw_data[energy]:
                    t = entry["temp"]
                    if t not in temp_groups:
                        temp_groups[t] = []
                    temp_groups[t].append(entry)
                if not temp_groups:
                    skipped_count += 1
                    continue

                energy_data = self.temperature_scan_data.setdefault(float(energy), {})
                for temp, repeats in temp_groups.items():
                    existing_repeats = list(energy_data.get(temp, {}).get("repeats", []))
                    merged_repeats = existing_repeats + list(repeats)
                    n = len(merged_repeats)
                    max_len = max(len(r["data"]) for r in merged_repeats)
                    avg_data = np.zeros(max_len)
                    count_arr = np.zeros(max_len)
                    for r in merged_repeats:
                        d = np.array(r["data"])
                        avg_data[: len(d)] += d
                        count_arr[: len(d)] += 1
                    count_arr[count_arr == 0] = 1
                    avg_data = avg_data / count_arr
                    avg_io = float(np.mean([r["io"] for r in merged_repeats]))
                    filenames = ", ".join(r["filename"] for r in merged_repeats)

                    energy_data[temp] = {
                        "repeats": merged_repeats,
                        "avg_data": avg_data.tolist(),
                        "avg_io": avg_io,
                        "filenames": filenames,
                        "repeat_count": n,
                        "peaks_info": None,
                    }

                temps_list = sorted(energy_data.keys())
                if temps_list:
                    best_peak_count = -1
                    best_peaks_info = None
                    peak_ranges = self.peak_ranges or None
                    for t in temps_list:
                        t_data = energy_data[t]["avg_data"]
                        t_peaks = self._detect_and_integrate_peaks(t_data, peak_ranges)
                        t_count = len(t_peaks)
                        if t_count > best_peak_count:
                            best_peak_count = t_count
                            best_peaks_info = t_peaks
                    peaks_info = best_peaks_info if best_peaks_info is not None else []

                    for temp in energy_data:
                        energy_data[temp]["peaks_info"] = peaks_info

                added_count += 1
                added_energies.add(float(energy))

            except Exception as e:
                if show_message:
                    QtWidgets.QMessageBox.critical(self, "错误", f"添加能量文件夹失败: {e}")
                skipped_count += 1

        if added_count > 0:
            self._refresh_loaded_temperature_scan_state(source_label)

        if added_count > 0:
            if show_message:
                QtWidgets.QMessageBox.information(
                    self,
                    "成功",
                    f"成功读取 {added_count} 个能量文件夹，合并为 {len(added_energies)} 个能量点的数据",
                )
            self.status_label.setText(
                f"已加载温度扫描原始数据: {added_count} 个文件夹 / {len(added_energies)} 个能量点"
            )
        if skipped_count > 0:
            if show_message:
                QtWidgets.QMessageBox.information(self, "提示", f"跳过了 {skipped_count} 个文件夹")
        if added_count == 0 and show_message:
            QtWidgets.QMessageBox.warning(self, "提示", "未从所选文件夹读取到有效温度扫描数据")
        return added_count

    def _add_energy_folder(self):
        dlg = QtWidgets.QFileDialog(self)
        dlg.setWindowTitle("选择温度扫描文件夹（可多选）")
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.Directory)
        dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        list_view = dlg.findChild(QtWidgets.QListView)
        if list_view:
            list_view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        tree_view = dlg.findChild(QtWidgets.QTreeView)
        if tree_view:
            tree_view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        if not dlg.exec():
            return
        folders = dlg.selectedFiles()
        if not folders:
            return
        self._load_energy_folders(folders, source_label="手动选择")

    def _clear_temperature_scan(self):
        if not self.temperature_scan_data:
            QtWidgets.QMessageBox.warning(self, "提示", "没有数据可清空")
            return
        if QtWidgets.QMessageBox.question(self, "确认", "确定要清空所有温度扫描数据吗？") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.temperature_scan_data = {}
        self.available_energies = []
        self.combo_energy_select.clear()
        self.combo_energy_select.addItem("全部能量")
        self.lbl_energy_count.setText("")
        self.ts_data_table.setRowCount(0)
        self.lbl_ts_folder.setText("未选择文件夹")
        self.lbl_ts_folder.setStyleSheet("color: rgba(232, 232, 232, 0.6);")
        self.parent_mf_by_energy = {}
        self.parent_signal_by_energy = {}
        self.parent_config_by_energy = {}
        self._refresh_energy_parent_table()
        QtWidgets.QMessageBox.information(self, "成功", "已清空所有温度扫描数据")

    def _remove_selected_energy(self):
        current_text = self.combo_energy_select.currentText()
        if current_text == "全部能量":
            QtWidgets.QMessageBox.warning(self, "提示", "请选择一个具体的能量")
            return
        try:
            energy = float(current_text.replace(" eV", ""))
            if energy not in self.temperature_scan_data:
                QtWidgets.QMessageBox.warning(self, "提示", "该能量数据不存在")
                return
            if QtWidgets.QMessageBox.question(self, "确认", f"确定要移除能量 {energy:.2f} eV 的数据吗？") != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            del self.temperature_scan_data[energy]
            self.available_energies = sorted(self.temperature_scan_data.keys())
            self.combo_energy_select.clear()
            self.combo_energy_select.addItem("全部能量")
            for e in self.available_energies:
                self.combo_energy_select.addItem(f"{e:.2f} eV")
            self.lbl_energy_count.setText(f"共 {len(self.available_energies)} 个能量点")
            self._refresh_ts_table()
            self._refresh_energy_parent_table()
            self.btn_remove_energy.setEnabled(False)
            QtWidgets.QMessageBox.information(self, "成功", f"已移除能量 {energy:.2f} eV 的数据")
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "提示", "无法解析能量值")

    def _on_energy_select(self, text):
        self.btn_remove_energy.setEnabled(text != "全部能量")

    def _refresh_ts_table(self):
        self.ts_data_table.setRowCount(0)
        for energy in self.available_energies:
            if energy not in self.temperature_scan_data:
                continue
            for temp in sorted(self.temperature_scan_data[energy].keys()):
                info = self.temperature_scan_data[energy][temp]
                peaks_info = info.get("peaks_info", [])
                peak_count = len([p for p in peaks_info if not p.get("overlapped", False)])
                overlapped_count = len([p for p in peaks_info if p.get("overlapped", False)])
                peak_info_text = f"{peak_count} 个峰"
                if overlapped_count > 0:
                    peak_info_text += f" ({overlapped_count} 个重合)"

                matched = info.get("matched_species", [])
                if matched:
                    peak_info_text += f" | {', '.join(matched[:5])}"
                    if len(matched) > 5:
                        peak_info_text += f" 等{len(matched)}个"

                row = self.ts_data_table.rowCount()
                self.ts_data_table.insertRow(row)
                self.ts_data_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{energy:.2f}"))
                self.ts_data_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{temp}"))
                self.ts_data_table.setItem(row, 2, QtWidgets.QTableWidgetItem(info["filenames"]))
                self.ts_data_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{info['avg_io']:.2f}"))
                self.ts_data_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{info['repeat_count']}"))
                self.ts_data_table.setItem(row, 5, QtWidgets.QTableWidgetItem(peak_info_text))
        # Auto-expand table group if data is present
        if hasattr(self, "ts_table_group"):
            self.ts_table_group.setChecked(self.ts_data_table.rowCount() > 0)

    def _detect_and_integrate_peaks(self, data, peak_ranges=None):
        peaks_info = []
        data_np = np.array(data)
        if len(data_np) == 0 or np.max(data_np) == 0:
            return []

        if peak_ranges:
            n = len(data_np)
            for mz, bounds in sorted(peak_ranges.items()):
                start, end = bounds
                start = max(0, int(start))
                end = min(n - 1, int(end))
                if start > end:
                    continue

                segment = data_np[start : end + 1]
                if len(segment) == 0:
                    continue

                peak_idx = start + int(np.argmax(segment))
                peak_integral = float(np.sum(segment))
                mz_est = self._apply_mz_calibration(peak_idx + 1)
                peaks_info.append({
                    "index": peak_idx,
                    "mz_raw": mz_est,
                    "mz_rounded": int(round(float(mz))),
                    "left_idx": start,
                    "right_idx": end,
                    "integral": peak_integral,
                    "overlapped": False,
                })

            peaks_info.sort(key=lambda x: x["mz_rounded"])
            return peaks_info

        detection_start = max(3000, 0)
        if detection_start >= len(data_np) - 1:
            return []

        n = len(data_np)
        local_max_indices = []
        for i in range(detection_start, n - 1):
            if data_np[i] > 3 and data_np[i] >= data_np[i - 1] and data_np[i] >= data_np[i + 1]:
                local_max_indices.append(i)

        local_max_indices.sort(key=lambda x: -data_np[x])

        used_indices: set[int] = set()
        final_max_indices: list[int] = []

        for max_idx in local_max_indices:
            if max_idx in used_indices:
                continue

            peak_val = data_np[max_idx]

            has_higher_nearby = False
            start_search = max(detection_start, max_idx - 30)
            end_search = min(n, max_idx + 31)

            for j in range(start_search, end_search):
                if j != max_idx and j in final_max_indices and data_np[j] > peak_val:
                    has_higher_nearby = True
                    break

            if has_higher_nearby:
                continue

            weak_tail_range = 90 if max_idx <= 15000 else 50
            is_weak_tail = False
            for prev_idx in final_max_indices:
                if max_idx > prev_idx and max_idx - prev_idx <= weak_tail_range:
                    if peak_val * 5 <= data_np[prev_idx]:
                        is_weak_tail = True
                        break

            if is_weak_tail:
                continue

            final_max_indices.append(max_idx)
            used_indices.add(max_idx)

            for j in range(max_idx - 10, max_idx + 11):
                if detection_start <= j < n:
                    used_indices.add(j)

        final_max_indices.sort()

        for pi in final_max_indices:
            mz_est = self._apply_mz_calibration(pi + 1)
            mz_rounded = round(mz_est)

            left_idx, right_idx = self._find_peak_bounds(pi, data)

            if right_idx - left_idx > 100:
                continue

            peak_integral = sum(data[left_idx : right_idx + 1])

            peaks_info.append({
                "index": pi,
                "mz_raw": mz_est,
                "mz_rounded": mz_rounded,
                "left_idx": left_idx,
                "right_idx": right_idx,
                "integral": peak_integral,
                "overlapped": False,
            })

        peaks_info.sort(key=lambda x: x["mz_rounded"])

        mz_groups: dict[int, list] = {}
        for peak in peaks_info:
            mz = peak["mz_rounded"]
            if mz not in mz_groups:
                mz_groups[mz] = []
            mz_groups[mz].append(peak)

        for mz, peaks in mz_groups.items():
            if len(peaks) > 1:
                peaks.sort(key=lambda x: -x["integral"])
                for peak in peaks[1:]:
                    peak["overlapped"] = True

        return peaks_info

    def _find_peak_bounds(self, peak_idx, data, min_threshold=0.1):
        max_val = data[peak_idx]
        threshold = max_val * min_threshold

        left_idx = peak_idx
        while left_idx > 0:
            left_idx -= 1
            if data[left_idx] <= threshold:
                left_idx = max(0, left_idx - 3)
                break

        right_idx = peak_idx
        while right_idx < len(data) - 1:
            right_idx += 1
            if data[right_idx] <= threshold:
                right_idx = min(len(data) - 1, right_idx + 3)
                break

        return left_idx, right_idx

    def _get_precomputed_signal_data(self, mz: int, energies_to_check: list[float]) -> dict[float, float]:
        result: dict[float, float] = {}
        target_mz = int(mz)
        for energy in energies_to_check:
            if energy not in self.temperature_scan_data:
                continue
            for temp, info in self.temperature_scan_data[energy].items():
                if temp in result:
                    continue
                signals = info.get("precomputed_signals", {})
                if target_mz not in signals:
                    continue
                try:
                    signal_val = float(signals[target_mz])
                except Exception:
                    continue
                if np.isfinite(signal_val):
                    result[float(temp)] = signal_val
        return result

    def _get_signal_from_scan_data(self, mz, energy=None):
        if not self.temperature_scan_data:
            return {}
        result: dict[float, float] = {}

        if energy is not None and energy in self.temperature_scan_data:
            energies_to_check = [energy]
        else:
            energies_to_check = self.available_energies

        target_peak = None

        for e in energies_to_check:
            if e not in self.temperature_scan_data:
                continue
            temps = list(self.temperature_scan_data[e].keys())
            if not temps:
                continue
            best_temp = None
            best_count = -1
            for t in temps:
                t_peaks = self.temperature_scan_data[e][t].get("peaks_info", [])
                if len(t_peaks) > best_count:
                    best_count = len(t_peaks)
                    best_temp = t
            if best_temp is None:
                continue
            peaks_info = self.temperature_scan_data[e][best_temp].get("peaks_info", [])

            for peak in peaks_info:
                if peak["mz_rounded"] == mz:
                    target_peak = peak
                    break
            if target_peak is not None:
                break

        if target_peak is None:
            return self._get_precomputed_signal_data(int(mz), energies_to_check)

        for e in energies_to_check:
            if e not in self.temperature_scan_data:
                continue
            for temp, info in self.temperature_scan_data[e].items():
                if temp in result:
                    continue
                data = info.get("avg_data", [])
                if data is None or len(data) == 0:
                    continue
                io = info.get("avg_io", 100.0)

                left_idx = max(0, int(target_peak["left_idx"]))
                right_idx = min(len(data) - 1, int(target_peak["right_idx"]))
                if right_idx < left_idx:
                    continue
                peak_integral = sum(data[left_idx : right_idx + 1])
                signal_val = peak_integral / io if io > 0 else peak_integral
                result[float(temp)] = signal_val

        if result:
            return result
        return self._get_precomputed_signal_data(int(mz), energies_to_check)

    def _species_record_matches(self, record: dict, mz: int, species_name: str | None = None) -> bool:
        if record.get("mz") != mz:
            return False
        if not species_name:
            return True
        return species_name in {record.get("species"), record.get("name")}

    def _find_species_record(self, mz: int, species_name: str | None = None) -> dict | None:
        for record in self.database:
            if self._species_record_matches(record, mz, species_name):
                return record
        if species_name:
            for record in self.database:
                if self._species_record_matches(record, mz):
                    return record
        return None

    def _recalculate_parent_mf_by_energy(self) -> None:
        if not self.available_energies or not self.parent_mf_results:
            self.parent_mf_by_energy = {}
            self.parent_signal_by_energy = {}
            self.parent_config_by_energy = {}
            return

        self.parent_mf_by_energy = {}
        self.parent_signal_by_energy = {}
        self.parent_config_by_energy = {}

        if not self.energy_parent_config:
            return

        main_parent_mz = self.spin_parent_mz.value()
        main_parent_energy = self.spin_parent_energy.value()
        main_parent_signal = self._get_signal_from_scan_data(main_parent_mz, main_parent_energy)
        if not main_parent_signal:
            return

        main_parent_mf_at_tm = self._mf_value_at_reference_temperature(self.parent_mf_results)
        if main_parent_mf_at_tm <= 0:
            return

        main_parent_species_name = self._selected_parent_species_name()
        for energy, cfg in sorted(self.energy_parent_config.items()):
            if energy not in self.temperature_scan_data:
                continue
            mz = int(cfg["mz"])
            signal_data = self._get_signal_from_scan_data(mz, energy)
            if not signal_data:
                continue

            species_name = self._resolved_reference_species_name(mz, cfg.get("species_name"))
            species = {
                "mz": mz,
                "species": species_name or "参考物种",
                "ie": self._species_ie(mz, species_name),
            }
            mf = self._calc_product_mf_auto(
                mz,
                species,
                main_parent_mz,
                float(main_parent_mz),
                main_parent_energy,
                main_parent_signal,
                main_parent_mf_at_tm,
                signal_data,
                calc_energy=energy,
                ref_species_name=main_parent_species_name,
            )
            if not mf:
                continue
            self.parent_mf_by_energy[energy] = mf
            self.parent_signal_by_energy[energy] = signal_data
            self.parent_config_by_energy[energy] = {
                "mz": mz,
                "species_name": species_name,
            }

    def _reference_parent_for_energy(
        self,
        energy: float,
    ) -> tuple[int, float, float, dict[float, float], float, str | None] | None:
        if self.energy_parent_config and not self.parent_mf_by_energy:
            self._recalculate_parent_mf_by_energy()

        if energy in self.energy_parent_config and energy not in self.parent_mf_by_energy:
            return None

        if energy in self.parent_mf_by_energy and energy in self.parent_signal_by_energy:
            cfg = self.parent_config_by_energy.get(energy, self._get_parent_config_for_energy(energy))
            mf = self.parent_mf_by_energy[energy]
            return (
                int(cfg["mz"]),
                float(cfg["mz"]),
                energy,
                self.parent_signal_by_energy[energy],
                self._mf_value_at_reference_temperature(mf),
                cfg.get("species_name"),
            )

        if self.parent_mf_results:
            parent_energy = self.spin_parent_energy.value()
            parent_mz = self.spin_parent_mz.value()
            parent_signal = self._get_signal_from_scan_data(parent_mz, parent_energy)
            if not parent_signal:
                return None
            return (
                parent_mz,
                float(parent_mz),
                parent_energy,
                parent_signal,
                self._mf_value_at_reference_temperature(self.parent_mf_results),
                self._selected_parent_species_name(),
            )

        return None

    def _reference_signal_for_candidate(
        self,
        mz: int,
        preferred_energy: float,
        fallback_energy: float,
        fallback_signal: dict[float, float] | None = None,
    ) -> tuple[float, dict[float, float]] | None:
        signal = self._get_signal_from_scan_data(mz, preferred_energy)
        if signal:
            return preferred_energy, signal
        if fallback_signal:
            return fallback_energy, fallback_signal
        for energy in sorted(self.available_energies):
            signal = self._get_signal_from_scan_data(mz, energy)
            if signal:
                return energy, signal
        return None

    def _reference_parent_for_species(
        self,
        energy: float,
        species_ie: float | None,
    ) -> tuple[int, float, float, dict[float, float], float, str | None] | None:
        if self.energy_parent_config and not self.parent_mf_by_energy:
            self._recalculate_parent_mf_by_energy()

        candidates: list[dict] = []

        if self.parent_mf_results:
            main_parent_mz = int(self.spin_parent_mz.value())
            main_parent_energy = float(self.spin_parent_energy.value())
            main_parent_species = self._selected_parent_species_name()
            main_parent_signal = self._get_signal_from_scan_data(main_parent_mz, main_parent_energy)
            signal_info = self._reference_signal_for_candidate(
                main_parent_mz,
                energy,
                main_parent_energy,
                main_parent_signal,
            )
            if signal_info is not None:
                signal_energy, signal_data = signal_info
                candidates.append(
                    {
                        "mz": main_parent_mz,
                        "species_name": main_parent_species,
                        "ie": self._species_ie(main_parent_mz, main_parent_species),
                        "configured_energy": main_parent_energy,
                        "signal_energy": signal_energy,
                        "signal_data": signal_data,
                        "mf_at_tm": self._mf_value_at_reference_temperature(self.parent_mf_results),
                    }
                )

        for configured_energy, mf in sorted(self.parent_mf_by_energy.items()):
            cfg = self.parent_config_by_energy.get(configured_energy, self._get_parent_config_for_energy(configured_energy))
            mz = int(cfg["mz"])
            species_name = cfg.get("species_name")
            signal_info = self._reference_signal_for_candidate(
                mz,
                energy,
                float(configured_energy),
                self.parent_signal_by_energy.get(configured_energy),
            )
            if signal_info is None:
                continue
            signal_energy, signal_data = signal_info
            candidates.append(
                {
                    "mz": mz,
                    "species_name": species_name,
                    "ie": self._species_ie(mz, species_name),
                    "configured_energy": float(configured_energy),
                    "signal_energy": signal_energy,
                    "signal_data": signal_data,
                    "mf_at_tm": self._mf_value_at_reference_temperature(mf),
                }
            )

        if not candidates:
            return None

        eligible: list[dict] = []
        if species_ie is not None and np.isfinite(species_ie):
            eligible = [
                candidate
                for candidate in candidates
                if candidate["ie"] is not None and candidate["ie"] <= species_ie
            ]

        if eligible:
            selected = min(
                eligible,
                key=lambda candidate: (
                    float(species_ie) - float(candidate["ie"]),
                    abs(float(candidate["signal_energy"]) - float(energy)),
                ),
            )
        else:
            exact_energy_candidates = [
                candidate
                for candidate in candidates
                if abs(float(candidate["configured_energy"]) - float(energy)) < 1e-9
                or abs(float(candidate["signal_energy"]) - float(energy)) < 1e-9
            ]
            selected = exact_energy_candidates[0] if exact_energy_candidates else candidates[0]

        return (
            int(selected["mz"]),
            float(selected["mz"]),
            float(selected["signal_energy"]),
            selected["signal_data"],
            float(selected["mf_at_tm"]),
            selected["species_name"],
        )

    def _calc_parent_mole_fraction(self):
        if not self.expansion_coefficients:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在参数设置中计算膨胀系数")
            return
        if not self._parent_mz_confirmed:
            QtWidgets.QMessageBox.warning(self, "提示", "请先确认母体分子离子 m/z，再开始计算。")
            return
        self.set_busy(True, "正在计算母体摩尔分数...")
        try:
            mz = self.spin_parent_mz.value()
            T0 = self.spin_parent_t0.value()
            X0 = self.spin_parent_mf0.value()
            energy = self.spin_parent_energy.value()
            if mz <= 0:
                QtWidgets.QMessageBox.warning(self, "提示", "请先设置母体质量数 m/z")
                return

            if energy not in self.temperature_scan_data:
                QtWidgets.QMessageBox.warning(
                    self, "提示",
                    f"未找到能量 {energy:.2f} eV 的温度扫描数据\n"
                    f"已加载的能量: {', '.join(f'{e:.2f} eV' for e in self.available_energies)}"
                )
                return

            signal_data = self._get_signal_from_scan_data(mz, energy)
            if not signal_data:
                QtWidgets.QMessageBox.warning(self, "提示", f"在能量 {energy:.2f} eV 下未找到质量数 {mz} 的峰")
                return

            self.parent_mf_results = calc_parent_mole_fraction(
                signal_data,
                reference_temperature=float(T0),
                parent_initial_mf=X0,
                expansion_coefficients=self._expansion_coefficients_for_energy(energy),
            )

            self.parent_result_table.setRowCount(0)
            for temp in sorted(self.parent_mf_results.keys()):
                row = self.parent_result_table.rowCount()
                self.parent_result_table.insertRow(row)
                self.parent_result_table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{int(temp)}"))
                self.parent_result_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{signal_data.get(temp, 0):.6f}"))
                self.parent_result_table.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{self._expansion_coefficient(temp, energy):.6f}"))
                self.parent_result_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{self.parent_mf_results[temp]:.6f}"))

            self._recalculate_parent_mf_by_energy()
            self._refresh_results_view()

            extra_msg = (
                f"\n已准备 {len(self.parent_mf_by_energy)} 个替代参考缓存"
                if self.parent_mf_by_energy
                else ""
            )
            QtWidgets.QMessageBox.information(self, "成功", f"计算了 {len(self.parent_mf_results)} 个温度点的母体摩尔分数{extra_msg}")
        finally:
            self.set_busy(False, "就绪")

    def _calculate_auto_mf(self):
        if not self.pie_species_data:
            QtWidgets.QMessageBox.warning(self, "提示", "请先加载PIE鉴定结果")
            return

        if not self.expansion_coefficients:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在参数设置中计算膨胀系数")
            return

        if not self.parent_mf_results:
            QtWidgets.QMessageBox.warning(self, "提示", "请先计算母体摩尔分数")
            return

        self.set_busy(True, "正在自动计算所有物种的摩尔分数...")
        try:
            self.txt_warnings.clear()
            warnings: list[str] = []

            species_by_mz: dict[int, list[dict]] = {}
            for d in self.pie_species_data:
                mz = d["mz"]
                if mz not in species_by_mz:
                    species_by_mz[mz] = []
                species_by_mz[mz].append(d)

            all_energies = sorted(self.available_energies)

            self.all_species_mf = {}
            self._recalculate_parent_mf_by_energy()

            parent_mz = self.spin_parent_mz.value()
            parent_mw = float(parent_mz)
            parent_energy = self.spin_parent_energy.value()
            parent_signal = self._get_signal_from_scan_data(parent_mz, parent_energy)
            parent_mf_at_tm = self._mf_value_at_reference_temperature(self.parent_mf_results)
            parent_species_name = self._selected_parent_species_name()

            parent_reference_keys = {(parent_mz, parent_species_name or "母体")}
            if parent_species_name is None:
                parent_reference_keys.update(
                    (parent_mz, species["species"])
                    for species in species_by_mz.get(parent_mz, [])
                )
            self.all_species_mf[(parent_mz, parent_species_name or "母体", parent_energy)] = self.parent_mf_results
            for energy, mf in self.parent_mf_by_energy.items():
                cfg = self.parent_config_by_energy.get(energy, self._get_parent_config_for_energy(energy))
                cfg_mz = int(cfg["mz"])
                label = cfg.get("species_name") or "参考物种"
                parent_reference_keys.add((cfg_mz, label))
                if cfg.get("species_name") is None:
                    parent_reference_keys.update(
                        (cfg_mz, species["species"])
                        for species in species_by_mz.get(cfg_mz, [])
                    )
                self.all_species_mf[(cfg_mz, label, energy)] = mf

            for mz, species_list in species_by_mz.items():
                active_species = [
                    species for species in species_list
                    if (mz, species["species"]) not in parent_reference_keys
                ]
                if not active_species:
                    continue

                if len(active_species) == 1:
                    species = active_species[0]
                    ie = species.get("ie", 0) or 0
                    usable_energies = [e for e in all_energies if e >= ie]

                    if not usable_energies:
                        warnings.append(f"质量数 {mz} 物种 {species['species']}: 没有高于电离能({ie:.2f} eV)的能量数据")
                        continue

                    calc_energy = select_calc_energy(ie, usable_energies)
                    signal_data = self._get_signal_from_scan_data(mz, calc_energy)
                    if not signal_data:
                        sorted_usable = sorted(usable_energies)
                        start_idx = sorted_usable.index(calc_energy) if calc_energy in sorted_usable else -1
                        for candidate_energy in sorted_usable[start_idx + 1:]:
                            signal_data = self._get_signal_from_scan_data(mz, candidate_energy)
                            if signal_data:
                                calc_energy = candidate_energy
                                break
                    if not signal_data:
                        warnings.append(
                            f"质量数 {mz} 物种 {species['species']}: "
                            f"在 {calc_energy:.2f} eV 及更高能量均没有信号数据"
                        )
                        continue

                    ref = self._reference_parent_for_species(calc_energy, ie)
                    if ref is None:
                        warnings.append(f"质量数 {mz} 物种 {species['species']}: 缺少 {calc_energy:.2f} eV 的可用参考物种")
                        continue

                    ref_mz, ref_mw, ref_energy, ref_signal, ref_mf_at_tm, ref_species_name = ref

                    mf = self._calc_product_mf_auto(
                        mz, species, ref_mz, ref_mw,
                        ref_energy, ref_signal, ref_mf_at_tm,
                        signal_data, calc_energy=calc_energy,
                        ref_species_name=ref_species_name,
                    )
                    if mf:
                        self.all_species_mf[(mz, species["species"], calc_energy)] = mf
                else:
                    species_list_sorted = sorted(
                        active_species,
                        key=lambda x: x.get("ie") if x.get("ie") is not None else float("inf"),
                    )
                    result = self._calc_multi_species_mf_auto(
                        mz, species_list_sorted, all_energies,
                        parent_mz, parent_mw, parent_energy,
                        parent_signal, parent_mf_at_tm,
                        ref_provider=self._reference_parent_for_species,
                    )
                    if result:
                        for key, mf in result.items():
                            if key != "warning" and (key[0], key[1]) not in parent_reference_keys:
                                self.all_species_mf[key] = mf
                        if "warning" in result:
                            warnings.append(result["warning"])

            self._update_auto_mf_table()

            if warnings:
                self.txt_warnings.setText("\n".join(warnings))

            parent_count, product_count = self._auto_result_counts()
            self.lbl_auto_status.setText(
                f"已计算 {len(self.all_species_mf)} 条结果（产物 {product_count}，母体参考 {parent_count}）"
            )
            self.lbl_auto_status.setStyleSheet("color: #6495ed;")

            self._plot_all_auto_mf()
            self._refresh_results_view()

            QtWidgets.QMessageBox.information(self, "完成", f"自动计算完成！\n共计算 {len(self.all_species_mf)} 条摩尔分数结果")
        except Exception as e:
            import traceback
            QtWidgets.QMessageBox.critical(self, "错误", f"计算摩尔分数时出错: {e}\n\n{traceback.format_exc()}")
        finally:
            self.set_busy(False, "就绪")

    def _calc_product_mf_auto(self, mz, species, ref_mz, ref_mw,
                               ref_energy, ref_signal_data, ref_mf_at_tm,
                               signal_data, calc_energy=None,
                               ref_species_name=None):
        ie = species.get("ie", 0) or 0
        species_name = species["species"]

        if calc_energy is None:
            usable_energies = [energy for energy in self.available_energies if energy >= ie]
            best_energy = select_calc_energy(ie, usable_energies) if usable_energies else None
        else:
            best_energy = calc_energy

        if best_energy is None:
            return None

        if not signal_data:
            return None

        species_obj = self._find_species_record(mz, species_name)

        ref_species_name = ref_species_name or species.get("ref_species") or species.get("ref_name")
        ref_obj = self._find_species_record(ref_mz, ref_species_name)

        if species_obj is None or ref_obj is None:
            return None

        energies_arr = species_obj.get("energies")
        cross_sections_arr = species_obj.get("cross_sections")
        if energies_arr is not None and cross_sections_arr is not None:
            sigma_i = _interpolate_cross_section(energies_arr, cross_sections_arr, best_energy)
        else:
            return None

        ref_energies = ref_obj.get("energies")
        ref_cross = ref_obj.get("cross_sections")
        if ref_energies is not None and ref_cross is not None:
            sigma_A = _interpolate_cross_section(ref_energies, ref_cross, ref_energy)
        else:
            return None

        if sigma_i == 0 or sigma_A == 0:
            return None

        D_i = calc_mass_discrimination(float(species.get("mw") or mz), self._mass_disc_exponent)
        D_A = calc_mass_discrimination(float(ref_mw), self._mass_disc_exponent)

        T_M = float(self.spin_parent_t0.value())
        if T_M not in ref_signal_data:
            T_M = max(ref_signal_data.keys()) if ref_signal_data else None
        if T_M is None:
            return None

        S_A_TM = ref_signal_data.get(T_M, 0)
        if S_A_TM == 0:
            return None

        lambda_TM = self._expansion_coefficient(T_M, ref_energy)
        results: dict[float, float] = {}
        for T, S_i in signal_data.items():
            lambda_T = self._expansion_coefficient(T, best_energy)
            X_i = (
                ref_mf_at_tm
                * (S_i / S_A_TM)
                * (sigma_A / sigma_i)
                * (D_A / D_i)
                * (lambda_TM / lambda_T)
            )
            results[T] = max(0.0, float(X_i))
        return results

    def _calc_multi_species_mf_auto(self, mz, species_list, energies,
                                     ref_mz, ref_mw, ref_energy,
                                     ref_signal_data, ref_mf_at_tm,
                                     ref_provider=None):
        result: dict = {}
        warnings: list[str] = []

        if not energies:
            warnings.append("没有可用能量数据")
            result["warning"] = f"质量数 {mz}: {'; '.join(warnings)}"
            return result

        sorted_energies = sorted(energies)
        energy_boundaries = [0] + sorted_energies

        species_groups: dict[int, list[dict]] = {}
        for s in species_list:
            ie = s.get("ie") if s.get("ie") is not None else float("inf")
            group_idx = None
            for i in range(len(energy_boundaries) - 1):
                lower = energy_boundaries[i]
                upper = energy_boundaries[i + 1]
                if lower < ie <= upper:
                    group_idx = i
                    break
            if group_idx is None:
                group_idx = len(energy_boundaries) - 1

            if group_idx not in species_groups:
                species_groups[group_idx] = []
            species_groups[group_idx].append(s)

        resolvable_species: list[dict] = []
        for group_idx, group in species_groups.items():
            if len(group) > 1:
                group.sort(key=lambda x: x.get("contribution", 0) or 0, reverse=True)
                best = group[0]
                resolvable_species.append(best)
                ignored_names = [s["species"] for s in group[1:]]
                upper_bound = (
                    f"{energy_boundaries[group_idx + 1]:.1f}"
                    if group_idx + 1 < len(energy_boundaries)
                    else "∞"
                )
                warnings.append(
                    f"电离能区间 ({energy_boundaries[group_idx]:.1f}-{upper_bound} eV) "
                    f"内存在 {len(group)} 个物种 ({', '.join(s['species'] for s in group)})，"
                    f"无法通过能量扫描分离，仅计算匹配系数最高的 {best['species']}，"
                    f"忽略了 {', '.join(ignored_names)}"
                )
            else:
                resolvable_species.append(group[0])

        resolvable_species.sort(key=lambda x: x.get("ie") if x.get("ie") is not None else float("inf"))

        if len(resolvable_species) <= 1:
            for species in resolvable_species:
                ie = species.get("ie") if species.get("ie") is not None else float("inf")
                if ie == float("inf"):
                    continue
                usable_energies = [e for e in sorted_energies if e >= ie]
                if not usable_energies:
                    continue
                calc_e = select_calc_energy(float(ie), usable_energies)
                if ref_provider is not None:
                    ref = ref_provider(calc_e, None if ie == float("inf") else float(ie))
                    if ref is not None:
                        ref_mz, ref_mw, ref_energy, ref_signal_data, ref_mf_at_tm, ref_species_name = ref
                    else:
                        continue
                else:
                    ref_species_name = None
                signal_data = self._get_signal_from_scan_data(mz, calc_e)
                if not signal_data:
                    continue
                mf = self._calc_product_mf_from_signal(
                    mz, species, calc_e, signal_data,
                    ref_mz, ref_mw, ref_energy,
                    ref_signal_data, ref_mf_at_tm,
                    ref_species_name=ref_species_name,
                )
                if mf:
                    result[(mz, species["species"], calc_e)] = mf
            if warnings:
                result["warning"] = f"质量数 {mz}: {'; '.join(warnings)}"
            return result

        try:
            energy_scan_data: dict[float, dict[float, float]] = {}
            for energy in sorted_energies:
                signal_data = self._get_signal_from_scan_data(mz, energy)
                if signal_data:
                    energy_scan_data[energy] = signal_data

            if not energy_scan_data:
                warnings.append("没有可用的温度扫描信号")
                result["warning"] = f"质量数 {mz}: {'; '.join(warnings)}"
                return result

            separated, pure_energies = separate_coexisting_species_signals(
                mz=mz,
                species_at_mz=[
                    {"species": s["species"], "ie": s.get("ie")}
                    for s in resolvable_species
                ],
                energy_scan_data=energy_scan_data,
                database=self.database,
                mz_index=self.mz_index,
            )

            for species in resolvable_species:
                species_name = species["species"]
                if species_name not in separated:
                    continue
                calc_e = pure_energies.get(species_name)
                if calc_e is None:
                    ie = species.get("ie") if species.get("ie") is not None else float("inf")
                    usable_energies = [e for e in sorted_energies if e >= ie]
                    if not usable_energies:
                        continue
                    calc_e = select_calc_energy(float(ie), usable_energies)
                if ref_provider is not None:
                    species_ie = species.get("ie") if species.get("ie") is not None else None
                    ref = ref_provider(calc_e, species_ie)
                    if ref is not None:
                        ref_mz, ref_mw, ref_energy, ref_signal_data, ref_mf_at_tm, ref_species_name = ref
                    else:
                        continue
                else:
                    ref_species_name = None
                mf = self._calc_product_mf_from_signal(
                    mz, species, calc_e, separated[species_name],
                    ref_mz, ref_mw, ref_energy,
                    ref_signal_data, ref_mf_at_tm,
                    ref_species_name=ref_species_name,
                )
                if mf:
                    result[(mz, species_name, calc_e)] = mf
        except Exception as exc:
            warnings.append(f"多物种信号分离失败，已回退到单能量计算 ({exc})")
            for species in resolvable_species:
                ie = species.get("ie") if species.get("ie") is not None else float("inf")
                if ie == float("inf"):
                    continue
                usable_energies = [e for e in sorted_energies if e >= ie]
                if not usable_energies:
                    continue
                calc_e = select_calc_energy(float(ie), usable_energies)
                if ref_provider is not None:
                    ref = ref_provider(calc_e, None if ie == float("inf") else float(ie))
                    if ref is not None:
                        ref_mz, ref_mw, ref_energy, ref_signal_data, ref_mf_at_tm, ref_species_name = ref
                    else:
                        continue
                else:
                    ref_species_name = None
                signal_data = self._get_signal_from_scan_data(mz, calc_e)
                if not signal_data:
                    continue
                mf = self._calc_product_mf_from_signal(
                    mz, species, calc_e, signal_data,
                    ref_mz, ref_mw, ref_energy,
                    ref_signal_data, ref_mf_at_tm,
                    ref_species_name=ref_species_name,
                )
                if mf:
                    result[(mz, species["species"], calc_e)] = mf

        unresolved_count = len(species_list) - len(
            set(k[1] for k in result if k != "warning")
        )
        ambiguous_count = sum(len(g) - 1 for g in species_groups.values() if len(g) > 1)
        if unresolved_count > ambiguous_count:
            warnings.append("部分可分辨物种未能计算摩尔分数")

        if warnings:
            result["warning"] = f"质量数 {mz}: {'; '.join(warnings)}"

        return result

    def _calc_product_mf_from_signal(self, mz, species, energy, signal_data,
                                      ref_mz, ref_mw, ref_energy,
                                      ref_signal_data, ref_mf_at_tm,
                                      ref_species_name=None):
        species_name = species["species"]

        species_obj = self._find_species_record(mz, species_name)

        ref_species_name = ref_species_name or species.get("ref_species") or species.get("ref_name")
        ref_obj = self._find_species_record(ref_mz, ref_species_name)
        if species_obj is None or ref_obj is None:
            return None

        energies_arr = species_obj.get("energies")
        cross_sections_arr = species_obj.get("cross_sections")
        if energies_arr is not None and cross_sections_arr is not None:
            sigma_i = _interpolate_cross_section(energies_arr, cross_sections_arr, energy)
        else:
            return None

        ref_energies = ref_obj.get("energies")
        ref_cross = ref_obj.get("cross_sections")
        if ref_energies is not None and ref_cross is not None:
            sigma_A = _interpolate_cross_section(ref_energies, ref_cross, ref_energy)
        else:
            return None

        if sigma_i == 0 or sigma_A == 0:
            return None

        D_i = calc_mass_discrimination(float(species.get("mw") or mz), self._mass_disc_exponent)
        D_A = calc_mass_discrimination(float(ref_mw), self._mass_disc_exponent)

        T_M = float(self.spin_parent_t0.value())
        if T_M not in ref_signal_data:
            T_M = max(ref_signal_data.keys()) if ref_signal_data else None
        if T_M is None:
            return None

        S_A_TM = ref_signal_data.get(T_M, 0)
        if S_A_TM == 0:
            return None

        lambda_TM = self._expansion_coefficient(T_M, ref_energy)
        results: dict[float, float] = {}
        for T, S_i in signal_data.items():
            if S_i <= 0:
                results[T] = 0.0
                continue
            lambda_T = self._expansion_coefficient(T, energy)
            X_i = (
                ref_mf_at_tm
                * (S_i / S_A_TM)
                * (sigma_A / sigma_i)
                * (D_A / D_i)
                * (lambda_TM / lambda_T)
            )
            results[T] = max(0.0, float(X_i))
        return results

    def _is_parent_auto_result_key(self, key: tuple[int, str, float]) -> bool:
        mz, species, energy = key
        cfg = self.parent_config_by_energy.get(energy)
        if cfg:
            cfg_label = cfg.get("species_name") or "参考物种"
            return int(cfg.get("mz", -1)) == mz and cfg_label == species

        parent_energy = self.spin_parent_energy.value() if hasattr(self, "spin_parent_energy") else None
        parent_mz = self.spin_parent_mz.value() if hasattr(self, "spin_parent_mz") else None
        parent_label = self._selected_parent_species_name() or "母体"
        return parent_energy == energy and parent_mz == mz and species == parent_label

    def _auto_result_counts(self) -> tuple[int, int]:
        parent_count = sum(1 for key in self.all_species_mf if self._is_parent_auto_result_key(key))
        return parent_count, max(0, len(self.all_species_mf) - parent_count)

    def _selected_auto_mf_key(self) -> tuple[int, str, float] | None:
        selected = self.auto_mf_table.selectedItems()
        if not selected:
            return None
        row = selected[0].row()
        mz_item = self.auto_mf_table.item(row, 1)
        species_item = self.auto_mf_table.item(row, 2)
        energy_item = self.auto_mf_table.item(row, 4)
        if not mz_item or not species_item or not energy_item:
            return None
        try:
            return (int(mz_item.text()), species_item.text(), float(energy_item.text()))
        except ValueError:
            return None

    def _auto_plot_keys_for_scope(self) -> list[tuple[int, str, float]]:
        scope = "products"
        if hasattr(self, "combo_auto_plot_scope"):
            scope = self.combo_auto_plot_scope.currentData() or "products"

        keys = list(self.all_species_mf.keys())
        if scope == "selected":
            selected = self._selected_auto_mf_key()
            return [selected] if selected in self.all_species_mf else []
        if scope == "parents":
            return [key for key in keys if self._is_parent_auto_result_key(key)]
        if scope == "products":
            product_keys = [key for key in keys if not self._is_parent_auto_result_key(key)]
            return product_keys or keys
        return keys

    def _update_auto_mf_table(self):
        self.auto_mf_table.setRowCount(0)
        sorted_items = sorted(
            self.all_species_mf.items(),
            key=lambda item: (
                self._is_parent_auto_result_key(item[0]),
                item[0][0],
                item[0][1],
                item[0][2],
            ),
        )
        for (mz, species, energy), mf in sorted_items:
            row = self.auto_mf_table.rowCount()
            self.auto_mf_table.insertRow(row)
            result_type = "母体参考" if self._is_parent_auto_result_key((mz, species, energy)) else "产物"
            self.auto_mf_table.setItem(row, 0, QtWidgets.QTableWidgetItem(result_type))
            self.auto_mf_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(mz)))
            self.auto_mf_table.setItem(row, 2, QtWidgets.QTableWidgetItem(species))

            ie = None
            for d in self.pie_species_data:
                if d["mz"] == mz and d["species"] == species:
                    ie = d.get("ie")
                    break
            self.auto_mf_table.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{ie:.2f}" if ie else "N/A"))
            self.auto_mf_table.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{energy:.2f}"))
            self.auto_mf_table.setItem(row, 5, QtWidgets.QTableWidgetItem("已计算"))

    def _on_auto_mf_selection_changed(self):
        key = self._selected_auto_mf_key()
        if key in self.all_species_mf:
            if hasattr(self, "combo_auto_plot_scope"):
                self.combo_auto_plot_scope.blockSignals(True)
                self._set_combo_current_data(self.combo_auto_plot_scope, "selected")
                self.combo_auto_plot_scope.blockSignals(False)
            self._plot_single_auto_mf(key)

    def _plot_all_auto_mf(self):
        if self.auto_mf_plot_widget is None:
            return
        if not self.all_species_mf:
            return

        self.auto_mf_plot_widget.clear_plot(xlabel="温度 (°C)", ylabel="摩尔分数")

        plot_keys = self._auto_plot_keys_for_scope()
        if not plot_keys:
            self.auto_mf_plot_widget.show_empty("没有可显示的曲线")
            return

        colors = [
            (100, 200, 255), (255, 150, 100), (100, 255, 150),
            (255, 100, 100), (255, 255, 100), (200, 150, 255),
            (255, 180, 220), (150, 255, 255), (255, 200, 150),
            (180, 255, 180),
        ]

        color_idx = 0
        species_energies: dict[str, list] = {}
        for mz, species, energy in plot_keys:
            if species not in species_energies:
                species_energies[species] = []
            species_energies[species].append((mz, energy))

        all_temps: list[float] = []
        all_values: list[float] = []
        x_ranges = []
        y_ranges = []
        for species, entries in species_energies.items():
            entries.sort(key=lambda x: x[1])
            for mz, energy in entries:
                key = (mz, species, energy)
                mf = self.all_species_mf[key]
                if not mf:
                    continue

                color = colors[color_idx % len(colors)]
                color_idx += 1

                sorted_temps = sorted(mf.keys())
                temps_arr = np.array(sorted_temps)
                mfs_arr = np.array([mf[t] for t in sorted_temps])
                all_temps.extend(float(t) for t in sorted_temps)
                all_values.extend(float(v) for v in mfs_arr)

                label = f"{species} ({energy:.1f}eV)"
                plot_x, plot_y = self.auto_mf_plot_widget.plot_series(
                    temps_arr,
                    mfs_arr,
                    color=color,
                    linewidth=2,
                    markersize=4.5,
                    label=label,
                )
                x_ranges.append(plot_x)
                y_ranges.append(plot_y)
        if all_temps and all_values:
            self.auto_mf_plot_widget.apply_data_limits(x_ranges, y_ranges, x_pad_min=50.0, y_pad_min=1e-9)
        self.auto_mf_plot_widget.finish(legend=True)

    def _plot_single_auto_mf(self, key):
        if self.auto_mf_plot_widget is None:
            return
        if key not in self.all_species_mf:
            return
        mf = self.all_species_mf[key]
        if not mf:
            return

        mz, species, energy = key

        self.auto_mf_plot_widget.clear_plot(xlabel="温度 (°C)", ylabel="摩尔分数")

        sorted_temps = sorted(mf.keys())
        temps_arr = np.array(sorted_temps)
        mfs_arr = np.array([mf[t] for t in sorted_temps])

        color = (100, 200, 255)
        label = f"m/z={mz} {species} @ {energy:.2f} eV"
        plot_x, plot_y = self.auto_mf_plot_widget.plot_series(
            temps_arr,
            mfs_arr,
            color=color,
            linewidth=2.8,
            markersize=5.5,
            label=label,
        )
        self.auto_mf_plot_widget.apply_data_limits([plot_x], [plot_y], x_pad_min=50.0, y_pad_min=1e-9)
        self.auto_mf_plot_widget.finish(legend=True)

    def _collect_all_results(self) -> dict[str, dict[float, float]]:
        return {series["label"]: series["values"] for series in self._collect_result_series()}

    def _add_result_series(
        self,
        series: list[dict[str, object]],
        seen_labels: set[str],
        result_type: str,
        label: str,
        values: dict[float, float],
        mz: int | None = None,
        species: str | None = None,
        energy: float | None = None,
    ) -> None:
        if not values or label in seen_labels:
            return
        seen_labels.add(label)
        series.append(
            {
                "type": result_type,
                "label": label,
                "values": values,
                "mz": mz,
                "species": species or label,
                "energy": energy,
            }
        )

    def _collect_auto_result_series(
        self,
        series: list[dict[str, object]],
        seen_labels: set[str],
    ) -> None:
        for mz, species, energy in self._auto_plot_keys_for_scope():
            mf = self.all_species_mf.get((mz, species, energy))
            if not mf:
                continue
            result_type = "母体参考" if self._is_parent_auto_result_key((mz, species, energy)) else "产物"
            name = f"{species}(m/z={mz},E={energy:.2f}eV)"
            self._add_result_series(
                series,
                seen_labels,
                result_type,
                name,
                mf,
                mz=mz,
                species=species,
                energy=energy,
            )

    def _collect_result_series(self) -> list[dict[str, object]]:
        series: list[dict[str, object]] = []
        seen_labels: set[str] = set()

        if hasattr(self, "all_species_mf") and self.all_species_mf:
            self._collect_auto_result_series(series, seen_labels)
            return series

        if self.parent_mf_results:
            mz = self.spin_parent_mz.value()
            species_name = self._selected_parent_species_name() or "母体"
            self._add_result_series(
                series,
                seen_labels,
                "母体",
                self._parent_result_label(mz, self._selected_parent_species_name()),
                self.parent_mf_results,
                mz=mz,
                species=species_name,
            )

        if self.parent_mf_by_energy:
            for energy, mf in sorted(self.parent_mf_by_energy.items()):
                cfg = self.parent_config_by_energy.get(energy, self._get_parent_config_for_energy(energy))
                mz = int(cfg["mz"])
                species_name = cfg.get("species_name") or "参考物种"
                self._add_result_series(
                    series,
                    seen_labels,
                    "母体参考",
                    self._parent_result_label(mz, cfg.get("species_name"), energy),
                    mf,
                    mz=mz,
                    species=species_name,
                    energy=energy,
                )

        for label, values in self.product_mf_results.items():
            self._add_result_series(series, seen_labels, "产物", label, values, species=label)

        for mz, species_data in self.isomeric_results.items():
            for species_name, mf_data in species_data.items():
                self._add_result_series(
                    series,
                    seen_labels,
                    "同分异构",
                    f"{species_name}(m/z={mz})",
                    mf_data,
                    mz=mz,
                    species=species_name,
                )

        return series

    def _refresh_results_view(self):
        self._update_results_table()
        self._plot_results_mf()

    def _update_results_table(self):
        result_series = self._collect_result_series()
        if not result_series:
            self.results_table.setRowCount(0)
            self.results_table.setColumnCount(0)
            if hasattr(self, "lbl_results_status"):
                self.lbl_results_status.setText("暂无结果")
            return

        all_temps = sorted(set().union(*(series["values"].keys() for series in result_series)))
        self.results_table.setRowCount(sum(len(series["values"]) for series in result_series))
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(["类型", "m/z", "物种/结果", "光子能量(eV)", "温度(°C)", "摩尔分数"])
        self.results_table.verticalHeader().setVisible(False)
        row = 0
        for series in result_series:
            values = series["values"]
            for temp in sorted(values.keys()):
                row_values = [
                    str(series["type"]),
                    "" if series["mz"] is None else str(series["mz"]),
                    str(series["species"]),
                    "" if series["energy"] is None else f"{float(series['energy']):.2f}",
                    f"{float(temp):.1f}",
                    f"{float(values[temp]):.6f}",
                ]
                for col, value in enumerate(row_values):
                    item = QtWidgets.QTableWidgetItem(value)
                    item.setToolTip(str(series["label"]))
                    self.results_table.setItem(row, col, item)
                row += 1

        header = self.results_table.horizontalHeader()
        for col in range(self.results_table.columnCount()):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for col, width in enumerate((70, 70, 280, 110, 90, 120)):
            self.results_table.setColumnWidth(col, width)
        if hasattr(self, "lbl_results_status"):
            self.lbl_results_status.setText(f"已显示 {len(result_series)} 条结果，{len(all_temps)} 个温度点")

    def _plot_results_mf(self):
        if self.mf_plot_widget is None:
            return

        result_series = self._collect_result_series()
        if hasattr(self, "mf_series_list"):
            self.mf_series_list.clear()
        if not result_series:
            self.mf_plot_widget.clear_plot(title="暂无摩尔分数结果", xlabel="温度 (°C)", ylabel="摩尔分数")
            return

        self.mf_plot_widget.clear_plot(title="摩尔分数-温度曲线", xlabel="温度 (°C)", ylabel="摩尔分数")

        colors = [
            (255, 100, 100), (100, 180, 255), (255, 200, 50),
            (180, 100, 255), (100, 255, 180), (255, 140, 50),
            (200, 255, 100), (255, 100, 200), (100, 220, 220),
            (220, 180, 140), (140, 100, 220), (220, 220, 100),
        ]

        all_temps = []
        all_mfs = []
        x_ranges = []
        y_ranges = []
        for idx, series in enumerate(result_series):
            color = colors[idx % len(colors)]
            name = str(series["label"])
            res = series["values"]
            sorted_temps = sorted(res.keys())
            all_temps.extend(sorted_temps)
            temps_arr = np.array(sorted_temps)
            mfs_arr = np.array([res[t] for t in sorted_temps])
            all_mfs.extend(mfs_arr.tolist())
            plot_x, plot_y = self.mf_plot_widget.plot_series(
                temps_arr,
                mfs_arr,
                color=color,
                linewidth=2,
                markersize=4.2,
            )
            x_ranges.append(plot_x)
            y_ranges.append(plot_y)
            if hasattr(self, "mf_series_list"):
                pixmap = QtGui.QPixmap(12, 12)
                pixmap.fill(QtGui.QColor(*color))
                item = QtWidgets.QListWidgetItem(QtGui.QIcon(pixmap), name)
                item.setToolTip(name)
                self.mf_series_list.addItem(item)

        if all_temps and all_mfs:
            self.mf_plot_widget.apply_data_limits(x_ranges, y_ranges, x_pad_min=50.0, y_pad_min=1e-9)
        self.mf_plot_widget.finish()

    def _export_results(self):
        all_results = self._collect_all_results()
        if not all_results:
            QtWidgets.QMessageBox.warning(self, "提示", "没有计算结果可导出")
            return
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "导出摩尔分数结果", "", "Excel Files (*.xlsx);;CSV Files (*.csv)"
        )
        if not file_path:
            return
        try:
            all_temps = sorted(set().union(*(r.keys() for r in all_results.values())))
            data = {"温度(°C)": all_temps}
            for name, res in all_results.items():
                data[name] = [res.get(t, 0) for t in all_temps]
            df = pd.DataFrame(data)
            if file_path.endswith(".xlsx"):
                df.to_excel(file_path, index=False)
            else:
                df.to_csv(file_path, index=False, encoding="utf-8-sig")
            record_project_artifact(
                self,
                "mole_fraction_result_file",
                file_path,
                message="摩尔分数结果已登记到项目管理",
            )
            QtWidgets.QMessageBox.information(self, "成功", "摩尔分数结果导出成功，并已登记到项目管理。")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"导出失败: {e}")

    def _export_plot(self):
        plot_widget = self.mf_plot_widget or self.auto_mf_plot_widget
        if plot_widget is None or plot_widget.figure is None:
            QtWidgets.QMessageBox.warning(self, "提示", "没有可导出的图表")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出摩尔分数-温度曲线图",
            str(ensure_output_dir("exports", "mole_fraction") / "mole_fraction_plot.png"),
            "PNG Images (*.png);;PDF Files (*.pdf)",
        )
        if not path:
            return
        if plot_widget.save_plot(path):
            QtWidgets.QMessageBox.information(self, "成功", f"曲线图已导出：{path}")
        else:
            QtWidgets.QMessageBox.warning(self, "错误", "导出图表失败")

    def set_temperature_scan_df(self, df: pd.DataFrame):
        self._temperature_scan_df = df

    def load_species_database(self, path: str | Path | None = None):
        if path is None:
            path = species_database_path()
        try:
            self.database, self.mz_index = load_species_database(path)
            self._loaded_database_path = str(Path(path))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "错误", f"加载数据库失败: {e}")
