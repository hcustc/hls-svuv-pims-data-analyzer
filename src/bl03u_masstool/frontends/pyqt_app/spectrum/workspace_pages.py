from __future__ import annotations

import os
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout

from bl03u_masstool.core.config import species_database_path
from bl03u_masstool.core.normalization import load_normalization_settings
from bl03u_masstool.core.project_lifecycle import (
    PROJECT_DIRECTORIES,
    PROJECT_SOURCE_SPECS,
    ProjectUIState,
    build_project_stage_statuses,
    collect_project_files,
    create_project_snapshot,
    ensure_project_structure,
    export_project_archive,
    get_project_ui_state,
    import_initial_project_data,
    next_project_stage,
    project_root,
    sanitize_project_slug,
)
from bl03u_masstool.core.project_settings import ProjectSettings, ProjectSettingsManager
from bl03u_masstool.frontends.pyqt_app.isotope.dialog import IsotopeAbundanceDialog
from bl03u_masstool.frontends.pyqt_app.mole_fraction.dialog import MoleFractionDialog
from bl03u_masstool.frontends.pyqt_app.nist.widget import IonizationEnergyLookupWidget
from bl03u_masstool.frontends.pyqt_app.normalization.widget import NormalizationSettingsWidget
from bl03u_masstool.frontends.pyqt_app.pics.dialog import PICSCalculatorDialog
from bl03u_masstool.frontends.pyqt_app.pie.dialog import PIESpeciesFitDialog
from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog


class ProjectImportDialog(QtWidgets.QDialog):
    """Small import wizard for copying initial project inputs into the project tree."""

    _directory_sources = {"sum_spectrum", "temperature_scan", "pie_scan"}
    _file_filters = {
        "single_spectrum": "质谱数据 (*.txt *.asc *.888);;所有文件 (*)",
        "sample_info": "样品信息 (*.xlsx *.xls *.csv *.tsv *.txt);;所有文件 (*)",
        "manual_peak": "Peak Files (*.yaml *.yml *.csv *.xlsx *.xls);;所有文件 (*)",
    }

    def __init__(self, settings: ProjectSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("原始数据导入向导")
        self.setMinimumWidth(760)
        self.path_edits: dict[str, QLineEdit] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        root_label = QtWidgets.QLabel(f"导入目标：{project_root(settings)}", self)
        root_label.setObjectName("ProjectHint")
        root_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(root_label)

        hint = QtWidgets.QLabel(
            "选择需要纳入项目生命周期管理的初始输入。导入会复制文件/目录到项目目录，并自动回填项目数据源。",
            self,
        )
        hint.setObjectName("ProjectHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)
        form.setColumnStretch(1, 1)
        for row, source_key in enumerate(
            ("single_spectrum", "sum_spectrum", "temperature_scan", "pie_scan", "sample_info", "manual_peak")
        ):
            spec = PROJECT_SOURCE_SPECS[source_key]
            label = QtWidgets.QLabel(spec.label, self)
            edit = QLineEdit(self)
            edit.setClearButtonEnabled(True)
            edit.setPlaceholderText("可留空")
            browse_btn = QPushButton("选择", self)
            browse_btn.setObjectName("BrowseButton")
            browse_btn.clicked.connect(lambda checked=False, key=source_key: self._browse_source(key))
            self.path_edits[source_key] = edit
            form.addWidget(label, row, 0)
            form.addWidget(edit, row, 1)
            form.addWidget(browse_btn, row, 2)
        layout.addLayout(form)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("导入到项目")
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_source(self, source_key: str) -> None:
        edit = self.path_edits[source_key]
        start_dir = self._dialog_start_dir(edit.text())
        if source_key in self._directory_sources:
            path = QFileDialog.getExistingDirectory(self, f"选择{PROJECT_SOURCE_SPECS[source_key].label}", start_dir)
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                f"选择{PROJECT_SOURCE_SPECS[source_key].label}",
                start_dir,
                self._file_filters.get(source_key, "所有文件 (*)"),
            )
        if path:
            edit.setText(path)

    def _dialog_start_dir(self, current: str) -> str:
        if current:
            path = Path(current).expanduser()
            if path.is_dir():
                return str(path)
            if path.parent.exists():
                return str(path.parent)
        return str(Path.home())

    def selected_sources(self) -> dict[str, str]:
        return {
            source_key: edit.text().strip()
            for source_key, edit in self.path_edits.items()
            if edit.text().strip()
        }


class WorkspacePagesMixin:
    def _build_workspace_pages(self):
        if hasattr(self, "workspace_stack"):
            return

        self.verticalLayout_9.removeItem(self.verticalLayout_7)

        self.page_nav = QtWidgets.QWidget(self.centralwidget)
        self.page_nav.setObjectName("PageNav")
        nav_layout = QHBoxLayout(self.page_nav)
        nav_layout.setContentsMargins(8, 6, 8, 6)
        nav_layout.setSpacing(2)

        self.workspace_stack = QtWidgets.QStackedWidget(self.centralwidget)
        self.workspace_stack.setObjectName("WorkspaceStack")

        self.spectrum_page = QtWidgets.QWidget(self.workspace_stack)
        self.spectrum_page.setObjectName("SpectrumPage")
        self.spectrum_page.setLayout(self.verticalLayout_7)
        self.workspace_stack.addWidget(self.spectrum_page)

        self.normalization_settings = load_normalization_settings()
        self.temperature_page = TemperatureScanDialog(
            self.current_calibration(),
            self.normalization_settings,
            self.workspace_stack,
        )
        self.temperature_page.set_project_settings(self.project_settings_manager.get())
        self.pie_page = PIESpeciesFitDialog(
            self.current_calibration(),
            self.normalization_settings,
            self.workspace_stack,
        )
        self.pie_page.set_project_settings(self.project_settings_manager.get())
        self.mole_fraction_page = MoleFractionDialog(
            self.current_calibration(),
            self.normalization_settings,
            self.workspace_stack,
        )
        self.mole_fraction_page.set_project_settings(self.project_settings_manager.get())
        self.ionization_page = IonizationEnergyLookupWidget(self.workspace_stack)
        self.isotope_page = IsotopeAbundanceDialog(self.workspace_stack)
        self.pics_page = PICSCalculatorDialog(
            self.current_calibration(),
            self.normalization_settings,
            self.workspace_stack,
        )
        self.pics_page.set_project_settings(self.project_settings_manager.get())
        self.project_page = QtWidgets.QWidget(self.workspace_stack)
        self.project_page.setObjectName("ProjectPage")
        self._build_project_page()

        pages = [
            ("project", "项目管理", self.project_page),
            ("spectrum", "质谱工作台", self.spectrum_page),
            ("temperature", "温度扫描", self.temperature_page),
            ("pie", "PIE拟合", self.pie_page),
            ("mole_fraction", "摩尔分数", self.mole_fraction_page),
            ("pics", "PICS计算", self.pics_page),
            ("ionization", "IE查询", self.ionization_page),
            ("isotope", "分子/同位素", self.isotope_page),
        ]
        # Pages that get a separator inserted AFTER them in the nav bar
        _nav_separators_after = {"spectrum", "mole_fraction"}

        self.page_buttons: dict[str, QtWidgets.QToolButton] = {}
        self.page_button_group = QtWidgets.QButtonGroup(self.page_nav)
        self.page_button_group.setExclusive(True)
        for index, (page_name, label, page) in enumerate(pages):
            if page is not self.spectrum_page:
                self.workspace_stack.addWidget(page)
            button = QtWidgets.QToolButton(self.page_nav)
            button.setText(label)
            button.setObjectName("WorkspaceTab")
            button.setCheckable(True)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setMinimumHeight(32)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Preferred,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            button.clicked.connect(lambda checked=False, name=page_name: self.switch_workspace_page(name))
            self.page_button_group.addButton(button, index)
            self.page_buttons[page_name] = button
            nav_layout.addWidget(button)
            if page_name in _nav_separators_after:
                sep = QtWidgets.QFrame(self.page_nav)
                sep.setFrameShape(QtWidgets.QFrame.Shape.VLine)
                sep.setObjectName("NavSeparator")
                sep.setFixedWidth(1)
                sep.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Fixed,
                    QtWidgets.QSizePolicy.Policy.Expanding,
                )
                nav_layout.addWidget(sep)
                nav_layout.addSpacing(2)
        nav_layout.addStretch(1)

        self.page_buttons["spectrum"].setChecked(True)
        self.verticalLayout_9.insertWidget(0, self.page_nav)
        self.verticalLayout_9.insertWidget(1, self.workspace_stack, stretch=1)
        self.workspace_stack.setCurrentWidget(self.spectrum_page)
        self._route_common_parameter_buttons()

        # Auto-pull tool page paths back to project management when edited
        self.lineEdit.editingFinished.connect(lambda: self._pull_path_from_tool("single"))
        self.folder_path.editingFinished.connect(lambda: self._pull_path_from_tool("sum"))
        # Note: temperature_page and pie_page are now read-only parameter display,
        # so no signal connections needed - parameters come from ProjectSettings

    def _build_project_page(self):
        page_layout = QVBoxLayout(self.project_page)
        page_layout.setContentsMargins(12, 12, 12, 12)
        page_layout.setSpacing(10)

        self.project_tabs = QtWidgets.QTabWidget(self.project_page)
        self.project_tabs.setObjectName("ProjectTabs")

        # --- Tab 1: 项目设置 (project identity only) ---
        self.project_identity_page = QtWidgets.QWidget(self.project_tabs)
        identity_layout = QVBoxLayout(self.project_identity_page)
        identity_layout.setContentsMargins(8, 8, 8, 8)
        identity_layout.setSpacing(10)
        self._build_project_identity_card(self.project_identity_page)
        self._build_lifecycle_card(self.project_identity_page)
        identity_layout.addWidget(self.project_identity_card)
        identity_layout.addWidget(self.project_lifecycle_card)
        identity_layout.addStretch(1)

        # --- Tab 2: 数据源 ---
        self.project_datasource_page = QtWidgets.QWidget(self.project_tabs)
        datasource_layout = QVBoxLayout(self.project_datasource_page)
        datasource_layout.setContentsMargins(8, 8, 8, 8)
        datasource_layout.setSpacing(10)
        self._build_datasource_card(self.project_datasource_page)
        datasource_layout.addWidget(self.datasource_card)
        datasource_layout.addStretch(1)

        # --- Tab 3: 产物管理 ---
        self.project_artifacts_page = QtWidgets.QWidget(self.project_tabs)
        artifacts_layout = QVBoxLayout(self.project_artifacts_page)
        artifacts_layout.setContentsMargins(8, 8, 8, 8)
        artifacts_layout.setSpacing(10)
        self._build_artifact_manager_card(self.project_artifacts_page)
        artifacts_layout.addWidget(self.artifact_manager_card, stretch=1)

        # --- Tab 4: 参数配置 (通用参数 + 功能参数 merged) ---
        self.project_common_page = QtWidgets.QWidget(self.project_tabs)
        common_layout = QVBoxLayout(self.project_common_page)
        common_layout.setContentsMargins(0, 0, 0, 0)
        common_scroll = QtWidgets.QScrollArea(self.project_common_page)
        common_scroll.setWidgetResizable(True)
        common_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        common_scroll_widget = QtWidgets.QWidget()
        common_scroll_layout = QVBoxLayout(common_scroll_widget)
        common_scroll_layout.setContentsMargins(8, 8, 8, 8)
        common_scroll_layout.setSpacing(12)

        # Section header: 通用参数
        common_section_title = QtWidgets.QLabel("通用参数（定标·归一化·寻峰）", common_scroll_widget)
        common_section_title.setObjectName("ProjectTitle")
        common_hint = QtWidgets.QLabel(
            "以下参数影响主工作台、温度扫描、PIE、摩尔分数和 PICS；保存后自动同步到各工具。",
            common_scroll_widget,
        )
        common_hint.setObjectName("ProjectHint")
        common_hint.setWordWrap(True)
        self.project_common_settings_widget = NormalizationSettingsWidget(
            self.normalization_settings,
            self.current_calibration(),
            common_scroll_widget,
        )
        self.project_common_settings_widget.save_button.clicked.connect(self.on_project_common_parameters_saved)
        self.project_common_settings_widget.settings_saved.connect(self.on_project_common_parameters_saved)

        # Divider
        divider = QtWidgets.QFrame(common_scroll_widget)
        divider.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        divider.setObjectName("NavSeparator")

        # Section header: 功能参数
        func_section_title = QtWidgets.QLabel("功能默认参数（PIE·温度扫描·PICS·摩尔分数）", common_scroll_widget)
        func_section_title.setObjectName("ProjectTitle")
        func_hint = QtWidgets.QLabel(
            '以下默认值在"应用到工具"或切换工具页时自动同步；保存后写入项目配置。',
            common_scroll_widget,
        )
        func_hint.setObjectName("ProjectHint")
        func_hint.setWordWrap(True)
        self._build_function_params_card(common_scroll_widget)

        common_scroll_layout.addWidget(common_section_title)
        common_scroll_layout.addWidget(common_hint)
        common_scroll_layout.addWidget(self.project_common_settings_widget)
        common_scroll_layout.addWidget(divider)
        common_scroll_layout.addWidget(func_section_title)
        common_scroll_layout.addWidget(func_hint)
        common_scroll_layout.addWidget(self.function_params_card)
        common_scroll_layout.addStretch(1)
        common_scroll.setWidget(common_scroll_widget)
        common_layout.addWidget(common_scroll)

        self.project_tabs.addTab(self.project_identity_page, "项目设置")
        self.project_tabs.addTab(self.project_datasource_page, "数据导入")
        self.project_tabs.addTab(self.project_artifacts_page, "产物管理")
        self.project_tabs.addTab(self.project_common_page, "参数配置")
        page_layout.addWidget(self.project_tabs, stretch=1)

        self.load_project_settings()
        self.refresh_project_parameter_summary()

    # ── Tab 1: Project Identity ──────────────────────────────────────────

    def _build_project_identity_card(self, parent):
        self.project_identity_card = QtWidgets.QFrame(parent)
        self.project_identity_card.setObjectName("ProjectCard")
        self.project_identity_card.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        card_layout = QVBoxLayout(self.project_identity_card)
        card_layout.setContentsMargins(10, 8, 10, 10)
        card_layout.setSpacing(10)

        hero = QtWidgets.QFrame(self.project_identity_card)
        hero.setObjectName("ProjectHero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(10, 8, 10, 8)
        hero_layout.setSpacing(10)

        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(4)
        title = QtWidgets.QLabel("项目概览", hero)
        title.setObjectName("ProjectTitle")
        self.project_status_label = QtWidgets.QLabel("", hero)
        self.project_status_label.setObjectName("ProjectStatus")
        self.project_status_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        hint = QtWidgets.QLabel("集中管理项目信息、数据源路径、通用参数和功能默认值。", hero)
        hint.setObjectName("ProjectHint")
        hint.setWordWrap(True)
        title_column.addWidget(title)
        title_column.addWidget(self.project_status_label)
        title_column.addWidget(hint)
        hero_layout.addLayout(title_column, stretch=1)

        action_bar = QtWidgets.QWidget(hero)
        action_bar.setObjectName("ProjectActionBar")
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(8, 6, 8, 6)
        action_layout.setSpacing(6)
        self.project_save_button = QPushButton("保存项目", action_bar)
        self.project_apply_button = QPushButton("应用到工具", action_bar)
        self.project_capture_button = QPushButton("读取当前路径", action_bar)
        self.project_save_button.setToolTip("保存当前项目页中的项目信息、数据源路径和功能默认参数。")
        self.project_apply_button.setToolTip("将项目路径和默认参数同步到主工作台、温度扫描、PIE 等工具页面，并保存项目配置。")
        self.project_capture_button.setToolTip("从当前各工具页面读取已选择的文件/目录，回填到项目数据源；不会自动保存。")
        for button in (self.project_save_button, self.project_apply_button, self.project_capture_button):
            button.setFixedHeight(28)
            action_layout.addWidget(button)
        self.project_apply_button.setObjectName("BrowseButton")
        self.project_capture_button.setObjectName("BrowseButton")
        hero_layout.addWidget(action_bar)
        card_layout.addWidget(hero)

        form_layout = QtWidgets.QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)

        self.project_name_edit = QLineEdit(self.project_identity_card)
        self.project_name_edit.setPlaceholderText("例如 C6F11O2H 温度扫描项目")
        self.project_system_edit = QLineEdit(self.project_identity_card)
        self.project_system_edit.setPlaceholderText("例如 C6F11O2H")
        self.project_description_edit = QLineEdit(self.project_identity_card)
        self.project_description_edit.setPlaceholderText("简要说明样品、批次或实验条件")
        self.project_output_dir_edit = QLineEdit(self.project_identity_card)
        self.project_output_dir_edit.setPlaceholderText("默认 output/Project_<项目名>")

        output_row = QtWidgets.QWidget(self.project_identity_card)
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(6)
        output_layout.addWidget(self.project_output_dir_edit, stretch=1)
        self.project_output_dir_button = QPushButton("选择", output_row)
        self.project_output_dir_button.setObjectName("BrowseButton")
        self.project_output_dir_button.setFixedHeight(28)
        output_layout.addWidget(self.project_output_dir_button)

        form_layout.addRow("项目名", self.project_name_edit)
        form_layout.addRow("实验体系", self.project_system_edit)
        form_layout.addRow("描述", self.project_description_edit)
        form_layout.addRow("项目目录", output_row)
        card_layout.addLayout(form_layout)

        self.project_save_button.clicked.connect(self.save_project_settings)
        self.project_apply_button.clicked.connect(self.apply_project_settings_to_tools)
        self.project_capture_button.clicked.connect(self.capture_current_project_paths)
        self.project_output_dir_button.clicked.connect(
            lambda: self.select_project_folder(self.project_output_dir_edit, "选择输出目录")
        )

    def _build_lifecycle_card(self, parent):
        self.project_lifecycle_card = QtWidgets.QFrame(parent)
        self.project_lifecycle_card.setObjectName("ProjectCard")

        card_layout = QVBoxLayout(self.project_lifecycle_card)
        card_layout.setContentsMargins(10, 8, 10, 10)
        card_layout.setSpacing(8)

        # Header: progress + recommended step
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header = QtWidgets.QLabel("生命周期进度", self.project_lifecycle_card)
        header.setObjectName("ProjectTitle")
        self.project_progress_label = QtWidgets.QLabel("", self.project_lifecycle_card)
        self.project_progress_label.setObjectName("ProjectStatus")
        header_row.addWidget(header)
        header_row.addWidget(self.project_progress_label, stretch=1)
        card_layout.addLayout(header_row)

        # Progress bar
        self.project_progress_bar = QtWidgets.QProgressBar(self.project_lifecycle_card)
        self.project_progress_bar.setMaximum(8)
        self.project_progress_bar.setMinimumHeight(24)
        self.project_progress_bar.setTextVisible(True)
        card_layout.addWidget(self.project_progress_bar)

        # Stage table
        self.project_stage_table = QtWidgets.QTableWidget(0, 3, self.project_lifecycle_card)
        self.project_stage_table.setHorizontalHeaderLabels(["阶段", "状态", "依据/操作"])
        self.project_stage_table.verticalHeader().setVisible(False)
        self.project_stage_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.project_stage_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.project_stage_table.setAlternatingRowColors(True)
        self.project_stage_table.setMinimumHeight(210)
        self.project_stage_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.project_stage_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.project_stage_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        card_layout.addWidget(self.project_stage_table)

        # Action bar: only main button (dynamic)
        action_bar = QtWidgets.QWidget(self.project_lifecycle_card)
        action_bar.setObjectName("ProjectActionBar")
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(8, 6, 8, 6)
        action_layout.setSpacing(6)

        self.project_main_action_button = QPushButton("", action_bar)
        self.project_main_action_button.setFixedHeight(28)
        self.project_main_action_button.setMinimumWidth(120)
        action_layout.addWidget(self.project_main_action_button)
        action_layout.addStretch(1)
        card_layout.addWidget(action_bar)

        # Connect main button to dynamic handler
        self.project_main_action_button.clicked.connect(self._on_project_main_action_clicked)

    def _build_artifact_manager_card(self, parent):
        self.artifact_manager_card = QtWidgets.QFrame(parent)
        self.artifact_manager_card.setObjectName("ProjectCard")

        card_layout = QVBoxLayout(self.artifact_manager_card)
        card_layout.setContentsMargins(10, 8, 10, 10)
        card_layout.setSpacing(8)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        title = QtWidgets.QLabel("项目文件树", self.artifact_manager_card)
        title.setObjectName("ProjectTitle")
        self.project_artifact_summary = QtWidgets.QLabel("", self.artifact_manager_card)
        self.project_artifact_summary.setObjectName("ProjectHint")
        self.project_artifact_summary.setWordWrap(True)
        header_row.addWidget(title)
        header_row.addWidget(self.project_artifact_summary, stretch=1)
        card_layout.addLayout(header_row)

        self.project_artifact_tree = QtWidgets.QTreeWidget(self.artifact_manager_card)
        self.project_artifact_tree.setHeaderLabels(["文件/目录", "阶段", "大小", "修改时间"])
        self.project_artifact_tree.setAlternatingRowColors(True)
        self.project_artifact_tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.project_artifact_tree.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.project_artifact_tree.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.project_artifact_tree.header().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.project_artifact_tree.header().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        card_layout.addWidget(self.project_artifact_tree, stretch=1)

        action_bar = QtWidgets.QWidget(self.artifact_manager_card)
        action_bar.setObjectName("ProjectActionBar")
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(8, 6, 8, 6)
        action_layout.setSpacing(6)
        self.project_refresh_artifacts_button = QPushButton("刷新", action_bar)
        self.project_open_selected_button = QPushButton("打开所选", action_bar)
        self.project_open_root_button = QPushButton("打开项目目录", action_bar)
        self.project_refresh_artifacts_button.setObjectName("BrowseButton")
        self.project_open_selected_button.setObjectName("BrowseButton")
        self.project_open_root_button.setObjectName("BrowseButton")
        for button in (
            self.project_refresh_artifacts_button,
            self.project_open_selected_button,
            self.project_open_root_button,
        ):
            button.setFixedHeight(28)
            action_layout.addWidget(button)
        action_layout.addStretch(1)
        card_layout.addWidget(action_bar)

        self.project_refresh_artifacts_button.clicked.connect(self.refresh_project_lifecycle)
        self.project_open_selected_button.clicked.connect(self.open_selected_project_artifact)
        self.project_open_root_button.clicked.connect(self.open_project_root_folder)

    # ── Tab 2: Data Sources ──────────────────────────────────────────────

    def _build_datasource_card(self, parent):
        self.datasource_card = QtWidgets.QFrame(parent)
        self.datasource_card.setObjectName("ProjectCard")
        self.datasource_card.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        card_layout = QVBoxLayout(self.datasource_card)
        card_layout.setContentsMargins(10, 8, 10, 10)
        card_layout.setSpacing(10)

        header = QtWidgets.QLabel("数据源路径", self.datasource_card)
        header.setObjectName("ProjectTitle")
        card_layout.addWidget(header)
        hint = QtWidgets.QLabel(
            '"读取当前路径"会从各工具页回填原始数据源；分析产物由上游工具导出后自动登记，也可在这里手动指定。',
            self.datasource_card,
        )
        hint.setObjectName("ProjectHint")
        hint.setWordWrap(True)
        card_layout.addWidget(hint)

        self.project_single_file_edit = QLineEdit(self.datasource_card)
        self.project_single_file_edit.setPlaceholderText("选择 .txt/.asc/.888 单谱文件")
        self.project_sum_folder_edit = QLineEdit(self.datasource_card)
        self.project_sum_folder_edit.setPlaceholderText("选择累计谱数据目录")
        self.project_temperature_folder_edit = QLineEdit(self.datasource_card)
        self.project_temperature_folder_edit.setPlaceholderText("选择温度扫描 txt 文件目录")
        self.project_pie_folder_edit = QLineEdit(self.datasource_card)
        self.project_pie_folder_edit.setPlaceholderText("选择 PIE 扫描目录")
        self.project_sample_info_edit = QLineEdit(self.datasource_card)
        self.project_sample_info_edit.setPlaceholderText("选择样品信息 xlsx/csv/txt 文件")
        self.project_pics_database_edit = QLineEdit(self.datasource_card)
        self.project_pics_database_edit.setPlaceholderText("选择 species_database.sqlite")
        self.project_manual_peak_edit = QLineEdit(self.datasource_card)
        self.project_manual_peak_edit.setPlaceholderText("选择 yaml/csv/xlsx 卡峰文件")
        self.project_temperature_result_edit = QLineEdit(self.datasource_card)
        self.project_temperature_result_edit.setPlaceholderText("温度扫描导出的 xlsx/csv 结果")
        self.project_pie_result_edit = QLineEdit(self.datasource_card)
        self.project_pie_result_edit.setPlaceholderText("PIE拟合导出的鉴定结果")
        self.project_mole_fraction_result_edit = QLineEdit(self.datasource_card)
        self.project_mole_fraction_result_edit.setPlaceholderText("摩尔分数导出的结果")

        def _browse_btn(text="选择"):
            b = QPushButton(text, self.datasource_card)
            b.setObjectName("BrowseButton")
            b.setFixedHeight(28)
            return b

        def _path_group(title: str):
            group = QtWidgets.QGroupBox(title, self.datasource_card)
            layout = QtWidgets.QGridLayout(group)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setHorizontalSpacing(8)
            layout.setVerticalSpacing(6)
            layout.setColumnStretch(1, 1)
            return group, layout

        def _add_path_row(layout, row: int, label: str, edit: QLineEdit, button: QPushButton):
            layout.addWidget(QtWidgets.QLabel(label, self.datasource_card), row, 0)
            layout.addWidget(edit, row, 1)
            layout.addWidget(button, row, 2)

        self.project_single_file_button = _browse_btn()
        self.project_sum_folder_button = _browse_btn()
        self.project_temperature_folder_button = _browse_btn()
        self.project_pie_folder_button = _browse_btn()
        self.project_sample_info_button = _browse_btn()
        self.project_database_button = _browse_btn()
        self.project_manual_peak_button = _browse_btn()
        self.project_temperature_result_button = _browse_btn()
        self.project_pie_result_button = _browse_btn()
        self.project_mole_fraction_result_button = _browse_btn()

        workbench_group, workbench_layout = _path_group("质谱工作台")
        _add_path_row(workbench_layout, 0, "单谱文件", self.project_single_file_edit, self.project_single_file_button)
        _add_path_row(workbench_layout, 1, "累计谱文件夹", self.project_sum_folder_edit, self.project_sum_folder_button)
        card_layout.addWidget(workbench_group)

        analysis_group, analysis_layout = _path_group("分析模块")
        _add_path_row(analysis_layout, 0, "温度扫描目录", self.project_temperature_folder_edit, self.project_temperature_folder_button)
        _add_path_row(analysis_layout, 1, "PIE扫描目录", self.project_pie_folder_edit, self.project_pie_folder_button)
        _add_path_row(analysis_layout, 2, "样品信息", self.project_sample_info_edit, self.project_sample_info_button)
        _add_path_row(analysis_layout, 3, "PICS数据库", self.project_pics_database_edit, self.project_database_button)
        _add_path_row(analysis_layout, 4, "手动卡峰文件", self.project_manual_peak_edit, self.project_manual_peak_button)
        card_layout.addWidget(analysis_group)

        artifact_group, artifact_layout = _path_group("分析产物")
        _add_path_row(artifact_layout, 0, "温度扫描结果", self.project_temperature_result_edit, self.project_temperature_result_button)
        _add_path_row(artifact_layout, 1, "PIE鉴定结果", self.project_pie_result_edit, self.project_pie_result_button)
        _add_path_row(artifact_layout, 2, "摩尔分数结果", self.project_mole_fraction_result_edit, self.project_mole_fraction_result_button)
        card_layout.addWidget(artifact_group)

        self.project_single_file_button.clicked.connect(self.select_project_single_file)
        self.project_sum_folder_button.clicked.connect(
            lambda: self.select_project_folder(self.project_sum_folder_edit, "选择累计谱文件夹")
        )
        self.project_temperature_folder_button.clicked.connect(
            lambda: self.select_project_folder(self.project_temperature_folder_edit, "选择温度扫描目录")
        )
        self.project_pie_folder_button.clicked.connect(
            lambda: self.select_project_folder(self.project_pie_folder_edit, "选择PIE扫描目录")
        )
        self.project_sample_info_button.clicked.connect(self.select_project_sample_info)
        self.project_database_button.clicked.connect(self.select_project_database)
        self.project_manual_peak_button.clicked.connect(self.select_project_manual_peak)
        self.project_temperature_result_button.clicked.connect(
            lambda: self.select_project_result_file(
                self.project_temperature_result_edit,
                "选择温度扫描结果文件",
                "Result Files (*.xlsx *.csv);;所有文件 (*)",
            )
        )
        self.project_pie_result_button.clicked.connect(
            lambda: self.select_project_result_file(
                self.project_pie_result_edit,
                "选择PIE鉴定结果文件",
                "Result Files (*.xlsx *.csv);;所有文件 (*)",
            )
        )
        self.project_mole_fraction_result_button.clicked.connect(
            lambda: self.select_project_result_file(
                self.project_mole_fraction_result_edit,
                "选择摩尔分数结果文件",
                "Result Files (*.xlsx *.csv);;所有文件 (*)",
            )
        )

        # Auto-save and push project paths when edited
        self.project_single_file_edit.editingFinished.connect(self._auto_save_datasource)
        self.project_sum_folder_edit.editingFinished.connect(self._auto_save_datasource)
        self.project_temperature_folder_edit.editingFinished.connect(self._auto_save_datasource)
        self.project_pie_folder_edit.editingFinished.connect(self._auto_save_datasource)
        self.project_sample_info_edit.editingFinished.connect(self._auto_save_datasource)
        self.project_pics_database_edit.editingFinished.connect(self._auto_save_datasource)
        self.project_manual_peak_edit.editingFinished.connect(self._auto_save_datasource)
        self.project_temperature_result_edit.editingFinished.connect(self._auto_save_datasource)
        self.project_pie_result_edit.editingFinished.connect(self._auto_save_datasource)
        self.project_mole_fraction_result_edit.editingFinished.connect(self._auto_save_datasource)

    # ── Tab 4: Function Params ───────────────────────────────────────────

    def _build_function_params_card(self, parent):
        self.function_params_card = QtWidgets.QFrame(parent)
        self.function_params_card.setObjectName("ProjectCard")

        card_layout = QVBoxLayout(self.function_params_card)
        card_layout.setContentsMargins(10, 8, 10, 10)
        card_layout.setSpacing(8)

        params_grid = QtWidgets.QGridLayout()
        params_grid.setContentsMargins(0, 0, 0, 0)
        params_grid.setHorizontalSpacing(10)
        params_grid.setVerticalSpacing(8)
        params_grid.setColumnStretch(0, 1)
        params_grid.setColumnStretch(1, 1)

        # -- PIE defaults --
        pie_group = QtWidgets.QGroupBox("PIE拟合默认值", self.function_params_card)
        pie_layout = QtWidgets.QGridLayout(pie_group)
        pie_layout.setHorizontalSpacing(8)
        pie_layout.setVerticalSpacing(6)

        self.fp_pie_energy_decimals = QtWidgets.QSpinBox()
        self.fp_pie_energy_decimals.setToolTip("光子能量保留的小数位数。值越大能量分组越细，典型值 1-2")
        self.fp_pie_energy_decimals.setRange(0, 6)
        self.fp_pie_energy_decimals.setValue(1)
        self.fp_pie_recursive = QtWidgets.QCheckBox("递归")
        self.fp_pie_recursive.setToolTip("启用后先用高能段数据拟合，再逐步扩展到低能区，提高低信号区拟合稳定性")
        self.fp_pie_prefer_gaussian = QtWidgets.QCheckBox("高斯积分")
        self.fp_pie_prefer_gaussian.setToolTip("使用高斯峰面积而非简单峰值强度作为信号量，更准确反映积分强度")
        self.fp_pie_prefer_gaussian.setChecked(True)
        self.fp_pie_multi_folder = QtWidgets.QCheckBox("多文件夹模式")
        self.fp_pie_multi_folder.setToolTip("启用后可从多个独立PIE扫描文件夹合并数据，用于不同能段的拼接实验")
        self.fp_pie_merge_method = QtWidgets.QComboBox()
        self.fp_pie_merge_method.setToolTip("低能段为主：以低能段信号为基准缩放其他段；简单拼接：直接按能量排序不缩放")
        self.fp_pie_merge_method.addItem("低能段为主", "low_energy_dominant")
        self.fp_pie_merge_method.addItem("第一组为主", "first_segment_dominant")
        self.fp_pie_merge_method.addItem("简单拼接", "mean")

        pie_layout.addWidget(QtWidgets.QLabel("能量小数位"), 0, 0)
        pie_layout.addWidget(self.fp_pie_energy_decimals, 0, 1)
        pie_layout.addWidget(self.fp_pie_recursive, 0, 2)
        pie_layout.addWidget(self.fp_pie_prefer_gaussian, 0, 3)
        pie_layout.addWidget(QtWidgets.QLabel("合并方法"), 1, 0)
        pie_layout.addWidget(self.fp_pie_merge_method, 1, 1)
        pie_layout.addWidget(self.fp_pie_multi_folder, 1, 2)
        params_grid.addWidget(pie_group, 0, 0)

        # -- Temperature Scan defaults --
        temp_group = QtWidgets.QGroupBox("温度扫描默认值", self.function_params_card)
        temp_layout = QtWidgets.QGridLayout(temp_group)
        temp_layout.setHorizontalSpacing(8)
        temp_layout.setVerticalSpacing(6)

        self.fp_temp_reference_mode = QtWidgets.QComboBox()
        self.fp_temp_reference_mode.setToolTip("Sum谱参考：所有温度累加后统一寻峰；独立参考：每个温度点独立寻峰")
        self.fp_temp_reference_mode.addItem("Sum谱参考", "sum")
        self.fp_temp_reference_mode.addItem("独立参考", "individual")
        self.fp_temp_prefer_gaussian = QtWidgets.QCheckBox("高斯积分")
        self.fp_temp_prefer_gaussian.setToolTip("使用高斯峰面积而非简单峰值强度，更准确反映积分强度")
        self.fp_temp_prefer_gaussian.setChecked(True)
        self.fp_temp_kr_mz = QtWidgets.QSpinBox()
        self.fp_temp_kr_mz.setToolTip("Kr（氪）定标物种的质量数，默认84，用于计算膨胀系数 λ(T)")
        self.fp_temp_kr_mz.setRange(1, 1000)
        self.fp_temp_kr_mz.setValue(84)

        temp_layout.addWidget(QtWidgets.QLabel("参考模式"), 0, 0)
        temp_layout.addWidget(self.fp_temp_reference_mode, 0, 1)
        temp_layout.addWidget(self.fp_temp_prefer_gaussian, 0, 2)
        temp_layout.addWidget(QtWidgets.QLabel("Kr 参考 m/z"), 1, 0)
        temp_layout.addWidget(self.fp_temp_kr_mz, 1, 1)
        params_grid.addWidget(temp_group, 0, 1)

        # -- PICS defaults --
        pics_group = QtWidgets.QGroupBox("PICS计算默认值", self.function_params_card)
        pics_layout = QtWidgets.QGridLayout(pics_group)
        pics_layout.setHorizontalSpacing(8)
        pics_layout.setVerticalSpacing(6)

        self.fp_pics_no_mz = QtWidgets.QSpinBox()
        self.fp_pics_no_mz.setToolTip("参考物种NO的质量数，用于PICS计算中信号比的分母")
        self.fp_pics_no_mz.setRange(1, 1000)
        self.fp_pics_no_mz.setValue(30)
        self.fp_pics_no_formula = QLineEdit()
        self.fp_pics_no_formula.setToolTip("参考物种NO的分子式，用于从数据库自动匹配光电离截面")
        self.fp_pics_no_formula.setText("NO")
        self.fp_pics_no_mf = QtWidgets.QDoubleSpinBox()
        self.fp_pics_no_mf.setToolTip("NO在反应器中的输入摩尔分数（已知量）")
        self.fp_pics_no_mf.setRange(0, 1)
        self.fp_pics_no_mf.setDecimals(6)
        self.fp_pics_no_mf.setValue(0.01)
        self.fp_pics_new_species_mf = QtWidgets.QDoubleSpinBox()
        self.fp_pics_new_species_mf.setToolTip("高于此摩尔分数的物种将被认定为'已观测到的新物种'")
        self.fp_pics_new_species_mf.setRange(0, 1)
        self.fp_pics_new_species_mf.setDecimals(6)
        self.fp_pics_new_species_mf.setValue(0.002)

        pics_layout.addWidget(QtWidgets.QLabel("NO m/z"), 0, 0)
        pics_layout.addWidget(self.fp_pics_no_mz, 0, 1)
        pics_layout.addWidget(QtWidgets.QLabel("NO分子式"), 0, 2)
        pics_layout.addWidget(self.fp_pics_no_formula, 0, 3)
        pics_layout.addWidget(QtWidgets.QLabel("NO摩尔分数"), 1, 0)
        pics_layout.addWidget(self.fp_pics_no_mf, 1, 1)
        pics_layout.addWidget(QtWidgets.QLabel("新物种阈值"), 1, 2)
        pics_layout.addWidget(self.fp_pics_new_species_mf, 1, 3)
        params_grid.addWidget(pics_group, 1, 0)

        # -- Mole Fraction defaults --
        mf_group = QtWidgets.QGroupBox("摩尔分数默认值", self.function_params_card)
        mf_layout = QtWidgets.QGridLayout(mf_group)
        mf_layout.setHorizontalSpacing(8)
        mf_layout.setVerticalSpacing(6)

        self.fp_mf_mass_disc_exponent = QtWidgets.QDoubleSpinBox()
        self.fp_mf_mass_disc_exponent.setToolTip("质量歧视因子公式 D_i = (MW/30)^n 中的指数n，值取决于离子源类型和质量分析器特性")
        self.fp_mf_mass_disc_exponent.setRange(0, 10)
        self.fp_mf_mass_disc_exponent.setDecimals(6)
        self.fp_mf_mass_disc_exponent.setValue(0.77897)
        self.fp_mf_parent_mz = QtWidgets.QSpinBox()
        self.fp_mf_parent_mz.setToolTip("母体物种（反应物）的质量数")
        self.fp_mf_parent_mz.setRange(1, 1000)
        self.fp_mf_parent_mz.setValue(128)
        self.fp_mf_parent_initial_mf = QtWidgets.QDoubleSpinBox()
        self.fp_mf_parent_initial_mf.setToolTip("母体物种在反应器入口的摩尔分数（已知或假设值）")
        self.fp_mf_parent_initial_mf.setRange(0, 1)
        self.fp_mf_parent_initial_mf.setDecimals(6)
        self.fp_mf_parent_initial_mf.setValue(0.002)
        self.fp_mf_photon_energy = QtWidgets.QDoubleSpinBox()
        self.fp_mf_photon_energy.setToolTip("实验使用的VUV光子能量 (eV)")
        self.fp_mf_photon_energy.setRange(0, 100)
        self.fp_mf_photon_energy.setDecimals(4)
        self.fp_mf_photon_energy.setValue(10.0)

        mf_layout.addWidget(QtWidgets.QLabel("质量歧视指数"), 0, 0)
        mf_layout.addWidget(self.fp_mf_mass_disc_exponent, 0, 1)
        mf_layout.addWidget(QtWidgets.QLabel("母体 m/z"), 0, 2)
        mf_layout.addWidget(self.fp_mf_parent_mz, 0, 3)
        mf_layout.addWidget(QtWidgets.QLabel("母体初始摩尔分数"), 1, 0)
        mf_layout.addWidget(self.fp_mf_parent_initial_mf, 1, 1)
        mf_layout.addWidget(QtWidgets.QLabel("光子能量 (eV)"), 1, 2)
        mf_layout.addWidget(self.fp_mf_photon_energy, 1, 3)
        params_grid.addWidget(mf_group, 1, 1)
        card_layout.addLayout(params_grid)

        # Save button for function params
        fp_action_bar = QtWidgets.QWidget(self.function_params_card)
        fp_action_bar.setObjectName("ProjectActionBar")
        fp_btn_row = QHBoxLayout(fp_action_bar)
        fp_btn_row.setContentsMargins(8, 6, 8, 6)
        fp_btn_row.setSpacing(8)
        fp_hint = QtWidgets.QLabel("保存后写入项目配置，并在应用或切换工具页时同步为默认参数。", fp_action_bar)
        fp_hint.setObjectName("ProjectHint")
        fp_hint.setWordWrap(True)
        fp_btn_row.addWidget(fp_hint, stretch=1)
        self.fp_save_button = QPushButton("保存功能参数", fp_action_bar)
        self.fp_save_button.clicked.connect(self.save_function_params)
        fp_btn_row.addWidget(self.fp_save_button)
        card_layout.addWidget(fp_action_bar)

    def _route_common_parameter_buttons(self) -> None:
        for page in (self.temperature_page, self.pie_page):
            button = getattr(page, "common_params_button", None)
            if button is None:
                continue
            try:
                button.clicked.disconnect()
            except (TypeError, RuntimeError):
                pass
            button.clicked.connect(self.open_common_parameters)

    # ── Settings manager helper ───────────────────────────────────────

    @property
    def project_settings_manager(self) -> ProjectSettingsManager:
        return ProjectSettingsManager()

    def _read_project_settings_to_ui(self, ps: ProjectSettings) -> None:
        """Populate all project page fields from a ProjectSettings instance."""
        self.project_name_edit.setText(ps.project_name)
        self.project_system_edit.setText(ps.system)
        self.project_description_edit.setText(ps.description)
        self.project_output_dir_edit.setText(ps.output_dir)
        self.project_single_file_edit.setText(ps.single_spectrum_file)
        self.project_sum_folder_edit.setText(ps.sum_spectrum_folder)
        self.project_temperature_folder_edit.setText(ps.temperature_scan_folder)
        self.project_pie_folder_edit.setText(ps.pie_scan_folder)
        self.project_sample_info_edit.setText(ps.sample_info_file)
        self.project_pics_database_edit.setText(ps.pics_database_path)
        self.project_manual_peak_edit.setText(ps.manual_peak_file)
        self.project_temperature_result_edit.setText(ps.temperature_scan_result_file)
        self.project_pie_result_edit.setText(ps.pie_identification_result_file)
        self.project_mole_fraction_result_edit.setText(ps.mole_fraction_result_file)

    def _collect_project_settings_from_ui(self) -> ProjectSettings:
        """Build a ProjectSettings from all UI fields (does not save)."""
        ps = self.project_settings_manager.get()
        ps.project_name = self.project_name_edit.text().strip()
        ps.system = self.project_system_edit.text().strip()
        ps.description = self.project_description_edit.text().strip()
        ps.output_dir = self.project_output_dir_edit.text().strip() or "output"
        ps.single_spectrum_file = self.project_single_file_edit.text().strip()
        ps.sum_spectrum_folder = self.project_sum_folder_edit.text().strip()
        ps.temperature_scan_folder = self.project_temperature_folder_edit.text().strip()
        ps.pie_scan_folder = self.project_pie_folder_edit.text().strip()
        ps.sample_info_file = self.project_sample_info_edit.text().strip()
        ps.pics_database_path = self.project_pics_database_edit.text().strip()
        ps.manual_peak_file = self.project_manual_peak_edit.text().strip()
        ps.temperature_scan_result_file = self.project_temperature_result_edit.text().strip()
        ps.pie_identification_result_file = self.project_pie_result_edit.text().strip()
        ps.mole_fraction_result_file = self.project_mole_fraction_result_edit.text().strip()
        return ps

    def load_project_settings(self) -> None:
        ps = self.project_settings_manager.get()
        default_pie_folder = "tests/fixtures/bl03u_sample/C6F11O2H/PIE_Scan/1050"
        default_database = str(species_database_path())
        if not ps.pie_scan_folder:
            ps.pie_scan_folder = default_pie_folder
        if not ps.pics_database_path:
            ps.pics_database_path = default_database
        self._read_project_settings_to_ui(ps)
        self._load_function_params_to_ui(ps)
        self.update_project_title()
        self.refresh_project_lifecycle()
        self.update_project_ui_state(ps)

    def _load_function_params_to_ui(self, ps: ProjectSettings) -> None:
        """Populate function params tab from ProjectSettings."""
        self.fp_pie_energy_decimals.setValue(ps.pie_energy_decimals)
        self.fp_pie_recursive.setChecked(ps.pie_recursive)
        self.fp_pie_prefer_gaussian.setChecked(ps.pie_prefer_gaussian)
        self.fp_pie_multi_folder.setChecked(ps.pie_multi_folder_mode)
        idx = self.fp_pie_merge_method.findData(ps.pie_merge_method)
        if idx >= 0:
            self.fp_pie_merge_method.setCurrentIndex(idx)
        idx = self.fp_temp_reference_mode.findData(ps.temp_reference_mode)
        if idx >= 0:
            self.fp_temp_reference_mode.setCurrentIndex(idx)
        self.fp_temp_prefer_gaussian.setChecked(ps.temp_prefer_gaussian)
        self.fp_temp_kr_mz.setValue(ps.temp_kr_mz)
        self.fp_pics_no_mz.setValue(ps.pics_no_mz)
        self.fp_pics_no_formula.setText(ps.pics_no_formula)
        self.fp_pics_no_mf.setValue(ps.pics_no_mf)
        self.fp_pics_new_species_mf.setValue(ps.pics_new_species_mf)
        self.fp_mf_mass_disc_exponent.setValue(ps.mf_mass_disc_exponent)
        self.fp_mf_parent_mz.setValue(ps.mf_parent_mz)
        self.fp_mf_parent_initial_mf.setValue(ps.mf_parent_initial_mf)
        self.fp_mf_photon_energy.setValue(ps.mf_photon_energy)

    def update_project_ui_state(self, ps: ProjectSettings | None = None) -> None:
        """Update main action button and progress bar based on project state"""
        if ps is None:
            ps = self.project_settings_manager.get()

        state = get_project_ui_state(ps)

        # Update main action button
        if hasattr(self, "project_main_action_button"):
            self.project_main_action_button.setText(state.button_text)
            self.project_main_action_button.setToolTip(state.description)

        # Update progress label and bar
        statuses = build_project_stage_statuses(ps)
        completed_count = sum(1 for s in statuses if s.completed)
        total_count = len(statuses)

        if hasattr(self, "project_progress_label"):
            self.project_progress_label.setText(f"已完成 {completed_count}/{total_count} 阶段")

        if hasattr(self, "project_progress_bar"):
            self.project_progress_bar.setValue(completed_count)

    def _on_project_main_action_clicked(self) -> None:
        """Handle main action button click based on current project state"""
        ps = self.project_settings_manager.get()
        state = get_project_ui_state(ps)

        if state == ProjectUIState.UNINITIALIZED:
            self.initialize_project_structure()
        elif state == ProjectUIState.AWAITING_DATA_IMPORT:
            self.project_tabs.setCurrentWidget(self.project_datasource_page)
        elif state == ProjectUIState.READY_TO_ANALYZE:
            self.switch_workspace_page("spectrum")
        elif state == ProjectUIState.ANALYSIS_IN_PROGRESS:
            # Show next recommended step
            next_status = next_project_stage(ps)
            if next_status:
                self.switch_workspace_page(next_status.nav_page)
        elif state == ProjectUIState.ANALYSIS_COMPLETE:
            self.project_tabs.setCurrentWidget(self.project_artifacts_page)
        """Write function params tab values into ProjectSettings."""
        ps.pie_energy_decimals = self.fp_pie_energy_decimals.value()
        ps.pie_recursive = self.fp_pie_recursive.isChecked()
        ps.pie_prefer_gaussian = self.fp_pie_prefer_gaussian.isChecked()
        ps.pie_multi_folder_mode = self.fp_pie_multi_folder.isChecked()
        ps.pie_merge_method = self.fp_pie_merge_method.currentData()
        ps.temp_reference_mode = self.fp_temp_reference_mode.currentData()
        ps.temp_prefer_gaussian = self.fp_temp_prefer_gaussian.isChecked()
        ps.temp_kr_mz = self.fp_temp_kr_mz.value()
        ps.pics_no_mz = self.fp_pics_no_mz.value()
        ps.pics_no_formula = self.fp_pics_no_formula.text().strip()
        ps.pics_no_mf = self.fp_pics_no_mf.value()
        ps.pics_new_species_mf = self.fp_pics_new_species_mf.value()
        ps.mf_mass_disc_exponent = self.fp_mf_mass_disc_exponent.value()
        ps.mf_parent_mz = self.fp_mf_parent_mz.value()
        ps.mf_parent_initial_mf = self.fp_mf_parent_initial_mf.value()
        ps.mf_photon_energy = self.fp_mf_photon_energy.value()

    def save_project_settings(self) -> None:
        ps = self._collect_project_settings_from_ui()
        self._collect_function_params_from_ui(ps)
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()
        self.update_project_title()
        self.refresh_project_lifecycle(ps)
        self.refresh_project_parameter_summary()
        self.update_project_ui_state(ps)
        self.statusbar.showMessage("项目设置已保存", 3000)

    def save_function_params(self) -> None:
        ps = self._collect_project_settings_from_ui()
        self._collect_function_params_from_ui(ps)
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()
        self._sync_project_settings_to_tool_pages(ps)
        self.update_project_title()
        self.refresh_project_lifecycle(ps)
        self.refresh_project_parameter_summary()
        self.statusbar.showMessage("功能参数已保存，项目摘要已更新", 3000)

    def _sync_project_settings_to_tool_pages(self, ps: ProjectSettings) -> None:
        calibration = self.current_calibration()
        if hasattr(self, "temperature_page"):
            self.temperature_page.normalization_settings = self.normalization_settings
            self.temperature_page.calibration = calibration
            self.temperature_page.set_project_settings(ps)
        if hasattr(self, "pie_page"):
            self.pie_page.normalization_settings = self.normalization_settings
            self.pie_page.calibration = calibration
            self.pie_page.set_project_settings(ps)
        if hasattr(self, "mole_fraction_page"):
            self.mole_fraction_page.normalization_settings = self.normalization_settings
            self.mole_fraction_page.calibration = calibration
            self.mole_fraction_page.set_project_settings(ps)
        if hasattr(self, "pics_page"):
            self.pics_page.normalization_settings = self.normalization_settings
            self.pics_page.calibration = calibration
            self.pics_page.set_project_settings(ps)

    def apply_project_settings_to_tools(self) -> None:
        ps = self._collect_project_settings_from_ui()
        self._collect_function_params_from_ui(ps)
        if ps.single_spectrum_file:
            self.lineEdit.setText(ps.single_spectrum_file)
        if ps.sum_spectrum_folder:
            self.folder_path.setText(ps.sum_spectrum_folder)
        # Temperature page and PIE page are now read-only parameter displays
        # No need to manually set folder_edit - parameters come from ProjectSettings
        if hasattr(self, "temperature_page"):
            self.temperature_page.set_project_settings(ps)
        if hasattr(self, "pie_page"):
            self.pie_page.set_project_settings(ps)
            if ps.pics_database_path and os.path.exists(ps.pics_database_path):
                self.pie_page.load_database(show_message=False)
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()
        self.update_project_title()
        self._sync_project_settings_to_tool_pages(ps)
        self.refresh_project_lifecycle(ps)
        self.refresh_project_parameter_summary()
        self.statusbar.showMessage("项目路径和默认参数已应用到当前工具并保存", 3000)

    def capture_current_project_paths(self) -> None:
        """Capture paths from editable tool pages (Spectrum).
        Temperature and PIE pages are read-only, so no capture needed."""
        self.project_single_file_edit.setText(self.lineEdit.text().strip())
        self.project_sum_folder_edit.setText(self.folder_path.text().strip())
        self.refresh_project_lifecycle()
        self.statusbar.showMessage('已从工具页面回填路径；请点击"保存项目"写入配置', 4000)

    def _auto_save_datasource(self) -> None:
        """Auto-save data source settings and sync to tool pages."""
        ps = self._collect_project_settings_from_ui()
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()
        # Push editable paths to Spectrum tool page
        val = self.project_single_file_edit.text().strip()
        if val:
            self.lineEdit.setText(val)
        val = self.project_sum_folder_edit.text().strip()
        if val:
            self.folder_path.setText(val)
        # Sync project settings to all tool pages (Temperature, PIE, etc.)
        self._sync_project_settings_to_tool_pages(ps)
        self.refresh_project_lifecycle(ps)

    def _push_path_to_tool(self, kind: str) -> None:
        """Push a single path field from project management to the corresponding editable tool page.
        Note: Temperature and PIE pages are read-only, so only Spectrum tool is updated."""
        if kind == "single":
            val = self.project_single_file_edit.text().strip()
            if val:
                self.lineEdit.setText(val)
        elif kind == "sum":
            val = self.project_sum_folder_edit.text().strip()
            if val:
                self.folder_path.setText(val)

    def _pull_path_from_tool(self, kind: str) -> None:
        """Pull a single path from an editable tool page back to project management.
        Note: Temperature and PIE pages are read-only, so only Spectrum tool is pulled."""
        if kind == "single":
            self.project_single_file_edit.setText(self.lineEdit.text().strip())
        elif kind == "sum":
            self.project_sum_folder_edit.setText(self.folder_path.text().strip())

    def select_project_single_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择单谱文件",
            self._dialog_start_dir(self.project_single_file_edit.text()),
            "质谱数据 (*.txt *.asc *.888);;所有文件 (*)",
        )
        if path:
            self.project_single_file_edit.setText(path)

    def select_project_folder(self, target: QLineEdit, title: str) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            title,
            self._dialog_start_dir(target.text()),
        )
        if folder:
            target.setText(folder)

    def select_project_database(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择PICS数据库",
            self._dialog_start_dir(self.project_pics_database_edit.text()),
            "SQLite Files (*.sqlite *.sqlite3 *.db);;所有文件 (*)",
        )
        if path:
            self.project_pics_database_edit.setText(path)

    def select_project_manual_peak(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择手动卡峰文件",
            self._dialog_start_dir(self.project_manual_peak_edit.text()),
            "Peak Files (*.yaml *.yml *.csv *.xlsx *.xls);;所有文件 (*)",
        )
        if path:
            self.project_manual_peak_edit.setText(path)

    def select_project_sample_info(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择样品信息文件",
            self._dialog_start_dir(self.project_sample_info_edit.text()),
            "样品信息 (*.xlsx *.xls *.csv *.tsv *.txt);;所有文件 (*)",
        )
        if path:
            self.project_sample_info_edit.setText(path)

    def select_project_result_file(self, target: QLineEdit, title: str, file_filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            self._dialog_start_dir(target.text()),
            file_filter,
        )
        if path:
            target.setText(path)

    # ── Lifecycle and artifact management ──────────────────────────────

    def _default_project_folder(self, ps: ProjectSettings) -> str:
        slug = sanitize_project_slug(ps.project_name or ps.system or "untitled")
        folder_name = slug if slug.lower().startswith("project") else f"Project_{slug}"
        return str(Path("output") / folder_name)

    def _collect_and_save_project_settings(self) -> ProjectSettings:
        ps = self._collect_project_settings_from_ui()
        self._collect_function_params_from_ui(ps)
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()
        return ps

    def initialize_project_structure(self) -> None:
        ps = self._collect_project_settings_from_ui()
        if (not ps.output_dir.strip() or ps.output_dir.strip() == "output") and (ps.project_name or ps.system):
            self.project_output_dir_edit.setText(self._default_project_folder(ps))
            ps = self._collect_project_settings_from_ui()
        self._collect_function_params_from_ui(ps)
        ensure_project_structure(ps)
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()
        self._read_project_settings_to_ui(ps)
        self._sync_project_settings_to_tool_pages(ps)
        self.refresh_project_lifecycle(ps)
        self.refresh_project_parameter_summary()
        self.update_project_ui_state(ps)
        self.statusbar.showMessage(f"项目目录已初始化：{project_root(ps)}", 5000)

    def open_project_import_wizard(self) -> None:
        ps = self._collect_project_settings_from_ui()
        if (not ps.output_dir.strip() or ps.output_dir.strip() == "output") and (ps.project_name or ps.system):
            self.project_output_dir_edit.setText(self._default_project_folder(ps))
            ps = self._collect_project_settings_from_ui()
        self._collect_function_params_from_ui(ps)
        ensure_project_structure(ps)
        dialog = ProjectImportDialog(ps, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        sources = dialog.selected_sources()
        if not sources:
            QtWidgets.QMessageBox.information(self, "导入向导", "未选择需要导入的文件或目录。")
            return
        try:
            results = import_initial_project_data(ps, sources)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "导入失败", str(exc))
            return
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()
        self._read_project_settings_to_ui(ps)
        self._sync_project_settings_to_tool_pages(ps)
        self.refresh_project_lifecycle(ps)
        self.refresh_project_parameter_summary()
        self.update_project_ui_state(ps)
        imported = "、".join(result.label for result in results)
        self.statusbar.showMessage(f"已导入：{imported}", 5000)

    def start_new_project_analysis(self) -> None:
        self.apply_project_settings_to_tools()
        self.switch_workspace_page("spectrum")
        self.statusbar.showMessage("已切换到质谱工作台，可开始新分析", 4000)

    def continue_next_project_stage(self) -> None:
        ps = self._collect_and_save_project_settings()
        status = next_project_stage(ps)
        if status is None:
            self.switch_workspace_page("project")
            if hasattr(self, "project_tabs"):
                self.project_tabs.setCurrentWidget(self.project_artifacts_page)
            self.statusbar.showMessage("全部阶段均已有记录，可检查产物或导出项目", 4000)
            return
        self.switch_workspace_page(status.nav_page)
        if status.key == "raw_data" and hasattr(self, "project_tabs"):
            self.project_tabs.setCurrentWidget(self.project_datasource_page)
        elif status.key in {"project_setup", "final_report"} and hasattr(self, "project_tabs"):
            target = self.project_identity_page if status.key == "project_setup" else self.project_artifacts_page
            self.project_tabs.setCurrentWidget(target)
        self.statusbar.showMessage(status.next_action, 5000)

    def create_project_version_snapshot(self) -> None:
        ps = self._collect_and_save_project_settings()
        try:
            snapshot_path = create_project_snapshot(ps, ps.project_name or ps.system or "snapshot")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "创建快照失败", str(exc))
            return
        self.refresh_project_lifecycle(ps)
        self.statusbar.showMessage(f"项目快照已创建：{snapshot_path}", 5000)

    def export_current_project(self) -> None:
        ps = self._collect_and_save_project_settings()
        default_name = f"{sanitize_project_slug(ps.project_name or ps.system or 'project')}_backup.zip"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "导出项目备份",
            str(project_root(ps).parent / default_name),
            "Zip Archive (*.zip)",
        )
        if not path:
            return
        try:
            archive_path = export_project_archive(ps, path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "导出失败", str(exc))
            return
        self.refresh_project_lifecycle(ps)
        self.statusbar.showMessage(f"项目已导出：{archive_path}", 5000)

    def refresh_project_lifecycle(self, ps: ProjectSettings | None = None) -> None:
        if ps is None:
            ps = self.project_settings_manager.get()
        if hasattr(self, "project_stage_table"):
            statuses = build_project_stage_statuses(ps)
            self.project_stage_table.setRowCount(len(statuses))
            for row, status in enumerate(statuses):
                state_text = "需检查" if status.warning else ("完成" if status.completed else "待处理")
                detail = status.detail
                for col, text in enumerate((status.label, state_text, detail)):
                    item = QtWidgets.QTableWidgetItem(text)
                    if col == 1:
                        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                    self.project_stage_table.setItem(row, col, item)
        if hasattr(self, "project_artifact_tree"):
            self._refresh_project_artifacts_tree(ps)

    def _refresh_project_artifacts_tree(self, ps: ProjectSettings) -> None:
        self.project_artifact_tree.clear()
        root_path = project_root(ps)
        root_item = QtWidgets.QTreeWidgetItem([root_path.name, "项目根目录", "", str(root_path)])
        root_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, str(root_path))
        self.project_artifact_tree.addTopLevelItem(root_item)

        records = collect_project_files(ps)
        records_by_section: dict[str, list] = {}
        for record in records:
            records_by_section.setdefault(record.section_key, []).append(record)
        for spec in PROJECT_DIRECTORIES:
            section_path = root_path / spec.relative_path
            section_records = records_by_section.get(spec.key, [])
            section_item = QtWidgets.QTreeWidgetItem(
                [f"{spec.label} ({len(section_records)})", spec.relative_path, "", spec.purpose]
            )
            section_item.setData(0, QtCore.Qt.ItemDataRole.UserRole, str(section_path))
            root_item.addChild(section_item)
            for record in section_records:
                name = record.relative_path
                if record.relative_path.startswith(f"{spec.relative_path}/"):
                    name = record.relative_path[len(spec.relative_path) + 1 :]
                if record.missing:
                    name = f"{name} (缺失)"
                elif record.registered:
                    name = f"{name} [已登记]"
                item = QtWidgets.QTreeWidgetItem(
                    [name, record.section_label, self._format_file_size(record.size_bytes), record.modified_at]
                )
                item.setToolTip(0, str(record.path))
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, str(record.path))
                section_item.addChild(item)
        root_item.setExpanded(True)
        for index in range(root_item.childCount()):
            root_item.child(index).setExpanded(True)
        total_size = sum(record.size_bytes for record in records if not record.missing)
        self.project_artifact_summary.setText(f"{len(records)} 个登记/扫描到的项目条目，合计 {self._format_file_size(total_size)}")

    def _format_file_size(self, size_bytes: int) -> str:
        size = float(size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
            size /= 1024
        return f"{size_bytes} B"

    def open_selected_project_artifact(self) -> None:
        item = self.project_artifact_tree.currentItem() if hasattr(self, "project_artifact_tree") else None
        if item is None:
            return
        path_text = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not path_text:
            return
        path = Path(path_text)
        if not path.exists():
            QtWidgets.QMessageBox.warning(self, "文件不存在", str(path))
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def open_project_root_folder(self) -> None:
        ps = self.project_settings_manager.get()
        root = project_root(ps)
        root.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(root)))

    def update_project_title(self) -> None:
        project_name = self.project_name_edit.text().strip()
        project_system = self.project_system_edit.text().strip()
        caption = project_name or project_system or "未命名项目"
        self.project_status_label.setText(f"当前项目: {caption}")
        window_title = "BL03U_MassSpectrumTool"
        if project_name:
            window_title = f"{window_title} - {project_name}"
        self.setWindowTitle(window_title)

    def refresh_project_parameter_summary(self) -> None:
        # 参数摘要已被移除，改为卡片化设计（第5步实现）
        # 此函数保留以保持向后兼容
        if not hasattr(self, "project_param_summary"):
            return
        try:
            ps = self.project_settings_manager.get()
            calibration = ps.to_calibration()
            light_map = {"io": "IO光电流", "beam_current": "Beam Current"}
            pie_map = {"first": "首点归一", "none": "逐点除光强", "off": "关闭"}
            peak_map = {"ensemble": "Ensemble融合检测", "prominence": "Prominence", "legacy": "传统局部极大", "cwt": "CWT小波"}
            temp_map = {"sum": "Sum谱参考", "individual": "独立参考"}
            merge_map = {
                "low_energy_dominant": "低能段为主",
                "first_segment_dominant": "第一组为主",
                "mean": "简单拼接",
            }
            next_status = next_project_stage(ps)
            next_step = next_status.next_action if next_status is not None else "检查产物并导出项目备份"
            summary = (
                f"下一步: {next_step}\n"
                "统一参数:\n"
                f"定标 A={calibration.a:.6g}, B={calibration.b:.6g}, C={calibration.c:.6g}; "
                f"光强来源={light_map.get(ps.light_source, ps.light_source)}; "
                f"温度光强归一化={'开' if ps.temperature_photon_normalize else '关'}; "
                f"PIE光强={pie_map.get(ps.pie_photon_mode, ps.pie_photon_mode)}; "
                f"质量歧视D={ps.mass_discrimination:.6g}; "
                f"主工作台寻峰={peak_map.get(ps.peak_algorithm, ps.peak_algorithm)}\n"
                "功能默认:\n"
                f"PIE 能量小数位={ps.pie_energy_decimals}, "
                f"递归={'开' if ps.pie_recursive else '关'}, "
                f"合并={merge_map.get(ps.pie_merge_method, ps.pie_merge_method)}; "
                f"温度参考={temp_map.get(ps.temp_reference_mode, ps.temp_reference_mode)}, "
                f"Kr m/z={ps.temp_kr_mz}; "
                f"PICS NO m/z={ps.pics_no_mz}; "
                f"母体 m/z={ps.mf_parent_mz}, 光子能量={ps.mf_photon_energy:.4g} eV\n"
                "分析产物:\n"
                f"温度扫描结果={'已登记' if ps.temperature_scan_result_file else '未登记'}; "
                f"PIE鉴定结果={'已登记' if ps.pie_identification_result_file else '未登记'}; "
                f"摩尔分数结果={'已登记' if ps.mole_fraction_result_file else '未登记'}"
            )
        except Exception as exc:
            summary = f"统一参数摘要读取失败: {exc}"
        self.project_param_summary.setText(summary)

    def on_project_common_parameters_saved(self) -> None:
        self.normalization_settings = load_normalization_settings()
        self.apply_config_defaults()
        calibration = self.current_calibration()
        ps = self.project_settings_manager.get()
        if hasattr(self, "temperature_page"):
            self.temperature_page.normalization_settings = self.normalization_settings
            self.temperature_page.calibration = calibration
            self.temperature_page.set_project_settings(ps)
        if hasattr(self, "pie_page"):
            self.pie_page.normalization_settings = self.normalization_settings
            self.pie_page.calibration = calibration
            self.pie_page.set_project_settings(ps)
        if hasattr(self, "mole_fraction_page"):
            self.mole_fraction_page.normalization_settings = self.normalization_settings
            self.mole_fraction_page.calibration = calibration
            self.mole_fraction_page.set_project_settings(ps)
        if hasattr(self, "pics_page"):
            self.pics_page.normalization_settings = self.normalization_settings
            self.pics_page.calibration = calibration
            self.pics_page.set_project_settings(ps)
        self.refresh_project_parameter_summary()
        self.statusbar.showMessage("通用参数已保存并同步到各工具", 3000)

    def switch_workspace_page(self, page_name: str):
        """Switch top-level workspace page and refresh shared calibration state."""
        page_map = {
            "project": self.project_page,
            "spectrum": self.spectrum_page,
            "temperature": self.temperature_page,
            "pie": self.pie_page,
            "mole_fraction": self.mole_fraction_page,
            "pics": self.pics_page,
            "ionization": self.ionization_page,
            "isotope": self.isotope_page,
        }
        page = page_map.get(page_name)
        if page is None:
            return
        if page_name == "project":
            self.load_project_settings()
        self.apply_config_defaults()
        calibration = self.current_calibration()
        ps = self.project_settings_manager.get()
        if hasattr(self, "temperature_page"):
            self.temperature_page.calibration = calibration
            self.temperature_page.set_project_settings(ps)
        if hasattr(self, "pie_page"):
            self.pie_page.calibration = calibration
            self.pie_page.set_project_settings(ps)
        if hasattr(self, "mole_fraction_page"):
            self.mole_fraction_page.calibration = calibration
            self.mole_fraction_page.set_project_settings(ps)
        if hasattr(self, "pics_page"):
            self.pics_page.calibration = calibration
            self.pics_page.set_project_settings(ps)
        self.workspace_stack.setCurrentWidget(page)
        if page_name in self.page_buttons:
            self.page_buttons[page_name].setChecked(True)
        if hasattr(self, "project_stage_table"):
            self.refresh_project_lifecycle(ps)
        if hasattr(self, "project_param_summary"):
            self.refresh_project_parameter_summary()

    def open_common_parameters(self):
        if hasattr(self, "project_common_settings_widget"):
            self.project_common_settings_widget.settings = self.normalization_settings
            self.project_common_settings_widget.calibration = self.current_calibration()
            self.project_common_settings_widget.load_from_settings()
        self.switch_workspace_page("project")
        if hasattr(self, "project_tabs"):
            self.project_tabs.setCurrentWidget(self.project_common_page)
