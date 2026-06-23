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
    ArtifactCategory,
    DataSourceValidationStatus,
    WorkflowProfile,
    analyze_workflow_capabilities,
    collect_project_files,
    create_project_snapshot,
    ensure_project_structure,
    export_project_archive,
    get_data_source_validation_status,
    next_project_stage,
    project_root,
    sanitize_project_slug,
    scan_project_artifacts,
    validate_all_data_sources,
)
from bl03u_masstool.core.project_settings import ProjectSettings, ProjectSettingsManager
from bl03u_masstool.frontends.pyqt_app.isotope.dialog import IsotopeAbundanceDialog
from bl03u_masstool.frontends.pyqt_app.progress_dialog import ProgressDialog
from bl03u_masstool.frontends.pyqt_app.worker import (
    ExportWorker,
    SnapshotWorker,
)
from bl03u_masstool.frontends.pyqt_app.worker_manager import WorkerManager
from bl03u_masstool.frontends.pyqt_app.mole_fraction.dialog import MoleFractionDialog
from bl03u_masstool.frontends.pyqt_app.nist.widget import IonizationEnergyLookupWidget
from bl03u_masstool.frontends.pyqt_app.normalization.widget import CommonParametersWidget
from bl03u_masstool.frontends.pyqt_app.pics.dialog import PICSCalculatorDialog
from bl03u_masstool.frontends.pyqt_app.pics.import_widget import PICSImportWidget
from bl03u_masstool.frontends.pyqt_app.pie.dialog import PIESpeciesFitDialog
from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog




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
        self.pics_import_page = PICSImportWidget(self.workspace_stack)
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
            ("pics_import", "PICS导入", self.pics_import_page),
            ("ionization", "IE查询", self.ionization_page),
            ("isotope", "分子/同位素", self.isotope_page),
        ]
        # Pages that get a separator inserted AFTER them in the nav bar
        _nav_separators_after = {"project", "mole_fraction"}

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

        # ProjectSettings remains the source of truth for managed data paths.
        # Tool-page paths are updated from project settings/imports, not pulled back implicitly.

    def _build_project_page(self):
        page_layout = QVBoxLayout(self.project_page)
        page_layout.setContentsMargins(12, 12, 12, 12)
        page_layout.setSpacing(10)

        self.project_tabs = QtWidgets.QTabWidget(self.project_page)
        self.project_tabs.setObjectName("ProjectTabs")

        # --- Tab 1: 项目设置 + 数据源 ---
        self.project_identity_page = QtWidgets.QWidget(self.project_tabs)

        settings_scroll = QtWidgets.QScrollArea(self.project_identity_page)
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        settings_scroll.setObjectName("ProjectSettingsScroll")

        settings_content = QtWidgets.QWidget(settings_scroll)
        identity_layout = QVBoxLayout(settings_content)
        identity_layout.setContentsMargins(8, 8, 8, 8)
        identity_layout.setSpacing(10)

        self._build_project_identity_card(self.project_identity_page)
        self._build_datasource_card(self.project_identity_page)
        identity_layout.addWidget(self.project_identity_card)
        identity_layout.addWidget(self.datasource_card)

        settings_scroll.setWidget(settings_content)
        settings_page_layout = QVBoxLayout(self.project_identity_page)
        settings_page_layout.setContentsMargins(0, 0, 0, 0)
        settings_page_layout.addWidget(settings_scroll)

        # --- Tab 2: 通用参数 ---
        self.project_common_parameters_widget = CommonParametersWidget(
            self.normalization_settings,
            self.current_calibration(),
            self.project_tabs,
        )
        self.project_common_parameters_widget.save_button.clicked.connect(self.on_project_common_parameters_saved)
        self.project_common_parameters_widget.settings_saved.connect(self.on_project_common_parameters_saved)
        # 立即设置ProjectSettings
        self.project_common_parameters_widget.set_project_settings(ProjectSettingsManager().get())

        # --- Tab 3: 功能默认参数 ---
        # 集成原有的 PeakDetectionWidget 和功能参数页面
        from bl03u_masstool.frontends.pyqt_app.normalization.widget import FunctionDefaultsWidget
        self.project_function_defaults_widget = FunctionDefaultsWidget(self.project_tabs)
        self.project_function_defaults_widget.settings_saved.connect(self.on_project_common_parameters_saved)
        self.project_function_defaults_widget.set_project_settings(ProjectSettingsManager().get())

        # --- Tab 4: 产物管理 ---
        self.project_artifacts_page = QtWidgets.QWidget(self.project_tabs)
        artifacts_layout = QVBoxLayout(self.project_artifacts_page)
        artifacts_layout.setContentsMargins(8, 8, 8, 8)
        artifacts_layout.setSpacing(10)
        self._build_artifact_manager_card(self.project_artifacts_page)
        artifacts_layout.addWidget(self.artifact_manager_card, stretch=1)

        self.project_tabs.addTab(self.project_identity_page, "项目设置")
        self.project_tabs.addTab(self.project_common_parameters_widget, "通用参数")
        self.project_tabs.addTab(self.project_function_defaults_widget, "功能默认参数")
        self.project_tabs.addTab(self.project_artifacts_page, "产物管理")
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

        # 新建项目按钮（突出显示）
        self.project_new_button = QPushButton("➕ 新建项目", action_bar)
        self.project_new_button.setObjectName("PrimaryButton")
        self.project_new_button.setToolTip("清空当前表单，开始创建新项目")
        self.project_new_button.setFixedHeight(32)
        self.project_new_button.setStyleSheet("""
            QPushButton#PrimaryButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 0px 16px;
                font-weight: bold;
            }
            QPushButton#PrimaryButton:hover {
                background-color: #45a049;
            }
            QPushButton#PrimaryButton:pressed {
                background-color: #3d8b40;
            }
        """)
        action_layout.addWidget(self.project_new_button)

        # 打开项目按钮
        self.project_open_button = QPushButton("📂 打开项目", action_bar)
        self.project_open_button.setObjectName("BrowseButton")
        self.project_open_button.setToolTip("打开已有项目配置文件")
        self.project_open_button.setFixedHeight(32)
        action_layout.addWidget(self.project_open_button)
        action_layout.addSpacing(12)

        self.project_save_and_apply_button = QPushButton("保存并应用", action_bar)

        self.project_save_and_apply_button.setToolTip(
            "保存项目设置 → 创建项目文件夹 → 同步参数到各工具页面。完整初始化和配置。"
        )

        self.project_save_and_apply_button.setFixedHeight(28)
        action_layout.addWidget(self.project_save_and_apply_button)

        self.project_save_and_apply_button.setObjectName("BrowseButton")
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

        self.project_new_button.clicked.connect(self.new_project)
        self.project_open_button.clicked.connect(self.open_project)
        self.project_save_and_apply_button.clicked.connect(self.save_and_apply_project_settings)
        self.project_output_dir_button.clicked.connect(
            lambda: self.select_project_folder(self.project_output_dir_edit, "选择输出目录")
        )

    def _build_artifact_manager_card(self, parent):
        self.artifact_manager_card = QtWidgets.QFrame(parent)
        self.artifact_manager_card.setObjectName("ProjectCard")

        card_layout = QVBoxLayout(self.artifact_manager_card)
        card_layout.setContentsMargins(10, 8, 10, 10)
        card_layout.setSpacing(12)

        # ── 产物摘要卡片 ──
        summary_card = QtWidgets.QFrame(self.artifact_manager_card)
        summary_card.setObjectName("ProjectCard")
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(10, 8, 10, 10)
        summary_layout.setSpacing(8)

        summary_header = QtWidgets.QLabel("产物摘要", summary_card)
        summary_header.setObjectName("ProjectTitle")
        self.artifact_summary_label = QtWidgets.QLabel("", summary_card)
        self.artifact_summary_label.setObjectName("ProjectStatus")
        self.artifact_summary_label.setWordWrap(True)

        summary_layout.addWidget(summary_header)
        summary_layout.addWidget(self.artifact_summary_label)
        card_layout.addWidget(summary_card)

        # ── 产物分类树 + 产物表格 ──
        content_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # 左侧分类树
        self.artifact_category_tree = QtWidgets.QTreeWidget(self.artifact_manager_card)
        self.artifact_category_tree.setHeaderLabels(["分类"])
        self.artifact_category_tree.setMaximumWidth(200)
        self.artifact_category_tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.artifact_category_tree.itemSelectionChanged.connect(self.on_artifact_category_changed)
        content_splitter.addWidget(self.artifact_category_tree)

        # 右侧产物表格
        self.artifact_product_table = QtWidgets.QTableWidget(self.artifact_manager_card)
        self.artifact_product_table.setColumnCount(5)
        self.artifact_product_table.setHorizontalHeaderLabels(["产物名称", "类型", "大小", "生成时间", "状态"])
        self.artifact_product_table.setAlternatingRowColors(True)
        self.artifact_product_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.artifact_product_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.artifact_product_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.artifact_product_table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.artifact_product_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.artifact_product_table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.artifact_product_table.horizontalHeader().setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        content_splitter.addWidget(self.artifact_product_table)
        content_splitter.setStretchFactor(0, 1)
        content_splitter.setStretchFactor(1, 3)

        card_layout.addWidget(content_splitter, stretch=1)

        # ── 操作按钮栏 ──
        action_bar = QtWidgets.QWidget(self.artifact_manager_card)
        action_bar.setObjectName("ProjectActionBar")
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(8, 6, 8, 6)
        action_layout.setSpacing(6)

        self.artifact_preview_button = QPushButton("预览", action_bar)
        self.artifact_open_dir_button = QPushButton("打开目录", action_bar)
        self.artifact_metadata_button = QPushButton("查看元数据", action_bar)
        self.artifact_snapshot_button = QPushButton("创建项目快照", action_bar)
        self.artifact_export_button = QPushButton("导出项目", action_bar)
        self.artifact_refresh_button = QPushButton("刷新", action_bar)

        for button in (
            self.artifact_preview_button,
            self.artifact_open_dir_button,
            self.artifact_metadata_button,
            self.artifact_snapshot_button,
            self.artifact_export_button,
            self.artifact_refresh_button,
        ):
            button.setFixedHeight(28)
            button.setObjectName("BrowseButton")
            action_layout.addWidget(button)

        action_layout.addStretch(1)
        card_layout.addWidget(action_bar)

        # 连接按钮信号
        self.artifact_preview_button.clicked.connect(self.preview_selected_artifact)
        self.artifact_open_dir_button.clicked.connect(self.open_artifact_directory)
        self.artifact_metadata_button.clicked.connect(self.show_artifact_metadata)
        self.artifact_snapshot_button.clicked.connect(self.create_artifact_snapshot)
        self.artifact_export_button.clicked.connect(self.export_project_artifacts)
        self.artifact_refresh_button.clicked.connect(self.refresh_project_artifacts_page)

    # ── Project data sources ─────────────────────────────────────────────

    def _build_datasource_card(self, parent):
        self.datasource_card = QtWidgets.QFrame(parent)
        self.datasource_card.setObjectName("ProjectCard")

        card_layout = QVBoxLayout(self.datasource_card)
        card_layout.setContentsMargins(10, 8, 10, 10)
        card_layout.setSpacing(10)

        # ── 顶部操作栏：标题、状态与导入入口 ──
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(10)

        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(3)

        datasource_title = QtWidgets.QLabel("项目数据源", self.datasource_card)
        datasource_title.setObjectName("ProjectTitle")
        datasource_hint = QtWidgets.QLabel(
            "通过导入向导把外部数据复制并登记到项目；下方路径可用于检查或微调已登记的数据源。",
            self.datasource_card,
        )
        datasource_hint.setObjectName("ProjectHint")
        datasource_hint.setWordWrap(True)
        self.datasource_status_label = QtWidgets.QLabel("", self.datasource_card)
        self.datasource_status_label.setObjectName("ProjectHint")
        self.datasource_status_label.setWordWrap(False)

        title_column.addWidget(datasource_title)
        title_column.addWidget(datasource_hint)
        title_column.addWidget(self.datasource_status_label)
        top_bar.addLayout(title_column, stretch=1)
        card_layout.addLayout(top_bar)

        # ── 工作流能力标签行 ──
        self.datasource_workflow_bar = QHBoxLayout()
        self.datasource_workflow_bar.setContentsMargins(0, 0, 0, 0)
        self.datasource_workflow_bar.setSpacing(6)
        self.datasource_workflow_bar.addStretch(1)
        card_layout.addLayout(self.datasource_workflow_bar)

        # 分割线
        sep = QtWidgets.QFrame(self.datasource_card)
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setObjectName("NavSeparator")
        card_layout.addWidget(sep)

        # ── 路径配置区：每行含内联状态 ──
        self.datasource_row_status_labels: dict[str, QtWidgets.QLabel] = {}

        self.project_single_file_edit = QLineEdit(self.datasource_card)
        self.project_single_file_edit.setPlaceholderText("选择 .txt/.asc/.888 单谱文件")
        self.project_sum_folder_edit = QLineEdit(self.datasource_card)
        self.project_sum_folder_edit.setPlaceholderText("选择累计谱数据目录")
        self.project_temperature_folder_edit = QLineEdit(self.datasource_card)
        self.project_temperature_folder_edit.setPlaceholderText("选择温度扫描 txt 文件目录")
        self.project_pie_folder_edit = QLineEdit(self.datasource_card)
        self.project_pie_folder_edit.setPlaceholderText("选择 PIE 扫描目录")
        self.project_manual_peak_edit = QLineEdit(self.datasource_card)
        self.project_manual_peak_edit.setPlaceholderText("选择 yaml/csv/xlsx 卡峰文件")

        def _browse_btn():
            b = QPushButton("选择", self.datasource_card)
            b.setObjectName("BrowseButton")
            b.setFixedHeight(26)
            b.setFixedWidth(44)
            return b

        def _status_label():
            lbl = QtWidgets.QLabel("—", self.datasource_card)
            lbl.setObjectName("ProjectHint")
            lbl.setFixedWidth(90)
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
            return lbl

        def _path_group(title: str):
            group = QtWidgets.QGroupBox(title, self.datasource_card)
            layout = QtWidgets.QGridLayout(group)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setHorizontalSpacing(6)
            layout.setVerticalSpacing(5)
            layout.setColumnStretch(1, 1)  # path edit stretches
            return group, layout

        def _add_path_row(layout, row: int, source_key: str, label: str, edit: QLineEdit, button: QPushButton):
            lbl = QtWidgets.QLabel(label, self.datasource_card)
            lbl.setFixedWidth(88)
            status_lbl = _status_label()
            self.datasource_row_status_labels[source_key] = status_lbl
            layout.addWidget(lbl, row, 0)
            layout.addWidget(edit, row, 1)
            layout.addWidget(button, row, 2)
            layout.addWidget(status_lbl, row, 3)

        self.project_single_file_button = _browse_btn()
        self.project_sum_folder_button = _browse_btn()
        self.project_temperature_folder_button = _browse_btn()
        self.project_pie_folder_button = _browse_btn()
        self.project_manual_peak_button = _browse_btn()

        workbench_group, workbench_layout = _path_group("质谱工作台")
        _add_path_row(workbench_layout, 0, "single_spectrum", "单谱文件", self.project_single_file_edit, self.project_single_file_button)
        _add_path_row(workbench_layout, 1, "sum_spectrum", "累计谱文件夹", self.project_sum_folder_edit, self.project_sum_folder_button)
        card_layout.addWidget(workbench_group)

        analysis_group, analysis_layout = _path_group("分析模块")
        _add_path_row(analysis_layout, 0, "temperature_scan", "温度扫描目录", self.project_temperature_folder_edit, self.project_temperature_folder_button)
        _add_path_row(analysis_layout, 1, "pie_scan", "PIE扫描目录", self.project_pie_folder_edit, self.project_pie_folder_button)
        _add_path_row(analysis_layout, 2, "manual_peak", "手动卡峰文件", self.project_manual_peak_edit, self.project_manual_peak_button)
        card_layout.addWidget(analysis_group)

        # ── 参考数据库（只读）──
        ref_group = QtWidgets.QGroupBox("参考数据库", self.datasource_card)
        ref_layout = QtWidgets.QGridLayout(ref_group)
        ref_layout.setContentsMargins(8, 8, 8, 8)
        ref_layout.setHorizontalSpacing(6)
        ref_layout.setVerticalSpacing(5)
        ref_layout.setColumnStretch(1, 1)

        ref_label = QtWidgets.QLabel("PICS截面数据库", self.datasource_card)
        ref_label.setFixedWidth(110)
        self.project_pics_database_display = QLineEdit(self.datasource_card)
        self.project_pics_database_display.setReadOnly(True)
        self.project_pics_database_display.setPlaceholderText("系统参考数据库（自动配置）")
        ref_layout.addWidget(ref_label, 0, 0)
        ref_layout.addWidget(self.project_pics_database_display, 0, 1)
        card_layout.addWidget(ref_group)

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
        self.project_manual_peak_button.clicked.connect(self.select_project_manual_peak)
        # Auto-save and push project paths when edited
        self.project_single_file_edit.editingFinished.connect(self._auto_save_datasource)
        self.project_sum_folder_edit.editingFinished.connect(self._auto_save_datasource)
        self.project_temperature_folder_edit.editingFinished.connect(self._auto_save_datasource)
        self.project_pie_folder_edit.editingFinished.connect(self._auto_save_datasource)
        self.project_manual_peak_edit.editingFinished.connect(self._auto_save_datasource)

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
        self.fp_mf_parent_initial_mf.setToolTip("母体物种在参考温度T₀处的摩尔分数（已知或假设值）")
        self.fp_mf_parent_initial_mf.setRange(0, 1)
        self.fp_mf_parent_initial_mf.setDecimals(6)
        self.fp_mf_parent_initial_mf.setValue(0.002)
        self.fp_mf_photon_energy = QtWidgets.QDoubleSpinBox()
        self.fp_mf_photon_energy.setToolTip("实验使用的VUV光子能量 (eV)")
        self.fp_mf_photon_energy.setRange(0, 100)
        self.fp_mf_photon_energy.setDecimals(4)
        self.fp_mf_photon_energy.setValue(10.0)
        self.fp_mf_reference_temperature = QtWidgets.QSpinBox()
        self.fp_mf_reference_temperature.setToolTip("参考温度T₀ (°C)：在此温度下母体摩尔分数为已知的初始值")
        self.fp_mf_reference_temperature.setRange(0, 2000)
        self.fp_mf_reference_temperature.setValue(550)

        mf_layout.addWidget(QtWidgets.QLabel("质量歧视指数"), 0, 0)
        mf_layout.addWidget(self.fp_mf_mass_disc_exponent, 0, 1)
        mf_layout.addWidget(QtWidgets.QLabel("母体 m/z"), 0, 2)
        mf_layout.addWidget(self.fp_mf_parent_mz, 0, 3)
        mf_layout.addWidget(QtWidgets.QLabel("母体初始摩尔分数"), 1, 0)
        mf_layout.addWidget(self.fp_mf_parent_initial_mf, 1, 1)
        mf_layout.addWidget(QtWidgets.QLabel("光子能量 (eV)"), 1, 2)
        mf_layout.addWidget(self.fp_mf_photon_energy, 1, 3)
        mf_layout.addWidget(QtWidgets.QLabel("参考温度 T₀ (°C)"), 2, 0)
        mf_layout.addWidget(self.fp_mf_reference_temperature, 2, 1)
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
        self.project_manual_peak_edit.setText(ps.manual_peak_file)
        # PICS database is read-only reference database
        if hasattr(self, "project_pics_database_display"):
            self.project_pics_database_display.setText(ps.pics_database_path)

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
        ps.manual_peak_file = self.project_manual_peak_edit.text().strip()
        # PICS database path is never modified from UI (read-only)
        return ps

    def load_project_settings(self) -> None:
        """加载项目配置。

        流程：
        1. 检查当前项目的输出目录（从UI或ProjectSettings）
        2. 如果项目配置存在，自动设置项目路径
        3. 加载项目级或全局配置
        4. 同步到所有UI元素
        """
        from pathlib import Path

        # Step 1: 获取当前项目的输出目录
        current_output_dir = self.project_output_dir_edit.text().strip()

        # 如果UI中没有项目路径，尝试从manager获取
        if not current_output_dir or current_output_dir == "output":
            # 先get一次（可能是全局配置）
            ps_temp = self.project_settings_manager.get()
            if ps_temp and ps_temp.output_dir and ps_temp.output_dir.strip() != "output":
                current_output_dir = ps_temp.output_dir

        # Step 2: 如果有有效的项目路径，设置它
        if current_output_dir and current_output_dir.strip() != "output":
            project_path = Path(current_output_dir)
            # 检查项目级配置文件是否存在
            project_config = project_path / "config" / "project.yaml"
            if project_config.exists():
                self.project_settings_manager.set_project_path(project_path)

        # Step 3: 现在加载配置（可能是项目级或全局的）
        ps = self.project_settings_manager.get()

        # Do NOT auto-fill paths here - only display what's actually saved in config
        # Users must use the import wizard to set up data sources
        self._read_project_settings_to_ui(ps)
        # NOTE: 功能参数现在在 FunctionDefaultsWidget 中管理
        # self._load_function_params_to_ui(ps)
        # 同步ProjectSettings到参数widgets
        if hasattr(self, "project_common_parameters_widget"):
            self.project_common_parameters_widget.set_project_settings(ps)
            self.project_common_parameters_widget.load_from_settings()  # 立即加载到UI
        if hasattr(self, "project_function_defaults_widget"):
            self.project_function_defaults_widget.set_project_settings(ps)
            self.project_function_defaults_widget.load_from_settings()  # 立即加载到UI
        self.update_project_title()
        self.refresh_project_lifecycle()
        self.refresh_project_datasource_page()

    def _load_function_params_to_ui(self, ps: ProjectSettings) -> None:
        """DEPRECATED: 功能参数现在在 FunctionDefaultsWidget 中管理"""
        # 保留此方法以维持向后兼容性，但内容已移到 FunctionDefaultsWidget
        pass

    def refresh_project_datasource_page(self, ps: ProjectSettings | None = None) -> None:
        """Refresh data source validation status on the data import page"""
        if ps is None:
            ps = self.project_settings_manager.get()

        validation_records = validate_all_data_sources(ps)
        validation_status = get_data_source_validation_status(ps)
        workflow_result = analyze_workflow_capabilities(ps)

        self.current_data_source_status = validation_status
        self.current_validation_records = validation_records
        self.current_workflow_result = workflow_result

        # 项目未初始化
        root_exists = project_root(ps).exists()
        if not root_exists:
            if hasattr(self, "datasource_status_label"):
                self.datasource_status_label.setText("请先在「项目设置」页初始化项目")
            self._clear_datasource_row_statuses()
            self._refresh_workflow_chips([])
            return

        # 顶部状态文字（单行简洁）
        if hasattr(self, "datasource_status_label"):
            valid_records = [r for r in validation_records if r.is_valid]
            invalid_records = [r for r in validation_records if r.path and not r.is_valid]
            total_files = sum(r.file_count for r in valid_records)

            if validation_status == DataSourceValidationStatus.UNCONFIGURED:
                status_text = "尚未配置数据源，请使用导入向导或在下方填入数据路径"
            elif validation_status == DataSourceValidationStatus.INVALID:
                status_text = f"[警告] {len(invalid_records)} 个路径失效，请重新导入"
            elif validation_status == DataSourceValidationStatus.PARTIAL:
                status_text = f"{len(valid_records)} 个数据源就绪，共 {total_files} 个文件"
            else:
                status_text = f"✓ 全部就绪 — {len(valid_records)} 个数据源，共 {total_files} 个文件"
            self.datasource_status_label.setText(status_text)

        # 每行内联状态
        record_map = {r.source_key: r for r in validation_records}
        key_map = {
            "single_spectrum": "single_spectrum",
            "sum_spectrum": "sum_spectrum",
            "temperature_scan": "temperature_scan",
            "pie_scan": "pie_scan",
            "manual_peak": "manual_peak",
        }
        if hasattr(self, "datasource_row_status_labels"):
            for ui_key, source_key in key_map.items():
                lbl = self.datasource_row_status_labels.get(ui_key)
                if lbl is None:
                    continue
                record = record_map.get(source_key)
                if record is None or not record.path:
                    lbl.setText("—")
                    lbl.setStyleSheet("")
                elif not record.is_valid:
                    lbl.setText("[警告] 路径失效")
                    lbl.setStyleSheet("color: #c0392b;")
                else:
                    count_text = f"{record.file_count}个文件" if record.file_count > 1 else "已配置"
                    lbl.setText(f"✓ {count_text}")
                    lbl.setStyleSheet("color: #27ae60;")

        # 工作流能力 chips
        available = []
        for profile in (workflow_result.available_workflows or []):
            if profile == WorkflowProfile.SPECTRUM_ONLY:
                available.append("质谱工作台")
            elif profile == WorkflowProfile.TEMPERATURE_SCAN:
                available.append("温度扫描")
            elif profile == WorkflowProfile.PIE_ANALYSIS:
                available.append("PIE拟合")
            elif profile == WorkflowProfile.FULL_ANALYSIS:
                available.append("完整流程")
        self._refresh_workflow_chips(available)

    def _clear_datasource_row_statuses(self) -> None:
        if hasattr(self, "datasource_row_status_labels"):
            for lbl in self.datasource_row_status_labels.values():
                lbl.setText("—")
                lbl.setStyleSheet("")

    def _refresh_workflow_chips(self, available: list[str]) -> None:
        if not hasattr(self, "datasource_workflow_bar"):
            return
        # 清空旧 chips（保留末尾 stretch）
        while self.datasource_workflow_bar.count() > 1:
            item = self.datasource_workflow_bar.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not available:
            return
        prefix = QtWidgets.QLabel("可用工作流：", None)
        prefix.setObjectName("ProjectHint")
        self.datasource_workflow_bar.insertWidget(0, prefix)
        for i, name in enumerate(available):
            chip = QtWidgets.QLabel(name, None)
            chip.setObjectName("WorkflowChip")
            chip.setStyleSheet(
                "QLabel { background: #27ae60; color: white; border-radius: 10px;"
                " padding: 1px 8px; font-size: 11px; }"
            )
            self.datasource_workflow_bar.insertWidget(i + 1, chip)

    def _collect_function_params_from_ui(self, ps: ProjectSettings) -> None:
        """Deprecated: Use FunctionDefaultsWidget.apply_to_settings() instead.

        This method was deprecated after Phase 3 UI refactoring when function
        parameters were moved to FunctionDefaultsWidget. It now delegates to the
        widget so older call sites still persist the current parameter edits.
        """
        if hasattr(self, "project_function_defaults_widget"):
            self.project_function_defaults_widget.apply_to_settings(ps)

    def new_project(self) -> None:
        """清空表单，准备创建新项目"""
        self.project_name_edit.clear()
        self.project_system_edit.clear()
        self.project_description_edit.clear()
        self.project_output_dir_edit.clear()
        self.project_name_edit.setFocus()
        self.statusbar.showMessage("已清空表单，请填写项目信息并点击'保存并应用'", 3000)

    def open_project(self) -> None:
        """打开已有项目（选择项目目录或配置文件）"""
        project_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择项目目录或包含 config/project.yaml 的文件夹",
            str(Path.home() / "Downloads"),
            QtWidgets.QFileDialog.Option.ShowDirsOnly
        )

        if not project_path:
            return

        project_path = Path(project_path)

        # Try to find config/project.yaml in the selected directory
        config_file = project_path / "config" / "project.yaml"
        if not config_file.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "配置文件不存在",
                f"在 {project_path} 中找不到 config/project.yaml\n\n"
                "请确保选择的是有效的项目目录。"
            )
            return

        try:
            # Load project settings from config file
            from bl03u_masstool.core.project_settings import load_project_settings
            ps = load_project_settings(config_file)

            # Set project path for manager
            self.project_settings_manager.set_project_path(project_path)

            # Display loaded settings in UI
            self.project_name_edit.setText(ps.project_name or "")
            self.project_system_edit.setText(ps.system or "")
            self.project_description_edit.setText(ps.description or "")
            self.project_output_dir_edit.setText(ps.output_dir or "")

            # Load to all parameter widgets
            self.project_common_parameters_widget.set_project_settings(ps)
            self.project_common_parameters_widget.load_from_settings()

            if hasattr(self, "project_function_defaults_widget"):
                self.project_function_defaults_widget.set_project_settings(ps)
                self.project_function_defaults_widget.load_from_settings()

            if hasattr(self, "project_peak_detection_widget"):
                self.project_peak_detection_widget.set_project_settings(ps)
                self.project_peak_detection_widget.load_from_settings()

            self.statusbar.showMessage(f"✓ 已加载项目：{ps.project_name}", 3000)

        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "加载项目失败",
                f"无法加载项目配置：\n{str(exc)}"
            )
            self.project_settings_manager.clear_project_path()

    def save_and_apply_project_settings(self) -> None:
        """Save project settings, create project structure, and sync to tools.

        This is the complete operation: initialize + apply to all tool pages.
        """
        ps = self._collect_project_settings_from_ui()
        ps = self._normalize_project_output_dir(ps)

        # Collect function parameters from FunctionDefaultsWidget
        if hasattr(self, "project_function_defaults_widget"):
            self.project_function_defaults_widget.apply_to_settings(ps)

        # Step 1: Create project directory structure
        try:
            ensure_project_structure(ps)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "初始化项目失败", str(exc))
            return

        # Step 2: Set project config path (before saving)
        from pathlib import Path
        self.project_settings_manager.set_project_path(Path(ps.output_dir))

        # Step 3: Save configuration to project-specific location
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()

        # Step 4: Read settings back to UI
        self._read_project_settings_to_ui(ps)

        # Step 5: Sync to tools
        self._apply_settings_to_tools(ps)

        # Step 6: Refresh UI
        self.update_project_title()
        self.refresh_project_lifecycle(ps)
        self.refresh_project_parameter_summary()
        self.statusbar.showMessage("[成功] 项目已保存、初始化并应用到工具", 3000)

    def initialize_project_structure(self) -> None:
        """Initialize project structure only (create folders, save config).

        Does NOT sync to tool pages. Use this for lightweight initialization.
        Use save_and_apply_project_settings() for complete setup including tools.
        """
        ps = self._collect_project_settings_from_ui()
        ps = self._normalize_project_output_dir(ps)

        # Collect function parameters from FunctionDefaultsWidget
        if hasattr(self, "project_function_defaults_widget"):
            self.project_function_defaults_widget.apply_to_settings(ps)

        # Create project directory structure
        try:
            ensure_project_structure(ps)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "初始化项目失败", str(exc))
            return

        # Set project config path (before saving)
        from pathlib import Path
        self.project_settings_manager.set_project_path(Path(ps.output_dir))

        # Save configuration to project-specific location
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()

        # Read settings back to UI
        self._read_project_settings_to_ui(ps)

        # Refresh UI (but don't sync to tools)
        self.update_project_title()
        self.refresh_project_lifecycle(ps)
        self.refresh_project_parameter_summary()
        self.statusbar.showMessage(f"✓ 项目已初始化：{project_root(ps)}", 3000)

    def apply_project_settings_to_tools(self) -> None:
        """Sync project settings to tool pages (light version, no save/init)."""
        ps = self._collect_project_settings_from_ui()
        self._collect_function_params_from_ui(ps)
        self._apply_settings_to_tools(ps)
        self.statusbar.showMessage("项目设置已应用到工具", 3000)

    def _apply_settings_to_tools(self, ps: ProjectSettings) -> None:
        """Internal method: sync project settings to all tool pages."""
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
        self._sync_project_settings_to_tool_pages(ps)

    def _sync_project_settings_to_tool_pages(self, ps: ProjectSettings) -> None:
        """Sync project settings to all tool pages (temperature, PIE, etc.)."""
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

    def save_function_params(self) -> None:
        """Save function parameters from the current project page."""
        ps = self._collect_project_settings_from_ui()
        self._collect_function_params_from_ui(ps)
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()

        # Phase 3 Step 2: Persist PIE configuration state after ProjectSettings save
        if hasattr(self, "pie_page"):
            success, error = self.pie_page.persist_per_mz_config_state()
            if not success:
                self.statusbar.showMessage(f"功能参数已保存，但 PIE 配置保存失败: {error}", 5000)
            else:
                self.statusbar.showMessage("功能参数已保存，项目摘要已更新", 3000)
        else:
            self.statusbar.showMessage("功能参数已保存，项目摘要已更新", 3000)

        self._sync_project_settings_to_tool_pages(ps)
        self.update_project_title()
        self.refresh_project_lifecycle(ps)
        self.refresh_project_parameter_summary()

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

        # Phase 3 Step 2: Persist PIE configuration state after ProjectSettings save
        if hasattr(self, "pie_page"):
            self.pie_page.persist_per_mz_config_state()

        # Sync project settings to all tool pages (Temperature, PIE, etc.)
        self._sync_project_settings_to_tool_pages(ps)
        self.refresh_project_lifecycle(ps)
        self.refresh_project_datasource_page(ps)

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

    def select_project_single_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择单谱文件",
            self._dialog_start_dir(self.project_single_file_edit.text()),
            "质谱数据 (*.txt *.asc *.888);;所有文件 (*)",
        )
        if path:
            self.project_single_file_edit.setText(path)
            self._auto_save_datasource()

    def select_project_folder(self, target: QLineEdit, title: str) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            title,
            self._dialog_start_dir(target.text()),
        )
        if folder:
            target.setText(folder)
            self._auto_save_datasource()

    def select_project_manual_peak(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择手动卡峰文件",
            self._dialog_start_dir(self.project_manual_peak_edit.text()),
            "Peak Files (*.yaml *.yml *.csv *.xlsx *.xls);;所有文件 (*)",
        )
        if path:
            self.project_manual_peak_edit.setText(path)
            self._auto_save_datasource()

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
        return str(Path("output") / self._project_default_folder_name(ps))

    def _project_folder_name(self, ps: ProjectSettings) -> str:
        return sanitize_project_slug(ps.project_name or ps.system or "untitled")

    def _project_default_folder_name(self, ps: ProjectSettings) -> str:
        slug = self._project_folder_name(ps)
        return slug if slug.lower().startswith("project") else f"Project_{slug}"

    def _normalize_project_output_dir(self, ps: ProjectSettings) -> ProjectSettings:
        """Resolve the project directory before initialization.

        The UI label is "项目目录", but users commonly pick an existing parent
        folder such as ~/Downloads and expect a new child folder named after the
        project. Existing project roots and matching explicit child paths are
        preserved.
        """
        raw_output = ps.output_dir.strip() if ps.output_dir else ""
        if (not raw_output or raw_output == "output") and (ps.project_name or ps.system):
            ps.output_dir = self._default_project_folder(ps)
        elif ps.project_name or ps.system:
            output_path = Path(raw_output).expanduser()
            config_file = output_path / "config" / "project.yaml"
            project_slug = self._project_folder_name(ps)
            default_name = self._project_default_folder_name(ps)
            matching_names = {project_slug, default_name}
            if output_path.exists() and output_path.is_dir() and not config_file.exists() and output_path.name not in matching_names:
                ps.output_dir = str(output_path / project_slug)

        ps.output_dir = str(project_root(ps))
        self.project_output_dir_edit.setText(ps.output_dir)
        return ps

    def _collect_and_save_project_settings(self) -> ProjectSettings:
        ps = self._collect_project_settings_from_ui()
        self._collect_function_params_from_ui(ps)
        # Set project config path before saving
        from pathlib import Path
        if ps.output_dir and ps.output_dir.strip() != "output":
            self.project_settings_manager.set_project_path(Path(ps.output_dir))
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()
        return ps

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
            self.project_tabs.setCurrentWidget(self.project_identity_page)
        elif status.key == "project_setup" and hasattr(self, "project_tabs"):
            self.project_tabs.setCurrentWidget(self.project_identity_page)
        self.statusbar.showMessage(status.next_action, 5000)

    def create_project_version_snapshot(self) -> None:
        ps = self._collect_and_save_project_settings()

        # Initialize worker manager if needed
        if not hasattr(self, "_worker_manager"):
            self._worker_manager = WorkerManager(self)

        # Create worker
        note = ps.project_name or ps.system or "snapshot"
        worker = SnapshotWorker(ps, note)

        # Connect signals
        worker.progress.connect(self._on_snapshot_progress)
        worker.finished.connect(self._on_snapshot_finished)
        worker.error.connect(self._on_snapshot_error)
        worker.cancelled.connect(self._on_snapshot_cancelled)

        # Show progress dialog
        self._snapshot_progress_dialog = ProgressDialog(self, "创建快照")
        self._snapshot_progress_dialog.rejected.connect(lambda: self._worker_manager.cancel())
        self._snapshot_progress_dialog.show()

        # Disable button
        if hasattr(self, "snapshot_button"):
            self.snapshot_button.setEnabled(False)

        # Start worker
        self._worker_manager.run_worker(worker)

    def _on_snapshot_progress(self, percent: int, message: str) -> None:
        """Update snapshot progress dialog."""
        if hasattr(self, "_snapshot_progress_dialog"):
            self._snapshot_progress_dialog.update(percent, message)

    def _on_snapshot_finished(self, result: dict) -> None:
        """Handle snapshot completion."""
        if hasattr(self, "_snapshot_progress_dialog"):
            self._snapshot_progress_dialog.close()

        if hasattr(self, "snapshot_button"):
            self.snapshot_button.setEnabled(True)

        if result.get("success"):
            self.refresh_project_lifecycle()
            self.statusbar.showMessage(f"项目快照已创建：{result['path']}", 5000)
        else:
            QtWidgets.QMessageBox.critical(self, "创建快照失败", result.get("error", "Unknown error"))

    def _on_snapshot_error(self, message: str) -> None:
        """Handle snapshot error."""
        if hasattr(self, "_snapshot_progress_dialog"):
            self._snapshot_progress_dialog.close()

        if hasattr(self, "snapshot_button"):
            self.snapshot_button.setEnabled(True)

        QtWidgets.QMessageBox.critical(self, "创建快照失败", message)

    def _on_snapshot_cancelled(self) -> None:
        """Handle snapshot cancellation."""
        if hasattr(self, "_snapshot_progress_dialog"):
            self._snapshot_progress_dialog.close()

        if hasattr(self, "snapshot_button"):
            self.snapshot_button.setEnabled(True)

        self.statusbar.showMessage("快照创建已取消", 3000)

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

        # Initialize worker manager if needed
        if not hasattr(self, "_worker_manager"):
            self._worker_manager = WorkerManager(self)

        # Create worker
        worker = ExportWorker(ps, path)

        # Connect signals
        worker.progress.connect(self._on_export_progress)
        worker.finished.connect(self._on_export_finished)
        worker.error.connect(self._on_export_error)
        worker.cancelled.connect(self._on_export_cancelled)

        # Show progress dialog
        self._export_progress_dialog = ProgressDialog(self, "导出项目")
        self._export_progress_dialog.rejected.connect(lambda: self._worker_manager.cancel())
        self._export_progress_dialog.show()

        # Disable button
        if hasattr(self, "export_button"):
            self.export_button.setEnabled(False)

        # Start worker
        self._worker_manager.run_worker(worker)

    def _on_export_progress(self, percent: int, message: str) -> None:
        """Update export progress dialog."""
        if hasattr(self, "_export_progress_dialog"):
            self._export_progress_dialog.update(percent, message)

    def _on_export_finished(self, result: dict) -> None:
        """Handle export completion."""
        if hasattr(self, "_export_progress_dialog"):
            self._export_progress_dialog.close()

        if hasattr(self, "export_button"):
            self.export_button.setEnabled(True)

        if result.get("success"):
            self.refresh_project_lifecycle()
            self.statusbar.showMessage(f"项目已导出：{result['path']}", 5000)
        else:
            QtWidgets.QMessageBox.critical(self, "导出失败", result.get("error", "Unknown error"))

    def _on_export_error(self, message: str) -> None:
        """Handle export error."""
        if hasattr(self, "_export_progress_dialog"):
            self._export_progress_dialog.close()

        if hasattr(self, "export_button"):
            self.export_button.setEnabled(True)

        QtWidgets.QMessageBox.critical(self, "导出失败", message)

    def _on_export_cancelled(self) -> None:
        """Handle export cancellation."""
        if hasattr(self, "_export_progress_dialog"):
            self._export_progress_dialog.close()

        if hasattr(self, "export_button"):
            self.export_button.setEnabled(True)

        self.statusbar.showMessage("项目导出已取消", 3000)

    def refresh_project_lifecycle(self, ps: ProjectSettings | None = None) -> None:
        if ps is None:
            ps = self.project_settings_manager.get()
        if hasattr(self, "artifact_category_tree"):
            self.refresh_project_artifacts_page(ps)

    def _format_file_size(self, size_bytes: int) -> str:
        size = float(size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024 or unit == "GB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
            size /= 1024
        return f"{size_bytes} B"

    def open_project_root_folder(self) -> None:
        ps = self.project_settings_manager.get()
        root = project_root(ps)
        root.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(root)))

    def refresh_project_artifacts_page(self, ps: ProjectSettings | None = None) -> None:
        """刷新产物管理页的所有内容"""
        if ps is None:
            ps = self.project_settings_manager.get()

        if not hasattr(self, "artifact_category_tree"):
            return

        # 扫描所有产物
        artifacts = scan_project_artifacts(ps)

        # 统计产物信息
        total_artifacts = len(artifacts)
        valid_count = sum(1 for a in artifacts if a.status == "有效")
        missing_count = sum(1 for a in artifacts if a.status == "缺失")
        incomplete_count = sum(1 for a in artifacts if a.status == "不完整")
        total_size = sum(a.size_bytes for a in artifacts if a.status == "有效")

        # 更新摘要标签
        summary_text = f"已登记 {total_artifacts} 个产物"
        if total_artifacts > 0:
            summary_text += f" | ✓ 有效 {valid_count}"
            if missing_count > 0:
                summary_text += f" | [!] 缺失 {missing_count}"
            if incomplete_count > 0:
                summary_text += f" | [x] 不完整 {incomplete_count}"
            summary_text += f" | 总大小 {self._format_file_size(total_size)}"
        self.artifact_summary_label.setText(summary_text)

        # 重建分类树
        self.artifact_category_tree.clear()
        category_items: dict[str, QtWidgets.QTreeWidgetItem] = {}

        for category in ArtifactCategory:
            item = QtWidgets.QTreeWidgetItem([category.label])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, category.key)
            self.artifact_category_tree.addTopLevelItem(item)
            category_items[category.key] = item

        # 按分类分组产物
        artifacts_by_category: dict[str, list] = {}
        for artifact in artifacts:
            artifacts_by_category.setdefault(artifact.category, []).append(artifact)

        # 存储产物到树的用户数据
        self._artifacts_data = artifacts
        self._artifacts_by_category = artifacts_by_category

        # 在每个分类下显示产物计数
        for category_key, category_items_list in artifacts_by_category.items():
            if category_key in category_items:
                count = len(category_items_list)
                item = category_items[category_key]
                text = item.text(0)
                if " (" not in text:
                    item.setText(0, f"{text} ({count})")

    def on_artifact_category_changed(self) -> None:
        """当选择分类时更新产物表格"""
        selected_items = self.artifact_category_tree.selectedItems()
        if not selected_items:
            self.artifact_product_table.setRowCount(0)
            return

        selected_item = selected_items[0]
        category_key = selected_item.data(0, QtCore.Qt.ItemDataRole.UserRole)

        if not hasattr(self, "_artifacts_by_category"):
            return

        artifacts = self._artifacts_by_category.get(category_key, [])

        # 填充表格
        self.artifact_product_table.setRowCount(len(artifacts))
        for row, artifact in enumerate(artifacts):
            artifact_name = Path(artifact.path).name

            # 产物名称
            name_item = QtWidgets.QTableWidgetItem(artifact_name)
            name_item.setToolTip(artifact.path)
            name_item.setData(QtCore.Qt.ItemDataRole.UserRole, artifact.path)
            self.artifact_product_table.setItem(row, 0, name_item)

            # 产物类型
            type_item = QtWidgets.QTableWidgetItem(artifact.artifact_type)
            self.artifact_product_table.setItem(row, 1, type_item)

            # 文件大小
            size_text = self._format_file_size(artifact.size_bytes) if artifact.size_bytes > 0 else "—"
            size_item = QtWidgets.QTableWidgetItem(size_text)
            size_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.artifact_product_table.setItem(row, 2, size_item)

            # 生成时间
            time_item = QtWidgets.QTableWidgetItem(artifact.generation_time[:10] if artifact.generation_time else "—")
            self.artifact_product_table.setItem(row, 3, time_item)

            # 状态
            status_item = QtWidgets.QTableWidgetItem(artifact.status)
            if artifact.status == "有效":
                status_item.setForeground(QtGui.QColor("green"))
            elif artifact.status == "缺失":
                status_item.setForeground(QtGui.QColor("red"))
            elif artifact.status == "不完整":
                status_item.setForeground(QtGui.QColor("orange"))
            self.artifact_product_table.setItem(row, 4, status_item)

    def preview_selected_artifact(self) -> None:
        """预览选定的产物"""
        selected = self.artifact_product_table.selectedIndexes()
        if not selected:
            return

        row = selected[0].row()
        artifact_path = self.artifact_product_table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
        if not artifact_path:
            return

        path = Path(artifact_path)
        if path.is_file() and path.suffix.lower() in (".xlsx", ".csv"):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
        else:
            QtWidgets.QMessageBox.information(self, "提示", "只能预览 XLSX 和 CSV 文件")

    def open_artifact_directory(self) -> None:
        """打开产物所在目录"""
        selected = self.artifact_product_table.selectedIndexes()
        if not selected:
            return

        row = selected[0].row()
        artifact_path = self.artifact_product_table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
        if not artifact_path:
            return

        path = Path(artifact_path).parent
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def show_artifact_metadata(self) -> None:
        """显示产物的元数据"""
        selected = self.artifact_product_table.selectedIndexes()
        if not selected:
            return

        row = selected[0].row()
        artifact_path = self.artifact_product_table.item(row, 0).data(QtCore.Qt.ItemDataRole.UserRole)
        if not artifact_path or not hasattr(self, "_artifacts_data"):
            return

        # 找到对应的产物
        artifact = None
        for a in self._artifacts_data:
            if a.path == artifact_path:
                artifact = a
                break

        if not artifact:
            return

        metadata_text = (
            f"产物类型: {artifact.artifact_type}\n"
            f"分类: {artifact.category}\n"
            f"路径: {artifact.path}\n"
            f"大小: {self._format_file_size(artifact.size_bytes)}\n"
            f"生成时间: {artifact.generation_time}\n"
            f"来源模块: {artifact.source_module}\n"
            f"状态: {artifact.status}\n"
            f"详情: {artifact.detail}"
        )

        QtWidgets.QMessageBox.information(self, "产物元数据", metadata_text)

    def create_artifact_snapshot(self) -> None:
        """创建项目快照"""
        ps = self.project_settings_manager.get()

        # 弹出输入对话框获取快照备注
        note, ok = QtWidgets.QInputDialog.getText(
            self, "创建项目快照", "请输入快照备注（可选）："
        )

        if not ok:
            return

        try:
            snapshot_path = create_project_snapshot(ps, note)
            self.statusbar.showMessage(f"快照已创建: {snapshot_path.name}", 4000)
            self.refresh_project_artifacts_page(ps)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "创建快照失败", f"错误: {str(e)}")

    def export_project_artifacts(self) -> None:
        """导出项目"""
        ps = self.project_settings_manager.get()

        # 弹出文件保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出项目",
            f"{sanitize_project_slug(ps.project_name or ps.system or 'project')}_export.zip",
            "ZIP 档案 (*.zip);;所有文件 (*)",
        )

        if not file_path:
            return

        try:
            export_path = export_project_archive(ps, file_path)
            QtWidgets.QMessageBox.information(self, "导出成功", f"项目已导出到:\n{export_path}")
            self.refresh_project_artifacts_page(ps)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "导出失败", f"错误: {str(e)}")

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
            "pics_import": self.pics_import_page,
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
        if hasattr(self, "project_param_summary"):
            self.refresh_project_parameter_summary()

    def open_common_parameters(self):
        if hasattr(self, "project_common_parameters_widget"):
            self.project_common_parameters_widget.settings = self.normalization_settings
            self.project_common_parameters_widget.calibration = self.current_calibration()
            # 设置ProjectSettings
            from bl03u_masstool.core.project_settings import ProjectSettingsManager
            ps = ProjectSettingsManager().get()
            self.project_common_parameters_widget.set_project_settings(ps)
            self.project_common_parameters_widget.load_from_settings()
        if hasattr(self, "project_peak_detection_widget"):
            ps = ProjectSettingsManager().get()
            self.project_peak_detection_widget.set_project_settings(ps)
            self.project_peak_detection_widget.load_from_settings()
        self.switch_workspace_page("project")
        if hasattr(self, "project_tabs"):
            self.project_tabs.setCurrentWidget(self.project_common_parameters_widget)
