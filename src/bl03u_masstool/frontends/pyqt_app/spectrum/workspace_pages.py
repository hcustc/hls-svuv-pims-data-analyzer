from __future__ import annotations

import os
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout

from bl03u_masstool.core.config import (
    PeakDetectionConfig,
    save_peak_detection_config,
    species_database_path,
)
from bl03u_masstool.core.normalization import load_normalization_settings
from bl03u_masstool.core.project_lifecycle import (
    PROJECT_DIRECTORIES,
    PROJECT_SOURCE_SPECS,
    DataSourceValidationStatus,
    WorkflowProfile,
    analyze_workflow_capabilities,
    collect_project_files,
    ensure_project_structure,
    get_data_source_validation_status,
    import_project_source,
    materialize_project_data_sources,
    next_project_stage,
    project_root,
    sanitize_project_slug,
    validate_all_data_sources,
)
from bl03u_masstool.core.project_settings import ProjectSettings, ProjectSettingsManager
from bl03u_masstool.frontends.pyqt_app.isotope.dialog import IsotopeAbundanceDialog
from bl03u_masstool.frontends.pyqt_app.progress_dialog import ProgressDialog
from bl03u_masstool.frontends.pyqt_app.worker import (
    ImportWorker,
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
            show_actions=False,
        )
        self.project_common_parameters_widget.settings_saved.connect(self.on_project_common_parameters_saved)
        # 立即设置ProjectSettings
        self.project_common_parameters_widget.set_project_settings(ProjectSettingsManager().get())

        # --- Tab 3: 功能默认参数 ---
        # 集成原有的 PeakDetectionWidget 和功能参数页面
        from bl03u_masstool.frontends.pyqt_app.normalization.widget import FunctionDefaultsWidget
        self.project_function_defaults_widget = FunctionDefaultsWidget(self.project_tabs)
        self.project_function_defaults_widget.settings_saved.connect(self.on_project_common_parameters_saved)
        self.project_function_defaults_widget.set_project_settings(ProjectSettingsManager().get())

        self.project_tabs.addTab(self.project_identity_page, "项目设置")
        self.project_tabs.addTab(self.project_common_parameters_widget, "通用参数")
        self.project_tabs.addTab(self.project_function_defaults_widget, "功能默认参数")
        self.project_tabs.currentChanged.connect(self._on_project_tab_changed)
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
        self.project_new_button = QPushButton("新建项目", action_bar)
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
        self.project_open_button = QPushButton("打开项目", action_bar)
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
        self.project_output_dir_edit.setPlaceholderText(
            "新建时选择父目录；保存后显示项目根目录"
        )

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
        form_layout.addRow("项目存放位置", output_row)
        card_layout.addLayout(form_layout)

        self.project_new_button.clicked.connect(self.new_project)
        self.project_open_button.clicked.connect(self.open_project)
        self.project_save_and_apply_button.clicked.connect(self.save_and_apply_project_settings)
        self.project_output_dir_button.clicked.connect(
            self.select_project_output_parent_folder
        )

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
            "原始实验目录由项目管理接管；单谱/累计谱是质谱工作台从项目内数据中选择的查看来源。",
            self.datasource_card,
        )
        datasource_hint.setObjectName("ProjectHint")
        datasource_hint.setWordWrap(True)

        title_column.addWidget(datasource_title)
        title_column.addWidget(datasource_hint)
        top_bar.addLayout(title_column, stretch=1)

        self.datasource_import_button = QPushButton("启动导入向导", self.datasource_card)
        self.datasource_import_button.setObjectName("PrimaryButton")
        self.datasource_import_button.setToolTip("选择原始数据源，并复制/登记到当前项目")
        self.datasource_import_button.setFixedHeight(30)
        top_bar.addWidget(self.datasource_import_button)
        card_layout.addLayout(top_bar)

        # 分割线
        sep = QtWidgets.QFrame(self.datasource_card)
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setObjectName("NavSeparator")
        card_layout.addWidget(sep)

        # ── 路径配置区：每行含内联状态 ──
        self.datasource_row_status_labels: dict[str, QtWidgets.QLabel] = {}

        self.project_temperature_folder_edit = QLineEdit(self.datasource_card)
        self.project_temperature_folder_edit.setPlaceholderText("选择温度扫描 txt 文件目录")
        self.project_pie_folder_edit = QLineEdit(self.datasource_card)
        self.project_pie_folder_edit.setPlaceholderText("选择 PIE 扫描目录")
        self.project_manual_peak_edit = QLineEdit(self.datasource_card)
        self.project_manual_peak_edit.setPlaceholderText("可导入旧卡峰文件，或由质谱工作台“保存到项目”生成")
        self.project_manual_peak_edit.setReadOnly(True)

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

        def _add_path_row(layout, row: int, source_key: str, label: str, edit: QLineEdit, button: QPushButton | None):
            lbl = QtWidgets.QLabel(label, self.datasource_card)
            lbl.setFixedWidth(88)
            status_lbl = _status_label()
            self.datasource_row_status_labels[source_key] = status_lbl
            layout.addWidget(lbl, row, 0)
            layout.addWidget(edit, row, 1)
            if button is not None:
                layout.addWidget(button, row, 2)
            layout.addWidget(status_lbl, row, 3)

        self.project_temperature_folder_button = _browse_btn()
        self.project_pie_folder_button = _browse_btn()

        analysis_group, analysis_layout = _path_group("原始扫描目录")
        _add_path_row(analysis_layout, 0, "temperature_scan", "温度扫描目录", self.project_temperature_folder_edit, self.project_temperature_folder_button)
        _add_path_row(analysis_layout, 1, "pie_scan", "PIE扫描目录", self.project_pie_folder_edit, self.project_pie_folder_button)
        card_layout.addWidget(analysis_group)

        artifact_group, artifact_layout = _path_group("项目产物")
        self.project_manual_peak_button = _browse_btn()
        self.project_manual_peak_button.setText("导入")
        _add_path_row(artifact_layout, 0, "manual_peak", "手动卡峰文件", self.project_manual_peak_edit, self.project_manual_peak_button)
        card_layout.addWidget(artifact_group)

        self.datasource_import_button.clicked.connect(self.import_project_datasource)
        self.project_temperature_folder_button.clicked.connect(
            lambda: self.select_project_folder(self.project_temperature_folder_edit, "选择温度扫描目录")
        )
        self.project_pie_folder_button.clicked.connect(
            lambda: self.select_project_folder(self.project_pie_folder_edit, "选择PIE扫描目录")
        )
        self.project_manual_peak_button.clicked.connect(self.import_project_manual_peak_file)
        # Auto-save and push project paths when edited
        self.project_temperature_folder_edit.editingFinished.connect(self._auto_save_datasource)
        self.project_pie_folder_edit.editingFinished.connect(self._auto_save_datasource)

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
        self.fp_mf_parent_mz.setRange(0, 1000)
        self.fp_mf_parent_mz.setSpecialValueText("未设置")
        self.fp_mf_parent_mz.setValue(0)
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
        self.project_temperature_folder_edit.setText(ps.temperature_scan_folder)
        self.project_pie_folder_edit.setText(ps.pie_scan_folder)
        self.project_manual_peak_edit.setText(ps.manual_peak_file)

    def _collect_project_settings_from_ui(self) -> ProjectSettings:
        """Build a ProjectSettings from all UI fields (does not save)."""
        ps = self.project_settings_manager.get()
        ps.project_name = self.project_name_edit.text().strip()
        ps.system = self.project_system_edit.text().strip()
        ps.description = self.project_description_edit.text().strip()
        ps.output_dir = self.project_output_dir_edit.text().strip() or "output"
        ps.temperature_scan_folder = self.project_temperature_folder_edit.text().strip()
        ps.pie_scan_folder = self.project_pie_folder_edit.text().strip()
        # PICS database path is never modified from UI (read-only)
        return ps

    def _collect_all_project_parameters_from_ui(self, ps: ProjectSettings) -> ProjectSettings:
        """Copy every project-owned parameter widget into the provided settings."""
        if hasattr(self, "project_common_parameters_widget"):
            self.project_common_parameters_widget.apply_to_settings(ps)
        if hasattr(self, "project_function_defaults_widget"):
            self.project_function_defaults_widget.apply_to_settings(ps)
        if hasattr(self, "project_peak_detection_widget"):
            self.project_peak_detection_widget.apply_to_settings(ps)
        return ps

    def _sync_project_page_edits_to_runtime(
        self,
        *,
        save_project: bool = True,
        sync_tools: bool = True,
    ) -> ProjectSettings:
        """Collect current project-page edits before another tool consumes settings."""
        ps = self._collect_project_settings_from_ui()
        self._collect_all_project_parameters_from_ui(ps)
        self.project_settings_manager.set(ps)
        self._sync_peak_detection_to_global_config(ps)
        if save_project and self.project_settings_manager.has_project_path():
            try:
                self.project_settings_manager.save()
            except Exception:
                pass
        self._apply_project_runtime_settings(ps)
        if sync_tools:
            self._sync_project_settings_to_tool_pages(ps)
        return ps

    def _on_project_tab_changed(self, _index: int) -> None:
        if not hasattr(self, "project_common_parameters_widget") or not hasattr(self, "project_function_defaults_widget"):
            return
        ps = self._sync_project_page_edits_to_runtime(save_project=True, sync_tools=True)
        self.refresh_project_parameter_summary()
        self.refresh_project_datasource_page(ps)

    @staticmethod
    def _sync_peak_detection_to_global_config(ps: ProjectSettings) -> None:
        """Write ProjectSettings peak detection values to the global config file.

        Project-scoped tools consume ProjectSettings directly. Keep the legacy
        global YAML in step so non-project consumers and older helper paths do
        not keep stale peak-detection defaults.
        """
        try:
            peak_config = PeakDetectionConfig(
                algorithm=str(ps.peak_algorithm),
                detection_min_idx=int(ps.detection_min_idx),
                threshold_end=float(ps.threshold_end),
                min_intensity=float(ps.min_intensity),
                nearby_peak_window=int(ps.nearby_peak_window),
                duplicate_window=int(ps.duplicate_window),
                weak_tail_early_window=int(ps.weak_tail_early_window),
                weak_tail_late_window=int(ps.weak_tail_late_window),
                weak_tail_ratio=float(ps.weak_tail_ratio),
                gaussian_window_max=int(ps.gaussian_window_max),
                gaussian_boundary_scale=float(ps.gaussian_boundary_scale),
                boundary_padding=int(ps.boundary_padding),
                prominence_ratio=float(ps.prominence_ratio),
                smoothing_window=int(ps.smoothing_window),
                smoothing_poly_order=int(ps.smoothing_poly_order),
                baseline_window=int(ps.baseline_window),
                baseline_percentile=float(ps.baseline_percentile),
                min_peak_width=int(ps.min_peak_width),
                max_peak_width=int(ps.max_peak_width),
                cwt_snr_threshold=float(ps.cwt_snr_threshold),
                cwt_wavelet_max_width=int(ps.cwt_wavelet_max_width),
                weak_tail_cutoff_idx=int(ps.weak_tail_cutoff_idx),
                vote_threshold=float(ps.vote_threshold),
                min_intensity_for_single_vote=float(ps.min_intensity_for_single_vote),
                mz_tolerance=float(ps.mz_tolerance),
            )
            save_peak_detection_config(peak_config)
        except Exception:
            pass

    def _load_project_settings_to_parameter_widgets(self, ps: ProjectSettings) -> None:
        if hasattr(self, "project_common_parameters_widget"):
            self.project_common_parameters_widget.set_project_settings(ps)
        if hasattr(self, "project_function_defaults_widget"):
            self.project_function_defaults_widget.set_project_settings(ps)
        if hasattr(self, "project_peak_detection_widget"):
            self.project_peak_detection_widget.set_project_settings(ps)

    def _apply_project_runtime_settings(self, ps: ProjectSettings) -> None:
        """Make the active desktop runtime use the project file as source of truth."""
        self.normalization_settings = ps.to_normalization_settings()
        calibration = ps.to_calibration()
        previous_plot_calibration = getattr(self, "current_plot_calibration", None)
        self.lineEdit_4.setText(f"{calibration.a:.6e}")
        self.lineEdit_5.setText(f"{calibration.b:.6e}")
        self.lineEdit_6.setText(f"{calibration.c:.6e}")
        if hasattr(self, "project_common_parameters_widget"):
            self.project_common_parameters_widget.settings = self.normalization_settings
            self.project_common_parameters_widget.calibration = calibration
        if getattr(self, "current_plot_x", None) is not None and self.current_plot_x.size:
            self.refresh_current_plot_calibration(previous_plot_calibration)
        elif hasattr(self, "p2"):
            self.refresh_plot_axis_mode()

    def load_project_settings(self) -> None:
        """加载项目配置。

        流程：
        1. 检查当前项目路径（从 UI 或 ProjectSettings）
        2. 如果项目配置存在，自动设置项目路径
        3. 加载项目级或全局配置
        4. 同步到所有UI元素
        """
        from pathlib import Path

        # Step 1: 获取当前项目路径
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
        self._load_project_settings_to_parameter_widgets(ps)
        self._apply_project_runtime_settings(ps)
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
        validation_status = get_data_source_validation_status(ps, validation_records)
        workflow_result = analyze_workflow_capabilities(ps, validation_records)

        self.current_data_source_status = validation_status
        self.current_validation_records = validation_records
        self.current_workflow_result = workflow_result

        # 项目未初始化
        root_exists = project_root(ps).exists()
        if not root_exists:
            self._clear_datasource_row_statuses()
            self._refresh_workflow_chips([])
            return

        # 每行内联状态
        record_map = {r.source_key: r for r in validation_records}
        key_map = {
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
                    lbl.setText("未登记")
                    lbl.setStyleSheet("")
                elif not record.is_valid:
                    lbl.setText("[警告] 路径失效")
                    lbl.setStyleSheet("color: #c0392b;")
                else:
                    count_text = record.detail.replace(" ", "") if record.file_count > 1 else "已配置"
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
        """Placeholder method - workflow display removed"""
        pass

    def _collect_function_params_from_ui(self, ps: ProjectSettings) -> None:
        """Deprecated: Use FunctionDefaultsWidget.apply_to_settings() instead.

        This method was deprecated after Phase 3 UI refactoring when function
        parameters were moved to FunctionDefaultsWidget. It now delegates to the
        widget so older call sites still persist the current parameter edits.
        """
        self._collect_all_project_parameters_from_ui(ps)

    def new_project(self) -> None:
        """清空表单，准备创建新项目"""
        self._creating_new_project = True
        self._opened_project_root = None
        self.project_settings_manager.clear_project_path()
        self.project_name_edit.clear()
        self.project_system_edit.clear()
        self.project_description_edit.clear()
        self.project_output_dir_edit.clear()
        self.project_name_edit.setFocus()
        self.statusbar.showMessage("已清空表单，请填写项目信息并点击'保存并应用'", 3000)

    def open_project(self) -> None:
        """打开已有项目（选择项目根目录或配置文件）"""
        project_path = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择项目根目录或包含 config/project.yaml 的文件夹",
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
                "请确保选择的是有效的项目根目录。"
            )
            return

        self.statusbar.showMessage("正在加载项目配置与数据源状态…")
        if hasattr(self, "project_open_button"):
            self.project_open_button.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

        try:
            # Load project settings from config file
            from bl03u_masstool.core.project_settings import load_project_settings
            ps = load_project_settings(config_file)
            self._creating_new_project = False
            self._opened_project_root = project_path.resolve()

            # Set project path for manager
            self.project_settings_manager.set_project_path(project_path)
            self.project_settings_manager.set(ps)

            # Display loaded settings in UI, including data-source paths.
            self._read_project_settings_to_ui(ps)

            # Load to all parameter widgets and desktop runtime.
            self._load_project_settings_to_parameter_widgets(ps)
            self._apply_project_runtime_settings(ps)

            self._apply_settings_to_tools(ps)
            self.update_project_title()
            self.refresh_project_lifecycle(ps)
            self.refresh_project_parameter_summary()
            self.refresh_project_datasource_page(ps)

            if getattr(self, "current_data_source_status", None) == DataSourceValidationStatus.UNCONFIGURED:
                self.statusbar.showMessage(
                    f"✓ 已加载项目：{ps.project_name or project_path.name}；尚未登记数据源，请点击“启动导入向导”",
                    7000,
                )
            else:
                self.statusbar.showMessage(f"✓ 已加载并应用项目：{ps.project_name or project_path.name}", 4000)

        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "加载项目失败",
                f"无法加载项目配置：\n{str(exc)}"
            )
            self.project_settings_manager.clear_project_path()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            if hasattr(self, "project_open_button"):
                self.project_open_button.setEnabled(True)

    def save_and_apply_project_settings(self) -> None:
        """Save project settings, create project structure, and sync to tools.

        This is the complete operation: initialize + apply to all tool pages.
        """
        ps = self._collect_project_settings_from_ui()
        ps = self._normalize_project_output_dir(ps)
        self._collect_all_project_parameters_from_ui(ps)
        self._sync_peak_detection_to_global_config(ps)

        # Step 1: Create project directory structure
        try:
            ensure_project_structure(ps)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "初始化项目失败", str(exc))
            return

        # Step 2: Bring registered raw data sources under the managed project folder.
        try:
            materialize_project_data_sources(ps, mode="copy")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "导入项目数据源失败", str(exc))
            return

        # Step 3: Set project config path (before saving)
        from pathlib import Path
        self.project_settings_manager.set_project_path(Path(ps.output_dir))

        # Step 4: Save configuration to project-specific location
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()
        self._creating_new_project = False
        self._opened_project_root = Path(ps.output_dir).resolve()

        # Step 5: Read settings back to UI
        self._read_project_settings_to_ui(ps)
        self._load_project_settings_to_parameter_widgets(ps)
        self._apply_project_runtime_settings(ps)

        # Step 6: Sync to tools
        self._apply_settings_to_tools(ps)

        # Step 7: Refresh UI
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
        self._collect_all_project_parameters_from_ui(ps)
        self._sync_peak_detection_to_global_config(ps)

        # Create project directory structure
        try:
            ensure_project_structure(ps)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "初始化项目失败", str(exc))
            return

        try:
            materialize_project_data_sources(ps, mode="copy")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "导入项目数据源失败", str(exc))
            return

        # Set project config path (before saving)
        from pathlib import Path
        self.project_settings_manager.set_project_path(Path(ps.output_dir))

        # Save configuration to project-specific location
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()
        self._creating_new_project = False
        self._opened_project_root = Path(ps.output_dir).resolve()

        # Read settings back to UI
        self._read_project_settings_to_ui(ps)
        self._load_project_settings_to_parameter_widgets(ps)
        self._apply_project_runtime_settings(ps)

        # Refresh UI (but don't sync to tools)
        self.update_project_title()
        self.refresh_project_lifecycle(ps)
        self.refresh_project_parameter_summary()
        self.statusbar.showMessage(f"✓ 项目已初始化：{project_root(ps)}", 3000)

    def apply_project_settings_to_tools(self) -> None:
        """Sync project settings to tool pages (light version, no save/init)."""
        ps = self._sync_project_page_edits_to_runtime(save_project=False, sync_tools=False)
        self._apply_settings_to_tools(ps)
        self.statusbar.showMessage("项目设置已应用到工具", 3000)

    def _apply_settings_to_tools(self, ps: ProjectSettings) -> None:
        """Internal method: sync project settings to all tool pages."""
        self._apply_project_runtime_settings(ps)
        if hasattr(self, "apply_project_spectrum_paths"):
            self.apply_project_spectrum_paths(ps, activate=True)
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
        if hasattr(self, "apply_project_spectrum_paths"):
            self.apply_project_spectrum_paths(ps)

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
        if hasattr(self, "apply_project_spectrum_paths"):
            self.apply_project_spectrum_paths(self._collect_project_settings_from_ui())

    def import_project_datasource(self) -> None:
        """Import a raw data source into the active project in the background."""
        ps = self._collect_project_settings_from_ui()
        ps = self._normalize_project_output_dir(ps)
        try:
            ensure_project_structure(ps)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "初始化项目失败", str(exc))
            return

        self.project_settings_manager.set_project_path(Path(ps.output_dir))
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()

        source_items = [
            ("温度扫描目录", "temperature_scan"),
            ("PIE扫描目录", "pie_scan"),
            ("手动卡峰文件", "manual_peak"),
        ]
        labels = [label for label, _source_key in source_items]
        label, ok = QtWidgets.QInputDialog.getItem(
            self,
            "导入项目数据源",
            "选择要导入的数据源类型：",
            labels,
            0,
            False,
        )
        if not ok or not label:
            return

        source_key = dict(source_items)[label]
        start_dir = self._dialog_start_dir(ps.output_dir)
        if source_key == "manual_peak":
            source_path, _ = QFileDialog.getOpenFileName(
                self,
                f"选择{label}",
                start_dir,
                "Peak Files (*.yaml *.yml *.csv *.xlsx *.xls);;所有文件 (*)",
            )
        else:
            source_path = QFileDialog.getExistingDirectory(self, f"选择{label}", start_dir)
        if not source_path:
            return

        if not hasattr(self, "_worker_manager"):
            self._worker_manager = WorkerManager(self)

        worker = ImportWorker(ps, source_path, source_key, mode="copy")
        worker.progress.connect(self._on_import_progress)
        worker.finished.connect(self._on_import_finished)
        worker.error.connect(self._on_import_error)
        worker.cancelled.connect(self._on_import_cancelled)

        self._import_progress_dialog = ProgressDialog(self, "导入项目数据源")
        self._import_progress_dialog.rejected.connect(lambda: self._worker_manager.cancel())
        self._import_progress_dialog.show()

        self.statusbar.showMessage(f"正在导入{label}…")
        self._worker_manager.run_worker(worker)

    def _on_import_progress(self, percent: int, message: str) -> None:
        if hasattr(self, "_import_progress_dialog"):
            self._import_progress_dialog.update(percent, message)

    def _on_import_finished(self, result: dict) -> None:
        if hasattr(self, "_import_progress_dialog"):
            self._import_progress_dialog.close()

        ps = self.project_settings_manager.get()
        field_name = result.get("field_name", "")
        destination = result.get("destination", "")
        if field_name and destination:
            setattr(ps, field_name, destination)
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()

        self._read_project_settings_to_ui(ps)
        self._apply_settings_to_tools(ps)
        self.update_project_title()
        self.refresh_project_lifecycle(ps)
        self.refresh_project_parameter_summary()
        self.refresh_project_datasource_page(ps)

        label = result.get("label") or "数据源"
        action = "链接" if result.get("mode") == "link" else "导入"
        self.statusbar.showMessage(f"✓ {label}已{action}并登记：{destination}", 5000)

    def _on_import_error(self, message: str) -> None:
        if hasattr(self, "_import_progress_dialog"):
            self._import_progress_dialog.close()
        QtWidgets.QMessageBox.critical(self, "导入数据源失败", message)
        self.statusbar.showMessage("导入数据源失败", 5000)

    def _on_import_cancelled(self) -> None:
        if hasattr(self, "_import_progress_dialog"):
            self._import_progress_dialog.close()
        self.statusbar.showMessage("数据源导入已取消", 3000)

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

    def select_project_output_parent_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择项目父目录",
            self._dialog_start_dir(self.project_output_dir_edit.text()),
        )
        if folder:
            self._creating_new_project = True
            self.project_output_dir_edit.setText(folder)

    def import_project_manual_peak_file(self) -> None:
        ps = self._collect_project_settings_from_ui()
        ps = self._normalize_project_output_dir(ps)
        try:
            ensure_project_structure(ps)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "初始化项目失败", str(exc))
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入手动卡峰文件",
            self._dialog_start_dir(self.project_manual_peak_edit.text() or ps.output_dir),
            "Peak Files (*.yaml *.yml *.csv *.xlsx *.xls);;所有文件 (*)",
        )
        if not path:
            return

        try:
            result = import_project_source(ps, path, "manual_peak", mode="copy")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "导入手动卡峰文件失败", str(exc))
            return

        self.project_settings_manager.set_project_path(Path(ps.output_dir))
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()

        self._read_project_settings_to_ui(ps)
        self._apply_settings_to_tools(ps)
        self.update_project_title()
        self.refresh_project_lifecycle(ps)
        self.refresh_project_parameter_summary()
        self.refresh_project_datasource_page(ps)
        self.statusbar.showMessage(f"✓ 手动卡峰文件已导入并登记：{result.destination}", 5000)

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
        """Resolve the project root before initialization.

        In new-project mode the UI field is a storage parent folder; saving
        creates <parent>/<project name>. Existing project roots are preserved
        only after open_project(); stale project markers in a parent folder
        must not make a new project write analysis/config directly there.
        """
        raw_output = ps.output_dir.strip() if ps.output_dir else ""
        if (not raw_output or raw_output == "output") and (ps.project_name or ps.system):
            ps.output_dir = self._default_project_folder(ps)
        elif ps.project_name or ps.system:
            output_path = Path(raw_output).expanduser()
            project_slug = self._project_folder_name(ps)
            default_name = self._project_default_folder_name(ps)
            matching_names = {project_slug, default_name}
            opened_root = getattr(self, "_opened_project_root", None)
            explicit_opened_root = False
            if opened_root is not None:
                try:
                    explicit_opened_root = output_path.resolve() == Path(opened_root).resolve()
                except OSError:
                    explicit_opened_root = False
            if (
                output_path.exists()
                and output_path.is_dir()
                and output_path.name not in matching_names
                and not explicit_opened_root
            ):
                ps.output_dir = str(output_path / project_slug)

        ps.output_dir = str(project_root(ps))
        self.project_output_dir_edit.setText(ps.output_dir)
        return ps

    def _collect_and_save_project_settings(self) -> ProjectSettings:
        ps = self._collect_project_settings_from_ui()
        self._collect_all_project_parameters_from_ui(ps)
        # Set project config path before saving
        from pathlib import Path
        if ps.output_dir and ps.output_dir.strip() != "output":
            self.project_settings_manager.set_project_path(Path(ps.output_dir))
        self.project_settings_manager.set(ps)
        self.project_settings_manager.save()
        self._apply_project_runtime_settings(ps)
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
            self.statusbar.showMessage("全部阶段均已有记录", 4000)
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
        pass

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
        ps = self.project_settings_manager.get()
        self._apply_project_runtime_settings(ps)
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
        self.refresh_project_parameter_summary()
        self.statusbar.showMessage("通用参数已保存并同步到各工具", 3000)

    def switch_workspace_page(self, page_name: str):
        """Switch top-level workspace page and refresh shared calibration state.

        When leaving the project page, saved edits are synced to all tool pages via
        _sync_project_page_edits_to_runtime (which calls _sync_project_settings_to_tool_pages).
        When switching between non-project tool pages, only shared state (calibration,
        normalization) is refreshed -- function-specific settings are not re-applied
        to avoid overwriting unsaved edits on those pages.
        """
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
        current_page = self.workspace_stack.currentWidget() if hasattr(self, "workspace_stack") else None
        if current_page is getattr(self, "project_page", None):
            # Leaving project page: save edits; sync_tools=True triggers
            # _sync_project_settings_to_tool_pages, which calls set_project_settings
            # on all tool pages with the freshly saved state.
            ps = self._sync_project_page_edits_to_runtime(save_project=True, sync_tools=True)
        elif page_name == "project":
            self.load_project_settings()
            ps = self.project_settings_manager.get()
        else:
            ps = self.project_settings_manager.get()
        if self.project_settings_manager.has_project_path():
            self._apply_project_runtime_settings(ps)
        else:
            self.apply_config_defaults()
            self.normalization_settings = load_normalization_settings()
        calibration = self.current_calibration()
        # Refresh shared calibration/normalization on all tool pages without
        # overwriting function-specific unsaved edits via set_project_settings.
        if hasattr(self, "temperature_page"):
            self.temperature_page.normalization_settings = self.normalization_settings
            self.temperature_page.calibration = calibration
        if hasattr(self, "pie_page"):
            self.pie_page.normalization_settings = self.normalization_settings
            self.pie_page.calibration = calibration
        if hasattr(self, "mole_fraction_page"):
            self.mole_fraction_page.normalization_settings = self.normalization_settings
            self.mole_fraction_page.calibration = calibration
        if hasattr(self, "pics_page"):
            self.pics_page.normalization_settings = self.normalization_settings
            self.pics_page.calibration = calibration
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
        if hasattr(self, "project_peak_detection_widget"):
            ps = ProjectSettingsManager().get()
            self.project_peak_detection_widget.set_project_settings(ps)
        self.switch_workspace_page("project")
        if hasattr(self, "project_tabs"):
            self.project_tabs.setCurrentWidget(self.project_common_parameters_widget)
