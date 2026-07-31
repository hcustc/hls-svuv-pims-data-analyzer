from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import logging
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout

from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.core.curve_database import (
    backfill_curve_peak_set_provenance,
    invalidate_curve_datasets_for_peak_source,
    project_curve_database_path,
)
from bl03u_masstool.core.config import (
    resolve_species_database_path,
)
from bl03u_masstool.core.peak_sets import (
    activate_peak_set,
    get_peak_set,
    import_peak_set,
    migrate_legacy_peak_file,
    normalize_peak_set_registry_paths,
    resolve_peak_set_path,
    verify_peak_set,
)
from bl03u_masstool.core.project_peak_generation import generate_project_peak_set
from bl03u_masstool.core.project_lifecycle import (
    DataSourceValidationStatus,
    ensure_project_structure,
    get_data_source_validation_status,
    project_root,
    sanitize_project_slug,
    validate_all_data_sources,
)
from bl03u_masstool.core.project_settings import (
    ProjectSettings,
    ProjectSettingsManager,
    derive_project_compatibility_fields,
    load_factory_project_settings,
)
from bl03u_masstool.frontends.pyqt_app.isotope.dialog import IsotopeAbundanceDialog
from bl03u_masstool.frontends.pyqt_app.isotope_correction.dialog import IsotopeCorrectionDialog
from bl03u_masstool.frontends.pyqt_app.progress_dialog import ProgressDialog
from bl03u_masstool.frontends.pyqt_app.worker import ImportWorker
from bl03u_masstool.frontends.pyqt_app.worker_manager import WorkerManager
from bl03u_masstool.frontends.pyqt_app.workers import WorkerThread
from bl03u_masstool.frontends.pyqt_app.mole_fraction.dialog import MoleFractionDialog
from bl03u_masstool.frontends.pyqt_app.nist.widget import IonizationEnergyLookupWidget
from bl03u_masstool.frontends.pyqt_app.normalization.widget import (
    AutoSelectDoubleSpinBox,
    CommonParametersWidget,
)
from bl03u_masstool.frontends.pyqt_app.pics.dialog import PICSCalculatorDialog
from bl03u_masstool.frontends.pyqt_app.pics.import_widget import PICSImportWidget
from bl03u_masstool.frontends.pyqt_app.pie.dialog import PIESpeciesFitDialog
from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog


logger = logging.getLogger(__name__)



class WorkspacePagesMixin:
    def _build_workspace_pages(self):
        if hasattr(self, "workspace_stack"):
            return

        self.verticalLayout_9.removeItem(self.verticalLayout_7)

        self.workspace_shell = QtWidgets.QWidget(self.centralwidget)
        self.workspace_shell.setObjectName("WorkspaceShell")
        shell_layout = QVBoxLayout(self.workspace_shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(8)

        self.page_nav = QtWidgets.QWidget(self.workspace_shell)
        self.page_nav.setObjectName("PageNav")
        self.page_nav.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        nav_layout = QHBoxLayout(self.page_nav)
        nav_layout.setContentsMargins(8, 6, 8, 6)
        nav_layout.setSpacing(6)

        self.workspace_stack = QtWidgets.QStackedWidget(self.workspace_shell)
        self.workspace_stack.setObjectName("WorkspaceStack")

        self.spectrum_page = QtWidgets.QWidget(self.workspace_stack)
        self.spectrum_page.setObjectName("SpectrumPage")
        self.spectrum_page.setLayout(self.verticalLayout_7)
        self.workspace_stack.addWidget(self.spectrum_page)

        self.normalization_settings = (
            load_factory_project_settings().to_normalization_settings()
        )
        self.temperature_page = TemperatureScanDialog(
            self.current_calibration(),
            self.normalization_settings,
            self.workspace_stack,
        )
        self.temperature_page.set_project_settings(
            self.project_settings_manager.snapshot(),
            activate_project_scope=self.project_settings_manager.has_project_path(),
            load_cached_results=False,
        )
        self.pie_page = PIESpeciesFitDialog(
            self.current_calibration(),
            self.normalization_settings,
            self.workspace_stack,
        )
        self.pie_page.set_project_settings(
            self.project_settings_manager.snapshot(),
            activate_project_scope=self.project_settings_manager.has_project_path(),
            load_cached_results=False,
        )
        self.isotope_correction_page = IsotopeCorrectionDialog(self.workspace_stack)
        self.isotope_correction_page.set_project_settings(
            self.project_settings_manager.snapshot(),
            activate_project_scope=self.project_settings_manager.has_project_path(),
        )
        self.mole_fraction_page = MoleFractionDialog(
            self.current_calibration(),
            self.normalization_settings,
            self.workspace_stack,
        )
        self.mole_fraction_page.set_project_settings(
            self.project_settings_manager.snapshot(),
            activate_project_scope=self.project_settings_manager.has_project_path(),
        )
        self.ionization_page = IonizationEnergyLookupWidget(self.workspace_stack)
        self.ionization_page.set_project_settings(self.project_settings_manager.snapshot())
        self.isotope_page = IsotopeAbundanceDialog(self.workspace_stack)
        self.isotope_page.set_project_settings(self.project_settings_manager.snapshot())
        self.pics_page = PICSCalculatorDialog(
            self.current_calibration(),
            self.normalization_settings,
            self.workspace_stack,
        )
        self.pics_page.set_project_settings(
            self.project_settings_manager.snapshot(),
            activate_project_scope=self.project_settings_manager.has_project_path(),
        )
        self.pics_import_page = PICSImportWidget(self.workspace_stack)
        self.pics_import_page.set_project_settings(self.project_settings_manager.snapshot())
        self.pics_import_page.import_completed.connect(
            lambda _result: self._refresh_pics_database_consumers()
        )
        self.project_page = QtWidgets.QWidget(self.workspace_stack)
        self.project_page.setObjectName("ProjectPage")
        self._build_project_page()

        pages = [
            ("project", "项目管理", self.project_page),
            ("spectrum", "质谱工作台", self.spectrum_page),
            ("temperature", "温度扫描", self.temperature_page),
            ("pie", "PIE拟合", self.pie_page),
            ("isotope_correction", "同位素贡献校正", self.isotope_correction_page),
            ("mole_fraction", "摩尔分数", self.mole_fraction_page),
            ("pics", "PICS计算", self.pics_page),
            ("pics_import", "PICS导入", self.pics_import_page),
            ("ionization", "IE查询", self.ionization_page),
            ("isotope", "分子式与质量分析", self.isotope_page),
        ]
        page_descriptions = {
            "project": "管理项目、数据源、共享参数和分析进度",
            "spectrum": "查看质谱、标定、寻峰和维护峰范围",
            "temperature": "生成并比较不同能量下的温度响应曲线",
            "pie": "拟合 PIE 曲线并识别候选物种",
            "isotope_correction": "按用户指定分子式后处理 PIE 或温度曲线的同位素贡献",
            "mole_fraction": "基于温扫和 PIE 结果计算物种摩尔分数",
            "pics": "计算物种光电离截面",
            "pics_import": "导入外部 PICS 数据",
            "ionization": "查询物种电离能",
            "isotope": "由分子式计算质量与同位素，或由质量搜索候选分子式",
        }
        page_lookup = {
            page_name: (index, label, page)
            for index, (page_name, label, page) in enumerate(pages)
        }
        nav_groups = [
            ("项目", ("project",)),
            ("数据处理", ("spectrum", "temperature", "pie", "isotope_correction", "mole_fraction")),
            ("PICS 与资料", ("pics", "pics_import", "ionization", "isotope")),
        ]

        self.page_buttons: dict[str, QtWidgets.QToolButton] = {}
        self.page_button_group = QtWidgets.QButtonGroup(self.page_nav)
        self.page_button_group.setExclusive(True)
        for page_name, _label, page in pages:
            if page is not self.spectrum_page:
                self.workspace_stack.addWidget(page)

        for group_index, (_group_title, group_pages) in enumerate(nav_groups):
            if group_index > 0:
                separator = QtWidgets.QFrame(self.page_nav)
                separator.setObjectName("NavSeparator")
                separator.setFrameShape(QtWidgets.QFrame.Shape.VLine)
                separator.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
                nav_layout.addWidget(separator)
            for page_name in group_pages:
                index, label, _page = page_lookup[page_name]
                button = QtWidgets.QToolButton(self.page_nav)
                button.setText(label)
                button.setObjectName("WorkspaceTab")
                button.setCheckable(True)
                button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
                button.setMinimumHeight(34)
                shortcut_text = f"Ctrl+{index + 1}"
                button.setToolTip(f"{page_descriptions[page_name]}（{shortcut_text}）")
                button.setAccessibleName(label)
                button.setAccessibleDescription(page_descriptions[page_name])
                button.setSizePolicy(
                    QtWidgets.QSizePolicy.Policy.Expanding,
                    QtWidgets.QSizePolicy.Policy.Fixed,
                )
                button.clicked.connect(lambda checked=False, name=page_name: self.switch_workspace_page(name))
                self.page_button_group.addButton(button, index)
                self.page_buttons[page_name] = button
                nav_layout.addWidget(button)
                shortcut = QtGui.QShortcut(QtGui.QKeySequence(shortcut_text), self)
                shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
                shortcut.activated.connect(lambda name=page_name: self.switch_workspace_page(name))
        self.page_buttons["spectrum"].setChecked(True)
        shell_layout.addWidget(self.page_nav)
        shell_layout.addWidget(self.workspace_stack, stretch=1)
        self.verticalLayout_9.insertWidget(0, self.workspace_shell, stretch=1)
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
        identity_layout.addStretch(1)

        settings_scroll.setWidget(settings_content)
        settings_page_layout = QVBoxLayout(self.project_identity_page)
        settings_page_layout.setContentsMargins(0, 0, 0, 0)
        settings_page_layout.addWidget(settings_scroll)

        # --- Tab 2: 定标与卡峰 ---
        self.project_baseline_page = QtWidgets.QWidget(self.project_tabs)
        self._build_project_baseline_page(self.project_baseline_page)

        # --- Tab 3: 项目分析参数 ---
        self.project_analysis_page = QtWidgets.QWidget(self.project_tabs)
        analysis_layout = QVBoxLayout(self.project_analysis_page)
        analysis_layout.setContentsMargins(6, 6, 6, 6)
        analysis_layout.setSpacing(8)

        self.project_common_parameters_widget = CommonParametersWidget(
            self.normalization_settings,
            self.current_calibration(),
            self.project_analysis_page,
            show_actions=False,
            show_calibration=False,
            show_kr_expansion=False,
            show_mass_response=False,
            show_element_filter=False,
            embedded=True,
        )
        self.project_common_parameters_widget.settings_saved.connect(
            self.on_project_common_parameters_saved
        )

        from bl03u_masstool.frontends.pyqt_app.normalization.widget import FunctionDefaultsWidget

        self.project_function_defaults_widget = FunctionDefaultsWidget(
            self.project_analysis_page,
            show_actions=False,
        )
        self.project_function_defaults_widget.settings_saved.connect(
            self.on_project_common_parameters_saved
        )
        self.project_function_defaults_widget.navigate_requested.connect(
            self.switch_workspace_page
        )

        # Keep FunctionDefaultsWidget's tab hierarchy intact.  Moving pages out
        # of a QTabWidget leaves their hidden state behind and caused the short
        # normalization card to be vertically centred in an otherwise empty
        # page.  The common card now lives directly at the top of the existing
        # "通用分析" page.
        self.project_analysis_tabs = (
            self.project_function_defaults_widget.tabs
        )
        self.project_common_analysis_page = self.project_analysis_tabs.widget(
            0
        )
        common_analysis_layout = self.project_common_analysis_page.layout()
        common_analysis_layout.insertWidget(
            0,
            self.project_common_parameters_widget,
        )
        analysis_layout.addWidget(
            self.project_function_defaults_widget,
            stretch=1,
        )
        analysis_action = QHBoxLayout()
        analysis_action.addStretch(1)
        self.project_analysis_save_button = QPushButton(
            "保存并应用项目参数",
            self.project_analysis_page,
        )
        self.project_analysis_save_button.setObjectName("PrimaryButton")
        self.project_analysis_save_button.clicked.connect(
            self.save_and_apply_project_settings
        )
        analysis_action.addWidget(self.project_analysis_save_button)
        self.project_analysis_unsaved_hint = QtWidgets.QLabel(
            "离开项目管理时如有未保存修改，将提示保存、放弃或取消。",
            self.project_analysis_page,
        )
        self.project_analysis_unsaved_hint.setObjectName("HintLabel")
        analysis_action.insertWidget(0, self.project_analysis_unsaved_hint)
        analysis_layout.addLayout(analysis_action)

        self.project_tabs.addTab(self.project_identity_page, "项目与数据")
        self.project_tabs.addTab(self.project_baseline_page, "定标与卡峰")
        self.project_tabs.addTab(self.project_analysis_page, "项目分析参数")
        self.project_tabs.currentChanged.connect(self._on_project_tab_changed)
        page_layout.addWidget(self.project_tabs, stretch=1)

        self.load_project_settings()
        if (
            self.project_settings_manager.has_project_path()
            and hasattr(self, "open_project_peak_ranges")
        ):
            self.open_project_peak_ranges(
                settings=self.project_settings_manager.snapshot(),
                prompt_before_replace=False,
                show_feedback=False,
            )

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
        hint = QtWidgets.QLabel("创建或打开项目，并统一管理项目目录和数据来源。", hero)
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
        self.project_new_button.setObjectName("BrowseButton")
        self.project_new_button.setToolTip("清空当前表单，开始创建新项目")
        self.project_new_button.setFixedHeight(32)
        action_layout.addWidget(self.project_new_button)

        # 打开项目按钮
        self.project_open_button = QPushButton("打开项目", action_bar)
        self.project_open_button.setObjectName("BrowseButton")
        self.project_open_button.setToolTip("打开已有项目配置文件")
        self.project_open_button.setFixedHeight(32)
        action_layout.addWidget(self.project_open_button)

        self.project_close_button = QPushButton("关闭项目", action_bar)
        self.project_close_button.setObjectName("BrowseButton")
        self.project_close_button.setToolTip("关闭当前项目，不删除项目文件")
        self.project_close_button.setFixedHeight(32)
        action_layout.addWidget(self.project_close_button)
        action_layout.addSpacing(12)

        self.project_save_and_apply_button = QPushButton("保存项目", action_bar)

        self.project_save_and_apply_button.setToolTip(
            "保存项目身份并创建项目结构；不会复制任何外部数据。"
        )

        self.project_save_and_apply_button.setFixedHeight(32)
        action_layout.addWidget(self.project_save_and_apply_button)

        self.project_save_and_apply_button.setObjectName("PrimaryButton")
        hero_layout.addWidget(action_bar)
        card_layout.addWidget(hero)

        form_layout = QtWidgets.QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(8)
        form_layout.setVerticalSpacing(6)

        self.project_name_edit = QLineEdit(self.project_identity_card)
        self.project_name_edit.setPlaceholderText("输入项目名称")
        self.project_system_edit = QLineEdit(self.project_identity_card)
        self.project_system_edit.setPlaceholderText("输入样品、体系或实验代号")
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
        self.project_close_button.clicked.connect(self.close_current_project)
        self.project_save_and_apply_button.clicked.connect(self.save_project)
        self.project_output_dir_button.clicked.connect(
            self.select_project_output_parent_folder
        )

    # ── Project data sources ─────────────────────────────────────────────

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
            "这里只显示项目内已登记的数据。外部数据只能通过“导入项目数据”复制并登记。",
            self.datasource_card,
        )
        datasource_hint.setObjectName("ProjectHint")
        datasource_hint.setWordWrap(True)

        title_column.addWidget(datasource_title)
        title_column.addWidget(datasource_hint)
        top_bar.addLayout(title_column, stretch=1)

        self.datasource_import_button = QPushButton("导入项目数据", self.datasource_card)
        self.datasource_import_button.setObjectName("PrimaryButton")
        self.datasource_import_button.setToolTip("选择原始数据源，并复制/登记到当前项目")
        self.datasource_import_button.setFixedHeight(30)
        top_bar.addWidget(self.datasource_import_button)
        card_layout.addLayout(top_bar)

        # 分割线
        sep = QtWidgets.QFrame(self.datasource_card)
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setObjectName("PanelSeparator")
        card_layout.addWidget(sep)

        # ── 路径配置区：每行含内联状态 ──
        self.datasource_row_status_labels: dict[str, QtWidgets.QLabel] = {}

        self.project_temperature_folder_edit = QLineEdit(self.datasource_card)
        self.project_temperature_folder_edit.setPlaceholderText("尚未导入温度扫描数据")
        self.project_temperature_folder_edit.setReadOnly(True)
        self.project_pie_folder_edit = QLineEdit(self.datasource_card)
        self.project_pie_folder_edit.setPlaceholderText("尚未选择 PIE 扫描目录")
        self.project_pie_folder_edit.setReadOnly(True)
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
        _add_path_row(analysis_layout, 1, "pie_scan", "PIE主目录", self.project_pie_folder_edit, self.project_pie_folder_button)

        self.project_pie_folders_list = QtWidgets.QListWidget(self.datasource_card)
        self.project_pie_folders_list.setMaximumHeight(92)
        self.project_pie_folders_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.project_pie_folders_list.setToolTip(
            "项目中的 PIE 能段目录。计算时会按实际光子能量排序，并用重叠能点的中位强度比缩放后续能段。"
        )
        pie_segment_label = QtWidgets.QLabel("PIE能段列表", self.datasource_card)
        pie_segment_label.setFixedWidth(88)
        analysis_layout.addWidget(pie_segment_label, 2, 0)
        analysis_layout.addWidget(self.project_pie_folders_list, 2, 1)

        pie_segment_actions = QtWidgets.QWidget(self.datasource_card)
        pie_segment_actions_layout = QVBoxLayout(pie_segment_actions)
        pie_segment_actions_layout.setContentsMargins(0, 0, 0, 0)
        pie_segment_actions_layout.setSpacing(4)
        self.project_pie_add_folder_button = QPushButton("添加能段", pie_segment_actions)
        self.project_pie_add_folder_button.setObjectName("BrowseButton")
        self.project_pie_remove_folder_button = QPushButton("移除", pie_segment_actions)
        self.project_pie_remove_folder_button.setObjectName("BrowseButton")
        self.project_pie_clear_folders_button = QPushButton("清空", pie_segment_actions)
        self.project_pie_clear_folders_button.setObjectName("BrowseButton")
        for button in (
            self.project_pie_add_folder_button,
            self.project_pie_remove_folder_button,
            self.project_pie_clear_folders_button,
        ):
            button.setFixedHeight(24)
            pie_segment_actions_layout.addWidget(button)
        analysis_layout.addWidget(pie_segment_actions, 2, 2)
        card_layout.addWidget(analysis_group)

        artifact_group, artifact_layout = _path_group("当前项目卡峰范围")
        self.project_baseline_artifact_group = artifact_group
        self.project_manual_peak_button = _browse_btn()
        self.project_manual_peak_button.setText("导入")
        _add_path_row(
            artifact_layout,
            0,
            "manual_peak",
            "当前卡峰范围",
            self.project_manual_peak_edit,
            self.project_manual_peak_button,
        )
        self.project_peak_set_generate_button = QPushButton(
            "从累计谱自动生成",
            self.datasource_card,
        )
        self.project_peak_set_generate_button.setObjectName("BrowseButton")
        self.project_peak_set_generate_button.setFixedHeight(26)
        self.project_peak_set_generate_button.setMaximumWidth(150)
        generation_label = QtWidgets.QLabel("创建卡峰范围", self.datasource_card)
        generation_label.setFixedWidth(88)
        generation_hint = QtWidgets.QLabel(
            "也可以在质谱工作台调整后保存到项目；历史快照由软件自动保留",
            self.datasource_card,
        )
        generation_hint.setObjectName("ProjectHint")
        generation_hint.setWordWrap(True)
        artifact_layout.addWidget(generation_label, 1, 0)
        artifact_layout.addWidget(self.project_peak_set_generate_button, 1, 1)
        artifact_layout.addWidget(generation_hint, 1, 2, 1, 2)
        self.datasource_import_button.clicked.connect(self.import_project_datasource)
        self.project_temperature_folder_button.clicked.connect(
            lambda: self.import_project_datasource("temperature_scan")
        )
        self.project_pie_folder_button.clicked.connect(
            lambda: self.import_project_datasource("pie_scan")
        )
        self.project_pie_add_folder_button.clicked.connect(
            lambda: self.import_project_datasource("pie_scan")
        )
        self.project_pie_remove_folder_button.clicked.connect(self.remove_project_pie_folders)
        self.project_pie_clear_folders_button.clicked.connect(self.clear_project_pie_folders)
        self.project_manual_peak_button.clicked.connect(self.import_project_manual_peak_file)
        self.project_peak_set_generate_button.clicked.connect(
            self.generate_project_peak_set_from_sum
        )

    def _build_project_baseline_page(self, parent: QtWidgets.QWidget) -> None:
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        calibration_group = QtWidgets.QGroupBox(
            "项目质量定标  m/z = A·x² + B·x + C",
            parent,
        )
        calibration_layout = QtWidgets.QGridLayout(calibration_group)
        self.project_calibration_edits = []
        for column, (label, name) in enumerate(
            (("A（二次项）", "a"), ("B（一次项）", "b"), ("C（常数项）", "c"))
        ):
            edit = AutoSelectDoubleSpinBox(calibration_group)
            edit.setObjectName(f"ProjectCalibration{name.upper()}Edit")
            edit.setRange(-1_000_000, 1_000_000)
            edit.setDecimals(18)
            edit.setSingleStep(0.000000001)
            edit.setButtonSymbols(
                QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons
            )
            calibration_layout.addWidget(QtWidgets.QLabel(label), 0, column)
            calibration_layout.addWidget(edit, 1, column)
            calibration_layout.setColumnStretch(column, 1)
            self.project_calibration_edits.append(edit)

        self.project_calibration_points_status = QtWidgets.QLabel(
            "定标点：0 个（由质谱工作台维护）",
            calibration_group,
        )
        self.project_calibration_points_status.setObjectName("ProjectHint")
        calibration_layout.addWidget(
            self.project_calibration_points_status,
            2,
            0,
            1,
            3,
        )
        layout.addWidget(calibration_group)
        layout.addWidget(self.project_baseline_artifact_group)
        layout.addStretch(1)

        action = QHBoxLayout()
        action.addStretch(1)
        self.project_baseline_save_button = QPushButton(
            "保存并应用项目参数",
            parent,
        )
        self.project_baseline_save_button.setObjectName("PrimaryButton")
        self.project_baseline_save_button.clicked.connect(
            self.save_and_apply_project_settings
        )
        action.addWidget(self.project_baseline_save_button)
        layout.addLayout(action)

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
        self._set_project_pie_folders(ps.effective_pie_scan_folders())
        self.project_manual_peak_edit.setText(ps.manual_peak_file)
        if hasattr(self, "project_calibration_edits"):
            for edit, value in zip(
                self.project_calibration_edits,
                (ps.cal_a, ps.cal_b, ps.cal_c),
            ):
                edit.setValue(value)
            self.project_calibration_points_status.setText(
                f"定标点：{len(ps.calibration_points)} 个（由质谱工作台维护）"
            )
        self._refresh_project_peak_sets(ps)

    def _migrate_project_peak_set_if_needed(
        self,
        ps: ProjectSettings,
    ) -> ProjectSettings:
        """Make a legacy project peak path project-owned and versioned."""
        if not self.project_settings_manager.has_project_path():
            return ps
        root = project_root(ps)
        normalize_peak_set_registry_paths(root)
        changed = False
        record = None
        if ps.active_peak_set_id:
            record, approved_path = activate_peak_set(root, ps.active_peak_set_id)
            configured = (
                resolve_peak_set_path(root, ps.manual_peak_file)
                if ps.manual_peak_file
                else None
            )
            if configured is None or configured != approved_path:
                ps.manual_peak_file = str(approved_path)
                changed = True
        elif ps.manual_peak_file:
            record = migrate_legacy_peak_file(
                root,
                ps.manual_peak_file,
                label=f"旧项目迁移 · {Path(ps.manual_peak_file).stem}",
            )
            if record is not None:
                approved_path = verify_peak_set(root, record)
                ps.active_peak_set_id = record.peak_set_id
                ps.manual_peak_file = str(approved_path)
                ps.temp_peak_source = "manual"
                changed = True

        if record is not None:
            backfill_curve_peak_set_provenance(
                project_curve_database_path(ps),
                peak_set_id=record.peak_set_id,
                peak_set_sha256=record.sha256,
                peak_set_origin=record.origin,
            )
        if changed:
            ps = self.project_settings_manager.replace_and_save(ps)
        return ps

    def _refresh_project_peak_sets(self, ps: ProjectSettings) -> None:
        """Refresh the single current peak-range status.

        Historical snapshots remain in the registry for provenance, but are
        intentionally not exposed as project files for users to manage.
        """
        status = self.datasource_row_status_labels.get("manual_peak")
        if status is None:
            return
        if not ps.manual_peak_file:
            status.setText("未设置")
            return
        try:
            record = get_peak_set(project_root(ps), ps.active_peak_set_id)
            if record is None:
                raise ValueError("missing active peak snapshot")
            verify_peak_set(project_root(ps), record)
            status.setText("✓ 当前")
        except (FileNotFoundError, ValueError):
            status.setText("⚠ 不可用")

    def _collect_project_settings_from_ui(self) -> ProjectSettings:
        """Build a ProjectSettings from all UI fields (does not save)."""
        ps = deepcopy(
            getattr(self, "_project_draft", None)
            or self.project_settings_manager.snapshot()
        )
        ps.project_name = self.project_name_edit.text().strip()
        ps.system = self.project_system_edit.text().strip()
        ps.description = self.project_description_edit.text().strip()
        ps.output_dir = self.project_output_dir_edit.text().strip() or "output"
        ps.temperature_scan_folder = self.project_temperature_folder_edit.text().strip()
        pie_folders = self._project_pie_folders_from_ui()
        if not (
            not ps.pie_scan_folders
            and pie_folders == ([ps.pie_scan_folder] if ps.pie_scan_folder else [])
        ):
            ps.pie_scan_folders = pie_folders
            ps.pie_scan_folder = pie_folders[0] if pie_folders else ""
        ps.pie_multi_folder_mode = len(pie_folders) > 1
        if hasattr(self, "project_calibration_edits"):
            ps.cal_a, ps.cal_b, ps.cal_c = (
                edit.value() for edit in self.project_calibration_edits
            )
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
        return derive_project_compatibility_fields(ps)

    def _sync_project_page_edits_to_runtime(
        self,
        *,
        save_project: bool = True,
        sync_tools: bool = True,
    ) -> ProjectSettings:
        """Compatibility helper that now updates only the shared project draft."""
        ps = self._collect_project_settings_from_ui()
        ps = self._collect_all_project_parameters_from_ui(ps)
        self._project_draft = deepcopy(ps)
        self._load_project_settings_to_parameter_widgets(self._project_draft)
        return deepcopy(self._project_draft)

    def _on_project_tab_changed(self, _index: int) -> None:
        if not hasattr(self, "project_common_parameters_widget") or not hasattr(self, "project_function_defaults_widget"):
            return
        ps = self._sync_project_page_edits_to_runtime(
            save_project=False,
            sync_tools=False,
        )
        if hasattr(self, "statusbar") and self._project_draft_is_dirty():
            self.statusbar.showMessage("项目参数草稿尚未保存", 2500)
        self.refresh_project_datasource_page(ps)

    def _load_project_settings_to_parameter_widgets(self, ps: ProjectSettings) -> None:
        if hasattr(self, "project_common_parameters_widget"):
            self.project_common_parameters_widget.set_project_settings(ps)
        if hasattr(self, "project_function_defaults_widget"):
            self.project_function_defaults_widget.set_project_settings(ps)
        if hasattr(self, "project_peak_detection_widget"):
            self.project_peak_detection_widget.set_project_settings(ps)

    def _apply_project_runtime_settings(self, ps: ProjectSettings) -> None:
        """Refresh project-owned settings without overwriting temporary spectra."""
        self.normalization_settings = ps.to_normalization_settings()
        calibration = ps.to_calibration()
        _, temporary_reset = self._ensure_temporary_spectrum_settings(
            ps,
            reset_if_context_changed=True,
        )
        if getattr(self, "spectrum_source_scope", "custom") == "project":
            self._activate_spectrum_calibration(
                calibration,
                self._settings_calibration_points(ps),
            )
        elif temporary_reset:
            self._activate_temporary_spectrum_calibration()
        if hasattr(self, "project_common_parameters_widget"):
            self.project_common_parameters_widget.settings = self.normalization_settings
            self.project_common_parameters_widget.calibration = calibration

    def _set_project_draft(
        self,
        settings: ProjectSettings,
        *,
        committed: bool,
    ) -> None:
        self._project_draft = deepcopy(settings)
        if committed:
            self._project_committed = deepcopy(settings)
        self._read_project_settings_to_ui(self._project_draft)
        self._load_project_settings_to_parameter_widgets(self._project_draft)

    def _project_draft_is_dirty(self) -> bool:
        committed = getattr(self, "_project_committed", None)
        draft = getattr(self, "_project_draft", None)
        if committed is None or draft is None:
            return False
        return asdict(draft) != asdict(committed)

    def _confirm_project_draft_resolution(self) -> bool:
        """Resolve unsaved edits before replacing or leaving the project draft."""
        if not hasattr(self, "project_common_parameters_widget"):
            return True
        self._sync_project_page_edits_to_runtime(
            save_project=False,
            sync_tools=False,
        )
        if not self._project_draft_is_dirty():
            return True

        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("项目参数尚未保存")
        box.setText("当前项目草稿包含未保存修改。")
        box.setInformativeText("请选择保存并应用、放弃修改或取消当前操作。")
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Save
            | QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel
        )
        box.button(QtWidgets.QMessageBox.StandardButton.Save).setText(
            "保存并应用"
        )
        box.button(QtWidgets.QMessageBox.StandardButton.Discard).setText(
            "放弃"
        )
        result = box.exec()
        if result == QtWidgets.QMessageBox.StandardButton.Cancel:
            return False
        if result == QtWidgets.QMessageBox.StandardButton.Discard:
            self._set_project_draft(self._project_committed, committed=True)
            return True
        if self.project_settings_manager.has_project_path():
            return bool(self.save_and_apply_project_settings())
        return bool(self.save_project())

    def load_project_settings(self) -> None:
        """Load the active project, or an isolated bundled factory snapshot."""
        ps = self.project_settings_manager.snapshot()
        if self.project_settings_manager.has_project_path():
            try:
                ps = self._migrate_project_peak_set_if_needed(ps)
            except Exception:
                logger.exception("Failed to migrate or verify project peak set")

        # Do NOT auto-fill paths here - only display what's actually saved in config
        # Users must use the import wizard to set up data sources
        self._set_project_draft(ps, committed=True)
        self._apply_project_runtime_settings(ps)
        self.update_project_title()
        self.refresh_project_datasource_page()

    def refresh_project_datasource_page(self, ps: ProjectSettings | None = None) -> None:
        """Refresh data source validation status on the data import page"""
        if ps is None:
            ps = self.project_settings_manager.snapshot()

        validation_records = validate_all_data_sources(ps)
        validation_status = get_data_source_validation_status(ps, validation_records)

        self.current_data_source_status = validation_status
        self.current_validation_records = validation_records

        # 项目未初始化
        root_exists = project_root(ps).exists()
        if not root_exists:
            self._clear_datasource_row_statuses()
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

    def _clear_datasource_row_statuses(self) -> None:
        if hasattr(self, "datasource_row_status_labels"):
            for lbl in self.datasource_row_status_labels.values():
                lbl.setText("—")
                lbl.setStyleSheet("")

    def _switch_tool_pages_to_standalone(self, ps: ProjectSettings) -> None:
        """Refresh tool pages after the project scope is removed."""
        self._load_project_settings_to_parameter_widgets(ps)
        self._temporary_spectrum_settings = None
        self._temporary_spectrum_context_key = None
        self.apply_config_defaults()
        self.normalization_settings = (
            load_factory_project_settings().to_normalization_settings()
        )
        calibration = self.current_calibration()

        if hasattr(self, "set_spectrum_source_scope"):
            self.set_spectrum_source_scope("custom", apply_project=False)
            self.lineEdit.setText(getattr(self, "_custom_single_spectrum_file", ""))
            self.folder_path.setText(getattr(self, "_custom_sum_spectrum_folder", ""))
        self._sync_project_settings_to_tool_pages(
            ps,
            activate_project_scope=False,
            calibration=calibration,
        )

    def _refresh_pics_database_consumers(self) -> None:
        """Reload every desktop consumer after PICS records are imported."""
        ps = self.project_settings_manager.snapshot()
        database_path = resolve_species_database_path(ps.pics_database_path)
        if hasattr(self, "pics_page"):
            self.pics_page.refresh_database()
        if hasattr(self, "pie_page"):
            self.pie_page.load_database(str(database_path), show_message=False)
        if hasattr(self, "mole_fraction_page"):
            self.mole_fraction_page.refresh_database()

    def new_project(self) -> None:
        """清空表单，准备创建新项目"""
        if not self._confirm_project_draft_resolution():
            return
        self._creating_new_project = True
        self._opened_project_root = None
        self.project_settings_manager.clear_project_path()
        ps = load_factory_project_settings()
        self._set_project_draft(ps, committed=True)
        self.project_name_edit.clear()
        self.project_system_edit.clear()
        self.project_description_edit.clear()
        self.project_output_dir_edit.clear()
        self.project_temperature_folder_edit.clear()
        self._set_project_pie_folders([])
        self.project_manual_peak_edit.clear()
        if hasattr(self, "detach_project_peak_ranges"):
            self.detach_project_peak_ranges()
        self._switch_tool_pages_to_standalone(ps)
        self._clear_datasource_row_statuses()
        self.project_name_edit.setFocus()
        self.update_project_title()
        if hasattr(self, "project_close_button"):
            self.project_close_button.setEnabled(False)
        self._sync_project_page_edits_to_runtime(save_project=False, sync_tools=False)
        self.statusbar.showMessage("已创建项目草稿，请填写项目信息并点击“保存项目”", 3000)

    def close_current_project(self) -> None:
        """Close the active project without touching files on disk."""
        if not self._confirm_project_draft_resolution():
            return
        self._creating_new_project = False
        self._opened_project_root = None
        self.project_settings_manager.clear_project_path()
        ps = load_factory_project_settings()
        self._set_project_draft(ps, committed=True)
        self.project_name_edit.clear()
        self.project_system_edit.clear()
        self.project_description_edit.clear()
        self.project_output_dir_edit.clear()
        if hasattr(self, "detach_project_peak_ranges"):
            self.detach_project_peak_ranges()
        self._switch_tool_pages_to_standalone(ps)

        self._clear_datasource_row_statuses()
        self.update_project_title()
        if hasattr(self, "project_close_button"):
            self.project_close_button.setEnabled(False)
        self.statusbar.showMessage("已关闭当前项目，工具页面已切换为临时数据", 3000)

    def open_project(self) -> None:
        """打开已有项目（选择项目根目录或配置文件）"""
        if not self._confirm_project_draft_resolution():
            return
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
            self._creating_new_project = False
            self._opened_project_root = project_path.resolve()

            # The selected folder is authoritative: resolve relative paths and
            # rebase absolute paths from legacy projects to this project root.
            ps = self.project_settings_manager.activate_project(project_path)
            try:
                ps = self._migrate_project_peak_set_if_needed(ps)
            except Exception as exc:
                logger.exception("Failed to migrate or verify project peak set")
                QtWidgets.QMessageBox.warning(
                    self,
                    "项目卡峰范围需要处理",
                    "旧项目卡峰文件未能自动迁移或当前卡峰范围校验失败。\n\n"
                    f"{exc}\n\n"
                    "PIE和温度曲线生成前将要求重新导入或自动生成卡峰范围。",
                )

            # Display loaded settings in UI, including data-source paths.
            self._set_project_draft(ps, committed=True)

            # Load to desktop runtime.
            self._apply_project_runtime_settings(ps)

            self._apply_settings_to_tools(ps)
            peak_ranges_loaded = False
            if hasattr(self, "open_project_peak_ranges"):
                peak_ranges_loaded = self.open_project_peak_ranges(
                    settings=ps,
                    prompt_before_replace=True,
                    show_feedback=False,
                )
            self.update_project_title()
            if hasattr(self, "project_close_button"):
                self.project_close_button.setEnabled(True)
            self.refresh_project_datasource_page(ps)

            if getattr(self, "current_data_source_status", None) == DataSourceValidationStatus.UNCONFIGURED:
                peak_message = "；已自动加载卡峰范围" if peak_ranges_loaded else ""
                self.statusbar.showMessage(
                    f"✓ 已加载项目：{ps.project_name or project_path.name}{peak_message}；"
                    "尚未登记数据源，请点击“导入项目数据”",
                    7000,
                )
            else:
                peak_message = "；已自动加载卡峰范围" if peak_ranges_loaded else ""
                self.statusbar.showMessage(
                    f"✓ 已加载并应用项目：{ps.project_name or project_path.name}"
                    f"{peak_message}",
                    4000,
                )

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

    def save_project(self) -> bool:
        """Create/save project identity without copying any external data."""
        draft = self._collect_project_settings_from_ui()
        draft = self._collect_all_project_parameters_from_ui(draft)
        draft = self._normalize_project_output_dir(draft)
        if not draft.project_name:
            QtWidgets.QMessageBox.warning(self, "项目名为空", "请先填写项目名。")
            return False

        if self.project_settings_manager.has_project_path():
            settings = self.project_settings_manager.snapshot()
            settings.project_name = draft.project_name
            settings.system = draft.system
            settings.description = draft.description
            settings.output_dir = str(
                self.project_settings_manager.get_project_config_path().parent.parent
            )
        else:
            settings = load_factory_project_settings()
            settings.project_name = draft.project_name
            settings.system = draft.system
            settings.description = draft.description
            settings.output_dir = draft.output_dir
        try:
            ensure_project_structure(settings)
            self.project_settings_manager.set_project_path(Path(settings.output_dir))
            saved = self.project_settings_manager.replace_and_save(settings)
        except Exception as exc:
            logger.exception("Failed to save project identity")
            QtWidgets.QMessageBox.critical(self, "保存项目失败", str(exc))
            return False

        self._creating_new_project = False
        self._opened_project_root = Path(saved.output_dir).resolve()
        self._set_project_draft(saved, committed=True)
        self._apply_settings_to_tools(deepcopy(saved))
        self.update_project_title()
        self.refresh_project_datasource_page(saved)
        if hasattr(self, "project_close_button"):
            self.project_close_button.setEnabled(True)
        self.statusbar.showMessage(
            "✓ 项目已保存；尚未复制任何数据，可继续“导入项目数据”",
            5000,
        )
        return True

    def save_and_apply_project_settings(self) -> bool:
        """Atomically save the shared baseline/analysis draft and broadcast it."""
        if not self.project_settings_manager.has_project_path():
            QtWidgets.QMessageBox.warning(
                self,
                "尚未保存项目",
                "请先在“项目与数据”中保存项目，再保存并应用项目参数。",
            )
            return False

        draft = self._collect_project_settings_from_ui()
        draft = self._collect_all_project_parameters_from_ui(draft)
        active_root = (
            self.project_settings_manager.get_project_config_path().parent.parent
        )
        draft.output_dir = str(active_root)
        self._project_draft = deepcopy(draft)
        try:
            saved = self.project_settings_manager.replace_and_save(draft)
        except Exception as exc:
            logger.exception("Failed to save the project parameter transaction")
            QtWidgets.QMessageBox.critical(
                self,
                "保存项目参数失败",
                f"{exc}\n\n已提交配置未改变，当前界面草稿仍保留。",
            )
            return False

        self._set_project_draft(saved, committed=True)
        self._apply_project_runtime_settings(saved)
        self._sync_project_settings_to_tool_pages(
            deepcopy(saved),
            activate_project_scope=None,
        )
        self.update_project_title()
        self.refresh_project_datasource_page(saved)
        self.statusbar.showMessage(
            "✓ 定标、卡峰和分析参数已保存并应用",
            4000,
        )
        return True

    def apply_analysis_settings_from_tool(
        self,
        page: str,
        edited_settings: ProjectSettings,
    ) -> ProjectSettings | None:
        """Persist one tool's whitelisted analysis parameters and broadcast them."""
        from bl03u_masstool.frontends.pyqt_app.temporary_analysis_settings import (
            ANALYSIS_PARAMETER_FIELDS,
        )

        if page not in ANALYSIS_PARAMETER_FIELDS:
            raise ValueError(f"Unknown analysis settings page: {page}")
        if not self.project_settings_manager.has_project_path():
            QtWidgets.QMessageBox.warning(
                self,
                "尚未打开项目",
                "项目参数只能保存到当前活动项目。",
            )
            return None
        patch = {
            field_name: deepcopy(getattr(edited_settings, field_name))
            for field_name in ANALYSIS_PARAMETER_FIELDS[page]
        }
        try:
            saved = self.project_settings_manager.update_module_settings(page, patch)
        except Exception as exc:
            logger.exception("Failed to save %s analysis settings", page)
            QtWidgets.QMessageBox.critical(
                self,
                "保存项目参数失败",
                str(exc),
            )
            return None

        self._set_project_draft(saved, committed=True)
        self._apply_project_runtime_settings(saved)
        self._sync_project_settings_to_tool_pages(
            deepcopy(saved),
            activate_project_scope=None,
        )
        self.update_project_title()
        self.refresh_project_datasource_page(saved)
        label = "温度扫描" if page == "temperature" else "PIE"
        self.statusbar.showMessage(
            f"✓ {label}项目参数已保存并同步",
            4000,
        )
        return deepcopy(saved)

    def _apply_settings_to_tools(self, ps: ProjectSettings) -> None:
        """Internal method: sync project settings to all tool pages."""
        self._apply_project_runtime_settings(ps)
        if hasattr(self, "apply_project_spectrum_paths"):
            self.apply_project_spectrum_paths(ps, activate=True)
        self._sync_project_settings_to_tool_pages(ps, activate_project_scope=True)

    def _sync_project_settings_to_tool_pages(
        self,
        ps: ProjectSettings,
        *,
        activate_project_scope: bool | None = None,
        calibration: Calibration | None = None,
    ) -> None:
        """Sync project settings to all tool pages (temperature, PIE, etc.)."""
        runtime_project_scope = (
            self.project_settings_manager.has_project_path()
            if activate_project_scope is None
            else activate_project_scope
        )
        if calibration is None:
            # ProjectSettings is authoritative only while a project is active;
            # Standalone tools continue to use the current session calibration.
            calibration = (
                ps.to_calibration()
                if runtime_project_scope
                else self.current_calibration()
            )
        source_scope_activation = (
            activate_project_scope if runtime_project_scope else False
        )
        if runtime_project_scope:
            self._invalidate_untraceable_project_curves(ps)
        current_widget = (
            self.workspace_stack.currentWidget()
            if hasattr(self, "workspace_stack")
            else None
        )
        if hasattr(self, "temperature_page"):
            self.temperature_page.normalization_settings = self.normalization_settings
            self.temperature_page.calibration = calibration
            self.temperature_page.set_project_settings(
                deepcopy(ps),
                activate_project_scope=source_scope_activation,
                load_cached_results=current_widget is self.temperature_page,
            )
        if hasattr(self, "pie_page"):
            self.pie_page.normalization_settings = self.normalization_settings
            self.pie_page.calibration = calibration
            self.pie_page.set_project_settings(
                deepcopy(ps),
                activate_project_scope=source_scope_activation,
                load_cached_results=current_widget is self.pie_page,
            )
        if hasattr(self, "isotope_correction_page"):
            self.isotope_correction_page.set_project_settings(
                deepcopy(ps),
                activate_project_scope=runtime_project_scope,
            )
        if hasattr(self, "mole_fraction_page"):
            self.mole_fraction_page.normalization_settings = self.normalization_settings
            self.mole_fraction_page.calibration = calibration
            self.mole_fraction_page.set_project_settings(
                deepcopy(ps),
                activate_project_scope=runtime_project_scope,
            )
        if hasattr(self, "pics_page"):
            self.pics_page.normalization_settings = self.normalization_settings
            self.pics_page.calibration = calibration
            self.pics_page.set_project_settings(
                deepcopy(ps),
                activate_project_scope=runtime_project_scope,
            )
        if hasattr(self, "pics_import_page"):
            self.pics_import_page.set_project_settings(deepcopy(ps))
        if hasattr(self, "ionization_page"):
            self.ionization_page.set_project_settings(deepcopy(ps))
        if hasattr(self, "isotope_page"):
            self.isotope_page.set_project_settings(deepcopy(ps))

    @staticmethod
    def _invalidate_untraceable_project_curves(ps: ProjectSettings) -> dict[str, int]:
        try:
            manual_peak_file = Path(str(ps.manual_peak_file or "")).expanduser()
            if ps.manual_peak_file and not manual_peak_file.is_absolute():
                manual_peak_file = project_root(ps) / manual_peak_file
            peak_set = (
                get_peak_set(project_root(ps), ps.active_peak_set_id)
                if ps.active_peak_set_id
                else None
            )
            return invalidate_curve_datasets_for_peak_source(
                project_curve_database_path(ps),
                manual_peak_file=manual_peak_file if ps.manual_peak_file else None,
                active_peak_set_id=ps.active_peak_set_id,
                peak_set_sha256=peak_set.sha256 if peak_set is not None else "",
            )
        except Exception:
            logger.exception("Failed to validate project curve peak provenance")
            return {}

    def _auto_save_datasource(self) -> None:
        """Refresh the shared project draft from the current data-source UI."""
        self._project_draft = self._collect_project_settings_from_ui()

    def _push_path_to_tool(self, kind: str) -> None:
        """Push a single path field from project management to the corresponding editable tool page.
        Note: Temperature and PIE pages are read-only, so only Spectrum tool is updated."""
        if hasattr(self, "apply_project_spectrum_paths"):
            self.apply_project_spectrum_paths(self._collect_project_settings_from_ui())

    def import_project_datasource(self, source_key: str | None = None) -> None:
        """Import a raw data source into the active project in the background."""
        if not self.project_settings_manager.has_project_path():
            QtWidgets.QMessageBox.warning(
                self,
                "尚未保存项目",
                "请先保存项目；导入操作不会替代项目创建。",
            )
            return
        ps = self.project_settings_manager.snapshot()

        source_items = [
            ("温度扫描目录", "temperature_scan"),
            ("PIE扫描目录", "pie_scan"),
            ("手动卡峰文件", "manual_peak"),
            ("Kr定标扫描目录", "kr_calibration"),
            ("Kr定标卡峰文件", "kr_calibration_peak"),
        ]
        source_labels = {key: label for label, key in source_items}
        if source_key is None:
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
        elif source_key not in source_labels:
            raise ValueError(f"Unknown project source type: {source_key}")
        else:
            label = source_labels[source_key]

        start_dir = self._dialog_start_dir(ps.output_dir)
        if source_key in {"manual_peak", "kr_calibration_peak"}:
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

        self._import_project_config_path = (
            self.project_settings_manager.get_project_config_path()
        )
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

        manager = self.project_settings_manager
        expected_config_path = getattr(self, "_import_project_config_path", None)
        project_is_active = manager.has_project_path()
        if project_is_active and expected_config_path is not None:
            project_is_active = (
                manager.get_project_config_path() == expected_config_path
            )
        self._import_project_config_path = None
        if not project_is_active:
            label = result.get("label") or "数据源"
            self.statusbar.showMessage(
                f"{label}处理已完成，但原项目已关闭或切换，结果未登记",
                6000,
            )
            return

        try:
            ps = manager.snapshot()
            field_name = result.get("field_name", "")
            destination = result.get("destination", "")
            if field_name and destination:
                setattr(ps, field_name, destination)
                if field_name == "manual_peak_file":
                    peak_set = import_peak_set(
                        project_root(ps),
                        destination,
                        label=Path(destination).stem,
                    )
                    ps.active_peak_set_id = peak_set.peak_set_id
                    ps.manual_peak_file = str(
                        verify_peak_set(project_root(ps), peak_set)
                    )
                    ps.temp_peak_source = "manual"
                if field_name == "pie_scan_folder":
                    pie_folders = ps.effective_pie_scan_folders()
                    if destination not in pie_folders:
                        pie_folders.append(destination)
                    ps.pie_scan_folders = pie_folders
                    ps.pie_scan_folder = pie_folders[0]
                    ps.pie_multi_folder_mode = len(pie_folders) > 1
            ps = manager.replace_and_save(ps)
        except Exception as exc:
            logger.exception("Failed to register imported data in the active project")
            QtWidgets.QMessageBox.critical(
                self,
                "登记导入结果失败",
                "数据处理已完成，但无法登记到当前项目。\n"
                f"错误信息：{exc}",
            )
            self.statusbar.showMessage("导入结果未能登记到项目", 6000)
            return
        self._invalidate_untraceable_project_curves(ps)

        data_fields = {
            "temperature_scan_folder",
            "pie_scan_folder",
            "pie_scan_folders",
            "pie_multi_folder_mode",
            "manual_peak_file",
            "active_peak_set_id",
            "kr_calibration_folder",
            "kr_calibration_peak_file",
            "temp_peak_source",
        }
        draft = deepcopy(getattr(self, "_project_draft", ps))
        for name in data_fields:
            setattr(draft, name, deepcopy(getattr(ps, name)))
        self._project_draft = draft
        self._project_committed = deepcopy(ps)
        self._read_project_settings_to_ui(draft)
        self._sync_project_settings_to_tool_pages(deepcopy(ps))
        self.update_project_title()
        self.refresh_project_datasource_page(ps)

        label = result.get("label") or "数据源"
        action = "链接" if result.get("mode") == "link" else "导入"
        self.statusbar.showMessage(f"✓ {label}已{action}并登记：{destination}", 5000)

    def _on_import_error(self, message: str) -> None:
        self._import_project_config_path = None
        if hasattr(self, "_import_progress_dialog"):
            self._import_progress_dialog.close()
        QtWidgets.QMessageBox.critical(self, "导入数据源失败", message)
        self.statusbar.showMessage("导入数据源失败", 5000)

    def _on_import_cancelled(self) -> None:
        self._import_project_config_path = None
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

    def _project_pie_folders_from_ui(self) -> list[str]:
        folders: list[str] = []
        seen: set[str] = set()
        if hasattr(self, "project_pie_folders_list"):
            for row in range(self.project_pie_folders_list.count()):
                item = self.project_pie_folders_list.item(row)
                value = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "").strip()
                if value and value not in seen:
                    seen.add(value)
                    folders.append(value)
        if not folders:
            primary = self.project_pie_folder_edit.text().strip()
            if primary:
                folders.append(primary)
        return folders

    def _set_project_pie_folders(self, folders: list[str]) -> None:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in folders:
            path = str(value or "").strip()
            if not path or path in seen:
                continue
            seen.add(path)
            normalized.append(path)

        self.project_pie_folders_list.clear()
        for index, path in enumerate(normalized, start=1):
            item = QtWidgets.QListWidgetItem(f"能段 {index}: {path}")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            self.project_pie_folders_list.addItem(item)
        self.project_pie_folder_edit.setText(normalized[0] if normalized else "")

    def select_project_pie_primary_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择PIE主扫描目录",
            self._dialog_start_dir(self.project_pie_folder_edit.text()),
        )
        if not folder:
            return
        folders = self._project_pie_folders_from_ui()
        if folders:
            folders[0] = folder
        else:
            folders = [folder]
        self._set_project_pie_folders(folders)
        self._auto_save_datasource()

    def add_project_pie_folder(self) -> None:
        folders = self._project_pie_folders_from_ui()
        start_path = folders[-1] if folders else self.project_output_dir_edit.text()
        folder = QFileDialog.getExistingDirectory(
            self,
            "添加PIE能段目录",
            self._dialog_start_dir(start_path),
        )
        if not folder:
            return
        if folder not in folders:
            folders.append(folder)
        self._set_project_pie_folders(folders)
        self._auto_save_datasource()

    def remove_project_pie_folders(self) -> None:
        selected_rows = {
            index.row() for index in self.project_pie_folders_list.selectedIndexes()
        }
        if not selected_rows:
            return
        folders = [
            folder
            for index, folder in enumerate(self._project_pie_folders_from_ui())
            if index not in selected_rows
        ]
        self._update_registered_pie_folders(folders)

    def clear_project_pie_folders(self) -> None:
        self._update_registered_pie_folders([])

    def _update_registered_pie_folders(self, folders: list[str]) -> None:
        """Cancel PIE registrations without deleting already copied files."""
        if not self.project_settings_manager.has_project_path():
            return
        committed = self.project_settings_manager.snapshot()
        committed.pie_scan_folders = list(folders)
        committed.pie_scan_folder = folders[0] if folders else ""
        saved = self.project_settings_manager.replace_and_save(committed)
        draft = deepcopy(getattr(self, "_project_draft", saved))
        draft.pie_scan_folders = list(saved.pie_scan_folders)
        draft.pie_scan_folder = saved.pie_scan_folder
        draft.pie_multi_folder_mode = saved.pie_multi_folder_mode
        self._project_draft = draft
        self._project_committed = deepcopy(saved)
        self._set_project_pie_folders(saved.effective_pie_scan_folders())
        self._sync_project_settings_to_tool_pages(deepcopy(saved))
        self.refresh_project_datasource_page(saved)
        self.statusbar.showMessage(
            "PIE 能段登记已更新；项目内已复制文件未删除",
            4000,
        )

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
        if not self.project_settings_manager.has_project_path():
            QtWidgets.QMessageBox.warning(self, "尚未保存项目", "请先保存项目。")
            return
        ps = self.project_settings_manager.snapshot()

        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入手动卡峰文件",
            self._dialog_start_dir(self.project_manual_peak_edit.text() or ps.output_dir),
            "Peak Files (*.yaml *.yml *.csv *.xlsx *.xls);;所有文件 (*)",
        )
        if not path:
            return

        try:
            peak_set = import_peak_set(
                project_root(ps),
                path,
                label=Path(path).stem,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "导入手动卡峰文件失败", str(exc))
            return

        ps.active_peak_set_id = peak_set.peak_set_id
        ps.manual_peak_file = str(verify_peak_set(project_root(ps), peak_set))
        ps.temp_peak_source = "manual"
        ps = self.project_settings_manager.replace_and_save(ps)
        self._invalidate_untraceable_project_curves(ps)

        self._read_project_settings_to_ui(ps)
        self._apply_settings_to_tools(ps)
        self.update_project_title()
        self.refresh_project_datasource_page(ps)
        self.statusbar.showMessage(
            f"✓ 已导入并设为当前项目卡峰范围：{peak_set.label}",
            5000,
        )
        if hasattr(self, "open_project_peak_ranges"):
            self.open_project_peak_ranges(
                settings=ps,
                prompt_before_replace=True,
                show_feedback=False,
            )

    def generate_project_peak_set_from_sum(self) -> None:
        if not self.project_settings_manager.has_project_path():
            QtWidgets.QMessageBox.warning(self, "尚未保存项目", "请先保存项目。")
            return
        ps = self.project_settings_manager.snapshot()
        source_folder = Path(str(ps.sum_spectrum_folder or "")).expanduser()
        if not source_folder.is_absolute():
            source_folder = project_root(ps) / source_folder
        if not source_folder.is_dir():
            QtWidgets.QMessageBox.warning(
                self,
                "无法自动生成卡峰范围",
                "项目累计谱目录不存在。请先在质谱工作台登记累计谱，"
                "或导入现有卡峰文件。",
            )
            return
        running = getattr(self, "_project_peak_generation_worker", None)
        if running is not None and running.isRunning():
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "从累计谱生成卡峰范围",
            "将使用项目累计质谱、项目定标和项目寻峰参数生成并保存"
            "当前项目卡峰范围。\n历史快照由软件自动保留。\n\n是否继续？",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        self.project_peak_set_generate_button.setEnabled(False)
        self.datasource_row_status_labels["manual_peak"].setText("正在自动寻峰…")
        worker = WorkerThread(
            lambda: generate_project_peak_set(
                ps,
                progress_callback=worker.report_progress,
            ),
            self,
        )
        self._project_peak_generation_worker = worker
        worker.progress.connect(
            lambda _value, message: self.datasource_row_status_labels[
                "manual_peak"
            ].setText(message)
        )

        def on_success(record) -> None:
            try:
                approved_path = verify_peak_set(project_root(ps), record)
                ps.active_peak_set_id = record.peak_set_id
                ps.manual_peak_file = str(approved_path)
                ps.temp_peak_source = "manual"
                saved = self.project_settings_manager.replace_and_save(ps)
                self._invalidate_untraceable_project_curves(saved)
                self._read_project_settings_to_ui(saved)
                self._apply_settings_to_tools(saved)
                self.refresh_project_datasource_page(saved)
                if hasattr(self, "open_project_peak_ranges"):
                    self.open_project_peak_ranges(
                        settings=saved,
                        prompt_before_replace=False,
                        show_feedback=False,
                    )
                self.statusbar.showMessage(
                    f"✓ 已自动生成当前项目卡峰范围：{record.label}",
                    5000,
                )
            except Exception as exc:
                logger.exception("Failed to activate generated peak set")
                QtWidgets.QMessageBox.critical(
                    self,
                    "保存项目卡峰范围失败",
                    str(exc),
                )

        def on_failure(message: str) -> None:
            QtWidgets.QMessageBox.critical(
                self,
                "自动生成项目卡峰范围失败",
                message,
            )
            self.datasource_row_status_labels["manual_peak"].setText("生成失败")

        worker.finished_with_result.connect(on_success)
        worker.failed.connect(on_failure)
        worker.finished.connect(
            lambda: self.project_peak_set_generate_button.setEnabled(True)
        )
        worker.finished.connect(
            lambda: setattr(self, "_project_peak_generation_worker", None)
        )
        worker.start()

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

    def open_project_root_folder(self) -> None:
        ps = self.project_settings_manager.snapshot()
        root = project_root(ps)
        root.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(root)))

    def update_project_title(self) -> None:
        project_name = self.project_name_edit.text().strip()
        project_system = self.project_system_edit.text().strip()
        caption = project_name or project_system
        has_project = self.project_settings_manager.has_project_path()
        if has_project:
            self.project_status_label.setText(f"当前项目: {caption or '未命名项目'}")
        elif caption:
            self.project_status_label.setText(f"新项目: {caption}（未保存）")
        else:
            self.project_status_label.setText("当前项目: 未打开项目")
        if hasattr(self, "project_close_button"):
            self.project_close_button.setEnabled(has_project)
        window_title = "BL03U_MassSpectrumTool"
        if project_name:
            window_title = f"{window_title} - {project_name}"
        self.setWindowTitle(window_title)

    def on_project_common_parameters_saved(self) -> None:
        self._sync_project_page_edits_to_runtime(
            save_project=False,
            sync_tools=False,
        )
        self.statusbar.showMessage(
            "参数已更新到项目草稿；点击“保存并应用项目参数”后生效",
            3500,
        )

    def switch_workspace_page(self, page_name: str):
        """Switch pages after explicitly resolving an unsaved project draft."""
        page_map = {
            "project": self.project_page,
            "spectrum": self.spectrum_page,
            "temperature": self.temperature_page,
            "pie": self.pie_page,
            "isotope_correction": self.isotope_correction_page,
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
            if (
                page is not current_page
                and not self._confirm_project_draft_resolution()
            ):
                if "project" in self.page_buttons:
                    self.page_buttons["project"].setChecked(True)
                return
            ps = self.project_settings_manager.snapshot()
        elif page_name == "project":
            self.load_project_settings()
            ps = self.project_settings_manager.snapshot()
        else:
            ps = self.project_settings_manager.snapshot()
        if self.project_settings_manager.has_project_path():
            self._apply_project_runtime_settings(ps)
        else:
            self.apply_config_defaults()
            self.normalization_settings = (
                load_factory_project_settings().to_normalization_settings()
            )
        calibration = (
            ps.to_calibration()
            if self.project_settings_manager.has_project_path()
            else self.current_calibration()
        )
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
        if (
            page_name in {"temperature", "pie"}
            and self.project_settings_manager.has_project_path()
            and hasattr(page, "ensure_project_cache_loaded")
        ):
            page.ensure_project_cache_loaded()
        if (
            page_name == "isotope_correction"
            and self.project_settings_manager.has_project_path()
            and hasattr(page, "ensure_project_source_loaded")
        ):
            # Revalidate the two upstream analysis fingerprints first. A dataset
            # that was conservatively marked stale can become valid again when
            # the complete current analysis key exactly matches its provenance.
            for upstream_page in (
                getattr(self, "temperature_page", None),
                getattr(self, "pie_page", None),
            ):
                if upstream_page is None or not hasattr(
                    upstream_page,
                    "ensure_project_cache_loaded",
                ):
                    continue
                upstream_page.ensure_project_cache_loaded()
                worker = getattr(upstream_page, "_autoload_worker", None)
                if worker is not None and worker.isRunning():
                    worker.finished.connect(page.refresh_project_sources)
            page.ensure_project_source_loaded()
        self.workspace_stack.setCurrentWidget(page)
        if page_name in self.page_buttons:
            self.page_buttons[page_name].setChecked(True)
    def open_common_parameters(self):
        if getattr(self, "spectrum_source_scope", "custom") == "custom":
            self.open_temporary_spectrum_parameters()
            return
        self.switch_workspace_page("project")
        if hasattr(self, "project_tabs"):
            self.project_tabs.setCurrentWidget(self.project_analysis_page)
        if hasattr(self, "project_analysis_tabs"):
            self.project_analysis_tabs.setCurrentIndex(0)
