from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import pyqtgraph as pg
from pyqtgraph import mkPen, LinearRegionItem, GraphicsLayoutWidget
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)
from scipy.optimize import curve_fit

from bl03u_masstool.frontends.pyqt_app.ui_massspec import Ui_MainWindow
from bl03u_masstool.frontends.pyqt_app.theme import get_plot_theme
from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.core.config import (
    load_calibration_config,
    load_calibration_points,
    load_peak_detection_config,
)
from bl03u_masstool.core.output_paths import ensure_output_dir
from bl03u_masstool.core.peak_detection import add_manual_peak as core_add_manual_peak
from bl03u_masstool.core.peak_detection import detect_peaks_by_algorithm
from bl03u_masstool.core.runtime_paths import resource_path
from bl03u_masstool.core.spectrum_io import read_bl03u_txt, sum_spectra
from bl03u_masstool.frontends.pyqt_app.spectrum.axis import SpectrumBottomAxis
from bl03u_masstool.frontends.pyqt_app.spectrum.peak_dialog import PeakDialog
from bl03u_masstool.frontends.pyqt_app.spectrum.workspace_pages import WorkspacePagesMixin


logger = logging.getLogger(__name__)


class MainWindow(WorkspacePagesMixin, Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.x_axis_mode = "mz"
        self.current_plot_x = np.array([], dtype=float)
        self.current_plot_axis_x = np.array([], dtype=float)
        self.current_plot_y = np.array([], dtype=float)
        self.current_plot_title = "质谱图"
        self.current_plot_calibration: Calibration | None = None
        self.plot_x_min: float | None = None
        self.plot_x_max: float | None = None
        self.p1 = None
        self.p2 = None
        self.p1_axis: SpectrumBottomAxis | None = None
        self.p2_axis: SpectrumBottomAxis | None = None
        self.selection_region = None
        self.spectrum_plot = None
        self._clamping_region = False
        self._updating_spectrum_y_range = False
        self._updating_peak_table = False
        self.spectrum_source_scope = self._default_spectrum_source_scope()
        self._custom_single_spectrum_file = self.lineEdit.text().strip()
        self._custom_sum_spectrum_folder = self.folder_path.text().strip()
        self.apply_config_defaults()
        self.configure_runtime_ui()
        self.current_time_offset = 0.0
        self.add_core_tools_launcher()
        self.pushButton.clicked.connect(self.plot_graph)
        self.pushButton_plot_graph_sum.clicked.connect(self.plot_graph_sum)
        self.setWindowIcon(QIcon(str(resource_path("icons/icon.png"))))
        self.setWindowTitle('BL03U_MassSpectrumTool')
        self.pushButton_3.clicked.connect(self.calculate)
        self.toolButton.clicked.connect(self.transfer)
        self.savePeakdata.clicked.connect(self.save)
        self.addPeak.clicked.connect(self.add_peak)
        self.pushButton_4.clicked.connect(self.auto_find_peaks)  # 自动寻峰按钮

        # 同步定标参数到 ProjectSettingsManager，防止切换页面后丢失修改
        self.lineEdit_4.editingFinished.connect(self._sync_calibration_to_project_settings)
        self.lineEdit_5.editingFinished.connect(self._sync_calibration_to_project_settings)
        self.lineEdit_6.editingFinished.connect(self._sync_calibration_to_project_settings)

        # 加载并显示图片
        self.original_pixmap = QPixmap(str(resource_path("icons/bjt.png")))
        if not self.original_pixmap.isNull():
            self.update_label_picture()

        # 连接 resize 事件以动态调整 label_picture 的大小
        self.widget_2.resizeEvent = self.on_widget_2_resize

        # # 设置 label 组件的图片
        # image_path = 'icons/bjt.png'  # 替换为您的图像文件路径
        # pixmap = QPixmap(image_path)
        # self.label_picture.setPixmap(pixmap)

        # 初始化图表相关的成员变量
        self.p1 = None
        self.p2 = None
        self.selection_region = None
        self.spectrum_plot = None

        # 添加一个定时器用于延迟更新
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(180)  # 设置延迟时间，单位为毫秒
        self.update_timer.timeout.connect(self.on_update_timeout)
        self.pending_update = False

        # 初始化文本框的可见性
        self.label_4.setVisible(False)  # 假设 label_4 是用于显示ABC结果的文本框
        self.pushButton_3.setVisible(False)

        # 连接选项卡切换事件
        self.peakResult.currentChanged.connect(self.on_tab_changed)

        # 初始化复选框状态
        self.checkBox_show_gaussian.setChecked(True)  # 默认显示高斯拟合图
        self.checkBox_show_gaussian.toggled.connect(self.on_gaussian_visibility_toggled)

        # 设置表格的上下文菜单
        self.peakData.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.peakData.customContextMenuRequested.connect(self.show_peak_context_menu)
        self.peakData.itemSelectionChanged.connect(self.on_peak_selection_changed)
        self.peakData.itemChanged.connect(self.on_peak_table_item_changed)

        # 添加清除按钮
        self.clearPeaksButton = QPushButton("清除峰值数据")
        self.clearPeaksButton.setText("清空峰值")
        self.clearPeaksButton.setMinimumWidth(0)
        self.clearPeaksButton.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.clearPeaksButton.setFixedHeight(28)
        self.horizontalLayout_4.addWidget(self.clearPeaksButton)
        self.clearPeaksButton.clicked.connect(self.clear_peak_data)
        self._add_peak_navigation_controls()

    def apply_config_defaults(self):
        try:
            previous_plot_calibration = self.current_plot_calibration
            calibration_points = None
            try:
                from bl03u_masstool.core.project_settings import ProjectSettingsManager

                manager = ProjectSettingsManager()
                if manager.has_project_path():
                    project_settings = manager.get()
                    calibration = project_settings.to_calibration()
                    calibration_points = project_settings.calibration_points
                else:
                    calibration = load_calibration_config()
            except Exception:
                calibration = load_calibration_config()
            self.lineEdit_4.setText(f"{calibration.a:.6e}")
            self.lineEdit_5.setText(f"{calibration.b:.6e}")
            self.lineEdit_6.setText(f"{calibration.c:.6e}")

            if calibration_points is None:
                points = load_calibration_points()
            else:
                points = [
                    (float(item["tof"]), float(item["mz"]))
                    for item in calibration_points
                    if "tof" in item and "mz" in item
                ]
            if points:
                self.region.setRowCount(len(points))
                for row, (tof, mz) in enumerate(points):
                    self.region.setItem(row, 0, QTableWidgetItem(f"{tof:g}"))
                    self.region.setItem(row, 1, QTableWidgetItem(f"{mz:g}"))
            if self.current_plot_x.size:
                self.refresh_current_plot_calibration(previous_plot_calibration)
            elif hasattr(self, "p2"):
                self.refresh_plot_axis_mode()
        except Exception as exc:
            logger.warning("Failed to load YAML configuration; using UI defaults", exc_info=True)

    def closeEvent(self, event) -> None:
        """Persist window geometry and splitter state before closing."""
        self._qsettings.setValue("window/geometry", self.saveGeometry())
        if hasattr(self, "main_splitter"):
            self._qsettings.setValue("window/splitter_state", self.main_splitter.saveState())
        super().closeEvent(event)

    def configure_runtime_ui(self):
        self.setAccessibleName("BL03U Mass Spectrum Tool")
        self._qsettings = QtCore.QSettings("BL03U", "MassSpecTool")
        self.resize(1280, 720)
        self.setMinimumSize(1024, 640)
        geometry = self._qsettings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(1280, 720)
        self.verticalLayout_9.setContentsMargins(10, 8, 10, 8)
        self.verticalLayout_9.setSpacing(8)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setSpacing(6)
        self.verticalLayout_7.setStretch(0, 0)
        self.verticalLayout_7.setStretch(1, 1)
        self.verticalLayout_8.setContentsMargins(6, 6, 6, 6)
        self.graph_layout.setContentsMargins(0, 0, 0, 0)
        self.graph_layout.setSpacing(0)
        self.label_picture.setText("")
        self.lineEdit.setClearButtonEnabled(True)
        self.folder_path.setClearButtonEnabled(True)
        self.tabWidget.setObjectName("SourceModeStateTabs")
        self.tabWidget.setDocumentMode(True)
        self.peakResult.setObjectName("PeakResultTabs")
        self.peakResult.setDocumentMode(True)
        self.peakResult.tabBar().setExpanding(False)
        self.statusbar.showMessage("就绪")

        self._rebuild_top_controls()
        self._rebuild_main_splitter()
        self._rebuild_side_panel()

        self.widget_3.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        self.widget_3.setMinimumWidth(340)
        self.widget_3.setMaximumWidth(560)

        for table in (self.peakData, self.region):
            table.setAlternatingRowColors(True)
            table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
            table.verticalHeader().setVisible(False)
            table.verticalHeader().setDefaultSectionSize(24)
            table.setWordWrap(False)

        self.peakData.setAccessibleName("卡峰范围表")
        self.region.setAccessibleName("质谱定标点表")

    def add_core_tools_launcher(self):
        """Build top-level workspace pages instead of launching tool dialogs."""
        self._build_workspace_pages()

    def _take_all_items(self, layout):
        while layout.count():
            layout.takeAt(0)

    def _add_peak_navigation_controls(self):
        self.peakNavigationPanel = QtWidgets.QWidget(self.widget_3)
        self.peakNavigationPanel.setObjectName("PeakNavigationPanel")
        navigation_layout = QHBoxLayout(self.peakNavigationPanel)
        navigation_layout.setContentsMargins(0, 0, 0, 0)
        navigation_layout.setSpacing(4)

        self.previousPeakButton = QPushButton("上一峰", self.peakNavigationPanel)
        self.nextPeakButton = QPushButton("下一峰", self.peakNavigationPanel)
        self.updatePeakRangeButton = QPushButton("更新范围", self.peakNavigationPanel)
        self.previousPeakButton.setToolTip("切换到表格中的上一个有效峰")
        self.nextPeakButton.setToolTip("切换到表格中的下一个有效峰")
        self.addPeak.setToolTip("根据当前框选区域添加一个峰，并按 m/z 插入表格")
        self.updatePeakRangeButton.setToolTip("根据当前框选区域重算当前峰，并写回卡峰范围")

        self.horizontalLayout_4.removeWidget(self.addPeak)
        for button in (self.previousPeakButton, self.nextPeakButton, self.addPeak, self.updatePeakRangeButton):
            button.setMinimumWidth(0)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            button.setFixedHeight(28)
        for button in (self.previousPeakButton, self.nextPeakButton):
            button.setObjectName("BrowseButton")

        navigation_layout.addWidget(self.previousPeakButton)
        navigation_layout.addWidget(self.nextPeakButton)
        navigation_layout.addWidget(self.addPeak)
        navigation_layout.addWidget(self.updatePeakRangeButton)
        self.verticalLayout.insertWidget(2, self.peakNavigationPanel)

        self.previousPeakButton.clicked.connect(self.select_previous_peak)
        self.nextPeakButton.clicked.connect(self.select_next_peak)
        self.updatePeakRangeButton.clicked.connect(self.update_current_peak_range_from_region)
        self.update_peak_navigation_state()

    def _rebuild_top_controls(self):
        self.widget.setObjectName("ControlBar")
        self.widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.widget.setMinimumHeight(88)
        self.widget.setMaximumHeight(104)
        self._take_all_items(self.horizontalLayout_17)
        self.horizontalLayout_17.setContentsMargins(8, 7, 8, 7)
        self.horizontalLayout_17.setSpacing(0)
        self.verticalLayout_9.removeWidget(self.checkBox_show_gaussian)

        toolbar_body = QtWidgets.QWidget(self.widget)
        toolbar_body.setObjectName("SpectrumToolbarBody")
        toolbar_body_layout = QVBoxLayout(toolbar_body)
        toolbar_body_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_body_layout.setSpacing(6)

        self.tabWidget.hide()
        source_panel = QtWidgets.QWidget(toolbar_body)
        source_panel.setObjectName("SourcePanel")
        source_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        source_layout = QHBoxLayout(source_panel)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(8)
        source_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)

        source_title = QtWidgets.QLabel("数据源", source_panel)
        source_title.setObjectName("ToolbarSectionTitle")
        source_layout.addWidget(source_title)

        scope_panel = QtWidgets.QWidget(source_panel)
        scope_panel.setObjectName("ModePanel")
        scope_panel.setFixedSize(218, 30)
        scope_layout = QHBoxLayout(scope_panel)
        scope_layout.setContentsMargins(0, 0, 0, 0)
        scope_layout.setSpacing(6)
        scope_label = QtWidgets.QLabel("来源", scope_panel)
        scope_label.setObjectName("ModeTitle")
        scope_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        scope_layout.addWidget(scope_label)
        scope_segment = QtWidgets.QWidget(scope_panel)
        scope_segment.setObjectName("ModeSegment")
        scope_segment_layout = QHBoxLayout(scope_segment)
        scope_segment_layout.setContentsMargins(0, 0, 0, 0)
        scope_segment_layout.setSpacing(3)
        self.projectSourceButton = QtWidgets.QToolButton(source_panel)
        self.projectSourceButton.setText("继承项目")
        self.projectSourceButton.setObjectName("ModeToggle")
        self.projectSourceButton.setCheckable(True)
        self.projectSourceButton.setFixedSize(72, 28)
        self.projectSourceButton.setToolTip("使用项目管理中登记的质谱工作台路径")
        self.customSourceButton = QtWidgets.QToolButton(source_panel)
        self.customSourceButton.setText("自选路径")
        self.customSourceButton.setObjectName("ModeToggle")
        self.customSourceButton.setCheckable(True)
        self.customSourceButton.setFixedSize(64, 28)
        self.customSourceButton.setToolTip("只在当前质谱工作台使用此路径，不写回项目配置")
        self.sourceScopeGroup = QtWidgets.QButtonGroup(source_panel)
        self.sourceScopeGroup.setExclusive(True)
        self.sourceScopeGroup.addButton(self.projectSourceButton, 0)
        self.sourceScopeGroup.addButton(self.customSourceButton, 1)
        scope_segment_layout.addWidget(self.projectSourceButton)
        scope_segment_layout.addWidget(self.customSourceButton)
        scope_layout.addWidget(scope_segment)
        source_layout.addWidget(scope_panel)

        mode_panel = QtWidgets.QWidget(source_panel)
        mode_panel.setObjectName("ModePanel")
        mode_panel.setFixedSize(174, 30)
        mode_layout = QHBoxLayout(mode_panel)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(6)
        mode_label = QtWidgets.QLabel("类型", mode_panel)
        mode_label.setObjectName("ModeTitle")
        mode_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        mode_layout.addWidget(mode_label)
        mode_segment = QtWidgets.QWidget(mode_panel)
        mode_segment.setObjectName("ModeSegment")
        mode_segment_layout = QHBoxLayout(mode_segment)
        mode_segment_layout.setContentsMargins(0, 0, 0, 0)
        mode_segment_layout.setSpacing(3)
        self.singleModeButton = QtWidgets.QToolButton(source_panel)
        self.singleModeButton.setText("单谱")
        self.singleModeButton.setObjectName("ModeToggle")
        self.singleModeButton.setCheckable(True)
        self.singleModeButton.setFixedSize(48, 28)
        self.singleModeButton.setChecked(self.tabWidget.currentIndex() == 0)
        self.sumModeButton = QtWidgets.QToolButton(source_panel)
        self.sumModeButton.setText("累计谱")
        self.sumModeButton.setObjectName("ModeToggle")
        self.sumModeButton.setCheckable(True)
        self.sumModeButton.setFixedSize(58, 28)
        self.sumModeButton.setChecked(self.tabWidget.currentIndex() != 0)
        self.sourceModeGroup = QtWidgets.QButtonGroup(source_panel)
        self.sourceModeGroup.setExclusive(True)
        self.sourceModeGroup.addButton(self.singleModeButton, 0)
        self.sourceModeGroup.addButton(self.sumModeButton, 1)
        mode_segment_layout.addWidget(self.singleModeButton)
        mode_segment_layout.addWidget(self.sumModeButton)
        mode_layout.addWidget(mode_segment)
        source_layout.addWidget(mode_panel)

        source_stack = QtWidgets.QStackedWidget(source_panel)
        source_stack.setObjectName("SourceStack")
        source_stack.setMinimumHeight(30)
        source_stack.setMaximumHeight(32)
        self.sourceStack = source_stack

        single_page = QtWidgets.QWidget(source_stack)
        single_layout = QHBoxLayout(single_page)
        single_layout.setContentsMargins(0, 0, 0, 0)
        single_layout.setSpacing(6)
        self.label.setText("文件")
        single_layout.addWidget(self.label)
        single_layout.addWidget(self.lineEdit, stretch=1)
        self.singleBrowseButton = QPushButton("选择", single_page)
        self.singleBrowseButton.setObjectName("BrowseButton")
        self.singleBrowseButton.setToolTip("选择单个质谱文件")
        self.singleBrowseButton.clicked.connect(self.choose_single_file)
        single_layout.addWidget(self.singleBrowseButton)
        single_layout.addWidget(self.pushButton)
        source_stack.addWidget(single_page)

        sum_page = QtWidgets.QWidget(source_stack)
        sum_layout = QHBoxLayout(sum_page)
        sum_layout.setContentsMargins(0, 0, 0, 0)
        sum_layout.setSpacing(6)
        self.label_18.setText("文件夹")
        sum_layout.addWidget(self.label_18)
        sum_layout.addWidget(self.folder_path, stretch=1)
        self.sumBrowseButton = QPushButton("选择", sum_page)
        self.sumBrowseButton.setObjectName("BrowseButton")
        self.sumBrowseButton.setToolTip("选择累计谱文件夹")
        self.sumBrowseButton.clicked.connect(self.choose_sum_folder)
        sum_layout.addWidget(self.sumBrowseButton)
        sum_layout.addWidget(self.pushButton_plot_graph_sum)
        source_stack.addWidget(sum_page)

        source_stack.setCurrentIndex(0 if self.singleModeButton.isChecked() else 1)
        self.sourceModeGroup.idClicked.connect(source_stack.setCurrentIndex)
        self.sourceModeGroup.idClicked.connect(self.tabWidget.setCurrentIndex)
        self.sourceScopeGroup.idClicked.connect(
            lambda button_id: self.set_spectrum_source_scope("project" if button_id == 0 else "custom")
        )
        self.lineEdit.textEdited.connect(self._remember_custom_spectrum_paths)
        self.folder_path.textEdited.connect(self._remember_custom_spectrum_paths)
        source_layout.addWidget(source_stack, stretch=1)

        for path_edit in (self.lineEdit, self.folder_path):
            path_edit.setMinimumWidth(280)
            path_edit.setMaximumWidth(16777215)
            path_edit.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
        for button in (self.singleBrowseButton, self.sumBrowseButton):
            button.setMinimumWidth(60)
            button.setFixedHeight(30)
        for button in (self.pushButton, self.pushButton_plot_graph_sum):
            button.setText("加载")
            button.setObjectName("PrimaryToolbarButton")
            button.setMinimumWidth(72)
            button.setFixedHeight(30)
        self._refresh_spectrum_source_controls()
        toolbar_body_layout.addWidget(source_panel)

        tool_panel = QtWidgets.QWidget(toolbar_body)
        tool_panel.setObjectName("CalibrationPanel")
        tool_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        tool_layout = QHBoxLayout(tool_panel)
        tool_layout.setContentsMargins(0, 0, 0, 0)
        tool_layout.setSpacing(8)
        tool_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)

        for widget in (self.label_17, self.label_5, self.lineEdit_4, self.label_6, self.lineEdit_5, self.label_7, self.lineEdit_6):
            widget.hide()

        self.label_2.setText("TOF")
        self.label_3.setText("m/z")
        secondary_title = QtWidgets.QLabel("显示与换算", tool_panel)
        secondary_title.setObjectName("ToolbarSectionTitle")
        tool_layout.addWidget(secondary_title)

        display_group = QtWidgets.QWidget(tool_panel)
        display_group.setObjectName("ToolbarGroup")
        display_layout = QHBoxLayout(display_group)
        display_layout.setContentsMargins(8, 3, 8, 3)
        display_layout.setSpacing(6)
        self.commonParamsButton = QPushButton("通用参数", display_group)
        self.commonParamsButton.setObjectName("BrowseButton")
        self.commonParamsButton.setToolTip("打开全局定标、归一化和自动寻峰参数")
        self.commonParamsButton.clicked.connect(self.open_common_parameters)
        self.commonParamsButton.setFixedHeight(30)
        display_layout.addWidget(self.commonParamsButton)
        self.checkBox_show_gaussian.setObjectName("InlineCheck")
        display_layout.addWidget(self.checkBox_show_gaussian)
        axis_label = QtWidgets.QLabel("横轴", display_group)
        self.xAxisModeCombo = QtWidgets.QComboBox(display_group)
        self.xAxisModeCombo.addItem("TOF", "tof")
        self.xAxisModeCombo.addItem("m/z", "mz")
        self.xAxisModeCombo.setFixedWidth(76)
        self.xAxisModeCombo.setCurrentIndex(1 if self.x_axis_mode == "mz" else 0)
        self.xAxisModeCombo.currentIndexChanged.connect(self.set_x_axis_mode)
        display_layout.addWidget(axis_label)
        display_layout.addWidget(self.xAxisModeCombo)
        tool_layout.addWidget(display_group)

        conversion_group = QtWidgets.QWidget(tool_panel)
        conversion_group.setObjectName("ToolbarGroup")
        conversion_layout = QHBoxLayout(conversion_group)
        conversion_layout.setContentsMargins(8, 3, 8, 3)
        conversion_layout.setSpacing(6)
        conversion_layout.addWidget(self.label_2)
        conversion_layout.addWidget(self.lineEdit_2)
        arrow_group = QtWidgets.QWidget(conversion_group)
        arrow_group.setObjectName("ArrowGroup")
        arrow_layout = QHBoxLayout(arrow_group)
        arrow_layout.setContentsMargins(0, 0, 0, 0)
        arrow_layout.setSpacing(4)
        self.toolButton_2.setToolTip("TOF 转 m/z")
        self.toolButton.setToolTip("m/z 转 TOF")
        arrow_layout.addWidget(self.toolButton_2)
        arrow_layout.addWidget(self.toolButton)
        conversion_layout.addWidget(arrow_group)
        conversion_layout.addWidget(self.label_3)
        conversion_layout.addWidget(self.lineEdit_3)
        tool_layout.addWidget(conversion_group)
        tool_layout.addStretch(1)
        toolbar_body_layout.addWidget(tool_panel)

        for edit in (self.lineEdit_4, self.lineEdit_5, self.lineEdit_6, self.lineEdit_2, self.lineEdit_3):
            edit.setMinimumHeight(26)
            edit.setMaximumHeight(28)
        self.lineEdit_4.setMinimumWidth(180)
        self.lineEdit_5.setMinimumWidth(120)
        self.lineEdit_6.setMinimumWidth(120)
        self.lineEdit_2.setMinimumWidth(112)
        self.lineEdit_2.setMaximumWidth(150)
        self.lineEdit_3.setMinimumWidth(112)
        self.lineEdit_3.setMaximumWidth(150)
        for button in (self.toolButton, self.toolButton_2):
            button.setObjectName("ArrowButton")
            button.setFixedSize(32, 28)
        self.horizontalLayout_17.addWidget(toolbar_body, stretch=1)

    def _default_spectrum_source_scope(self) -> str:
        try:
            from bl03u_masstool.core.project_settings import ProjectSettingsManager

            manager = ProjectSettingsManager()
            if manager.has_project_path():
                project_settings = manager.get()
                if project_settings.single_spectrum_file or project_settings.sum_spectrum_folder:
                    return "project"
        except Exception:
            pass
        return "custom"

    def _remember_custom_spectrum_paths(self, _text: str | None = None) -> None:
        if getattr(self, "spectrum_source_scope", "custom") != "custom":
            return
        self._custom_single_spectrum_file = self.lineEdit.text().strip()
        self._custom_sum_spectrum_folder = self.folder_path.text().strip()

    def set_spectrum_source_scope(self, scope: str, *, apply_project: bool = True) -> None:
        scope = "project" if scope == "project" else "custom"
        previous_scope = getattr(self, "spectrum_source_scope", "custom")
        if previous_scope == "custom" and scope == "project":
            self._remember_custom_spectrum_paths()
        self.spectrum_source_scope = scope

        if hasattr(self, "projectSourceButton"):
            self.projectSourceButton.setChecked(scope == "project")
        if hasattr(self, "customSourceButton"):
            self.customSourceButton.setChecked(scope == "custom")

        if scope == "project" and apply_project:
            try:
                from bl03u_masstool.core.project_settings import ProjectSettingsManager

                self.apply_project_spectrum_paths(ProjectSettingsManager().get())
            except Exception:
                pass
        elif scope == "custom" and previous_scope == "project":
            if self._custom_single_spectrum_file:
                self.lineEdit.setText(self._custom_single_spectrum_file)
            if self._custom_sum_spectrum_folder:
                self.folder_path.setText(self._custom_sum_spectrum_folder)

        self._refresh_spectrum_source_controls()

    def _refresh_spectrum_source_controls(self) -> None:
        use_project = getattr(self, "spectrum_source_scope", "custom") == "project"
        if hasattr(self, "projectSourceButton"):
            self.projectSourceButton.setChecked(use_project)
        if hasattr(self, "customSourceButton"):
            self.customSourceButton.setChecked(not use_project)
        for edit, project_placeholder, custom_placeholder in (
            (self.lineEdit, "项目未登记单谱文件", "选择或输入单谱文件路径"),
            (self.folder_path, "项目未登记累计谱文件夹", "选择或输入累计谱文件夹路径"),
        ):
            edit.setReadOnly(use_project)
            edit.setClearButtonEnabled(not use_project)
            edit.setPlaceholderText(project_placeholder if use_project else custom_placeholder)
        for button in (
            getattr(self, "singleBrowseButton", None),
            getattr(self, "sumBrowseButton", None),
        ):
            if button is not None:
                button.setEnabled(not use_project)

    def apply_project_spectrum_paths(self, project_settings, *, activate: bool = False) -> None:
        has_project_source = bool(
            getattr(project_settings, "single_spectrum_file", "")
            or getattr(project_settings, "sum_spectrum_folder", "")
        )
        if activate and has_project_source:
            self.set_spectrum_source_scope("project", apply_project=False)
        if getattr(self, "spectrum_source_scope", "custom") != "project":
            self._refresh_spectrum_source_controls()
            return
        self.lineEdit.setText(getattr(project_settings, "single_spectrum_file", ""))
        self.folder_path.setText(getattr(project_settings, "sum_spectrum_folder", ""))
        self._refresh_spectrum_source_controls()

    def set_x_axis_mode(self):
        combo = getattr(self, "xAxisModeCombo", None)
        new_mode = str(combo.currentData()) if combo is not None else "mz"
        if new_mode == self.x_axis_mode:
            self.refresh_plot_axis_mode()
            return

        old_mode = self.x_axis_mode
        region_tof = None
        view_tof = None
        if self.current_plot_x.size:
            if self.selection_region is not None:
                region_tof = self._axis_range_to_tof(
                    *self.selection_region.getRegion(),
                    mode=old_mode,
                )
            if self.p2 is not None:
                view_tof = self._axis_range_to_tof(*self.p2.viewRange()[0], mode=old_mode)

        self.x_axis_mode = new_mode
        if self.current_plot_x.size == 0:
            self.refresh_plot_axis_mode()
            return

        x_values = self.current_plot_x.copy()
        y_values = self.current_plot_y.copy()
        title = self.current_plot_title
        try:
            self.clear_plot_area()
            self.setup_plots(x_values, y_values, title=title)
            if region_tof is not None and self.selection_region is not None:
                self.selection_region.setRegion(self._tof_range_to_axis_range(*region_tof))
            if view_tof is not None and self.p2 is not None:
                left, right = self._tof_range_to_axis_range(*view_tof)
                left, right = self._clamp_region_values(left, right)
                if right > left:
                    self.p2.setXRange(left, right, padding=0)
                    self.update_spectrum_y_range_for_visible_x((left, right))
            self.update_plot()
            self.update_region_readout()
            if self.p1 is not None and self.checkBox_show_gaussian.isChecked():
                self.update_gaussian_fit()
        except Exception as exc:
            QMessageBox.warning(self, "Warning", f"切换横轴失败: {exc}")

    def refresh_plot_axis_mode(self) -> None:
        axis_label = "m/z" if self.x_axis_mode == "mz" else "TOF"
        calibration = self.current_calibration()
        for axis in (self.p1_axis, self.p2_axis):
            if axis is not None:
                axis.set_display_mode(self.x_axis_mode, calibration)
        for plot in (self.p1, self.p2):
            if plot is not None:
                plot.setLabel("bottom", axis_label)

    def _tof_to_axis_x(self, tof, mode: str | None = None):
        axis_mode = self.x_axis_mode if mode is None else mode
        tof_values = np.asarray(tof, dtype=float)
        if axis_mode == "mz":
            converted = self.current_calibration().tof_to_mz(tof_values)
        else:
            converted = tof_values
        return float(converted) if np.ndim(converted) == 0 else converted

    def _axis_x_to_tof(self, axis_x: float, mode: str | None = None) -> float:
        axis_mode = self.x_axis_mode if mode is None else mode
        if axis_mode == "mz":
            return self.current_calibration().mz_to_tof(float(axis_x))
        return float(axis_x)

    def _axis_range_to_tof(
        self,
        left: float,
        right: float,
        mode: str | None = None,
    ) -> tuple[float, float]:
        left_tof = self._axis_x_to_tof(left, mode)
        right_tof = self._axis_x_to_tof(right, mode)
        if right_tof < left_tof:
            left_tof, right_tof = right_tof, left_tof
        return left_tof, right_tof

    def _tof_range_to_axis_range(self, left_tof: float, right_tof: float) -> tuple[float, float]:
        left = float(self._tof_to_axis_x(left_tof))
        right = float(self._tof_to_axis_x(right_tof))
        if right < left:
            left, right = right, left
        return left, right

    def _tof_bounds_to_indices(self, left_tof: float, right_tof: float) -> tuple[int, int]:
        if right_tof < left_tof:
            left_tof, right_tof = right_tof, left_tof
        left_idx = int(np.searchsorted(self.current_plot_x, left_tof, side="left"))
        right_idx = int(np.searchsorted(self.current_plot_x, right_tof, side="right") - 1)
        left_idx = max(0, min(left_idx, self.current_plot_y.size - 1))
        right_idx = max(left_idx, min(right_idx, self.current_plot_y.size - 1))
        return left_idx, right_idx

    def _current_axis_x_values(self) -> np.ndarray:
        if self.current_plot_x.size == 0:
            return np.array([], dtype=float)
        return np.asarray(self._tof_to_axis_x(self.current_plot_x), dtype=float)

    def _rebuild_main_splitter(self):
        self.widget_2.setObjectName("PlotPanel")
        self.widget_3.setObjectName("SidePanel")
        self._take_all_items(self.horizontalLayout_13)
        self.horizontalLayout_13.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_13.setSpacing(0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self.centralwidget)
        splitter.setObjectName("MainSplitter")
        splitter.addWidget(self.widget_2)
        splitter.addWidget(self.widget_3)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, True)
        splitter.setSizes([880, 400])
        self.horizontalLayout_13.addWidget(splitter)
        self.main_splitter = splitter
        splitter_state = self._qsettings.value("window/splitter_state")
        if splitter_state is not None:
            splitter.restoreState(splitter_state)

    def _rebuild_side_panel(self):
        self.widget_3.setMinimumWidth(340)
        self.widget_3.setMaximumWidth(560)
        self._take_all_items(self.verticalLayout)
        self.verticalLayout.setContentsMargins(6, 6, 6, 6)
        self.verticalLayout.setSpacing(6)

        summary = QtWidgets.QWidget(self.widget_3)
        summary.setObjectName("PeakSummary")
        summary.setMaximumHeight(58)
        summary_layout = QtWidgets.QGridLayout(summary)
        summary_layout.setContentsMargins(6, 4, 6, 4)
        summary_layout.setHorizontalSpacing(6)
        summary_layout.setVerticalSpacing(3)

        self.label_12.setText("峰位 TOF")
        self.label_16.setText("m/z")
        self.label_13.setText("左边界")
        self.label_15.setText("右边界")
        for label in (self.label_12, self.label_16, self.label_13, self.label_15):
            label.setObjectName("ReadoutLabel")
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        for value in (self.label_8, self.label_9, self.label_10, self.label_11):
            value.setObjectName("ReadoutValue")
            value.setMinimumHeight(23)
            value.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        summary_layout.addWidget(self.label_12, 0, 0)
        summary_layout.addWidget(self.label_8, 0, 1)
        summary_layout.addWidget(self.label_16, 0, 2)
        summary_layout.addWidget(self.label_9, 0, 3)
        summary_layout.addWidget(self.label_13, 1, 0)
        summary_layout.addWidget(self.label_10, 1, 1)
        summary_layout.addWidget(self.label_15, 1, 2)
        summary_layout.addWidget(self.label_11, 1, 3)
        summary_layout.setColumnStretch(1, 1)
        summary_layout.setColumnStretch(3, 1)

        self.label_14.hide()
        self.verticalLayout.addWidget(summary)

        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(4)
        self.savePeakdata.setText("导出")
        self.pushButton_4.setText("自动寻峰")
        self.horizontalLayout_4.insertWidget(0, self.pushButton_4)
        for button in (self.pushButton_4, self.addPeak, self.savePeakdata):
            button.setMinimumWidth(0)
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            button.setFixedHeight(28)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout.addWidget(self.peakResult, stretch=1)
        self.verticalLayout.addWidget(self.label_4)
        self.verticalLayout.addLayout(self.horizontalLayout_7)

    def _dialog_start_dir(self, current_path: str) -> str:
        current_path = current_path.strip().strip('"')
        if current_path and os.path.isdir(current_path):
            return os.path.abspath(current_path)
        if current_path and os.path.isfile(current_path):
            return os.path.dirname(os.path.abspath(current_path))
        return os.getcwd()

    def choose_single_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择单谱文件",
            self._dialog_start_dir(self.lineEdit.text()),
            "质谱数据 (*.txt *.asc *.888);;所有文件 (*)",
        )
        if file_path:
            self.lineEdit.setText(file_path)
            self._remember_custom_spectrum_paths()

    def choose_sum_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择累计谱文件夹",
            self._dialog_start_dir(self.folder_path.text()),
        )
        if folder_path:
            self.folder_path.setText(folder_path)
            self._remember_custom_spectrum_paths()

    def current_calibration(self):
        return Calibration(
            a=float(self.lineEdit_4.text()),
            b=float(self.lineEdit_5.text()),
            c=float(self.lineEdit_6.text()),
        )

    @staticmethod
    def _calibration_close(left: Calibration | None, right: Calibration | None) -> bool:
        if left is None or right is None:
            return left is right
        return bool(
            np.allclose(
                [left.a, left.b, left.c],
                [right.a, right.b, right.c],
                rtol=1e-12,
                atol=1e-18,
            )
        )

    def _axis_x_to_tof_with_calibration(
        self,
        axis_x: float,
        calibration: Calibration,
        mode: str | None = None,
    ) -> float:
        axis_mode = self.x_axis_mode if mode is None else mode
        if axis_mode == "mz":
            return calibration.mz_to_tof(float(axis_x))
        return float(axis_x)

    def _axis_range_to_tof_with_calibration(
        self,
        left: float,
        right: float,
        calibration: Calibration,
        mode: str | None = None,
    ) -> tuple[float, float]:
        left_tof = self._axis_x_to_tof_with_calibration(left, calibration, mode)
        right_tof = self._axis_x_to_tof_with_calibration(right, calibration, mode)
        if right_tof < left_tof:
            left_tof, right_tof = right_tof, left_tof
        return left_tof, right_tof

    def _recalibrate_peak_table_mz(self) -> None:
        calibration = self.current_calibration()
        self._updating_peak_table = True
        try:
            for row in range(self.peakData.rowCount()):
                time = self._table_float(row, 1)
                if time is None or not np.isfinite(time):
                    continue
                self.peakData.setItem(
                    row,
                    2,
                    QTableWidgetItem(f"{float(calibration.tof_to_mz(time)):.2f}"),
                )
        finally:
            self._updating_peak_table = False

    def refresh_current_plot_calibration(self, previous_calibration: Calibration | None = None) -> None:
        """Re-project the loaded spectrum after calibration coefficients change."""
        new_calibration = self.current_calibration()
        old_calibration = previous_calibration or self.current_plot_calibration
        self.current_plot_calibration = new_calibration
        self.refresh_plot_axis_mode()

        if self.current_plot_x.size == 0 or self.current_plot_y.size == 0:
            return

        view_tof = None
        region_tof = None
        if old_calibration is not None:
            try:
                if self.selection_region is not None:
                    region_tof = self._axis_range_to_tof_with_calibration(
                        *self.selection_region.getRegion(),
                        old_calibration,
                    )
                if self.p2 is not None:
                    view_tof = self._axis_range_to_tof_with_calibration(
                        *self.p2.viewRange()[0],
                        old_calibration,
                    )
            except Exception:
                region_tof = None
                view_tof = None

        axis_x_values = self._current_axis_x_values()
        if axis_x_values.size == 0 or not np.any(np.isfinite(axis_x_values)):
            return
        self.current_plot_axis_x = axis_x_values
        self.plot_x_min = float(np.nanmin(axis_x_values))
        self.plot_x_max = float(np.nanmax(axis_x_values))

        if self.spectrum_plot is not None:
            self.spectrum_plot.setData(axis_x_values, self.current_plot_y)

        y_min = float(np.nanmin(self.current_plot_y))
        y_max = float(np.nanmax(self.current_plot_y))
        y_padding = max(1.0, (y_max - y_min) * 0.08)
        x_range = max(1e-9, self.plot_x_max - self.plot_x_min)
        if self.p2 is not None:
            self._apply_plot_limits(self.p2, y_min - y_padding, y_max + y_padding, x_range)
        if self.p1 is not None:
            self._apply_plot_limits(self.p1, y_min - y_padding, y_max + y_padding, x_range)

        if self.selection_region is not None:
            self.selection_region.setBounds((self.plot_x_min, self.plot_x_max))
            if region_tof is not None:
                self.selection_region.setRegion(self._tof_range_to_axis_range(*region_tof))
            else:
                self.selection_region.setRegion(
                    self._clamp_region_values(*self.selection_region.getRegion())
                )

        if self.p2 is not None and view_tof is not None:
            left, right = self._tof_range_to_axis_range(*view_tof)
            left, right = self._clamp_region_values(left, right)
            if right > left:
                self.p2.setXRange(left, right, padding=0)
                self.update_spectrum_y_range_for_visible_x((left, right))

        if not self._calibration_close(old_calibration, new_calibration):
            self._recalibrate_peak_table_mz()
        self.update_plot()
        self.update_region_readout()
        if self.p1 is not None and self.checkBox_show_gaussian.isChecked():
            self.update_gaussian_fit()

    def _sync_calibration_to_project_settings(self):
        """将当前 UI 中的定标参数同步写到 ProjectSettingsManager。

        防止用户直接在质谱工作台修改 lineEdit_4/5/6 或通过 calculate()
        拟合新参数后，切换页面时被 _apply_project_runtime_settings 覆盖。
        """
        try:
            from bl03u_masstool.core.project_settings import ProjectSettingsManager
            from bl03u_masstool.core.config import save_calibration_config

            calibration = self.current_calibration()
            save_calibration_config(calibration)

            manager = ProjectSettingsManager()
            if manager.has_project_path():
                ps = manager.get()
                ps.cal_a = calibration.a
                ps.cal_b = calibration.b
                ps.cal_c = calibration.c
                manager.save()
            self.refresh_current_plot_calibration()
        except Exception:
            logger.debug("Calibration sync skipped", exc_info=True)

    def load_image(self, image_path):
        pixmap = QPixmap(str(resource_path(image_path)))

        if pixmap.isNull():
            logger.warning("Failed to load image: %s", image_path)
        else:
            # 根据 label_picture 的大小调整图片大小并保持宽高比
            scaled_pixmap = pixmap.scaled(
                self.widget_2.size(),
                aspectRatioMode=QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                transformMode=QtCore.Qt.TransformationMode.SmoothTransformation
            )
            self.label_picture.setPixmap(scaled_pixmap)

    
    def update_label_picture(self):
        """更新 label_picture 的图片"""
        if self.label_picture is None:
            return
        target_size = self.label_picture.size()
        target_size.setWidth(max(1, min(520, int(target_size.width() * 0.42))))
        target_size.setHeight(max(1, min(170, int(target_size.height() * 0.26))))
        scaled_pixmap = self.original_pixmap.scaled(
            target_size,
            aspectRatioMode=QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                transformMode=QtCore.Qt.TransformationMode.SmoothTransformation
        )
        self.label_picture.setPixmap(scaled_pixmap)

    def on_widget_2_resize(self, event):
        """当 widget_2 大小改变时调整 label_picture 的大小"""
        # 使用 setScaledContents 来自动调整图片大小，而不是手动设置几何形状
        self.update_label_picture()

    def clear_widgets(self):
        """清除旧的小部件"""
        if self.label_picture is not None:
            self.graph_layout.removeWidget(self.label_picture)
            self.label_picture.setParent(None)
            self.label_picture.deleteLater()
            self.label_picture = None
        self.clear_plot_area()
        self.current_plot_x = np.array([], dtype=float)
        self.current_plot_axis_x = np.array([], dtype=float)
        self.current_plot_y = np.array([], dtype=float)
        self.current_plot_calibration = None
        self.current_plot_title = "质谱图"
        self.plot_x_min = None
        self.plot_x_max = None
        self.update_peak_navigation_state()

    def clear_plot_area(self):
        """清除当前绘图区，不触发文件重载。"""
        if hasattr(self, "update_timer"):
            self.update_timer.stop()
            self.pending_update = False
        while self.graph_layout.count():
            item = self.graph_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()
        self.p1 = None
        self.p2 = None
        self.p1_axis = None
        self.p2_axis = None
        self.selection_region = None
        self.spectrum_plot = None

    def setup_plots(self, x_data, y_data, title="质谱图"):
        """设置并绘制图表"""
        pg.setConfigOptions(antialias=False)

        x_values = np.asarray(x_data, dtype=float)
        y_values = np.asarray(y_data, dtype=float)
        if x_values.size == 0 or y_values.size == 0:
            raise ValueError("谱图没有可绘制的数据")
        point_count = min(x_values.size, y_values.size)
        x_values = x_values[:point_count]
        y_values = y_values[:point_count]
        self.current_plot_x = x_values
        axis_x_values = self._current_axis_x_values()
        if axis_x_values.size == 0 or not np.any(np.isfinite(axis_x_values)):
            raise ValueError("谱图横坐标无效")
        self.current_plot_axis_x = axis_x_values
        self.current_plot_y = y_values
        self.current_plot_title = title
        self.current_plot_calibration = self.current_calibration()
        self.plot_x_min = float(np.nanmin(axis_x_values))
        self.plot_x_max = float(np.nanmax(axis_x_values))
        x_range = max(1e-9, self.plot_x_max - self.plot_x_min)
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        y_padding = max(1.0, (y_max - y_min) * 0.08)

        plot_theme = get_plot_theme()
        win = GraphicsLayoutWidget()
        win.setBackground(plot_theme.background)
        self.graph_layout.addWidget(win)
        self.p1 = None
        self.p1_axis = None

        # 检查复选框状态，决定是否显示高斯拟合图
        show_gaussian = self.checkBox_show_gaussian.isChecked()
        if show_gaussian:
            self.p1_axis = SpectrumBottomAxis()
            self.p1 = win.addPlot(title="高斯拟合图", axisItems={"bottom": self.p1_axis})
            self.p1.setLabel('left', 'COUNTS')
            self.p1.showGrid(x=True, y=True, alpha=0.25)
            self._apply_plot_limits(self.p1, y_min - y_padding, y_max + y_padding, x_range)
            win.nextRow()

        self.p2_axis = SpectrumBottomAxis()
        self.p2 = win.addPlot(title=title, axisItems={"bottom": self.p2_axis})
        self.p2.setLabel('left', 'COUNTS')
        self.p2.showGrid(x=True, y=True, alpha=0.25)
        self._apply_plot_limits(self.p2, y_min - y_padding, y_max + y_padding, x_range)
        self.spectrum_plot = self.p2.plot(axis_x_values, y_values, pen=mkPen(plot_theme.spectrum_curve, width=1.2), name='质谱图')
        self._optimize_curve(self.spectrum_plot)
        self._configure_spectrum_plot_interaction()

        # 添加 LinearRegionItem 到 p2，设置移动模式和边界约束
        peak_idx = int(np.nanargmax(y_values))
        peak_center = float(axis_x_values[peak_idx])
        finite_axis_x = axis_x_values[np.isfinite(axis_x_values)]
        if finite_axis_x.size > 1:
            spacing_values = np.diff(np.sort(finite_axis_x))
            spacing_values = spacing_values[np.isfinite(spacing_values) & (spacing_values > 0)]
            data_spacing = float(np.median(spacing_values)) if spacing_values.size else 1.0
        else:
            data_spacing = 1.0
        min_half_width = max(data_spacing * 6, 0.02 if self.x_axis_mode == "mz" else 4.0)
        half_width = max(min_half_width, x_range * 0.001)
        region_left = max(self.plot_x_min, peak_center - half_width)
        region_right = min(self.plot_x_max, peak_center + half_width)
        self.selection_region = LinearRegionItem(
            values=[region_left, region_right],
            bounds=(self.plot_x_min, self.plot_x_max),
            movable=True,  # 允许整体移动
            brush=pg.mkBrush(color=plot_theme.region_brush)
        )
        self.selection_region.setZValue(20)
        self.p2.addItem(self.selection_region, ignoreBounds=True)

        self.refresh_plot_axis_mode()
        self.update_region_readout()
        if show_gaussian:
            self.update_gaussian_fit()

        self.selection_region.sigRegionChanged.connect(self.update_region_readout)
        if show_gaussian:
            if hasattr(self.selection_region, "sigRegionChangeFinished"):
                self.selection_region.sigRegionChangeFinished.connect(self.update_gaussian_fit)
            else:
                self.selection_region.sigRegionChanged.connect(self.update_gaussian_fit)
            self.p1.sigRangeChanged.connect(lambda window, viewRange: self.update_region(viewRange))
        self.update_peak_navigation_state()

    def _apply_plot_limits(self, plot, y_min: float, y_max: float, x_range: float) -> None:
        if self.plot_x_min is None or self.plot_x_max is None:
            return
        y_span = max(1.0, y_max - y_min)
        plot.setLimits(
            xMin=self.plot_x_min,
            xMax=self.plot_x_max,
            maxXRange=x_range,
            minXRange=max(1e-6 if self.x_axis_mode == "mz" else 1.0, x_range / 100_000),
            yMin=y_min - y_span,
            yMax=y_max + y_span,
        )
        plot.setXRange(self.plot_x_min, self.plot_x_max, padding=0)
        plot.setYRange(y_min, y_max, padding=0)

    def _configure_spectrum_plot_interaction(self) -> None:
        if self.p2 is None:
            return
        self.p2.setMouseEnabled(x=True, y=False)
        sig_x_range_changed = getattr(self.p2, "sigXRangeChanged", None)
        if sig_x_range_changed is not None:
            sig_x_range_changed.connect(
                lambda _plot, x_range: self.update_spectrum_y_range_for_visible_x(x_range)
            )
        else:
            self.p2.sigRangeChanged.connect(
                lambda _plot, view_range: self.update_spectrum_y_range_for_visible_x(view_range[0])
            )

    def update_spectrum_y_range_for_visible_x(self, x_range=None) -> None:
        if self.p2 is None or self._updating_spectrum_y_range:
            return
        if x_range is None:
            x_range = self.p2.viewRange()[0]
        self._set_spectrum_y_range_for_x_range(float(x_range[0]), float(x_range[1]))

    def _set_spectrum_y_range_for_x_range(
        self,
        x_left: float,
        x_right: float,
        include_y: float | None = None,
    ) -> None:
        if self.p2 is None or self.current_plot_axis_x.size == 0 or self.current_plot_y.size == 0:
            return
        if not (np.isfinite(x_left) and np.isfinite(x_right)):
            return
        if x_right < x_left:
            x_left, x_right = x_right, x_left

        start = int(np.searchsorted(self.current_plot_axis_x, x_left, side="left"))
        end = int(np.searchsorted(self.current_plot_axis_x, x_right, side="right"))
        start = max(0, min(start, self.current_plot_y.size - 1))
        end = max(start + 1, min(end, self.current_plot_y.size))
        segment_y = self.current_plot_y[start:end]
        segment_y = segment_y[np.isfinite(segment_y)]
        if segment_y.size == 0:
            return

        y_min = float(np.nanmin(segment_y))
        y_max = float(np.nanmax(segment_y))
        if include_y is not None and np.isfinite(include_y):
            y_max = max(y_max, float(include_y))
        y_span = max(1.0, y_max - y_min)
        lower = y_min - y_span * 0.08
        upper = y_max + y_span * 0.20
        self._updating_spectrum_y_range = True
        try:
            self.p2.setYRange(lower, upper, padding=0)
        finally:
            self._updating_spectrum_y_range = False

    @staticmethod
    def _optimize_curve(curve) -> None:
        for method_name, args, kwargs in (
            ("setClipToView", (True,), {}),
            ("setDownsampling", (), {"auto": True, "method": "peak"}),
        ):
            method = getattr(curve, method_name, None)
            if method is not None:
                try:
                    method(*args, **kwargs)
                except TypeError:
                    pass

    def plot_graph(self):
        """从单个文件绘制图表"""
        try:
            self.clear_widgets()
            file_path = self.lineEdit.text().strip('"')
            spectrum = read_bl03u_txt(file_path, trim_start=4000)
            self.current_time_offset = float(spectrum.x[0]) if len(spectrum.x) else 0.0
            self.setup_plots(spectrum.x, spectrum.y)
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'无法加载文件: {str(e)}')

    def plot_graph_sum(self):
        """从文件夹中的多个文件绘制累积图表"""
        try:
            self.clear_widgets()
            folder_path = self.folder_path.text().strip('"')
            if not os.path.exists(folder_path):
                raise FileNotFoundError("文件夹不存在")

            spectrum = sum_spectra(folder_path, suffixes=(".txt",), trim_start=4000)
            self.current_time_offset = float(spectrum.x[0]) if len(spectrum.x) else 0.0
            self.setup_plots(spectrum.x, spectrum.y, title="累加质谱图")
        except Exception as e:
            QMessageBox.warning(self, 'Warning', str(e))

    def on_gaussian_visibility_toggled(self, checked: bool):
        """切换高斯拟合图时复用当前谱图数据，避免重新读取文件。"""
        if self.current_plot_x.size == 0 or self.current_plot_y.size == 0:
            return

        region = self.selection_region.getRegion() if self.selection_region is not None else None
        view_range = self.p2.viewRange() if self.p2 is not None else None
        x_values = self.current_plot_x.copy()
        y_values = self.current_plot_y.copy()
        title = self.current_plot_title

        try:
            self.clear_plot_area()
            self.setup_plots(x_values, y_values, title=title)
            if region is not None and self.selection_region is not None:
                self.selection_region.setRegion(self._clamp_region_values(*region))
            self._restore_plot_view_range(view_range)
            self.update_plot()
            self.update_region_readout()
            if checked:
                self.update_gaussian_fit()
        except Exception as exc:
            QMessageBox.warning(self, "Warning", f"切换高斯拟合图失败: {exc}")

    def _restore_plot_view_range(self, view_range) -> None:
        if self.p2 is None or not view_range:
            return
        try:
            left, right = self._clamp_region_values(view_range[0][0], view_range[0][1])
            if right > left:
                self.p2.setXRange(left, right, padding=0)
                self.update_spectrum_y_range_for_visible_x((left, right))
        except (TypeError, ValueError, IndexError):
            return

    def update_gaussian_fit(self):
        """准备更新高斯拟合图"""
        if not self.pending_update:
            self.pending_update = True
            self.update_timer.start()

    def update_region_readout(self):
        if self.selection_region is None:
            return
        left, right = self._clamp_region_values(*self.selection_region.getRegion())
        current_left, current_right = self.selection_region.getRegion()
        if (left, right) != (current_left, current_right) and not self._clamping_region:
            self._clamping_region = True
            try:
                self.selection_region.setRegion([left, right])
            finally:
                self._clamping_region = False
        center_axis = (left + right) / 2
        try:
            center_tof = self._axis_x_to_tof(center_axis)
            left_tof, right_tof = self._axis_range_to_tof(left, right)
            center_mz = float(self.current_calibration().tof_to_mz(center_tof))
        except Exception:
            return
        self.label_8.setText(f"{center_tof:.2f}")
        self.label_9.setText(f"{center_mz:.5f}")
        self.label_10.setText(f"{left_tof:.2f}")
        self.label_11.setText(f"{right_tof:.2f}")

    def _clamp_region_values(self, left: float, right: float) -> tuple[float, float]:
        if self.plot_x_min is None or self.plot_x_max is None:
            return float(left), float(right)
        left = max(self.plot_x_min, min(float(left), self.plot_x_max))
        right = max(self.plot_x_min, min(float(right), self.plot_x_max))
        if right < left:
            left, right = right, left
        if right == left:
            right = min(self.plot_x_max, left + 1.0)
            left = max(self.plot_x_min, right - 1.0)
        return left, right

    def _table_float(self, row: int, column: int) -> float | None:
        item = self.peakData.item(row, column)
        if item is None:
            return None
        text = item.text().strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _peak_row_values(self, row: int) -> tuple[float, float, float] | None:
        time = self._table_float(row, 1)
        mz = self._table_float(row, 2)
        intensity = self._table_float(row, 3)
        if time is None or mz is None or intensity is None:
            return None
        if not (np.isfinite(time) and np.isfinite(mz) and np.isfinite(intensity)):
            return None
        return time, mz, intensity

    def _valid_peak_rows(self) -> list[int]:
        return [row for row in range(self.peakData.rowCount()) if self._peak_row_values(row) is not None]

    def on_peak_selection_changed(self):
        self.focus_selected_peak()
        self.update_peak_navigation_state()

    def on_peak_table_item_changed(self, item: QTableWidgetItem):
        if self._updating_peak_table:
            return
        self.update_plot()
        self.update_peak_navigation_state()

    def update_peak_navigation_state(self):
        if not hasattr(self, "previousPeakButton"):
            return
        rows = self._valid_peak_rows()
        current = self.peakData.currentRow()
        has_current = current in rows
        self.previousPeakButton.setEnabled(
            bool(rows) if not has_current else any(row < current for row in rows)
        )
        self.nextPeakButton.setEnabled(
            bool(rows) if not has_current else any(row > current for row in rows)
        )
        self.updatePeakRangeButton.setEnabled(
            has_current
            and self.selection_region is not None
            and self.current_plot_y.size > 0
        )

    def _select_peak_row(self, row: int) -> None:
        if row < 0 or row >= self.peakData.rowCount():
            return
        self.peakData.setCurrentCell(row, 0)
        self.peakData.selectRow(row)
        item = self.peakData.item(row, 0)
        if item is not None:
            self.peakData.scrollToItem(
                item,
                QtWidgets.QAbstractItemView.ScrollHint.PositionAtCenter,
            )
        self.update_peak_navigation_state()

    def select_previous_peak(self):
        rows = self._valid_peak_rows()
        current = self.peakData.currentRow()
        if current not in rows and rows:
            self._select_peak_row(rows[-1])
            return
        previous_rows = [row for row in rows if row < current]
        if previous_rows:
            self._select_peak_row(previous_rows[-1])

    def select_next_peak(self):
        rows = self._valid_peak_rows()
        current = self.peakData.currentRow()
        if current not in rows and rows:
            self._select_peak_row(rows[0])
            return
        next_rows = [row for row in rows if row > current]
        if next_rows:
            self._select_peak_row(next_rows[0])

    def _set_peak_table_row(self, row: int, peak_data: dict) -> None:
        self.peakData.setItem(row, 0, QTableWidgetItem(str(peak_data["species"])))
        self.peakData.setItem(row, 1, QTableWidgetItem(f"{float(peak_data['time']):.2f}"))
        self.peakData.setItem(row, 2, QTableWidgetItem(f"{float(peak_data['mz']):.2f}"))
        self.peakData.setItem(row, 3, QTableWidgetItem(f"{float(peak_data['intensity']):.2f}"))
        self.peakData.setItem(row, 4, QTableWidgetItem(f"{float(peak_data['left']):.2f}"))
        self.peakData.setItem(row, 5, QTableWidgetItem(f"{float(peak_data['right']):.2f}"))

    def _peak_row_is_empty(self, row: int) -> bool:
        for column in range(self.peakData.columnCount()):
            item = self.peakData.item(row, column)
            if item is not None and item.text().strip():
                return False
        return True

    def _peak_row_data(self, row: int) -> dict | None:
        values = self._peak_row_values(row)
        if values is None:
            return None

        time, mz, intensity = values
        species_item = self.peakData.item(row, 0)
        species = species_item.text().strip() if species_item and species_item.text().strip() else "Unknown"
        left = self._table_float(row, 4)
        right = self._table_float(row, 5)
        return {
            "species": species,
            "time": time,
            "mz": mz,
            "intensity": intensity,
            "left": time if left is None else left,
            "right": time if right is None else right,
        }

    def _set_peak_table_text_row(self, row: int, values: list[str]) -> None:
        for column, value in enumerate(values[: self.peakData.columnCount()]):
            if value:
                self.peakData.setItem(row, column, QTableWidgetItem(value))

    def _insert_peak_data_sorted(self, peak_data: dict) -> int:
        entries = []
        trailing_rows = []
        for row in range(self.peakData.rowCount()):
            row_data = self._peak_row_data(row)
            if row_data is not None:
                entries.append((False, row_data))
            elif not self._peak_row_is_empty(row):
                trailing_rows.append([
                    self.peakData.item(row, column).text() if self.peakData.item(row, column) else ""
                    for column in range(self.peakData.columnCount())
                ])

        entries.append((True, peak_data))
        entries.sort(key=lambda entry: (float(entry[1]["mz"]), float(entry[1]["time"])))

        self.peakData.setUpdatesEnabled(False)
        self.peakData.setRowCount(0)
        insert_row = 0
        for is_new, row_data in entries:
            row = self.peakData.rowCount()
            self.peakData.insertRow(row)
            self._set_peak_table_row(row, row_data)
            if is_new:
                insert_row = row

        for values in trailing_rows:
            row = self.peakData.rowCount()
            self.peakData.insertRow(row)
            self._set_peak_table_text_row(row, values)

        self.peakData.setUpdatesEnabled(True)
        return insert_row

    def _peak_from_current_region(self):
        if self.selection_region is None or self.current_plot_y.size == 0:
            return None
        min_x, max_x = self._clamp_region_values(*self.selection_region.getRegion())
        left_tof, right_tof = self._axis_range_to_tof(min_x, max_x)
        left_idx, right_idx = self._tof_bounds_to_indices(left_tof, right_tof)
        return core_add_manual_peak(
            self.current_plot_y,
            left_idx,
            right_idx,
            calibration=self.current_calibration(),
            time_offset=self.current_time_offset,
        )

    def update_current_peak_range_from_region(self):
        row = self.peakData.currentRow()
        if row < 0 or self._peak_row_values(row) is None:
            QMessageBox.warning(self, "提示", "请先在卡峰范围表中选择一个峰。")
            return

        peak = self._peak_from_current_region()
        if peak is None:
            QMessageBox.warning(self, "提示", "当前框选范围无效，无法更新卡峰范围。")
            return

        species_item = self.peakData.item(row, 0)
        species = species_item.text() if species_item and species_item.text().strip() else peak.species
        peak_data = {
            "species": species,
            "time": peak.time,
            "mz": peak.mz,
            "intensity": peak.intensity,
            "left": peak.left_bound + self.current_time_offset,
            "right": peak.right_bound + self.current_time_offset,
        }
        new_row = self.update_peak_in_table(row, peak_data)
        self._select_peak_row(new_row)
        self.update_plot()
        self.update_region_readout()
        if self.p1 is not None and self.checkBox_show_gaussian.isChecked():
            self.update_gaussian_fit()
        self.statusbar.showMessage("已根据当前框选范围更新当前峰", 3000)

    def focus_selected_peak(self):
        """根据右侧峰表当前行，将主图定位到对应峰位。"""
        if self._updating_peak_table or self.p2 is None or self.selection_region is None:
            return
        row = self.peakData.currentRow()
        if row < 0:
            return

        peak_time = self._table_float(row, 1)
        peak_intensity = self._table_float(row, 3)
        left = self._table_float(row, 4)
        right = self._table_float(row, 5)
        if peak_time is None or not np.isfinite(peak_time):
            return

        full_tof_range = self._tof_x_range()
        fallback_half_width = max(2.0, full_tof_range * 0.0025)
        if left is None or right is None or not np.isfinite(left) or not np.isfinite(right) or left == right:
            left = peak_time - fallback_half_width
            right = peak_time + fallback_half_width

        left, right = self._clamp_tof_range(left, right)
        self.selection_region.setRegion(self._tof_range_to_axis_range(left, right))
        self.update_region_readout()
        self._center_plot_on_peak(peak_time, peak_intensity, left, right)
        if self.p1 is not None and self.checkBox_show_gaussian.isChecked():
            self.update_gaussian_fit()

    def _plot_x_range(self) -> float:
        if self.plot_x_min is None or self.plot_x_max is None:
            return 1.0
        return max(1e-9, self.plot_x_max - self.plot_x_min)

    def _tof_x_range(self) -> float:
        if self.current_plot_x.size == 0:
            return 1.0
        return max(1.0, float(np.nanmax(self.current_plot_x) - np.nanmin(self.current_plot_x)))

    def _clamp_tof_range(self, left: float, right: float) -> tuple[float, float]:
        if self.current_plot_x.size == 0:
            return float(left), float(right)
        tof_min = float(np.nanmin(self.current_plot_x))
        tof_max = float(np.nanmax(self.current_plot_x))
        left = max(tof_min, min(float(left), tof_max))
        right = max(tof_min, min(float(right), tof_max))
        if right < left:
            left, right = right, left
        if right == left:
            right = min(tof_max, left + 1.0)
            left = max(tof_min, right - 1.0)
        return left, right

    def _center_plot_on_peak(
        self,
        peak_time: float,
        peak_intensity: float | None,
        left: float,
        right: float,
    ) -> None:
        if self.p2 is None or self.plot_x_min is None or self.plot_x_max is None:
            return

        full_range = self._plot_x_range()
        left_axis, right_axis = self._tof_range_to_axis_range(left, right)
        peak_axis = float(self._tof_to_axis_x(peak_time))
        selected_width = max(1e-9, abs(right_axis - left_axis))
        minimum_view_width = 0.05 if self.x_axis_mode == "mz" else 10.0
        view_width = min(full_range, max(selected_width * 6, full_range * 0.02, minimum_view_width))
        center = max(self.plot_x_min, min(peak_axis, self.plot_x_max))
        x_left = center - view_width / 2
        x_right = center + view_width / 2
        if x_left < self.plot_x_min:
            x_right += self.plot_x_min - x_left
            x_left = self.plot_x_min
        if x_right > self.plot_x_max:
            x_left -= x_right - self.plot_x_max
            x_right = self.plot_x_max
        x_left = max(self.plot_x_min, x_left)
        x_right = min(self.plot_x_max, x_right)
        if x_right > x_left:
            self.p2.setXRange(x_left, x_right, padding=0)
            self._set_spectrum_y_range_for_x_range(x_left, x_right, peak_intensity)

    def on_update_timeout(self):
        """实际执行更新高斯拟合图"""
        self.update_timer.stop()
        self.pending_update = False

        if self.selection_region and self.p1 and self.p2 and self.current_plot_x.size:
            try:
                region_left, region_right = self._clamp_region_values(*self.selection_region.getRegion())
                minX, maxX = self._axis_range_to_tof(region_left, region_right)
                x_start = int(np.searchsorted(self.current_plot_x, minX, side="left"))
                x_end = int(np.searchsorted(self.current_plot_x, maxX, side="right"))
                x_start = max(0, min(x_start, self.current_plot_x.size))
                x_end = max(x_start, min(x_end, self.current_plot_x.size))
                x_data = self.current_plot_x[x_start:x_end]
                y_data = self.current_plot_y[x_start:x_end]

                if len(x_data) != len(y_data):
                    logger.warning("Gaussian fit skipped because x/y data lengths differ")
                    return

                if len(x_data) < 3 or len(y_data) < 3:
                    logger.warning("Gaussian fit skipped because selected region has too few points")
                    return

                # 数据预处理：移除基线
                baseline = np.min(y_data)
                y_data = y_data - baseline
                if np.sum(y_data) <= 0:
                    return

                # 提供更合理的初始猜测值
                max_y = np.max(y_data)
                mean_x = np.sum(x_data * y_data) / np.sum(y_data)  # 加权平均作为中心点
                sigma_guess = (maxX - minX) / 4  # 使用区域宽度的1/4作为初始sigma
                initial_guess = [mean_x, sigma_guess, max_y]

                # 设置更宽松的参数边界
                bounds = (
                    [min(x_data), 0.1, 0],  # 下界
                    [max(x_data), (maxX - minX) / 2, max_y * 2]  # 上界
                )

                # 使用更稳健的拟合方法
                popt, pcov = curve_fit(
                    self.func_gaosi, 
                    x_data, 
                    y_data, 
                    p0=initial_guess, 
                    bounds=bounds, 
                    maxfev=10000,  # 增加最大迭代次数
                    method='trf'  # 使用更稳健的拟合方法
                )

                readout_labels = (self.label_8, self.label_9, self.label_10, self.label_11)
                if any(label is None for label in readout_labels):
                    return
                self.label_8.setText(str(round(popt[0], 2)))  # 使用拟合得到的中心点
                self.label_9.setText(f"{float(self.current_calibration().tof_to_mz(popt[0])):.5f}")
                self.label_10.setText(str(round(minX, 2)))
                self.label_11.setText(str(round(maxX, 2)))

                self.p1.clear()
                x_interp = np.linspace(x_data[0], x_data[-1], 1000)
                y_fit = self.func_gaosi(x_interp, *popt) + baseline  # 添加回基线
                x_interp_axis = self._tof_to_axis_x(x_interp)
                plot_theme = get_plot_theme()
                self.p1.plot(x_interp_axis, y_fit, pen=mkPen(plot_theme.fit_curve, width=2), name='拟合图')
                self.p1.setXRange(region_left, region_right, padding=0)
                y_min = float(np.nanmin(y_fit))
                y_max = float(np.nanmax(y_fit))
                y_span = max(1.0, y_max - y_min)
                self.p1.setYRange(y_min - y_span * 0.08, y_max + y_span * 0.20, padding=0)

            except Exception:
                logger.exception("Gaussian fit failed")

    @staticmethod
    def func_gaosi(x, miu, sigma, amplitude):
        """高斯函数"""
        return amplitude * np.exp(-(x - miu)**2 / (2 * sigma**2))

    # 读取表格2数据
    def read_table_data(self):
        table = self.findChild(QTableWidget, "peakData")
        data = {
            'Species': [],
            '飞行时间': [],
            '质量数 (m/z)': [],
            '强度': [],
            '左边界': [],
            '右边界': []
        }

        for row in range(table.rowCount()):
            # 检查该行是否有任何数据
            has_data = False
            for col in range(table.columnCount()):
                item = table.item(row, col)
                if item and item.text().strip():
                    has_data = True
                    break
            
            if not has_data:
                continue

            # 读取每一列的数据
            for col in range(table.columnCount()):
                item = table.item(row, col)
                value = item.text().strip() if item and item.text().strip() else ""
                
                # 根据列名进行数据类型转换
                if col in [1, 2, 3, 4, 5]:  # 数值类型的列
                    try:
                        value = float(value) if value else 0.0
                    except ValueError:
                        value = 0.0
                
                data[table.horizontalHeaderItem(col).text()].append(value)

        df = pd.DataFrame(data)
        return df

    # 读取表格2数据
    def read_table_data_1(self):
        table = self.findChild(QTableWidget, "region")
        column1_data = []
        column2_data = []

        for row in range(table.rowCount()):
            item1 = table.item(row, 0)  # 第一列数据
            item2 = table.item(row, 1)  # 第二列数据

            if item1 is not None and item1.text():
                column1_data.append(item1.text())

            if item2 is not None and item2.text():
                column2_data.append(item2.text())

        df = pd.DataFrame({
            'time': column1_data,
            'mass': column2_data
        })
        return df

        # print(df)

    def calculate(self):
        """计算飞行时间质谱定标参数 A、B、C"""
        try:
            # 读取定标数据
            df = self.read_table_data_1()
            
            # 确保有至少3个点
            if len(df) < 3:
                QMessageBox.warning(self, "警告", "至少需要3个定标点才能进行二次拟合！")
                return
            
            # 准备数据点列表 [(tof, mz), ...]
            points = []
            for _, row in df.iterrows():
                try:
                    tof = float(row['time'])
                    mz = float(row['mass'])
                    points.append((tof, mz))
                except (ValueError, TypeError):
                    continue
            
            if len(points) < 3:
                QMessageBox.warning(self, "警告", "有效的定标点不足3个，请检查数据！")
                return
            
            # 使用核心库中的拟合函数
            from bl03u_masstool.core.calibration import fit_quadratic_calibration, score_quadratic_calibration
            calibration = fit_quadratic_calibration(points)
            r2 = score_quadratic_calibration(points, calibration)
            
            # 更新界面显示
            self.lineEdit_4.setText(f"{calibration.a:.6e}")
            self.lineEdit_5.setText(f"{calibration.b:.6e}")
            self.lineEdit_6.setText(f"{calibration.c:.6e}")
            
            # 显示结果
            result_text = (
                f"m/z = {calibration.c:.8g} + {calibration.b:.8g}×TOF + {calibration.a:.8g}×TOF²\n"
                f"R² = {r2:.6f}"
            )
            self.label_4.setText(result_text)
            
            # 保存到配置文件
            try:
                from bl03u_masstool.core.config import save_calibration_config, save_calibration_points
                save_calibration_config(calibration)
                save_calibration_points(points)
            except Exception:
                logger.exception("Failed to save calibration configuration")

            # 同步到 ProjectSettingsManager，防止切换页面后参数被覆盖
            self._sync_calibration_to_project_settings()
            
            # 显示成功消息
            QMessageBox.information(self, "成功", f"定标完成！\nR² = {r2:.6f}")
            
        except Exception as e:
            logger.exception("Failed to calculate calibration parameters")
            QMessageBox.critical(self, "错误", f"计算定标参数时发生错误：{str(e)}")

    def transfer(self):
        time = self.findChild(QLineEdit, "lineEdit_2")
        t = float(time.text())
        m = self.current_calibration().tof_to_mz(t)
        lineEdit = self.findChild(QLineEdit, "lineEdit_3")  # 找到名为lineEdit_7的QLineEdit对象
        lineEdit.setText(str(m))  # 将值设置为QLineEdit的文本内容

    def save(self):
        try:
            df = self.read_table_data()
            if df.empty:
                QMessageBox.warning(self, "警告", "没有数据可以保存！")
                return
                
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存文件",
                str(ensure_output_dir("exports", "peak_ranges") / "peak_ranges.csv"),
                "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*.*)"
            )
            
            if file_path:
                if file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False, encoding='utf-8-sig')
                elif file_path.endswith('.xlsx'):
                    df.to_excel(file_path, index=False)
                else:
                    df.to_csv(file_path, index=False, encoding='utf-8-sig')
                QMessageBox.information(self, "成功", "文件保存成功！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存文件时发生错误：{str(e)}")

    def add_peak(self):
        """添加峰值到表格"""
        try:
            # 获取当前选择区域的范围
            if self.selection_region is None:
                raise ValueError("请先加载谱图并选择卡峰区域")
            minX, maxX = self._clamp_region_values(*self.selection_region.getRegion())
            left_tof, right_tof = self._axis_range_to_tof(minX, maxX)
            left_idx, right_idx = self._tof_bounds_to_indices(left_tof, right_tof)
            peak = core_add_manual_peak(
                self.current_plot_y,
                left_idx,
                right_idx,
                calibration=self.current_calibration(),
                time_offset=self.current_time_offset,
            )
            if peak is None:
                raise ValueError("选择区域无效")
            
            peak_data = {
                "species": peak.species,
                "time": peak.time,
                "mz": peak.mz,
                "intensity": peak.intensity,
                "left": peak.left_bound + self.current_time_offset,
                "right": peak.right_bound + self.current_time_offset,
            }
            self._updating_peak_table = True
            try:
                row = self._insert_peak_data_sorted(peak_data)
            finally:
                self._updating_peak_table = False
            self._select_peak_row(row)
            self.update_plot()
            
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'添加峰值失败: {str(e)}')

    def auto_find_peaks(self):
        """自动寻峰功能"""
        try:
            # 获取当前图表数据
            if self.spectrum_plot is None or self.spectrum_plot.yData is None:
                QMessageBox.warning(self, "提示", "请先加载谱图数据。")
                return
            y_data = self.spectrum_plot.yData
            peak_config = self.current_peak_detection_config()
            peaks = self.detect_peaks_for_config(y_data, peak_config)

            # 在表格中显示检测到的峰值
            self.display_peaks(peaks)

            # 表格是峰标注的单一数据源，避免重复叠加旧标注。
            self.update_plot()

        except Exception as e:
            QMessageBox.warning(self, 'Error', f'自动寻峰失败: {str(e)}')

    def current_peak_detection_config(self):
        try:
            from bl03u_masstool.core.project_settings import ProjectSettingsManager

            manager = ProjectSettingsManager()
            if manager.has_project_path():
                return manager.get().to_peak_detection_config()
        except Exception:
            pass
        return load_peak_detection_config()

    def detect_peaks_for_config(self, y_data, peak_config):
        peak_kwargs = peak_config.to_peak_kwargs()
        return detect_peaks_by_algorithm(
            y_data,
            calibration=self.current_calibration(),
            start_idx=0,
            end_idx=len(y_data),
            time_offset=self.current_time_offset,
            **peak_kwargs,
        )

    def display_peaks(self, peaks):
        """在表格中显示检测到的峰值"""
        table = self.findChild(QTableWidget, "peakData")
        table.setUpdatesEnabled(False)
        self._updating_peak_table = True
        try:
            table.setRowCount(len(peaks))

            # 设置表格的列标题
            headers = ["Species", "飞行时间", "质量数 (m/z)", "强度", "左边界", "右边界"]
            table.setHorizontalHeaderLabels(headers)

            sorted_peaks = sorted(peaks, key=lambda peak: (float(peak.mz), float(peak.time)))
            for i, peak in enumerate(sorted_peaks):
                # 按列顺序设置数据
                table.setItem(i, 0, QTableWidgetItem("Unknown"))  # Species 列
                table.setItem(i, 1, QTableWidgetItem(f"{peak.time:.2f}"))  # 飞行时间列
                table.setItem(i, 2, QTableWidgetItem(f"{peak.mz:.2f}"))    # 质量数列
                table.setItem(i, 3, QTableWidgetItem(f"{peak.intensity:.2f}"))  # 强度列
                table.setItem(i, 4, QTableWidgetItem(f"{peak.left_bound + self.current_time_offset:.2f}"))  # 左边界列
                table.setItem(i, 5, QTableWidgetItem(f"{peak.right_bound + self.current_time_offset:.2f}"))  # 右边界列
        finally:
            self._updating_peak_table = False
            table.setUpdatesEnabled(True)
        self.update_peak_navigation_state()

    def update_region(self, viewRange):
        """更新选择区域"""
        minX, maxX = viewRange[0]
        if self.selection_region is not None:
            left, right = self._clamp_region_values(minX, maxX)
            self.selection_region.setRegion([left, right])

    def on_tab_changed(self):
        """选项卡切换事件处理"""
        # 检查当前选项卡是否为"质谱定标"
        if self.peakResult.tabText(self.peakResult.currentIndex()) == "质谱定标":
            self.label_4.setVisible(True)  # 显示文本框
            self.pushButton_3.setVisible(True)
        else:
            self.label_4.setVisible(False)  # 隐藏文本框
            self.pushButton_3.setVisible(False)

    def show_peak_context_menu(self, position):
        """显示峰值表格的上下文菜单"""
        menu = QMenu()
        add_action = menu.addAction("添加峰值")
        delete_action = menu.addAction("删除峰值")
        edit_action = menu.addAction("修改峰值")
        
        # 获取点击的行
        row = self.peakData.rowAt(position.y())
        
        # 根据是否选中行来启用/禁用菜单项
        delete_action.setEnabled(row >= 0)
        edit_action.setEnabled(row >= 0)
        
        action = menu.exec(self.peakData.mapToGlobal(position))
        
        if action == add_action:
            self.add_peak_manually()
        elif action == delete_action and row >= 0:
            self.delete_peak(row)
        elif action == edit_action and row >= 0:
            self.edit_peak(row)

    def add_peak_manually(self):
        """手动添加峰值"""
        dialog = PeakDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            peak_data = dialog.get_peak_data()
            self.add_peak_to_table(peak_data)
            self.update_plot()

    def delete_peak(self, row):
        """删除峰值"""
        reply = QMessageBox.question(self, '确认删除', 
                                   '确定要删除这个峰值吗？',
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                   QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.peakData.removeRow(row)
            self.update_plot()  # 更新图表
            self.update_peak_navigation_state()

    def edit_peak(self, row):
        """编辑峰值"""
        current_data = []
        for col in range(self.peakData.columnCount()):
            item = self.peakData.item(row, col)
            current_data.append(item.text() if item else "")
            
        dialog = PeakDialog(self, current_data)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            peak_data = dialog.get_peak_data()
            new_row = self.update_peak_in_table(row, peak_data)
            self._select_peak_row(new_row)
            self.update_plot()  # 更新图表
            self.update_peak_navigation_state()

    def add_peak_to_table(self, peak_data):
        """将峰值添加到表格"""
        self._updating_peak_table = True
        try:
            row = self._insert_peak_data_sorted(peak_data)
        finally:
            self._updating_peak_table = False
        self._select_peak_row(row)

    def update_peak_in_table(self, row, peak_data):
        """更新表格中的峰值"""
        self._updating_peak_table = True
        try:
            self.peakData.removeRow(row)
            new_row = self._insert_peak_data_sorted(peak_data)
        finally:
            self._updating_peak_table = False
        self.update_peak_navigation_state()
        return new_row

    def update_plot(self):
        """更新图表显示"""
        if self.p2 is None:
            return
        # 清除现有标注
        for item in self.p2.items[:]:
            if isinstance(item, pg.TextItem):
                self.p2.removeItem(item)
        
        # 重新添加所有峰值标注
        for row in range(self.peakData.rowCount()):
            values = self._peak_row_values(row)
            if values is None:
                continue
            time, mz, intensity = values
            text_item = pg.TextItem(text=f'{mz:.2f}', color='red', anchor=(0.5, 1.5))
            text_item.setPos(float(self._tof_to_axis_x(time)), intensity)
            self.p2.addItem(text_item)

    def clear_peak_data(self):
        """清除峰值数据表格中的所有数据"""
        try:
            # 清除表格数据
            self._updating_peak_table = True
            try:
                self.peakData.setRowCount(0)
            finally:
                self._updating_peak_table = False
            
            # 清除图表上的峰值标注
            if self.p2:
                for item in self.p2.items[:]:
                    if isinstance(item, pg.TextItem):
                        self.p2.removeItem(item)
            
            # 清除高斯拟合图
            if self.p1:
                self.p1.clear()
            
            # 清除标签显示
            time_max = self.findChild(QLabel, "label_8")
            mass = self.findChild(QLabel, "label_9")
            peak_left = self.findChild(QLabel, "label_10")
            peak_right = self.findChild(QLabel, "label_11")
            
            if time_max: time_max.setText("")
            if mass: mass.setText("")
            if peak_left: peak_left.setText("")
            if peak_right: peak_right.setText("")
            self.update_peak_navigation_state()
            
            QMessageBox.information(self, "成功", "峰值数据已清除")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"清除数据时发生错误：{str(e)}")
