from __future__ import annotations

import os

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import QMainWindow, QTableWidgetItem, QTableWidget, QLabel, QLineEdit, QGraphicsPixmapItem, \
    QGraphicsScene, QGraphicsView, QMessageBox, QMenu, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QDialogButtonBox, QFileDialog
import pyqtgraph as pg
from pyqtgraph import mkPen, LinearRegionItem, GraphicsLayoutWidget
import numpy as np
from PyQt6 import QtCore
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtGui import QPixmap

from frontends.pyqt_app.ui_massspec import Ui_MainWindow
from core.calibration import Calibration
from core.config import load_calibration_config, load_calibration_points, load_peak_detection_config
from core.output_paths import ensure_output_dir
from core.peak_detection import add_manual_peak as core_add_manual_peak
from core.peak_detection import detect_peaks_in_range
from core.peak_detection import detect_peaks_prominence
from core.spectrum_io import read_bl03u_txt, sum_spectra
from core.normalization import load_normalization_settings
from frontends.pyqt_app.dialogs import (
    CommonParametersDialog,
    CoreToolsDialog,
    IonizationEnergyLookupWidget,
    IsotopeAbundanceDialog,
    PIESpeciesFitDialog,
    TemperatureScanDialog,
)


class SpectrumBottomAxis(pg.AxisItem):
    """Bottom axis that keeps plot data in TOF while displaying TOF or m/z labels."""

    def __init__(self, orientation: str = "bottom"):
        super().__init__(orientation=orientation)
        self.display_mode = "tof"
        self.calibration = Calibration()

    def set_display_mode(self, mode: str, calibration: Calibration) -> None:
        self.display_mode = mode if mode in {"tof", "mz"} else "tof"
        self.calibration = calibration
        self.picture = None
        self.update()

    def tickStrings(self, values, scale, spacing):
        if self.display_mode != "mz":
            return super().tickStrings(values, scale, spacing)

        try:
            mz_values = self.calibration.tof_to_mz(np.asarray(values, dtype=float))
        except Exception:
            return ["" for _ in values]
        return [self._format_tick(float(value)) for value in np.atleast_1d(mz_values)]

    @staticmethod
    def _format_tick(value: float) -> str:
        if not np.isfinite(value):
            return ""
        abs_value = abs(value)
        if abs_value >= 100:
            return f"{value:.0f}"
        if abs_value >= 10:
            return f"{value:.1f}"
        if abs_value >= 1:
            return f"{value:.2f}"
        return f"{value:.4f}"


class PeakDialog(QDialog):
    """峰值编辑对话框"""
    def __init__(self, parent=None, peak_data=None):
        super().__init__(parent)
        self.setWindowTitle("编辑峰值")
        self.setup_ui()
        if peak_data:
            self.set_peak_data(peak_data)

    def setup_ui(self):
        layout = QFormLayout()
        
        # 创建输入框
        self.species_edit = QLineEdit()
        self.time_edit = QLineEdit()
        self.mz_edit = QLineEdit()
        self.intensity_edit = QLineEdit()
        self.left_edit = QLineEdit()
        self.right_edit = QLineEdit()
        
        # 添加到布局
        layout.addRow("Species:", self.species_edit)
        layout.addRow("飞行时间:", self.time_edit)
        layout.addRow("质量数 (m/z):", self.mz_edit)
        layout.addRow("强度:", self.intensity_edit)
        layout.addRow("左边界:", self.left_edit)
        layout.addRow("右边界:", self.right_edit)
        
        # 添加确定和取消按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            QtCore.Qt.Orientation.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addWidget(buttons)
        self.setLayout(main_layout)

    def set_peak_data(self, peak_data):
        """设置对话框中的峰值数据"""
        self.species_edit.setText(str(peak_data[0]))
        self.time_edit.setText(str(peak_data[1]))
        self.mz_edit.setText(str(peak_data[2]))
        self.intensity_edit.setText(str(peak_data[3]))
        self.left_edit.setText(str(peak_data[4]))
        self.right_edit.setText(str(peak_data[5]))

    def get_peak_data(self):
        """获取对话框中的峰值数据"""
        return {
            'species': self.species_edit.text(),
            'time': float(self.time_edit.text()),
            'mz': float(self.mz_edit.text()),
            'intensity': float(self.intensity_edit.text()),
            'left': float(self.left_edit.text()),
            'right': float(self.right_edit.text())
        }

class MainWindow(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.x_axis_mode = "tof"
        self.current_plot_x = np.array([], dtype=float)
        self.current_plot_y = np.array([], dtype=float)
        self.current_plot_title = "质谱图"
        self.plot_x_min: float | None = None
        self.plot_x_max: float | None = None
        self.p1 = None
        self.p2 = None
        self.p1_axis: SpectrumBottomAxis | None = None
        self.p2_axis: SpectrumBottomAxis | None = None
        self.selection_region = None
        self.spectrum_plot = None
        self._clamping_region = False
        self._updating_peak_table = False
        self.apply_config_defaults()
        self.configure_runtime_ui()
        self.current_time_offset = 0.0
        self.add_core_tools_launcher()
        self.pushButton.clicked.connect(self.plot_graph)
        self.pushButton_plot_graph_sum.clicked.connect(self.plot_graph_sum)
        self.setWindowIcon(QIcon('icons/icon.png'))
        self.setWindowTitle('BL03U_MassSpectrumTool')
        self.pushButton_3.clicked.connect(self.calculate)
        self.toolButton.clicked.connect(self.transfer)
        self.savePeakdata.clicked.connect(self.save)
        self.addPeak.clicked.connect(self.add_peak)
        self.pushButton_4.clicked.connect(self.auto_find_peaks)  # 自动寻峰按钮

        
        # 加载并显示图片
        self.original_pixmap = QPixmap('icons/bjt.png')  # 替换为您的图像文件路径
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
        self.peakData.itemSelectionChanged.connect(self.focus_selected_peak)

        # 添加更新按钮
        self.updatePeaks = QPushButton("更新峰标注")
        self.updatePeaks.setText("更新标注")
        self.updatePeaks.setMinimumWidth(0)
        self.updatePeaks.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.updatePeaks.setFixedHeight(28)
        self.horizontalLayout_4.addWidget(self.updatePeaks)
        self.updatePeaks.clicked.connect(self.update_all_peaks)

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

    def apply_config_defaults(self):
        try:
            calibration = load_calibration_config()
            self.lineEdit_4.setText(f"{calibration.a:.12g}")
            self.lineEdit_5.setText(f"{calibration.b:.12g}")
            self.lineEdit_6.setText(f"{calibration.c:.12g}")

            points = load_calibration_points()
            if points:
                self.region.setRowCount(len(points))
                for row, (tof, mz) in enumerate(points):
                    self.region.setItem(row, 0, QTableWidgetItem(f"{tof:g}"))
                    self.region.setItem(row, 1, QTableWidgetItem(f"{mz:g}"))
            if hasattr(self, "p2"):
                self.refresh_plot_axis_mode()
        except Exception as exc:
            print(f"加载YAML配置失败，使用界面默认值: {exc}")

    def configure_runtime_ui(self):
        self.setAccessibleName("BL03U Mass Spectrum Tool")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 760)
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
        self.tabWidget.setDocumentMode(True)
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
        self.widget_3.setMaximumWidth(460)

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

    def _build_workspace_pages(self):
        if hasattr(self, "workspace_stack"):
            return

        self.verticalLayout_9.removeItem(self.verticalLayout_7)

        self.page_nav = QtWidgets.QWidget(self.centralwidget)
        self.page_nav.setObjectName("PageNav")
        nav_layout = QHBoxLayout(self.page_nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(6)

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
        self.pie_page = PIESpeciesFitDialog(
            self.current_calibration(),
            self.normalization_settings,
            self.workspace_stack,
        )
        self.ionization_page = IonizationEnergyLookupWidget(self.workspace_stack)
        self.isotope_page = IsotopeAbundanceDialog(self.workspace_stack)

        pages = [
            ("spectrum", "质谱工作台", self.spectrum_page),
            ("temperature", "温度扫描", self.temperature_page),
            ("pie", "PIE拟合", self.pie_page),
            ("ionization", "IE查询", self.ionization_page),
            ("isotope", "分子/同位素", self.isotope_page),
        ]
        self.page_buttons: dict[str, QtWidgets.QToolButton] = {}
        self.page_button_group = QtWidgets.QButtonGroup(self.page_nav)
        self.page_button_group.setExclusive(True)
        for index, (page_name, label, page) in enumerate(pages):
            if page is not self.spectrum_page:
                self.workspace_stack.addWidget(page)
            button = QtWidgets.QToolButton(self.page_nav)
            button.setText(label)
            button.setObjectName("PageCard")
            button.setCheckable(True)
            button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            button.setFixedHeight(34)
            button.clicked.connect(lambda checked=False, name=page_name: self.switch_workspace_page(name))
            self.page_button_group.addButton(button, index)
            self.page_buttons[page_name] = button
            nav_layout.addWidget(button)
        nav_layout.addStretch(1)

        self.page_buttons["spectrum"].setChecked(True)
        self.verticalLayout_9.insertWidget(0, self.page_nav)
        self.verticalLayout_9.insertWidget(1, self.workspace_stack, stretch=1)
        self.workspace_stack.setCurrentWidget(self.spectrum_page)

    def switch_workspace_page(self, page_name: str):
        """Switch top-level workspace page and refresh shared calibration state."""
        page_map = {
            "spectrum": self.spectrum_page,
            "temperature": self.temperature_page,
            "pie": self.pie_page,
            "ionization": self.ionization_page,
            "isotope": self.isotope_page,
        }
        page = page_map.get(page_name)
        if page is None:
            return
        self.apply_config_defaults()
        calibration = self.current_calibration()
        if hasattr(self, "temperature_page"):
            self.temperature_page.calibration = calibration
        if hasattr(self, "pie_page"):
            self.pie_page.calibration = calibration
        self.workspace_stack.setCurrentWidget(page)
        if page_name in self.page_buttons:
            self.page_buttons[page_name].setChecked(True)

    def open_common_parameters(self):
        dialog = CommonParametersDialog(self.normalization_settings, self.current_calibration(), self)
        dialog.exec()
        self.apply_config_defaults()
        calibration = self.current_calibration()
        if hasattr(self, "temperature_page"):
            self.temperature_page.calibration = calibration
        if hasattr(self, "pie_page"):
            self.pie_page.calibration = calibration

    def add_core_tools_launcher_dialog_buttons(self):
        """Legacy dialog launchers kept for reference; not used by the main window."""
        actions = [
            ("通用参数设置", "normalization", "维护质量定标、归一化和Kr膨胀系数参数"),
            ("温度扫描", "temperature", "打开温度扫描曲线可视化"),
            ("PIE拟合", "pie", "打开PIE曲线提取与PICS拟合"),
            ("同位素", "isotope", "打开同位素丰度计算"),
        ]
        self.coreToolButtons = []
        for text, tab_name, tooltip in actions:
            button = QPushButton(text)
            button.setObjectName("WorkflowButton")
            button.setToolTip(tooltip)
            button.setFixedHeight(30)
            button.clicked.connect(lambda checked=False, name=tab_name: self.launch_core_tool(name))
            self.coreToolButtons.append(button)

        if hasattr(self, "workflow_actions_layout"):
            for button in self.coreToolButtons:
                self.workflow_actions_layout.addWidget(button)
        else:
            tools_layout = QHBoxLayout()
            tools_layout.addStretch()
            for button in self.coreToolButtons:
                tools_layout.addWidget(button)
            self.verticalLayout_9.insertLayout(0, tools_layout)

    def _take_all_items(self, layout):
        while layout.count():
            layout.takeAt(0)

    def _rebuild_top_controls(self):
        self.widget.setObjectName("ControlBar")
        self.widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.widget.setMinimumHeight(68)
        self.widget.setMaximumHeight(76)
        self._take_all_items(self.horizontalLayout_17)
        self.horizontalLayout_17.setContentsMargins(8, 5, 8, 5)
        self.horizontalLayout_17.setSpacing(8)
        self.verticalLayout_9.removeWidget(self.checkBox_show_gaussian)

        self.tabWidget.hide()
        source_panel = QtWidgets.QWidget(self.widget)
        source_panel.setObjectName("SourcePanel")
        source_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        source_layout = QHBoxLayout(source_panel)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(6)
        source_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)

        mode_panel = QtWidgets.QWidget(source_panel)
        mode_panel.setObjectName("ModePanel")
        mode_panel.setFixedSize(174, 30)
        mode_layout = QHBoxLayout(mode_panel)
        mode_layout.setContentsMargins(0, 0, 0, 0)
        mode_layout.setSpacing(6)
        mode_label = QtWidgets.QLabel("模式", mode_panel)
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
        source_layout.addWidget(source_stack, stretch=1)

        for path_edit in (self.lineEdit, self.folder_path):
            path_edit.setMinimumWidth(280)
            path_edit.setMaximumWidth(16777215)
        for button in (self.singleBrowseButton, self.sumBrowseButton):
            button.setMinimumWidth(60)
            button.setFixedHeight(30)
        for button in (self.pushButton, self.pushButton_plot_graph_sum):
            button.setText("加载")
            button.setMinimumWidth(72)
            button.setFixedHeight(30)
        self.horizontalLayout_17.addWidget(source_panel, stretch=7)

        tool_panel = QtWidgets.QWidget(self.widget)
        tool_panel.setObjectName("CalibrationPanel")
        tool_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        tool_layout = QVBoxLayout(tool_panel)
        tool_layout.setContentsMargins(0, 0, 0, 0)
        tool_layout.setSpacing(4)

        for widget in (self.label_17, self.label_5, self.lineEdit_4, self.label_6, self.lineEdit_5, self.label_7, self.lineEdit_6):
            widget.hide()

        conversion_row = QHBoxLayout()
        conversion_row.setContentsMargins(0, 0, 0, 0)
        conversion_row.setSpacing(6)

        self.label_2.setText("飞行时间")
        self.label_3.setText("质量数")
        self.commonParamsButton = QPushButton("通用参数", tool_panel)
        self.commonParamsButton.setObjectName("BrowseButton")
        self.commonParamsButton.setToolTip("打开全局定标、归一化和自动寻峰参数")
        self.commonParamsButton.clicked.connect(self.open_common_parameters)
        self.commonParamsButton.setFixedHeight(30)
        conversion_row.addWidget(self.commonParamsButton)
        conversion_row.addSpacing(4)
        conversion_row.addWidget(self.checkBox_show_gaussian)
        conversion_row.addSpacing(8)
        axis_label = QtWidgets.QLabel("横轴", tool_panel)
        self.xAxisModeCombo = QtWidgets.QComboBox(tool_panel)
        self.xAxisModeCombo.addItem("TOF", "tof")
        self.xAxisModeCombo.addItem("m/z", "mz")
        self.xAxisModeCombo.setFixedWidth(76)
        self.xAxisModeCombo.setCurrentIndex(1 if self.x_axis_mode == "mz" else 0)
        self.xAxisModeCombo.currentIndexChanged.connect(self.set_x_axis_mode)
        conversion_row.addWidget(axis_label)
        conversion_row.addWidget(self.xAxisModeCombo)
        conversion_row.addSpacing(8)
        conversion_row.addWidget(self.label_2)
        conversion_row.addWidget(self.lineEdit_2, stretch=1)
        arrow_group = QtWidgets.QWidget(tool_panel)
        arrow_group.setObjectName("ArrowGroup")
        arrow_layout = QHBoxLayout(arrow_group)
        arrow_layout.setContentsMargins(0, 0, 0, 0)
        arrow_layout.setSpacing(4)
        arrow_layout.addWidget(self.toolButton_2)
        arrow_layout.addWidget(self.toolButton)
        conversion_row.addWidget(arrow_group)
        conversion_row.addWidget(self.label_3)
        conversion_row.addWidget(self.lineEdit_3, stretch=1)
        conversion_row.addStretch(1)
        tool_layout.addLayout(conversion_row)
        tool_layout.setAlignment(conversion_row, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.checkBox_show_gaussian.setObjectName("InlineCheck")

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
        self.horizontalLayout_17.addWidget(tool_panel, stretch=6)

    def set_x_axis_mode(self):
        combo = getattr(self, "xAxisModeCombo", None)
        self.x_axis_mode = str(combo.currentData()) if combo is not None else "tof"
        self.refresh_plot_axis_mode()

    def refresh_plot_axis_mode(self) -> None:
        axis_label = "m/z" if self.x_axis_mode == "mz" else "TOF"
        calibration = self.current_calibration()
        for axis in (self.p1_axis, self.p2_axis):
            if axis is not None:
                axis.set_display_mode(self.x_axis_mode, calibration)
        for plot in (self.p1, self.p2):
            if plot is not None:
                plot.setLabel("bottom", axis_label)

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
        splitter.setCollapsible(1, False)
        splitter.setSizes([1120, 360])
        self.horizontalLayout_13.addWidget(splitter)
        self.main_splitter = splitter

    def _rebuild_side_panel(self):
        self.widget_3.setMinimumWidth(340)
        self.widget_3.setMaximumWidth(460)
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

    def choose_sum_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "选择累计谱文件夹",
            self._dialog_start_dir(self.folder_path.text()),
        )
        if folder_path:
            self.folder_path.setText(folder_path)

    def current_calibration(self):
        return Calibration(
            a=float(self.lineEdit_4.text()),
            b=float(self.lineEdit_5.text()),
            c=float(self.lineEdit_6.text()),
        )

    def launch_core_tool(self, initial_tab: str = "normalization"):
        """Open one core tool tab directly from the main toolbar."""
        try:
            dialog = CoreToolsDialog(self.current_calibration(), self, initial_tab=initial_tab)
            dialog.exec()
            self.apply_config_defaults()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动核心处理工具失败：{str(e)}")

    def load_image(self, image_path):
        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            print(f"无法加载图片: {image_path}")
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
        self.current_plot_y = np.array([], dtype=float)
        self.current_plot_title = "质谱图"
        self.plot_x_min = None
        self.plot_x_max = None

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
        self.current_plot_y = y_values
        self.current_plot_title = title
        self.plot_x_min = float(np.nanmin(x_values))
        self.plot_x_max = float(np.nanmax(x_values))
        x_range = max(1.0, self.plot_x_max - self.plot_x_min)
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        y_padding = max(1.0, (y_max - y_min) * 0.08)

        win = GraphicsLayoutWidget()
        win.setBackground("#ffffff")
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
        self.spectrum_plot = self.p2.plot(x_values, y_values, pen=mkPen('#2563eb', width=1.2), name='质谱图')
        self._optimize_curve(self.spectrum_plot)

        # 添加 LinearRegionItem 到 p2，设置移动模式和边界约束
        peak_idx = int(np.nanargmax(y_values))
        peak_center = float(x_values[peak_idx])
        half_width = max(4.0, x_range * 0.001)
        region_left = max(self.plot_x_min, peak_center - half_width)
        region_right = min(self.plot_x_max, peak_center + half_width)
        self.selection_region = LinearRegionItem(
            values=[region_left, region_right],
            bounds=(self.plot_x_min, self.plot_x_max),
            movable=True,  # 允许整体移动
            brush=pg.mkBrush(color=(128, 128, 128, 50))  # 半透明灰色
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

    def _apply_plot_limits(self, plot, y_min: float, y_max: float, x_range: float) -> None:
        if self.plot_x_min is None or self.plot_x_max is None:
            return
        plot.setLimits(
            xMin=self.plot_x_min,
            xMax=self.plot_x_max,
            maxXRange=x_range,
            minXRange=max(1.0, x_range / 100_000),
            yMin=y_min,
            yMax=y_max,
        )
        plot.setXRange(self.plot_x_min, self.plot_x_max, padding=0)
        plot.setYRange(y_min, y_max, padding=0)

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
            y_left, y_right = float(view_range[1][0]), float(view_range[1][1])
            if np.isfinite(y_left) and np.isfinite(y_right) and y_right > y_left:
                self.p2.setYRange(y_left, y_right, padding=0)
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
        center = (left + right) / 2
        self.label_8.setText(f"{center:.2f}")
        self.label_9.setText(f"{float(self.current_calibration().tof_to_mz(center)):.2f}")
        self.label_10.setText(f"{left:.2f}")
        self.label_11.setText(f"{right:.2f}")

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

        full_range = self._plot_x_range()
        fallback_half_width = max(2.0, full_range * 0.0025)
        if left is None or right is None or not np.isfinite(left) or not np.isfinite(right) or left == right:
            left = peak_time - fallback_half_width
            right = peak_time + fallback_half_width

        left, right = self._clamp_region_values(left, right)
        self.selection_region.setRegion([left, right])
        self.update_region_readout()
        self._center_plot_on_peak(peak_time, peak_intensity, left, right)
        if self.p1 is not None and self.checkBox_show_gaussian.isChecked():
            self.update_gaussian_fit()

    def _plot_x_range(self) -> float:
        if self.plot_x_min is None or self.plot_x_max is None:
            return 1.0
        return max(1.0, self.plot_x_max - self.plot_x_min)

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
        selected_width = max(1.0, abs(right - left))
        view_width = min(full_range, max(selected_width * 6, full_range * 0.02, 10.0))
        center = max(self.plot_x_min, min(float(peak_time), self.plot_x_max))
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

        if self.current_plot_x.size == 0 or self.current_plot_y.size == 0:
            return
        start = int(np.searchsorted(self.current_plot_x, x_left, side="left"))
        end = int(np.searchsorted(self.current_plot_x, x_right, side="right"))
        start = max(0, min(start, self.current_plot_y.size - 1))
        end = max(start + 1, min(end, self.current_plot_y.size))
        segment_y = self.current_plot_y[start:end]
        segment_y = segment_y[np.isfinite(segment_y)]
        if segment_y.size == 0:
            return

        y_min = float(np.nanmin(segment_y))
        y_max = float(np.nanmax(segment_y))
        if peak_intensity is not None and np.isfinite(peak_intensity):
            y_max = max(y_max, float(peak_intensity))
        y_span = max(1.0, y_max - y_min)
        self.p2.setYRange(y_min - y_span * 0.08, y_max + y_span * 0.20, padding=0)

    def on_update_timeout(self):
        """实际执行更新高斯拟合图"""
        self.update_timer.stop()
        self.pending_update = False

        if self.selection_region and self.p1 and self.p2 and self.current_plot_x.size:
            try:
                minX, maxX = self._clamp_region_values(*self.selection_region.getRegion())
                x_start = int(np.searchsorted(self.current_plot_x, minX, side="left"))
                x_end = int(np.searchsorted(self.current_plot_x, maxX, side="right"))
                x_start = max(0, min(x_start, self.current_plot_x.size))
                x_end = max(x_start, min(x_end, self.current_plot_x.size))
                x_data = self.current_plot_x[x_start:x_end]
                y_data = self.current_plot_y[x_start:x_end]

                if len(x_data) != len(y_data):
                    print("警告：X 和 Y 数据长度不匹配")
                    return

                if len(x_data) < 3 or len(y_data) < 3:
                    print("警告：选择区域内的数据点不足，无法进行拟合。")
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
                self.label_9.setText(str(round(float(self.current_calibration().tof_to_mz(popt[0])), 2)))
                self.label_10.setText(str(round(minX, 2)))
                self.label_11.setText(str(round(maxX, 2)))

                self.p1.clear()
                x_interp = np.linspace(x_data[0], x_data[-1], 1000)
                y_fit = self.func_gaosi(x_interp, *popt) + baseline  # 添加回基线
                self.p1.plot(x_interp, y_fit, pen=mkPen('r', width=2), name='拟合图')

            except Exception as e:
                print(f"高斯拟合错误：{e}")

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
        df = self.read_table_data_1()
        poly = PolynomialFeatures(degree=2)  # 二次多项式
        x = poly.fit_transform(df[['time']])
        y = df['mass'].astype(float)  # 将y转换为浮点类型
        # print(y)
        model = LinearRegression()
        model.fit(x, y)

        # 获取系数
        coefficients = model.coef_
        intercept = model.intercept_
        y_pred = model.predict(x)
        # print(y_pred)
        # 计算残差
        # residuals = y - y_pred
        # print(residuals)
        # 计算R平方（COD）
        r2 = model.score(x, y)
        # print(r2)
        #
        # 提取回归系数
        a, b, c = coefficients[1], coefficients[2], intercept
        # print(a, b, c)
        # # 格式化输出结果
        # print("y =", f"{c}" + " + " f"{a}" "x" + " + " f"{b}" "x²")
        # print(f"R²(COD) = {r2}")

        # 在标签上设置文本内容
        result_text = f"y = {c} + {a}x + {b}x²\n, R²(COD) = {r2}"
        label = self.findChild(QLabel, "label_4")
        label.setText(result_text)
        # print("Column 1 data:", column1_data)
        # print("Column 2 data:", column2_data)

    def transfer(self):
        time = self.findChild(QLineEdit, "lineEdit_2")
        t = float(time.text())
        m = self.current_calibration().tof_to_mz(t)
        lineEdit = self.findChild(QLineEdit, "lineEdit_3")  # 找到名为lineEdit_7的QLineEdit对象
        lineEdit.setText(str(m))  # 将值设置为QLineEdit的文本内容

    def transfer_2(self):
        time = self.findChild(QLabel, "label_8")
        return self.current_calibration().tof_to_mz(float(time.text()))

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
            minX, maxX = self.selection_region.getRegion()
            left_idx = int(max(0, round(minX - self.current_time_offset)))
            right_idx = int(min(len(self.spectrum_plot.yData) - 1, round(maxX - self.current_time_offset)))
            peak = core_add_manual_peak(
                self.spectrum_plot.yData,
                left_idx,
                right_idx,
                calibration=self.current_calibration(),
                time_offset=self.current_time_offset,
            )
            if peak is None:
                raise ValueError("选择区域无效")
            
            # 在表格中显示
            row = self.peakData.rowCount()
            self.peakData.insertRow(row)
            
            # 按列顺序设置数据
            self.peakData.setItem(row, 0, QTableWidgetItem("Unknown"))  # Species 列
            self.peakData.setItem(row, 1, QTableWidgetItem(f"{peak.time:.2f}"))  # 飞行时间列
            self.peakData.setItem(row, 2, QTableWidgetItem(f"{peak.mz:.2f}"))    # 质量数列
            self.peakData.setItem(row, 3, QTableWidgetItem(f"{peak.intensity:.2f}"))  # 强度列
            self.peakData.setItem(row, 4, QTableWidgetItem(f"{peak.left_bound + self.current_time_offset:.2f}"))  # 左边界列
            self.peakData.setItem(row, 5, QTableWidgetItem(f"{peak.right_bound + self.current_time_offset:.2f}"))  # 右边界列
            
            # 添加标注
            text_item = pg.TextItem(text=f'{peak.mz:.2f}', color='red', anchor=(0.5, 1.5))
            text_item.setPos(peak.time, peak.intensity)
            self.p2.addItem(text_item)
            
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'添加峰值失败: {str(e)}')

    def auto_find_peaks(self):
        """自动寻峰功能"""
        try:
            # 获取当前图表数据
            y_data = self.spectrum_plot.yData
            peak_config = load_peak_detection_config()
            peaks = self.detect_peaks_for_config(y_data, peak_config)

            # 在表格中显示检测到的峰值
            self.display_peaks(peaks)

            # 在图表上标注检测到的峰值
            self.annotate_peaks_on_plot(peaks)

        except Exception as e:
            QMessageBox.warning(self, 'Error', f'自动寻峰失败: {str(e)}')

    def detect_peaks_for_config(self, y_data, peak_config):
        algorithm = getattr(peak_config, "algorithm", "legacy")
        common = {
            "calibration": self.current_calibration(),
            "start_idx": 0,
            "end_idx": len(y_data),
            "detection_min_idx": peak_config.detection_min_idx,
            "time_offset": self.current_time_offset,
            "threshold_end": peak_config.threshold_end,
            "min_intensity": peak_config.min_intensity,
            "duplicate_window": peak_config.duplicate_window,
            "gaussian_window_max": peak_config.gaussian_window_max,
            "gaussian_boundary_scale": peak_config.gaussian_boundary_scale,
            "boundary_padding": peak_config.boundary_padding,
        }
        if algorithm == "legacy":
            return detect_peaks_in_range(
                y_data,
                nearby_peak_window=peak_config.nearby_peak_window,
                weak_tail_early_window=peak_config.weak_tail_early_window,
                weak_tail_late_window=peak_config.weak_tail_late_window,
                weak_tail_ratio=peak_config.weak_tail_ratio,
                **common,
            )
        if algorithm == "cwt":
            try:
                from core.cwt_peak_detection import CwtPeakDetectionConfig, detect_peaks_cwt
            except ImportError as exc:
                raise RuntimeError("CWT寻峰需要安装 PyWavelets") from exc
            detection_start = max(0, int(peak_config.detection_min_idx))
            cwt_config = CwtPeakDetectionConfig(
                window_size=peak_config.smoothing_window,
                poly_order=peak_config.smoothing_poly_order,
                prominence_ratio=peak_config.prominence_ratio,
                min_peak_distance=max(1, peak_config.duplicate_window),
                min_peak_width=peak_config.min_peak_width,
                max_peak_width=peak_config.max_peak_width,
                baseline_percentile=peak_config.baseline_percentile,
                baseline_window_factor=max(
                    1,
                    int(peak_config.baseline_window / max(1, peak_config.smoothing_window)),
                ),
            )
            return detect_peaks_cwt(
                y_data[detection_start:],
                calibration=self.current_calibration(),
                start_idx=detection_start,
                time_offset=self.current_time_offset,
                config=cwt_config,
            )
        return detect_peaks_prominence(
            y_data,
            prominence_ratio=peak_config.prominence_ratio,
            smoothing_window=peak_config.smoothing_window,
            smoothing_poly_order=peak_config.smoothing_poly_order,
            baseline_window=peak_config.baseline_window,
            baseline_percentile=peak_config.baseline_percentile,
            min_peak_width=peak_config.min_peak_width,
            max_peak_width=peak_config.max_peak_width,
            **common,
        )

    def display_peaks(self, peaks):
        """在表格中显示检测到的峰值"""
        table = self.findChild(QTableWidget, "peakData")
        self._updating_peak_table = True
        try:
            table.setRowCount(len(peaks))

            # 设置表格的列标题
            headers = ["Species", "飞行时间", "质量数 (m/z)", "强度", "左边界", "右边界"]
            table.setHorizontalHeaderLabels(headers)

            for i, peak in enumerate(peaks):
                # 按列顺序设置数据
                table.setItem(i, 0, QTableWidgetItem("Unknown"))  # Species 列
                table.setItem(i, 1, QTableWidgetItem(f"{peak.time:.2f}"))  # 飞行时间列
                table.setItem(i, 2, QTableWidgetItem(f"{peak.mz:.2f}"))    # 质量数列
                table.setItem(i, 3, QTableWidgetItem(f"{peak.intensity:.2f}"))  # 强度列
                table.setItem(i, 4, QTableWidgetItem(f"{peak.left_bound + self.current_time_offset:.2f}"))  # 左边界列
                table.setItem(i, 5, QTableWidgetItem(f"{peak.right_bound + self.current_time_offset:.2f}"))  # 右边界列
        finally:
            self._updating_peak_table = False

    @staticmethod
    def mz_to_time(mz: float, a: float, b: float, c: float) -> float:
        """将质量数转换为飞行时间"""
        # 这里假设 a, b, c 是已知的时间系数
        # 需要求解二次方程 a + b * time + c * time^2 = mz
        # 这可以通过求解二次方程来实现
        discriminant = b**2 - 4*c*(a - mz)
        if discriminant < 0:
            raise ValueError("无解")
        time1 = (-b + np.sqrt(discriminant)) / (2*c)
        time2 = (-b - np.sqrt(discriminant)) / (2*c)
        # 返回正值的解
        return time1 if time1 > 0 else time2

    def annotate_peaks_on_plot(self, peaks):
        """在质谱图上标注检测到的峰值"""
        if self.p2 is None:
            return
        for peak in peaks:
            try:
                text_item = pg.TextItem(text=f'{peak.mz:.2f}', color='red', anchor=(0.5, 1.5))
                text_item.setPos(peak.time, peak.intensity)  # 使用飞行时间作为x坐标
                self.p2.addItem(text_item)
            except Exception as e:
                print(f"无法标注峰值 {peak.mz}: {e}")

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

    def calculate_abc(self):
        """计算ABC参数"""
        df = self.read_table_data_1()
        poly = PolynomialFeatures(degree=2)  # 二次多项式
        x = poly.fit_transform(df[['time']])
        y = df['mass'].astype(float)  # 将y转换为浮点类型

        model = LinearRegression()
        model.fit(x, y)

        # 获取系数
        coefficients = model.coef_
        intercept = model.intercept_
        r2 = model.score(x, y)

        # 提取回归系数
        a, b, c = coefficients[1], coefficients[2], intercept

        # 在标签上设置文本内容
        result_text = f"y = {c} + {a}x + {b}x²\n, R²(COD) = {r2}"
        label = self.findChild(QLabel, "label_4")
        label.setText(result_text)

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
            self.annotate_peak(peak_data)

    def delete_peak(self, row):
        """删除峰值"""
        reply = QMessageBox.question(self, '确认删除', 
                                   '确定要删除这个峰值吗？',
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                   QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.peakData.removeRow(row)
            self.update_plot()  # 更新图表

    def edit_peak(self, row):
        """编辑峰值"""
        current_data = []
        for col in range(self.peakData.columnCount()):
            item = self.peakData.item(row, col)
            current_data.append(item.text() if item else "")
            
        dialog = PeakDialog(self, current_data)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            peak_data = dialog.get_peak_data()
            self.update_peak_in_table(row, peak_data)
            self.update_plot()  # 更新图表

    def add_peak_to_table(self, peak_data):
        """将峰值添加到表格"""
        self._updating_peak_table = True
        row = self.peakData.rowCount()
        try:
            self.peakData.insertRow(row)
            self.peakData.setItem(row, 0, QTableWidgetItem(peak_data['species']))
            self.peakData.setItem(row, 1, QTableWidgetItem(f"{peak_data['time']:.2f}"))
            self.peakData.setItem(row, 2, QTableWidgetItem(f"{peak_data['mz']:.2f}"))
            self.peakData.setItem(row, 3, QTableWidgetItem(f"{peak_data['intensity']:.2f}"))
            self.peakData.setItem(row, 4, QTableWidgetItem(f"{peak_data['left']:.2f}"))
            self.peakData.setItem(row, 5, QTableWidgetItem(f"{peak_data['right']:.2f}"))
        finally:
            self._updating_peak_table = False

    def update_peak_in_table(self, row, peak_data):
        """更新表格中的峰值"""
        self._updating_peak_table = True
        try:
            self.peakData.setItem(row, 0, QTableWidgetItem(peak_data['species']))
            self.peakData.setItem(row, 1, QTableWidgetItem(f"{peak_data['time']:.2f}"))
            self.peakData.setItem(row, 2, QTableWidgetItem(f"{peak_data['mz']:.2f}"))
            self.peakData.setItem(row, 3, QTableWidgetItem(f"{peak_data['intensity']:.2f}"))
            self.peakData.setItem(row, 4, QTableWidgetItem(f"{peak_data['left']:.2f}"))
            self.peakData.setItem(row, 5, QTableWidgetItem(f"{peak_data['right']:.2f}"))
        finally:
            self._updating_peak_table = False

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
            text_item.setPos(time, intensity)
            self.p2.addItem(text_item)

    def annotate_peak(self, peak_data):
        """在图表上标注单个峰值"""
        if self.p2 is None:
            return
        try:
            text_item = pg.TextItem(text=f"{peak_data['mz']:.2f}", color='red', anchor=(0.5, 1.5))
            text_item.setPos(peak_data['time'], peak_data['intensity'])
            self.p2.addItem(text_item)
        except Exception as e:
            print(f"无法标注峰值: {e}")

    def update_all_peaks(self):
        """更新所有峰值标注"""
        try:
            if self.p2 is None:
                return
            # 清除现有标注
            for item in self.p2.items[:]:
                if isinstance(item, pg.TextItem):
                    self.p2.removeItem(item)
            
            # 从表格中读取所有峰值数据并重新标注
            for row in range(self.peakData.rowCount()):
                values = self._peak_row_values(row)
                if values is None:
                    continue
                time, mz, intensity = values
                text_item = pg.TextItem(text=f'{mz:.2f}', color='red', anchor=(0.5, 1.5))
                text_item.setPos(time, intensity)
                self.p2.addItem(text_item)
                
        except Exception as e:
            QMessageBox.warning(self, 'Error', f'更新峰值标注失败: {str(e)}')

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
            
            QMessageBox.information(self, "成功", "峰值数据已清除")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"清除数据时发生错误：{str(e)}")
