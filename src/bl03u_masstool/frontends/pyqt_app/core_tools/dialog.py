from __future__ import annotations


from PyQt6 import QtWidgets

from bl03u_masstool.core.calibration import Calibration
from bl03u_masstool.core.normalization import load_normalization_settings

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - only used when optional plotting is unavailable
    pg = None

from bl03u_masstool.frontends.pyqt_app.isotope.dialog import IsotopeAbundanceDialog
from bl03u_masstool.frontends.pyqt_app.mole_fraction.dialog import MoleFractionDialog
from bl03u_masstool.frontends.pyqt_app.nist.widget import IonizationEnergyLookupWidget
from bl03u_masstool.frontends.pyqt_app.normalization.widget import CommonParametersWidget
from bl03u_masstool.frontends.pyqt_app.pics.dialog import PICSCalculatorDialog
from bl03u_masstool.frontends.pyqt_app.pics.import_widget import PICSImportWidget
from bl03u_masstool.frontends.pyqt_app.pie.dialog import PIESpeciesFitDialog
from bl03u_masstool.frontends.pyqt_app.temperature.dialog import TemperatureScanDialog

class CoreToolsDialog(QtWidgets.QDialog):
    def __init__(self, calibration: Calibration, parent=None, initial_tab: str = "normalization"):
        super().__init__(parent)
        self.setWindowTitle("BL03U核心处理工具")
        self.resize(1280, 800)
        self.normalization_settings = load_normalization_settings()
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        tab_indexes = {
            "normalization": self.tabs.addTab(CommonParametersWidget(self.normalization_settings, calibration, self), "通用参数"),
            "temperature": self.tabs.addTab(TemperatureScanDialog(calibration, self.normalization_settings, self), "温度扫描"),
            "pie": self.tabs.addTab(PIESpeciesFitDialog(calibration, self.normalization_settings, self), "PIE物种拟合"),
            "mole_fraction": self.tabs.addTab(MoleFractionDialog(calibration, self.normalization_settings, self), "摩尔分数"),
            "pics": self.tabs.addTab(PICSCalculatorDialog(calibration, self.normalization_settings, self), "PICS计算"),
            "pics_import": self.tabs.addTab(PICSImportWidget(self), "PICS导入"),
            "ionization": self.tabs.addTab(IonizationEnergyLookupWidget(self), "电离能查询"),
            "isotope": self.tabs.addTab(IsotopeAbundanceDialog(self), "分子式与质量分析"),
        }
        self.tabs.setCurrentIndex(tab_indexes.get(initial_tab, 0))
        layout.addWidget(self.tabs)
