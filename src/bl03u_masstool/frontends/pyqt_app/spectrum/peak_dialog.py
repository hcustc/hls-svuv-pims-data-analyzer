from __future__ import annotations

from PyQt6 import QtCore
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QVBoxLayout


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
