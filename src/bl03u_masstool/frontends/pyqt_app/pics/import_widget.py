"""PICS Import Widget - Standalone tab for importing PICS data from external files."""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from bl03u_masstool.core.config import species_database_path


class PICSImportWidget(QtWidgets.QWidget):
    """Standalone widget for importing external PICS data into the species database.

    Supports CSV, XLSX, and other tabular formats. Provides file selection,
    preview, and upsert mode writing to the local species database.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PICSImportPage")
        self._init_ui()

    def _init_ui(self):
        """Initialize the import interface."""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # ========== Header Section ==========
        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)

        title = QtWidgets.QLabel("🗂️ PICS 数据库导入工具")
        title.setObjectName("ProjectTitle")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        header_layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "导入外部光电离截面 (PICS) 数据到本地PICS截面数据库，支持 CSV、XLSX 等格式"
        )
        subtitle.setObjectName("ProjectHint")
        subtitle.setWordWrap(True)
        header_layout.addWidget(subtitle)

        main_layout.addWidget(header)

        # ========== Divider ==========
        divider1 = QtWidgets.QFrame()
        divider1.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        divider1.setObjectName("NavSeparator")
        main_layout.addWidget(divider1)

        # ========== File Selection Section ==========
        file_group = QtWidgets.QGroupBox("1️⃣ 选择数据文件")
        file_group.setObjectName("ImportSection")
        file_layout = QtWidgets.QVBoxLayout(file_group)

        file_info = QtWidgets.QLabel(
            "支持格式：\n"
            "  • 宽表：每行一物种，能量值作为列名（如 10.5 eV、11.0 eV）\n"
            "  • 长表：每行一个能量点，包含 mz、name、energy_ev、cross_section 列"
        )
        file_info.setWordWrap(True)
        file_info.setStyleSheet("color: #6495ed; background-color: #f0f7ff; padding: 8px; border-radius: 4px;")
        file_layout.addWidget(file_info)

        btn_import = QtWidgets.QPushButton("📂 选择 XLSX / CSV / TSV 文件…")
        btn_import.setObjectName("WorkflowButton")
        btn_import.setFixedHeight(36)
        btn_import.setToolTip("打开文件选择对话框")
        btn_import.clicked.connect(self._import_pics_from_file)
        file_layout.addWidget(btn_import)

        main_layout.addWidget(file_group)

        # ========== Preview Section ==========
        preview_group = QtWidgets.QGroupBox("2️⃣ 导入预览与确认")
        preview_group.setObjectName("ImportSection")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)

        preview_note = QtWidgets.QLabel(
            "导入时使用 upsert 模式：\n"
            "  • 相同物种 + 相同 m/z 的记录将被更新\n"
            "  • 其他现有记录保留不变"
        )
        preview_note.setWordWrap(True)
        preview_note.setStyleSheet("color: #6495ed; background-color: #f0f7ff; padding: 8px; border-radius: 4px;")
        preview_layout.addWidget(preview_note)

        self.preview_table = QtWidgets.QTableWidget()
        self.preview_table.setColumnCount(3)
        self.preview_table.setHorizontalHeaderLabels(["物种", "m/z", "能量点数"])
        self.preview_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.preview_table.setMaximumHeight(150)
        self.preview_table.setVisible(False)
        preview_layout.addWidget(self.preview_table)

        self.preview_status = QtWidgets.QLabel("等待文件选择…")
        self.preview_status.setWordWrap(True)
        preview_layout.addWidget(self.preview_status)

        main_layout.addWidget(preview_group)

        # ========== Import Result Section ==========
        result_group = QtWidgets.QGroupBox("3️⃣ 导入结果")
        result_group.setObjectName("ImportSection")
        result_layout = QtWidgets.QVBoxLayout(result_group)

        self.result_label = QtWidgets.QLabel("暂无导入记录")
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        result_layout.addWidget(self.result_label)

        main_layout.addWidget(result_group)

        # ========== Stretch ==========
        main_layout.addStretch()

    def _import_pics_from_file(self):
        """Handle PICS file import workflow."""
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择 PICS 数据文件",
            "",
            "数据文件 (*.xlsx *.xls *.csv *.tsv *.txt);;所有文件 (*)",
        )
        if not file_path:
            return

        # Parse file
        try:
            from bl03u_masstool.core.pics_import import parse_pics_upload, write_pics_records

            self.preview_status.setText(f"解析中…")
            self.preview_status.setStyleSheet("color: #6495ed;")

            content = Path(file_path).read_bytes()
            records = parse_pics_upload(content, Path(file_path).name)

            # Show preview table
            self.preview_table.setRowCount(0)
            for r in records[:20]:
                row = self.preview_table.rowCount()
                self.preview_table.insertRow(row)
                self.preview_table.setItem(row, 0, QtWidgets.QTableWidgetItem(r["species"]))
                self.preview_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(r["mz"])))
                self.preview_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(len(r["energies"]))))

            self.preview_table.setVisible(True)

            if len(records) > 20:
                preview_text = f"解析到 {len(records)} 个物种（显示前20个），还有 {len(records) - 20} 个…"
            else:
                preview_text = f"解析到 {len(records)} 个物种"

            self.preview_status.setText(preview_text)
            self.preview_status.setStyleSheet("color: #4ecdc4;")

        except Exception as e:
            self.preview_status.setText(f"❌ 文件解析失败：{str(e)}")
            self.preview_status.setStyleSheet("color: #ff6b6b;")
            self.preview_table.setVisible(False)
            QtWidgets.QMessageBox.warning(self, "解析失败", f"无法解析文件：\n{e}")
            return

        # Show confirmation dialog
        lines = [f"即将导入 {len(records)} 个物种\n"]
        for r in records[:15]:
            ie_str = f"IE={r['ie']} eV" if r['ie'] else "IE 未知"
            lines.append(f"  m/z={r['mz']}  {r['species']}  {ie_str}  ({len(r['energies'])} 点)")
        if len(records) > 15:
            lines.append(f"  … 还有 {len(records) - 15} 个物种")
        lines.append("\n⚠️ 相同物种和 m/z 的现有记录将被替换")
        lines.append("是否继续导入？")

        reply = QtWidgets.QMessageBox.question(
            self,
            "确认导入 PICS 数据",
            "\n".join(lines),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            self.preview_status.setText("导入已取消")
            self.preview_status.setStyleSheet("color: #6495ed;")
            return

        # Write to database
        try:
            db_path = species_database_path()
            result = write_pics_records(records, db_path, mode="upsert")

            result_text = (
                f"✅ 导入成功\n\n"
                f"新增物种: {result['inserted_species']}\n"
                f"更新物种: {result['replaced_species']}\n"
                f"总数据点: {result['inserted_points']}"
            )
            self.result_label.setText(result_text)
            self.result_label.setStyleSheet("color: #4ecdc4; background-color: #f0fff7; padding: 12px; border-radius: 4px;")

            QtWidgets.QMessageBox.information(
                self,
                "✅ 导入完成",
                result_text,
            )
        except Exception as e:
            result_text = f"❌ 导入失败：{str(e)}"
            self.result_label.setText(result_text)
            self.result_label.setStyleSheet("color: #ff6b6b; background-color: #fff0f5; padding: 12px; border-radius: 4px;")
            QtWidgets.QMessageBox.warning(self, "导入失败", f"写入数据库失败：\n{e}")

