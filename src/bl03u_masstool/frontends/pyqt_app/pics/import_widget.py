"""PICS Import Widget - Standalone tab for importing PICS data from external files."""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtWidgets

from bl03u_masstool.core.config import species_database_path


class PICSImportWidget(QtWidgets.QWidget):
    """Standalone widget for importing external PICS data into the species database.

    Supports CSV, XLSX, and other tabular formats. Provides file selection,
    preview, and upsert mode writing to the local species database.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        """Initialize the import interface."""
        layout = QtWidgets.QVBoxLayout(self)

        # Title and description
        title = QtWidgets.QLabel("外部 PICS 数据导入")
        title.setObjectName("ProjectTitle")
        layout.addWidget(title)

        hint = QtWidgets.QLabel(
            "从 CSV / XLSX 文件导入物种的光电离截面数据到本地数据库。\n"
            "支持宽表（每行一物种，能量值作列名）和长表（每行一能量点）格式。"
        )
        hint.setObjectName("ProjectHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        # Import group
        import_group = QtWidgets.QGroupBox("导入外部 PICS 数据")
        import_layout = QtWidgets.QVBoxLayout(import_group)

        import_note = QtWidgets.QLabel(
            "选择包含光电离截面数据的文件，程序会解析文件内容并提供预览确认。\n"
            "导入时使用 upsert 模式：相同物种和 m/z 的记录将被更新，其他记录保留。"
        )
        import_note.setWordWrap(True)
        import_note.setStyleSheet("color: #6495ed;")
        import_layout.addWidget(import_note)

        btn_import = QtWidgets.QPushButton("选择文件导入 PICS 到数据库…")
        btn_import.setObjectName("WorkflowButton")
        btn_import.setToolTip("打开文件选择对话框，选择 XLSX/CSV/TSV/TXT 文件")
        btn_import.clicked.connect(self._import_pics_from_file)
        import_layout.addWidget(btn_import)

        layout.addWidget(import_group)

        # Status area
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addStretch()

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

            content = Path(file_path).read_bytes()
            records = parse_pics_upload(content, Path(file_path).name)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "解析失败", f"无法解析文件：\n{e}")
            self.status_label.setText("❌ 文件解析失败")
            self.status_label.setStyleSheet("color: #ff6b6b;")
            return

        # Show preview dialog
        lines = [f"解析到 {len(records)} 个物种：\n"]
        for r in records[:20]:
            ie_str = f"IE={r['ie']} eV" if r['ie'] else "IE 未知"
            lines.append(f"  m/z={r['mz']}  {r['species']}  {ie_str}  ({len(r['energies'])} 点)")
        if len(records) > 20:
            lines.append(f"  … 还有 {len(records) - 20} 个物种")
        lines.append("\n是否以 upsert 模式写入本地数据库？\n（同名同 m/z 物种将被替换）")

        reply = QtWidgets.QMessageBox.question(
            self,
            "确认导入",
            "\n".join(lines),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            self.status_label.setText("⊘ 导入已取消")
            self.status_label.setStyleSheet("color: #6495ed;")
            return

        # Write to database
        try:
            db_path = species_database_path()
            result = write_pics_records(records, db_path, mode="upsert")
            msg = (
                f"✓ 导入完成\n"
                f"写入物种: {result['inserted_species']}\n"
                f"替换旧记录: {result['replaced_species']}\n"
                f"数据点: {result['inserted_points']}"
            )
            QtWidgets.QMessageBox.information(self, "导入完成", msg)
            self.status_label.setText(
                f"✓ 最后导入: {len(records)} 个物种 | "
                f"新增: {result['inserted_species']} | "
                f"更新: {result['replaced_species']}"
            )
            self.status_label.setStyleSheet("color: #4ecdc4;")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "写入失败", f"写入数据库失败：\n{e}")
            self.status_label.setText("❌ 数据库写入失败")
            self.status_label.setStyleSheet("color: #ff6b6b;")
