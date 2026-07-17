"""Standalone workflow for previewing and importing external PICS tables."""

from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from bl03u_masstool.core.config import resolve_species_database_path
from bl03u_masstool.core.project_settings import ProjectSettings
from bl03u_masstool.frontends.pyqt_app.workers import WorkerThread


class PICSImportWidget(QtWidgets.QWidget):
    """Parse, preview, and explicitly confirm PICS database imports."""

    import_completed = QtCore.pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("PICSImportPage")
        self._records: list[dict] = []
        self._selected_path = ""
        self._worker: WorkerThread | None = None
        self.project_settings: ProjectSettings | None = None
        self._init_ui()

    def set_project_settings(self, project_settings: ProjectSettings) -> None:
        self.project_settings = project_settings

    def database_path(self) -> Path:
        configured_path = (
            self.project_settings.pics_database_path
            if self.project_settings is not None
            else None
        )
        return resolve_species_database_path(configured_path)

    @staticmethod
    def _set_status(label: QtWidgets.QLabel, text: str, status: str = "") -> None:
        label.setText(text)
        label.setProperty("status", status)
        label.style().unpolish(label)
        label.style().polish(label)

    def _init_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QtWidgets.QWidget(self)
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        title_column = QtWidgets.QVBoxLayout()
        title = QtWidgets.QLabel("导入外部 PICS 数据", header)
        title.setObjectName("ProjectTitle")
        subtitle = QtWidgets.QLabel("选择表格并检查解析结果，确认后写入本地 PICS 数据库。", header)
        subtitle.setObjectName("ProjectHint")
        title_column.addWidget(title)
        title_column.addWidget(subtitle)
        header_layout.addLayout(title_column, 1)
        self.open_calculator_button = QtWidgets.QPushButton("前往 PICS 计算", header)
        self.open_calculator_button.setObjectName("BrowseButton")
        self.open_calculator_button.clicked.connect(self._open_calculator)
        header_layout.addWidget(self.open_calculator_button)
        root.addWidget(header)

        file_group = QtWidgets.QGroupBox("1. 选择并解析文件", self)
        file_layout = QtWidgets.QGridLayout(file_group)
        file_layout.setHorizontalSpacing(8)
        file_layout.setVerticalSpacing(6)
        self.file_path_edit = QtWidgets.QLineEdit(file_group)
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("选择 CSV、TSV、TXT、XLSX 或 XLS 文件")
        self.choose_file_button = QtWidgets.QPushButton("选择文件", file_group)
        self.choose_file_button.setObjectName("PrimaryButton")
        self.choose_file_button.clicked.connect(self._import_pics_from_file)
        file_layout.addWidget(QtWidgets.QLabel("数据文件"), 0, 0)
        file_layout.addWidget(self.file_path_edit, 0, 1)
        file_layout.addWidget(self.choose_file_button, 0, 2)
        format_hint = QtWidgets.QLabel(
            "支持长表（mz、name、energy_ev、cross_section）和宽表（能量作为列名）；每个物种至少需要 2 个能量点。",
            file_group,
        )
        format_hint.setObjectName("HintLabel")
        format_hint.setWordWrap(True)
        file_layout.addWidget(format_hint, 1, 1, 1, 2)
        file_layout.setColumnStretch(1, 1)
        root.addWidget(file_group)

        preview_group = QtWidgets.QGroupBox("2. 检查导入内容", self)
        preview_layout = QtWidgets.QVBoxLayout(preview_group)
        preview_toolbar = QtWidgets.QHBoxLayout()
        preview_toolbar.addWidget(QtWidgets.QLabel("写入方式"))
        self.import_mode_combo = QtWidgets.QComboBox(preview_group)
        self.import_mode_combo.addItem("更新相同物种，保留其他记录（推荐）", "upsert")
        self.import_mode_combo.addItem("全部追加为新记录", "append")
        self.import_mode_combo.currentIndexChanged.connect(self._refresh_mode_hint)
        preview_toolbar.addWidget(self.import_mode_combo)
        self.mode_hint_label = QtWidgets.QLabel("", preview_group)
        self.mode_hint_label.setObjectName("HintLabel")
        preview_toolbar.addWidget(self.mode_hint_label, 1)
        self.confirm_import_button = QtWidgets.QPushButton("确认导入数据库", preview_group)
        self.confirm_import_button.setObjectName("PrimaryButton")
        self.confirm_import_button.setEnabled(False)
        self.confirm_import_button.clicked.connect(self._write_preview_records)
        preview_toolbar.addWidget(self.confirm_import_button)
        preview_layout.addLayout(preview_toolbar)

        self.preview_status = QtWidgets.QLabel("尚未选择文件", preview_group)
        self.preview_status.setObjectName("InlineStatusLabel")
        preview_layout.addWidget(self.preview_status)
        self.preview_table = QtWidgets.QTableWidget(0, 5, preview_group)
        self.preview_table.setHorizontalHeaderLabels(["物种", "m/z", "电离能 (eV)", "能量范围 (eV)", "数据点"])
        self.preview_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.preview_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.preview_table.setAlternatingRowColors(True)
        header_view = self.preview_table.horizontalHeader()
        header_view.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header_view.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        preview_layout.addWidget(self.preview_table, 1)
        root.addWidget(preview_group, 1)

        result_bar = QtWidgets.QWidget(self)
        result_bar.setObjectName("ProjectActionBar")
        result_layout = QtWidgets.QHBoxLayout(result_bar)
        result_layout.setContentsMargins(10, 7, 10, 7)
        result_layout.addWidget(QtWidgets.QLabel("导入状态"))
        self.result_label = QtWidgets.QLabel("等待导入", result_bar)
        self.result_label.setObjectName("InlineStatusLabel")
        result_layout.addWidget(self.result_label, 1)
        root.addWidget(result_bar)
        self._refresh_mode_hint()

    def _refresh_mode_hint(self) -> None:
        if self.import_mode_combo.currentData() == "append":
            self.mode_hint_label.setText("相同物种也会新增一份记录")
        else:
            self.mode_hint_label.setText("按物种名称、m/z 和电离能匹配并替换")

    def _open_calculator(self) -> None:
        window = self.window()
        if hasattr(window, "switch_workspace_page"):
            window.switch_workspace_page("pics")

    def _set_busy(self, busy: bool, message: str) -> None:
        self.choose_file_button.setDisabled(busy)
        self.import_mode_combo.setDisabled(busy)
        self.confirm_import_button.setDisabled(busy or not self._records)
        self._set_status(self.preview_status, message, "busy" if busy else "")

    def _import_pics_from_file(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择 PICS 数据文件",
            "",
            "数据文件 (*.xlsx *.xls *.csv *.tsv *.txt);;所有文件 (*)",
        )
        if not file_path:
            return
        self._selected_path = file_path
        self.file_path_edit.setText(file_path)
        self._records = []
        self.preview_table.setRowCount(0)
        self._set_busy(True, "正在解析文件…")

        def _parse() -> list[dict]:
            from bl03u_masstool.core.pics_import import parse_pics_upload

            path = Path(file_path)
            return parse_pics_upload(path.read_bytes(), path.name)

        self._worker = WorkerThread(_parse, self)
        self._worker.finished_with_result.connect(self._show_preview)
        self._worker.failed.connect(self._show_parse_error)
        self._worker.start()

    def _show_preview(self, records: object) -> None:
        self._records = list(records if isinstance(records, list) else [])
        self.preview_table.setRowCount(len(self._records))
        total_points = 0
        for row, record in enumerate(self._records):
            energies = [float(value) for value in record.get("energies", [])]
            total_points += len(energies)
            energy_range = "—" if not energies else f"{min(energies):.3f}–{max(energies):.3f}"
            values = (
                str(record.get("species", "")),
                str(record.get("mz", "")),
                "—" if record.get("ie") is None else f"{float(record['ie']):.4f}",
                energy_range,
                str(len(energies)),
            )
            for column, value in enumerate(values):
                self.preview_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        self._set_busy(False, "")
        self.confirm_import_button.setEnabled(bool(self._records))
        self._set_status(
            self.preview_status,
            f"解析完成：{len(self._records)} 个物种，{total_points} 个截面数据点",
            "success",
        )
        self._set_status(self.result_label, "请检查预览后确认导入")

    def _show_parse_error(self, message: str) -> None:
        self._records = []
        self._set_busy(False, "")
        self.confirm_import_button.setEnabled(False)
        self._set_status(self.preview_status, f"文件解析失败：{message}", "error")
        self._set_status(self.result_label, "未写入数据库", "error")

    def _write_preview_records(self) -> None:
        if not self._records:
            self._set_status(self.result_label, "请先选择并成功解析数据文件", "error")
            return
        records = list(self._records)
        mode = str(self.import_mode_combo.currentData())
        database_path = self.database_path()
        self._set_busy(True, "正在写入数据库…")

        def _write() -> dict:
            from bl03u_masstool.core.pics_import import write_pics_records

            return write_pics_records(records, database_path, mode=mode)

        self._worker = WorkerThread(_write, self)
        self._worker.finished_with_result.connect(self._show_import_result)
        self._worker.failed.connect(self._show_import_error)
        self._worker.start()

    def _show_import_result(self, result: object) -> None:
        values = dict(result if isinstance(result, dict) else {})
        self._set_busy(False, "")
        message = (
            f"导入完成：写入 {values.get('inserted_species', 0)} 个物种，"
            f"更新 {values.get('replaced_species', 0)} 条旧记录，"
            f"共 {values.get('inserted_points', 0)} 个数据点"
        )
        self._set_status(self.preview_status, "当前预览已写入数据库", "success")
        self._set_status(self.result_label, message, "success")
        self.import_completed.emit(values)

    def _show_import_error(self, message: str) -> None:
        self._set_busy(False, "")
        self._set_status(self.preview_status, "数据库写入失败", "error")
        self._set_status(self.result_label, f"导入失败：{message}", "error")
