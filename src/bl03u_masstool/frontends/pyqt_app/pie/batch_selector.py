from __future__ import annotations

from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from bl03u_masstool.core.curve_database import (
    CurveDataset,
    list_curve_datasets_read_only,
)


class CurveBatchSelectorDialog(QtWidgets.QDialog):
    """Select a view batch without implicitly changing the project current batch."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        curve_type: str = "pie",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.database_path = Path(database_path)
        self.curve_type = str(curve_type)
        self.datasets: list[CurveDataset] = []
        self.requested_action = ""
        self.selected_dataset_id: int | None = None
        type_text = "PIE" if self.curve_type == "pie" else "温度扫描"
        self.setWindowTitle(f"{type_text}分析批次")
        self.resize(920, 480)

        layout = QtWidgets.QVBoxLayout(self)
        hint = QtWidgets.QLabel(
            "“加载批次”只改变本页查看对象；“设为项目当前”才修改项目默认批次。"
        )
        hint.setObjectName("HintLabel")
        layout.addWidget(hint)

        self.table = QtWidgets.QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                "dataset_id",
                "状态",
                "角色",
                "生成时间",
                "通道/点",
                "数据源",
                "卡峰集",
                "analysis_key",
            ]
        )
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.doubleClicked.connect(lambda _index: self._finish("load"))
        layout.addWidget(self.table, stretch=1)

        buttons = QtWidgets.QHBoxLayout()
        self.load_button = QtWidgets.QPushButton("加载批次")
        self.load_button.setObjectName("PrimaryButton")
        self.load_button.clicked.connect(lambda: self._finish("load"))
        self.current_button = QtWidgets.QPushButton("设为项目当前")
        self.current_button.clicked.connect(lambda: self._finish("set_current"))
        self.stale_button = QtWidgets.QPushButton("标记过期")
        self.stale_button.clicked.connect(lambda: self._finish("mark_stale"))
        self.archive_button = QtWidgets.QPushButton("归档")
        self.archive_button.clicked.connect(lambda: self._finish("archive"))
        buttons.addStretch()
        buttons.addWidget(self.load_button)
        buttons.addWidget(self.current_button)
        buttons.addWidget(self.stale_button)
        buttons.addWidget(self.archive_button)
        close_button = QtWidgets.QPushButton("关闭")
        close_button.clicked.connect(self.reject)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        self._load()

    def _load(self) -> None:
        try:
            self.datasets = list_curve_datasets_read_only(
                self.database_path,
                curve_type=self.curve_type,
                include_stale=True,
            )
        except Exception as exc:
            self.datasets = []
            QtWidgets.QMessageBox.critical(self, "批次读取失败", str(exc))
        self.table.setRowCount(len(self.datasets))
        for row_index, dataset in enumerate(self.datasets):
            provenance = dataset.metadata.get("analysis_provenance")
            if not isinstance(provenance, dict):
                provenance = {}
            peak_set = str(provenance.get("active_peak_set_id") or "")
            if not peak_set:
                peak = provenance.get("manual_peak_file")
                if isinstance(peak, dict):
                    peak_set = str(peak.get("name") or peak.get("path") or "")
            state = (
                "当前"
                if dataset.is_current
                else {
                    "valid": "有效历史",
                    "stale": "过期",
                    "archived": "归档",
                }.get(dataset.validity_status, dataset.validity_status)
            )
            values = (
                str(dataset.dataset_id),
                state,
                dataset.dataset_role,
                dataset.created_at,
                f"{dataset.channel_count}/{dataset.point_count}",
                dataset.source_label.replace("\n", "；"),
                peak_set or "未登记",
                dataset.analysis_key,
            )
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(
                    QtCore.Qt.ItemDataRole.UserRole,
                    dataset.dataset_id,
                )
                self.table.setItem(row_index, column, item)
        if self.datasets:
            current = next(
                (
                    index
                    for index, dataset in enumerate(self.datasets)
                    if dataset.is_current
                ),
                0,
            )
            self.table.selectRow(current)
        self.table.resizeColumnsToContents()
        self._update_buttons()

    def _selected_dataset(self) -> CurveDataset | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        index = rows[0].row()
        return self.datasets[index] if index < len(self.datasets) else None

    def _update_buttons(self) -> None:
        dataset = self._selected_dataset()
        can_load = dataset is not None and dataset.validity_status != "archived"
        self.load_button.setEnabled(can_load)
        self.current_button.setEnabled(
            dataset is not None and dataset.validity_status == "valid"
        )
        self.stale_button.setEnabled(
            dataset is not None and dataset.validity_status == "valid"
        )
        self.archive_button.setEnabled(
            dataset is not None and dataset.validity_status != "archived"
        )

    def _finish(self, action: str) -> None:
        dataset = self._selected_dataset()
        if dataset is None:
            return
        self.requested_action = action
        self.selected_dataset_id = dataset.dataset_id
        self.accept()
