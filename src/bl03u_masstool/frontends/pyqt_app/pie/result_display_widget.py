"""
ResultDisplayWidget - 结果详情展示面板

职责：
- 展示曲线数据表格
- 展示物种贡献明细表格
- 显示拟合结果的有效性状态
- 导出鉴定结果操作

设计原则：
- 仅负责结果的展示和用户交互
- 通过方法接收数据更新（不通过信号）
- 通过信号向上报告用户操作
- 根据结果状态自动启用/禁用导出按钮
"""

from PyQt6 import QtCore, QtGui, QtWidgets

from bl03u_masstool.core.pie_analysis import species_ionization_energy_value


class CopyableTableWidget(QtWidgets.QTableWidget):
    """Read-only-friendly table with spreadsheet-compatible copying."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.ActionsContextMenu)
        copy_selected_action = QtGui.QAction("复制所选", self)
        copy_selected_action.setShortcut(
            QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Copy)
        )
        copy_selected_action.setShortcutContext(
            QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        copy_selected_action.triggered.connect(self.copy_selected_to_clipboard)
        self.addAction(copy_selected_action)

        copy_all_action = QtGui.QAction("复制全部", self)
        copy_all_action.triggered.connect(self.copy_all_to_clipboard)
        self.addAction(copy_all_action)

    def copy_all_to_clipboard(self) -> None:
        headers = [
            self.horizontalHeaderItem(column).text()
            if self.horizontalHeaderItem(column) is not None
            else ""
            for column in range(self.columnCount())
        ]
        lines = ["\t".join(headers)]
        lines.extend(
            "\t".join(
                self.item(row, column).text()
                if self.item(row, column) is not None
                else ""
                for column in range(self.columnCount())
            )
            for row in range(self.rowCount())
        )
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))

    def copy_selected_to_clipboard(self) -> None:
        indexes = self.selectedIndexes()
        if not indexes:
            return
        selected = {(index.row(), index.column()) for index in indexes}
        rows = range(
            min(row for row, _column in selected),
            max(row for row, _column in selected) + 1,
        )
        columns = range(
            min(column for _row, column in selected),
            max(column for _row, column in selected) + 1,
        )
        text = "\n".join(
            "\t".join(
                self.item(row, column).text()
                if (row, column) in selected and self.item(row, column) is not None
                else ""
                for column in columns
            )
            for row in rows
        )
        QtWidgets.QApplication.clipboard().setText(text)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.matches(QtGui.QKeySequence.StandardKey.Copy):
            self.copy_selected_to_clipboard()
            event.accept()
            return
        super().keyPressEvent(event)


class ResultDisplayWidget(QtWidgets.QWidget):
    """结果详情面板 - 曲线数据和物种贡献明细"""

    # ---- 信号定义 ----
    visibility_changed = QtCore.pyqtSignal(bool)    # 展开/收起时发出

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- 大小策略和约束 ----
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding  # 竖向可扩展
        )
        self.setMinimumHeight(120)  # 最小高度（header + 一点表格）
        self.setMaximumHeight(16777215)  # 允许最大化

        # ---- 初始隐藏 ----
        self.setVisible(False)

        # ---- 面板样式钩子（具体样式由应用级 QSS 统一管理） ----
        self.setObjectName("ResultDisplayPanel")

        # ---- 内部状态 ----
        self._current_status = "UNFITTED"

        # ---- UI构建 ----
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)  # 添加内边距
        main_layout.setSpacing(6)

        # Header: "结果详情" [状态] + [导出按钮] (with background)
        header_frame = QtWidgets.QFrame()
        header_frame.setObjectName("ResultPanelHeader")
        header_frame.setFixedHeight(32)
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(8, 4, 8, 4)
        header.setSpacing(8)
        title = QtWidgets.QLabel("详细数据")
        title.setObjectName("ReadoutLabel")
        header.addWidget(title)

        # ► 状态指示器
        self.status_indicator = QtWidgets.QLabel("")  # 显示状态文字
        self.status_indicator.setObjectName("ResultStatus")
        self.status_indicator.setFixedWidth(100)
        self.status_indicator.setToolTip("结果有效性状态: COMPLETED / OBSOLETE / FAILED")
        header.addWidget(self.status_indicator)

        header.addStretch()

        self.metrics_label = QtWidgets.QLabel("")
        self.metrics_label.setObjectName("ResultMetrics")
        self.metrics_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        header.addWidget(self.metrics_label)

        main_layout.addWidget(header_frame)

        # Tab widget (curve_data, fit_table)
        self.result_tabs = QtWidgets.QTabWidget()
        self.result_tabs.setMinimumHeight(120)

        # ---- Tab 1: 曲线数据 ----
        self.curve_table = CopyableTableWidget()
        self.curve_table.setWordWrap(False)
        self.curve_table.setAlternatingRowColors(True)
        self.curve_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.result_tabs.addTab(self.curve_table, "曲线数据")

        # ---- Tab 2: 物种贡献明细 ----
        self.fit_table = CopyableTableWidget()
        self.fit_table.setWordWrap(False)
        self.fit_table.setAlternatingRowColors(True)
        self.fit_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.fit_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.result_tabs.addTab(self.fit_table, "物种贡献明细")

        main_layout.addWidget(self.result_tabs, stretch=1)

    # ---- 数据更新方法 ----

    def update_curve_data(self, data: dict):
        """
        更新曲线数据表格

        Args:
            data: 包含能量、强度、拟合曲线等数据的字典
        """
        if not data:
            self.curve_table.setRowCount(0)
            return

        # 提取数据
        energies = data.get("energies", [])
        experimental = data.get("experimental", [])
        total_fit = data.get("total_fit", [])
        residual = data.get("residual", [])

        n_rows = max(
            len(energies) if hasattr(energies, '__len__') else 0,
            len(experimental) if hasattr(experimental, '__len__') else 0,
            len(total_fit) if hasattr(total_fit, '__len__') else 0,
            len(residual) if hasattr(residual, '__len__') else 0,
        )

        self.curve_table.setColumnCount(4)
        self.curve_table.setHorizontalHeaderLabels(["能量(eV)", "实验值", "拟合值", "残差"])
        self.curve_table.setRowCount(n_rows)

        for row in range(n_rows):
            # 能量
            e_val = energies[row] if row < len(energies) else ""
            e_text = f"{float(e_val):.4f}" if e_val != "" else ""
            energy_item = QtWidgets.QTableWidgetItem(e_text)
            energy_item.setFlags(energy_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.curve_table.setItem(row, 0, energy_item)

            # 实验值
            exp_val = experimental[row] if row < len(experimental) else ""
            exp_text = f"{float(exp_val):.6f}" if exp_val != "" else ""
            exp_item = QtWidgets.QTableWidgetItem(exp_text)
            exp_item.setFlags(exp_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.curve_table.setItem(row, 1, exp_item)

            # 拟合值
            fit_val = total_fit[row] if row < len(total_fit) else ""
            fit_text = f"{float(fit_val):.6f}" if fit_val != "" else ""
            fit_item = QtWidgets.QTableWidgetItem(fit_text)
            fit_item.setFlags(fit_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.curve_table.setItem(row, 2, fit_item)

            # 残差
            res_val = residual[row] if row < len(residual) else ""
            res_text = f"{float(res_val):.6f}" if res_val != "" else ""
            res_item = QtWidgets.QTableWidgetItem(res_text)
            res_item.setFlags(res_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.curve_table.setItem(row, 3, res_item)

        self.curve_table.resizeColumnsToContents()
        self.curve_table.horizontalHeader().setStretchLastSection(True)

    def update_fit_table(self, results: dict):
        """
        更新物种贡献明细表格

        Args:
            results: 拟合结果字典，包含 'species' 列表
        """
        if not results:
            self.fit_table.setRowCount(0)
            self.metrics_label.setText("")
            return

        species_list = results.get("species", [])
        r_squared = results.get("r_squared", 0.0)
        rmse = results.get("rmse", 0.0)
        mae = results.get("mae", 0.0)
        self.metrics_label.setText(f"R² {r_squared:.4f}   RMSE {rmse:.4g}   MAE {mae:.4g}")

        self.fit_table.setColumnCount(5)
        self.fit_table.setHorizontalHeaderLabels(["物种", "系数", "贡献(%)", "IE(eV)", "m/z"])
        self.fit_table.setRowCount(len(species_list) + 1)  # +1 for summary row

        # 物种行
        for row, species in enumerate(species_list):
            name = species.get("species", "")
            coeff = species.get("coefficient", 0.0)
            contrib_pct = float(
                species.get("contribution_percent", species.get("contribution", 0.0)) or 0.0
            )
            ie = species_ionization_energy_value(species)
            mz = species.get("mz", "")

            # 物种名
            name_item = QtWidgets.QTableWidgetItem(str(name))
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.fit_table.setItem(row, 0, name_item)

            # 系数
            coeff_item = QtWidgets.QTableWidgetItem(f"{float(coeff):.6f}")
            coeff_item.setFlags(coeff_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.fit_table.setItem(row, 1, coeff_item)

            # 贡献%
            contrib_item = QtWidgets.QTableWidgetItem(f"{contrib_pct:.2f}%")
            contrib_item.setFlags(contrib_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.fit_table.setItem(row, 2, contrib_item)

            # IE
            try:
                ie_value = float(ie)
                ie_text = f"{ie_value:.4f}" if ie_value > 0 else "未查到"
            except (TypeError, ValueError):
                ie_text = {
                    "pending": "查询中…",
                    "failed": "查询失败",
                }.get(str(species.get("ie_query_status", "")), "未查到")
            ie_item = QtWidgets.QTableWidgetItem(ie_text)
            ie_item.setToolTip(
                f"来源: {species.get('ie_source') or '未提供'}\n"
                f"{species.get('ie_message') or ''}"
            )
            ie_item.setFlags(ie_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.fit_table.setItem(row, 3, ie_item)

            # m/z
            mz_item = QtWidgets.QTableWidgetItem(str(mz))
            mz_item.setFlags(mz_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.fit_table.setItem(row, 4, mz_item)

        # 汇总行
        summary_row = len(species_list)
        summary_cells = [
            f"拟合统计",
            f"R²={r_squared:.4f}",
            f"RMSE={rmse:.6f}",
            f"MAE={mae:.6f}",
            ""
        ]
        for col, cell_text in enumerate(summary_cells):
            item = QtWidgets.QTableWidgetItem(cell_text)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            item.setBackground(QtCore.Qt.GlobalColor.lightGray)
            self.fit_table.setItem(summary_row, col, item)

        self.fit_table.resizeColumnsToContents()
        self.fit_table.horizontalHeader().setStretchLastSection(True)

    def clear_data(self):
        """清空所有表格数据"""
        self.curve_table.setRowCount(0)
        self.fit_table.setRowCount(0)
        self.metrics_label.setText("")
        self.set_result_status("UNFITTED")

    # ---- 状态管理方法 ----

    def set_result_status(self, status: str) -> None:
        """
        设置结果有效性状态

        Args:
            status: "UNFITTED" | "PREVIEW" | "COMPLETED" | "OBSOLETE" | "FAILED" | "CONFIRMED"
        """
        self._current_status = status

        status_text = {
            "UNFITTED": "",
            "PREVIEW": "实时预览",
            "COMPLETED": "[已拟合]",
            "OBSOLETE": "[结果过期]",
            "FAILED": "[拟合失败]",
            "CONFIRMED": "[✓ 已确认]",
        }
        self.status_indicator.setText(status_text.get(status, ""))

        self.status_indicator.setProperty("status", status.lower())
        self.status_indicator.style().unpolish(self.status_indicator)
        self.status_indicator.style().polish(self.status_indicator)

    def get_result_status(self) -> str:
        """获取当前结果状态"""
        return self._current_status

    # ---- 可见性管理 ----

    def setVisible(self, visible: bool):
        """重写setVisible以发出信号"""
        super().setVisible(visible)
        self.visibility_changed.emit(visible)

    def is_visible(self) -> bool:
        """检查面板是否可见"""
        return super().isVisible()
