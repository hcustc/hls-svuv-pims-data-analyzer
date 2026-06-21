"""
FittingControlWidget - 拟合配置面板

职责：
- 强制物种管理（输入、标签、移除）
- 候选物种配置（选择、系数模式、系数编辑）
- 所有拟合前的参数设置

设计原则：
- 仅负责UI展示和用户输入收集
- 通过信号向上报告配置变更
- 通过方法接收数据更新（不通过信号）
"""

from PyQt6 import QtCore, QtWidgets

from bl03u_masstool.frontends.pyqt_app.common.widgets import FlowLayout


class FittingControlWidget(QtWidgets.QWidget):
    """拟合配置面板 - 强制物种和候选物种管理"""

    # ---- 信号定义 ----
    force_species_added = QtCore.pyqtSignal(str)          # 添加单个强制物种
    force_species_removed = QtCore.pyqtSignal(str)        # 移除单个强制物种
    force_species_cleared = QtCore.pyqtSignal()           # 清除所有强制物种

    coefficient_mode_changed = QtCore.pyqtSignal(str)     # 系数模式: auto/locked_fit/manual
    candidate_selection_changed = QtCore.pyqtSignal(list) # 选中的候选物种ID列表
    candidate_coefficient_changed = QtCore.pyqtSignal(dict) # {candidate_id: coefficient_value}

    # ---- 控制变更信号 ----
    candidates_import_requested = QtCore.pyqtSignal()     # 导入系数按钮点击
    candidates_zeroed = QtCore.pyqtSignal()               # 清零按钮点击

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- 大小策略和约束 ----
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum  # 竖向不扩展，只取所需高度
        )
        self.setMinimumHeight(100)
        self.setMaximumHeight(16777215)  # 允许展开但不强制

        # ---- 内部状态 ----
        self._force_species: list[str] = []
        self._candidate_data: list[dict] = []
        self._candidate_updating = False

        # ---- UI构建 ----
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)

        # Header: "拟合配置"
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("拟合配置")
        title.setObjectName("ReadoutLabel")
        header.addWidget(title)
        header.addStretch()

        # Tab widget
        self.control_tabs = QtWidgets.QTabWidget()
        self.control_tabs.setMinimumHeight(100)

        # ---- Tab 1: 强制物种 ----
        force_panel = QtWidgets.QWidget()
        force_panel_layout = QtWidgets.QVBoxLayout(force_panel)
        force_panel_layout.setContentsMargins(0, 4, 0, 0)
        force_panel_layout.setSpacing(6)

        force_input_row = QtWidgets.QHBoxLayout()
        force_input_row.setSpacing(6)
        self.force_input = QtWidgets.QLineEdit()
        self.force_input.setPlaceholderText("输入物种名称后回车添加...")
        self.force_input.returnPressed.connect(self._add_force_from_input)
        force_input_row.addWidget(self.force_input, stretch=1)
        add_force_btn = QtWidgets.QPushButton("添加")
        add_force_btn.setObjectName("BrowseButton")
        add_force_btn.setFixedHeight(28)
        add_force_btn.clicked.connect(self._add_force_from_input)
        force_input_row.addWidget(add_force_btn)
        force_panel_layout.addLayout(force_input_row)

        tag_container = QtWidgets.QWidget()
        tag_container.setObjectName("TagCloud")
        self.force_tag_layout = FlowLayout(tag_container, margin=0, spacing=4)
        tag_container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        force_panel_layout.addWidget(tag_container, stretch=1)
        self.control_tabs.addTab(force_panel, "强制物种")

        # ---- Tab 2: 候选物种 ----
        candidate_panel = QtWidgets.QWidget()
        candidate_layout = QtWidgets.QVBoxLayout(candidate_panel)
        candidate_layout.setContentsMargins(0, 4, 0, 0)
        candidate_layout.setSpacing(4)

        self.candidate_table = QtWidgets.QTableWidget()
        self.candidate_table.setColumnCount(6)
        self.candidate_table.setHorizontalHeaderLabels(["选择", "物种", "m/z", "IE(eV)", "系数", "锁定"])
        self.candidate_table.setWordWrap(False)
        self.candidate_table.setAlternatingRowColors(True)
        self.candidate_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidate_table.horizontalHeader().setStretchLastSection(True)

        # 候选控制栏 - 分两层以降低密度
        # 第一层：系数模式选择
        candidate_mode_row = QtWidgets.QHBoxLayout()
        candidate_mode_row.setSpacing(6)
        candidate_mode_row.addWidget(QtWidgets.QLabel("系数模式:"))
        self.coefficient_mode_combo = QtWidgets.QComboBox()
        self.coefficient_mode_combo.addItem("自动拟合", "auto")
        self.coefficient_mode_combo.addItem("锁定已选", "locked_fit")
        self.coefficient_mode_combo.addItem("手动系数", "manual")
        self.coefficient_mode_combo.currentIndexChanged.connect(self._on_coefficient_mode_changed)
        candidate_mode_row.addWidget(self.coefficient_mode_combo)
        candidate_mode_row.addStretch()

        # 第二层：候选物种控制按钮
        candidate_action_row = QtWidgets.QHBoxLayout()
        candidate_action_row.setSpacing(6)
        self.candidate_select_all_btn = QtWidgets.QPushButton("全选")
        self.candidate_select_all_btn.clicked.connect(lambda: self._set_all_candidates_checked(True))
        self.candidate_clear_btn = QtWidgets.QPushButton("清除选择")
        self.candidate_clear_btn.setToolTip("取消所有候选物种的勾选")
        self.candidate_clear_btn.clicked.connect(lambda: self._set_all_candidates_checked(False))
        self.candidate_import_coeff_btn = QtWidgets.QPushButton("导入系数")
        self.candidate_import_coeff_btn.setToolTip("从当前拟合结果导入候选物种系数")
        self.candidate_import_coeff_btn.clicked.connect(self._import_coefficients_from_fit)
        self.candidate_zero_coeff_btn = QtWidgets.QPushButton("系数清零")
        self.candidate_zero_coeff_btn.setToolTip("将所有候选物种系数值清零")
        self.candidate_zero_coeff_btn.clicked.connect(self._zero_all_coefficients)
        for btn in (self.candidate_select_all_btn, self.candidate_clear_btn,
                     self.candidate_import_coeff_btn, self.candidate_zero_coeff_btn):
            btn.setObjectName("BrowseButton")
        candidate_action_row.addWidget(self.candidate_select_all_btn)
        candidate_action_row.addWidget(self.candidate_clear_btn)
        candidate_action_row.addWidget(self.candidate_import_coeff_btn)
        candidate_action_row.addWidget(self.candidate_zero_coeff_btn)
        candidate_action_row.addStretch()

        # 组织两层为垂直控制栏
        candidate_controls_container = QtWidgets.QVBoxLayout()
        candidate_controls_container.setSpacing(4)
        candidate_controls_container.addLayout(candidate_mode_row)
        candidate_controls_container.addLayout(candidate_action_row)

        candidate_layout.addLayout(candidate_controls_container)
        candidate_layout.addWidget(self.candidate_table, stretch=1)

        self.control_tabs.addTab(candidate_panel, "候选物种")

        main_layout.addLayout(header)
        main_layout.addWidget(self.control_tabs, stretch=1)

    # ---- 强制物种方法 ----

    def _add_force_from_input(self):
        """Add species from the input field to the force-fit tag cloud."""
        name = self.force_input.text().strip()
        if name and name not in self.get_force_species():
            current = list(self.get_force_species())
            current.append(name)
            self._force_species = current
            self._rebuild_force_tags()
            self.force_species_added.emit(name)
        self.force_input.clear()

    def add_force_species(self, species_name: str):
        """Add a force-fit species (called externally, e.g. from m/z list)."""
        current = list(self.get_force_species())
        if species_name not in current:
            current.append(species_name)
            self._force_species = current
            self._rebuild_force_tags()
            self.force_species_added.emit(species_name)

    def _remove_force_species_name(self, name: str):
        """Remove a specific species from the force-fit list by name."""
        current = list(self.get_force_species())
        if name in current:
            current.remove(name)
            self._force_species = current
            self._rebuild_force_tags()
            self.force_species_removed.emit(name)

    def get_force_species(self) -> list[str]:
        """Return current force-fit species list."""
        return list(self._force_species)

    def set_force_species(self, species: list[str]):
        """Set force-fit species list (called externally for initialization)."""
        self._force_species = list(species)
        self._rebuild_force_tags()

    def _rebuild_force_tags(self):
        """Rebuild the tag chips in the flow layout from _force_species."""
        # Clear existing tag widgets
        while self.force_tag_layout.count():
            item = self.force_tag_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for name in self.get_force_species():
            tag = QtWidgets.QFrame()
            tag.setObjectName("ForceTag")
            tag.setFixedHeight(24)
            tag_layout = QtWidgets.QHBoxLayout(tag)
            tag_layout.setContentsMargins(6, 1, 2, 1)
            tag_layout.setSpacing(2)

            label = QtWidgets.QLabel(name)
            label.setObjectName("ForceTagText")
            tag_layout.addWidget(label)

            close_btn = QtWidgets.QPushButton("\u2715")
            close_btn.setObjectName("TagCloseButton")
            close_btn.setFixedSize(16, 16)
            close_btn.clicked.connect(lambda checked=False, n=name: self._remove_force_species_name(n))
            tag_layout.addWidget(close_btn)

            self.force_tag_layout.addWidget(tag)

        # Force re-layout of the tag container
        container = self.force_tag_layout.parent()
        if container is not None:
            container.updateGeometry()

    # ---- 候选物种方法 ----

    def populate_candidate_table(self, mz: int, filtered_db: list[dict]):
        """根据选中的m/z填充候选物种表格"""
        self._candidate_updating = True
        try:
            candidates = [item for item in filtered_db if item.get("mz") == mz]
            self._candidate_data = candidates
            self.candidate_table.setRowCount(len(candidates))
            for row, species in enumerate(candidates):
                # 选择 checkbox
                check_widget = QtWidgets.QWidget()
                check_layout = QtWidgets.QHBoxLayout(check_widget)
                check_layout.setContentsMargins(0, 0, 0, 0)
                check_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                chk = QtWidgets.QCheckBox()
                chk.setChecked(True)
                chk.stateChanged.connect(lambda state, r=row: self._on_candidate_changed(r))
                check_layout.addWidget(chk)
                self.candidate_table.setCellWidget(row, 0, check_widget)

                # 物种名称
                name_item = QtWidgets.QTableWidgetItem(str(species.get("species", "")))
                name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.candidate_table.setItem(row, 1, name_item)

                # m/z
                mz_item = QtWidgets.QTableWidgetItem(str(species.get("mz", "")))
                mz_item.setFlags(mz_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.candidate_table.setItem(row, 2, mz_item)

                # IE
                ie = species.get("ionization_energy")
                ie_text = f"{ie:.4f}" if ie is not None else ""
                ie_item = QtWidgets.QTableWidgetItem(ie_text)
                ie_item.setFlags(ie_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                self.candidate_table.setItem(row, 3, ie_item)

                # 系数 spinbox（减小高度和内边距以节省空间）
                coeff_spin = QtWidgets.QDoubleSpinBox()
                coeff_spin.setRange(0, 1e6)
                coeff_spin.setDecimals(6)
                coeff_spin.setValue(0.0)
                coeff_spin.setEnabled(False)
                coeff_spin.setFixedHeight(24)  # 固定高度，避免占用过多空间
                coeff_spin.setContentsMargins(0, 0, 0, 0)
                coeff_spin.valueChanged.connect(lambda val, r=row: self._on_candidate_changed(r))
                self.candidate_table.setCellWidget(row, 4, coeff_spin)

                # 锁定 checkbox
                lock_widget = QtWidgets.QWidget()
                lock_layout = QtWidgets.QHBoxLayout(lock_widget)
                lock_layout.setContentsMargins(0, 0, 0, 0)
                lock_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                lock_chk = QtWidgets.QCheckBox()
                lock_chk.setEnabled(False)
                lock_chk.stateChanged.connect(lambda state, r=row: self._on_candidate_changed(r))
                lock_layout.addWidget(lock_chk)
                self.candidate_table.setCellWidget(row, 5, lock_widget)

            # 设置列宽策略
            header = self.candidate_table.horizontalHeader()
            # 固定宽列：选择、m/z、IE、锁定
            self.candidate_table.setColumnWidth(0, 40)   # 选择 checkbox
            self.candidate_table.setColumnWidth(2, 50)   # m/z
            self.candidate_table.setColumnWidth(3, 70)   # IE
            self.candidate_table.setColumnWidth(5, 40)   # 锁定 checkbox

            # 自适应列：物种名称、系数
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)  # 物种名称
            header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.Stretch)  # 系数
        finally:
            self._candidate_updating = False

    def _on_candidate_changed(self, row: int):
        """候选表格变化时发出信号"""
        if self._candidate_updating:
            return
        self.candidate_selection_changed.emit(self._get_selected_candidate_ids())

    def _on_coefficient_mode_changed(self, index: int):
        """系数模式切换"""
        mode = self.coefficient_mode_combo.currentData()
        for row in range(self.candidate_table.rowCount()):
            coeff_spin = self.candidate_table.cellWidget(row, 4)
            if isinstance(coeff_spin, QtWidgets.QDoubleSpinBox):
                coeff_spin.setEnabled(mode != "auto")
            lock_widget = self.candidate_table.cellWidget(row, 5)
            if lock_widget:
                lock_chk = lock_widget.findChild(QtWidgets.QCheckBox)
                if lock_chk:
                    lock_chk.setEnabled(mode == "locked_fit")
                    if mode != "locked_fit":
                        lock_chk.setChecked(False)
        self.coefficient_mode_changed.emit(mode)

    def _set_all_candidates_checked(self, checked: bool):
        """全选或清空候选物种"""
        self._candidate_updating = True
        try:
            for row in range(self.candidate_table.rowCount()):
                check_widget = self.candidate_table.cellWidget(row, 0)
                if check_widget:
                    chk = check_widget.findChild(QtWidgets.QCheckBox)
                    if chk:
                        chk.setChecked(checked)
        finally:
            self._candidate_updating = False
        self.candidate_selection_changed.emit(self._get_selected_candidate_ids())

    def _import_coefficients_from_fit(self):
        """从当前拟合结果导入系数到候选面板（信号通知dialog）"""
        self.candidates_import_requested.emit()

    def import_coefficients(self, species_list: list[dict]):
        """接收dialog传入的拟合结果物种列表并更新系数"""
        if not species_list:
            return
        self._candidate_updating = True
        try:
            coeff_by_name = {sp.get("species"): float(sp.get("coefficient", 0)) for sp in species_list}
            for row in range(self.candidate_table.rowCount()):
                name_item = self.candidate_table.item(row, 1)
                if name_item and name_item.text() in coeff_by_name:
                    coeff_spin = self.candidate_table.cellWidget(row, 4)
                    if isinstance(coeff_spin, QtWidgets.QDoubleSpinBox):
                        coeff_spin.setValue(coeff_by_name[name_item.text()])
        finally:
            self._candidate_updating = False

    def _zero_all_coefficients(self):
        """将所有候选物种系数清零"""
        self._candidate_updating = True
        try:
            for row in range(self.candidate_table.rowCount()):
                coeff_spin = self.candidate_table.cellWidget(row, 4)
                if isinstance(coeff_spin, QtWidgets.QDoubleSpinBox):
                    coeff_spin.setValue(0.0)
        finally:
            self._candidate_updating = False
        self.candidates_zeroed.emit()

    def get_candidates(self) -> list[dict]:
        """获取当前候选物种列表"""
        return list(self._candidate_data)

    def set_candidates(self, candidates: list[dict]):
        """设置候选物种列表（初始化时）"""
        self._candidate_data = list(candidates)

    def get_coefficient_mode(self) -> str:
        """获取当前系数模式"""
        return self.coefficient_mode_combo.currentData()

    def _get_selected_candidate_ids(self) -> list[int]:
        """获取选中的候选物种ID列表"""
        selected_ids = []
        for row in range(self.candidate_table.rowCount()):
            check_widget = self.candidate_table.cellWidget(row, 0)
            chk = check_widget.findChild(QtWidgets.QCheckBox) if check_widget else None
            if chk and chk.isChecked() and row < len(self._candidate_data):
                species = self._candidate_data[row]
                species_id = int(species.get("id", row + 1))
                selected_ids.append(species_id)
        return selected_ids

    def clear_ui(self):
        """清空所有UI内容（切换m/z或清除拟合时）"""
        self._force_species = []
        self._candidate_data = []
        self.force_input.clear()
        self._rebuild_force_tags()
        self.candidate_table.setRowCount(0)
