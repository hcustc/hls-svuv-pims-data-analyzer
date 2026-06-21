"""
FittingControlWidget - 拟合配置面板

职责：
- 统一的拟合物种配置界面
- 合并锁定候选物种和候选物种到单一表格
- 支持物种来源标记（自动/手动）和锁定状态管理（锁定/普通）
- 所有拟合前的参数设置

设计原则：
- 仅负责UI展示和用户输入收集
- 通过信号向上报告配置变更
- 通过方法接收数据更新（不通过信号）
- 单一界面，无Tab切换

锁定候选语义：
- 被锁定的物种始终进入拟合候选集合，不被自动筛选移除
- 其系数由优化器自由决定，可为0或接近0
- "锁定"只表示用户要求算法考虑该物种，不表示已鉴别或必须参与
- 详见 forced_species_analysis.md
"""

from PyQt6 import QtCore, QtWidgets
import numpy as np


class FittingControlWidget(QtWidgets.QWidget):
    """拟合配置面板 - 统一的拟合物种配置"""

    # ---- 信号定义 ----
    # 新信号：锁定候选相关（主要使用）
    locked_candidate_added = QtCore.pyqtSignal(str)          # 添加单个锁定候选
    locked_candidate_removed = QtCore.pyqtSignal(str)        # 移除单个锁定候选
    candidate_lock_toggled = QtCore.pyqtSignal(str)          # 物种锁定状态切换

    # 向后兼容的信号（别名，保留但不建议使用）
    force_species_added = locked_candidate_added             # 别名
    force_species_removed = locked_candidate_removed         # 别名
    force_species_cleared = QtCore.pyqtSignal()              # 清除所有锁定候选（保留）

    # 统一的配置变更信号
    coefficient_mode_changed = QtCore.pyqtSignal(str)        # 系数模式: auto/locked_fit/manual
    species_config_changed = QtCore.pyqtSignal()             # 物种配置整体改变
    candidate_selection_changed = QtCore.pyqtSignal(list)    # 选中的候选物种ID列表

    # ---- 控制变更信号 ----
    candidates_import_requested = QtCore.pyqtSignal()        # 导入系数按钮点击
    candidates_zeroed = QtCore.pyqtSignal()                  # 清零按钮点击
    species_forced_toggled = QtCore.pyqtSignal(str)          # 向后兼容：物种状态切换（改用candidate_lock_toggled）

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- 大小策略和约束 ----
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.setMinimumHeight(100)
        self.setMaximumHeight(16777215)

        # ---- 面板样式 ----
        self.setObjectName("FittingConfigPanel")
        self.setStyleSheet("""
            #FittingConfigPanel {
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                background-color: #ffffff;
            }
        """)

        # ---- 内部状态 ----
        self._locked_species: list[str] = []                   # 主要使用：锁定候选物种名称列表
        self._unified_species_data: list[dict] = []            # 统一的物种数据
        self._current_mz: int | None = None                    # 当前m/z
        self._updating = False                                 # 防止递归更新

        # ---- UI构建 ----
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Header: "拟合配置"
        header_frame = QtWidgets.QFrame()
        header_frame.setObjectName("PanelHeader")
        header_frame.setStyleSheet("""
            #PanelHeader {
                background-color: #f1f5f9;
                border-bottom: 1px solid #e2e8f0;
                border-radius: 3px;
                padding: 4px 0px;
                margin-bottom: 4px;
            }
        """)
        header_frame.setFixedHeight(32)
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(8, 4, 8, 4)
        header.setSpacing(0)
        title = QtWidgets.QLabel("拟合配置")
        title.setObjectName("ReadoutLabel")
        header.addWidget(title)
        header.addStretch()
        main_layout.addWidget(header_frame)

        # ---- 顶部：物种输入面板 ----
        input_row = QtWidgets.QHBoxLayout()
        input_row.setSpacing(6)
        self.species_input = QtWidgets.QLineEdit()
        self.species_input.setPlaceholderText("输入物种名称后按回车或点击\"添加候选\"来添加候选物种...")
        self.species_input.returnPressed.connect(self._add_species_from_input)
        input_row.addWidget(self.species_input, stretch=1)

        add_species_btn = QtWidgets.QPushButton("添加候选")
        add_species_btn.setObjectName("BrowseButton")
        add_species_btn.setFixedHeight(28)
        add_species_btn.clicked.connect(self._add_species_from_input)
        input_row.addWidget(add_species_btn)
        main_layout.addLayout(input_row)

        # ---- 中间：控制栏（系数模式 + 按钮）----
        control_row1 = QtWidgets.QHBoxLayout()
        control_row1.setSpacing(6)
        control_row1.addWidget(QtWidgets.QLabel("系数模式:"))

        self.coefficient_mode_combo = QtWidgets.QComboBox()
        self.coefficient_mode_combo.addItem("自动拟合", "auto")
        self.coefficient_mode_combo.addItem("锁定已选", "locked_fit")
        self.coefficient_mode_combo.addItem("手动系数", "manual")
        self.coefficient_mode_combo.currentIndexChanged.connect(self._on_coefficient_mode_changed)
        control_row1.addWidget(self.coefficient_mode_combo)
        control_row1.addStretch()
        main_layout.addLayout(control_row1)

        # 按钮行
        control_row2 = QtWidgets.QHBoxLayout()
        control_row2.setSpacing(6)

        self.select_all_btn = QtWidgets.QPushButton("全选")
        self.select_all_btn.clicked.connect(lambda: self._set_all_rows_checked(True))
        self.select_all_btn.setObjectName("BrowseButton")
        control_row2.addWidget(self.select_all_btn)

        self.clear_selection_btn = QtWidgets.QPushButton("清除选择")
        self.clear_selection_btn.setToolTip("取消所有物种的勾选")
        self.clear_selection_btn.clicked.connect(lambda: self._set_all_rows_checked(False))
        self.clear_selection_btn.setObjectName("BrowseButton")
        control_row2.addWidget(self.clear_selection_btn)

        self.import_coeff_btn = QtWidgets.QPushButton("导入系数")
        self.import_coeff_btn.setToolTip("从当前拟合结果导入系数")
        self.import_coeff_btn.clicked.connect(lambda: self.candidates_import_requested.emit())
        self.import_coeff_btn.setObjectName("BrowseButton")
        control_row2.addWidget(self.import_coeff_btn)

        self.zero_coeff_btn = QtWidgets.QPushButton("系数清零")
        self.zero_coeff_btn.setToolTip("将所有系数值清零")
        self.zero_coeff_btn.clicked.connect(self._zero_all_coefficients)
        self.zero_coeff_btn.setObjectName("BrowseButton")
        control_row2.addWidget(self.zero_coeff_btn)

        control_row2.addStretch()
        main_layout.addLayout(control_row2)

        # ---- 下方：简化物种表格（4列：启用 | 物种(※) | 系数 | 操作） ----
        self.species_table = QtWidgets.QTableWidget()
        self.species_table.setColumnCount(4)
        self.species_table.setHorizontalHeaderLabels([
            "启用", "物种", "系数", "操作"
        ])
        self.species_table.setWordWrap(False)
        self.species_table.setAlternatingRowColors(True)
        self.species_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.species_table.horizontalHeader().setStretchLastSection(False)
        main_layout.addWidget(self.species_table, stretch=1)

    # ---- 物种数据管理方法 ----

    def _add_species_from_input(self):
        """从输入框添加物种为普通候选，用户可后续点击按钮锁定"""
        name = self.species_input.text().strip()
        if not name:
            return

        # 检查该物种是否已存在于当前m/z的候选中
        already_exists = any(
            species.get('species') == name and species.get('mz') == self._current_mz
            for species in self._unified_species_data
        )

        if not already_exists:
            # 添加新的普通候选物种（默认不锁定）
            new_species = {
                'id': self._generate_new_id(),
                'species': name,
                'mz': self._current_mz,
                'ionization_energy': 0.0,
                'source': 'manual',
                'is_locked': False,  # 新添加的候选默认为普通状态
                'is_enabled': True,
                'coefficient': 0.0,
            }
            self._unified_species_data.append(new_species)

            self._refresh_table_from_data()
            self.species_config_changed.emit()

        self.species_input.clear()

    def populate_unified_species_table(self, mz: int, filtered_db: list[dict], locked_species: list[str]):
        """
        填充统一的物种表格
        - 合并自动候选物种和锁定候选物种
        - 标记来源和锁定状态
        """
        self._current_mz = mz
        self._updating = True
        try:
            # 收集自动候选物种
            auto_candidates = [item for item in filtered_db if item.get("mz") == mz]

            # 合并锁定候选物种
            self._unified_species_data = self._merge_species_lists(
                auto_candidates, locked_species
            )

            # 刷新表格
            self._refresh_table_from_data()
        finally:
            self._updating = False

    def _merge_species_lists(self, auto_candidates: list[dict], locked_species: list[str]) -> list[dict]:
        """
        合并自动候选物种和锁定候选物种
        - 相同species且mz相同视为重复
        - 重复时合并为单一条目，标记为'automatic+manual'
        - is_locked标记：该物种是否被用户锁定
        """
        result = []
        seen_species = {}  # key: species_name

        # 先处理自动候选
        for auto_item in auto_candidates:
            species_name = auto_item.get('species')
            merged_item = {
                'id': auto_item.get('id', self._generate_new_id()),
                'species': species_name,
                'mz': auto_item.get('mz'),
                'ionization_energy': auto_item.get('ionization_energy', 0.0),
                'source': 'automatic',
                'is_locked': species_name in locked_species,  # 检查是否在锁定列表中
                'is_enabled': True,
                'coefficient': 0.0,
                'cross_sections': auto_item.get('cross_sections', np.array([])),
                'energies': auto_item.get('energies', np.array([])),
            }

            # 如果在锁定列表中，标记来源为混合
            if species_name in locked_species:
                merged_item['source'] = 'automatic+manual'

            result.append(merged_item)
            seen_species[species_name] = merged_item

        # 处理锁定列表中未在auto中的
        for locked_name in locked_species:
            if locked_name not in seen_species:
                # 新建手动物种条目
                manual_item = {
                    'id': self._generate_new_id(),
                    'species': locked_name,
                    'mz': self._current_mz,
                    'ionization_energy': 0.0,
                    'source': 'manual',
                    'is_locked': True,
                    'is_enabled': True,
                    'coefficient': 0.0,
                    'cross_sections': np.array([]),
                    'energies': np.array([]),
                }
                result.append(manual_item)
                seen_species[locked_name] = manual_item

        return result

    def _generate_new_id(self) -> int:
        """生成新的物种ID"""
        if not self._unified_species_data:
            return 1
        max_id = max((sp.get('id', 0) for sp in self._unified_species_data), default=0)
        return max_id + 1

    def _refresh_table_from_data(self):
        """从_unified_species_data刷新表格显示"""
        self._updating = True
        try:
            self.species_table.setRowCount(len(self._unified_species_data))

            # 更新header显示当前m/z
            if self._current_mz is not None:
                self.species_table.horizontalHeader().setSectionResizeMode(
                    0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents
                )
                self.species_table.setHorizontalHeaderLabels([
                    "启用", f"物种 (m/z={self._current_mz})", "系数", "操作"
                ])

            header = self.species_table.horizontalHeader()
            # 设置列宽（简化后的4列）
            self.species_table.setColumnWidth(0, 40)   # 启用 checkbox
            self.species_table.setColumnWidth(2, 120)  # 系数
            self.species_table.setColumnWidth(3, 50)   # 操作 (删除按钮)

            # 自适应列
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)  # 物种名称

            for row, species in enumerate(self._unified_species_data):
                self._populate_table_row(row, species)
        finally:
            self._updating = False

    def _populate_table_row(self, row: int, species: dict):
        """填充表格的一行（简化为4列）"""
        # Col 0: 启用 checkbox
        enable_widget = QtWidgets.QWidget()
        enable_layout = QtWidgets.QHBoxLayout(enable_widget)
        enable_layout.setContentsMargins(0, 0, 0, 0)
        enable_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        enable_chk = QtWidgets.QCheckBox()
        enable_chk.setChecked(species.get('is_enabled', True))
        enable_chk.stateChanged.connect(lambda state, r=row: self._on_row_changed(r))
        enable_layout.addWidget(enable_chk)
        self.species_table.setCellWidget(row, 0, enable_widget)

        # Col 1: 物种名称 + 切换按钮 (※)
        species_widget = QtWidgets.QWidget()
        species_layout = QtWidgets.QHBoxLayout(species_widget)
        species_layout.setContentsMargins(4, 0, 4, 0)
        species_layout.setSpacing(4)

        # 物种名称 (QLabel)
        species_name = species.get('species', '')
        name_label = QtWidgets.QLabel(species_name)
        name_label.setWordWrap(False)
        species_layout.addWidget(name_label, stretch=1)

        # 切换按钮 (※) - 锁定候选标记
        # 注意：新实现"锁定候选"（不被筛选移除，系数正常优化）
        # 详见 locked_candidates_refactor_plan.md
        toggle_btn = QtWidgets.QPushButton("※")
        toggle_btn.setFixedSize(24, 24)
        toggle_btn.setFlat(False)
        toggle_btn.setToolTip("点击切换锁定/普通状态")
        toggle_btn.clicked.connect(lambda checked=False, r=row: self._toggle_locked_status(r))

        # 根据锁定状态设置按钮样式
        # 向后兼容：读取新字段is_locked，如果不存在则尝试读取旧字段is_forced
        is_locked = species.get('is_locked', species.get('is_forced', False))
        if is_locked:
            # 锁定候选物种：琥珀色背景，视觉上突出
            toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #fbbf24;
                    border: 1px solid #f59e0b;
                    border-radius: 3px;
                    font-weight: bold;
                    color: #92400e;
                }
                QPushButton:hover {
                    background-color: #fcd34d;
                }
            """)
        else:
            # 普通候选物种：灰色背景，低视觉强调
            toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e5e7eb;
                    border: 1px solid #d1d5db;
                    border-radius: 3px;
                }
                QPushButton:hover {
                    background-color: #f3f4f6;
                }
            """)

        species_layout.addWidget(toggle_btn)

        # 添加Tooltip：m/z、IE、来源、锁定状态
        ie = species.get('ionization_energy', 0.0)
        ie_text = f"{float(ie):.4f}" if ie else "N/A"
        source = species.get('source', '自动')
        locked = "是" if species.get('is_locked', False) else "否"
        tooltip_text = (
            f"物种: {species_name}\n"
            f"m/z: {species.get('mz', 'N/A')}\n"
            f"IE: {ie_text} eV\n"
            f"来源: {source}\n"
            f"锁定: {locked}"
        )
        name_label.setToolTip(tooltip_text)
        toggle_btn.setToolTip(tooltip_text + "\n\n点击切换锁定/普通状态")

        self.species_table.setCellWidget(row, 1, species_widget)

        # Col 2: 系数（QDoubleSpinBox）
        coeff_spin = QtWidgets.QDoubleSpinBox()
        coeff_spin.setRange(0, 1e6)
        coeff_spin.setDecimals(6)
        coeff_spin.setValue(species.get('coefficient', 0.0))
        mode = self.coefficient_mode_combo.currentData()
        coeff_spin.setEnabled(mode != "auto")
        coeff_spin.setFixedHeight(24)
        coeff_spin.setContentsMargins(0, 0, 0, 0)
        coeff_spin.valueChanged.connect(lambda val, r=row: self._on_row_changed(r))
        self.species_table.setCellWidget(row, 2, coeff_spin)

        # Col 3: 操作 (删除按钮)
        remove_widget = QtWidgets.QWidget()
        remove_layout = QtWidgets.QHBoxLayout(remove_widget)
        remove_layout.setContentsMargins(0, 0, 0, 0)
        remove_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        remove_btn = QtWidgets.QPushButton("✕")
        remove_btn.setFixedSize(24, 24)
        remove_btn.setToolTip("移除物种")
        remove_btn.clicked.connect(lambda checked=False, r=row: self._remove_row(r))
        remove_layout.addWidget(remove_btn)
        self.species_table.setCellWidget(row, 3, remove_widget)

    def _toggle_locked_status(self, row: int):
        """切换物种的锁定状态"""
        if 0 <= row < len(self._unified_species_data):
            species = self._unified_species_data[row]
            # 向后兼容：迁移旧字段到新字段
            if 'is_forced' in species and 'is_locked' not in species:
                species['is_locked'] = species.pop('is_forced')

            species['is_locked'] = not species.get('is_locked', False)
            species_name = species.get('species')

            # 更新_locked_species列表
            if species['is_locked'] and species_name not in self._locked_species:
                self._locked_species.append(species_name)
                self.locked_candidate_added.emit(species_name)
            elif not species['is_locked'] and species_name in self._locked_species:
                self._locked_species.remove(species_name)
                self.locked_candidate_removed.emit(species_name)

            # 刷新按钮样式 (Col 1 的切换按钮)
            species_widget = self.species_table.cellWidget(row, 1)
            if species_widget:
                toggle_btn = species_widget.findChild(QtWidgets.QPushButton)
                if toggle_btn:
                    is_locked = species.get('is_locked', False)
                    if is_locked:
                        toggle_btn.setStyleSheet("""
                            QPushButton {
                                background-color: #fbbf24;
                                border: 1px solid #f59e0b;
                                border-radius: 3px;
                                font-weight: bold;
                                color: #92400e;
                            }
                            QPushButton:hover {
                                background-color: #fcd34d;
                            }
                        """)
                    else:
                        toggle_btn.setStyleSheet("""
                            QPushButton {
                                background-color: #e5e7eb;
                                border: 1px solid #d1d5db;
                                border-radius: 3px;
                            }
                            QPushButton:hover {
                                background-color: #f3f4f6;
                            }
                        """)

            self.species_config_changed.emit()
            self.species_forced_toggled.emit(species_name)

    def _remove_row(self, row: int):
        """从表格移除物种"""
        if 0 <= row < len(self._unified_species_data):
            species = self._unified_species_data.pop(row)
            species_name = species.get('species')

            # 从锁定列表中移除
            if species_name in self._locked_species:
                self._locked_species.remove(species_name)
                self.locked_candidate_removed.emit(species_name)

            self._refresh_table_from_data()
            self.species_config_changed.emit()

    def _on_row_changed(self, row: int):
        """表格行改变时更新数据"""
        if self._updating or row < 0 or row >= self.species_table.rowCount():
            return

        species = self._unified_species_data[row]

        # 更新启用状态
        enable_widget = self.species_table.cellWidget(row, 0)
        if enable_widget:
            enable_chk = enable_widget.findChild(QtWidgets.QCheckBox)
            if enable_chk:
                species['is_enabled'] = enable_chk.isChecked()

        # 更新系数 (Col 2)
        coeff_widget = self.species_table.cellWidget(row, 2)
        if isinstance(coeff_widget, QtWidgets.QDoubleSpinBox):
            species['coefficient'] = coeff_widget.value()

        self.species_config_changed.emit()

    def _on_coefficient_mode_changed(self, index: int):
        """系数模式改变时更新表格控件状态"""
        mode = self.coefficient_mode_combo.currentData()

        for row in range(self.species_table.rowCount()):
            # 更新系数输入框启用状态 (Col 2)
            coeff_widget = self.species_table.cellWidget(row, 2)
            if isinstance(coeff_widget, QtWidgets.QDoubleSpinBox):
                coeff_widget.setEnabled(mode != "auto")

        self.coefficient_mode_changed.emit(mode)

    def _set_all_rows_checked(self, checked: bool):
        """全选或清空所有物种的启用状态"""
        self._updating = True
        try:
            for row in range(self.species_table.rowCount()):
                enable_widget = self.species_table.cellWidget(row, 0)
                if enable_widget:
                    enable_chk = enable_widget.findChild(QtWidgets.QCheckBox)
                    if enable_chk:
                        enable_chk.setChecked(checked)
                        self._unified_species_data[row]['is_enabled'] = checked
        finally:
            self._updating = False
        self.species_config_changed.emit()

    def _zero_all_coefficients(self):
        """将所有系数清零"""
        self._updating = True
        try:
            for row in range(self.species_table.rowCount()):
                coeff_widget = self.species_table.cellWidget(row, 2)  # Col 2: 系数
                if isinstance(coeff_widget, QtWidgets.QDoubleSpinBox):
                    coeff_widget.setValue(0.0)
                    self._unified_species_data[row]['coefficient'] = 0.0
        finally:
            self._updating = False
        self.candidates_zeroed.emit()

    def import_coefficients(self, species_list: list[dict]):
        """接收导入的系数并更新表格"""
        if not species_list:
            return
        self._updating = True
        try:
            coeff_by_name = {sp.get("species"): float(sp.get("coefficient", 0)) for sp in species_list}
            for row in range(self.species_table.rowCount()):
                # 获取物种名 (从 Col 1 的 widget 中)
                species_widget = self.species_table.cellWidget(row, 1)
                if species_widget:
                    name_label = species_widget.findChild(QtWidgets.QLabel)
                    if name_label and name_label.text() in coeff_by_name:
                        coeff_spin = self.species_table.cellWidget(row, 2)  # Col 2: 系数
                        if isinstance(coeff_spin, QtWidgets.QDoubleSpinBox):
                            coeff_spin.setValue(coeff_by_name[name_label.text()])
                            self._unified_species_data[row]['coefficient'] = coeff_by_name[name_label.text()]
        finally:
            self._updating = False

    # ---- 新方法：锁定候选管理 ----

    def get_locked_species(self) -> list[str]:
        """获取锁定候选物种列表（主要方法）"""
        return list(self._locked_species)

    def set_locked_species(self, species: list[str]):
        """设置锁定候选物种列表（主要方法）"""
        self._locked_species = list(species)

    def lock_candidate(self, species_name: str):
        """锁定单个候选物种（主要方法）

        被锁定的物种将不被自动筛选移除，其系数由优化器自由决定
        """
        if species_name not in self._locked_species:
            self._locked_species.append(species_name)
            self.locked_candidate_added.emit(species_name)
            # 触发表格更新
            self._refresh_table_from_data()
            self.species_config_changed.emit()

    def unlock_candidate(self, species_name: str):
        """解锁单个候选物种（主要方法）"""
        if species_name in self._locked_species:
            self._locked_species.remove(species_name)
            self.locked_candidate_removed.emit(species_name)
            self._refresh_table_from_data()
            self.species_config_changed.emit()

    # ---- 向后兼容方法 ----

    def get_force_species(self) -> list[str]:
        """获取锁定候选物种列表（向后兼容，推荐使用get_locked_species）"""
        return self.get_locked_species()

    def set_force_species(self, species: list[str]):
        """设置锁定候选物种列表（向后兼容，推荐使用set_locked_species）"""
        self.set_locked_species(species)

    def add_force_species(self, species_name: str):
        """锁定单个候选物种（向后兼容，推荐使用lock_candidate）"""
        self.lock_candidate(species_name)

    def get_candidates(self) -> list[dict]:
        """获取当前候选物种列表（向后兼容）"""
        return list(self._unified_species_data)

    def set_candidates(self, candidates: list[dict]):
        """设置候选物种列表（向后兼容）"""
        self._unified_species_data = list(candidates)

    def get_coefficient_mode(self) -> str:
        """获取当前系数模式"""
        return self.coefficient_mode_combo.currentData()

    def _get_selected_candidate_ids(self) -> list[int]:
        """获取选中的候选物种ID列表"""
        selected_ids = []
        for row in range(self.species_table.rowCount()):
            enable_widget = self.species_table.cellWidget(row, 0)
            enable_chk = enable_widget.findChild(QtWidgets.QCheckBox) if enable_widget else None
            if enable_chk and enable_chk.isChecked() and row < len(self._unified_species_data):
                species = self._unified_species_data[row]
                species_id = int(species.get("id", row + 1))
                selected_ids.append(species_id)
        return selected_ids

    def clear_ui(self):
        """清空UI内容"""
        self._locked_species = []
        self._unified_species_data = []
        self.species_input.clear()
        self.species_table.setRowCount(0)

    # ---- 向后兼容属性 ----

    @property
    def _force_species(self) -> list[str]:
        """向后兼容：返回_locked_species（只读属性别名）"""
        return self._locked_species

    @property
    def _candidate_data(self):
        """向后兼容：返回_unified_species_data"""
        return self._unified_species_data

    @_candidate_data.setter
    def _candidate_data(self, value):
        """向后兼容：设置_unified_species_data"""
        self._unified_species_data = value

    @property
    def candidate_table(self):
        """向后兼容：返回species_table"""
        return self.species_table

    @property
    def _candidate_updating(self):
        """向后兼容：返回_updating"""
        return self._updating

    @_candidate_updating.setter
    def _candidate_updating(self, value):
        """向后兼容：设置_updating"""
        self._updating = value
