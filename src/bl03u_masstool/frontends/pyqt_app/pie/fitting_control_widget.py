"""
FittingControlWidget - 拟合配置面板

职责：
- 统一的拟合物种配置界面
- 显示当前 m/z 的候选物种列表（来源：PICS 数据库）
- 支持物种启用/禁用、锁定/解锁、系数配置
- 所有拟合前的参数设置

设计原则：
- 候选物种 100% 由当前 m/z 对应的 PICS 数据库查询结果提供
- 不支持在此界面临时添加物种（缺少 PICS 数据会导致拟合条件不完整）
- 若某 m/z 无候选，显示空状态并引导用户前往 PICS 导入
- 仅负责UI展示和用户输入收集
- 通过信号向上报告配置变更
- 通过方法接收数据更新（不通过信号）
- 单一界面，无Tab切换

锁定候选语义：
- 被锁定的物种在自动筛选中不被移除，始终进入拟合候选集合
- 其系数由优化器自由决定，可为0或接近0
- "锁定"只表示用户要求算法考虑该物种，不表示已鉴别或必须参与
- PICS 数据完整性只决定物种是否具备拟合条件，不代表物种已被确认存在
- 物种鉴别由实验曲线与拟合结果支持
- 详见 locked_candidates_refactor_plan.md
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
    species_config_changed = QtCore.pyqtSignal()             # 物种配置整体改变
    candidate_selection_changed = QtCore.pyqtSignal(list)    # 选中的候选物种ID列表
    fit_readiness_changed = QtCore.pyqtSignal(bool, str)      # 是否可拟合 + 不可拟合原因

    # ---- 控制变更信号 ----
    candidates_zeroed = QtCore.pyqtSignal()                  # 清零按钮点击
    ie_query_requested = QtCore.pyqtSignal()                 # 查询/重试当前候选的缺失 IE
    pics_import_requested = QtCore.pyqtSignal()              # PICS 导入按钮点击（空状态）
    species_forced_toggled = QtCore.pyqtSignal(str)          # 向后兼容：物种状态切换（改用candidate_lock_toggled）

    # ---- 结果操作信号 ----
    confirmation_requested = QtCore.pyqtSignal()             # 确认鉴定按钮点击
    export_requested = QtCore.pyqtSignal()                   # 导出鉴定结果按钮点击

    def __init__(self, parent=None):
        super().__init__(parent)

        # ---- 大小策略和约束 ----
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum
        )
        self.setMinimumHeight(100)
        self.setMaximumHeight(16777215)

        # ---- 面板样式钩子（具体样式由应用级 QSS 统一管理） ----
        self.setObjectName("FittingConfigPanel")

        # ---- 内部状态 ----
        self._locked_species: list[str] = []                   # 主要使用：锁定候选物种名称列表
        self._unified_species_data: list[dict] = []            # 统一的物种数据
        self._current_mz: int | None = None                    # 当前m/z
        self._updating = False                                 # 防止递归更新
        self._candidates_loaded = False                        # 是否已执行过 PICS 查询

        # ---- UI构建 ----
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Header: "拟合配置"
        header_frame = QtWidgets.QFrame()
        header_frame.setObjectName("PanelHeader")
        header_frame.setFixedHeight(32)
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(8, 4, 8, 4)
        header.setSpacing(0)
        title = QtWidgets.QLabel("拟合配置")
        title.setObjectName("ReadoutLabel")
        header.addWidget(title)
        header.addStretch()
        self.current_mz_label = QtWidgets.QLabel("未选择 m/z")
        self.current_mz_label.setObjectName("PieMzBadge")
        header.addWidget(self.current_mz_label)
        main_layout.addWidget(header_frame)

        # ---- 空状态提示（显示初始或查询为空状态） ----
        self.empty_state_widget = QtWidgets.QWidget()
        empty_layout = QtWidgets.QVBoxLayout(self.empty_state_widget)
        empty_layout.setContentsMargins(16, 16, 16, 16)
        empty_layout.setSpacing(12)

        empty_icon = QtWidgets.QLabel("[i]")
        empty_icon.setObjectName("PieEmptyIcon")
        empty_icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_icon)
        self.empty_icon = empty_icon  # 保存引用以便更新

        empty_title = QtWidgets.QLabel("拟合配置")
        empty_title.setObjectName("PieEmptyTitle")
        empty_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(empty_title)
        self.empty_title = empty_title  # 保存引用以便更新

        empty_text = QtWidgets.QLabel()
        empty_text.setObjectName("ProjectHint")
        empty_text.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_text.setWordWrap(True)
        empty_layout.addWidget(empty_text)
        self.empty_text = empty_text  # 保存引用以便更新

        empty_layout.addSpacing(8)

        self.goto_pics_btn = QtWidgets.QPushButton("前往 PICS 导入")
        self.goto_pics_btn.setObjectName("BrowseButton")
        self.goto_pics_btn.setFixedHeight(32)
        self.goto_pics_btn.clicked.connect(lambda: self.pics_import_requested.emit())
        empty_layout.addWidget(self.goto_pics_btn, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_layout.addStretch()

        main_layout.addWidget(self.empty_state_widget, stretch=1)

        candidate_summary_row = QtWidgets.QHBoxLayout()
        candidate_summary_row.setSpacing(6)
        self.candidate_summary_label = QtWidgets.QLabel("0 个候选 · 0 个启用")
        self.candidate_summary_label.setObjectName("CandidateSummary")
        candidate_summary_row.addWidget(self.candidate_summary_label, stretch=1)
        self.fit_mode_label = QtWidgets.QLabel("自动 NNLS")
        self.fit_mode_label.setObjectName("FitModeBadge")
        candidate_summary_row.addWidget(self.fit_mode_label)
        main_layout.addLayout(candidate_summary_row)

        self.lock_hint_label = QtWidgets.QLabel("※ 锁定：自动筛选时保留该候选，系数仍由拟合决定")
        self.lock_hint_label.setObjectName("HintLabel")
        self.lock_hint_label.setWordWrap(True)
        main_layout.addWidget(self.lock_hint_label)

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

        self.zero_coeff_btn = QtWidgets.QPushButton("系数清零")
        self.zero_coeff_btn.setToolTip("将所有系数值清零")
        self.zero_coeff_btn.clicked.connect(self._zero_all_coefficients)
        self.zero_coeff_btn.setObjectName("BrowseButton")
        control_row2.addWidget(self.zero_coeff_btn)

        self.ie_query_btn = QtWidgets.QPushButton("查询缺失 IE")
        self.ie_query_btn.setToolTip("通过本地物种库 / NIST WebBook 查询当前候选中缺失的电离能")
        self.ie_query_btn.clicked.connect(self.ie_query_requested.emit)
        self.ie_query_btn.setObjectName("BrowseButton")
        control_row2.addWidget(self.ie_query_btn)

        control_row2.addStretch()
        main_layout.addLayout(control_row2)

        # ---- 下方：物种表格（启用 | 物种(※) | IE | 系数 | 操作） ----
        self.species_table = QtWidgets.QTableWidget()
        self.species_table.setColumnCount(5)
        self.species_table.setHorizontalHeaderLabels([
            "启用", "物种", "IE (eV)", "系数", "操作"
        ])
        self.species_table.setWordWrap(False)
        self.species_table.setAlternatingRowColors(True)
        self.species_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.species_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.species_table.verticalHeader().setVisible(False)
        self.species_table.verticalHeader().setDefaultSectionSize(30)
        self.species_table.horizontalHeader().setStretchLastSection(False)
        main_layout.addWidget(self.species_table, stretch=1)

        # ---- 结果摘要区（拟合后显示）----
        # 分隔线
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setStyleSheet("color: #e2e8f0;")
        main_layout.addWidget(separator)

        self.result_section = QtWidgets.QWidget()
        self.result_section.setVisible(False)
        result_layout = QtWidgets.QVBoxLayout(self.result_section)
        result_layout.setContentsMargins(0, 4, 0, 4)
        result_layout.setSpacing(4)

        # 结果标题行：标签 + R² + 状态标签
        result_header = QtWidgets.QHBoxLayout()
        result_header.setSpacing(6)
        result_title = QtWidgets.QLabel("拟合结果")
        result_title.setObjectName("ReadoutLabel")
        result_header.addWidget(result_title)

        self.result_r2_label = QtWidgets.QLabel("")
        self.result_r2_label.setObjectName("ReadoutValue")
        result_header.addWidget(self.result_r2_label)

        result_header.addStretch()

        self.result_status_label = QtWidgets.QLabel("")
        self.result_status_label.setFixedWidth(70)
        self.result_status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        result_header.addWidget(self.result_status_label)
        result_layout.addLayout(result_header)

        # 物种贡献摘要表格（4列：物种名、IE、贡献%、系数）
        self.fit_result_table = QtWidgets.QTableWidget()
        self.fit_result_table.setColumnCount(4)
        self.fit_result_table.setHorizontalHeaderLabels(["物种", "IE(eV)", "贡献%", "系数"])
        self.fit_result_table.setWordWrap(False)
        self.fit_result_table.setAlternatingRowColors(True)
        self.fit_result_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.fit_result_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.fit_result_table.setMaximumHeight(160)
        self.fit_result_table.setMinimumHeight(60)
        self.fit_result_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.fit_result_table.setColumnWidth(1, 72)
        self.fit_result_table.setColumnWidth(2, 60)
        self.fit_result_table.setColumnWidth(3, 80)
        result_layout.addWidget(self.fit_result_table)

        # 操作按钮行
        action_row = QtWidgets.QHBoxLayout()
        action_row.setSpacing(6)

        self.confirm_btn = QtWidgets.QPushButton("✓ 确认鉴定")
        self.confirm_btn.setObjectName("WorkflowButton")
        self.confirm_btn.setToolTip("将当前拟合结果确认为该 m/z 的鉴定结论")
        self.confirm_btn.setFixedHeight(28)
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self.confirmation_requested.emit)
        action_row.addWidget(self.confirm_btn)

        self.export_result_btn = QtWidgets.QPushButton("导出鉴定结果")
        self.export_result_btn.setObjectName("ExportButton")
        self.export_result_btn.setToolTip("导出所有已拟合的 m/z 鉴定结果到 Excel")
        self.export_result_btn.setFixedHeight(28)
        self.export_result_btn.setEnabled(False)
        self.export_result_btn.clicked.connect(self.export_requested.emit)
        action_row.addWidget(self.export_result_btn)

        result_layout.addLayout(action_row)
        main_layout.addWidget(self.result_section)

        # 初始化空状态显示
        self._update_ui_state()

    # ---- 物种数据管理方法 ----

    @staticmethod
    def _coerce_ie(species: dict) -> float | None:
        value = species.get("ie")
        if value is None:
            value = species.get("ionization_energy")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if np.isfinite(numeric) and numeric > 0 else None

    @classmethod
    def _ie_display_text(cls, species: dict) -> str:
        value = cls._coerce_ie(species)
        if value is not None:
            return f"{value:.4f}"
        status = str(species.get("ie_query_status", "not_queried"))
        return {
            "pending": "查询中…",
            "not_found": "未查到",
            "failed": "查询失败",
        }.get(status, "待查询")

    def populate_unified_species_table(self, mz: int, filtered_db: list[dict], locked_species: list[str]):
        """
        填充统一的物种表格
        - 显示当前 m/z 的 PICS 数据库候选物种
        - 标记来源和锁定状态
        - 若无候选物种，显示空状态提示

        此方法在 PIE 曲线生成且 m/z 确定后被调用
        标记候选查询已执行（_candidates_loaded = True）
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

            # 标记查询已执行
            self._candidates_loaded = True

            # 刷新表格并管理空状态
            self._update_ui_state()
        finally:
            self._updating = False
        self._refresh_candidate_summary()

    def _update_ui_state(self):
        """
        根据候选物种和查询状态来显示或隐藏空状态提示

        状态1: 前置数据未准备（初始或清空）
          - _candidates_loaded = False
          - 显示空状态："请先生成 PIE 曲线"（[i]）

        状态2: 查询已执行但无结果
          - _candidates_loaded = True，_unified_species_data 为空
          - 显示空状态："当前 m/z 未找到候选物种"（[!] + PICS 导入按钮）

        状态3: 有候选物种
          - _unified_species_data 不为空
          - 显示表格，隐藏空状态
        """
        has_candidates = len(self._unified_species_data) > 0

        if has_candidates:
            # 有候选物种 → 显示表格
            self.empty_state_widget.hide()
            self.species_table.show()
            self._refresh_table_from_data()
        elif self._candidates_loaded:
            # 查询已执行但无结果 → 显示"未找到候选物种"空状态
            self.species_table.setRowCount(0)
            self.empty_icon.setText("[!]")
            self.empty_title.setText("当前 m/z 未找到候选物种")
            self.empty_text.setText(
                "PICS 数据库中尚未收录该质荷比的物种或缺少截面数据。\n"
                "请先前往 PICS 导入添加对应物种及完整的截面数据。"
            )
            self.goto_pics_btn.show()
            self.empty_state_widget.show()
            self.species_table.hide()
        else:
            # 前置数据未准备 → 显示"请先生成 PIE 曲线"空状态
            self.species_table.setRowCount(0)
            self.empty_icon.setText("[i]")
            self.empty_title.setText("等待 PIE 曲线")
            self.empty_text.setText(
                "请先在上方选择数据源并生成 PIE 曲线。\n"
                "随后在左侧选择 m/z，对应候选物种将自动显示在此。"
            )
            self.goto_pics_btn.hide()
            self.empty_state_widget.show()
            self.species_table.hide()
        self.current_mz_label.setText(
            f"m/z {self._current_mz}" if self._current_mz is not None else "未选择 m/z"
        )
        self._refresh_candidate_summary()
        self._refresh_ie_query_button()

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
            ionization_energy = self._coerce_ie(auto_item)
            merged_item = dict(auto_item)
            merged_item.update({
                'id': auto_item.get('id', self._generate_new_id()),
                'species': species_name,
                'mz': auto_item.get('mz'),
                'ie': ionization_energy,
                'ionization_energy': ionization_energy,
                'ie_source': auto_item.get('ie_source', 'PICS数据库' if ionization_energy is not None else ''),
                'ie_query_status': auto_item.get(
                    'ie_query_status', 'available' if ionization_energy is not None else 'not_queried'
                ),
                'source': 'automatic',
                'is_locked': species_name in locked_species,  # 检查是否在锁定列表中
                'is_enabled': True,
                'coefficient': 0.0,
                'cross_sections': auto_item.get('cross_sections', np.array([])),
                'energies': auto_item.get('energies', np.array([])),
            })

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
                    'ie': None,
                    'ionization_energy': None,
                    'ie_source': '',
                    'ie_query_status': 'not_queried',
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
                    "启用", f"物种 (m/z={self._current_mz})", "IE (eV)", "系数", "操作"
                ])

            header = self.species_table.horizontalHeader()
            # 设置列宽
            self.species_table.setColumnWidth(0, 40)   # 启用 checkbox
            self.species_table.setColumnWidth(2, 78)   # IE
            self.species_table.setColumnWidth(3, 105)  # 系数
            self.species_table.setColumnWidth(4, 42)   # 操作 (删除按钮)

            # 自适应列
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)  # 物种名称

            for row, species in enumerate(self._unified_species_data):
                self._populate_table_row(row, species)
        finally:
            self._updating = False

    def _populate_table_row(self, row: int, species: dict):
        """填充表格的一行。"""
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

        # 根据锁定状态设置动态属性，具体视觉由应用级 QSS 管理。
        # 向后兼容：读取新字段is_locked，如果不存在则尝试读取旧字段is_forced
        is_locked = species.get('is_locked', species.get('is_forced', False))
        toggle_btn.setObjectName("LockCandidateButton")
        toggle_btn.setProperty("locked", is_locked)

        species_layout.addWidget(toggle_btn)

        # 添加Tooltip：m/z、IE、来源、锁定状态
        ie_text = self._ie_display_text(species)
        ie_source = species.get('ie_source') or "未提供"
        ie_message = species.get('ie_message') or ""
        source = species.get('source', '自动')
        locked = "是" if species.get('is_locked', False) else "否"
        tooltip_text = (
            f"物种: {species_name}\n"
            f"m/z: {species.get('mz', 'N/A')}\n"
            f"IE: {ie_text}{' eV' if self._coerce_ie(species) is not None else ''}\n"
            f"IE来源: {ie_source}\n"
            + (f"IE备注: {ie_message}\n" if ie_message else "") +
            f"来源: {source}\n"
            f"锁定: {locked}"
        )
        name_label.setToolTip(tooltip_text)
        toggle_btn.setToolTip(tooltip_text + "\n\n点击切换锁定/普通状态")

        self.species_table.setCellWidget(row, 1, species_widget)

        # Col 2: IE（只读文本，查询状态也在此显示）
        ie_item = QtWidgets.QTableWidgetItem(ie_text)
        ie_item.setFlags(ie_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        ie_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ie_item.setToolTip(tooltip_text)
        self.species_table.setItem(row, 2, ie_item)

        # Col 3: 系数（QDoubleSpinBox）
        coeff_spin = QtWidgets.QDoubleSpinBox()
        coeff_spin.setRange(0, 1e6)
        coeff_spin.setDecimals(6)
        coeff_spin.setValue(species.get('coefficient', 0.0))
        coeff_spin.setToolTip("保持 0 时由 NNLS 自动求解；输入正值后按手动系数拟合")
        coeff_spin.setFixedHeight(24)
        coeff_spin.setContentsMargins(0, 0, 0, 0)
        coeff_spin.valueChanged.connect(lambda val, r=row: self._on_row_changed(r))
        self.species_table.setCellWidget(row, 3, coeff_spin)

        # Col 4: 操作 (删除按钮)
        remove_widget = QtWidgets.QWidget()
        remove_layout = QtWidgets.QHBoxLayout(remove_widget)
        remove_layout.setContentsMargins(0, 0, 0, 0)
        remove_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        remove_btn = QtWidgets.QPushButton("✕")
        remove_btn.setObjectName("IconButton")
        remove_btn.setFixedSize(24, 24)
        remove_btn.setToolTip("移除物种")
        remove_btn.clicked.connect(lambda checked=False, r=row: self._remove_row(r))
        remove_layout.addWidget(remove_btn)
        self.species_table.setCellWidget(row, 4, remove_widget)

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

            # 刷新按钮动态属性 (Col 1 的切换按钮)
            species_widget = self.species_table.cellWidget(row, 1)
            if species_widget:
                toggle_btn = species_widget.findChild(QtWidgets.QPushButton)
                if toggle_btn:
                    is_locked = species.get('is_locked', False)
                    toggle_btn.setProperty("locked", is_locked)
                    toggle_btn.style().unpolish(toggle_btn)
                    toggle_btn.style().polish(toggle_btn)

            self._refresh_candidate_summary()
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
            self._emit_selection_state()
            self._refresh_ie_query_button()
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

        # 更新系数 (Col 3)
        coeff_widget = self.species_table.cellWidget(row, 3)
        if isinstance(coeff_widget, QtWidgets.QDoubleSpinBox):
            species['coefficient'] = coeff_widget.value()

        self._emit_selection_state()
        self.species_config_changed.emit()

    def _set_all_rows_checked(self, checked: bool):
        """全选或清空所有物种的启用状态"""
        if not self._unified_species_data:
            return
        self._updating = True
        try:
            for row in range(min(self.species_table.rowCount(), len(self._unified_species_data))):
                enable_widget = self.species_table.cellWidget(row, 0)
                if enable_widget:
                    enable_chk = enable_widget.findChild(QtWidgets.QCheckBox)
                    if enable_chk:
                        enable_chk.setChecked(checked)
                        self._unified_species_data[row]['is_enabled'] = checked
        finally:
            self._updating = False
        self._emit_selection_state()
        self.species_config_changed.emit()

    def _zero_all_coefficients(self):
        """将所有系数清零"""
        self._updating = True
        try:
            for row in range(self.species_table.rowCount()):
                coeff_widget = self.species_table.cellWidget(row, 3)  # Col 3: 系数
                if isinstance(coeff_widget, QtWidgets.QDoubleSpinBox):
                    coeff_widget.setValue(0.0)
                    self._unified_species_data[row]['coefficient'] = 0.0
        finally:
            self._updating = False
        self._refresh_candidate_summary()
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
                        coeff_spin = self.species_table.cellWidget(row, 3)  # Col 3: 系数
                        if isinstance(coeff_spin, QtWidgets.QDoubleSpinBox):
                            coeff_spin.setValue(coeff_by_name[name_label.text()])
                            self._unified_species_data[row]['coefficient'] = coeff_by_name[name_label.text()]
        finally:
            self._updating = False
        self._refresh_candidate_summary()

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

    def update_ionization_energy_for_ids(
        self,
        species_ids: set[int],
        *,
        value: float | None,
        status: str,
        source: str = "",
        message: str = "",
    ) -> None:
        """Update IE cells without rebuilding coefficient/check-box widgets."""
        for row, species in enumerate(self._unified_species_data):
            try:
                species_id = int(species.get("id", -1))
            except (TypeError, ValueError):
                continue
            if species_id not in species_ids:
                continue
            species["ie"] = value
            species["ionization_energy"] = value
            species["ie_query_status"] = status
            species["ie_source"] = source
            species["ie_message"] = message
            item = self.species_table.item(row, 2)
            tooltip = (
                f"物种: {species.get('species', '')}\n"
                f"IE来源: {source or '未提供'}\n"
                f"{message}"
            )
            if item is not None:
                item.setText(self._ie_display_text(species))
                item.setToolTip(tooltip)
            species_widget = self.species_table.cellWidget(row, 1)
            if species_widget is not None:
                children = species_widget.findChildren(QtWidgets.QLabel)
                children.extend(species_widget.findChildren(QtWidgets.QPushButton))
                for child in children:
                    child.setToolTip(tooltip)
        self._refresh_ie_query_button()

    def _refresh_ie_query_button(self) -> None:
        has_missing = any(self._coerce_ie(species) is None for species in self._unified_species_data)
        has_pending = any(
            species.get("ie_query_status") == "pending" for species in self._unified_species_data
        )
        self.ie_query_btn.setEnabled(bool(self._unified_species_data) and has_missing and not has_pending)

    def _refresh_candidate_summary(self) -> None:
        candidate_count = len(self._unified_species_data)
        selected_ids = self._get_selected_candidate_ids() if candidate_count else []
        selected_id_set = set(selected_ids)
        locked_count = sum(bool(species.get("is_locked")) for species in self._unified_species_data)
        manual_mode = False
        for species in self._unified_species_data:
            try:
                species_id = int(species.get("id", -1))
                coefficient = float(species.get("coefficient", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if species_id in selected_id_set and coefficient > 0.0:
                manual_mode = True
                break

        self.candidate_summary_label.setText(
            f"{candidate_count} 个候选 · {len(selected_ids)} 个启用 · {locked_count} 个锁定"
        )
        self.fit_mode_label.setText("手动系数" if manual_mode else "自动 NNLS")
        self.fit_mode_label.setProperty("mode", "manual" if manual_mode else "auto")
        self.fit_mode_label.style().unpolish(self.fit_mode_label)
        self.fit_mode_label.style().polish(self.fit_mode_label)

        has_candidates = candidate_count > 0
        self.select_all_btn.setEnabled(has_candidates and len(selected_ids) < candidate_count)
        self.clear_selection_btn.setEnabled(has_candidates and bool(selected_ids))
        self.zero_coeff_btn.setEnabled(has_candidates and manual_mode)
        ready, reason = self.get_fit_readiness()
        self.fit_readiness_changed.emit(ready, reason)

    def _emit_selection_state(self) -> None:
        selected_ids = self._get_selected_candidate_ids()
        self._refresh_candidate_summary()
        self.candidate_selection_changed.emit(selected_ids)

    def get_fit_readiness(self) -> tuple[bool, str]:
        """Return whether the current candidate configuration can be fitted."""
        if not self._candidates_loaded or self._current_mz is None:
            return False, "请先选择一条 m/z 曲线"
        if not self._unified_species_data:
            return False, "当前 m/z 没有可用的 PICS 候选"
        if not self._get_selected_candidate_ids():
            return False, "请至少启用一个候选物种"
        return True, ""

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
        """清空UI内容，重置为初始状态"""
        self._locked_species = []
        self._unified_species_data = []
        self._candidates_loaded = False  # 重置查询标志
        self.species_table.setRowCount(0)
        self.clear_fit_result()
        self._update_ui_state()

    # ---- 结果摘要区方法 ----

    def show_fit_result(self, fit_model: dict, status: str = "COMPLETED"):
        """
        填充并显示右侧结果摘要区

        Args:
            fit_model: 拟合结果字典（含 species, r_squared 等）
            status: "COMPLETED" | "OBSOLETE" | "CONFIRMED"
        """
        species_list = fit_model.get("species", [])
        r2 = fit_model.get("r_squared", 0.0)

        # 更新 R² 标签（带颜色）
        if r2 >= 0.8:
            color = "#166534"
        elif r2 >= 0.5:
            color = "#92400e"
        else:
            color = "#dc2626"
        self.result_r2_label.setText(
            f"<span style='color:{color}; font-weight:bold;'>R²={r2:.4f}</span>"
        )
        self.result_r2_label.setTextFormat(QtCore.Qt.TextFormat.RichText)

        # 填充物种贡献表格
        self.fit_result_table.setRowCount(len(species_list))
        total_contrib = sum(s.get("contribution_percent", 0.0) for s in species_list)
        if total_contrib <= 0:
            total_contrib = 1.0
        for row, sp in enumerate(species_list):
            name = str(sp.get("species", ""))
            ie = self._coerce_ie(sp)
            ie_text = f"{ie:.4f}" if ie is not None else {
                "pending": "查询中…",
                "failed": "查询失败",
            }.get(str(sp.get("ie_query_status", "")), "未查到")
            contrib = float(sp.get("contribution_percent", 0.0))
            coeff = float(sp.get("coefficient", 0.0))

            name_item = QtWidgets.QTableWidgetItem(name)
            ie_item = QtWidgets.QTableWidgetItem(ie_text)
            ie_item.setToolTip(
                f"来源: {sp.get('ie_source') or '未提供'}\n"
                f"{sp.get('ie_message') or ''}"
            )
            contrib_item = QtWidgets.QTableWidgetItem(f"{contrib:.1f}%")
            coeff_item = QtWidgets.QTableWidgetItem(f"{coeff:.4f}")

            ie_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            contrib_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            coeff_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            self.fit_result_table.setItem(row, 0, name_item)
            self.fit_result_table.setItem(row, 1, ie_item)
            self.fit_result_table.setItem(row, 2, contrib_item)
            self.fit_result_table.setItem(row, 3, coeff_item)

        self.fit_result_table.resizeRowsToContents()

        # 更新状态标签与结果操作可用性
        self.set_fit_result_status(status)

        # 显示结果区
        self.result_section.setVisible(True)

    def clear_fit_result(self):
        """隐藏结果摘要区并清空内容"""
        self.fit_result_table.setRowCount(0)
        self.result_r2_label.setText("")
        self.result_status_label.setText("")
        self.confirm_btn.setText("✓ 确认鉴定")
        self.confirm_btn.setEnabled(False)
        self.export_result_btn.setEnabled(False)
        self.result_section.setVisible(False)

    def set_fit_confirmed(self, confirmed: bool):
        """更新确认状态显示"""
        if confirmed:
            self.set_fit_result_status("CONFIRMED")
            self.confirm_btn.setText("↺ 重新确认")
        else:
            self.set_fit_result_status("COMPLETED")
            self.confirm_btn.setText("✓ 确认鉴定")

    def set_fit_result_status(self, status: str) -> None:
        """Update result validity and guard actions that require a committed fit."""
        self._set_result_status_label(status)
        committed = status in {"COMPLETED", "CONFIRMED"}
        self.confirm_btn.setEnabled(committed)
        if status == "CONFIRMED":
            self.confirm_btn.setText("↺ 重新确认")
        elif status == "COMPLETED":
            self.confirm_btn.setText("✓ 确认鉴定")
        self.export_result_btn.setEnabled(committed)
        if status in {"PREVIEW", "OBSOLETE"}:
            self.confirm_btn.setToolTip("参数已变化，请先重新拟合后再确认鉴定")
        else:
            self.confirm_btn.setToolTip("将当前拟合结果确认为该 m/z 的鉴定结论")

    def set_export_available(self, available: bool) -> None:
        """Synchronize all-results export with the dialog-level result state."""
        self.export_result_btn.setEnabled(bool(available))

    def _set_result_status_label(self, status: str):
        """设置结果状态标签文字和颜色"""
        config = {
            "COMPLETED": ("[已拟合]", "#1d4ed8"),
            "OBSOLETE":  ("[结果过期]", "#92400e"),
            "CONFIRMED": ("[✓ 已确认]", "#166534"),
            "PREVIEW":   ("[参数预览]", "#7c3aed"),
            "FAILED":    ("[失败]", "#dc2626"),
        }
        text, color = config.get(status, ("", "#374151"))
        self.result_status_label.setText(
            f"<span style='color:{color}; font-weight:bold;'>{text}</span>"
        )
        self.result_status_label.setTextFormat(QtCore.Qt.TextFormat.RichText)

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
