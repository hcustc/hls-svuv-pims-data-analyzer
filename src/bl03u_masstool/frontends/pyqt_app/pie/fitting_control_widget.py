"""
FittingControlWidget - 拟合配置面板

职责：
- 统一的拟合物种配置界面
- 显示当前 m/z 的候选物种列表（来源：PICS 数据库）
- 支持物种启用/禁用、锁定/解锁
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

from functools import lru_cache
from io import BytesIO

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from bl03u_masstool.frontends.pyqt_app.common.widgets import ElidedLabel, StateGlyph


@lru_cache(maxsize=256)
def _formula_from_smiles(smiles: str) -> str:
    """Return a molecular formula for a valid SMILES string."""
    normalized = str(smiles or "").strip()
    if not normalized:
        return ""
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import rdMolDescriptors

        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(normalized)
        return rdMolDescriptors.CalcMolFormula(molecule) if molecule is not None else ""
    except Exception:
        return ""


@lru_cache(maxsize=256)
def _structure_png(smiles: str, width: int = 320, height: int = 200) -> bytes | None:
    """Render a SMILES string to PNG bytes without writing a temporary file."""
    normalized = str(smiles or "").strip()
    if not normalized:
        return None
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import Draw

        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(normalized)
        if molecule is None:
            return None
        image = Draw.MolToImage(
            molecule,
            size=(int(width), int(height)),
            fitImage=True,
        )
        stream = BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue()
    except Exception:
        return None


class _SpeciesIdentityWidget(QtWidgets.QWidget):
    """Clickable identity block that also reports hover lifecycle."""

    activated = QtCore.pyqtSignal()
    hover_entered = QtCore.pyqtSignal(QtCore.QPoint)
    hover_left = QtCore.pyqtSignal()

    def enterEvent(self, event: QtGui.QEnterEvent) -> None:
        self.hover_entered.emit(QtGui.QCursor.pos())
        super().enterEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        self.hover_left.emit()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.activated.emit()
        super().mousePressEvent(event)


class _SpeciesHoverCard(QtWidgets.QFrame):
    """Non-activating floating card used for molecular structure previews."""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        flags = (
            QtCore.Qt.WindowType.ToolTip
            | QtCore.Qt.WindowType.FramelessWindowHint
        )
        super().__init__(parent, flags)
        self.setObjectName("SpeciesHoverCard")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground)
        self.setMinimumWidth(370)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self.structure_label = QtWidgets.QLabel("暂无结构数据")
        self.structure_label.setObjectName("HoverStructureCanvas")
        self.structure_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.structure_label.setFixedSize(150, 104)
        self.structure_label.setWordWrap(True)
        layout.addWidget(self.structure_label)

        details = QtWidgets.QVBoxLayout()
        details.setContentsMargins(0, 0, 0, 0)
        details.setSpacing(4)

        kicker = QtWidgets.QLabel("物种结构")
        kicker.setObjectName("SpeciesHoverKicker")
        details.addWidget(kicker)

        self.name_label = QtWidgets.QLabel()
        self.name_label.setObjectName("SpeciesHoverName")
        self.name_label.setWordWrap(True)
        details.addWidget(self.name_label)

        self.formula_label = QtWidgets.QLabel()
        self.formula_label.setObjectName("SpeciesHoverFormulaBadge")
        details.addWidget(self.formula_label)

        self.ie_label = QtWidgets.QLabel()
        self.ie_label.setObjectName("SpeciesHoverMeta")
        details.addWidget(self.ie_label)

        self.hint_label = QtWidgets.QLabel()
        self.hint_label.setObjectName("SpeciesHoverHint")
        self.hint_label.setWordWrap(True)
        details.addWidget(self.hint_label)
        details.addStretch()
        layout.addLayout(details, stretch=1)

    def show_species(
        self,
        species: dict,
        *,
        formula: str,
        ie_text: str,
        anchor: QtCore.QPoint,
    ) -> None:
        name = str(
            species.get("species") or species.get("name") or "未命名物种"
        ).strip()
        smiles = str(species.get("smiles") or "").strip()
        image_bytes = _structure_png(smiles) if smiles else None

        self.name_label.setText(name)
        self.formula_label.setText(f"分子式：{formula or '未提供'}")
        self.ie_label.setText(
            f"IE：{ie_text}{' eV' if ie_text and ie_text[0].isdigit() else ''}"
        )
        self.structure_label.setPixmap(QtGui.QPixmap())
        if image_bytes:
            pixmap = QtGui.QPixmap()
            if pixmap.loadFromData(image_bytes, "PNG"):
                self.structure_label.setText("")
                self.structure_label.setPixmap(
                    pixmap.scaled(
                        self.structure_label.size() - QtCore.QSize(8, 8),
                        QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                        QtCore.Qt.TransformationMode.SmoothTransformation,
                    )
                )
                self.hint_label.setText("二维结构来自数据库 SMILES")
            else:
                self.structure_label.setText("结构图加载失败")
                self.hint_label.setText("无法加载结构图，请检查物种元数据")
        elif smiles:
            self.structure_label.setText("SMILES\n无法解析")
            self.hint_label.setText("SMILES 无法解析，请检查物种元数据")
        else:
            self.structure_label.setText(
                f"{formula}\n暂无二维结构" if formula else "暂无结构数据"
            )
            self.hint_label.setText(
                "数据库未提供 SMILES，当前仅显示分子式"
                if formula
                else "数据库未提供分子式或 SMILES"
            )

        self.setToolTip(
            f"物种: {name}\n分子式: {formula or '未提供'}\n"
            f"SMILES: {smiles or '未提供'}"
        )
        self.adjustSize()
        screen = QtGui.QGuiApplication.screenAt(anchor)
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QtCore.QRect()
        position = anchor + QtCore.QPoint(14, 18)
        if available.isValid():
            if position.x() + self.width() > available.right():
                position.setX(anchor.x() - self.width() - 14)
            if position.y() + self.height() > available.bottom():
                position.setY(anchor.y() - self.height() - 12)
            position.setX(max(available.left(), position.x()))
            position.setY(max(available.top(), position.y()))
        self.move(position)
        self.show()
        self.raise_()


class FittingControlWidget(QtWidgets.QWidget):
    """拟合配置面板 - 统一的拟合物种配置"""

    # ---- 信号定义 ----
    # 新信号：锁定候选相关（主要使用）
    locked_candidate_added = QtCore.pyqtSignal(str)          # 添加单个锁定候选
    locked_candidate_removed = QtCore.pyqtSignal(str)        # 移除单个锁定候选
    candidate_lock_toggled = QtCore.pyqtSignal(str)          # 物种锁定状态切换

    # 统一的配置变更信号
    species_config_changed = QtCore.pyqtSignal()             # 物种配置整体改变
    candidate_selection_changed = QtCore.pyqtSignal(list)    # 选中的候选物种ID列表
    fit_readiness_changed = QtCore.pyqtSignal(bool, str)      # 是否可拟合 + 不可拟合原因

    # ---- 控制变更信号 ----
    ie_query_requested = QtCore.pyqtSignal()                 # 查询/重试当前候选的缺失 IE
    pics_import_requested = QtCore.pyqtSignal()              # PICS 导入按钮点击（空状态）

    # ---- 结果操作信号 ----
    confirmation_requested = QtCore.pyqtSignal()             # 确认鉴定按钮点击
    export_requested = QtCore.pyqtSignal()                   # 导出鉴定结果按钮点击
    result_details_requested = QtCore.pyqtSignal()            # 查看拟合明细

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
        self._current_mz_display: str | None = None            # 精确峰曲线显示标签
        self._updating = False                                 # 防止递归更新
        self._candidates_loaded = False                        # 是否已执行过 PICS 查询
        self._pending_hover_species: dict | None = None
        self._pending_hover_position = QtCore.QPoint()
        self.species_hover_card = _SpeciesHoverCard(self)
        self._species_hover_timer = QtCore.QTimer(self)
        self._species_hover_timer.setSingleShot(True)
        self._species_hover_timer.setInterval(250)
        self._species_hover_timer.timeout.connect(self._show_pending_species_hover)

        # ---- UI构建 ----
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # Header: candidate scope for the selected curve.
        header_frame = QtWidgets.QFrame()
        header_frame.setObjectName("PanelHeader")
        header_frame.setFixedHeight(32)
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(8, 4, 8, 4)
        header.setSpacing(0)
        title = QtWidgets.QLabel("候选物种")
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
        empty_layout.addStretch()

        empty_icon = StateGlyph("candidate")
        empty_layout.addWidget(
            empty_icon,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
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

        self.candidate_controls_widget = QtWidgets.QWidget()
        candidate_controls_layout = QtWidgets.QVBoxLayout(self.candidate_controls_widget)
        candidate_controls_layout.setContentsMargins(0, 0, 0, 0)
        candidate_controls_layout.setSpacing(6)

        candidate_summary_row = QtWidgets.QHBoxLayout()
        candidate_summary_row.setSpacing(6)
        self.candidate_summary_label = QtWidgets.QLabel("0 个候选 · 0 个启用")
        self.candidate_summary_label.setObjectName("CandidateSummary")
        self.candidate_summary_label.setToolTip(
            "拟合时使用非负最小二乘（NNLS）自动计算每个已选候选的贡献；"
            "所有贡献系数都不会小于 0。"
        )
        candidate_summary_row.addWidget(self.candidate_summary_label, stretch=1)
        self.fit_mode_label = QtWidgets.QLabel("自动拟合")
        self.fit_mode_label.setObjectName("FitModeBadge")
        self.fit_mode_label.setToolTip(
            "自动拟合使用非负最小二乘（NNLS）计算候选物种贡献。"
        )
        self.fit_mode_label.hide()
        self.lock_hint_label = QtWidgets.QLabel(
            "“保留”表示自动筛选时不移除该候选；贡献仍由自动拟合计算。"
        )
        self.lock_hint_label.setObjectName("HintLabel")
        self.lock_hint_label.hide()

        self.ie_query_btn = QtWidgets.QPushButton("查询缺失 IE")
        self.ie_query_btn.setToolTip("通过本地物种库 / NIST WebBook 查询当前候选中缺失的电离能")
        self.ie_query_btn.clicked.connect(self.ie_query_requested.emit)
        self.ie_query_btn.setObjectName("BrowseButton")
        self.ie_query_btn.hide()

        self.candidate_actions_btn = QtWidgets.QToolButton()
        self.candidate_actions_btn.setText("批量操作…")
        self.candidate_actions_btn.setObjectName("CommandMenuButton")
        self.candidate_actions_btn.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup
        )
        self.candidate_actions_menu = QtWidgets.QMenu(self.candidate_actions_btn)
        self.select_all_action = self.candidate_actions_menu.addAction("启用全部候选")
        self.select_all_action.triggered.connect(lambda: self._set_all_rows_checked(True))
        self.clear_selection_action = self.candidate_actions_menu.addAction("取消全部候选")
        self.clear_selection_action.triggered.connect(
            lambda: self._set_all_rows_checked(False)
        )
        self.candidate_actions_menu.addSeparator()
        self.ie_query_action = self.candidate_actions_menu.addAction("查询缺失电离能")
        self.ie_query_action.triggered.connect(self.ie_query_requested.emit)
        self.candidate_actions_btn.setMenu(self.candidate_actions_menu)
        self.candidate_actions_btn.setToolTip("批量处理当前候选物种")
        candidate_summary_row.addWidget(self.candidate_actions_btn)
        candidate_controls_layout.addLayout(candidate_summary_row)

        # ---- 下方：物种表格（启用 | 物种 | IE） ----
        self.species_table = QtWidgets.QTableWidget()
        self.species_table.setColumnCount(3)
        self.species_table.setHorizontalHeaderLabels([
            "启用", "候选物种", "IE (eV)"
        ])
        self.species_table.setWordWrap(False)
        self.species_table.setAlternatingRowColors(True)
        self.species_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.species_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.species_table.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        self.species_table.verticalHeader().setVisible(False)
        self.species_table.verticalHeader().setDefaultSectionSize(46)
        self.species_table.verticalHeader().setMinimumSectionSize(46)
        self.species_table.horizontalHeader().setStretchLastSection(False)
        candidate_controls_layout.addWidget(self.species_table, stretch=1)
        main_layout.addWidget(self.candidate_controls_widget, stretch=1)

        # ---- 结果摘要区（拟合后显示）----
        # 分隔线
        self.panel_separator = QtWidgets.QFrame()
        self.panel_separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.panel_separator.setObjectName("PanelSeparator")
        self.panel_separator.setVisible(False)
        main_layout.addWidget(self.panel_separator)

        self.result_section = QtWidgets.QWidget()
        self.result_section.setVisible(False)
        result_layout = QtWidgets.QVBoxLayout(self.result_section)
        result_layout.setContentsMargins(0, 4, 0, 4)
        result_layout.setSpacing(4)

        # Single result summary: state, fit metric and details.
        result_header = QtWidgets.QHBoxLayout()
        result_header.setSpacing(6)
        result_title = QtWidgets.QLabel("结果")
        result_title.setObjectName("ReadoutLabel")
        result_header.addWidget(result_title)

        self.result_status_label = QtWidgets.QLabel("")
        self.result_status_label.setObjectName("ResultStatus")
        self.result_status_label.setMinimumWidth(48)
        result_header.addWidget(self.result_status_label)

        self.result_r2_label = QtWidgets.QLabel("")
        self.result_r2_label.setObjectName("ResultMetrics")
        result_header.addWidget(self.result_r2_label)

        result_header.addStretch()

        self.result_details_btn = QtWidgets.QPushButton("详情")
        self.result_details_btn.setObjectName("ResultDetailLink")
        self.result_details_btn.setFixedHeight(24)
        self.result_details_btn.clicked.connect(self.result_details_requested.emit)
        result_header.addWidget(self.result_details_btn)
        result_layout.addLayout(result_header)

        self.result_contribution_label = ElidedLabel("")
        self.result_contribution_label.setObjectName("HintLabel")
        self.result_contribution_label.setMinimumWidth(0)
        self.result_contribution_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        result_layout.addWidget(self.result_contribution_label)

        # 操作按钮行
        action_row = QtWidgets.QHBoxLayout()
        action_row.setSpacing(6)

        self.confirm_btn = QtWidgets.QPushButton("确认结果")
        self.confirm_btn.setObjectName("WorkflowButton")
        self.confirm_btn.setToolTip("将当前拟合结果确认为该 m/z 的鉴定结论")
        self.confirm_btn.setFixedHeight(28)
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.clicked.connect(self.confirmation_requested.emit)
        action_row.addWidget(self.confirm_btn)

        self.export_result_btn = QtWidgets.QPushButton("导出鉴定")
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

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        self._hide_species_hover()
        super().hideEvent(event)

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

    @staticmethod
    def _formula_display_text(species: dict) -> str:
        formula = str(species.get("formula") or "").strip()
        if formula:
            return formula
        return _formula_from_smiles(str(species.get("smiles") or ""))

    def populate_unified_species_table(
        self,
        mz: int,
        filtered_db: list[dict],
        locked_species: list[str],
        *,
        display_mz: str | None = None,
    ):
        """
        填充统一的物种表格
        - 显示当前 m/z 的 PICS 数据库候选物种
        - 标记来源和锁定状态
        - 若无候选物种，显示空状态提示

        此方法在 PIE 曲线生成且 m/z 确定后被调用
        标记候选查询已执行（_candidates_loaded = True）
        """
        self._current_mz = mz
        self._current_mz_display = str(display_mz) if display_mz else str(mz)
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
            self.candidate_controls_widget.show()
            self.panel_separator.setVisible(self.result_section.isVisible())
            self._refresh_table_from_data()
        elif self._candidates_loaded:
            # 查询已执行但无结果 → 显示"未找到候选物种"空状态
            self.species_table.setRowCount(0)
            self.empty_icon.set_kind("warning")
            self.empty_icon.show()
            self.empty_title.setText("当前 m/z 未找到候选物种")
            self.empty_text.setText(
                "PICS 数据库中尚未收录该质荷比的物种或缺少截面数据。\n"
                "请先前往 PICS 导入添加对应物种及完整的截面数据。"
            )
            self.goto_pics_btn.show()
            self.empty_state_widget.show()
            self.candidate_controls_widget.hide()
            self.panel_separator.hide()
            self._hide_species_hover()
        else:
            # 前置数据未准备 → 显示"请先生成 PIE 曲线"空状态
            self.species_table.setRowCount(0)
            self.empty_icon.hide()
            self.empty_title.setText("尚未选择曲线")
            self.empty_text.setText("从左侧选择一条 m/z 曲线后显示候选物种。")
            self.goto_pics_btn.hide()
            self.empty_state_widget.show()
            self.candidate_controls_widget.hide()
            self.panel_separator.hide()
            self._hide_species_hover()
        self.current_mz_label.setText(
            f"m/z {self._current_mz_display}"
            if self._current_mz is not None
            else "未选择 m/z"
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
        previous_row = self.species_table.currentRow()
        self._updating = True
        try:
            self.species_table.setRowCount(len(self._unified_species_data))

            # 更新header显示当前m/z
            if self._current_mz is not None:
                self.species_table.setHorizontalHeaderLabels([
                    "启用", "候选物种", "IE (eV)"
                ])

            header = self.species_table.horizontalHeader()
            header.setSectionsMovable(False)
            header.setMinimumSectionSize(36)
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Fixed)
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Fixed)
            self.species_table.setColumnWidth(0, 40)
            self.species_table.setColumnWidth(2, 84)

            for row, species in enumerate(self._unified_species_data):
                self.species_table.setRowHeight(row, 46)
                self._populate_table_row(row, species)
        finally:
            self._updating = False
        if self._unified_species_data:
            target_row = previous_row if 0 <= previous_row < len(self._unified_species_data) else 0
            self.species_table.setCurrentCell(target_row, 1)

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

        # Col 1: species identity + optional keep-in-auto-selection control.
        species_widget = _SpeciesIdentityWidget()
        species_widget.activated.connect(
            lambda r=row: self._select_candidate_row(r)
        )
        species_widget.hover_entered.connect(
            lambda position, sp=species: self._schedule_species_hover(sp, position)
        )
        species_widget.hover_left.connect(self._hide_species_hover)
        species_layout = QtWidgets.QHBoxLayout(species_widget)
        species_layout.setContentsMargins(4, 2, 4, 2)
        species_layout.setSpacing(4)

        # 物种名称 + 常驻分子式
        species_name = species.get('species', '')
        identity_layout = QtWidgets.QVBoxLayout()
        identity_layout.setContentsMargins(0, 0, 0, 0)
        identity_layout.setSpacing(0)
        name_label = QtWidgets.QLabel(species_name)
        name_label.setObjectName("SpeciesNameLabel")
        name_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        name_label.setWordWrap(False)
        name_label.setMinimumWidth(0)
        name_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        identity_layout.addWidget(name_label)

        formula = self._formula_display_text(species)
        formula_label = QtWidgets.QLabel(formula or "分子式未知")
        formula_label.setObjectName("SpeciesFormulaLabel")
        formula_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        formula_label.setWordWrap(False)
        formula_label.setMinimumWidth(0)
        formula_label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        identity_layout.addWidget(formula_label)
        species_layout.addLayout(identity_layout, stretch=1)

        # "保留" only protects the candidate from automatic filtering. It
        # does not force a non-zero contribution.
        is_locked = species.get('is_locked', species.get('is_forced', False))
        toggle_btn = QtWidgets.QPushButton("已保留" if is_locked else "保留")
        toggle_btn.setFixedSize(46, 22)
        toggle_btn.setFlat(False)
        toggle_btn.clicked.connect(lambda checked=False, r=row: self._toggle_locked_status(r))

        # 根据锁定状态设置动态属性，具体视觉由应用级 QSS 管理。
        # 向后兼容：读取新字段is_locked，如果不存在则尝试读取旧字段is_forced
        toggle_btn.setObjectName("LockCandidateButton")
        toggle_btn.setProperty("locked", is_locked)

        species_layout.addWidget(toggle_btn)

        # 添加Tooltip：m/z、IE、来源、锁定状态
        ie_text = self._ie_display_text(species)
        ie_source = species.get('ie_source') or "未提供"
        ie_message = species.get('ie_message') or ""
        source = species.get('source', '自动')
        locked = "是" if species.get('is_locked', False) else "否"
        smiles = str(species.get("smiles") or "").strip()
        tooltip_text = (
            f"物种: {species_name}\n"
            f"分子式: {formula or '未提供'}\n"
            f"SMILES: {smiles or '未提供'}\n"
            f"m/z: {species.get('mz', 'N/A')}\n"
            f"IE: {ie_text}{' eV' if self._coerce_ie(species) is not None else ''}\n"
            f"IE来源: {ie_source}\n"
            + (f"IE备注: {ie_message}\n" if ie_message else "") +
            f"来源: {source}\n"
            f"锁定: {locked}"
        )
        toggle_btn.setToolTip(
            tooltip_text
            + "\n\n保留后，自动筛选不会移除该候选；贡献仍由自动拟合计算。"
        )

        self.species_table.setCellWidget(row, 1, species_widget)

        # Col 2: IE（只读文本，查询状态也在此显示）
        ie_item = QtWidgets.QTableWidgetItem(ie_text)
        ie_item.setFlags(ie_item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        ie_item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ie_item.setToolTip(tooltip_text)
        self.species_table.setItem(row, 2, ie_item)

    def _select_candidate_row(self, row: int) -> None:
        if 0 <= row < self.species_table.rowCount():
            self.species_table.setCurrentCell(row, 1)

    def _schedule_species_hover(
        self,
        species: dict,
        position: QtCore.QPoint,
    ) -> None:
        self._pending_hover_species = species
        self._pending_hover_position = QtCore.QPoint(position)
        self._species_hover_timer.start()

    def _show_pending_species_hover(self) -> None:
        species = self._pending_hover_species
        if species is None or not self.isVisible():
            return
        self.species_hover_card.show_species(
            species,
            formula=self._formula_display_text(species),
            ie_text=self._ie_display_text(species),
            anchor=self._pending_hover_position,
        )

    def _hide_species_hover(self) -> None:
        self._species_hover_timer.stop()
        self._pending_hover_species = None
        self.species_hover_card.hide()

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
                    toggle_btn.setText("已保留" if is_locked else "保留")
                    toggle_btn.setProperty("locked", is_locked)
                    toggle_btn.style().unpolish(toggle_btn)
                    toggle_btn.style().polish(toggle_btn)

            self._refresh_candidate_summary()
            self.species_config_changed.emit()

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

    # ---- 锁定候选管理 ----

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

    def get_candidate_data(self) -> list[dict]:
        """Return an isolated snapshot of the current candidate rows."""
        return list(self._unified_species_data)

    def set_candidate_data(self, candidates: list[dict]) -> None:
        """Replace candidate row data for state restoration and tests."""
        self._unified_species_data = list(candidates)

    @property
    def candidate_update_in_progress(self) -> bool:
        """Whether candidate widgets are being synchronized programmatically."""
        return bool(self._updating)

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
        self.ie_query_action.setEnabled(bool(self._unified_species_data) and has_missing and not has_pending)

    def _refresh_candidate_summary(self) -> None:
        candidate_count = len(self._unified_species_data)
        selected_ids = self._get_selected_candidate_ids() if candidate_count else []
        locked_count = sum(bool(species.get("is_locked")) for species in self._unified_species_data)

        self.candidate_summary_label.setText(
            f"候选 {candidate_count} · 已选 {len(selected_ids)}"
            + (f" · 保留 {locked_count}" if locked_count else "")
        )
        self.fit_mode_label.setText("自动拟合")
        self.fit_mode_label.setProperty("mode", "auto")
        self.fit_mode_label.style().unpolish(self.fit_mode_label)
        self.fit_mode_label.style().polish(self.fit_mode_label)

        has_candidates = candidate_count > 0
        self.select_all_action.setEnabled(has_candidates and len(selected_ids) < candidate_count)
        self.clear_selection_action.setEnabled(has_candidates and bool(selected_ids))
        self.candidate_actions_btn.setEnabled(has_candidates)
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
        self._current_mz = None
        self._current_mz_display = None
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
            f"<span style='color:{color}; font-weight:bold;'>R² = {r2:.4f}</span>"
        )
        self.result_r2_label.setTextFormat(QtCore.Qt.TextFormat.RichText)

        significant = sorted(
            (
                species
                for species in species_list
                if float(species.get("contribution_percent", 0.0) or 0.0) >= 0.05
            ),
            key=lambda species: -float(
                species.get("contribution_percent", 0.0) or 0.0
            ),
        )
        if significant:
            summary = " · ".join(
                f"{species.get('species', '未命名')} "
                f"{float(species.get('contribution_percent', 0.0) or 0.0):.1f}%"
                for species in significant[:3]
            )
            if len(significant) > 3:
                summary += f" · 另 {len(significant) - 3} 个"
            self.result_contribution_label.setText(f"贡献：{summary}")
        else:
            self.result_contribution_label.setText("未得到有效物种贡献")
        self.result_contribution_label.setToolTip(
            "\n".join(
                f"{species.get('species', '未命名')}："
                f"贡献 {float(species.get('contribution_percent', 0.0) or 0.0):.2f}% · "
                f"系数 {float(species.get('coefficient', 0.0) or 0.0):.6g}"
                for species in significant
            )
        )

        # 更新状态标签与结果操作可用性
        self.set_fit_result_status(status)

        # 显示结果区
        self.panel_separator.setVisible(True)
        self.result_section.setVisible(True)

    def clear_fit_result(self):
        """隐藏结果摘要区并清空内容"""
        self.result_r2_label.setText("")
        self.result_status_label.setText("")
        self.result_contribution_label.setText("")
        self.result_contribution_label.setToolTip("")
        self.confirm_btn.setText("确认结果")
        self.confirm_btn.setEnabled(False)
        self.export_result_btn.setEnabled(False)
        self.panel_separator.setVisible(False)
        self.result_section.setVisible(False)
        self._hide_species_hover()

    def set_fit_confirmed(self, confirmed: bool):
        """更新确认状态显示"""
        if confirmed:
            self.set_fit_result_status("CONFIRMED")
            self.confirm_btn.setText("重新确认")
        else:
            self.set_fit_result_status("COMPLETED")
            self.confirm_btn.setText("确认结果")

    def set_fit_result_status(self, status: str) -> None:
        """Update result validity and guard actions that require a committed fit."""
        self._set_result_status_label(status)
        committed = status in {"COMPLETED", "CONFIRMED"}
        self.confirm_btn.setEnabled(committed)
        if status == "CONFIRMED":
            self.confirm_btn.setText("重新确认")
        elif status == "COMPLETED":
            self.confirm_btn.setText("确认结果")
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
            "COMPLETED": ("已拟合", "completed"),
            "OBSOLETE":  ("需重拟合", "obsolete"),
            "CONFIRMED": ("已确认", "confirmed"),
            "PREVIEW":   ("实时预览", "preview"),
            "FAILED":    ("失败", "failed"),
        }
        text, visual_status = config.get(status, ("", ""))
        self.result_status_label.setText(text)
        self.result_status_label.setProperty("status", visual_status)
        self.result_status_label.style().unpolish(self.result_status_label)
        self.result_status_label.style().polish(self.result_status_label)
