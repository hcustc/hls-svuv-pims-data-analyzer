from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .isotope import calculate_isotope_distribution, formula_nominal_mass


CORRECTION_MODES = {
    "manual_fraction": "按母峰比例",
    "max_compatible": "最大非负相容贡献",
}

# Isotope pairs are molecule-specific scientific assumptions.  Projects may
# configure them explicitly, but the application must not silently impose one
# compound's channels on unrelated data.
DEFAULT_ISOTOPE_QC_PAIRS: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class IsotopeHypothesis:
    """One user-supplied molecular-formula hypothesis for a parent mass channel."""

    formula: str
    parent_mz: int
    fraction: float = 1.0
    label: str = ""

    def __post_init__(self) -> None:
        formula = str(self.formula or "").strip()
        if not formula:
            raise ValueError("分子式不能为空")
        parent_mz = int(round(float(self.parent_mz)))
        if parent_mz <= 0:
            raise ValueError("母峰 m/z 必须为正整数")
        nominal_mass = formula_nominal_mass(formula)
        if nominal_mass != parent_mz:
            raise ValueError(
                f"{formula} 的名义质量为 {nominal_mass}，与母峰 m/z {parent_mz} 不一致"
            )
        fraction = float(self.fraction)
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("母峰比例必须在 0 到 1 之间")
        object.__setattr__(self, "formula", formula)
        object.__setattr__(self, "parent_mz", parent_mz)
        object.__setattr__(self, "fraction", fraction)
        object.__setattr__(self, "label", str(self.label or "").strip())

    @property
    def display_name(self) -> str:
        return self.label or f"{self.formula}@{self.parent_mz}"


@dataclass(frozen=True)
class IsotopeCorrectionResult:
    """Numerical outputs for isotope contribution post-processing."""

    axis_name: str
    mode: str
    curve_table: pd.DataFrame
    component_table: pd.DataFrame
    source_table: pd.DataFrame
    pattern_table: pd.DataFrame
    pattern_rank: int
    pattern_condition_number: float
    diagnostic_table: pd.DataFrame
    sensitivity_table: pd.DataFrame
    scientific_note: str


def nominal_isotope_pattern(
    formula: str,
    *,
    max_shift: int = 6,
    min_ratio: float = 0.0,
) -> dict[int, float]:
    """Return nominal isotope ratios relative to the formula's base peak.

    The returned keys are integer nominal-mass shifts.  This is intended for
    unit-mass PIE and temperature-scan channels; it does not model instrument
    resolution or multiply charged ions.
    """

    max_shift = int(max_shift)
    min_ratio = float(min_ratio)
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    if min_ratio < 0:
        raise ValueError("min_ratio must be non-negative")

    base_mass = formula_nominal_mass(formula)
    rows = calculate_isotope_distribution(formula, min_percent=0.0)
    abundance_by_mass = {
        int(round(float(row["mass"]))): float(row["abundance"])
        for row in rows
    }
    base_abundance = abundance_by_mass.get(base_mass, 0.0)
    if base_abundance <= 0:
        raise ValueError(f"无法确定 {formula} 的基准同位素峰")

    pattern: dict[int, float] = {}
    for shift in range(max_shift + 1):
        ratio = abundance_by_mass.get(base_mass + shift, 0.0) / base_abundance
        if ratio > min_ratio or shift == 0:
            pattern[shift] = float(ratio)
    return pattern


def infer_curve_columns(data: pd.DataFrame, source_type: str) -> dict[str, object]:
    """Suggest axis, mass, and signal columns for exported PIE/temperature data."""

    source_type = str(source_type or "").strip().lower()
    if source_type not in {"temperature", "pie"}:
        raise ValueError("source_type must be 'temperature' or 'pie'")
    if data.empty and not len(data.columns):
        raise ValueError("结果表没有可读取的列")

    axis_candidates = (
        ("temperature", "温度")
        if source_type == "temperature"
        else ("energy", "photon_energy", "光子能量")
    )
    mass_candidates = ("mz_rounded", "mz", "m/z", "质量数 (m/z)", "质量数(m/z)", "质量数")
    signal_candidates = (
        ("area", "normalized_area", "photon_normalized_area", "raw_area", "强度")
        if source_type == "temperature"
        else (
            "merged_intensity",
            "io_time_normalized_intensity",
            "normalized_intensity",
            "photon_normalized_intensity",
            "raw_area",
            "intensity",
            "强度",
        )
    )

    axis_column = _first_present_column(data, axis_candidates)
    mass_column = _first_present_column(data, mass_candidates)
    if axis_column is None:
        expected = "temperature" if source_type == "temperature" else "energy"
        raise ValueError(f"结果表缺少 {expected} 轴列")
    if mass_column is None:
        raise ValueError("结果表缺少 m/z 列")

    metadata_columns = {
        "channel_id",
        "mz",
        "mz_rounded",
        "m/z",
        "peak_track",
        "left_bound",
        "right_bound",
        "photon_energy",
        "scan_energy",
        "reference_temperature",
        "expansion_lambda",
        "file_count",
    }
    numeric_columns = [
        str(column)
        for column in data.columns
        if column not in {axis_column, mass_column}
        and str(column) not in metadata_columns
        and pd.to_numeric(data[column], errors="coerce").notna().any()
    ]
    ordered_signals = [
        candidate
        for candidate in signal_candidates
        if candidate in numeric_columns
    ]
    ordered_signals.extend(column for column in numeric_columns if column not in ordered_signals)
    if not ordered_signals:
        raise ValueError("结果表没有可用的数值信号列")
    return {
        "axis_column": axis_column,
        "mass_column": mass_column,
        "signal_columns": ordered_signals,
        "preferred_signal_column": ordered_signals[0],
    }


def prepare_curve_matrix(
    data: pd.DataFrame,
    *,
    axis_column: str,
    intensity_column: str,
    mass_column: str | None = None,
    mz_min: int | None = None,
    mz_max: int | None = None,
    aggregation: str = "mean",
    peak_track_column: str | None = None,
    peak_track_selection: Mapping[int, object] | None = None,
) -> pd.DataFrame:
    """Convert a long exported result table into axis × nominal-mass signals."""

    if aggregation not in {"mean", "sum"}:
        raise ValueError("aggregation must be 'mean' or 'sum'")
    if axis_column not in data:
        raise ValueError(f"结果表缺少坐标列: {axis_column}")
    if intensity_column not in data:
        raise ValueError(f"结果表缺少信号列: {intensity_column}")
    resolved_mass_column = mass_column or ("mz_rounded" if "mz_rounded" in data else "mz")
    if resolved_mass_column not in data:
        raise ValueError(f"结果表缺少质量列: {resolved_mass_column}")

    track_column = peak_track_column
    if track_column is None and "peak_track" in data.columns:
        track_column = "peak_track"
    if track_column is not None and track_column not in data.columns:
        raise ValueError(f"结果表缺少峰轨道列: {track_column}")

    working_data: dict[str, object] = {
        "_axis": pd.to_numeric(data[axis_column], errors="coerce"),
        "_mz": pd.to_numeric(data[resolved_mass_column], errors="coerce"),
        "_intensity": pd.to_numeric(data[intensity_column], errors="coerce"),
    }
    if track_column is not None:
        working_data["_peak_track"] = data[track_column]
    elif {"left_bound", "right_bound"}.issubset(data.columns):
        working_data["_peak_track"] = list(
            zip(data["left_bound"].tolist(), data["right_bound"].tolist())
        )
        track_column = "__peak_geometry__"
    working = pd.DataFrame(
        working_data
    ).dropna()
    if working.empty:
        raise ValueError("所选坐标、质量和信号列没有有效数值")

    working["_nominal_mz"] = np.rint(working["_mz"]).astype(int)
    selection = {
        int(round(float(key))): value
        for key, value in dict(peak_track_selection or {}).items()
    }
    if "_peak_track" in working.columns:
        collision_masses = [
            int(mass)
            for mass, group in working.groupby("_nominal_mz", sort=True)
            if group["_peak_track"].nunique(dropna=True) > 1
        ]
        missing_selection = [
            mass for mass in collision_masses if mass not in selection
        ]
        if missing_selection:
            masses_text = "、".join(str(value) for value in missing_selection)
            raise ValueError(
                "同一名义质量存在多个精确峰轨道，不能自动平均："
                f"m/z {masses_text}。请为每个碰撞质量选择具体 peak_track。"
            )
        if selection:
            keep = np.ones(len(working), dtype=bool)
            for mass, selected_track in selection.items():
                mass_mask = working["_nominal_mz"].to_numpy() == mass
                track_mask = (
                    working["_peak_track"].astype(str).to_numpy()
                    == str(selected_track)
                )
                keep &= ~mass_mask | track_mask
            working = working.loc[keep]
            if working.empty:
                raise ValueError("所选 peak_track 没有可用数据")
    if mz_min is not None:
        working = working[working["_nominal_mz"] >= int(mz_min)]
    if mz_max is not None:
        working = working[working["_nominal_mz"] <= int(mz_max)]
    if working.empty:
        raise ValueError("所选质量范围内没有数据")

    matrix = working.pivot_table(
        index="_axis",
        columns="_nominal_mz",
        values="_intensity",
        aggfunc=aggregation,
        sort=True,
    )
    matrix = matrix.sort_index().sort_index(axis=1)
    matrix.index.name = axis_column
    matrix.columns = [int(column) for column in matrix.columns]
    matrix.columns.name = "mz"
    return matrix


def evaluate_isotope_qc_pairs(
    curve_matrix: pd.DataFrame,
    pairs: Sequence[Mapping[str, object]] = DEFAULT_ISOTOPE_QC_PAIRS,
    *,
    relative_tolerance: float = 0.20,
    min_valid_points: int = 3,
) -> pd.DataFrame:
    """Compare observed isotope pairs with formula-derived theoretical ratios."""
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative")
    rows: list[dict[str, object]] = []
    for pair in pairs:
        light_mz = int(pair["light_mz"])
        heavy_mz = int(pair["heavy_mz"])
        formula = str(pair["formula"])
        shift = heavy_mz - light_mz
        if shift <= 0:
            raise ValueError("同位素核验重峰质量必须大于轻峰质量")
        theoretical = float(
            nominal_isotope_pattern(formula, max_shift=shift).get(shift, 0.0)
        )
        status = "pass"
        reason = ""
        valid_count = 0
        observed_median = float("nan")
        observed_mad = float("nan")
        relative_error = float("nan")
        if light_mz not in curve_matrix.columns or heavy_mz not in curve_matrix.columns:
            status = "missing"
            reason = "缺少轻峰或重峰通道"
        else:
            light = pd.to_numeric(curve_matrix[light_mz], errors="coerce").to_numpy(
                dtype=float
            )
            heavy = pd.to_numeric(curve_matrix[heavy_mz], errors="coerce").to_numpy(
                dtype=float
            )
            valid = (
                np.isfinite(light)
                & np.isfinite(heavy)
                & (light > 0)
                & (heavy >= 0)
            )
            ratios = heavy[valid] / light[valid]
            valid_count = int(ratios.size)
            if valid_count < int(min_valid_points):
                status = "fail"
                reason = (
                    f"有效点数 {valid_count} 少于要求 {int(min_valid_points)}"
                )
            else:
                observed_median = float(np.median(ratios))
                observed_mad = float(
                    np.median(np.abs(ratios - observed_median))
                )
                relative_error = (
                    abs(observed_median - theoretical) / theoretical
                    if theoretical > 0
                    else float("inf")
                )
                if relative_error > float(relative_tolerance):
                    status = "fail"
                    reason = (
                        f"相对偏差 {relative_error:.1%} 超过"
                        f" {float(relative_tolerance):.1%}"
                    )
        rows.append(
            {
                "light_mz": light_mz,
                "heavy_mz": heavy_mz,
                "formula": formula,
                "theoretical_heavy_to_light": theoretical,
                "theoretical_light_to_heavy": (
                    1.0 / theoretical if theoretical > 0 else float("inf")
                ),
                "observed_heavy_to_light_median": observed_median,
                "observed_ratio_mad": observed_mad,
                "relative_error": relative_error,
                "valid_point_count": valid_count,
                "status": status,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def apply_isotope_correction(
    curve_matrix: pd.DataFrame,
    hypotheses: Sequence[IsotopeHypothesis],
    *,
    mode: str = "manual_fraction",
) -> IsotopeCorrectionResult:
    """Calculate formula-dependent isotope contributions without interpretation.

    ``manual_fraction`` scales every hypothesis by the configured fraction of
    its observed parent channel. ``max_compatible`` maximizes the total
    theoretical isotope contribution at each axis point while requiring every
    modeled contribution and every remaining signal to stay non-negative.

    The parent ``M`` channel is a source curve, not an isotope interference
    against itself.  It is therefore retained in the corrected output; only
    positive mass shifts (M+1, M+2, ...) are subtracted from their target
    channels.
    """

    mode = str(mode or "").strip()
    if mode not in CORRECTION_MODES:
        raise ValueError(f"未知计算模式: {mode}")
    hypothesis_list = list(hypotheses)
    if not hypothesis_list:
        raise ValueError("请至少添加一个分子式假设")
    if curve_matrix.empty:
        raise ValueError("曲线矩阵为空")

    matrix = curve_matrix.copy()
    try:
        matrix.columns = [int(round(float(column))) for column in matrix.columns]
    except (TypeError, ValueError) as exc:
        raise ValueError("曲线矩阵的质量通道必须是数值") from exc
    if len(set(matrix.columns)) != len(matrix.columns):
        raise ValueError("曲线矩阵包含重复的质量通道")
    matrix = matrix.sort_index(axis=1)
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("所选曲线存在缺失值或非有限数，请调整质量范围或数据聚合方式")
    if mode == "max_compatible" and np.any(values < 0):
        raise ValueError("最大非负相容模式不能处理负信号，请先检查背景扣除结果")

    masses = [int(value) for value in matrix.columns]
    mass_index = {mass: index for index, mass in enumerate(masses)}
    for hypothesis in hypothesis_list:
        if hypothesis.parent_mz not in mass_index:
            raise ValueError(
                f"{hypothesis.display_name} 的母峰 m/z {hypothesis.parent_mz} 不在所选质量范围内"
            )

    pattern_matrix, pattern_table = _build_pattern_matrix(hypothesis_list, masses)
    subtraction_matrix = pattern_matrix.copy()
    for hypothesis_index, hypothesis in enumerate(hypothesis_list):
        subtraction_matrix[
            mass_index[hypothesis.parent_mz],
            hypothesis_index,
        ] = 0.0
    if not np.any(subtraction_matrix > 0):
        raise ValueError("当前分子式在所选质量范围内没有理论同位素贡献")

    if mode == "manual_fraction":
        fractions_by_parent: dict[int, float] = {}
        for hypothesis in hypothesis_list:
            fractions_by_parent[hypothesis.parent_mz] = (
                fractions_by_parent.get(hypothesis.parent_mz, 0.0)
                + hypothesis.fraction
            )
        overallocated = [
            parent_mz
            for parent_mz, total_fraction in fractions_by_parent.items()
            if total_fraction > 1.0 + 1e-12
        ]
        if overallocated:
            masses_text = "、".join(str(value) for value in sorted(overallocated))
            raise ValueError(f"同一母峰的手动比例总和不能超过 100%：m/z {masses_text}")
        source_amplitudes = np.column_stack(
            [
                values[:, mass_index[hypothesis.parent_mz]] * hypothesis.fraction
                for hypothesis in hypothesis_list
            ]
        )
    else:
        source_amplitudes = np.vstack(
            [
                _maximum_compatible_amplitudes(pattern_matrix, observed)
                for observed in values
            ]
        )

    predicted = source_amplitudes @ subtraction_matrix.T
    residual = values - predicted
    axis_name = str(matrix.index.name or "axis")
    curve_table = _build_curve_table(
        matrix.index,
        masses,
        values,
        predicted,
        residual,
        axis_name=axis_name,
        mode=mode,
    )
    component_table, source_table = _build_component_tables(
        matrix.index,
        masses,
        hypothesis_list,
        subtraction_matrix,
        source_amplitudes,
        axis_name=axis_name,
        mode=mode,
    )

    rank = int(np.linalg.matrix_rank(pattern_matrix))
    condition_number = (
        float(np.linalg.cond(pattern_matrix))
        if rank == len(hypothesis_list)
        else float("inf")
    )
    diagnostic_table = _build_numerical_diagnostic_table(
        matrix.index,
        values,
        pattern_matrix,
        source_amplitudes,
        axis_name=axis_name,
        rank=rank,
        condition_number=condition_number,
    )
    sensitivity_table = _build_coefficient_sensitivity_table(
        matrix.index,
        values,
        pattern_matrix,
        source_amplitudes,
        hypothesis_list,
        axis_name=axis_name,
        mode=mode,
    )
    return IsotopeCorrectionResult(
        axis_name=axis_name,
        mode=mode,
        curve_table=curve_table,
        component_table=component_table,
        source_table=source_table,
        pattern_table=pattern_table,
        pattern_rank=rank,
        pattern_condition_number=condition_number,
        diagnostic_table=diagnostic_table,
        sensitivity_table=sensitivity_table,
        scientific_note=(
            "质量峰存在可重复的系统质量偏移是可预见的；精确质量偏差不能单独用于"
            "排除分子式。归属还必须结合标定兼容性、同位素闭合和PIE形状。"
        ),
    )


def _first_present_column(data: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    columns = {str(column).strip().lower(): str(column) for column in data.columns}
    for candidate in candidates:
        matched = columns.get(str(candidate).strip().lower())
        if matched is not None:
            return matched
    return None


def _build_pattern_matrix(
    hypotheses: Sequence[IsotopeHypothesis],
    masses: Sequence[int],
) -> tuple[np.ndarray, pd.DataFrame]:
    max_shift = max(
        0,
        max(
            (mass - hypothesis.parent_mz for hypothesis in hypotheses for mass in masses),
            default=0,
        ),
    )
    pattern_matrix = np.zeros((len(masses), len(hypotheses)), dtype=float)
    rows: list[dict] = []
    for hypothesis_index, hypothesis in enumerate(hypotheses):
        pattern = nominal_isotope_pattern(hypothesis.formula, max_shift=max_shift)
        hypothesis_id = f"H{hypothesis_index + 1}"
        for mass_index, mass in enumerate(masses):
            shift = int(mass - hypothesis.parent_mz)
            ratio = float(pattern.get(shift, 0.0)) if shift >= 0 else 0.0
            pattern_matrix[mass_index, hypothesis_index] = ratio
            if ratio > 0:
                rows.append(
                    {
                        "hypothesis": hypothesis_id,
                        "label": hypothesis.display_name,
                        "formula": hypothesis.formula,
                        "parent_mz": hypothesis.parent_mz,
                        "fraction": hypothesis.fraction,
                        "mz": int(mass),
                        "mass_shift": shift,
                        "isotope_ratio": ratio,
                    }
                )
    return pattern_matrix, pd.DataFrame(
        rows,
        columns=[
            "hypothesis",
            "label",
            "formula",
            "parent_mz",
            "fraction",
            "mz",
            "mass_shift",
            "isotope_ratio",
        ],
    )


def _maximum_compatible_amplitudes(pattern_matrix: np.ndarray, observed: np.ndarray) -> np.ndarray:
    hypothesis_count = pattern_matrix.shape[1]
    if hypothesis_count == 1:
        ratios = pattern_matrix[:, 0]
        active = ratios > 0
        if not np.any(active):
            return np.zeros(1, dtype=float)
        return np.asarray([float(np.min(observed[active] / ratios[active]))], dtype=float)

    from scipy.optimize import linprog

    explained_per_amplitude = pattern_matrix.sum(axis=0)
    result = linprog(
        c=-explained_per_amplitude,
        A_ub=pattern_matrix,
        b_ub=observed,
        bounds=[(0.0, None)] * hypothesis_count,
        method="highs",
    )
    if not result.success:
        raise ValueError(f"最大非负相容贡献求解失败: {result.message}")
    return np.asarray(result.x, dtype=float)


def _build_numerical_diagnostic_table(
    axis_values,
    observed: np.ndarray,
    pattern_matrix: np.ndarray,
    source_amplitudes: np.ndarray,
    *,
    axis_name: str,
    rank: int,
    condition_number: float,
) -> pd.DataFrame:
    full_modeled = source_amplitudes @ pattern_matrix.T
    closure = observed - full_modeled
    rows: list[dict[str, object]] = []
    for axis_index, axis_value in enumerate(axis_values):
        observed_norm = float(np.linalg.norm(observed[axis_index], ord=1))
        closure_norm = float(np.linalg.norm(closure[axis_index], ord=1))
        rows.append(
            {
                axis_name: float(axis_value),
                "observed_total": float(np.sum(observed[axis_index])),
                "full_modeled_total": float(np.sum(full_modeled[axis_index])),
                "closure_l1": closure_norm,
                "relative_closure_error": (
                    closure_norm / observed_norm
                    if observed_norm > 0
                    else float("nan")
                ),
                "minimum_closure_residual": float(np.min(closure[axis_index])),
                "matrix_rank": int(rank),
                "condition_number": float(condition_number),
            }
        )
    return pd.DataFrame(rows)


def _build_coefficient_sensitivity_table(
    axis_values,
    observed: np.ndarray,
    pattern_matrix: np.ndarray,
    source_amplitudes: np.ndarray,
    hypotheses: Sequence[IsotopeHypothesis],
    *,
    axis_name: str,
    mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for axis_index, axis_value in enumerate(axis_values):
        base = source_amplitudes[axis_index]
        if mode == "max_compatible":
            elasticities = np.zeros(len(hypotheses), dtype=float)
            for mass_index, observed_value in enumerate(observed[axis_index]):
                delta = max(abs(float(observed_value)) * 0.01, 1e-9)
                perturbed = observed[axis_index].copy()
                perturbed[mass_index] += delta
                shifted = _maximum_compatible_amplitudes(
                    pattern_matrix,
                    perturbed,
                )
                for hypothesis_index, base_value in enumerate(base):
                    denominator = max(abs(float(base_value)), 1e-12)
                    input_fraction = delta / max(
                        abs(float(observed_value)),
                        delta,
                    )
                    elasticity = (
                        abs(float(shifted[hypothesis_index] - base_value))
                        / denominator
                        / input_fraction
                    )
                    elasticities[hypothesis_index] = max(
                        elasticities[hypothesis_index],
                        elasticity,
                    )
        else:
            elasticities = np.ones(len(hypotheses), dtype=float)
        for hypothesis_index, hypothesis in enumerate(hypotheses):
            rows.append(
                {
                    axis_name: float(axis_value),
                    "hypothesis": f"H{hypothesis_index + 1}",
                    "label": hypothesis.display_name,
                    "formula": hypothesis.formula,
                    "parent_mz": hypothesis.parent_mz,
                    "source_amplitude": float(base[hypothesis_index]),
                    "max_coefficient_elasticity": float(
                        elasticities[hypothesis_index]
                    ),
                    "mode": mode,
                }
            )
    return pd.DataFrame(rows)


def _build_curve_table(
    axis_values,
    masses: Sequence[int],
    observed: np.ndarray,
    predicted: np.ndarray,
    residual: np.ndarray,
    *,
    axis_name: str,
    mode: str,
) -> pd.DataFrame:
    rows = []
    for axis_index, axis_value in enumerate(axis_values):
        for mass_index, mass in enumerate(masses):
            observed_value = float(observed[axis_index, mass_index])
            contribution_value = float(predicted[axis_index, mass_index])
            residual_value = float(residual[axis_index, mass_index])
            if observed_value != 0:
                contribution_percent = contribution_value / observed_value * 100.0
                residual_percent = residual_value / observed_value * 100.0
            else:
                contribution_percent = float("nan")
                residual_percent = float("nan")
            rows.append(
                {
                    axis_name: float(axis_value),
                    "mz": int(mass),
                    "observed": observed_value,
                    "isotope_contribution": contribution_value,
                    "residual": residual_value,
                    "isotope_contribution_percent": contribution_percent,
                    "residual_percent": residual_percent,
                    "mode": mode,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            axis_name,
            "mz",
            "observed",
            "isotope_contribution",
            "residual",
            "isotope_contribution_percent",
            "residual_percent",
            "mode",
        ],
    )


def _build_component_tables(
    axis_values,
    masses: Sequence[int],
    hypotheses: Sequence[IsotopeHypothesis],
    pattern_matrix: np.ndarray,
    source_amplitudes: np.ndarray,
    *,
    axis_name: str,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_rows: list[dict] = []
    source_rows: list[dict] = []
    for axis_index, axis_value in enumerate(axis_values):
        for hypothesis_index, hypothesis in enumerate(hypotheses):
            hypothesis_id = f"H{hypothesis_index + 1}"
            amplitude = float(source_amplitudes[axis_index, hypothesis_index])
            source_rows.append(
                {
                    axis_name: float(axis_value),
                    "hypothesis": hypothesis_id,
                    "label": hypothesis.display_name,
                    "formula": hypothesis.formula,
                    "parent_mz": hypothesis.parent_mz,
                    "fraction": hypothesis.fraction,
                    "source_amplitude": amplitude,
                    "mode": mode,
                }
            )
            for mass_index, mass in enumerate(masses):
                ratio = float(pattern_matrix[mass_index, hypothesis_index])
                if ratio <= 0:
                    continue
                component_rows.append(
                    {
                        axis_name: float(axis_value),
                        "hypothesis": hypothesis_id,
                        "label": hypothesis.display_name,
                        "formula": hypothesis.formula,
                        "parent_mz": hypothesis.parent_mz,
                        "mz": int(mass),
                        "isotope_ratio": ratio,
                        "source_amplitude": amplitude,
                        "contribution": amplitude * ratio,
                        "mode": mode,
                    }
                )
    return (
        pd.DataFrame(
            component_rows,
            columns=[
                axis_name,
                "hypothesis",
                "label",
                "formula",
                "parent_mz",
                "mz",
                "isotope_ratio",
                "source_amplitude",
                "contribution",
                "mode",
            ],
        ),
        pd.DataFrame(
            source_rows,
            columns=[
                axis_name,
                "hypothesis",
                "label",
                "formula",
                "parent_mz",
                "fraction",
                "source_amplitude",
                "mode",
            ],
        ),
    )


__all__ = [
    "CORRECTION_MODES",
    "DEFAULT_ISOTOPE_QC_PAIRS",
    "IsotopeCorrectionResult",
    "IsotopeHypothesis",
    "apply_isotope_correction",
    "evaluate_isotope_qc_pairs",
    "infer_curve_columns",
    "nominal_isotope_pattern",
    "prepare_curve_matrix",
]
