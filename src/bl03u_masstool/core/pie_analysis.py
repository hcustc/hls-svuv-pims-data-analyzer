from __future__ import annotations

from pathlib import Path
import logging
import re
import sqlite3

import numpy as np
import pandas as pd

from .calibration import Calibration
from .integration import baseline_corrected_area, gaussian_area
from .normalization import extract_light_intensity
from .peak_ranges import load_peak_ranges, peak_ranges_to_peaks
from .peak_detection import detect_peaks_by_algorithm, fit_gaussian
from .spectrum_io import Spectrum, extract_first_number, read_spectrum


logger = logging.getLogger(__name__)

EV_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*eV", re.IGNORECASE)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS species (
    id INTEGER PRIMARY KEY,
    mz INTEGER NOT NULL,
    name TEXT NOT NULL,
    ionization_energy REAL,
    formula TEXT,
    elements TEXT,
    smiles TEXT
);

CREATE TABLE IF NOT EXISTS pic_cross_sections (
    id INTEGER PRIMARY KEY,
    species_id INTEGER NOT NULL,
    energy_ev REAL NOT NULL,
    cross_section REAL NOT NULL,
    FOREIGN KEY (species_id) REFERENCES species(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_species_mz ON species(mz);
CREATE INDEX IF NOT EXISTS idx_pics_species_energy ON pic_cross_sections(species_id, energy_ev);
"""


def load_species_database(path: str | Path) -> tuple[list[dict], dict[int, list[int]]]:
    """Load the PIE PICS database from SQLite."""
    path = Path(path)
    if path.suffix.lower() not in {".sqlite", ".sqlite3", ".db"}:
        raise ValueError("PIE PICS database must be a SQLite file")
    return load_species_database_sqlite(path)


def save_species_database_sqlite(database: list[dict], sqlite_path: str | Path, *, overwrite: bool = True) -> Path:
    """Write parsed PICS species records to a normalized SQLite database."""
    sqlite_path = Path(sqlite_path)
    if overwrite and sqlite_path.exists():
        sqlite_path.unlink()
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_SQL)
        for species_id, item in enumerate(database, start=1):
            conn.execute(
                "INSERT INTO species (id, mz, name, ionization_energy, formula, elements, smiles) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    species_id, 
                    int(item["mz"]), 
                    str(item["species"]), 
                    item.get("ie"), 
                    item.get("formula"), 
                    item.get("elements"), 
                    item.get("smiles"),
                ),
            )
            energies = np.asarray(item["energies"], dtype=float)
            cross_sections = np.asarray(item["cross_sections"], dtype=float)
            conn.executemany(
                "INSERT INTO pic_cross_sections (species_id, energy_ev, cross_section) VALUES (?, ?, ?)",
                [(species_id, float(e), float(c)) for e, c in zip(energies, cross_sections)],
            )
        conn.commit()
    return sqlite_path


def load_species_database_sqlite(path: str | Path) -> tuple[list[dict], dict[int, list[int]]]:
    """Load PICS species records from SQLite into the core fitting structure."""
    database: list[dict] = []
    mz_index: dict[int, list[int]] = {}
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        species_rows = conn.execute(
            "SELECT id, mz, name, ionization_energy, formula, elements, smiles FROM species ORDER BY id"
        ).fetchall()
        for species_row in species_rows:
            pic_rows = conn.execute(
                "SELECT energy_ev, cross_section FROM pic_cross_sections WHERE species_id = ? ORDER BY energy_ev",
                (species_row["id"],),
            ).fetchall()
            if not pic_rows:
                continue
            index = len(database)
            mz = int(species_row["mz"])
            database.append({
                "id": int(species_row["id"]),
                "mz": mz,
                "species": species_row["name"],
                "ie": species_row["ionization_energy"],
                "formula": species_row["formula"],
                "elements": species_row["elements"],
                "smiles": species_row["smiles"],
                "energies": np.array([row["energy_ev"] for row in pic_rows]),
                "cross_sections": np.array([row["cross_section"] for row in pic_rows]),
            })
            mz_index.setdefault(mz, []).append(index)
    return database, mz_index


def query_species_by_mz(path: str | Path, mz: int) -> list[dict]:
    """Query candidate species records for one integer m/z from SQLite."""
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return [{
            "id": row["id"],
            "mz": row["mz"],
            "species": row["name"],
            "ie": row["ionization_energy"],
            "formula": row["formula"],
            "elements": row["elements"],
            "smiles": row["smiles"],
        } for row in conn.execute(
            "SELECT id, mz, name, ionization_energy, formula, elements, smiles FROM species WHERE mz = ? ORDER BY id",
            (int(mz),)
        )]


def extract_photon_energy(metadata_lines: list[str], path: str | Path | None = None, fallback: float | None = None) -> float:
    """Extract nominal photon energy for PIE grouping.

    Priority:
    1. File metadata (most accurate) - Energy field in spectrum header
    2. Directory path (fallback) - e.g., "8.0eV" in directory name
    3. Fallback value
    """
    # First try: extract from file metadata (most accurate)
    for line in metadata_lines:
        if "energy" in line.lower():
            value = extract_first_number(line)
            if value is not None:
                return value

    # Second try: extract from path (useful for legacy folder structures)
    if path is not None:
        path_obj = Path(path)
        for part in reversed(path_obj.parts):
            match = EV_RE.search(part)
            if match:
                return float(match.group(1))

    # Last resort: fallback
    if fallback is not None:
        return float(fallback)
    raise ValueError("cannot extract photon energy")


def extract_io_current(metadata_lines: list[str], fallback: float = 1.0) -> float:
    """Extract IO current from spectrum header metadata."""
    return extract_light_intensity(metadata_lines, "io", fallback)


def _iter_pie_files(folder: str | Path, suffixes: tuple[str, ...], recursive: bool) -> list[Path]:
    folder = Path(folder)
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in suffixes)


def _is_blank_spectrum_path(path: Path) -> bool:
    text = " ".join((*path.parts[-3:], path.stem)).lower()
    return "blank" in text or "background" in text or "空白" in text


def _group_spectra_by_energy(
    folder: str | Path,
    *,
    suffixes: tuple[str, ...],
    recursive: bool,
    energy_decimals: int,
    light_source: str,
) -> list[dict]:
    grouped: dict[float, list[tuple[Path, Spectrum, float, float]]] = {}
    blank_grouped: dict[float, list[tuple[Path, Spectrum, float, float]]] = {}
    files = _iter_pie_files(folder, suffixes, recursive)
    for idx, path in enumerate(files):
        spectrum = read_spectrum(path, header_lines=10 if path.suffix.lower() == ".txt" else None, trim_start=0)
        if len(spectrum.y) == 0:
            continue
        energy = extract_photon_energy(spectrum.metadata_lines, path, fallback=float(idx))
        energy_key = round(energy, energy_decimals)
        io_current = extract_light_intensity(spectrum.metadata_lines, light_source)
        target = blank_grouped if _is_blank_spectrum_path(path) else grouped
        target.setdefault(energy_key, []).append((path, spectrum, energy, io_current))
    groups = []
    for energy_key, items in grouped.items():
        blank_items = blank_grouped.get(energy_key, [])
        min_len = min(
            [len(item[1].y) for item in items]
            + [len(item[1].y) for item in blank_items]
        )
        y_mean = np.mean([item[1].y[:min_len] for item in items], axis=0)
        blank_file_count = len(blank_items)
        if blank_items:
            blank_mean = np.mean([item[1].y[:min_len] for item in blank_items], axis=0)
            y_mean = y_mean - blank_mean
        first_x = items[0][1].x
        x_values = first_x[:min_len] if len(first_x) >= min_len else np.arange(1, min_len + 1, dtype=float)
        groups.append({
            "energy": float(np.mean([item[2] for item in items])),
            "energy_key": energy_key,
            "io": float(np.mean([item[3] for item in items])),
            "light_source": light_source,
            "file_count": len(items),
            "files": [item[0].name for item in items],
            "blank_file_count": blank_file_count,
            "blank_files": [item[0].name for item in blank_items],
            "background_subtracted": bool(blank_items),
            "spectrum": Spectrum(
                x=np.asarray(x_values, dtype=float),
                y=np.asarray(y_mean, dtype=float),
                metadata_lines=[],
                path=str(folder),
            ),
        })
    return sorted(groups, key=lambda x: x["energy"])


def analyze_pie_folder(
    folder: str | Path,
    *,
    calibration: Calibration = Calibration(),
    suffixes: tuple[str, ...] = (".txt", ".asc", ".888"),
    recursive: bool = True,
    energy_decimals: int = 1,
    algorithm: str = "legacy",
    threshold_end: float = 2,
    min_intensity: float = 3,
    detection_min_idx: int = 3000,
    nearby_peak_window: int = 30,
    duplicate_window: int = 20,
    weak_tail_early_window: int = 90,
    weak_tail_late_window: int = 50,
    weak_tail_ratio: float = 5,
    gaussian_window_max: int = 30,
    gaussian_boundary_scale: float = 1.5,
    boundary_padding: int = 2,
    prominence_ratio: float = 0.005,
    smoothing_window: int = 5,
    smoothing_poly_order: int = 2,
    baseline_window: int = 301,
    baseline_percentile: float = 5.0,
    min_peak_width: int = 1,
    max_peak_width: int = 80,
    prefer_gaussian: bool = True,
    manual_peak_path: str | Path | None = None,
    photon_normalize: bool = True,
    photon_reference_mode: str = "first",
    mass_discrimination: float = 1.0,
    light_source: str = "io",
    target_mz_values: list[int] | None = None,
    vote_threshold: float = 0.667,
    min_intensity_for_single_vote: float = 5.0,
    mz_tolerance: float = 0.2,
    cwt_snr_threshold: float = 0.02,
    cwt_wavelet_max_width: int = 30,
    weak_tail_cutoff_idx: int = 15000,
) -> pd.DataFrame:
    """Generate experimental PIE curves from a folder of energy-resolved spectra."""
    groups = _group_spectra_by_energy(
        folder,
        suffixes=suffixes,
        recursive=recursive,
        energy_decimals=energy_decimals,
        light_source=light_source,
    )
    if not groups:
        return pd.DataFrame(columns=[
            "energy", "file_count", "io", "light_source", "mz", "mz_rounded",
            "species", "photon_normalized_intensity", "normalized_intensity",
            "raw_area", "left_bound", "right_bound", "blank_file_count",
            "background_subtracted",
        ])
    if manual_peak_path:
        reference_group = max(groups, key=lambda x: x["energy"])
        reference_peaks = peak_ranges_to_peaks(load_peak_ranges(manual_peak_path, calibration=calibration))
        reference_source = Path(manual_peak_path).name
    else:
        reference_group = max(groups, key=lambda x: x["energy"])
        reference_spectrum = reference_group["spectrum"]
        reference_peaks = detect_peaks_by_algorithm(
            reference_spectrum.y,
            algorithm=algorithm,
            calibration=calibration,
            start_idx=0,
            end_idx=len(reference_spectrum.y),
            detection_min_idx=detection_min_idx,
            threshold_end=threshold_end,
            min_intensity=min_intensity,
            nearby_peak_window=nearby_peak_window,
            duplicate_window=duplicate_window,
            weak_tail_early_window=weak_tail_early_window,
            weak_tail_late_window=weak_tail_late_window,
            weak_tail_ratio=weak_tail_ratio,
            gaussian_window_max=gaussian_window_max,
            gaussian_boundary_scale=gaussian_boundary_scale,
            boundary_padding=boundary_padding,
            prominence_ratio=prominence_ratio,
            smoothing_window=smoothing_window,
            smoothing_poly_order=smoothing_poly_order,
            baseline_window=baseline_window,
            baseline_percentile=baseline_percentile,
            min_peak_width=min_peak_width,
            max_peak_width=max_peak_width,
            vote_threshold=vote_threshold,
            min_intensity_for_single_vote=min_intensity_for_single_vote,
            mz_tolerance=mz_tolerance,
            cwt_snr_threshold=cwt_snr_threshold,
            cwt_wavelet_max_width=cwt_wavelet_max_width,
            weak_tail_cutoff_idx=weak_tail_cutoff_idx,
        )
        reference_source = "auto"
    if target_mz_values:
        target_set = {int(round(v)) for v in target_mz_values}
        reference_peaks = [p for p in reference_peaks if int(round(p.mz)) in target_set]
    base_io = groups[0]["io"] if groups[0]["io"] > 0 else 1.0
    denominator = float(mass_discrimination)
    if denominator <= 0:
        raise ValueError("mass_discrimination must be positive")
    rows = []
    for group in groups:
        spectrum = group["spectrum"]
        for peak in reference_peaks:
            center_idx = int(round(peak.index))
            raw_area = baseline_corrected_area(spectrum.y, peak.left_bound, peak.right_bound)
            if prefer_gaussian:
                window_size = max(5, min(30, peak.right_bound - peak.left_bound + 5))
                fit = fit_gaussian(spectrum.y, center_idx, window_size)
                if fit is not None:
                    raw_area = gaussian_area(fit.amplitude, fit.fwhm)
            photon_normalized = raw_area
            if photon_normalize:
                photon_normalized = raw_area / group["io"] if group["io"] > 0 else 0.0
                if photon_reference_mode == "first":
                    photon_normalized *= base_io
                elif photon_reference_mode != "none":
                    raise ValueError("photon_reference_mode must be 'first' or 'none'")
            normalized = photon_normalized / denominator
            rows.append({
                "energy": group["energy"],
                "file_count": group["file_count"],
                "io": group["io"],
                "light_source": group["light_source"],
                "blank_file_count": group.get("blank_file_count", 0),
                "background_subtracted": bool(group.get("background_subtracted", False)),
                "reference_energy": reference_group["energy"],
                "reference_source": reference_source,
                "mz": peak.mz,
                "mz_rounded": int(round(peak.mz)),
                "species": peak.species,
                "photon_normalized_intensity": photon_normalized,
                "normalized_intensity": normalized,
                "raw_area": raw_area,
                "left_bound": peak.left_bound,
                "right_bound": peak.right_bound,
            })
    return pd.DataFrame(rows)


def build_pie_curves(analysis_df: pd.DataFrame) -> dict[int, dict]:
    """Convert PIE analysis rows into m/z keyed curve objects."""
    curves: dict[int, dict] = {}
    if analysis_df.empty:
        return curves
    working = analysis_df.copy()
    if "raw_area" not in working:
        working["raw_area"] = working["normalized_intensity"]
    if "photon_normalized_intensity" not in working:
        working["photon_normalized_intensity"] = working["normalized_intensity"]
    if "species" not in working:
        working["species"] = ""
    for mz, group in working.groupby("mz_rounded"):
        group = group.sort_values("energy").reset_index(drop=True)
        group["energy_rounded"] = group["energy"].round(2)
        ordered = group.groupby("energy_rounded", as_index=False).agg(
            energy=("energy", "mean"),
            normalized_intensity=("normalized_intensity", "mean"),
            raw_area=("raw_area", "mean"),
            photon_normalized_intensity=("photon_normalized_intensity", "mean"),
            species=("species", "first"),
            mz=("mz", "mean"),
            file_count=("file_count", "first"),
            io=("io", "first"),
        ).sort_values("energy")
        curves[int(mz)] = {
            "mz": int(mz),
            "mz_exact_mean": float(ordered["mz"].mean()),
            "species": str(ordered["species"].iloc[0]) if "species" in ordered else "",
            "energies": ordered["energy"].astype(float).tolist(),
            "intensities": ordered["normalized_intensity"].astype(float).tolist(),
            "rows": ordered,
        }
    return curves


def _solve_nonnegative_coefficients(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    try:
        from scipy.optimize import nnls
        coeffs, _ = nnls(design, target)
    except Exception:
        coeffs, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        coeffs = np.maximum(0, coeffs)
    return np.asarray(coeffs, dtype=float)


def _clean_coefficient_map(coefficients: dict[int, float] | None) -> dict[int, float]:
    cleaned: dict[int, float] = {}
    for key, value in (coefficients or {}).items():
        try:
            species_id = int(key)
            coefficient = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(coefficient):
            cleaned[species_id] = max(0.0, coefficient)
    return cleaned


def fit_species_combination_with_curve(
    species_list: list[dict],
    experimental_energies,
    experimental_intensities,
    *,
    coefficient_mode: str = "fit",
    coefficients: dict[int, float] | None = None,
    locked_species_ids: list[int] | None = None,
) -> dict:
    """Fit one experimental PIE curve using SQLite PICS records."""
    energies = np.asarray(experimental_energies, dtype=float)
    intensities = np.asarray(experimental_intensities, dtype=float)
    point_count = min(energies.size, intensities.size)
    energies = energies[:point_count]
    intensities = intensities[:point_count]
    valid = np.isfinite(energies) & np.isfinite(intensities)
    energies = energies[valid]
    intensities = intensities[valid]
    coefficient_mode = coefficient_mode if coefficient_mode in {"fit", "manual", "locked_fit"} else "fit"
    empty_model = {
        "energies": energies.tolist(),
        "experimental": intensities.tolist(),
        "fitted": [],
        "residuals": [],
        "r_squared": 0.0,
        "rmse": 0.0,
        "mae": 0.0,
        "candidate_count": len(species_list),
        "coefficient_mode": coefficient_mode,
        "locked_species_ids": [],
        "species": [],
    }
    if not species_list or energies.size == 0 or intensities.size == 0:
        return empty_model
    active_species: list[dict] = []
    active_species_ids: list[int] = []
    design_columns: list[np.ndarray] = []
    for species in species_list:
        pic_energies = np.asarray(species["energies"], dtype=float)
        pic_sections = np.asarray(species["cross_sections"], dtype=float)
        pic_count = min(pic_energies.size, pic_sections.size)
        pic_energies = pic_energies[:pic_count]
        pic_sections = pic_sections[:pic_count]
        pic_valid = np.isfinite(pic_energies) & np.isfinite(pic_sections)
        pic_energies = pic_energies[pic_valid]
        pic_sections = pic_sections[pic_valid]
        if pic_energies.size < 2:
            continue
        order = np.argsort(pic_energies)
        basis = np.interp(
            energies, pic_energies[order], pic_sections[order], left=0.0, right=0.0
        )
        basis = np.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(basis > 0):
            active_species.append(species)
            active_species_ids.append(int(species.get("id", len(active_species_ids) + 1)))
            design_columns.append(basis)
    if not design_columns:
        return empty_model
    design = np.column_stack(design_columns)
    coefficient_map = _clean_coefficient_map(coefficients)
    locked_ids = {int(value) for value in (locked_species_ids or [])}
    coeffs = np.zeros(len(active_species), dtype=float)
    if coefficient_mode == "manual":
        coeffs = np.asarray([
            coefficient_map.get(species_id, 0.0) for species_id in active_species_ids
        ], dtype=float)
    elif coefficient_mode == "locked_fit":
        locked_indices = [idx for idx, sid in enumerate(active_species_ids) if sid in locked_ids]
        free_indices = [idx for idx, sid in enumerate(active_species_ids) if sid not in locked_ids]
        for idx in locked_indices:
            coeffs[idx] = coefficient_map.get(active_species_ids[idx], 0.0)
        residual_target = intensities - design[:, locked_indices] @ coeffs[locked_indices] if locked_indices else intensities
        if free_indices:
            coeffs[free_indices] = _solve_nonnegative_coefficients(design[:, free_indices], residual_target)
            for idx in locked_indices:
                if coeffs[idx] <= 0 and np.any(design[:, idx] > 0):
                    coeffs[idx] = np.finfo(float).eps
    else:
        coeffs = _solve_nonnegative_coefficients(design, intensities)
    fitted = design @ coeffs
    residuals = intensities - fitted
    ss_tot = float(np.sum((intensities - np.mean(intensities)) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean(residuals ** 2)) if residuals.size else 0.0)
    mae = float(np.mean(np.abs(residuals)) if residuals.size else 0.0)
    fitted_area = float(np.sum(fitted))
    merged: dict[tuple[str, float | None], dict] = {}
    for idx, species in enumerate(active_species):
        if coeffs[idx] > 0.001 or active_species_ids[idx] in locked_ids:
            component = design[:, idx] * coeffs[idx]
            key = (species["species"], species.get("ie"))
            if key not in merged:
                merged[key] = {
                    "mz": int(species["mz"]),
                    "ids": [],
                    "coefficients_by_id": {},
                    "species": species["species"],
                    "ie": species.get("ie"),
                    "formula": species.get("formula"),
                    "elements": species.get("elements"),
                    "smiles": species.get("smiles"),
                    "coefficient": 0.0,
                    "component_area": 0.0,
                    "component_intensities": np.zeros_like(component),
                    "r_squared": float(r_squared),
                }
            merged[key]["ids"].append(active_species_ids[idx])
            merged[key]["coefficients_by_id"][active_species_ids[idx]] = float(coeffs[idx])
            merged[key]["coefficient"] += float(coeffs[idx])
            merged[key]["component_area"] += float(np.sum(component))
            merged[key]["component_intensities"] = merged[key]["component_intensities"] + component
    results = []
    for item in merged.values():
        contribution = 100.0 * item["component_area"] / fitted_area if fitted_area > 0 else 0.0
        results.append({
            "mz": item["mz"],
            "ids": item["ids"],
            "coefficients_by_id": item["coefficients_by_id"],
            "species": item["species"],
            "ie": item["ie"],
            "formula": item.get("formula"),
            "elements": item.get("elements"),
            "smiles": item.get("smiles"),
            "coefficient": item["coefficient"],
            "contribution_percent": contribution,
            "r_squared": item["r_squared"],
            "component_intensities": item["component_intensities"].tolist(),
        })
    results = sorted(results, key=lambda x: -x["coefficient"])
    return {
        "energies": energies.tolist(),
        "experimental": intensities.tolist(),
        "fitted": fitted.tolist(),
        "residuals": residuals.tolist(),
        "r_squared": float(r_squared),
        "rmse": rmse,
        "mae": mae,
        "candidate_count": len(active_species),
        "coefficient_mode": coefficient_mode,
        "locked_species_ids": sorted(locked_ids & set(active_species_ids)),
        "species": results,
    }


def fit_species_combination(species_list: list[dict], experimental_energies, experimental_intensities) -> list[dict]:
    model = fit_species_combination_with_curve(species_list, experimental_energies, experimental_intensities)
    return [{
        k: v for k, v in item.items() if k not in {"component_intensities"}
    } for item in model["species"]]


def identify_species_for_mz_with_curve(
    database: list[dict],
    mz: int,
    energies,
    intensities,
    *,
    forced_species: list[str] | None = None,
) -> dict:
    candidates = [item for item in database if item["mz"] == int(mz)]
    forced_names = {name for name in (forced_species or []) if name}
    if forced_names:
        locked_ids = [
            int(item.get("id", index + 1))
            for index, item in enumerate(candidates)
            if item.get("species") in forced_names
        ]
        return fit_species_combination_with_curve(
            candidates,
            energies,
            intensities,
            coefficient_mode="locked_fit",
            coefficients={species_id: np.finfo(float).eps for species_id in locked_ids},
            locked_species_ids=locked_ids,
        )
    return fit_species_combination_with_curve(candidates, energies, intensities)


def _empty_pie_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "energy", "file_count", "io", "light_source", "mz", "mz_rounded",
        "species", "photon_normalized_intensity", "normalized_intensity",
        "raw_area", "left_bound", "right_bound", "blank_file_count",
        "background_subtracted",
    ])


def _aggregate_segment_by_energy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    seg_df = df.sort_values("energy").reset_index(drop=True).copy()
    seg_df["energy_rounded"] = seg_df["energy"].round(2)
    agg_spec = {
        "energy": ("energy", "mean"),
        "mz_rounded": ("mz_rounded", "first"),
        "normalized_intensity": ("normalized_intensity", "mean"),
        "raw_area": ("raw_area", "mean"),
        "photon_normalized_intensity": ("photon_normalized_intensity", "mean"),
        "mz": ("mz", "mean"),
        "species": ("species", "first"),
        "file_count": ("file_count", "first"),
        "io": ("io", "first"),
        "light_source": ("light_source", "first"),
        "left_bound": ("left_bound", "first"),
        "right_bound": ("right_bound", "first"),
    }
    for optional in (
        "source_folder_idx",
        "source_folder",
        "reference_energy",
        "reference_source",
        "blank_file_count",
        "background_subtracted",
    ):
        if optional in seg_df.columns:
            agg_spec[optional] = (optional, "first")
    return seg_df.groupby("energy_rounded", as_index=False).agg(**agg_spec)


def _scale_factor_from_overlap(ref_df: pd.DataFrame, seg_df: pd.DataFrame) -> float | None:
    ref_energies = set(ref_df["energy_rounded"].values)
    seg_energies = set(seg_df["energy_rounded"].values)
    overlapping_energies = ref_energies & seg_energies
    if not overlapping_energies:
        return None
    merged_overlap = pd.merge(
        ref_df[ref_df["energy_rounded"].isin(overlapping_energies)][["energy_rounded", "normalized_intensity"]],
        seg_df[seg_df["energy_rounded"].isin(overlapping_energies)][["energy_rounded", "normalized_intensity"]],
        on="energy_rounded",
        suffixes=("_ref", "_seg"),
    )
    if merged_overlap.empty:
        return None
    ref_intensities = merged_overlap["normalized_intensity_ref"].values
    seg_intensities = merged_overlap["normalized_intensity_seg"].values
    valid_mask = (seg_intensities > 0) & (ref_intensities > 0)
    if np.sum(valid_mask) == 0:
        return None
    return float(np.median(ref_intensities[valid_mask] / seg_intensities[valid_mask]))


def merge_pie_segments(
    analysis_dfs: list[pd.DataFrame],
    *,
    merge_method: str = "low_energy_dominant",
) -> pd.DataFrame:
    """Merge PIE analysis DataFrames from multiple energy segments.

    Segments are scaled in energy order. For 3+ segments, each unscaled segment
    is matched against an already-scaled overlapping segment, so chained overlaps
    (A overlaps B, B overlaps C) are handled even when C does not overlap A.
    """
    if not analysis_dfs:
        return _empty_pie_dataframe()
    if len(analysis_dfs) == 1:
        return analysis_dfs[0].copy()

    all_mz: set[int] = set()
    for df in analysis_dfs:
        if not df.empty and "mz_rounded" in df.columns:
            all_mz.update(df["mz_rounded"].dropna().astype(int).unique())
    if not all_mz:
        return _empty_pie_dataframe()

    merged_rows: list[dict] = []
    for target_mz in sorted(all_mz):
        mz_segments = []
        for df_idx, df in enumerate(analysis_dfs):
            if df.empty or "mz_rounded" not in df.columns:
                continue
            seg_df = df[df["mz_rounded"] == target_mz].copy()
            if seg_df.empty:
                continue
            seg_df = _aggregate_segment_by_energy(seg_df)
            mz_segments.append({
                "df": seg_df,
                "min_energy": float(seg_df["energy"].min()),
                "max_energy": float(seg_df["energy"].max()),
                "segment_idx": df_idx,
            })

        if not mz_segments:
            continue
        if len(mz_segments) == 1:
            merged_rows.extend(mz_segments[0]["df"].drop(columns=["energy_rounded"], errors="ignore").to_dict("records"))
            continue

        mz_segments.sort(key=lambda x: x["min_energy"])

        if merge_method == "mean":
            combined = pd.concat([s["df"] for s in mz_segments], ignore_index=True)
            final_combined = _aggregate_segment_by_energy(combined)
            merged_rows.extend(final_combined.drop(columns=["energy_rounded"], errors="ignore").to_dict("records"))
            continue

        if merge_method == "first_segment_dominant":
            first = next((s for s in mz_segments if s["segment_idx"] == 0), mz_segments[0])
            remaining = [s for s in mz_segments if s is not first]
            ordered_segments = [first] + sorted(remaining, key=lambda x: x["min_energy"])
        else:
            ordered_segments = mz_segments

        scaled_segments: list[pd.DataFrame] = []
        for segment in ordered_segments:
            seg_df = segment["df"].copy()
            if not scaled_segments:
                seg_df["source_segment"] = segment["segment_idx"]
                seg_df["scale_factor"] = 1.0
                scaled_segments.append(seg_df)
                continue

            best_scale: float | None = None
            best_overlap_count = -1
            for ref_df in scaled_segments:
                overlap_count = len(set(ref_df["energy_rounded"].values) & set(seg_df["energy_rounded"].values))
                if overlap_count <= 0:
                    continue
                candidate = _scale_factor_from_overlap(ref_df, seg_df)
                if candidate is not None and overlap_count > best_overlap_count:
                    best_scale = candidate
                    best_overlap_count = overlap_count

            scale_factor = 1.0 if best_scale is None else best_scale
            if best_scale is None:
                logger.debug("No overlap found for m/z %s segment %s; leaving scale factor at 1", target_mz, segment["segment_idx"])
            for column in ("normalized_intensity", "photon_normalized_intensity", "raw_area"):
                if column in seg_df.columns:
                    seg_df[column] *= scale_factor
            seg_df["source_segment"] = segment["segment_idx"]
            seg_df["scale_factor"] = scale_factor
            scaled_segments.append(seg_df)

        combined = pd.concat(scaled_segments, ignore_index=True)
        final_combined = _aggregate_segment_by_energy(combined)
        merged_rows.extend(final_combined.drop(columns=["energy_rounded"], errors="ignore").to_dict("records"))

    if not merged_rows:
        return _empty_pie_dataframe()
    return pd.DataFrame(merged_rows).sort_values(["mz_rounded", "energy"]).reset_index(drop=True)


def analyze_multiple_pie_folders(
    folders: list[str | Path],
    *,
    calibration: Calibration = Calibration(),
    suffixes: tuple[str, ...] = (".txt", ".asc", ".888"),
    recursive: bool = True,
    energy_decimals: int = 1,
    algorithm: str = "legacy",
    threshold_end: float = 2,
    min_intensity: float = 3,
    detection_min_idx: int = 3000,
    nearby_peak_window: int = 30,
    duplicate_window: int = 20,
    weak_tail_early_window: int = 90,
    weak_tail_late_window: int = 50,
    weak_tail_ratio: float = 5,
    gaussian_window_max: int = 30,
    gaussian_boundary_scale: float = 1.5,
    boundary_padding: int = 2,
    prominence_ratio: float = 0.005,
    smoothing_window: int = 5,
    smoothing_poly_order: int = 2,
    baseline_window: int = 301,
    baseline_percentile: float = 5.0,
    min_peak_width: int = 1,
    max_peak_width: int = 80,
    prefer_gaussian: bool = True,
    manual_peak_path: str | Path | None = None,
    photon_normalize: bool = True,
    photon_reference_mode: str = "first",
    mass_discrimination: float = 1.0,
    light_source: str = "io",
    merge_method: str = "low_energy_dominant",
    vote_threshold: float = 0.667,
    min_intensity_for_single_vote: float = 5.0,
    mz_tolerance: float = 0.2,
    cwt_snr_threshold: float = 0.02,
    cwt_wavelet_max_width: int = 30,
    weak_tail_cutoff_idx: int = 15000,
) -> pd.DataFrame:
    """Analyze multiple PIE folders and merge them with overlap scaling."""
    all_spectra_by_folder: list[tuple[int, Path, list[dict]]] = []
    all_spectra_flat: list[dict] = []

    for folder_idx, folder in enumerate(folders):
        folder = Path(folder)
        groups = _group_spectra_by_energy(
            folder,
            suffixes=suffixes,
            recursive=recursive,
            energy_decimals=energy_decimals,
            light_source=light_source,
        )
        all_spectra_by_folder.append((folder_idx, folder, groups))
        all_spectra_flat.extend(groups)

    if not all_spectra_flat:
        return _empty_pie_dataframe()

    ref_group = max(all_spectra_flat, key=lambda g: g["energy"])
    if manual_peak_path:
        reference_peaks = peak_ranges_to_peaks(load_peak_ranges(manual_peak_path, calibration=calibration))
        reference_source = Path(manual_peak_path).name
    else:
        reference_peaks = detect_peaks_by_algorithm(
            ref_group["spectrum"].y,
            algorithm=algorithm,
            calibration=calibration,
            start_idx=0,
            end_idx=len(ref_group["spectrum"].y),
            detection_min_idx=detection_min_idx,
            threshold_end=threshold_end,
            min_intensity=min_intensity,
            nearby_peak_window=nearby_peak_window,
            duplicate_window=duplicate_window,
            weak_tail_early_window=weak_tail_early_window,
            weak_tail_late_window=weak_tail_late_window,
            weak_tail_ratio=weak_tail_ratio,
            gaussian_window_max=gaussian_window_max,
            gaussian_boundary_scale=gaussian_boundary_scale,
            boundary_padding=boundary_padding,
            prominence_ratio=prominence_ratio,
            smoothing_window=smoothing_window,
            smoothing_poly_order=smoothing_poly_order,
            baseline_window=baseline_window,
            baseline_percentile=baseline_percentile,
            min_peak_width=min_peak_width,
            max_peak_width=max_peak_width,
            vote_threshold=vote_threshold,
            min_intensity_for_single_vote=min_intensity_for_single_vote,
            mz_tolerance=mz_tolerance,
            cwt_snr_threshold=cwt_snr_threshold,
            cwt_wavelet_max_width=cwt_wavelet_max_width,
            weak_tail_cutoff_idx=weak_tail_cutoff_idx,
        )
        reference_source = "auto"

    all_groups_sorted = sorted(all_spectra_flat, key=lambda g: g["energy"])
    base_io = all_groups_sorted[0]["io"] if all_groups_sorted[0]["io"] > 0 else 1.0
    denominator = float(mass_discrimination)
    if denominator <= 0:
        raise ValueError("mass_discrimination must be positive")

    all_rows = []
    for folder_idx, folder, groups in all_spectra_by_folder:
        for group in groups:
            spectrum = group["spectrum"]
            energy = group["energy"]
            io_current = group["io"]
            file_count = group["file_count"]

            for peak in reference_peaks:
                raw_area = baseline_corrected_area(spectrum.y, peak.left_bound, peak.right_bound)
                if prefer_gaussian:
                    window_size = max(5, min(30, peak.right_bound - peak.left_bound + 5))
                    fit = fit_gaussian(spectrum.y, int(round(peak.index)), window_size)
                    if fit is not None:
                        raw_area = gaussian_area(fit.amplitude, fit.fwhm)

                photon_normalized = raw_area
                if photon_normalize:
                    photon_normalized = raw_area / io_current if io_current > 0 else 0.0
                    if photon_reference_mode == "first":
                        photon_normalized *= base_io
                normalized = photon_normalized / denominator

                all_rows.append({
                    "energy": energy,
                    "file_count": file_count,
                    "io": io_current,
                    "light_source": light_source,
                    "blank_file_count": group.get("blank_file_count", 0),
                    "background_subtracted": bool(group.get("background_subtracted", False)),
                    "reference_energy": ref_group["energy"],
                    "reference_source": reference_source,
                    "source_folder_idx": folder_idx,
                    "source_folder": str(folder),
                    "mz": peak.mz,
                    "mz_rounded": int(round(peak.mz)),
                    "species": peak.species,
                    "photon_normalized_intensity": photon_normalized,
                    "normalized_intensity": normalized,
                    "raw_area": raw_area,
                    "left_bound": peak.left_bound,
                    "right_bound": peak.right_bound,
                })

    if not all_rows:
        return _empty_pie_dataframe()

    temp_df = pd.DataFrame(all_rows)
    analysis_dfs = [
        temp_df[temp_df["source_folder_idx"] == folder_idx].copy().reset_index(drop=True)
        for folder_idx, _folder, _groups in all_spectra_by_folder
    ]
    return merge_pie_segments(analysis_dfs, merge_method=merge_method)

def identify_species_for_mz(database: list[dict], mz: int, energies, intensities) -> list[dict]:
    model = identify_species_for_mz_with_curve(database, mz, energies, intensities)
    return [{
        k: v for k, v in item.items() if k not in {"component_intensities"}
    } for item in model["species"]]
