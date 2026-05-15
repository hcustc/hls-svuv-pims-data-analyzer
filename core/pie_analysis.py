from __future__ import annotations

from pathlib import Path
import re
import sqlite3

import numpy as np
import pandas as pd

from .calibration import Calibration
from .integration import baseline_corrected_area, gaussian_area
from .normalization import extract_light_intensity
from .peak_ranges import load_peak_ranges, peak_ranges_to_peaks
from .peak_detection import detect_peaks_in_range, fit_gaussian
from .spectrum_io import Spectrum, extract_first_number, read_spectrum


EV_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*eV", re.IGNORECASE)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS species (
    id INTEGER PRIMARY KEY,
    mz INTEGER NOT NULL,
    name TEXT NOT NULL,
    ionization_energy REAL
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
                "INSERT INTO species (id, mz, name, ionization_energy) VALUES (?, ?, ?, ?)",
                (species_id, int(item["mz"]), str(item["species"]), item.get("ie")),
            )
            energies = np.asarray(item["energies"], dtype=float)
            cross_sections = np.asarray(item["cross_sections"], dtype=float)
            conn.executemany(
                "INSERT INTO pic_cross_sections (species_id, energy_ev, cross_section) VALUES (?, ?, ?)",
                [
                    (species_id, float(energy), float(cross_section))
                    for energy, cross_section in zip(energies, cross_sections)
                ],
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
            "SELECT id, mz, name, ionization_energy FROM species ORDER BY id"
        ).fetchall()
        for species_row in species_rows:
            pic_rows = conn.execute(
                """
                SELECT energy_ev, cross_section
                FROM pic_cross_sections
                WHERE species_id = ?
                ORDER BY energy_ev
                """,
                (species_row["id"],),
            ).fetchall()
            if not pic_rows:
                continue
            index = len(database)
            mz = int(species_row["mz"])
            database.append(
                {
                    "mz": mz,
                    "species": species_row["name"],
                    "ie": species_row["ionization_energy"],
                    "energies": np.array([row["energy_ev"] for row in pic_rows], dtype=np.float32),
                    "cross_sections": np.array([row["cross_section"] for row in pic_rows], dtype=np.float32),
                }
            )
            mz_index.setdefault(mz, []).append(index)
    return database, mz_index


def query_species_by_mz(path: str | Path, mz: int) -> list[dict]:
    """Query candidate species records for one integer m/z from SQLite."""
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return [
            {
                "id": row["id"],
                "mz": row["mz"],
                "species": row["name"],
                "ie": row["ionization_energy"],
            }
            for row in conn.execute(
                "SELECT id, mz, name, ionization_energy FROM species WHERE mz = ? ORDER BY id",
                (int(mz),),
            )
        ]


def extract_photon_energy(metadata_lines: list[str], path: str | Path | None = None, fallback: float | None = None) -> float:
    """Extract nominal photon energy for PIE grouping."""
    if path is not None:
        path_obj = Path(path)
        for part in reversed(path_obj.parts):
            match = EV_RE.search(part)
            if match:
                return float(match.group(1))

    for line in metadata_lines:
        if "energy" in line.lower():
            value = extract_first_number(line)
            if value is not None:
                return value

    if fallback is not None:
        return float(fallback)
    raise ValueError("cannot extract photon energy")


def extract_io_current(metadata_lines: list[str], fallback: float = 1.0) -> float:
    """Extract IO current from spectrum header metadata."""
    return extract_light_intensity(metadata_lines, "io", fallback)


def _iter_pie_files(folder: str | Path, suffixes: tuple[str, ...], recursive: bool) -> list[Path]:
    folder = Path(folder)
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in suffixes)


def _group_spectra_by_energy(
    folder: str | Path,
    *,
    suffixes: tuple[str, ...],
    recursive: bool,
    energy_decimals: int,
    light_source: str,
) -> list[dict]:
    grouped: dict[float, list[tuple[Path, Spectrum, float, float]]] = {}
    files = _iter_pie_files(folder, suffixes, recursive)
    for idx, path in enumerate(files):
        spectrum = read_spectrum(path, header_lines=10 if path.suffix.lower() == ".txt" else None, trim_start=0)
        if len(spectrum.y) == 0:
            continue
        energy = extract_photon_energy(spectrum.metadata_lines, path, fallback=float(idx))
        energy_key = round(energy, energy_decimals)
        io_current = extract_light_intensity(spectrum.metadata_lines, light_source)
        grouped.setdefault(energy_key, []).append((path, spectrum, energy, io_current))

    groups = []
    for energy_key, items in grouped.items():
        min_len = min(len(item[1].y) for item in items)
        y_sum = np.sum([item[1].y[:min_len] for item in items], axis=0)
        groups.append(
            {
                "energy": float(np.mean([item[2] for item in items])),
                "energy_key": energy_key,
                "io": float(np.mean([item[3] for item in items])),
                "light_source": light_source,
                "file_count": len(items),
                "files": [item[0].name for item in items],
                "spectrum": Spectrum(
                    x=np.arange(1, min_len + 1, dtype=float),
                    y=y_sum,
                    metadata_lines=[],
                    path=str(folder),
                ),
            }
        )
    return sorted(groups, key=lambda item: item["energy"])


def analyze_pie_folder(
    folder: str | Path,
    *,
    calibration: Calibration = Calibration(),
    suffixes: tuple[str, ...] = (".txt", ".asc", ".888"),
    recursive: bool = True,
    energy_decimals: int = 1,
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
    prefer_gaussian: bool = True,
    manual_peak_path: str | Path | None = None,
    photon_normalize: bool = True,
    photon_reference_mode: str = "first",
    mass_discrimination: float = 1.0,
    light_source: str = "io",
    target_mz_values: list[int] | None = None,
) -> pd.DataFrame:
    """Generate experimental PIE curves from a folder of energy-resolved spectra.

    Files are grouped by nominal photon energy, spectra at the same energy are
    summed, reference peaks are detected on the highest-energy group, and all
    energy groups are integrated using those reference bounds. TXT, ASC, and
    text-like 888 files are considered by default. Intensities are normalized by
    IO current to the lowest-energy group's IO when IO metadata is available.
    """
    groups = _group_spectra_by_energy(
        folder,
        suffixes=suffixes,
        recursive=recursive,
        energy_decimals=energy_decimals,
        light_source=light_source,
    )
    if not groups:
        return pd.DataFrame(
            columns=[
                "energy",
                "file_count",
                "io",
                "light_source",
                "mz",
                "mz_rounded",
                "species",
                "photon_normalized_intensity",
                "normalized_intensity",
                "raw_area",
                "left_bound",
                "right_bound",
            ]
        )

    if manual_peak_path:
        reference_group = max(groups, key=lambda item: item["energy"])
        reference_peaks = peak_ranges_to_peaks(load_peak_ranges(manual_peak_path, calibration=calibration))
        reference_source = Path(manual_peak_path).name
    else:
        reference_group = max(groups, key=lambda item: item["energy"])
        reference_spectrum = reference_group["spectrum"]
        reference_peaks = detect_peaks_in_range(
            reference_spectrum.y,
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
        )
        reference_source = "auto"

    if target_mz_values:
        target_set = {int(round(value)) for value in target_mz_values}
        reference_peaks = [
            peak for peak in reference_peaks if int(round(peak.mz)) in target_set
        ]

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
            rows.append(
                {
                    "energy": group["energy"],
                    "file_count": group["file_count"],
                    "io": group["io"],
                    "light_source": group["light_source"],
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
                }
            )
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
        ordered = (
            group.groupby("energy", as_index=False)
            .agg(
                normalized_intensity=("normalized_intensity", "sum"),
                raw_area=("raw_area", "sum"),
                photon_normalized_intensity=("photon_normalized_intensity", "sum"),
                species=("species", "first"),
                mz=("mz", "mean"),
                file_count=("file_count", "first"),
                io=("io", "first"),
            )
            .sort_values("energy")
        )
        curves[int(mz)] = {
            "mz": int(mz),
            "mz_exact_mean": float(ordered["mz"].mean()),
            "species": str(ordered["species"].iloc[0]) if "species" in ordered else "",
            "energies": ordered["energy"].astype(float).tolist(),
            "intensities": ordered["normalized_intensity"].astype(float).tolist(),
            "rows": ordered,
        }
    return curves


def fit_species_combination_with_curve(species_list: list[dict], experimental_energies, experimental_intensities) -> dict:
    """Fit one experimental PIE curve using SQLite PICS records.

    The database stores cross sections on each species' native photon-energy
    grid. For fitting, each candidate PICS curve is interpolated onto the
    experimental energy grid and solved as a non-negative linear combination.
    """
    energies = np.asarray(experimental_energies, dtype=float)
    intensities = np.asarray(experimental_intensities, dtype=float)
    point_count = min(energies.size, intensities.size)
    energies = energies[:point_count]
    intensities = intensities[:point_count]
    valid = np.isfinite(energies) & np.isfinite(intensities)
    energies = energies[valid]
    intensities = intensities[valid]
    empty_model = {
        "energies": energies.tolist(),
        "experimental": intensities.tolist(),
        "fitted": [],
        "r_squared": 0.0,
        "candidate_count": len(species_list),
        "species": [],
    }
    if not species_list or energies.size == 0 or intensities.size == 0:
        return empty_model

    active_species: list[dict] = []
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
            energies,
            pic_energies[order],
            pic_sections[order],
            left=0.0,
            right=0.0,
        )
        basis = np.nan_to_num(basis, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(basis > 0):
            active_species.append(species)
            design_columns.append(basis)

    if not design_columns:
        return empty_model

    design = np.column_stack(design_columns)
    try:
        from scipy.optimize import nnls

        coeffs, _ = nnls(design, intensities)
    except Exception:
        coeffs, _, _, _ = np.linalg.lstsq(design, intensities, rcond=None)
        coeffs = np.maximum(0, coeffs)
    fitted = design @ coeffs
    ss_tot = float(np.sum((intensities - np.mean(intensities)) ** 2))
    ss_res = float(np.sum((intensities - fitted) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    fitted_area = float(np.sum(fitted))
    merged: dict[tuple[str, float | None], dict] = {}
    for idx, species in enumerate(active_species):
        if coeffs[idx] > 0.001:
            component = design[:, idx] * coeffs[idx]
            key = (species["species"], species.get("ie"))
            if key not in merged:
                merged[key] = {
                    "mz": int(species["mz"]),
                    "species": species["species"],
                    "ie": species.get("ie"),
                    "coefficient": 0.0,
                    "component_area": 0.0,
                    "component_intensities": np.zeros_like(component),
                    "r_squared": float(r_squared),
                }
            merged[key]["coefficient"] += float(coeffs[idx])
            merged[key]["component_area"] += float(np.sum(component))
            merged[key]["component_intensities"] = merged[key]["component_intensities"] + component

    results = []
    for item in merged.values():
        contribution = 100.0 * item["component_area"] / fitted_area if fitted_area > 0 else 0.0
        results.append(
            {
                "mz": item["mz"],
                "species": item["species"],
                "ie": item["ie"],
                "coefficient": item["coefficient"],
                "contribution_percent": contribution,
                "r_squared": item["r_squared"],
                "component_intensities": item["component_intensities"].tolist(),
            }
        )
    results = sorted(results, key=lambda item: -item["coefficient"])
    return {
        "energies": energies.tolist(),
        "experimental": intensities.tolist(),
        "fitted": fitted.tolist(),
        "r_squared": float(r_squared),
        "candidate_count": len(active_species),
        "species": results,
    }


def fit_species_combination(species_list: list[dict], experimental_energies, experimental_intensities) -> list[dict]:
    model = fit_species_combination_with_curve(species_list, experimental_energies, experimental_intensities)
    return [
        {
            key: value
            for key, value in item.items()
            if key not in {"component_intensities"}
        }
        for item in model["species"]
    ]


def identify_species_for_mz_with_curve(database: list[dict], mz: int, energies, intensities) -> dict:
    return fit_species_combination_with_curve(
        [item for item in database if item["mz"] == int(mz)],
        energies,
        intensities,
    )


def identify_species_for_mz(database: list[dict], mz: int, energies, intensities) -> list[dict]:
    model = identify_species_for_mz_with_curve(database, mz, energies, intensities)
    return [
        {
            key: value
            for key, value in item.items()
            if key not in {"component_intensities"}
        }
        for item in model["species"]
    ]
