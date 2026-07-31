from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re
import sqlite3

import numpy as np
import pandas as pd

from .calibration import Calibration
from .integration import integrate_peak_with_method, resolve_integration_method
from .isotope_correction import (
    DEFAULT_ISOTOPE_QC_PAIRS,
    evaluate_isotope_qc_pairs,
)
from .normalization import extract_acquisition_time_s, extract_light_intensity
from .peak_ranges import load_peak_ranges, peak_ranges_to_peaks
from .peak_detection import detect_peaks_by_algorithm
from .spectrum_io import Spectrum, extract_first_number, find_filename_replicate_groups, filename_replicate_key, read_spectrum


logger = logging.getLogger(__name__)

EV_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*eV", re.IGNORECASE)
PIE_SPECTRUM_SUFFIXES = (".txt", ".asc", ".888")


@dataclass(frozen=True)
class PieSegmentSummary:
    """Lightweight description used by the PIE source selector."""

    folder: Path
    file_count: int
    min_energy: float | None = None
    max_energy: float | None = None


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
                    item.get("ie") if item.get("ie") is not None else item.get("ionization_energy"),
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
            ionization_energy = species_row["ionization_energy"]
            database.append({
                "id": int(species_row["id"]),
                "mz": mz,
                "species": species_row["name"],
                # Keep both names because the fitting core historically uses ``ie``
                # while the PyQt candidate model used ``ionization_energy``.
                "ie": ionization_energy,
                "ionization_energy": ionization_energy,
                "ie_source": "PICS数据库" if ionization_energy is not None else "",
                "ie_query_status": "available" if ionization_energy is not None else "not_queried",
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


def _looks_like_single_energy_directory(name: str) -> bool:
    """Return whether a directory label represents one energy point, not a segment range."""
    if re.fullmatch(r"\s*[-+]?\d+(?:\.\d+)?\s*(?:eV)?\s*", name, re.IGNORECASE):
        return True
    if "ev" not in name.lower():
        return False
    return len(re.findall(r"[-+]?\d+(?:\.\d+)?", name)) == 1


def discover_pie_segment_folders(
    folder: str | Path,
    *,
    suffixes: tuple[str, ...] = PIE_SPECTRUM_SUFFIXES,
) -> list[Path]:
    """Resolve a temporary PIE source into one dataset or several segment folders.

    A normal PIE dataset often contains one child directory per energy point
    (for example ``8.0eV/`` and ``8.1eV/``). Those must remain one recursive
    source. A parent whose immediate non-energy children each contain several
    spectrum files is treated as a multi-segment container.
    """
    root = Path(folder).expanduser()
    if not root.is_dir():
        return [root]

    direct_files = [
        path
        for path in root.iterdir()
        if path.is_file()
        and path.suffix.lower() in suffixes
        and not _is_blank_spectrum_path(path)
    ]
    if direct_files:
        return [root]

    candidates: list[tuple[Path, list[Path]]] = []
    for child in sorted(
        (path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")),
        key=lambda path: path.name.lower(),
    ):
        files = [
            path
            for path in _iter_pie_files(child, suffixes, recursive=True)
            if not _is_blank_spectrum_path(path)
        ]
        if files:
            candidates.append((child, files))

    if len(candidates) < 2:
        return [root]
    if all(_looks_like_single_energy_directory(child.name) for child, _files in candidates):
        return [root]
    if not all(len(files) >= 2 for _child, files in candidates):
        return [root]
    return [child for child, _files in candidates]


def _read_spectrum_header_preview(path: Path, *, max_lines: int = 32) -> list[str]:
    """Read only the header-sized prefix needed for energy-range previews."""
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            lines: list[str] = []
            with path.open("r", encoding=encoding) as handle:
                for index, line in enumerate(handle):
                    if index >= max_lines:
                        break
                    lines.append(line.strip())
            return [line for line in lines if line]
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return []


def inspect_pie_source_segments(
    folder: str | Path,
    *,
    suffixes: tuple[str, ...] = PIE_SPECTRUM_SUFFIXES,
) -> list[PieSegmentSummary]:
    """Discover selectable PIE segments and summarize their energy coverage.

    The operation only reads a short prefix from each spectrum file, so the GUI
    can show energy ranges without loading full spectra before analysis.
    """
    summaries: list[PieSegmentSummary] = []
    for segment_folder in discover_pie_segment_folders(folder, suffixes=suffixes):
        files = [
            path
            for path in _iter_pie_files(segment_folder, suffixes, recursive=True)
            if not _is_blank_spectrum_path(path)
        ]
        if not files:
            continue
        energies: list[float] = []
        for path in files:
            try:
                energies.append(
                    extract_photon_energy(_read_spectrum_header_preview(path), path)
                )
            except (OSError, UnicodeDecodeError, ValueError):
                continue
        summaries.append(
            PieSegmentSummary(
                folder=segment_folder,
                file_count=len(files),
                min_energy=min(energies) if energies else None,
                max_energy=max(energies) if energies else None,
            )
        )

    return sorted(
        summaries,
        key=lambda item: (
            item.min_energy is None,
            item.min_energy if item.min_energy is not None else float("inf"),
            item.folder.name.lower(),
        ),
    )


def _is_blank_spectrum_path(path: Path) -> bool:
    stem = path.stem.lower()
    if "blank" in stem or "background" in stem or "空白" in stem:
        return True
    blank_dir_names = {"blank", "blanks", "background", "backgrounds", "空白"}
    return any(part.lower() in blank_dir_names for part in path.parts[-3:-1])


def _aligned_x_values(items: list[tuple], min_len: int) -> np.ndarray:
    if not items:
        return np.arange(1, min_len + 1, dtype=float)
    reference_path, reference_spectrum, *_ = items[0]
    reference_x = np.asarray(reference_spectrum.x[:min_len], dtype=float)
    if reference_x.size != min_len:
        raise ValueError(f"spectrum x axis is shorter than y data: {reference_path}")
    for path, spectrum, *_ in items[1:]:
        x_values = np.asarray(spectrum.x[:min_len], dtype=float)
        if x_values.size != min_len or not np.allclose(x_values, reference_x, rtol=1e-7, atol=1e-9, equal_nan=True):
            raise ValueError(
                "PIE spectra in the same energy group have mismatched x axes: "
                f"{reference_path.name} vs {path.name}"
            )
    return reference_x


def _aggregate_spectrum_stack(
    values: list[np.ndarray],
    *,
    replicate_mode: str,
) -> np.ndarray:
    if not values:
        return np.array([], dtype=float)
    stack = np.asarray(values, dtype=float)
    if replicate_mode == "sum":
        return np.sum(stack, axis=0)
    return np.mean(stack, axis=0)


def _subtract_optional_background(
    sample_values: list[np.ndarray],
    blank_values: list[np.ndarray],
    *,
    replicate_mode: str,
) -> np.ndarray:
    result = _aggregate_spectrum_stack(
        sample_values,
        replicate_mode=replicate_mode,
    )
    if blank_values:
        result = result - _aggregate_spectrum_stack(
            blank_values,
            replicate_mode=replicate_mode,
        )
    return np.asarray(result, dtype=float)


def _group_spectra_by_energy(
    folder: str | Path,
    *,
    suffixes: tuple[str, ...],
    recursive: bool,
    energy_decimals: int,
    light_source: str,
    replicate_mode: str = "off",
    normalize_by_time: bool = True,
    normalize_by_light: bool = True,
) -> list[dict]:
    replicate_mode = "sum" if replicate_mode == "sum" else "mean" if replicate_mode == "mean" else "off"
    grouped: dict[object, list[tuple[Path, Spectrum, float, float, float | None]]] = {}
    blank_grouped: dict[float, list[tuple[Path, Spectrum, float, float, float | None]]] = {}
    files = _iter_pie_files(folder, suffixes, recursive)
    filename_replicates = find_filename_replicate_groups(files) if replicate_mode != "off" else {}
    filename_replicate_energy_keys: dict[tuple[Path, str], set[float]] = {}
    spectra: list[tuple[Path, Spectrum, float, float, float, float | None]] = []
    for path in files:
        spectrum = read_spectrum(path, header_lines=10 if path.suffix.lower() == ".txt" else None, trim_start=0)
        if len(spectrum.y) == 0:
            continue
        try:
            energy = extract_photon_energy(spectrum.metadata_lines, path)
        except ValueError:
            logger.warning("Skipping PIE spectrum with no photon energy: %s", path)
            continue
        energy_key = round(energy, energy_decimals)
        io_current = extract_light_intensity(
            spectrum.metadata_lines,
            light_source,
            fallback=float("nan"),
        )
        if normalize_by_light and (
            not np.isfinite(io_current) or float(io_current) <= 0
        ):
            raise ValueError(
                f"PIE谱缺少有效{light_source}，不能执行光强归一化：{path}"
            )
        acquisition_time_s = extract_acquisition_time_s(spectrum.metadata_lines)
        if normalize_by_time and (
            acquisition_time_s is None
            or not np.isfinite(acquisition_time_s)
            or acquisition_time_s <= 0
        ):
            raise ValueError(
                "PIE谱缺少有效扫描时间，已停止计算："
                f"{path}。只有在项目设置中明确关闭按扫描时间归一化后才能继续。"
            )
        spectra.append(
            (
                path,
                spectrum,
                energy,
                energy_key,
                float(io_current),
                acquisition_time_s,
            )
        )
        repeat_key = filename_replicate_key(path)
        if repeat_key is not None and repeat_key in filename_replicates:
            filename_replicate_energy_keys.setdefault(repeat_key, set()).add(energy_key)

    valid_filename_replicates = {
        key
        for key, energy_keys in filename_replicate_energy_keys.items()
        if len(energy_keys) <= 2
    }

    for path, spectrum, energy, energy_key, io_current, acquisition_time_s in spectra:
        if _is_blank_spectrum_path(path):
            blank_grouped.setdefault(energy_key, []).append(
                (path, spectrum, energy, io_current, acquisition_time_s)
            )
            continue
        if replicate_mode == "off":
            grouped.setdefault(("file", path), []).append(
                (path, spectrum, energy, io_current, acquisition_time_s)
            )
            continue
        repeat_key = filename_replicate_key(path)
        if repeat_key is not None and repeat_key in valid_filename_replicates:
            grouped.setdefault(("filename", repeat_key), []).append(
                (path, spectrum, energy, io_current, acquisition_time_s)
            )
        else:
            grouped.setdefault(("energy", energy_key), []).append(
                (path, spectrum, energy, io_current, acquisition_time_s)
            )
    groups = []
    for group_key, items in grouped.items():
        energy_key = round(float(np.mean([item[2] for item in items])), energy_decimals)
        blank_items = blank_grouped.get(energy_key, [])
        min_len = min(
            [len(item[1].y) for item in items]
            + [len(item[1].y) for item in blank_items]
        )
        x_values = _aligned_x_values(items + blank_items, min_len)
        raw_stack = [np.asarray(item[1].y[:min_len], dtype=float) for item in items]
        raw_blank_stack = [
            np.asarray(item[1].y[:min_len], dtype=float) for item in blank_items
        ]

        def normalization_denominators(
            source_items: list[tuple],
            *,
            by_time: bool,
            by_light: bool,
        ) -> list[float]:
            denominators: list[float] = []
            for item in source_items:
                denominator = 1.0
                acquisition_time = item[4]
                if by_time:
                    if (
                        acquisition_time is None
                        or not np.isfinite(acquisition_time)
                        or acquisition_time <= 0
                    ):
                        return []
                    denominator *= float(acquisition_time)
                if by_light:
                    current = float(item[3])
                    if not np.isfinite(current) or current <= 0:
                        return []
                    denominator *= current
                denominators.append(denominator)
            return denominators

        def normalized_stack(
            source_items: list[tuple],
            *,
            by_time: bool,
            by_light: bool,
        ) -> list[np.ndarray]:
            denominators = normalization_denominators(
                source_items,
                by_time=by_time,
                by_light=by_light,
            )
            if source_items and not denominators:
                return []
            raw_values = [
                np.asarray(item[1].y[:min_len], dtype=float)
                for item in source_items
            ]
            result = [
                values / denominator
                for values, denominator in zip(
                    raw_values,
                    denominators,
                    strict=True,
                )
            ]
            if (
                replicate_mode == "sum"
                and result
                and (by_time or by_light)
            ):
                total_exposure = float(np.sum(denominators))
                if not np.isfinite(total_exposure) or total_exposure <= 0:
                    return []
                return [
                    np.sum(np.asarray(raw_values, dtype=float), axis=0)
                    / total_exposure
                ]
            return result

        count_rate_stack = normalized_stack(
            items,
            by_time=True,
            by_light=False,
        )
        count_rate_blank_stack = normalized_stack(
            blank_items,
            by_time=True,
            by_light=False,
        )
        io_stack = normalized_stack(items, by_time=False, by_light=True)
        io_blank_stack = normalized_stack(
            blank_items,
            by_time=False,
            by_light=True,
        )
        io_time_stack = normalized_stack(items, by_time=True, by_light=True)
        io_time_blank_stack = normalized_stack(
            blank_items,
            by_time=True,
            by_light=True,
        )
        analysis_stack = normalized_stack(
            items,
            by_time=normalize_by_time,
            by_light=normalize_by_light,
        )
        analysis_blank_stack = normalized_stack(
            blank_items,
            by_time=normalize_by_time,
            by_light=normalize_by_light,
        )
        sample_analysis_denominators = normalization_denominators(
            items,
            by_time=normalize_by_time,
            by_light=normalize_by_light,
        )
        blank_analysis_denominators = normalization_denominators(
            blank_items,
            by_time=normalize_by_time,
            by_light=normalize_by_light,
        )
        y_value = _subtract_optional_background(
            analysis_stack,
            analysis_blank_stack,
            replicate_mode=replicate_mode,
        )
        raw_sample_y_value = _aggregate_spectrum_stack(
            raw_stack,
            replicate_mode=replicate_mode,
        )
        raw_blank_y_value = (
            _aggregate_spectrum_stack(
                raw_blank_stack,
                replicate_mode=replicate_mode,
            )
            if raw_blank_stack
            else np.zeros(min_len, dtype=float)
        )
        raw_y_value = raw_sample_y_value - raw_blank_y_value
        count_rate_y_value = (
            _subtract_optional_background(
                count_rate_stack,
                count_rate_blank_stack,
                replicate_mode=replicate_mode,
            )
            if count_rate_stack
            else np.full(min_len, np.nan, dtype=float)
        )
        io_y_value = (
            _subtract_optional_background(
                io_stack,
                io_blank_stack,
                replicate_mode=replicate_mode,
            )
            if io_stack
            else np.full(min_len, np.nan, dtype=float)
        )
        io_time_y_value = (
            _subtract_optional_background(
                io_time_stack,
                io_time_blank_stack,
                replicate_mode=replicate_mode,
            )
            if io_time_stack
            else np.full(min_len, np.nan, dtype=float)
        )
        blank_file_count = len(blank_items)
        uses_filename_grouping = bool(group_key[0] == "filename")
        warning = ""
        if replicate_mode != "off" and not uses_filename_grouping and len(items) > 1:
            warning = (
                "未识别到文件名末尾采集序号，已退回按能量分组的旧逻辑处理重复文件。"
            )
        acquisition_times = [
            float(item[4])
            for item in items
            if item[4] is not None and np.isfinite(item[4]) and item[4] > 0
        ]
        groups.append({
            "energy": float(np.mean([item[2] for item in items])),
            "energy_key": energy_key,
            "io": float(np.mean([item[3] for item in items])),
            "light_source": light_source,
            "file_count": len(items),
            "files": [
                (
                    item[0].relative_to(Path(folder)).as_posix()
                    if item[0].is_relative_to(Path(folder))
                    else item[0].name
                )
                for item in items
            ],
            "replicate_mode": replicate_mode,
            "replicate_grouping": "filename" if uses_filename_grouping else "energy",
            "replicate_warning": warning,
            "blank_file_count": blank_file_count,
            "blank_files": [item[0].name for item in blank_items],
            "background_subtracted": bool(blank_items),
            "acquisition_time_s": (
                float(np.mean(acquisition_times)) if acquisition_times else np.nan
            ),
            "total_acquisition_time_s": (
                float(np.sum(acquisition_times)) if acquisition_times else np.nan
            ),
            "acquisition_time_min_s": (
                float(np.min(acquisition_times)) if acquisition_times else np.nan
            ),
            "acquisition_time_max_s": (
                float(np.max(acquisition_times)) if acquisition_times else np.nan
            ),
            "acquisition_times_s": acquisition_times,
            "normalize_by_time": bool(normalize_by_time),
            "normalize_by_light": bool(normalize_by_light),
            "_raw_sample_values": raw_stack,
            "_raw_blank_values": raw_blank_stack,
            "_sample_analysis_denominators": sample_analysis_denominators,
            "_blank_analysis_denominators": blank_analysis_denominators,
            "spectrum": Spectrum(
                x=np.asarray(x_values, dtype=float),
                y=np.asarray(y_value, dtype=float),
                metadata_lines=[],
                path=str(folder),
            ),
            "raw_spectrum": Spectrum(
                x=np.asarray(x_values, dtype=float),
                y=np.asarray(raw_y_value, dtype=float),
                metadata_lines=[],
                path=str(folder),
            ),
            "raw_sample_spectrum": Spectrum(
                x=np.asarray(x_values, dtype=float),
                y=np.asarray(raw_sample_y_value, dtype=float),
                metadata_lines=[],
                path=str(folder),
            ),
            "raw_blank_spectrum": Spectrum(
                x=np.asarray(x_values, dtype=float),
                y=np.asarray(raw_blank_y_value, dtype=float),
                metadata_lines=[],
                path=str(folder),
            ),
            "count_rate_spectrum": Spectrum(
                x=np.asarray(x_values, dtype=float),
                y=np.asarray(count_rate_y_value, dtype=float),
                metadata_lines=[],
                path=str(folder),
            ),
            "io_normalized_spectrum": Spectrum(
                x=np.asarray(x_values, dtype=float),
                y=np.asarray(io_y_value, dtype=float),
                metadata_lines=[],
                path=str(folder),
            ),
            "io_time_normalized_spectrum": Spectrum(
                x=np.asarray(x_values, dtype=float),
                y=np.asarray(io_time_y_value, dtype=float),
                metadata_lines=[],
                path=str(folder),
            ),
        })
    return sorted(groups, key=lambda x: x["energy"])


def _poisson_peak_window_snr(group: dict, peak) -> float:
    """Estimate peak-window SNR from gross sample and blank counts.

    The configured integration method may remove a baseline or fit a Gaussian,
    neither of which is a Poisson count.  Noise is therefore estimated from
    summed detector counts in the same fixed peak window.  For averaged
    replicates, variance is propagated for the mean of independent counts.
    """
    replicate_mode = str(group.get("replicate_mode", "off"))
    normalization_applied = bool(
        group.get("normalize_by_time", False)
        or group.get("normalize_by_light", False)
    )

    def component(
        spectra: list[np.ndarray],
        denominators: list[float],
    ) -> tuple[float, float]:
        if not spectra:
            return 0.0, 0.0
        counts = np.asarray(
            [
                max(
                    float(
                        integrate_peak_with_method(
                            values,
                            peak,
                            prefer_gaussian=False,
                            integration_method="sum_counts",
                        )[0]
                    ),
                    0.0,
                )
                for values in spectra
            ],
            dtype=float,
        )
        exposure = np.asarray(denominators, dtype=float)
        if exposure.size != counts.size or np.any(exposure <= 0):
            return float("nan"), float("nan")
        if replicate_mode == "sum":
            if not normalization_applied:
                return float(np.sum(counts)), float(np.sum(counts))
            total_exposure = float(np.sum(exposure))
            return (
                float(np.sum(counts) / total_exposure),
                float(np.sum(counts) / total_exposure**2),
            )
        count = len(counts)
        signal = float(np.mean(counts / exposure))
        variance = float(np.sum(counts / exposure**2) / count**2)
        return signal, variance

    sample_signal, sample_variance = component(
        list(group.get("_raw_sample_values", [])),
        list(group.get("_sample_analysis_denominators", [])),
    )
    blank_signal, blank_variance = component(
        list(group.get("_raw_blank_values", [])),
        list(group.get("_blank_analysis_denominators", [])),
    )
    variance = sample_variance + blank_variance
    if not np.isfinite(variance) or variance <= 0:
        return float("nan")
    return float((sample_signal - blank_signal) / np.sqrt(variance))


def _summarize_integration_methods(values) -> str:
    methods = sorted({str(value) for value in values if str(value)})
    if not methods:
        return ""
    if len(methods) == 1:
        return methods[0]
    return "mixed"


def _merge_metadata_values(values) -> list:
    merged: list = []
    for value in values:
        if isinstance(value, (list, tuple, set, np.ndarray)):
            candidates = list(value)
        elif value is None or (
            isinstance(value, float) and not np.isfinite(value)
        ):
            candidates = []
        else:
            candidates = [value]
        for candidate in candidates:
            if candidate not in merged:
                merged.append(candidate)
    return merged


def normalize_integration_method(value: str | None, *, prefer_gaussian: bool | None = None) -> str:
    return resolve_integration_method(
        value,
        prefer_gaussian=bool(prefer_gaussian),
    )


def _analyze_pie_sources(
    folders: list[str | Path],
    *,
    calibration: Calibration,
    suffixes: tuple[str, ...],
    recursive: bool,
    energy_decimals: int,
    algorithm: str,
    threshold_end: float,
    min_intensity: float,
    detection_min_idx: int,
    nearby_peak_window: int,
    duplicate_window: int,
    weak_tail_early_window: int,
    weak_tail_late_window: int,
    weak_tail_ratio: float,
    gaussian_window_max: int,
    gaussian_boundary_scale: float,
    boundary_padding: int,
    prominence_ratio: float,
    smoothing_window: int,
    smoothing_poly_order: int,
    baseline_window: int,
    baseline_percentile: float,
    min_peak_width: int,
    max_peak_width: int,
    prefer_gaussian: bool,
    integration_method: str | None,
    manual_peak_path: str | Path | None,
    photon_normalize: bool,
    photon_reference_mode: str,
    normalize_by_time: bool,
    mass_discrimination: float,
    light_source: str,
    target_mz_values: list[int] | None,
    replicate_mode: str,
    vote_threshold: float,
    min_intensity_for_single_vote: float,
    mz_tolerance: float,
    cwt_snr_threshold: float,
    cwt_wavelet_max_width: int,
    weak_tail_cutoff_idx: int,
    merge_method: str | None,
    isotope_qc_pairs: list[dict] | None,
) -> pd.DataFrame:
    """Shared PIE analysis pipeline for one or many source folders."""
    grouped_sources: list[tuple[int, Path, list[dict]]] = []
    all_groups: list[dict] = []
    for folder_idx, folder_value in enumerate(folders):
        folder = Path(folder_value)
        groups = _group_spectra_by_energy(
            folder,
            suffixes=suffixes,
            recursive=recursive,
            energy_decimals=energy_decimals,
            light_source=light_source,
            replicate_mode=replicate_mode,
            normalize_by_time=normalize_by_time,
            normalize_by_light=photon_normalize,
        )
        grouped_sources.append((folder_idx, folder, groups))
        all_groups.extend(groups)

    if not all_groups:
        return _empty_pie_dataframe()

    reference_group = max(all_groups, key=lambda group: group["energy"])
    if manual_peak_path:
        reference_peaks = peak_ranges_to_peaks(
            load_peak_ranges(manual_peak_path, calibration=calibration)
        )
        reference_source = Path(manual_peak_path).name
    else:
        # Detection thresholds are defined in detector counts.  Run peak
        # finding on the unnormalized reference spectrum, then integrate all
        # provenance layers through the same fixed windows.
        reference_spectrum = reference_group["raw_spectrum"]
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
        target_set = {int(round(value)) for value in target_mz_values}
        reference_peaks = [
            peak for peak in reference_peaks
            if int(round(peak.mz)) in target_set
        ]

    if photon_normalize and photon_reference_mode not in {"first", "none"}:
        raise ValueError("photon_reference_mode must be 'first' or 'none'")
    denominator = float(mass_discrimination)
    if denominator <= 0:
        raise ValueError("mass_discrimination must be positive")
    ordered_groups = sorted(all_groups, key=lambda group: group["energy"])
    base_io = ordered_groups[0]["io"] if ordered_groups[0]["io"] > 0 else 1.0
    configured_method = normalize_integration_method(
        integration_method,
        prefer_gaussian=prefer_gaussian,
    )
    include_source = merge_method is not None
    rows: list[dict] = []
    for folder_idx, folder, groups in grouped_sources:
        for group in groups:
            for peak in reference_peaks:
                analysis_area, actual_method = integrate_peak_with_method(
                    group["spectrum"].y,
                    peak,
                    prefer_gaussian=prefer_gaussian,
                    integration_method=configured_method,
                )
                raw_area, _raw_method = integrate_peak_with_method(
                    group["raw_spectrum"].y,
                    peak,
                    prefer_gaussian=prefer_gaussian,
                    integration_method=configured_method,
                )

                def integrate_optional_spectrum(name: str) -> float:
                    values = np.asarray(group[name].y, dtype=float)
                    if not np.any(np.isfinite(values)):
                        return float("nan")
                    area, _method = integrate_peak_with_method(
                        values,
                        peak,
                        prefer_gaussian=prefer_gaussian,
                        integration_method=configured_method,
                    )
                    return float(area)

                count_rate = integrate_optional_spectrum("count_rate_spectrum")
                io_normalized = integrate_optional_spectrum(
                    "io_normalized_spectrum"
                )
                io_time_normalized = integrate_optional_spectrum(
                    "io_time_normalized_spectrum"
                )
                photon_normalized = float(analysis_area)
                if photon_normalize and photon_reference_mode == "first":
                    photon_normalized *= base_io
                normalized_intensity = photon_normalized / denominator
                signal_to_noise = _poisson_peak_window_snr(group, peak)
                row = {
                    "energy": group["energy"],
                    "file_count": group["file_count"],
                    "source_files": list(group.get("files", [])),
                    "source_spectra": [
                        str((Path(folder) / str(file_name)).resolve())
                        for file_name in group.get("files", [])
                    ],
                    "io": group["io"],
                    "light_source": group["light_source"],
                    "acquisition_time_s": group.get("acquisition_time_s", np.nan),
                    "total_acquisition_time_s": group.get(
                        "total_acquisition_time_s",
                        np.nan,
                    ),
                    "acquisition_time_min_s": group.get(
                        "acquisition_time_min_s",
                        np.nan,
                    ),
                    "acquisition_time_max_s": group.get(
                        "acquisition_time_max_s",
                        np.nan,
                    ),
                    "acquisition_times_s": list(
                        group.get("acquisition_times_s", [])
                    ),
                    "time_normalized": bool(normalize_by_time),
                    "io_normalized": bool(photon_normalize),
                    "replicate_mode": group.get("replicate_mode", replicate_mode),
                    "replicate_grouping": group.get("replicate_grouping", ""),
                    "replicate_warning": group.get("replicate_warning", ""),
                    "blank_file_count": group.get("blank_file_count", 0),
                    "background_subtracted": bool(group.get("background_subtracted", False)),
                    "reference_energy": reference_group["energy"],
                    "reference_source": reference_source,
                    "mz": peak.mz,
                    "mz_rounded": int(round(peak.mz)),
                    "species": peak.species,
                    "photon_normalized_intensity": photon_normalized,
                    "normalized_intensity": normalized_intensity,
                    "merged_intensity": normalized_intensity,
                    "raw_area": float(raw_area),
                    "count_rate": count_rate,
                    "io_normalized_intensity": io_normalized,
                    "io_time_normalized_intensity": io_time_normalized,
                    "signal_to_noise": signal_to_noise,
                    "signal_to_noise_method": "poisson_peak_window",
                    "integration_method": actual_method,
                    "left_bound": peak.left_bound,
                    "right_bound": peak.right_bound,
                }
                if include_source:
                    row["source_folder_idx"] = folder_idx
                    row["source_folder"] = str(folder)
                rows.append(row)

    if not rows:
        return _empty_pie_dataframe()
    analysis_df = pd.DataFrame(rows)
    if merge_method is None:
        return _attach_isotope_qc(
            analysis_df,
            isotope_qc_pairs=isotope_qc_pairs,
        )
    segment_frames = [
        analysis_df[analysis_df["source_folder_idx"] == folder_idx].copy().reset_index(drop=True)
        for folder_idx, _folder, _groups in grouped_sources
    ]
    merged = merge_pie_segments(segment_frames, merge_method=merge_method)
    return _attach_isotope_qc(
        merged,
        isotope_qc_pairs=isotope_qc_pairs,
    )


def _attach_isotope_qc(
    analysis_df: pd.DataFrame,
    *,
    isotope_qc_pairs: list[dict] | None,
) -> pd.DataFrame:
    result = analysis_df.copy()
    pairs = list(
        DEFAULT_ISOTOPE_QC_PAIRS
        if isotope_qc_pairs is None
        else isotope_qc_pairs
    )
    if result.empty or not pairs:
        result["isotope_qc_status"] = "not_evaluated"
        result.attrs.update(analysis_df.attrs)
        result.attrs["isotope_qc"] = []
        result.attrs["isotope_qc_status"] = "not_evaluated"
        return result
    signal_column = (
        "merged_intensity"
        if "merged_intensity" in result.columns
        else "normalized_intensity"
    )
    identity_columns = [
        column
        for column in ("left_bound", "right_bound")
        if column in result.columns
    ]
    collision_masses: list[int] = []
    qc_masses = {
        int(pair[key])
        for pair in pairs
        for key in ("light_mz", "heavy_mz")
        if key in pair
    }
    if len(identity_columns) == 2:
        for nominal_mz, group in result.groupby("mz_rounded", sort=True):
            if (
                int(nominal_mz) in qc_masses
                and len(group[identity_columns].drop_duplicates()) > 1
            ):
                collision_masses.append(int(nominal_mz))
    if collision_masses:
        masses_text = "、".join(str(value) for value in collision_masses)
        qc_records = [
            {
                "status": "fail",
                "reason": (
                    "同一名义质量存在多个精确峰轨道，必须先选择 peak_track："
                    f"m/z {masses_text}"
                ),
            }
        ]
        overall = "fail"
    else:
        working = result.copy()
        working["_energy_qc"] = pd.to_numeric(
            working["energy"],
            errors="coerce",
        ).round(6)
        matrix = working.pivot_table(
            index="_energy_qc",
            columns="mz_rounded",
            values=signal_column,
            aggfunc="mean",
            sort=True,
        )
        qc_table = evaluate_isotope_qc_pairs(matrix, pairs)
        qc_records = qc_table.to_dict("records")
        evaluated = qc_table[qc_table["status"] != "missing"]
        if evaluated.empty:
            overall = "not_evaluated"
        elif (evaluated["status"] == "fail").any():
            overall = "fail"
        else:
            overall = "pass"
    previous_attrs = dict(analysis_df.attrs)
    result["isotope_qc_status"] = overall
    result.attrs.update(previous_attrs)
    result.attrs["isotope_qc"] = qc_records
    result.attrs["isotope_qc_status"] = overall
    return result


def analyze_pie_folder(
    folder: str | Path,
    *,
    calibration: Calibration = Calibration(),
    suffixes: tuple[str, ...] = PIE_SPECTRUM_SUFFIXES,
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
    integration_method: str | None = None,
    manual_peak_path: str | Path | None = None,
    photon_normalize: bool = True,
    photon_reference_mode: str = "first",
    normalize_by_time: bool = True,
    mass_discrimination: float = 1.0,
    light_source: str = "io",
    target_mz_values: list[int] | None = None,
    replicate_mode: str = "off",
    vote_threshold: float = 0.667,
    min_intensity_for_single_vote: float = 5.0,
    mz_tolerance: float = 0.2,
    cwt_snr_threshold: float = 0.02,
    cwt_wavelet_max_width: int = 30,
    weak_tail_cutoff_idx: int = 15000,
    isotope_qc_pairs: list[dict] | None = None,
) -> pd.DataFrame:
    """Generate experimental PIE curves from a folder of energy-resolved spectra."""
    return _analyze_pie_sources(
        [folder],
        calibration=calibration,
        suffixes=suffixes,
        recursive=recursive,
        energy_decimals=energy_decimals,
        algorithm=algorithm,
        threshold_end=threshold_end,
        min_intensity=min_intensity,
        detection_min_idx=detection_min_idx,
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
        prefer_gaussian=prefer_gaussian,
        integration_method=integration_method,
        manual_peak_path=manual_peak_path,
        photon_normalize=photon_normalize,
        photon_reference_mode=photon_reference_mode,
        normalize_by_time=normalize_by_time,
        mass_discrimination=mass_discrimination,
        light_source=light_source,
        target_mz_values=target_mz_values,
        replicate_mode=replicate_mode,
        vote_threshold=vote_threshold,
        min_intensity_for_single_vote=min_intensity_for_single_vote,
        mz_tolerance=mz_tolerance,
        cwt_snr_threshold=cwt_snr_threshold,
        cwt_wavelet_max_width=cwt_wavelet_max_width,
        weak_tail_cutoff_idx=weak_tail_cutoff_idx,
        merge_method=None,
        isotope_qc_pairs=isotope_qc_pairs,
    )


def _pie_peak_track_series(group: pd.DataFrame) -> pd.Series:
    """Assign stable ordered peak tracks within one nominal-mass group.

    A reference peak can drift slightly in calibrated m/z between energies or
    independently analysed segments. Its order among the resolved peaks stays
    stable, so rank unique peak windows at each energy instead of requiring
    bit-for-bit equality of the calibrated mass.
    """
    if group.empty:
        return pd.Series(dtype="int64", index=group.index)

    base = group.reset_index().rename(columns={"index": "_original_index"})
    has_complete_bounds = all(
        column in base.columns and base[column].notna().all()
        for column in ("left_bound", "right_bound")
    )
    identity_columns = (
        ["left_bound", "right_bound"] if has_complete_bounds else ["mz"]
    )
    identities = base[["energy", *identity_columns]].drop_duplicates()
    if "mz" in identity_columns:
        identities["_peak_sort_mz"] = pd.to_numeric(
            identities["mz"],
            errors="coerce",
        )
    else:
        mz_means = (
            base.groupby(
                ["energy", *identity_columns],
                dropna=False,
                as_index=False,
            )["mz"]
            .mean()
            .rename(columns={"mz": "_peak_sort_mz"})
        )
        identities = identities.merge(
            mz_means,
            on=["energy", *identity_columns],
            how="left",
        )
    identities = identities.sort_values(
        ["energy", "_peak_sort_mz", *identity_columns],
        kind="stable",
    )
    identities["_pie_peak_track"] = identities.groupby(
        "energy",
        sort=False,
    ).cumcount()
    tracked = base.merge(
        identities[["energy", *identity_columns, "_pie_peak_track"]],
        on=["energy", *identity_columns],
        how="left",
        sort=False,
    )
    return (
        tracked.set_index("_original_index")["_pie_peak_track"]
        .reindex(group.index)
        .astype("int64")
    )


def build_pie_curves(analysis_df: pd.DataFrame) -> dict[int | float, dict]:
    """Convert PIE analysis rows into precise-peak PIE curve objects.

    Every curve is identified by its mean calibrated m/z. PICS candidates remain
    indexed separately by ``mz_rounded`` so broad integer-mass lookup never
    replaces or reduces the precision of the experimental curve identity.
    """
    curves: dict[int | float, dict] = {}
    if analysis_df.empty:
        return curves
    working = analysis_df.copy()
    if "raw_area" not in working:
        working["raw_area"] = working["normalized_intensity"]
    if "photon_normalized_intensity" not in working:
        working["photon_normalized_intensity"] = working["normalized_intensity"]
    if "species" not in working:
        working["species"] = ""
    if "integration_method" not in working:
        working["integration_method"] = ""
    for nominal_mz, nominal_group in working.groupby("mz_rounded"):
        nominal_group = nominal_group.copy()
        nominal_group["_pie_peak_track"] = _pie_peak_track_series(nominal_group)
        peak_groups = list(
            nominal_group.groupby(
                "_pie_peak_track",
                dropna=False,
                sort=False,
            )
        )
        has_nominal_collision = len(peak_groups) > 1
        for _peak_identity, group in peak_groups:
            group = group.sort_values("energy").reset_index(drop=True)
            replicate_modes = set(group.get("replicate_mode", pd.Series(dtype=object)).dropna().astype(str))
            if replicate_modes == {"off"}:
                ordered = group.copy()
            else:
                group["energy_rounded"] = group["energy"].round(2)
                curve_agg: dict[str, tuple[str, object]] = {
                    "energy": ("energy", "mean"),
                    "normalized_intensity": ("normalized_intensity", "mean"),
                    "raw_area": ("raw_area", "mean"),
                    "photon_normalized_intensity": (
                        "photon_normalized_intensity",
                        "mean",
                    ),
                    "integration_method": (
                        "integration_method",
                        _summarize_integration_methods,
                    ),
                    "species": ("species", "first"),
                    "mz": ("mz", "mean"),
                    "file_count": ("file_count", "first"),
                    "io": ("io", "first"),
                }
                for column in (
                    "merged_intensity",
                    "count_rate",
                    "io_normalized_intensity",
                    "io_time_normalized_intensity",
                    "acquisition_time_s",
                    "total_acquisition_time_s",
                    "acquisition_time_min_s",
                    "acquisition_time_max_s",
                    "signal_to_noise",
                    "shared_scale_factor",
                    "shared_scale_log_mad",
                ):
                    if column in group.columns:
                        curve_agg[column] = (column, "mean")
                for column in (
                    "source_folder",
                    "source_segment",
                    "time_normalized",
                    "io_normalized",
                    "signal_to_noise_method",
                    "segment_scale_qc",
                ):
                    if column in group.columns:
                        curve_agg[column] = (column, "first")
                ordered = (
                    group.groupby("energy_rounded", as_index=False)
                    .agg(**curve_agg)
                    .sort_values("energy")
                )
            exact_mz = float(ordered["mz"].mean())
            curve_key = exact_mz
            if curve_key in curves:
                raise ValueError(
                    "PIE curve identity collision for precise m/z "
                    f"{exact_mz:.12g}; peak bounds must uniquely identify each curve"
                )
            ordered = ordered.drop(columns=["_pie_peak_track"], errors="ignore")
            curves[curve_key] = {
                "mz": exact_mz,
                "mz_rounded": int(nominal_mz),
                "mz_exact_mean": exact_mz,
                "curve_key": curve_key,
                "peak_track": int(_peak_identity),
                "has_nominal_collision": has_nominal_collision,
                "species": str(ordered["species"].iloc[0]) if "species" in ordered else "",
                "energies": ordered["energy"].astype(float).tolist(),
                "intensities": ordered[
                    "merged_intensity"
                    if "merged_intensity" in ordered
                    else "normalized_intensity"
                ].astype(float).tolist(),
                "rows": ordered,
            }
    return curves


def build_pie_ratio_curve(
    numerator_rows: pd.DataFrame,
    denominator_rows: pd.DataFrame,
    *,
    intensity_column: str = "merged_intensity",
    min_snr: float = 3.0,
    energy_decimals: int = 4,
) -> pd.DataFrame:
    """Align two PIE channels and calculate a low-SNR-masked ratio curve."""
    if intensity_column not in numerator_rows.columns:
        intensity_column = "normalized_intensity"
    if intensity_column not in denominator_rows.columns:
        intensity_column = "normalized_intensity"
    required = {"energy", intensity_column}
    if not required.issubset(numerator_rows.columns) or not required.issubset(
        denominator_rows.columns
    ):
        raise ValueError("PIE比值需要能量列和强度列")

    def prepare(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        prepared = pd.DataFrame(
            {
                "energy_key": pd.to_numeric(
                    frame["energy"],
                    errors="coerce",
                ).round(int(energy_decimals)),
                f"energy_{suffix}": pd.to_numeric(
                    frame["energy"],
                    errors="coerce",
                ),
                f"intensity_{suffix}": pd.to_numeric(
                    frame[intensity_column],
                    errors="coerce",
                ),
                f"snr_{suffix}": _snr_values(frame),
            }
        )
        return prepared.groupby("energy_key", as_index=False).mean(numeric_only=True)

    merged = pd.merge(
        prepare(numerator_rows, "numerator"),
        prepare(denominator_rows, "denominator"),
        on="energy_key",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "energy",
                "numerator",
                "denominator",
                "ratio",
                "valid",
                "mask_reason",
            ]
        )
    merged["energy"] = (
        merged["energy_numerator"] + merged["energy_denominator"]
    ) / 2.0
    numerator = merged["intensity_numerator"].to_numpy(dtype=float)
    denominator = merged["intensity_denominator"].to_numpy(dtype=float)
    numerator_snr = merged["snr_numerator"].to_numpy(dtype=float)
    denominator_snr = merged["snr_denominator"].to_numpy(dtype=float)
    finite = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & np.isfinite(numerator_snr)
        & np.isfinite(denominator_snr)
    )
    positive_denominator = denominator > 0
    snr_valid = (
        (numerator_snr >= float(min_snr))
        & (denominator_snr >= float(min_snr))
    )
    valid = finite & positive_denominator & snr_valid
    ratio = np.full(len(merged), np.nan, dtype=float)
    ratio[valid] = numerator[valid] / denominator[valid]
    reasons = np.full(len(merged), "", dtype=object)
    reasons[~finite] = "非有限值"
    reasons[finite & ~positive_denominator] = "分母不为正"
    reasons[finite & positive_denominator & ~snr_valid] = "低SNR"
    return pd.DataFrame(
        {
            "energy": merged["energy"].to_numpy(dtype=float),
            "numerator": numerator,
            "denominator": denominator,
            "numerator_snr": numerator_snr,
            "denominator_snr": denominator_snr,
            "ratio": ratio,
            "valid": valid,
            "mask_reason": reasons,
        }
    ).sort_values("energy").reset_index(drop=True)


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


def species_ionization_energy_value(species: dict) -> object:
    """Return IE using the legacy and PyQt schema aliases."""
    value = species.get("ie")
    return species.get("ionization_energy") if value is None else value


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
        # locked_fit模式：将指定物种系数锁定为 coefficients[id]，然后用 NNLS 优化其余物种
        # 注意：identify_species_for_mz_with_curve 已不再使用此模式
        # 此模式仍可由 UI 手动配置（coefficient_mode_combo = "锁定已选"）时触发
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
        # 结果包含条件：
        # 1. 系数>0.001（普通物种，贡献度判断）
        # 2. OR 该物种在locked_ids中（锁定候选，不被自动筛除，系数由 NNLS 正常优化）
        # TODO: 改进为贡献度范数判断，而非绝对系数阈值（见 locked_candidates_refactor_plan.md Phase 4）
        if coeffs[idx] > 0.001 or active_species_ids[idx] in locked_ids:
            component = design[:, idx] * coeffs[idx]
            ionization_energy = species_ionization_energy_value(species)
            key = (species["species"], ionization_energy)
            if key not in merged:
                merged[key] = {
                    "mz": int(species["mz"]),
                    "ids": [],
                    "coefficients_by_id": {},
                    "species": species["species"],
                    "ie": ionization_energy,
                    "ionization_energy": ionization_energy,
                    "ie_source": species.get("ie_source", ""),
                    "ie_query_status": species.get("ie_query_status", ""),
                    "ie_message": species.get("ie_message", ""),
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
            "ionization_energy": item["ionization_energy"],
            "ie_source": item.get("ie_source", ""),
            "ie_query_status": item.get("ie_query_status", ""),
            "ie_message": item.get("ie_message", ""),
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
    locked_species: list[str] | None = None,
    forced_species: list[str] | None = None,  # 向后兼容别名
) -> dict:
    """
    识别给定m/z的物种并拟合曲线

    Args:
        locked_species: 锁定候选物种名称列表
                       锁定物种不被自动筛选移除，系数由优化器正常决定
                       "锁定"不表示已鉴别或系数必须非零
        forced_species: 向后兼容别名，等同于 locked_species

    当存在锁定候选物种时，这些物种的系数由 NNLS 自由优化，不强制为 eps。
    锁定只保证它们出现在结果集合中（无论系数是否 > 阈值）。
    物种鉴别由实验曲线与拟合结果共同支持。
    """
    # 向后兼容：如果只传了 forced_species，使用它
    if locked_species is None and forced_species is not None:
        locked_species = forced_species

    candidates = [item for item in database if item["mz"] == int(mz)]
    locked_names = {name for name in (locked_species or []) if name}
    if locked_names:
        locked_ids = [
            int(item.get("id", index + 1))
            for index, item in enumerate(candidates)
            if item.get("species") in locked_names
        ]
        return fit_species_combination_with_curve(
            candidates,
            energies,
            intensities,
            locked_species_ids=locked_ids,
            # 不传 coefficient_mode="locked_fit"，使用默认 "fit"（NNLS 正常优化）
            # locked_species_ids 仅用于结果包含逻辑，不影响系数优化
        )
    return fit_species_combination_with_curve(candidates, energies, intensities)


def _empty_pie_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "energy", "file_count", "io", "light_source", "replicate_mode",
        "replicate_grouping", "replicate_warning", "reference_energy",
        "reference_source", "mz", "mz_rounded",
        "species", "photon_normalized_intensity", "normalized_intensity",
        "merged_intensity", "raw_area", "count_rate",
        "io_normalized_intensity", "io_time_normalized_intensity",
        "acquisition_time_s", "total_acquisition_time_s",
        "acquisition_time_min_s", "acquisition_time_max_s", "time_normalized",
        "io_normalized", "signal_to_noise", "signal_to_noise_method",
        "integration_method",
        "left_bound", "right_bound", "blank_file_count",
        "background_subtracted", "shared_scale_factor",
        "shared_scale_log_mad", "segment_scale_qc",
        "source_files", "source_folders", "source_spectra",
    ])


def _segment_energy_aggregation_spec(
    seg_df: pd.DataFrame,
    *,
    include_mz_rounded: bool,
) -> dict[str, tuple[str, object]]:
    if "integration_method" not in seg_df:
        seg_df["integration_method"] = ""
    if "signal_to_noise_method" not in seg_df:
        seg_df["signal_to_noise_method"] = ""
    agg_spec: dict[str, tuple[str, object]] = {
        "energy": ("energy", "mean"),
        "normalized_intensity": ("normalized_intensity", "mean"),
        "raw_area": ("raw_area", "mean"),
        "photon_normalized_intensity": ("photon_normalized_intensity", "mean"),
        "integration_method": ("integration_method", _summarize_integration_methods),
        "signal_to_noise_method": (
            "signal_to_noise_method",
            _summarize_integration_methods,
        ),
        "mz": ("mz", "mean"),
        "species": ("species", "first"),
        "file_count": ("file_count", "first"),
        "io": ("io", "first"),
        "light_source": ("light_source", "first"),
        "left_bound": ("left_bound", "first"),
        "right_bound": ("right_bound", "first"),
    }
    for numeric_column in (
        "merged_intensity",
        "count_rate",
        "io_normalized_intensity",
        "io_time_normalized_intensity",
        "acquisition_time_s",
        "total_acquisition_time_s",
        "signal_to_noise",
    ):
        if numeric_column in seg_df.columns:
            agg_spec[numeric_column] = (numeric_column, "mean")
    if "acquisition_time_min_s" in seg_df.columns:
        agg_spec["acquisition_time_min_s"] = (
            "acquisition_time_min_s",
            "min",
        )
    if "acquisition_time_max_s" in seg_df.columns:
        agg_spec["acquisition_time_max_s"] = (
            "acquisition_time_max_s",
            "max",
        )
    if include_mz_rounded:
        agg_spec = {
            "energy": agg_spec.pop("energy"),
            "mz_rounded": ("mz_rounded", "first"),
            **agg_spec,
        }
    for optional in (
        "source_folder_idx",
        "source_folder",
        "reference_energy",
        "reference_source",
        "blank_file_count",
        "background_subtracted",
        "time_normalized",
        "io_normalized",
        "source_segment",
        "scale_factor",
        "shared_scale_factor",
        "shared_scale_log_mad",
        "shared_scale_overlap_energy_count",
        "shared_scale_channel_count",
        "shared_scale_observation_count",
        "shared_scale_rejected_count",
        "segment_scale_qc",
    ):
        if optional in seg_df.columns:
            agg_spec[optional] = (optional, "first")
    for sequence_column in (
        "source_files",
        "source_spectra",
        "acquisition_times_s",
        "source_folders",
    ):
        if sequence_column in seg_df.columns:
            agg_spec[sequence_column] = (
                sequence_column,
                _merge_metadata_values,
            )
    if "source_folder" in seg_df.columns:
        agg_spec["source_folders"] = (
            "source_folder",
            _merge_metadata_values,
        )
    return agg_spec


def _aggregate_segment_by_energy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    seg_df = df.sort_values("energy").reset_index(drop=True).copy()
    seg_df["energy_rounded"] = seg_df["energy"].round(2)
    agg_spec = _segment_energy_aggregation_spec(
        seg_df,
        include_mz_rounded=True,
    )
    return seg_df.groupby("energy_rounded", as_index=False).agg(**agg_spec)


def _pie_peak_identity_series(df: pd.DataFrame) -> pd.Series:
    """Return ordered precise-peak track IDs independently for each nominal m/z."""
    identities = pd.Series("", index=df.index, dtype=object)
    for nominal_mz, group in df.groupby("mz_rounded", sort=False):
        tracks = _pie_peak_track_series(group)
        identities.loc[group.index] = tracks.map(
            lambda value: f"{int(nominal_mz)}:track:{int(value)}"
        )
    return identities


def _aggregate_segments_by_mz_energy(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate every m/z-energy group in one Pandas operation.

    ``merge_pie_segments`` historically called ``_aggregate_segment_by_energy``
    hundreds of times—once per m/z and segment. Grouping by both keys at once is
    mathematically equivalent and avoids most of the DataFrame construction cost.
    """
    if df.empty:
        return df.copy()
    seg_df = df.sort_values(["mz_rounded", "energy"]).reset_index(drop=True).copy()
    if "_pie_peak_key" not in seg_df.columns or seg_df["_pie_peak_key"].isna().any():
        seg_df["_pie_peak_key"] = _pie_peak_identity_series(seg_df)
    seg_df["energy_rounded"] = seg_df["energy"].round(2)
    agg_spec = _segment_energy_aggregation_spec(
        seg_df,
        include_mz_rounded=False,
    )
    result = seg_df.groupby(
        ["mz_rounded", "_pie_peak_key", "energy_rounded"],
        as_index=False,
        sort=True,
    ).agg(**agg_spec)
    ordered_columns = [
        "energy",
        "mz_rounded",
        "_pie_peak_key",
        "energy_rounded",
        *[
            column
            for column in result.columns
            if column not in {"energy", "mz_rounded", "_pie_peak_key", "energy_rounded"}
        ],
    ]
    return result[ordered_columns]


def _snr_values(frame: pd.DataFrame) -> pd.Series:
    if "signal_to_noise" in frame.columns:
        return pd.to_numeric(frame["signal_to_noise"], errors="coerce")
    raw = pd.to_numeric(
        frame.get("raw_area", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    return np.sqrt(raw.clip(lower=0.0))


def _shared_scale_from_overlap(
    ref_df: pd.DataFrame,
    seg_df: pd.DataFrame,
    *,
    min_overlap_energies: int,
    min_channels: int,
    min_snr: float,
) -> dict[str, object]:
    """Estimate one robust factor shared by every peak in a segment."""
    join_keys = ["mz_rounded", "_pie_peak_key", "energy_rounded"]
    ref = ref_df.copy()
    seg = seg_df.copy()
    ref["_scale_snr"] = _snr_values(ref)
    seg["_scale_snr"] = _snr_values(seg)
    overlap = pd.merge(
        ref[
            [
                *join_keys,
                "normalized_intensity",
                "_scale_snr",
            ]
        ],
        seg[
            [
                *join_keys,
                "normalized_intensity",
                "_scale_snr",
            ]
        ],
        on=join_keys,
        suffixes=("_ref", "_seg"),
    )
    if overlap.empty:
        return {
            "valid": False,
            "reason": "没有共同的能量-峰轨道观测",
            "overlap_energy_count": 0,
            "channel_count": 0,
            "observation_count": 0,
            "rejected_count": 0,
        }
    ref_intensity = pd.to_numeric(
        overlap["normalized_intensity_ref"],
        errors="coerce",
    ).to_numpy(dtype=float)
    seg_intensity = pd.to_numeric(
        overlap["normalized_intensity_seg"],
        errors="coerce",
    ).to_numpy(dtype=float)
    ref_snr = pd.to_numeric(overlap["_scale_snr_ref"], errors="coerce").to_numpy(
        dtype=float
    )
    seg_snr = pd.to_numeric(overlap["_scale_snr_seg"], errors="coerce").to_numpy(
        dtype=float
    )
    valid = (
        np.isfinite(ref_intensity)
        & np.isfinite(seg_intensity)
        & np.isfinite(ref_snr)
        & np.isfinite(seg_snr)
        & (ref_intensity > 0)
        & (seg_intensity > 0)
        & (ref_snr >= float(min_snr))
        & (seg_snr >= float(min_snr))
    )
    qualified = overlap.loc[valid].copy()
    energy_count = int(qualified["energy_rounded"].nunique())
    channel_count = int(qualified["_pie_peak_key"].nunique())
    observation_count = int(len(qualified))
    if energy_count < int(min_overlap_energies) or channel_count < int(min_channels):
        return {
            "valid": False,
            "reason": (
                f"有效重叠仅 {energy_count} 个能量点、{channel_count} 条曲线"
                f"（要求至少 {min_overlap_energies} 个能量点、{min_channels} 条"
                f"SNR≥{min_snr:g} 曲线）"
            ),
            "overlap_energy_count": energy_count,
            "channel_count": channel_count,
            "observation_count": observation_count,
            "rejected_count": int(len(overlap) - observation_count),
        }

    log_ratios = np.log(ref_intensity[valid] / seg_intensity[valid])
    initial_median = float(np.median(log_ratios))
    absolute_deviation = np.abs(log_ratios - initial_median)
    initial_mad = float(np.median(absolute_deviation))
    if initial_mad > 0:
        robust_sigma = 1.4826 * initial_mad
        retained = absolute_deviation <= 3.5 * robust_sigma
    else:
        retained = np.ones(log_ratios.size, dtype=bool)
    retained_rows = qualified.loc[retained]
    retained_energy_count = int(retained_rows["energy_rounded"].nunique())
    retained_channel_count = int(retained_rows["_pie_peak_key"].nunique())
    if (
        retained_energy_count < int(min_overlap_energies)
        or retained_channel_count < int(min_channels)
    ):
        return {
            "valid": False,
            "reason": (
                "稳健离群剔除后证据不足："
                f"{retained_energy_count} 个能量点、{retained_channel_count} 条曲线"
            ),
            "overlap_energy_count": retained_energy_count,
            "channel_count": retained_channel_count,
            "observation_count": int(np.sum(retained)),
            "rejected_count": int(len(overlap) - np.sum(retained)),
        }
    retained_log_ratios = log_ratios[retained]
    log_median = float(np.median(retained_log_ratios))
    log_mad = float(np.median(np.abs(retained_log_ratios - log_median)))
    return {
        "valid": True,
        "factor": float(np.exp(log_median)),
        "log_median": log_median,
        "log_mad": log_mad,
        "overlap_energy_count": retained_energy_count,
        "channel_count": retained_channel_count,
        "observation_count": int(retained_log_ratios.size),
        "rejected_count": int(len(overlap) - retained_log_ratios.size),
        "reason": "",
    }


def merge_pie_segments(
    analysis_dfs: list[pd.DataFrame],
    *,
    merge_method: str = "low_energy_dominant",
    min_overlap_energies: int = 3,
    min_scale_channels: int = 3,
    min_scale_snr: float = 10.0,
) -> pd.DataFrame:
    """Merge PIE analysis DataFrames from multiple energy segments.

    One factor is estimated per segment and shared by every peak track.  This
    preserves isotope ratios.  Scaling is blocked when overlap/SNR evidence is
    insufficient instead of silently substituting a factor of 1.
    """
    if not analysis_dfs:
        return _empty_pie_dataframe()
    if len(analysis_dfs) == 1:
        single = analysis_dfs[0].copy()
        if "merged_intensity" not in single.columns:
            single["merged_intensity"] = single["normalized_intensity"]
        single["source_segment"] = 0
        single["scale_factor"] = 1.0
        single["shared_scale_factor"] = 1.0
        single["shared_scale_log_mad"] = 0.0
        single["segment_scale_qc"] = "single_segment"
        single.attrs["segment_scaling_diagnostics"] = [
            {
                "segment": 0,
                "factor": 1.0,
                "qc_status": "single_segment",
            }
        ]
        return single

    prepared_segments: list[dict[str, object]] = []
    for df_idx, df in enumerate(analysis_dfs):
        if df.empty or "mz_rounded" not in df.columns:
            continue
        prepared = _aggregate_segments_by_mz_energy(df)
        if "merged_intensity" not in prepared.columns:
            prepared["merged_intensity"] = prepared["normalized_intensity"]
        prepared_segments.append(
            {
                "df": prepared,
                "segment_idx": df_idx,
                "min_energy": float(prepared["energy"].min()),
                "max_energy": float(prepared["energy"].max()),
            }
        )
    if not prepared_segments:
        return _empty_pie_dataframe()

    prepared_segments.sort(key=lambda item: float(item["min_energy"]))
    if merge_method == "first_segment_dominant":
        first = next(
            (
                segment
                for segment in prepared_segments
                if int(segment["segment_idx"]) == 0
            ),
            prepared_segments[0],
        )
        ordered_segments = [
            first,
            *[
                segment
                for segment in prepared_segments
                if segment is not first
            ],
        ]
    else:
        ordered_segments = list(prepared_segments)

    diagnostics: list[dict[str, object]] = []
    scaled_segments: list[pd.DataFrame] = []
    if merge_method == "mean":
        for segment in ordered_segments:
            seg_df = segment["df"].copy()
            segment_idx = int(segment["segment_idx"])
            seg_df["source_segment"] = segment_idx
            seg_df["scale_factor"] = 1.0
            seg_df["shared_scale_factor"] = 1.0
            seg_df["shared_scale_log_mad"] = 0.0
            seg_df["segment_scale_qc"] = "unscaled_mean"
            scaled_segments.append(seg_df)
            diagnostics.append(
                {
                    "segment": segment_idx,
                    "factor": 1.0,
                    "qc_status": "unscaled_mean",
                }
            )
    else:
        for position, segment in enumerate(ordered_segments):
            seg_df = segment["df"].copy()
            segment_idx = int(segment["segment_idx"])
            if position == 0:
                diagnostic = {
                    "valid": True,
                    "factor": 1.0,
                    "log_mad": 0.0,
                    "overlap_energy_count": 0,
                    "channel_count": 0,
                    "observation_count": 0,
                    "rejected_count": 0,
                    "qc_status": "reference_segment",
                    "segment": segment_idx,
                    "reference_segment": None,
                }
            else:
                candidates: list[tuple[dict[str, object], int]] = []
                failed: list[dict[str, object]] = []
                for reference_position, reference_df in enumerate(scaled_segments):
                    candidate = _shared_scale_from_overlap(
                        reference_df,
                        seg_df,
                        min_overlap_energies=min_overlap_energies,
                        min_channels=min_scale_channels,
                        min_snr=min_scale_snr,
                    )
                    reference_segment = int(
                        ordered_segments[reference_position]["segment_idx"]
                    )
                    candidate["reference_segment"] = reference_segment
                    if bool(candidate.get("valid")):
                        candidates.append((candidate, reference_segment))
                    else:
                        failed.append(candidate)
                if not candidates:
                    best_failed = max(
                        failed,
                        key=lambda item: (
                            int(item.get("overlap_energy_count", 0)),
                            int(item.get("channel_count", 0)),
                            int(item.get("observation_count", 0)),
                        ),
                        default={"reason": "没有可比较的已缩放能段"},
                    )
                    raise ValueError(
                        f"PIE能段 {segment_idx} 共享缩放QC失败："
                        f"{best_failed.get('reason')}。未使用静默1.0因子。"
                    )
                diagnostic, reference_segment = max(
                    candidates,
                    key=lambda item: (
                        int(item[0].get("overlap_energy_count", 0)),
                        int(item[0].get("channel_count", 0)),
                        int(item[0].get("observation_count", 0)),
                    ),
                )
                diagnostic = dict(diagnostic)
                diagnostic.update(
                    {
                        "qc_status": "pass",
                        "segment": segment_idx,
                        "reference_segment": reference_segment,
                    }
                )
            scale_factor = float(diagnostic["factor"])
            seg_df["merged_intensity"] = (
                pd.to_numeric(seg_df["normalized_intensity"], errors="coerce")
                * scale_factor
            )
            # ``normalized_intensity`` remains the backward-compatible plotting
            # field, while all explicitly raw/rate columns retain their original
            # unscaled values.
            seg_df["normalized_intensity"] = seg_df["merged_intensity"]
            seg_df["source_segment"] = segment_idx
            seg_df["scale_factor"] = scale_factor
            seg_df["shared_scale_factor"] = scale_factor
            seg_df["shared_scale_log_mad"] = float(
                diagnostic.get("log_mad", 0.0)
            )
            seg_df["shared_scale_overlap_energy_count"] = int(
                diagnostic.get("overlap_energy_count", 0)
            )
            seg_df["shared_scale_channel_count"] = int(
                diagnostic.get("channel_count", 0)
            )
            seg_df["shared_scale_observation_count"] = int(
                diagnostic.get("observation_count", 0)
            )
            seg_df["shared_scale_rejected_count"] = int(
                diagnostic.get("rejected_count", 0)
            )
            seg_df["segment_scale_qc"] = str(diagnostic["qc_status"])
            scaled_segments.append(seg_df)
            diagnostics.append(dict(diagnostic))

    combined = pd.concat(scaled_segments, ignore_index=True)
    final_combined = _aggregate_segments_by_mz_energy(combined)
    result = (
        final_combined.drop(
            columns=["energy_rounded", "_pie_peak_key"],
            errors="ignore",
        )
        .sort_values(["mz_rounded", "mz", "energy"])
        .reset_index(drop=True)
    )
    result.attrs["segment_scaling_diagnostics"] = diagnostics
    result.attrs["segment_scaling_policy"] = {
        "version": 3,
        "strategy": "shared_robust_log_median",
        "min_overlap_energies": int(min_overlap_energies),
        "min_channels": int(min_scale_channels),
        "min_snr": float(min_scale_snr),
        "snr_method": "poisson_peak_window",
    }
    return result


def analyze_multiple_pie_folders(
    folders: list[str | Path],
    *,
    calibration: Calibration = Calibration(),
    suffixes: tuple[str, ...] = PIE_SPECTRUM_SUFFIXES,
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
    integration_method: str | None = None,
    manual_peak_path: str | Path | None = None,
    photon_normalize: bool = True,
    photon_reference_mode: str = "first",
    normalize_by_time: bool = True,
    mass_discrimination: float = 1.0,
    light_source: str = "io",
    merge_method: str = "low_energy_dominant",
    replicate_mode: str = "off",
    vote_threshold: float = 0.667,
    min_intensity_for_single_vote: float = 5.0,
    mz_tolerance: float = 0.2,
    cwt_snr_threshold: float = 0.02,
    cwt_wavelet_max_width: int = 30,
    weak_tail_cutoff_idx: int = 15000,
    isotope_qc_pairs: list[dict] | None = None,
) -> pd.DataFrame:
    """Analyze multiple PIE folders and merge them with overlap scaling."""
    return _analyze_pie_sources(
        folders,
        calibration=calibration,
        suffixes=suffixes,
        recursive=recursive,
        energy_decimals=energy_decimals,
        algorithm=algorithm,
        threshold_end=threshold_end,
        min_intensity=min_intensity,
        detection_min_idx=detection_min_idx,
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
        prefer_gaussian=prefer_gaussian,
        integration_method=integration_method,
        manual_peak_path=manual_peak_path,
        photon_normalize=photon_normalize,
        photon_reference_mode=photon_reference_mode,
        normalize_by_time=normalize_by_time,
        mass_discrimination=mass_discrimination,
        light_source=light_source,
        target_mz_values=None,
        replicate_mode=replicate_mode,
        vote_threshold=vote_threshold,
        min_intensity_for_single_vote=min_intensity_for_single_vote,
        mz_tolerance=mz_tolerance,
        cwt_snr_threshold=cwt_snr_threshold,
        cwt_wavelet_max_width=cwt_wavelet_max_width,
        weak_tail_cutoff_idx=weak_tail_cutoff_idx,
        merge_method=merge_method,
        isotope_qc_pairs=isotope_qc_pairs,
    )

def identify_species_for_mz(database: list[dict], mz: int, energies, intensities) -> list[dict]:
    model = identify_species_for_mz_with_curve(database, mz, energies, intensities)
    return [{
        k: v for k, v in item.items() if k not in {"component_intensities"}
    } for item in model["species"]]
