from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
import math
import os
from pathlib import Path
import re
import sqlite3
import shutil
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi import FastAPI, Request
from fastapi import HTTPException
from pydantic import BaseModel
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from core.calibration import Calibration
from core.config import (
    PROJECT_ROOT,
    load_calibration_config,
    load_peak_detection_config,
    species_database_path,
)
from core.isotope import calculate_isotope_distribution
from core.normalization import load_normalization_settings
from core.peak_detection import detect_peaks_in_range
from core.pie_analysis import (
    SCHEMA_SQL,
    analyze_pie_folder,
    build_pie_curves,
    fit_species_combination_with_curve,
    identify_species_for_mz_with_curve,
    load_species_database,
)
from core.spectrum_io import read_bl03u_txt, sum_spectra


app = FastAPI(title="BL03U MassSpectrumTool API")
WEB_ROOT = PROJECT_ROOT / "frontends" / "web_app" / "static"
if WEB_ROOT.exists():
    app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")

EXECUTOR = ThreadPoolExecutor(max_workers=2)
JOBS_LOCK = threading.Lock()


@dataclass
class PieJob:
    id: str
    status: str = "queued"
    message: str = "等待计算"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    analysis_df: Any = None
    curves: Dict[int, dict] = field(default_factory=dict)
    fits: Dict[int, dict] = field(default_factory=dict)
    database_path: str = ""
    source_folder: str = ""
    peak_source: str = ""


JOBS: Dict[str, PieJob] = {}


@dataclass
class PicsLibrary:
    id: str
    database_path: str
    source_file: str
    created_at: float = field(default_factory=time.time)
    parsed_species: int = 0
    inserted_points: int = 0


PICS_LIBRARIES_LOCK = threading.Lock()
PICS_LIBRARIES: Dict[str, PicsLibrary] = {}
PICS_LIBRARY_TTL_SECONDS = int(float(os.environ.get("BL03U_PICS_LIBRARY_TTL_HOURS", "24")) * 3600)
MAX_UPLOAD_BYTES = int(float(os.environ.get("BL03U_MAX_UPLOAD_MB", "20")) * 1024 * 1024)
DEFAULT_JOB_TTL_HOURS = 6
DEFAULT_MAX_JOBS = 100


class PathAccessError(ValueError):
    """Raised when a user-supplied path escapes configured data roots."""


class IsotopePayload(BaseModel):
    formula: str
    min_percent: float = 0.01


class PieStartPayload(BaseModel):
    folder: str
    database: Optional[str] = None
    library_id: Optional[str] = None
    peak_source: str = "auto"
    manual_peak_path: Optional[str] = None
    target_mz: Optional[str] = None
    recursive: bool = True
    energy_decimals: int = 1
    gaussian: bool = True
    photon_mode: Optional[str] = None
    light_source: Optional[str] = None
    mass_discrimination: Optional[float] = None


class PieFitPayload(BaseModel):
    species_ids: Optional[List[int]] = None
    coefficient_mode: str = "fit"
    coefficients: Optional[Dict[int, float]] = None
    locked_species_ids: Optional[List[int]] = None


def _json_records(df) -> List[Dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return []
    return df.where(df.notna(), None).to_dict(orient="records")


def _float_or_none(value) -> Optional[float]:
    return None if value is None else float(value)


def _clean_column_name(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _find_column(columns, aliases: List[str]) -> Optional[str]:
    cleaned = {_clean_column_name(column): column for column in columns}
    for alias in aliases:
        key = _clean_column_name(alias)
        if key in cleaned:
            return cleaned[key]
    return None


def _parse_energy_column(value: object) -> Optional[float]:
    text = str(value).strip().lower().replace("ev", "")
    try:
        return float(text)
    except ValueError:
        return None


def _parse_mz_column(value: object) -> Optional[float]:
    text = str(value).strip().lower()
    direct = _parse_energy_column(text)
    if direct is not None:
        return direct
    text = text.replace("m/z", " ").replace("mz", " ").replace("mass", " ")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def _parse_mz_values(value: Optional[str]) -> List[int]:
    if not value:
        return []
    values = []
    for token in re.split(r"[,，;；\s]+", value.strip()):
        if not token:
            continue
        try:
            values.append(int(round(float(token))))
        except ValueError:
            raise ValueError(f"无法解析目标 m/z: {token}")
    return sorted(set(values))


def _resolve_root_path(path: Union[str, Path]) -> Path:
    path = Path(path).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve(strict=False)


def _path_is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _allowed_data_roots() -> List[Path]:
    roots = [_resolve_root_path(PROJECT_ROOT)]
    configured = os.environ.get("BL03U_ALLOWED_DATA_ROOTS", "").strip()
    if configured:
        for token in configured.split(os.pathsep):
            token = token.strip()
            if token:
                roots.append(_resolve_root_path(token))

    unique_roots: List[Path] = []
    for root in roots:
        if root not in unique_roots:
            unique_roots.append(root)
    return unique_roots


def _resolve_path(path: Union[str, Path]) -> Path:
    resolved = _resolve_root_path(path)
    if any(_path_is_inside(resolved, root) for root in _allowed_data_roots()):
        return resolved
    allowed = ", ".join(str(root) for root in _allowed_data_roots())
    raise PathAccessError(f"路径不在允许的数据目录内: {resolved}。允许目录: {allowed}")


def _default_database_path() -> Path:
    return species_database_path()


def _admin_token() -> str:
    return os.environ.get("BL03U_ADMIN_TOKEN", "").strip()


def _verify_admin_token(request: Request) -> None:
    expected = _admin_token()
    if not expected:
        raise HTTPException(status_code=403, detail="服务器未配置 BL03U_ADMIN_TOKEN，禁止写入维护库")
    token = request.headers.get("x-admin-token", "").strip()
    auth = request.headers.get("authorization", "").strip()
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    if token != expected:
        raise HTTPException(status_code=403, detail="管理员 token 无效")


def _get_pics_library(library_id: Optional[str]) -> Optional[PicsLibrary]:
    if not library_id:
        return None
    _cleanup_expired_pics_libraries()
    with PICS_LIBRARIES_LOCK:
        return PICS_LIBRARIES.get(library_id)


def _cleanup_expired_pics_libraries() -> None:
    if PICS_LIBRARY_TTL_SECONDS <= 0:
        return
    now = time.time()
    expired: List[PicsLibrary] = []
    with PICS_LIBRARIES_LOCK:
        for library_id, library in list(PICS_LIBRARIES.items()):
            if now - library.created_at > PICS_LIBRARY_TTL_SECONDS:
                expired.append(library)
                PICS_LIBRARIES.pop(library_id, None)
    for library in expired:
        try:
            Path(library.database_path).unlink(missing_ok=True)
        except Exception:
            pass


def _check_upload_size(content: bytes) -> None:
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"上传文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB 限制")


def _database_path_from_request(
    *,
    database: Optional[str] = None,
    library_id: Optional[str] = None,
) -> Path:
    library = _get_pics_library(library_id)
    if library_id and library is None:
        raise HTTPException(status_code=404, detail="PICS library not found")
    if library is not None:
        path = Path(library.database_path)
    else:
        try:
            path = _resolve_path(database or species_database_path())
        except PathAccessError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="SQLite PICS database not found")
    return path


def _get_job(job_id: str) -> PieJob:
    _cleanup_jobs()
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def _update_job(job_id: str, **values) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        for key, value in values.items():
            setattr(job, key, value)
        job.updated_at = time.time()


def _job_ttl_seconds() -> int:
    return int(float(os.environ.get("BL03U_PIE_JOB_TTL_HOURS", str(DEFAULT_JOB_TTL_HOURS))) * 3600)


def _max_jobs() -> int:
    return max(1, int(os.environ.get("BL03U_MAX_PIE_JOBS", str(DEFAULT_MAX_JOBS))))


def _cleanup_jobs() -> None:
    ttl_seconds = _job_ttl_seconds()
    max_jobs = _max_jobs()
    now = time.time()
    with JOBS_LOCK:
        if ttl_seconds > 0:
            for job_id, job in list(JOBS.items()):
                if job.status in {"done", "error"} and now - job.updated_at > ttl_seconds:
                    JOBS.pop(job_id, None)

        overflow = len(JOBS) - max_jobs
        if overflow <= 0:
            return
        removable = sorted(
            (
                job
                for job in JOBS.values()
                if job.status in {"done", "error"}
            ),
            key=lambda item: item.updated_at,
        )
        for job in removable[:overflow]:
            JOBS.pop(job.id, None)


def _curve_payload(curve: dict) -> Dict[str, Any]:
    return {
        "mz": int(curve["mz"]),
        "mz_exact_mean": float(curve.get("mz_exact_mean", curve["mz"])),
        "species": curve.get("species", ""),
        "energies": [float(value) for value in curve["energies"]],
        "intensities": [float(value) for value in curve["intensities"]],
        "rows": _json_records(curve.get("rows")),
    }


@lru_cache(maxsize=8)
def _load_species_database_cached(path: str, mtime: float) -> List[dict]:
    _ = mtime
    database, _index = load_species_database(path)
    return database


def _read_pics_upload_table(content: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    buffer = BytesIO(content)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(buffer)
    if suffix == ".tsv":
        return pd.read_csv(buffer, sep="\t")
    if suffix in {".csv", ".txt", ""}:
        return pd.read_csv(buffer)
    raise ValueError("只支持 CSV、TSV、TXT、XLSX、XLS 文件")


def _read_pie_curve_upload_table(content: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(BytesIO(content))
    if suffix not in {".csv", ".tsv", ".txt", ""}:
        raise ValueError("只支持 CSV、TSV、TXT、XLSX、XLS 文件")

    attempts = (
        {"sep": None, "engine": "python"},
        {"sep": "\t"},
        {"sep": ","},
        {"sep": r"\s+", "engine": "python"},
    )
    last_error: Optional[Exception] = None
    for options in attempts:
        try:
            df = pd.read_csv(BytesIO(content), **options)
            if df.shape[1] > 1:
                return df
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(BytesIO(content))


def _analysis_rows_from_long_pie_table(df: pd.DataFrame, filename: str) -> List[Dict[str, Any]]:
    mz_col = _find_column(df.columns, ["mz", "m/z", "mass", "mass_number", "质量数"])
    energy_col = _find_column(
        df.columns,
        ["energy_ev", "energy", "photon_energy", "photon energy", "e_ev", "ev", "光子能量"],
    )
    intensity_col = _find_column(
        df.columns,
        [
            "intensity",
            "normalized_intensity",
            "signal",
            "area",
            "raw_area",
            "integral",
            "peak_area",
            "积分",
            "强度",
        ],
    )
    species_col = _find_column(df.columns, ["species", "name", "formula", "物种", "名称", "分子式"])
    if not all([mz_col, energy_col, intensity_col]):
        return []

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        mz = pd.to_numeric(row[mz_col], errors="coerce")
        energy = pd.to_numeric(row[energy_col], errors="coerce")
        intensity = pd.to_numeric(row[intensity_col], errors="coerce")
        if pd.isna(mz) or pd.isna(energy) or pd.isna(intensity):
            continue
        species = "" if not species_col or pd.isna(row[species_col]) else str(row[species_col]).strip()
        rows.append(
            {
                "energy": float(energy),
                "file_count": 1,
                "io": 1.0,
                "light_source": "uploaded",
                "reference_energy": float(energy),
                "reference_source": filename,
                "mz": float(mz),
                "mz_rounded": int(round(float(mz))),
                "species": species,
                "photon_normalized_intensity": float(intensity),
                "normalized_intensity": float(intensity),
                "raw_area": float(intensity),
                "left_bound": None,
                "right_bound": None,
                "source_file": filename,
            }
        )
    return rows


def _analysis_rows_from_wide_pie_table(df: pd.DataFrame, filename: str) -> List[Dict[str, Any]]:
    energy_col = _find_column(
        df.columns,
        ["energy_ev", "energy", "photon_energy", "photon energy", "e_ev", "ev", "光子能量"],
    )
    if not energy_col and len(df.columns) >= 2:
        first_col = df.columns[0]
        numeric = pd.to_numeric(df[first_col], errors="coerce")
        if int(numeric.notna().sum()) > 0:
            energy_col = first_col
    if not energy_col:
        return []

    skip_cols = {energy_col}
    for alias_group in (
        ["species", "name", "formula", "物种", "名称", "分子式"],
        ["io", "beam_current", "light_source"],
    ):
        col = _find_column(df.columns, alias_group)
        if col:
            skip_cols.add(col)

    rows: List[Dict[str, Any]] = []
    for column in df.columns:
        if column in skip_cols:
            continue
        mz_value = _parse_mz_column(column)
        if mz_value is None:
            continue
        for _, row in df.iterrows():
            energy = pd.to_numeric(row[energy_col], errors="coerce")
            intensity = pd.to_numeric(row[column], errors="coerce")
            if pd.isna(energy) or pd.isna(intensity):
                continue
            rows.append(
                {
                    "energy": float(energy),
                    "file_count": 1,
                    "io": 1.0,
                    "light_source": "uploaded",
                    "reference_energy": float(energy),
                    "reference_source": filename,
                    "mz": float(mz_value),
                    "mz_rounded": int(round(float(mz_value))),
                    "species": "",
                    "photon_normalized_intensity": float(intensity),
                    "normalized_intensity": float(intensity),
                    "raw_area": float(intensity),
                    "left_bound": None,
                    "right_bound": None,
                    "source_file": filename,
                }
            )
    return rows


def _parse_uploaded_pie_curves(content: bytes, filename: str) -> pd.DataFrame:
    if not content:
        raise ValueError("上传文件为空")
    df = _read_pie_curve_upload_table(content, filename)
    if df.empty:
        raise ValueError("上传文件没有数据行")
    rows = _analysis_rows_from_long_pie_table(df, filename)
    if not rows:
        rows = _analysis_rows_from_wide_pie_table(df, filename)
    if not rows:
        raise ValueError("无法识别PIE曲线表格。长表需要 mz/energy/intensity；宽表需要 energy 列和多个 m/z 列。")
    return pd.DataFrame(rows)


def _records_from_long_table(df: pd.DataFrame) -> List[Dict[str, Any]]:
    mz_col = _find_column(df.columns, ["mz", "m/z", "mass", "mass_number", "质量数"])
    name_col = _find_column(df.columns, ["name", "species", "formula", "molecule", "物种", "名称", "分子式"])
    ie_col = _find_column(df.columns, ["ionization_energy", "ie", "ionization energy", "电离能", "ie_ev"])
    energy_col = _find_column(df.columns, ["energy_ev", "energy", "photon_energy", "photon energy", "e_ev", "光子能量"])
    cross_col = _find_column(
        df.columns,
        ["cross_section", "cross section", "pics", "sigma", "cross", "截面", "光电离截面"],
    )
    if not all([mz_col, name_col, energy_col, cross_col]):
        return []

    working = df[[mz_col, name_col, energy_col, cross_col] + ([ie_col] if ie_col else [])].copy()
    working = working.rename(
        columns={
            mz_col: "mz",
            name_col: "species",
            energy_col: "energy",
            cross_col: "cross_section",
        }
    )
    if ie_col:
        working = working.rename(columns={ie_col: "ie"})
    else:
        working["ie"] = None
    working["mz"] = pd.to_numeric(working["mz"], errors="coerce")
    working["energy"] = pd.to_numeric(working["energy"], errors="coerce")
    working["cross_section"] = pd.to_numeric(working["cross_section"], errors="coerce")
    working["ie"] = pd.to_numeric(working["ie"], errors="coerce")
    working["species"] = working["species"].astype(str).str.strip()
    working = working.dropna(subset=["mz", "energy", "cross_section"])
    working = working[working["species"] != ""]

    records = []
    for (mz, species, ie), group in working.groupby(["mz", "species", "ie"], dropna=False):
        points = (
            group.groupby("energy", as_index=False)["cross_section"]
            .mean()
            .sort_values("energy")
        )
        if len(points) < 2:
            continue
        records.append(
            {
                "mz": int(round(float(mz))),
                "species": str(species),
                "ie": None if pd.isna(ie) else float(ie),
                "energies": points["energy"].astype(float).tolist(),
                "cross_sections": points["cross_section"].astype(float).tolist(),
            }
        )
    return records


def _records_from_wide_table(df: pd.DataFrame) -> List[Dict[str, Any]]:
    mz_col = _find_column(df.columns, ["mz", "m/z", "mass", "mass_number", "质量数"])
    name_col = _find_column(df.columns, ["name", "species", "formula", "molecule", "物种", "名称", "分子式"])
    ie_col = _find_column(df.columns, ["ionization_energy", "ie", "ionization energy", "电离能", "ie_ev"])
    if not mz_col or not name_col:
        return []
    metadata_cols = {mz_col, name_col}
    if ie_col:
        metadata_cols.add(ie_col)

    energy_columns = []
    for column in df.columns:
        if column in metadata_cols:
            continue
        energy = _parse_energy_column(column)
        if energy is not None:
            energy_columns.append((column, energy))
    if not energy_columns:
        return []

    records = []
    for _, row in df.iterrows():
        try:
            mz = int(round(float(row[mz_col])))
        except Exception:
            continue
        species = str(row[name_col]).strip()
        if not species:
            continue
        ie = None
        if ie_col and not pd.isna(row[ie_col]):
            ie = float(row[ie_col])
        energies = []
        cross_sections = []
        for column, energy in energy_columns:
            value = pd.to_numeric(row[column], errors="coerce")
            if pd.isna(value):
                continue
            energies.append(float(energy))
            cross_sections.append(float(value))
        if len(energies) < 2:
            continue
        order = sorted(range(len(energies)), key=lambda idx: energies[idx])
        records.append(
            {
                "mz": mz,
                "species": species,
                "ie": ie,
                "energies": [energies[idx] for idx in order],
                "cross_sections": [cross_sections[idx] for idx in order],
            }
        )
    return records


def _parse_pics_upload(content: bytes, filename: str) -> List[Dict[str, Any]]:
    if not content:
        raise ValueError("上传文件为空")
    df = _read_pics_upload_table(content, filename)
    if df.empty:
        raise ValueError("上传文件没有数据行")
    records = _records_from_long_table(df)
    if not records:
        records = _records_from_wide_table(df)
    if not records:
        raise ValueError(
            "无法识别PICS表格。长表需要 mz/name/energy_ev/cross_section；宽表需要 mz/name 和多个能量列。"
        )
    return records


def _backup_database(database_path: Path) -> Optional[Path]:
    if not database_path.exists():
        return None
    backup_dir = database_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{database_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}{database_path.suffix}"
    shutil.copy2(database_path, backup_path)
    return backup_path


def _matching_species_ids(conn: sqlite3.Connection, record: Dict[str, Any]) -> List[int]:
    if record["ie"] is None:
        rows = conn.execute(
            """
            SELECT id FROM species
            WHERE mz = ? AND LOWER(name) = LOWER(?) AND ionization_energy IS NULL
            """,
            (int(record["mz"]), str(record["species"])),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id FROM species
            WHERE mz = ? AND LOWER(name) = LOWER(?) AND ABS(ionization_energy - ?) < 1e-6
            """,
            (int(record["mz"]), str(record["species"]), float(record["ie"])),
        ).fetchall()
    return [int(row[0]) for row in rows]


def _write_pics_records_to_path(
    records: List[Dict[str, Any]],
    database_path: Path,
    mode: str,
    confirm_overwrite: bool,
    *,
    backup: bool,
) -> Dict[str, Any]:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = _backup_database(database_path) if backup else None
    inserted_species = 0
    replaced_species = 0
    inserted_points = 0
    mode = mode if mode in {"upsert", "append", "overwrite_all"} else "upsert"
    if mode == "overwrite_all" and not confirm_overwrite:
        raise ValueError("覆盖整个PICS库需要确认")

    with sqlite3.connect(database_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_SQL)
        if mode == "overwrite_all":
            conn.execute("DELETE FROM pic_cross_sections")
            conn.execute("DELETE FROM species")
        for record in records:
            if mode == "upsert":
                ids = _matching_species_ids(conn, record)
                if ids:
                    replaced_species += len(ids)
                    conn.executemany("DELETE FROM species WHERE id = ?", [(species_id,) for species_id in ids])

            cursor = conn.execute(
                "INSERT INTO species (mz, name, ionization_energy) VALUES (?, ?, ?)",
                (int(record["mz"]), str(record["species"]), record["ie"]),
            )
            species_id = int(cursor.lastrowid)
            rows = [
                (species_id, float(energy), float(cross_section))
                for energy, cross_section in zip(record["energies"], record["cross_sections"])
                if np_is_finite(energy) and np_is_finite(cross_section)
            ]
            if len(rows) < 2:
                conn.execute("DELETE FROM species WHERE id = ?", (species_id,))
                continue
            conn.executemany(
                "INSERT INTO pic_cross_sections (species_id, energy_ev, cross_section) VALUES (?, ?, ?)",
                rows,
            )
            inserted_species += 1
            inserted_points += len(rows)
        conn.commit()
    _load_species_database_cached.cache_clear()
    return {
        "mode": mode,
        "database": str(database_path),
        "backup": "" if backup_path is None else str(backup_path),
        "parsed_species": len(records),
        "inserted_species": inserted_species,
        "replaced_species": replaced_species,
        "inserted_points": inserted_points,
    }


def _write_pics_records(records: List[Dict[str, Any]], mode: str, confirm_overwrite: bool) -> Dict[str, Any]:
    return _write_pics_records_to_path(
        records,
        _default_database_path(),
        mode,
        confirm_overwrite,
        backup=True,
    )


def _create_session_pics_library(records: List[Dict[str, Any]], filename: str) -> Dict[str, Any]:
    _cleanup_expired_pics_libraries()
    library_id = uuid.uuid4().hex
    database_path = Path(tempfile.gettempdir()) / "bl03u_mstool_pics" / f"{library_id}.sqlite"
    result = _write_pics_records_to_path(
        records,
        database_path,
        mode="overwrite_all",
        confirm_overwrite=True,
        backup=False,
    )
    library = PicsLibrary(
        id=library_id,
        database_path=str(database_path),
        source_file=filename,
        parsed_species=int(result["parsed_species"]),
        inserted_points=int(result["inserted_points"]),
    )
    with PICS_LIBRARIES_LOCK:
        PICS_LIBRARIES[library_id] = library
    result.update(
        {
            "scope": "session",
            "library_id": library_id,
            "source_file": filename,
        }
    )
    return result


def np_is_finite(value: object) -> bool:
    try:
        return bool(pd.notna(value) and math.isfinite(float(value)))
    except Exception:
        return False


def _pie_summary(analysis_df: pd.DataFrame, curves: Dict[int, dict]) -> Dict[str, Any]:
    energy_count = int(analysis_df["energy"].nunique()) if not analysis_df.empty else 0
    file_count = (
        int(analysis_df.groupby("energy")["file_count"].first().sum())
        if not analysis_df.empty and "file_count" in analysis_df
        else 0
    )
    return {
        "curve_count": len(curves),
        "energy_count": energy_count,
        "row_count": len(analysis_df),
        "file_count": file_count,
        "mz_values": sorted(curves),
    }


def _query_fit_candidates(database_path: Path, mz: int) -> List[Dict[str, Any]]:
    with sqlite3.connect(database_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                s.id,
                s.mz,
                s.name,
                s.ionization_energy,
                COUNT(p.id) AS point_count,
                MIN(p.energy_ev) AS energy_min,
                MAX(p.energy_ev) AS energy_max,
                MAX(p.cross_section) AS cross_section_max
            FROM species s
            LEFT JOIN pic_cross_sections p ON p.species_id = s.id
            WHERE s.mz = ?
            GROUP BY s.id, s.mz, s.name, s.ionization_energy
            ORDER BY s.name, s.id
            """,
            (int(mz),),
        ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "mz": int(row["mz"]),
            "name": row["name"],
            "ionization_energy": _float_or_none(row["ionization_energy"]),
            "point_count": int(row["point_count"] or 0),
            "energy_min": _float_or_none(row["energy_min"]),
            "energy_max": _float_or_none(row["energy_max"]),
            "cross_section_max": _float_or_none(row["cross_section_max"]),
        }
        for row in rows
    ]


def _load_species_records_by_ids(database_path: Path, species_ids: List[int]) -> List[dict]:
    unique_ids = sorted({int(value) for value in species_ids})
    if not unique_ids:
        return []
    placeholders = ",".join("?" for _ in unique_ids)
    records: List[dict] = []
    with sqlite3.connect(database_path) as conn:
        conn.row_factory = sqlite3.Row
        species_rows = conn.execute(
            f"""
            SELECT id, mz, name, ionization_energy
            FROM species
            WHERE id IN ({placeholders})
            ORDER BY mz, name, id
            """,
            unique_ids,
        ).fetchall()
        for species_row in species_rows:
            points = conn.execute(
                """
                SELECT energy_ev, cross_section
                FROM pic_cross_sections
                WHERE species_id = ?
                ORDER BY energy_ev
                """,
                (int(species_row["id"]),),
            ).fetchall()
            if len(points) < 2:
                continue
            records.append(
                {
                    "id": int(species_row["id"]),
                    "mz": int(species_row["mz"]),
                    "species": species_row["name"],
                    "ie": _float_or_none(species_row["ionization_energy"]),
                    "energies": [float(point["energy_ev"]) for point in points],
                    "cross_sections": [float(point["cross_section"]) for point in points],
                }
            )
    return records


def _parse_species_ids(value: Optional[str]) -> List[int]:
    if not value:
        return []
    ids: List[int] = []
    for token in re.split(r"[,，;；\s]+", value.strip()):
        if not token:
            continue
        try:
            ids.append(int(token))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"无法解析候选 PICS id: {token}")
    return sorted(set(ids))


def _run_pie_job(job_id: str, payload: PieStartPayload) -> None:
    try:
        _update_job(job_id, status="running", message="读取配置与文件路径")
        folder = _resolve_path(payload.folder)
        try:
            database_path = _database_path_from_request(
                database=payload.database,
                library_id=payload.library_id,
            )
        except HTTPException as exc:
            if payload.library_id:
                raise FileNotFoundError("临时 PICS 工作库不存在，请重新上传 PICS 数据")
            raise FileNotFoundError(str(exc.detail))
        if not folder.exists() or not folder.is_dir():
            raise FileNotFoundError(f"PIE扫描文件夹不存在: {folder}")

        manual_peak_path: Optional[str] = None
        peak_source = payload.peak_source
        if peak_source == "manual":
            if not payload.manual_peak_path:
                raise ValueError("手动卡峰模式需要提供卡峰文件")
            manual_peak_file = _resolve_path(payload.manual_peak_path)
            if not manual_peak_file.exists() or not manual_peak_file.is_file():
                raise FileNotFoundError(f"手动卡峰文件不存在: {manual_peak_file}")
            manual_peak_path = str(manual_peak_file)

        peak_config = load_peak_detection_config()
        calibration = load_calibration_config()
        normalization = load_normalization_settings()
        photon_mode = payload.photon_mode or normalization.pie_photon_mode
        light_source = payload.light_source or normalization.light_source
        mass_discrimination = (
            float(payload.mass_discrimination)
            if payload.mass_discrimination is not None
            else float(normalization.mass_discrimination)
        )
        target_mz_values = _parse_mz_values(payload.target_mz)

        _update_job(job_id, message="正在生成PIE曲线")
        analysis_df = analyze_pie_folder(
            folder,
            calibration=calibration,
            recursive=payload.recursive,
            energy_decimals=payload.energy_decimals,
            threshold_end=peak_config.threshold_end,
            min_intensity=peak_config.min_intensity,
            detection_min_idx=peak_config.detection_min_idx,
            nearby_peak_window=peak_config.nearby_peak_window,
            duplicate_window=peak_config.duplicate_window,
            weak_tail_early_window=peak_config.weak_tail_early_window,
            weak_tail_late_window=peak_config.weak_tail_late_window,
            weak_tail_ratio=peak_config.weak_tail_ratio,
            gaussian_window_max=peak_config.gaussian_window_max,
            gaussian_boundary_scale=peak_config.gaussian_boundary_scale,
            boundary_padding=peak_config.boundary_padding,
            prefer_gaussian=payload.gaussian,
            manual_peak_path=manual_peak_path,
            photon_normalize=photon_mode != "off",
            photon_reference_mode="none" if photon_mode == "off" else photon_mode,
            mass_discrimination=mass_discrimination,
            light_source=light_source,
            target_mz_values=target_mz_values,
        )
        curves = build_pie_curves(analysis_df)
        summary = _pie_summary(analysis_df, curves)
        _update_job(
            job_id,
            status="done",
            message="完成",
            summary=summary,
            analysis_df=analysis_df,
            curves=curves,
            database_path=str(database_path),
            source_folder=str(folder),
            peak_source=peak_source,
        )
    except Exception as exc:
        _update_job(job_id, status="error", message="失败", error=str(exc))


@app.get("/", include_in_schema=False)
def web_index():
    index = WEB_ROOT / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="web frontend not found")
    return FileResponse(index)


@app.post("/api/pie/start")
def start_pie_job(payload: PieStartPayload):
    _cleanup_jobs()
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = PieJob(id=job_id)
    EXECUTOR.submit(_run_pie_job, job_id, payload)
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/pie/upload_curve")
async def upload_pie_curve(
    request: Request,
    filename: str = "pie_curve.csv",
    database: Optional[str] = None,
    library_id: Optional[str] = None,
):
    try:
        database_path = _database_path_from_request(database=database, library_id=library_id)
    except HTTPException as exc:
        if library_id:
            raise HTTPException(status_code=404, detail="临时 PICS 工作库不存在，请重新上传 PICS 数据")
        raise exc
    try:
        content = await request.body()
        _check_upload_size(content)
        analysis_df = _parse_uploaded_pie_curves(content, filename)
        curves = build_pie_curves(analysis_df)
        if not curves:
            raise ValueError("上传文件没有可用 PIE 曲线")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    summary = _pie_summary(analysis_df, curves)
    _cleanup_jobs()
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = PieJob(
            id=job_id,
            status="done",
            message="完成",
            summary=summary,
            analysis_df=analysis_df,
            curves=curves,
            database_path=str(database_path),
            source_folder=filename,
            peak_source="uploaded_curve",
        )
    return {"job_id": job_id, "status": "done", "summary": summary}


@app.get("/api/pie/progress/{job_id}")
def pie_progress(job_id: str):
    job = _get_job(job_id)
    return {
        "job_id": job.id,
        "status": job.status,
        "message": job.message,
        "error": job.error,
        "elapsed": round(time.time() - job.created_at, 3),
        "summary": job.summary if job.status == "done" else {},
    }


@app.get("/api/pie/result/{job_id}")
def pie_result(job_id: str):
    job = _get_job(job_id)
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"job status is {job.status}")
    return {
        "job_id": job.id,
        "summary": job.summary,
        "source_folder": job.source_folder,
        "peak_source": job.peak_source,
    }


@app.get("/api/pie/curve/{job_id}/{mz}")
def pie_curve(job_id: str, mz: int):
    job = _get_job(job_id)
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"job status is {job.status}")
    curve = job.curves.get(int(mz))
    if curve is None:
        raise HTTPException(status_code=404, detail="curve not found")
    return {
        "curve": _curve_payload(curve),
        "fit": job.fits.get(int(mz)),
    }


@app.get("/api/pie/candidates/{job_id}/{mz}")
def pie_fit_candidates(job_id: str, mz: int):
    job = _get_job(job_id)
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"job status is {job.status}")
    database_path = Path(job.database_path)
    if not database_path.exists() or not database_path.is_file():
        raise HTTPException(status_code=404, detail="SQLite PICS库不存在")
    return {"rows": _query_fit_candidates(database_path, int(mz))}


@app.get("/api/pie/candidate_curves/{job_id}/{mz}")
def pie_candidate_curves(job_id: str, mz: int, species_ids: str = ""):
    job = _get_job(job_id)
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"job status is {job.status}")
    database_path = Path(job.database_path)
    if not database_path.exists() or not database_path.is_file():
        raise HTTPException(status_code=404, detail="SQLite PICS库不存在")
    selected_ids = _parse_species_ids(species_ids)
    if not selected_ids:
        selected_ids = [row["id"] for row in _query_fit_candidates(database_path, int(mz))]
    rows = [
        {
            "id": int(item["id"]),
            "mz": int(item["mz"]),
            "species": item["species"],
            "ie": item.get("ie"),
            "energies": [float(value) for value in item["energies"]],
            "cross_sections": [float(value) for value in item["cross_sections"]],
        }
        for item in _load_species_records_by_ids(database_path, selected_ids)
        if int(item["mz"]) == int(mz)
    ]
    return {"rows": rows}


@app.post("/api/pie/fit/{job_id}/{mz}")
def fit_pie_curve(job_id: str, mz: int, payload: Optional[PieFitPayload] = None):
    job = _get_job(job_id)
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"job status is {job.status}")
    curve = job.curves.get(int(mz))
    if curve is None:
        raise HTTPException(status_code=404, detail="curve not found")
    database_path = Path(job.database_path)
    selected_ids = payload.species_ids if payload and payload.species_ids else []
    if selected_ids:
        species = _load_species_records_by_ids(database_path, selected_ids)
        if not species:
            raise HTTPException(status_code=400, detail="未找到可拟合的 PICS 候选")
        wrong_mz = sorted({int(item["mz"]) for item in species if int(item["mz"]) != int(mz)})
        if wrong_mz:
            raise HTTPException(status_code=400, detail=f"选择的 PICS m/z 与当前曲线不一致: {wrong_mz}")
        fit_model = fit_species_combination_with_curve(
            species,
            curve["energies"],
            curve["intensities"],
            coefficient_mode=payload.coefficient_mode if payload else "fit",
            coefficients=payload.coefficients if payload else None,
            locked_species_ids=payload.locked_species_ids if payload else None,
        )
        fit_model["selection_mode"] = "manual"
        fit_model["selected_species_ids"] = [int(value) for value in selected_ids]
    else:
        database = _load_species_database_cached(str(database_path), database_path.stat().st_mtime)
        fit_model = identify_species_for_mz_with_curve(
            database,
            int(mz),
            curve["energies"],
            curve["intensities"],
        )
        fit_model["selection_mode"] = "auto"
        fit_model["selected_species_ids"] = []
    with JOBS_LOCK:
        job.fits[int(mz)] = fit_model
        job.updated_at = time.time()
    return {"fit": fit_model}


@app.get("/api/pie/export/{job_id}")
def pie_export(job_id: str):
    job = _get_job(job_id)
    if job.status != "done":
        raise HTTPException(status_code=409, detail=f"job status is {job.status}")
    return {"rows": _json_records(job.analysis_df)}


@app.get("/api/pics/search")
def pics_search(
    database: Optional[str] = None,
    library_id: Optional[str] = None,
    mz: Optional[int] = None,
    tolerance: int = 0,
    name: str = "",
    ie_min: Optional[float] = None,
    ie_max: Optional[float] = None,
    limit: int = 100,
):
    database_path = _database_path_from_request(database=database, library_id=library_id)
    limit = max(1, min(int(limit), 500))
    clauses = []
    params: List[Any] = []
    if mz is not None:
        tolerance = max(0, int(tolerance))
        clauses.append("s.mz BETWEEN ? AND ?")
        params.extend([int(mz) - tolerance, int(mz) + tolerance])
    if name.strip():
        clauses.append("LOWER(s.name) LIKE ?")
        params.append(f"%{name.strip().lower()}%")
    if ie_min is not None:
        clauses.append("s.ionization_energy >= ?")
        params.append(float(ie_min))
    if ie_max is not None:
        clauses.append("s.ionization_energy <= ?")
        params.append(float(ie_max))
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    sql = f"""
        SELECT
            s.id,
            s.mz,
            s.name,
            s.ionization_energy,
            COUNT(p.id) AS point_count,
            MIN(p.energy_ev) AS energy_min,
            MAX(p.energy_ev) AS energy_max,
            MAX(p.cross_section) AS cross_section_max
        FROM species s
        LEFT JOIN pic_cross_sections p ON p.species_id = s.id
        {where_sql}
        GROUP BY s.id, s.mz, s.name, s.ionization_energy
        ORDER BY s.mz, s.name, s.id
        LIMIT ?
    """
    params.append(limit)
    with sqlite3.connect(database_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    return {
        "rows": [
            {
                "id": int(row["id"]),
                "mz": int(row["mz"]),
                "name": row["name"],
                "ionization_energy": _float_or_none(row["ionization_energy"]),
                "point_count": int(row["point_count"] or 0),
                "energy_min": _float_or_none(row["energy_min"]),
                "energy_max": _float_or_none(row["energy_max"]),
                "cross_section_max": _float_or_none(row["cross_section_max"]),
            }
            for row in rows
        ]
    }


@app.get("/api/pics/species/{species_id}")
def pics_species(
    species_id: int,
    database: Optional[str] = None,
    library_id: Optional[str] = None,
):
    database_path = _database_path_from_request(database=database, library_id=library_id)
    with sqlite3.connect(database_path) as conn:
        conn.row_factory = sqlite3.Row
        species = conn.execute(
            "SELECT id, mz, name, ionization_energy FROM species WHERE id = ?",
            (int(species_id),),
        ).fetchone()
        if species is None:
            raise HTTPException(status_code=404, detail="species not found")
        points = conn.execute(
            """
            SELECT energy_ev, cross_section
            FROM pic_cross_sections
            WHERE species_id = ?
            ORDER BY energy_ev
            """,
            (int(species_id),),
        ).fetchall()
    return {
        "species": {
            "id": int(species["id"]),
            "mz": int(species["mz"]),
            "name": species["name"],
            "ionization_energy": _float_or_none(species["ionization_energy"]),
        },
        "points": [
            {
                "energy_ev": float(point["energy_ev"]),
                "cross_section": float(point["cross_section"]),
            }
            for point in points
        ],
    }


@app.post("/api/pics/upload")
async def pics_upload(
    request: Request,
    filename: str = "pics.csv",
    scope: str = "session",
    mode: str = "upsert",
    confirm_overwrite: bool = False,
):
    content = await request.body()
    _check_upload_size(content)
    try:
        records = _parse_pics_upload(content, filename)
        if scope == "server":
            _verify_admin_token(request)
            result = _write_pics_records(records, mode=mode, confirm_overwrite=confirm_overwrite)
            result["scope"] = "server"
            result["library_id"] = ""
        else:
            result = _create_session_pics_library(records, filename)
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.post("/isotope")
def isotope(payload: IsotopePayload):
    return {"distribution": calculate_isotope_distribution(payload.formula, min_percent=payload.min_percent)}


@app.post("/spectrum/read")
async def read_spectrum(request: Request, filename: str = "spectrum.txt"):
    suffix = Path(filename).suffix or ".txt"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(await request.body())
            temp_path = handle.name
        spectrum = read_bl03u_txt(temp_path)
        return {"x": spectrum.x.tolist(), "y": spectrum.y.tolist()}
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


@app.post("/spectrum/peaks")
async def detect_peaks(
    request: Request,
    filename: str = "spectrum.txt",
    a: float = 3.66334e-7,
    b: float = 0.000637719,
    c: float = 0.272489072,
):
    suffix = Path(filename).suffix or ".txt"
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(await request.body())
            temp_path = handle.name
        spectrum = read_bl03u_txt(temp_path)
        offset = float(spectrum.x[0]) if len(spectrum.x) else 0.0
        peaks = detect_peaks_in_range(
            spectrum.y,
            calibration=Calibration(a, b, c),
            detection_min_idx=0,
            time_offset=offset,
        )
        return {"peaks": [peak.to_dict() for peak in peaks]}
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


@app.get("/spectrum/sum")
def sum_folder(folder: str):
    try:
        folder_path = _resolve_path(folder)
        spectrum = sum_spectra(folder_path)
    except PathAccessError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"x": spectrum.x.tolist(), "y": spectrum.y.tolist()}
