from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from io import BytesIO
import logging
import math
import os
from pathlib import Path
import re
import sqlite3
import shutil
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi import FastAPI, Request
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from bl03u_masstool.api.jobs import (
    DEFAULT_JOB_TTL_HOURS,
    DEFAULT_MAX_JOBS,
    JOBS,
    JOBS_LOCK,
    PICS_LIBRARIES,
    PICS_LIBRARIES_LOCK,
    PICS_LIBRARY_TTL_SECONDS,
    PicsLibrary,
    PieJob,
    cleanup_expired_pics_libraries as _cleanup_expired_pics_libraries,
    cleanup_jobs as _cleanup_jobs,
    get_job as _get_job,
    get_pics_library as _get_pics_library,
    job_ttl_seconds as _job_ttl_seconds,
    max_jobs as _max_jobs,
    update_job as _update_job,
)
from bl03u_masstool.api.schemas import IsotopePayload, PieFitPayload, PieStartPayload
from bl03u_masstool.core.analysis_artifacts import build_analysis_manifest, build_pie_evidence_objects
from bl03u_masstool.core.config import (
    PROJECT_ROOT,
    load_calibration_config,
    load_peak_detection_config,
    species_database_path,
)
from bl03u_masstool.core.isotope import calculate_isotope_distribution
from bl03u_masstool.core.normalization import load_normalization_settings
from bl03u_masstool.core.peak_detection import detect_peaks_in_range
from bl03u_masstool.core.pie_analysis import (
    SCHEMA_SQL,
    analyze_pie_folder,
    build_pie_curves,
    fit_species_combination_with_curve,
    identify_species_for_mz_with_curve,
    load_species_database,
)
from bl03u_masstool.core.pics_import import (
    _clean_column_name,
    _find_column,
    _parse_energy_column,
    parse_pics_upload as _parse_pics_upload,
    write_pics_records,
)
from bl03u_masstool.core.spectrum_io import read_bl03u_txt, sum_spectra
from bl03u_masstool.logging_config import configure_logging


configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="BL03U MassSpectrumTool API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PACKAGE_ROOT / "frontends" / "web_app" / "static"
if WEB_ROOT.exists():
    app.mount("/static", StaticFiles(directory=WEB_ROOT), name="static")


@app.middleware("http")
async def log_unhandled_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        logger.exception("Unhandled API error: %s %s", request.method, request.url.path)
        raise

EXECUTOR = ThreadPoolExecutor(max_workers=2)
MAX_UPLOAD_BYTES = int(float(os.environ.get("BL03U_MAX_UPLOAD_MB", "20")) * 1024 * 1024)


class PathAccessError(ValueError):
    """Raised when a user-supplied path escapes configured data roots."""


def _json_records(df) -> List[Dict[str, Any]]:
    if df is None or getattr(df, "empty", True):
        return []
    return df.where(df.notna(), None).to_dict(orient="records")


def _float_or_none(value) -> Optional[float]:
    return None if value is None else float(value)


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


def _backup_database(database_path: Path) -> Optional[Path]:
    if not database_path.exists():
        return None
    backup_dir = database_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{database_path.stem}-{time.strftime('%Y%m%d-%H%M%S')}{database_path.suffix}"
    shutil.copy2(database_path, backup_path)
    return backup_path


def _write_pics_records_to_path(
    records: List[Dict[str, Any]],
    database_path: Path,
    mode: str,
    confirm_overwrite: bool,
    *,
    backup: bool,
) -> Dict[str, Any]:
    backup_path = _backup_database(database_path) if backup else None
    result = write_pics_records(
        records, database_path, mode=mode, confirm_overwrite=confirm_overwrite
    )
    _load_species_database_cached.cache_clear()
    result["database"] = str(database_path)
    result["backup"] = "" if backup_path is None else str(backup_path)
    result["parsed_species"] = len(records)
    return result


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


def _pie_artifacts(
    *,
    source_label: str,
    peak_source: str,
    database_reference: Dict[str, Any],
    payload: PieStartPayload | None,
    analysis_df: pd.DataFrame,
    curves: Dict[int, dict],
    fits: Dict[int, dict] | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    parameters: Dict[str, Any] = {"peak_source": peak_source, **database_reference}
    if payload is not None:
        parameters.update(
            {
                "recursive": payload.recursive,
                "energy_decimals": payload.energy_decimals,
                "prefer_gaussian": payload.gaussian,
                "manual_peak_path": payload.manual_peak_path,
                "target_mz": payload.target_mz,
                "photon_mode": payload.photon_mode,
                "light_source": payload.light_source,
                "mass_discrimination": payload.mass_discrimination,
            }
        )
    manifest = build_analysis_manifest(
        analysis_type="pie",
        input_path=source_label,
        parameters=parameters,
        analysis_df=analysis_df,
        curves=curves,
    )
    evidence = build_pie_evidence_objects(curves, fits or {})
    return manifest, evidence


def _public_path_label(path: str | Path) -> str:
    path_obj = Path(path)
    try:
        resolved = path_obj.resolve(strict=False)
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return resolved.name or path_obj.name or str(path)


def _database_artifact_reference(
    *,
    database: Optional[str] = None,
    library_id: Optional[str] = None,
    resolved_path: Path,
) -> Dict[str, Any]:
    if library_id:
        return {"database_scope": "session", "library_id": library_id}
    if database:
        return {"database_scope": "custom", "database_name": Path(database).name or resolved_path.name}
    return {"database_scope": "server", "database_name": resolved_path.name}


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
        peak_kwargs = peak_config.to_peak_kwargs()
        analysis_df = analyze_pie_folder(
            folder,
            calibration=calibration,
            recursive=payload.recursive,
            energy_decimals=payload.energy_decimals,
            **peak_kwargs,
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
        manifest, evidence = _pie_artifacts(
            source_label=_public_path_label(folder),
            peak_source=peak_source,
            database_reference=_database_artifact_reference(
                database=payload.database,
                library_id=payload.library_id,
                resolved_path=database_path,
            ),
            payload=payload,
            analysis_df=analysis_df,
            curves=curves,
        )
        _update_job(
            job_id,
            status="done",
            message="完成",
            summary=summary,
            analysis_df=analysis_df,
            curves=curves,
            database_path=str(database_path),
            source_folder=_public_path_label(folder),
            peak_source=peak_source,
            manifest=manifest,
            evidence=evidence,
        )
    except Exception as exc:
        _update_job(job_id, status="error", message="失败", error=str(exc))


def _register_routes() -> None:
    from bl03u_masstool.api.routes import isotope as isotope_routes
    from bl03u_masstool.api.routes import pics as pics_routes
    from bl03u_masstool.api.routes import pie as pie_routes
    from bl03u_masstool.api.routes import spectrum as spectrum_routes
    from bl03u_masstool.api.routes import web as web_routes

    route_deps = sys.modules[__name__]
    app.include_router(web_routes.create_router(route_deps))
    app.include_router(pie_routes.create_router(route_deps))
    app.include_router(pics_routes.create_router(route_deps))
    app.include_router(isotope_routes.create_router(route_deps))
    app.include_router(spectrum_routes.create_router(route_deps))


_register_routes()
