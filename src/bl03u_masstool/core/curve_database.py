from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .project_settings import ProjectSettings


SCHEMA_VERSION = 5
CURVE_TYPES = {"temperature", "pie"}
DATASET_ROLES = {"observed", "corrected", "derived"}
VALIDITY_STATUSES = {"valid", "stale", "archived"}


@dataclass(frozen=True)
class CurveDataset:
    dataset_id: int
    dataset_key: str
    curve_type: str
    name: str
    source_label: str
    created_at: str
    updated_at: str
    channel_count: int
    point_count: int
    is_current: bool
    stale_reason: str
    dataset_group: str
    analysis_key: str
    dataset_role: str
    parent_dataset_id: int | None
    validity_status: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CurveDatasetState:
    dataset_id: int
    curve_type: str
    is_current: bool
    validity_status: str
    stale_reason: str
    source_label: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class CurveChannelSummary:
    channel_id: int
    dataset_id: int
    exact_mz: float
    nominal_mz: int
    peak_track: int | None
    left_bound: float | None
    right_bound: float | None
    species_label: str
    point_count: int
    energy_count: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CurveChannel:
    channel_id: int
    dataset_id: int
    curve_type: str
    exact_mz: float
    nominal_mz: int
    peak_track: int | None
    left_bound: float | None
    right_bound: float | None
    species_label: str
    rows: pd.DataFrame


def project_curve_database_path(settings: ProjectSettings) -> Path:
    """Return the project-owned curve database path.

    The import is intentionally local to avoid making project settings depend on
    the lifecycle module.
    """
    configured = str(getattr(settings, "curve_database_path", "") or "").strip()
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_absolute():
            return configured_path
        from .project_lifecycle import project_root

        return project_root(settings) / configured_path
    from .project_lifecycle import project_root

    return project_root(settings) / "analysis" / "curve_data.sqlite"


def _connect(path: str | Path) -> sqlite3.Connection:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def _connect_readonly(path: str | Path) -> sqlite3.Connection:
    database_path = Path(path)
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    # ``immutable=1`` guarantees the isolated Demo cannot create -shm/-wal
    # sidecars.  A live WAL must still be honored for newly written project
    # batches, so use ordinary read-only mode while that sidecar exists.
    immutable = "" if database_path.with_name(database_path.name + "-wal").exists() else "&immutable=1"
    connection = sqlite3.connect(
        f"file:{database_path.resolve().as_posix()}?mode=ro{immutable}",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def _database_user_version(path: Path) -> int:
    if not path.is_file():
        return 0
    connection = sqlite3.connect(path)
    try:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])
    finally:
        connection.close()


def _backup_database_before_migration(path: Path, from_version: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = path.with_name(
        f"{path.name}.v{from_version}-backup-{timestamp}.sqlite"
    )
    suffix = 1
    while backup_path.exists():
        backup_path = path.with_name(
            f"{path.name}.v{from_version}-backup-{timestamp}-{suffix}.sqlite"
        )
        suffix += 1
    source = sqlite3.connect(path)
    destination = sqlite3.connect(backup_path)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    return backup_path


def initialize_curve_database(path: str | Path) -> Path:
    database_path = Path(path)
    previous_version = _database_user_version(database_path)
    if database_path.is_file() and 0 < previous_version < SCHEMA_VERSION:
        _backup_database_before_migration(database_path, previous_version)
    with _connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS curve_datasets (
                dataset_id INTEGER PRIMARY KEY,
                dataset_key TEXT NOT NULL UNIQUE,
                curve_type TEXT NOT NULL CHECK (curve_type IN ('temperature', 'pie')),
                name TEXT NOT NULL,
                source_label TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL DEFAULT '',
                is_current INTEGER NOT NULL DEFAULT 1,
                stale_reason TEXT NOT NULL DEFAULT '',
                dataset_group TEXT NOT NULL DEFAULT '',
                analysis_key TEXT NOT NULL DEFAULT '',
                dataset_role TEXT NOT NULL DEFAULT 'observed',
                parent_dataset_id INTEGER
                    REFERENCES curve_datasets(dataset_id),
                validity_status TEXT NOT NULL DEFAULT 'valid',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS mass_channels (
                channel_id INTEGER PRIMARY KEY,
                dataset_id INTEGER NOT NULL
                    REFERENCES curve_datasets(dataset_id) ON DELETE CASCADE,
                exact_mz REAL NOT NULL,
                nominal_mz INTEGER NOT NULL,
                peak_track INTEGER,
                left_bound REAL,
                right_bound REAL,
                species_label TEXT NOT NULL DEFAULT '',
                UNIQUE(dataset_id, exact_mz)
            );

            CREATE INDEX IF NOT EXISTS idx_mass_channels_exact_mz
                ON mass_channels(exact_mz);
            CREATE INDEX IF NOT EXISTS idx_mass_channels_dataset_exact_mz
                ON mass_channels(dataset_id, exact_mz);
            CREATE INDEX IF NOT EXISTS idx_mass_channels_nominal_mz
                ON mass_channels(nominal_mz);

            CREATE TABLE IF NOT EXISTS curve_points (
                point_id INTEGER PRIMARY KEY,
                channel_id INTEGER NOT NULL
                    REFERENCES mass_channels(channel_id) ON DELETE CASCADE,
                point_order INTEGER NOT NULL,
                axis_value REAL NOT NULL,
                photon_energy REAL,
                signals_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_curve_points_channel_axis
                ON curve_points(channel_id, axis_value);

            CREATE TABLE IF NOT EXISTS species_assignments (
                assignment_id INTEGER PRIMARY KEY,
                channel_id INTEGER NOT NULL
                    REFERENCES mass_channels(channel_id) ON DELETE CASCADE,
                assignment_status TEXT NOT NULL DEFAULT 'candidate'
                    CHECK (assignment_status IN ('candidate', 'confirmed', 'rejected')),
                species_name TEXT NOT NULL DEFAULT '',
                formula TEXT NOT NULL DEFAULT '',
                species_database_ids TEXT NOT NULL DEFAULT '[]',
                ionization_energy REAL,
                coefficient REAL,
                contribution_percent REAL,
                r_squared REAL,
                source TEXT NOT NULL DEFAULT 'pie_fit',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_species_assignments_channel
                ON species_assignments(channel_id);
            CREATE INDEX IF NOT EXISTS idx_species_assignments_formula
                ON species_assignments(formula);

            CREATE TABLE IF NOT EXISTS pie_fit_states (
                fit_state_id INTEGER PRIMARY KEY,
                dataset_id INTEGER NOT NULL
                    REFERENCES curve_datasets(dataset_id) ON DELETE CASCADE,
                channel_id INTEGER NOT NULL
                    REFERENCES mass_channels(channel_id) ON DELETE CASCADE,
                config_json TEXT NOT NULL DEFAULT '{}',
                result_json TEXT NOT NULL DEFAULT '{}',
                config_hash TEXT NOT NULL DEFAULT '',
                fit_status TEXT NOT NULL DEFAULT 'draft',
                updated_at TEXT NOT NULL,
                UNIQUE(dataset_id, channel_id)
            );

            CREATE INDEX IF NOT EXISTS idx_pie_fit_states_dataset_channel
                ON pie_fit_states(dataset_id, channel_id);
            """
        )
        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(curve_datasets)"
            ).fetchall()
        }
        if "content_hash" not in columns:
            connection.execute(
                "ALTER TABLE curve_datasets "
                "ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''"
            )
        if "is_current" not in columns:
            connection.execute(
                "ALTER TABLE curve_datasets "
                "ADD COLUMN is_current INTEGER NOT NULL DEFAULT 1"
            )
        if "stale_reason" not in columns:
            connection.execute(
                "ALTER TABLE curve_datasets "
                "ADD COLUMN stale_reason TEXT NOT NULL DEFAULT ''"
            )
        if "dataset_group" not in columns:
            connection.execute(
                "ALTER TABLE curve_datasets "
                "ADD COLUMN dataset_group TEXT NOT NULL DEFAULT ''"
            )
        if "analysis_key" not in columns:
            connection.execute(
                "ALTER TABLE curve_datasets "
                "ADD COLUMN analysis_key TEXT NOT NULL DEFAULT ''"
            )
        if "dataset_role" not in columns:
            connection.execute(
                "ALTER TABLE curve_datasets "
                "ADD COLUMN dataset_role TEXT NOT NULL DEFAULT 'observed'"
            )
        if "parent_dataset_id" not in columns:
            connection.execute(
                "ALTER TABLE curve_datasets ADD COLUMN parent_dataset_id INTEGER"
            )
        if "validity_status" not in columns:
            connection.execute(
                "ALTER TABLE curve_datasets "
                "ADD COLUMN validity_status TEXT NOT NULL DEFAULT 'valid'"
            )
        connection.execute(
            """
            UPDATE curve_datasets
            SET dataset_group = CASE
                    WHEN dataset_group = '' THEN dataset_key
                    ELSE dataset_group
                END,
                analysis_key = CASE
                    WHEN analysis_key = '' THEN
                        CASE
                            WHEN content_hash != '' THEN content_hash
                            ELSE dataset_key
                        END
                    ELSE analysis_key
                END,
                validity_status = CASE
                    WHEN validity_status = 'valid'
                         AND is_current = 0
                         AND stale_reason != ''
                    THEN 'stale'
                    ELSE validity_status
                END
            """
        )
        if previous_version == 3:
            legacy_rows = connection.execute(
                """
                SELECT dataset_id, metadata_json, stale_reason
                FROM curve_datasets
                """
            ).fetchall()
            for legacy_row in legacy_rows:
                try:
                    legacy_metadata = json.loads(
                        legacy_row["metadata_json"] or "{}"
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    legacy_metadata = {}
                provenance = legacy_metadata.get("analysis_provenance")
                has_provenance = isinstance(provenance, dict) and bool(provenance)
                cache_key = (
                    str(provenance.get("cache_key") or "")
                    if has_provenance
                    else ""
                )
                if cache_key:
                    connection.execute(
                        """
                        UPDATE curve_datasets
                        SET analysis_key = ?, validity_status = 'valid'
                        WHERE dataset_id = ?
                        """,
                        (cache_key, int(legacy_row["dataset_id"])),
                    )
                elif not has_provenance:
                    connection.execute(
                        """
                        UPDATE curve_datasets
                        SET is_current = 0, validity_status = 'stale',
                            stale_reason = CASE
                                WHEN stale_reason = ''
                                THEN '迁移批次缺少完整分析溯源'
                                ELSE stale_reason
                            END
                        WHERE dataset_id = ?
                        """,
                        (int(legacy_row["dataset_id"]),),
                    )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_curve_datasets_group_current
            ON curve_datasets(dataset_group, is_current)
            """
        )
        if previous_version < 5:
            connection.execute(
                "DROP INDEX IF EXISTS idx_curve_datasets_analysis_key"
            )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_curve_datasets_analysis_key
            ON curve_datasets(dataset_group, analysis_key)
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
                idx_curve_datasets_analysis_content
            ON curve_datasets(dataset_group, analysis_key, content_hash)
            """
        )
        connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return database_path


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _first_finite(rows: pd.DataFrame, columns: Iterable[str]) -> float | None:
    for column in columns:
        if column not in rows:
            continue
        values = pd.to_numeric(rows[column], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    return None


def _signal_columns(rows: pd.DataFrame, curve_type: str) -> list[str]:
    preferred = (
        (
            "area",
            "raw_area",
            "photon_normalized_area",
            "normalized_area",
            "expansion_lambda",
            "reference_temperature",
        )
        if curve_type == "temperature"
        else (
            "normalized_intensity",
            "raw_area",
            "photon_normalized_intensity",
        )
    )
    result = [column for column in preferred if column in rows]
    if result:
        return result
    excluded = {
        "energy",
        "temperature",
        "mz",
        "mz_rounded",
        "photon_energy",
        "scan_energy",
        "left_bound",
        "right_bound",
    }
    return [
        str(column)
        for column in rows.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(rows[column])
    ]


def _curve_content_hash(
    curve_type: str,
    curves: Mapping[int | float, Mapping[str, Any]],
) -> str:
    digest = hashlib.sha256(curve_type.encode("utf-8"))
    for curve_key, curve in sorted(curves.items(), key=lambda item: float(item[0])):
        digest.update(f"{float(curve_key):.17g}".encode("ascii"))
        digest.update(
            f"|{float(curve.get('mz_exact_mean', curve.get('mz', curve_key))):.17g}"
            .encode("ascii")
        )
        rows = curve.get("rows")
        if isinstance(rows, pd.DataFrame):
            stable = rows.copy()
            stable.attrs = {}
            stable = stable.reindex(sorted(stable.columns), axis=1)
            digest.update(
                stable.to_json(
                    orient="split",
                    date_format="iso",
                    double_precision=15,
                    default_handler=str,
                ).encode("utf-8")
            )
        else:
            digest.update(
                _json_dumps(
                    {
                        "axis": curve.get(
                            "temperatures"
                            if curve_type == "temperature"
                            else "energies",
                            [],
                        ),
                        "values": curve.get(
                            "areas"
                            if curve_type == "temperature"
                            else "intensities",
                            [],
                        ),
                    }
                ).encode("utf-8")
            )
    return digest.hexdigest()


def store_curve_dataset(
    path: str | Path,
    *,
    curve_type: str,
    curves: Mapping[int | float, Mapping[str, Any]],
    dataset_key: str,
    name: str,
    source_label: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> int:
    """Insert or atomically replace one logical curve dataset.

    ``dataset_key`` identifies the analysis slot (for example
    ``temperature:project``). Re-running that analysis replaces its curve
    channels and points instead of accumulating indistinguishable copies.
    """
    if curve_type not in CURVE_TYPES:
        raise ValueError(f"Unsupported curve type: {curve_type}")
    if not str(dataset_key).strip():
        raise ValueError("dataset_key cannot be empty")
    initialize_curve_database(path)
    timestamp = datetime.now(timezone.utc).isoformat()
    content_hash = _curve_content_hash(curve_type, curves)

    with _connect(path) as connection:
        existing = connection.execute(
            """
            SELECT dataset_id, created_at, content_hash
            FROM curve_datasets WHERE dataset_key = ?
            """,
            (dataset_key,),
        ).fetchone()
        if existing is None:
            cursor = connection.execute(
                """
                INSERT INTO curve_datasets(
                    dataset_key, curve_type, name, source_label, content_hash,
                    is_current, stale_reason, dataset_group, analysis_key,
                    dataset_role, validity_status, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, '', ?, ?, 'observed', 'valid', ?, ?, ?)
                """,
                (
                    dataset_key,
                    curve_type,
                    name,
                    source_label,
                    content_hash,
                    dataset_key,
                    content_hash or dataset_key,
                    _json_dumps(dict(metadata or {})),
                    timestamp,
                    timestamp,
                ),
            )
            dataset_id = int(cursor.lastrowid)
        else:
            dataset_id = int(existing["dataset_id"])
            if str(existing["content_hash"] or "") == content_hash:
                connection.execute(
                    """
                    UPDATE curve_datasets
                    SET name = ?, source_label = ?, metadata_json = ?,
                        is_current = 1, stale_reason = '',
                        validity_status = 'valid', updated_at = ?
                    WHERE dataset_id = ?
                    """,
                    (
                        name,
                        source_label,
                        _json_dumps(dict(metadata or {})),
                        timestamp,
                        dataset_id,
                    ),
                )
                return dataset_id
            connection.execute(
                """
                UPDATE curve_datasets
                SET curve_type = ?, name = ?, source_label = ?,
                    content_hash = ?, is_current = 1, stale_reason = '',
                    validity_status = 'valid', metadata_json = ?, updated_at = ?
                WHERE dataset_id = ?
                """,
                (
                    curve_type,
                    name,
                    source_label,
                    content_hash,
                    _json_dumps(dict(metadata or {})),
                    timestamp,
                    dataset_id,
                ),
            )
            connection.execute(
                "DELETE FROM mass_channels WHERE dataset_id = ?",
                (dataset_id,),
            )

        for fallback_track, (curve_key, curve) in enumerate(
            sorted(curves.items(), key=lambda item: float(item[0]))
        ):
            rows = curve.get("rows")
            if not isinstance(rows, pd.DataFrame) or rows.empty:
                axis_name = "temperature" if curve_type == "temperature" else "energy"
                values_name = "areas" if curve_type == "temperature" else "intensities"
                rows = pd.DataFrame(
                    {
                        axis_name: list(curve.get(
                            "temperatures" if curve_type == "temperature" else "energies",
                            [],
                        )),
                        (
                            "area"
                            if curve_type == "temperature"
                            else "normalized_intensity"
                        ): list(curve.get(values_name, [])),
                    }
                )
            rows = rows.reset_index(drop=True)
            exact_mz = float(
                curve.get("mz_exact_mean", curve.get("mz", curve_key))
            )
            nominal_mz = int(
                round(float(curve.get("mz_rounded", exact_mz)))
            )
            peak_track_value = curve.get(
                "temperature_peak_track",
                curve.get("temperature_peak_cluster", fallback_track),
            )
            try:
                peak_track = int(peak_track_value)
            except (TypeError, ValueError):
                peak_track = fallback_track
            left_bound = _first_finite(rows, ("left_bound",))
            right_bound = _first_finite(rows, ("right_bound",))
            channel_cursor = connection.execute(
                """
                INSERT INTO mass_channels(
                    dataset_id, exact_mz, nominal_mz, peak_track,
                    left_bound, right_bound, species_label
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    exact_mz,
                    nominal_mz,
                    peak_track,
                    left_bound,
                    right_bound,
                    str(curve.get("species", "") or ""),
                ),
            )
            channel_id = int(channel_cursor.lastrowid)
            axis_column = "temperature" if curve_type == "temperature" else "energy"
            signal_columns = _signal_columns(rows, curve_type)
            point_records: list[tuple[Any, ...]] = []
            for point_order, row in rows.iterrows():
                axis_value = _finite_float(row.get(axis_column))
                if axis_value is None:
                    continue
                photon_energy = _finite_float(
                    row.get("photon_energy", row.get("scan_energy"))
                )
                signals = {
                    column: _finite_float(row.get(column))
                    for column in signal_columns
                }
                signals = {
                    key: value for key, value in signals.items() if value is not None
                }
                point_metadata = {
                    key: _json_value(row.get(key))
                    for key in (
                        "integration_method",
                        "file_count",
                        "replicate_mode",
                        "scan_folder",
                        "curve_class",
                        "curve_class_label",
                        "curve_class_reason",
                        "replicate_grouping",
                        "replicate_warning",
                        "reference_source",
                    )
                    if key in rows
                }
                point_records.append(
                    (
                        channel_id,
                        int(point_order),
                        axis_value,
                        photon_energy,
                        _json_dumps(signals),
                        _json_dumps(point_metadata),
                    )
                )
            connection.executemany(
                """
                INSERT INTO curve_points(
                    channel_id, point_order, axis_value, photon_energy,
                    signals_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                point_records,
            )
    return dataset_id


def store_curve_dataset_version(
    path: str | Path,
    *,
    curve_type: str,
    curves: Mapping[int | float, Mapping[str, Any]],
    dataset_group: str,
    analysis_key: str,
    name: str,
    source_label: str = "",
    dataset_role: str = "observed",
    parent_dataset_id: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    make_current: bool = True,
) -> int:
    """Store an immutable batch, deduplicated by input key and result content.

    ``analysis_key`` describes the input conditions. Repeating those conditions
    may still produce a different payload (for example after an algorithm
    update), so only an identical ``content_hash`` reuses the old batch.
    Different content creates a new immutable batch under the same analysis key.
    """
    if curve_type not in CURVE_TYPES:
        raise ValueError(f"Unsupported curve type: {curve_type}")
    group = str(dataset_group).strip()
    key = str(analysis_key).strip()
    if not group:
        raise ValueError("dataset_group cannot be empty")
    if not key:
        raise ValueError("analysis_key cannot be empty")
    if dataset_role not in DATASET_ROLES:
        raise ValueError(f"Unsupported dataset role: {dataset_role}")

    initialize_curve_database(path)
    content_hash = _curve_content_hash(curve_type, curves)
    with _connect(path) as connection:
        matching_content = connection.execute(
            """
            SELECT dataset_id, content_hash
            FROM curve_datasets
            WHERE dataset_group = ? AND analysis_key = ?
              AND content_hash = ?
            ORDER BY dataset_id
            LIMIT 1
            """,
            (group, key, content_hash),
        ).fetchone()
        if matching_content is not None:
            dataset_id = int(matching_content["dataset_id"])
            if make_current:
                set_current_curve_dataset(path, dataset_id)
            return dataset_id
        prior_versions = connection.execute(
            """
            SELECT dataset_id, content_hash
            FROM curve_datasets
            WHERE dataset_group = ? AND analysis_key = ?
            ORDER BY dataset_id
            """,
            (group, key),
        ).fetchall()

    run_token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    dataset_key = f"{group}:{run_token}:{uuid.uuid4().hex[:10]}"
    stored_metadata = dict(metadata or {})
    if prior_versions:
        stored_metadata["analysis_key_collision"] = {
            "detected": True,
            "prior_dataset_ids": [
                int(row["dataset_id"]) for row in prior_versions
            ],
            "prior_content_hashes": [
                str(row["content_hash"] or "") for row in prior_versions
            ],
            "new_content_hash": content_hash,
        }
    dataset_id = store_curve_dataset(
        path,
        curve_type=curve_type,
        curves=curves,
        dataset_key=dataset_key,
        name=name,
        source_label=source_label,
        metadata=stored_metadata,
    )
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        with _connect(path) as connection:
            if parent_dataset_id is not None:
                parent = connection.execute(
                    "SELECT dataset_id FROM curve_datasets WHERE dataset_id = ?",
                    (int(parent_dataset_id),),
                ).fetchone()
                if parent is None:
                    raise ValueError(
                        f"Parent curve dataset {int(parent_dataset_id)} does not exist"
                    )
            if make_current:
                connection.execute(
                    """
                    UPDATE curve_datasets
                    SET is_current = 0
                    WHERE dataset_group = ? AND dataset_id != ?
                    """,
                    (group, dataset_id),
                )
            connection.execute(
                """
                UPDATE curve_datasets
                SET dataset_group = ?, analysis_key = ?, dataset_role = ?,
                    parent_dataset_id = ?, validity_status = 'valid',
                    is_current = ?, stale_reason = '', updated_at = ?
                WHERE dataset_id = ?
                """,
                (
                    group,
                    key,
                    dataset_role,
                    parent_dataset_id,
                    int(bool(make_current)),
                    timestamp,
                    dataset_id,
                ),
            )
    except Exception:
        with _connect(path) as cleanup:
            cleanup.execute(
                "DELETE FROM curve_datasets WHERE dataset_id = ?",
                (dataset_id,),
            )
        raise
    return dataset_id


def list_curve_datasets(
    path: str | Path,
    *,
    curve_type: str | None = None,
    include_stale: bool = False,
) -> list[CurveDataset]:
    if not Path(path).is_file():
        return []
    # Existing projects may contain a database created by schema v1/v2.
    # Read paths must migrate it before referencing newer columns such as
    # ``is_current`` and ``stale_reason``.
    initialize_curve_database(path)
    query = """
        SELECT d.*,
               COUNT(DISTINCT c.channel_id) AS channel_count,
               COUNT(p.point_id) AS point_count
        FROM curve_datasets d
        LEFT JOIN mass_channels c ON c.dataset_id = d.dataset_id
        LEFT JOIN curve_points p ON p.channel_id = c.channel_id
    """
    conditions: list[str] = []
    parameters_list: list[Any] = []
    if curve_type is not None:
        conditions.append("d.curve_type = ?")
        parameters_list.append(curve_type)
    if not include_stale:
        conditions.append("d.is_current = 1")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " GROUP BY d.dataset_id ORDER BY d.updated_at DESC, d.dataset_id DESC"
    with _connect(path) as connection:
        rows = connection.execute(query, tuple(parameters_list)).fetchall()
    return [
        CurveDataset(
            dataset_id=int(row["dataset_id"]),
            dataset_key=str(row["dataset_key"]),
            curve_type=str(row["curve_type"]),
            name=str(row["name"]),
            source_label=str(row["source_label"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            channel_count=int(row["channel_count"]),
            point_count=int(row["point_count"]),
            is_current=bool(row["is_current"]),
            stale_reason=str(row["stale_reason"] or ""),
            dataset_group=str(row["dataset_group"] or row["curve_type"]),
            analysis_key=str(row["analysis_key"] or row["dataset_key"]),
            dataset_role=str(row["dataset_role"] or "observed"),
            parent_dataset_id=(
                int(row["parent_dataset_id"])
                if row["parent_dataset_id"] is not None
                else None
            ),
            validity_status=str(row["validity_status"] or "valid"),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )
        for row in rows
    ]


def list_curve_datasets_read_only(
    path: str | Path,
    *,
    curve_type: str | None = None,
    include_stale: bool = True,
) -> list[CurveDataset]:
    """Inspect dataset headers without migrating or otherwise writing the DB."""
    if not Path(path).is_file():
        return []
    with _connect_readonly(path) as connection:
        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(curve_datasets)"
            ).fetchall()
        }
        required = {"dataset_id", "dataset_key", "curve_type", "name"}
        if not required.issubset(columns):
            raise ValueError("Unsupported curve database schema")
        optional = {
            "source_label": "''",
            "created_at": "''",
            "updated_at": "''",
            "is_current": "1",
            "stale_reason": "''",
            "dataset_group": "curve_type",
            "analysis_key": "dataset_key",
            "dataset_role": "'observed'",
            "parent_dataset_id": "NULL",
            "validity_status": (
                "CASE WHEN COALESCE(stale_reason, '') != '' "
                "THEN 'stale' ELSE 'valid' END"
                if "stale_reason" in columns
                else "'valid'"
            ),
            "metadata_json": "'{}'",
        }
        selected = [
            name if name in columns else f"{fallback} AS {name}"
            for name, fallback in optional.items()
        ]
        query = (
            "SELECT dataset_id, dataset_key, curve_type, name, "
            + ", ".join(selected)
            + " FROM curve_datasets"
        )
        parameters: list[Any] = []
        conditions: list[str] = []
        if curve_type is not None:
            conditions.append("curve_type = ?")
            parameters.append(str(curve_type))
        if not include_stale and "is_current" in columns:
            conditions.append("is_current = 1")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY updated_at DESC, dataset_id DESC"
        dataset_rows = connection.execute(query, parameters).fetchall()
        results: list[CurveDataset] = []
        for row in dataset_rows:
            counts = connection.execute(
                """
                SELECT COUNT(DISTINCT c.channel_id) AS channel_count,
                       COUNT(p.point_id) AS point_count
                FROM mass_channels c
                LEFT JOIN curve_points p ON p.channel_id = c.channel_id
                WHERE c.dataset_id = ?
                """,
                (int(row["dataset_id"]),),
            ).fetchone()
            results.append(
                CurveDataset(
                    dataset_id=int(row["dataset_id"]),
                    dataset_key=str(row["dataset_key"]),
                    curve_type=str(row["curve_type"]),
                    name=str(row["name"]),
                    source_label=str(row["source_label"] or ""),
                    created_at=str(row["created_at"] or ""),
                    updated_at=str(row["updated_at"] or ""),
                    channel_count=int(counts["channel_count"]),
                    point_count=int(counts["point_count"]),
                    is_current=bool(row["is_current"]),
                    stale_reason=str(row["stale_reason"] or ""),
                    dataset_group=str(row["dataset_group"] or row["curve_type"]),
                    analysis_key=str(row["analysis_key"] or row["dataset_key"]),
                    dataset_role=str(row["dataset_role"] or "observed"),
                    parent_dataset_id=(
                        int(row["parent_dataset_id"])
                        if row["parent_dataset_id"] is not None
                        else None
                    ),
                    validity_status=str(row["validity_status"] or "valid"),
                    metadata=json.loads(row["metadata_json"] or "{}"),
                )
            )
    return results


def get_curve_dataset_state_read_only(
    path: str | Path,
    dataset_id: int,
) -> CurveDatasetState | None:
    """Read one dataset's identity and validity without scanning curve points."""
    if not Path(path).is_file():
        return None
    with _connect_readonly(path) as connection:
        columns = {
            str(row["name"])
            for row in connection.execute(
                "PRAGMA table_info(curve_datasets)"
            ).fetchall()
        }
        if not {"dataset_id", "curve_type"}.issubset(columns):
            return None
        selected = {
            "is_current": "is_current" if "is_current" in columns else "1",
            "stale_reason": (
                "stale_reason" if "stale_reason" in columns else "''"
            ),
            "validity_status": (
                "validity_status"
                if "validity_status" in columns
                else (
                    "CASE WHEN COALESCE(stale_reason, '') != '' "
                    "THEN 'stale' ELSE 'valid' END"
                    if "stale_reason" in columns
                    else "'valid'"
                )
            ),
            "source_label": (
                "source_label" if "source_label" in columns else "''"
            ),
            "created_at": "created_at" if "created_at" in columns else "''",
            "updated_at": "updated_at" if "updated_at" in columns else "''",
        }
        row = connection.execute(
            """
            SELECT dataset_id, curve_type,
                   {is_current} AS is_current,
                   {validity_status} AS validity_status,
                   {stale_reason} AS stale_reason,
                   {source_label} AS source_label,
                   {created_at} AS created_at,
                   {updated_at} AS updated_at
            FROM curve_datasets
            WHERE dataset_id = ?
            """.format(**selected),
            (int(dataset_id),),
        ).fetchone()
    if row is None:
        return None
    return CurveDatasetState(
        dataset_id=int(row["dataset_id"]),
        curve_type=str(row["curve_type"]),
        is_current=bool(row["is_current"]),
        validity_status=str(row["validity_status"] or "valid"),
        stale_reason=str(row["stale_reason"] or ""),
        source_label=str(row["source_label"] or ""),
        created_at=str(row["created_at"] or ""),
        updated_at=str(row["updated_at"] or ""),
    )


def set_current_curve_dataset(path: str | Path, dataset_id: int) -> None:
    """Set one valid batch as the current dataset for its dataset group."""
    initialize_curve_database(path)
    timestamp = datetime.now(timezone.utc).isoformat()
    with _connect(path) as connection:
        dataset = connection.execute(
            """
            SELECT dataset_group, validity_status
            FROM curve_datasets WHERE dataset_id = ?
            """,
            (int(dataset_id),),
        ).fetchone()
        if dataset is None:
            raise ValueError(f"Curve dataset {int(dataset_id)} does not exist")
        if str(dataset["validity_status"]) != "valid":
            raise ValueError("Only a valid curve dataset can become current")
        group = str(dataset["dataset_group"])
        connection.execute(
            """
            UPDATE curve_datasets
            SET is_current = 0, updated_at = ?
            WHERE dataset_group = ?
            """,
            (timestamp, group),
        )
        connection.execute(
            """
            UPDATE curve_datasets
            SET is_current = 1, stale_reason = '', updated_at = ?
            WHERE dataset_id = ?
            """,
            (timestamp, int(dataset_id)),
        )


def set_curve_dataset_validity(
    path: str | Path,
    *,
    dataset_id: int,
    validity_status: str,
    reason: str = "",
) -> None:
    if validity_status not in VALIDITY_STATUSES:
        raise ValueError(f"Unsupported validity status: {validity_status}")
    initialize_curve_database(path)
    with _connect(path) as connection:
        cursor = connection.execute(
            """
            UPDATE curve_datasets
            SET validity_status = ?, stale_reason = ?,
                is_current = CASE WHEN ? = 'valid' THEN is_current ELSE 0 END,
                updated_at = ?
            WHERE dataset_id = ?
            """,
            (
                validity_status,
                str(reason),
                validity_status,
                datetime.now(timezone.utc).isoformat(),
                int(dataset_id),
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"Curve dataset {int(dataset_id)} does not exist")


def reactivate_curve_dataset_for_analysis(
    path: str | Path,
    *,
    dataset_id: int,
    analysis_key: str,
) -> bool:
    """Restore a stale dataset only when its full recorded analysis key matches.

    This supports reversible cache invalidation: if project inputs and analysis
    parameters return to exactly the fingerprint that produced a stored batch,
    that batch can safely become valid/current again without recomputation.
    Archived datasets are never restored automatically.
    """
    expected_key = str(analysis_key or "").strip()
    if not expected_key:
        return False
    initialize_curve_database(path)
    timestamp = datetime.now(timezone.utc).isoformat()
    with _connect(path) as connection:
        row = connection.execute(
            """
            SELECT dataset_group, analysis_key, validity_status, metadata_json
            FROM curve_datasets
            WHERE dataset_id = ?
            """,
            (int(dataset_id),),
        ).fetchone()
        if row is None or str(row["validity_status"]) == "archived":
            return False
        metadata = json.loads(row["metadata_json"] or "{}")
        provenance = (
            metadata.get("analysis_provenance")
            if isinstance(metadata, dict)
            else {}
        )
        recorded_key = str(row["analysis_key"] or "")
        provenance_key = (
            str(provenance.get("cache_key") or "")
            if isinstance(provenance, dict)
            else ""
        )
        if expected_key not in {recorded_key, provenance_key}:
            return False
        group = str(row["dataset_group"])
        connection.execute(
            """
            UPDATE curve_datasets
            SET is_current = 0, updated_at = ?
            WHERE dataset_group = ?
            """,
            (timestamp, group),
        )
        connection.execute(
            """
            UPDATE curve_datasets
            SET validity_status = 'valid', is_current = 1,
                stale_reason = '', updated_at = ?
            WHERE dataset_id = ?
            """,
            (timestamp, int(dataset_id)),
        )
    return True


def list_curve_channels(
    path: str | Path,
    dataset_id: int,
) -> list[CurveChannelSummary]:
    """List channel metadata without reading any rows from ``curve_points``."""
    with _connect_readonly(path) as connection:
        dataset = connection.execute(
            "SELECT dataset_id FROM curve_datasets WHERE dataset_id = ?",
            (int(dataset_id),),
        ).fetchone()
        if dataset is None:
            raise ValueError(f"Curve dataset {int(dataset_id)} does not exist")
        rows = connection.execute(
            """
            WITH point_summary AS (
                SELECT channel_id,
                       COUNT(*) AS point_count,
                       COUNT(DISTINCT photon_energy) AS energy_count,
                       MIN(point_order) AS first_point_order
                FROM curve_points
                GROUP BY channel_id
            )
            SELECT c.channel_id, c.dataset_id, c.exact_mz, c.nominal_mz,
                   c.peak_track, c.left_bound, c.right_bound, c.species_label,
                   COALESCE(s.point_count, 0) AS point_count,
                   COALESCE(s.energy_count, 0) AS energy_count,
                   COALESCE(first_point.metadata_json, '{}')
                       AS channel_metadata_json
            FROM mass_channels c
            LEFT JOIN point_summary s ON s.channel_id = c.channel_id
            LEFT JOIN curve_points first_point
                   ON first_point.channel_id = c.channel_id
                  AND first_point.point_order = s.first_point_order
            WHERE c.dataset_id = ?
            ORDER BY c.exact_mz, c.peak_track, c.channel_id
            """,
            (int(dataset_id),),
        ).fetchall()
    return [_channel_summary_from_row(row) for row in rows]


def find_curve_channels(
    path: str | Path,
    dataset_id: int,
    exact_mz: float,
    tolerance: float,
) -> list[CurveChannelSummary]:
    """Find precise-mass channels within an explicit absolute tolerance."""
    if float(tolerance) < 0:
        raise ValueError("tolerance must be non-negative")
    with _connect_readonly(path) as connection:
        rows = connection.execute(
            """
            SELECT c.channel_id, c.dataset_id, c.exact_mz, c.nominal_mz,
                   c.peak_track, c.left_bound, c.right_bound, c.species_label,
                   (
                       SELECT COUNT(*)
                       FROM curve_points p
                       WHERE p.channel_id = c.channel_id
                   ) AS point_count,
                   (
                       SELECT COUNT(DISTINCT p.photon_energy)
                       FROM curve_points p
                       WHERE p.channel_id = c.channel_id
                         AND p.photon_energy IS NOT NULL
                   ) AS energy_count,
                   COALESCE(
                       (
                           SELECT p.metadata_json
                           FROM curve_points p
                           WHERE p.channel_id = c.channel_id
                           ORDER BY p.point_order
                           LIMIT 1
                       ),
                       '{}'
                   ) AS channel_metadata_json
            FROM mass_channels c
            WHERE c.dataset_id = ? AND ABS(c.exact_mz - ?) <= ?
            ORDER BY ABS(c.exact_mz - ?), c.channel_id
            """,
            (
                int(dataset_id),
                float(exact_mz),
                float(tolerance),
                float(exact_mz),
            ),
        ).fetchall()
    return [_channel_summary_from_row(row) for row in rows]


def _channel_summary_from_row(row: sqlite3.Row) -> CurveChannelSummary:
    try:
        metadata = json.loads(row["channel_metadata_json"] or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return CurveChannelSummary(
        channel_id=int(row["channel_id"]),
        dataset_id=int(row["dataset_id"]),
        exact_mz=float(row["exact_mz"]),
        nominal_mz=int(row["nominal_mz"]),
        peak_track=(
            int(row["peak_track"]) if row["peak_track"] is not None else None
        ),
        left_bound=(
            float(row["left_bound"]) if row["left_bound"] is not None else None
        ),
        right_bound=(
            float(row["right_bound"]) if row["right_bound"] is not None else None
        ),
        species_label=str(row["species_label"] or ""),
        point_count=int(row["point_count"]),
        energy_count=int(row["energy_count"]),
        metadata=metadata,
    )


def load_curve_channel(path: str | Path, channel_id: int) -> CurveChannel:
    """Load exactly one channel and its ordered point rows."""
    with _connect_readonly(path) as connection:
        channel = connection.execute(
            """
            SELECT c.*, d.curve_type
            FROM mass_channels c
            JOIN curve_datasets d ON d.dataset_id = c.dataset_id
            WHERE c.channel_id = ?
            """,
            (int(channel_id),),
        ).fetchone()
        if channel is None:
            raise ValueError(f"Curve channel {int(channel_id)} does not exist")
        points = connection.execute(
            """
            SELECT point_order, axis_value, photon_energy,
                   signals_json, metadata_json
            FROM curve_points
            WHERE channel_id = ?
            ORDER BY point_order
            """,
            (int(channel_id),),
        ).fetchall()

    curve_type = str(channel["curve_type"])
    axis_column = "temperature" if curve_type == "temperature" else "energy"
    records: list[dict[str, Any]] = []
    for point in points:
        record: dict[str, Any] = {
            axis_column: float(point["axis_value"]),
            "mz": float(channel["exact_mz"]),
            "mz_rounded": int(channel["nominal_mz"]),
        }
        if point["photon_energy"] is not None:
            record["photon_energy"] = float(point["photon_energy"])
        if channel["left_bound"] is not None:
            record["left_bound"] = float(channel["left_bound"])
        if channel["right_bound"] is not None:
            record["right_bound"] = float(channel["right_bound"])
        record.update(json.loads(point["signals_json"] or "{}"))
        record.update(json.loads(point["metadata_json"] or "{}"))
        records.append(record)
    return CurveChannel(
        channel_id=int(channel["channel_id"]),
        dataset_id=int(channel["dataset_id"]),
        curve_type=curve_type,
        exact_mz=float(channel["exact_mz"]),
        nominal_mz=int(channel["nominal_mz"]),
        peak_track=(
            int(channel["peak_track"])
            if channel["peak_track"] is not None
            else None
        ),
        left_bound=(
            float(channel["left_bound"])
            if channel["left_bound"] is not None
            else None
        ),
        right_bound=(
            float(channel["right_bound"])
            if channel["right_bound"] is not None
            else None
        ),
        species_label=str(channel["species_label"] or ""),
        rows=pd.DataFrame.from_records(records),
    )


def mark_curve_datasets_stale(
    path: str | Path,
    *,
    curve_type: str,
    reason: str,
) -> int:
    if curve_type not in CURVE_TYPES:
        raise ValueError(f"Unsupported curve type: {curve_type}")
    if not Path(path).is_file():
        return 0
    initialize_curve_database(path)
    with _connect(path) as connection:
        cursor = connection.execute(
            """
            UPDATE curve_datasets
            SET is_current = 0, stale_reason = ?,
                validity_status = 'stale', updated_at = ?
            WHERE curve_type = ? AND is_current = 1
            """,
            (
                str(reason),
                datetime.now(timezone.utc).isoformat(),
                curve_type,
            ),
        )
        return int(cursor.rowcount)


def invalidate_curve_datasets_for_peak_source(
    path: str | Path,
    *,
    manual_peak_file: str | Path | None,
    active_peak_set_id: str = "",
    peak_set_sha256: str = "",
) -> dict[str, int]:
    """Invalidate current datasets whose recorded peak source is not reproducible.

    This is deliberately a light-weight project-open check: the peak-range file
    is normally small, whereas scanning every raw PIE/temperature folder belongs
    to the lazy cache check performed when the corresponding page is opened.
    """
    datasets = list_curve_datasets(path, include_stale=False)
    if not datasets:
        return {}

    expected: dict[str, Any] | None = None
    configured_path = str(manual_peak_file or "").strip()
    if configured_path:
        peak_path = Path(configured_path).expanduser()
        if peak_path.is_file():
            digest = hashlib.sha256()
            with peak_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            stat = peak_path.stat()
            expected = {
                "path": str(peak_path),
                "size": stat.st_size,
                "sha256": digest.hexdigest(),
            }
        else:
            expected = {"path": str(peak_path), "missing": True}

    expected_peak_set_id = str(active_peak_set_id or "").strip()
    expected_peak_sha = str(peak_set_sha256 or "").strip().lower()
    stale_types: dict[str, str] = {}
    for dataset in datasets:
        provenance = dataset.metadata.get("analysis_provenance")
        if not isinstance(provenance, dict) or not provenance:
            stale_types[dataset.curve_type] = "曲线缺少卡峰文件溯源信息"
            continue
        recorded = provenance.get("manual_peak_file")
        recorded_peak_set_id = str(
            provenance.get("active_peak_set_id") or ""
        ).strip()
        recorded_sha = (
            str(recorded.get("sha256") or "").strip().lower()
            if isinstance(recorded, dict)
            else ""
        )
        if expected_peak_set_id:
            if (
                recorded_peak_set_id == expected_peak_set_id
                and recorded_sha
                and recorded_sha == expected_peak_sha
            ):
                continue
            stale_types[dataset.curve_type] = "项目卡峰集版本已变化"
            continue
        if expected is None:
            if recorded or provenance.get("peak_source") == "manual_peak_file":
                stale_types[dataset.curve_type] = "项目卡峰来源已变化"
            continue
        if (
            not isinstance(recorded, dict)
            or recorded.get("missing")
            or expected.get("missing")
            or str(recorded.get("path") or "") != str(expected.get("path") or "")
            or int(recorded.get("size") or -1) != int(expected.get("size") or -2)
            or str(recorded.get("sha256") or "") != str(expected.get("sha256") or "")
        ):
            stale_types[dataset.curve_type] = "项目卡峰文件或其内容已变化"

    return {
        curve_type: mark_curve_datasets_stale(
            path,
            curve_type=curve_type,
            reason=reason,
        )
        for curve_type, reason in stale_types.items()
    }


def backfill_curve_peak_set_provenance(
    path: str | Path,
    *,
    peak_set_id: str,
    peak_set_sha256: str,
    peak_set_origin: str = "",
) -> int:
    """Attach a migrated peak-set identity to matching legacy curve metadata."""
    database_path = Path(path)
    if not database_path.is_file():
        return 0
    expected_sha = str(peak_set_sha256 or "").strip().lower()
    expected_id = str(peak_set_id or "").strip()
    if not expected_id or not expected_sha:
        return 0

    changed = 0
    with _connect(database_path) as connection:
        rows = connection.execute(
            "SELECT dataset_id, metadata_json FROM curve_datasets"
        ).fetchall()
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            provenance = metadata.get("analysis_provenance")
            if not isinstance(provenance, dict):
                continue
            recorded = provenance.get("manual_peak_file")
            if not isinstance(recorded, dict):
                continue
            recorded_sha = str(recorded.get("sha256") or "").strip().lower()
            if recorded_sha != expected_sha:
                continue
            if str(provenance.get("active_peak_set_id") or "") == expected_id:
                continue
            provenance["active_peak_set_id"] = expected_id
            if peak_set_origin:
                provenance["peak_set_origin"] = str(peak_set_origin)
            metadata["analysis_provenance"] = provenance
            connection.execute(
                """
                UPDATE curve_datasets
                SET metadata_json = ?, updated_at = ?
                WHERE dataset_id = ?
                """,
                (
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                    int(row["dataset_id"]),
                ),
            )
            changed += 1
    return changed


def _load_curve_dataset_from_connection(
    connection: sqlite3.Connection,
    dataset_id: int,
) -> pd.DataFrame:
    dataset = connection.execute(
        "SELECT curve_type FROM curve_datasets WHERE dataset_id = ?",
        (int(dataset_id),),
    ).fetchone()
    if dataset is None:
        raise ValueError(f"Curve dataset {dataset_id} does not exist")
    rows = connection.execute(
        """
        SELECT c.channel_id, c.exact_mz, c.nominal_mz, c.peak_track,
               c.left_bound, c.right_bound, c.species_label,
               p.point_order, p.axis_value, p.photon_energy,
               p.signals_json, p.metadata_json
        FROM mass_channels c
        JOIN curve_points p ON p.channel_id = c.channel_id
        WHERE c.dataset_id = ?
        ORDER BY c.exact_mz, p.point_order
        """,
        (int(dataset_id),),
    ).fetchall()
    curve_type = str(dataset["curve_type"])
    axis_column = "temperature" if curve_type == "temperature" else "energy"
    records: list[dict[str, Any]] = []
    for row in rows:
        record = {
            "channel_id": int(row["channel_id"]),
            "mz": float(row["exact_mz"]),
            "mz_rounded": int(row["nominal_mz"]),
            "peak_track": row["peak_track"],
            "left_bound": row["left_bound"],
            "right_bound": row["right_bound"],
            "species": str(row["species_label"] or ""),
            axis_column: float(row["axis_value"]),
        }
        if row["photon_energy"] is not None:
            record["photon_energy"] = float(row["photon_energy"])
        record.update(json.loads(row["signals_json"] or "{}"))
        record.update(json.loads(row["metadata_json"] or "{}"))
        records.append(record)
    return pd.DataFrame.from_records(records)


def load_curve_dataset(path: str | Path, dataset_id: int) -> pd.DataFrame:
    """Load one dataset as a post-processing friendly wide DataFrame."""
    initialize_curve_database(path)
    with _connect(path) as connection:
        return _load_curve_dataset_from_connection(connection, dataset_id)


def load_curve_dataset_read_only(path: str | Path, dataset_id: int) -> pd.DataFrame:
    """Load one current-schema dataset without migrating or writing the database."""
    with _connect_readonly(path) as connection:
        return _load_curve_dataset_from_connection(connection, dataset_id)


def replace_species_assignments(
    path: str | Path,
    *,
    dataset_id: int,
    exact_mz: float,
    assignments: Iterable[Mapping[str, Any]],
    assignment_status: str,
    r_squared: float | None = None,
    source: str = "pie_fit",
    tolerance: float = 1e-6,
) -> int:
    if assignment_status not in {"candidate", "confirmed", "rejected"}:
        raise ValueError(f"Unsupported assignment status: {assignment_status}")
    timestamp = datetime.now(timezone.utc).isoformat()
    with _connect(path) as connection:
        channel = connection.execute(
            """
            SELECT channel_id, ABS(exact_mz - ?) AS distance
            FROM mass_channels
            WHERE dataset_id = ? AND ABS(exact_mz - ?) <= ?
            ORDER BY distance
            LIMIT 1
            """,
            (float(exact_mz), int(dataset_id), float(exact_mz), float(tolerance)),
        ).fetchone()
        if channel is None:
            raise ValueError(
                f"No stored curve channel matches exact m/z {float(exact_mz):.12g}"
            )
        channel_id = int(channel["channel_id"])
        connection.execute(
            "DELETE FROM species_assignments WHERE channel_id = ? AND source = ?",
            (channel_id, source),
        )
        count = 0
        for assignment in assignments:
            ids = assignment.get("ids")
            if ids is None:
                ids = [assignment.get("id")] if assignment.get("id") is not None else []
            connection.execute(
                """
                INSERT INTO species_assignments(
                    channel_id, assignment_status, species_name, formula,
                    species_database_ids, ionization_energy, coefficient,
                    contribution_percent, r_squared, source, metadata_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    channel_id,
                    assignment_status,
                    str(assignment.get("species") or assignment.get("name") or ""),
                    str(assignment.get("formula") or ""),
                    _json_dumps(list(ids)),
                    _finite_float(
                        assignment.get(
                            "ionization_energy",
                            assignment.get("ie"),
                        )
                    ),
                    _finite_float(assignment.get("coefficient")),
                    _finite_float(assignment.get("contribution_percent")),
                    _finite_float(r_squared),
                    source,
                    _json_dumps({}),
                    timestamp,
                ),
            )
            count += 1
    return count


def list_species_assignments(
    path: str | Path,
    *,
    dataset_id: int,
    exact_mz: float | None = None,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    query = """
        SELECT a.assignment_id, a.assignment_status, a.species_name, a.formula,
               a.species_database_ids, a.ionization_energy, a.coefficient,
               a.contribution_percent, a.r_squared, a.source, a.created_at,
               c.channel_id, c.exact_mz, c.nominal_mz
        FROM species_assignments a
        JOIN mass_channels c ON c.channel_id = a.channel_id
        WHERE c.dataset_id = ?
    """
    parameters: list[Any] = [int(dataset_id)]
    if exact_mz is not None:
        query += " AND ABS(c.exact_mz - ?) <= ?"
        parameters.extend([float(exact_mz), float(tolerance)])
    query += " ORDER BY c.exact_mz, a.assignment_id"
    with _connect(path) as connection:
        rows = connection.execute(query, parameters).fetchall()
    records = [dict(row) for row in rows]
    for record in records:
        record["species_database_ids"] = json.loads(
            record.get("species_database_ids") or "[]"
        )
    return pd.DataFrame.from_records(records)


def save_pie_fit_state(
    path: str | Path,
    *,
    dataset_id: int,
    channel_id: int,
    config: Mapping[str, Any],
    result: Mapping[str, Any],
    config_hash: str = "",
    fit_status: str = "draft",
) -> None:
    """Persist PIE fitting state under the immutable dataset/channel identity."""
    initialize_curve_database(path)
    timestamp = datetime.now(timezone.utc).isoformat()
    with _connect(path) as connection:
        owner = connection.execute(
            """
            SELECT 1
            FROM mass_channels
            WHERE channel_id = ? AND dataset_id = ?
            """,
            (int(channel_id), int(dataset_id)),
        ).fetchone()
        if owner is None:
            raise ValueError("The channel does not belong to the selected dataset")
        connection.execute(
            """
            INSERT INTO pie_fit_states(
                dataset_id, channel_id, config_json, result_json,
                config_hash, fit_status, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id, channel_id) DO UPDATE SET
                config_json = excluded.config_json,
                result_json = excluded.result_json,
                config_hash = excluded.config_hash,
                fit_status = excluded.fit_status,
                updated_at = excluded.updated_at
            """,
            (
                int(dataset_id),
                int(channel_id),
                _json_dumps(dict(config)),
                _json_dumps(dict(result)),
                str(config_hash),
                str(fit_status),
                timestamp,
            ),
        )


def load_pie_fit_state(
    path: str | Path,
    *,
    dataset_id: int,
    channel_id: int,
) -> dict[str, Any] | None:
    initialize_curve_database(path)
    with _connect(path) as connection:
        row = connection.execute(
            """
            SELECT config_json, result_json, config_hash, fit_status, updated_at
            FROM pie_fit_states
            WHERE dataset_id = ? AND channel_id = ?
            """,
            (int(dataset_id), int(channel_id)),
        ).fetchone()
    if row is None:
        return None
    return {
        "config": json.loads(row["config_json"] or "{}"),
        "result": json.loads(row["result_json"] or "{}"),
        "config_hash": str(row["config_hash"] or ""),
        "fit_status": str(row["fit_status"] or ""),
        "updated_at": str(row["updated_at"] or ""),
    }


def list_pie_fit_states(
    path: str | Path,
    *,
    dataset_id: int,
) -> list[dict[str, Any]]:
    initialize_curve_database(path)
    with _connect(path) as connection:
        rows = connection.execute(
            """
            SELECT f.channel_id, c.exact_mz, f.config_json, f.result_json,
                   f.config_hash, f.fit_status, f.updated_at
            FROM pie_fit_states f
            JOIN mass_channels c ON c.channel_id = f.channel_id
            WHERE f.dataset_id = ?
            ORDER BY c.exact_mz, f.channel_id
            """,
            (int(dataset_id),),
        ).fetchall()
    return [
        {
            "channel_id": int(row["channel_id"]),
            "exact_mz": float(row["exact_mz"]),
            "config": json.loads(row["config_json"] or "{}"),
            "result": json.loads(row["result_json"] or "{}"),
            "config_hash": str(row["config_hash"] or ""),
            "fit_status": str(row["fit_status"] or ""),
            "updated_at": str(row["updated_at"] or ""),
        }
        for row in rows
    ]
