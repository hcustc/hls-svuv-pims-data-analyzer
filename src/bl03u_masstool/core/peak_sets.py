from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


REGISTRY_VERSION = 1


@dataclass(frozen=True)
class PeakSet:
    peak_set_id: str
    label: str
    peak_file: str
    manifest_file: str
    sha256: str
    size: int
    created_at: str
    origin: str
    metadata: dict[str, Any]


def peak_sets_directory(project_dir: str | Path) -> Path:
    return Path(project_dir) / "analysis" / "spectrum" / "manual_peaks" / "peak_sets"


def peak_set_registry_path(project_dir: str | Path) -> Path:
    return peak_sets_directory(project_dir) / "registry.json"


def _project_relative_path(project_dir: str | Path, path: str | Path) -> str:
    project_path = Path(project_dir).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_path / candidate
    resolved = candidate.resolve()
    try:
        return resolved.relative_to(project_path).as_posix()
    except ValueError:
        return str(resolved)


def resolve_peak_set_path(project_dir: str | Path, path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(project_dir).expanduser() / candidate
    return candidate.resolve()


def _read_registry(project_dir: str | Path) -> dict[str, Any]:
    path = peak_set_registry_path(project_dir)
    if not path.is_file():
        return {"version": REGISTRY_VERSION, "peak_sets": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"卡峰集注册表无法读取：{path}") from exc
    if int(data.get("version", -1)) != REGISTRY_VERSION:
        raise ValueError(f"不支持的卡峰集注册表版本：{data.get('version')}")
    if not isinstance(data.get("peak_sets"), list):
        raise ValueError("卡峰集注册表缺少 peak_sets 列表")
    return data


def _write_registry(project_dir: str | Path, data: Mapping[str, Any]) -> Path:
    path = peak_set_registry_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(data), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def list_peak_sets(project_dir: str | Path) -> list[PeakSet]:
    records: list[PeakSet] = []
    for item in _read_registry(project_dir)["peak_sets"]:
        if not isinstance(item, dict):
            continue
        try:
            records.append(
                PeakSet(
                    peak_set_id=str(item["peak_set_id"]),
                    label=str(item.get("label") or item["peak_set_id"]),
                    peak_file=str(item["peak_file"]),
                    manifest_file=str(item.get("manifest_file") or ""),
                    sha256=str(item["sha256"]),
                    size=int(item["size"]),
                    created_at=str(item["created_at"]),
                    origin=str(item.get("origin") or "unknown"),
                    metadata=dict(item.get("metadata") or {}),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(records, key=lambda record: record.created_at, reverse=True)


def get_peak_set(project_dir: str | Path, peak_set_id: str) -> PeakSet | None:
    target = str(peak_set_id or "").strip()
    return next(
        (record for record in list_peak_sets(project_dir) if record.peak_set_id == target),
        None,
    )


def verify_peak_set(project_dir: str | Path, peak_set: PeakSet) -> Path:
    path = resolve_peak_set_path(project_dir, peak_set.peak_file)
    approved_directory = peak_sets_directory(project_dir).resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(approved_directory):
        raise ValueError(f"卡峰集文件超出项目批准目录：{path}")
    if not path.is_file():
        raise FileNotFoundError(f"卡峰集文件不存在：{path}")
    content = path.read_bytes()
    if len(content) != peak_set.size or hashlib.sha256(content).hexdigest() != peak_set.sha256:
        raise ValueError(f"卡峰集内容已被外部修改：{peak_set.label}")
    return resolved_path


def create_peak_set(
    project_dir: str | Path,
    *,
    content: bytes,
    extension: str,
    label: str,
    origin: str,
    manifest: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PeakSet:
    """Create and register a new immutable project peak-set version."""
    if not content:
        raise ValueError("不能创建空卡峰集")
    suffix = extension if str(extension).startswith(".") else f".{extension}"
    suffix = suffix.lower()
    if suffix not in {".csv", ".yaml", ".yml", ".xlsx", ".xls"}:
        raise ValueError(f"不支持的卡峰文件格式：{suffix}")

    project_path = Path(project_dir)
    directory = peak_sets_directory(project_path)
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()
    created_at = datetime.now(timezone.utc).isoformat()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    peak_set_id = f"peak-{timestamp}-{digest[:8]}"
    peak_path = directory / f"{peak_set_id}{suffix}"
    manifest_path = peak_path.with_suffix(".manifest.yaml")
    if peak_path.exists():
        raise FileExistsError(f"卡峰集版本已存在：{peak_path}")

    peak_path.write_bytes(content)
    manifest_data = dict(manifest or {})
    manifest_data.update(
        {
            "version": max(2, int(manifest_data.get("version", 1))),
            "peak_set_id": peak_set_id,
            "peak_file": peak_path.name,
            "label": str(label or peak_set_id),
            "origin": str(origin or "unknown"),
            "approved": True,
            "created_at": created_at,
            "sha256": digest,
            "size": len(content),
        }
    )
    manifest_path.write_text(
        yaml.safe_dump(manifest_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    record = PeakSet(
        peak_set_id=peak_set_id,
        label=str(label or peak_set_id),
        peak_file=_project_relative_path(project_path, peak_path),
        manifest_file=_project_relative_path(project_path, manifest_path),
        sha256=digest,
        size=len(content),
        created_at=created_at,
        origin=str(origin or "unknown"),
        metadata=dict(metadata or {}),
    )
    registry = _read_registry(project_path)
    registry["peak_sets"].append(asdict(record))
    try:
        _write_registry(project_path, registry)
    except Exception:
        peak_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return record


def import_peak_set(
    project_dir: str | Path,
    source: str | Path,
    *,
    label: str | None = None,
) -> PeakSet:
    source_path = Path(source).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"卡峰文件不存在：{source_path}")
    source_manifest_path = source_path.with_suffix(".manifest.yaml")
    manifest: dict[str, Any] = {}
    if source_manifest_path.is_file():
        loaded = yaml.safe_load(source_manifest_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            manifest = loaded
    manifest["imported_from"] = str(source_path)
    peak_count: int | None = None
    try:
        if source_path.suffix.lower() == ".csv":
            peak_count = int(len(pd.read_csv(source_path)))
        elif source_path.suffix.lower() in {".xlsx", ".xls"}:
            peak_count = int(len(pd.read_excel(source_path)))
        elif source_path.suffix.lower() in {".yaml", ".yml"}:
            loaded = yaml.safe_load(source_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                peak_count = len(loaded)
            elif isinstance(loaded, dict):
                rows = loaded.get("peaks") or loaded.get("peak_ranges")
                if isinstance(rows, list):
                    peak_count = len(rows)
    except Exception:
        peak_count = None
    metadata: dict[str, Any] = {"imported_from": str(source_path)}
    if peak_count is not None:
        metadata["peak_count"] = peak_count
    return create_peak_set(
        project_dir,
        content=source_path.read_bytes(),
        extension=source_path.suffix,
        label=label or source_path.stem,
        origin="imported",
        manifest=manifest,
        metadata=metadata,
    )


def find_peak_set_by_sha256(
    project_dir: str | Path,
    sha256: str,
) -> PeakSet | None:
    digest = str(sha256 or "").strip().lower()
    if not digest:
        return None
    for record in list_peak_sets(project_dir):
        if record.sha256.lower() != digest:
            continue
        try:
            verify_peak_set(project_dir, record)
        except (FileNotFoundError, ValueError):
            continue
        return record
    return None


def migrate_legacy_peak_file(
    project_dir: str | Path,
    configured_peak_file: str | Path | None,
    *,
    label: str | None = None,
) -> PeakSet | None:
    """Copy a pre-versioning project peak file into the immutable registry.

    The legacy path is never returned directly. Identical content already
    registered in the project is reused so opening the same old project does
    not create a new version on every launch.
    """
    configured = str(configured_peak_file or "").strip()
    if not configured:
        return None
    source = resolve_peak_set_path(project_dir, configured)
    if not source.is_file():
        raise FileNotFoundError(f"旧项目卡峰文件不存在：{source}")
    content = source.read_bytes()
    existing = find_peak_set_by_sha256(
        project_dir,
        hashlib.sha256(content).hexdigest(),
    )
    if existing is not None:
        return existing
    return import_peak_set(
        project_dir,
        source,
        label=label or f"旧项目迁移 · {source.stem}",
    )


def activate_peak_set(
    project_dir: str | Path,
    peak_set_id: str,
) -> tuple[PeakSet, Path]:
    record = get_peak_set(project_dir, peak_set_id)
    if record is None:
        raise ValueError(f"项目中不存在卡峰集：{peak_set_id}")
    return record, verify_peak_set(project_dir, record)


def resolve_active_peak_file(
    project_dir: str | Path,
    *,
    active_peak_set_id: str,
    configured_peak_file: str | Path | None,
) -> Path | None:
    """Resolve the only peak file that project analyses are allowed to consume."""
    active_id = str(active_peak_set_id or "").strip()
    configured = str(configured_peak_file or "").strip()
    if not active_id:
        return None
    _record, approved_path = activate_peak_set(project_dir, active_id)
    if configured:
        configured_path = resolve_peak_set_path(project_dir, configured)
        if configured_path.resolve() != approved_path.resolve():
            raise ValueError(
                "项目配置中的卡峰文件与当前批准卡峰集不一致，请在项目管理中重新激活卡峰集。"
            )
    return approved_path


def normalize_peak_set_registry_paths(project_dir: str | Path) -> bool:
    """Rewrite project-local absolute registry paths as portable relatives."""
    registry = _read_registry(project_dir)
    changed = False
    for item in registry["peak_sets"]:
        if not isinstance(item, dict):
            continue
        for field_name in ("peak_file", "manifest_file"):
            raw = str(item.get(field_name) or "").strip()
            if not raw:
                continue
            normalized = _project_relative_path(project_dir, raw)
            if normalized != raw:
                item[field_name] = normalized
                changed = True
    if changed:
        _write_registry(project_dir, registry)
    return changed
