"""In-memory background job and temporary library state for the API."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

from fastapi import HTTPException


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
    manifest: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PicsLibrary:
    id: str
    database_path: str
    source_file: str
    created_at: float = field(default_factory=time.time)
    parsed_species: int = 0
    inserted_points: int = 0


JOBS_LOCK = threading.Lock()
JOBS: Dict[str, PieJob] = {}

PICS_LIBRARIES_LOCK = threading.Lock()
PICS_LIBRARIES: Dict[str, PicsLibrary] = {}
PICS_LIBRARY_TTL_SECONDS = int(float(os.environ.get("BL03U_PICS_LIBRARY_TTL_HOURS", "24")) * 3600)
DEFAULT_JOB_TTL_HOURS = 6
DEFAULT_MAX_JOBS = 100


def get_pics_library(library_id: Optional[str]) -> Optional[PicsLibrary]:
    if not library_id:
        return None
    cleanup_expired_pics_libraries()
    with PICS_LIBRARIES_LOCK:
        return PICS_LIBRARIES.get(library_id)


def cleanup_expired_pics_libraries() -> None:
    if PICS_LIBRARY_TTL_SECONDS <= 0:
        return
    now = time.time()
    expired: List[PicsLibrary] = []
    with PICS_LIBRARIES_LOCK:
        for library_id, library in list(PICS_LIBRARIES.items()):
            if now - library.created_at > PICS_LIBRARY_TTL_SECONDS:
                expired.append(library)
        for library in expired:
            library_id = library.id
            PICS_LIBRARIES.pop(library_id, None)
    for library in expired:
        try:
            Path(library.database_path).unlink(missing_ok=True)
        except Exception:
            pass


def get_job(job_id: str) -> PieJob:
    cleanup_jobs()
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


def update_job(job_id: str, **values) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        for key, value in values.items():
            setattr(job, key, value)
        job.updated_at = time.time()


def job_ttl_seconds() -> int:
    return int(float(os.environ.get("BL03U_PIE_JOB_TTL_HOURS", str(DEFAULT_JOB_TTL_HOURS))) * 3600)


def max_jobs() -> int:
    return max(1, int(os.environ.get("BL03U_MAX_PIE_JOBS", str(DEFAULT_MAX_JOBS))))


def cleanup_jobs() -> None:
    ttl = job_ttl_seconds()
    limit = max_jobs()
    now = time.time()
    with JOBS_LOCK:
        if ttl > 0:
            for job_id, job in list(JOBS.items()):
                if job.status in {"done", "error"} and now - job.updated_at > ttl:
                    JOBS.pop(job_id, None)
        overflow = len(JOBS) - limit
        if overflow <= 0:
            return
        removable = sorted(
            (
                (job.updated_at, job_id)
                for job_id, job in JOBS.items()
                if job.status in {"done", "error"}
            )
        )
        for _, job_id in removable[:overflow]:
            JOBS.pop(job_id, None)


__all__ = [
    "DEFAULT_JOB_TTL_HOURS",
    "DEFAULT_MAX_JOBS",
    "JOBS",
    "JOBS_LOCK",
    "PICS_LIBRARIES",
    "PICS_LIBRARIES_LOCK",
    "PICS_LIBRARY_TTL_SECONDS",
    "PicsLibrary",
    "PieJob",
    "cleanup_expired_pics_libraries",
    "cleanup_jobs",
    "get_job",
    "get_pics_library",
    "job_ttl_seconds",
    "max_jobs",
    "update_job",
]
