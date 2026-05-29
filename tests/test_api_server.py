from __future__ import annotations

from pathlib import Path
import sqlite3
import time

import numpy as np
from fastapi.testclient import TestClient

from api import server
from core.spectrum_io import Spectrum


PICS_UPLOAD = (
    b"mz,name,energy_ev,cross_section,ionization_energy\n"
    b"18,Water,11.0,0.1,12.6\n"
    b"18,Water,12.0,0.2,12.6\n"
)


def _client() -> TestClient:
    return TestClient(server.app)


def _write_test_pics_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(server.SCHEMA_SQL)
        conn.execute(
            "INSERT INTO species (id, mz, name, ionization_energy) VALUES (?, ?, ?, ?)",
            (1, 18, "Water", 12.6),
        )
        conn.executemany(
            "INSERT INTO pic_cross_sections (species_id, energy_ev, cross_section) VALUES (?, ?, ?)",
            [(1, 11.0, 0.1), (1, 12.0, 0.2)],
        )
        conn.commit()
    return path


def test_api_rejects_database_path_outside_allowed_roots(tmp_path, monkeypatch):
    monkeypatch.delenv("BL03U_ALLOWED_DATA_ROOTS", raising=False)
    outside_database = tmp_path / "outside.sqlite"

    response = _client().get("/api/pics/search", params={"database": str(outside_database)})

    assert response.status_code == 400
    assert "allowed" in response.json()["detail"].lower() or "允许" in response.json()["detail"]


def test_api_allows_configured_data_root_for_database(tmp_path, monkeypatch):
    monkeypatch.setenv("BL03U_ALLOWED_DATA_ROOTS", str(tmp_path))
    database = _write_test_pics_db(tmp_path / "species.sqlite")

    response = _client().get("/api/pics/search", params={"database": str(database), "mz": 18})

    assert response.status_code == 200
    rows = response.json()["rows"]
    assert len(rows) == 1
    assert rows[0]["name"] == "Water"


def test_sum_folder_rejects_outside_allowed_root(tmp_path, monkeypatch):
    monkeypatch.delenv("BL03U_ALLOWED_DATA_ROOTS", raising=False)

    response = _client().get("/spectrum/sum", params={"folder": str(tmp_path)})

    assert response.status_code == 400
    assert "允许" in response.json()["detail"]


def test_spectrum_upload_temp_file_is_removed(monkeypatch):
    paths: list[Path] = []

    def fake_read_bl03u_txt(path):
        temp_path = Path(path)
        assert temp_path.exists()
        paths.append(temp_path)
        return Spectrum(
            x=np.array([1.0, 2.0]),
            y=np.array([3.0, 4.0]),
            metadata_lines=[],
            path=str(temp_path),
        )

    monkeypatch.setattr(server, "read_bl03u_txt", fake_read_bl03u_txt)

    response = _client().post("/spectrum/read", params={"filename": "spectrum.txt"}, content=b"1\n2\n")

    assert response.status_code == 200
    assert paths
    assert not paths[0].exists()


def test_pics_server_upload_requires_and_accepts_admin_token(tmp_path, monkeypatch):
    monkeypatch.delenv("BL03U_ADMIN_TOKEN", raising=False)
    client = _client()

    rejected = client.post(
        "/api/pics/upload",
        params={"scope": "server", "filename": "pics.csv"},
        content=PICS_UPLOAD,
    )
    assert rejected.status_code == 403

    database = tmp_path / "maintained.sqlite"
    monkeypatch.setenv("BL03U_ADMIN_TOKEN", "secret")
    monkeypatch.setattr(server, "_default_database_path", lambda: database)

    accepted = client.post(
        "/api/pics/upload",
        params={"scope": "server", "filename": "pics.csv"},
        headers={"x-admin-token": "secret"},
        content=PICS_UPLOAD,
    )

    assert accepted.status_code == 200
    assert accepted.json()["scope"] == "server"
    assert database.exists()


def test_pics_session_upload_returns_searchable_library_id():
    response = _client().post(
        "/api/pics/upload",
        params={"scope": "session", "filename": "pics.csv"},
        content=PICS_UPLOAD,
    )

    assert response.status_code == 200
    library_id = response.json()["library_id"]
    try:
        search = _client().get("/api/pics/search", params={"library_id": library_id, "mz": 18})
        assert search.status_code == 200
        assert search.json()["rows"][0]["name"] == "Water"
    finally:
        with server.PICS_LIBRARIES_LOCK:
            library = server.PICS_LIBRARIES.pop(library_id, None)
        if library is not None:
            Path(library.database_path).unlink(missing_ok=True)


def test_upload_pie_curve_uses_allowed_database_root(tmp_path, monkeypatch):
    monkeypatch.setenv("BL03U_ALLOWED_DATA_ROOTS", str(tmp_path))
    database = _write_test_pics_db(tmp_path / "species.sqlite")
    curve = b"mz,energy,intensity\n18,11.0,1.0\n18,12.0,2.0\n"

    response = _client().post(
        "/api/pie/upload_curve",
        params={"database": str(database), "filename": "curve.csv"},
        content=curve,
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    curve_response = _client().get(f"/api/pie/curve/{job_id}/18")
    assert curve_response.status_code == 200
    assert curve_response.json()["curve"]["energies"] == [11.0, 12.0]


def test_cleanup_jobs_removes_stale_completed_jobs(monkeypatch):
    monkeypatch.setenv("BL03U_PIE_JOB_TTL_HOURS", "0.001")
    old_job = server.PieJob(id="stale-test-job", status="done")
    old_job.updated_at = time.time() - 10
    active_job = server.PieJob(id="active-test-job", status="running")
    active_job.updated_at = time.time() - 10

    try:
        with server.JOBS_LOCK:
            server.JOBS[old_job.id] = old_job
            server.JOBS[active_job.id] = active_job
        server._cleanup_jobs()
        with server.JOBS_LOCK:
            assert old_job.id not in server.JOBS
            assert active_job.id in server.JOBS
    finally:
        with server.JOBS_LOCK:
            server.JOBS.pop(old_job.id, None)
            server.JOBS.pop(active_job.id, None)
