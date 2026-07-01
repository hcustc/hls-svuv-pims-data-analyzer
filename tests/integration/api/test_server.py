from __future__ import annotations

import json
import time
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from bl03u_masstool.api import app as server
from bl03u_masstool.core.config import PeakDetectionConfig
from bl03u_masstool.core.spectrum_io import Spectrum


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


def test_web_root_redirects_to_static_index():
    response = _client().get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/static/index.html"

    static_response = _client().get("/static/index.html")
    assert static_response.status_code == 200
    assert "HSL SVUV-PIMS Data Analyzer" in static_response.text


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


def test_detect_peaks_endpoint_uses_calibration(monkeypatch):
    class FakePeak:
        def to_dict(self):
            return {"mz": 18.0}

    def fake_read_bl03u_txt(path):
        return Spectrum(
            x=np.array([100.0, 101.0]),
            y=np.array([3.0, 8.0]),
            metadata_lines=[],
            path=str(path),
        )

    def fake_detect_peaks_in_range(y, *, calibration, detection_min_idx, time_offset):
        assert calibration.a == 1.0
        assert calibration.b == 2.0
        assert calibration.c == 3.0
        assert detection_min_idx == 0
        assert time_offset == 100.0
        return [FakePeak()]

    monkeypatch.setattr(server, "read_bl03u_txt", fake_read_bl03u_txt)
    monkeypatch.setattr(server, "detect_peaks_in_range", fake_detect_peaks_in_range)

    response = _client().post(
        "/spectrum/peaks",
        params={"filename": "spectrum.txt", "a": 1.0, "b": 2.0, "c": 3.0},
        content=b"1\n2\n",
    )

    assert response.status_code == 200
    assert response.json()["peaks"] == [{"mz": 18.0}]


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
            # Windows may hold a brief lock on the SQLite file after the
            # search request above. Give it a moment and retry once.
            db_path = Path(library.database_path)
            for attempt in (1, 2):
                try:
                    db_path.unlink(missing_ok=True)
                except PermissionError:
                    if attempt == 1:
                        time.sleep(0.5)
                    else:
                        raise


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

    artifacts_response = _client().get(f"/api/pie/artifacts/{job_id}")
    assert artifacts_response.status_code == 200
    artifacts = artifacts_response.json()
    assert artifacts["manifest"]["analysis_type"] == "pie"
    assert artifacts["summary"]["curve_count"] == 1
    assert artifacts["evidence"]["18"]["confidence_level"] == "unfitted"
    assert artifacts["manifest"]["input_path"] == "curve.csv"
    assert artifacts["manifest"]["parameters"]["database_scope"] == "custom"
    assert artifacts["manifest"]["parameters"]["database_name"] == "species.sqlite"
    assert "gaussian" not in artifacts["manifest"]["parameters"]
    artifacts_text = json.dumps(artifacts, ensure_ascii=False)
    assert str(database) not in artifacts_text
    assert str(tmp_path) not in artifacts_text

    fit_response = _client().post(f"/api/pie/fit/{job_id}/18")
    assert fit_response.status_code == 200
    fitted_artifacts = _client().get(f"/api/pie/artifacts/{job_id}").json()
    assert fitted_artifacts["evidence"]["18"]["candidate_count"] == 1
    assert fitted_artifacts["evidence"]["18"]["fit"] is not None


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


def test_pie_start_job_passes_peak_detection_config(tmp_path, monkeypatch):
    captured: dict[str, object] = {}
    database = _write_test_pics_db(tmp_path / "species.sqlite")

    peak_config = PeakDetectionConfig(
        algorithm="ensemble",
        detection_min_idx=17,
        threshold_end=1.5,
        min_intensity=2.5,
        prominence_ratio=0.012,
        smoothing_window=9,
        baseline_window=401,
        min_peak_width=2,
        max_peak_width=44,
        vote_threshold=0.75,
        min_intensity_for_single_vote=8.0,
        mz_tolerance=0.33,
    )

    def fake_analyze_pie_folder(folder, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(server, "_database_path_from_request", lambda **kwargs: database)
    monkeypatch.setattr(server, "_resolve_path", lambda value: Path(value))
    monkeypatch.setattr(server, "load_peak_detection_config", lambda: peak_config)
    monkeypatch.setattr(
        server,
        "load_normalization_settings",
        lambda: SimpleNamespace(pie_photon_mode="first", light_source="io", mass_discrimination=1.0),
    )
    monkeypatch.setattr(server, "analyze_pie_folder", fake_analyze_pie_folder)
    monkeypatch.setattr(server, "build_pie_curves", lambda df: {})
    monkeypatch.setattr(server, "_pie_summary", lambda df, curves: {})
    monkeypatch.setattr(server, "_pie_artifacts", lambda **kwargs: ({}, {}))

    job_id = "peak-config-test"
    payload = server.PieStartPayload(folder=str(tmp_path))
    try:
        with server.JOBS_LOCK:
            server.JOBS[job_id] = server.PieJob(id=job_id)
        server._run_pie_job(job_id, payload)
        job = server._get_job(job_id)
        assert job.status == "done"
    finally:
        with server.JOBS_LOCK:
            server.JOBS.pop(job_id, None)

    assert captured["algorithm"] == "ensemble"
    assert captured["detection_min_idx"] == 17
    assert captured["prominence_ratio"] == 0.012
    assert captured["smoothing_window"] == 9
    assert captured["baseline_window"] == 401
    assert captured["min_peak_width"] == 2
    assert captured["max_peak_width"] == 44
    assert captured["vote_threshold"] == 0.75
    assert captured["min_intensity_for_single_vote"] == 8.0
    assert captured["mz_tolerance"] == 0.33
