from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bl03u_masstool.core.peak_detection import Peak
from bl03u_masstool.core.peak_sets import verify_peak_set
from bl03u_masstool.core.project_peak_generation import generate_project_peak_set
from bl03u_masstool.core.project_settings import ProjectSettings


def _write_spectrum(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [f"header {index}" for index in range(10)]
    values = ["1.0" for _index in range(4020)]
    path.write_text("\n".join([*header, *values]), encoding="utf-8")


def test_generate_project_peak_set_persists_detected_ranges(
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    source = project / "raw_data" / "sum"
    _write_spectrum(source / "scan.txt")
    settings = ProjectSettings(
        output_dir=str(project),
        sum_spectrum_folder=str(source),
        detection_min_idx=0,
    )
    monkeypatch.setattr(
        "bl03u_masstool.core.project_peak_generation.detect_peaks_by_algorithm",
        lambda *_args, **_kwargs: [
            Peak(
                index=2,
                time=4003.0,
                mz=28.031,
                intensity=10.0,
                fwhm=4.0,
                left_bound=0,
                right_bound=4,
            )
        ],
    )

    record = generate_project_peak_set(settings)

    approved = verify_peak_set(project, record)
    assert record.origin == "auto_generated"
    assert not Path(record.peak_file).is_absolute()
    assert record.metadata["peak_count"] == 1
    assert "Unknown,4003,28.031,4001,4005" in approved.read_text(
        encoding="utf-8-sig"
    )
    manifest = yaml.safe_load(
        approved.with_suffix(".manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["source"]["mode"] == "sum"
    assert manifest["source"]["fingerprint"]["file_count"] == 1
    assert manifest["created_from"]["tool"] == "project_analysis_preflight"


def test_generate_project_peak_set_requires_registered_sum_folder(tmp_path):
    settings = ProjectSettings(
        output_dir=str(tmp_path / "project"),
        sum_spectrum_folder=str(tmp_path / "missing"),
    )

    with pytest.raises(FileNotFoundError, match="累计谱目录不存在"):
        generate_project_peak_set(settings)
