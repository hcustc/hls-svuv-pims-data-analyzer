from __future__ import annotations

from pathlib import Path

import pytest

from bl03u_masstool.core.peak_sets import (
    activate_peak_set,
    create_peak_set,
    import_peak_set,
    list_peak_sets,
    migrate_legacy_peak_file,
    resolve_active_peak_file,
    verify_peak_set,
)


def test_peak_sets_are_versioned_and_never_overwritten(tmp_path):
    project = tmp_path / "project"
    first_content = b"label,mz,left_bound,right_bound\nA,10,8,12\n"
    second_content = b"label,mz,left_bound,right_bound\nA,10,7,13\n"

    first = create_peak_set(
        project,
        content=first_content,
        extension=".csv",
        label="first approved",
        origin="spectrum_workbench",
    )
    second = create_peak_set(
        project,
        content=second_content,
        extension=".csv",
        label="second approved",
        origin="spectrum_workbench",
    )

    assert first.peak_set_id != second.peak_set_id
    assert first.peak_file != second.peak_file
    assert not Path(first.peak_file).is_absolute()
    assert verify_peak_set(project, first).read_bytes() == first_content
    assert verify_peak_set(project, second).read_bytes() == second_content
    assert [item.peak_set_id for item in list_peak_sets(project)] == [
        second.peak_set_id,
        first.peak_set_id,
    ]


def test_activation_refuses_peak_set_modified_outside_application(tmp_path):
    project = tmp_path / "project"
    record = create_peak_set(
        project,
        content=b"label,mz\nA,10\n",
        extension=".csv",
        label="approved",
        origin="imported",
    )
    verify_peak_set(project, record).write_bytes(b"label,mz\nB,11\n")

    with pytest.raises(ValueError, match="外部修改"):
        activate_peak_set(project, record.peak_set_id)


def test_import_peak_set_copies_source_as_approved_version(tmp_path):
    project = tmp_path / "project"
    source = tmp_path / "legacy.csv"
    source.write_text("label,mz\nA,10\n", encoding="utf-8")

    record = import_peak_set(project, source)
    source.write_text("label,mz\nB,11\n", encoding="utf-8")

    assert verify_peak_set(project, record).read_text(encoding="utf-8") == "label,mz\nA,10\n"
    active, active_path = activate_peak_set(project, record.peak_set_id)
    assert active == record
    assert active_path.read_text(encoding="utf-8") == "label,mz\nA,10\n"


def test_active_peak_set_id_is_authoritative_over_configured_path(tmp_path):
    project = tmp_path / "project"
    record = create_peak_set(
        project,
        content=b"label,mz\nA,10\n",
        extension=".csv",
        label="approved",
        origin="spectrum_workbench",
    )
    other = tmp_path / "candidate.csv"
    other.write_text("label,mz\nB,11\n", encoding="utf-8")

    assert resolve_active_peak_file(
        project,
        active_peak_set_id=record.peak_set_id,
        configured_peak_file=record.peak_file,
    ) == verify_peak_set(project, record)
    with pytest.raises(ValueError, match="不一致"):
        resolve_active_peak_file(
            project,
            active_peak_set_id=record.peak_set_id,
            configured_peak_file=other,
        )


def test_unversioned_project_peak_path_is_not_returned_directly(tmp_path):
    legacy = tmp_path / "legacy.csv"
    legacy.write_text("label,mz\nA,10\n", encoding="utf-8")

    assert resolve_active_peak_file(
        tmp_path / "project",
        active_peak_set_id="",
        configured_peak_file=legacy,
    ) is None


def test_legacy_peak_file_is_copied_once_and_reused_by_digest(tmp_path):
    project = tmp_path / "project"
    legacy = tmp_path / "legacy.csv"
    legacy.write_text("label,mz\nA,10\n", encoding="utf-8")

    first = migrate_legacy_peak_file(project, legacy)
    second = migrate_legacy_peak_file(project, legacy)

    assert first is not None
    assert second is not None
    assert first.peak_set_id == second.peak_set_id
    assert len(list_peak_sets(project)) == 1
    assert verify_peak_set(project, first).parent == (
        project / "analysis" / "spectrum" / "manual_peaks" / "peak_sets"
    ).resolve()
