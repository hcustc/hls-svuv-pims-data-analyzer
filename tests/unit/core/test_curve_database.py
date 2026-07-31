import sqlite3

import numpy as np
import pandas as pd
import pytest

from bl03u_masstool.core import curve_database as curve_database_module
from bl03u_masstool.core.curve_database import (
    backfill_curve_peak_set_provenance,
    find_curve_channels,
    get_curve_dataset_state_read_only,
    invalidate_curve_datasets_for_peak_source,
    list_curve_channels,
    list_curve_datasets,
    list_curve_datasets_read_only,
    list_pie_fit_states,
    list_species_assignments,
    load_curve_channel,
    load_curve_dataset,
    mark_curve_datasets_stale,
    project_curve_database_path,
    reactivate_curve_dataset_for_analysis,
    replace_species_assignments,
    save_pie_fit_state,
    set_curve_dataset_validity,
    set_current_curve_dataset,
    store_curve_dataset,
    store_curve_dataset_version,
)
from bl03u_masstool.core.curve_repository import (
    RepositoryCurveMapping,
    SQLiteCurveRepository,
    validate_repository_round_trip,
)
from bl03u_masstool.core.project_settings import ProjectSettings


def _pie_curves():
    return {
        227.587157: {
            "mz": 227.587157,
            "mz_rounded": 228,
            "rows": pd.DataFrame(
                {
                    "energy": [7.0, 8.0],
                    "normalized_intensity": [0.0, 0.2],
                    "raw_area": [0.0, 2.0],
                    "source_spectra": [
                        ["/segment-low/7eV.txt"],
                        ["/segment-low/8eV.txt", "/segment-high/8eV.txt"],
                    ],
                    "left_bound": [24076, 24076],
                    "right_bound": [24086, 24086],
                }
            ),
        },
        228.023117: {
            "mz": 228.023117,
            "mz_rounded": 228,
            "rows": pd.DataFrame(
                {
                    "energy": [7.0, 8.0],
                    "normalized_intensity": [1.0, 10.0],
                    "raw_area": [10.0, 100.0],
                    "left_bound": [24091, 24091],
                    "right_bound": [24118, 24118],
                }
            ),
        },
    }


def test_curve_database_keeps_precise_peaks_separate(tmp_path):
    path = tmp_path / "curves.sqlite"
    dataset_id = store_curve_dataset(
        path,
        curve_type="pie",
        curves=_pie_curves(),
        dataset_key="pie:project",
        name="PIE",
    )

    datasets = list_curve_datasets(path, curve_type="pie")
    assert [(item.dataset_id, item.channel_count, item.point_count) for item in datasets] == [
        (dataset_id, 2, 4)
    ]
    loaded = load_curve_dataset(path, dataset_id)
    assert sorted(loaded["mz"].unique()) == pytest.approx([227.587157, 228.023117])
    assert loaded.groupby("mz").size().tolist() == [2, 2]
    assert loaded["mz_rounded"].unique().tolist() == [228]
    merged_source_row = loaded[
        np.isclose(loaded["mz"], 227.587157)
        & np.isclose(loaded["energy"], 8.0)
    ].iloc[0]
    assert merged_source_row["source_spectra"] == [
        "/segment-low/8eV.txt",
        "/segment-high/8eV.txt",
    ]

    with sqlite3.connect(path) as connection:
        indexes = {
            row[1]
            for row in connection.execute("PRAGMA index_list(mass_channels)")
        }
    assert "idx_mass_channels_exact_mz" in indexes


def test_reloading_unchanged_dataset_preserves_species_assignments(tmp_path):
    path = tmp_path / "curves.sqlite"
    curves = _pie_curves()
    dataset_id = store_curve_dataset(
        path,
        curve_type="pie",
        curves=curves,
        dataset_key="pie:project",
        name="PIE",
    )
    replace_species_assignments(
        path,
        dataset_id=dataset_id,
        exact_mz=228.023117,
        assignments=[
            {
                "species": "Chrysene",
                "formula": "C18H12",
                "coefficient": 0.8,
                "contribution_percent": 75.0,
            }
        ],
        assignment_status="confirmed",
        r_squared=0.99,
    )

    same_id = store_curve_dataset(
        path,
        curve_type="pie",
        curves=curves,
        dataset_key="pie:project",
        name="PIE refreshed",
    )

    assert same_id == dataset_id
    assignments = list_species_assignments(
        path,
        dataset_id=dataset_id,
        exact_mz=228.023117,
    )
    assert assignments.loc[0, "formula"] == "C18H12"
    assert assignments.loc[0, "assignment_status"] == "confirmed"


def test_changed_curve_replaces_old_channels_and_assignments(tmp_path):
    path = tmp_path / "curves.sqlite"
    curves = _pie_curves()
    dataset_id = store_curve_dataset(
        path,
        curve_type="pie",
        curves=curves,
        dataset_key="pie:project",
        name="PIE",
    )
    replace_species_assignments(
        path,
        dataset_id=dataset_id,
        exact_mz=228.023117,
        assignments=[{"species": "candidate"}],
        assignment_status="candidate",
    )
    curves[228.023117]["rows"].loc[1, "normalized_intensity"] = 11.0

    store_curve_dataset(
        path,
        curve_type="pie",
        curves=curves,
        dataset_key="pie:project",
        name="PIE",
    )

    assert list_species_assignments(path, dataset_id=dataset_id).empty
    loaded = load_curve_dataset(path, dataset_id)
    assert loaded.loc[
        np.isclose(loaded["mz"], 228.023117)
        & (loaded["energy"] == 8.0),
        "normalized_intensity",
    ].iloc[0] == pytest.approx(11.0)


def test_project_curve_database_uses_project_analysis_directory(tmp_path):
    settings = ProjectSettings(output_dir=str(tmp_path / "project"))
    assert project_curve_database_path(settings) == (
        tmp_path / "project" / "analysis" / "curve_data.sqlite"
    )


def test_project_curve_database_resolves_configured_relative_path_from_project(
    tmp_path,
):
    settings = ProjectSettings(
        output_dir=str(tmp_path / "project"),
        curve_database_path="analysis/custom_curves.sqlite",
    )
    assert project_curve_database_path(settings) == (
        tmp_path / "project" / "analysis" / "custom_curves.sqlite"
    )


def test_stale_dataset_can_only_reactivate_for_exact_analysis_key(tmp_path):
    path = tmp_path / "curves.sqlite"
    dataset_id = store_curve_dataset_version(
        path,
        curve_type="pie",
        curves=_pie_curves(),
        dataset_group="pie:project",
        analysis_key="analysis-key-1",
        name="PIE",
    )
    mark_curve_datasets_stale(
        path,
        curve_type="pie",
        reason="analysis inputs changed",
    )

    assert not reactivate_curve_dataset_for_analysis(
        path,
        dataset_id=dataset_id,
        analysis_key="different-key",
    )
    stale = get_curve_dataset_state_read_only(path, dataset_id)
    assert stale is not None
    assert stale.validity_status == "stale"

    assert reactivate_curve_dataset_for_analysis(
        path,
        dataset_id=dataset_id,
        analysis_key="analysis-key-1",
    )
    restored = get_curve_dataset_state_read_only(path, dataset_id)
    assert restored is not None
    assert restored.validity_status == "valid"
    assert restored.is_current


def test_stale_curve_dataset_is_hidden_until_recomputed(tmp_path):
    path = tmp_path / "curves.sqlite"
    curves = _pie_curves()
    dataset_id = store_curve_dataset(
        path,
        curve_type="pie",
        curves=curves,
        dataset_key="pie:project",
        name="PIE",
    )

    assert mark_curve_datasets_stale(
        path,
        curve_type="pie",
        reason="manual peak file changed",
    ) == 1
    assert list_curve_datasets(path, curve_type="pie") == []
    stale = list_curve_datasets(path, curve_type="pie", include_stale=True)
    assert stale[0].dataset_id == dataset_id
    assert stale[0].is_current is False
    assert stale[0].stale_reason == "manual peak file changed"

    same_id = store_curve_dataset(
        path,
        curve_type="pie",
        curves=curves,
        dataset_key="pie:project",
        name="PIE recomputed",
    )
    assert same_id == dataset_id
    current = list_curve_datasets(path, curve_type="pie")
    assert current[0].is_current is True
    assert current[0].stale_reason == ""


def test_peak_source_validation_invalidates_untraceable_and_changed_curves(tmp_path):
    path = tmp_path / "curves.sqlite"
    peak_file = tmp_path / "peak_ranges.csv"
    peak_file.write_text("mz,left,right\n228,10,20\n", encoding="utf-8")
    curves = _pie_curves()

    store_curve_dataset(
        path,
        curve_type="pie",
        curves=curves,
        dataset_key="pie:project",
        name="legacy PIE",
    )
    assert invalidate_curve_datasets_for_peak_source(
        path,
        manual_peak_file=peak_file,
    ) == {"pie": 1}

    import hashlib

    digest = hashlib.sha256(peak_file.read_bytes()).hexdigest()
    store_curve_dataset(
        path,
        curve_type="pie",
        curves=curves,
        dataset_key="pie:project",
        name="traceable PIE",
        metadata={
            "analysis_provenance": {
                "peak_source": "manual_peak_file",
                "manual_peak_file": {
                    "path": str(peak_file),
                    "size": peak_file.stat().st_size,
                    "sha256": digest,
                },
            }
        },
    )
    assert invalidate_curve_datasets_for_peak_source(
        path,
        manual_peak_file=peak_file,
    ) == {}

    peak_file.write_text("mz,left,right\n228,11,21\n", encoding="utf-8")
    assert invalidate_curve_datasets_for_peak_source(
        path,
        manual_peak_file=peak_file,
    ) == {"pie": 1}


def test_listing_migrates_legacy_database_before_querying_current_state(tmp_path):
    path = tmp_path / "legacy_curves.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE curve_datasets (
                dataset_id INTEGER PRIMARY KEY,
                dataset_key TEXT NOT NULL UNIQUE,
                curve_type TEXT NOT NULL,
                name TEXT NOT NULL,
                source_label TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE mass_channels (
                channel_id INTEGER PRIMARY KEY,
                dataset_id INTEGER NOT NULL,
                exact_mz REAL NOT NULL,
                nominal_mz INTEGER NOT NULL,
                peak_track INTEGER,
                left_bound REAL,
                right_bound REAL,
                species_label TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE curve_points (
                point_id INTEGER PRIMARY KEY,
                channel_id INTEGER NOT NULL,
                point_order INTEGER NOT NULL,
                axis_value REAL NOT NULL,
                photon_energy REAL,
                signals_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            INSERT INTO curve_datasets(
                dataset_key, curve_type, name, source_label, content_hash,
                metadata_json, created_at, updated_at
            ) VALUES (
                'pie:project', 'pie', 'Legacy PIE', '', '', '{}',
                '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z'
            );
            """
        )

    read_only_datasets = list_curve_datasets_read_only(path, curve_type="pie")
    assert len(read_only_datasets) == 1
    assert read_only_datasets[0].dataset_group == "pie:project"
    with sqlite3.connect(path) as connection:
        pre_migration_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(curve_datasets)")
        }
    assert "dataset_group" not in pre_migration_columns

    datasets = list_curve_datasets(path, curve_type="pie")
    assert len(datasets) == 1
    assert datasets[0].name == "Legacy PIE"
    assert datasets[0].is_current is True
    assert datasets[0].stale_reason == ""
    with sqlite3.connect(path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(curve_datasets)")
        }
        version = connection.execute("PRAGMA user_version").fetchone()[0]
    assert {"is_current", "stale_reason"}.issubset(columns)
    assert version == 5


def test_lazy_channel_queries_keep_precise_identity_and_point_order(tmp_path):
    path = tmp_path / "curves.sqlite"
    dataset_id = store_curve_dataset_version(
        path,
        curve_type="pie",
        curves=_pie_curves(),
        dataset_group="pie:project",
        analysis_key="analysis-a",
        name="PIE batch A",
    )

    summaries = list_curve_channels(path, dataset_id)
    assert [summary.point_count for summary in summaries] == [2, 2]
    matches = find_curve_channels(path, dataset_id, 228.0231, 0.0001)
    assert [match.exact_mz for match in matches] == pytest.approx([228.023117])

    channel = load_curve_channel(path, matches[0].channel_id)
    assert channel.dataset_id == dataset_id
    assert channel.rows["energy"].tolist() == [7.0, 8.0]
    assert channel.rows["normalized_intensity"].tolist() == [1.0, 10.0]


def test_channel_summary_exposes_lightweight_curve_metadata(tmp_path):
    path = tmp_path / "temperature-curves.sqlite"
    rows = pd.DataFrame(
        {
            "temperature": [650.0, 750.0, 650.0, 750.0],
            "scan_energy": [8.0, 8.0, 9.0, 9.0],
            "mz": [70.01] * 4,
            "mz_rounded": [70] * 4,
            "area": [1.0, 2.0, 1.5, 2.5],
            "curve_class": ["formation"] * 4,
            "curve_class_label": ["生成(升高)"] * 4,
            "curve_class_reason": ["测试"] * 4,
        }
    )
    dataset_id = store_curve_dataset_version(
        path,
        curve_type="temperature",
        curves={
            70.01: {
                "mz": 70.01,
                "mz_rounded": 70,
                "temperatures": rows["temperature"].tolist(),
                "areas": rows["area"].tolist(),
                "rows": rows,
            }
        },
        dataset_group="temperature:project",
        analysis_key="temperature-summary",
        name="Temperature summary",
    )

    summary = list_curve_channels(path, dataset_id)[0]

    assert summary.point_count == 4
    assert summary.energy_count == 2
    assert summary.metadata["curve_class"] == "formation"
    assert summary.metadata["curve_class_label"] == "生成(升高)"


def test_immutable_analysis_batches_do_not_replace_previous_fit_chain(tmp_path):
    path = tmp_path / "curves.sqlite"
    curves_a = _pie_curves()
    dataset_a = store_curve_dataset_version(
        path,
        curve_type="pie",
        curves=curves_a,
        dataset_group="pie:project",
        analysis_key="analysis-a",
        name="PIE batch A",
    )
    curves_b = _pie_curves()
    curves_b[228.023117]["rows"].loc[1, "normalized_intensity"] = 11.0
    dataset_b = store_curve_dataset_version(
        path,
        curve_type="pie",
        curves=curves_b,
        dataset_group="pie:project",
        analysis_key="analysis-a",
        name="PIE batch B",
    )

    assert dataset_b != dataset_a
    datasets = list_curve_datasets(path, curve_type="pie", include_stale=True)
    assert {dataset.dataset_id for dataset in datasets} == {dataset_a, dataset_b}
    assert next(item for item in datasets if item.dataset_id == dataset_b).is_current
    assert not next(item for item in datasets if item.dataset_id == dataset_a).is_current
    collision = next(
        item for item in datasets if item.dataset_id == dataset_b
    ).metadata["analysis_key_collision"]
    assert collision["prior_dataset_ids"] == [dataset_a]
    assert load_curve_dataset(path, dataset_a).loc[
        lambda frame: np.isclose(frame["mz"], 228.023117)
        & np.isclose(frame["energy"], 8.0),
        "normalized_intensity",
    ].iloc[0] == pytest.approx(10.0)

    set_current_curve_dataset(path, dataset_a)
    current = list_curve_datasets(path, curve_type="pie")
    assert [item.dataset_id for item in current] == [dataset_a]


def test_repository_round_trip_and_lru_mapping(tmp_path):
    path = tmp_path / "curves.sqlite"
    curves = _pie_curves()
    dataset_id = store_curve_dataset_version(
        path,
        curve_type="pie",
        curves=curves,
        dataset_group="pie:project",
        analysis_key="analysis-a",
        name="PIE batch A",
    )
    repository = SQLiteCurveRepository(path, dataset_id, cache_size=1)
    report = validate_repository_round_trip(curves, repository)
    mapping = RepositoryCurveMapping(repository)

    assert report.channel_count == 2
    assert report.point_count == 4
    assert mapping.max_point_count == 2
    assert mapping.metadata(228.023117)["channel_id"] > 0
    assert mapping[228.023117]["intensities"] == [1.0, 10.0]


def test_read_only_listing_does_not_migrate_schema(tmp_path):
    path = tmp_path / "legacy-readonly.sqlite"
    store_curve_dataset(
        path,
        curve_type="pie",
        curves=_pie_curves(),
        dataset_key="pie:project",
        name="PIE",
    )
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 3")

    datasets = list_curve_datasets_read_only(path, curve_type="pie")

    assert len(datasets) == 1
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3


def test_dataset_state_read_returns_validity_without_loading_points(tmp_path):
    path = tmp_path / "state.sqlite"
    dataset_id = store_curve_dataset_version(
        path,
        curve_type="pie",
        curves=_pie_curves(),
        dataset_group="pie:project",
        analysis_key="state-check",
        name="State check",
    )

    state = get_curve_dataset_state_read_only(path, dataset_id)

    assert state is not None
    assert state.dataset_id == dataset_id
    assert state.curve_type == "pie"
    assert state.validity_status == "valid"
    assert state.is_current is True


def test_v3_migration_backs_up_and_marks_untraceable_batch_stale(tmp_path):
    path = tmp_path / "v3.sqlite"
    store_curve_dataset(
        path,
        curve_type="pie",
        curves=_pie_curves(),
        dataset_key="pie:project",
        name="PIE",
    )
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 3")
        connection.execute(
            "UPDATE curve_datasets SET metadata_json = '{}'"
        )

    datasets = list_curve_datasets(path, curve_type="pie", include_stale=True)

    assert datasets[0].validity_status == "stale"
    assert datasets[0].is_current is False
    assert "缺少完整分析溯源" in datasets[0].stale_reason
    backups = list(tmp_path.glob("v3.sqlite.v3-backup-*.sqlite"))
    assert len(backups) == 1
    list_curve_datasets(path, curve_type="pie", include_stale=True)
    assert list(tmp_path.glob("v3.sqlite.v3-backup-*.sqlite")) == backups


def test_v4_migration_replaces_analysis_only_unique_constraint(tmp_path):
    path = tmp_path / "v4.sqlite"
    curves_a = _pie_curves()
    dataset_a = store_curve_dataset_version(
        path,
        curve_type="pie",
        curves=curves_a,
        dataset_group="pie:project",
        analysis_key="analysis-a",
        name="A",
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "DROP INDEX IF EXISTS idx_curve_datasets_analysis_content"
        )
        connection.execute(
            "DROP INDEX IF EXISTS idx_curve_datasets_analysis_key"
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX idx_curve_datasets_analysis_key
            ON curve_datasets(dataset_group, analysis_key)
            """
        )
        connection.execute("PRAGMA user_version = 4")

    curves_b = _pie_curves()
    curves_b[228.023117]["rows"].loc[1, "normalized_intensity"] = 11.0
    dataset_b = store_curve_dataset_version(
        path,
        curve_type="pie",
        curves=curves_b,
        dataset_group="pie:project",
        analysis_key="analysis-a",
        name="B",
    )

    assert dataset_b != dataset_a
    assert len(list(tmp_path.glob("v4.sqlite.v4-backup-*.sqlite"))) == 1
    with sqlite3.connect(path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        indexes = {
            row[1]: bool(row[2])
            for row in connection.execute(
                "PRAGMA index_list(curve_datasets)"
            )
        }
    assert version == 5
    assert indexes["idx_curve_datasets_analysis_key"] is False
    assert indexes["idx_curve_datasets_analysis_content"] is True


def test_pie_fit_state_is_isolated_by_dataset_and_channel(tmp_path):
    path = tmp_path / "fit-state.sqlite"
    dataset_a = store_curve_dataset_version(
        path,
        curve_type="pie",
        curves=_pie_curves(),
        dataset_group="pie:project",
        analysis_key="analysis-a",
        name="A",
    )
    curves_b = _pie_curves()
    curves_b[228.023117]["rows"].loc[1, "normalized_intensity"] = 11.0
    dataset_b = store_curve_dataset_version(
        path,
        curve_type="pie",
        curves=curves_b,
        dataset_group="pie:project",
        analysis_key="analysis-b",
        name="B",
    )
    channel_a = find_curve_channels(path, dataset_a, 228.023117, 1e-9)[0]
    channel_b = find_curve_channels(path, dataset_b, 228.023117, 1e-9)[0]
    save_pie_fit_state(
        path,
        dataset_id=dataset_a,
        channel_id=channel_a.channel_id,
        config={"locked_ids": [1]},
        result={"r_squared": 0.91},
    )
    save_pie_fit_state(
        path,
        dataset_id=dataset_b,
        channel_id=channel_b.channel_id,
        config={"locked_ids": [2]},
        result={"r_squared": 0.82},
    )

    states_a = list_pie_fit_states(path, dataset_id=dataset_a)
    states_b = list_pie_fit_states(path, dataset_id=dataset_b)
    assert states_a[0]["config"]["locked_ids"] == [1]
    assert states_b[0]["config"]["locked_ids"] == [2]
    assert states_a[0]["result"]["r_squared"] == pytest.approx(0.91)
    assert states_b[0]["result"]["r_squared"] == pytest.approx(0.82)


def test_failed_version_metadata_step_leaves_no_partial_dataset(tmp_path):
    path = tmp_path / "no-partial.sqlite"
    with pytest.raises(ValueError, match="Parent curve dataset"):
        store_curve_dataset_version(
            path,
            curve_type="pie",
            curves=_pie_curves(),
            dataset_group="pie:project",
            analysis_key="analysis-a",
            name="A",
            parent_dataset_id=999,
        )
    assert list_curve_datasets(path, curve_type="pie", include_stale=True) == []


def test_stale_and_archive_keep_batch_but_remove_it_from_current(tmp_path):
    path = tmp_path / "validity.sqlite"
    dataset_id = store_curve_dataset_version(
        path,
        curve_type="pie",
        curves=_pie_curves(),
        dataset_group="pie:project",
        analysis_key="analysis-a",
        name="A",
    )
    set_curve_dataset_validity(
        path,
        dataset_id=dataset_id,
        validity_status="stale",
        reason="test",
    )
    assert list_curve_datasets(path, curve_type="pie") == []
    stale = list_curve_datasets(path, curve_type="pie", include_stale=True)
    assert stale[0].validity_status == "stale"
    assert stale[0].channel_count == 2

    set_curve_dataset_validity(
        path,
        dataset_id=dataset_id,
        validity_status="archived",
        reason="archived",
    )
    archived = list_curve_datasets(path, curve_type="pie", include_stale=True)
    assert archived[0].validity_status == "archived"
    assert archived[0].channel_count == 2


def test_database_lock_failure_does_not_create_partial_batch(tmp_path, monkeypatch):
    path = tmp_path / "locked.sqlite"
    original_id = store_curve_dataset_version(
        path,
        curve_type="pie",
        curves=_pie_curves(),
        dataset_group="pie:project",
        analysis_key="analysis-a",
        name="A",
    )
    lock = sqlite3.connect(path)
    lock.execute("PRAGMA journal_mode = WAL")
    lock.execute("BEGIN IMMEDIATE")

    def fast_timeout_connect(database_path):
        connection = sqlite3.connect(database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 50")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    monkeypatch.setattr(
        curve_database_module,
        "_connect",
        fast_timeout_connect,
    )
    curves_b = _pie_curves()
    curves_b[228.023117]["rows"].loc[1, "normalized_intensity"] = 11.0
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            store_curve_dataset_version(
                path,
                curve_type="pie",
                curves=curves_b,
                dataset_group="pie:project",
                analysis_key="analysis-b",
                name="B",
            )
    finally:
        lock.rollback()
        lock.close()

    datasets = list_curve_datasets(path, curve_type="pie", include_stale=True)
    assert [dataset.dataset_id for dataset in datasets] == [original_id]


def test_recomputed_stale_batch_can_be_validated_then_reactivated(tmp_path):
    path = tmp_path / "reactivate.sqlite"
    curves = _pie_curves()
    dataset_id = store_curve_dataset_version(
        path,
        curve_type="pie",
        curves=curves,
        dataset_group="pie:project",
        analysis_key="analysis-a",
        name="A",
    )
    set_curve_dataset_validity(
        path,
        dataset_id=dataset_id,
        validity_status="stale",
        reason="peak source changed",
    )

    same_id = store_curve_dataset_version(
        path,
        curve_type="pie",
        curves=curves,
        dataset_group="pie:project",
        analysis_key="analysis-a",
        name="A recomputed",
        make_current=False,
    )
    repository = SQLiteCurveRepository(path, same_id)
    validate_repository_round_trip(curves, repository)
    set_curve_dataset_validity(
        path,
        dataset_id=same_id,
        validity_status="valid",
    )
    set_current_curve_dataset(path, same_id)

    current = list_curve_datasets(path, curve_type="pie")
    assert [dataset.dataset_id for dataset in current] == [dataset_id]
    assert current[0].validity_status == "valid"
    assert current[0].stale_reason == ""


def test_matching_legacy_curve_provenance_is_backfilled_with_peak_set_id(tmp_path):
    path = tmp_path / "backfill.sqlite"
    dataset_id = store_curve_dataset_version(
        path,
        curve_type="pie",
        curves=_pie_curves(),
        dataset_group="pie:project",
        analysis_key="legacy-analysis",
        name="legacy",
        metadata={
            "analysis_provenance": {
                "peak_source": "manual_peak_file",
                "manual_peak_file": {
                    "path": "/old/project/peaks.csv",
                    "sha256": "abc123",
                    "size": 10,
                },
            }
        },
    )

    assert backfill_curve_peak_set_provenance(
        path,
        peak_set_id="peak-new",
        peak_set_sha256="abc123",
        peak_set_origin="imported",
    ) == 1
    dataset = next(
        item
        for item in list_curve_datasets(path, curve_type="pie")
        if item.dataset_id == dataset_id
    )
    provenance = dataset.metadata["analysis_provenance"]
    assert provenance["active_peak_set_id"] == "peak-new"
    assert provenance["peak_set_origin"] == "imported"
