from __future__ import annotations

import yaml
import pytest

from bl03u_masstool.core.project_settings import (
    ProjectSettings,
    ProjectSettingsManager,
    load_project_settings,
    save_project_settings,
)


def test_project_settings_default_parent_mz_is_unset():
    assert ProjectSettings().mf_parent_mz == 0


def test_project_settings_round_trip_preserves_nested_yaml_fields(tmp_path):
    path = tmp_path / "project.yaml"
    settings = ProjectSettings(
        project_name="Flame Study",
        system="C6F11O2H",
        description="PIE UI defaults",
        output_dir="out",
        single_spectrum_file="data/single.txt",
        sum_spectrum_folder="data/sum",
        temperature_scan_folder="data/temp",
        pie_scan_folder="data/pie",
        pie_scan_folders=["data/pie-low", "data/pie-high"],
        pics_database_path="database/species_database.sqlite",
        manual_peak_file="config/manual.yaml",
        temperature_scan_result_file="output/temperature.xlsx",
        pie_identification_result_file="output/pie_identification.xlsx",
        mole_fraction_result_file="output/mole_fraction.xlsx",
        cal_a=1.2e-7,
        cal_b=2.3e-4,
        cal_c=0.45,
        calibration_points=[{"tof": 100.0, "mz": 28.0}],
        light_source="beam_current",
        temperature_photon_normalize=False,
        temperature_kr_correct=True,
        pie_photon_mode="each",
        kr_calibration_folder="data/kr",
        kr_calibration_peak_file="config/peak.yaml",
        expansion_factors={14.6: {550.0: 1.0, 600.0: 1.1}, 14.7: {550.0: 1.0, 600.0: 1.2}},
        selected_elements=["C", "H", "O"],
        peak_algorithm="cwt",
        detection_min_idx=123,
        threshold_end=4.5,
        min_intensity=6.7,
        pie_energy_decimals=2,
        pie_recursive=False,
        pie_prefer_gaussian=False,
        pie_integration_method="baseline",
        pie_multi_folder_mode=True,
        pie_merge_method="mean",
        temp_reference_mode="kr",
        temp_prefer_gaussian=False,
        temp_integration_method="baseline",
        temp_kr_mz=86,
        temp_curve_class_change_threshold=0.33,
        temp_curve_class_peak_fraction=0.72,
        pics_no_mz=31,
        pics_no_formula="15NO",
        pics_no_mf=0.02,
        pics_new_species_mf=0.003,
        mf_md_preset="30 Torr (Catalysis)",
        mf_mass_disc_exponent=0.7,
        mf_parent_mz=130,
        mf_parent_initial_mf=0.004,
        mf_reference_temperature=575.0,
        mf_reference_species_mz=44,
        mf_reference_species_tm=590.0,
        mf_reference_species_mf_at_tm=0.005,
        mf_photon_energy=11.2,
        mf_kr_data={550.0: 12.3},
    )

    save_project_settings(settings, path)
    loaded = load_project_settings(path)

    assert loaded.project_name == settings.project_name
    assert loaded.system == settings.system
    assert loaded.description == settings.description
    assert loaded.output_dir == settings.output_dir
    assert loaded.single_spectrum_file == settings.single_spectrum_file
    assert loaded.sum_spectrum_folder == settings.sum_spectrum_folder
    assert loaded.temperature_scan_folder == settings.temperature_scan_folder
    assert loaded.pie_scan_folder == settings.pie_scan_folder
    assert loaded.pie_scan_folders == settings.pie_scan_folders
    assert loaded.effective_pie_scan_folders() == settings.pie_scan_folders
    assert loaded.pics_database_path == settings.pics_database_path
    assert loaded.manual_peak_file == settings.manual_peak_file
    assert loaded.temperature_scan_result_file == settings.temperature_scan_result_file
    assert loaded.pie_identification_result_file == settings.pie_identification_result_file
    assert loaded.mole_fraction_result_file == settings.mole_fraction_result_file
    assert loaded.cal_a == settings.cal_a
    assert loaded.cal_b == settings.cal_b
    assert loaded.cal_c == settings.cal_c
    assert loaded.calibration_points == settings.calibration_points
    assert loaded.light_source == settings.light_source
    assert loaded.temperature_photon_normalize is False
    assert loaded.temperature_kr_correct is True
    assert loaded.pie_photon_mode == settings.pie_photon_mode
    saved_yaml = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "mass_discrimination" not in saved_yaml["general_parameters"]["normalization"]
    assert loaded.kr_calibration_folder == settings.kr_calibration_folder
    assert loaded.kr_calibration_peak_file == settings.kr_calibration_peak_file
    assert loaded.expansion_factors == settings.expansion_factors
    assert loaded.selected_elements == settings.selected_elements
    assert loaded.peak_algorithm == settings.peak_algorithm
    assert loaded.detection_min_idx == settings.detection_min_idx
    assert loaded.threshold_end == settings.threshold_end
    assert loaded.min_intensity == settings.min_intensity
    assert loaded.pie_energy_decimals == settings.pie_energy_decimals
    assert loaded.pie_recursive is False
    assert loaded.pie_prefer_gaussian is False
    assert loaded.pie_integration_method == "baseline"
    assert loaded.pie_multi_folder_mode is True
    assert loaded.pie_merge_method == settings.pie_merge_method
    assert loaded.temp_reference_mode == settings.temp_reference_mode
    assert loaded.temp_prefer_gaussian is False
    assert loaded.temp_integration_method == "baseline"
    assert loaded.temp_kr_mz == settings.temp_kr_mz
    assert loaded.temp_curve_class_change_threshold == settings.temp_curve_class_change_threshold
    assert loaded.temp_curve_class_peak_fraction == settings.temp_curve_class_peak_fraction
    assert loaded.pics_no_mz == settings.pics_no_mz
    assert loaded.pics_no_formula == settings.pics_no_formula
    assert loaded.pics_no_mf == settings.pics_no_mf
    assert loaded.pics_new_species_mf == settings.pics_new_species_mf
    assert loaded.mf_md_preset == settings.mf_md_preset
    assert loaded.mf_mass_disc_exponent == settings.mf_mass_disc_exponent
    assert loaded.mf_parent_mz == settings.mf_parent_mz
    assert loaded.mf_parent_initial_mf == settings.mf_parent_initial_mf
    assert loaded.mf_reference_temperature == settings.mf_reference_temperature
    assert loaded.mf_reference_species_mz == settings.mf_reference_species_mz
    assert loaded.mf_reference_species_tm == settings.mf_reference_species_tm
    assert loaded.mf_reference_species_mf_at_tm == settings.mf_reference_species_mf_at_tm
    assert loaded.mf_photon_energy == settings.mf_photon_energy
    assert loaded.mf_kr_data == settings.mf_kr_data
    assert saved_yaml["data_sources"]["pie_scan_folders"] == ["data/pie-low", "data/pie-high"]


def test_project_settings_legacy_single_pie_folder_remains_effective():
    settings = ProjectSettings(pie_scan_folder="data/legacy-pie")

    assert settings.effective_pie_scan_folders() == ["data/legacy-pie"]


def test_project_scoped_settings_store_relative_paths_and_rebase_after_move(tmp_path):
    original_root = tmp_path / "computer-a" / "Project_Portable"
    config_path = original_root / "config" / "project.yaml"
    external_database = tmp_path / "shared" / "species.sqlite"
    settings = ProjectSettings(
        project_name="Portable",
        output_dir=str(original_root),
        temperature_scan_folder=str(original_root / "raw_data" / "temperature_scan" / "run-1"),
        pie_scan_folder=str(original_root / "raw_data" / "pie_scan" / "low"),
        pie_scan_folders=[
            str(original_root / "raw_data" / "pie_scan" / "low"),
            str(original_root / "raw_data" / "pie_scan" / "high"),
        ],
        pics_database_path=str(external_database),
        pie_identification_result_file=str(original_root / "analysis" / "pie" / "identification.xlsx"),
    )

    save_project_settings(settings, config_path)

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["project"]["output_dir"] == "."
    assert saved["data_sources"]["temperature_scan_folder"] == "raw_data/temperature_scan/run-1"
    assert saved["data_sources"]["pie_scan_folders"] == [
        "raw_data/pie_scan/low",
        "raw_data/pie_scan/high",
    ]
    assert saved["data_sources"]["pics_database_path"] == str(external_database)
    assert saved["analysis_artifacts"]["pie_identification_result_file"] == (
        "analysis/pie/identification.xlsx"
    )

    moved_root = tmp_path / "computer-b" / "Project_Portable"
    moved_config = moved_root / "config" / "project.yaml"
    moved_config.parent.mkdir(parents=True)
    moved_config.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

    loaded = load_project_settings(moved_config)

    assert loaded.output_dir == str(moved_root)
    assert loaded.temperature_scan_folder == str(moved_root / "raw_data" / "temperature_scan" / "run-1")
    assert loaded.pie_scan_folder == str(moved_root / "raw_data" / "pie_scan" / "low")
    assert loaded.pie_scan_folders == [
        str(moved_root / "raw_data" / "pie_scan" / "low"),
        str(moved_root / "raw_data" / "pie_scan" / "high"),
    ]
    assert loaded.pics_database_path == str(external_database)
    assert loaded.pie_identification_result_file == str(
        moved_root / "analysis" / "pie" / "identification.xlsx"
    )


def test_project_scoped_settings_rebase_legacy_windows_absolute_paths(tmp_path):
    project_root = tmp_path / "migrated" / "Project_C6H5ClO"
    config_path = project_root / "config" / "project.yaml"
    config_path.parent.mkdir(parents=True)
    old_root = r"C:\Users\Administrator\Desktop\output\Project_C6H5ClO"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "Legacy", "output_dir": old_root},
                "data_sources": {
                    "temperature_scan_folder": old_root + r"\raw_data\temperature_scan\温度扫描",
                    "pie_scan_folder": old_root + r"\raw_data\pie_scan\PIE_低能段",
                    "pie_scan_folders": [
                        old_root + r"\raw_data\pie_scan\PIE_低能段",
                        old_root + r"\raw_data\pie_scan\PIE_高能段",
                    ],
                    "pics_database_path": r"D:\shared\species.sqlite",
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    loaded = load_project_settings(config_path)

    assert loaded.output_dir == str(project_root)
    assert loaded.temperature_scan_folder == str(
        project_root / "raw_data" / "temperature_scan" / "温度扫描"
    )
    assert loaded.pie_scan_folders == [
        str(project_root / "raw_data" / "pie_scan" / "PIE_低能段"),
        str(project_root / "raw_data" / "pie_scan" / "PIE_高能段"),
    ]
    assert loaded.pics_database_path == r"D:\shared\species.sqlite"

    save_project_settings(loaded, config_path)
    migrated = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert migrated["project"]["output_dir"] == "."
    assert migrated["data_sources"]["pie_scan_folders"] == [
        "raw_data/pie_scan/PIE_低能段",
        "raw_data/pie_scan/PIE_高能段",
    ]
    assert migrated["data_sources"]["pics_database_path"] == r"D:\shared\species.sqlite"


def test_project_settings_manager_uses_portable_storage_but_absolute_runtime_paths(tmp_path):
    project_root = tmp_path / "Project_Manager"
    temperature_folder = project_root / "raw_data" / "temperature_scan" / "run"
    manager = ProjectSettingsManager()
    manager.clear_project_path()
    try:
        manager.set_project_path(project_root)
        manager.set(
            ProjectSettings(
                project_name="Manager",
                output_dir=str(project_root),
                temperature_scan_folder=str(temperature_folder),
            )
        )

        config_path = manager.save()
        saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        reloaded = manager.reload()

        assert saved["project"]["output_dir"] == "."
        assert saved["data_sources"]["temperature_scan_folder"] == "raw_data/temperature_scan/run"
        assert reloaded.output_dir == str(project_root)
        assert reloaded.temperature_scan_folder == str(temperature_folder)
    finally:
        manager.clear_project_path()


def test_load_project_settings_accepts_saved_yaml_key_names(tmp_path):
    path = tmp_path / "project.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "Saved Name", "system": "Saved System"},
                "analysis_artifacts": {
                    "temperature_scan_result_file": "out/temp.xlsx",
                    "pie_identification_result_file": "out/pie.xlsx",
                    "mole_fraction_result_file": "out/mf.xlsx",
                },
                "calibration": {"a": 1.0, "b": 2.0, "c": 3.0, "points": [{"tof": 1, "mz": 2}]},
                "peak_detection": {"algorithm": "cwt", "detection_min_idx": 321},
                "function_params": {"pie": {"energy_decimals": 3}},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_project_settings(path)

    assert loaded.project_name == "Saved Name"
    assert loaded.system == "Saved System"
    assert loaded.temperature_scan_result_file == "out/temp.xlsx"
    assert loaded.pie_identification_result_file == "out/pie.xlsx"
    assert loaded.mole_fraction_result_file == "out/mf.xlsx"
    assert loaded.cal_a == 1.0
    assert loaded.cal_b == 2.0
    assert loaded.cal_c == 3.0
    assert loaded.calibration_points == [{"tof": 1, "mz": 2}]
    assert loaded.peak_algorithm == "cwt"
    assert loaded.detection_min_idx == 321
    assert loaded.pie_energy_decimals == 3


def test_load_project_settings_accepts_legacy_mass_discrimination(tmp_path):
    path = tmp_path / "legacy-project.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "general_parameters": {
                    "normalization": {
                        "light_source": "beam_current",
                        "mass_discrimination": 0.42,
                    },
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    loaded = load_project_settings(path)

    assert loaded.light_source == "beam_current"
    assert loaded.mass_discrimination == pytest.approx(0.42)


def test_normalization_settings_preserve_project_mass_discrimination():
    settings = ProjectSettings(mass_discrimination=0.42)

    assert settings.to_normalization_settings().mass_discrimination == pytest.approx(0.42)


def test_load_project_settings_warns_when_file_missing(tmp_path, caplog):
    missing_path = tmp_path / "missing-project.yaml"

    with caplog.at_level("WARNING"):
        loaded = load_project_settings(missing_path)

    assert isinstance(loaded, ProjectSettings)
    assert "Project settings file not found" in caplog.text
