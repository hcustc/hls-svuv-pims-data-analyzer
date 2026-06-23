from __future__ import annotations

import yaml

from bl03u_masstool.core.project_settings import ProjectSettings, load_project_settings, save_project_settings


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
        mass_discrimination=0.9,
        kr_calibration_folder="data/kr",
        kr_calibration_peak_file="config/peak.yaml",
        expansion_factors={550.0: 1.0, 600.0: 1.1},
        selected_elements=["C", "H", "O"],
        peak_algorithm="cwt",
        detection_min_idx=123,
        threshold_end=4.5,
        min_intensity=6.7,
        pie_energy_decimals=2,
        pie_recursive=False,
        pie_prefer_gaussian=False,
        pie_multi_folder_mode=True,
        pie_merge_method="mean",
        temp_reference_mode="kr",
        temp_prefer_gaussian=False,
        temp_kr_mz=86,
        pics_no_mz=31,
        pics_no_formula="15NO",
        pics_no_mf=0.02,
        pics_new_species_mf=0.003,
        mf_mass_disc_exponent=0.7,
        mf_parent_mz=130,
        mf_parent_initial_mf=0.004,
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
    assert loaded.mass_discrimination == settings.mass_discrimination
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
    assert loaded.pie_multi_folder_mode is True
    assert loaded.pie_merge_method == settings.pie_merge_method
    assert loaded.temp_reference_mode == settings.temp_reference_mode
    assert loaded.temp_prefer_gaussian is False
    assert loaded.temp_kr_mz == settings.temp_kr_mz
    assert loaded.pics_no_mz == settings.pics_no_mz
    assert loaded.pics_no_formula == settings.pics_no_formula
    assert loaded.pics_no_mf == settings.pics_no_mf
    assert loaded.pics_new_species_mf == settings.pics_new_species_mf
    assert loaded.mf_mass_disc_exponent == settings.mf_mass_disc_exponent
    assert loaded.mf_parent_mz == settings.mf_parent_mz
    assert loaded.mf_parent_initial_mf == settings.mf_parent_initial_mf
    assert loaded.mf_photon_energy == settings.mf_photon_energy
    assert loaded.mf_kr_data == settings.mf_kr_data


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
