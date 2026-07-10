from types import SimpleNamespace

import pandas as pd

from bl03u_masstool.core.export_pie_results import export_pie_results_to_excel


def test_export_pie_results_keeps_missing_ie_blank(tmp_path):
    dialog = SimpleNamespace(
        all_fit_results={
            30: {
                "success": True,
                "r_squared": 0.9,
                "model": {
                    "species": [
                        {
                            "species": "Unknown isomer",
                            "ie": None,
                            "coefficient": 1.0,
                            "contribution_percent": 100.0,
                        }
                    ]
                },
            }
        }
    )
    output_path = tmp_path / "pie_results.xlsx"

    result = export_pie_results_to_excel(dialog, output_path)

    assert result["success"] is True
    exported = pd.read_excel(output_path)
    assert pd.isna(exported.loc[0, "电离能(eV)"])


def test_export_pie_results_falls_back_to_ionization_energy_alias(tmp_path):
    dialog = SimpleNamespace(
        all_fit_results={
            44: {
                "success": True,
                "r_squared": 0.95,
                "model": {
                    "species": [
                        {
                            "species": "CO2",
                            "ie": None,
                            "ionization_energy": 13.777,
                            "coefficient": 2.0,
                            "contribution_percent": 100.0,
                        }
                    ]
                },
            }
        }
    )
    output_path = tmp_path / "pie_results.xlsx"

    result = export_pie_results_to_excel(dialog, output_path)

    assert result["success"] is True
    exported = pd.read_excel(output_path)
    assert exported.loc[0, "电离能(eV)"] == 13.777
