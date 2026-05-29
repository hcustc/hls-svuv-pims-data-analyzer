---
name: bl03u-temperature-scan
description: Use when modifying BL03U temperature scan analysis, temperature curve building, IO or beam-current normalization, Kr expansion correction, or scan exports.
---

# Skill: BL03U Temperature Scan

## When to use

- Changing temperature-dependent spectrum analysis.
- Building or exporting temperature curves by m/z.
- Changing IO or beam-current normalization.
- Changing Kr expansion correction or mass discrimination correction.
- Supporting manual peak files in temperature scans.
- Updating the temperature scan PyQt dialog.

## Code map

- `core/temperature_scan.py`
  - `extract_temperature`
  - `extract_photon_energy`
  - `analyze_temperature_folder`
  - `_build_reference_spectrum`
  - `_apply_temperature_normalization`
  - `compute_kr_expansion_factors`
  - `build_temperature_curves`
- `core/normalization.py`
  - `extract_light_intensity`
  - Normalization setting helpers.
- `core/integration.py`
- `core/peak_ranges.py`
- `core/peak_detection.py`
- `frontends/pyqt_app/dialogs.py`
  - `TemperatureScanDialog`
- `core/output_paths.py`
  - Temperature export/image directories.
- `tests/test_core_smoke.py`
  - Temperature curve grouping, manual peak, IO normalization, and beam-current normalization tests.

## Core workflow

- Temperature scan reads BL03U `.txt` files with headers and usually needs untrimmed data.
- Temperature is extracted from metadata lines matching temperature labels.
- Photon energy is extracted from metadata lines containing energy labels.
- IO or beam-current normalization routes through `extract_light_intensity`.
- Reference peaks come from either a manual peak file or detected peaks from a reference spectrum.
- `reference_mode` supports summed-spectrum and max-temperature reference strategies.
- Each spectrum integrates the same reference peak bounds to build comparable curves.
- Result tables include raw, photon-normalized, expansion-normalized, and final area columns.
- `build_temperature_curves` groups rows by rounded m/z and sums areas per temperature.

## Gotchas

- Manual peak files should bypass reference auto-detection but still use calibration.
- Kr correction needs the configured Kr m/z present and positive.
- `mass_discrimination` must remain positive.
- Do not assume IO current; workflows may use beam current instead.
- Preserve output column names consumed by UI, tests, and exports.
- File ordering and m/z grouping affect curve reproducibility.

## Validation

For future temperature scan changes, run:

```bash
python -m pytest tests/test_core_smoke.py
```

Add focused tests for temperature extraction, empty folder schemas, reference modes, Kr correction error paths, explicit expansion factor mapping, manual peak files, and beam-current normalization. For GUI changes, run `python main.py`, open the temperature scan dialog, analyze a folder, and verify curves/exports.