---
name: bl03u-peak-detection
description: Use when modifying BL03U peak detection, Gaussian fitting, manual peak ranges, peak integration boundaries, peak detection YAML settings, or peak table outputs.
---

# Skill: BL03U Peak Detection

## When to use

- Changing automatic peak finding.
- Changing Gaussian fitting or peak boundary logic.
- Adding or modifying manual peak ranges.
- Changing integration behavior or peak output tables.
- Tuning peak detection YAML settings.
- Wiring peak behavior into GUI, API, temperature scan, or PIE workflows.

## Code map

- `core/peak_detection.py`
  - `GaussianFit`
  - `Peak`
  - `gaussian`
  - `fit_gaussian`
  - `detect_peaks_in_range`
  - `add_manual_peak`
  - `peaks_to_dataframe`
- `core/integration.py`
  - `integrate_peak`
  - Baseline and Gaussian integration helpers.
- `core/peak_ranges.py`
  - Manual peak range loading and conversion.
- `config/peak_detection.yaml`
- `config/peak_integration.yaml`
- `core/config.py`
  - `PeakDetectionConfig`
  - `load_peak_detection_config`
- `frontends/pyqt_app/main_window.py`
  - Auto/manual peak UI wiring.
- `frontends/pyqt_app/dialogs.py`
  - Analysis parameter dialogs.
- `api/server.py`
  - Spectrum peak endpoint and PIE job peak settings.
- `tests/test_core_smoke.py`
  - Peak detection, integration, and manual peak range tests.

## Core workflow

- `detect_peaks_in_range` converts y-data to NumPy, applies the detection start index, and finds local maxima above `min_intensity`.
- Nearby duplicates and weak shoulder/tail peaks are suppressed before final output.
- Peak left/right boundaries are found from `threshold_end` and may be refined by Gaussian fit.
- `Peak.time` is TOF plus optional offset.
- `Peak.mz` comes from the active `Calibration` object.
- Gaussian boundary behavior uses `gaussian_boundary_scale` and `boundary_padding`.
- Manual peak addition chooses the max intensity in the selected range and marks `is_auto=False`.
- `peaks_to_dataframe` defines user-facing peak columns; update all callers/tests if these change.

## Gotchas

- Keep boundary indices valid and inclusive.
- `detection_min_idx` may default high for real BL03U data; tests often override it to detect synthetic peaks.
- Gaussian fitting can fail and should fall back gracefully instead of crashing.
- Do not make core peak detection depend on PyQt widgets or FastAPI request objects.
- Preserve supported manual range formats when changing `core/peak_ranges.py`.
- Integration may re-fit Gaussian against the current spectrum; avoid stale-fit assumptions.

## Validation

For future peak detection changes, run:

```bash
python -m pytest tests/test_core_smoke.py
```

Add focused tests for simple peaks, Gaussian failure fallback, duplicate/tail suppression, manual range loading, and Gaussian/non-Gaussian integration. Manually validate GUI auto peak detection, manual peak addition, and the API spectrum peak endpoint when frontend/API wiring changes.