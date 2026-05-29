---
name: bl03u-calibration-config
description: Use when modifying BL03U TOF-to-m/z calibration, YAML configuration loading/saving, calibration points, app paths, or default analysis settings.
---

# Skill: BL03U Calibration and Config

## When to use

- Changing TOF-to-m/z calibration coefficients or fitting.
- Adding or changing YAML config keys.
- Loading or saving calibration, peak detection, integration, normalization, or app config.
- Changing project-relative paths or default database/config paths.
- Wiring config defaults into the PyQt GUI or FastAPI API.

## Code map

- `core/calibration.py`
  - `Calibration`
  - `tof_to_mz`
  - `mz_to_tof`
  - `fit_quadratic_calibration`
  - `score_quadratic_calibration`
- `core/config.py`
  - `PROJECT_ROOT`
  - `CONFIG_ROOT`
  - `project_path`
  - `load_yaml`
  - `load_app_config`
  - `species_database_path`
  - `load_calibration_config`
  - `save_calibration_config`
  - `load_calibration_points`
  - `load_peak_detection_config`
  - `save_peak_detection_config`
  - `load_peak_integration_config`
- `config/app.yaml`
- `config/calibration.yaml`
- `config/calibration_points.yaml`
- `config/peak_detection.yaml`
- `config/peak_integration.yaml`
- `config/normalization.yaml`
- `frontends/pyqt_app/main_window.py`
  - `apply_config_defaults`
- `api/server.py`
  - Loads calibration and detection settings for API workflows.
- `tests/test_core_smoke.py`
  - Calibration and config smoke coverage.

## Core workflow

- Calibration equation is `mz = c + b * tof + a * tof^2`.
- `Calibration` stores quadratic coefficients and exposes TOF/m/z conversion helpers.
- `mz_to_tof` solves the positive root and handles near-linear cases.
- `fit_quadratic_calibration` needs at least three `(tof, mz)` points.
- `core.config.PROJECT_ROOT` is derived from `core/config.py`, so config/database paths should stay repository-relative unless explicitly overridden.
- `app.yaml` points to default config and database locations.
- Dataclass defaults, YAML defaults, GUI defaults, and API defaults should stay aligned.

## Gotchas

- Do not silently change the calibration equation order or coefficient meaning.
- YAML roots should remain mappings; invalid roots should fail clearly.
- Preserve config backward compatibility unless all callers and tests are updated.
- Saving config can create parent directories; avoid redirecting user data unexpectedly.
- Keep path handling cross-platform and avoid hard-coded absolute paths.

## Validation

For future calibration/config changes, run:

```bash
python -m pytest tests/test_core_smoke.py
```

Add focused tests for calibration round trips, invalid YAML roots, missing optional config values, and relative/absolute path behavior. For GUI changes, launch `python main.py` and verify calibration fields populate from YAML.