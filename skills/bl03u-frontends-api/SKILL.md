---
name: bl03u-frontends-api
description: Use when modifying BL03U PyQt GUI behavior, FastAPI endpoints, static web UI, dialogs, worker threading, or frontend-to-core integration.
---

# Skill: BL03U Frontends and API

## When to use

- Changing PyQt window layout, signals, slots, or event handlers.
- Changing dialogs or long-running worker behavior.
- Changing FastAPI endpoints or request/response models.
- Changing static web UI HTML, CSS, or JavaScript.
- Keeping GUI, API, and core behavior consistent.

## Code map

- `main.py`
  - PyQt entry point.
- `frontends/pyqt_app/main_window.py`
  - `MainWindow`
  - Spectrum loading, plotting, calibration, peak table actions, and menu/tool wiring.
- `frontends/pyqt_app/dialogs.py`
  - Dialogs for temperature scan, isotope, PIE/PICS, and analysis parameters.
- `frontends/pyqt_app/workers.py`
  - Background worker patterns.
- `frontends/pyqt_app/ui_massspec.py`
  - Large/generated UI class.
- `frontends/pyqt_app/theme.py`
- `api/server.py`
  - FastAPI app, static mount, background job state, upload parsing, and all API endpoints.
- `frontends/web_app/static/index.html`
- `frontends/web_app/static/app.js`
- `frontends/web_app/static/styles.css`
- `frontends/web_app/README.md`
- `requirements.txt`

## Core workflow

- PyQt and FastAPI should call core modules instead of duplicating scientific logic.
- `MainWindow.apply_config_defaults` loads calibration and calibration point YAML into the GUI.
- `MainWindow` wires plotting, spectrum summing, calibration, saving, manual peaks, and automatic peak detection.
- Dialogs expose core tools for isotope calculation, temperature scan, PIE/PICS fitting, and parameters.
- FastAPI mounts static assets from `frontends/web_app/static`.
- The static web UI is plain HTML/CSS/JavaScript and calls `api/server.py` endpoints directly.
- Long PIE operations run as background jobs with job IDs and polling endpoints.

## Gotchas

- Do not block the PyQt UI thread with long analysis.
- Keep scientific algorithms in `core/`, not in PyQt widgets, FastAPI handlers, or static JavaScript.
- Preserve API response shapes consumed by `frontends/web_app/static/app.js`.
- Avoid breaking Chinese UI labels/messages unless intentionally updating localization.
- `api/server.py` combines endpoints, upload parsing, job state, and PICS DB writes; localize changes carefully.
- Prefer wrapper logic in `main_window.py` or `dialogs.py` over manual edits to large/generated `ui_massspec.py` when possible.
- Static web app has no build step; validation is runtime/manual plus API tests.

## Validation

For future frontend/API changes, run:

```bash
python -m pytest tests/test_core_smoke.py
```

Manual PyQt smoke check:

```bash
python main.py
```

Load a spectrum, sum a folder, run auto peak detection, and open core tool dialogs.

Manual FastAPI/web smoke check:

```bash
uvicorn api.server:app --reload
```

Open the served web UI and exercise isotope, spectrum read/peaks, PIE/PICS search, upload, fitting, progress, and export flows. Add FastAPI `TestClient` tests for changed endpoints when response shapes or error handling change.