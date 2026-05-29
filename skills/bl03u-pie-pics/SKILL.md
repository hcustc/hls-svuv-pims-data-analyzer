---
name: bl03u-pie-pics
description: Use when modifying BL03U PIE curve generation, PICS fitting, species SQLite database queries/uploads, ionization-energy lookup, or web PICS workflows.
---

# Skill: BL03U PIE and PICS

## When to use

- Generating PIE curves from energy-resolved spectra.
- Changing PICS database schema, query, upload, or fitting behavior.
- Changing species candidate selection or ionization-energy lookup.
- Supporting manual, automatic, or locked fit coefficients.
- Updating web PIE/PICS upload, search, fitting, export, or progress workflows.

## Code map

- `core/pie_analysis.py`
  - `SCHEMA_SQL`
  - `load_species_database`
  - `save_species_database_sqlite`
  - `load_species_database_sqlite`
  - `query_species_by_mz`
  - `extract_photon_energy`
  - `analyze_pie_folder`
  - `build_pie_curves`
  - `identify_species_for_mz_with_curve`
  - `fit_species_combination_with_curve`
- `api/server.py`
  - `PieJob`
  - `PicsLibrary`
  - PIE job endpoints.
  - PICS search/species/upload endpoints.
- `database/species_database.sqlite`
- `frontends/web_app/static/index.html`
- `frontends/web_app/static/app.js`
- `frontends/web_app/static/styles.css`
- `frontends/web_app/README.md`
- `frontends/pyqt_app/dialogs.py`
  - PIE/PICS dialogs.
- `tests/test_core_smoke.py`
  - SQLite roundtrip, PICS fitting, manual coefficient, locked coefficient, PIE folder, and uploaded-curve tests.

## Core workflow

- PIE analysis groups spectra by photon energy.
- Photon energy may come from path segments such as `13.0eV` or from metadata.
- Spectra at the same rounded energy are summed before integration.
- Reference peaks are detected automatically or loaded from manual peak files.
- Integrated intensities can be photon-normalized by the selected light source.
- PIE curves are built per rounded m/z.
- SQLite PICS schema includes `species` and `pic_cross_sections` tables.
- PICS fitting interpolates candidate cross-section curves onto the experimental energy grid.
- Fit modes include automatic non-negative fitting, manual coefficients, and locked coefficients.
- FastAPI uses background jobs for PIE analysis and in-memory temporary PICS libraries with TTL.

## Gotchas

- Treat `database/species_database.sqlite` as a maintained data artifact; do not destructively rewrite it without explicit approval.
- Server-library uploads require `BL03U_ADMIN_TOKEN`.
- Temporary PICS libraries are session-scoped and expire by `BL03U_PICS_LIBRARY_TTL_HOURS`.
- Upload file size is controlled by `BL03U_MAX_UPLOAD_MB`.
- Preserve long-table and wide-table uploaded PIE parsing behavior.
- Candidate species IDs in API payloads may arrive as strings; parse carefully.
- Be careful with mutable global job state and locks in `api/server.py`.

## Validation

For future PIE/PICS changes, run:

```bash
python -m pytest tests/test_core_smoke.py
```

Add focused tests for SQLite schema round trips, PICS candidate query by m/z, no-candidate fitting, manual coefficients, locked species IDs, uploaded PIE long/wide parsing, and PICS upload parsing. For web changes, run `uvicorn api.server:app --reload`, open the web UI, exercise PIE/PICS fitting, search, and temporary upload. For admin upload changes, verify missing-token rejection and token-authenticated success.