---
name: bl03u-spectrum-io
description: Use when changing BL03U spectrum file parsing, TXT/ASC/888 input handling, folder spectrum loading, summed spectra, metadata extraction, or output path conventions.
---

# Skill: BL03U Spectrum IO

## When to use

- Reading or parsing BL03U mass spectrum files.
- Handling `.txt`, `.asc`, or `.888` inputs.
- Changing metadata/header behavior.
- Changing folder spectrum loading or summed spectrum behavior.
- Fixing import/export paths for spectrum-derived output.

## Code map

- `core/spectrum_io.py`
  - `Spectrum`
  - `read_spectrum`
  - `read_bl03u_txt`
  - `list_spectrum_files`
  - `read_folder_spectra`
  - `sum_spectra`
  - `extract_header_numbers`
  - `extract_first_number`
- `core/output_paths.py`
  - Output subdirectory conventions.
- `api/server.py`
  - Spectrum read/sum/peak endpoints.
- `frontends/pyqt_app/main_window.py`
  - GUI spectrum loading, plotting, summed spectra, and peak table integration.
- `data/examples/example-mass-spectrum.txt`
  - Example input data.
- `tests/test_core_smoke.py`
  - Smoke tests that indirectly validate parser behavior.

## Core workflow

- `read_spectrum` separates numeric data lines from non-numeric metadata lines.
- A one-column numeric file becomes intensity-only data with a synthetic x-axis.
- A two-or-more-column numeric file uses the first column as `x` and second column as `y`.
- BL03U `.txt` files commonly use a 10-line header.
- `read_bl03u_txt` applies BL03U defaults such as header handling and trim behavior.
- Folder readers sort files and support suffix filters.
- `sum_spectra` aligns spectra by truncating all y-arrays to the shortest length before summing.

## Gotchas

- Do not assume every spectrum file has two columns.
- Preserve metadata lines; temperature, photon energy, IO current, and beam current extraction depend on headers.
- Be careful with `trim_start`: temperature and PIE workflows may need untrimmed data.
- Keep suffix handling consistent between core, GUI, and API.
- Keep scientific parsing logic in `core/spectrum_io.py`, not in frontend code.

## Validation

For future code changes involving spectrum IO, run:

```bash
python -m pytest tests/test_core_smoke.py
```

Add focused tests for one-column files, two-column files, metadata preservation, 10-line BL03U headers, and empty/non-numeric files when changing parser behavior. For API changes, manually check the spectrum read, sum, and peak endpoints from `api/server.py`.