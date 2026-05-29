---
name: bl03u-isotope-tools
description: Use when modifying BL03U molecular formula parsing, isotope abundance calculations, isotope API responses, or isotope GUI tools.
---

# Skill: BL03U Isotope Tools

## When to use

- Adding supported elements or isotope masses/abundances.
- Changing molecular formula parsing.
- Changing abundance thresholds or output shape.
- Wiring isotope calculation results into FastAPI or PyQt dialogs.
- Debugging isotope results for molecular formulas.

## Code map

- `core/isotope.py`
  - `ISOTOPES`
  - `parse_formula`
  - `calculate_isotope_distribution`
- `api/server.py`
  - Isotope payload and endpoint.
- `frontends/pyqt_app/dialogs.py`
  - `IsotopeAbundanceDialog`
- `tests/test_core_smoke.py`
  - Isotope distribution smoke test.

## Core workflow

- Formula parsing supports element symbols matching `[A-Z][a-z]?` with optional integer counts.
- Missing element counts default to 1.
- Unsupported elements raise `ValueError`.
- Element isotope distributions are cached with `lru_cache`.
- Molecular distributions are built by repeated convolution of element distributions.
- Output rows contain `mass`, `abundance`, and `percent`.
- `min_percent` filters low-abundance results.

## Gotchas

- The current parser does not support parentheses, charges, isotope labels, decimal counts, or hydrates unless explicitly extended.
- Preserve deterministic sorted output by mass.
- Large formulas can grow combinatorially; avoid unnecessary expansion.
- If adding parser grammar, add tests for both old and new behavior.
- API and GUI may depend on the `mass`, `abundance`, and `percent` keys.

## Validation

For future isotope changes, run:

```bash
python -m pytest tests/test_core_smoke.py
```

Add focused tests for formulas such as `H2O`, `CO2`, and BL03U-relevant species; unsupported elements; `min_percent`; and multi-character elements such as `Cl`, `Br`, and `Si`. For API/UI changes, manually test the isotope endpoint and `IsotopeAbundanceDialog`.