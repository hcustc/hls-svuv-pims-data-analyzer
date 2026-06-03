from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "output"

OUTPUT_SUBDIRS = (
    ("exports", "calibration"),
    ("exports", "peak_ranges"),
    ("exports", "temperature"),
    ("exports", "pie"),
    ("images", "temperature"),
    ("images", "energy"),
    ("logs",),
)


def ensure_output_dir(*parts: str) -> Path:
    path = OUTPUT_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_structure() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for parts in OUTPUT_SUBDIRS:
        ensure_output_dir(*parts)
    return OUTPUT_ROOT
