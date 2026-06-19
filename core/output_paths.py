from __future__ import annotations

from .runtime_paths import writable_path


OUTPUT_ROOT = writable_path("output")

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
