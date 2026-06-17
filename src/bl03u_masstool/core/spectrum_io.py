from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np


FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass
class Spectrum:
    x: np.ndarray
    y: np.ndarray
    metadata_lines: list[str]
    path: str | None = None

    def trimmed(self, start: int) -> "Spectrum":
        start = max(0, int(start))
        if start == 0:
            return self
        return Spectrum(
            x=np.arange(start + 1, start + 1 + max(0, len(self.y) - start), dtype=float),
            y=self.y[start:],
            metadata_lines=self.metadata_lines,
            path=self.path,
        )


def _read_lines(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return handle.readlines()


def parse_numeric_rows(lines: Iterable[str]) -> tuple[list[str], np.ndarray]:
    """Parse numeric rows from a BL03U spectrum text-like file.

    Lines with one numeric token are treated as intensity-only rows. Lines with
    two or more numeric tokens are retained as numeric rows so two-column files
    can preserve their x axis.
    """
    metadata: list[str] = []
    rows: list[list[float]] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        values = [float(match.group(0)) for match in FLOAT_RE.finditer(line)]
        if values and _line_is_numeric_like(line, values):
            rows.append(values)
        else:
            metadata.append(line)

    if not rows:
        return metadata, np.array([], dtype=float)

    max_cols = max(len(row) for row in rows)
    padded = [row + [np.nan] * (max_cols - len(row)) for row in rows]
    return metadata, np.array(padded, dtype=float)


def _line_is_numeric_like(line: str, values: list[float]) -> bool:
    stripped = FLOAT_RE.sub("", line)
    stripped = stripped.replace(",", "").replace("\t", "").replace(" ", "")
    return not stripped and bool(values)


def read_spectrum(
    path: str | Path,
    *,
    header_lines: int | None = None,
    trim_start: int = 0,
    x_start: int = 1,
) -> Spectrum:
    """Read txt/asc/888 spectrum files into x/y arrays."""
    path = Path(path)
    lines = _read_lines(path)
    data_lines = lines[header_lines:] if header_lines is not None else lines
    metadata, rows = parse_numeric_rows(data_lines)
    if header_lines is not None:
        metadata = [line.strip() for line in lines[:header_lines] if line.strip()] + metadata

    if rows.size == 0:
        x = np.array([], dtype=float)
        y = np.array([], dtype=float)
    elif rows.ndim == 2 and rows.shape[1] >= 2 and not np.all(np.isnan(rows[:, 1])):
        x = rows[:, 0].astype(float)
        y = rows[:, 1].astype(float)
    else:
        y = rows[:, 0].astype(float) if rows.ndim == 2 else rows.astype(float)
        x = np.arange(x_start, x_start + len(y), dtype=float)

    if trim_start:
        trim_start = max(0, int(trim_start))
        x = np.arange(trim_start + x_start, trim_start + x_start + max(0, len(y) - trim_start), dtype=float)
        y = y[trim_start:]

    return Spectrum(x=x, y=y, metadata_lines=metadata, path=str(path))


def read_bl03u_txt(path: str | Path, *, trim_start: int = 4000) -> Spectrum:
    """Read current BL03U TXT files with the usual 10-line header."""
    return read_spectrum(path, header_lines=10, trim_start=trim_start, x_start=1)


def list_spectrum_files(folder: str | Path, suffixes: tuple[str, ...] = (".txt", ".asc", ".888")) -> list[Path]:
    folder = Path(folder)
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def read_folder_spectra(
    folder: str | Path,
    *,
    suffixes: tuple[str, ...] = (".txt",),
    trim_start: int = 4000,
) -> list[Spectrum]:
    return [read_spectrum(path, header_lines=10 if path.suffix.lower() == ".txt" else None, trim_start=trim_start) for path in list_spectrum_files(folder, suffixes)]


def sum_spectra(
    folder: str | Path,
    *,
    suffixes: tuple[str, ...] = (".txt",),
    trim_start: int = 4000,
) -> Spectrum:
    spectra = read_folder_spectra(folder, suffixes=suffixes, trim_start=trim_start)
    if not spectra:
        raise ValueError("no spectrum files found")
    min_len = min(len(item.y) for item in spectra)
    if min_len == 0:
        raise ValueError("spectrum files contain no numeric data")
    y_sum = np.sum([item.y[:min_len] for item in spectra], axis=0)
    x = spectra[0].x[:min_len]
    return Spectrum(x=x, y=y_sum, metadata_lines=[], path=str(folder))


def extract_header_numbers(path: str | Path, max_lines: int = 10) -> list[float]:
    lines = _read_lines(path)[:max_lines]
    values: list[float] = []
    for line in lines:
        match = FLOAT_RE.search(line)
        if match:
            values.append(float(match.group(0)))
    return values


def extract_first_number(line: str) -> float | None:
    match = FLOAT_RE.search(line)
    return float(match.group(0)) if match else None

