"""PICS data import: parse and write to the species SQLite database.

This module contains pure-Python logic for reading PICS tables from files
(CSV / TSV / XLSX) and writing the resulting records into the SQLite database.
It has no dependency on FastAPI so it can be called from both the PyQt desktop
application and the web API backend.

Supported table formats
-----------------------
Long table  – one energy point per row.
    Required columns: mz, name, energy_ev, cross_section
    Optional column:  ionization_energy

Wide table  – one species per row, energy values as column names.
    Required columns: mz, name
    Optional column:  ionization_energy
    Energy columns:   any column whose name parses as a float (with optional
                      "eV" suffix), e.g. ``10.5``, ``10.5eV``, ``10.5 eV``

Write modes
-----------
upsert       Replace matching species (same mz + name + ie), keep the rest.
append       Insert all records as new entries regardless of duplicates.
overwrite_all  Delete everything then insert.  Requires confirm=True.
"""

from __future__ import annotations

import math
import re
import sqlite3
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Column-name helpers
# ---------------------------------------------------------------------------

def _clean_column_name(value: object) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _find_column(columns, aliases: List[str]) -> Optional[str]:
    cleaned = {_clean_column_name(c): c for c in columns}
    for alias in aliases:
        key = _clean_column_name(alias)
        if key in cleaned:
            return cleaned[key]
    return None


def _parse_energy_column(value: object) -> Optional[float]:
    """Return float eV if *value* is a numeric string (with optional 'eV')."""
    text = str(value).strip().lower().replace("ev", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------

def read_pics_table(content: bytes, filename: str) -> pd.DataFrame:
    """Parse *content* bytes into a DataFrame based on *filename* extension."""
    suffix = Path(filename).suffix.lower()
    buf = BytesIO(content)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(buf)
    if suffix == ".tsv":
        return pd.read_csv(buf, sep="\t")
    if suffix in {".csv", ".txt", ""}:
        return pd.read_csv(buf)
    raise ValueError(f"不支持的文件格式 {suffix!r}，请使用 CSV、TSV、TXT、XLSX 或 XLS")


# ---------------------------------------------------------------------------
# Table-format parsers
# ---------------------------------------------------------------------------

def _records_from_long_table(df: pd.DataFrame) -> List[Dict[str, Any]]:
    mz_col    = _find_column(df.columns, ["mz", "m/z", "mass", "mass_number", "质量数"])
    name_col  = _find_column(df.columns, ["name", "species", "formula", "molecule", "物种", "名称", "分子式"])
    ie_col    = _find_column(df.columns, ["ionization_energy", "ie", "ionization energy", "电离能", "ie_ev"])
    energy_col = _find_column(
        df.columns, ["energy_ev", "energy", "photon_energy", "photon energy", "e_ev", "光子能量"]
    )
    cross_col = _find_column(
        df.columns,
        ["cross_section", "cross section", "pics", "sigma", "cross", "截面", "光电离截面"],
    )
    if not all([mz_col, name_col, energy_col, cross_col]):
        return []

    cols = [mz_col, name_col, energy_col, cross_col] + ([ie_col] if ie_col else [])
    working = df[cols].copy().rename(
        columns={mz_col: "mz", name_col: "species", energy_col: "energy", cross_col: "cross_section"}
    )
    if ie_col:
        working = working.rename(columns={ie_col: "ie"})
    else:
        working["ie"] = None

    working["mz"]           = pd.to_numeric(working["mz"],           errors="coerce")
    working["energy"]       = pd.to_numeric(working["energy"],       errors="coerce")
    working["cross_section"]= pd.to_numeric(working["cross_section"],errors="coerce")
    working["ie"]           = pd.to_numeric(working["ie"],           errors="coerce")
    working["species"]      = working["species"].astype(str).str.strip()
    working = working.dropna(subset=["mz", "energy", "cross_section"])
    working = working[working["species"] != ""]

    records: List[Dict[str, Any]] = []
    for (mz, species, ie), group in working.groupby(["mz", "species", "ie"], dropna=False):
        points = (
            group.groupby("energy", as_index=False)["cross_section"]
            .mean()
            .sort_values("energy")
        )
        if len(points) < 2:
            continue
        records.append(
            {
                "mz":            int(round(float(mz))),
                "species":       str(species),
                "ie":            None if pd.isna(ie) else float(ie),
                "energies":      points["energy"].astype(float).tolist(),
                "cross_sections":points["cross_section"].astype(float).tolist(),
            }
        )
    return records


def _records_from_wide_table(df: pd.DataFrame) -> List[Dict[str, Any]]:
    mz_col   = _find_column(df.columns, ["mz", "m/z", "mass", "mass_number", "质量数"])
    name_col = _find_column(df.columns, ["name", "species", "formula", "molecule", "物种", "名称", "分子式"])
    ie_col   = _find_column(df.columns, ["ionization_energy", "ie", "ionization energy", "电离能", "ie_ev"])
    if not mz_col or not name_col:
        return []

    meta_cols = {mz_col, name_col}
    if ie_col:
        meta_cols.add(ie_col)

    energy_columns = [
        (col, energy)
        for col in df.columns
        if col not in meta_cols and (energy := _parse_energy_column(col)) is not None
    ]
    if not energy_columns:
        return []

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            mz = int(round(float(row[mz_col])))
        except Exception:
            continue
        species = str(row[name_col]).strip()
        if not species:
            continue
        ie: Optional[float] = None
        if ie_col and not pd.isna(row[ie_col]):
            try:
                ie = float(row[ie_col])
            except Exception:
                ie = None

        energies: list[float] = []
        cross_sections: list[float] = []
        for col, energy in energy_columns:
            val = pd.to_numeric(row[col], errors="coerce")
            if pd.isna(val):
                continue
            energies.append(energy)
            cross_sections.append(float(val))
        if len(energies) < 2:
            continue

        order = sorted(range(len(energies)), key=lambda i: energies[i])
        records.append(
            {
                "mz":             mz,
                "species":        species,
                "ie":             ie,
                "energies":       [energies[i] for i in order],
                "cross_sections": [cross_sections[i] for i in order],
            }
        )
    return records


# ---------------------------------------------------------------------------
# Public parse entry point
# ---------------------------------------------------------------------------

def parse_pics_upload(content: bytes, filename: str) -> List[Dict[str, Any]]:
    """Parse *content* and return a list of species records.

    Each record has keys: mz, species, ie, energies, cross_sections.
    Raises ValueError if the file is empty or the format is unrecognised.
    """
    if not content:
        raise ValueError("上传文件为空")
    df = read_pics_table(content, filename)
    if df.empty:
        raise ValueError("文件中没有数据行")

    # Skip description rows: drop rows where mz column is non-numeric text
    records = _records_from_long_table(df)
    if not records:
        records = _records_from_wide_table(df)
    if not records:
        raise ValueError(
            "无法识别 PICS 表格格式。\n"
            "长表需要列：mz、name、energy_ev、cross_section\n"
            "宽表需要列：mz、name，并以能量值（如 10.5eV）作为后续列名"
        )
    return records


# ---------------------------------------------------------------------------
# Database write
# ---------------------------------------------------------------------------

def _is_finite(value: object) -> bool:
    try:
        return bool(pd.notna(value) and math.isfinite(float(value)))
    except Exception:
        return False


def _matching_species_ids(
    conn: sqlite3.Connection, record: Dict[str, Any]
) -> List[int]:
    """Return IDs of existing species that match mz + name + ie."""
    if record["ie"] is None:
        rows = conn.execute(
            "SELECT id FROM species "
            "WHERE mz = ? AND LOWER(name) = LOWER(?) AND ionization_energy IS NULL",
            (int(record["mz"]), str(record["species"])),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id FROM species "
            "WHERE mz = ? AND LOWER(name) = LOWER(?) AND ABS(ionization_energy - ?) < 1e-6",
            (int(record["mz"]), str(record["species"]), float(record["ie"])),
        ).fetchall()
    return [int(r[0]) for r in rows]


def write_pics_records(
    records: List[Dict[str, Any]],
    database_path: Path,
    *,
    mode: str = "upsert",
    confirm_overwrite: bool = False,
) -> Dict[str, Any]:
    """Write *records* into the SQLite PICS database at *database_path*.

    Parameters
    ----------
    records:
        Parsed species records from ``parse_pics_upload``.
    database_path:
        Writable path to the SQLite database file.
    mode:
        ``"upsert"`` (default), ``"append"``, or ``"overwrite_all"``.
    confirm_overwrite:
        Must be ``True`` when *mode* is ``"overwrite_all"``.

    Returns
    -------
    dict with keys: mode, inserted_species, replaced_species, inserted_points.
    """
    if mode not in {"upsert", "append", "overwrite_all"}:
        mode = "upsert"
    if mode == "overwrite_all" and not confirm_overwrite:
        raise ValueError("overwrite_all 模式需要 confirm_overwrite=True 才能执行")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    inserted_species = 0
    replaced_species = 0
    inserted_points  = 0

    from .pie_analysis import SCHEMA_SQL

    with sqlite3.connect(database_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_SQL)
        if mode == "overwrite_all":
            conn.execute("DELETE FROM pic_cross_sections")
            conn.execute("DELETE FROM species")

        for record in records:
            # Build cross-section rows first to validate before deleting old records
            rows = [
                (float(e), float(cs))
                for e, cs in zip(record["energies"], record["cross_sections"])
                if _is_finite(e) and _is_finite(cs)
            ]
            if len(rows) < 2:
                # Skip records with insufficient valid data points
                continue

            if mode == "upsert":
                ids = _matching_species_ids(conn, record)
                if ids:
                    replaced_species += len(ids)
                    conn.executemany(
                        "DELETE FROM species WHERE id = ?",
                        [(sid,) for sid in ids],
                    )

            cursor = conn.execute(
                "INSERT INTO species (mz, name, ionization_energy) VALUES (?, ?, ?)",
                (int(record["mz"]), str(record["species"]), record["ie"]),
            )
            species_id = int(cursor.lastrowid)

            conn.executemany(
                "INSERT INTO pic_cross_sections (species_id, energy_ev, cross_section) "
                "VALUES (?, ?, ?)",
                [(species_id, e, cs) for e, cs in rows],
            )
            inserted_species += 1
            inserted_points  += len(rows)

        conn.commit()

    return {
        "mode":             mode,
        "inserted_species": inserted_species,
        "replaced_species": replaced_species,
        "inserted_points":  inserted_points,
    }
