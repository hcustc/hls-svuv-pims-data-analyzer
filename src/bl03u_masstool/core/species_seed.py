"""Build the default PICS SQLite database from reviewable seed files."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from .runtime_paths import resource_path


DEFAULT_SCHEMA_PATH = Path("resources/pics/schema.sql")
DEFAULT_SEED_PATH = Path("resources/pics/species_seed.csv")


def default_species_schema_path() -> Path:
    return resource_path(DEFAULT_SCHEMA_PATH)


def default_species_seed_path() -> Path:
    return resource_path(DEFAULT_SEED_PATH)


def build_species_database_from_seed(
    output_path: str | Path,
    *,
    seed_path: str | Path | None = None,
    schema_path: str | Path | None = None,
    overwrite: bool = True,
) -> Path:
    """Create a normalized SQLite PICS database from the CSV seed."""

    output = Path(output_path)
    seed = Path(seed_path) if seed_path is not None else default_species_seed_path()
    schema = Path(schema_path) if schema_path is not None else default_species_schema_path()

    if not seed.exists():
        raise FileNotFoundError(f"PICS seed file not found: {seed}")
    if not schema.exists():
        raise FileNotFoundError(f"PICS schema file not found: {schema}")
    if output.exists() and not overwrite:
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    seen_species: set[int] = set()
    with sqlite3.connect(output) as conn, seed.open("r", encoding="utf-8", newline="") as handle:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(schema.read_text(encoding="utf-8"))
        reader = csv.DictReader(handle)
        for row in reader:
            species_id = int(row["species_id"])
            if species_id not in seen_species:
                conn.execute(
                    """
                    INSERT INTO species (id, mz, name, ionization_energy, formula, elements, smiles)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        species_id,
                        int(row["mz"]),
                        row["name"],
                        _optional_float(row["ionization_energy"]),
                        _optional_text(row["formula"]),
                        _optional_text(row["elements"]),
                        _optional_text(row["smiles"]),
                    ),
                )
                seen_species.add(species_id)
            conn.execute(
                """
                INSERT INTO pic_cross_sections (species_id, energy_ev, cross_section)
                VALUES (?, ?, ?)
                """,
                (species_id, float(row["energy_ev"]), float(row["cross_section"])),
            )
        conn.commit()
    return output


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _optional_float(value: str | None) -> float | None:
    text = _optional_text(value)
    return None if text is None else float(text)
