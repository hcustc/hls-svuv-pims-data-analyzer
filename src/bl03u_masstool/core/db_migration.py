"""SQLite schema version management for the PICS species database.

Each schema change increments SCHEMA_VERSION and adds a migration block in
_MIGRATIONS.  On every startup ``ensure_database_up_to_date`` is called by
``species_database_path``; it either builds the database from seed (if missing)
or runs any pending migrations (if the stored user_version is behind).

Adding a migration
------------------
1. Bump ``SCHEMA_VERSION``.
2. Add an entry to ``_MIGRATIONS``::

       SCHEMA_VERSION: 2,
       _MIGRATIONS: {
           ...
           2: \"\"\"ALTER TABLE species ADD COLUMN source TEXT;\"\"\",
       }

The SQL is executed inside a single transaction per version step.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# Increment this whenever the schema or maintained reference data changes and
# add a migration entry below.
SCHEMA_VERSION: int = 3

# Maps target_version -> SQL to bring the database from (target-1) to target.
# Version 1 is the initial schema; it has no migration SQL because a missing
# database is always rebuilt from seed rather than migrated from version 0.
_MIGRATIONS: dict[int, str] = {
    1: "",  # Initial schema – handled by build_species_database_from_seed.
    2: """
        UPDATE species
        SET formula = 'C10H22', smiles = 'CCCCCCCCCC'
        WHERE name COLLATE NOCASE = 'n-Decane';

        UPDATE species
        SET formula = 'C11H10', smiles = 'Cc1cccc2ccccc12'
        WHERE name COLLATE NOCASE = '1-Methylnaphthalene';

        UPDATE species
        SET formula = 'C11H10', smiles = 'Cc1ccc2ccccc2c1'
        WHERE name COLLATE NOCASE = '2-Methylnaphthalene';
    """,
    3: """
        UPDATE species
        SET formula = 'CH3', smiles = '[CH3]'
        WHERE name COLLATE NOCASE = 'Methyl radical';
    """,
}


def get_user_version(conn: sqlite3.Connection) -> int:
    """Return the schema version stored in the database (0 if unset)."""
    return conn.execute("PRAGMA user_version").fetchone()[0]


def set_user_version(conn: sqlite3.Connection, version: int) -> None:
    # PRAGMA user_version does not accept bound parameters.
    conn.execute(f"PRAGMA user_version = {version:d}")


def ensure_database_up_to_date(path: Path) -> None:
    """Guarantee the database at *path* exists and is at SCHEMA_VERSION.

    - If the file does not exist: build from seed and stamp the version.
    - If the stored version equals SCHEMA_VERSION: nothing to do.
    - If the stored version is behind: run migrations sequentially.
    - If the stored version is ahead: raise RuntimeError (app is too old).
    """
    if not path.exists():
        _build_and_stamp(path)
        return

    with sqlite3.connect(path) as conn:
        db_version = get_user_version(conn)

    if db_version == SCHEMA_VERSION:
        return

    if db_version > SCHEMA_VERSION:
        raise RuntimeError(
            f"Database schema version {db_version} is newer than this "
            f"application supports ({SCHEMA_VERSION}). "
            "Please upgrade the application."
        )

    _run_migrations(path, from_version=db_version)


def _build_and_stamp(path: Path) -> None:
    """Build a fresh database from seed and write the version stamp."""
    from .species_seed import build_species_database_from_seed

    build_species_database_from_seed(path)
    # build_species_database_from_seed already stamps the version; nothing else needed.


def _run_migrations(path: Path, from_version: int) -> None:
    """Apply all migrations from *from_version + 1* up to SCHEMA_VERSION."""
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        for version in range(from_version + 1, SCHEMA_VERSION + 1):
            sql = _MIGRATIONS.get(version, "")
            if sql.strip():
                conn.executescript(sql)
            set_user_version(conn, version)
        conn.commit()
