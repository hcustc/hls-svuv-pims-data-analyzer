"""Build the default PICS SQLite database from seed resources."""

from __future__ import annotations

import argparse
from pathlib import Path

from bl03u_masstool.core.config import load_app_config
from bl03u_masstool.core.runtime_paths import ensure_user_copy
from bl03u_masstool.core.species_seed import (
    build_species_database_from_seed,
    default_species_schema_path,
    default_species_seed_path,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="SQLite output path. Defaults to the configured writable PICS database path.",
    )
    parser.add_argument("--seed", type=Path, default=default_species_seed_path(), help="CSV seed file.")
    parser.add_argument("--schema", type=Path, default=default_species_schema_path(), help="SQL schema file.")
    parser.add_argument("--no-overwrite", action="store_true", help="Keep an existing output database.")
    args = parser.parse_args(argv)

    if args.output:
        output = args.output
    else:
        app = load_app_config()
        value = app.get("database", {}).get("pics", "database/species_database.sqlite")
        output = ensure_user_copy(value)
    path = build_species_database_from_seed(
        output,
        seed_path=args.seed,
        schema_path=args.schema,
        overwrite=not args.no_overwrite,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
