# Test Layout

- `unit/`: fast tests for pure functions, parsers, fitting logic, and small API helpers.
- `integration/`: tests that exercise FastAPI, CLI subprocesses, or real BL03U sample data.
- `fixtures/small/`: tiny deterministic fixtures that should remain easy to review.
- `fixtures/bl03u_sample/`: optional local real-data regression samples, ignored by Git.

Do not add large raw experimental datasets directly to this tree. Keep only tiny,
deterministic fixtures in Git. Real-data integration tests are marked
`real_data` and skip automatically when the local data directory is absent.

To run real-data regression tests locally, either keep a local copy at
`tests/fixtures/bl03u_sample/` or point pytest to another location:

```bash
BL03U_REAL_DATA_DIR=/path/to/bl03u_sample python -m pytest -m real_data
```

For shared large datasets, prefer release assets, DVC, Git LFS, object storage,
an internal NAS, or a dedicated data repository with checksums/manifests.
