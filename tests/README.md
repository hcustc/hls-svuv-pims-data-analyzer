# Test Layout

- `unit/`: fast tests for pure functions, parsers, fitting logic, and small API helpers.
- `integration/`: tests that exercise FastAPI, CLI subprocesses, or real BL03U sample data.
- `fixtures/small/`: tiny deterministic fixtures that should remain easy to review.
- `fixtures/bl03u_sample/`: minimal real-data regression samples.

Do not add large raw experimental datasets directly to this tree. Prefer release
assets, DVC, Git LFS, or a dedicated data repository for full-size datasets.
