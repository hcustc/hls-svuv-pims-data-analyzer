# Test Fixtures

This directory separates small deterministic fixtures from BL03U instrument
sample data.

- `small/`: tiny text or CSV fixtures suitable for fast unit tests.
- `bl03u_sample/`: a minimal real-data regression sample kept in-repository.

Larger raw datasets should live outside the Git repository, for example as
release assets, DVC remotes, Git LFS assets, or a dedicated data repository.
