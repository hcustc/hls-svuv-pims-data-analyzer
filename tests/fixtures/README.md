# Test Fixtures

This directory separates small deterministic fixtures from BL03U instrument
sample data.

- `small/`: tiny text or CSV fixtures suitable for fast unit tests.
- `bl03u_sample/`: optional local real-data regression sample ignored by Git.

Real-data integration tests look for `bl03u_sample/` here by default and skip
when it is absent. You can also set `BL03U_REAL_DATA_DIR=/path/to/bl03u_sample`
to keep the data elsewhere.

Larger raw datasets should live outside the Git repository, for example as
release assets, DVC remotes, Git LFS assets, object storage, an internal NAS, or
a dedicated data repository with checksums/manifests.
