# Keep Raw Datasets immutable

Imported instrument data is an immutable Raw Dataset with stable identity, source provenance, and content checksums. Software and Agents must never edit or replace it in place; a re-import creates a new data version, and every transformation is stored separately as a Derived Artifact.
