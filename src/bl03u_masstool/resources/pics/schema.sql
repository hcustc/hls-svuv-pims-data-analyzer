CREATE TABLE IF NOT EXISTS species (
    id INTEGER PRIMARY KEY,
    mz INTEGER NOT NULL,
    name TEXT NOT NULL,
    ionization_energy REAL,
    formula TEXT,
    elements TEXT,
    smiles TEXT
);

CREATE TABLE IF NOT EXISTS pic_cross_sections (
    id INTEGER PRIMARY KEY,
    species_id INTEGER NOT NULL,
    energy_ev REAL NOT NULL,
    cross_section REAL NOT NULL,
    FOREIGN KEY (species_id) REFERENCES species(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_species_mz ON species(mz);
CREATE INDEX IF NOT EXISTS idx_pics_species_energy ON pic_cross_sections(species_id, energy_ev);
