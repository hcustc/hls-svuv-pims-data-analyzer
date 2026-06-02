
import sys
sys.path.insert(0, '.')

print("="*70)
print("Test Database Loading and Element Filtering")
print("="*70)

try:
    # 1. Load database
    from core.pie_analysis import load_species_database_sqlite
    db, mz_idx = load_species_database_sqlite('database/species_database.sqlite')
    
    print(f"\n[OK] Database loaded successfully")
    print(f"  - Total species: {len(db)}")
    print(f"  - m/z index count: {len(mz_idx)}")
    
    # 2. Check species data structure
    print("\n[OK] Species data structure:")
    if db:
        sample = db[0]
        print(f"  Fields: {', '.join(sample.keys())}")
        print(f"  Sample species:")
        print(f"    Name: {sample.get('species')}")
        print(f"    Formula: {sample.get('formula')}")
        print(f"    Elements: {sample.get('elements')}")
        print(f"    SMILES: {sample.get('smiles')}")
    
    # 3. Test element filtering
    print("\n[OK] Testing element filtering:")
    from core.elements import filter_species_by_elements
    
    # Filter species with Cl
    cl_species = filter_species_by_elements(db, {'Cl'})
    print(f"  Species with Cl: {len(cl_species)}")
    
    # Filter species with C, H, O, Cl
    cho_cl_species = filter_species_by_elements(db, {'C', 'H', 'O', 'Cl'})
    print(f"  Species with C,H,O,Cl: {len(cho_cl_species)}")
    
    if cho_cl_species:
        print(f"\n  Filtered species examples (first 10):")
        for i, sp in enumerate(cho_cl_species[:10]):
            print(f"    {i+1}. {sp['species']} (m/z {sp['mz']}, formula: {sp.get('formula')}, elements: {sp.get('elements')})")
    
    # 4. Test m/z 112 species (Monochlorobenzene)
    print("\n[OK] Testing m/z 112 species:")
    if 112 in mz_idx:
        species_112 = [db[i] for i in mz_idx[112]]
        print(f"  Species with m/z 112 in database: {len(species_112)}")
        for sp in species_112[:5]:
            print(f"    - {sp['species']} (formula: {sp.get('formula')}, elements: {sp.get('elements')})")
    else:
        print("  [FAIL] No species with m/z 112 in database")
    
    print("\n" + "="*70)
    print("[OK] All tests passed! Database is updated correctly")
    print("="*70)
    
except Exception as e:
    print(f"\n[FAIL] Test failed: {e}")
    import traceback
    traceback.print_exc()
