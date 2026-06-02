
from core.pie_analysis import load_species_database
from core.elements import filter_species_by_elements

db, idx = load_species_database('database/species_database.sqlite')

print(f'Total species: {len(db)}')

# 测试只包含C, H, O的元素
selected_elements = {'C', 'H', 'O'}
filtered = filter_species_by_elements(db, selected_elements)

print(f'Filtered (C, H, O): {len(filtered)}')

if filtered:
    print('\nSample filtered:')
    for i in range(min(10, len(filtered))):
        print(f'  {filtered[i]["species"]} ({filtered[i]["formula"]}) - {filtered[i]["elements"]}')

