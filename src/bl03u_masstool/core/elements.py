import re
from typing import Set

ELEMENT_SYMBOLS = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
}

ELEMENT_PATTERN = re.compile(r'([A-Z][a-z]?)(\d*)')

def parse_formula(formula: str) -> dict[str, int]:
    """Parse a molecular formula and return element counts."""
    if not formula or not isinstance(formula, str):
        return {}
    
    elements = {}
    matches = ELEMENT_PATTERN.findall(formula)
    
    for element, count in matches:
        if element in ELEMENT_SYMBOLS:
            count = int(count) if count else 1
            elements[element] = elements.get(element, 0) + count
    
    return elements

def get_elements_from_formula(formula: str) -> Set[str]:
    """Extract unique elements from a molecular formula."""
    elements_dict = parse_formula(formula)
    return set(elements_dict.keys())

def filter_species_by_elements(species_list: list[dict], selected_elements: Set[str]) -> list[dict]:
    """Filter species by their elemental composition."""
    if not selected_elements:
        return species_list
    
    filtered = []
    for species in species_list:
        formula = species.get('formula', '')
        species_elements = get_elements_from_formula(formula)
        
        if not species_elements:
            filtered.append(species)
            continue
        
        if species_elements <= selected_elements:
            filtered.append(species)
    
    return filtered

def get_all_elements_from_database(species_list: list[dict]) -> Set[str]:
    """Get all unique elements from a species database."""
    all_elements = set()
    for species in species_list:
        formula = species.get('formula', '')
        if formula:
            all_elements.update(get_elements_from_formula(formula))
    return sorted(all_elements)

COMMON_ELEMENTS = ['H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si']
