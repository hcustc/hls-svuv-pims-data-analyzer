from __future__ import annotations

import re
from functools import lru_cache


ISOTOPES = {
    "H": [(1, 0.999885), (2, 0.000115)],
    "C": [(12, 0.9893), (13, 0.0107)],
    "N": [(14, 0.99632), (15, 0.00368)],
    "O": [(16, 0.99757), (17, 0.00038), (18, 0.00205)],
    "F": [(19, 1.0)],
    "Cl": [(35, 0.7576), (37, 0.2424)],
    "Br": [(79, 0.5069), (81, 0.4931)],
    "I": [(127, 1.0)],
    "S": [(32, 0.9502), (33, 0.0075), (34, 0.0421), (36, 0.0002)],
    "P": [(31, 1.0)],
    "Si": [(28, 0.9223), (29, 0.0467), (30, 0.0310)],
    "Fe": [(54, 0.05845), (56, 0.91754), (57, 0.02119), (58, 0.00282)],
    "Cu": [(63, 0.6917), (65, 0.3083)],
    "Zn": [(64, 0.4863), (66, 0.2790), (67, 0.0410), (68, 0.1875), (70, 0.0062)],
    "Mg": [(24, 0.7899), (25, 0.1000), (26, 0.1101)],
    "Ca": [(40, 0.96941), (42, 0.00647), (43, 0.00135), (44, 0.02086), (46, 0.00004), (48, 0.00187)],
    "K": [(39, 0.932581), (40, 0.000117), (41, 0.067302)],
    "Na": [(23, 1.0)],
    "Li": [(6, 0.0759), (7, 0.9241)],
    "B": [(10, 0.199), (11, 0.801)],
    "Al": [(27, 1.0)],
    "Ga": [(69, 0.60108), (71, 0.39892)],
    "Ge": [(70, 0.2033), (72, 0.2732), (73, 0.0776), (74, 0.3673), (76, 0.0786)],
    "As": [(75, 1.0)],
    "Se": [(74, 0.0089), (76, 0.0937), (77, 0.0763), (78, 0.2378), (80, 0.4961), (82, 0.0872)],
    "Kr": [(78, 0.0035), (80, 0.0228), (82, 0.1159), (83, 0.1150), (84, 0.5690), (86, 0.1738)],
    "Xe": [(124, 0.0009), (126, 0.0009), (128, 0.0192), (129, 0.2644), (130, 0.0408), (131, 0.2118), (132, 0.2689), (134, 0.1044), (136, 0.0887)],
    "Hg": [(196, 0.0015), (198, 0.0100), (199, 0.1687), (200, 0.2310), (201, 0.1318), (202, 0.2986), (204, 0.1614)],
    "Pb": [(204, 0.014), (206, 0.241), (207, 0.221), (208, 0.524)],
    "Sn": [(112, 0.0097), (114, 0.0066), (115, 0.0034), (116, 0.1454), (117, 0.0768), (118, 0.2422), (119, 0.0859), (120, 0.3258), (122, 0.0463), (124, 0.0564)],
}


def parse_formula(formula: str) -> dict[str, int]:
    matches = re.findall(r"([A-Z][a-z]?)(\d*)", formula.strip())
    if not matches:
        raise ValueError("invalid molecular formula")
    composition: dict[str, int] = {}
    for element, count_text in matches:
        if element not in ISOTOPES:
            raise ValueError(f"unknown element: {element}")
        composition[element] = composition.get(element, 0) + (int(count_text) if count_text else 1)
    return composition


@lru_cache(maxsize=256)
def _element_distribution(element: str, count: int) -> tuple[tuple[float, float], ...]:
    distribution = {0.0: 1.0}
    for _ in range(count):
        next_distribution = {}
        for current_mass, current_abundance in distribution.items():
            for isotope_mass, isotope_abundance in ISOTOPES[element]:
                mass = current_mass + isotope_mass
                next_distribution[mass] = next_distribution.get(mass, 0.0) + current_abundance * isotope_abundance
        distribution = next_distribution
    return tuple(sorted(distribution.items()))


def calculate_isotope_distribution(formula: str, *, min_percent: float = 0.01) -> list[dict]:
    composition = parse_formula(formula)
    distribution = {0.0: 1.0}
    for element, count in composition.items():
        element_dist = dict(_element_distribution(element, count))
        next_distribution = {}
        for mass_a, abundance_a in distribution.items():
            for mass_b, abundance_b in element_dist.items():
                mass = mass_a + mass_b
                next_distribution[mass] = next_distribution.get(mass, 0.0) + abundance_a * abundance_b
        distribution = next_distribution

    total = sum(distribution.values())
    rows = []
    for mass, abundance in sorted(distribution.items()):
        percent = abundance / total * 100 if total else 0.0
        if percent >= min_percent:
            rows.append({"mass": float(mass), "abundance": float(abundance), "percent": float(percent)})
    return rows

