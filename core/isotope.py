from __future__ import annotations

import re
from functools import lru_cache
from typing import Mapping


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

MONOISOTOPIC_MASSES = {
    "H": 1.00782503223,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Cl": 34.968852682,
    "Br": 78.9183376,
    "I": 126.9044719,
    "S": 31.9720711744,
    "P": 30.97376199842,
    "Si": 27.97692653465,
    "Fe": 55.93493633,
    "Cu": 62.92959772,
    "Zn": 63.92914201,
    "Mg": 23.985041697,
    "Ca": 39.962590863,
    "K": 38.9637064864,
    "Na": 22.989769282,
    "Li": 7.0160034366,
    "B": 11.00930536,
    "Al": 26.98153853,
    "Ga": 68.9255735,
    "Ge": 73.921177761,
    "As": 74.92159457,
    "Se": 79.9165218,
    "Kr": 83.9114977282,
    "Xe": 131.9041550856,
    "Hg": 201.9706434,
    "Pb": 207.9766525,
    "Sn": 119.90220163,
}

FORMULA_ORDER = ("C", "H")


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


def formula_to_string(composition: Mapping[str, int]) -> str:
    elements = [element for element in FORMULA_ORDER if int(composition.get(element, 0)) > 0]
    elements.extend(
        sorted(element for element, count in composition.items() if element not in FORMULA_ORDER and int(count) > 0)
    )
    parts = []
    for element in elements:
        count = int(composition[element])
        parts.append(element if count == 1 else f"{element}{count}")
    return "".join(parts)


def composition_nominal_mass(composition: Mapping[str, int]) -> int:
    mass = 0
    for element, count in composition.items():
        if element not in ISOTOPES:
            raise ValueError(f"unknown element: {element}")
        nominal_isotope = max(ISOTOPES[element], key=lambda item: item[1])[0]
        mass += int(nominal_isotope) * int(count)
    return mass


def formula_nominal_mass(formula: str) -> int:
    return composition_nominal_mass(parse_formula(formula))


def composition_monoisotopic_mass(composition: Mapping[str, int]) -> float:
    mass = 0.0
    for element, count in composition.items():
        if element not in MONOISOTOPIC_MASSES:
            raise ValueError(f"unknown element: {element}")
        mass += MONOISOTOPIC_MASSES[element] * int(count)
    return float(mass)


def formula_monoisotopic_mass(formula: str) -> float:
    return composition_monoisotopic_mass(parse_formula(formula))


def parse_element_count_ranges(text: str) -> dict[str, tuple[int, int]]:
    """Parse element ranges such as 'C:0-20,H:0-60,O:0-10'."""
    ranges: dict[str, tuple[int, int]] = {}
    for raw_item in re.split(r"[,;]\s*", text.strip()):
        item = raw_item.strip()
        if not item:
            continue
        match = re.fullmatch(r"([A-Z][a-z]?)\s*(?::|=)\s*(\d+)(?:\s*-\s*(\d+))?", item)
        if not match:
            raise ValueError(f"invalid element range: {item}")
        element = match.group(1)
        if element not in MONOISOTOPIC_MASSES:
            raise ValueError(f"unknown element: {element}")
        min_count = int(match.group(2))
        max_count = int(match.group(3) or match.group(2))
        if min_count > max_count:
            raise ValueError(f"invalid element range: {item}")
        ranges[element] = (min_count, max_count)
    if not ranges:
        raise ValueError("at least one element range is required")
    return ranges


def _tolerance_to_da(target_mass: float, tolerance: float, tolerance_unit: str) -> float:
    target_mass = float(target_mass)
    tolerance = float(tolerance)
    if target_mass <= 0:
        raise ValueError("target_mass must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    unit = tolerance_unit.strip().lower()
    if unit in {"da", "u", "amu"}:
        return tolerance
    if unit == "ppm":
        return target_mass * tolerance / 1_000_000
    raise ValueError("tolerance_unit must be 'Da' or 'ppm'")


def generate_formula_candidates(
    target_mass: float,
    *,
    tolerance: float = 0.5,
    tolerance_unit: str = "Da",
    element_ranges: Mapping[str, tuple[int, int]],
    max_results: int = 200,
) -> list[dict]:
    """Generate formula candidates from mass and explicit element count ranges."""
    tolerance_da = _tolerance_to_da(target_mass, tolerance, tolerance_unit)
    lower = float(target_mass) - tolerance_da
    upper = float(target_mass) + tolerance_da
    max_results = max(1, min(int(max_results), 10_000))
    scan_limit = max_results * 20

    elements = [
        (element, int(bounds[0]), int(bounds[1]), MONOISOTOPIC_MASSES[element])
        for element, bounds in element_ranges.items()
    ]
    for element, min_count, max_count, _ in elements:
        if min_count < 0 or max_count < 0 or min_count > max_count:
            raise ValueError(f"invalid element range for {element}")

    suffix_min = [0.0] * (len(elements) + 1)
    suffix_max = [0.0] * (len(elements) + 1)
    for idx in range(len(elements) - 1, -1, -1):
        _, min_count, max_count, mono_mass = elements[idx]
        suffix_min[idx] = suffix_min[idx + 1] + min_count * mono_mass
        suffix_max[idx] = suffix_max[idx + 1] + max_count * mono_mass

    results: list[dict] = []

    def visit(index: int, current: dict[str, int], current_mass: float) -> None:
        if len(results) >= scan_limit:
            return
        if index == len(elements):
            if lower <= current_mass <= upper and any(value > 0 for value in current.values()):
                formula = formula_to_string(current)
                error_da = current_mass - float(target_mass)
                results.append(
                    {
                        "formula": formula,
                        "nominal_mass": composition_nominal_mass(current),
                        "monoisotopic_mass": float(current_mass),
                        "error_da": float(error_da),
                        "error_ppm": float(error_da / float(target_mass) * 1_000_000),
                        "composition": {key: value for key, value in current.items() if value > 0},
                    }
                )
            return

        element, min_count, max_count, mono_mass = elements[index]
        for count in range(min_count, max_count + 1):
            next_mass = current_mass + count * mono_mass
            if next_mass + suffix_min[index + 1] > upper:
                break
            if next_mass + suffix_max[index + 1] < lower:
                continue
            if count > 0:
                current[element] = count
            else:
                current.pop(element, None)
            visit(index + 1, current, next_mass)
        current.pop(element, None)

    visit(0, {}, 0.0)
    return sorted(results, key=lambda item: (abs(item["error_da"]), item["formula"]))[:max_results]


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
