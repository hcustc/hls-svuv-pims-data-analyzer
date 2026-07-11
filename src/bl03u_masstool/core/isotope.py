from __future__ import annotations

from dataclasses import dataclass
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

ISOTOPE_EXACT_MASSES = {
    ("H", 1): 1.00782503223,
    ("H", 2): 2.01410177812,
    ("C", 12): 12.0,
    ("C", 13): 13.00335483507,
    ("N", 14): 14.00307400443,
    ("N", 15): 15.00010889888,
    ("O", 16): 15.99491461957,
    ("O", 17): 16.9991317565,
    ("O", 18): 17.99915961286,
    ("S", 32): 31.9720711744,
    ("S", 33): 32.9714589098,
    ("S", 34): 33.967867004,
    ("S", 36): 35.96708071,
    ("Cl", 35): 34.968852682,
    ("Cl", 37): 36.965902602,
    ("Br", 79): 78.9183376,
    ("Br", 81): 80.9162897,
}

FORMULA_ORDER = ("C", "H")
HYDRATE_SEPARATORS = {"·", "•", "."}


@dataclass(frozen=True)
class ParsedFormula:
    composition: dict[str, int]
    monoisotopic_mass: float
    nominal_mass: int


def _merge_composition(target: dict[str, int], source: Mapping[str, int], multiplier: int = 1) -> None:
    for element, count in source.items():
        value = int(count) * int(multiplier)
        if value:
            target[element] = target.get(element, 0) + value


def _strip_charge_suffix(formula: str) -> str:
    text = formula.strip()
    for pattern in (r"\^\d*[+-]$", r"\^[+-]\d*$", r"[+-]\d+$", r"[+-]$"):
        cleaned = re.sub(pattern, "", text)
        if cleaned != text:
            return cleaned
    return text


def _split_formula_segments(formula: str) -> list[str]:
    segments: list[str] = []
    start = 0
    depth = 0
    bracket_depth = 0
    for index, char in enumerate(formula):
        if char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif bracket_depth == 0 and char == "(":
            depth += 1
        elif bracket_depth == 0 and char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unmatched ')' in molecular formula")
        elif depth == 0 and bracket_depth == 0 and char in HYDRATE_SEPARATORS:
            segment = formula[start:index]
            if not segment:
                raise ValueError("empty hydrate segment in molecular formula")
            segments.append(segment)
            start = index + 1
    if depth != 0:
        raise ValueError("unmatched '(' in molecular formula")
    if bracket_depth != 0:
        raise ValueError("unmatched '[' in molecular formula")
    tail = formula[start:]
    if not tail:
        raise ValueError("empty hydrate segment in molecular formula")
    segments.append(tail)
    return segments


class _FormulaParser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def parse(self) -> ParsedFormula:
        parsed = self._parse_group(stop_char=None)
        if self.pos != len(self.text):
            raise ValueError(f"invalid molecular formula near: {self.text[self.pos:]}")
        if not parsed.composition:
            raise ValueError("invalid molecular formula")
        return parsed

    def _parse_group(self, stop_char: str | None) -> ParsedFormula:
        composition: dict[str, int] = {}
        monoisotopic_mass = 0.0
        nominal_mass = 0
        while self.pos < len(self.text):
            char = self.text[self.pos]
            if stop_char and char == stop_char:
                break
            if char == "(":
                self.pos += 1
                inner = self._parse_group(stop_char=")")
                if self.pos >= len(self.text) or self.text[self.pos] != ")":
                    raise ValueError("unmatched '(' in molecular formula")
                self.pos += 1
                count = self._parse_count()
                _merge_composition(composition, inner.composition, count)
                monoisotopic_mass += inner.monoisotopic_mass * count
                nominal_mass += inner.nominal_mass * count
                continue
            if char == "[":
                element, mass_number = self._parse_isotope_label()
                count = self._parse_count()
                composition[element] = composition.get(element, 0) + count
                monoisotopic_mass += _isotope_exact_mass(element, mass_number) * count
                nominal_mass += int(mass_number) * count
                continue
            if char.isupper():
                element = self._parse_element()
                count = self._parse_count()
                composition[element] = composition.get(element, 0) + count
                monoisotopic_mass += MONOISOTOPIC_MASSES[element] * count
                nominal_isotope = max(ISOTOPES[element], key=lambda item: item[1])[0]
                nominal_mass += int(nominal_isotope) * count
                continue
            raise ValueError(f"invalid molecular formula near: {self.text[self.pos:]}")
        return ParsedFormula(composition, float(monoisotopic_mass), int(nominal_mass))

    def _parse_count(self) -> int:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos].isdigit():
            self.pos += 1
        if start == self.pos:
            return 1
        count = int(self.text[start:self.pos])
        if count <= 0:
            raise ValueError("formula counts must be positive integers")
        return count

    def _parse_element(self) -> str:
        start = self.pos
        self.pos += 1
        if self.pos < len(self.text) and self.text[self.pos].islower():
            self.pos += 1
        element = self.text[start:self.pos]
        if element not in ISOTOPES:
            raise ValueError(f"unknown element: {element}")
        return element

    def _parse_isotope_label(self) -> tuple[str, int]:
        match = re.match(r"\[(\d+)([A-Z][a-z]?)\]", self.text[self.pos :])
        if not match:
            raise ValueError(f"invalid isotope label near: {self.text[self.pos:]}")
        mass_number = int(match.group(1))
        element = match.group(2)
        if element not in ISOTOPES:
            raise ValueError(f"unknown element: {element}")
        self.pos += len(match.group(0))
        return element, mass_number


def _isotope_exact_mass(element: str, mass_number: int) -> float:
    return float(ISOTOPE_EXACT_MASSES.get((element, int(mass_number)), mass_number))


def _parse_segment_multiplier(segment: str) -> tuple[int, str]:
    match = re.match(r"(\d+)(?=[A-Z(\[])", segment)
    if not match:
        return 1, segment
    multiplier = int(match.group(1))
    if multiplier <= 0:
        raise ValueError("hydrate segment multipliers must be positive integers")
    return multiplier, segment[match.end() :]


def _parse_formula_detail(formula: str) -> ParsedFormula:
    text = _strip_charge_suffix(re.sub(r"\s+", "", formula or ""))
    if not text:
        raise ValueError("invalid molecular formula")
    composition: dict[str, int] = {}
    monoisotopic_mass = 0.0
    nominal_mass = 0
    for segment in _split_formula_segments(text):
        multiplier, body = _parse_segment_multiplier(segment)
        if not body:
            raise ValueError("empty hydrate segment in molecular formula")
        parsed = _FormulaParser(body).parse()
        _merge_composition(composition, parsed.composition, multiplier)
        monoisotopic_mass += parsed.monoisotopic_mass * multiplier
        nominal_mass += parsed.nominal_mass * multiplier
    return ParsedFormula(composition, float(monoisotopic_mass), int(nominal_mass))


def parse_formula(formula: str) -> dict[str, int]:
    return dict(_parse_formula_detail(formula).composition)


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
    return _parse_formula_detail(formula).nominal_mass


def composition_monoisotopic_mass(composition: Mapping[str, int]) -> float:
    mass = 0.0
    for element, count in composition.items():
        if element not in MONOISOTOPIC_MASSES:
            raise ValueError(f"unknown element: {element}")
        mass += MONOISOTOPIC_MASSES[element] * int(count)
    return float(mass)


def formula_monoisotopic_mass(formula: str) -> float:
    return _parse_formula_detail(formula).monoisotopic_mass


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


def formula_double_bond_equivalent(composition: Mapping[str, int]) -> float | None:
    """Return the basic CHN/halogen double-bond equivalent when applicable."""
    supported = {"C", "H", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
    if any(element not in supported for element, count in composition.items() if int(count) > 0):
        return None
    carbon = int(composition.get("C", 0))
    hydrogen = int(composition.get("H", 0))
    nitrogen = int(composition.get("N", 0))
    halogens = sum(int(composition.get(element, 0)) for element in ("F", "Cl", "Br", "I"))
    return float((2 * carbon + 2 + nitrogen - hydrogen - halogens) / 2.0)


def _passes_basic_formula_rules(composition: Mapping[str, int]) -> bool:
    dbe = formula_double_bond_equivalent(composition)
    return dbe is None or dbe >= 0


def generate_formula_candidates(
    target_mass: float,
    *,
    tolerance: float = 0.5,
    tolerance_unit: str = "Da",
    element_ranges: Mapping[str, tuple[int, int]],
    max_results: int = 200,
    mass_mode: str = "monoisotopic",
    apply_chemical_rules: bool = False,
) -> list[dict]:
    """Generate formula candidates from nominal or monoisotopic mass.

    ``mass_mode="nominal"`` is intended for an integer mass number from a
    unit-mass spectrum.  The default ``"monoisotopic"`` mode preserves the
    historical exact-mass search behavior.
    """
    mass_mode = str(mass_mode or "monoisotopic").strip().lower()
    if mass_mode not in {"nominal", "monoisotopic"}:
        raise ValueError("mass_mode must be 'nominal' or 'monoisotopic'")
    tolerance_da = _tolerance_to_da(target_mass, tolerance, tolerance_unit)
    lower = float(target_mass) - tolerance_da
    upper = float(target_mass) + tolerance_da
    max_results = max(1, min(int(max_results), 10_000))
    scan_limit = max_results * 20

    elements = []
    for element, bounds in element_ranges.items():
        search_mass = (
            float(max(ISOTOPES[element], key=lambda item: item[1])[0])
            if mass_mode == "nominal"
            else MONOISOTOPIC_MASSES[element]
        )
        elements.append((element, int(bounds[0]), int(bounds[1]), search_mass))
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
                if apply_chemical_rules and not _passes_basic_formula_rules(current):
                    return
                formula = formula_to_string(current)
                nominal_mass = composition_nominal_mass(current)
                monoisotopic_mass = composition_monoisotopic_mass(current)
                dbe = formula_double_bond_equivalent(current)
                error_da = current_mass - float(target_mass)
                results.append(
                    {
                        "formula": formula,
                        "nominal_mass": nominal_mass,
                        "monoisotopic_mass": monoisotopic_mass,
                        "matched_mass": float(current_mass),
                        "mass_mode": mass_mode,
                        "dbe": dbe,
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
