from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re
import socket
import sqlite3
import time
from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlparse

from .config import species_database_path
from .isotope import ISOTOPES, parse_formula

try:
    from bs4 import BeautifulSoup, Comment
except Exception:  # pragma: no cover - dependency error is raised at runtime
    BeautifulSoup = None
    Comment = None

try:
    import requests
    from urllib3.util import connection as urllib3_connection
except Exception:  # pragma: no cover - dependency error is raised at runtime
    requests = None
    urllib3_connection = None


NIST_BASE_URL = "https://webbook.nist.gov"
NIST_SEARCH_URL = f"{NIST_BASE_URL}/cgi/cbook.cgi"
KJ_PER_MOL_PER_EV = 96.48533212331002


@dataclass(frozen=True)
class NistIonizationEnergy:
    value: float
    uncertainty: float | None = None
    units: str = "eV"
    method: str | None = None
    reference: str | None = None
    reference_url: str | None = None
    comment: str | None = None
    quantity: str | None = None
    source: str = "webbook"

    def formatted_value(self) -> str:
        if self.uncertainty is None:
            return f"{self.value:.4f} {self.units}"
        return f"{self.value:.4f} ± {self.uncertainty:g} {self.units}"


@dataclass(frozen=True)
class NistCompoundIonization:
    nist_id: str | None
    name: str | None
    formula: str | None
    cas_rn: str | None
    url: str | None
    ion_energetics_url: str | None
    evaluated_ie: NistIonizationEnergy | None = None
    determinations: tuple[NistIonizationEnergy, ...] = ()
    message: str = ""

    @property
    def best_ie(self) -> NistIonizationEnergy | None:
        if self.evaluated_ie is not None:
            return self.evaluated_ie
        if self.determinations:
            return self.determinations[0]
        return None


@dataclass(frozen=True)
class NistWebBookResult:
    query: str
    search_type: str
    requested_url: str
    compounds: tuple[NistCompoundIonization, ...] = ()
    selected_compound: NistCompoundIonization | None = None
    lost: bool = False
    message: str = ""

    @property
    def best_ie(self) -> NistIonizationEnergy | None:
        if self.selected_compound is None:
            return None
        return self.selected_compound.best_ie


@dataclass(frozen=True)
class LocalIonizationEnergyMatch:
    species_id: int
    name: str
    mz: int
    ionization_energy_ev: float


def _require_bs4() -> None:
    if BeautifulSoup is None:
        raise RuntimeError("NIST WebBook 查询需要 beautifulsoup4，请先安装该依赖。")


def _require_requests() -> None:
    if requests is None:
        raise RuntimeError("NIST WebBook 查询需要 requests，请先安装该依赖。")


def _fix_nist_html(html: str) -> str:
    return html.replace("clss=", "class=").replace("\xa0", " ")


def _clean_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _absolute_url(href: str | None) -> str | None:
    if not href:
        return None
    return urljoin(NIST_BASE_URL, href)


def _query_param(url: str | None, key: str) -> str | None:
    if not url:
        return None
    values = parse_qs(urlparse(url).query).get(key)
    return values[0] if values else None


def _parse_value_uncertainty(text: str | None) -> tuple[float | None, float | None]:
    if not text:
        return None, None
    normalized = _clean_text(text).replace("+/-", "±").replace("\u00b1", "±")
    value_match = re.search(r"[-+]?\d+(?:\.\d+)?", normalized)
    value = float(value_match.group(0)) if value_match else None
    uncertainty_match = re.search(r"±\s*([-+]?\d+(?:\.\d+)?)", normalized)
    uncertainty = float(uncertainty_match.group(1)) if uncertainty_match else None
    return value, uncertainty


def _is_ionization_energy_quantity(quantity: str | None) -> bool:
    normalized = _clean_text(quantity).lower()
    return bool(re.search(r"\bie\b", normalized)) or "ionization energy" in normalized


def _to_electron_volts(row: NistIonizationEnergy) -> NistIonizationEnergy:
    units = _clean_text(row.units).lower()
    if units in {"ev", "electron volt", "electron volts"}:
        return row
    if units in {"kj/mol", "kj mol-1", "kj mol^-1", "kj mol⁻¹", "kilojoule/mol", "kilojoules/mol"}:
        comment = row.comment or ""
        suffix = f"converted from {row.value:g} {row.units}"
        comment = f"{comment}; {suffix}" if comment else suffix
        return replace(
            row,
            value=row.value / KJ_PER_MOL_PER_EV,
            uncertainty=None if row.uncertainty is None else row.uncertainty / KJ_PER_MOL_PER_EV,
            units="eV",
            comment=comment,
        )
    return row


def infer_nist_search_type(query: str) -> str:
    normalized = query.strip()
    if re.fullmatch(r"\d{2,7}-\d{2}-\d", normalized):
        return "id"
    if re.fullmatch(r"C\d{2,}", normalized, re.IGNORECASE):
        return "id"
    if normalized.startswith("InChI="):
        return "inchi"
    compact = normalized.replace(" ", "")
    if re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", compact):
        return "formula"
    return "name"


def build_nist_search_url(query: str, search_type: str = "auto") -> str:
    resolved_type = infer_nist_search_type(query) if search_type == "auto" else search_type
    if resolved_type == "inchi":
        return f"{NIST_BASE_URL}/cgi/inchi/{quote(query.strip(), safe='')}"
    param_key = {
        "formula": "Formula",
        "name": "Name",
        "id": "ID",
    }.get(resolved_type)
    if param_key is None:
        raise ValueError(f"不支持的 NIST WebBook 查询类型: {search_type}")
    params = {param_key: query.strip(), "Units": "SI", "cIE": "on"}
    if resolved_type == "formula":
        params["NoIon"] = "on"
    return f"{NIST_SEARCH_URL}?{urlencode(params)}"


def lookup_local_ionization_energy(
    query: str,
    *,
    db_path: str | Path | None = None,
    allow_fuzzy: bool = False,
) -> LocalIonizationEnergyMatch | None:
    path = Path(db_path) if db_path is not None else species_database_path()
    if not path.exists():
        return None
    normalized = query.strip()
    if not normalized:
        return None
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        row = conn.execute(
            """
            SELECT id, name, mz, ionization_energy
            FROM species
            WHERE lower(name) = lower(?)
              AND ionization_energy IS NOT NULL
            ORDER BY id
            LIMIT 1
            """,
            (normalized,),
        ).fetchone()
        if row is None and allow_fuzzy:
            row = conn.execute(
                """
                SELECT id, name, mz, ionization_energy
                FROM species
                WHERE lower(name) LIKE lower(?)
                  AND ionization_energy IS NOT NULL
                ORDER BY length(name), id
                LIMIT 1
                """,
                (f"%{normalized}%",),
            ).fetchone()
    if row is None:
        return None
    return LocalIonizationEnergyMatch(
        species_id=int(row[0]),
        name=str(row[1]),
        mz=int(row[2]),
        ionization_energy_ev=float(row[3]),
    )


def nominal_mass_from_formula(formula: str) -> int:
    composition = parse_formula(formula)
    mass = 0
    for element, count in composition.items():
        nominal_isotope = max(ISOTOPES[element], key=lambda item: item[1])[0]
        mass += int(nominal_isotope) * int(count)
    return mass


def lookup_local_ionization_candidates_by_mz(
    mz: int,
    *,
    db_path: str | Path | None = None,
    limit: int = 25,
) -> tuple[LocalIonizationEnergyMatch, ...]:
    path = Path(db_path) if db_path is not None else species_database_path()
    if not path.exists():
        return tuple()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            """
            SELECT id, name, mz, ionization_energy
            FROM species
            WHERE mz = ?
              AND ionization_energy IS NOT NULL
            ORDER BY id
            LIMIT ?
            """,
            (int(mz), int(limit)),
        ).fetchall()
    return tuple(
        LocalIonizationEnergyMatch(
            species_id=int(row[0]),
            name=str(row[1]),
            mz=int(row[2]),
            ionization_energy_ev=float(row[3]),
        )
        for row in rows
    )


def local_match_to_compound(
    match: LocalIonizationEnergyMatch,
    *,
    source: str = "local_database",
    message: str = "本地物种数据库命中。",
) -> NistCompoundIonization:
    ie = NistIonizationEnergy(
        value=match.ionization_energy_ev,
        units="eV",
        method="local_database",
        reference="species_database.sqlite",
        comment=f"m/z={match.mz}; species id={match.species_id}",
        quantity="IE",
        source=source,
    )
    return NistCompoundIonization(
        nist_id=f"local:{match.species_id}",
        name=match.name,
        formula=None,
        cas_rn=None,
        url=None,
        ion_energetics_url=None,
        evaluated_ie=ie,
        determinations=(),
        message=message,
    )


def local_match_to_result(query: str, match: LocalIonizationEnergyMatch) -> NistWebBookResult:
    compound = local_match_to_compound(match, message="本地物种数据库命中，未访问 NIST WebBook。")
    return NistWebBookResult(
        query=query.strip(),
        search_type="local_database",
        requested_url="local:species_database.sqlite",
        compounds=(compound,),
        selected_compound=compound,
        message=f"本地物种数据库命中: {match.name}，未访问 NIST WebBook。",
    )


def _is_compound_page(soup: Any) -> bool:
    header = soup.find("h1", id="Top")
    return bool(header and header.find_next("ul"))


def _compound_id_from_comment(soup: Any) -> str | None:
    if Comment is None:
        return None
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        match = re.search(r"/cgi/.*\?(?:Form|ID)=([^&]+)&", str(comment).replace("\n", ""))
        if match:
            return match.group(1)
    return None


def _extract_data_refs(soup: Any) -> dict[str, str]:
    mask_map = {
        "1": "cTG",
        "2": "cTC",
        "4": "cTP",
        "8": "cTR",
        "10": "cSO",
        "20": "cIE",
        "40": "cIC",
        "80": "cIR",
        "100": "cTZ",
        "200": "cMS",
        "400": "cUV",
        "800": "cES",
        "1000": "cDI",
        "2000": "cGC",
    }
    refs: dict[str, str] = {}
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        mask = re.search(r"Mask=(\d+)", href)
        text = _clean_text(anchor.get_text(" ", strip=True))
        key = mask_map.get(mask.group(1), text) if mask else text
        if key == "cIE" or "ion energetics" in text.lower():
            refs["cIE"] = _absolute_url(href) or href
    return refs


def parse_compound_page(html: str, *, page_url: str | None = None, fallback_id: str | None = None) -> dict[str, Any]:
    _require_bs4()
    soup = BeautifulSoup(html, "html.parser")
    if not _is_compound_page(soup):
        raise ValueError("NIST WebBook 返回页不是单一化合物页面。")

    header = soup.find("h1", id="Top")
    info = header.find_next("ul")
    page_id = _query_param(page_url, "ID")
    nist_id = page_id or fallback_id or _compound_id_from_comment(soup)
    if nist_id is None:
        for anchor in info.find_all("a", href=True):
            nist_id = _query_param(anchor["href"], "ID")
            if nist_id:
                break

    formula = None
    cas_rn = None
    for item in info.find_all("li"):
        text = _clean_text(item.get_text(" ", strip=True))
        if text.startswith("Formula:"):
            formula = re.sub(r"\bMonomer\b", "", text.removeprefix("Formula:")).strip()
        elif text.startswith("CAS Registry Number:"):
            cas_rn = text.removeprefix("CAS Registry Number:").strip()

    data_refs = _extract_data_refs(soup)
    return {
        "nist_id": nist_id,
        "name": _clean_text(header.get_text(" ", strip=True)) or None,
        "formula": formula or None,
        "cas_rn": cas_rn or None,
        "data_refs": data_refs,
        "url": page_url or (f"{NIST_SEARCH_URL}?ID={quote(nist_id)}&Units=SI" if nist_id else None),
        "soup": soup,
    }


def parse_search_result_ids(html: str) -> tuple[tuple[str, ...], bool]:
    _require_bs4()
    soup = BeautifulSoup(html, "html.parser")
    if _is_compound_page(soup):
        info = parse_compound_page(html)
        nist_id = info.get("nist_id")
        return ((nist_id,) if nist_id else tuple()), False

    ids: list[str] = []
    seen: set[str] = set()
    result_list = soup.find("ol")
    anchors = result_list.find_all("a", href=True) if result_list else soup.find_all("a", href=True)
    for anchor in anchors:
        href = anchor["href"]
        if "/cgi/cbook.cgi" not in href and "cbook.cgi" not in href:
            continue
        nist_id = _query_param(href, "ID")
        if nist_id and nist_id not in seen:
            ids.append(nist_id)
            seen.add(nist_id)
    lost = "due to the large number of matching species" in soup.get_text(" ", strip=True).lower()
    return tuple(ids), lost


def parse_ionization_energy_summary(html: str) -> tuple[NistIonizationEnergy, ...]:
    _require_bs4()
    soup = BeautifulSoup(html, "html.parser")
    header = soup.find("h2", id="Ion-Energetics")
    if header is None:
        header = soup.find("h2", string=re.compile("Gas phase ion energetics", re.IGNORECASE))
    table = header.find_next("table", class_=re.compile("data")) if header else None
    if table is None:
        return tuple()

    rows: list[NistIonizationEnergy] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        quantity = _clean_text(cells[0].get_text(" ", strip=True))
        raw_value = _clean_text(cells[1].get_text(" ", strip=True))
        value, uncertainty = _parse_value_uncertainty(raw_value)
        if value is None:
            continue
        units = _clean_text(cells[2].get_text(" ", strip=True)) if len(cells) > 2 else "eV"
        method = _clean_text(cells[3].get_text(" ", strip=True)) if len(cells) > 3 else None
        reference = None
        reference_url = None
        if len(cells) > 4:
            ref_anchor = cells[4].find("a", href=True)
            reference = _clean_text(ref_anchor.get_text(" ", strip=True)) if ref_anchor else _clean_text(cells[4].get_text(" ", strip=True))
            reference_url = _absolute_url(ref_anchor["href"]) if ref_anchor else None
        comment = _clean_text(cells[5].get_text(" ", strip=True)) if len(cells) > 5 else None
        rows.append(
            NistIonizationEnergy(
                value=value,
                uncertainty=uncertainty,
                units=units or "eV",
                method=method or None,
                reference=reference or None,
                reference_url=reference_url,
                comment=comment or None,
                quantity=quantity or None,
                source="summary",
            )
        )
    return tuple(rows)


def parse_ionization_energy_determinations(html: str) -> tuple[NistIonizationEnergy, ...]:
    _require_bs4()
    soup = BeautifulSoup(html, "html.parser")
    header = soup.find("h3", string=re.compile("Ionization energy determinations", re.IGNORECASE))
    table = header.find_next("table", class_=re.compile("data")) if header else None
    if table is None:
        return tuple()

    rows: list[NistIonizationEnergy] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        value, uncertainty = _parse_value_uncertainty(cells[0].get_text(" ", strip=True))
        if value is None:
            continue
        method_anchor = cells[1].find("a", href=True)
        method = _clean_text(method_anchor.get_text(" ", strip=True)) if method_anchor else _clean_text(cells[1].get_text(" ", strip=True))
        reference = None
        reference_url = None
        if len(cells) > 2:
            ref_anchor = cells[2].find("a", href=True)
            reference = _clean_text(ref_anchor.get_text(" ", strip=True)) if ref_anchor else _clean_text(cells[2].get_text(" ", strip=True))
            reference_url = _absolute_url(ref_anchor["href"]) if ref_anchor else None
        comment = _clean_text(cells[3].get_text(" ", strip=True)) if len(cells) > 3 else None
        rows.append(
            NistIonizationEnergy(
                value=value,
                uncertainty=uncertainty,
                units="eV",
                method=method or None,
                reference=reference or None,
                reference_url=reference_url,
                comment=comment or None,
                quantity="IE",
                source="determination",
            )
        )
    return tuple(rows)


def pick_evaluated_ie(summary_rows: tuple[NistIonizationEnergy, ...]) -> NistIonizationEnergy | None:
    for row in summary_rows:
        quantity = (row.quantity or "").lower()
        if _is_ionization_energy_quantity(row.quantity) and "evaluated" in quantity:
            return _to_electron_volts(replace(row, source="evaluated"))
    for row in summary_rows:
        if _is_ionization_energy_quantity(row.quantity):
            return _to_electron_volts(row)
    return None


class NistWebBookClient:
    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_candidates: int = 10,
        retries: int = 3,
        retry_delay: float = 1.5,
        force_ipv4: bool = True,
        local_first: bool = True,
        local_db_path: str | Path | None = None,
    ):
        self.timeout = float(timeout)
        self.max_candidates = int(max_candidates)
        self.retries = max(1, int(retries))
        self.retry_delay = max(0.0, float(retry_delay))
        self.force_ipv4 = bool(force_ipv4)
        self.local_first = bool(local_first)
        self.local_db_path = Path(local_db_path) if local_db_path is not None else None

    def _fetch_html(self, url: str) -> tuple[str, str]:
        _require_requests()
        headers = {
            "User-Agent": "BL03U-MassSpectrumTool/1.0 (+https://webbook.nist.gov/)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "close",
        }
        timeout = (self.timeout, self.timeout)
        last_error: Exception | None = None
        old_allowed_gai_family = None
        for attempt in range(1, self.retries + 1):
            try:
                session = requests.Session()
                if self.force_ipv4 and urllib3_connection is not None:
                    old_allowed_gai_family = urllib3_connection.allowed_gai_family
                    urllib3_connection.allowed_gai_family = lambda: socket.AF_INET
                response = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
                response.raise_for_status()
                return _fix_nist_html(response.text), response.url
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                reason = exc.response.reason if exc.response is not None else str(exc)
                raise RuntimeError(f"NIST WebBook HTTP {status}: {reason}") from exc
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(self.retry_delay * attempt)
                    continue
            finally:
                if old_allowed_gai_family is not None and urllib3_connection is not None:
                    urllib3_connection.allowed_gai_family = old_allowed_gai_family
                    old_allowed_gai_family = None
                try:
                    session.close()
                except Exception:
                    pass
        raise RuntimeError(f"无法连接 NIST WebBook（已重试 {self.retries} 次）: {last_error}") from last_error

    def _load_compound_by_id(self, nist_id: str) -> NistCompoundIonization:
        compound_url = f"{NIST_SEARCH_URL}?{urlencode({'ID': nist_id, 'Units': 'SI'})}"
        html, final_url = self._fetch_html(compound_url)
        return self._compound_from_html(html, final_url, fallback_id=nist_id)

    def _load_candidate_by_id(self, nist_id: str) -> NistCompoundIonization:
        try:
            return self._load_compound_by_id(nist_id)
        except Exception as exc:
            return NistCompoundIonization(
                nist_id=nist_id,
                name=None,
                formula=None,
                cas_rn=None,
                url=f"{NIST_SEARCH_URL}?{urlencode({'ID': nist_id, 'Units': 'SI'})}",
                ion_energetics_url=None,
                message=f"候选条目加载失败: {exc}",
            )

    def _compound_from_html(
        self,
        html: str,
        page_url: str,
        *,
        fallback_id: str | None = None,
    ) -> NistCompoundIonization:
        info = parse_compound_page(html, page_url=page_url, fallback_id=fallback_id)
        ion_url = info["data_refs"].get("cIE")
        ie_html = html
        if not parse_ionization_energy_summary(html) and not parse_ionization_energy_determinations(html):
            if ion_url:
                ie_html, _ = self._fetch_html(ion_url)
        summary = parse_ionization_energy_summary(ie_html)
        determinations = parse_ionization_energy_determinations(ie_html)
        evaluated = pick_evaluated_ie(summary)
        message = ""
        if ion_url is None:
            message = "该 WebBook 条目没有 Gas phase ion energetics 数据链接。"
        elif evaluated is None and not determinations:
            message = "WebBook 有离子能量学页面，但未解析到 IE 表。"
        return NistCompoundIonization(
            nist_id=info.get("nist_id"),
            name=info.get("name"),
            formula=info.get("formula"),
            cas_rn=info.get("cas_rn"),
            url=info.get("url"),
            ion_energetics_url=ion_url,
            evaluated_ie=evaluated,
            determinations=determinations,
            message=message,
        )

    def query_ionization_energy(self, query: str, *, search_type: str = "auto") -> NistWebBookResult:
        normalized = query.strip()
        if not normalized:
            raise ValueError("请输入要查询的物种名称、分子式、CAS号或 NIST ID。")
        resolved_type = infer_nist_search_type(normalized) if search_type == "auto" else search_type
        local_formula_candidates: tuple[NistCompoundIonization, ...] = tuple()
        if self.local_first:
            local_match = lookup_local_ionization_energy(
                normalized,
                db_path=self.local_db_path,
                allow_fuzzy=resolved_type == "name",
            )
            if local_match is not None:
                return local_match_to_result(normalized, local_match)
            if resolved_type == "formula":
                try:
                    nominal_mz = nominal_mass_from_formula(normalized)
                    local_formula_candidates = tuple(
                        local_match_to_compound(
                            match,
                            source="local_mz_candidate",
                            message=f"本地物种数据库按分子式名义质量 m/z={nominal_mz} 命中候选；请人工确认结构。",
                        )
                        for match in lookup_local_ionization_candidates_by_mz(
                            nominal_mz,
                            db_path=self.local_db_path,
                            limit=self.max_candidates,
                        )
                    )
                except ValueError:
                    local_formula_candidates = tuple()
        search_url = build_nist_search_url(normalized, resolved_type)
        html, final_url = self._fetch_html(search_url)

        _require_bs4()
        soup = BeautifulSoup(html, "html.parser")
        if _is_compound_page(soup):
            compound = self._compound_from_html(html, final_url)
            compounds = self._merge_local_and_webbook_compounds(local_formula_candidates, (compound,))
            message = "WebBook 单一条目查询完成。"
            if compound.best_ie is None:
                message = compound.message or "WebBook 条目中未找到可用 IE。"
            if local_formula_candidates:
                message = f"本地库按 m/z 返回 {len(local_formula_candidates)} 个候选；{message}"
            return NistWebBookResult(
                query=normalized,
                search_type=resolved_type,
                requested_url=final_url,
                compounds=compounds,
                selected_compound=compound if len(compounds) == 1 else None,
                message=message,
            )

        ids, lost = parse_search_result_ids(html)
        if not ids:
            if local_formula_candidates:
                selected = local_formula_candidates[0] if len(local_formula_candidates) == 1 else None
                return NistWebBookResult(
                    query=normalized,
                    search_type=resolved_type,
                    requested_url=final_url,
                    compounds=local_formula_candidates,
                    selected_compound=selected,
                    lost=lost,
                    message=(
                        "NIST WebBook 未返回匹配物种；"
                        f"本地库按 m/z 返回 {len(local_formula_candidates)} 个候选，请人工确认结构。"
                    ),
                )
            return NistWebBookResult(
                query=normalized,
                search_type=resolved_type,
                requested_url=final_url,
                lost=lost,
                message="NIST WebBook 未返回匹配物种。请检查名称/分子式，或改用 CAS 号、NIST ID。",
            )

        webbook_compounds = tuple(self._load_candidate_by_id(nist_id) for nist_id in ids[: self.max_candidates])
        compounds = self._merge_local_and_webbook_compounds(local_formula_candidates, webbook_compounds)
        selected = compounds[0] if len(compounds) == 1 else None
        if len(ids) > self.max_candidates:
            message = f"WebBook 返回 {len(ids)} 个候选，当前只加载前 {self.max_candidates} 个。"
        elif len(webbook_compounds) == 1:
            message = "WebBook 单一候选查询完成。"
        else:
            message = "WebBook 返回多个候选；分子式查询可能对应多个同分异构体，请按名称、CAS 或 NIST ID 确认具体物种。"
        if local_formula_candidates:
            message = f"本地库按 m/z 返回 {len(local_formula_candidates)} 个候选；{message}"
        return NistWebBookResult(
            query=normalized,
            search_type=resolved_type,
            requested_url=final_url,
            compounds=compounds,
            selected_compound=selected,
            lost=lost,
            message=message,
        )

    def _merge_local_and_webbook_compounds(
        self,
        local_formula_candidates: tuple[NistCompoundIonization, ...],
        webbook_compounds: tuple[NistCompoundIonization, ...],
    ) -> tuple[NistCompoundIonization, ...]:
        if not self.local_first or not webbook_compounds:
            return local_formula_candidates + webbook_compounds

        local_by_webbook_name: list[NistCompoundIonization] = []
        seen_local_ids: set[str] = set()
        for compound in webbook_compounds:
            if not compound.name:
                continue
            match = lookup_local_ionization_energy(compound.name, db_path=self.local_db_path, allow_fuzzy=False)
            if match is None:
                continue
            local_compound = local_match_to_compound(
                match,
                source="local_database",
                message="本地物种数据库与 WebBook 候选名称精确匹配。",
            )
            if local_compound.nist_id not in seen_local_ids:
                local_by_webbook_name.append(local_compound)
                seen_local_ids.add(str(local_compound.nist_id))

        remaining_local_formula = tuple(
            compound for compound in local_formula_candidates if str(compound.nist_id) not in seen_local_ids
        )
        return tuple(local_by_webbook_name) + remaining_local_formula + webbook_compounds


def default_nist_webbook_client() -> NistWebBookClient:
    return NistWebBookClient()
