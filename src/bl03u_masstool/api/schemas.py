"""Request payload schemas for the BL03U API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class IsotopePayload(BaseModel):
    formula: str
    min_percent: float = 0.01


class PieStartPayload(BaseModel):
    folder: str
    database: Optional[str] = None
    library_id: Optional[str] = None
    peak_source: str = "auto"
    manual_peak_path: Optional[str] = None
    target_mz: Optional[str] = None
    recursive: bool = True
    energy_decimals: int = 1
    gaussian: bool = False
    integration_method: str = "sum_counts"
    photon_mode: Optional[str] = None
    light_source: Optional[str] = None
    mass_discrimination: Optional[float] = None
    replicate_mode: str = "off"


class PieFitPayload(BaseModel):
    species_ids: Optional[List[int]] = None
    coefficient_mode: str = "fit"
    coefficients: Optional[Dict[int, float]] = None
    locked_species_ids: Optional[List[int]] = None


__all__ = ["IsotopePayload", "PieFitPayload", "PieStartPayload"]
