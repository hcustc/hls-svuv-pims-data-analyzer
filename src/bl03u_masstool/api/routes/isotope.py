"""Isotope calculation routes."""

from __future__ import annotations

from fastapi import APIRouter

from bl03u_masstool.api.schemas import IsotopePayload


def create_router(deps) -> APIRouter:
    router = APIRouter()

    @router.post("/isotope")
    def isotope(payload: IsotopePayload):
        return {
            "distribution": deps.calculate_isotope_distribution(
                payload.formula,
                min_percent=payload.min_percent,
            )
        }

    return router
