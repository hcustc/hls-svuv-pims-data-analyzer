"""Spectrum file and peak-detection routes."""

from __future__ import annotations

from pathlib import Path
import tempfile

from fastapi import APIRouter, HTTPException, Request

from bl03u_masstool.core.calibration import Calibration


def create_router(deps) -> APIRouter:
    router = APIRouter()

    @router.post("/spectrum/read")
    async def read_spectrum(request: Request, filename: str = "spectrum.txt"):
        suffix = Path(filename).suffix or ".txt"
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(await request.body())
                temp_path = handle.name
            spectrum = deps.read_bl03u_txt(temp_path)
            return {"x": spectrum.x.tolist(), "y": spectrum.y.tolist()}
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    @router.post("/spectrum/peaks")
    async def detect_peaks(
        request: Request,
        filename: str = "spectrum.txt",
        a: float = 3.66334e-7,
        b: float = 0.000637719,
        c: float = 0.272489072,
    ):
        suffix = Path(filename).suffix or ".txt"
        temp_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(await request.body())
                temp_path = handle.name
            spectrum = deps.read_bl03u_txt(temp_path)
            offset = float(spectrum.x[0]) if len(spectrum.x) else 0.0
            peaks = deps.detect_peaks_in_range(
                spectrum.y,
                calibration=Calibration(a, b, c),
                detection_min_idx=0,
                time_offset=offset,
            )
            return {"peaks": [peak.to_dict() for peak in peaks]}
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    @router.get("/spectrum/sum")
    def sum_folder(folder: str):
        try:
            folder_path = deps._resolve_path(folder)
            spectrum = deps.sum_spectra(folder_path)
        except deps.PathAccessError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"x": spectrum.x.tolist(), "y": spectrum.y.tolist()}

    return router
