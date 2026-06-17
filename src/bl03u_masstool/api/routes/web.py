"""Web frontend routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from starlette.responses import RedirectResponse


def create_router(deps) -> APIRouter:
    router = APIRouter()

    @router.get("/", include_in_schema=False)
    def web_index():
        index = deps.WEB_ROOT / "index.html"
        if not index.exists():
            raise HTTPException(status_code=404, detail="web frontend not found")
        return RedirectResponse(url="/static/index.html")

    return router
