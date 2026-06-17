"""PICS database search and upload routes."""

from __future__ import annotations

import sqlite3
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Request


def create_router(deps) -> APIRouter:
    router = APIRouter()

    @router.get("/api/pics/search")
    def pics_search(
        database: Optional[str] = None,
        library_id: Optional[str] = None,
        mz: Optional[int] = None,
        tolerance: int = 0,
        name: str = "",
        ie_min: Optional[float] = None,
        ie_max: Optional[float] = None,
        limit: int = 100,
    ):
        database_path = deps._database_path_from_request(database=database, library_id=library_id)
        limit = max(1, min(int(limit), 500))
        clauses = []
        params: List[Any] = []
        if mz is not None:
            tolerance = max(0, int(tolerance))
            clauses.append("s.mz BETWEEN ? AND ?")
            params.extend([int(mz) - tolerance, int(mz) + tolerance])
        if name.strip():
            clauses.append("LOWER(s.name) LIKE ?")
            params.append(f"%{name.strip().lower()}%")
        if ie_min is not None:
            clauses.append("s.ionization_energy >= ?")
            params.append(float(ie_min))
        if ie_max is not None:
            clauses.append("s.ionization_energy <= ?")
            params.append(float(ie_max))
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT
                s.id,
                s.mz,
                s.name,
                s.ionization_energy,
                COUNT(p.id) AS point_count,
                MIN(p.energy_ev) AS energy_min,
                MAX(p.energy_ev) AS energy_max,
                MAX(p.cross_section) AS cross_section_max
            FROM species s
            LEFT JOIN pic_cross_sections p ON p.species_id = s.id
            {where_sql}
            GROUP BY s.id, s.mz, s.name, s.ionization_energy
            ORDER BY s.mz, s.name, s.id
            LIMIT ?
        """
        params.append(limit)
        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
        return {
            "rows": [
                {
                    "id": int(row["id"]),
                    "mz": int(row["mz"]),
                    "name": row["name"],
                    "ionization_energy": deps._float_or_none(row["ionization_energy"]),
                    "point_count": int(row["point_count"] or 0),
                    "energy_min": deps._float_or_none(row["energy_min"]),
                    "energy_max": deps._float_or_none(row["energy_max"]),
                    "cross_section_max": deps._float_or_none(row["cross_section_max"]),
                }
                for row in rows
            ]
        }

    @router.get("/api/pics/species/{species_id}")
    def pics_species(
        species_id: int,
        database: Optional[str] = None,
        library_id: Optional[str] = None,
    ):
        database_path = deps._database_path_from_request(database=database, library_id=library_id)
        with sqlite3.connect(database_path) as conn:
            conn.row_factory = sqlite3.Row
            species = conn.execute(
                "SELECT id, mz, name, ionization_energy FROM species WHERE id = ?",
                (int(species_id),),
            ).fetchone()
            if species is None:
                raise HTTPException(status_code=404, detail="species not found")
            points = conn.execute(
                """
                SELECT energy_ev, cross_section
                FROM pic_cross_sections
                WHERE species_id = ?
                ORDER BY energy_ev
                """,
                (int(species_id),),
            ).fetchall()
        return {
            "species": {
                "id": int(species["id"]),
                "mz": int(species["mz"]),
                "name": species["name"],
                "ionization_energy": deps._float_or_none(species["ionization_energy"]),
            },
            "points": [
                {
                    "energy_ev": float(point["energy_ev"]),
                    "cross_section": float(point["cross_section"]),
                }
                for point in points
            ],
        }

    @router.post("/api/pics/upload")
    async def pics_upload(
        request: Request,
        filename: str = "pics.csv",
        scope: str = "session",
        mode: str = "upsert",
        confirm_overwrite: bool = False,
    ):
        content = await request.body()
        deps._check_upload_size(content)
        try:
            records = deps._parse_pics_upload(content, filename)
            if scope == "server":
                deps._verify_admin_token(request)
                result = deps._write_pics_records(records, mode=mode, confirm_overwrite=confirm_overwrite)
                result["scope"] = "server"
                result["library_id"] = ""
            else:
                result = deps._create_session_pics_library(records, filename)
        except Exception as exc:
            if isinstance(exc, HTTPException):
                raise exc
            raise HTTPException(status_code=400, detail=str(exc))
        return result

    return router
