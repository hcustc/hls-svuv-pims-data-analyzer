"""PIE analysis routes."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Optional
import uuid

from fastapi import APIRouter, HTTPException, Request

from bl03u_masstool.api.schemas import PieFitPayload, PieStartPayload


def create_router(deps) -> APIRouter:
    router = APIRouter()

    @router.post("/api/pie/start")
    def start_pie_job(payload: PieStartPayload):
        deps._cleanup_jobs()
        job_id = uuid.uuid4().hex
        with deps.JOBS_LOCK:
            deps.JOBS[job_id] = deps.PieJob(id=job_id)
        deps.EXECUTOR.submit(deps._run_pie_job, job_id, payload)
        return {"job_id": job_id, "status": "queued"}

    @router.post("/api/pie/upload_curve")
    async def upload_pie_curve(
        request: Request,
        filename: str = "pie_curve.csv",
        database: Optional[str] = None,
        library_id: Optional[str] = None,
    ):
        try:
            database_path = deps._database_path_from_request(database=database, library_id=library_id)
        except HTTPException as exc:
            if library_id:
                raise HTTPException(status_code=404, detail="临时 PICS 工作库不存在，请重新上传 PICS 数据")
            raise exc
        try:
            content = await request.body()
            deps._check_upload_size(content)
            analysis_df = deps._parse_uploaded_pie_curves(content, filename)
            curves = deps.build_pie_curves(analysis_df)
            if not curves:
                raise ValueError("上传文件没有可用 PIE 曲线")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        summary = deps._pie_summary(analysis_df, curves)
        manifest, evidence = deps._pie_artifacts(
            source_label=Path(filename).name,
            peak_source="uploaded_curve",
            database_reference=deps._database_artifact_reference(
                database=database,
                library_id=library_id,
                resolved_path=database_path,
            ),
            payload=None,
            analysis_df=analysis_df,
            curves=curves,
        )
        deps._cleanup_jobs()
        job_id = uuid.uuid4().hex
        with deps.JOBS_LOCK:
            deps.JOBS[job_id] = deps.PieJob(
                id=job_id,
                status="done",
                message="完成",
                summary=summary,
                analysis_df=analysis_df,
                curves=curves,
                database_path=str(database_path),
                source_folder=Path(filename).name,
                peak_source="uploaded_curve",
                manifest=manifest,
                evidence=evidence,
            )
        return {"job_id": job_id, "status": "done", "summary": summary}

    @router.get("/api/pie/progress/{job_id}")
    def pie_progress(job_id: str):
        job = deps._get_job(job_id)
        return {
            "job_id": job.id,
            "status": job.status,
            "message": job.message,
            "error": job.error,
            "elapsed": round(time.time() - job.created_at, 3),
            "summary": job.summary if job.status == "done" else {},
        }

    @router.get("/api/pie/result/{job_id}")
    def pie_result(job_id: str):
        job = deps._get_job(job_id)
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"job status is {job.status}")
        return {
            "job_id": job.id,
            "summary": job.summary,
            "source_folder": job.source_folder,
            "peak_source": job.peak_source,
        }

    @router.get("/api/pie/artifacts/{job_id}")
    def pie_artifacts(job_id: str):
        job = deps._get_job(job_id)
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"job status is {job.status}")
        return {
            "job_id": job.id,
            "summary": job.summary,
            "manifest": job.manifest,
            "evidence": job.evidence,
        }

    @router.get("/api/pie/curve/{job_id}/{mz}")
    def pie_curve(job_id: str, mz: int):
        job = deps._get_job(job_id)
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"job status is {job.status}")
        curve = job.curves.get(int(mz))
        if curve is None:
            raise HTTPException(status_code=404, detail="curve not found")
        return {
            "curve": deps._curve_payload(curve),
            "fit": job.fits.get(int(mz)),
        }

    @router.get("/api/pie/candidates/{job_id}/{mz}")
    def pie_fit_candidates(job_id: str, mz: int):
        job = deps._get_job(job_id)
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"job status is {job.status}")
        database_path = Path(job.database_path)
        if not database_path.exists() or not database_path.is_file():
            raise HTTPException(status_code=404, detail="SQLite PICS截面数据库不存在")
        return {"rows": deps._query_fit_candidates(database_path, int(mz))}

    @router.get("/api/pie/candidate_curves/{job_id}/{mz}")
    def pie_candidate_curves(job_id: str, mz: int, species_ids: str = ""):
        job = deps._get_job(job_id)
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"job status is {job.status}")
        database_path = Path(job.database_path)
        if not database_path.exists() or not database_path.is_file():
            raise HTTPException(status_code=404, detail="SQLite PICS截面数据库不存在")
        selected_ids = deps._parse_species_ids(species_ids)
        if not selected_ids:
            selected_ids = [row["id"] for row in deps._query_fit_candidates(database_path, int(mz))]
        rows = [
            {
                "id": int(item["id"]),
                "mz": int(item["mz"]),
                "species": item["species"],
                "ie": item.get("ie"),
                "energies": [float(value) for value in item["energies"]],
                "cross_sections": [float(value) for value in item["cross_sections"]],
            }
            for item in deps._load_species_records_by_ids(database_path, selected_ids)
            if int(item["mz"]) == int(mz)
        ]
        return {"rows": rows}

    @router.post("/api/pie/fit/{job_id}/{mz}")
    def fit_pie_curve(job_id: str, mz: int, payload: Optional[PieFitPayload] = None):
        job = deps._get_job(job_id)
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"job status is {job.status}")
        curve = job.curves.get(int(mz))
        if curve is None:
            raise HTTPException(status_code=404, detail="curve not found")
        database_path = Path(job.database_path)
        selected_ids = payload.species_ids if payload and payload.species_ids else []
        if selected_ids:
            species = deps._load_species_records_by_ids(database_path, selected_ids)
            if not species:
                raise HTTPException(status_code=400, detail="未找到可拟合的 PICS 候选")
            wrong_mz = sorted({int(item["mz"]) for item in species if int(item["mz"]) != int(mz)})
            if wrong_mz:
                raise HTTPException(status_code=400, detail=f"选择的 PICS m/z 与当前曲线不一致: {wrong_mz}")
            fit_model = deps.fit_species_combination_with_curve(
                species,
                curve["energies"],
                curve["intensities"],
                coefficient_mode=payload.coefficient_mode if payload else "fit",
                coefficients=payload.coefficients if payload else None,
                locked_species_ids=payload.locked_species_ids if payload else None,
            )
            fit_model["selection_mode"] = "manual"
            fit_model["selected_species_ids"] = [int(value) for value in selected_ids]
        else:
            database = deps._load_species_database_cached(str(database_path), database_path.stat().st_mtime)
            fit_model = deps.identify_species_for_mz_with_curve(
                database,
                int(mz),
                curve["energies"],
                curve["intensities"],
            )
            fit_model["selection_mode"] = "auto"
            fit_model["selected_species_ids"] = []
        with deps.JOBS_LOCK:
            job.fits[int(mz)] = fit_model
            job.evidence = deps.build_pie_evidence_objects(job.curves, job.fits)
            job.updated_at = time.time()
        return {"fit": fit_model}

    @router.get("/api/pie/export/{job_id}")
    def pie_export(job_id: str):
        job = deps._get_job(job_id)
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"job status is {job.status}")
        return {"rows": deps._json_records(job.analysis_df)}

    return router
