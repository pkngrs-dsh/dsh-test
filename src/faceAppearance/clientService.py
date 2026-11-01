"""Versioned HTTP contract for clients consuming a complete assembled asset."""

from typing import Annotated, Literal

from fastapi import File, Form, HTTPException, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException


class ClientError(BaseModel):
    code: str
    message: str
    retryable: bool = False


class ErrorResponse(BaseModel):
    apiVersion: Literal["1"] = "1"
    error: ClientError


class TemplateIdentity(BaseModel):
    templateId: str
    templateHashes: dict[str, str]


class UploadLimits(BaseModel):
    minImages: int
    maxImages: int
    maxImageBytes: int
    maxTotalBytes: int


class AssetFormat(BaseModel):
    format: Literal["zip"] = "zip"
    schemaVersion: Literal[1] = 1


class Capabilities(BaseModel):
    apiVersion: Literal["1"] = "1"
    ready: bool
    acceptingJobs: bool
    template: TemplateIdentity | None
    limits: UploadLimits
    output: AssetFormat
    assemblySourceStage: Literal["fusion", "wildcap"]
    pollAfterSeconds: int


class AssetSummary(BaseModel):
    schemaVersion: Literal[1]
    templateId: str
    templateHashes: dict[str, str]
    coordinates: dict
    binding: dict
    counts: dict[str, int]


class AssetDownload(BaseModel):
    url: str
    manifestUrl: str
    sha256: str
    sizeBytes: int


class ClientJob(BaseModel):
    apiVersion: Literal["1"] = "1"
    jobId: str
    status: Literal["queued", "running", "succeeded", "failed"]
    createdAt: str
    startedAt: str | None = None
    finishedAt: str | None = None
    numImages: int
    statusUrl: str
    pollAfterSeconds: int
    download: AssetDownload | None = None
    asset: AssetSummary | None = None
    error: ClientError | None = None
    metrics: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def clientFailure(code: str, message: str, statusCode: int = 422):
    return HTTPException(statusCode, detail={"code": code, "message": message,
                                            "retryable": statusCode in {429, 503}})


def jobResponse(record: dict) -> ClientJob:
    jobId = record["jobId"]
    active = record["status"] in {"queued", "running"}
    summary = record.get("clientAsset") if record["status"] == "succeeded" else None
    error = record.get("error")
    clientError = None
    if error:
        code = error.get("code", "reconstruction_failed")
        # Detailed exception text and filesystem paths remain in the local job record.
        messages = {"service_restarted": "The service restarted before this job completed. Submit a new job.",
                    "invalid_client_asset": "The generated client asset did not pass package validation."}
        clientError = ClientError(code=code, message=messages.get(
            code, "Reconstruction failed. Inspect the photos or contact the service operator with the job ID."))
    result = record.get("result") or {}
    metrics = {key: result[key] for key in ("loadSeconds", "inferenceSeconds", "stageSeconds",
               "peakAllocatedGiB", "acceptedCameraCount") if key in result}
    return ClientJob(jobId=jobId, status=record["status"], createdAt=record["createdAt"],
                     startedAt=record.get("startedAt"), finishedAt=record.get("finishedAt"),
                     numImages=record["numImages"], statusUrl=f"/v1/jobs/{jobId}",
                     pollAfterSeconds=2 if active else 0, error=clientError,
                     asset=AssetSummary(**summary) if summary else None,
                     download=AssetDownload(url=f"/v1/jobs/{jobId}/download",
                         manifestUrl=f"/v1/jobs/{jobId}/manifest", **summary["archive"]) if summary else None,
                     metrics=metrics,
                     warnings=["client_rig_not_validated", "appearance_not_physically_calibrated",
                               "unobserved_regions_use_template"])


def registerClientApi(app, config, getQueue, submitImages, downloadArtifact,
                      *, maxImageBytes: int, maxTotalBytes: int):
    """Install v1 routes over the same serialized GPU execution queue."""
    errors = {code: {"model": ErrorResponse} for code in (401, 404, 409, 413, 422, 429, 500, 503)}

    def template():
        assets = getattr(getQueue().pipeline, "assemblyAssets", None)
        manifest = getattr(assets, "manifest", None)
        if not isinstance(manifest, dict):
            return None
        return {key: manifest[key] for key in ("templateId", "templateHashes")}

    def available():
        worker = getQueue()
        return bool(config.enableAssembly and worker.pipeline.loaded and template()
                    and (config.assemblySourceStage != "wildcap" or (
                        config.enableWildCap and getattr(worker.pipeline, "wildCapAdapter", None) is not None)))

    @app.exception_handler(StarletteHTTPException)
    async def httpError(request, exception):
        if not request.url.path.startswith("/v1/"):
            return JSONResponse(status_code=exception.status_code,
                                content={"detail": exception.detail}, headers=exception.headers)
        detail = exception.detail
        codes = {404: "not_found", 409: "result_not_ready", 413: "upload_too_large",
                 422: "validation_error", 429: "queue_full", 503: "service_unavailable"}
        error = detail if isinstance(detail, dict) else {
            "code": codes.get(exception.status_code, "request_failed"), "message": str(detail),
            "retryable": exception.status_code in {429, 503}}
        return JSONResponse(status_code=exception.status_code,
                            content=ErrorResponse(error=ClientError(**error)).model_dump(),
                            headers=exception.headers)

    @app.exception_handler(RequestValidationError)
    async def invalidRequest(request, exception):
        if request.url.path.startswith("/v1/"):
            return JSONResponse(status_code=422, content=ErrorResponse(error=ClientError(
                code="validation_error", message="Expected multipart form data with images and an optional templateId."
            )).model_dump())
        from fastapi.exception_handlers import request_validation_exception_handler
        return await request_validation_exception_handler(request, exception)

    @app.exception_handler(Exception)
    async def unexpectedError(request, exception):
        import logging
        logging.getLogger(__name__).exception("Unhandled service request failure")
        if request.url.path.startswith("/v1/"):
            return JSONResponse(status_code=500, content=ErrorResponse(error=ClientError(
                code="internal_error", message="The service could not complete the request.")).model_dump(),
                headers={"Cache-Control": "no-store"})
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

    @app.get("/v1/capabilities", tags=["client"], responses=errors, response_model=Capabilities)
    def capabilities():
        queue = getQueue()
        ready = available()
        return {"apiVersion": "1", "ready": ready,
                "acceptingJobs": ready and queue.accepting and not queue.pending.full(),
                "template": template(),
                "limits": {"minImages": config.minViews, "maxImages": config.maxViews,
                           "maxImageBytes": maxImageBytes, "maxTotalBytes": maxTotalBytes},
                "output": {"format": "zip", "schemaVersion": 1},
                "assemblySourceStage": config.assemblySourceStage, "pollAfterSeconds": 2}

    @app.post("/v1/jobs", status_code=202, response_model=ClientJob, tags=["client"], responses=errors)
    async def createJob(response: Response, images: Annotated[list[UploadFile], File()],
                        templateId: Annotated[str | None, Form()] = None):
        try:
            if not available():
                raise clientFailure("client_service_unavailable", "Client assembly assets are not available.", 503)
            if templateId is not None and templateId != template()["templateId"]:
                raise clientFailure("template_mismatch", "The requested template is not served by this instance.")
            record = await submitImages(images, "assembly", clientContract=True)
            result = jobResponse(record)
            response.headers["Location"] = result.statusUrl
            response.headers["Retry-After"] = "2"
            return result
        finally:
            for upload in images:
                await upload.close()

    def clientRecord(jobId):
        record = getQueue().get(jobId)
        if record is None or not record.get("clientContract"):
            raise clientFailure("not_found", "Unknown client job.", 404)
        return record

    @app.get("/v1/jobs/{jobId}", response_model=ClientJob, tags=["client"], responses=errors)
    def getJob(jobId: str, response: Response):
        result = jobResponse(clientRecord(jobId))
        if result.pollAfterSeconds:
            response.headers["Retry-After"] = str(result.pollAfterSeconds)
        return result

    def completedRecord(jobId):
        record = clientRecord(jobId)
        if record["status"] != "succeeded":
            raise clientFailure("job_failed" if record["status"] == "failed" else "result_not_ready",
                                "A successful job is required to download the asset.", 409)
        return record

    from .webPreview import registerClientPreview
    registerClientPreview(app, completedRecord, downloadArtifact, errors=errors)

    @app.get("/v1/jobs/{jobId}/download", tags=["client"], responses={**errors, 200: {
        "description": "Verified client asset ZIP", "content": {"application/zip": {
            "schema": {"type": "string", "format": "binary"}}}}},
             response_class=Response)
    def downloadJob(jobId: str):
        record = completedRecord(jobId)
        response = downloadArtifact(jobId, "assembly/clientAsset.zip")
        response.media_type = "application/zip"
        response.headers["Content-Type"] = "application/zip"
        response.headers["ETag"] = '"' + record["clientAsset"]["archive"]["sha256"] + '"'
        return response

    @app.get("/v1/jobs/{jobId}/manifest", tags=["client"], responses=errors)
    def manifestJob(jobId: str):
        completedRecord(jobId)
        return downloadArtifact(jobId, "assembly/manifest.json")
