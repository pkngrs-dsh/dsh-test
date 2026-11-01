"""Local FastAPI service with one preloaded pipeline and a bounded job queue.

Importing this module or creating an application does not import the pipeline,
load model weights, or initialize CUDA. Run one ASGI process per GPU. Uploaded
images and outputs remain below outputRoot. Optional Bearer authentication applies
to both client and legacy APIs. The static /viewer shell is public so users can
enter their token; use one ASGI process per GPU.
"""

from __future__ import annotations

import hashlib
import secrets
import queue
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Callable, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer
from PIL import Image
from starlette.concurrency import run_in_threadpool

from .config import PipelineConfig
from .jobs import JobQueue, timestamp, writeJobRecord


maxImageBytes = 25 * 1024 * 1024
maxTotalBytes = 200 * 1024 * 1024
uploadChunkBytes = 1024 * 1024
StopAfter = Literal["geometry", "delight", "fusion", "wildcap", "assembly"]


def createApp(config: PipelineConfig, pipelineFactory: Callable | None = None,
              outputRoot: Path | None = None, queueCapacity: int = 2,
              apiToken: str | None = None) -> FastAPI:
    """Create a local application; the default Pipeline is imported at startup."""
    if isinstance(queueCapacity, bool) or not isinstance(queueCapacity, int) or queueCapacity < 1:
        raise ValueError("queueCapacity must be a positive integer")
    if apiToken is not None and not apiToken.strip():
        raise ValueError("apiToken must be nonempty when configured")
    outputRoot = Path(outputRoot or config.projectRoot / "runs/service").resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        factory = pipelineFactory
        if factory is None:
            from .pipeline import Pipeline

            factory = Pipeline
        pipeline = factory(config)
        outputRoot.mkdir(parents=True, exist_ok=True)
        jobQueue = JobQueue(pipeline, outputRoot, queueCapacity)
        # Lock before model loading, so a duplicate owner fails without using GPU memory.
        jobQueue.acquireOwnership()
        try:
            await run_in_threadpool(pipeline.loadModels)
            if not pipeline.loaded:
                raise RuntimeError("Pipeline did not finish loading its models")
            from .startupWarmup import runStartupWarmup
            app.state.startupWarmup = await run_in_threadpool(runStartupWarmup, pipeline, config)
            jobQueue.start()
        except BaseException:
            jobQueue.releaseOwnership()
            raise
        app.state.pipeline = pipeline
        app.state.jobQueue = jobQueue
        try:
            yield
        finally:
            await run_in_threadpool(jobQueue.close)

    app = FastAPI(title="FaceAppearance client reconstruction", version="1.0.0", lifespan=lifespan,
                  dependencies=[Depends(HTTPBearer(auto_error=False))] if apiToken else [])

    from .webPreview import registerViewer
    registerViewer(app)

    def getQueue() -> JobQueue:
        jobQueue = getattr(app.state, "jobQueue", None)
        if jobQueue is None:
            raise HTTPException(503, detail="Models are not ready; wait for service startup")
        return jobQueue

    @app.get("/health")
    def health():
        return {**getQueue().health(), "startupWarmup": app.state.startupWarmup}

    async def acceptJob(images: list[UploadFile], stopAfter: str,
                        *, clientContract: bool = False):
        jobQueue = getQueue()
        if stopAfter == "wildcap" and not config.enableWildCap:
            raise HTTPException(422, detail="WildCap was not enabled and preloaded for this service")
        if stopAfter == "assembly":
            if not config.enableAssembly or getattr(jobQueue.pipeline, "assemblyAssets", None) is None:
                raise HTTPException(422, detail="Assembly assets were not enabled and preloaded for this service")
            if config.assemblySourceStage == "wildcap" and (not config.enableWildCap
                    or getattr(jobQueue.pipeline, "wildCapAdapter", None) is None):
                raise HTTPException(422, detail="The selected assembly source requires preloaded WildCap")
        if not config.minViews <= len(images) <= config.maxViews:
            raise HTTPException(422, detail=f"Expected {config.minViews}–{config.maxViews} distinct images; "
                                            f"received {len(images)}")
        if not jobQueue.accepting:
            raise HTTPException(503, detail="Service is shutting down")
        if jobQueue.pending.full():
            raise HTTPException(429, detail="Reconstruction queue is full; retry after a job finishes",
                                headers={"Retry-After": "5"})
        jobId = uuid.uuid4().hex
        incomingDir = outputRoot / f".incoming-{jobId}"
        inputDir = incomingDir / "inputs"
        inputDir.mkdir(parents=True)
        totalBytes = 0
        contentHashes = set()
        imagePaths, originalNames = [], []
        try:
            for imageIndex, upload in enumerate(images):
                # Client filenames never become filesystem paths.
                imagePath = inputDir / f"image{imageIndex:02d}.image"
                imageBytes = 0
                contentHash = hashlib.sha256()
                with imagePath.open("wb") as imageFile:
                    while chunk := await upload.read(uploadChunkBytes):
                        imageBytes += len(chunk)
                        totalBytes += len(chunk)
                        if imageBytes > maxImageBytes:
                            raise HTTPException(413, detail=f"Image {imageIndex + 1} exceeds the 25 MiB limit")
                        if totalBytes > maxTotalBytes:
                            raise HTTPException(413, detail="Combined image data exceeds the 200 MiB limit")
                        contentHash.update(chunk)
                        imageFile.write(chunk)
                if imageBytes == 0:
                    raise HTTPException(422, detail=f"Image {imageIndex + 1} is empty")
                digest = contentHash.hexdigest()
                if digest in contentHashes:
                    raise HTTPException(422, detail=f"Image {imageIndex + 1} duplicates an earlier image; "
                                                   "supply distinct views")
                contentHashes.add(digest)
                imagePaths.append(imagePath)
                originalNames.append(Path(upload.filename or f"image{imageIndex:02d}").name)

            from .data import loadViews

            try:
                await run_in_threadpool(loadViews, imagePaths, minViews=config.minViews, maxViews=config.maxViews)
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                message = str(error).replace(str(inputDir), "uploaded images")
                raise HTTPException(422, detail=f"Invalid image set: {message}") from error
            try:
                return jobQueue.submit(jobId, incomingDir, originalNames, stopAfter,
                                       clientContract=clientContract)
            except queue.Full as error:
                raise HTTPException(429, detail="Reconstruction queue is full; retry after a job finishes",
                                    headers={"Retry-After": "5"}) from error
            except RuntimeError as error:
                raise HTTPException(503, detail=str(error)) from error
        finally:
            for upload in images:
                await upload.close()
            if incomingDir.exists():
                shutil.rmtree(incomingDir)

    async def submitImages(images: list[UploadFile], stopAfter: str,
                           *, clientContract: bool = False):
        try:
            return await acceptJob(images, stopAfter, clientContract=clientContract)
        finally:
            for upload in images:
                await upload.close()

    @app.post("/jobs", status_code=202)
    async def submitJob(images: Annotated[list[UploadFile], File()],
                        stopAfter: Annotated[StopAfter, Form()] = "fusion"):
        return await submitImages(images, stopAfter)

    @app.get("/jobs/{jobId}")
    def getJob(jobId: str):
        record = getQueue().get(jobId)
        if record is None:
            raise HTTPException(404, detail="Unknown job")
        return record

    @app.get("/jobs/{jobId}/artifacts/{filename:path}")
    def downloadArtifact(jobId: str, filename: str):
        record = getQueue().get(jobId)
        if record is None:
            raise HTTPException(404, detail="Unknown job")
        if record["status"] != "succeeded":
            raise HTTPException(409, detail="Artifacts are available after the job succeeds")
        jobRoot = (outputRoot / jobId).resolve()
        artifactRoot = (jobRoot / "output").resolve()
        if not jobRoot.is_relative_to(outputRoot) or not artifactRoot.is_relative_to(jobRoot):
            raise HTTPException(404, detail="Artifact not found")
        requestedPath = (artifactRoot / filename).resolve()
        if not requestedPath.is_relative_to(artifactRoot) or not requestedPath.is_file():
            raise HTTPException(404, detail="Artifact not found")
        allowedNames = {item["filename"] for item in record["artifacts"]}
        if filename not in allowedNames:
            raise HTTPException(404, detail="Artifact not found")
        return FileResponse(requestedPath, filename=requestedPath.name)

    from .clientService import registerClientApi
    registerClientApi(app, config, getQueue, submitImages, downloadArtifact,
                      maxImageBytes=maxImageBytes, maxTotalBytes=maxTotalBytes)

    @app.middleware("http")
    async def authorize(request, callNext):
        viewerShell = request.url.path == "/viewer" or request.url.path.startswith("/viewer/")
        if apiToken is not None and not viewerShell:
            supplied = request.headers.get("authorization", "")
            if not secrets.compare_digest(supplied.encode("utf-8"),
                                          ("Bearer " + apiToken).encode("utf-8")):
                return JSONResponse(status_code=401, content={"apiVersion": "1", "error": {
                    "code": "unauthorized", "message": "A valid Bearer token is required", "retryable": False}},
                    headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"})
        response = await callNext(request)
        if request.url.path.startswith("/v1/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    return app
