"""Browser previews derived from the same verified ZIP used by native clients.

No files are extracted, no template axes are changed, and no rig compatibility
is inferred. Smooth preview normals are computed before UV/material splitting
so coincident render vertices share the source vertex's normal.
"""

from pathlib import Path

import numpy as np
from fastapi import HTTPException, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .clientImport import ClientAssetImportError, buildRenderMesh, loadClientAsset


PREVIEW_WARNINGS = ["client_rig_not_validated", "appearance_not_physically_calibrated",
                    "unobserved_regions_use_template", "preview_normals_computed_from_geometry"]


def registerViewer(app):
    """Serve the bundled viewer; StaticFiles confines reads to this directory."""
    @app.get("/viewer", include_in_schema=False)
    def viewerRedirect():
        return RedirectResponse("/viewer/", status_code=307)

    app.mount("/viewer", StaticFiles(directory=Path(__file__).with_name("web"), html=True),
              name="viewer")


def buildPreview(asset, jobId: str) -> dict:
    """Return JSON-compatible render attributes, preserving triangle order."""
    source = asset.mesh
    render = buildRenderMesh(asset)
    vertices, faces = source["vertices"], source["faces"]
    edgesA = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edgesB = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    faceNormals = np.cross(edgesA, edgesB)
    normals = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], faceNormals)
    lengths = np.linalg.norm(normals, axis=1)
    np.divide(normals, lengths[:, None], out=normals, where=lengths[:, None] > 0)

    groups = []
    for faceIndex, materialIndex in enumerate(render["materialIds"]):
        materialIndex = int(materialIndex)
        if groups and groups[-1]["materialIndex"] == materialIndex:
            groups[-1]["count"] += 3
        else:
            groups.append({"start": faceIndex * 3, "count": 3, "materialIndex": materialIndex})

    materials = []
    for material in asset.materials["materials"]:
        texture = material["baseColorTexture"]
        identity = asset.manifest["files"][texture["path"]]
        materials.append({key: material[key] for key in ("index", "name", "roughnessFactor",
                          "metallicFactor", "alphaMode", "alphaCutoff", "baseColorFactor")})
        materials[-1]["texture"] = {
            "url": f"/v1/jobs/{jobId}/textures/{material['index']}",
            "sha256": identity["sha256"], "sizeBytes": identity["bytes"], "colorSpace": "sRGB"}

    warnings = PREVIEW_WARNINGS.copy()
    if np.any(lengths[render["sourceVertexIds"]] == 0):
        warnings.append("preview_contains_zero_length_normals")
    return {
        "schemaVersion": 1, "jobId": jobId, "templateId": asset.manifest["templateId"],
        "templateHashes": asset.manifest["templateHashes"],
        "counts": {"sourceVertices": len(vertices), "renderVertices": len(render["vertices"]),
                   "faces": len(faces), "materials": len(materials)},
        "coordinates": {"lengthUnit": "meter", "uvOrigin": "vDown"},
        "mesh": {"positions": render["vertices"].ravel().tolist(),
                 "normals": normals[render["sourceVertexIds"]].ravel().tolist(),
                 "uvs": render["uvs"].ravel().tolist(),
                 "indices": render["faces"].ravel().tolist(),
                 "sourceVertexIds": render["sourceVertexIds"].tolist(), "groups": groups},
        "materials": materials, "warnings": warnings}


def registerClientPreview(app, completedRecord, downloadArtifact, *, errors):
    """Expose only geometry and packaged textures of successful client jobs."""
    def verifiedAsset(jobId):
        record = completedRecord(jobId)
        # Reuse the artifact allowlist and containment check before opening the ZIP.
        archivePath = downloadArtifact(jobId, "assembly/clientAsset.zip").path
        try:
            summary = record["clientAsset"]
            return loadClientAsset(
                archivePath, expectedTemplateId=summary["templateId"],
                expectedTemplateHashes=summary["templateHashes"],
                expectedArchiveSha256=summary["archive"]["sha256"],
                expectedArchiveBytes=summary["archive"]["sizeBytes"])
        except (ClientAssetImportError, KeyError, TypeError) as error:
            raise HTTPException(409, detail={"code": "invalid_client_asset",
                "message": "The saved client asset did not pass package validation.",
                "retryable": False}) from error

    @app.get("/v1/jobs/{jobId}/preview", tags=["client"], responses=errors)
    def previewJob(jobId: str):
        return buildPreview(verifiedAsset(jobId), jobId)

    @app.get("/v1/jobs/{jobId}/textures/{materialIndex}", tags=["client"], responses={**errors, 200: {
        "description": "Verified packaged base color texture", "content": {"image/png": {
            "schema": {"type": "string", "format": "binary"}}}}}, response_class=Response)
    def previewTexture(jobId: str, materialIndex: int):
        asset = verifiedAsset(jobId)
        materials = asset.materials["materials"]
        if not 0 <= materialIndex < len(materials):
            raise HTTPException(404, detail={"code": "not_found", "message": "Unknown material.",
                                              "retryable": False})
        texturePath = materials[materialIndex]["baseColorTexture"]["path"]
        identity = asset.manifest["files"][texturePath]
        return Response(asset.payloads[texturePath], media_type="image/png",
                        headers={"ETag": '"' + identity["sha256"] + '"',
                                 "X-Content-Type-Options": "nosniff"})
