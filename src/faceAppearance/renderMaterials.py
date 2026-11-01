"""Prepare saved reconstruction materials for an explicitly experimental render.

No model inference, GPU allocation, rendering or file writes occur here. Both
meshes stay in the reconstructed UFF frame. PyTorch3D supplies smooth geometry
normals; per-face OBJ UV derivatives supply T/B directions. PyTorch3D does not
thereby certify the tangent basis or green-channel convention of WildCap's prior.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .atlasTransfer import transferTexture
from .normalTransfer import buildCornerTangentFrames, convertUvs, rasterizeTangentFrames, transferTangentNormals
from .registration import RegistrationAssets
from .texture import validateTriangles


def pytorch3dVertexNormals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Get exactly Meshes.verts_normals_packed() on CPU, with original indices."""
    import torch
    from pytorch3d.structures import Meshes

    vertices, faces = validateTriangles(vertices, faces, 3)
    with torch.no_grad():
        mesh = Meshes(verts=[torch.tensor(vertices, dtype=torch.float32, device="cpu")],
                      faces=[torch.tensor(faces, dtype=torch.int64, device="cpu")])
        return mesh.verts_normals_packed().detach().numpy().copy()


def buildRenderBasis(geometry: dict, resolution: int, *, uvOrigin: str) -> dict:
    """Return renderer corner TBN and its matching atlas, using PyTorch3D N.

    cornerFrames has shape (F,3,3,3): face, corner, XYZ, T/B/N column.
    A renderer must barycentrically interpolate these corners, normalize N,
    Gram-Schmidt T against N, then form B=handedness*cross(N,T), as the atlas
    rasterizer does. Do not linearly blend TBN without re-orthogonalization.
    """
    faces = np.asarray(geometry["faces"])
    uvFaces = geometry.get("uvFaces", faces)
    normals = pytorch3dVertexNormals(geometry["vertices"], faces)
    corners = buildCornerTangentFrames(geometry["vertices"], faces, geometry["uvs"], uvFaces,
                                       uvOrigin=uvOrigin, cornerNormals=normals[faces])
    atlas = rasterizeTangentFrames(corners, geometry["uvs"], uvFaces, resolution,
                                   uvOrigin=uvOrigin, validFaces=geometry.get("validFaceMask"))
    return {"vertexNormals": normals, "corners": corners, "atlas": atlas}


def transferRenderNormal(sourceGeometry: dict, targetGeometry: dict, sourceNormal: np.ndarray,
                         sourceCoverage: np.ndarray, atlasMap: dict, *, sourceConvention: str) -> dict:
    """ICT OBJ V-up signed normals -> UFF native V-down, target OpenGL tangent.

    Geometry inputs must be in the same reconstructed UFF object frame. The
    caller explicitly chooses an experimental openGL or directX source meaning.
    Neutral normals outside validMask are render fallbacks, never observations.
    Source and destination square atlas resolutions can differ (e.g. 512/1024).
    """
    if sourceConvention not in {"openGL", "directX"}:
        raise ValueError("Choose the experimental sourceConvention 'openGL' or 'directX'")
    sourceNormal = np.asarray(sourceNormal)
    targetShape = np.shape(atlasMap["validMask"])
    if (sourceNormal.ndim != 3 or sourceNormal.shape[-1] != 3
            or sourceNormal.shape[0] != sourceNormal.shape[1]
            or len(targetShape) != 2 or targetShape[0] != targetShape[1]):
        raise ValueError("Normal transfer requires square source RGB and destination atlases")
    sourceBasis = buildRenderBasis(sourceGeometry, sourceNormal.shape[0], uvOrigin="vUp")
    targetBasis = buildRenderBasis(targetGeometry, targetShape[0], uvOrigin="vDown")
    transferred = transferTangentNormals(sourceNormal, sourceCoverage, atlasMap,
                                         sourceBasis["atlas"], targetBasis["atlas"],
                                         sourceConvention=sourceConvention, targetConvention="openGL",
                                         inputEncoding="signed")
    normals = transferred["tangentNormals"].copy()
    normals[~transferred["validMask"]] = [0, 0, 1]
    return {"tangentNormal": normals, "validMask": transferred["validMask"],
            "objectNormal": transferred["objectNormals"],
            "targetVertexNormals": targetBasis["vertexNormals"],
            "targetCornerFrames": targetBasis["corners"]["cornerFrames"],
            "targetFaceHandedness": targetBasis["corners"]["faceHandedness"],
            "targetFaceValidMask": targetBasis["corners"]["faceValidMask"],
            "metadata": {**transferred["diagnostics"], "status": "experimentalRenderPreparation",
                         "sourceConventionIsAssumed": True, "sourceConventionVerified": False,
                         "geometryNormals": "pytorch3d.structures.Meshes.verts_normals_packed; CPU float32",
                         "basis": "perFaceObjUvDerivativesGramSchmidtWithPytorch3dSmoothNormals",
                         "sourceUvOrigin": "vUp", "targetUvOrigin": "vDown",
                         "sourceResolution": sourceNormal.shape[0], "targetResolution": targetShape[0],
                         "invalidNormalFallback": [0, 0, 1], "modelInferenceRepeated": False}}


def prepareRenderMaterials(projectRoot: Path, resultDir: Path,
                           sourceConvention: str = "openGL") -> dict:
    """Read saved pipeline arrays; prepare UFF maps without changing the run.

    baseAlbedo retains native OpenDelight values. wildCapAlbedo replaces the
    qualified region with gamma2.2 WildCap diffuse resampled in linear light;
    its fallback remains native and the composite is explicitly a preview.
    Both geometry.uvs (native V-down) and geometry.objUvs (V-up for TexturesUV)
    are returned. All image arrays use top-left row order.
    """
    projectRoot = Path(projectRoot).resolve()
    resultDir = Path(resultDir)
    resultDir = (resultDir if resultDir.is_absolute() else projectRoot / resultDir).resolve()
    geometryPath, texturePath = resultDir / "geometry.npz", resultDir / "texture.npz"
    materialPath = resultDir / "wildcapIct/materials.npz"
    with np.load(geometryPath, allow_pickle=False) as data:
        geometry = {key: data[key] for key in ("vertices", "faces", "uvs")}
    with np.load(texturePath, allow_pickle=False) as data:
        baseAlbedo, baseCoverage = data["albedo"], data["coverageMask"]
        basePreview = data["previewAlbedo"] if "previewAlbedo" in data.files else baseAlbedo.copy()
    with np.load(materialPath, allow_pickle=False) as data:
        materials = {key: data[key] for key in ("diffuse", "specular", "tangentNormal", "validMask")}
    if (baseAlbedo.ndim != 3 or baseAlbedo.shape[-1] != 3
            or baseAlbedo.shape[0] != baseAlbedo.shape[1] or baseCoverage.shape != baseAlbedo.shape[:2]
            or not np.isfinite(baseAlbedo).all()):
        raise ValueError("Saved UFF base texture/coverage has incompatible dimensions or values")
    configPath = resultDir / "config.json"
    config = json.loads(configPath.read_text()) if configPath.exists() else {}
    assetRoot = projectRoot / config.get("registrationAsset", "assets/registration/wrapV1")
    assets = RegistrationAssets(assetRoot, baseAlbedo.shape[0])
    reportPath = resultDir / "result.json"
    if reportPath.exists():
        report = json.loads(reportPath.read_text())
        if report.get("status") != "succeeded":
            raise ValueError("Render preparation requires a successfully completed saved run")
        sourceHash = report.get("wildCapDiagnostics", {}).get("registrationResultSha256")
        if sourceHash is not None and sourceHash != assets.manifest["resultSha256"]:
            raise ValueError("Saved materials use a different registered ICT surface")
    ictGeometry = assets.toIctGeometry(geometry)
    targetGeometry = {**geometry, "validFaceMask": assets.geometry["uffValidFaces"]}
    mapping = assets.atlasMap("ictToUff")
    normal = transferRenderNormal(ictGeometry, targetGeometry, materials["tangentNormal"],
                                  materials["validMask"], mapping, sourceConvention=sourceConvention)
    diffuse = transferTexture(materials["diffuse"], materials["validMask"], mapping,
                              materialKind="color", colorEncoding="gamma2.2")
    specular = transferTexture(materials["specular"], materials["validMask"], mapping,
                               materialKind="scalar")
    materialMask = diffuse["validMask"] & specular["validMask"]
    if not materialMask.any() or not normal["validMask"].any():
        raise ValueError("No valid material/normal pixels survived experimental render preparation")
    composite = baseAlbedo.copy()
    composite[materialMask] = diffuse["texture"][materialMask]
    outputGeometry = {**geometry, "uvFaces": geometry["faces"].copy(),
                      "objUvs": convertUvs(geometry["uvs"], sourceOrigin="vDown", targetOrigin="vUp"),
                      "vertexNormals": normal["targetVertexNormals"],
                      "cornerTangentFrames": normal["targetCornerFrames"],
                      "faceTangentHandedness": normal["targetFaceHandedness"],
                      "faceTangentValidMask": normal["targetFaceValidMask"],
                      "geometryFrame": "uvFaceFusionCanonical", "uvOrigin": "vDown"}
    metadata = {**normal["metadata"], "geometryFrame": "uvFaceFusionCanonical",
                "registrationResultSha256": assets.manifest["resultSha256"],
                "inputSha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (geometryPath, texturePath, materialPath)},
                "diffuseTransfer": diffuse["diagnostics"], "specularTransfer": specular["diagnostics"],
                "baseColorEncoding": "OpenDelightNative",
                "wildCapColorEncoding": "gamma2.2 on materialValidMask; native fallback; preview only",
                "normalValidPixelCount": int(normal["validMask"].sum()),
                "materialValidPixelCount": int(materialMask.sum()), "specularFallback": 0,
                "basePreviewHoleFill": "saved nearest-observation preview; not measured",
                "warnings": ["WildCapSourceTangentConventionAndBakerAreUnverified",
                             "Pytorch3dGeometryNormalsDoNotCertifyWildCapTangentBasis",
                             "RoughnessAndSubsurfaceScatteringAreNotEstimated"]}
    return {"geometry": outputGeometry, "ictGeometry": ictGeometry,
            "baseAlbedo": baseAlbedo, "basePreviewAlbedo": basePreview,
            "wildCapAlbedo": composite, "diffuse": diffuse["texture"], "specular": specular["texture"],
            "tangentNormal": normal["tangentNormal"], "objectNormal": normal["objectNormal"],
            "normalValidMask": normal["validMask"], "materialValidMask": materialMask,
            "baseCoverageMask": baseCoverage, "metadata": metadata}
