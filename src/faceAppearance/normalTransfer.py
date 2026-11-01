"""Transfer tangent normals through a shared geometry coordinate system on CPU.

Atlas arrays use top-left/V-down coordinates. Tangent derivatives are always
computed from OBJ-style V-up UVs: UFF native V-down UVs are explicitly converted
before derivatives, whereas ICT OBJ UVs are converted only for rasterization.

The basis is per-face UV derivatives with corner/texel Gram-Schmidt; it is NOT
claimed to be MikkTSpace or the undocumented basis of the WildCap training data.
WildCap's run_wildcap.py generates normal channels but its photometric calls use
object_normal_lr_4k, not those channels (lines 393-398, 438). Its mesh renderer
only interpolates vertex normals. The generic, unused split-sum shading-normal
helper cannot establish the saved normal maps' convention. Therefore callers
MUST supply sourceConvention; source='wildcap' is intentionally unsupported.

Primary-source cross-check: DoRA's paper (arxiv.org/html/2506.03478v1,
sections 4.1 and 6.6) describes ICT-registered tangent normal maps but does not
specify their green sign or tangent generator. github.com/yxuhan/DoRA currently
publishes a release plan rather than that renderer/training implementation.
WildCap (arxiv.org/html/2512.11237v2, section 3.2) explicitly shades using coarse
geometry normals while predicting detailed normals through its prior.

Meshes, geometry normals, and both atlases must refer to the same object frame.
This module neither estimates nor silently applies an ICT/UFF global transform.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .texture import buildUvLookup, samplePixels, validateTriangles, vertexNormals


def normalizeVectors(vectors: np.ndarray, epsilon: float = 1e-12
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Normalize finite vectors and return validity, leaving zero vectors zero."""
    vectors = np.asarray(vectors, dtype=np.float64)
    lengths = np.linalg.norm(vectors, axis=-1)
    validMask = np.isfinite(vectors).all(axis=-1) & (lengths > epsilon)
    normalized = np.zeros_like(vectors)
    np.divide(vectors, lengths[..., None], out=normalized, where=validMask[..., None])
    return normalized, validMask


def convertUvs(uvs: np.ndarray, *, sourceOrigin: str, targetOrigin: str) -> np.ndarray:
    """Convert only the V axis; never interpret an image flip as a normal flip."""
    if sourceOrigin not in {"vUp", "vDown"} or targetOrigin not in {"vUp", "vDown"}:
        raise ValueError("UV origin must be explicitly 'vUp' or 'vDown'")
    uvs = np.asarray(uvs, dtype=np.float64)
    if uvs.ndim != 2 or uvs.shape[1] != 2 or not np.isfinite(uvs).all():
        raise ValueError("Expected finite UV coordinates shaped (N,2)")
    result = uvs.copy()
    if sourceOrigin != targetOrigin:
        result[:, 1] = 1 - result[:, 1]
    return result


def buildCornerTangentFrames(vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray,
                            uvFaces: np.ndarray, *, uvOrigin: str,
                            normalMode: str = "smooth", cornerNormals: np.ndarray | None = None
                            ) -> dict[str, Any]:
    """Build (face, corner, XYZ, TBN-column) frames preserving mirrored UVs.

    T points along increasing OBJ U. B points along increasing OBJ V after
    orthogonalization. ``handedness`` is sign(dot(cross(N,T), dP/dV)); B equals
    that sign times cross(N,T), so mirrored UV islands are not silently flipped.
    Degenerate geometry/UVs or zero corner normals invalidate the whole face.

    Smooth defaults to area-weighted vertex normals. Supply cornerNormals with
    shape (F,3,3) to match a specific renderer's normals or hard-edge splits.
    Tangents are deliberately not averaged across UV seams or mirrored islands.
    """
    vertices, faces = validateTriangles(vertices, faces, 3)
    uvs, uvFaces = validateTriangles(uvs, uvFaces, 2)
    if len(uvFaces) != len(faces):
        raise ValueError("Geometry and UV faces must correspond in the same order")
    if normalMode not in {"smooth", "flat"}:
        raise ValueError("normalMode must be 'smooth' or 'flat'")
    objUvs = convertUvs(uvs, sourceOrigin=uvOrigin, targetOrigin="vUp")
    triangles = vertices[faces]
    uvTriangles = objUvs[uvFaces]
    firstEdge = triangles[:, 1] - triangles[:, 0]
    secondEdge = triangles[:, 2] - triangles[:, 0]
    firstUvEdge = uvTriangles[:, 1] - uvTriangles[:, 0]
    secondUvEdge = uvTriangles[:, 2] - uvTriangles[:, 0]
    determinants = firstUvEdge[:, 0] * secondUvEdge[:, 1] - firstUvEdge[:, 1] * secondUvEdge[:, 0]
    faceNormals, geometryValid = normalizeVectors(np.cross(firstEdge, secondEdge))
    faceValid = geometryValid & (np.abs(determinants) > 1e-12)
    tangentDerivative = np.zeros_like(firstEdge)
    bitangentDerivative = np.zeros_like(firstEdge)
    tangentNumerator = firstEdge * secondUvEdge[:, 1:2] - secondEdge * firstUvEdge[:, 1:2]
    bitangentNumerator = secondEdge * firstUvEdge[:, 0:1] - firstEdge * secondUvEdge[:, 0:1]
    np.divide(tangentNumerator, determinants[:, None], out=tangentDerivative, where=faceValid[:, None])
    np.divide(bitangentNumerator, determinants[:, None], out=bitangentDerivative, where=faceValid[:, None])

    if cornerNormals is None:
        if normalMode == "smooth":
            normalCorners = vertexNormals(vertices, faces)[faces]
        else:
            normalCorners = np.repeat(faceNormals[:, None, :], 3, axis=1)
    else:
        normalCorners = np.asarray(cornerNormals, dtype=np.float64)
        if normalCorners.shape != (len(faces), 3, 3) or not np.isfinite(normalCorners).all():
            raise ValueError("cornerNormals must be finite and shaped (F,3,3)")
    normalCorners, normalValid = normalizeVectors(normalCorners)
    tangents = np.broadcast_to(tangentDerivative[:, None], normalCorners.shape)
    tangents = tangents - np.sum(tangents * normalCorners, axis=-1, keepdims=True) * normalCorners
    tangents, tangentValid = normalizeVectors(tangents)
    crossVectors = np.cross(normalCorners, tangents)
    orientation = np.sum(crossVectors * bitangentDerivative[:, None], axis=-1)
    signs = np.sign(orientation)
    cornerValid = normalValid & tangentValid & (np.abs(orientation) > 1e-12)
    # Conflicting normals/handedness within a face cannot define one atlas frame.
    faceValid &= cornerValid.all(axis=1) & (signs == signs[:, :1]).all(axis=1)
    bitangents = crossVectors * signs[..., None]
    frames = np.stack([tangents, bitangents, normalCorners], axis=-1)
    frames[~faceValid] = 0
    handedness = np.where(faceValid, signs[:, 0], 0).astype(np.int8)
    return {"cornerFrames": frames.astype(np.float32), "faceValidMask": faceValid,
            "faceHandedness": handedness, "uvOrigin": uvOrigin,
            "basisDefinition": "perFaceObjUvDerivativesGramSchmidt; positiveU positiveV normal",
            "normalMode": "explicitCornerNormals" if cornerNormals is not None else normalMode}


def rasterizeTangentFrames(cornerFrames: dict[str, Any], uvs: np.ndarray, uvFaces: np.ndarray,
                          resolution: int, *, uvOrigin: str, lookup: dict | None = None,
                          validFaces: np.ndarray | None = None) -> dict[str, Any]:
    """Rasterize and re-orthogonalize per-corner TBN, with top-left array origin."""
    rasterUvs = convertUvs(uvs, sourceOrigin=uvOrigin, targetOrigin="vDown")
    _, uvFaces = validateTriangles(rasterUvs, uvFaces, 2)
    if uvOrigin != cornerFrames["uvOrigin"]:
        raise ValueError("Raster UV origin differs from the corner frame definition")
    frames = np.asarray(cornerFrames["cornerFrames"], dtype=np.float64)
    faceValid = np.asarray(cornerFrames["faceValidMask"], dtype=bool).copy()
    handedness = np.asarray(cornerFrames["faceHandedness"], dtype=np.int8)
    if frames.shape != (len(uvFaces), 3, 3, 3) or faceValid.shape != (len(uvFaces),):
        raise ValueError("Corner frames do not match UV faces")
    if validFaces is not None:
        validFaces = np.asarray(validFaces)
        if validFaces.shape != faceValid.shape or validFaces.dtype != np.bool_:
            raise ValueError("validFaces must be a boolean mask for every face")
        faceValid &= validFaces
    lookup = buildUvLookup(rasterUvs, uvFaces, resolution) if lookup is None else lookup
    faceIds = np.asarray(lookup["faceIds"])
    validMask = np.asarray(lookup["validMask"], dtype=bool).copy()
    barycentrics = np.asarray(lookup["barycentrics"], dtype=np.float64)
    if (faceIds.shape != (resolution, resolution) or validMask.shape != faceIds.shape
            or barycentrics.shape != (*faceIds.shape, 3)):
        raise ValueError("UV lookup dimensions do not match requested resolution")
    if ((faceIds[validMask] < 0).any() or (faceIds[validMask] >= len(frames)).any()
            or not np.isfinite(barycentrics).all()
            or not np.allclose(barycentrics[validMask].sum(axis=-1), 1, atol=1e-5)
            or (barycentrics[validMask] < -1e-6).any()):
        raise ValueError("UV lookup contains invalid face indices or barycentrics")
    validMask[validMask] &= faceValid[faceIds[validMask]]
    ids = faceIds[validMask]
    interpolated = np.sum(frames[ids] * barycentrics[validMask, :, None, None], axis=1)
    normals, normalValid = normalizeVectors(interpolated[..., 2])
    tangents = interpolated[..., 0]
    tangents -= np.sum(tangents * normals, axis=-1, keepdims=True) * normals
    tangents, tangentValid = normalizeVectors(tangents)
    bitangents = np.cross(normals, tangents) * handedness[ids, None]
    pixelValid = normalValid & tangentValid
    pixelFrames = np.stack([tangents, bitangents, normals], axis=-1)
    pixelFrames[~pixelValid] = 0
    atlasFrames = np.zeros((resolution, resolution, 3, 3), np.float32)
    atlasFrames[validMask] = pixelFrames
    pixelHandedness = np.zeros((resolution, resolution), np.int8)
    pixelHandedness[validMask] = np.where(pixelValid, handedness[ids], 0)
    validMask[validMask] &= pixelValid
    return {"frames": atlasFrames, "validMask": validMask, "handedness": pixelHandedness,
            "faceIds": faceIds.copy(), "arrayOrigin": "vDown", "derivativeOrigin": "vUp",
            "basisDefinition": cornerFrames["basisDefinition"], "normalMode": cornerFrames["normalMode"]}


def buildTangentAtlas(vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray,
                      uvFaces: np.ndarray, resolution: int, *, uvOrigin: str,
                      normalMode: str = "smooth", cornerNormals: np.ndarray | None = None,
                      lookup: dict | None = None, validFaces: np.ndarray | None = None) -> dict[str, Any]:
    """Convenience mesh-to-atlas entry point; no weights, torch or GPU required."""
    corners = buildCornerTangentFrames(vertices, faces, uvs, uvFaces, uvOrigin=uvOrigin,
                                       normalMode=normalMode, cornerNormals=cornerNormals)
    return rasterizeTangentFrames(corners, uvs, uvFaces, resolution, uvOrigin=uvOrigin,
                                  lookup=lookup, validFaces=validFaces)


def normalYSign(convention: str) -> float:
    """Require the source's green-channel meaning; never infer it from a name."""
    signs = {"positiveV": 1.0, "negativeV": -1.0, "openGL": 1.0, "directX": -1.0}
    if convention not in signs:
        raise ValueError("Normal convention must be explicitly positiveV/openGL or negativeV/directX; "
                         "WildCap's source convention is not verified")
    return signs[convention]


def tangentToObject(normals: np.ndarray, frames: np.ndarray, *, convention: str) -> np.ndarray:
    """Decode signed tangent vectors using TBN columns in a shared object frame."""
    normals, frames = np.asarray(normals, np.float64), np.asarray(frames, np.float64)
    if normals.shape[-1:] != (3,) or frames.shape != (*normals.shape[:-1], 3, 3):
        raise ValueError("Normal and frame dimensions differ")
    localNormals = normals * np.array([1, normalYSign(convention), 1])
    objectNormals = np.einsum("...ij,...j->...i", frames, localNormals)
    return normalizeVectors(objectNormals)[0].astype(np.float32)


def objectToTangent(normals: np.ndarray, frames: np.ndarray, *, convention: str) -> np.ndarray:
    """Encode object vectors with the transpose of the orthonormal TBN frame."""
    normals, frames = np.asarray(normals, np.float64), np.asarray(frames, np.float64)
    if normals.shape[-1:] != (3,) or frames.shape != (*normals.shape[:-1], 3, 3):
        raise ValueError("Normal and frame dimensions differ")
    localNormals = np.einsum("...ij,...i->...j", frames, normals)
    localNormals *= np.array([1, normalYSign(convention), 1])
    return normalizeVectors(localNormals)[0].astype(np.float32)


def transferTangentNormals(normalMap: np.ndarray, sourceCoverage: np.ndarray, atlasMap: dict,
                           sourceFrames: dict, targetFrames: dict, *, sourceConvention: str,
                           targetConvention: str = "positiveV", inputEncoding: str = "signed"
                           ) -> dict[str, Any]:
    """Tangent -> object -> mapped UV sample -> destination tangent, preserving support.

    atlasMap follows atlasTransfer.compileAtlasMap: each destination texel stores
    its source UV in V-down image coordinates. Source and target frames must be
    built on meshes in the SAME geometry frame, with matching image resolutions.
    ``signed`` input uses [-1,1]; ``unorm`` explicitly decodes a [0,1] RGB normal
    image as 2*RGB-1. Returned tangent/object vectors are signed, normalized, and
    zero outside validMask. No coverage hole is filled with an invented normal.

    A caller may explicitly select an experimental WildCap convention; this
    mathematical transfer does not establish that the selection is correct.
    """
    normalYSign(sourceConvention)
    normalYSign(targetConvention)
    normalMap = np.asarray(normalMap, dtype=np.float64)
    sourceCoverage = np.asarray(sourceCoverage)
    sourceBasis = np.asarray(sourceFrames["frames"])
    sourceFrameMask = np.asarray(sourceFrames["validMask"], dtype=bool)
    targetBasis = np.asarray(targetFrames["frames"])
    targetFrameMask = np.asarray(targetFrames["validMask"], dtype=bool)
    mappingMask = np.asarray(atlasMap["validMask"], dtype=bool)
    mappedUvs = np.asarray(atlasMap["sourceUvMap"])
    if (normalMap.ndim != 3 or normalMap.shape[-1] != 3 or min(normalMap.shape[:2]) < 2
            or sourceCoverage.shape != normalMap.shape[:2]
            or sourceFrameMask.shape != sourceCoverage.shape
            or sourceBasis.shape != (*sourceCoverage.shape, 3, 3)):
        raise ValueError("Source normal, coverage and tangent-frame dimensions differ")
    if (mappingMask.ndim != 2 or targetFrameMask.shape != mappingMask.shape
            or targetBasis.shape != (*mappingMask.shape, 3, 3)
            or mappedUvs.shape != (*mappingMask.shape, 2)):
        raise ValueError("Destination atlas mapping and tangent-frame dimensions differ")
    if not all(np.isfinite(value).all() for value in (normalMap, sourceCoverage, sourceBasis,
                                                    targetBasis, mappedUvs)):
        raise ValueError("Normal transfer inputs must be finite")
    if (sourceCoverage < 0).any() or (sourceCoverage > 1).any():
        raise ValueError("sourceCoverage must be within [0,1]")
    if inputEncoding == "unorm":
        if (normalMap < 0).any() or (normalMap > 1).any():
            raise ValueError("Unorm normal images must be within [0,1]")
        normalMap = 2 * normalMap - 1
    elif inputEncoding != "signed":
        raise ValueError("inputEncoding must be 'signed' or 'unorm'")
    elif (np.abs(normalMap) > 1 + 1e-6).any():
        raise ValueError("Signed normal images must be within [-1,1]")
    normalizedInput, normalValid = normalizeVectors(normalMap)
    sourceMask = (sourceCoverage >= 0.999) & sourceFrameMask & normalValid
    sourceObject = tangentToObject(normalizedInput, sourceBasis, convention=sourceConvention)
    sourceObject[~sourceMask] = 0
    candidateMask = mappingMask & targetFrameMask & (mappedUvs >= 0).all(axis=-1) & (mappedUvs <= 1).all(axis=-1)
    sourcePixels = mappedUvs[candidateMask] * [normalMap.shape[1] - 1, normalMap.shape[0] - 1]
    sampledSupport = samplePixels(sourceMask.astype(np.float32), sourcePixels) >= 0.999
    sampledObject, sampledNormalValid = normalizeVectors(samplePixels(sourceObject, sourcePixels))
    sampledSupport &= sampledNormalValid
    sampledTangent = objectToTangent(sampledObject, targetBasis[candidateMask], convention=targetConvention)
    sampledTangent[~sampledSupport] = 0
    sampledObject[~sampledSupport] = 0
    targetNormals = np.zeros((*mappingMask.shape, 3), np.float32)
    objectNormals = np.zeros_like(targetNormals)
    validMask = np.zeros(mappingMask.shape, dtype=bool)
    targetNormals[candidateMask] = sampledTangent
    objectNormals[candidateMask] = sampledObject
    validMask[candidateMask] = sampledSupport
    return {"tangentNormals": targetNormals, "objectNormals": objectNormals, "validMask": validMask,
            "diagnostics": {"sourceConvention": sourceConvention, "targetConvention": targetConvention,
                            "inputEncoding": inputEncoding, "outputEncoding": "signed",
                            "sourceValidTexels": int(sourceMask.sum()), "mappedValidTexels": int(validMask.sum()),
                            "basisDefinition": "perFaceObjUvDerivativesGramSchmidt; not certified MikkTSpace",
                            "wildCapSourceConventionVerified": False}}
