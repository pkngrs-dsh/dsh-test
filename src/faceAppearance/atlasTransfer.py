"""Offline atlas compilation and masked scalar/color texture resampling.

Array UVs always use a top-left origin. ICT OBJ UVs are converted explicitly.
Tangent-space normals must not be passed to transferTexture: they require TBN
basis conversion, which is a separate operation.
"""

import numpy as np

from .correspondence import SurfaceIndex, qualifyMapping
from .texture import buildUvLookup, interpolateSurface, samplePixels, vertexNormals


def compileAtlasMap(sourceVertices: np.ndarray, sourceFaces: np.ndarray, sourceUvs: np.ndarray,
                    sourceUvFaces: np.ndarray, targetVertices: np.ndarray, targetFaces: np.ndarray,
                    targetUvs: np.ndarray, targetUvFaces: np.ndarray, resolution: int,
                    sourceValidFaces: np.ndarray, targetValidFaces: np.ndarray,
                    maxDistance: float = 0.01, minNormalDot: float = 0.5) -> dict:
    """Compile destination texels → source image UV using registered surfaces."""
    lookup = buildUvLookup(targetUvs, targetUvFaces, resolution)
    atlasMask = lookup["validMask"]
    targetMask = atlasMask.copy()
    targetMask[atlasMask] &= np.asarray(targetValidFaces, bool)[lookup["faceIds"][atlasMask]]
    surface = interpolateSurface(targetVertices, targetFaces, lookup)
    normals = interpolateSurface(vertexNormals(targetVertices, targetFaces), targetFaces, lookup)
    queryPoints = surface[targetMask]
    index = SurfaceIndex(sourceVertices, sourceFaces)
    mapping = qualifyMapping(index.query(queryPoints), index, normals[targetMask], maxDistance, minNormalDot)
    qualified = mapping["validMask"] & np.asarray(sourceValidFaces, bool)[mapping["triangleIds"]]
    mappedUvs = np.sum(np.asarray(sourceUvs)[sourceUvFaces[mapping["triangleIds"]]]
                       * mapping["barycentricWeights"][..., None], axis=1)
    uvMap = np.zeros((resolution, resolution, 2), np.float32)
    validMask = np.zeros((resolution, resolution), bool)
    distanceMap = np.zeros((resolution, resolution), np.float32)
    uvMap[targetMask] = mappedUvs
    validMask[targetMask] = qualified
    distanceMap[targetMask] = mapping["distances"]
    return {"sourceUvMap": uvMap, "validMask": validMask, "distanceMap": distanceMap,
            "atlasMask": atlasMask, "targetFaceIds": lookup["faceIds"]}


def transferTexture(texture: np.ndarray, sourceCoverage: np.ndarray, atlasMap: dict,
                    *, materialKind: str = "color", colorEncoding: str = "native") -> dict:
    """Resample color/scalar data and preserve missing-data support exactly.

    native preserves the caller's numerical encoding, including unknown
    OpenDelight values. gamma2.2 explicitly decodes colors before bilinear
    sampling, then restores that encoding. Scalars always remain linear data.
    """
    if materialKind not in {"color", "scalar"}:
        raise ValueError("Only color/scalar data; tangent normals require a basis transformation")
    if colorEncoding not in {"native", "gamma2.2"}:
        raise ValueError("colorEncoding must be 'native' or 'gamma2.2'")
    if materialKind == "scalar" and colorEncoding != "native":
        raise ValueError("Scalar data must use native encoding; gamma2.2 is only for color")
    texture, sourceCoverage = np.asarray(texture), np.asarray(sourceCoverage)
    if texture.ndim not in (2, 3) or sourceCoverage.shape != texture.shape[:2]:
        raise ValueError("Texture and coverage dimensions differ")
    if not np.isfinite(texture).all() or not np.isfinite(sourceCoverage).all():
        raise ValueError("Texture/coverage must be finite")
    if ((sourceCoverage < 0).any() or (sourceCoverage > 1).any()):
        raise ValueError("Coverage must be within [0,1]")
    sampleTexture = texture.astype(np.float32)
    if colorEncoding == "gamma2.2":
        if (sampleTexture < 0).any():
            raise ValueError("gamma2.2 color values must be nonnegative")
        sampleTexture = np.power(sampleTexture, 2.2)
    targetShape = atlasMap["validMask"].shape
    sourceUvs = atlasMap["sourceUvMap"][atlasMap["validMask"]]
    positions = sourceUvs * [texture.shape[1] - 1, texture.shape[0] - 1]
    support = samplePixels(sourceCoverage.astype(np.float32), positions) >= 0.999
    values = samplePixels(sampleTexture, positions)
    if colorEncoding == "gamma2.2":
        values = np.power(values, 1 / 2.2)
    result = np.zeros((*targetShape, *texture.shape[2:]), np.float32)
    validMask = np.zeros(targetShape, bool)
    values[~support] = 0
    result[atlasMap["validMask"]] = values
    validMask[atlasMap["validMask"]] = support
    return {"texture": result, "validMask": validMask,
            "diagnostics": {"materialKind": materialKind, "colorEncoding": colorEncoding,
                            "interpolationSpace": "linearLight" if colorEncoding == "gamma2.2" else "native"}}
