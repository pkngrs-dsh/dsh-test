"""Offline UV correction using the same legacy samples as skin deformation.

The field is a per-vertex offset to existing per-corner UVs, so original seams
survive. Unsupported or folded corrected triangles are excluded, not filled.
This reconciles two old correspondence assets; it does not certify anatomy.
"""

import numpy as np

from .texture import buildUvLookup, samplePixels


def signedDoubleAreas(corners):
    first, second = corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0]
    return first[:, 0]*second[:, 1]-first[:, 1]*second[:, 0]


def compileUvCorrection(arrays: dict, skinUvs: np.ndarray, skinUvFaces: np.ndarray,
                        samples: dict, deformer) -> tuple[dict, dict]:
    """Return replacement atlas lookup arrays plus diagnostics; never mutate input.

    deformer supplies a preloaded ``transferValues`` surface-sample solver. It
    must use the same NICP faces, sample rows and static mask as ``samples``.
    UVs use top-left V-down. The field and residuals are dimensionless UV units.
    """
    faces = arrays["nicpFaces"]
    skinUvs, skinUvFaces = np.asarray(skinUvs), np.asarray(skinUvFaces)
    if (skinUvs.ndim != 2 or skinUvs.shape[1] != 2 or skinUvs.dtype.kind != "f"
            or not np.isfinite(skinUvs).all() or skinUvFaces.shape != faces.shape
            or skinUvFaces.dtype.kind not in "iu" or np.any(skinUvFaces < 0) or np.any(skinUvFaces >= len(skinUvs))):
        raise ValueError("Expected finite native UVs and independent UV faces matching NICP topology")
    corners = skinUvs[skinUvFaces]
    sampleFaceIds, bary = np.asarray(samples["sampleNicpFaceIds"]), np.asarray(samples["sampleNicpWeights"])
    sampleUvs, sampleWeights = np.asarray(samples["sampleUvs"]), np.asarray(samples["sampleWeights"])
    if (sampleFaceIds.ndim != 1 or sampleFaceIds.dtype.kind not in "iu" or not len(sampleFaceIds)
            or np.any(sampleFaceIds < 0) or np.any(sampleFaceIds >= len(faces))
            or bary.shape != (len(sampleFaceIds), 3) or bary.dtype.kind != "f" or not np.isfinite(bary).all()
            or np.any(bary < 0) or not np.allclose(bary.sum(axis=1), 1., rtol=0, atol=1e-6)
            or sampleUvs.shape != (len(sampleFaceIds), 2) or sampleUvs.dtype.kind != "f" or not np.isfinite(sampleUvs).all()
            or sampleWeights.shape != sampleFaceIds.shape or sampleWeights.dtype.kind != "f"
            or not np.isfinite(sampleWeights).all() or np.any(sampleWeights < 0) or not np.any(sampleWeights > 0)):
        raise ValueError("Invalid UV surface constraints")
    if (not np.array_equal(deformer.faces, faces) or not np.array_equal(deformer.staticMask, arrays["staticMask"])
            or not np.array_equal(deformer.sampleFaceIds, sampleFaceIds)
            or not np.array_equal(deformer.sampleBarycentrics, bary)
            or not np.array_equal(deformer.sampleWeights, sampleWeights)):
        raise ValueError("UV correction must use the matching preloaded surface-sample system")
    current = (corners[sampleFaceIds]*bary[..., None]).sum(axis=1)
    desiredDelta = sampleUvs-current
    correction = deformer.transferValues(np.zeros((len(arrays["nicpToFullIds"]), 2)), desiredDelta)
    correctedCorners = corners+correction[faces]
    beforeAreas, afterAreas = signedDoubleAreas(corners), signedDoubleAreas(correctedCorners)
    areaRatio = np.divide(afterAreas, beforeAreas, out=np.zeros_like(afterAreas), where=np.abs(beforeAreas) > 1e-12)
    faceValid = (areaRatio >= .1) & (areaRatio <= 10.)
    # Match full template faces by their original oriented indices, just as in
    # the existing compiler. A full-mesh prefix assumption is never used.
    fullLookup = {tuple(face): index for index, face in enumerate(arrays["faces"])}
    nicpFullFaces = arrays["nicpToFullIds"][faces]
    nicpToFull = np.array([fullLookup.get(tuple(face), -1) for face in nicpFullFaces])
    if np.any(nicpToFull < 0) or len(np.unique(nicpToFull)) != len(faces):
        raise ValueError("NICP must be a unique oriented-face subset of the full template")
    height, width = arrays["atlasValidMask"].shape
    if height != width:
        raise ValueError("Prepared atlas must be square")
    headFaceIds = np.flatnonzero(arrays["materialIds"] == 0)
    lookup = buildUvLookup(arrays["uvs"], arrays["uvFaces"][headFaceIds], height)
    fullToNicp = np.full(len(arrays["faces"]), -1, np.int64)
    fullToNicp[nicpToFull] = np.arange(len(faces))
    mappedFaceIds = np.full((height, width), -1, np.int64)
    headMask = lookup["validMask"]
    mappedFaceIds[headMask] = fullToNicp[headFaceIds[lookup["faceIds"][headMask]]]
    valid = arrays["atlasValidMask"].copy() & (mappedFaceIds >= 0)
    valid[valid] &= faceValid[mappedFaceIds[valid]]
    correctedUvs = np.zeros((height, width, 2), np.float32)
    correctedUvs[valid] = (correctedCorners[mappedFaceIds[valid]]*lookup["barycentrics"][valid, :, None]).sum(axis=1)
    # Re-check actual UFF support after moving sample coordinates. Previous
    # support cannot be reused across holes or outside the native chart.
    sourceSupport = buildUvLookup(arrays["sourceUvs"], arrays["sourceFaces"], height)["validMask"]
    supported = samplePixels(sourceSupport.astype(np.float32), correctedUvs[valid]*(height-1)) >= .999
    valid[valid] = supported
    correctedUvs[~valid] = 0
    predictedDelta = (correction[faces[sampleFaceIds]]*bary[..., None]).sum(axis=1)
    supportedSamples = samples["sampleWeights"] > 0
    errorBefore = np.linalg.norm(desiredDelta[supportedSamples], axis=1)
    errorAfter = np.linalg.norm((desiredDelta-predictedDelta)[supportedSamples], axis=1)
    diagnostics = {"method": "legacyBarycentricUvOffsetField", "uvConvention": "vDown",
        "constraintCount": int(supportedSamples.sum()), "referenceResolution": height,
        "medianSampleErrorBeforeTexels": float(np.median(errorBefore)*(height-1)),
        "medianSampleErrorAfterTexels": float(np.median(errorAfter)*(height-1)),
        "p95SampleErrorBeforeTexels": float(np.percentile(errorBefore, 95)*(height-1)),
        "p95SampleErrorAfterTexels": float(np.percentile(errorAfter, 95)*(height-1)),
        "maxVertexCorrectionTexels": float(np.linalg.norm(correction, axis=1).max()*(height-1)),
        "baseEligibleAtlasTexels": int(arrays["atlasValidMask"].sum()), "correctedEligibleAtlasTexels": int(valid.sum()),
        "rejectedUvFaceCount": int((~faceValid).sum()),
        "staticVertexCorrectionsExactlyZero": bool(np.array_equal(correction[arrays["staticMask"]], np.zeros_like(correction[arrays["staticMask"]]))),
        "anatomicalAccuracyCertified": False}
    return {"atlasSourceUvs": correctedUvs, "atlasValidMask": valid,
            "uvCorrectionPerNicpVertex": correction, "correctedUvFaceValid": faceValid}, diagnostics
