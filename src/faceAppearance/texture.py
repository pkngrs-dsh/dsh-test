"""CPU UV lookup, depth-tested view projection and conservative albedo fusion.

UVFaceFusion's native UV arrays use V downwards, with (0, 0) at image top left.
An integer raster pixel center (x, y) corresponds to UV (x/(W-1), y/(H-1)).
No gamma conversion is performed: OpenDelight's native encoding is retained.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def validateTriangles(vertices: np.ndarray, faces: np.ndarray, dimensions: int) -> tuple[np.ndarray, np.ndarray]:
    vertices, faces = np.asarray(vertices, dtype=np.float64), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != dimensions or not np.isfinite(vertices).all():
        raise ValueError(f"Expected finite vertices shaped (N, {dimensions})")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise ValueError("Expected integer triangle indices shaped (F, 3)")
    if len(vertices) == 0 or len(faces) == 0 or np.min(faces) < 0 or np.max(faces) >= len(vertices):
        raise ValueError("Triangle indices must address a nonempty vertex array")
    return vertices, faces.astype(np.int64, copy=False)


def trianglePixels(triangle: np.ndarray, height: int, width: int
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Rasterize one projected triangle at integer pixel centers, either winding."""
    minX, minY = np.maximum(np.ceil(np.min(triangle, axis=0) - 1e-7), [0, 0]).astype(np.int64)
    maxX, maxY = np.minimum(np.floor(np.max(triangle, axis=0) + 1e-7), [width - 1, height - 1]).astype(np.int64)
    if minX > maxX or minY > maxY:
        return None
    first, second, third = triangle
    denominator = ((second[1] - third[1]) * (first[0] - third[0])
                   + (third[0] - second[0]) * (first[1] - third[1]))
    if abs(denominator) < 1e-12:
        return None
    ys, xs = np.mgrid[minY:maxY + 1, minX:maxX + 1]
    firstWeight = ((second[1] - third[1]) * (xs - third[0])
                   + (third[0] - second[0]) * (ys - third[1])) / denominator
    secondWeight = ((third[1] - first[1]) * (xs - third[0])
                    + (first[0] - third[0]) * (ys - third[1])) / denominator
    weights = np.stack([firstWeight, secondWeight, 1 - firstWeight - secondWeight], axis=-1)
    inside = np.min(weights, axis=-1) >= -1e-7
    if not inside.any():
        return None
    return ys[inside], xs[inside], weights[inside]


def buildUvLookup(uvs: np.ndarray, faces: np.ndarray, resolution: int) -> dict[str, np.ndarray]:
    """Map each UV pixel to its first covering triangle and vertex barycentrics."""
    uvs, faces = validateTriangles(uvs, faces, 2)
    if not isinstance(resolution, (int, np.integer)) or resolution < 2:
        raise ValueError("resolution must be an integer >= 2")
    faceIds = np.full((resolution, resolution), -1, dtype=np.int32)
    barycentrics = np.zeros((resolution, resolution, 3), dtype=np.float32)
    rasterUvs = uvs * (resolution - 1)
    for faceIndex, face in enumerate(faces):
        pixels = trianglePixels(rasterUvs[face], resolution, resolution)
        if pixels is None:
            continue
        ys, xs, weights = pixels
        available = faceIds[ys, xs] == -1
        faceIds[ys[available], xs[available]] = faceIndex
        barycentrics[ys[available], xs[available]] = weights[available]
    return {"faceIds": faceIds, "barycentrics": barycentrics, "validMask": faceIds >= 0}


def interpolateSurface(vertices: np.ndarray, faces: np.ndarray,
                       lookup: dict[str, np.ndarray]) -> np.ndarray:
    """Interpolate mesh XYZ at valid UV pixels; invalid pixels remain zero."""
    vertices, faces = validateTriangles(vertices, faces, 3)
    faceIds = np.asarray(lookup["faceIds"])
    validMask = np.asarray(lookup["validMask"], dtype=bool)
    weights = np.asarray(lookup["barycentrics"])
    if validMask.shape != faceIds.shape or weights.shape != (*faceIds.shape, 3):
        raise ValueError("Lookup arrays have inconsistent shapes")
    if np.any(faceIds[validMask] < 0) or np.any(faceIds[validMask] >= len(faces)):
        raise ValueError("Lookup contains invalid face indices")
    if not np.isfinite(weights).all():
        raise ValueError("Lookup barycentrics must be finite")
    surface = np.zeros((*faceIds.shape, 3), dtype=np.float32)
    triangleVertices = vertices[faces[faceIds[validMask]]]
    surface[validMask] = np.sum(triangleVertices * weights[validMask, :, None], axis=1)
    return surface


def rasterizeDepth(cameraVertices: np.ndarray, faces: np.ndarray,
                   intrinsics: np.ndarray, height: int, width: int) -> np.ndarray:
    """Perspective-correct camera-space Z buffer; background is positive infinity.

    Triangles crossing the camera near plane are omitted conservatively. Head
    reconstruction cameras must place the whole observed head in front of them.
    Both windings write depth, so opposite-facing geometry can still occlude.
    """
    cameraVertices, faces = validateTriangles(cameraVertices, faces, 3)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    if height < 1 or width < 1:
        raise ValueError("Raster dimensions must be positive")
    depth = np.full((height, width), np.inf, dtype=np.float32)
    projected = cameraVertices @ intrinsics.T
    visibleVertices = (cameraVertices[:, 2] > 1e-8) & (np.abs(projected[:, 2]) > 1e-8)
    pixels = np.zeros((len(cameraVertices), 2), dtype=np.float64)
    pixels[visibleVertices] = projected[visibleVertices, :2] / projected[visibleVertices, 2:3]
    for face in faces:
        if not visibleVertices[face].all():
            continue
        rasterPixels = trianglePixels(pixels[face], height, width)
        if rasterPixels is None:
            continue
        ys, xs, weights = rasterPixels
        interpolatedDepth = 1.0 / np.sum(weights / cameraVertices[face, 2], axis=1)
        depth[ys, xs] = np.minimum(depth[ys, xs], interpolatedDepth)
    return depth


def vertexNormals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals; triangle winding must face outwards."""
    triangleVertices = vertices[faces]
    faceNormals = np.cross(triangleVertices[:, 1] - triangleVertices[:, 0],
                           triangleVertices[:, 2] - triangleVertices[:, 0])
    normals = np.zeros_like(vertices, dtype=np.float64)
    for cornerIndex in range(3):
        np.add.at(normals, faces[:, cornerIndex], faceNormals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 1e-12)


def weightedColorMedian(colors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-channel weighted median over views; missing observations have zero weight."""
    medians = np.zeros(colors.shape[1:], dtype=np.float32)
    for channelIndex in range(3):
        order = np.argsort(colors[:, :, channelIndex], axis=0)
        sortedWeights = np.take_along_axis(weights, order, axis=0)
        cumulativeWeights = np.cumsum(sortedWeights, axis=0)
        medianIndices = np.argmax(cumulativeWeights >= cumulativeWeights[-1:] * 0.5, axis=0)
        sortedColors = np.take_along_axis(colors[:, :, channelIndex], order, axis=0)
        medians[:, channelIndex] = np.take_along_axis(sortedColors, medianIndices[None], axis=0)[0]
    return medians



def samplePixels(image: np.ndarray, imagePoints: np.ndarray) -> np.ndarray:
    """Bilinear samples with zero border and source-range preservation.

    Float32 accumulation can overshoot a constant source by one ULP. Restore
    the convex-combination bound per channel, including the zero border. The
    bound comes from the source, not [0,1], so signed/HDR data stays meaningful
    and invalid RGB sources are not silently normalized into valid colors.
    """
    image = np.asarray(image, dtype=np.float32)
    imagePoints = np.asarray(imagePoints, dtype=np.float64)
    if image.ndim not in (2, 3) or imagePoints.ndim != 2 or imagePoints.shape[1] != 2:
        raise ValueError("Expected an HW/HWC image and Nx2 sampling positions")
    if not np.isfinite(imagePoints).all():
        raise ValueError("Sampling positions must be finite")
    x0 = np.floor(imagePoints[:, 0]).astype(np.int64)
    y0 = np.floor(imagePoints[:, 1]).astype(np.int64)
    dx, dy = imagePoints[:, 0] - x0, imagePoints[:, 1] - y0
    sampled = np.zeros((len(imagePoints), *image.shape[2:]), dtype=np.float32)
    for xOffset, yOffset, weights in ((0, 0, (1 - dx) * (1 - dy)), (1, 0, dx * (1 - dy)),
                                     (0, 1, (1 - dx) * dy), (1, 1, dx * dy)):
        xs, ys = x0 + xOffset, y0 + yOffset
        inside = (xs >= 0) & (ys >= 0) & (xs < image.shape[1]) & (ys < image.shape[0])
        values = image[ys[inside], xs[inside]]
        sampled[inside] += values * (weights[inside, None] if image.ndim == 3 else weights[inside])
    if image.size:
        minimum = np.minimum(image.min(axis=(0, 1)), 0)
        maximum = np.maximum(image.max(axis=(0, 1)), 0)
        np.clip(sampled, minimum, maximum, out=sampled)
    return sampled


def fuseTexture(geometry: dict[str, Any], delight: dict[str, Any], cameras: list[dict[str, Any]],
                resolution: int = 512, colorBalance: bool = True,
                *, uvLookup: dict | None = None, samplingMode: str = "projection") -> dict[str, Any]:
    """Fuse only masked, front-facing, depth-visible OpenDelight observations.

    geometry: vertices (V,3), faces (F,3), uvs (V,2).
    delight: albedoImages (N,H,W,3), validMasks (N,H,W).
    Cameras project canonical row vectors as xyz @ rotation.T + translation;
    intrinsics operate in the exact image frame used by albedoImages.

    albedo is unfilled, previewAlbedo fills unseen valid-atlas pixels by nearest
    observed texel. coverageMask always describes real support, independent of
    RGB intensity. confidence is a heuristic agreement/view-quality score, not
    a probability or a physically calibrated material uncertainty.
    """
    from scipy.ndimage import distance_transform_edt

    if samplingMode not in {"projection", "denseUv"}:
        raise ValueError("samplingMode must be projection or denseUv")
    vertices, faces = validateTriangles(geometry["vertices"], geometry["faces"], 3)
    uvs = np.asarray(geometry["uvs"])
    if uvs.shape != (len(vertices), 2):
        raise ValueError("Expected one UV per geometry vertex")
    images = np.asarray(delight["albedoImages"], dtype=np.float32)
    masks = np.asarray(delight["validMasks"])
    if images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] == 0 or min(images.shape[1:3]) < 2:
        raise ValueError("albedoImages must have shape (N, H, W, 3), with N >= 1 and H,W >= 2")
    if masks.shape != images.shape[:3] or len(cameras) != len(images):
        raise ValueError("Masks and cameras must match albedo image count and dimensions")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("albedoImages must be finite floats in [0, 1]")
    if masks.dtype.kind not in "biuf" or not np.isfinite(masks).all() or masks.min() < 0 or masks.max() > 1:
        raise ValueError("validMasks must contain finite values in [0, 1]")

    lookup = uvLookup if uvLookup is not None else buildUvLookup(uvs, faces, resolution)
    if lookup["faceIds"].shape != (resolution, resolution):
        raise ValueError("Preloaded UV lookup does not match requested resolution")
    atlasMask = lookup["validMask"]
    surface = interpolateSurface(vertices, faces, lookup)[atlasMask].astype(np.float64)
    surfaceNormals = interpolateSurface(vertexNormals(vertices, faces), faces, lookup)[atlasMask].astype(np.float64)
    normalLengths = np.linalg.norm(surfaceNormals, axis=1, keepdims=True)
    surfaceNormals = np.divide(surfaceNormals, normalLengths, out=np.zeros_like(surfaceNormals), where=normalLengths > 1e-12)
    viewCount, imageHeight, imageWidth = images.shape[:3]
    if samplingMode == "denseUv" and np.asarray(geometry.get("uvMaps")).shape != (*images.shape[:3], 2):
        raise ValueError("denseUv sampling requires matching per-view predicted UV maps")
    queryY, queryX = np.nonzero(atlasMask)
    queryUvs = np.stack([queryX, queryY], axis=-1) / (resolution - 1)
    sampleCount = len(surface)
    sampledColors = np.zeros((viewCount, sampleCount, 3), dtype=np.float32)
    sampleWeights = np.zeros((viewCount, sampleCount), dtype=np.float32)
    diagnostics: dict[str, Any] = {"samplingMode": samplingMode, "uvConvention": "vDownIntegerPixelCenters", "colorSpace": "opendelightNative",
                                   "colorBalanceMethod": "nativeEncodingMultiplicativeOverlap" if colorBalance else "disabled",
                                   "confidenceMeaning": "heuristicViewQualityAndAgreementNotProbability",
                                   "views": [], "warnings": []}
    meshExtent = float(np.max(np.ptp(vertices, axis=0)))
    for viewIndex, camera in enumerate(cameras):
        viewDiagnostic: dict[str, Any] = {"viewIndex": viewIndex, "status": "rejected", "visibleTexelCount": 0}
        diagnostics["views"].append(viewDiagnostic)
        if not camera.get("valid", False):
            viewDiagnostic["reason"] = "invalidCamera"
            continue
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
        rotation = np.asarray(camera["rotation"], dtype=np.float64)
        translation = np.asarray(camera["translation"], dtype=np.float64)
        cameraError = float(camera.get("medianErrorPixels", 0))
        if (intrinsics.shape != (3, 3) or rotation.shape != (3, 3) or translation.shape != (3,)
                or not all(np.isfinite(value).all() for value in (intrinsics, rotation, translation))
                or not np.isfinite(cameraError) or cameraError < 0):
            raise ValueError(f"Camera {viewIndex} has malformed or nonfinite parameters")
        if sampleCount == 0:
            viewDiagnostic["reason"] = "emptyUvAtlas"
            continue
        cameraSurface = surface @ rotation.T + translation
        projected = cameraSurface @ intrinsics.T
        inFront = (cameraSurface[:, 2] > 1e-8) & (np.abs(projected[:, 2]) > 1e-8)
        imagePoints = np.full((sampleCount, 2), -1.0, dtype=np.float64)
        imagePoints[inFront] = projected[inFront, :2] / projected[inFront, 2:3]
        insideImage = (inFront & (imagePoints[:, 0] >= 0) & (imagePoints[:, 0] <= imageWidth - 1)
                       & (imagePoints[:, 1] >= 0) & (imagePoints[:, 1] <= imageHeight - 1))
        cameraNormals = surfaceNormals @ rotation.T
        viewDirections = -cameraSurface / np.maximum(np.linalg.norm(cameraSurface, axis=1, keepdims=True), 1e-12)
        cosines = np.maximum(np.sum(cameraNormals * viewDirections, axis=1), 0)
        candidateMask = insideImage & (cosines > 0.10)
        candidateIndices = np.flatnonzero(candidateMask)
        if not len(candidateIndices):
            viewDiagnostic["reason"] = "noFrontFacingProjection"
            continue
        cameraVertices = vertices @ rotation.T + translation
        depth = rasterizeDepth(cameraVertices, faces, intrinsics, imageHeight, imageWidth)
        candidatePoints = imagePoints[candidateIndices]
        nearestX = np.clip(np.rint(candidatePoints[:, 0]).astype(int), 0, imageWidth - 1)
        nearestY = np.clip(np.rint(candidatePoints[:, 1]).astype(int), 0, imageHeight - 1)
        depthSamples = depth[nearestY, nearestX]
        # The small allowance accommodates rounded pixel centers and smooth
        # surface interpolation; it is relative to the canonical mesh scale.
        depthTolerance = max(meshExtent * 0.005, 1e-7)
        depthVisible = np.isfinite(depthSamples) & (np.abs(cameraSurface[candidateIndices, 2] - depthSamples) <= depthTolerance)
        colorPoints = candidatePoints
        semanticValid = np.ones(len(candidateIndices), dtype=bool)
        if samplingMode == "denseUv":
            from .surfaceSampling import invertUvMap
            colorPoints, semanticValid = invertUvMap(geometry["uvMaps"][viewIndex], masks[viewIndex],
                                                     queryUvs[candidateIndices], candidatePoints)
            viewDiagnostic["semanticAcceptedCount"] = int(semanticValid.sum())
            if semanticValid.any():
                viewDiagnostic["medianAlignmentCorrectionPixels"] = float(np.median(
                    np.linalg.norm(colorPoints[semanticValid] - candidatePoints[semanticValid], axis=1)))
        maskSupport = samplePixels(masks[viewIndex].astype(np.float32), colorPoints)
        validSamples = depthVisible & semanticValid & (maskSupport >= 0.999)
        colors = samplePixels(images[viewIndex], colorPoints)
        retainedIndices = candidateIndices[validSamples]
        sampledColors[viewIndex, retainedIndices] = colors[validSamples]
        cameraQuality = 1.0 / (1.0 + (cameraError / 4.0) ** 2)
        sampleWeights[viewIndex, retainedIndices] = (cosines[retainedIndices] ** 4) * cameraQuality
        viewDiagnostic.update(status="ok", projectedTexelCount=int(len(candidateIndices)),
                              depthVisibleTexelCount=int(depthVisible.sum()),
                              visibleTexelCount=int(len(retainedIndices)), cameraQuality=cameraQuality,
                              medianErrorPixels=cameraError, depthTolerance=depthTolerance)
        if not len(retainedIndices):
            viewDiagnostic.update(status="rejected", reason="noValidDepthAndMaskSupport")

    colorGains = np.ones((viewCount, 3), dtype=np.float32)
    referenceView = int(np.argmax(sampleWeights.sum(axis=1)))
    if colorBalance and sampleCount:
        for viewIndex in range(viewCount):
            if viewIndex == referenceView:
                continue
            overlap = (sampleWeights[referenceView] > 0.05) & (sampleWeights[viewIndex] > 0.05)
            if np.count_nonzero(overlap) < 32:
                continue
            for channelIndex in range(3):
                referenceValues = sampledColors[referenceView, overlap, channelIndex]
                sourceValues = sampledColors[viewIndex, overlap, channelIndex]
                # Black texels remain valid observations. Only the gain fit
                # omits values too close to zero or saturation for stable ratios.
                stable = (referenceValues > 0.03) & (sourceValues > 0.03) & (referenceValues < 0.97) & (sourceValues < 0.97)
                if np.count_nonzero(stable) >= 32:
                    gain = np.exp(np.median(np.log(referenceValues[stable]) - np.log(sourceValues[stable])))
                    colorGains[viewIndex, channelIndex] = np.clip(gain, 2 / 3, 1.5)
        sampledColors = np.clip(sampledColors * colorGains[:, None, :], 0, 1)
    diagnostics.update(colorReferenceView=referenceView, colorGains=colorGains.tolist())
    medians = weightedColorMedian(sampledColors, sampleWeights)
    colorResiduals = np.sqrt(np.mean((sampledColors - medians[None]) ** 2, axis=-1))
    robustWeights = sampleWeights / (1.0 + (colorResiduals / 0.06) ** 2)
    totalWeights = robustWeights.sum(axis=0)
    observed = totalWeights > 1e-8
    fusedSamples = np.divide(np.sum(sampledColors * robustWeights[:, :, None], axis=0), totalWeights[:, None],
                             out=np.zeros((sampleCount, 3), dtype=np.float32), where=totalWeights[:, None] > 1e-8)
    weightedResiduals = np.divide(np.sum(colorResiduals * robustWeights, axis=0), totalWeights,
                                  out=np.zeros(sampleCount, dtype=np.float32), where=observed)
    confidenceSamples = np.clip(sampleWeights.max(axis=0), 0, 1) * np.exp(-weightedResiduals / 0.1)
    confidenceSamples[~observed] = 0
    selectedViews = np.argmax(robustWeights, axis=0).astype(np.int16)
    selectedViews[~observed] = -1
    albedo = np.zeros((resolution, resolution, 3), dtype=np.float32)
    coverageMask = np.zeros((resolution, resolution), dtype=bool)
    confidence = np.zeros((resolution, resolution), dtype=np.float32)
    sourceIndex = np.full((resolution, resolution), -1, dtype=np.int16)
    albedo[atlasMask], coverageMask[atlasMask] = fusedSamples, observed
    confidence[atlasMask], sourceIndex[atlasMask] = confidenceSamples, selectedViews
    previewAlbedo = albedo.copy()
    if observed.any():
        _, nearestIndices = distance_transform_edt(~coverageMask, return_indices=True)
        previewMissing = atlasMask & ~coverageMask
        previewAlbedo[previewMissing] = albedo[nearestIndices[0][previewMissing], nearestIndices[1][previewMissing]]
    else:
        diagnostics["warnings"].append("No depth-visible masked observations; no albedo has been reconstructed.")
    diagnostics.update(atlasTexelCount=int(sampleCount), observedTexelCount=int(observed.sum()),
                       atlasCoverage=float(observed.mean()) if sampleCount else 0.0,
                       missingTexelCount=int((~observed).sum()),
                       previewHoleFill="nearestObservedTexelWithinAtlasOnlyNotMeasuredAlbedo")
    return {"albedo": albedo, "previewAlbedo": previewAlbedo, "coverageMask": coverageMask,
            "confidence": confidence, "sourceIndex": sourceIndex, "diagnostics": diagnostics}
