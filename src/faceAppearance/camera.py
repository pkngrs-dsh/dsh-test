"""Fit canonical-space cameras using image→UV semantic correspondences.

VGGT's world cameras cannot project UVFaceFusion's canonical mesh directly.
Every accepted camera is checked against correspondences withheld from fitting.
"""

from __future__ import annotations

import numpy as np


def projectPoints(points: np.ndarray, intrinsics: np.ndarray,
                  rotation: np.ndarray, translation: np.ndarray):
    cameraPoints = points @ rotation.T + np.asarray(translation).reshape(1, 3)
    projected = cameraPoints @ intrinsics.T
    return projected[:, :2] / np.maximum(projected[:, 2:3], 1e-9), cameraPoints[:, 2]


def fitCamera(objectPoints: np.ndarray, imagePoints: np.ndarray,
              intrinsics: np.ndarray, maxErrorPixels: float = 12, seed: int = 2026) -> dict:
    import cv2

    objectPoints = np.asarray(objectPoints, dtype=np.float64)
    imagePoints = np.asarray(imagePoints, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64).copy()
    count = len(objectPoints)
    if count < 60:
        return {"valid": False, "reason": "Fewer than 60 surface correspondences", "pointCount": count}
    generator = np.random.default_rng(seed)
    order = generator.permutation(count)
    heldOut = order[:max(12, count // 5)]
    fitting = order[len(heldOut):]
    cv2.setRNGSeed(seed)
    success, rotationVector, translation, inliers = cv2.solvePnPRansac(
        objectPoints[fitting], imagePoints[fitting], intrinsics, None,
        iterationsCount=300, reprojectionError=maxErrorPixels, confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP)
    if not success or inliers is None or len(inliers) < 30:
        return {"valid": False, "reason": "Robust camera fitting failed", "pointCount": count}
    accepted = fitting[inliers[:, 0]]
    rotationVector, translation = cv2.solvePnPRefineLM(
        objectPoints[accepted], imagePoints[accepted], intrinsics, None,
        rotationVector, translation)
    rotation = cv2.Rodrigues(rotationVector)[0]
    projected, depth = projectPoints(objectPoints, intrinsics, rotation, translation)
    errors = np.linalg.norm(projected - imagePoints, axis=1)
    heldOutMedian = float(np.median(errors[heldOut]))
    inlierFraction = len(accepted) / len(fitting)
    positiveFraction = float((depth > 0).mean())
    valid = (heldOutMedian <= maxErrorPixels and inlierFraction >= 0.4 and positiveFraction > 0.99)
    return {"valid": bool(valid), "reason": "accepted" if valid else "Camera quality gate failed",
            "intrinsics": intrinsics, "rotation": rotation, "translation": translation.reshape(3),
            "pointCount": count, "inlierFraction": float(inlierFraction),
            "positiveDepthFraction": positiveFraction, "medianErrorPixels": float(np.median(errors)),
            "heldOutMedianErrorPixels": heldOutMedian,
            "heldOutP90ErrorPixels": float(np.percentile(errors[heldOut], 90)),
            "diagnosticImagePoints": imagePoints[heldOut[:200]],
            "diagnosticProjectedPoints": projected[heldOut[:200]]}


def fitViewCameras(geometry: dict, maxErrorPixels: float = 12, seed: int = 2026,
                   *, uvLookup: dict | None = None):
    from .texture import buildUvLookup, interpolateSurface

    lookup = uvLookup if uvLookup is not None else buildUvLookup(geometry["uvs"], geometry["faces"], 512)
    if lookup["faceIds"].shape != (512, 512):
        raise ValueError("Camera fitting requires a 512 pixel UV lookup")
    surface = interpolateSurface(geometry["vertices"], geometry["faces"], lookup)
    cameras = []
    for viewIndex, uvMap in enumerate(geometry["uvMaps"]):
        imageHeight, imageWidth = uvMap.shape[:2]
        yy, xx = np.mgrid[0:imageHeight:5, 0:imageWidth:5]
        predictedUv = uvMap[yy, xx]
        valid = (geometry["faceMasks"][viewIndex, yy, xx] > 0.5)
        valid &= np.isfinite(predictedUv).all(axis=-1) & (predictedUv >= 0).all(axis=-1)
        valid &= (predictedUv <= 1).all(axis=-1)
        safeUv = np.nan_to_num(predictedUv, nan=0.0, posinf=0.0, neginf=0.0)
        uvPixels = np.rint(np.clip(safeUv, 0, 1) * 511).astype(np.int32)
        valid &= lookup["validMask"][uvPixels[..., 1], uvPixels[..., 0]]
        points = surface[uvPixels[..., 1], uvPixels[..., 0]][valid]
        pixels = np.stack([xx, yy], axis=-1)[valid]
        camera = fitCamera(points, pixels, geometry["cameraIntrinsics"][viewIndex],
                           maxErrorPixels, seed + viewIndex)
        camera["viewIndex"] = viewIndex
        cameras.append(camera)
    return cameras
