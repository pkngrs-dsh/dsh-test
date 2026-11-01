"""Strict image ingestion with reversible pixel coordinates and content hashes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .faceCrop import cropFaceSquare, validateCropOptions
from .imageColor import readSrgbImage

imageSuffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def readImage(path: Path, imageSize: int = 512, *, cropPolicy: str = "centerSquare",
              faceDetector: Callable | None = None, cropMargin: float = 1.35) -> tuple[np.ndarray, dict]:
    """Read a photo without model loading; faceSquare requires a supplied detector.

    The detector receives EXIF-oriented, ICC-managed original-size sRGB uint8 pixels and must
    return exactly one face. Its errors propagate; no center-crop fallback is
    used. ``inputToWorking`` always starts in that oriented original frame.
    """
    import cv2

    validateCropOptions(cropPolicy, cropMargin)
    if isinstance(imageSize, bool) or not isinstance(imageSize, (int, np.integer)) or imageSize < 2:
        raise ValueError("imageSize must be an integer >= 2")
    if cropPolicy == "faceSquare" and not callable(faceDetector):
        raise ValueError("faceSquare requires a faceDetector; no centerSquare fallback is performed")
    path = Path(path).resolve()
    contentHash = hashlib.sha256(path.read_bytes()).hexdigest()
    source, colorManagement = readSrgbImage(path)
    with source:
        width, height = source.size
        if min(width, height) < 64:
            raise ValueError(f"Image is too small: {path.name}")
        if cropPolicy == "faceSquare":
            oriented = np.array(source, dtype=np.uint8, copy=True)
            detection = faceDetector(oriented)
            croppedArray, cropMetadata = cropFaceSquare(oriented, detection, cropMargin)
            left, top, _, _ = cropMetadata["cropBoxXyxy"]
            cropSize = cropMetadata["cropSize"]
            cropped = Image.fromarray(croppedArray)
        else:
            cropSize = min(width, height)
            left, top = (width - cropSize) // 2, (height - cropSize) // 2
            cropped = source.crop((left, top, left + cropSize, top + cropSize))
            cropMetadata = {"cropBoxXyxy": [left, top, left + cropSize, top + cropSize],
                            "cropSize": cropSize, "paddingLeftTopRightBottom": [0, 0, 0, 0],
                            "paddingMode": "none", "qualityFlags": []}
        resized = np.asarray(cropped.resize((imageSize, imageSize), Image.Resampling.LANCZOS))
    image = resized.astype(np.float32) / 255.0
    scale = imageSize / cropSize
    imageTransform = np.array([[scale, 0, scale * (0.5 - left) - 0.5],
                               [0, scale, scale * (0.5 - top) - 0.5], [0, 0, 1]])
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    metadata = {"path": str(path), "name": path.name, "sha256": contentHash,
                "orientedSize": [width, height], "inputToWorking": imageTransform.tolist(),
                "workingToInput": np.linalg.inv(imageTransform).tolist(),
                "workingSize": [imageSize, imageSize],
                "cropPolicy": "exifTransposeThenFaceSquare" if cropPolicy == "faceSquare" else "exifTransposeThenCenterSquare",
                "requestedCropPolicy": cropPolicy, "colorManagement": colorManagement,
                "blurProxy": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
                "darkFraction": float((gray < 8).mean()),
                "brightFraction": float((gray > 247).mean())}
    metadata.update(cropMetadata)
    if "faceDetection" in metadata:
        landmarks = np.asarray(metadata["faceDetection"]["landmarks"])
        mapped = np.column_stack([landmarks, np.ones(len(landmarks))]) @ imageTransform.T
        metadata["faceLandmarksWorking"] = mapped[:, :2].tolist()
    return image, metadata


def loadViews(paths: list[Path], minViews: int = 4, maxViews: int = 16, *,
              cropPolicy: str = "centerSquare", faceDetector: Callable | None = None,
              cropMargin: float = 1.35):
    if not minViews <= len(paths) <= maxViews:
        raise ValueError(f"Expected {minViews}–{maxViews} views, received {len(paths)}")
    images, metadata, hashes = [], [], set()
    for path in paths:
        image, record = readImage(path, cropPolicy=cropPolicy, faceDetector=faceDetector, cropMargin=cropMargin)
        if record["sha256"] in hashes:
            raise ValueError(f"Duplicate photo: {path.name}; supply distinct views")
        hashes.add(record["sha256"])
        images.append(image)
        metadata.append(record)
    return np.stack(images), metadata


def listImages(directory: Path) -> list[Path]:
    return sorted(path for path in Path(directory).iterdir()
                  if path.is_file() and path.suffix.lower() in imageSuffixes)


def auditDataset(directory: Path) -> dict:
    groups, failedFiles, uniqueHashes = {}, [], set()
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in imageSuffixes:
            continue
        group = str(path.parent.relative_to(directory))
        record = groups.setdefault(group, {"validFiles": [], "duplicateFiles": []})
        try:
            _, item = readImage(path)
            record["validFiles"].append(path.name)
            if item["sha256"] in uniqueHashes:
                record["duplicateFiles"].append(path.name)
            uniqueHashes.add(item["sha256"])
        except (OSError, ValueError) as error:
            failedFiles.append({"path": str(path), "error": str(error)})
    return {"sourceRoot": str(Path(directory).resolve()), "groupCount": len(groups),
            "validFileCount": sum(len(g["validFiles"]) for g in groups.values()),
            "distinctContentCount": len(uniqueHashes), "failedFileCount": len(failedFiles),
            "failedFiles": failedFiles, "groups": groups}
