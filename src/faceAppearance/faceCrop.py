"""Full-face square crops in EXIF-oriented image coordinates, without models.

Detector boxes use continuous XYXY image coordinates. Crop rectangles use
integer, right/bottom-exclusive bounds and may extend outside the source.
Only edge replication fills that missing area; it is never observed detail.
"""

from __future__ import annotations

from numbers import Real

import numpy as np


def validateCropOptions(cropPolicy: str, cropMargin: float) -> None:
    if not isinstance(cropPolicy, str) or cropPolicy not in {"centerSquare", "faceSquare"}:
        raise ValueError("cropPolicy must be centerSquare or faceSquare")
    if (isinstance(cropMargin, (bool, np.bool_)) or not isinstance(cropMargin, Real)
            or not np.isfinite(cropMargin) or cropMargin < 1):
        raise ValueError("cropMargin must be a finite number >= 1 to preserve the complete face")


def validateFaceDetection(detection: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Validate the selected face; missing or malformed coordinates fail."""
    if not isinstance(detection, dict):
        raise ValueError("faceDetector must return one detection dictionary")
    try:
        box = np.asarray(detection["boxXyxy"], dtype=np.float64)
        landmarks = np.asarray(detection["landmarks"], dtype=np.float64)
        score = detection["score"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Face detection requires boxXyxy, landmarks and score") from error
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Face boxXyxy must contain four finite coordinates")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Face boxXyxy must have positive width and height")
    if landmarks.ndim != 2 or landmarks.shape[1] != 2 or len(landmarks) == 0 or not np.isfinite(landmarks).all():
        raise ValueError("Face landmarks must be a nonempty finite Nx2 array")
    if (isinstance(score, (bool, np.bool_)) or not isinstance(score, Real)
            or not np.isfinite(score) or not 0 <= score <= 1):
        raise ValueError("Face detection score must be finite and in [0, 1]")
    return box, landmarks, float(score)


def _placeCrop(minimum: float, maximum: float, side: int, imageExtent: int) -> int:
    """Minimize padding by shifting the crop, while retaining the face bounds."""
    allowedLow = int(np.ceil(maximum)) - side
    allowedHigh = int(np.floor(minimum))
    preferred = int(np.floor((minimum + maximum - side) / 2))
    # For a crop larger than the image, every position in this interval retains
    # all available source pixels with the smallest possible padding area.
    paddingLow, paddingHigh = min(0, imageExtent - side), max(0, imageExtent - side)
    feasibleLow, feasibleHigh = max(allowedLow, paddingLow), min(allowedHigh, paddingHigh)
    if feasibleLow <= feasibleHigh:
        return min(max(preferred, feasibleLow), feasibleHigh)
    return allowedHigh if allowedHigh < paddingLow else allowedLow


def cropFaceSquare(image: np.ndarray, detection: dict,
                   cropMargin: float = 1.35) -> tuple[np.ndarray, dict]:
    """Crop a detected face and landmarks with margin, preserving their frame.

    ``image`` is the original EXIF-oriented RGB uint8 array. Landmarks outside
    the detector box expand the reference bounds so they are also retained.
    Boundary flags describe limitations already present in the original image.
    A crop may move away from the box center to avoid unnecessary padding.
    """
    validateCropOptions("faceSquare", cropMargin)
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or min(image.shape[:2]) < 1:
        raise ValueError("Face crop requires an EXIF-oriented RGB uint8 image")
    box, landmarks, score = validateFaceDetection(detection)
    height, width = image.shape[:2]
    if box[2] <= 0 or box[3] <= 0 or box[0] >= width or box[1] >= height:
        raise ValueError("Detected face box does not overlap the original image")
    # Landmark coordinates are pixel centers. Include their containing pixels
    # even when a landmark lies exactly on the detector's exclusive box edge.
    landmarkLow = np.floor(landmarks.min(axis=0))
    landmarkHigh = np.floor(landmarks.max(axis=0)) + 1
    reference = np.array([min(box[0], float(landmarkLow[0])),
                          min(box[1], float(landmarkLow[1])),
                          max(box[2], float(landmarkHigh[0])),
                          max(box[3], float(landmarkHigh[1]))])
    requestedSide = float(np.max(reference[2:] - reference[:2])) * float(cropMargin)
    if not np.isfinite(requestedSide) or requestedSide > 4 * max(width, height):
        raise ValueError("Requested face crop exceeds four times the original image extent; check detection coordinates or cropMargin")
    # Integer rounding must not lose a fraction of either box edge at margin=1.
    side = max(int(np.ceil(requestedSide)),
               int(np.max(np.ceil(reference[2:]) - np.floor(reference[:2]))))
    left = _placeCrop(reference[0], reference[2], side, width)
    top = _placeCrop(reference[1], reference[3], side, height)
    right, bottom = left + side, top + side
    padding = [max(0, -left), max(0, -top), max(0, right - width), max(0, bottom - height)]
    cropped = image[max(0, top):min(height, bottom), max(0, left):min(width, right)]
    if any(padding):
        cropped = np.pad(cropped, ((padding[1], padding[3]), (padding[0], padding[2]), (0, 0)), mode="edge")
    flags = []
    if box[0] <= 0 or box[1] <= 0 or box[2] >= width - 1 or box[3] >= height - 1:
        flags.append("faceBoxTouchesOriginalBoundary")
    if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
        flags.append("faceBoxOutsideOriginalImage")
    if (landmarks < 0).any() or (landmarks[:, 0] > width - 1).any() or (landmarks[:, 1] > height - 1).any():
        flags.append("landmarksOutsideOriginalImage")
    if np.any(reference != box):
        flags.append("landmarksExpandDetectorBox")
    if any(padding):
        flags.append("edgePaddingApplied")
    metadata = {
        "cropBoxXyxy": [left, top, right, bottom],
        "cropSize": side,
        "cropMargin": float(cropMargin),
        "cropReferenceBoxXyxy": reference.tolist(),
        "paddingLeftTopRightBottom": padding,
        "paddingMode": "edge" if any(padding) else "none",
        "faceDetection": {"boxXyxy": box.tolist(), "landmarks": landmarks.tolist(), "score": score},
        "qualityFlags": flags,
    }
    if "selection" in detection:
        metadata["faceSelection"] = dict(detection["selection"])
    return np.ascontiguousarray(cropped), metadata


def workingImageValidMask(record: dict) -> np.ndarray:
    """Mark working integer pixel centers that map inside observed source pixels.

    The inverse of ``inputToWorking`` maps into the EXIF-oriented original image.
    Source centers lie in [0,width-1] x [0,height-1]. Thus edge-replicated padding
    is excluded even though its RGB values resemble the adjacent face pixels.
    This tests centers, not the full Lanczos interpolation footprint.
    """
    try:
        orientedSize = np.asarray(record["orientedSize"])
        workingSize = np.asarray(record["workingSize"])
        transform = np.asarray(record["inputToWorking"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Valid pixel mask requires orientedSize, workingSize and inputToWorking") from error
    for size in (orientedSize, workingSize):
        if size.shape != (2,) or size.dtype.kind not in "iu" or (size < 1).any():
            raise ValueError("Image sizes must contain positive integer width and height")
    if transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise ValueError("inputToWorking must be a finite invertible 3x3 matrix")
    try:
        inverse = np.linalg.inv(transform)
    except np.linalg.LinAlgError as error:
        raise ValueError("inputToWorking must be a finite invertible 3x3 matrix") from error
    originalWidth, originalHeight = orientedSize
    workingWidth, workingHeight = workingSize
    ys, xs = np.mgrid[:workingHeight, :workingWidth]
    original = np.stack([xs, ys, np.ones_like(xs)], axis=-1) @ inverse.T
    denominator = original[..., 2]
    finite = np.isfinite(original).all(axis=-1) & (np.abs(denominator) > 1e-12)
    originalX = np.divide(original[..., 0], denominator, out=np.zeros_like(denominator), where=finite)
    originalY = np.divide(original[..., 1], denominator, out=np.zeros_like(denominator), where=finite)
    tolerance = 1e-7  # Keep exactly-on-boundary centers after floating-point inversion.
    return (finite & (originalX >= -tolerance) & (originalX <= originalWidth - 1 + tolerance)
            & (originalY >= -tolerance) & (originalY <= originalHeight - 1 + tolerance))
