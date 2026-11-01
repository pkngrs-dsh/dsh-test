"""Deterministic primary-face selection shared by detection adapters."""

from __future__ import annotations

import numpy as np


def selectLargestFaceIndex(boxes, scores) -> int:
    """Select the largest XYXY box, then confidence and original index on ties.

    All detections must have finite coordinates, positive width and height, and
    finite confidence in [0, 1]. Invalid detections fail instead of silently
    changing which person is selected. Inputs are never modified.
    """
    try:
        boxes = np.asarray(boxes)
        scores = np.asarray(scores)
    except (TypeError, ValueError) as error:
        raise ValueError("Face boxes and scores must be numeric arrays") from error
    # FACER represents an empty rects collection as a one-dimensional tensor.
    if boxes.shape in {(0,), (0, 4)} and scores.shape == (0,):
        raise ValueError("No faces detected")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Face boxes must have shape (N, 4)")
    if scores.shape != (len(boxes),):
        raise ValueError("Face scores must have shape (N,) matching the boxes")
    if boxes.dtype.kind not in "iuf" or scores.dtype.kind not in "iuf":
        raise ValueError("Face boxes and scores must contain real numeric values")
    boxes = boxes.astype(np.float64, copy=False)
    scores = scores.astype(np.float64, copy=False)
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Face boxes and scores must contain finite values")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Face scores must be in [0, 1]")
    with np.errstate(over="ignore", invalid="ignore"):
        sizes = boxes[:, 2:] - boxes[:, :2]
        areas = sizes[:, 0] * sizes[:, 1]
    if not np.isfinite(sizes).all() or not np.isfinite(areas).all():
        raise ValueError("Face box widths, heights and areas must be finite")
    if (sizes <= 0).any() or (areas <= 0).any():
        raise ValueError("Face boxes must have positive width, height and area")
    largest = np.flatnonzero(areas == areas.max())
    return int(largest[np.argmax(scores[largest])])
