"""Strict legacy surface-sample adapter, independent of registration solvers.

Each JSON record identifies one point on an NICP triangle, not a vertex in the
new UFF mesh. UV input is explicit array V-down. This module never infers or
changes UV direction and never turns out-of-triangle weights into valid ones.
"""

import json
from pathlib import Path

import numpy as np

from .correspondence import SurfaceIndex, sampleSurface
from .texture import validateTriangles


def _uniqueObject(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON sample key: {key}")
        result[key] = value
    return result


def readLegacyBarycentric(path, nicpFaces, *, expectedSampleCount=5346, boundaryTolerance=0.):
    """Parse numerically ordered sample IDs and preserve the original weights.

    Malformed tables raise. Finite weights outside the convex triangle are
    returned unchanged but marked invalid; caller must apply the validity mask.
    A tolerance only handles numerical boundary noise, never renormalization.
    """
    records = json.loads(Path(path).read_text(), object_pairs_hook=_uniqueObject)
    if not isinstance(records, dict) or not records:
        raise ValueError("Barycentric JSON must be a nonempty keyed object")
    if not isinstance(expectedSampleCount, int) or expectedSampleCount < 1:
        raise ValueError("Expected sample count must be positive")
    if not np.isfinite(boundaryTolerance) or boundaryTolerance < 0:
        raise ValueError("Boundary tolerance must be finite and nonnegative")
    ordered = {}
    for key, value in records.items():
        if not key.isascii() or not key.isdecimal():
            raise ValueError(f"Sample key must be a nonnegative decimal integer: {key}")
        index = int(key)
        if index in ordered:
            raise ValueError(f"Multiple keys identify sample {index}")
        ordered[index] = value
    if set(ordered) != set(range(expectedSampleCount)):
        raise ValueError("Sample keys must cover exactly 0..expectedSampleCount-1")
    rows = []
    for sampleId in range(expectedSampleCount):
        row = ordered[sampleId]
        if (not isinstance(row, list) or len(row) != 3
                or any(isinstance(value, bool) or not isinstance(value, (float, int)) for value in row)):
            raise ValueError(f"Sample {sampleId} must contain numeric faceId,u,v")
        rows.append(row)
    rows = np.asarray(rows, np.float64)
    if not np.isfinite(rows).all() or (rows[:, 0] != np.floor(rows[:, 0])).any():
        raise ValueError("Face indices must be integers and all values must be finite")
    nicpFaces = np.asarray(nicpFaces)
    if nicpFaces.ndim != 2 or nicpFaces.shape[1] != 3 or nicpFaces.dtype.kind not in "iu":
        raise ValueError("NICP faces must be integer triangles")
    if (nicpFaces < 0).any():
        raise ValueError("NICP vertex indices cannot be negative")
    faceIds = rows[:, 0].astype(np.int64)
    if (faceIds < 0).any() or (faceIds >= len(nicpFaces)).any():
        raise ValueError("Sample face index is outside NICP topology")
    weights = np.c_[rows[:, 1:], 1-rows[:, 1:].sum(axis=1)]
    valid = (weights.min(1) >= -boundaryTolerance) & (weights.max(1) <= 1+boundaryTolerance)
    return {"sampleIds": np.arange(expectedSampleCount, dtype=np.int64),
            "sampleNicpFaceIds": faceIds, "sampleNicpWeights": weights,
            "sampleBaryValidMask": valid}


def compileLegacyUvSamples(sampleUvs, sourceVertices, sourceFaces, sourceUvs, *, maxUvDistance=1e-5):
    """Query explicit V-down samples on UFF's UV triangles, retaining face IDs.

    Degenerate UV faces are excluded from the search without renumbering the
    returned original geometry face IDs. Missing UV support is marked invalid;
    its nearest mapping remains available for diagnostics only.
    """
    sourceVertices, sourceFaces = validateTriangles(sourceVertices, sourceFaces, 3)
    sourceUvs, _ = validateTriangles(sourceUvs, sourceFaces, 2)
    sampleUvs = np.asarray(sampleUvs, np.float64)
    if sampleUvs.ndim != 2 or sampleUvs.shape[1] != 2 or not np.isfinite(sampleUvs).all():
        raise ValueError("Sample UVs must be finite (N,2) V-down coordinates")
    if len(sourceUvs) != len(sourceVertices):
        raise ValueError("Expected one UFF UV per geometry vertex")
    if not np.isfinite(maxUvDistance) or maxUvDistance < 0:
        raise ValueError("UV distance threshold must be finite and nonnegative")
    planar = np.c_[sourceUvs, np.zeros(len(sourceUvs))]
    triangles = planar[sourceFaces]
    lengths = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
    originalFaceIds = np.flatnonzero(lengths > 1e-14)
    if not len(originalFaceIds):
        raise ValueError("Source UV chart has no nondegenerate triangles")
    mapping = SurfaceIndex(planar, sourceFaces[originalFaceIds]).query(np.c_[sampleUvs, np.zeros(len(sampleUvs))])
    mapping["triangleIds"] = originalFaceIds[mapping["triangleIds"]]
    return {"sampleUvs": sampleUvs, "sourceTriangleIds": mapping["triangleIds"].astype(np.int64),
            "sourceBarycentricWeights": mapping["barycentricWeights"],
            "sourceUvDistances": mapping["distances"],
            "sampleUvValidMask": mapping["distances"] <= maxUvDistance,
            "sampleSourcePoints": sampleSurface(sourceVertices, sourceFaces, mapping)}
