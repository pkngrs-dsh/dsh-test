"""Preloaded CPU displacement solve from surface samples to fixed mesh indices.

Geometry deformation uses the caller's centimeter frame; generic attribute
transfer keeps the caller's value units. This module neither aligns coordinate
frames nor changes topology or guards triangle inversions. The caller must apply
any registration and geometric safety policy around this solve.
"""

from __future__ import annotations

from numbers import Real
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized


def _floatArray(values, name: str, shape: tuple | None = None) -> np.ndarray:
    array = np.asarray(values)
    if (array.dtype.kind != "f" or not np.isfinite(array).all()
            or (shape is not None and array.shape != shape)):
        raise ValueError(f"{name} must be a finite floating-point array"+(f" shaped {shape}" if shape is not None else ""))
    copied = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(copied).all():
        raise ValueError(f"{name} is outside the float64 range")
    return copied


def _indexArray(values, name: str, shape: tuple, upperBound: int) -> np.ndarray:
    array = np.asarray(values)
    if (array.shape != shape or array.dtype.kind not in "iu"
            or np.any(array < 0) or np.any(array >= upperBound)):
        raise ValueError(f"{name} must contain integer indices shaped {shape} in [0,{upperBound})")
    return np.array(array, dtype=np.int64, copy=True)


def _nonnegativeScalar(value, name: str, *, positive=False) -> float:
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} scalar")
    return float(value)


def _freezeSparse(matrix: sparse.csr_matrix) -> None:
    for values in (matrix.data, matrix.indices, matrix.indptr):
        values.setflags(write=False)


def _uniformLaplacian(vertexCount: int, faces: np.ndarray) -> sparse.csr_matrix:
    edges = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1), axis=0)
    rows, columns = np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(vertexCount, vertexCount))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    # Isolated vertices have zero Laplacian; the rest prior anchors their displacement.
    return (sparse.diags((degree > 0).astype(np.float64))
            - sparse.diags(1/np.maximum(degree, 1)) @ adjacency).tocsr()


class BarycentricDeformer:
    """Factor one fixed weighted surface-sample system for repeated CPU solves.

    ``C`` is available as ``sampleMatrix`` and its rest samples as
    ``restSamplePoints``. Sample weights enter the objective once, not squared::

        dataWeight * sum(sampleWeights * ||C @ displacement - residual||²)
        + smoothWeight * ||L @ displacement||² + restWeight * ||displacement||²

    Here ``residual = targetPoints - C @ restVertices``. Static displacement is
    exactly zero. No minimum of three samples or non-collinearity is required:
    this is a displacement solve, not a similarity fit. At least one sample
    weight must be positive. All-static meshes return a fresh copy of the rest
    mesh. With zero regularization, unconstrained free vertices cause an explicit
    singular-system error during construction.

    The uniform Laplacian and factorized operator are independent of rest XYZ.
    ``transferValues`` reuses them for attributes such as two-channel UV deltas.
    """

    def __init__(self, restVertices, faces, sampleFaceIds, sampleBarycentrics,
                 sampleWeights, staticMask, *, dataWeight=20., smoothWeight=1., restWeight=.03):
        started = perf_counter()
        vertices = _floatArray(restVertices, "restVertices")
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3:
            raise ValueError("restVertices must be a nonempty Nx3 mesh with at least three vertices")
        rawFaces = np.asarray(faces)
        if rawFaces.ndim != 2 or rawFaces.shape[1:] != (3,) or len(rawFaces) < 1:
            raise ValueError("faces must be a nonempty Fx3 triangle array")
        triangles = _indexArray(rawFaces, "faces", rawFaces.shape, len(vertices))
        if np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0):
            raise ValueError("faces may not repeat a vertex within a triangle")
        rawSampleIds = np.asarray(sampleFaceIds)
        if rawSampleIds.ndim != 1 or len(rawSampleIds) < 1:
            raise ValueError("sampleFaceIds must contain at least one surface sample")
        sampleIds = _indexArray(rawSampleIds, "sampleFaceIds", rawSampleIds.shape, len(triangles))
        barycentrics = _floatArray(sampleBarycentrics, "sampleBarycentrics", (len(sampleIds), 3))
        if (np.any(barycentrics < 0) or np.any(barycentrics > 1)
                or not np.allclose(barycentrics.sum(axis=1), 1., rtol=0, atol=1e-6)):
            raise ValueError("sampleBarycentrics must be nonnegative unit-sum triangle coordinates")
        weights = _floatArray(sampleWeights, "sampleWeights", (len(sampleIds),))
        if np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("sampleWeights must be nonnegative with at least one positive observation")
        static = np.asarray(staticMask)
        if static.shape != (len(vertices),) or static.dtype.kind != "b":
            raise ValueError("staticMask must be a boolean array with one entry per rest vertex")
        static = static.copy()
        self.dataWeight = _nonnegativeScalar(dataWeight, "dataWeight", positive=True)
        self.smoothWeight = _nonnegativeScalar(smoothWeight, "smoothWeight")
        self.restWeight = _nonnegativeScalar(restWeight, "restWeight")
        self.restVertices, self.faces = vertices, triangles
        self.sampleFaceIds, self.sampleBarycentrics, self.sampleWeights = sampleIds, barycentrics, weights
        self.staticMask = static
        self.freeVertexIds = np.flatnonzero(~static)
        self.positiveSampleMask = weights > 0
        self.sampleMatrix = sparse.csr_matrix((barycentrics.ravel(),
            (np.repeat(np.arange(len(sampleIds)), 3), triangles[sampleIds].ravel())),
            shape=(len(sampleIds), len(vertices)))
        self.laplacian = _uniformLaplacian(len(vertices), triangles)
        self.restSamplePoints = np.asarray(self.sampleMatrix @ vertices)
        if not np.isfinite(self.restSamplePoints).all():
            raise ValueError("Rest surface samples overflow float64")
        weightedMatrix = self.sampleMatrix.multiply(weights[:, None])
        operator = (self.dataWeight*(self.sampleMatrix.T @ weightedMatrix)
                    + self.smoothWeight*(self.laplacian.T @ self.laplacian)
                    + self.restWeight*sparse.eye(len(vertices), format="csr"))
        self.freeOperator = operator[self.freeVertexIds][:, self.freeVertexIds].tocsr()
        if not np.isfinite(self.freeOperator.data).all():
            raise ValueError("Weighted deformation operator overflows float64")
        self._solve = None
        if len(self.freeVertexIds):
            try:
                self._solve = factorized(self.freeOperator.tocsc())
            except RuntimeError as error:
                raise ValueError("Deformation system is singular; constrain free vertices or increase restWeight") from error
        for values in (self.restVertices, self.faces, self.sampleFaceIds, self.sampleBarycentrics,
                       self.sampleWeights, self.staticMask, self.freeVertexIds,
                       self.positiveSampleMask, self.restSamplePoints):
            values.setflags(write=False)
        for matrix in (self.sampleMatrix, self.laplacian, self.freeOperator):
            _freezeSparse(matrix)
        self.preloadSeconds = perf_counter()-started

    def transferValues(self, restValues, targetValues) -> np.ndarray:
        """Transfer arbitrary float NxD vertex values with the existing factorization.

        ``D`` is any positive channel count, and targets have shape MxD. Solve
        for a change from ``restValues`` using ``targetValues - C @ restValues``;
        static vertices keep their original attribute values exactly. Inputs
        must be finite even for zero-weight samples, which never affect the
        solve. The isolated float64 result keeps the caller's units. There is
        no range clamp, UV-domain inference or geometric safety operation.
        """
        rest = _floatArray(restValues, "restValues")
        if rest.ndim != 2 or rest.shape[0] != len(self.restVertices) or rest.shape[1] < 1:
            raise ValueError("restValues must be NxD with one row per vertex and at least one channel")
        targets = _floatArray(targetValues, "targetValues", (len(self.sampleFaceIds), rest.shape[1]))
        restSamples = self.sampleMatrix @ rest
        if not np.isfinite(restSamples).all():
            raise ValueError("Rest attribute samples overflow float64")
        positive = self.positiveSampleMask
        residual = np.zeros_like(targets)
        residual[positive] = targets[positive]-restSamples[positive]
        if not np.isfinite(residual).all():
            raise ValueError("Target attribute residual overflows float64")
        displacement = np.zeros_like(rest)
        if self._solve is not None:
            rhs = self.dataWeight*(self.sampleMatrix.T @ (self.sampleWeights[:, None]*residual))
            if not np.isfinite(rhs).all():
                raise ValueError("Weighted attribute target overflows float64")
            displacement[self.freeVertexIds] = self._solve(rhs[self.freeVertexIds])
        transferred = rest+displacement
        if not np.isfinite(transferred).all():
            raise ValueError("Attribute solve returned nonfinite values")
        # Hard constraints are explicit even when the solver has no free vertices.
        transferred[self.staticMask] = rest[self.staticMask]
        return transferred

    def deform(self, targetPoints) -> tuple[np.ndarray, dict]:
        """Return an isolated float64 mesh in centimeters and sample-fit diagnostics.

        Target points must be finite even for zero-weight samples. Zero-weight
        samples do not enter the solve or residual statistics. This raw solution
        needs the caller's geometric safety checks before downstream use.
        """
        started = perf_counter()
        targets = _floatArray(targetPoints, "targetPoints", self.restSamplePoints.shape)
        vertices = self.transferValues(self.restVertices, targets)
        displacement = vertices-self.restVertices
        positive = self.positiveSampleMask
        residual = np.zeros_like(targets)
        residual[positive] = targets[positive]-self.restSamplePoints[positive]
        beforeErrors = np.hypot.reduce(residual[positive], axis=1)
        afterErrors = np.hypot.reduce((self.sampleMatrix @ vertices)[positive]-targets[positive], axis=1)
        if not np.isfinite(beforeErrors).all() or not np.isfinite(afterErrors).all():
            raise ValueError("Sample residual diagnostics overflow float64")
        diagnostics = {"lengthUnit": "centimeter", "sampleCount": len(targets),
            "positiveSampleCount": int(positive.sum()), "vertexCount": len(vertices),
            "freeVertexCount": len(self.freeVertexIds), "staticVertexCount": int(self.staticMask.sum()),
            "allVerticesStatic": len(self.freeVertexIds) == 0,
            "sampleResidualPopulation": "unweighted statistics over positive-weight samples only",
            "sampleResidualMedianBeforeCm": float(np.median(beforeErrors)),
            "sampleResidualP95BeforeCm": float(np.percentile(beforeErrors, 95)),
            "sampleResidualMedianAfterCm": float(np.median(afterErrors)),
            "sampleResidualP95AfterCm": float(np.percentile(afterErrors, 95)),
            "maxDisplacementCm": float(np.hypot.reduce(displacement, axis=1).max()),
            "dataWeight": self.dataWeight, "smoothWeight": self.smoothWeight, "restWeight": self.restWeight,
            "preloadSeconds": self.preloadSeconds, "deformSeconds": perf_counter()-started,
            "geometryGuardApplied": False}
        return vertices, diagnostics
