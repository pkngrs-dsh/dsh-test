"""Generic differentiable mesh guards for a supplied bounded template patch.

Constraints use every interior-incident edge/face and every patch control
vertex. No subject identity, exceptional vertex, or failed element is accepted
as input. A feasible result is not an anatomical or photographic acceptance.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .localSurfaceRegistration import _scalar


DEFAULT_POLICY = {
    'minimumRestEdgeRatio': .25, 'maximumRestEdgeRatio': 2.,
    'minimumRestAreaRatio': .1, 'maximumRestAreaRatio': 3.,
    'minimumRestQualityRatio': .4, 'minimumBaselineNormalCosine': .3,
    'restNormalCosineFloor': 0.,
    'interiorDisplacementLimit': 2., 'rimDisplacementLimit': .5,
}


def _vertices(value, name, shape=None):
    array = np.asarray(value, dtype=np.float64)
    if (array.ndim != 2 or array.shape[1] != 3 or len(array) < 3
            or (shape is not None and array.shape != shape) or not np.isfinite(array).all()):
        raise ValueError(f'{name} must be a finite Nx3 array matching the reference topology')
    return array.copy()


def _indices(value, name, bound):
    indices = np.asarray(value)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices >= bound) or len(np.unique(indices)) != len(indices)):
        raise ValueError(f'{name} must contain unique valid integer indices')
    return indices.astype(np.int64, copy=True)


def _triangleMetrics(vertices, faces):
    points = vertices[faces]
    u, v = points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]
    cross = np.cross(u, v)
    crossLength = np.linalg.norm(cross, axis=1)
    squaredEdgeSum = np.sum(u*u + v*v + (v-u)**2, axis=1)
    if (not np.isfinite(crossLength).all() or not np.isfinite(squaredEdgeSum).all()
            or np.any(crossLength <= 0) or np.any(squaredEdgeSum <= 0)):
        raise ValueError('Guarded faces must have finite nonzero area and edge length')
    normal = cross/crossLength[:, None]
    quality = 2*np.sqrt(3.)*crossLength/squaredEdgeSum
    return {'points': points, 'u': u, 'v': v, 'cross': cross, 'crossLength': crossLength,
            'edgeSquaredSum': squaredEdgeSum, 'area': crossLength*.5, 'normal': normal, 'quality': quality}


def buildMeshShapeGuardSpec(
    baselineVertices, restVertices, faces, interiorVertexIds, rimVertexIds, *,
    lengthScale, depthOperator, depthLower, depthUpper, policy=None,
):
    """Freeze a data-only specification of constraints ``values >= 0``.

    All coordinates, depth bounds and ``lengthScale`` share target-frame mesh
    units. lengthScale is ONE reference physical length: pass the target-frame
    size of 1 mm when using default displacement limits (interior 2, rim .5).
    Changing length units requires scaling it and both depth bounds as well.

    The patch consists of disjoint interior/rim IDs. Every edge and face touching
    an interior vertex is guarded, and all its endpoints must be in this patch.
    No element-exclusion list exists. Depth rows may involve any patch vertices
    and XYZ axes, but cannot introduce support outside the patch. The caller
    owns their physical meaning and direction; the generic builder checks only
    their sparse linear form and initial feasibility.

    Bounds use baseline/rest envelopes, retaining preexisting violations without
    allowing them to worsen: edges [min(base,.25rest),max(base,2rest)], areas
    [min(base,.1rest),max(base,3rest)], quality >= min(qBase,.4qRest), cosine to
    baseline >= .3, and cosine to rest >= min(oldDot,0). Quality is
    4*sqrt(3)*triangleArea/sum(edgeLengthSquared), in [0,1]. The caller supplies
    signed depth hulls explicitly, usually [min(dBase,dRest,0),max(...)].

    Edge lengths and depth divide by lengthScale; area divides by lengthScale
    squared. Displacement uses squared length divided by lengthScale squared.
    Quality and normal cosines are dimensionless. Direct edge/area residuals
    retain physical tolerances and gradients near thin or compressed elements.
    Zero-area guarded triangles are rejected, including during evaluation:
    neither a fabricated normal nor a finite surrogate Jacobian is returned.

    Global rotation/translation and joint scaling preserve feasible geometry
    when depthOperator/bounds are transformed consistently. Normal coefficients
    are frozen references, not subject-specific truth. Collision, exterior
    geometry, observation residual and rig acceptance remain separate gates.
    """
    baseline = _vertices(baselineVertices, 'baselineVertices')
    rest = _vertices(restVertices, 'restVertices', baseline.shape)
    triangles = np.asarray(faces)
    if (triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or triangles.dtype.kind not in 'iu' or np.any(triangles < 0)
            or np.any(triangles >= len(baseline))
            or np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0)):
        raise ValueError('faces must contain valid nondegenerate integer triangle indices')
    triangles = triangles.astype(np.int64, copy=True)
    interior = _indices(interiorVertexIds, 'interiorVertexIds', len(baseline))
    rim = _indices(rimVertexIds, 'rimVertexIds', len(baseline))
    if np.intersect1d(interior, rim).size:
        raise ValueError('Interior and rim vertices must be disjoint')
    controls = np.sort(np.r_[interior, rim])
    interiorMask = np.zeros(len(baseline), bool); interiorMask[interior] = True
    patchMask = np.zeros(len(baseline), bool); patchMask[controls] = True
    faceIds = np.flatnonzero(interiorMask[triangles].any(axis=1))
    patchFaces = triangles[faceIds]
    if not len(faceIds) or not patchMask[patchFaces].all():
        raise ValueError('Every interior-incident face must be contained in the supplied patch')
    allEdges = np.unique(np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]],
        triangles[:, [2, 0]]]), axis=1), axis=0)
    edges = allEdges[interiorMask[allEdges].any(axis=1)]
    if not len(edges) or not patchMask[edges].all():
        raise ValueError('Every interior-incident edge must be contained in the supplied patch')
    scale = _scalar(lengthScale, 'lengthScale', positive=True)
    effective = DEFAULT_POLICY.copy()
    if policy is not None:
        if not isinstance(policy, dict) or set(policy)-set(effective):
            raise ValueError('Unknown mesh guard policy entries')
        effective.update(policy)
    for name in effective:
        effective[name] = _scalar(effective[name], name, positive=name not in (
            'minimumBaselineNormalCosine', 'restNormalCosineFloor'))
    if (effective['minimumRestEdgeRatio'] > effective['maximumRestEdgeRatio']
            or effective['minimumRestAreaRatio'] > effective['maximumRestAreaRatio']
            or effective['minimumRestQualityRatio'] > 1
            or effective['minimumBaselineNormalCosine'] > 1
            or effective['restNormalCosineFloor'] > 1):
        raise ValueError('Inconsistent mesh guard ratios or cosine thresholds')
    baselineMetrics, restMetrics = _triangleMetrics(baseline, patchFaces), _triangleMetrics(rest, patchFaces)
    baselineLengths = np.linalg.norm(baseline[edges[:, 1]]-baseline[edges[:, 0]], axis=1)
    restLengths = np.linalg.norm(rest[edges[:, 1]]-rest[edges[:, 0]], axis=1)
    if (not np.isfinite(baselineLengths).all() or not np.isfinite(restLengths).all()
            or np.any(baselineLengths <= 0) or np.any(restLengths <= 0)):
        raise ValueError('Guarded edges must have finite positive reference lengths')
    if (not sparse.issparse(depthOperator) or depthOperator.format not in ('csr', 'csc', 'coo')
            or depthOperator.dtype.kind not in 'fiu' or len(depthOperator.shape) != 2
            or depthOperator.shape[0] < 1 or depthOperator.shape[1] != baseline.size
            or not np.isfinite(depthOperator.data).all()):
        raise ValueError('depthOperator must be a finite real sparse Mx(3N) matrix')
    depth = sparse.csr_matrix(depthOperator, dtype=np.float64, copy=True)
    depth.sum_duplicates(); depth.eliminate_zeros()
    if not np.isfinite(depth.data).all() or not patchMask[depth.indices//3].all():
        raise ValueError('Depth operator must have finite support contained in the patch')
    lower, upper = np.asarray(depthLower, float), np.asarray(depthUpper, float)
    if (lower.shape != (depth.shape[0],) or upper.shape != lower.shape
            or not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper)):
        raise ValueError('Depth bounds must be finite ordered vectors matching depthOperator')
    baselineDepths = depth@baseline.ravel()
    if np.any(baselineDepths < lower-1e-12*scale) or np.any(baselineDepths > upper+1e-12*scale):
        raise ValueError('Baseline must lie inside supplied depth bounds')
    # Derive old cosine using the same expression as evaluation to avoid a
    # rounding-only negative residual at a grandfathered orientation boundary.
    oldDot = np.sum(baselineMetrics['cross']*restMetrics['normal'], axis=1)/baselineMetrics['crossLength']
    limits = np.where(interiorMask[controls], effective['interiorDisplacementLimit'], effective['rimDisplacementLimit'])
    spec = {'schemaVersion': 1, 'vertexCount': len(baseline), 'baselineVertices': baseline,
        'restVertices': rest, 'faces': triangles, 'interiorVertexIds': interior, 'rimVertexIds': rim,
        'controlVertexIds': controls, 'edgeVertexIds': edges, 'faceVertexIds': patchFaces, 'faceIds': faceIds,
        'lengthScale': scale, 'policy': effective, 'depthOperator': depth,
        'depthLower': lower.copy(), 'depthUpper': upper.copy(),
        'edgeLower': np.minimum(baselineLengths, effective['minimumRestEdgeRatio']*restLengths),
        'edgeUpper': np.maximum(baselineLengths, effective['maximumRestEdgeRatio']*restLengths),
        'areaLower': np.minimum(baselineMetrics['area'], effective['minimumRestAreaRatio']*restMetrics['area']),
        'areaUpper': np.maximum(baselineMetrics['area'], effective['maximumRestAreaRatio']*restMetrics['area']),
        'qualityLower': np.minimum(baselineMetrics['quality'], effective['minimumRestQualityRatio']*restMetrics['quality']),
        'baselineNormals': baselineMetrics['normal'], 'restNormals': restMetrics['normal'],
        'baselineNormalCosineLower': effective['minimumBaselineNormalCosine'],
        'restNormalCosineLower': np.minimum(oldDot, effective['restNormalCosineFloor']),
        'displacementLimits': limits*scale,
        'diagnostics': {'method': 'uniformInteriorIncidentMeshShapeGuards',
            'edgeCount': len(edges), 'faceCount': len(patchFaces), 'controlVertexCount': len(controls),
            'depthRowCount': depth.shape[0], 'exceptionalElementListsUsed': False,
            'physicalLengthReference': 'oneCallerSpecifiedPhysicalLength', 'columnOrder': 'vertexMajorXYZ',
            'zeroAreaBehavior': 'rejectWithoutSurrogateNormalOrJacobian', 'qualityAccepted': False}}
    baselineValues = evaluateMeshShapeConstraints(baseline, spec, withJacobian=False)['values']
    if baselineValues.min() < -1e-10:
        raise ValueError('Derived mesh guard constraints do not admit the supplied baseline')
    return spec


def _crossGradient(metrics, gradientCross):
    """Chain a scalar cross-product gradient to the triangle's three vertices."""
    u, v = metrics['u'], metrics['v']
    return np.stack([np.cross(gradientCross, v-u), np.cross(v, gradientCross),
                     np.cross(gradientCross, u)], axis=1)


def evaluateMeshShapeConstraints(vertices, guardSpec, withJacobian=True):
    """Return finite dimensionless values (>=0 feasible) and optional CSR J.

    ``groups`` maps each constraint name to a half-open row span. Jacobian
    columns use full-template vertex-major XYZ order. Interior-incident edges
    and faces are deterministic for a given topology; the specification fixes
    their bounds throughout optimization. No candidate-dependent row pruning
    or surrogate values for degenerate triangles is performed.
    """
    if not isinstance(guardSpec, dict) or guardSpec.get('schemaVersion') != 1:
        raise ValueError('Expected a mesh shape guard specification')
    if type(withJacobian) is not bool:
        raise ValueError('withJacobian must be boolean')
    spec = guardSpec
    current = _vertices(vertices, 'vertices', (spec['vertexCount'], 3))
    scale = spec['lengthScale']
    # Evaluate in reference-length units to keep both objective values and
    # intermediate geometric calculations invariant to unit changes.
    normalized = current/scale
    edges, faces, controls = spec['edgeVertexIds'], spec['faceVertexIds'], spec['controlVertexIds']
    edgeVector = normalized[edges[:, 1]]-normalized[edges[:, 0]]
    edgeLengths = np.linalg.norm(edgeVector, axis=1)
    metrics = _triangleMetrics(normalized, faces)
    areas = metrics['area']
    q = metrics['quality']
    values, rowIndices, columnIndices, entries, groups = [], [], [], [], {}
    offset = 0

    def append(name, value, ids=None, gradient=None, linear=None):
        nonlocal offset
        value = np.asarray(value, float)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError(f'Nonfinite or malformed mesh constraint: {name}')
        groups[name] = {'start': offset, 'stop': offset+len(value)}
        values.append(value)
        if withJacobian:
            if linear is not None:
                matrix = linear.tocoo()
                rowIndices.append(matrix.row+offset); columnIndices.append(matrix.col); entries.append(matrix.data)
            else:
                if gradient.shape != (*ids.shape, 3) or not np.isfinite(gradient).all():
                    raise ValueError(f'Nonfinite or malformed mesh Jacobian: {name}')
                rowIndices.append(np.repeat(np.arange(len(value))+offset, ids.shape[1]*3))
                columnIndices.append((3*ids[:, :, None]+np.arange(3)).ravel())
                entries.append(gradient.ravel())
        offset += len(value)

    edgeGradient = np.stack([-edgeVector/edgeLengths[:, None], edgeVector/edgeLengths[:, None]], axis=1)/scale if withJacobian else None
    append('edgeLower', edgeLengths-spec['edgeLower']/scale, edges, edgeGradient)
    append('edgeUpper', spec['edgeUpper']/scale-edgeLengths, edges, -edgeGradient if withJacobian else None)
    areaGradient = _crossGradient(metrics, .5*metrics['normal'])/scale if withJacobian else None
    append('areaLower', areas-spec['areaLower']/scale**2, faces, areaGradient)
    append('areaUpper', spec['areaUpper']/scale**2-areas, faces, -areaGradient if withJacobian else None)
    qualityGradient = None
    if withJacobian:
        points = metrics['points']
        sumGradient = 2*(3*points-points.sum(axis=1, keepdims=True))
        qualityGradient = (_crossGradient(metrics, (2*np.sqrt(3.)/metrics['edgeSquaredSum'])[:, None]*metrics['normal'])
            - (q/metrics['edgeSquaredSum'])[:, None, None]*sumGradient)/scale
    append('triangleQuality', q-spec['qualityLower'], faces, qualityGradient)
    for name, normals, lower in (
        ('normalToBaseline', spec['baselineNormals'], spec['baselineNormalCosineLower']),
        ('normalToRest', spec['restNormals'], spec['restNormalCosineLower'])):
        cosine = np.sum(metrics['cross']*normals, axis=1)/metrics['crossLength']
        gradient = None
        if withJacobian:
            gradientCross = (normals-cosine[:, None]*metrics['normal'])/metrics['crossLength'][:, None]
            gradient = _crossGradient(metrics, gradientCross)/scale
        append(name, cosine-lower, faces, gradient)
    depth = spec['depthOperator']@current.ravel()
    append('depthLower', (depth-spec['depthLower'])/scale, linear=spec['depthOperator']/scale)
    append('depthUpper', (spec['depthUpper']-depth)/scale, linear=-spec['depthOperator']/scale)
    delta = (current[controls]-spec['baselineVertices'][controls])/scale
    append('controlDisplacement', (spec['displacementLimits']/scale)**2-np.sum(delta**2, axis=1),
           controls[:, None], (-2*delta[:, None, :]/scale) if withJacobian else None)
    result = {'values': np.concatenate(values), 'groups': groups}
    if withJacobian:
        jacobian = sparse.csr_matrix((np.concatenate(entries), (np.concatenate(rowIndices), np.concatenate(columnIndices))),
            shape=(offset, current.size))
        jacobian.sum_duplicates(); jacobian.eliminate_zeros()
        if not np.isfinite(jacobian.data).all():
            raise ValueError('Mesh constraint Jacobian is nonfinite')
        result['jacobian'] = jacobian
    return result
