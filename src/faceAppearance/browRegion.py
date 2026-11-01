"""Conservative, model-free surface weights around photographic eyebrow proxies.

Parser labels 2/3 describe image eyebrows, not anatomical brow-ridge ground
truth. Photo distances use original pixels and an explicit eye-distance scale.
Protection uses shortest paths along 3D mesh edges, normalized by the mesh's
maximum axis span. Canonical coordinates are not calibrated physical units.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, label, map_coordinates
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .texture import vertexNormals


def _smoothStep(value):
    value = np.clip(value, 0., 1.)
    return value*value*(3.-2.*value)


def _graphDistance(graph, sourceMask):
    indices = np.flatnonzero(sourceMask)
    if not len(indices):
        return np.full(graph.shape[0], np.inf)
    return np.asarray(dijkstra(graph, directed=False, indices=indices, min_only=True))


def buildBrowRegion(vertices, faces, camera, labelMap, confidenceMap,
                   vertexVisibility, eyeBoundaryIds, *, eyeDistancePixels,
                   minConfidence=.8, coreRadiusEyeFraction=.06,
                   outerRadiusEyeFraction=.36, eyeProtectedDistanceFraction=.02,
                   eyeFalloffDistanceFraction=.06):
    """Select a smooth visible-front ROI while fixing eye and mesh boundaries.

    ``camera`` is an OpenCV camera in the same EXIF-oriented original-photo
    frame as the semantic maps. ``vertexVisibility`` must be a caller-computed
    depth-tested boolean V mask; this function additionally checks depth,
    frustum and front-facing normals. ``eyeBoundaryIds`` is a nonempty 1-D
    integer list of actual topological boundary vertices, normally both eyes.

    High-confidence eyebrow components need at least one visible projected
    vertex before they can seed the distance field. Photo-distance weights are
    one through ``coreRadiusEyeFraction`` and smoothly fall to zero at
    ``outerRadiusEyeFraction``. Eye graph distance is strictly protected through
    ``eyeProtectedDistanceFraction`` and smoothly releases at
    ``eyeFalloffDistanceFraction``, both as fractions of maximum mesh axis span.
    All other mesh boundaries, and transitions into invisible/back-facing/
    out-of-frame vertices, receive a smooth taper over .01 of that span.
    Hard-protected vertices are exactly zero.

    No inputs are modified. Missing/unreliable eyebrow support returns a valid
    all-zero region with an explicit diagnostic status, not guessed anatomy.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    labels, confidence = np.asarray(labelMap), np.asarray(confidenceMap)
    visibility, eyeIds = np.asarray(vertexVisibility), np.asarray(eyeBoundaryIds)
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3
            or not np.isfinite(vertices).all() or faces.ndim != 2
            or faces.shape[1:] != (3,) or not len(faces) or faces.dtype.kind not in 'iu'
            or faces.min() < 0 or faces.max() >= len(vertices)
            or np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0)):
        raise ValueError('Expected finite Vx3 vertices and valid integer triangle faces')
    faces = faces.astype(np.int64, copy=False)
    if (labels.ndim != 2 or min(labels.shape) < 2 or labels.dtype.kind not in 'iu'
            or labels.min() < 0 or labels.max() > 18 or confidence.shape != labels.shape
            or confidence.dtype.kind not in 'fiu' or not np.isfinite(confidence).all()
            or confidence.min() < 0 or confidence.max() > 1):
        raise ValueError('Expected standard 19-label maps and finite confidence in [0,1]')
    if visibility.shape != (len(vertices),) or visibility.dtype != np.bool_:
        raise ValueError('vertexVisibility must be a boolean V mask')
    if (eyeIds.ndim != 1 or not len(eyeIds) or eyeIds.dtype.kind not in 'iu'
            or eyeIds.min() < 0 or eyeIds.max() >= len(vertices)):
        raise ValueError('eyeBoundaryIds must be a nonempty integer vertex-index list')
    eyeIds = np.unique(eyeIds.astype(np.int64, copy=False))
    try:
        intrinsics = np.asarray(camera['intrinsics'], dtype=np.float64)
        rotation = np.asarray(camera['rotation'], dtype=np.float64)
        translation = np.asarray(camera['translation'], dtype=np.float64)
    except (TypeError, KeyError) as error:
        raise ValueError('Expected an OpenCV intrinsics/rotation/translation camera') from error
    if (intrinsics.shape != (3, 3) or rotation.shape != (3, 3) or translation.size != 3
            or not np.isfinite(intrinsics).all() or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all() or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0
            or not np.allclose(intrinsics[2], [0., 0., 1.], atol=1e-8)
            or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-5)):
        raise ValueError('Invalid finite OpenCV camera or proper rotation')
    scalars = (eyeDistancePixels, minConfidence, coreRadiusEyeFraction,
               outerRadiusEyeFraction, eyeProtectedDistanceFraction, eyeFalloffDistanceFraction)
    if (any(isinstance(value, (bool, np.bool_)) or not np.isscalar(value) for value in scalars)
            or not np.isfinite(scalars).all() or eyeDistancePixels <= 0 or not 0 <= minConfidence <= 1
            or not 0 <= coreRadiusEyeFraction < outerRadiusEyeFraction
            or not 0 <= eyeProtectedDistanceFraction < eyeFalloffDistanceFraction):
        raise ValueError('Invalid finite positive eye scale or ordered ROI/protection fractions')
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
        raise ValueError('Duplicate triangles make boundary protection ambiguous')
    cross = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                     vertices[faces[:, 2]]-vertices[faces[:, 0]])
    if np.any(np.linalg.norm(cross, axis=1) == 0):
        raise ValueError('Degenerate triangles cannot define front-facing surface normals')
    edges, counts = np.unique(np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]],
                                                       faces[:, [2, 0]])), axis=1), axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError('Non-manifold edges make boundary protection ambiguous')
    boundary = np.zeros(len(vertices), bool)
    boundary[np.unique(edges[counts == 1])] = True
    if not boundary[eyeIds].all():
        raise ValueError('eyeBoundaryIds must address actual mesh boundary vertices')
    cameraVertices = vertices@rotation.T+translation.reshape(3)
    positive = cameraVertices[:, 2] > 1e-8
    projected = cameraVertices@intrinsics.T
    pixels = np.full((len(vertices), 2), np.nan)
    pixels[positive] = projected[positive, :2]/projected[positive, 2:3]
    projectable = positive & np.isfinite(pixels).all(1)
    height, width = labels.shape
    inside = projectable & (pixels[:, 0] >= 0) & (pixels[:, 0] <= width-1) & (pixels[:, 1] >= 0) & (pixels[:, 1] <= height-1)
    viewDirection = -cameraVertices/np.maximum(np.linalg.norm(cameraVertices, axis=1, keepdims=True), 1e-12)
    front = np.sum((vertexNormals(vertices, faces)@rotation.T)*viewDirection, axis=1) > 0
    eligible = visibility & inside & front
    headSpan = float(np.ptp(vertices, axis=0).max())
    graphEdges = edges
    lengths = np.linalg.norm(vertices[graphEdges[:, 0]]-vertices[graphEdges[:, 1]], axis=1)
    lengths = np.maximum(lengths, np.finfo(np.float64).eps*headSpan)
    graph = coo_matrix((np.r_[lengths, lengths],
                       (np.r_[graphEdges[:, 0], graphEdges[:, 1]], np.r_[graphEdges[:, 1], graphEdges[:, 0]])),
                      shape=(len(vertices), len(vertices))).tocsr()
    eyeMask = np.zeros(len(vertices), bool); eyeMask[eyeIds] = True
    eyeDistance = _graphDistance(graph, eyeMask)
    boundaryDistance = _graphDistance(graph, boundary)
    eligibilityDistance = _graphDistance(graph, ~eligible)
    ids = np.flatnonzero(eligible)
    nearest = np.floor(pixels[ids]+.5).astype(int)
    seeds = np.zeros(len(vertices), bool)
    acceptedPixels = np.zeros(labels.shape, bool)
    acceptedComponents, seedsByLabel = {}, {}
    for labelId in (2, 3):
        components, _ = label((labels == labelId) & (confidence >= minConfidence))
        sampled = components[nearest[:, 1], nearest[:, 0]]
        componentIds = np.unique(sampled[sampled > 0])
        acceptedPixels |= np.isin(components, componentIds) if len(componentIds) else False
        labelSeeds = ids[sampled > 0]
        seeds[labelSeeds] = True
        acceptedComponents[str(labelId)] = componentIds.tolist()
        seedsByLabel[str(labelId)] = int(len(labelSeeds))
    photoDistance = np.full(len(vertices), np.inf)
    if acceptedPixels.any():
        distanceField = distance_transform_edt(~acceptedPixels)
        photoDistance[ids] = map_coordinates(distanceField, pixels[ids, ::-1].T, order=1, mode='nearest')
    coreRadius, outerRadius = eyeDistancePixels*np.asarray((coreRadiusEyeFraction, outerRadiusEyeFraction))
    protectedRadius, releaseRadius = headSpan*np.asarray((eyeProtectedDistanceFraction, eyeFalloffDistanceFraction))
    boundaryFeatherRadius = .01*headSpan
    photoWeight = 1.-_smoothStep((photoDistance-coreRadius)/(outerRadius-coreRadius))
    eyeWeight = _smoothStep((eyeDistance-protectedRadius)/(releaseRadius-protectedRadius))
    boundaryWeight = _smoothStep(boundaryDistance/boundaryFeatherRadius)
    eligibilityWeight = _smoothStep(eligibilityDistance/boundaryFeatherRadius)
    protected = boundary | ~eligible | (eyeDistance <= protectedRadius)
    weights = (photoWeight*eyeWeight*boundaryWeight*eligibilityWeight).astype(np.float32)
    weights[protected] = 0.
    if not seeds.any():
        weights[:] = 0.
    diagnostics = {'status': 'ready' if weights.any() else ('noReliableEyebrowSeeds' if not seeds.any() else 'fullyProtected'),
        'vertexCount': len(vertices), 'eligibleVertexCount': int(eligible.sum()), 'eyebrowSeedCount': int(seeds.sum()),
        'eyebrowSeedCountByLabel': seedsByLabel, 'acceptedPhotoComponentIdsByLabel': acceptedComponents,
        'activeVertexCount': int((weights > 0).sum()), 'fullyWeightedVertexCount': int((weights == 1).sum()),
        'partialWeightVertexCount': int(((weights > 0) & (weights < 1)).sum()),
        'protectedVertexCount': int(protected.sum()), 'meshBoundaryVertexCount': int(boundary.sum()),
        'declaredEyeBoundaryVertexCount': len(eyeIds), 'eyeDistancePixels': float(eyeDistancePixels),
        'minimumSemanticConfidence': float(minConfidence), 'coreRadiusPixels': float(coreRadius),
        'outerRadiusPixels': float(outerRadius), 'headMaximumAxisSpanCanonical': headSpan,
        'eyeProtectedRadiusCanonical': float(protectedRadius), 'eyeReleaseRadiusCanonical': float(releaseRadius),
        'eyeProtectedDistanceFraction': float(eyeProtectedDistanceFraction),
        'eyeFalloffDistanceFraction': float(eyeFalloffDistanceFraction),
        'boundaryAndVisibilityFeatherCanonical': boundaryFeatherRadius,
        'boundaryAndVisibilityFeatherFractionOfHeadSpan': .01,
        'graphDistanceMetric': 'shortest mesh-edge path, each edge weighted by canonical 3D Euclidean length',
        'smoothing': 'cubic smoothstep with zero endpoint derivatives; multiplicative photo/eye/boundary/visibility tapers',
        'inputVisibilityTrusted': 'caller supplied depth-tested mask; depth/frustum/front-facing checks additionally enforced',
        'limits': ['Image eyebrow semantics are a proxy, not verified anatomical brow-ridge annotations.',
                   'Canonical graph distances are not calibrated physical units such as millimeters.',
                   'Weights select a region; they do not by themselves prevent self-intersection or establish fairing quality.']}
    return {'regionWeights': weights, 'eyebrowSeedMask': seeds, 'protectedMask': protected,
            'eligibleMask': eligible, 'meshBoundaryMask': boundary, 'photoDistancePixels': photoDistance,
            'eyeGraphDistanceCanonical': eyeDistance, 'boundaryGraphDistanceCanonical': boundaryDistance,
            'eligibilityGraphDistanceCanonical': eligibilityDistance, 'diagnostics': diagnostics}
