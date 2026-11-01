"""Ordered eye-corner and eyelid correspondences, independent of detector library.

Photo contours use original-image XY pixels. Both upper/lower curves always run
from the inner to the outer corner. Automatic semantic endpoints are explicitly
proxies, not anatomical ground truth. A template loop is bound once; subsequent
views cannot silently change its vertex endpoints or swap its upper/lower paths.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
from scipy.ndimage import gaussian_filter1d, label as connectedComponents, map_coordinates

from .anatomicalBoundaries import extractMeshBoundaryLoops, meshTopologyHash
from .camera import projectPoints


def _samplePolyline(points, sampleCount, confidence=None):
    points = np.asarray(points, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or not np.isfinite(points).all() or not isinstance(sampleCount, int) or sampleCount < 3):
        raise ValueError('Expected a finite XY polyline and at least three samples')
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if distance[-1] <= 1e-8:
        raise ValueError('Eye curve must have nonzero length')
    keep = np.r_[True, np.diff(distance) > 1e-9]
    parameters = np.linspace(0, distance[-1], sampleCount)
    result = np.stack([np.interp(parameters, distance[keep], points[keep, axis]) for axis in (0, 1)], 1)
    if confidence is None:
        return result
    confidence = np.asarray(confidence, float)
    if (confidence.shape != (len(points),) or not np.isfinite(confidence).all()
            or np.any((confidence < 0) | (confidence > 1))):
        raise ValueError('Curve confidence must be a matching finite vector in [0,1]')
    return result, np.interp(parameters, distance[keep], confidence[keep])


def _curveTangents(curve):
    tangent = np.gradient(np.asarray(curve, float), axis=0)
    return tangent/np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)


def eyeContourMetrics(innerCorner, outerCorner, upperCurve, lowerCurve):
    """Measure image-space corner aperture and endpoint directions, in degrees."""
    inner, outer = np.asarray(innerCorner, float), np.asarray(outerCorner, float)
    upper, lower = np.asarray(upperCurve, float), np.asarray(lowerCurve, float)
    if (inner.shape != (2,) or outer.shape != (2,) or upper.ndim != 2 or upper.shape[1] != 2
            or lower.shape != upper.shape or len(upper) < 3
            or not all(np.isfinite(value).all() for value in (inner, outer, upper, lower))):
        raise ValueError('Invalid eye metric inputs')
    def angle(first, second):
        denominator = np.linalg.norm(first)*np.linalg.norm(second)
        return float(np.degrees(np.arccos(np.clip(np.dot(first, second)/max(denominator, 1e-12), -1, 1))))
    # Several samples damp single raster-pixel jaggies without moving endpoints.
    step = min(3, len(upper)-2)
    upperInner, lowerInner = upper[step]-inner, lower[step]-inner
    upperOuter, lowerOuter = upper[-step-1]-outer, lower[-step-1]-outer
    chord = outer-inner
    width = float(np.linalg.norm(chord))
    return {'cornerDistancePixels': width,
            'innerApertureDegrees': angle(upperInner, lowerInner),
            'outerApertureDegrees': angle(upperOuter, lowerOuter),
            'cornerAxisDegrees': float(np.degrees(np.arctan2(chord[1], chord[0]))),
            'upperInnerTangentDegrees': float(np.degrees(np.arctan2(upperInner[1], upperInner[0]))),
            'lowerInnerTangentDegrees': float(np.degrees(np.arctan2(lowerInner[1], lowerInner[0]))),
            'upperOuterInwardTangentDegrees': float(np.degrees(np.arctan2(upperOuter[1], upperOuter[0]))),
            'lowerOuterInwardTangentDegrees': float(np.degrees(np.arctan2(lowerOuter[1], lowerOuter[0]))),
            'maximumAperturePixels': float(np.linalg.norm(upper-lower, axis=1).max()),
            'meaning': 'Image-space proxy geometry; angles are not anatomical ground truth'}


def makeEyeContourObservation(labelId, innerCorner, outerCorner, upperCurve, lowerCurve, *,
                              upperConfidence=None, lowerConfidence=None, cornerConfidence=None,
                              sampleCount=24, source='explicitCurveObservation', minAspect=.10,
                              maxAspect=.85, minCornerDistance=8.):
    """Normalize a semantic/detector/manual observation into the same contract.

    Curves must already be labelled upper/lower and ordered inner-to-outer.
    Endpoints farther than one pixel from their declared corner are rejected;
    valid endpoints are snapped exactly to the explicit corner after resampling.
    Supplied confidence is provenance-dependent, not a calibrated probability.
    """
    inner, outer = np.asarray(innerCorner, float), np.asarray(outerCorner, float)
    upper, lower = np.asarray(upperCurve, float), np.asarray(lowerCurve, float)
    if (not isinstance(labelId, (int, np.integer)) or isinstance(labelId, bool)
            or inner.shape != (2,) or outer.shape != (2,) or not np.isfinite(inner).all()
            or not np.isfinite(outer).all() or not np.isfinite([minAspect, maxAspect, minCornerDistance]).all()
            or not 0 <= minAspect < maxAspect or minCornerDistance <= 0):
        raise ValueError('Invalid eye label, corner coordinates or shape gates')
    for curve in (upper, lower):
        if (curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 3 or not np.isfinite(curve).all()
                or np.linalg.norm(curve[0]-inner) > 1 or np.linalg.norm(curve[-1]-outer) > 1):
            raise ValueError('Eye curves must run from the declared inner corner to outer corner')
    upperConfidence = np.ones(len(upper)) if upperConfidence is None else upperConfidence
    lowerConfidence = np.ones(len(lower)) if lowerConfidence is None else lowerConfidence
    upper, upperConfidence = _samplePolyline(upper, sampleCount, upperConfidence)
    lower, lowerConfidence = _samplePolyline(lower, sampleCount, lowerConfidence)
    corners = np.asarray([upperConfidence[0], upperConfidence[-1]]) if cornerConfidence is None else np.asarray(cornerConfidence, float)
    if corners.shape != (2,) or not np.isfinite(corners).all() or np.any((corners < 0) | (corners > 1)):
        raise ValueError('Expected two finite corner confidence values in [0,1]')
    upper[[0, -1]], lower[[0, -1]] = np.stack([inner, outer]), np.stack([inner, outer])
    metrics = eyeContourMetrics(inner, outer, upper, lower)
    width = metrics['cornerDistancePixels']
    if width < minCornerDistance:
        return {'labelId': int(labelId), 'valid': False, 'reason': 'eyeTooSmallOrForeshortened', 'diagnostics': metrics}
    chord = (outer-inner)/width
    downward = np.array([-chord[1], chord[0]])
    downward *= 1 if downward[1] >= 0 else -1
    aperture = (lower-upper)@downward
    aspect = float(aperture.max()/width)
    if aspect < minAspect:
        return {'labelId': int(labelId), 'valid': False, 'reason': 'closedOrUncertainEyeAperture', 'diagnostics': {**metrics, 'aspect': aspect}}
    if aspect > maxAspect or np.mean(aperture[1:-1] < -.25) > .1:
        return {'labelId': int(labelId), 'valid': False, 'reason': 'swappedCrossingOrImplausibleEyelids', 'diagnostics': {**metrics, 'aspect': aspect}}
    return {'labelId': int(labelId), 'valid': True, 'reason': 'accepted', 'source': str(source),
            'innerCorner': inner.copy(), 'outerCorner': outer.copy(),
            'upperCurve': upper, 'lowerCurve': lower, 'upperConfidence': upperConfidence,
            'lowerConfidence': lowerConfidence, 'cornerConfidence': corners,
            'diagnostics': {**metrics, 'aspect': aspect}}


def convertCachedEyeLandmarks(cache, cacheReport, *, sampleCount=24, minimumRawPeak=.6):
    """Adapt the explicit STFR/OpenCV cache contract without loading its model.

    Input eyes are image-left/image-right, each with lower/upper curves ordered
    image-left endpoint to image-right endpoint. Semantic identity is supplied
    by the cache's verified imageSideToSemanticLabel mapping, never guessed from
    anatomical label names. minimumRawPeak is a conservative experimental whole-
    eye gate, not an estimate of visibility or detector accuracy.
    """
    curves = np.asarray(cache['eyelidCurves'], float)
    corners = np.asarray(cache['corners'], float)
    confidence = np.asarray(cache['curveConfidence'], float)
    peaks = np.asarray(cache['rawPeakScores'], float)
    validPoints = np.asarray(cache.get('pointValidMask', np.ones(confidence.shape, bool)))
    metadata = cacheReport.get('metadata', {})
    mapping = cacheReport.get('imageSideToSemanticLabel', {})
    quality = cacheReport.get('quality', [])
    if (curves.ndim != 4 or curves.shape[:2] != (2, 2) or curves.shape[-1] != 2 or curves.shape[2] < 3
            or corners.shape != (2, 2, 2) or confidence.shape != curves.shape[:-1]
            or peaks.shape != confidence.shape or validPoints.shape != confidence.shape or validPoints.dtype != np.bool_
            or not all(np.isfinite(value).all() for value in (curves, corners, confidence, peaks))
            or np.any((confidence < 0) | (confidence > 1)) or not np.isfinite(minimumRawPeak) or minimumRawPeak < 0
            or metadata.get('eyeOrder') != ['imageLeft', 'imageRight']
            or metadata.get('curveOrder') != ['lower', 'upper']
            or metadata.get('cornerOrder') != ['imageLeftEndpoint', 'imageRightEndpoint']
            or set(mapping) != {'imageLeft', 'imageRight'} or len(set(mapping.values())) != 2
            or len(quality) != 2 or len(metadata.get('originalSize', [])) != 2):
        raise ValueError('Unsupported or invalid cached eye-landmark contract')
    eyes = {}
    for index, side in enumerate(('imageLeft', 'imageRight')):
        labelId = mapping[side]
        if not isinstance(labelId, int):
            raise ValueError('Cached semantic eye identity must be an integer label')
        minimum = float(peaks[index].min())
        if not quality[index].get('valid', False) or not validPoints[index].all() or minimum < minimumRawPeak:
            eyes[labelId] = {'labelId': labelId, 'valid': False, 'reason': 'uncertainOrPotentiallyOccludedLandmarkEye',
                'source': 'STFRMergeFanEye/OpenCvCpu', 'diagnostics': {'minimumRawPeak': minimum,
                    'minimumRawPeakGate': minimumRawPeak, 'detectorQuality': quality[index],
                    'confidenceMeaning': 'Heuristic gate; low score may reflect occlusion, not calibrated probability'}}
            continue
        reverse = index == 0
        order = np.arange(curves.shape[2]-1, -1, -1) if reverse else np.arange(curves.shape[2])
        inner, outer = corners[index, 1 if reverse else 0], corners[index, 0 if reverse else 1]
        upper, lower = curves[index, 1, order].copy(), curves[index, 0, order].copy()
        upperConfidence, lowerConfidence = confidence[index, 1, order], confidence[index, 0, order]
        # The cache corners are the mean of tiny upper/lower endpoint offsets.
        # Keep that explicit shared endpoint instead of inventing two canthi.
        upper[[0, -1]], lower[[0, -1]] = np.stack([inner, outer]), np.stack([inner, outer])
        cornerConfidence = np.minimum(upperConfidence[[0, -1]], lowerConfidence[[0, -1]])
        observation = makeEyeContourObservation(labelId, inner, outer, upper, lower,
            upperConfidence=upperConfidence, lowerConfidence=lowerConfidence, cornerConfidence=cornerConfidence,
            sampleCount=sampleCount, source='STFRMergeFanEye/OpenCvCpu')
        observation['diagnostics'].update({'minimumRawPeak': minimum, 'minimumRawPeakGate': minimumRawPeak,
            'imageSide': side, 'semanticIdentitySource': 'cacheVerifiedSemanticCenterAssignment',
            'detectorQuality': quality[index], 'confidenceMeaning': 'Uncalibrated heatmap peak weighting',
            'cornerMeaning': 'Detector endpoint prediction, not anatomical ground truth'})
        eyes[labelId] = observation
    return {'schemaVersion': 1, 'eyes': eyes, 'imageShape': list(reversed(metadata['originalSize'])),
            'coordinateFrame': 'originalExifOrientedPixels', 'source': 'STFRMergeFanEye/OpenCvCpu',
            'diagnostics': {'minimumRawPeakGate': minimumRawPeak, 'modelVisibilityNotAssumed': True}}


def _walkLoop(loop, first, last, direction):
    count = len(loop)
    positions = (first+direction*np.arange(count)) % count
    return np.asarray(loop)[positions[:int(np.flatnonzero(positions == last)[0])+1]]


def extractEyeContours(labelMap, confidence, *, eyeLabelIds=(4, 5), sampleCount=24,
                       faceCenter=None, viewYawDegrees=None, maxAbsYawDegrees=65.,
                       minArea=32, minMeanConfidence=.75, minComponentFraction=.85,
                       minAspect=.10, maxAspect=.85, minCornerDistance=12., smoothingSigma=.6):
    """Extract separate largest eye contours from an existing semantic map.

    Two PCA-axis extremes are automatic corner proxies. The one nearer the
    supplied faceCenter (or nose-label centroid) is called inner. Low-confidence,
    fragmented, clipped, tiny, closed and explicitly too-oblique observations
    fail with a reason. Segmentation alone cannot prove absence of occlusion;
    callers must still depth-test each bound 3D point in its photo camera.
    """
    import cv2

    labels, confidence = np.asarray(labelMap), np.asarray(confidence, float)
    if (labels.ndim != 2 or labels.dtype.kind not in 'iu' or min(labels.shape) < 3
            or confidence.shape != labels.shape or not np.isfinite(confidence).all()
            or np.any((confidence < 0) | (confidence > 1)) or len(set(eyeLabelIds)) != len(eyeLabelIds)
            or not isinstance(sampleCount, int) or sampleCount < 6
            or not isinstance(minArea, int) or minArea < 3
            or not np.isfinite([minMeanConfidence, minComponentFraction, maxAbsYawDegrees, smoothingSigma]).all()
            or not 0 <= minMeanConfidence <= 1 or not 0 < minComponentFraction <= 1
            or not 0 < maxAbsYawDegrees < 90 or smoothingSigma < 0):
        raise ValueError('Invalid semantic maps or contour extraction configuration')
    eyes = {}
    if viewYawDegrees is not None and (not np.isfinite(viewYawDegrees) or abs(viewYawDegrees) > maxAbsYawDegrees):
        return {'schemaVersion': 1, 'eyes': {int(labelId): {'labelId': int(labelId), 'valid': False,
            'reason': 'excessiveOrUnknownYaw'} for labelId in eyeLabelIds}, 'imageShape': list(labels.shape)}
    if faceCenter is None:
        yy, xx = np.nonzero(labels == 10)
        if len(xx):
            faceCenter = np.array([np.median(xx), np.median(yy)])
        else:
            centers = []
            for labelId in eyeLabelIds:
                yy, xx = np.nonzero(labels == labelId)
                if len(xx):
                    centers.append([np.mean(xx), np.mean(yy)])
            if len(centers) == 2:
                faceCenter = np.mean(centers, axis=0)
    if faceCenter is None or np.shape(faceCenter) != (2,) or not np.isfinite(faceCenter).all():
        return {'schemaVersion': 1, 'eyes': {int(labelId): {'labelId': int(labelId), 'valid': False,
            'reason': 'missingReliableFaceCenter'} for labelId in eyeLabelIds}, 'imageShape': list(labels.shape)}
    faceCenter = np.asarray(faceCenter, float)
    for labelId in eyeLabelIds:
        def reject(reason, **diagnostics):
            eyes[int(labelId)] = {'labelId': int(labelId), 'valid': False, 'reason': reason, 'diagnostics': diagnostics}
        region = labels == labelId
        components, count = connectedComponents(region)
        if not count:
            reject('missingEyeLabel')
            continue
        areas = np.bincount(components.ravel())[1:]
        largest = int(np.argmax(areas))+1
        component = components == largest
        area, fraction = int(component.sum()), float(component.sum()/region.sum())
        if area < minArea or fraction < minComponentFraction:
            reject('smallOrFragmentedEyeRegion', area=area, componentFraction=fraction)
            continue
        if component[0].any() or component[-1].any() or component[:, 0].any() or component[:, -1].any():
            reject('eyeTouchesPhotoBoundary', area=area)
            continue
        meanConfidence = float(confidence[component].mean())
        if meanConfidence < minMeanConfidence:
            reject('lowRegionConfidence', meanConfidence=meanConfidence)
            continue
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        boundary = max(contours, key=len)[:, 0].astype(float)
        if len(boundary) < 8:
            reject('insufficientBoundarySamples')
            continue
        _, _, axes = np.linalg.svd(boundary-boundary.mean(0), full_matrices=False)
        axis = axes[0]*(1 if axes[0, 0] >= 0 else -1)
        if abs(axis[0]) < np.cos(np.radians(50)):
            reject('implausibleEyeAxisTilt')
            continue
        along = (boundary-boundary.mean(0))@axis
        endpoints = [int(np.argmin(along)), int(np.argmax(along))]
        nearer = int(np.argmin([np.linalg.norm(boundary[index]-faceCenter) for index in endpoints]))
        innerIndex, outerIndex = endpoints[nearer], endpoints[1-nearer]
        firstPath = _walkLoop(np.arange(len(boundary)), innerIndex, outerIndex, 1)
        secondPath = _walkLoop(np.arange(len(boundary)), innerIndex, outerIndex, -1)
        chord = boundary[outerIndex]-boundary[innerIndex]
        downward = np.array([-chord[1], chord[0]])
        downward *= 1 if downward[1] >= 0 else -1
        if np.mean((boundary[firstPath]-boundary[innerIndex])@downward) > np.mean((boundary[secondPath]-boundary[innerIndex])@downward):
            firstPath, secondPath = secondPath, firstPath
        curves, curveConfidence = [], []
        for path in (firstPath, secondPath):
            points = boundary[path]
            values = confidence[points[:, 1].astype(int), points[:, 0].astype(int)]
            sampled, sampledConfidence = _samplePolyline(points, sampleCount, values)
            if smoothingSigma > 0:
                sampled = gaussian_filter1d(sampled, smoothingSigma, axis=0, mode='nearest')
                sampled[[0, -1]] = boundary[[innerIndex, outerIndex]]
            curves.append(sampled)
            curveConfidence.append(sampledConfidence)
        cornerPixels = boundary[[innerIndex, outerIndex]].astype(int)
        observation = makeEyeContourObservation(int(labelId), boundary[innerIndex], boundary[outerIndex],
            curves[0], curves[1], upperConfidence=curveConfidence[0], lowerConfidence=curveConfidence[1],
            cornerConfidence=confidence[cornerPixels[:, 1], cornerPixels[:, 0]], sampleCount=sampleCount,
            source='semanticLargestComponentPcaEndpointProxy', minAspect=minAspect, maxAspect=maxAspect,
            minCornerDistance=minCornerDistance)
        observation['diagnostics'].update({'regionArea': area, 'meanRegionConfidence': meanConfidence,
            'componentFraction': fraction, 'pcaAxis': axis.tolist(),
            'cornerMeaning': 'Automatic semantic contour extremes, not measured anatomical canthi',
            'smoothingSigmaInCurveSamples': smoothingSigma, 'endpointsLockedDuringSmoothing': True})
        eyes[int(labelId)] = observation
    return {'schemaVersion': 1, 'imageShape': list(labels.shape), 'eyes': eyes,
            'faceCenter': faceCenter, 'coordinateFrame': 'originalExifOrientedPixels'}


def _project(vertices, camera):
    return projectPoints(vertices, camera['intrinsics'], camera['rotation'], camera['translation'])


def _bindMeshPath(path, projectedVertices, vertexDepth, sampleCount):
    points = projectedVertices[path]
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if arc[-1] <= 1e-8 or np.any(vertexDepth[path] <= 0) or np.any(np.diff(arc) <= 1e-9):
        raise ValueError('Template eyelid path has collapsed projection or nonpositive depth')
    query = np.linspace(0, arc[-1], sampleCount)
    segments = np.minimum(np.searchsorted(arc, query, side='right')-1, len(path)-2)
    segments = np.maximum(segments, 0)
    fraction = (query-arc[segments])/(arc[segments+1]-arc[segments])
    indices = np.stack([path[segments], path[segments+1]], 1)
    barycentric = np.stack([1-fraction, fraction], 1)/vertexDepth[indices]
    barycentric /= barycentric.sum(1, keepdims=True)
    return {'vertexIndices': indices, 'barycentric': barycentric, 'pathVertexIndices': path.copy()}


def bindTemplateEyeContours(vertices, faces, eyeLoops, referenceCamera, referenceContours, *,
                            sampleCount=24, cornerOverrides=None, maxCenterDistanceFraction=1.):
    """Freeze template corners, upper/lower paths and one-to-one eye label IDs.

    Default endpoints are projected PCA extremes. ``cornerOverrides`` can supply
    reviewed {name: {innerVertex: int, outerVertex: int}} to replace these proxies.
    Only complete actual topology boundary loops are accepted. The label mapping
    is chosen jointly and bijectively from reference centers, then stays fixed.
    """
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    extracted = extractMeshBoundaryLoops(vertices, faces)
    if (not isinstance(sampleCount, int) or sampleCount < 6 or len(eyeLoops) != 2
            or not np.isfinite(maxCenterDistanceFraction) or maxCenterDistanceFraction <= 0):
        raise ValueError('Expected two eye loops and valid binding configuration')
    referenceEyes = {int(key): value for key, value in referenceContours['eyes'].items() if value.get('valid')}
    if len(referenceEyes) != 2:
        raise ValueError('Both reliable reference eye observations are required for initial binding')
    projected, depth = _project(vertices, referenceCamera)
    names = list(eyeLoops)
    loops = {}
    for name, loop in eyeLoops.items():
        loop = np.asarray(loop)
        if (loop.ndim != 1 or loop.dtype.kind not in 'iu' or len(np.unique(loop)) != len(loop)
                or not any(set(loop.tolist()) == set(value.tolist()) for value in extracted['loops'])):
            raise ValueError('Template eye path must be one complete real topology boundary loop')
        # Check that caller order itself follows the boundary, not a shuffled set.
        canonical = next(value for value in extracted['loops'] if set(loop.tolist()) == set(value.tolist()))
        edges = {tuple(sorted(pair)) for pair in zip(canonical, np.roll(canonical, -1))}
        if any(tuple(sorted(pair)) not in edges for pair in zip(loop, np.roll(loop, -1))):
            raise ValueError('Template eye loop vertex order must follow connected boundary edges')
        loops[name] = loop.astype(np.int64)
    centers = [projected[loops[name]].mean(0) for name in names]
    labels = list(referenceEyes)
    assigned = min(permutations(labels), key=lambda permutation: sum(np.linalg.norm(
        centers[index]-(np.asarray(referenceEyes[label]['innerCorner'])+np.asarray(referenceEyes[label]['outerCorner']))/2)
        for index, label in enumerate(permutation)))
    cornerOverrides = {} if cornerOverrides is None else cornerOverrides
    results = {}
    for name, labelId, center in zip(names, assigned, centers):
        loop, observation = loops[name], referenceEyes[labelId]
        observedCenter = (np.asarray(observation['innerCorner'])+np.asarray(observation['outerCorner']))/2
        observedWidth = np.linalg.norm(np.asarray(observation['outerCorner'])-np.asarray(observation['innerCorner']))
        if np.linalg.norm(center-observedCenter) > maxCenterDistanceFraction*observedWidth:
            raise ValueError('Reference template and eye observation are too far apart for reliable label mapping')
        points = projected[loop]
        if name in cornerOverrides:
            innerVertex, outerVertex = (cornerOverrides[name][key] for key in ('innerVertex', 'outerVertex'))
            if innerVertex not in loop or outerVertex not in loop or innerVertex == outerVertex:
                raise ValueError('Reviewed eye corners must be distinct vertices on the corresponding boundary')
            innerIndex, outerIndex = int(np.flatnonzero(loop == innerVertex)[0]), int(np.flatnonzero(loop == outerVertex)[0])
            source = 'explicitReviewedTemplateCornerOverride'
        else:
            _, _, axes = np.linalg.svd(points-points.mean(0), full_matrices=False)
            along = (points-points.mean(0))@axes[0]
            candidates = [int(np.argmin(along)), int(np.argmax(along))]
            innerIndex = min(candidates, key=lambda index: np.linalg.norm(points[index]-observation['innerCorner']))
            outerIndex = candidates[1] if innerIndex == candidates[0] else candidates[0]
            source = 'referenceProjectedPcaExtremesProxy'
        upperPath = _walkLoop(loop, innerIndex, outerIndex, 1)
        lowerPath = _walkLoop(loop, innerIndex, outerIndex, -1)
        chord = points[outerIndex]-points[innerIndex]
        downward = np.array([-chord[1], chord[0]])
        downward *= 1 if downward[1] >= 0 else -1
        if np.mean((projected[upperPath]-points[innerIndex])@downward) > np.mean((projected[lowerPath]-points[innerIndex])@downward):
            upperPath, lowerPath = lowerPath, upperPath
        results[name] = {'labelId': int(labelId), 'cornerSource': source,
            'innerVertex': int(loop[innerIndex]), 'outerVertex': int(loop[outerIndex]),
            'innerCorner': {'vertexIndices': np.array([[loop[innerIndex]]]), 'barycentric': np.ones((1, 1))},
            'outerCorner': {'vertexIndices': np.array([[loop[outerIndex]]]), 'barycentric': np.ones((1, 1))},
            'upper': _bindMeshPath(upperPath, projected, depth, sampleCount),
            'lower': _bindMeshPath(lowerPath, projected, depth, sampleCount)}
    return {'schemaVersion': 1, 'topologySha256': meshTopologyHash(len(vertices), faces),
            'sampleCount': sampleCount, 'eyes': results,
            'diagnostics': {'cornerBindingFrozen': True, 'labelMappingFrozen': True,
                            'bothCurveDirections': 'innerToOuter', 'anatomicalGroundTruth': False}}


def validateEyeContourObservation(observation, *, sampleCount=24, labelId=None):
    """Validate canonical sampled targets without moving or resampling any point.

    Detector adapters normalize once with makeEyeContourObservation. Fitting,
    metrics and overlays then share exactly those arrays, including their sample
    spacing. This function also accepts their portable JSON list representation.
    Noncanonical sample counts or mismatched corner endpoints fail explicitly.
    """
    if not isinstance(sampleCount, int) or sampleCount < 6 or not observation.get('valid'):
        raise ValueError('A valid canonical eye observation and at least six samples are required')
    if labelId is not None and observation.get('labelId') != labelId:
        raise ValueError('Eye observation identity does not match its fixed template binding')
    shapes = {'innerCorner': (2,), 'outerCorner': (2,),
              'upperCurve': (sampleCount, 2), 'lowerCurve': (sampleCount, 2),
              'upperConfidence': (sampleCount,), 'lowerConfidence': (sampleCount,), 'cornerConfidence': (2,)}
    result = dict(observation)
    for key, shape in shapes.items():
        value = np.asarray(observation.get(key), dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'Canonical eye observation has invalid {key} shape or values')
        if 'Confidence' in key and np.any((value < 0) | (value > 1)):
            raise ValueError('Canonical eye confidence must be in [0,1]')
        result[key] = value.copy()
    endpoints = np.stack([result['innerCorner'], result['outerCorner']])
    for part in ('upperCurve', 'lowerCurve'):
        if not np.allclose(result[part][[0, -1]], endpoints, rtol=0, atol=1e-6):
            raise ValueError('Canonical eye curve endpoints must equal its declared corners')
    if np.linalg.norm(result['outerCorner']-result['innerCorner']) <= 1e-8:
        raise ValueError('Canonical eye corners must be distinct')
    return result


def buildEyeContourConstraints(vertices, faces, cameras, templateBinding, photoContours, vertexVisibility, *,
                               cornerWeight=4., curveWeight=1., minimumConfidence=.4,
                               maxTargetDistance=28., minimumCurveSamples=4):
    """Create explicit point/ordered-segment constraints on frozen eye topology.

    Curves and corners are separate groups. Interior curve samples exclude the
    duplicate endpoints. ``segmentPairs`` indexes only surviving originally
    adjacent samples; visibility gaps are never joined. ``targetTangents`` and
    ``sampleIndices`` are available to a direction-aware geometry solver.
    """
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    visibility = np.asarray(vertexVisibility)
    if (templateBinding.get('schemaVersion') != 1
            or templateBinding.get('topologySha256') != meshTopologyHash(len(vertices), faces)
            or len(cameras) != len(photoContours) or visibility.shape != (len(cameras), len(vertices))
            or visibility.dtype != np.bool_ or not np.isfinite([cornerWeight, curveWeight, minimumConfidence, maxTargetDistance]).all()
            or min(cornerWeight, curveWeight, maxTargetDistance) <= 0 or not 0 <= minimumConfidence <= 1
            or not isinstance(minimumCurveSamples, int) or minimumCurveSamples < 2):
        raise ValueError('Invalid frozen eye binding, observations, visibility or fit configuration')
    constraints, reports = [], []
    sampleCount = int(templateBinding['sampleCount'])
    for eyeName, eye in templateBinding['eyes'].items():
        for viewIndex, (camera, photo) in enumerate(zip(cameras, photoContours)):
            observed = photo['eyes'].get(eye['labelId'], photo['eyes'].get(str(eye['labelId'])))
            if observed is None or not observed.get('valid'):
                reports.append({'eye': eyeName, 'viewIndex': viewIndex, 'status': 'rejectedPhotoEye',
                                'reason': 'missing' if observed is None else observed.get('reason', 'invalid')})
                continue
            # Preserve exactly the arrays displayed and measured by the caller.
            # Repeating arc-length sampling here would shift the fit target.
            normalized = validateEyeContourObservation(observed, sampleCount=sampleCount, labelId=eye['labelId'])
            for part in ('innerCorner', 'outerCorner', 'upper', 'lower'):
                binding = eye[part]
                indices, barycentric = np.asarray(binding['vertexIndices']), np.asarray(binding['barycentric'], float)
                if (indices.ndim != 2 or indices.shape[1] not in (1, 2) or indices.dtype.kind not in 'iu'
                        or not len(indices) or indices.min() < 0 or indices.max() >= len(vertices)
                        or barycentric.shape != indices.shape or not np.isfinite(barycentric).all()
                        or np.any(barycentric < 0) or not np.allclose(barycentric.sum(1), 1)):
                    raise ValueError('Invalid fixed eye surface binding')
                isCurve = part in ('upper', 'lower')
                if isCurve:
                    targets = normalized[part+'Curve']
                    confidence = normalized[part+'Confidence']
                    selected = np.arange(1, sampleCount-1)
                    targetTangents = _curveTangents(targets)
                    weight = curveWeight
                else:
                    targets = np.asarray(normalized[part])[None]
                    confidence = normalized['cornerConfidence'][0 if part == 'innerCorner' else 1:1 if part == 'innerCorner' else 2]
                    selected = np.arange(1)
                    targetTangents = None
                    weight = cornerWeight
                if len(indices) != len(targets):
                    raise ValueError('Template and photo curve sample counts differ')
                points = (vertices[indices]*barycentric[..., None]).sum(1)
                pixels, depth = _project(points, camera)
                distance = np.linalg.norm(pixels-targets, axis=1)
                valid = (visibility[viewIndex, indices].all(1) & (depth > 0)
                         & (confidence >= minimumConfidence) & (distance <= maxTargetDistance)
                         & np.isfinite(pixels).all(1))
                if 'imageShape' in photo:
                    height, width = photo['imageShape']
                    valid &= ((targets[:, 0] >= 0) & (targets[:, 0] <= width-1)
                              & (targets[:, 1] >= 0) & (targets[:, 1] <= height-1))
                retained = selected[valid[selected]]
                report = {'eye': eyeName, 'labelId': eye['labelId'], 'viewIndex': viewIndex, 'part': part,
                          'candidateCount': len(selected), 'sampleCount': len(retained),
                          'invisibleCount': int((~visibility[viewIndex, indices].all(1)[selected]).sum()),
                          'lowConfidenceCount': int((confidence[selected] < minimumConfidence).sum()),
                          'excessiveDistanceCount': int((distance[selected] > maxTargetDistance).sum())}
                reports.append(report)
                if len(retained) < (minimumCurveSamples if isCurve else 1):
                    report['status'] = 'insufficientReliableCorrespondences'
                    continue
                report.update({'status': 'accepted', 'initialMeanErrorPixels': float(distance[retained].mean())})
                constraint = {'name': f'{eyeName}.{part}', 'viewIndex': viewIndex,
                    'vertexIndices': indices[retained].copy(), 'barycentric': barycentric[retained].copy(),
                    'weights': confidence[retained]*weight, 'targetPixels': targets[retained].copy(),
                    # Explicit segmentPairs, not a shortcut joining every kept
                    # point, preserve gaps caused by occlusion/confidence gates.
                    'orderedCurve': False, 'curveOrder': 'innerToOuter' if isCurve else 'singleCorner',
                    'sampleIndices': retained.copy()}
                if isCurve:
                    continuous = np.flatnonzero(np.diff(retained) == 1)
                    constraint['segmentPairs'] = np.stack([continuous, continuous+1], 1)
                    constraint['targetTangents'] = targetTangents[retained].copy()
                constraints.append(constraint)
    return {'constraints': constraints, 'diagnostics': {'groups': reports, 'constraintCount': len(constraints),
        'explicitCornerCorrespondence': True, 'upperLowerCorrespondenceSeparate': True,
        'bindingFrozen': True, 'visibilityFrozen': True, 'cornerWeight': cornerWeight,
        'limits': 'Automatic photo/template corner proxies remain uncertain; no anatomical ground truth asserted'}}
