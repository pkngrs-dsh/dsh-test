import copy
import json

import numpy as np
import pytest

from faceAppearance.eyeContours import (bindTemplateEyeContours, buildEyeContourConstraints,
    convertCachedEyeLandmarks, extractEyeContours, makeEyeContourObservation, validateEyeContourObservation)


def photoEye(labelId, center, width=12., opening=5., samples=13):
    theta = np.linspace(0, np.pi, samples)
    direction = 1 if labelId == 4 else -1
    x = center[0]+direction*width*np.cos(theta)
    upper = np.stack([x, center[1]-opening*np.sin(theta)], 1)
    lower = np.stack([x, center[1]+opening*np.sin(theta)], 1)
    return makeEyeContourObservation(labelId, upper[0], upper[-1], upper, lower, sampleCount=24)


def semanticMaps():
    labels = np.zeros((100, 120), np.uint8)
    yy, xx = np.mgrid[:100, :120]
    labels[((xx-35)/12)**2+((yy-45)/5)**2 <= 1] = 4
    labels[((xx-85)/12)**2+((yy-45)/5)**2 <= 1] = 5
    labels[57:67, 58:63] = 10
    return labels, np.full(labels.shape, .95)


def templateEyes():
    vertices, faces, loops = [], [], {}
    count = 16
    angle = np.linspace(0, 2*np.pi, count, endpoint=False)
    for name, center in [('imageLeftTemplate', (35, 50)), ('imageRightTemplate', (85, 50))]:
        base = len(vertices)
        for horizontal, vertical in ((18, 9), (12, 4)):
            vertices.extend(np.stack([center[0]+horizontal*np.cos(angle),
                                      center[1]+vertical*np.sin(angle), np.ones(count)], 1))
        for index in range(count):
            nxt = (index+1) % count
            faces.extend([[base+index, base+nxt, base+count+index],
                          [base+nxt, base+count+nxt, base+count+index]])
        loops[name] = np.arange(base+count, base+count*2)
    camera = {'intrinsics': np.eye(3), 'rotation': np.eye(3), 'translation': np.zeros(3)}
    photo = {'eyes': {4: photoEye(4, (35, 50)), 5: photoEye(5, (85, 50))}, 'imageShape': [100, 120]}
    return np.array(vertices), np.array(faces), loops, camera, photo


def testSemanticEyesRemainSeparateAndCurvesRunInnerToOuter():
    labels, confidence = semanticMaps()
    result = extractEyeContours(labels, confidence)
    assert all(eye['valid'] for eye in result['eyes'].values())
    assert result['eyes'][4]['innerCorner'][0] > result['eyes'][4]['outerCorner'][0]
    assert result['eyes'][5]['innerCorner'][0] < result['eyes'][5]['outerCorner'][0]
    for eye in result['eyes'].values():
        for curve in ('upperCurve', 'lowerCurve'):
            np.testing.assert_array_equal(eye[curve][0], eye['innerCorner'])
            np.testing.assert_array_equal(eye[curve][-1], eye['outerCorner'])
        assert eye['upperCurve'][12, 1] < eye['lowerCurve'][12, 1]
        assert eye['source'] == 'semanticLargestComponentPcaEndpointProxy'
        assert not 'groundTruth' in eye


def testSemanticSmoothingKeepsEndpointsFixed():
    labels, confidence = semanticMaps()
    rough = extractEyeContours(labels, confidence, smoothingSigma=0)
    smooth = extractEyeContours(labels, confidence, smoothingSigma=1.5)
    for labelId in (4, 5):
        np.testing.assert_array_equal(rough['eyes'][labelId]['innerCorner'], smooth['eyes'][labelId]['innerCorner'])
        np.testing.assert_array_equal(rough['eyes'][labelId]['outerCorner'], smooth['eyes'][labelId]['outerCorner'])


@pytest.mark.parametrize('failure', ['lowConfidence', 'fragmented', 'border', 'closed', 'yaw'])
def testUnreliableSemanticEyesAreRejected(failure):
    labels, confidence = semanticMaps()
    kwargs = {}
    if failure == 'lowConfidence':
        confidence[labels == 4] = .3
    elif failure == 'fragmented':
        labels[10:20, 10:20] = 4
    elif failure == 'border':
        labels[labels == 4] = 0
        labels[40:50, :20] = 4
    elif failure == 'closed':
        labels[labels == 4] = 0
        labels[45, 23:48] = 4
        kwargs['minArea'] = 10
    else:
        kwargs['viewYawDegrees'] = 75
    result = extractEyeContours(labels, confidence, **kwargs)
    assert not result['eyes'][4]['valid']


def testExplicitCurvesRejectWrongOrderAndSwappedUpperLower():
    eye = photoEye(4, (35, 50))
    with pytest.raises(ValueError, match='inner corner'):
        makeEyeContourObservation(4, eye['innerCorner'], eye['outerCorner'], eye['upperCurve'][::-1], eye['lowerCurve'])
    reversedLids = makeEyeContourObservation(4, eye['innerCorner'], eye['outerCorner'], eye['lowerCurve'], eye['upperCurve'])
    assert not reversedLids['valid']


def testTemplateCornersAndUpperLowerPathsAreFrozenAndBijective():
    vertices, faces, loops, camera, photo = templateEyes()
    binding = bindTemplateEyeContours(vertices, faces, loops, camera, photo)
    assert binding['eyes']['imageLeftTemplate']['labelId'] == 4
    assert binding['eyes']['imageRightTemplate']['labelId'] == 5
    for eye in binding['eyes'].values():
        for part in ('upper', 'lower'):
            path = eye[part]['pathVertexIndices']
            assert path[0] == eye['innerVertex'] and path[-1] == eye['outerVertex']
        upper = vertices[eye['upper']['pathVertexIndices'], 1].mean()
        lower = vertices[eye['lower']['pathVertexIndices'], 1].mean()
        assert upper < lower
    # Same physical loops in opposite traversal order yield the same endpoint IDs.
    opposite = bindTemplateEyeContours(vertices, faces, {k: v[::-1] for k, v in loops.items()}, camera, photo)
    for name in loops:
        assert binding['eyes'][name]['innerVertex'] == opposite['eyes'][name]['innerVertex']
        assert binding['eyes'][name]['outerVertex'] == opposite['eyes'][name]['outerVertex']


def testTemplateExplicitCornerOverridesAndShuffledPathsValidation():
    vertices, faces, loops, camera, photo = templateEyes()
    base = bindTemplateEyeContours(vertices, faces, loops, camera, photo)
    eye = base['eyes']['imageLeftTemplate']
    overrides = {'imageLeftTemplate': {'innerVertex': eye['innerVertex'], 'outerVertex': eye['outerVertex']}}
    explicit = bindTemplateEyeContours(vertices, faces, loops, camera, photo, cornerOverrides=overrides)
    assert explicit['eyes']['imageLeftTemplate']['cornerSource'] == 'explicitReviewedTemplateCornerOverride'
    shuffled = copy.deepcopy(loops)
    shuffled['imageLeftTemplate'][[1, 5]] = shuffled['imageLeftTemplate'][[5, 1]]
    with pytest.raises(ValueError, match='connected'):
        bindTemplateEyeContours(vertices, faces, shuffled, camera, photo)


def testBindingsAndPhotoObservationsCanBeReloadedFromPortableJson():
    vertices, faces, loops, camera, photo = templateEyes()
    savedPhoto = json.loads(json.dumps(photo, default=lambda value: value.tolist()))
    binding = bindTemplateEyeContours(vertices, faces, loops, camera, savedPhoto)
    savedBinding = json.loads(json.dumps(binding, default=lambda value: value.tolist()))
    result = buildEyeContourConstraints(vertices, faces, [camera], savedBinding, [savedPhoto], np.ones((1, len(vertices)), bool))
    assert len(result['constraints']) == 8
    for constraint in result['constraints']:
        name, part = constraint['name'].split('.')
        labelId = savedBinding['eyes'][name]['labelId']
        observation = savedPhoto['eyes'][str(labelId)]
        expected = (np.asarray(observation[part+'Curve'])[constraint['sampleIndices']]
                    if part in ('upper', 'lower') else np.asarray(observation[part])[None])
        np.testing.assert_array_equal(constraint['targetPixels'], expected)


def testBuildUsesCanonicalTargetsExactlyWithoutSecondArcLengthSampling():
    vertices, faces, loops, camera, photo = templateEyes()
    binding = bindTemplateEyeContours(vertices, faces, loops, camera, photo)
    # Deliberately nonuniform but valid sample spacing makes repeat resampling
    # observable; the normalized target is an immutable part of the experiment.
    photo['eyes'][4]['upperCurve'][5:10, 0] += .12
    initial = copy.deepcopy(photo)
    result = buildEyeContourConstraints(vertices, faces, [camera], binding, [photo], np.ones((1, len(vertices)), bool))
    for constraint in result['constraints']:
        name, part = constraint['name'].split('.')
        eye = initial['eyes'][binding['eyes'][name]['labelId']]
        expected = (eye[part+'Curve'][constraint['sampleIndices']]
                    if part in ('upper', 'lower') else eye[part][None])
        np.testing.assert_array_equal(constraint['targetPixels'], expected)
    np.testing.assert_array_equal(photo['eyes'][4]['upperCurve'], initial['eyes'][4]['upperCurve'])


@pytest.mark.parametrize('corruption', ['sampleCount', 'cornerMismatch', 'identity', 'confidence'])
def testCanonicalValidationRejectsMalformedTargetsInsteadOfRepairingThem(corruption):
    eye = photoEye(4, (35, 50))
    if corruption == 'sampleCount':
        eye['upperCurve'] = eye['upperCurve'][:-1]
    elif corruption == 'cornerMismatch':
        eye['upperCurve'][0] += [0, .02]
    elif corruption == 'identity':
        eye['labelId'] = 5
    else:
        eye['upperConfidence'][5] = -1
    with pytest.raises(ValueError):
        validateEyeContourObservation(eye, sampleCount=24, labelId=4)


def testConstraintGroupsUseExplicitCornersAndIndependentOrderedLids():
    vertices, faces, loops, camera, photo = templateEyes()
    binding = bindTemplateEyeContours(vertices, faces, loops, camera, photo)
    result = buildEyeContourConstraints(vertices, faces, [camera], binding, [photo], np.ones((1, len(vertices)), bool))
    assert len(result['constraints']) == 8
    for constraint in result['constraints']:
        if constraint['name'].endswith('Corner'):
            assert len(constraint['targetPixels']) == 1
            assert 'segmentPairs' not in constraint
            assert constraint['weights'][0] == pytest.approx(4.)
        else:
            assert len(constraint['targetPixels']) == 22
            assert constraint['segmentPairs'].shape == (21, 2)
            assert np.all(np.diff(constraint['sampleIndices'][constraint['segmentPairs']], axis=1) == 1)
            np.testing.assert_allclose(np.linalg.norm(constraint['targetTangents'], axis=1), 1.)


def testVisibilityGapsNeverCreateFalseAdjacentSegments():
    vertices, faces, loops, camera, photo = templateEyes()
    binding = bindTemplateEyeContours(vertices, faces, loops, camera, photo)
    visibility = np.ones((1, len(vertices)), bool)
    hiddenVertex = binding['eyes']['imageLeftTemplate']['upper']['pathVertexIndices'][4]
    visibility[0, hiddenVertex] = False
    result = buildEyeContourConstraints(vertices, faces, [camera], binding, [photo], visibility)
    upper = next(value for value in result['constraints'] if value['name'] == 'imageLeftTemplate.upper')
    assert len(upper['sampleIndices']) < 22
    assert (np.diff(upper['sampleIndices']) > 1).any()
    assert np.all(np.diff(upper['sampleIndices'][upper['segmentPairs']], axis=1) == 1)


def testRejectedPhotoEyeAndLowConfidenceCornerDoNotBecomeConstraints():
    vertices, faces, loops, camera, photo = templateEyes()
    binding = bindTemplateEyeContours(vertices, faces, loops, camera, photo)
    modified = copy.deepcopy(photo)
    modified['eyes'][5] = {'valid': False, 'reason': 'occluded'}
    modified['eyes'][4]['cornerConfidence'][0] = .1
    result = buildEyeContourConstraints(vertices, faces, [camera], binding, [modified], np.ones((1, len(vertices)), bool))
    assert len(result['constraints']) == 3
    assert all(value['name'].startswith('imageLeftTemplate') for value in result['constraints'])
    assert not any(value['name'].endswith('innerCorner') for value in result['constraints'])


def testTopologyChangeCannotReuseBindings():
    vertices, faces, loops, camera, photo = templateEyes()
    binding = bindTemplateEyeContours(vertices, faces, loops, camera, photo)
    with pytest.raises(ValueError, match='Invalid frozen eye binding'):
        buildEyeContourConstraints(vertices, faces[::-1], [camera], binding, [photo], np.ones((1, len(vertices)), bool))


def cachedLandmarks():
    curves, corners = [], []
    for labelId, center in ((4, (35, 50)), (5, (85, 50))):
        eye = photoEye(labelId, center, samples=12)
        # Cache contract always image-left-to-right, independent of eye side.
        order = np.arange(23, -1, -1) if labelId == 4 else np.arange(24)
        curves.append(np.stack([eye['lowerCurve'][order], eye['upperCurve'][order]]))
        corners.append(np.stack([curves[-1][0, 0], curves[-1][0, -1]]))
    arrays = {'eyelidCurves': np.array(curves), 'corners': np.array(corners),
              'curveConfidence': np.full((2, 2, 24), .9), 'rawPeakScores': np.full((2, 2, 24), .9),
              'pointValidMask': np.ones((2, 2, 24), bool)}
    report = {'metadata': {'eyeOrder': ['imageLeft', 'imageRight'], 'curveOrder': ['lower', 'upper'],
              'cornerOrder': ['imageLeftEndpoint', 'imageRightEndpoint'], 'originalSize': [120, 100]},
              'imageSideToSemanticLabel': {'imageLeft': 4, 'imageRight': 5},
              'quality': [{'valid': True}, {'valid': True}]}
    return arrays, report


def testLandmarkCacheConvertsExplicitEyeAndCurveOrder():
    arrays, report = cachedLandmarks()
    result = convertCachedEyeLandmarks(arrays, report)
    assert result['imageShape'] == [100, 120]
    assert result['eyes'][4]['innerCorner'][0] > result['eyes'][4]['outerCorner'][0]
    assert result['eyes'][5]['innerCorner'][0] < result['eyes'][5]['outerCorner'][0]
    for eye in result['eyes'].values():
        assert eye['valid'] and eye['upperCurve'][12, 1] < eye['lowerCurve'][12, 1]
        assert eye['upperCurve'].shape == (24, 2)


def testLandmarkCacheWholeEyeGateRejectsUncertainFarEye():
    arrays, report = cachedLandmarks()
    arrays['rawPeakScores'][0, 1, 3] = .495
    result = convertCachedEyeLandmarks(arrays, report, minimumRawPeak=.6)
    assert not result['eyes'][4]['valid']
    assert result['eyes'][5]['valid']
    assert result['eyes'][4]['diagnostics']['minimumRawPeak'] == pytest.approx(.495)


def testLandmarkCacheCannotGuessMissingIdentityMapping():
    arrays, report = cachedLandmarks()
    report.pop('imageSideToSemanticLabel')
    with pytest.raises(ValueError, match='contract'):
        convertCachedEyeLandmarks(arrays, report)
