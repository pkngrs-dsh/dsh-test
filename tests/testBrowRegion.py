import copy

import numpy as np
import pytest

from faceAppearance.browRegion import buildBrowRegion


def fixtureData(*, ySpacing=1.):
    yy, xx = np.mgrid[:21, :21]
    vertices = np.column_stack([xx.ravel()+10., yy.ravel()*ySpacing+10., np.ones(xx.size)])
    faces = []
    for y in range(20):
        for x in range(20):
            first = y*21+x
            # Normals point toward the camera at the origin, along negative Z.
            faces.extend([[first, first+22, first+1], [first, first+21, first+22]])
    labels = np.zeros((64, 64), np.uint8)
    labels[15, 20] = 2
    camera = {'intrinsics': np.eye(3), 'rotation': np.eye(3), 'translation': np.zeros(3)}
    return {'vertices': vertices, 'faces': np.asarray(faces), 'camera': camera,
            'labelMap': labels, 'confidenceMap': np.ones(labels.shape),
            'vertexVisibility': np.ones(len(vertices), bool), 'eyeBoundaryIds': np.arange(420, 441),
            'eyeDistancePixels': 10.}


def testPhotoCoreAndFalloffAreSmoothAndBounded():
    data = fixtureData()
    result = buildBrowRegion(**data, coreRadiusEyeFraction=.1, outerRadiusEyeFraction=.5)
    weights = result['regionWeights'].reshape(21, 21)
    assert weights.dtype == np.float32
    assert np.isfinite(weights).all() and np.min(weights) >= 0 and np.max(weights) <= 1
    assert weights[5, 10] == weights[5, 11] == 1
    assert weights[5, 12] == pytest.approx(1-(.25**2*(3-2*.25)))
    assert weights[5, 14] > weights[5, 15] == 0
    assert result['eyebrowSeedMask'].sum() == 1
    assert result['diagnostics']['status'] == 'ready'


@pytest.mark.parametrize('mode', ['missing', 'lowConfidence', 'invisible', 'unprojectedComponent'])
def testNoTrustedVisibleEyebrowProducesEmptyRegion(mode):
    data = fixtureData()
    if mode == 'missing':
        data['labelMap'][:] = 0
    elif mode == 'lowConfidence':
        data['confidenceMap'][:] = .79
    elif mode == 'invisible':
        data['vertexVisibility'][:] = False
    else:
        data['labelMap'][:] = 0
        data['labelMap'][45, 45] = 2
    result = buildBrowRegion(**data)
    assert not result['regionWeights'].any() and not result['eyebrowSeedMask'].any()
    assert result['diagnostics']['status'] == 'noReliableEyebrowSeeds'


def testUnseenSemanticComponentsCannotExpandASeededRegion():
    data = fixtureData()
    baseline = buildBrowRegion(**data)
    # Same label, detached component adjacent to the mesh but not under any vertex.
    data['labelMap'][9, 20] = 2
    result = buildBrowRegion(**data)
    np.testing.assert_array_equal(result['regionWeights'], baseline['regionWeights'])
    assert len(result['diagnostics']['acceptedPhotoComponentIdsByLabel']['2']) == 1


def testEyeProtectionUsesCanonicalGraphDistanceAndExactThresholds():
    data = fixtureData()
    data['labelMap'][:] = 2
    data['eyeBoundaryIds'] = np.arange(21)
    result = buildBrowRegion(**data, eyeProtectedDistanceFraction=.1, eyeFalloffDistanceFraction=.3)
    weights = result['regionWeights'].reshape(21, 21)
    distance = result['eyeGraphDistanceCanonical'].reshape(21, 21)
    assert distance[2, 10] == 2 and distance[6, 10] == 6
    assert weights[2, 10] == 0 and result['protectedMask'].reshape(21, 21)[2, 10]
    assert weights[3, 10] == pytest.approx(.15625)
    assert weights[4, 10] == pytest.approx(.5)
    assert weights[6, 10] == 1
    assert result['diagnostics']['eyeProtectedRadiusCanonical'] == 2
    assert result['diagnostics']['eyeReleaseRadiusCanonical'] == 6
    # Moving the surface farther from the camera compresses projected edges;
    # the protection distances still come from unchanged canonical 3D edges.
    compressed = copy.deepcopy(data)
    compressed['camera']['translation'][2] = 9
    other = buildBrowRegion(**compressed, eyeProtectedDistanceFraction=.1, eyeFalloffDistanceFraction=.3)
    np.testing.assert_array_equal(other['eyeGraphDistanceCanonical'], result['eyeGraphDistanceCanonical'])
    np.testing.assert_array_equal(other['regionWeights'], result['regionWeights'])


def testEveryMeshBoundaryIsStrictlyFixedAndFeatheredInwards():
    data = fixtureData(ySpacing=.1)
    data['labelMap'][:] = 2
    result = buildBrowRegion(**data)
    weights = result['regionWeights'].reshape(21, 21)
    assert not result['regionWeights'][result['meshBoundaryMask']].any()
    assert result['meshBoundaryMask'].sum() == 80
    assert weights[1, 10] == pytest.approx(.5)
    assert weights[2, 10] == pytest.approx(1.)
    assert result['diagnostics']['boundaryAndVisibilityFeatherCanonical'] == pytest.approx(.2)


@pytest.mark.parametrize('mode', ['behind', 'outside', 'backface', 'hidden'])
def testNonvisibleOrOutOfFrustumVerticesNeverReceiveWeight(mode):
    data = fixtureData()
    data['labelMap'][:] = 2
    if mode == 'behind':
        data['camera']['translation'][2] = -2
    elif mode == 'outside':
        data['camera']['intrinsics'][0, 2] = 100
    elif mode == 'backface':
        data['faces'] = data['faces'][:, ::-1]
    else:
        data['vertexVisibility'][:] = False
    result = buildBrowRegion(**data)
    assert not result['regionWeights'].any()
    assert result['protectedMask'].all()
    assert not result['eligibleMask'].any()


def testVisibilityTransitionHasItsOwnSmoothFeather():
    data = fixtureData(ySpacing=.1)
    data['labelMap'][:] = 2
    data['eyeBoundaryIds'] = np.arange(21)
    data['vertexVisibility'].reshape(21, 21)[16] = False
    result = buildBrowRegion(**data)
    weights = result['regionWeights'].reshape(21, 21)
    assert weights[16, 10] == 0
    assert weights[15, 10] == pytest.approx(.5)
    assert weights[14, 10] == pytest.approx(1.)


def testInputsAndTopologyAreNeverMutated():
    data = fixtureData()
    before = copy.deepcopy(data)
    buildBrowRegion(**data)
    for name in data:
        if isinstance(data[name], np.ndarray):
            np.testing.assert_array_equal(data[name], before[name])
    for name in data['camera']:
        np.testing.assert_array_equal(data['camera'][name], before['camera'][name])


@pytest.mark.parametrize('failure', [
    'negativeScale', 'nanScale', 'boolScale', 'reversedRadii', 'negativeProtection',
    'invalidConfidence', 'wrongLabel', 'floatLabels', 'floatFaces', 'badFaceIndex',
    'repeatedTriangleIndex', 'degenerateFace', 'duplicateFace', 'wrongVisibility',
    'interiorEyeIds', 'emptyEyeIds', 'floatEyeIds', 'improperCamera', 'badIntrinsics'])
def testInvalidContractsFailClosed(failure):
    data = fixtureData()
    if failure == 'negativeScale': data['eyeDistancePixels'] = -1
    elif failure == 'nanScale': data['eyeDistancePixels'] = np.nan
    elif failure == 'boolScale': data['eyeDistancePixels'] = True
    elif failure == 'reversedRadii': data.update(coreRadiusEyeFraction=.4, outerRadiusEyeFraction=.3)
    elif failure == 'negativeProtection': data['eyeProtectedDistanceFraction'] = -.1
    elif failure == 'invalidConfidence': data['confidenceMap'][0, 0] = np.nan
    elif failure == 'wrongLabel': data['labelMap'][0, 0] = 19
    elif failure == 'floatLabels': data['labelMap'] = data['labelMap'].astype(float)
    elif failure == 'floatFaces': data['faces'] = data['faces'].astype(float)
    elif failure == 'badFaceIndex': data['faces'][0, 0] = len(data['vertices'])
    elif failure == 'repeatedTriangleIndex': data['faces'][0, 0] = data['faces'][0, 1]
    elif failure == 'degenerateFace': data['vertices'][22] = data['vertices'][1]
    elif failure == 'duplicateFace': data['faces'] = np.vstack([data['faces'], data['faces'][0]])
    elif failure == 'wrongVisibility': data['vertexVisibility'] = data['vertexVisibility'].astype(int)
    elif failure == 'interiorEyeIds': data['eyeBoundaryIds'] = np.array([100])
    elif failure == 'emptyEyeIds': data['eyeBoundaryIds'] = np.empty(0, int)
    elif failure == 'floatEyeIds': data['eyeBoundaryIds'] = data['eyeBoundaryIds'].astype(float)
    elif failure == 'improperCamera': data['camera']['rotation'][0, 0] = -1
    else: data['camera']['intrinsics'][0, 0] = 0
    with pytest.raises(ValueError):
        buildBrowRegion(**data)
