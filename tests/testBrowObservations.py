import numpy as np
import pytest

from faceAppearance.browObservations import fuseNormalObservations, fuseDepthObservations, sampleNormalObservations


def testHeldOutCannotAffectNormalTarget():
    normals = np.array([[[0., 0., 1.]], [[1., 0., 0.]], [[0., 0., 1.]], [[0., 0., -1.]]])
    result = fuseNormalObservations(normals, np.ones((4, 1)), [0, 2, 3])
    np.testing.assert_allclose(result['normalTargets'], [[0, 0, 1]])
    assert result['normalSupportCount'][0] == 2
    assert not result['acceptedNormalViews'][1].any()
    normals[1] = [0, 1, 0]
    other = fuseNormalObservations(normals, np.ones((4, 1)), [0, 2, 3])
    np.testing.assert_array_equal(result['normalTargets'], other['normalTargets'])


def testNoAgreementNoFabricatedObservation():
    normals = np.eye(3)[:, None]
    result = fuseNormalObservations(normals, np.ones((3, 1)), [0, 1, 2])
    assert result['normalWeights'][0] == 0
    assert not result['normalTargets'].any()


def testConsensusThresholdIsRadiusRatherThanPairwiseBound():
    angles = np.radians([-20., 0., 20.])
    normals = np.stack([np.sin(angles), np.zeros(3), np.cos(angles)], axis=1)[:, None]
    result = fuseNormalObservations(normals, np.ones((3, 1)), [0, 1, 2], referenceWeight=1.)
    assert result['normalSupportCount'][0] == 3
    np.testing.assert_allclose(result['normalTargets'], [[0, 0, 1]], atol=1e-12)
    assert 'not a pairwise bound' in result['diagnostics']['agreementMeaning']


def testDepthLateralAndScatterRejectionAndUnits():
    vertices = np.array([[0., 0, 1], [1, 0, 1], [0, 1, 1]])
    directions = np.tile([0., 0, 1], (3, 1))
    partials = np.tile(vertices, (4, 1, 1))
    partials[0, :, 2] += .010
    partials[2, :, 2] += .012
    partials[3, :, 2] += .011
    partials[1] += 100  # held-out values cannot change the targets.
    partials[[0, 2, 3], 1, 0] += .1  # off-ray correspondence failure.
    partials[3, 2, 2] -= .04  # a disagreeing depth fails the scatter gate.
    result = fuseDepthObservations(vertices, directions, partials, np.ones((4, 3), bool), [0, 2, 3])
    assert result['depthTargets'][0] == pytest.approx(.011)
    assert result['depthWeights'][0] > 0
    assert not result['depthWeights'][1:].any()


def testOriginalToWorkingAndSkinGate():
    vertices = np.array([[1., 1, 1], [1, 3, 1], [3, 1, 1]])
    faces = np.array([[0, 1, 2]])  # toward camera (negative z).
    camera = {'intrinsics': np.eye(3), 'rotation': np.eye(3), 'translation': np.zeros(3)}
    normalMap = np.tile([0., 0, -1], (3, 3, 1))
    labels = np.ones((5, 5), np.uint8)
    labels[3, 1] = 2  # eyebrow hair is not trusted skin surface.
    result = sampleNormalObservations(vertices, faces, [camera], [np.diag([.5, .5, 1])],
        [normalMap], [np.ones((3, 3), bool)], [labels], [np.ones((5, 5))], [np.ones(3, bool)])
    np.testing.assert_allclose(result['samplingPixels'][0], [[.5, .5], [.5, 1.5], [1.5, .5]])
    assert result['weightsByView'][0, 0] > 0
    assert result['weightsByView'][0, 1] == 0
    assert result['weightsByView'][0, 2] > 0
    np.testing.assert_allclose(result['normalsByView'][0, 0], [0, 0, -1])


@pytest.mark.parametrize('indices', [[0, 0], [0, 3], [-1, 0]])
def testInvalidViewSelection(indices):
    with pytest.raises(ValueError):
        fuseNormalObservations(np.ones((3, 2, 3)), np.ones((3, 2)), indices)


@pytest.mark.parametrize('weight', [np.nan, np.inf, -1.])
def testInvalidReferenceConfidenceRejected(weight):
    with pytest.raises(ValueError):
        fuseNormalObservations(np.ones((3, 2, 3)), np.ones((3, 2)), [0, 1, 2], referenceWeight=weight)
