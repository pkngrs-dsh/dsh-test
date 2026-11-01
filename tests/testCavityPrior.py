import numpy as np
import pytest

from faceAppearance.cavityPrior import buildCavityDepthPrior

FACES = np.array([[0, 1, 4], [1, 2, 4], [2, 5, 4], [5, 0, 4], [0, 5, 3], [5, 2, 3]])


def meshes():
    rest = np.array([[-1., -1., 0.], [1., -1., 0.], [1., 1., 0.], [-1., 1., 0.],
                     [0., 0., 3.], [.4, 0., 1.]])
    current = rest.copy()
    current[4:, 2] = [.5, 2.]
    return current, rest


def prior(current, rest, **kwargs):
    return buildCavityDepthPrior(current, rest, np.arange(4), np.array([4, 5]),
        patchFaces=FACES, inwardDirection=[0., 0., 1.], maxAddedDepth=1., **kwargs)


def testRestoresOnlyMissingDepthWithoutFlatteningDeeperVerticesOrEditingInputs():
    current, rest = meshes()
    first, second = current.copy(), rest.copy()
    result = prior(current, rest, restorationFraction=.5)
    np.testing.assert_allclose(result['addedDepth'], [1., 0.])
    np.testing.assert_allclose(result['targetPoints'], [[0., 0., 1.5], [.4, 0., 2.]])
    np.testing.assert_array_equal(current, first)
    np.testing.assert_array_equal(rest, second)
    assert result['diagnostics']['qualityAccepted'] is False


def testRigidCoordinateChangePreservesDepthAndTransformsTargets():
    current, rest = meshes()
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    translation = np.array([3., 7., -4.])
    baseline = prior(current, rest)
    transformed = buildCavityDepthPrior(current@rotation.T+translation, rest@rotation.T+translation,
        np.arange(4), np.array([4, 5]), inwardDirection=np.array([0., 0., 1.])@rotation.T,
        patchFaces=FACES, maxAddedDepth=1.)
    np.testing.assert_allclose(transformed['addedDepth'], baseline['addedDepth'])
    np.testing.assert_allclose(transformed['targetPoints'], baseline['targetPoints']@rotation.T+translation)


def testUnitsScaleTogether():
    current, rest = meshes()
    original = prior(current, rest)
    scaled = buildCavityDepthPrior(1000*current, 1000*rest, np.arange(4), np.array([4, 5]),
        patchFaces=FACES, inwardDirection=[0., 0., 1.], maxAddedDepth=1000.)
    np.testing.assert_allclose(scaled['targetPoints']/1000, original['targetPoints'])


@pytest.mark.parametrize('direction', [[1., 0., 0.], [0., 0., 0.], [0., np.nan, 1.]])
def testRejectsAmbiguousOrInvalidInwardDirection(direction):
    current, rest = meshes()
    with pytest.raises(ValueError):
        buildCavityDepthPrior(current, rest, np.arange(4), np.array([4, 5]),
            patchFaces=FACES, inwardDirection=direction, maxAddedDepth=1.)


def testRejectsOverlappingRegionsAndDegenerateRim():
    current, rest = meshes()
    with pytest.raises(ValueError):
        buildCavityDepthPrior(current, rest, np.arange(4), np.array([3, 4]),
            patchFaces=FACES, inwardDirection=[0., 0., 1.], maxAddedDepth=1.)
    current[:4, 1] = 0.
    with pytest.raises(ValueError):
        prior(current, rest)


def testNonplanarBoundaryIsSubtractedWithoutInventingCavityDepth():
    current, rest = meshes()
    # Saddle rim has a zero-tilt best plane. The interior harmonic extension
    # is 0.4/3 at both interior nodes; apply it to both depth fields.
    baseline = prior(current, rest)
    for vertices in (current, rest):
        vertices[:4, 2] += [.4, -.4, .4, -.4]
        vertices[4:, 2] += .4/3
    result = prior(current, rest)
    np.testing.assert_allclose(result['beforeDepth'], baseline['beforeDepth'], atol=1e-12)
    np.testing.assert_allclose(result['scaledRestDepth'], baseline['scaledRestDepth'], atol=1e-12)


def testWrongPatchBoundaryIsRejected():
    current, rest = meshes()
    with pytest.raises(ValueError, match='boundary matching'):
        buildCavityDepthPrior(current, rest, np.arange(4), np.array([4, 5]),
            patchFaces=FACES[:-1], inwardDirection=[0., 0., 1.], maxAddedDepth=1.)
