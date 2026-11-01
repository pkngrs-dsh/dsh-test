"""Independent semantic/order and exact legacy-surface regression checks."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from faceAppearance.clientLandmarks import loadClientLandmarks, sampleClientLandmarks
from faceAppearance.semanticEyelidBinding import (
    buildSemanticEyelidBinding, sampleSemanticEyelids, projectSemanticEyelids,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def original():
    with np.load(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1/template.npz') as data:
        vertices = data['vertices'][data['nicpToFullIds']]
        faces = data['nicpFaces'].copy()
        fullVertices = data['vertices'].copy()
        skinIds = data['nicpToFullIds'].copy()
    landmarks = loadClientLandmarks(faces)
    return vertices, faces, landmarks, fullVertices, skinIds


@pytest.fixture(scope='module')
def binding(original):
    return buildSemanticEyelidBinding(original[1], original[2])


def testExactOriginalRowsAndUniqueSampleCount(original, binding):
    landmarks = original[2]
    ids = []
    for eye in binding['eyes'].values():
        for lid in ('upper', 'lower'):
            row = eye[lid]
            ids.extend(row['mpIds'].tolist())
            assert np.array_equal(row['faceIds'], landmarks['faceIds'][row['mpIds']])
            assert np.array_equal(row['vertexIndices'], landmarks['vertexIndices'][row['mpIds']])
            assert np.array_equal(row['barycentricWeights'], landmarks['barycentricWeights'][row['mpIds']])
            assert not row['vertexIndices'].flags.writeable
    assert len(ids) == 36
    assert len(set(ids)) == binding['uniqueSampleCount'] == 32


def testIndependentSemanticOrderAndSharedCorners(original, binding):
    expected = {
        'subjectRight': {
            'upper': [133, 173, 157, 158, 159, 160, 161, 246, 33],
            'lower': [133, 155, 154, 153, 145, 144, 163, 7, 33]},
        'subjectLeft': {
            'upper': [362, 398, 384, 385, 386, 387, 388, 466, 263],
            'lower': [362, 382, 381, 380, 374, 373, 390, 249, 263]},
    }
    samples = sampleSemanticEyelids(original[0], binding)
    for name, paths in expected.items():
        for lid, ids in paths.items():
            assert np.array_equal(binding['eyes'][name][lid]['mpIds'], ids)
            assert np.array_equal(samples[name][lid][0], samples[name]['innerCorner'])
            assert np.array_equal(samples[name][lid][-1], samples[name]['outerCorner'])
        upper, lower = samples[name]['upper'], samples[name]['lower']
        assert np.all(upper[1:-1, 1] > lower[1:-1, 1])
        assert abs(samples[name]['innerCorner'][0]) < abs(samples[name]['outerCorner'][0])


def testSamplingEqualsLegacyAndCommutesWithDeformation(original, binding):
    vertices = original[0].copy()
    vertices += np.random.default_rng(42).normal(0, .02, vertices.shape)
    preserved = vertices.copy()
    legacy = sampleClientLandmarks(vertices, original[2])
    samples = sampleSemanticEyelids(vertices, binding)
    for name, eye in binding['eyes'].items():
        for role, row in eye.items():
            expected = legacy[row['mpIds']]
            if role.endswith('Corner'):
                expected = expected[0]
            assert np.array_equal(samples[name][role], expected)
    assert np.array_equal(vertices, preserved)


def testNonSkinMovementCannotChangeBindingOrSamples(original, binding):
    full, skinIds = original[3].copy(), original[4]
    nonSkin = np.ones(len(full), bool)
    nonSkin[skinIds] = False
    before = sampleSemanticEyelids(full[skinIds], binding)
    full[nonSkin] += [100., -200., 300.]
    after = sampleSemanticEyelids(full[skinIds], binding)
    rebuilt = buildSemanticEyelidBinding(original[1])
    assert rebuilt['bindingSha256'] == binding['bindingSha256']
    for name in before:
        for role in before[name]:
            assert np.array_equal(before[name][role], after[name][role])


def testCandidateDoesNotClaimSurfacePathOrAnatomicalTruth(binding):
    assert binding['reviewStatus'] == 'unreviewedLegacySemanticCandidate'
    assert binding['surfacePathCreated'] is False
    assert binding['anatomicalAccuracyValidated'] is False
    for eye in binding['eyes'].values():
        for role in ('upper', 'lower'):
            assert len(eye[role]['mpIds']) == 9
    assert 'surfacePaths' not in binding


def testWrongNumberedTopologyRejected(original):
    faces = original[1].copy()
    faces[[0, 1]] = faces[[1, 0]]
    with pytest.raises(ValueError, match='topology|order'):
        buildSemanticEyelidBinding(faces)


@pytest.mark.parametrize('field', ['sourceSha256', 'nicpFacesHash', 'barycentricWeights', 'vertexIndices'])
def testStaleOrReboundSuppliedSourceRejected(original, field):
    supplied = deepcopy(original[2])
    if field.endswith('Hash') or field.endswith('Sha256'):
        supplied[field] = '0'*64
    else:
        supplied[field] = supplied[field].copy()
        supplied[field][133, 0] += .001 if field == 'barycentricWeights' else 1
    with pytest.raises(ValueError, match='source'):
        buildSemanticEyelidBinding(original[1], supplied)


@pytest.mark.parametrize('mutation', ['reverse', 'corner', 'source', 'weights'])
def testTamperedCandidateRejectedOnSample(original, binding, mutation):
    candidate = deepcopy(binding)
    row = candidate['eyes']['subjectRight']['upper']
    if mutation == 'reverse':
        row['mpIds'] = row['mpIds'][::-1].copy()
    elif mutation == 'corner':
        candidate['eyes']['subjectRight']['innerCorner']['vertexIndices'][0, 0] += 1
    elif mutation == 'source':
        candidate['sourceSha256'] = '0'*64
    else:
        row['barycentricWeights'][2, 0] += .001
        row['barycentricWeights'][2, 1] -= .001
    with pytest.raises(ValueError):
        sampleSemanticEyelids(original[0], candidate)


def testProjectionIsOnlyExplicitCameraProjection(original, binding):
    camera = dict(intrinsics=np.array([[100., 0., 50.], [0., 120., 40.], [0., 0., 1.]]),
        rotation=np.eye(3), translation=np.array([0., -168., 20.]))
    projected = projectSemanticEyelids(original[0], binding, camera)
    sampled = sampleSemanticEyelids(original[0], binding)
    for name, eye in sampled.items():
        for role, points in eye.items():
            cameraPoints = points+camera['translation']
            expected = cameraPoints @ camera['intrinsics'].T
            expected = expected[..., :2]/expected[..., 2, None]
            assert np.array_equal(projected[name][role]['pixels'], expected)
            assert np.asarray(projected[name][role]['positiveDepthMask']).all()


def testBehindCameraReportedAsUnprojectable(original, binding):
    camera = dict(intrinsics=np.eye(3), rotation=np.eye(3), translation=np.array([0., 0., -100.]))
    projected = projectSemanticEyelids(original[0], binding, camera)
    for eye in projected.values():
        for row in eye.values():
            assert np.isnan(row['pixels']).all()
            assert not np.asarray(row['positiveDepthMask']).any()


@pytest.mark.parametrize('badVertices', [np.zeros((2794, 3)), np.full((2795, 3), np.nan)])
def testInvalidGeometryRejected(binding, badVertices):
    with pytest.raises(ValueError, match='vertices'):
        sampleSemanticEyelids(badVertices, binding)
