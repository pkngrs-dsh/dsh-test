"""The optional mesh field keeps the existing solver gates and camera frame."""
from copy import deepcopy

import numpy as np
import pytest

from faceAppearance.eyeRefinement import refineEyeGeometry
from faceAppearance.fixedBoundaryDeformation import buildFixedBoundaryDeformationBasis
from faceAppearance.frontLandmarkRefinement import refineFrontLandmarks
from testEyeRefinement import makeGrid, makeCameras, pointConstraint
from testEyeRefinementHooks import setupFit


def testEvidenceCaptureDoesNotChangeLegacyNumericsOrStepSelection():
    args, options = setupFit()
    old = refineEyeGeometry(*args, **options)
    result = refineEyeGeometry(*args, **options, captureSolverEvidence=True)
    np.testing.assert_array_equal(old['vertices'], result['vertices'])
    np.testing.assert_array_equal(old['controlOffsets'], result['controlOffsets'])
    assert old['history'] == result['history']
    assert old['diagnostics']['rejections'] == result['diagnostics']['rejections']
    evidence = result['solverEvidence']
    assert evidence['trials']
    for trial in evidence['trials']:
        np.testing.assert_array_equal(trial['trialCoefficients'], trial['beforeCoefficients']
            + trial['stepScale']*trial['coefficientStep'])
        if trial['evaluation'] is not None:
            evaluation = trial['evaluation']
            assert evaluation['objective'] == pytest.approx(sum(evaluation['objectiveComponents'].values()), abs=1e-12)
    reconstructed = evidence['referenceVertices'] + evidence['maxDisplacement']*(
        (evidence['basis'] @ evidence['finalCoefficients']) @ evidence['axes'].T)
    np.testing.assert_array_equal(reconstructed, result['vertices'])


def frontSetup():
    vertices, faces = makeGrid()
    camera = makeCameras()[1]
    fixed = np.flatnonzero(np.max(np.abs(vertices[:, :2]), axis=1) > 1.3)
    indices = np.array([220, 221, 241])
    target = vertices[indices] + np.array([.06, .04]) @ camera['rotation'][:2]
    constraint = pointConstraint(indices, target, camera)
    return vertices, faces, camera, fixed, constraint


def testNewFrontFieldCoordinatesMovementWhilePreservingFixedBoundaryAndDepth():
    vertices, faces, camera, fixed, constraint = frontSetup()
    options = dict(fixedVertexIds=fixed, supportRadius=2.5, maxDisplacement=.2,
        iterations=20, controlCount=3, captureSolverEvidence=True)
    old = refineFrontLandmarks(vertices, faces, camera, [constraint], **options)
    result = refineFrontLandmarks(vertices, faces, camera, [constraint],
        deformationMode='fixedBoundaryBiharmonic', **options)
    assert result['diagnostics']['fixedBoundary']['effectiveSupportRadius'] == 2.5
    assert old['diagnostics']['fixedBoundary']['effectiveSupportRadius'] < 2.5
    np.testing.assert_array_equal(old['controlPoints'], result['controlPoints'])
    np.testing.assert_array_equal(result['vertices'][fixed], vertices[fixed])
    assert np.count_nonzero(np.linalg.norm(result['vertices']-vertices, axis=1) > 1e-8) > 100
    np.testing.assert_allclose((result['vertices']-vertices) @ camera['rotation'][2], 0, atol=1e-15)
    assert result['diagnostics']['finalMeanErrorPixels'] < result['diagnostics']['initialMeanErrorPixels']
    assert result['solverEvidence']['fieldConstruction'] is not None


def externalSetup():
    vertices, faces, camera, fixed, constraint = frontSetup()
    field = buildFixedBoundaryDeformationBasis(vertices, faces, vertices[[220,221,241]],
        fixedVertexIds=fixed, supportRadius=2.5, controlCount=3)
    return (vertices, faces, [camera], [constraint]), field


@pytest.mark.parametrize('value', [True, '0.1', [.1], np.nan, 0., 1.1])
def testFrontRejectsInvalidSmoothingConfiguration(value):
    vertices, faces, camera, fixed, constraint = frontSetup()
    with pytest.raises(ValueError, match='smoothingRadiusFraction'):
        refineFrontLandmarks(vertices, faces, camera, [constraint], fixedVertexIds=fixed,
            deformationMode='fixedBoundaryBiharmonic', smoothingRadiusFraction=value)


@pytest.mark.parametrize('mutation', ['reference', 'faces', 'mask', 'nonfinite', 'rowBound', 'normalization', 'budget'])
def testExternalBasisRejectsStaleIdentityAndInvalidFields(mutation):
    args, field = externalSetup()
    field = deepcopy(field)
    options = dict(controlCount=3)
    if mutation == 'reference': field['referenceVertices'][0,0] += .001
    elif mutation == 'faces': field['faces'][0] = field['faces'][0,::-1]
    elif mutation == 'mask': field['activeVertexMask'][0] = ~field['activeVertexMask'][0]
    elif mutation == 'nonfinite': field['basis'][220,0] = np.nan
    elif mutation == 'rowBound': field['basis'][220,0] = 2.
    elif mutation == 'normalization': field['normalization'] = [1.]
    elif mutation == 'budget': options['controlCount'] = 2
    with pytest.raises(ValueError):
        refineEyeGeometry(*args, deformationBasis=field, **options)


def testFieldRetainsExtraSafetyAtEveryTrialAndSeparatesGateFailures():
    args, field = externalSetup()
    vertices = args[0]
    direction = args[2][0]['rotation'][0]
    seen = []
    def safety(candidate):
        seen.append(candidate.copy())
        return dict(accepted=bool((candidate[220]-vertices[220]) @ direction <= .005))
    result = refineEyeGeometry(*args, deformationBasis=field, controlCount=3,
        supportRadius=2.5, maxDisplacement=.2, deformationFrame='referenceImagePlane',
        iterations=10, additionalSafety=safety, captureSolverEvidence=True)
    trials = result['solverEvidence']['trials']
    rejected = [item for item in trials if item['safety']['geometricSafetyAccepted']
        and not item['safety']['additionalSafety']['accepted']]
    assert rejected
    assert all(item['decision'] == 'safetyRejected' and item['evaluation'] is None for item in rejected)
    assert (result['vertices'][220]-vertices[220]) @ direction <= .005
    assert any(np.array_equal(candidate, result['vertices']) for candidate in seen)
