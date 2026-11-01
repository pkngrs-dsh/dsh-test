"""Optional physical residuals and trial-step safety in the existing solver."""
import numpy as np
import pytest
from scipy import sparse

from faceAppearance.eyeRefinement import refineEyeGeometry
from testEyeRefinement import makeGrid, makeCameras, pointConstraint


def setupFit(dtype=np.float64):
    vertices, faces = makeGrid(dtype=dtype)
    cameras = makeCameras()[:1]
    constraint = pointConstraint([220], vertices[[220]]+[.15, .04, 0], cameras[0])
    return (vertices, faces, cameras, [constraint]), dict(supportRadius=2.5,
        maxDisplacement=.3, iterations=25, deformationFrame='referenceImagePlane')


def testNoneHooksLeaveLegacyNumericsIdentical():
    args, options = setupFit()
    before = refineEyeGeometry(*args, **options)
    after = refineEyeGeometry(*args, **options, additionalResidual=None, additionalSafety=None)
    np.testing.assert_array_equal(before['vertices'], after['vertices'])
    assert before['diagnostics'] == after['diagnostics']
    assert before['history'] == after['history']


def testSparseVertexResidualChangesSolutionWithCorrectParameterChain():
    args, options = setupFit()
    count = len(args[0]); offset = args[0][220, 0]
    def residual(vertices, withJacobian):
        assert not vertices.flags.writeable
        result = {'residual': np.array([10*(vertices[220, 0]-offset)])}
        if withJacobian:
            result['jacobian'] = sparse.csr_matrix(([10.], ([0], [660])), shape=(1, 3*count))
        return result
    ordinary = refineEyeGeometry(*args, **options)
    constrained = refineEyeGeometry(*args, **options, additionalResidual=residual)
    # Camera derivative is 20 px/world unit, pixelScale is 4, so the extra
    # x stiffness 100 trades against photo stiffness 25: optimal x ~ .03.
    shift = constrained['vertices'][220]-args[0][220]
    assert .025 < shift[0] < .035
    assert ordinary['vertices'][220, 0]-offset > .12
    assert constrained['diagnostics']['additionalHooks']['residualCount'] == 1


@pytest.mark.parametrize('dtype', [np.float64, np.float32])
def testHardHookRejectsStepsAndChecksRoundedOutput(dtype):
    args, options = setupFit(dtype)
    seen = []
    def safety(vertices):
        seen.append(vertices.copy())
        return {'accepted': bool(vertices[220, 0] <= .02)}
    result = refineEyeGeometry(*args, **options, additionalSafety=safety)
    assert 0 < result['vertices'][220, 0] <= .02
    assert result['diagnostics']['additionalHooks']['safetyRejections'] > 0
    assert any(np.array_equal(item, result['vertices'].astype(float)) for item in seen)


def testUnsafeBaselineCannotBypassHook():
    args, options = setupFit()
    with pytest.raises(ValueError, match='Initial mesh'):
        refineEyeGeometry(*args, **options, additionalSafety=lambda v: {'accepted': False})


@pytest.mark.parametrize('callback', [lambda v,j: {}, lambda v,j: {'residual': [np.nan]},
    lambda v,j: {'residual': [[0.]]}, lambda v,j: {'residual': [0.], 'jacobian': np.zeros((1,1323))}])
def testInvalidResidualRejected(callback):
    args, options = setupFit()
    with pytest.raises(ValueError):
        refineEyeGeometry(*args, **options, additionalResidual=callback)


def testCallbacksCannotMutateSolverCoordinates():
    args, options = setupFit()
    original = args[0].copy()
    def mutation(vertices, withJacobian):
        vertices[0, 0] = 2
    with pytest.raises(ValueError, match='read-only'):
        refineEyeGeometry(*args, **options, additionalResidual=mutation)
    np.testing.assert_array_equal(args[0], original)


@pytest.mark.parametrize('returned', [{}, {'accepted': 1}, {'accepted': 'false'}])
def testSafetyMustReturnExplicitBoolean(returned):
    args, options = setupFit()
    with pytest.raises(ValueError, match='boolean'):
        refineEyeGeometry(*args, **options, additionalSafety=lambda v: returned)
