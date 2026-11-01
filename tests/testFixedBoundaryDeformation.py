"""Representation tests independent of photographs and acceptance thresholds."""
import numpy as np
import pytest
from scipy import sparse

import faceAppearance.fixedBoundaryDeformation as module
from faceAppearance.eyeRefinement import buildEyeDeformationBasis
from faceAppearance.fixedBoundaryDeformation import buildFixedBoundaryDeformationBasis


def meshGrid(size=13):
    grid = np.linspace(-1., 1., size)
    x, y = np.meshgrid(grid, grid)
    vertices = np.column_stack([x.ravel(), y.ravel(), np.zeros(size*size)])
    faces = []
    for row in range(size-1):
        for col in range(size-1):
            first = row*size+col
            faces.extend([[first, first+1, first+size], [first+1, first+size+1, first+size]])
    fixed = np.flatnonzero((np.abs(vertices[:, 0]) == 1) | (np.abs(vertices[:, 1]) == 1))
    return vertices, np.asarray(faces, np.int64), fixed


def rebuild(value):
    return sparse.csr_matrix((value['data'], value['indices'], value['indptr']), shape=tuple(value['shape']))


def build(vertices=None, faces=None, fixed=None, points=None, **options):
    if vertices is None:
        vertices, faces, fixed = meshGrid()
    if points is None:
        points = np.array([[-.31, .21, 0.], [.23, -.13, 0.], [.41, .29, 0.]])
    return buildFixedBoundaryDeformationBasis(vertices, faces, points,
        fixedVertexIds=fixed, supportRadius=options.pop('supportRadius', .8), **options)


def testExactBoundaryAndEquationWithEnergyCertificate():
    vertices, faces, fixed = meshGrid()
    result = build(vertices, faces, fixed)
    evidence = result['evidence']
    free = evidence['freeVertexIds']
    matrix = rebuild(evidence['systemMatrix'])
    bending = rebuild(evidence['bending'])
    stiffness = rebuild(evidence['stiffness'])
    raw = evidence['rawBasis']
    driving = evidence['drivingBasis']
    mass = evidence['lumpedMass']
    np.testing.assert_allclose(bending.toarray(), (stiffness.T @ sparse.diags(1/mass) @ stiffness).toarray(), atol=1e-12)
    # Full rows before free-column restriction preserve the fixed-boundary
    # slope term. Restricting K first is a different, weaker objective.
    croppedK = stiffness[free][:, free]
    croppedQ = croppedK.T @ sparse.diags(1/mass[free]) @ croppedK
    assert np.max(np.abs(bending[free][:, free].toarray()-croppedQ.toarray())) > .1
    assert np.array_equal(result['referenceVertices'], vertices)
    assert np.array_equal(result['faces'], faces)
    assert np.array_equal(result['basis'][fixed], np.zeros((len(fixed), 3)))
    assert np.array_equal(result['activeVertexMask'], np.any(result['basis'] != 0, axis=1))
    np.testing.assert_allclose(matrix @ raw[free], evidence['rightHandSide'], atol=1e-12)
    np.testing.assert_allclose(result['basis']*result['normalization'], raw, atol=1e-15)
    assert result['diagnostics']['solveRelativeResidual'] < 1e-12
    assert np.abs(result['basis']).sum(1).max() <= 1+1e-14
    # The computed projection beats the tempting but nonsmooth operation of
    # simply zeroing original drivers at the fixed boundary.
    clipped = driving.copy()
    clipped[fixed] = 0
    def energy(value):
        delta = value[free]-driving[free]
        return float((mass[free, None]*delta**2).sum() + .1**4*(value*(bending @ value)).sum())
    assert energy(raw) < energy(clipped)
    np.testing.assert_allclose(rebuild(evidence['stiffness']).sum(1), 0., atol=1e-14)
    eigenvalues = np.linalg.eigvalsh(matrix.toarray())
    assert eigenvalues.min() > 0


def testNearbyFixedControlDoesNotCollapseRemoteField():
    vertices, faces, fixed = meshGrid(21)
    points = np.array([[-.999, 0., 0.], [.5, 0., 0.]])
    requested = .7
    original = buildEyeDeformationBasis(vertices, points, supportRadius=requested, controlCount=32)
    cap = np.linalg.norm(original['controlPoints'][:, None]-vertices[fixed][None], axis=2).min()*(1-1e-12)
    old = buildEyeDeformationBasis(vertices, points, supportRadius=cap, controlCount=32)
    new = build(vertices, faces, fixed, points, supportRadius=requested)
    assert old['activeVertexMask'].sum() <= 1
    assert new['activeVertexMask'].sum() > 200
    # More than a machine-noise tail participates near the second control.
    remote = (vertices[:, 0] > .2) & (vertices[:, 0] < .8) & (np.abs(vertices[:, 1]) < .3)
    assert (np.linalg.norm(new['basis'][remote], axis=1) > .01).sum() > 10
    assert new['diagnostics']['meaningfulActiveVertexCount'] > 200
    np.testing.assert_array_equal(new['controlPoints'], original['controlPoints'])


def testDisconnectedFixedAccessoryHasNoEffectAndNoProximityBridge():
    vertices, faces, fixed = meshGrid()
    points = np.array([[-.31, .21, 0.], [.23, -.13, 0.]])
    reference = build(vertices, faces, fixed, points)
    accessory = np.array([[-.01, -.01, 1e-4], [.01, -.01, 1e-4], [0., .01, 1e-4]])
    count = len(vertices)
    combined = build(np.r_[vertices, accessory], np.r_[faces, [[count, count+1, count+2]]],
        np.r_[fixed, np.arange(count, count+3)], points)
    np.testing.assert_allclose(combined['basis'][:count], reference['basis'], rtol=1e-10, atol=1e-12)
    assert not combined['basis'][count:].any()
    stiffness = rebuild(combined['evidence']['stiffness'])
    assert stiffness[:count, count:].nnz == 0
    assert rebuild(combined['evidence']['bending'])[:count, count:].nnz == 0


def testTriangleTopologyDoesNotBridgeCloseOpposingSheets():
    vertices, faces, fixed = meshGrid(7)
    count = len(vertices)
    other = vertices+np.array([0., 0., .001])
    # Both sheets are movable inside their own fixed boundary. Euclidean
    # drivers correlate them, but the actual operator has no cross-sheet edge.
    result = build(np.r_[vertices, other], np.r_[faces, faces+count], np.r_[fixed, fixed+count])
    assert rebuild(result['evidence']['stiffness'])[:count, count:].nnz == 0
    assert rebuild(result['evidence']['bending'])[:count, count:].nnz == 0


@pytest.mark.parametrize('scale', [1e-5, .07, 13., 1e5])
def testScaleAndTranslationEquivariance(scale):
    vertices, faces, fixed = meshGrid(9)
    points = np.array([[-.31, .21, 0.], [.23, -.13, 0.], [.41, .29, 0.]])
    reference = build(vertices, faces, fixed, points)
    translation = scale*np.array([13.7, -4.1, 1.3])
    transformed = build(vertices*scale+translation, faces, fixed, points*scale+translation,
                        supportRadius=.8*scale)
    np.testing.assert_allclose(transformed['basis'], reference['basis'], rtol=2e-10, atol=2e-12)
    np.testing.assert_allclose(transformed['controlPoints'], reference['controlPoints']*scale+translation)


def testRigidRotationAndVertexNumberingEquivariance():
    vertices, faces, fixed = meshGrid(9)
    points = np.array([[-.31, .21, 0.], [.23, -.13, 0.], [.41, .29, 0.]])
    reference = build(vertices, faces, fixed, points)
    angle = .73
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    transformed = build(vertices @ rotation.T, faces, fixed, points @ rotation.T)
    np.testing.assert_allclose(transformed['basis'], reference['basis'], rtol=2e-10, atol=2e-12)
    permutation = np.random.default_rng(17).permutation(len(vertices))
    inverse = np.empty(len(vertices), np.int64)
    inverse[permutation] = np.arange(len(vertices))
    renumbered = build(vertices[permutation], inverse[faces], inverse[fixed], points)
    np.testing.assert_allclose(renumbered['basis'][inverse], reference['basis'], rtol=2e-10, atol=2e-12)


def testThinObtuseTrianglesRetainFiniteElementWeights():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [.2, .002, 0.], [1., 1., 0.]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    result = build(vertices, faces, np.array([0, 1, 3]), np.array([[.2, .002, 0.]]))
    diagnostics = result['diagnostics']
    assert diagnostics['triangleDiagnostics']['negativeCotangentCount'] > 0
    assert diagnostics['triangleDiagnostics']['minimumTriangleQuality'] < .01
    assert diagnostics['numericalRidge'] == 0
    assert diagnostics['cotangentClipping'] is False
    assert diagnostics['solveRelativeResidual'] < 1e-10
    stiffness = rebuild(result['evidence']['stiffness']).toarray()
    assert np.linalg.eigvalsh(stiffness).min() > -1e-10


def testEmptyControlsAndAllFixedReturnHonestZeroBasis():
    vertices, faces, fixed = meshGrid(7)
    empty = build(vertices, faces, fixed, np.empty((0, 3)))
    assert empty['basis'].shape == (len(vertices), 0)
    assert not empty['activeVertexMask'].any()
    assert empty['diagnostics']['status'] == 'noControls'
    full = build(vertices, faces, np.arange(len(vertices)))
    assert not full['basis'].any()
    assert full['diagnostics']['status'] == 'allVerticesFixed'
    assert full['diagnostics']['numericalRank'] == 0


def testFixedIsolatedVertexIsAllowedButFreeIsolatedVertexIsRejected():
    vertices, faces, fixed = meshGrid(7)
    combined = np.r_[vertices, [[0., 0., 1.]]]
    reference = build(vertices, faces, fixed)
    result = build(combined, faces, np.r_[fixed, len(vertices)])
    np.testing.assert_allclose(result['basis'][:-1], reference['basis'], rtol=1e-10, atol=1e-12)
    assert not result['basis'][-1].any()
    with pytest.raises(ValueError, match='isolated'):
        build(combined, faces, fixed)


@pytest.mark.parametrize('options', [dict(controlCount=True), dict(controlCount=0),
    dict(supportRadius=0), dict(supportRadius=np.nan), dict(smoothingRadiusFraction=0),
    dict(smoothingRadiusFraction=np.inf), dict(smoothingRadiusFraction=True),
    dict(supportRadius='bad'), dict(smoothingRadiusFraction='bad')])
def testInvalidNumericConfigurationRejected(options):
    with pytest.raises(ValueError):
        build(**options)


def testBadIndicesDegeneracyAndWorkBudgetsRejected(monkeypatch):
    vertices, faces, fixed = meshGrid(7)
    with pytest.raises(ValueError, match='distinct'):
        build(vertices, faces, np.r_[fixed, fixed[0]])
    with pytest.raises(ValueError, match='one-dimensional'):
        build(vertices, faces, np.empty((0, 2), int))
    with pytest.raises(ValueError, match='Triangle indices'):
        build(vertices, np.array([[0, 0, 1]]), fixed)
    with pytest.raises(ValueError, match='degenerate'):
        build(vertices, np.array([[0, 1, 2]]), fixed)
    monkeypatch.setattr(module, 'MAX_DENSE_ELEMENTS', 1)
    with pytest.raises(ValueError, match='work budget'):
        build(vertices, faces, fixed)


def testSparseProductBudgetIsCheckedBeforeMultiplication(monkeypatch):
    monkeypatch.setattr(module, 'MAX_SPARSE_PRODUCT_TERMS', 1)
    with pytest.raises(ValueError, match='sparse product'):
        build()
