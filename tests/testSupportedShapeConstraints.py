import copy
import numpy as np
import pytest
from scipy import sparse

from faceAppearance.supportedShapeConstraints import (
    buildSupportedShapeConstraintSpec, evaluateSupportedShapeConstraints,
)
from faceAppearance.guardedMeshQuadratic import solveGuardedMeshQuadratic


def fixture():
    vertices = np.asarray([[x, y, .025*x*y] for y in range(4) for x in range(5)], float)
    vertices = np.vstack((vertices, [9., 9., 2.]))  # isolated unsupported vertex
    faces = []
    for y in range(3):
        for x in range(4):
            a = 5*y+x
            faces.extend(((a, a+1, a+6), (a, a+6, a+5)))
    faces = np.asarray(faces, np.int64)
    sampled = np.setdiff1d(np.arange(len(vertices)), [6, 7, 9, 20])
    c = sparse.csr_matrix((np.ones(len(sampled)), (np.arange(len(sampled)), sampled)),
        shape=(len(sampled), len(vertices)))
    return dict(referenceVertices=vertices, faces=faces, sampleMatrix=c,
        staticVertexIds=np.asarray([0, 4, 15, 19]), delegatedVertexIds=np.asarray([9]),
        protectedVertexIds=np.asarray([10]), lengthScale=1.)


def build(**overrides):
    inputs = fixture()
    inputs.update(overrides)
    return buildSupportedShapeConstraintSpec(**inputs)


def canonicalRows(spec, result, vertexNewToOld=None, faceNewToOld=None):
    vertexNewToOld = np.arange(spec['vertexCount']) if vertexNewToOld is None else vertexNewToOld
    faceNewToOld = np.arange(len(spec['faces'])) if faceNewToOld is None else faceNewToOld
    rows = []
    for name, span in result['groups'].items():
        if name.startswith('edge'):
            ids = np.sort(vertexNewToOld[spec['edgeVertexIds']], axis=1)
            order = np.lexsort(ids.T[::-1])
        elif name.startswith('area') or name.startswith('normal'):
            order = np.argsort(faceNewToOld[spec['faceIds']])
        else:
            order = np.argsort(vertexNewToOld[spec['controlVertexIds']])
        rows.extend(span['start'] + order)
    rows = np.asarray(rows, int)
    columns = (3*np.argsort(vertexNewToOld)[:, None] + np.arange(3)).ravel()
    return result['values'][rows], result['jacobian'][rows][:, columns].toarray()


def testSupportTopologyAndDelegationAreRebuiltWithoutSpecialIds():
    spec = build()
    np.testing.assert_array_equal(np.flatnonzero(spec['unsupportedFreeMask']), [6, 7, 20])
    assert {frozenset(v) for v in spec['unsupportedComponentVertexIds']} == {frozenset([6, 7]), frozenset([20])}
    np.testing.assert_array_equal(spec['hardFixedVertexIds'], [0, 4, 6, 7, 15, 19, 20])
    assert 9 not in spec['hardFixedVertexIds']  # explicitly delegated elsewhere
    edges = {tuple(sorted(pair)) for f in fixture()['faces'] for pair in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}
    neighbors = {i for edge in edges if any(j in (6, 7, 20) for j in edge) for i in edge}
    expectedRing = sorted(neighbors - {0, 4, 6, 7, 9, 15, 19, 20})
    np.testing.assert_array_equal(spec['transitionVertexIds'], expectedRing)
    assert 10 in spec['controlVertexIds']
    allowed = np.union1d(spec['hardFixedVertexIds'], spec['controlVertexIds'])
    assert np.isin(spec['edgeVertexIds'], allowed).all()
    assert np.isin(spec['faceVertexIds'], allowed).all()
    assert spec['diagnostics']['supportThreshold'] == 0
    assert not spec['diagnostics']['allExteriorConstraintsIncluded']


def testIndependentPhysicalValuesAndAnalyticJacobianFiniteDifference():
    spec = build()
    current = spec['referenceVertices'].copy()
    rng = np.random.default_rng(51)
    current += rng.normal(size=current.shape)*.004
    result = evaluateSupportedShapeConstraints(current, spec)
    edge = spec['edgeVertexIds']
    lengths = np.linalg.norm(current[edge[:, 1]]-current[edge[:, 0]], axis=1)
    triangles = current[spec['faceVertexIds']]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1)*.5
    cosine = np.sum(cross/(2*areas[:, None])*spec['safetyReferenceNormals'], axis=1)
    controls = spec['controlVertexIds']
    expected = np.r_[lengths-spec['edgeLower'], spec['edgeUpper']-lengths,
        areas-spec['areaLower'], spec['areaUpper']-areas,
        cosine-spec['policy']['minimumNormalCosine'],
        spec['displacementLimits']**2-np.sum((current[controls]-spec['safetyReferenceVertices'][controls])**2, axis=1)]
    np.testing.assert_allclose(result['values'], expected, atol=1e-14)
    numeric = np.empty(result['jacobian'].shape)
    epsilon = 2e-6
    for column in range(current.size):
        plus, minus = current.copy(), current.copy()
        plus.ravel()[column] += epsilon
        minus.ravel()[column] -= epsilon
        numeric[:, column] = (evaluateSupportedShapeConstraints(plus, spec, False)['values']
            - evaluateSupportedShapeConstraints(minus, spec, False)['values'])/(2*epsilon)
    np.testing.assert_allclose(result['jacobian'].toarray(), numeric, rtol=2e-5, atol=3e-10)
    assert 'jacobian' not in evaluateSupportedShapeConstraints(current, spec, False)


def testReferenceIsFeasibleAndProtectedLimitTakesPrecedence():
    spec = build(protectedVertexIds=np.asarray([1, 10]))
    result = evaluateSupportedShapeConstraints(spec['referenceVertices'], spec)
    assert result['values'].min() >= 0
    assert dict(zip(spec['controlVertexIds'], spec['displacementLimits']))[1] == .1
    moved = spec['referenceVertices'].copy(); moved[1, 2] += .11
    values = evaluateSupportedShapeConstraints(moved, spec, False)
    span = values['groups']['controlDisplacement']
    local = np.flatnonzero(spec['controlVertexIds'] == 1)[0]
    assert values['values'][span['start'] + local] < 0


def testSafetyReferenceIsNotResetToStartingGeometry():
    data = fixture()
    safety = data['referenceVertices'].copy()
    data['referenceVertices'] = safety.copy()
    data['referenceVertices'][10, 2] += .06
    spec = buildSupportedShapeConstraintSpec(**data, safetyReferenceVertices=safety)
    moved = spec['referenceVertices'].copy(); moved[10, 2] += .05
    result = evaluateSupportedShapeConstraints(moved, spec)
    row = result['groups']['controlDisplacement']['start'] + np.flatnonzero(spec['controlVertexIds'] == 10)[0]
    np.testing.assert_allclose(result['values'][row], .1**2-.11**2)


def testObservationOrderingReplicationExplicitZerosAndSparseFormatDoNotChangeDomains():
    original = fixture()
    spec = buildSupportedShapeConstraintSpec(**original)
    c = original['sampleMatrix']
    repeated = sparse.vstack((c[::-1], c, c[::-1])).tocoo()
    # Two half-weight entries preserve the same exact affine row without an
    # implementation-side normalization; explicit zeros never create support.
    duplicated = sparse.coo_matrix((np.r_[repeated.data*.5, repeated.data*.5, 0.],
        (np.r_[repeated.row, repeated.row, 0], np.r_[repeated.col, repeated.col, 20])), shape=repeated.shape)
    other = build(sampleMatrix=duplicated)
    for key in ('supportedMask', 'unsupportedFreeMask', 'hardFixedVertexIds', 'transitionVertexIds',
                'controlVertexIds', 'edgeVertexIds', 'faceIds'):
        np.testing.assert_array_equal(spec[key], other[key])
    np.testing.assert_array_equal(evaluateSupportedShapeConstraints(spec['referenceVertices'], spec)['values'],
        evaluateSupportedShapeConstraints(other['referenceVertices'], other)['values'])


def testArbitrarilySmallNonzeroSamplingIsStillAvailability():
    data = fixture()
    c = data['sampleMatrix'].tolil()
    c[0, 0] -= 1e-13
    c[0, 6] = 1e-13
    spec = build(sampleMatrix=c.tocsr())
    assert spec['supportedMask'][6]
    assert 6 not in spec['hardFixedVertexIds']


def testVertexAndFaceRenumberingCovariance():
    data = fixture(); spec = buildSupportedShapeConstraintSpec(**data)
    rng = np.random.default_rng(82)
    order = rng.permutation(len(data['referenceVertices']))
    inverse = np.argsort(order)
    faceOrder = rng.permutation(len(data['faces']))
    permuted = {**data, 'referenceVertices': data['referenceVertices'][order],
        'faces': inverse[data['faces'][faceOrder]], 'sampleMatrix': data['sampleMatrix'][:, order],
        **{key: inverse[data[key]] for key in ('staticVertexIds', 'delegatedVertexIds', 'protectedVertexIds')}}
    other = buildSupportedShapeConstraintSpec(**permuted)
    assert set(order[other['hardFixedVertexIds']]) == set(spec['hardFixedVertexIds'])
    assert {frozenset(order[v]) for v in other['unsupportedComponentVertexIds']} == {frozenset(v) for v in spec['unsupportedComponentVertexIds']}
    current = data['referenceVertices'] + rng.normal(size=data['referenceVertices'].shape)*.003
    values, jacobian = canonicalRows(spec, evaluateSupportedShapeConstraints(current, spec))
    newValues, newJacobian = canonicalRows(other, evaluateSupportedShapeConstraints(current[order], other), order, faceOrder)
    np.testing.assert_allclose(newValues, values, atol=1e-14)
    np.testing.assert_allclose(newJacobian, jacobian, atol=1e-14)


def testTranslationRotationAndPhysicalScaleCovarianceOfValuesAndJacobian():
    data = fixture(); spec = buildSupportedShapeConstraintSpec(**data)
    rotation, _ = np.linalg.qr(np.random.default_rng(19).normal(size=(3, 3)))
    assert np.linalg.det(rotation) > 0
    translation, scale = np.array([21., -31., 3.]), 7.2
    transformed = {**data, 'referenceVertices': scale*data['referenceVertices']@rotation.T+translation,
        'lengthScale': scale*data['lengthScale']}
    other = buildSupportedShapeConstraintSpec(**transformed)
    current = data['referenceVertices'].copy(); current[1, 2] += .02
    original = evaluateSupportedShapeConstraints(current, spec)
    moved = evaluateSupportedShapeConstraints(scale*current@rotation.T+translation, other)
    np.testing.assert_allclose(moved['values'], original['values'], atol=1e-13)
    mappedJacobian = (moved['jacobian'].toarray().reshape(-1, len(current), 3)@rotation*scale).reshape(original['jacobian'].shape)
    np.testing.assert_allclose(mappedJacobian, original['jacobian'].toarray(), atol=1e-13)


def testExactSolverFixesUnsupportedCoordinatesWhileSupportedFitRemainsActive():
    spec = build()
    reference = spec['referenceVertices']
    fixed, controls = spec['hardFixedVertexIds'], spec['controlVertexIds']
    target = reference.copy(); target[:, 2] += .015
    expected = target.copy(); expected[fixed] = reference[fixed]
    result = solveGuardedMeshQuadratic(reference, sparse.eye(reference.size, format='csr'), target.ravel(),
        controls, fixed, lambda v: evaluateSupportedShapeConstraints(v, spec), lengthScale=1.)
    assert result['diagnostics']['solverAccepted']
    np.testing.assert_array_equal(result['rawVertices'][fixed], reference[fixed])
    np.testing.assert_allclose(result['rawVertices'], expected, atol=3e-7)
    assert np.max(abs(result['rawVertices'][controls]-reference[controls])) > .01


def testAllSupportedEmptyLocalDomainAndProtectedOnlyAreWellDefined():
    identity = sparse.eye(len(fixture()['referenceVertices']), format='csr')
    spec = build(sampleMatrix=identity, protectedVertexIds=np.array([], np.int64))
    assert not spec['unsupportedFreeMask'].any()
    result = evaluateSupportedShapeConstraints(spec['referenceVertices'], spec)
    assert result['values'].shape == (0,)
    assert result['jacobian'].shape == (0, spec['referenceVertices'].size)
    protectedOnly = build(sampleMatrix=identity)
    assert len(protectedOnly['controlVertexIds']) == 1
    assert len(evaluateSupportedShapeConstraints(protectedOnly['referenceVertices'], protectedOnly)['values']) == 1


def testNoSamplesPreservesUnsupportedComponentsIncludingIsolatedVertices():
    n = len(fixture()['referenceVertices'])
    spec = build(sampleMatrix=sparse.csr_matrix((0, n)))
    assert not spec['supportedMask'].any()
    assert not len(spec['controlVertexIds'])
    assert 20 in spec['hardFixedVertexIds'] and 9 not in spec['hardFixedVertexIds']
    assert np.isfinite(evaluateSupportedShapeConstraints(spec['referenceVertices'], spec)['values']).all()


def testInputsRemainUnchangedAndOutputDoesNotAliasThem():
    data = fixture(); saved = copy.deepcopy(data)
    spec = buildSupportedShapeConstraintSpec(**data)
    evaluateSupportedShapeConstraints(data['referenceVertices'], spec)
    for key, value in data.items():
        if sparse.issparse(value):
            assert (value-saved[key]).nnz == 0
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, saved[key])
    spec['referenceVertices'][0] += 1
    np.testing.assert_array_equal(data['referenceVertices'], saved['referenceVertices'])


@pytest.mark.parametrize('change,match', [
    ({'lengthScale': 0}, 'positive'),
    ({'lengthScale': True}, 'positive'),
    ({'sampleMatrix': np.eye(21)}, 'sparse'),
    ({'sampleMatrix': sparse.eye(20)}, 'sparse'),
    ({'staticVertexIds': np.array([0, 0])}, 'unique'),
    ({'delegatedVertexIds': np.array([21])}, 'valid'),
    ({'protectedVertexIds': np.array([1.])}, 'integer'),
    ({'policy': {'notAPolicy': 2}}, 'Unknown'),
    ({'policy': {'minimumEdgeRatio': 1.1}}, 'inclusive'),
    ({'policy': {'maximumAreaRatio': .9}}, 'inclusive'),
    ({'policy': {'minimumNormalCosine': 1.01}}, 'Cosine'),
    ({'policy': {'maximumProtectedDisplacement': .6}}, 'no-looser'),
])
def testInvalidContractIsRejected(change, match):
    with pytest.raises(ValueError, match=match):
        build(**change)


@pytest.mark.parametrize('kind', ['negative', 'nonaffine', 'nan'])
def testInvalidSamplingNeverSilentlyNormalizes(kind):
    c = fixture()['sampleMatrix'].copy()
    c.data[0] = {'negative': -1., 'nonaffine': .9, 'nan': np.nan}[kind]
    with pytest.raises(ValueError, match='sampleMatrix'):
        build(sampleMatrix=c)


def testExactOriginalStaticFrameAndInheritedProtectedBudgetAreRequired():
    data = fixture(); safety = data['referenceVertices'].copy()
    data['referenceVertices'] = safety.copy(); data['referenceVertices'][0, 0] += 1e-12
    with pytest.raises(ValueError, match='static coordinates'):
        buildSupportedShapeConstraintSpec(**data, safetyReferenceVertices=safety)
    data['referenceVertices'] = safety.copy(); data['referenceVertices'][10, 2] += .11
    with pytest.raises(ValueError, match='displacement limits'):
        buildSupportedShapeConstraintSpec(**data, safetyReferenceVertices=safety)
    data['referenceVertices'] = safety.copy(); data['referenceVertices'][20, 2] += .51
    with pytest.raises(ValueError, match='displacement limits'):
        buildSupportedShapeConstraintSpec(**data, safetyReferenceVertices=safety)


def testDegenerateGuardedTriangleAndInvalidCandidateFailExplicitly():
    spec = build()
    triangle = spec['faceVertexIds'][0]
    current = spec['referenceVertices'].copy(); current[triangle[1]] = current[triangle[0]]
    with pytest.raises(ValueError, match='positive finite|nonzero area'):
        evaluateSupportedShapeConstraints(current, spec)
    bad = spec['referenceVertices'].copy(); bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        evaluateSupportedShapeConstraints(bad, spec)
    with pytest.raises(ValueError, match='boolean'):
        evaluateSupportedShapeConstraints(spec['referenceVertices'], spec, 1)


def testShapeConstraintsRejectReferenceRelativeStretchWithoutChangingTopology():
    spec = build()
    candidate = spec['referenceVertices'].copy()
    candidate[spec['transitionVertexIds'][0], 0] += .5
    result = evaluateSupportedShapeConstraints(candidate, spec)
    edgeEnd = result['groups']['edgeUpper']['stop']
    assert result['values'][:edgeEnd].min() < 0
    np.testing.assert_array_equal(spec['faces'], fixture()['faces'])
