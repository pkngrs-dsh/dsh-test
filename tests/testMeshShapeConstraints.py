"""Analytic Jacobian and invariant guard tests on synthetic mesh patches."""
from copy import deepcopy

import numpy as np
import pytest
from scipy import sparse

from faceAppearance.meshShapeConstraints import buildMeshShapeGuardSpec, evaluateMeshShapeConstraints


def problem():
    yy, xx = np.meshgrid(np.linspace(-1., 1., 7), np.linspace(-1., 1., 7), indexing='ij')
    rest = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(49)])
    faces = []
    for row in range(6):
        for column in range(6):
            i = row*7+column
            faces.extend([[i, i+1, i+7], [i+1, i+8, i+7]])
    grid = np.arange(49).reshape(7, 7)
    interior = grid[2:5, 2:5].ravel()
    rim = np.setdiff1d(grid[1:6, 1:6].ravel(), interior)
    rest[interior, 2] = [.06, .12, .07, .13, .2, .11, .08, .1, .05]
    baseline = rest.copy(); baseline[interior, 2] *= .4
    baseline[interior, :2] += np.random.default_rng(17).normal(size=(len(interior), 2))*.012
    depth = np.zeros((len(interior), baseline.size))
    depth[np.arange(len(interior)), 3*interior+2] = 1.
    depth[:, 3*rim+2] = -1/len(rim)
    lower = np.minimum(np.minimum(depth@baseline.ravel(), depth@rest.ravel()), 0.)
    upper = np.maximum(np.maximum(depth@baseline.ravel(), depth@rest.ravel()), 0.)
    return {'baselineVertices': baseline, 'restVertices': rest, 'faces': np.asarray(faces),
        'interiorVertexIds': interior, 'rimVertexIds': rim, 'lengthScale': .2,
        'depthOperator': sparse.csr_matrix(depth), 'depthLower': lower, 'depthUpper': upper}


def rows(result, name):
    group = result['groups'][name]
    return slice(group['start'], group['stop'])


def testAnalyticJacobianMatchesCenteredFiniteDifferencesForEveryConstraintAndCoordinate():
    args = problem(); spec = buildMeshShapeGuardSpec(**args)
    candidate = args['baselineVertices'] + np.random.default_rng(53).normal(size=(49, 3))*.009
    analytic = evaluateMeshShapeConstraints(candidate, spec)
    assert sparse.isspmatrix_csr(analytic['jacobian'])
    step = args['lengthScale']*2e-6
    numeric = np.empty(analytic['jacobian'].shape)
    for column in range(candidate.size):
        plus, minus = candidate.copy(), candidate.copy()
        plus.ravel()[column] += step; minus.ravel()[column] -= step
        numeric[:, column] = (evaluateMeshShapeConstraints(plus, spec, False)['values']
            - evaluateMeshShapeConstraints(minus, spec, False)['values'])/(2*step)
    np.testing.assert_allclose(analytic['jacobian'].toarray(), numeric, atol=3e-8, rtol=3e-5)
    for name in analytic['groups']:
        assert np.linalg.norm(numeric[rows(analytic, name)]) > 1e-5
    assert analytic['jacobian'].shape[1] == candidate.size
    assert set(analytic['groups']) == {'edgeLower', 'edgeUpper', 'areaLower', 'areaUpper',
        'triangleQuality', 'normalToBaseline', 'normalToRest', 'depthLower', 'depthUpper', 'controlDisplacement'}


def testBaselineFeasibleAndBoundsMatchIndependentGeometry():
    args = problem(); spec = buildMeshShapeGuardSpec(**args)
    evaluation = evaluateMeshShapeConstraints(args['baselineVertices'], spec)
    assert evaluation['values'].min() >= -1e-12
    assert 'jacobian' not in evaluateMeshShapeConstraints(args['baselineVertices'], spec, False)
    interior = set(args['interiorVertexIds'].tolist())
    expectedFaces = [index for index, face in enumerate(args['faces']) if any(int(v) in interior for v in face)]
    np.testing.assert_array_equal(spec['faceIds'], expectedFaces)
    expectedEdges = sorted({tuple(sorted([int(a), int(b)])) for face in args['faces'] for a, b in
        [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] if int(a) in interior or int(b) in interior})
    np.testing.assert_array_equal(spec['edgeVertexIds'], expectedEdges)
    for index, (a, b) in enumerate(expectedEdges):
        old = np.linalg.norm(args['baselineVertices'][a]-args['baselineVertices'][b])
        rest = np.linalg.norm(args['restVertices'][a]-args['restVertices'][b])
        assert spec['edgeLower'][index] == pytest.approx(min(old, .25*rest))
        assert spec['edgeUpper'][index] == pytest.approx(max(old, 2*rest))
    for index, face in enumerate(args['faces'][expectedFaces]):
        values = []
        for vertices in (args['baselineVertices'], args['restVertices']):
            a, b, c = vertices[face]
            area = np.linalg.norm(np.cross(b-a, c-a))*.5
            quality = 4*np.sqrt(3)*area/(np.sum((b-a)**2)+np.sum((c-b)**2)+np.sum((a-c)**2))
            values.append((area, quality))
        assert spec['areaLower'][index] == pytest.approx(min(values[0][0], .1*values[1][0]))
        assert spec['areaUpper'][index] == pytest.approx(max(values[0][0], 3*values[1][0]))
        assert spec['qualityLower'][index] == pytest.approx(min(values[0][1], .4*values[1][1]))
    protected = np.setdiff1d(np.arange(49), spec['controlVertexIds'])
    protectedCoordinates = (3*protected[:, None]+np.arange(3)).ravel()
    assert evaluation['jacobian'][:, protectedCoordinates].nnz == 0


@pytest.mark.parametrize('scale', [.001, 1., 1000.])
def testSimilarityChangesJacobianFrameButNotValuesOrFeasibility(scale):
    args = problem(); original = buildMeshShapeGuardSpec(**args)
    trial = args['baselineVertices'] + np.random.default_rng(31).normal(size=(49, 3))*.008
    first = evaluateMeshShapeConstraints(trial, original)
    rotation, _ = np.linalg.qr(np.random.default_rng(41).normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0: rotation[:, 0] *= -1
    translation = np.array([1.8, -.3, .7])*scale
    transformed = deepcopy(args)
    for key in ('baselineVertices', 'restVertices'):
        transformed[key] = args[key]@rotation*scale+translation
    transformed['lengthScale'] *= scale
    transformed['depthOperator'] = (args['depthOperator']@sparse.kron(sparse.eye(49), rotation)).tocsr()
    shift = transformed['depthOperator']@np.tile(translation, 49)
    transformed['depthLower'] = args['depthLower']*scale+shift
    transformed['depthUpper'] = args['depthUpper']*scale+shift
    second = evaluateMeshShapeConstraints(trial@rotation*scale+translation, buildMeshShapeGuardSpec(**transformed))
    np.testing.assert_allclose(second['values'], first['values'], atol=1e-12, rtol=1e-11)
    expectedJacobian = (first['jacobian'].toarray().reshape(-1, 49, 3)@rotation/scale).reshape(first['jacobian'].shape)
    np.testing.assert_allclose(second['jacobian'].toarray(), expectedJacobian, atol=2e-10/scale, rtol=1e-10)


def testVertexRenumberingAndInteriorOrderCannotChangeConstraintSets():
    args = problem(); original = buildMeshShapeGuardSpec(**args)
    rng = np.random.default_rng(67)
    trial = args['baselineVertices'] + rng.normal(size=(49, 3))*.007
    first = evaluateMeshShapeConstraints(trial, original)
    order = rng.permutation(49); inverse = np.argsort(order)
    changed = deepcopy(args)
    for key in ('baselineVertices', 'restVertices'): changed[key] = args[key][order]
    changed['faces'] = inverse[args['faces']]
    changed['interiorVertexIds'] = inverse[args['interiorVertexIds'][::-1]]
    changed['rimVertexIds'] = inverse[args['rimVertexIds'][::-1]]
    columns = (3*order[:, None]+np.arange(3)).ravel()
    changed['depthOperator'] = args['depthOperator'][:, columns]
    secondSpec = buildMeshShapeGuardSpec(**changed)
    second = evaluateMeshShapeConstraints(trial[order], secondSpec)
    for group in first['groups']:
        np.testing.assert_allclose(np.sort(second['values'][rows(second, group)]),
                                   np.sort(first['values'][rows(first, group)]), atol=1e-13)
    # Compare actual row identities as well as values, including the Jacobian.
    edgeLookup = {tuple(edge): i for i, edge in enumerate(original['edgeVertexIds'])}
    edgeOrder = np.array([edgeLookup[tuple(sorted(order[edge]))] for edge in secondSpec['edgeVertexIds']])
    controlLookup = {int(v): i for i, v in enumerate(original['controlVertexIds'])}
    controlOrder = np.array([controlLookup[int(v)] for v in order[secondSpec['controlVertexIds']]])
    for group in first['groups']:
        sourceRows = np.arange(first['groups'][group]['start'], first['groups'][group]['stop'])
        if group.startswith('edge'): sourceRows = sourceRows[edgeOrder]
        if group == 'controlDisplacement': sourceRows = sourceRows[controlOrder]
        np.testing.assert_allclose(second['jacobian'][rows(second, group)].toarray(),
            first['jacobian'][sourceRows][:, columns].toarray(), atol=1e-12)


def thinTriangleProblem(reverseBaseline=False):
    rest = np.array([[0., 0., 0.], [1., 0., 0.], [.5, .8, 0.]])
    baseline = np.array([[0., 0., 0.], [1., 0., 0.], [.999, .0001, 0.]])
    if reverseBaseline: baseline[:, 1] *= -1
    depth = sparse.csr_matrix(([-.5, -.5, 1.], ([0, 0, 0], [2, 5, 8])), shape=(1, 9))
    return {'baselineVertices': baseline, 'restVertices': rest, 'faces': np.array([[0, 1, 2]]),
        'interiorVertexIds': np.array([2]), 'rimVertexIds': np.array([0, 1]), 'lengthScale': .1,
        'depthOperator': depth, 'depthLower': np.array([0.]), 'depthUpper': np.array([0.])}


@pytest.mark.parametrize('reverseBaseline', [False, True])
def testGrandfatheredThinOrReversedBaselineIsFeasibleButCannotFurtherCollapse(reverseBaseline):
    args = thinTriangleProblem(reverseBaseline)
    spec = buildMeshShapeGuardSpec(**args)
    first = evaluateMeshShapeConstraints(args['baselineVertices'], spec)
    assert first['values'].min() >= -1e-12
    assert spec['areaLower'][0] == pytest.approx(.00005)
    assert spec['restNormalCosineLower'][0] == (-1 if reverseBaseline else 0)
    collapsed = args['baselineVertices'].copy(); collapsed[2, 1] *= .5
    worse = evaluateMeshShapeConstraints(collapsed, spec)
    assert worse['values'][rows(worse, 'areaLower')][0] < 0
    assert worse['values'][rows(worse, 'triangleQuality')][0] < 0
    assert np.isfinite(worse['jacobian'].data).all()


def testThinFaceAreaAndEdgeJacobiansRetainPhysicalGradientScale():
    args = thinTriangleProblem(); spec = buildMeshShapeGuardSpec(**args)
    result = evaluateMeshShapeConstraints(args['baselineVertices'], spec)
    areaGradient = result['jacobian'][rows(result, 'areaLower')].toarray()[0]
    # d(area / L²)/d(vertex2.y) = half base length / L².
    assert areaGradient[7] == pytest.approx(.5/.1**2)
    edgeGradients = result['jacobian'][rows(result, 'edgeLower')].toarray()
    assert np.linalg.norm(edgeGradients[0]) == pytest.approx(np.sqrt(2)/.1)


def testInteriorAndRimDisplacementCapsApplyToAllControlsUniformly():
    args = problem(); spec = buildMeshShapeGuardSpec(**args)
    for key, cap in [('interiorVertexIds', 2.), ('rimVertexIds', .5)]:
        for vertex in args[key]:
            moved = args['baselineVertices'].copy()
            # Rigidly translating the full patch avoids a degenerate face when
            # probing each cap; every row still has its independent limit.
            moved[spec['controlVertexIds'], 2] += args['lengthScale']*(cap+1e-3)
            result = evaluateMeshShapeConstraints(moved, spec)
            row = int(np.flatnonzero(spec['controlVertexIds'] == vertex)[0])
            assert result['values'][rows(result, 'controlDisplacement')][row] < 0


def testInputArraysAndSparseDepthAreNotMutatedOrAliased():
    args = problem(); snapshot = deepcopy(args)
    spec = buildMeshShapeGuardSpec(**args)
    evaluateMeshShapeConstraints(args['baselineVertices'], spec)
    for key in args:
        if sparse.issparse(args[key]): np.testing.assert_array_equal(args[key].toarray(), snapshot[key].toarray())
        else: np.testing.assert_array_equal(args[key], snapshot[key])
    spec['baselineVertices'][:] = 0
    spec['depthOperator'].data[:] = 0
    np.testing.assert_array_equal(args['baselineVertices'], snapshot['baselineVertices'])
    np.testing.assert_array_equal(args['depthOperator'].toarray(), snapshot['depthOperator'].toarray())


@pytest.mark.parametrize('damage', ['nanVertex', 'wrongShape', 'zeroArea'])
def testInvalidCandidateGeometryIsRejectedWithoutInventingNormals(damage):
    args = problem(); spec = buildMeshShapeGuardSpec(**args)
    candidate = args['baselineVertices'].copy()
    if damage == 'nanVertex': candidate[0, 0] = np.nan
    elif damage == 'wrongShape': candidate = candidate[:-1]
    else:
        face = spec['faceVertexIds'][0]
        candidate[face] = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    for withJacobian in (False, True):
        with pytest.raises(ValueError): evaluateMeshShapeConstraints(candidate, spec, withJacobian)


@pytest.mark.parametrize('damage', ['overlap', 'incompleteRim', 'badFace', 'duplicateIds', 'zeroScale',
    'nonfiniteScale', 'unknownPolicy', 'reversedPolicy', 'badQuality', 'zeroLimit', 'denseDepth',
    'outsideDepth', 'nanDepth', 'wrongDepthShape', 'reversedDepthBounds', 'baselineOutsideDepth', 'nanRest'])
def testInvalidGuardSpecificationsAreRejected(damage):
    args = problem()
    if damage == 'overlap': args['rimVertexIds'][0] = args['interiorVertexIds'][0]
    elif damage == 'incompleteRim': args['rimVertexIds'] = args['rimVertexIds'][1:]
    elif damage == 'badFace': args['faces'][0, 0] = 49
    elif damage == 'duplicateIds': args['interiorVertexIds'][0] = args['interiorVertexIds'][1]
    elif damage == 'zeroScale': args['lengthScale'] = 0.
    elif damage == 'nonfiniteScale': args['lengthScale'] = np.inf
    elif damage == 'unknownPolicy': args['policy'] = {'excludedVertexIds': [0]}
    elif damage == 'reversedPolicy': args['policy'] = {'minimumRestEdgeRatio': 3., 'maximumRestEdgeRatio': 2.}
    elif damage == 'badQuality': args['policy'] = {'minimumRestQualityRatio': 1.1}
    elif damage == 'zeroLimit': args['policy'] = {'rimDisplacementLimit': 0.}
    elif damage == 'denseDepth': args['depthOperator'] = args['depthOperator'].toarray()
    elif damage == 'outsideDepth':
        args['depthOperator'] = args['depthOperator'].tolil(); args['depthOperator'][0, 0] = 1.; args['depthOperator'] = args['depthOperator'].tocsr()
    elif damage == 'nanDepth': args['depthOperator'].data[0] = np.nan
    elif damage == 'wrongDepthShape': args['depthLower'] = args['depthLower'][:, None]
    elif damage == 'reversedDepthBounds': args['depthLower'] = args['depthUpper']+1
    elif damage == 'baselineOutsideDepth': args['depthLower'] = args['depthUpper']*.99
    elif damage == 'nanRest': args['restVertices'][0, 0] = np.nan
    with pytest.raises(ValueError): buildMeshShapeGuardSpec(**args)
