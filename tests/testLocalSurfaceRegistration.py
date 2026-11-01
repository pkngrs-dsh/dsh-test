"""Independent contracts for a local, frozen-correspondence surface solve."""
import copy

import numpy as np
import pytest

from faceAppearance.localSurfaceRegistration import solveLocalSurfaceIncrement


def mesh(size=9):
    yy, xx = np.meshgrid(np.linspace(-1., 1., size), np.linspace(-1., 1., size), indexing='ij')
    vertices = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(size*size)])
    faces = []
    for y in range(size-1):
        for x in range(size-1):
            i = y*size+x
            faces.extend([[i, i+1, i+size], [i+1, i+size+1, i+size]])
    return vertices, np.asarray(faces)


def problem():
    vertices, faces = mesh()
    radius = np.linalg.norm(vertices[:, :2], axis=1)
    active = radius < .9
    ids = np.flatnonzero(active)
    # An analytic smooth bump, independent of the implementation's RBF basis.
    envelope = (1-(radius[ids]/.9)**2)**2
    desired = envelope[:, None]*[.012, -.008, .02]
    return {
        'vertices': vertices, 'faces': faces, 'sampleVertexIndices': ids[:, None],
        'sampleBarycentric': np.ones((len(ids), 1)), 'targetPoints': vertices[ids]+desired,
        'targetNormals': np.tile([0., 0., 1.], (len(ids), 1)), 'weights': np.ones(len(ids)),
        'activeMask': active, 'maxDisplacement': .05, 'controlRadius': .4, 'controlCount': 24,
        'displacementWeight': 1e-6, 'strainWeight': 1e-5,
    }


def testRecoverKnownLowFrequencyThreeDimensionalShapeAndFreezeInputs():
    args = problem()
    snapshot = copy.deepcopy(args)
    proposal, diagnostics = solveLocalSurfaceIncrement(**args)
    ids = args['sampleVertexIndices'][:, 0]
    error = proposal[ids]-args['targetPoints']
    assert np.linalg.norm(error, axis=1).mean() < .0005
    assert np.max(np.abs(error), axis=0).max() < .001
    assert diagnostics['proposal']['weightedNormalRms'] < diagnostics['before']['weightedNormalRms']*.05
    assert diagnostics['proposal']['weightedTangentRms'] < diagnostics['before']['weightedTangentRms']*.05
    np.testing.assert_array_equal(proposal[~args['activeMask']], args['vertices'][~args['activeMask']])
    assert diagnostics['fixedVerticesExactlyUnchanged']
    assert diagnostics['linearizedSolveCount'] == 1
    assert diagnostics['correspondencesUpdated'] is False
    assert diagnostics['qualityAccepted'] is False
    for name, value in snapshot.items():
        np.testing.assert_array_equal(args[name], value)


def testUniformCapPreservesTheWholeFieldDirection():
    args = problem()
    first, firstDiagnostics = solveLocalSurfaceIncrement(**args)
    assert firstDiagnostics['boundScale'] == 1
    ids = args['sampleVertexIndices'][:, 0]
    args['targetPoints'] = args['vertices'][ids] + 10*(args['targetPoints']-args['vertices'][ids])
    second, secondDiagnostics = solveLocalSurfaceIncrement(**args)
    assert 0 < secondDiagnostics['boundScale'] < 1
    np.testing.assert_allclose(second-args['vertices'],
        (first-args['vertices'])*10*secondDiagnostics['boundScale'], atol=1e-12)
    assert np.linalg.norm(second-args['vertices'], axis=1).max() == pytest.approx(args['maxDisplacement'])


def testLengthUnitAndNormalMagnitudeDoNotChangeThePhysicalSolution():
    args = problem()
    original, _ = solveLocalSurfaceIncrement(**args)
    factor = 1000.
    scaled = copy.deepcopy(args)
    for name in ('vertices', 'targetPoints', 'maxDisplacement', 'controlRadius'):
        scaled[name] = scaled[name]*factor
    scaled['targetNormals'] *= 7
    scaled['weights'] *= 19
    converted, _ = solveLocalSurfaceIncrement(**scaled)
    np.testing.assert_allclose(converted/factor, original, atol=1e-11)


def testVeryShortEdgeMovesCoherentlyDespiteConflictingPointTargets():
    vertices, faces = mesh(5)
    vertices[13] = [1e-5, 0., 0.]
    active = np.zeros((5, 5), bool)
    active[1:4, 1:4] = True
    active = active.ravel()
    ids = np.flatnonzero(active)
    targets = vertices[ids]+[0., 0., .01]
    targets[np.flatnonzero(ids == 12)[0], 2] = .04
    targets[np.flatnonzero(ids == 13)[0], 2] = -.02
    proposal, report = solveLocalSurfaceIncrement(vertices, faces, ids[:, None], np.ones((len(ids), 1)),
        targets, np.tile([0., 0., 1.], (len(ids), 1)), np.ones(len(ids)), activeMask=active,
        maxDisplacement=.05, controlRadius=.6, controlCount=6)
    delta = proposal-vertices
    assert np.linalg.norm(delta[12]-delta[13]) < 1e-5
    assert delta[[12, 13], 2].mean() > .001
    np.testing.assert_array_equal(proposal[~active], vertices[~active])
    assert report['maximumRelativeEdgeLengthChange'] < .05


def testConvexTriangleSamplesFollowTheirOriginalSurfaceIndices():
    args = problem()
    active = args['activeMask']
    triangles = args['faces'][active[args['faces']].all(1)]
    bary = np.tile([.2, .3, .5], (len(triangles), 1))
    points = (args['vertices'][triangles]*bary[..., None]).sum(1)
    args.update(sampleVertexIndices=triangles, sampleBarycentric=bary,
        targetPoints=points+[0., 0., .006], targetNormals=np.tile([0., 0., 1.], (len(points), 1)),
        weights=np.ones(len(points)))
    proposal, report = solveLocalSurfaceIncrement(**args)
    expected = (proposal[triangles]*bary[..., None]).sum(1)
    np.testing.assert_array_equal(report['proposalPoints'], expected)
    assert report['proposal']['weightedNormalRms'] < report['before']['weightedNormalRms']*.15


@pytest.mark.parametrize('damage', [
    'nanVertices', 'nanTargets', 'nanNormals', 'zeroNormals', 'nanWeights', 'negativeWeights',
    'allZeroWeights', 'floatIndices', 'outOfBoundsIndices', 'negativeBary', 'wrongBarySum',
    'wrongTargetShape', 'noActive', 'softActive', 'invalidRadius', 'invalidCap',
    'booleanControls', 'negativeNormalWeight', 'negativeTangentWeight', 'nanStrainWeight',
])
def testInvalidGeometryCorrespondenceOrSettingsAreRejected(damage):
    args = problem()
    if damage == 'nanVertices': args['vertices'][0, 0] = np.nan
    elif damage == 'nanTargets': args['targetPoints'][0, 0] = np.nan
    elif damage == 'nanNormals': args['targetNormals'][0, 0] = np.inf
    elif damage == 'zeroNormals': args['targetNormals'][0] = 0
    elif damage == 'nanWeights': args['weights'][0] = np.nan
    elif damage == 'negativeWeights': args['weights'][0] = -1
    elif damage == 'allZeroWeights': args['weights'][:] = 0
    elif damage == 'floatIndices': args['sampleVertexIndices'] = args['sampleVertexIndices'].astype(float)
    elif damage == 'outOfBoundsIndices': args['sampleVertexIndices'][0] = len(args['vertices'])
    elif damage == 'negativeBary': args['sampleBarycentric'][0] = -1
    elif damage == 'wrongBarySum': args['sampleBarycentric'][0] = .5
    elif damage == 'wrongTargetShape': args['targetPoints'] = args['targetPoints'][:, :2]
    elif damage == 'noActive': args['activeMask'][:] = False
    elif damage == 'softActive': args['activeMask'] = args['activeMask'].astype(float)
    elif damage == 'invalidRadius': args['controlRadius'] = np.inf
    elif damage == 'invalidCap': args['maxDisplacement'] = 0
    elif damage == 'booleanControls': args['controlCount'] = True
    elif damage == 'negativeNormalWeight': args['normalWeight'] = -1
    elif damage == 'negativeTangentWeight': args['tangentWeight'] = -1
    elif damage == 'nanStrainWeight': args['strainWeight'] = np.nan
    with pytest.raises(ValueError):
        solveLocalSurfaceIncrement(**args)


def testSampleWithNoActiveBarycentricSupportIsRejectedEvenIfAnIndexIsActive():
    args = problem()
    activeId = int(np.flatnonzero(args['activeMask'])[0])
    inactiveId = int(np.flatnonzero(~args['activeMask'])[0])
    args.update(sampleVertexIndices=np.array([[activeId, inactiveId]]),
        sampleBarycentric=np.array([[0., 1.]]), targetPoints=np.array([[0., 0., 0.]]),
        targetNormals=np.array([[0., 0., 1.]]), weights=np.ones(1))
    with pytest.raises(ValueError, match='no movable surface support'):
        solveLocalSurfaceIncrement(**args)


def testZeroTargetProducesExactUnchangedVertices():
    args = problem()
    args['targetPoints'] = args['vertices'][args['sampleVertexIndices'][:, 0]].copy()
    proposal, report = solveLocalSurfaceIncrement(**args)
    np.testing.assert_array_equal(proposal, args['vertices'])
    assert report['maximumDisplacement'] == 0
    assert report['proposal']['weightedPointRms'] == 0
