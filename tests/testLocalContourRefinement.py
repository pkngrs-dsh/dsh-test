"""Behavioral contracts for one frozen-contour geometry proposal."""
import copy

import numpy as np
import pytest

from faceAppearance.camera import projectPoints
from faceAppearance.localContourRefinement import refineLocalContours


def mesh(xs=None, ys=None):
    xs = np.linspace(-1., 1., 9) if xs is None else np.asarray(xs)
    ys = np.linspace(-1., 1., 9) if ys is None else np.asarray(ys)
    xx, yy = np.meshgrid(xs, ys)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    faces = []
    for row in range(len(ys)-1):
        for column in range(len(xs)-1):
            i = row*len(xs)+column
            faces.extend([[i, i+1, i+len(xs)], [i+1, i+len(xs)+1, i+len(xs)]])
    return vertices, np.asarray(faces)


def cameras():
    angle = .4
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                         [-np.sin(angle), 0., np.cos(angle)]])
    return [{'intrinsics': np.array([[300., 0., 256.], [0., 300., 256.], [0., 0., 1.]]),
             'rotation': r, 'translation': np.array([0., 0., 5.])} for r in (np.eye(3), rotation)]


def contour(indices, targetPoints, camera, view=0, normal=(1., 0.)):
    return {'viewIndex': view, 'vertexIndices': np.asarray(indices)[:, None],
            'barycentric': np.ones((len(indices), 1)),
            'targetPixels': projectPoints(targetPoints, **camera)[0],
            'normalPixels': np.tile(normal, (len(indices), 1)), 'weights': np.ones(len(indices))}


def problem():
    vertices, faces = mesh()
    radial = np.linalg.norm(vertices[:, :2], axis=1)
    active = radial < .9
    ids = np.flatnonzero(active)
    # Analytic smooth field independent of the solver basis.
    target = vertices[ids]+(1-(radial[ids]/.9)**2)[:, None]**2*[.012, -.008, .006]
    views = cameras()
    groups = [contour(ids, target, camera, view, normal) for view, camera in enumerate(views)
              for normal in ((1., 0.), (0., 1.))]
    return {'vertices': vertices, 'faces': faces, 'cameras': views, 'constraints': groups,
            'activeMask': active, 'maxDisplacement': .05, 'controlCount': 24, 'controlRadius': .4,
            'displacementWeight': 1e-7, 'strainWeight': 1e-6}


def assertUnchanged(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assertUnchanged(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for item, previous in zip(actual, expected):
            assertUnchanged(item, previous)
    else:
        np.testing.assert_array_equal(actual, expected)


def testRecoverKnownSmoothPerspectiveDeformationWithoutChangingInputs():
    args = problem()
    snapshot = copy.deepcopy(args)
    proposal, report = refineLocalContours(**args)
    ids = args['constraints'][0]['vertexIndices'][:, 0]
    radial = np.linalg.norm(args['vertices'][ids, :2], axis=1)
    expected = args['vertices'][ids]+(1-(radial/.9)**2)[:, None]**2*[.012, -.008, .006]
    assert np.linalg.norm(proposal[ids]-expected, axis=1).mean() < .0005
    for group in report['contourGroups']:
        assert group['proposal']['weightedNormalRmsPixels'] < group['before']['weightedNormalRmsPixels']*.05
    assert report['linearizedSolveCount'] == 1
    assert report['qualityAccepted'] is False
    assert report['correspondencesUpdated'] is False
    assert report['scalesFrozen'] is True
    assert report['fixedVerticesExactlyUnchanged'] is True
    np.testing.assert_array_equal(proposal[~args['activeMask']], args['vertices'][~args['activeMask']])
    assertUnchanged(args, snapshot)


def testTangentialTargetMotionAndNormalSignHaveNoInfluence():
    args = problem()
    args['constraints'] = args['constraints'][:1]
    first, _ = refineLocalContours(**args)
    args['constraints'][0]['targetPixels'][:, 1] += np.linspace(-90., 140., len(args['constraints'][0]['weights']))
    args['constraints'][0]['normalPixels'] *= -17
    second, report = refineLocalContours(**args)
    np.testing.assert_allclose(second, first, atol=2e-11)
    group = report['contourGroups'][0]
    assert group['proposal']['weightedNormalRmsPixels'] < group['before']['weightedNormalRmsPixels']


def testUniformCapPreservesTheEntireFieldInsteadOfClippingVertices():
    args = problem()
    first, initial = refineLocalContours(**args)
    assert initial['boundScale'] == 1
    for group in args['constraints']:
        points = args['vertices'][group['vertexIndices'][:, 0]]
        before = projectPoints(points, **args['cameras'][group['viewIndex']])[0]
        group['targetPixels'] = before+20*(group['targetPixels']-before)
    second, report = refineLocalContours(**args)
    assert 0 < report['boundScale'] < 1
    np.testing.assert_allclose(second-args['vertices'],
                               (first-args['vertices'])*20*report['boundScale'], atol=1e-12)
    assert np.linalg.norm(second-args['vertices'], axis=1).max() == pytest.approx(args['maxDisplacement'])


def testPhotoResolutionMeshUnitsAndGlobalPhotoWeightsAreInvariant():
    args = problem()
    first, _ = refineLocalContours(**args)
    scaled = copy.deepcopy(args)
    for name in ('vertices', 'maxDisplacement', 'controlRadius'):
        scaled[name] *= 1000
    for camera in scaled['cameras']:
        camera['translation'] *= 1000
        camera['intrinsics'][:2] *= 2
    for group in scaled['constraints']:
        group['targetPixels'] *= 2
        group['weights'] *= 19
    second, _ = refineLocalContours(**scaled)
    np.testing.assert_allclose(second/1000, first, atol=1e-11)


def testDepthWeightChangesReferenceStrengthAndZeroDisablesIt():
    args = problem()
    ids = args['constraints'][0]['vertexIndices'][:, 0]
    baseline, _ = refineLocalContours(**args)
    target = args['vertices'][ids].copy()
    target[:, 2] = -.01*(1-(np.linalg.norm(target[:, :2], axis=1)/.9)**2)**2
    args['depthConstraints'] = {'vertexIndices': ids[:, None], 'barycentric': np.ones((len(ids), 1)),
        'targetPoints': target, 'targetNormals': np.tile([0., 0., 1.], (len(ids), 1)), 'weights': np.ones(len(ids))}
    disabled, zeroReport = refineLocalContours(**args, depthWeight=0.)
    strong, strongReport = refineLocalContours(**args, depthWeight=100.)
    np.testing.assert_array_equal(disabled, baseline)
    assert strong[40, 2] < -.009
    assert strongReport['depth']['proposal']['weightedPlaneRms'] < zeroReport['depth']['proposal']['weightedPlaneRms']*.05
    scaled = copy.deepcopy(args)
    scaled['depthConstraints']['weights'] *= 23
    scaled['depthConstraints']['targetNormals'] *= -7
    equivalent, _ = refineLocalContours(**scaled, depthWeight=100.)
    np.testing.assert_allclose(equivalent, strong, atol=1e-12)


def testVeryShortEdgeCoMovesDespiteOpposingPhotoTargets():
    vertices, faces = mesh(xs=[-1., -.5, 0., 1e-5, .5, 1.], ys=np.linspace(-1., 1., 5))
    active = (np.abs(vertices[:, 0]) < .9) & (np.abs(vertices[:, 1]) < .9)
    ids = np.flatnonzero(active)
    target = vertices[ids]+[.01, 0., 0.]
    target[np.flatnonzero(ids == 14)[0], 0] = .04
    target[np.flatnonzero(ids == 15)[0], 0] = -.02
    camera = cameras()[0]
    proposal, report = refineLocalContours(vertices, faces, [camera], [contour(ids, target, camera)],
        activeMask=active, maxDisplacement=.05, controlCount=6, controlRadius=.6)
    delta = proposal-vertices
    assert np.linalg.norm(delta[14]-delta[15]) < 1e-6
    assert delta[[14, 15], 0].mean() > .001
    assert report['maximumRelativeEdgeLengthChange'] < .05
    np.testing.assert_array_equal(proposal[~active], vertices[~active])


def testTriangleBarycentricSamplesAreEvaluatedOnOriginalTopology():
    args = problem()
    triangles = args['faces'][args['activeMask'][args['faces']].all(1)]
    bary = np.tile([.2, .3, .5], (len(triangles), 1))
    points = (args['vertices'][triangles]*bary[..., None]).sum(1)
    group = contour(np.arange(len(points)), points+[.003, 0., 0.], args['cameras'][0])
    group.update(vertexIndices=triangles, barycentric=bary)
    args['constraints'] = [group]
    proposal, report = refineLocalContours(**args)
    np.testing.assert_array_equal(report['contourGroups'][0]['proposal']['points'],
                                  (proposal[triangles]*bary[..., None]).sum(1))
    assert report['contourGroups'][0]['proposal']['weightedNormalRmsPixels'] < .03


@pytest.mark.parametrize('damage', [
    'emptyConstraints', 'nanVertices', 'nanPixels', 'zeroNormals', 'nanNormals', 'negativeWeights',
    'zeroWeights', 'nanWeights', 'floatIndices', 'outOfRangeIndices', 'negativeBary', 'wrongBarySum',
    'wrongPixelsShape', 'missingField', 'booleanView', 'badView', 'noActive', 'softActive',
    'badRadius', 'badCap', 'booleanControls', 'nanDisplacementWeight', 'negativeStrainWeight',
    'negativeDepthWeight', 'nanCamera', 'improperCamera', 'nonpositiveCameraDepth',
])
def testInvalidInputsAreRejected(damage):
    args = problem()
    group = args['constraints'][0]
    if damage == 'emptyConstraints': args['constraints'] = []
    elif damage == 'nanVertices': args['vertices'][0, 0] = np.nan
    elif damage == 'nanPixels': group['targetPixels'][0, 0] = np.inf
    elif damage == 'zeroNormals': group['normalPixels'][0] = 0
    elif damage == 'nanNormals': group['normalPixels'][0, 0] = np.nan
    elif damage == 'negativeWeights': group['weights'][0] = -1
    elif damage == 'zeroWeights': group['weights'][:] = 0
    elif damage == 'nanWeights': group['weights'][0] = np.nan
    elif damage == 'floatIndices': group['vertexIndices'] = group['vertexIndices'].astype(float)
    elif damage == 'outOfRangeIndices': group['vertexIndices'][0] = len(args['vertices'])
    elif damage == 'negativeBary': group['barycentric'][0] = -1
    elif damage == 'wrongBarySum': group['barycentric'][0] = .5
    elif damage == 'wrongPixelsShape': group['targetPixels'] = group['targetPixels'][:, :1]
    elif damage == 'missingField': del group['normalPixels']
    elif damage == 'booleanView': group['viewIndex'] = True
    elif damage == 'badView': group['viewIndex'] = 2
    elif damage == 'noActive': args['activeMask'][:] = False
    elif damage == 'softActive': args['activeMask'] = args['activeMask'].astype(float)
    elif damage == 'badRadius': args['controlRadius'] = 0
    elif damage == 'badCap': args['maxDisplacement'] = np.inf
    elif damage == 'booleanControls': args['controlCount'] = False
    elif damage == 'nanDisplacementWeight': args['displacementWeight'] = np.nan
    elif damage == 'negativeStrainWeight': args['strainWeight'] = -1
    elif damage == 'negativeDepthWeight': args['depthWeight'] = -1
    elif damage == 'nanCamera': args['cameras'][0]['intrinsics'][0, 0] = np.nan
    elif damage == 'improperCamera': args['cameras'][0]['rotation'][0, 0] = 2
    elif damage == 'nonpositiveCameraDepth': args['cameras'][0]['translation'][2] = 0
    with pytest.raises(ValueError):
        refineLocalContours(**args)


def testInactiveBarycentricSupportIsRejectedEvenWhenAnIndexIsActive():
    args = problem()
    activeId, inactiveId = int(np.flatnonzero(args['activeMask'])[0]), int(np.flatnonzero(~args['activeMask'])[0])
    group = contour([inactiveId], args['vertices'][[inactiveId]], args['cameras'][0])
    group.update(vertexIndices=np.array([[activeId, inactiveId]]), barycentric=np.array([[0., 1.]]))
    args['constraints'] = [group]
    with pytest.raises(ValueError, match='no movable surface support'):
        refineLocalContours(**args)


def testAlreadyMatchingContoursLeaveEveryVertexExactlyUnchanged():
    args = problem()
    for group in args['constraints']:
        group['targetPixels'] = projectPoints(args['vertices'][group['vertexIndices'][:, 0]],
                                             **args['cameras'][group['viewIndex']])[0]
    proposal, report = refineLocalContours(**args)
    np.testing.assert_array_equal(proposal, args['vertices'])
    assert report['maximumDisplacement'] == 0


@pytest.mark.parametrize('damage', ['nanTargets', 'zeroNormals', 'negativeWeights', 'wrongBarySum', 'noMovableSupport'])
def testDepthAnchorsAreValidatedIndependently(damage):
    args = problem()
    ids = args['constraints'][0]['vertexIndices'][:, 0]
    depth = {'vertexIndices': ids[:, None], 'barycentric': np.ones((len(ids), 1)),
             'targetPoints': args['vertices'][ids].copy(),
             'targetNormals': np.tile([0., 0., 1.], (len(ids), 1)), 'weights': np.ones(len(ids))}
    if damage == 'nanTargets': depth['targetPoints'][0, 0] = np.nan
    elif damage == 'zeroNormals': depth['targetNormals'][0] = 0
    elif damage == 'negativeWeights': depth['weights'][0] = -1
    elif damage == 'wrongBarySum': depth['barycentric'][0] = .5
    elif damage == 'noMovableSupport': depth['vertexIndices'][0] = np.flatnonzero(~args['activeMask'])[0]
    with pytest.raises(ValueError):
        refineLocalContours(**args, depthConstraints=depth)
