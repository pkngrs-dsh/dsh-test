"""Photo-only observation mapping, independent of fitted geometry and gates."""
from copy import deepcopy

import numpy as np
import pytest

from faceAppearance.eyeObservationAlignment import (
    AXIS_TOLERANCE, MAX_CURVE_POINTS, alignCoarseEyeTargets,
    alignEyeLandmarkTargets,
)


def observations():
    x = np.array([0., 2., 5., 8., 10.])
    source = {}
    points = []
    for name, offset in (('imageLeft', 0.), ('imageRight', 20.)):
        source[name] = dict(upper=np.column_stack([x+offset, [0., -2., -3., -2., 0.]]),
            lower=np.column_stack([x+offset, [0., 1., 2., 1., 0.]]))
        points.extend(np.array([[-.2, .3], [3., -.4], [7., -.6],
            [10.2, -.3], [7., .7], [3., .5]])+[offset, 0.])
    return source, np.asarray(points)


def coarseGroups(points, subset=False):
    groups = []
    for name, ids in (('imageLeftEye', [41, 36, 38, 40] if subset else [41, 39, 37, 40, 36, 38]),
                      ('imageRightEye', [45, 44, 42, 47] if subset else [43, 46, 45, 47, 44, 42])):
        ids = np.asarray(ids, np.int32); count = len(ids)
        groups.append(dict(name=name, ibugIds=ids, targetPixels=points[ids-36].copy(),
            vertexIndices=np.arange(count*3, dtype=np.int32).reshape(count, 3),
            barycentric=np.tile([.2, .3, .5], (count, 1)), weights=np.linspace(.5, 1., count),
            confidence=np.linspace(0., 1., count), opaqueMetadata=dict(untouched=[name, 123])))
    return groups[::-1]


def assertSame(first, second):
    if isinstance(first, np.ndarray):
        assert isinstance(second, np.ndarray) and first.dtype == second.dtype
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first: assertSame(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert type(first) is type(second) and len(first) == len(second)
        for left, right in zip(first, second): assertSame(left, right)
    else:
        assert first == second


def testInteriorProjectsOnlyNormalAndCornersAreExplicitException():
    source, original = observations()
    table = alignEyeLandmarkTargets(source, original)
    expected = original.copy()
    for offset in (0, 6):
        expected[offset:offset+6, 1] = [0., -7/3, -7/3, 0., 4/3, 4/3]
        expected[offset, 0] += .2; expected[offset+3, 0] -= .2
    np.testing.assert_allclose(table['targetPixels'], expected, atol=1e-14)
    np.testing.assert_array_equal(table['originalTargetPixels'], original)
    np.testing.assert_array_equal(table['ibugIds'], np.arange(36, 48))
    for point in table['evidence']['points']:
        row = point['ibugId']-36
        if point['role'] == 'canthus':
            assert point['canthusAlongException'] and abs(point['alongDeltaPixels']) > .19
        else:
            assert not point['canthusAlongException']
            assert point['originalAlong'] == point['targetAlong'] and point['alongDeltaPixels'] == 0
            np.testing.assert_array_equal(original[row, 0], table['targetPixels'][row, 0])
            lid = source[point['eye']][point['role']]
            first, second = lid[point['sourceSegment']]; fraction = point['segmentFraction']
            np.testing.assert_allclose(table['targetPixels'][row], first*(1-fraction)+second*fraction, atol=1e-14)
    assert table['evidence']['geometryUsed'] is False
    assert table['evidence']['candidateProjectionUsed'] is False
    assert table['evidence']['sourceSampleOrderChanged'] is False


def testInputsAreUnchangedAndAllReturnedArraysAreIndependent():
    source, original = observations(); groups = coarseGroups(original, subset=True)
    sourceBefore, originalBefore, groupsBefore = deepcopy(source), original.copy(), deepcopy(groups)
    table = alignEyeLandmarkTargets(source, original)
    aligned = alignCoarseEyeTargets(source, groups, mappedTable=table)
    assertSame(source, sourceBefore); assertSame(original, originalBefore); assertSame(groups, groupsBefore)
    assertSame(table, alignEyeLandmarkTargets(source, original))
    aligned['coarseGroups'][0]['vertexIndices'][0, 0] = 999
    aligned['coarseGroups'][0]['targetPixels'][:] = 999
    aligned['mappedTable']['targetPixels'][:] = 999
    assertSame(groups, groupsBefore)
    assertSame(table, alignEyeLandmarkTargets(source, original))
    table['evidence']['sourcePhotoEyes']['imageLeft']['upper'][:] = 999
    table['originalTargetPixels'][:] = 999
    assertSame(source, sourceBefore); assertSame(original, originalBefore)


def testShuffledSubsetGroupsReuseOneCompleteTableAndPreserveEveryOtherField():
    source, original = observations(); groups = coarseGroups(original, subset=True)
    table = alignEyeLandmarkTargets(source, original)
    result = alignCoarseEyeTargets(source, groups, mappedTable=table)
    assert result['evidence']['tableSha256'] == table['tableSha256']
    assert result['evidence']['fieldsChanged'] == ['targetPixels']
    for before, after, reference in zip(groups, result['coarseGroups'], result['evidence']['groups']):
        for key in before:
            if key != 'targetPixels': assertSame(before[key], after[key])
        rows = before['ibugIds']-36
        np.testing.assert_array_equal(after['targetPixels'], table['targetPixels'][rows])
        np.testing.assert_array_equal(reference['tableRows'], rows)
    complete = alignCoarseEyeTargets(source, coarseGroups(original))
    fromOriginal = alignCoarseEyeTargets(source, groups, eyeLandmarks=original)
    assertSame(complete['mappedTable'], table)
    assertSame(fromOriginal, result)


@pytest.mark.parametrize('angle', [0., .3, 2.1, -2.6])
@pytest.mark.parametrize('scale', [1e-6, 1., 1e6])
def testProperSimilarityEquivariance(angle, scale):
    source, original = observations(); table = alignEyeLandmarkTargets(source, original)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    translation = scale*np.array([15.3, -27.1])
    transform = lambda value: scale*value @ rotation.T+translation
    movedSource = {name: {lid: transform(curve) for lid, curve in eye.items()} for name, eye in source.items()}
    moved = alignEyeLandmarkTargets(movedSource, transform(original))
    np.testing.assert_allclose(moved['targetPixels'], transform(table['targetPixels']), rtol=2e-14, atol=scale*2e-13)
    for first, second in zip(table['evidence']['points'], moved['evidence']['points']):
        assert first['role'] == second['role']
        np.testing.assert_allclose(second['originalAlong'], first['originalAlong'], atol=2e-14)
        np.testing.assert_allclose(second['normalDeltaPixels'], first['normalDeltaPixels']*scale, atol=scale*2e-13)


def testDifferentDenseEndpointPositionsStillUseTheirMidpoint():
    source, original = observations()
    for eye in source.values():
        eye['upper'][0] += [-.1, -.2]; eye['lower'][0] += [.1, .2]
        eye['upper'][-1] += [-.1, -.2]; eye['lower'][-1] += [.1, .2]
    table = alignEyeLandmarkTargets(source, original)
    for name, rows in (('imageLeft', (0, 3)), ('imageRight', (6, 9))):
        expected = .5*source[name]['upper'][[0, -1]]+.5*source[name]['lower'][[0, -1]]
        np.testing.assert_array_equal(table['targetPixels'][list(rows)], expected)
        assert (table['evidence']['axes'][name]['endpointGapPixels'] > 0).all()


@pytest.mark.parametrize('change', ['upperReverse', 'bothReverse', 'swapLids', 'duplicateAxis', 'backfold', 'cross', 'interiorTouch', 'degenerateAxis'])
def testMalformedDenseSemanticsAreRejectedWithoutSortingOrRepair(change):
    source, original = observations(); eye = source['imageLeft']
    if change == 'upperReverse': eye['upper'] = eye['upper'][::-1].copy()
    elif change == 'bothReverse':
        for lid in eye: eye[lid] = eye[lid][::-1].copy()
    elif change == 'swapLids': eye['upper'], eye['lower'] = eye['lower'], eye['upper']
    elif change == 'duplicateAxis': eye['upper'][2, 0] = eye['upper'][1, 0]
    elif change == 'backfold': eye['upper'][2, 0] = eye['upper'][1, 0]-.1
    elif change == 'cross': eye['upper'][2, 1] = 2.1
    elif change == 'interiorTouch': eye['upper'][2, 1] = eye['lower'][2, 1]
    elif change == 'degenerateAxis':
        for lid in eye: eye[lid][-1] = eye[lid][0]
    before = deepcopy(source)
    with pytest.raises(ValueError): alignEyeLandmarkTargets(source, original)
    assertSame(source, before)


def testUnionKnotsDetectCrossingWithDifferentSampling():
    source, original = observations()
    # Upper samples alone miss the lower V reaching above the upper curve.
    source['imageLeft']['lower'] = np.array([[0., 0.], [2., 1.], [3., -3.], [5., 2.], [8., 1.], [10., 0.]])
    with pytest.raises(ValueError, match='intersecting'): alignEyeLandmarkTargets(source, original)


@pytest.mark.parametrize('s', [-.01, 0., 1., 1.01])
def testOutsideDenseCanthusDomainIsRejectedWithoutClamping(s):
    source, original = observations()
    original[1, 0] = 10*s; original[0, 0] = -20.; original[3, 0] = 20.
    if s >= 1: original[2, 0] = 10*s+.1
    with pytest.raises(ValueError, match='no clamping'): alignEyeLandmarkTargets(source, original)


def testActualLidDomainAlsoLimitsInteriorPoints():
    source, original = observations()
    source['imageLeft']['upper'][0, 0] = .2
    source['imageLeft']['lower'][0, 0] = -.2
    original[1, 0] = .1
    with pytest.raises(ValueError, match='no clamping'): alignEyeLandmarkTargets(source, original)


def testOriginalIBugOrderIsSemanticRatherThanAnUnorderedSet():
    source, original = observations(); original[[1, 2]] = original[[2, 1]]
    with pytest.raises(ValueError, match='original iBUG point order'): alignEyeLandmarkTargets(source, original)


def testTableAuthenticationRejectsStaleOrModifiedSourcesTargetsAndEvidence():
    source, original = observations(); groups = coarseGroups(original, subset=True)
    table = alignEyeLandmarkTargets(source, original)
    for key in ('targetPixels', 'tableSha256', 'evidence'):
        stale = deepcopy(table)
        if key == 'targetPixels': stale[key][1, 1] += .01
        elif key == 'tableSha256': stale[key] = '0'*64
        else: stale[key]['geometryUsed'] = True
        with pytest.raises(ValueError, match='stale'): alignCoarseEyeTargets(source, groups, mappedTable=stale)
    changedSource = deepcopy(source); changedSource['imageLeft']['upper'][1, 1] -= .1
    with pytest.raises(ValueError, match='stale'): alignCoarseEyeTargets(changedSource, groups, mappedTable=table)
    groups[0]['targetPixels'][0, 0] += .001
    with pytest.raises(ValueError, match='same original'): alignCoarseEyeTargets(source, groups, mappedTable=table)


def testMissingFullTableCannotBeSilentlyInferredFromSparseSubset():
    source, original = observations(); groups = coarseGroups(original, subset=True)
    with pytest.raises(ValueError, match='full 12'): alignCoarseEyeTargets(source, groups)
    with pytest.raises(ValueError, match='either'):
        alignCoarseEyeTargets(source, groups, mappedTable=alignEyeLandmarkTargets(source, original), eyeLandmarks=original)


@pytest.mark.parametrize('change', ['wrongShape', 'nonFinite', 'complex', 'missingEye', 'tooShort', 'overBudget'])
def testInvalidPhotoInputsAreRejected(change):
    source, original = observations()
    if change == 'wrongShape': original = original[:11]
    elif change == 'nonFinite': source['imageLeft']['upper'][2, 0] = np.nan
    elif change == 'complex': original = original.astype(complex)
    elif change == 'missingEye': source.pop('imageRight')
    elif change == 'tooShort': source['imageLeft']['upper'] = source['imageLeft']['upper'][:2]
    elif change == 'overBudget': source['imageLeft']['upper'] = np.zeros((MAX_CURVE_POINTS+1, 2))
    with pytest.raises(ValueError): alignEyeLandmarkTargets(source, original)


@pytest.mark.parametrize('change', ['missingTargets', 'wrongName', 'duplicateGroup', 'duplicateId', 'wrongEyeId', 'floatId', 'zeroWeight', 'nanConfidence', 'largeConfidence', 'unpairedBinding', 'invalidBary', 'negativeIndex'])
def testInvalidGroupsCannotChangeObservationSelection(change):
    source, original = observations(); groups = coarseGroups(original)
    group = groups[0]
    if change == 'missingTargets': group.pop('targetPixels')
    elif change == 'wrongName': group['name'] = 'anatomicLeft'
    elif change == 'duplicateGroup': groups[1] = deepcopy(group)
    elif change == 'duplicateId': group['ibugIds'][0] = group['ibugIds'][1]
    elif change == 'wrongEyeId': group['ibugIds'][0] = 36
    elif change == 'floatId': group['ibugIds'] = group['ibugIds'].astype(float)
    elif change == 'zeroWeight': group['weights'][0] = 0
    elif change == 'nanConfidence': group['confidence'][0] = np.nan
    elif change == 'largeConfidence': group['confidence'][0] = 1.01
    elif change == 'unpairedBinding': group.pop('barycentric')
    elif change == 'invalidBary': group['barycentric'][0] = [0., 0., 0.]
    elif change == 'negativeIndex': group['vertexIndices'][0, 0] = -1
    with pytest.raises(ValueError): alignCoarseEyeTargets(source, groups, eyeLandmarks=original)


def testToleranceIsOneFixedDimensionlessArithmeticConstant():
    source, original = observations(); table = alignEyeLandmarkTargets(source, original)
    assert AXIS_TOLERANCE == 64*np.finfo(np.float64).eps
    assert table['evidence']['monotonicityToleranceInEyeWidths'] == AXIS_TOLERANCE
    assert table['evidence']['upperLowerSeparationToleranceInEyeWidths'] == AXIS_TOLERANCE
