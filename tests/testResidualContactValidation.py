"""Prospective controls for residual eye fitting and contact-plane validation."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('residualContactValidation', ROOT / 'tools/validateResidualContactExperiment.py')
AUDIT = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(AUDIT)


def coarseFixture():
    vertices = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [2., 0., 1.], [3., 0., 1.], [2., 1., 1.]])
    weights = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [.5, .5, 0.], [.5, 0., .5], [0., .5, .5]])
    constraints = {}
    for name, triangle in zip(AUDIT.COARSE_EYES, ([0, 1, 2], [3, 4, 5])):
        ids = np.tile(triangle, (6, 1))
        points = (vertices[ids] * weights[..., None]).sum(1)
        constraints[name] = dict(name=name, vertexIndices=ids, barycentric=weights.copy(), targetPixels=points[:, :2].copy())
    camera = dict(intrinsics=np.eye(3), rotation=np.eye(3), translation=np.zeros(3))
    return vertices, constraints, camera


def maskFixture():
    x = np.linspace(10, 20, 6)
    photos = {name: dict(upper=np.c_[x, np.full(6, 10)], lower=np.c_[x, np.full(6, 20)]) for name in AUDIT.EYES}
    baseline, candidate = {'eyes': {}}, {'eyes': {}}
    for name in AUDIT.EYES:
        old = np.zeros((32, 32), bool); old[10:21, 12:23] = True
        new = np.zeros((32, 32), bool); new[10:21, 10:21] = True
        def measurement(mask):
            return dict(roi=dict(origin=[0., 0.], scale=1., shape=[32, 32]), roiMasks=dict(eye=mask), status='accepted', reasons=[])
        baseline['eyes'][name] = measurement(old); candidate['eyes'][name] = measurement(new)
    return baseline, candidate, photos


def contactFixture():
    vertices = np.array([[0., 0., .1], [-1., -1., 0.], [1., -1., 0.], [0., 1., 0.]])
    return vertices, dict(sourceVertexIds=np.array([0]), targetVertexIds=np.array([[1, 2, 3]]),
        targetBarycentric=np.array([[.25, .25, .5]]), referenceNormals=np.array([[0., 0., 1.]]),
        minimumGap=0., lengthScale=1., tolerance=1e-8)


class ResidualContactValidationTests(unittest.TestCase):
    def testExactCompletedCoarseBaselinePasses(self):
        vertices, constraints, camera = coarseFixture()
        self.assertTrue(AUDIT.coarseEyePreservationAudit(vertices, vertices, constraints, camera, imageScale=1.)['accepted'])

    def testMeanPreservedCannotHideOneBadCoarsePoint(self):
        vertices, constraints, camera = coarseFixture(); candidate = vertices.copy(); candidate[0, 0] += .1
        report = AUDIT.coarseEyePreservationAudit(vertices, candidate, constraints, camera, imageScale=1.)
        self.assertTrue(report['eyes']['imageLeftEye']['checks']['meanNotWorse'])
        self.assertFalse(report['eyes']['imageLeftEye']['checks']['maximumNotWorse'])
        self.assertFalse(report['accepted'])

    def testCoarseAllowanceCannotAccumulateAcrossResidualRounds(self):
        vertices, constraints, camera = coarseFixture()
        first = vertices.copy(); first[:, 0] += .04
        second = vertices.copy(); second[:, 0] += .08
        self.assertTrue(AUDIT.coarseEyePreservationAudit(vertices, first, constraints, camera, imageScale=1.)['accepted'])
        self.assertFalse(AUDIT.coarseEyePreservationAudit(vertices, second, constraints, camera, imageScale=1.)['accepted'])

    def testOneEyeCannotStandInForBothEyes(self):
        vertices, constraints, camera = coarseFixture(); del constraints['imageRightEye']
        with self.assertRaises(ValueError): AUDIT.coarseEyePreservationAudit(vertices, vertices, constraints, camera, imageScale=1.)

    def testLegacySideBudgetIsNotResetAtCompletedBaseline(self):
        vertices, constraints, camera = coarseFixture()
        entries = list(constraints.values())
        binding = dict(ibugIds=np.arange(36, 48), vertexIndices=np.concatenate([entry['vertexIndices'] for entry in entries]),
            barycentricWeights=np.concatenate([entry['barycentric'] for entry in entries]))
        photo = np.zeros((68, 2)); photo[36:48] = np.concatenate([entry['targetPixels'] for entry in entries])
        observations = dict(points=np.stack([photo, photo]), accepted=np.ones(2, bool), confidence=np.ones((2, 68)))
        completed = vertices.copy(); completed[:, 0] += .7
        candidate = vertices.copy(); candidate[:, 0] += 1.1
        result = AUDIT.legacyResidualObservationAudit(vertices, completed, candidate, np.array([[0, 1, 2], [3, 4, 5]]),
            binding, [camera, camera], [dict(workingSize=[512, 512])]*2, observations, 0,
            masks=[np.ones(12, bool), np.ones(12, bool)])
        self.assertFalse(result['side'][0]['accepted'])
        self.assertAlmostEqual(result['side'][0]['changeAgainstCompletedBaselinePixels'], .4)
        self.assertFalse(result['side'][0]['baselineAlreadyExceedsOriginalBudget'])

    def testUnchangedBaselineCanStillFailHistoricalFullGate(self):
        baseline = {(0, 1), (1, 2)}
        anchors = dict(originalPureDirect={(0, 1)}, currentBounded=baseline, completedCoarseBaseline=baseline)
        report = AUDIT.historicalCollisionAudit(baseline, baseline, anchors)
        self.assertTrue(report['incrementalNoNewPairs']); self.assertFalse(report['allHistoricalAnchorsAccepted'])
        self.assertEqual(report['anchors']['originalPureDirect']['inheritedPairsStillPresent'], [(1, 2)])

    def testRepairingOldPairDoesNotPayForNewPair(self):
        baseline = {(0, 1), (1, 2)}; candidate = {(0, 1), (2, 3)}
        anchors = dict(originalPureDirect={(0, 1)}, currentBounded=baseline, completedCoarseBaseline=baseline)
        report = AUDIT.historicalCollisionAudit(baseline, candidate, anchors)
        self.assertFalse(report['incrementalNoNewPairs'])
        self.assertEqual(report['anchors']['originalPureDirect']['repairedHistoricalPairs'], [(1, 2)])

    def testHistoricalDefectsCanBeActuallyRemoved(self):
        baseline = {(0, 1), (1, 2)}; candidate = {(0, 1)}
        anchors = dict(originalPureDirect=candidate, currentBounded=baseline, completedCoarseBaseline=baseline)
        self.assertTrue(AUDIT.historicalCollisionAudit(baseline, candidate, anchors)['allHistoricalAnchorsAccepted'])

    def testAllThreeCompleteAnchorsRemainRequired(self):
        with self.assertRaises(ValueError): AUDIT.historicalCollisionAudit(set(), set(), {})

    def testCompleteMaskImprovementPassesOriginalGate(self):
        before, after, photos = maskFixture()
        self.assertTrue(AUDIT.fullApertureRasterAudit(before, after, photos, originalPixelScale=1.)['accepted'])

    def testFragmentedBaselineStillHasDefinedMaskError(self):
        before, after, photos = maskFixture()
        for name in AUDIT.EYES:
            before['eyes'][name]['status'] = 'rejected'; before['eyes'][name]['reasons'] = ['eye_fragmented_or_too_small']
            before['eyes'][name]['roiMasks']['eye'][2, 2] = True
        report = AUDIT.fullApertureRasterAudit(before, after, photos, originalPixelScale=1.)
        self.assertTrue(report['effectDefined']); self.assertTrue(report['accepted'])

    def testPerfectProxyCannotWaiveRejectedFullContour(self):
        before, after, photos = maskFixture(); after['eyes']['imageRight']['status'] = 'rejected'
        report = AUDIT.fullApertureRasterAudit(before, after, photos, originalPixelScale=1.)
        self.assertTrue(report['originalApertureGate']['accepted']); self.assertFalse(report['accepted'])

    def testEmptyBaselineDoesNotInventHundredPercentBenefit(self):
        before, after, photos = maskFixture(); before['eyes']['imageLeft']['roiMasks']['eye'][:] = False
        report = AUDIT.fullApertureRasterAudit(before, after, photos, originalPixelScale=1.)
        self.assertFalse(report['effectDefined']); self.assertFalse(report['accepted'])

    def testChangedRasterCannotCountAsGeometryImprovement(self):
        before, after, photos = maskFixture(); after['eyes']['imageLeft']['roi']['origin'][0] = .001
        report = AUDIT.fullApertureRasterAudit(before, after, photos, originalPixelScale=1.)
        self.assertFalse(report['accepted']); self.assertIn('imageLeft:changedRasterIdentity', report['reasons'])

    def testNewApertureFragmentIsStillRejected(self):
        before, after, photos = maskFixture(); after['eyes']['imageRight']['roiMasks']['eye'][2, 2] = True
        self.assertFalse(AUDIT.fullApertureRasterAudit(before, after, photos, originalPixelScale=1.)['accepted'])

    def testContactGapEvaluatesBothMovingSurfaces(self):
        before, arguments = contactFixture(); after = before.copy(); after[1:, 2] += .2
        report = AUDIT.frozenContactGapAudit(before, after, **arguments)
        self.assertFalse(report['accepted']); self.assertAlmostEqual(report['candidateGap'][0], -.1)

    def testInheritedNegativeGapCanImproveButCannotWorsen(self):
        before, arguments = contactFixture(); before[0, 2] = -.1
        better = before.copy(); better[0, 2] = -.05
        worse = before.copy(); worse[0, 2] = -.11
        self.assertTrue(AUDIT.frozenContactGapAudit(before, better, **arguments)['accepted'])
        self.assertFalse(AUDIT.frozenContactGapAudit(before, worse, **arguments)['accepted'])

    def testContactGapJointRigidMotionIsInvariant(self):
        before, arguments = contactFixture(); after = before.copy(); after[0, 2] -= .05
        first = AUDIT.frozenContactGapAudit(before, after, **arguments)
        rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        transformed = {**arguments, 'referenceNormals': arguments['referenceNormals'] @ rotation.T}
        second = AUDIT.frozenContactGapAudit(before @ rotation.T + [3, 5, 2], after @ rotation.T + [3, 5, 2], **transformed)
        np.testing.assert_allclose(first['candidateGap'], second['candidateGap'], atol=1e-14)
        self.assertEqual(first['accepted'], second['accepted'])

    def testEmptyContactDomainCannotBeCalledContactEvidence(self):
        before, arguments = contactFixture()
        arguments.update(sourceVertexIds=np.empty(0, np.int64), targetVertexIds=np.empty((0, 3), np.int64),
            targetBarycentric=np.empty((0, 3)), referenceNormals=np.empty((0, 3)))
        result = AUDIT.frozenContactGapAudit(before, before, **arguments)
        self.assertTrue(result['accepted']); self.assertTrue(result['emptyContactDomain'])

    def testContactNormalMustBeUnitAndNotSilentlyFlipped(self):
        before, arguments = contactFixture(); arguments['referenceNormals'] *= 2
        with self.assertRaises(ValueError): AUDIT.frozenContactGapAudit(before, before, **arguments)

    def testContactSpecRebuildsNearestFeatureAndBothSurfaceOperator(self):
        from faceAppearance.surfaceContactConstraints import buildSurfaceContactSpec
        vertices, _ = contactFixture(); faces = np.array([[1, 2, 3]])
        groups = [dict(name='templateSurface', sourceVertexIds=np.array([0]), targetFaceIds=np.array([0]),
            normalSign=1, orientationProvenance='synthetic outward +Z target winding')]
        policy = dict(lengthScale=1., searchRadius=2., activationDistance=.2, minimumGap=0., weight=1.)
        spec = buildSurfaceContactSpec(vertices, faces, groups, **policy)
        report = AUDIT.contactSpecAudit(spec, vertices, faces, groups, policy, savedOperator=spec['gapOperator'])
        self.assertTrue(report['validationPassed'])
        unilateral = spec['gapOperator'].copy().tolil(); unilateral[:, 3:] = 0
        self.assertFalse(AUDIT.contactSpecAudit(spec, vertices, faces, groups, policy, savedOperator=unilateral.tocsr())['validationPassed'])

    def testContactSpecDoesNotAcceptFlippedNormalOrResetLowerBound(self):
        from faceAppearance.surfaceContactConstraints import buildSurfaceContactSpec
        vertices, _ = contactFixture(); vertices[0, 2] = -.1; faces = np.array([[1, 2, 3]])
        groups = [dict(name='templateSurface', sourceVertexIds=np.array([0]), targetFaceIds=np.array([0]),
            normalSign=1, orientationProvenance='synthetic outward +Z target winding')]
        policy = dict(lengthScale=1., searchRadius=2., activationDistance=.2, minimumGap=0., weight=1.)
        spec = buildSurfaceContactSpec(vertices, faces, groups, **policy)
        spec['lowerGaps'][:] = 0
        self.assertFalse(AUDIT.contactSpecAudit(spec, vertices, faces, groups, policy, savedOperator=spec['gapOperator'])['validationPassed'])
        spec = buildSurfaceContactSpec(vertices, faces, groups, **policy)
        spec['selectionEvidence'][0]['nearestNormals'] *= -1
        self.assertFalse(AUDIT.contactSpecAudit(spec, vertices, faces, groups, policy, savedOperator=spec['gapOperator'])['validationPassed'])

    def testClosestFeatureProofIncludesLongTrianglesNotJustNearCentroids(self):
        point = np.array([0., 0., .1])
        triangles = np.array([[[-.1, -.1, 0.], [100., -.1, 0.], [-.1, 100., 0.]],
                              [[-1., -1., -1.], [1., -1., -1.], [0., 1., -1.]]])
        bary, distances, _ = AUDIT._closestFeatureIndependent(point, triangles)
        self.assertEqual(int(np.argmin(distances)), 0)
        self.assertAlmostEqual(distances[0], .1)
        np.testing.assert_allclose(bary[0] @ triangles[0], [0, 0, 0], atol=1e-14)

    def testAccessoryCannotBecomeMovableThroughEyeCornerException(self):
        vertices, _, _ = coarseFixture()
        scene = dict(vertices=vertices, faces=np.array([[0, 1, 2], [3, 4, 5]]),
            uvs=np.zeros((3, 2)), uvFaces=np.array([[0, 1, 2], [0, 1, 2]]), materialIds=np.array([0, 0]), nicpToFullIds=np.arange(3))
        mask = np.array([True, False, False, True, False, False])
        with self.assertRaises(ValueError): AUDIT.frozenAssemblyAudit(scene, copy.deepcopy(scene), mask)

    def testRemoteSkinAndNonSkinAreExactlyFixed(self):
        vertices, _, _ = coarseFixture()
        scene = dict(vertices=vertices, faces=np.array([[0, 1, 2], [3, 4, 5]]),
            uvs=np.zeros((3, 2)), uvFaces=np.array([[0, 1, 2], [0, 1, 2]]), materialIds=np.array([0, 2]), nicpToFullIds=np.arange(3))
        after = copy.deepcopy(scene); after['vertices'][1, 0] = np.nextafter(after['vertices'][1, 0], np.inf)
        report = AUDIT.frozenAssemblyAudit(scene, after, np.array([True, False, False, False, False, False]))
        self.assertFalse(report['accepted'])

    def testJsonEvidenceKeysDoNotChangeAlphaMaterialIdentity(self):
        self.assertTrue(AUDIT.sameEvidence({3: np.array([0., 1.])}, {'3': np.array([0., 1.])}))
        self.assertFalse(AUDIT.sameEvidence({3: np.array([0., 1.])}, {'2': np.array([0., 1.])}))
        self.assertFalse(AUDIT.sameEvidence({3: 0, '3': 1}, {'3': 1}))

    def testPerRoundMaskDecisionDoesNotInventFinalCumulativeTolerance(self):
        def score(left, right):
            return dict(meanSymmetricBoundaryDistanceOriginalPixels=(left + right) / 2,
                eyes={name: dict(symmetricBoundaryMeanOriginalPixels=value,
                    symmetricBoundaryP95OriginalPixels=value, intersectionOverUnion=.8,
                    visibleComponentCount=1, secondaryComponentAreaFraction=0.)
                    for name, value in zip(AUDIT.EYES, (left, right))})
        start, middle, final = score(1., 3.), score(1.1, 2.5), score(1.2, 2.)
        rounds = [AUDIT.apertureMetricAudit(a, b)['accepted'] for a, b in ((start, middle), (middle, final))]
        self.assertEqual(rounds, [True, True])
        self.assertFalse(AUDIT.apertureMetricAudit(start, final)['accepted'])
        self.assertTrue(AUDIT.residualAcceptanceFromRounds(True, True, 'acceptedLocalExperiment', rounds))
        self.assertFalse(AUDIT.residualAcceptanceFromRounds(False, True, 'acceptedLocalExperiment', rounds))
        self.assertFalse(AUDIT.residualAcceptanceFromRounds(True, False, 'noImprovingSafeCandidate', []))

    def testFullGeometryGateDoesNotIgnoreAccessoryVertex(self):
        vertices, _, camera = coarseFixture()
        scene = dict(faces=np.array([[0, 1, 2], [3, 4, 5]]), nicpToFullIds=np.arange(3))
        candidate = vertices.copy(); candidate[4, 0] += .001
        gate = AUDIT.fullLocalGeometryAudit(vertices, candidate, scene, np.arange(3, 6), camera)
        self.assertFalse(gate['accepted']); self.assertFalse(gate['checks']['exactlyFixed'])

    def testFullGeometryGateRetainsDepthConstraint(self):
        vertices, _, camera = coarseFixture()
        scene = dict(faces=np.array([[0, 1, 2], [3, 4, 5]]), nicpToFullIds=np.arange(6))
        candidate = vertices.copy(); candidate[0, 2] += .001
        gate = AUDIT.fullLocalGeometryAudit(vertices, candidate, scene, np.empty(0, int), camera)
        self.assertFalse(gate['accepted']); self.assertFalse(gate['checks']['referenceDepth'])

    def testExportCannotRewriteUnchangedObjLineForRoundoff(self):
        from types import SimpleNamespace
        vertices, _, _ = coarseFixture()
        mesh = dict(vertices=vertices / 100, faces=np.array([[0, 1, 2], [3, 4, 5]]),
            uvs=np.zeros((3, 2)), uvFaces=np.array([[0, 1, 2], [0, 1, 2]]), materialIds=np.array([0, 2]))
        obj = ''.join('v ' + ' '.join(str(x) for x in row) + '\n' for row in mesh['vertices']).encode()
        source = SimpleNamespace(mesh=mesh, payloads={'face_nicp.obj': obj, 'head_color.png': b'original'},
            manifest={'binding': {'vertexOrderPreserved': True}, 'materials': []})
        candidate = copy.deepcopy(source); candidate.payloads['face_nicp.obj'] = obj.replace(b'v ', b'v  ', 1)
        raw = dict(rotation=np.eye(3), translation=np.zeros(3), scale=1.)
        canonical = mesh['vertices'] * 100
        identity = dict(changedVertexMask=np.zeros(6, bool), canonicalVertices=canonical)
        gate = AUDIT.exportIdentityAudit(source, candidate, identity, canonical, canonical, np.zeros(6, bool), raw, {'changedVertexCount': 0})
        self.assertFalse(gate['validationPassed']); self.assertFalse(gate['checks']['fixedAndNonVertexObjLinesExact'])


if __name__ == '__main__': unittest.main()
