"""Acceptance-regression tests; no real case fitting or candidate selection."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('featureValidation', ROOT / 'tools/validateFeatureGeometryExperiment.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def scene():
    return dict(vertices=np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]]),
        faces=np.array([[0, 1, 2]]), uvs=np.array([[0., 0.], [1., 0.], [0., 1.]]),
        uvFaces=np.array([[0, 1, 2]]), materialIds=np.array([0]), nicpToFullIds=np.arange(3))


def aperture(mean=2.):
    row = dict(symmetricBoundaryMeanOriginalPixels=mean, symmetricBoundaryP95OriginalPixels=mean + 1,
        intersectionOverUnion=.8, visibleComponentCount=1, secondaryComponentAreaFraction=0.)
    return dict(meanSymmetricBoundaryDistanceOriginalPixels=mean,
        eyes=dict(imageLeft=row.copy(), imageRight=row.copy()))


class FeatureGeometryValidationTests(unittest.TestCase):
    def testNestedEvidencePreservesDenseArraysWithoutPickle(self):
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / 'evidence'
            values = np.arange(12).reshape(2, 2, 3)
            np.savez_compressed(stem.with_suffix('.npz'), **{'root/masks/0': values})
            stem.with_suffix('.json').write_text(json.dumps({'masks': [
                {'arrayKey': 'root/masks/0', 'shape': [2, 2, 3], 'dtype': str(values.dtype)}]}))
            np.testing.assert_array_equal(AUDIT.readEvidence(stem)['masks'][0], values)

    def testEvidenceMetadataMismatchIsNotSilentlyLoaded(self):
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / 'evidence'
            np.savez_compressed(stem.with_suffix('.npz'), x=np.ones(3))
            stem.with_suffix('.json').write_text(json.dumps({'arrayKey': 'x', 'shape': [4], 'dtype': 'float64'}))
            with self.assertRaisesRegex(ValueError, 'metadata mismatch'):
                AUDIT.readEvidence(stem)

    def testUnreferencedEvidenceIsRejected(self):
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / 'evidence'
            np.savez_compressed(stem.with_suffix('.npz'), hidden=np.ones(3))
            stem.with_suffix('.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'unreferenced'):
                AUDIT.readEvidence(stem)

    def testLowerTotalCollisionCountCanStillIntroduceNewPair(self):
        result = AUDIT.intersectionSetAudit({(0, 1), (2, 3)}, {(0, 4)})
        self.assertFalse(result['accepted'])
        self.assertEqual(result['newPairs'], [(0, 4)])
        self.assertEqual(len(result['removedPairs']), 2)

    def testInheritedPairsAreReportedWithoutBeingCalledNew(self):
        result = AUDIT.intersectionSetAudit({(0, 1), (2, 3)}, {(1, 0)})
        self.assertTrue(result['accepted'])
        self.assertEqual(result['inheritedPairs'], [(0, 1)])

    def testStrictCollisionGateIncludesEyeAndAlphaMaterials(self):
        head = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.]])
        before = np.r_[head, head + [0, 0, 2]]
        after = np.r_[head, [[.5, .5, -1.], [.5, .5, 1.], [1.5, .5, 0.]]]
        faces = np.arange(6).reshape(2, 3)
        for material in (2, 3):
            result = AUDIT.strictIntersectionAudit(before, after, faces, np.array([0, material]))
            self.assertFalse(result['accepted'])
            self.assertEqual(result['newPairs'], [(0, 1)])
            self.assertEqual(result['newPairsByMaterialPair'], {f'0/{material}': 1})
            self.assertEqual(result['excludedMaterialIds'], [])

    def testStrictCollisionRetainsSharedIndexExclusion(self):
        vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [1., 1., 0.]])
        result = AUDIT.strictIntersectionAudit(vertices, vertices, np.array([[0, 1, 2], [0, 1, 3]]))
        self.assertEqual(result['baselinePairs'], [])

    def testCanonicalFixedVerticesUseExactComparisonNotRoundtripTolerance(self):
        before = scene(); after = copy.deepcopy(before)
        after['vertices'][0, 0] += 1e-15
        result = AUDIT.sceneIdentityAudit(before, after, fixedVertexIds=[0])
        self.assertFalse(result['accepted'])
        self.assertFalse(result['checks']['fixedExactlyUnchanged'])

    def testProtectedAccessoryMovementIsRejected(self):
        before = scene(); after = copy.deepcopy(before)
        after['vertices'][1, 0] += .1
        self.assertFalse(AUDIT.sceneIdentityAudit(before, after, protectedVertexIds=[1])['accepted'])

    def testMatchingCountsDoNotProveTopologyOrBindingIdentity(self):
        for key in ('faces', 'uvFaces', 'nicpToFullIds'):
            before = scene(); after = copy.deepcopy(before)
            after[key] = after[key][..., ::-1]
            self.assertFalse(AUDIT.sceneIdentityAudit(before, after)['accepted'], key)

    def testUvAndMaterialChangesAreRejected(self):
        for key in ('uvs', 'materialIds'):
            before = scene(); after = copy.deepcopy(before)
            after[key] = after[key] + 1
            self.assertFalse(AUDIT.sceneIdentityAudit(before, after)['accepted'], key)

    def testOriginalDataFitRetainsTwoPercentMseLimit(self):
        operator = sparse.eye(3)
        target = np.zeros((3, 3)); before = np.ones((3, 3))
        good = np.ones((3, 3)) * np.sqrt(1.019)
        bad = np.ones((3, 3)) * np.sqrt(1.021)
        self.assertTrue(AUDIT.originalDataFitAudit(operator, target, before, good)['accepted'])
        self.assertFalse(AUDIT.originalDataFitAudit(operator, target, before, bad)['accepted'])

    def testPerfectZeroDataFitDoesNotHideAnyNewError(self):
        operator = sparse.eye(3); target = np.zeros((3, 3))
        self.assertTrue(AUDIT.originalDataFitAudit(operator, target, target, target)['accepted'])
        result = AUDIT.originalDataFitAudit(operator, target, target, np.ones((3, 3)) * 1e-15)
        self.assertFalse(result['accepted'])
        self.assertIsNone(result['mseRatio'])

    def testSourceTargetsMustComeFromOriginalSourceFrame(self):
        source = scene()
        bary = np.array([[.2, .3, .5], [.6, .1, .3]])
        targets = bary @ source['vertices']
        arguments = (source['vertices'], source['faces'], np.zeros(2, int), bary)
        valid = np.ones(2, bool)
        self.assertTrue(AUDIT.sourceCorrespondenceAudit(*arguments, targets, valid, identityTolerance=1e-12)['accepted'])
        self.assertFalse(AUDIT.sourceCorrespondenceAudit(*arguments, targets + .1, valid, identityTolerance=1e-12)['accepted'])

    def testSourceBarycentricCannotRescaleTheSampleSilently(self):
        source = scene(); bary = np.array([[.4, .6, 1.]])
        result = AUDIT.sourceCorrespondenceAudit(source['vertices'], source['faces'], np.zeros(1, int), bary,
            bary @ source['vertices'], np.ones(1, bool), identityTolerance=1e-12)
        self.assertFalse(result['accepted'])

    def testNoDifferentialEvidenceDoesNotReportPerfectImprovement(self):
        for weights, baseline in ((np.zeros(2), np.ones((2, 3))), (np.ones(2), np.zeros((2, 3)))):
            result = AUDIT.differentialFitAudit(sparse.eye(2), sparse.eye(2), np.zeros((2, 3)), baseline,
                np.zeros((2, 3)), reliabilityWeights=weights, reliabilityProvenance='uniform engineering weight')
            self.assertFalse(result['effectDefined'])
            self.assertIsNone(result['improvementFraction'])
            self.assertFalse(result['anatomicalTruthEstablished'])

    def testSourceDifferentialFitCannotReplaceOriginalDataFit(self):
        # A constant translation has zero pair differential while the original
        # observation error grows. Both diagnostics must remain available.
        difference = sparse.csr_matrix([[1., -1.]])
        baseline = np.ones((2, 3)); candidate = baseline + 10
        fit = AUDIT.differentialFitAudit(difference, sparse.eye(2), np.zeros((2, 3)), baseline, candidate,
            reliabilityWeights=np.ones(1), reliabilityProvenance='uniform engineering weight')
        original = AUDIT.originalDataFitAudit(sparse.eye(2), np.zeros((2, 3)), baseline, candidate)
        self.assertEqual(fit['candidateWeightedRms'], 0)
        self.assertFalse(original['accepted'])

    def testApertureImprovementRequiresBothEyesNotTextureProxyStatus(self):
        before, after = aperture(), aperture(1.9)
        self.assertTrue(AUDIT.apertureMetricAudit(before, after)['accepted'])
        after['eyes']['imageRight']['visibleComponentCount'] = 2
        after['textureContours'] = {'status': 'accepted'}
        self.assertFalse(AUDIT.apertureMetricAudit(before, after)['accepted'])

    def testApertureRequiresActualBenefitAndFiniteMeasurements(self):
        before = aperture()
        self.assertFalse(AUDIT.apertureMetricAudit(before, aperture())['accepted'])
        self.assertFalse(AUDIT.apertureMetricAudit(before, {'status': 'accepted'})['accepted'])
        after = aperture(1.9); after['eyes']['imageLeft']['intersectionOverUnion'] = np.nan
        self.assertFalse(AUDIT.apertureMetricAudit(before, after)['accepted'])


if __name__ == '__main__':
    unittest.main()
