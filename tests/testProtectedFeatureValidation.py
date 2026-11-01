"""Input-derived protection and controlled-assembly acceptance regressions."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('protectedValidation', ROOT / 'tools/validateProtectedFeatureExperiment.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def supportInputs():
    faces = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    sample = sparse.csr_matrix([[1., 0, 0, 0, 0, 0], [0, 1., 0, 0, 0, 0]])
    return sample, faces


def fullScene():
    return dict(vertices=np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.],
                                   [0., 0., 2.], [1., 0., 2.], [0., 1., 2.]]),
        faces=np.array([[0, 1, 2], [3, 4, 5]]), uvs=np.array([[0., 0.], [1., 0.], [0., 1.]]),
        uvFaces=np.array([[0, 1, 2], [0, 1, 2]]), materialIds=np.array([0, 0]), nicpToFullIds=np.arange(3))


class ProtectedFeatureValidationTests(unittest.TestCase):
    def testStructuralSupportIsNotSquaredConfidence(self):
        _, faces = supportInputs()
        c = sparse.csr_matrix([[1., 1e-300, 0, 0, 0, 0]])
        domain = AUDIT.structuralSupportDomain(c, faces)
        np.testing.assert_array_equal(np.flatnonzero(domain['supportedMask']), [0, 1])
        self.assertFalse(domain['confidenceEstimated'])

    def testExplicitSparseZerosDoNotCreateSourceSupport(self):
        _, faces = supportInputs()
        c = sparse.csr_matrix((np.array([1., 0.]), (np.array([0, 0]), np.array([0, 5]))), shape=(1, 6))
        domain = AUDIT.structuralSupportDomain(c, faces)
        np.testing.assert_array_equal(np.flatnonzero(domain['supportedMask']), [0])

    def testUnsupportedProtectionExcludesOriginalStaticAndDelegatedCavity(self):
        c, faces = supportInputs()
        d = AUDIT.structuralSupportDomain(c, faces, originalStaticIds=[5], delegatedCavityIds=[4], eyeMouthIds=[3, 4])
        np.testing.assert_array_equal(d['unsupportedFixedVertexIds'], [2, 3])
        np.testing.assert_array_equal(d['transitionVertexIds'], [0, 1])
        np.testing.assert_array_equal(d['hardFixedVertexIds'], [2, 3, 5])
        np.testing.assert_array_equal(d['controlVertexIds'], [0, 1, 4])

    def testObservationReplicationDoesNotChangeProtection(self):
        c, faces = supportInputs()
        a = AUDIT.structuralSupportDomain(c, faces)
        b = AUDIT.structuralSupportDomain(sparse.vstack([c, c, c]), faces)
        for key in ('supportedMask', 'unsupportedFixedMask', 'transitionMask', 'controlVertexIds'):
            np.testing.assert_array_equal(a[key], b[key])

    def testVertexRelabelingCannotChangeThePhysicalDomain(self):
        c, faces = supportInputs()
        permutation = np.array([4, 2, 0, 5, 1, 3]); inverse = np.argsort(permutation)
        a = AUDIT.structuralSupportDomain(c, faces)
        b = AUDIT.structuralSupportDomain(c[:, permutation], inverse[faces])
        for key in ('supportedMask', 'unsupportedFixedMask', 'transitionMask'):
            np.testing.assert_array_equal(b[key], a[key][permutation])

    def testProtectionIsTopologicalNotCoordinateOrCaseDependent(self):
        c, faces = supportInputs()
        d = AUDIT.structuralSupportDomain(c, faces)
        self.assertFalse(d['candidateDependent'])
        self.assertNotIn('vertices', d)
        self.assertNotIn('caseId', d)

    def testInvalidAffineOrSignedSamplingIsRejected(self):
        _, faces = supportInputs()
        for row in ([.5, 0, 0, 0, 0, 0], [1.1, -.1, 0, 0, 0, 0]):
            with self.assertRaises(ValueError):
                AUDIT.structuralSupportDomain(sparse.csr_matrix([row]), faces)

    def testEmptyObservationsMeanNoStructuralSupport(self):
        _, faces = supportInputs()
        domain = AUDIT.structuralSupportDomain(sparse.csr_matrix((0, 6)), faces, originalStaticIds=[5], delegatedCavityIds=[4])
        self.assertFalse(domain['supportedMask'].any())
        np.testing.assert_array_equal(domain['unsupportedFixedVertexIds'], [0, 1, 2, 3])

    def testDelegatedFacesStayUnderTheirSeparateGuard(self):
        sample, faces = supportInputs()
        domain = AUDIT.structuralSupportDomain(sample, faces, delegatedCavityIds=[4])
        self.assertTrue(domain['incidentProtectedFaceMask'][2])
        self.assertFalse(domain['guardedFaceMask'][2])
        self.assertFalse(domain['guardedEdgeMask'][np.any(domain['edges'] == 4, axis=1)].any())

    def testProtectedDelegatedControlsRemainInTheUnion(self):
        sample, faces = supportInputs()
        domain = AUDIT.structuralSupportDomain(sample, faces, delegatedCavityIds=[4], eyeMouthIds=[4])
        self.assertIn(4, domain['localControlVertexIds'])
        self.assertIn(4, domain['controlVertexIds'])

    def testIndependentSupportSpecRebuildsBoundsAndRejectsRelaxation(self):
        from faceAppearance.supportedShapeConstraints import buildSupportedShapeConstraintSpec
        sample, faces = supportInputs()
        vertices = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.],
                             [1., 1., 1.], [0., 2., 1.], [1., 2., 1.]])
        static = np.array([5]); delegated = np.array([4]); protected = np.array([1])
        raw = dict(unit=1., faces=faces, staticIds=static, eyeMouthIds=protected, pureDirect=vertices)
        system = dict(a=raw, controls=delegated, C=sample)
        spec = buildSupportedShapeConstraintSpec(vertices, faces, sample, static,
            delegatedVertexIds=delegated, protectedVertexIds=protected, lengthScale=1.)
        solve = dict(startVertices=vertices, fixedVertexIds=spec['hardFixedVertexIds'],
            controlVertexIds=np.union1d(delegated, spec['controlVertexIds']))
        passed = AUDIT.supportSpecAudit(system, vertices, vertices, spec, solve)
        self.assertTrue(passed['validationPassed']); self.assertTrue(passed['constraintAccepted'])
        spec['edgeLower'] *= .5
        self.assertFalse(AUDIT.supportSpecAudit(system, vertices, vertices, spec, solve)['validationPassed'])

    def testUnsupportedCoordinatesStayExactlyAtBoundedBaseline(self):
        c, faces = supportInputs(); d = AUDIT.structuralSupportDomain(c, faces)
        before = np.arange(18, dtype=float).reshape(6, 3); after = before.copy()
        after[3, 0] = np.nextafter(after[3, 0], np.inf)
        self.assertFalse(AUDIT.hardSupportProtectionAudit(before, after, d)['accepted'])
        np.testing.assert_array_equal(AUDIT.hardSupportProtectionAudit(before, after, d)['unexpectedChangedVertexIds'], [3])

    def testDelegatedCavityIsNotAccidentallyHardFrozen(self):
        c, faces = supportInputs(); d = AUDIT.structuralSupportDomain(c, faces, delegatedCavityIds=[3])
        before = np.zeros((6, 3)); after = before.copy(); after[3] = 1
        result = AUDIT.hardSupportProtectionAudit(before, after, d)
        self.assertTrue(result['accepted'])
        np.testing.assert_array_equal(result['delegatedCavityChangedVertexIds'], [3])

    def testFixedAttachmentIncludesMaterialZeroInnerCorner(self):
        before = fullScene(); after = copy.deepcopy(before)
        after['vertices'][3, 0] += 1e-15
        self.assertFalse(AUDIT.fixedAttachmentAudit(before, after, after['vertices'][:3])['accepted'])

    def testOnlyTheMappedSkinMayChange(self):
        before = fullScene(); after = copy.deepcopy(before)
        after['vertices'][0, 0] += .01
        self.assertTrue(AUDIT.fixedAttachmentAudit(before, after, after['vertices'][:3])['accepted'])
        self.assertFalse(AUDIT.fixedAttachmentAudit(before, after, before['vertices'][:3])['accepted'])

    def testFixedSkinAndUvIdentityRemainRequired(self):
        before = fullScene(); after = copy.deepcopy(before); after['vertices'][0, 0] += .01
        self.assertFalse(AUDIT.fixedAttachmentAudit(before, after, after['vertices'][:3], staticSkinIds=[0])['accepted'])
        after = copy.deepcopy(before); after['uvFaces'] = after['uvFaces'][:, ::-1]
        self.assertFalse(AUDIT.fixedAttachmentAudit(before, after, after['vertices'][:3])['accepted'])

    def testPairedBaselineNoRegressionDoesNotWaiveOldPureGate(self):
        scene = fullScene(); crossing = scene['vertices'].copy()
        crossing[3:] = [[.2, .2, 0.], [.2, .2, 2.], [.7, .2, 1.]]
        anchors = {name: {'vertices': crossing, 'faces': scene['faces']}
                   for name in ('currentBounded', 'pairedBaseline')}
        anchors['originalPureDirect'] = {'vertices': scene['vertices'], 'faces': scene['faces']}
        result = AUDIT.threeAnchorFullAudit(crossing, scene['faces'], np.array([0, 2]), anchors)
        self.assertTrue(result['noRegressionAgainstPairedBaseline'])
        self.assertFalse(result['accepted'])
        self.assertEqual(result['anchors']['originalPureDirect']['newPairs'], [(0, 1)])

    def testAllThreeHistoricalAnchorsAreMandatory(self):
        scene = fullScene()
        with self.assertRaises(ValueError):
            AUDIT.threeAnchorFullAudit(scene['vertices'], scene['faces'], scene['materialIds'], {})

    def testLocalEyeStageCannotLeakOutsideSupport(self):
        before = np.zeros((4, 3)); after = before.copy(); after[3, 0] = 1e-15
        result = AUDIT.localEyeStageIdentityAudit(before, after, np.array([True, True, False, False]))
        self.assertFalse(result['accepted'])
        np.testing.assert_array_equal(result['changedOutsideSupportIds'], [3])

    def testLocalEyeStagePreservesPriorMouthChanges(self):
        before = np.ones((4, 3)); after = before.copy(); after[0] += .1
        result = AUDIT.localEyeStageIdentityAudit(before, after, np.array([True, False, False, False]))
        self.assertTrue(result['accepted'])
        self.assertFalse(AUDIT.localEyeStageIdentityAudit(before, after, np.ones(4, bool), originalStaticIds=[0])['accepted'])

    def testPairedResidualUsesSameFrozenBindingAndTargets(self):
        vertices = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
        ids = np.array([[0, 1, 2]]); bary = np.array([[.2, .3, .5]])
        camera = dict(intrinsics=np.eye(3), rotation=np.eye(3), translation=np.zeros(3))
        target = np.array([[.3, .5]])
        good = AUDIT.semanticTargetErrors(vertices, ids, bary, target, camera)
        shifted = vertices + [1, 0, 0]
        bad = AUDIT.semanticTargetErrors(shifted, ids, bary, target, camera)
        result = AUDIT.pairedSemanticAlignmentAudit(bad, good)
        self.assertTrue(result['diagnosticOnly']); self.assertNotIn('accepted', result)
        self.assertFalse(AUDIT.pairedSemanticAlignmentAudit(bad, good, tolerancePixels=.05)['accepted'])
        other = copy.deepcopy(good); other['targetPixels'] += .1
        with self.assertRaises(ValueError):
            AUDIT.pairedSemanticAlignmentAudit(bad, other)

    def testSemanticBarycentricFloatRoundingIsPreservedNotNormalized(self):
        vertices = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
        weights = np.array([[.2, .3, .5 + 2e-8]])
        original = weights.copy()
        camera = dict(intrinsics=np.eye(3), rotation=np.eye(3), translation=np.zeros(3))
        result = AUDIT.semanticTargetErrors(vertices, np.array([[0, 1, 2]]), weights, np.zeros((1, 2)), camera)
        self.assertTrue(np.isfinite(result['meanPixels']))
        np.testing.assert_array_equal(weights, original)
        self.assertNotEqual(weights.sum(), 1.)

    def testEyeSupportUsesBindingDepthAndOriginalCanthusWidth(self):
        from faceAppearance.semanticEyeSupport import buildSemanticEyeSupport
        _, faces = supportInputs()
        vertices = np.array([[0., 0., 2.], [.1, 0., 2.], [0., .1, 2.],
                             [.1, .1, 2.], [0., .2, 2.], [.1, .2, 2.]])
        indices = faces[np.array([0, 0, 1, 1, 2, 3])]
        weights = np.tile([.2, .3, .5], (6, 1)); ids = np.arange(36, 42)
        binding = dict(ibugIds=ids, vertexIndices=indices, barycentricWeights=weights)
        pixels = np.ones((68, 2)); pixels[36] = [10, 10]; pixels[39] = [30, 10]
        camera = dict(intrinsics=np.diag([100., 100., 1.]), rotation=np.eye(3), translation=np.zeros(3))
        observation = dict(pixels=pixels, confidence=np.ones(68), workingRgb=np.zeros((64, 64, 3)))
        selected = np.ones(6, bool); fixed = np.array([5])
        support = buildSemanticEyeSupport(vertices, faces, binding, observation, camera,
            eyeIbugIds=ids, eligibleMask=selected, fixedVertexIds=fixed)
        evidence = dict(frozenSupportReferenceVertices=vertices, vertexIndices=indices,
            barycentricWeights=weights, ibugIds=ids, selectedBindingMask=selected)
        result = AUDIT.eyeSupportAudit(evidence, support, faces, camera, pixels, np.ones(68), (64, 64), fixed, .5)
        self.assertTrue(result['validationPassed'])
        self.assertFalse(result['movableMask'][5])
        support['diagnostics']['localCameraDepth'] += .01
        self.assertFalse(AUDIT.eyeSupportAudit(evidence, support, faces, camera, pixels, np.ones(68), (64, 64), fixed, .5)['validationPassed'])


if __name__ == '__main__':
    unittest.main()
