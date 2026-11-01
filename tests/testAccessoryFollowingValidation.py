"""Independent contracts for a fixed template eye-accessory allowlist."""
from __future__ import annotations

import importlib.util
from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('accessoryFollowingAudit', ROOT/'tools/validateAccessoryFollowingExperiment.py')
AUDIT = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(AUDIT)


def sceneFixture():
    return dict(vertices=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
        [.3, .3, .2], [.8, .8, .2], [0., 0., 1.]]), faces=np.array([[0, 1, 2]]),
        uvs=np.array([[0., 0.], [1., 0.], [0., 1.]]), uvFaces=np.array([[0, 1, 2]]),
        materialIds=np.array([0]), nicpToFullIds=np.array([0, 1, 2]))


class AccessoryMappingAuditTests(unittest.TestCase):
    def setUp(self):
        self.base = sceneFixture()['vertices']; self.followers = np.array([3, 4])
        self.sources = np.array([[0, 1, 2], [0, 1, 2]])
        self.weights = np.array([[.4, .3, .3], [.1, .7, .2]])

    def positions(self, candidate, **kwargs):
        return AUDIT.independentAccessoryPositions(self.base, candidate,
            kwargs.get('followers', self.followers), kwargs.get('sources', self.sources), kwargs.get('weights', self.weights))

    def testNoDisplacementIsExactIdentity(self):
        np.testing.assert_array_equal(self.positions(self.base), self.base[self.followers])

    def testTotalBaselineMappingIgnoresCandidateFollowerValues(self):
        candidate = self.base.copy(); candidate[1] += [.1, -.2, 0.]
        expected = self.base[self.followers] + self.weights[:, 1, None]*np.array([.1, -.2, 0.])
        candidate[self.followers] += 10
        np.testing.assert_allclose(self.positions(candidate), expected, atol=1e-15, rtol=0)

    def testMappingComposesTotalSkinDisplacementsWithoutDrift(self):
        first = self.base.copy(); first[0] += [.1, .2, 0]
        second = first.copy(); second[2] += [-.2, .1, 0]
        second[self.followers] = self.positions(first)
        expected = self.base[self.followers] + self.weights[:, 0, None]*[.1, .2, 0] + self.weights[:, 2, None]*[-.2, .1, 0]
        np.testing.assert_allclose(self.positions(second), expected, atol=1e-15, rtol=0)

    def testNoFollowerSourcesOrIndependentFollowerDegreesOfFreedom(self):
        sources = self.sources.copy(); sources[0, 0] = 3
        with self.assertRaises(ValueError): self.positions(self.base, sources=sources)

    def testRejectNonconvexNonfiniteOrWrongWeights(self):
        for weights in (np.array([[-.1, .6, .5], [.1, .7, .2]]), self.weights*2,
                        self.weights*np.nan, self.weights[:, :2]):
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError): self.positions(self.base, weights=weights)


class WhitelistAssemblyAuditTests(unittest.TestCase):
    def setUp(self):
        self.scene = sceneFixture(); self.mask = np.array([True, True, False, False, False, False])
        self.followers = np.array([3, 4])

    def audit(self, candidate, followers=None):
        return AUDIT.whitelistAssemblyAudit(self.scene, candidate, self.mask,
            self.followers if followers is None else followers, staticFullIds=[2])

    def testAllowedFollowersMoveWithoutFalseAllNonSkinFixedClaim(self):
        candidate = {**self.scene, 'vertices': self.scene['vertices'].copy()}
        candidate['vertices'][3, 1] += .1
        result = self.audit(candidate)
        self.assertTrue(result['accepted']); self.assertFalse(result['allNonSkinExactlyFixed'])
        self.assertFalse(result['allNonSkinFixedIsRequired'])
        np.testing.assert_array_equal(result['effectiveFixedVertexIds'], [2, 5])

    def testEyeOrAnyNonwhitelistedPartMovementFails(self):
        candidate = {**self.scene, 'vertices': self.scene['vertices'].copy()}
        candidate['vertices'][5, 1] += .001
        self.assertFalse(self.audit(candidate)['accepted'])

    def testProtectedSkinRemainsExact(self):
        candidate = {**self.scene, 'vertices': self.scene['vertices'].copy()}
        candidate['vertices'][2, 1] += .001
        self.assertFalse(self.audit(candidate)['accepted'])

    def testTopologyOrUvChangeFails(self):
        for key in ('faces', 'uvs', 'uvFaces', 'materialIds', 'nicpToFullIds'):
            candidate = {**self.scene, key: self.scene[key].copy()}; candidate[key].flat[0] += 1
            with self.subTest(key=key): self.assertFalse(self.audit(candidate)['accepted'])

    def testSkinCannotBeDeclaredFollower(self):
        with self.assertRaises(ValueError): self.audit(self.scene, followers=np.array([0, 3]))

    def testDuplicateFollowerRejected(self):
        with self.assertRaises(ValueError): self.audit(self.scene, followers=np.array([3, 3]))


class OriginalTemplateBindingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from faceAppearance.eyeAccessoryFollowing import buildEyeAccessoryBinding, DEFAULT_POLICY
        path=ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1'
        cls.arrays=dict(np.load(path/'template.npz'))
        cls.components=json.loads((path/'components.json').read_text())
        cls.policy=DEFAULT_POLICY.copy()
        cls.binding=buildEyeAccessoryBinding(cls.arrays,cls.components,policy=cls.policy)

    def proof(self, binding):
        return AUDIT.accessoryBindingAudit(binding,self.arrays,self.components,self.policy)

    def testOriginalTemplateAllSixPartsAnd264SurfaceRowsPass(self):
        value=self.proof(self.binding)
        self.assertTrue(value['validationPassed'])
        self.assertEqual(sum(len(row['rows']) for row in value['components']),264)

    def testChangingDriverWeightsFailsEvenWithNewDigest(self):
        changed=deepcopy(self.binding); weights=changed['components'][0]['barycentricWeights'][0]
        weights[:]=[.2,.3,.5]
        changed['bindingSha256']=AUDIT.evidenceIdentity({k:v for k,v in changed.items() if k!='bindingSha256'})
        self.assertFalse(self.proof(changed)['validationPassed'])

    def testCrossLidDomainFailsEvenWithNewDigest(self):
        changed=deepcopy(self.binding)
        part=next(row for row in changed['components'] if row['role']=='upper')
        eye=next(row for row in changed['eyeDomains'] if row['name']==part['eyeName'])
        part['allowedFaceIds']=eye['sourceFaceDomains']['lower'].copy()
        changed['bindingSha256']=AUDIT.evidenceIdentity({k:v for k,v in changed.items() if k!='bindingSha256'})
        self.assertFalse(self.proof(changed)['validationPassed'])

    def testWhitelistRejectsEyebrowRelabeling(self):
        components=deepcopy(self.components)
        next(row for row in components if row['name']=='male00SG3.component001')['label']='eyebrow'
        with self.assertRaises(ValueError): AUDIT.templateAccessoryWhitelist(self.arrays,components)

    def testSpecificationAndScalarOperatorIndependentlyAgree(self):
        from faceAppearance.eyeAccessoryFollowing import prepareEyeAccessoryFollowing,eyeAccessoryDisplacementOperator,applyEyeAccessoryFollowing
        arrays=self.arrays; skin=arrays['nicpToFullIds']; base=arrays['vertices']
        fixed=np.setdiff1d(np.arange(len(base)),skin)
        support=dict(fixedVertexIds=fixed,movableMask=np.isin(np.arange(len(base)),skin))
        spec=prepareEyeAccessoryFollowing(self.binding,base,arrays['faces'],skin,fixed)
        operator=eyeAccessoryDisplacementOperator(spec)
        proof=AUDIT.followingSpecificationAudit(spec,self.binding,base,arrays,support,operator)
        self.assertTrue(proof['validationPassed'])
        candidate=base.copy(); candidate[skin,:2]+=np.random.default_rng(5).normal(0,.01,(len(skin),2))
        expected=AUDIT.independentlyMapFollowing(spec,candidate)
        actual=applyEyeAccessoryFollowing(spec,candidate)
        self.assertEqual(expected.tobytes(),actual.tobytes())
        self.assertTrue(AUDIT.followingRelationAudit(spec,actual)['validationPassed'])
        changed=operator.copy().tolil(); changed[spec['followerVertexIds'][0],skin[0]]+=.01
        self.assertFalse(AUDIT.followingSpecificationAudit(spec,self.binding,base,arrays,support,changed.tocsr())['validationPassed'])


class SkippedObservationReplayAuditTests(unittest.TestCase):
    def setUp(self):
        self.base=np.arange(12,dtype=float).reshape(4,3)
        self.support=dict(fixedVertexIds=np.array([0,3]),movableMask=np.array([False,True,True,False]))
        self.contours=dict(mask=np.array([[True,False],[False,True]]),status='baseline')
        self.error='imageLeft: intersecting lids'
        self.evidence=dict(vertices=self.base.copy(),rawCandidateVertices=self.base.copy(),solverVertices=self.base.copy(),
            candidateProposals=[],solverEvidence=[],support=deepcopy(self.support),beforeContours=deepcopy(self.contours),afterContours=deepcopy(self.contours),
            diagnostics=dict(status='skippedInvalidUnifiedObservation',candidateAccepted=False,accepted=False,
                contactOptimizationEnabled=False,observationAlignmentError=self.error))

    def audit(self, candidate):
        return AUDIT.skippedObservationReplayAudit(self.evidence,candidate,self.base,self.support,self.contours,self.error)

    def testActualSkipDoesNotRequireFabricatedSolverContours(self):
        proof=self.audit(deepcopy(self.evidence))
        self.assertTrue(proof['validationPassed']);self.assertTrue(proof['solverWasSkipped'])

    def testChangedBaselineOutputRejected(self):
        value=deepcopy(self.evidence);value['vertices'][0,0]+=.001
        self.assertFalse(self.audit(value)['validationPassed'])

    def testExtraFitFieldRejected(self):
        value=deepcopy(self.evidence);value['solverContours']=self.contours
        self.assertFalse(self.audit(value)['validationPassed'])

    def testUnexplainedInputFailureRejected(self):
        value=deepcopy(self.evidence);value['diagnostics']['observationAlignmentError']='another reason'
        self.assertFalse(self.audit(value)['validationPassed'])

    def testSolverProposalOrCaptureCannotHideInSkip(self):
        for key in ('candidateProposals','solverEvidence'):
            value=deepcopy(self.evidence);value[key]=[dict(round=1)]
            with self.subTest(key=key):self.assertFalse(self.audit(value)['validationPassed'])

    def testChangedSupportOrMaskRejected(self):
        value=deepcopy(self.evidence);value['support']['fixedVertexIds'][0]=1
        self.assertFalse(self.audit(value)['validationPassed'])
        value=deepcopy(self.evidence);value['afterContours']['mask'][0,0]=False
        self.assertFalse(self.audit(value)['validationPassed'])


if __name__ == '__main__': unittest.main()
