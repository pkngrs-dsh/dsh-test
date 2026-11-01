"""Independent acceptance contracts for fixed-boundary smooth eye fields."""
import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('boundaryFieldValidation', ROOT / 'tools/validateBoundaryFieldExperiment.py')
AUDIT = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(AUDIT)


def meshFixture():
    vertices = np.array([[x, y, 1.] for y in range(3) for x in range(3)], float)
    faces = []
    for y in range(2):
        for x in range(2):
            a = y * 3 + x; faces.extend([[a, a + 1, a + 3], [a + 1, a + 4, a + 3]])
    fixed = np.setdiff1d(np.arange(9), [4])
    return vertices, np.asarray(faces), fixed


def syntheticField():
    vertices, faces, fixed = meshFixture()
    system = AUDIT.independentBoundaryFieldSystem(vertices, faces, vertices[[1, 4, 7]],
        fixedVertexIds=fixed, supportRadius=2., controlCount=3, smoothingRadiusFraction=.1)
    raw = np.zeros_like(system['compactRaw']); free = system['freeVertexIds']
    raw[free] = np.linalg.solve(system['operator'].toarray(), system['rhs'])
    normalization = max(1., np.abs(raw).sum(1).max(initial=0))
    return system, raw / normalization, normalization


class BoundaryFieldValidationTests(unittest.TestCase):
    def classify(self, **overrides):
        args = dict(producerAccepted=True, geometryChanged=True, incrementalSafe=True,
            absoluteEligible=True, finalBaselineEffect={'effectDefined': True, 'accepted': True}, evidenceValidated=True)
        args.update(overrides); return AUDIT.classifyBoundaryCandidate(**args)

    def testPerRoundSuccessCannotWaiveFinalBaselineMaskGate(self):
        result = self.classify(finalBaselineEffect={'effectDefined': True, 'accepted': False})
        self.assertTrue(result['producerAccepted']); self.assertFalse(result['improvementAccepted'])
        self.assertFalse(result['changingAbsoluteEligible'])

    def testUndefinedEffectNeverMeansImprovement(self):
        self.assertFalse(self.classify(finalBaselineEffect={'effectDefined': False, 'accepted': True})['improvementAccepted'])

    def testUnchangedModelIsNotSuccessfulImprovement(self):
        self.assertFalse(self.classify(geometryChanged=False)['improvementAccepted'])

    def testInheritedFailureRemainsSeparateFromLocalImprovement(self):
        result = self.classify(absoluteEligible=False)
        self.assertTrue(result['improvementAccepted']); self.assertFalse(result['changingAbsoluteEligible'])

    def testIndependentEvidenceCannotBeWaived(self):
        self.assertFalse(self.classify(evidenceValidated=False)['improvementAccepted'])

    def testRepresentationHasExactBoundaryAndCameraPlane(self):
        system, basis, _ = syntheticField()
        report = AUDIT.fieldRepresentationAudit(system['referenceVertices'], basis, np.eye(3)[:, :2],
            fixedVertexIds=system['fixedVertexIds'], maxDisplacement=.1, cameraRotation=np.eye(3))
        self.assertTrue(report['validationPassed']); self.assertEqual(report['activeVertexCount'], 1)
        self.assertFalse(report['constructionFormulaAudited'])

    def testInactiveFieldIsReportedWithoutInventedSupportFallback(self):
        vertices, _, fixed = meshFixture(); matrix = np.zeros((9, 3))
        report = AUDIT.fieldRepresentationAudit(vertices, matrix, np.eye(3)[:, :2], fixedVertexIds=fixed,
            maxDisplacement=.1, cameraRotation=np.eye(3))
        self.assertTrue(report['validationPassed']); self.assertTrue(report['ineffectiveField'])
        self.assertEqual(report['numericalRank'], 0)

    def testTinyFixedRowLeakIsStillRejected(self):
        system, basis, _ = syntheticField(); basis[0, 0] = 1e-30
        report = AUDIT.fieldRepresentationAudit(system['referenceVertices'], basis, np.eye(3)[:, :2],
            fixedVertexIds=system['fixedVertexIds'], maxDisplacement=.1, cameraRotation=np.eye(3))
        self.assertFalse(report['validationPassed'])

    def testChangingToWorldDepthFreedomIsRejected(self):
        system, basis, _ = syntheticField()
        report = AUDIT.fieldRepresentationAudit(system['referenceVertices'], basis, np.eye(3)[:, [0, 2]],
            fixedVertexIds=system['fixedVertexIds'], maxDisplacement=.1, cameraRotation=np.eye(3))
        self.assertFalse(report['validationPassed'])

    def testRawClippingDoesNotMasqueradeAsHardBoundaryField(self):
        vertices, _, fixed = meshFixture(); basis = np.ones((9, 1)); before = np.zeros((1, 2)); step = np.array([[.1, 0.]])
        raw = AUDIT.reconstructFieldTrial(vertices, basis, np.eye(3)[:, :2], .1, step)
        raw[fixed] = vertices[fixed]
        report = AUDIT.trialFieldEvidenceAudit(vertices, basis, np.eye(3)[:, :2], maxDisplacement=.1,
            coefficientsBefore=before, coefficientStep=step, stepScale=1., trialCoefficients=step,
            rawVertices=raw, fixedVertexIds=fixed)
        self.assertFalse(report['validationPassed']); self.assertTrue(report['clippedRawGeometryDetected'])

    def testCoefficientBacktrackAndRawGeometryAreReconstructed(self):
        system, basis, _ = syntheticField(); before = np.zeros((basis.shape[1], 2)); step = np.ones_like(before)
        proposed = before + .5 * step
        raw = AUDIT.reconstructFieldTrial(system['referenceVertices'], basis, np.eye(3)[:, :2], .1, proposed)
        report = AUDIT.trialFieldEvidenceAudit(system['referenceVertices'], basis, np.eye(3)[:, :2], maxDisplacement=.1,
            coefficientsBefore=before, coefficientStep=step, stepScale=.5, trialCoefficients=proposed,
            rawVertices=raw, fixedVertexIds=system['fixedVertexIds'])
        self.assertTrue(report['validationPassed'])
        proposed[0, 0] += 1e-12
        report = AUDIT.trialFieldEvidenceAudit(system['referenceVertices'], basis, np.eye(3)[:, :2], maxDisplacement=.1,
            coefficientsBefore=before, coefficientStep=step, stepScale=.5, trialCoefficients=proposed,
            rawVertices=raw, fixedVertexIds=system['fixedVertexIds'])
        self.assertFalse(report['validationPassed'])

    def testSparseAndDenseRepresentTheSameField(self):
        system, basis, _ = syntheticField(); coefficients = np.ones((basis.shape[1], 2))
        dense = AUDIT.reconstructFieldTrial(system['referenceVertices'], basis, np.eye(3)[:, :2], .1, coefficients)
        sparseResult = AUDIT.reconstructFieldTrial(system['referenceVertices'], sparse.csr_matrix(basis), np.eye(3)[:, :2], .1, coefficients)
        np.testing.assert_allclose(dense, sparseResult, atol=1e-15, rtol=0)

    def testAcceptedTrialCannotSkipAnUncomfortableGate(self):
        gates = {'shape': {'evaluated': True, 'accepted': True}, 'contact': {'evaluated': False, 'accepted': None}}
        self.assertFalse(AUDIT.trialGateTraceAudit(gates, requiredGateNames=list(gates), accepted=True, rejectionReasons=[])['validationPassed'])

    def testRejectedTrialRetainsFailureAndDoesNotInventLaterPass(self):
        gates = {'shape': {'evaluated': True, 'accepted': False}, 'objective': {'evaluated': False, 'accepted': None}}
        self.assertTrue(AUDIT.trialGateTraceAudit(gates, requiredGateNames=list(gates), accepted=False, rejectionReasons=['shape'])['validationPassed'])
        gates['objective']['accepted'] = True
        self.assertFalse(AUDIT.trialGateTraceAudit(gates, requiredGateNames=list(gates), accepted=False, rejectionReasons=['shape'])['validationPassed'])

    def testEveryEvaluatedRejectionRemainsNamed(self):
        gates = {k: {'evaluated': True, 'accepted': False} for k in ('shape', 'contact')}
        self.assertFalse(AUDIT.trialGateTraceAudit(gates, requiredGateNames=list(gates), accepted=False, rejectionReasons=['shape'])['validationPassed'])

    def testIndependentFEMMatchesSavedSolutionWithoutSolvingInAudit(self):
        system, basis, scale = syntheticField()
        report = AUDIT.boundaryFieldConstructionAudit(system, basis, system['selectedControlPoints'], scale)
        self.assertTrue(report['validationPassed']); self.assertLess(report['equationBackwardError'], 1e-14)
        self.assertFalse(system['systemSolved'])

    def testRestrictingLaplacianBeforeSquaringIsADifferentField(self):
        system, _, _ = syntheticField(); free = system['freeVertexIds']
        smallK = system['stiffness'][free][:, free]; mass = system['lumpedMass'][free]
        wrong = sparse.diags(mass) + .1 ** 4 * (smallK.T @ sparse.diags(1 / mass) @ smallK)
        raw = np.zeros_like(system['compactRaw']); raw[free] = np.linalg.solve(wrong.toarray(), system['rhs'])
        scale = max(1., np.abs(raw).sum(1).max(initial=0))
        report = AUDIT.boundaryFieldConstructionAudit(system, raw / scale, system['selectedControlPoints'], scale)
        self.assertFalse(report['validationPassed']); self.assertFalse(report['checks']['freeFullFEMEquation'])

    def testChangingNormalizationDoesNotPassTheSameObjective(self):
        system, basis, scale = syntheticField()
        self.assertFalse(AUDIT.boundaryFieldConstructionAudit(system, basis, system['selectedControlPoints'], scale + 1)['validationPassed'])

    def testObtuseCotangentIsNotSilentlyClamped(self):
        vertices = np.array([[0., 0., 1.], [2., 0., 1.], [.2, .1, 1.]])
        system = AUDIT.independentBoundaryFieldSystem(vertices, np.array([[0, 1, 2]]), vertices[[0]],
            fixedVertexIds=np.array([0]), supportRadius=2.)
        k = system['stiffness'].toarray()
        self.assertGreater(k[0, 1], 0.)
        np.testing.assert_allclose(k.sum(1), 0., atol=1e-14)

    def testUniformUnitsDoNotChangeDimensionlessFEM(self):
        vertices, faces, fixed = meshFixture(); controls = vertices[[1, 4, 7]]
        a = AUDIT.independentBoundaryFieldSystem(vertices, faces, controls, fixedVertexIds=fixed, supportRadius=2.)
        b = AUDIT.independentBoundaryFieldSystem(vertices * 17 + [5, 2, -3], faces, controls * 17 + [5, 2, -3],
            fixedVertexIds=fixed, supportRadius=34.)
        np.testing.assert_allclose(a['operator'].toarray(), b['operator'].toarray(), atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(a['rhs'], b['rhs'], atol=1e-13, rtol=1e-13)

    def testDegenerateMeshDoesNotUseFallbackField(self):
        with self.assertRaises(ValueError):
            AUDIT.independentBoundaryFieldSystem(np.zeros((3, 3)), np.array([[0, 1, 2]]), np.zeros((1, 3)),
                fixedVertexIds=np.empty(0, int), supportRadius=1.)

    def testSavedProductionFormulaAgainstIndependentGradientAssembly(self):
        from faceAppearance.fixedBoundaryDeformation import buildFixedBoundaryDeformationBasis
        vertices, faces, fixed = meshFixture()
        field = buildFixedBoundaryDeformationBasis(vertices, faces, vertices[[1, 4, 7]],
            fixedVertexIds=fixed, supportRadius=2., controlCount=3)
        report = AUDIT.savedBoundaryFieldAudit(field['evidence'], field['basis'],
            requestedControlCount=3, supportRadius=2., expectedFixedVertexIds=fixed)
        self.assertTrue(report['validationPassed'], report['checks'])

    def testSavedMatrixTamperingIsDetectedEvenWhenBasisPasses(self):
        from faceAppearance.fixedBoundaryDeformation import buildFixedBoundaryDeformationBasis
        vertices, faces, fixed = meshFixture()
        field = buildFixedBoundaryDeformationBasis(vertices, faces, vertices[[1, 4, 7]],
            fixedVertexIds=fixed, supportRadius=2., controlCount=3)
        field['evidence']['bending']['data'][0] += .01
        report = AUDIT.savedBoundaryFieldAudit(field['evidence'], field['basis'],
            requestedControlCount=3, supportRadius=2., expectedFixedVertexIds=fixed)
        self.assertFalse(report['validationPassed']); self.assertFalse(report['checks']['bendingIndependentFEM'])

    def testIsolatedFixedVertexDoesNotInventMassOrMotion(self):
        vertices, faces, fixed = meshFixture(); vertices = np.r_[vertices, [[5., 5., 1.]]]
        system = AUDIT.independentBoundaryFieldSystem(vertices, faces, vertices[[4]],
            fixedVertexIds=np.r_[fixed, 9], supportRadius=2.)
        self.assertEqual(system['lumpedMass'][9], 0)
        self.assertEqual(system['fullCurvature'].getrow(9).nnz, 0)
        with self.assertRaises(ValueError):
            AUDIT.independentBoundaryFieldSystem(vertices, faces, vertices[[4]], fixedVertexIds=fixed, supportRadius=2.)

    def testPhotoAndRegularizerObjectiveReconstruction(self):
        vertices = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
        faces = np.array([[0, 1, 2]]); basis = np.array([[1.], [0.], [0.]])
        regularizer = AUDIT.independentFieldRegularizer(vertices, faces, basis, np.array([True, False, False]),
            maxDisplacement=.1, displacementWeight=.002, strainWeight=.03, controlWeight=.0001)
        np.testing.assert_allclose(regularizer, np.eye(2) * .0024, atol=1e-15)
        row = dict(viewIndex=0, vertexIndices=np.array([[0]]), barycentric=np.array([[1.]]),
            originalPoints=vertices[[0]], basis=np.array([[1.]]), weights=np.array([2.]), normalizedWeights=np.array([1.]),
            targetPixels=np.array([[.1, 0.]]), transform=np.eye(2)[None], segmentPairs=np.empty((0, 2), int))
        evidence = dict(referenceVertices=vertices, basis=basis, axes=np.eye(3)[:, :2], maxDisplacement=.1, preparedConstraints=[row])
        camera = dict(intrinsics=np.eye(3), rotation=np.eye(3), translation=np.zeros(3))
        evaluated = AUDIT.independentSolverObjective(evidence, np.array([[.1, 0.]]), camera,
            regularizer=regularizer, pixelScale=1., curveWeight=0., additionalEnergy=.02)
        self.assertAlmostEqual(evaluated['objectiveComponents']['regularization'], .000012)
        self.assertAlmostEqual(evaluated['objectiveComponents']['photoAndCurve'], .00405)
        self.assertAlmostEqual(evaluated['objective'], .024062)
        self.assertAlmostEqual(evaluated['meanErrorPixels'], .09)
        self.assertTrue(evaluated['preparedBindingsValidated'])
        row['normalizedWeights'] = np.array([.5])
        self.assertFalse(AUDIT.independentSolverObjective(evidence, np.array([[.1, 0.]]), camera,
            regularizer=regularizer, pixelScale=1., curveWeight=0.)['preparedBindingsValidated'])


if __name__ == '__main__': unittest.main()

class CapturedSolverIntegrationTests(unittest.TestCase):
    def testBothModesPassFullSyntheticCapturedRoundAudit(self):
        from unittest.mock import patch
        from faceAppearance.frontLandmarkRefinement import refineFrontLandmarks
        vertices, faces, fixed = meshFixture()
        camera=dict(rotation=np.eye(3),translation=np.array([0.,0.,10.]),intrinsics=np.diag([100.,100.,1.]))
        ids=np.repeat([[4]],5,axis=0)
        pixels=vertices[ids[:,0],:2]*100/11
        rows=[dict(name='synthetic',viewIndex=0,vertexIndices=ids,barycentric=np.ones_like(ids,float),targetPixels=pixels+np.array([.002,0.]),weights=np.ones(5))]
        coarse=[dict(name=name,vertexIndices=np.repeat([[4,4,4]],4,axis=0),barycentric=np.tile([1.,0.,0.],(4,1)),targetPixels=np.repeat(pixels[[0]],4,axis=0)) for name in AUDIT.COARSE_EYES]
        support=dict(fixedVertexIds=fixed,movableMask=~np.isin(np.arange(9),fixed))
        spec=dict(sourceVertexIds=np.empty(0,int),targetVertexIds=np.empty((0,3),int),targetBarycentric=np.empty((0,3)),normals=np.empty((0,3)),minimumGap=0.,lengthScale=1.,weight=1.)
        operator=sparse.csr_matrix((0,27))
        def safety(v):
            coarseCheck=AUDIT.coarseEyePreservationAudit(vertices,v,coarse,camera,imageScale=1.)
            contact=AUDIT.frozenContactGapAudit(vertices,v,spec['sourceVertexIds'],spec['targetVertexIds'],spec['targetBarycentric'],spec['normals'],0.,lengthScale=1.,tolerance=1e-8)
            return dict(accepted=coarseCheck['accepted'] and contact['accepted'],coarseAlignment=coarseCheck,contact=contact)
        scene=dict(nicpToFullIds=np.arange(9),faces=faces,materialIds=np.zeros(len(faces),int),uvs=None,uvFaces=None,materials=None,alphaTextures=None)
        prepared=dict(cornerFaceIds=np.empty(0,int),photoEyes={})
        config=dict(fieldPolicy=dict(legacyMode='compact',candidateMode='fixedBoundaryBiharmonic'))
        for mode,branch in (('compact','legacyResidualReplay'),('fixedBoundaryBiharmonic','boundaryFieldContact')):
            fitted=refineFrontLandmarks(vertices,faces,camera,rows,fixedVertexIds=fixed,supportRadius=.24,controlCount=32,maxDisplacement=.06,
                minConfidence=0.,iterations=3,maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3,curveWeight=0.,deformationMode=mode,captureSolverEvidence=True,additionalSafety=safety)
            record=dict(round=1,fullVertexIds=np.arange(9),fullFaceIds=np.arange(len(faces)),fullReferenceVertices=vertices.copy(),fullFixedVertexIds=fixed,evidence=fitted['solverEvidence'])
            raw=dict(solverEvidence=[record],diagnostics=dict(fast=dict(stages=[dict(round=1,solver=fitted['diagnostics'])])),candidateProposals=[])
            with patch('faceAppearance.fastFrontEyeGeometry._constraints',return_value=(rows,{})):
                report=AUDIT.auditSolverFields(branch,prepared,raw,camera,vertices,scene,support,coarse,1.,spec,operator,lambda _: {},config)
            self.assertTrue(report['validationPassed'],str(report['rounds'][0]['checks'])+str(report['rounds'][0]['trials']))
