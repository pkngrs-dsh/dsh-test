"""Prospective independent contracts for changing only frozen eye observations."""
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('eyeObservationAudit',ROOT/'tools/validateEyeObservationExperiment.py')
AUDIT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(AUDIT)


def fixture():
    photos={};targets=[]
    for name,offset in zip(AUDIT.EYES,(np.array([10.,20.]),np.array([40.,25.]))):
        x=np.array([0.,2.,8.,10.]);h=np.array([0.,2.,2.,0.])
        photos[name]=dict(upper=np.c_[x,-h]+offset,lower=np.c_[x,h]+offset)
        targets.extend(np.array([[-1.,.5],[3.,-1.],[7.,-1.],[11.,.5],[7.,1.],[3.,1.]])+offset)
    return photos,np.asarray(targets)


def groups(table):
    result=[]
    for start,name in ((36,'imageLeftEye'),(42,'imageRightEye')):
        ids=np.arange(start,start+6)[[0,1,3,4]]
        result.append(dict(name=name,ibugIds=ids,vertexIndices=np.tile([0,1,2],(4,1)),
            barycentric=np.tile([.2,.3,.5],(4,1)),weights=np.arange(1,5,dtype=float),confidence=np.ones(4),
            targetPixels=table[ids-36].copy()))
    return result


class EyeObservationValidationTests(unittest.TestCase):
    def testOneUniqueTablePreservesInternalAxisAndMovesCornersToMidpoints(self):
        photo,old=fixture();value=AUDIT.independentUnifiedTargets(photo,old)
        np.testing.assert_array_equal(value['targetPixels'][[1,2,4,5,7,8,10,11],0],old[[1,2,4,5,7,8,10,11],0])
        for eye,start in zip(AUDIT.EYES,(0,6)):
            np.testing.assert_array_equal(value['targetPixels'][start],(photo[eye]['upper'][0]+photo[eye]['lower'][0])/2)
            np.testing.assert_array_equal(value['targetPixels'][start+3],(photo[eye]['upper'][-1]+photo[eye]['lower'][-1])/2)
            self.assertEqual(value['evidence'][eye]['points'][1]['segmentIndex'],1)
        self.assertFalse(value['anatomicalTruthEstablished'])

    def testRotationsAndTranslationsAreCovariantWithoutSortingX(self):
        photo,old=fixture();before=AUDIT.independentUnifiedTargets(photo,old)['targetPixels']
        theta=2.1;R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        t=np.array([21.,-9.]);converted={name:{k:v@R.T+t for k,v in lids.items()} for name,lids in photo.items()}
        after=AUDIT.independentUnifiedTargets(converted,old@R.T+t)['targetPixels']
        np.testing.assert_allclose(after,before@R.T+t,rtol=0,atol=2e-14)

    def testUniformScaleUsesTheSameAxisParameter(self):
        photo,old=fixture();before=AUDIT.independentUnifiedTargets(photo,old)
        after=AUDIT.independentUnifiedTargets({n:{k:v*7 for k,v in p.items()} for n,p in photo.items()},old*7)
        np.testing.assert_allclose(after['targetPixels'],before['targetPixels']*7,rtol=0,atol=1e-13)
        self.assertEqual(after['evidence']['imageLeft']['points'][1]['originalAxisCoordinate'],.3)

    def testArbitraryTableOrderJoinsByIbugId(self):
        photo,old=fixture();permutation=np.array([11,5,9,2,1,4,0,6,10,8,7,3])
        canonical=AUDIT.independentUnifiedTargets(photo,old)
        shuffled=AUDIT.independentUnifiedTargets(photo,old[permutation],ibugIds=AUDIT.EYE_IDS[permutation])
        np.testing.assert_array_equal(shuffled['targetPixels'],canonical['targetPixels'][permutation])

    def testSourceArraysAreNeverMutated(self):
        photo,old=fixture();saved=copy.deepcopy(photo);savedOld=old.copy()
        value=AUDIT.independentUnifiedTargets(photo,old);value['targetPixels'][0]=0
        np.testing.assert_array_equal(old,savedOld)
        for name in photo:
            for lid in photo[name]:np.testing.assert_array_equal(photo[name][lid],saved[name][lid])

    def testMissingOrDuplicateIdsCannotGenerateASecondPartialTable(self):
        photo,old=fixture()
        with self.assertRaises(ValueError):AUDIT.independentUnifiedTargets(photo,old[:-1],ibugIds=AUDIT.EYE_IDS[:-1])
        ids=AUDIT.EYE_IDS.copy();ids[-1]=46
        with self.assertRaises(ValueError):AUDIT.independentUnifiedTargets(photo,old,ibugIds=ids)

    def testReversedOrRepeatedDenseAxisIsNotRepairedBySorting(self):
        for change in ('reverse','duplicate'):
            photo,old=fixture()
            if change=='reverse':photo['imageLeft']['upper'][1:3]=photo['imageLeft']['upper'][2:0:-1]
            else:photo['imageLeft']['upper'][2,0]=photo['imageLeft']['upper'][1,0]
            with self.assertRaises(ValueError):AUDIT.independentUnifiedTargets(photo,old)

    def testCrossingLidsAreRejected(self):
        photo,old=fixture();photo['imageLeft']['upper'][1,1]=23
        with self.assertRaises(ValueError):AUDIT.independentUnifiedTargets(photo,old)

    def testOutOfDomainInternalPointIsNotClampedOrExtrapolated(self):
        photo,old=fixture();old[1,0]=9.9
        with self.assertRaises(ValueError):AUDIT.independentUnifiedTargets(photo,old)

    def testReversedSemanticOrderIsRejected(self):
        photo,old=fixture();old[[1,2]]=old[[2,1]]
        with self.assertRaises(ValueError):AUDIT.independentUnifiedTargets(photo,old)

    def testOnlyReferenceEyesChangeAndConfidenceIsExact(self):
        photo,old=fixture();table=AUDIT.independentUnifiedTargets(photo,old)
        original=dict(points=np.zeros((4,68,2)),confidence=np.ones((4,68)),accepted=np.ones(4,bool))
        original['points'][3,36:48]=old;aligned=copy.deepcopy(original);aligned['points'][3,36:48]=table['targetPixels']
        self.assertTrue(AUDIT.observationReplacementAudit(original,aligned,3,table)['validationPassed'])
        aligned['confidence'][0,0]=.9
        self.assertFalse(AUDIT.observationReplacementAudit(original,aligned,3,table)['validationPassed'])

    def testWrongReferenceOrMouthChangeIsRejected(self):
        photo,old=fixture();table=AUDIT.independentUnifiedTargets(photo,old)
        original=dict(points=np.zeros((4,68,2)),confidence=np.ones((4,68)),accepted=np.ones(4,bool));original['points'][3,36:48]=old
        aligned=copy.deepcopy(original);aligned['points'][3,36:48]=table['targetPixels'];aligned['points'][3,48,0]=1
        self.assertFalse(AUDIT.observationReplacementAudit(original,aligned,3,table)['validationPassed'])

    def testCoarseSubsetMustReuseUniqueTableAndKeepBindingsWeights(self):
        photo,old=fixture();table=AUDIT.independentUnifiedTargets(photo,old);source=groups(old);aligned=groups(table['targetPixels'])
        self.assertTrue(AUDIT.coarseTableReuseAudit(source,aligned,table)['validationPassed'])
        aligned[0]['barycentric'][0]=[1.,0.,0.]
        self.assertFalse(AUDIT.coarseTableReuseAudit(source,aligned,table)['validationPassed'])

    def testNewGateCannotRelabelOriginalFailureAsAcceptance(self):
        result=AUDIT.classifyObservationCandidate(producerAccepted=True,changed=True,activeObservationAccepted=True,
            physicalIncrementalSafe=True,legacyOriginalAccepted=False,legacyAbsoluteEligible=False,
            finalBaselineEffectAccepted=True,evidenceValidated=True)
        self.assertTrue(result['diagnosticImprovementAccepted']);self.assertFalse(result['changingAbsoluteEligible'])
        self.assertFalse(result['improvementPreservesOriginalPolicy']);self.assertFalse(result['clientReady'])

    def testFinalBaselineMaskIsStillRequired(self):
        result=AUDIT.classifyObservationCandidate(producerAccepted=True,changed=True,activeObservationAccepted=True,
            physicalIncrementalSafe=True,legacyOriginalAccepted=True,legacyAbsoluteEligible=True,
            finalBaselineEffectAccepted=False,evidenceValidated=True)
        self.assertFalse(result['diagnosticImprovementAccepted'])

    def testUnchangedBaselineCannotCountAsImprovement(self):
        result=AUDIT.classifyObservationCandidate(producerAccepted=True,changed=False,activeObservationAccepted=True,
            physicalIncrementalSafe=True,legacyOriginalAccepted=True,legacyAbsoluteEligible=True,
            finalBaselineEffectAccepted=True,evidenceValidated=True)
        self.assertFalse(result['diagnosticImprovementAccepted'])

class ProducerTableContractTests(unittest.TestCase):
    def testSyntheticProducerTablePassesIndependentIntersectionProof(self):
        from faceAppearance.eyeObservationAlignment import alignEyeLandmarkTargets
        photo,old=fixture();table=alignEyeLandmarkTargets(photo,old)
        report=AUDIT.alignmentTableAudit(photo,old,table)
        self.assertTrue(report['validationPassed'],report['checks'])

    def testTamperedTargetsOrSegmentEvidenceAreRejected(self):
        from faceAppearance.eyeObservationAlignment import alignEyeLandmarkTargets
        photo,old=fixture();table=alignEyeLandmarkTargets(photo,old)
        tampered=copy.deepcopy(table);tampered['targetPixels'][1,1]+=.01
        self.assertFalse(AUDIT.alignmentTableAudit(photo,old,tampered)['validationPassed'])
        tampered=copy.deepcopy(table);tampered['evidence']['points'][1]['sourceSegment']+=1
        self.assertFalse(AUDIT.alignmentTableAudit(photo,old,tampered)['validationPassed'])
