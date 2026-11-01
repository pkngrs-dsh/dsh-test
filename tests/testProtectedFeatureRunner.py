"""Exercise the paired gate on serialized public-stage evidence."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from runProtectedFeatureExperiment import pairedEyeAudit


def fixture(offset=0.):
    vertices=np.array([[0.,0.,2.],[1.,0.,2.],[0.,1.,2.]])
    vertices[:,0]+=offset
    evidence={'ibugIds':[36,37], 'vertexIndices':[[0,1,2],[0,1,2]],
        'barycentricWeights':[[1.,0.,0.],[0.,1.,0.]],'targetPixels':[[0.,0.],[.5,0.]],
        'acceptedAfter':vertices.tolist()}
    return {'diagnostics':{'stagedLandmarks':{'stages':[{'name':name,'stageEvidence':evidence.copy()}
        for name in ('imageLeftEye','imageRightEye')]}}}

CAMERA={'intrinsics':np.eye(3),'rotation':np.eye(3),'translation':np.zeros(3)}


def testSerializedEqualStagesPass():
    result=pairedEyeAudit(fixture(),fixture(),CAMERA,1.)
    assert result['accepted']
    assert result['eyes']['imageLeftEye']['meanChangePixels']==0.


def testSameTargetsWorseFitFailsFrozenTolerance():
    result=pairedEyeAudit(fixture(.11),fixture(),CAMERA,1.)
    assert not result['accepted']
    assert result['eyes']['imageLeftEye']['meanChangePixels']>.05


def testDifferentTargetsCannotPassByReducingError():
    local=fixture()
    local['diagnostics']['stagedLandmarks']['stages'][0]['stageEvidence']['targetPixels']=[[.1,0.],[.5,0.]]
    result=pairedEyeAudit(local,fixture(),CAMERA,1.)
    assert not result['accepted']
    assert result['eyes']['imageLeftEye']['reason']=='pairedStageTargetsChanged'
