"""Localize the original semantic eye solver without changing its targets."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

import faceAppearance.clientGeometryRefinement as refinement
from faceAppearance.camera import projectPoints
from faceAppearance.semanticEyeSupport import (buildSemanticEyeSupport,
    normalizeSemanticEyeSupportPolicy, semanticEyeFitMetrics)


def fixture(disconnected=False):
    nx,ny=31,21
    xx,yy=np.meshgrid(np.linspace(-1.5,1.5,nx),np.linspace(-1.,1.,ny))
    vertices=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,5.)]
    faces=[]
    for row in range(ny-1):
        for col in range(nx-1):
            first=row*nx+col
            faces.extend([[first,first+1,first+nx],[first+1,first+nx+1,first+nx]])
    points=np.array([[-.9,0.],[-.7,-.1],[-.5,-.1],[-.3,0.],[-.5,.1],[-.7,.1],
        [.3,0.],[.5,-.1],[.7,-.1],[.9,0.],[.7,.1],[.5,.1]])
    anchors=np.argmin(np.linalg.norm(vertices[None,:,:2]-points[:,None,:],axis=-1),axis=1)
    indices=np.c_[anchors,anchors+1,anchors+nx]
    binding=dict(ibugIds=np.arange(36,48),vertexIndices=indices,
        barycentricWeights=np.tile([1.,0.,0.],(12,1)),validMask=np.ones(12,bool))
    camera=dict(intrinsics=np.array([[100.,0.,128.],[0.,100.,128.],[0.,0.,1.]]),rotation=np.eye(3),translation=np.zeros(3))
    pixels=np.full((68,2),128.)
    pixels[36:48]=projectPoints(vertices[anchors],**camera)[0]+[.4,0.]
    observation=dict(pixels=pixels,confidence=np.ones(68),workingRgb=np.zeros((256,256,3),np.uint8))
    border=(xx==xx.min())|(xx==xx.max())|(yy==yy.min())|(yy==yy.max())
    fixed=np.flatnonzero(border.ravel())
    if disconnected:
        first=len(vertices);vertices=np.r_[vertices,vertices[indices[0]]+[0,0,.001]]
        faces.append([first,first+1,first+2])
    return vertices,np.asarray(faces,np.int64),binding,observation,camera,fixed


def support(args,**kwargs):
    v,f,b,o,c,fixed=args
    return buildSemanticEyeSupport(v,f,b,o,c,eyeIbugIds=range(36,42),eligibleMask=np.ones(12,bool),fixedVertexIds=fixed,**kwargs)


def runStages(args,**kwargs):
    v,f,b,o,c,fixed=args
    return refinement._stagedLandmarks(v,f,fixed,b,[c],[o],[np.ones(12,bool)],0,
        {'reliable':True,'reasons':[]},**kwargs)


def testGeodesicBandIsFrozenInSkinTopologyAndProtectsFarRegions():
    args=fixture(disconnected=True);result=support(args)
    assert result['diagnostics']['status']=='supported'
    assert result['diagnostics']['photoEyeWidthCanonical']==pytest.approx(.6)
    assert result['diagnostics']['geodesicRadiusCanonical']==pytest.approx(.3)
    assert not result['movableMask'][-3:].any()  # Nearby but disconnected sheet.
    assert not result['movableMask'][args[-1]].any()
    vertices=args[0]
    assert not result['movableMask'][vertices[:,1]>.7].any()
    assert not result['movableMask'][vertices[:,1]<-.7].any()
    assert not result['movableMask'][vertices[:,0]>.8].any()
    assert result['movableMask'][result['seedVertexIds']].all()
    assert np.isfinite(result['geodesicDistance']).all()
    assert not result['geodesicDistanceMeasuredMask'][-3:].any()
    assert (result['geodesicDistance'][~result['geodesicDistanceMeasuredMask']]==0).all()


@pytest.mark.parametrize('scale',[.1,10.])
def testSupportIsCovariantToCanonicalUnitsRotationTranslation(scale):
    args=fixture();baseline=support(args);changed=list(deepcopy(args))
    angle=.37
    rotation=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    shift=np.array([2.,-3.,.7])
    changed[0]=args[0]@rotation.T*scale+shift
    changed[4]['rotation']=rotation.T;changed[4]['translation']=-shift@rotation
    result=support(changed)
    np.testing.assert_array_equal(result['movableMask'],baseline['movableMask'])
    assert result['diagnostics']['geodesicRadiusCanonical']==pytest.approx(baseline['diagnostics']['geodesicRadiusCanonical']*scale)


def testSupportIsCovariantToVertexAndBindingOrder():
    args=list(fixture());before=support(args)
    permutation=np.random.default_rng(81).permutation(len(args[0]));inverse=np.argsort(permutation)
    args[0]=args[0][permutation];args[1]=inverse[args[1]];args[2]['vertexIndices']=inverse[args[2]['vertexIndices']]
    args[-1]=inverse[args[-1]]
    order=np.random.default_rng(83).permutation(12)
    for key in args[2]:args[2][key]=args[2][key][order]
    after=support(args)
    np.testing.assert_array_equal(after['movableMask'][inverse],before['movableMask'])


def testFixedSeedProtectionOverridesTheLocalDomain():
    args=list(fixture());first=support(args)
    seed=int(first['seedVertexIds'][0]);args[-1]=np.r_[args[-1],seed]
    after=support(args)
    assert not after['movableMask'][seed]


@pytest.mark.parametrize('kind',['lowConfidence','offImage','zeroWidth','noEligible','allFixed'])
def testUnreliableLocalSupportIsExplicitlySkippedWithoutWidening(kind):
    v,f,b,o,c,fixed=fixture();eligible=np.ones(12,bool)
    if kind=='lowConfidence':o['confidence'][36]=.49
    if kind=='offImage':o['pixels'][36]=[-1.,128.]
    if kind=='zeroWidth':o['pixels'][39]=o['pixels'][36]
    if kind=='noEligible':eligible[:6]=False
    if kind=='allFixed':fixed=np.arange(len(v))
    result=buildSemanticEyeSupport(v,f,b,o,c,eyeIbugIds=range(36,42),eligibleMask=eligible,fixedVertexIds=fixed)
    assert result['diagnostics']['status']=='unsupported'
    assert result['diagnostics']['reasons']
    assert not result['movableMask'].any()
    assert result['fixedMask'].all()


@pytest.mark.parametrize('policy',[True,{'unknown':1},{'radiusEyeWidthFraction':True},
    {'radiusEyeWidthFraction':float('nan')},{'radiusEyeWidthFraction':0},{'radiusEyeWidthFraction':1.1}])
def testInvalidPolicyFails(policy):
    with pytest.raises(ValueError):normalizeSemanticEyeSupportPolicy(policy)


def testIndependentSemanticErrorsUseTheOriginalTargetsAndBarycentricOrder():
    v,f,b,o,c,fixed=fixture();mask=np.isin(b['ibugIds'],range(36,42));snapshot=deepcopy((v,b,o,c,mask))
    result=semanticEyeFitMetrics(v,b,o,c,mask)
    assert result['count']==6
    np.testing.assert_allclose(result['errorsPixels'],.4,atol=1e-12)
    np.testing.assert_array_equal(result['ibugIds'],np.arange(36,42))
    for value,prior in ((v,snapshot[0]),(o['pixels'],snapshot[2]['pixels']),(b['vertexIndices'],snapshot[1]['vertexIndices'])):
        np.testing.assert_array_equal(value,prior)


def testDiagnosticCaptureDoesNotChangeLegacyGeometryOrDefaultOutput():
    args=fixture()
    plain,_,plainReport=runStages(args)
    captured,_,report=runStages(args,captureSemanticEyeEvidence=True)
    np.testing.assert_array_equal(captured,plain)
    assert all('stageEvidence' not in row for row in plainReport['stages'])
    for stage in report['stages'][1:]:
        evidence=stage['stageEvidence']
        assert evidence['supportMode']=='legacyGlobalCompactField'
        assert evidence['before'].shape==args[0].shape
        assert len(evidence['targetPixels'])==6
        assert evidence['afterFit']['meanErrorPixels']<evidence['beforeFit']['meanErrorPixels']
    np.testing.assert_array_equal(report['stages'][1]['stageEvidence']['acceptedAfter'],report['stages'][2]['stageEvidence']['before'])


def testRealOldSixPointSolverMovesOnlyFrozenLocalDomainsAndRetainsAlignment():
    args=fixture();original=deepcopy(args)
    final,fit,report=runStages(args,semanticEyeSupportPolicy={'radiusEyeWidthFraction':.5})
    assert np.any(final!=args[0])
    assert fit.all()
    for stage in report['stages'][1:]:
        assert stage['status']=='accepted'
        support=stage['localSupport'];evidence=stage['stageEvidence']
        np.testing.assert_array_equal(evidence['frozenSupportReferenceVertices'],args[0])
        np.testing.assert_array_equal(evidence['acceptedAfter'][support['fixedMask']],evidence['before'][support['fixedMask']])
        np.testing.assert_array_equal(evidence['targetPixels'],args[3]['pixels'][evidence['ibugIds']])
        assert evidence['afterFit']['meanErrorPixels']<evidence['beforeFit']['meanErrorPixels']
        assert evidence['afterFit']['meanErrorPixels']<.05
        assert all(attempt['localSupportGeometry']['accepted'] for attempt in stage['attempts'])
        assert all(attempt['localEyeAlignment']['accepted'] for attempt in stage['attempts'])
    np.testing.assert_array_equal(final[args[-1]],args[0][args[-1]])
    np.testing.assert_array_equal(args[0],original[0])
    np.testing.assert_array_equal(args[3]['pixels'],original[3]['pixels'])


def testInvalidSolverMotionOutsideLocalDomainIsRejectedRatherThanClamped(monkeypatch):
    args=fixture()
    def malicious(vertices,*a,**kwargs):
        candidate=vertices.copy()
        # A far cheek outside either eye domain, but not an original static row.
        candidate[18*31+15,0]+=.001
        return {'vertices':candidate,'diagnostics':{}}
    monkeypatch.setattr(refinement,'refineFrontLandmarks',malicious)
    final,_,report=runStages(args,semanticEyeSupportPolicy={})
    np.testing.assert_array_equal(final,args[0])
    for stage in report['stages'][1:]:
        assert stage['status']=='unchangedAfterGates'
        assert all(not attempt['localSupportGeometry']['accepted'] for attempt in stage['attempts'])
        assert not np.array_equal(stage['stageEvidence']['proposal'],stage['stageEvidence']['before'])


def testLocalEyeMeanAndMaximumCannotRegressWhileCombinedEyeMeanHidesIt(monkeypatch):
    args=fixture();binding=args[2]
    def away(vertices,*a,**kwargs):
        candidate=vertices.copy();targets=a[2][0]['vertexIndices'][:,0]
        candidate[targets,0]-=.004
        return {'vertices':candidate,'diagnostics':{}}
    monkeypatch.setattr(refinement,'refineFrontLandmarks',away)
    final,_,report=runStages(args,semanticEyeSupportPolicy={})
    # Initial large steps regress 0.08px; tiny old backtracks remain below the
    # established 0.05px allowance and may pass. The rejection itself is kept.
    for stage in report['stages'][1:]:
        assert not stage['attempts'][0]['localEyeAlignment']['accepted']
        assert stage['stageEvidence']['afterFit']['meanErrorPixels']<=stage['stageEvidence']['beforeFit']['meanErrorPixels']+.05*.5+1e-12


def testLocalPolicyCannotSilentlyDisableTheOriginalSixPointStages():
    with pytest.raises(ValueError,match='existing six-point'):
        runStages(fixture(),semanticEyeRefinement=False,semanticEyeSupportPolicy={})


def testPublicLocalOptionWritesFinitePortableEvidenceAndKeepsStaticIdentity(monkeypatch,tmp_path):
    v,f,b,o,c,fixed=fixture()
    originalCount=len(v)
    v=np.r_[v,np.tile([0.,0.,5.],(2795-originalCount,1))]
    static=np.zeros(2795,bool);static[fixed]=True;static[originalCount:]=True
    assets=SimpleNamespace(geometryMethod='directSurface',directArrays={},
        geometryTransfer={'restWeight':0.,'staticConstraint':'hard'},arrays={
            'vertices':v.copy(),'nicpToFullIds':np.arange(2795),'nicpFaces':f.copy(),'staticMask':static})
    monkeypatch.setattr(refinement,'loadClientLandmarks',lambda *a,**kw:{'isOriginalBinding':True})
    monkeypatch.setattr(refinement,'ibugClientLandmarks',lambda *a,**kw:b)
    monkeypatch.setattr(refinement,'_visible',lambda *a,**kw:np.ones(12,bool))
    monkeypatch.setattr(refinement,'mouthObservationQuality',lambda *a,**kw:{'reliable':True,'reasons':[]})
    monkeypatch.setattr(refinement,'_refineCrease',lambda vertices,*a,**kw:(vertices,{'status':'notPartOfThisContractTest'}))
    result=refinement.refineRegisteredClientGeometry(assets,v,f,
        {'rotation':np.eye(3),'translation':np.zeros(3),'scale':1.},[c],[{'workingSize':[256,256]}],
        photoPaths=[None],delight={},projectRoot=tmp_path,outputDirectory=tmp_path/'local',
        referenceViewIndex=0,cachedObservations={'observations':[o],'reports':[{'viewIndex':0,'status':'detected'}]},
        semanticEyeSupportPolicy={'radiusEyeWidthFraction':.5},captureSemanticEyeEvidence=True)
    saved=json.loads((tmp_path/'local/geometryRefinement.json').read_text(),
        parse_constant=lambda value:pytest.fail(f'Nonfinite serialized constant: {value}'))
    np.testing.assert_array_equal(result['vertices'][static],v[static])
    for stage in saved['stagedLandmarks']['stages'][1:]:
        assert stage['status']=='accepted'
        support=stage['localSupport'];evidence=stage['stageEvidence']
        assert np.isfinite(np.asarray(support['geodesicDistance'],float)).all()
        assert not any(support['geodesicDistanceMeasuredMask'][originalCount:])
        assert len(evidence['ibugIds'])==len(evidence['targetPixels'])==6
        assert len(evidence['acceptedAfter'])==2795
        fixedMask=np.asarray(support['fixedMask'],bool)
        np.testing.assert_array_equal(np.asarray(evidence['acceptedAfter'])[fixedMask],np.asarray(evidence['before'])[fixedMask])
    # The returned diagnostics obey the same strict JSON contract as callers.
    json.dumps(result['diagnostics'],allow_nan=False)
