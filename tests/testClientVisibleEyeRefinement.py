"""Opt-in aperture field: frozen topology support and strict full-assembly gates."""
import copy

import numpy as np
import pytest

import faceAppearance.clientVisibleEyeRefinement as visible
import faceAppearance.fastFrontEyeGeometry as fast
from faceAppearance.clientGeometrySafety import intersectionPairs


def makeInputs():
    xx, yy = np.meshgrid(np.linspace(-1., 1., 21), np.linspace(-.5, .5, 11))
    vertices = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, 5.)]
    faces = []
    for row in range(10):
        for column in range(20):
            first = row*21+column
            faces.extend([[first, first+1, first+21], [first+1, first+22, first+21]])
    gridCount = len(vertices)
    # A topologically disconnected skin layer is closer than the band radius.
    vertices = np.r_[vertices, [[-.49,-.09,5.01],[-.48,-.09,5.01],[-.49,-.08,5.01]]]
    faces.append([gridCount,gridCount+1,gridCount+2])
    skin = np.arange(len(vertices))
    skinFaceCount = len(faces)
    for center in ((-3.,-3.,6.), (3.,-3.,6.), (-4.,4.,6.), (4.,4.,6.)):
        first = len(vertices)
        vertices = np.r_[vertices,np.asarray(center)+[[0.,0.,0.],[.1,0.,0.],[0.,.1,0.]]]
        faces.append([first,first+1,first+2])
    faces = np.asarray(faces,np.int64)
    scene = dict(vertices=vertices, faces=faces, nicpToFullIds=skin,
        materialIds=np.r_[np.zeros(skinFaceCount+2,np.int64),2,3],
        uvs=vertices[:,:2].copy(), uvFaces=faces.copy(),
        materials=[{'alphaMode':'OPAQUE'}]*3+[{'alphaMode':'MASK'}],alphaTextures={})
    corners = np.arange(skinFaceCount,skinFaceCount+2)
    camera = dict(intrinsics=np.array([[100.,0.,128.],[0.,100.,128.],[0.,0.,1.]]),
        rotation=np.eye(3), translation=np.zeros(3))
    eyes, measurement = {}, {'eyes':{},'diagnostics':{}}
    for name, center, column in [('imageLeft',118.,5),('imageRight',138.,14)]:
        eyes[name] = dict(upper=np.array([[center-6.,128.],[center-2.,125.],[center+2.,125.],[center+6.,128.]]),
            lower=np.array([[center-6.,128.],[center-2.,131.],[center+2.,131.],[center+6.,128.]]))
        aperture = np.zeros((9,9),bool);aperture[3:6,3:6] = True
        bindable = ~aperture
        faceIds = np.full((9,9), 2*(4*20+column),np.int64)
        measurement['eyes'][name] = dict(status='accepted',roiMasks=dict(eye=aperture,
            eligibleBindableSkin=bindable,faceIds=faceIds),diagnostics={})
    return scene, camera, eyes, measurement, corners


def supportFor(inputs, **kwargs):
    scene,camera,eyes,measurement,corners=inputs
    return visible.buildClientVisibleEyeSupport(scene,camera,eyes,measurement,cornerFaceIds=corners,**kwargs)


def testTopologySupportDoesNotJumpToNearbyDisconnectedSkinOrAccessories():
    inputs=makeInputs();scene=inputs[0]
    support=supportFor(inputs)
    assert support['diagnostics']['status']=='supported'
    assert support['movableMask'].any()
    assert not support['movableMask'][231:234].any()
    assert not support['movableMask'][-6:].any()  # Eye and alpha card.
    assert support['movableMask'][np.unique(scene['faces'][inputs[-1]])].all()
    assert not support['movableMask'][0]  # Far cheek/brow stays fixed.
    for eye in support['eyes'].values():
        assert eye['geodesicRadius']==pytest.approx(.18)
        assert np.all(eye['distance'][eye['movableSkinVertexIds']]<=.18+1e-12)
        assert len(eye['movableSkinVertexIds'])>len(eye['seedVertexIds'])
    assert support['diagnostics']['candidateDependentSupport'] is False


def testExplicitProtectionWinsEvenAtApertureSeedAndCanthus():
    inputs=makeInputs();initial=supportFor(inputs)
    seed=int(initial['eyes']['imageLeft']['seedVertexIds'][0])
    corner=int(initial['cornerVertexIds'][0])
    support=supportFor(inputs,fixedVertexIds=[seed],protectedVertexIds=[corner])
    assert not support['movableMask'][[seed,corner]].any()
    assert set([seed,corner]).issubset(support['fixedVertexIds'])


@pytest.mark.parametrize('scale',[.1,10.])
def testSupportScalesWithFrozenCanonicalFrameWithoutWorldRadiusConstants(scale):
    inputs=makeInputs();before=supportFor(inputs)
    changed=copy.deepcopy(inputs)
    shift=np.array([2.,-3.,7.])
    changed[0]['vertices']=changed[0]['vertices']*scale+shift
    changed[1]['translation']=-shift
    after=supportFor(changed)
    np.testing.assert_array_equal(after['movableMask'],before['movableMask'])
    for name in after['eyes']:
        assert after['eyes'][name]['eyeWidth']==pytest.approx(before['eyes'][name]['eyeWidth']*scale)


def testSupportIsCovariantToVertexReindexing():
    inputs=makeInputs();before=supportFor(inputs)
    scene=inputs[0]
    permutation=np.random.default_rng(71).permutation(len(scene['vertices']))
    inverse=np.argsort(permutation)
    scene['vertices']=scene['vertices'][permutation]
    scene['faces']=inverse[scene['faces']]
    scene['nicpToFullIds']=inverse[scene['nicpToFullIds']]
    after=supportFor(inputs)
    np.testing.assert_array_equal(after['movableMask'][inverse],before['movableMask'])


@pytest.mark.parametrize('policy',[{'unknown':1},{'geodesicBandEyeWidthFraction':True},
    {'geodesicBandEyeWidthFraction':float('nan')},{'geodesicBandEyeWidthFraction':0},
    {'geodesicBandEyeWidthFraction':.6},{'rounds':True},{'rounds':3},
    {'iterationsPerRound':21},{'iterationsPerRound':1.5}])
def testUnsupportedPolicyRejected(policy):
    with pytest.raises(ValueError): supportFor(makeInputs(),policy=policy)


@pytest.mark.parametrize('field',['fixedVertexIds','protectedVertexIds'])
def testInvalidProtectionIndicesRejected(field):
    for ids in ([0,0],[-1],[100000],[1.5]):
        with pytest.raises(ValueError): supportFor(makeInputs(),**{field:ids})


def testMissingOneEyeSupportSkipsWithoutInvokingSolver(monkeypatch):
    inputs=makeInputs();scene,camera,eyes,measurement,corners=inputs
    measurement['eyes']['imageLeft'].pop('roiMasks')
    monkeypatch.setattr(visible,'extractVisibleEyeContours',lambda **kw:measurement)
    monkeypatch.setattr(visible,'refineFastFrontEyes',lambda *a,**kw:pytest.fail('No support must skip fitting'))
    result=visible.refineClientVisibleEyes(scene,camera,eyes,imageShape=(256,256),cornerFaceIds=corners)
    assert result['diagnostics']['status']=='skippedInsufficientVisibleSupport'
    np.testing.assert_array_equal(result['vertices'],scene['vertices'])
    assert not result['support']['movableMask'].any()


def testRejectedCameraCannotLoseItsStatusThroughCameraNormalization():
    inputs=makeInputs();inputs[1]['valid']=False
    with pytest.raises(ValueError,match='rejected photo camera'): supportFor(inputs)
    scene,camera,eyes,_,corners=inputs
    with pytest.raises(ValueError,match='rejected photo camera'):
        visible.refineClientVisibleEyes(scene,camera,eyes,imageShape=(256,256),cornerFaceIds=corners)


def rasterInputs():
    vertices,faces,materials,skin,corners=[],[],[],[],[]
    angle=np.linspace(0.,2*np.pi,32,endpoint=False)
    octahedron=np.array([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.],[0.,0.,1.],[0.,0.,-1.]])
    octaFaces=np.array([[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]])
    eyes={}
    for name,center in [('imageLeft',-.7),('imageRight',.7)]:
        first=len(vertices)
        for width,height in ((.60,.35),(.25,.10)):
            vertices.extend(np.c_[center+width*np.cos(angle),height*np.sin(angle),np.full(32,4.5)])
        skin.extend(range(first,len(vertices)))
        for i in range(32):
            j=(i+1)%32
            # The camera looks along +z; visible head faces point toward -z.
            faces.extend([[first+i,first+32+i,first+j],[first+j,first+32+i,first+32+j]])
            materials.extend([0,0])
        first=len(vertices);vertices.extend(octahedron*.4+[center,0.,5.])
        faces.extend(octaFaces+first);materials.extend([2]*8)
        first=len(vertices);direction=1 if center<0 else -1
        vertices.extend(np.array([[.20,-.025,0.],[.24,0.,0.],[.20,.025,0.]])*[direction,1.,1.]+[center,0.,4.49])
        corners.append(len(faces));faces.append([first,first+1,first+2]);materials.append(0)
        arc=np.linspace(np.pi,0.,12)
        eyes[name]=dict(upper=np.c_[128+(center+.25*np.cos(arc))/4.5*100,128-.11*np.sin(arc)/4.5*100],
            lower=np.c_[128+(center+.25*np.cos(arc))/4.5*100,128+.11*np.sin(arc)/4.5*100])
    vertices=np.asarray(vertices);faces=np.asarray(faces,np.int64)
    scene=dict(vertices=vertices,faces=faces,materialIds=np.asarray(materials,np.int64),
        nicpToFullIds=np.asarray(skin,np.int64),uvs=vertices[:,:2],uvFaces=faces.copy(),
        materials=[{'alphaMode':'OPAQUE'}]*3,alphaTextures={})
    camera=dict(intrinsics=np.array([[100.,0.,128.],[0.,100.,128.],[0.,0.,1.]]),rotation=np.eye(3),translation=np.zeros(3))
    return scene,camera,eyes,np.asarray(corners,np.int64)


def testCompleteSyntheticAssemblyUsesRealRasterBindingSolverAndSafety():
    scene,camera,eyes,corners=rasterInputs()
    original=copy.deepcopy(scene)
    result=visible.refineClientVisibleEyes(scene,camera,eyes,imageShape=(256,256),cornerFaceIds=corners,
        policy={'rounds':1,'iterationsPerRound':1})
    assert result['diagnostics']['fastSolverInvoked']
    assert result['support']['diagnostics']['status']=='supported'
    assert set(result['beforeContours']['eyes'])=={'imageLeft','imageRight'}
    assert all(eye['roiMasks']['eye'].any() for eye in result['beforeContours']['eyes'].values())
    assert result['candidateProposals']
    assert all('strict' in proposal for proposal in result['candidateProposals'])
    np.testing.assert_array_equal(result['vertices'][result['support']['fixedMask']],original['vertices'][result['support']['fixedMask']])
    assert not (intersectionPairs(result['vertices'],scene['faces'])-intersectionPairs(original['vertices'],scene['faces']))
    for key in ('vertices','faces','uvs','uvFaces'):
        np.testing.assert_array_equal(scene[key],original[key])


def fakeFit(monkeypatch, inputs, *, corruptFixed=False, rejectedContours=False, emptyContours=False,
            lastRejected=False):
    scene,camera,eyes,measurement,corners=inputs
    support=supportFor(inputs)
    moving=int(support['eyes']['imageLeft']['seedVertexIds'][0])
    proposed=scene['vertices'].copy();proposed[moving,0]+=.0005
    if corruptFixed: proposed[-1,0]+=.0005
    def fit(vertices,faces,materialIds,camera,photoEyes,**kwargs):
        first=kwargs['candidateGate'](proposed,dict(round=1,scale=1.,geometry={'accepted':True}))
        stages=[dict(round=1,status='accepted' if first['accepted'] else 'noImprovingSafeCandidate',
            selectedScale=1.,attempts=[dict(scale=1.)])]
        if lastRejected:
            bad=proposed.copy();bad[-1,0]+=.0005
            second=kwargs['candidateGate'](bad,dict(round=2,scale=.25,geometry={'accepted':True}))
            assert not second['accepted']
            stages.append(dict(round=2,status='noImprovingSafeCandidate',attempts=[dict(scale=.25)]))
        contours=copy.deepcopy(measurement)
        if rejectedContours: contours['eyes']['imageLeft']['status']='rejected'
        if emptyContours: contours['eyes']={}
        return dict(vertices=proposed,afterContours=contours,textureContours=measurement,
            diagnostics=dict(status='acceptedLocalExperiment',stages=stages))
    monkeypatch.setattr(visible,'extractVisibleEyeContours',lambda **kw:measurement)
    monkeypatch.setattr(visible,'refineFastFrontEyes',fit)
    return proposed


def testWrapperReturnsSafeGeometryAndPreservesAllInputIdentities(monkeypatch):
    inputs=makeInputs();original=copy.deepcopy(inputs)
    proposal=fakeFit(monkeypatch,inputs)
    scene,camera,eyes,_,corners=inputs
    result=visible.refineClientVisibleEyes(scene,camera,eyes,imageShape=(256,256),cornerFaceIds=corners)
    assert result['diagnostics']['accepted']
    np.testing.assert_array_equal(result['vertices'],proposal)
    assert result['candidateProposals'][0]['strict']['accepted']
    for key in ('vertices','faces','materialIds','nicpToFullIds','uvs','uvFaces'):
        np.testing.assert_array_equal(scene[key],original[0][key])
    for key in camera: np.testing.assert_array_equal(camera[key],original[1][key])
    for name in eyes:
        for key in eyes[name]: np.testing.assert_array_equal(eyes[name][key],original[2][name][key])
    np.testing.assert_array_equal(result['vertices'][result['support']['fixedMask']],scene['vertices'][result['support']['fixedMask']])


def testIndependentFinalCheckRejectsSolverThatReturnsMovedAccessory(monkeypatch):
    inputs=makeInputs();fakeFit(monkeypatch,inputs,corruptFixed=True)
    scene,camera,eyes,_,corners=inputs
    result=visible.refineClientVisibleEyes(scene,camera,eyes,imageShape=(256,256),cornerFaceIds=corners)
    assert not result['diagnostics']['accepted']
    assert not result['candidateProposals'][0]['strict']['accepted']
    np.testing.assert_array_equal(result['vertices'],scene['vertices'])
    assert not np.array_equal(result['rawCandidateVertices'],result['vertices'])


@pytest.mark.parametrize('kw',[{'rejectedContours':True},{'emptyContours':True}])
def testGoodTextureProxyCannotSubstituteForCompleteApertureAcceptance(monkeypatch,kw):
    inputs=makeInputs();fakeFit(monkeypatch,inputs,**kw)
    scene,camera,eyes,_,corners=inputs
    result=visible.refineClientVisibleEyes(scene,camera,eyes,imageShape=(256,256),cornerFaceIds=corners)
    assert not result['diagnostics']['accepted']
    assert 'unreliableCompleteCandidateContours' in result['diagnostics']['strict']['reasons']
    assert result['textureContours'] is None
    np.testing.assert_array_equal(result['vertices'],scene['vertices'])


def testLastRejectedProposalDoesNotReplaceLastAcceptedSolverIterate(monkeypatch):
    inputs=makeInputs();proposal=fakeFit(monkeypatch,inputs,lastRejected=True)
    scene,camera,eyes,_,corners=inputs
    result=visible.refineClientVisibleEyes(scene,camera,eyes,imageShape=(256,256),cornerFaceIds=corners)
    assert result['diagnostics']['accepted']
    np.testing.assert_array_equal(result['vertices'],proposal)
    np.testing.assert_array_equal(result['solverVertices'],proposal)
    assert not np.array_equal(result['rawCandidateVertices'],proposal)
    assert len(result['candidateProposals'])==2
    assert result['candidateProposals'][0]['selectedByFastRound']
    assert not result['candidateProposals'][1]['selectedByFastRound']


def collisionInputs(material):
    original=np.array([[-.2,0.,5.],[-.1,0.,5.],[-.2,.2,5.],
        [10.,0.,5.],[11.,0.,5.],[10.,1.,5.],
        [0.,-.1,4.9],[0.,.3,4.9],[0.,.1,5.1]])
    faces=np.arange(9).reshape(-1,3)
    scene=dict(vertices=original,faces=faces,nicpToFullIds=np.arange(6),
        materialIds=np.array([0,0,material]),materials=[{'alphaMode':'OPAQUE'}]*3+[{'alphaMode':'MASK'}])
    camera=dict(intrinsics=np.eye(3),rotation=np.eye(3),translation=np.zeros(3))
    candidate=original.copy();candidate[:3,0]+=.15
    return scene,camera,candidate


@pytest.mark.parametrize('material',[0,2,3])
def testStrictFullCollisionDetectsSkinEyesAndAlphaCards(material):
    scene,camera,candidate=collisionInputs(material)
    baseline=intersectionPairs(scene['vertices'],scene['faces'])
    assert not baseline
    checks=visible._strictCandidateChecks(scene['vertices'],candidate,scene,camera,
        {'fixedVertexIds':np.arange(3,9)},baseline)
    assert checks['geometry']['accepted']  # Rigid translation, unchanged depth/edge/area.
    assert not checks['accepted']
    assert checks['fullAssembly']['newPairs']==[(0,2)]
    assert checks['fullAssembly']['excludedMaterialIds']==[]


def testUnchangedBaselineContactsAreNotNewCollisions():
    scene,camera,candidate=collisionInputs(2)
    baseline=intersectionPairs(candidate,scene['faces'])
    assert baseline=={(0,2)}
    checks=visible._strictCandidateChecks(candidate,candidate.copy(),scene,camera,
        {'fixedVertexIds':np.arange(9)},baseline)
    assert checks['accepted']
    assert checks['fullAssembly']['newPairs']==[]


def mockFastInternals(monkeypatch, *, oldGeometryAccepted=True):
    original=np.array([[0.,0.,5.],[1.,0.,5.],[0.,1.,5.],
        [2.,0.,5.],[2.1,0.,5.],[2.,.1,5.]])
    def measure(vertices,*a,**kw):
        return {'marker':float(np.asarray(vertices)[0,0]),'eyes':{'imageLeft':{},'imageRight':{}}}
    def score(measured,*a,**kw):
        value=1.-20*measured['marker']
        return {'meanSymmetricBoundaryDistanceOriginalPixels':value,
            'eyes':{name:dict(symmetricBoundaryMeanOriginalPixels=value,
                symmetricBoundaryP95OriginalPixels=value,intersectionOverUnion=.5,
                visibleComponentCount=1,secondaryComponentAreaFraction=0.) for name in ('imageLeft','imageRight')}}
    def fit(vertices,*a,**kw):
        proposal=vertices.copy();proposal[:,0]+=.01
        return dict(vertices=proposal,diagnostics={})
    monkeypatch.setattr(fast,'extractVisibleEyeContours',measure)
    monkeypatch.setattr(fast,'evaluateApertureMask',score)
    monkeypatch.setattr(fast,'_constraints',lambda *a,**kw:([{}],{}))
    monkeypatch.setattr(fast,'refineFrontLandmarks',fit)
    monkeypatch.setattr(fast,'checkFrontGeometry',lambda *a,**kw:{'accepted':oldGeometryAccepted,'reasons':[]})
    monkeypatch.setattr(fast,'_eyeContactSafety',lambda *a,**kw:{'accepted':True})
    monkeypatch.setattr(fast,'_newIntersections',lambda *a,**kw:{'newPairCount':0})
    return original


def invokeFast(original,**kwargs):
    return fast.refineFastFrontEyes(original,np.array([[0,1,2],[3,4,5]]),np.array([0,0]),
        dict(rotation=np.eye(3)),{'imageLeft':{},'imageRight':{}},skinVertexIds=np.arange(3),
        cornerFaceIds=np.array([1]),imageShape=(256,256),rounds=1,**kwargs)


def testOptionalFastGateBacktracksAndRecordsEveryProposal(monkeypatch):
    original=mockFastInternals(monkeypatch)
    attempts=[]
    def gate(vertices,context):
        assert not vertices.flags.writeable
        attempts.append((vertices.copy(),copy.deepcopy(context)))
        return {'accepted':context['scale']==.25}
    result=invokeFast(original,candidateGate=gate)
    assert [row[1]['scale'] for row in attempts]==[1.,.5,.25]
    assert result['diagnostics']['stages'][0]['selectedScale']==.25
    np.testing.assert_allclose(result['vertices']-original,np.tile([.0025,0.,0.],(len(original),1)))


def testFastOptionalGateCannotWaiveOrMutateOldGeometryRejection(monkeypatch):
    original=mockFastInternals(monkeypatch,oldGeometryAccepted=False)
    def gate(vertices,context):
        context['geometry']['accepted']=True
        return {'accepted':True}
    result=invokeFast(original,candidateGate=gate)
    np.testing.assert_array_equal(result['vertices'],original)
    assert all(not row['geometry']['accepted'] for row in result['diagnostics']['stages'][0]['attempts'])


def testOmittingFastGateRetainsOriginalSelection(monkeypatch):
    original=mockFastInternals(monkeypatch)
    result=invokeFast(original)
    assert result['diagnostics']['stages'][0]['selectedScale']==1.
    assert 'externalCandidateGate' not in result['diagnostics']['stages'][0]['attempts'][0]
    np.testing.assert_allclose(result['vertices']-original,np.tile([.01,0.,0.],(len(original),1)))


@pytest.mark.parametrize('value',[None,True,{}, {'accepted':1}])
def testFastCandidateGateRequiresExplicitBooleanReport(monkeypatch,value):
    original=mockFastInternals(monkeypatch)
    with pytest.raises(ValueError,match='boolean accepted'):
        invokeFast(original,candidateGate=lambda *a:value)
