"""Explicit initial-state replay preserves identity and recomputes mesh evidence."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import faceAppearance.clientGeometryRefinement as refinement


@pytest.fixture
def scene(monkeypatch, tmp_path):
    rest=np.zeros((2795,3),float)
    rest[:6]=[[0,0,0],[.4,0,0],[0,.4,0],[1,0,0],[1.4,0,0],[1,.4,0]]
    faces=np.array([[0,1,2],[3,4,5]])
    static=np.zeros(2795,bool);static[:3]=True
    angle=.23;c,s=np.cos(angle),np.sin(angle)
    transform={'rotation':np.array([[c,-s,0],[s,c,0],[0,0,1]]),
               'translation':np.array([.13,-.27,3.1]),'scale':.73}
    canonical=(rest@transform['rotation'])*transform['scale']+transform['translation']
    calls=[]
    def deform(geometry):
        calls.append('registration')
        return {'vertices':rest.copy()},None,{'templateToUff':deepcopy(transform)}
    assets=SimpleNamespace(geometryMethod='directSurface',directArrays={},
        geometryTransfer={'restWeight':0.,'staticConstraint':'hard'},
        arrays={'vertices':rest,'nicpToFullIds':np.arange(2795),
                'nicpFaces':faces,'staticMask':static},_deform=deform)
    binding={'ibugIds':np.array([48]),'vertexIndices':np.array([[3,4,5]]),
             'barycentricWeights':np.array([[.3,.3,.4]]),'validMask':np.ones(1,bool)}
    monkeypatch.setattr(refinement,'loadClientLandmarks',lambda *a,**k:{'isOriginalBinding':True})
    monkeypatch.setattr(refinement,'ibugClientLandmarks',lambda *a,**k:binding)
    cameras=[{'valid':True,'intrinsics':[[100,0,128],[0,100,128],[0,0,1]],
              'rotation':np.eye(3).tolist(),'translation':[0,0,0]}]
    inputs=[{'workingSize':[256,256],'sha256':'a'*64}]
    cache={'observations':[None],'reports':[{'viewIndex':0,'status':'unavailable'}]}
    arguments=dict(photoPaths=[None],delight={},projectRoot=tmp_path,
                   outputDirectory=tmp_path/'result',referenceViewIndex=None,
                   cachedObservations=cache)
    return SimpleNamespace(assets=assets,rest=rest,vertices=canonical,faces=faces,
        transform=transform,cameras=cameras,inputs=inputs,cache=cache,arguments=arguments,
        calls=calls,static=static)


def run(scene, **changes):
    arguments={**scene.arguments,**changes}
    return refinement.refineRegisteredClientGeometry(scene.assets,scene.vertices,scene.faces,
        scene.transform,scene.cameras,scene.inputs,**arguments)


def testExplicitCanonicalInputIsNotRegisteredOrRoundTripped(scene):
    # A one-ULP input difference demonstrates that accepted static coordinates
    # survive exactly instead of being replaced by another matrix product.
    scene.vertices[0,0]=np.nextafter(scene.vertices[0,0],np.inf)
    original=scene.vertices.copy()
    result=run(scene)
    assert scene.calls==[]
    np.testing.assert_array_equal(result['vertices'],original)
    np.testing.assert_array_equal(result['mouthState']['beforeCanonical'],original)
    np.testing.assert_array_equal(result['mouthState']['afterCanonical'],original)
    np.testing.assert_array_equal(result['templateVertices'][scene.static],scene.rest[scene.static])
    assert result['diagnostics']['directRegistration']['initialRegistrationProvided']
    with np.load(scene.arguments['outputDirectory']/'clientGeometry.npz') as saved:
        np.testing.assert_array_equal(saved['vertices'],original)
    np.testing.assert_array_equal(scene.vertices,original)


def testLegacyDefaultAndExplicitEntryShareUnchangedNoReferenceResult(scene,monkeypatch,tmp_path):
    monkeypatch.setattr(refinement,'_observations',lambda *a:([None],deepcopy(scene.cache['reports'])))
    legacy=refinement.refineClientGeometry(scene.assets,{},scene.cameras,scene.inputs,
        photoPaths=[None],delight={},projectRoot=tmp_path,outputDirectory=tmp_path/'legacy',
        referenceViewIndex=None)
    supplied=run(scene)
    for key in ('vertices','templateVertices','directTemplateVertices','faces'):
        # The supplied canonical input must be inverted for template output;
        # canonical output and static template rows remain exact.
        if key in ('templateVertices','directTemplateVertices'):
            np.testing.assert_allclose(legacy[key],supplied[key],rtol=0,atol=1e-15)
        else:np.testing.assert_array_equal(legacy[key],supplied[key])
    assert scene.calls==['registration']
    assert 'initialRegistrationProvided' not in legacy['diagnostics']['directRegistration']
    before=deepcopy(legacy['diagnostics']);after=deepcopy(supplied['diagnostics'])
    before.pop('seconds');after.pop('seconds')
    after['directRegistration'].pop('initialRegistrationProvided')
    assert before==after


def validCache(scene):
    return {'observations':[{'pixels':np.full((68,2),128.),'confidence':np.ones(68)*.9,
                             'workingRgb':np.full((256,256,3),120,np.uint8)}],
            'reports':[{'viewIndex':0,'status':'detected','source':'providedWorkingPhotograph',
                        'sourceSha256':'b'*64}]}


def testVisibilityAndFreshStagesReceiveEachExactInputNotPreviousDelta(scene,monkeypatch,tmp_path):
    seen=[];visible=[]
    monkeypatch.setattr(refinement,'_visible',lambda vertices,*a:(visible.append(vertices.copy()) or np.ones(1,bool)))
    monkeypatch.setattr(refinement,'mouthObservationQuality',lambda *a:{'reliable':True,'reasons':[]})
    def stages(before,*args):
        seen.append(before.copy())
        after=before.copy()
        # Deliberately depends on the incoming mesh, making a cached historical
        # increment distinguishable from executing this stage again.
        after[4,1]+=before[4,0]*.001
        args[4][0]['pixels'][0,0]=42.
        return after,np.ones(1,bool),{'stages':[]}
    monkeypatch.setattr(refinement,'_stagedLandmarks',stages)
    def crease(current,*a,**k):
        assert np.array_equal(k['cumulativeReference'],seen[-1])
        return current,{'status':'skipped'}
    monkeypatch.setattr(refinement,'_refineCrease',crease)
    cache=validCache(scene);sourceCache=deepcopy(cache)
    first=run(scene,cachedObservations=cache,referenceViewIndex=0,outputDirectory=tmp_path/'first')
    scene.vertices[4,0]+=.005
    second=run(scene,cachedObservations=cache,referenceViewIndex=0,outputDirectory=tmp_path/'second')
    for index,result in enumerate((first,second)):
        np.testing.assert_array_equal(visible[index],seen[index])
        np.testing.assert_array_equal(result['mouthState']['beforeCanonical'],seen[index])
        np.testing.assert_array_equal(result['vertices'][scene.static],seen[index][scene.static])
        assert result['diagnostics']['combinedGeometryCheck']['accepted']
        assert result['diagnostics']['newIntersectionPairs']==0
    assert first['vertices'][4,1]-seen[0][4,1] != second['vertices'][4,1]-seen[1][4,1]
    np.testing.assert_array_equal(cache['observations'][0]['pixels'],sourceCache['observations'][0]['pixels'])
    assert scene.calls==[]


@pytest.mark.parametrize('mutation,message',[
    ('faceOrder','face order'),('winding','face order'),('static','transformed rest'),
    ('nonfinite','finite floating'),('integer','finite floating'),('reflection','proper'),
    ('nonrotation','proper'),('scale','proper'),('translation','proper')])
def testInvalidGeometryIdentityAndFrameAreRejectedBeforeStages(scene,mutation,message):
    if mutation=='faceOrder':scene.faces=scene.faces[::-1]
    elif mutation=='winding':scene.faces=scene.faces[:,::-1]
    elif mutation=='static':scene.vertices[0,0]+=.0001
    elif mutation=='nonfinite':scene.vertices[3,0]=np.nan
    elif mutation=='integer':scene.vertices=scene.vertices.astype(int)
    elif mutation=='reflection':scene.transform['rotation']=np.diag([-1,1,1])
    elif mutation=='nonrotation':scene.transform['rotation']=np.eye(3)*1.01
    elif mutation=='scale':scene.transform['scale']=-1.
    elif mutation=='translation':scene.transform['translation']=[0,0,np.inf]
    with pytest.raises(ValueError,match=message):run(scene)
    assert scene.calls==[]


def testRegistrationDiagnosticFrameMismatchIsRejected(scene):
    previous=deepcopy(scene.transform);previous['translation'][0]+=.01
    with pytest.raises(ValueError,match='disagree'):
        run(scene,registrationDiagnostics={'templateToUff':previous})


@pytest.mark.parametrize('mutation,message',[
    ('count','count'),('view','ordered'),('pixels','68-point'),('confidence','68-point'),
    ('rgbSize','workingSize'),('rgbEncoding','workingSize'),('camera','valid current camera')])
def testInvalidSharedObservationSchemaFailsExplicitly(scene,mutation,message):
    cache=validCache(scene)
    if mutation=='count':cache['reports']=[]
    elif mutation=='view':cache['reports'][0]['viewIndex']=1
    elif mutation=='pixels':cache['observations'][0]['pixels'][0,0]=np.nan
    elif mutation=='confidence':cache['observations'][0]['confidence']=np.ones(67)
    elif mutation=='rgbSize':cache['observations'][0]['workingRgb']=np.zeros((128,128,3),np.uint8)
    elif mutation=='rgbEncoding':cache['observations'][0]['workingRgb']=np.zeros((256,256,3),float)
    elif mutation=='camera':scene.cameras[0]['valid']=False
    with pytest.raises(ValueError,match=message):run(scene,cachedObservations=cache)


def testPublicObservationEntryPreservesViewOrderAndRejectedCamera(scene,monkeypatch):
    scene.cameras[0]['valid']=False
    def observe(project,photoPaths,workingPaths,inputs,cameras):
        assert cameras==[None]
        assert photoPaths==[None] and workingPaths==['working.png']
        return [None],[{'viewIndex':0,'status':'unavailable'}]
    monkeypatch.setattr(refinement,'_observations',observe)
    result=refinement.observeClientGeometryPhotos('.', [None],scene.inputs,scene.cameras,
                                                  workingPhotoPaths=['working.png'])
    assert result==scene.cache


def testVisibleEyeReplacementSkipsOnlySemanticEyeSolvers(scene,monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(refinement,'_visible',lambda *a:np.ones(1,bool))
    monkeypatch.setattr(refinement,'mouthObservationQuality',lambda *a:{'reliable':True,'reasons':[]})
    def stages(before,*args,**options):
        calls.append(options)
        return before.copy(),np.ones(1,bool),{'stages':[]}
    monkeypatch.setattr(refinement,'_stagedLandmarks',stages)
    monkeypatch.setattr(refinement,'_refineCrease',lambda current,*a,**k:(current,{'status':'retained'}))
    result=run(scene,referenceViewIndex=0,cachedObservations=validCache(scene),semanticEyeRefinement=False)
    assert calls==[{'semanticEyeRefinement':False}]
    np.testing.assert_array_equal(result['vertices'][scene.static],scene.vertices[scene.static])
    with pytest.raises(ValueError,match='boolean'):run(scene,semanticEyeRefinement=1)
