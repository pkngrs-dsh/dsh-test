"""Independent artifact/coordinate/support audit for local fast eye experiments."""
from __future__ import annotations
import argparse,io,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.colmapTexture import _destinationUvLookup
from faceAppearance.surfaceVisibility import ProjectedSurfaceVisibility
from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs,extractVisibleEyeContours


def require(value,message):
    if not value:raise ValueError(message)

def loadnpz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def canonical(asset):
    s=asset.manifest['assemblyDiagnostics']['templateToUff']
    return asset.mesh['vertices']*100@np.asarray(s['rotation'])*float(s['scale'])+s['translation']

def photoOutside(xy,polygon,margin):
    inside=np.zeros(len(xy),bool); distance2=np.full(len(xy),np.inf)
    for first,last in zip(polygon,np.roll(polygon,-1,axis=0)):
        v=last-first; f=np.clip(((xy-first)*v).sum(1)/max(v@v,1e-20),0,1)
        distance2=np.minimum(distance2,((xy-first-f[:,None]*v)**2).sum(1))
        if abs(v[1])>1e-15:
            inside^=((first[1]>xy[:,1])!=(last[1]>xy[:,1]))&(xy[:,0]<first[0]+(xy[:,1]-first[1])*v[0]/v[1])
    return ~inside&(distance2>=margin**2-1e-8)


def audit(directory):
    report=json.loads((directory/'report.json').read_text())
    require(report['status']=='completeLocalExperiment','Experiment incomplete')
    source=Path(report['paths']['source']); observations=Path(report['paths']['observations'])
    original=_load(Path(report['paths']['archive']))
    assets={n:_load(directory/n/'legacyAsset.zip') for n in ('baseline','geometry','combined')}
    g=loadnpz(directory/'geometry.npz'); skin=g['skinVertexIds']; before,after=g['before'],g['after']
    require(np.isfinite(before).all() and np.isfinite(after).all(),'Nonfinite geometry')
    require(np.allclose(canonical(original),before,rtol=0,atol=1e-12),'Original coordinate frame mismatch')
    arrays=AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1').arrays
    require(np.array_equal(skin,arrays['nicpToFullIds']),'NICP index map mismatch')
    for name,asset in assets.items():
        require(np.allclose(canonical(asset),before if name=='baseline' else after,rtol=0,atol=1e-12),name+' ZIP coordinate mismatch')
        require(all(np.array_equal(asset.mesh[k],original.mesh[k]) for k in ('faces','uvs','uvFaces','materialIds')),name+' topology/UV/material mismatch')
        require(all(asset.payloads[k]==original.payloads[k] for k in original.payloads if k not in ('face_nicp.obj','head_color.png','face_nicp.png')),name+' other texture/material bytes changed')
    require(np.array_equal(before[g['fixedVertexIds']],after[g['fixedVertexIds']]),'Protected skin moved')
    cornerVertices=np.unique(g['faces'][g['cornerFaceIds']]); allowed=np.unique(np.r_[skin,cornerVertices])
    frozen=np.ones(len(before),bool);frozen[allowed]=False
    require(np.array_equal(before[frozen],after[frozen]),'Unrelated eyes/teeth/lash vertices moved')
    if report['geometry'].get('status')=='noImprovingSafeCandidate':
        require(np.array_equal(before,after),'Rejected geometry did not preserve original')
    require(assets['geometry'].payloads['face_nicp.obj']==assets['combined'].payloads['face_nicp.obj'],'Texture branch moved geometry')
    require(assets['baseline'].payloads['face_nicp.obj']==original.payloads['face_nicp.obj'],'Baseline OBJ bytes changed')
    payload=lambda a:np.asarray(Image.open(io.BytesIO(a.payloads['head_color.png']))).copy()
    originalP,beforeP,afterP=map(payload,(original,assets['geometry'],assets['combined']))
    textureStatus=next(row for row in report['branches'] if row['id']=='combined')['texture']
    if not (directory/'textureProvenance.npz').exists():
        require(np.array_equal(beforeP,afterP),'Skipped texture still changed colors')
        return {'caseId':report['caseId'],'status':'passedTextureSkipped','reason':textureStatus,'geometryFinite':True,'topologyUvsFixed':True}
    p=loadnpz(directory/'textureProvenance.npz'); alpha=p['alpha']; modified=(alpha>0)|p['filterPadding']
    require(np.isfinite(alpha).all() and (alpha>=0).all() and (alpha<=1).all(),'Invalid alpha')
    require(np.isfinite(p['linearRGB']).all() and (p['linearRGB']>=0).all() and (p['linearRGB']<=1).all(),'Invalid texture RGB')
    require(np.array_equal(beforeP[~modified],afterP[~modified]),'Texture outside edit changed')
    require(not (modified&~p['writableMask']).any() and not(modified&p['aliasMask']).any(),'Protected or aliased UV edited')
    require(not(p['observedMask']&p['estimatedMask']).any() and np.array_equal(modified,p['observedMask']|p['estimatedMask']),'Provenance masks mismatch')
    # Recompute full template chart exclusions, independent of sparse lookup.
    size=afterP.shape[0]; member=np.zeros(len(before),bool);member[skin]=True
    other=(original.mesh['materialIds']==0)&~member[original.mesh['faces']].all(1)
    excluded=dict(vertices=before,faces=original.mesh['faces'][other],uvs=original.mesh['uvs'],uvFaces=original.mesh['uvFaces'][other],uvConvention='topLeftPixelEdges')
    protected=binary_dilation(_destinationUvLookup(excluded,size)[0]['validMask'],iterations=2)
    require(not(modified&protected).any() and np.array_equal(originalP[protected],afterP[protected]),'Accessory UV protection changed')
    query=p['sourceXY'][modified]
    require(np.isfinite(query).all(),'Nonfinite accepted source')
    sourceCache=loadnpz(Path(report['paths']['delight'])/'delight.npz'); valid=sourceCache['validMasks'][0]
    floors=np.floor(query).astype(int);h,w=valid.shape
    require((floors>=0).all() and (floors<[w-1,h-1]).all(),'Source out of bounds')
    sourcePixels=np.unique(np.concatenate([floors+[dx,dy] for dy in (0,1) for dx in (0,1)]),axis=0)
    require((valid[sourcePixels[:,1],sourcePixels[:,0]]>=.999).all(),'Invalid real source support')
    obs=loadnpz(observations/'view00.npz')['eyelidCurvesWorking']
    for eye in obs:
        polygon=np.concatenate([eye[1],eye[0][::-1]])
        require(photoOutside(sourcePixels,polygon,textureStatus['sourceSafetyWorkingPixels']).all(),'Source footprint touches photographic eye opening')
    # Rebuild actual surface points from immutable UV triangle barycentrics.
    direct=loadnpz(source/'directClient/geometry.npz'); faces=direct['faces']; uvFaces=direct['uvFaces']; uv=direct['uvs']*size-.5
    yy,xx=np.nonzero(alpha>0);ids=p['surfaceFaceIds'][yy,xx]
    require((ids>=0).all() and (ids<len(faces)).all(),'Invalid sampled face ids')
    tri=uv[uvFaces[ids]]; first,second,third=tri[:,0],tri[:,1],tri[:,2]
    denominator=(second[:,1]-third[:,1])*(first[:,0]-third[:,0])+(third[:,0]-second[:,0])*(first[:,1]-third[:,1])
    require((abs(denominator)>1e-12).all(),'Degenerate source UV face')
    b0=((second[:,1]-third[:,1])*(xx-third[:,0])+(third[:,0]-second[:,0])*(yy-third[:,1]))/denominator
    b1=((third[:,1]-first[:,1])*(xx-third[:,0])+(first[:,0]-third[:,0])*(yy-third[:,1]))/denominator
    weights=np.c_[b0,b1,1-b0-b1]
    require((weights>=-1e-7).all(),'Sample outside owning UV triangle')
    pts=np.sum(after[skin][faces[ids]]*weights[:,:,None],axis=1)
    camera=json.loads((source/'cameras.json').read_text())[0];R=np.asarray(camera['rotation']);t=np.asarray(camera['translation']);K=np.asarray(camera['intrinsics'])
    pts=pts@R.T+t; proj=pts@K.T; xy=proj[:,:2]/proj[:,2:]
    require(np.allclose(xy,p['projectedXY'][yy,xx],rtol=0,atol=7e-5),'Recorded projection differs from actual exported mesh')
    # Independently render the actual exported assembly to validate the texture
    # proxy; full aperture rejection is retained separately and is not erased.
    contourAudit={'performed':False,'reason':'Legacy pilot lacks textureBinding.json'}
    if (directory/'textureBinding.json').exists():
        saved=json.loads((directory/'textureBinding.json').read_text())
        loaded=loadLegacyEyeBindingInputs(directory/'combined/legacyAsset.zip',skin)
        loaded.pop('nicpToFullIds')
        eyes={name:{'upper':obs[i,1],'lower':obs[i,0]} for i,name in enumerate(('imageLeft','imageRight'))}
        measured=extractVisibleEyeContours(**loaded,nicpToFullIds=skin,camera=camera,photoEyes=eyes,
            imageShape=valid.shape,minimumEyeRasterWidth=256,maximumRoiSize=768)
        maximumContourError=0.
        for name in eyes:
            require(saved['eyes'][name]['status']=='accepted' and measured['eyes'][name]['status']=='accepted','Texture proxy rejected')
            for part in ('upper','lower'):
                stored=np.asarray(saved['eyes'][name][part]['sourcePixels']);actual=measured['eyes'][name][part]['sourcePixels']
                require(stored.shape==eyes[name][part].shape and np.isfinite(stored).all(),'Invalid proxy curve')
                require(stored[0,0]<stored[-1,0] and (np.linalg.norm(np.diff(stored,axis=0),axis=1)>1e-7).all(),'Degenerate/unordered proxy curve')
                error=float(abs(stored-actual).max());maximumContourError=max(maximumContourError,error)
                require(error<=1e-6,'Texture proxy differs from actual exported model')
        contourAudit={'performed':True,'status':'passed','maximumErrorWorkingPixels':maximumContourError,
            'scope':'True visible eye-material proxy, not complete eye-and-corner aperture'}
    opaque=original.mesh['faces'][original.mesh['materialIds']!=3]
    visibility=ProjectedSurfaceVisibility(after@R.T+t,opaque,K,valid.shape)
    depth=visibility.closestDepth(pts)['depth'];extent=np.ptp(after[skin],axis=0).max()
    require(np.isfinite(depth).all() and (abs(depth-pts[:,2])<=extent*1e-6).all(),'Edited center occluded in opaque scene')
    return dict(caseId=report['caseId'],status='passed',centerSamples=len(xx),filterPadding=int(p['filterPadding'].sum()),
        uniqueSourcePixels=len(sourcePixels),sourcePixelsValidOutsideEyes=True,sourceProvenanceDisjointComplete=True,
        zipCoordinatesAndTopologyVerified=True,unrelatedGeometryFixed=True,protectedUvsExact=True,
        actualProjectionMaximumError=float(abs(xy-p['projectedXY'][yy,xx]).max(initial=0)),
        exactOpaqueSceneVisibilityPassed=True,visibilityLimitation='MASK cards excluded; this audit does not claim per-fragment alpha-aware visibility',
        geometryStatus=report['geometry'].get('status'),textureContourAudit=contourAudit,fullGenerationMeasured=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT/'runs/fastFrontEyes20261012');parser.add_argument('--case',action='append');args=parser.parse_args()
    cases=args.case or ('currentMale','highPitch4','cyanWall4','outdoorSideLight4')
    rows=[]
    for case in cases:
        result=audit(args.root/case);rows.append(result)
        (args.root/case/'artifactAudit.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.root/'artifactAudit.json').write_text(json.dumps({'status':'passed','cases':rows},indent=2)+'\n')
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
