"""Read-only, fixed-camera photo semantic audit of cached registration stages.

No detector, UFF, camera fit, geometry fit, nearest-surface rebinding, or GPU.
Eye scores compare independently annotated skin-hole curves with the same
frozen photo curve. They are topology-independent 2-D proxies, not final visible
aperture or anatomical ground truth. Mouth/nose have no audited common UFF
binding, so only fixed client-stage point errors are reported for those parts.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from faceAppearance.anatomicalBoundaries import extractMeshBoundaryLoops, validateBoundaryAsset
from faceAppearance.camera import projectPoints
from faceAppearance.clientLandmarks import loadClientLandmarks, ibugClientLandmarks, sampleClientLandmarks
from faceAppearance.surfaceVisibility import ProjectedSurfaceVisibility


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def plain(x):
    if isinstance(x, dict): return {k: plain(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [plain(v) for v in x]
    if isinstance(x, np.ndarray): return plain(x.tolist())
    if isinstance(x, np.generic): return x.item()
    return x


def write(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False))


def project(vertices, camera):
    return projectPoints(vertices, *[np.asarray(camera[k]) for k in ('intrinsics','rotation','translation')])


def resample(poly, spacing=.25):
    following = np.roll(poly, -1, axis=0)
    length = np.linalg.norm(following-poly, axis=1)
    cumulative = np.r_[0., np.cumsum(length)]
    query = np.linspace(0, cumulative[-1], max(16,int(np.ceil(cumulative[-1]/spacing))), endpoint=False)
    index = np.clip(np.searchsorted(cumulative, query, side='right')-1, 0, len(poly)-1)
    ratio = (query-cumulative[index])/np.maximum(length[index], 1e-12)
    return poly[index]+(following[index]-poly[index])*ratio[:,None]


def distances(points, poly):
    start = poly; delta = np.roll(poly,-1,axis=0)-poly
    t = np.clip(np.einsum('nmd,md->nm',points[:,None]-start,delta)/np.maximum(np.sum(delta*delta,1),1e-20),0,1)
    return np.linalg.norm(points[:,None]-start-t[...,None]*delta,axis=-1).min(1)


def crossings(poly):
    count=0
    def cross(a,b): return a[0]*b[1]-a[1]*b[0]
    for i in range(len(poly)):
        a,b=poly[i],poly[(i+1)%len(poly)]
        for j in range(i+2,len(poly)):
            if i==0 and j==len(poly)-1: continue
            c,d=poly[j],poly[(j+1)%len(poly)]
            count += (cross(b-a,c-a)*cross(b-a,d-a)<-1e-12 and cross(d-c,a-c)*cross(d-c,b-c)<-1e-12)
    return int(count)


def curve_metrics(model, photo, width, originalScale):
    a,b=distances(resample(model),photo),distances(resample(photo),model)
    symmetrical=float((a.mean()+b.mean())*.5)
    crossed=crossings(model)
    iou=None; ratio=None
    if not crossed:
        lo=np.floor(np.r_[model,photo].min(0))-2
        hi=np.ceil(np.r_[model,photo].max(0))+2
        shape=tuple(((hi-lo)*4+1).astype(int)[::-1]); masks=[]
        for p in (model,photo):
            mask=np.zeros(shape,np.uint8)
            cv2.fillPoly(mask,[np.rint((p-lo)*4).astype(np.int32)],1)
            masks.append(mask.astype(bool))
        intersection=(masks[0]&masks[1]).sum();union=(masks[0]|masks[1]).sum()
        iou=float(intersection/union);ratio=float(masks[0].sum()/masks[1].sum())
    def area_and_center(poly):
        nxt=np.roll(poly,-1,axis=0)
        cross=poly[:,0]*nxt[:,1]-nxt[:,0]*poly[:,1]
        signed=cross.sum()*.5
        if abs(signed)<1e-8:return 0.,poly.mean(0)
        return abs(float(signed)),np.sum((poly+nxt)*cross[:,None],axis=0)/(6*signed)
    modelArea,modelCenter=area_and_center(model);photoArea,photoCenter=area_and_center(photo)
    # Photo polygon is upper left-to-right followed by reversed lower curve.
    right=photo[len(photo)//2-1];left=photo[0]
    along=(right-left)/np.linalg.norm(right-left);normal=np.array([-along[1],along[0]])
    modelHeight=float(np.ptp(model@normal));photoHeight=float(np.ptp(photo@normal))
    return {'symmetricCurveMeanWorkingPixels':symmetrical,
        'symmetricCurveMeanOriginalPixels':symmetrical*originalScale,
        'symmetricCurveMeanEyeWidths':symmetrical/width,
        'modelToPhotoMeanWorkingPixels':float(a.mean()),'photoToModelMeanWorkingPixels':float(b.mean()),
        'bidirectionalP95WorkingPixels':float(np.percentile(np.r_[a,b],95)),
        'polygonIoU4x':iou,'projectedSkinHoleAreaOverPhotoEye':ratio,
        'polygonAreaWorkingPixelsSquared':modelArea,'photoPolygonAreaWorkingPixelsSquared':photoArea,
        'analyticPolygonAreaOverPhotoEye':modelArea/photoArea,
        'photoAxisHeightWorkingPixels':modelHeight,'photoAxisHeightOriginalPixels':modelHeight*originalScale,
        'photoAxisTargetHeightWorkingPixels':photoHeight,'photoAxisHeightOverPhoto':modelHeight/photoHeight,
        'photoAxisHeightDefinition':'Extent perpendicular to the frozen photo upper-curve canthus axis; same axis for every branch',
        'polygonCenterWorkingPixels':modelCenter,'photoPolygonCenterWorkingPixels':photoCenter,
        'centerOffsetWorkingPixels':modelCenter-photoCenter,
        'centerErrorWorkingPixels':float(np.linalg.norm(modelCenter-photoCenter)),
        'properSelfCrossings':crossed,'distanceSamplingStepWorkingPixels':.25}


def boundary_visibility(vertices, faces, loop, camera):
    # Source 3-D edge samples; no triangle/semantic correspondence is invented.
    edge=vertices[loop]; f=np.linspace(0,1,8,endpoint=False)
    points=(edge[:,None,:]*(1-f[None,:,None])+np.roll(edge,-1,axis=0)[:,None,:]*f[None,:,None]).reshape(-1,3)
    cv=vertices@np.asarray(camera['rotation']).T+camera['translation']
    cp=points@np.asarray(camera['rotation']).T+camera['translation']
    visibility=ProjectedSurfaceVisibility(cv,faces,np.asarray(camera['intrinsics']),(512,512))
    depth=visibility.closestDepth(cp)['depth'];eps=np.ptp(vertices,axis=0).max()*1e-6
    visible=np.isfinite(depth)&(np.abs(depth-cp[:,2])<=eps)
    return {'sampleCount':len(points),'visibleSkinBoundarySamples':int(visible.sum()),
        'visibleSkinBoundaryFraction':float(visible.mean()),'sampling':'8 fixed points per 3-D boundary edge',
        'scope':'Own skin mesh only; no client eye/corner/accessory geometry is inserted into UFF'}


def case_paths(case):
    if case=='currentMale':
        source=ROOT/'runs/liveJaw20261012/source/reconstruction'
        obs=ROOT/'runs/denseClientEyes20261012/observations'
    else:
        base=ROOT/'runs/eyelidTextureBatch20261012'/case
        source=base/'baseline/reconstruction';obs=base/'observations'
    return source,obs,ROOT/'runs/fastFrontEyes20261012/final'/case


def overlay(photo, predictions, target, output, case):
    width=512; panels=[]
    bounds=np.r_[target[0],target[1]];lo=bounds.min(0)-[30,25];hi=bounds.max(0)+[30,25]
    box=tuple(np.r_[np.maximum(lo,0),np.minimum(hi,512)].astype(int))
    try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    except OSError:font=ImageFont.load_default()
    for name, eyes in predictions.items():
        image=photo.copy();draw=ImageDraw.Draw(image)
        for curve in target:
            p=[tuple(v) for v in curve];draw.line(p+[p[0]],fill=(40,255,80),width=2)
        for curve in eyes:
            p=[tuple(v) for v in curve];draw.line(p+[p[0]],fill=(255,75,75),width=1)
        crop=image.crop(box);crop=crop.resize((width,round(crop.height*width/crop.width)))
        panel=Image.new('RGB',(width,crop.height+45),(22,28,31));panel.paste(crop,(0,45))
        ImageDraw.Draw(panel).text((10,10),name,fill='white',font=font);panels.append(panel)
    columns=min(3,len(panels));rows=(len(panels)+columns-1)//columns;ph=max(p.height for p in panels)
    grid=Image.new('RGB',(columns*width,rows*ph+45),(22,28,31))
    ImageDraw.Draw(grid).text((10,10),f'{case} | green=frozen photo / red=skin-hole rim; same camera',fill='white',font=font)
    for i,p in enumerate(panels):grid.paste(p,((i%columns)*width,45+(i//columns)*ph))
    grid.save(output/'eyeSkinRims.png')


def run(case, output, currentUff=None):
    source,obs,final=case_paths(case);out=output/case;out.mkdir(parents=True,exist_ok=True)
    manifest=read(obs/'manifest.json');inputs=read(source/'inputs.json');cameras=read(source/'cameras.json');camera=cameras[0]
    if not camera['valid']: raise ValueError('Frozen reference camera was rejected')
    if sha(source/'cameras.json')!=manifest['camerasSha256'] or sha(source/'inputs.json')!=manifest['inputsSha256']:
        raise ValueError('Cached photo observations do not belong to these cameras/crops')
    observed=arrays(obs/'view00.npz');metadata=read(obs/'view00.json')
    if metadata['inputSha256']!=inputs[0]['sha256'] or not np.array_equal(observed['inputToWorking'],np.asarray(inputs[0]['inputToWorking'])):
        raise ValueError('Photo identity/crop mismatch')
    target=[np.r_[observed['eyelidCurvesWorking'][i,1],observed['eyelidCurvesWorking'][i,0,::-1]] for i in range(2)]
    widths=np.linalg.norm(observed['cornersWorking'][:,1]-observed['cornersWorking'][:,0],axis=1)
    inv=np.asarray(inputs[0]['workingToInput']);sv=np.linalg.svd(inv[:2,:2],compute_uv=False)
    if not np.allclose(sv,sv[0]):raise ValueError('Original pixel conversion needs isotropic saved crop')
    originalScale=float(sv.mean())
    c=arrays(source/'directClient/refinement/clientGeometry.npz');faces=c['faces']
    direct=c['directTemplateVertices']@c['templateToUffRotation']*c['templateToUffScale']+c['templateToUffTranslation']
    stable=arrays(source/'directClient/geometry.npz');end=arrays(final/'geometry.npz')
    branches={'clientDirect':(direct,faces),'clientPostLandmarks':(c['vertices'],faces),
        'clientAfterJaw':(stable['vertices'],faces),'clientFinalEyes':(end['after'][end['skinVertexIds']],faces)}
    uffPath=source/'geometry.npz'
    proofPath=ROOT/'runs/registrationTransfer20261013/metrics'/case/'report.json'
    proof=read(proofPath) if proofPath.exists() else None
    if case=='currentMale' and currentUff is not None:uffPath=Path(currentUff).resolve()
    elif case=='currentMale' and not uffPath.exists() and proof is not None:
        uffPath=Path(proof['sourceFiles']['uff']['path'])
    unavailable=None
    if uffPath.exists():
        if (proof is None or sha(uffPath)!=proof['sourceFiles']['uff']['sha256']
                or sha(source/'cameras.json')!=proof['sourceFiles']['cameras']['sha256']
                or proof['reproduction']['maximumCanonicalVertexDifference']>=1e-10):
            raise ValueError('Matching UFF/camera identity needs the independent direct-registration replay proof')
        u=arrays(uffPath);branches={'uffFrozen':(u['vertices'],u['faces']),**branches}
    else:unavailable='Matching original UFF mesh missing. Other runs of the same photos are not substituted.'
    assetPath=ROOT/'assets/anatomical/uff6986/boundaries.json';boundaryAsset=read(assetPath)
    clientRest=arrays(ROOT/'assets/clientTemplates/legacyMaleDirectV1/template.npz')
    if not np.array_equal(faces,clientRest['nicpFaces']):raise ValueError('Wrong client topology')
    clientLoops=extractMeshBoundaryLoops(clientRest['vertices'][clientRest['nicpToFullIds']],faces)
    if clientLoops['diagnostics']['loopVertexCounts']!=[40,10,10]:raise ValueError('Unexpected audited client skin holes')
    binding=ibugClientLandmarks(loadClientLandmarks(faces),groups=('eyes','mouth','nose'))
    ids=binding['ibugIds'];target68=observed['faceLandmarksWorking'][ids];confidence=observed['faceRawPeakScores'][ids]
    frozenValid=binding['validMask']&np.isfinite(target68).all(1)&(confidence>=np.where(ids>=48,.4,.5))
    eyeRecords={};pointsRecords={};predictions={};saved={}
    for name,(vertices,ff) in branches.items():
        if name=='uffFrozen':
            named=validateBoundaryAsset(vertices,ff,boundaryAsset)
            loops=[named['positiveXEye'],named['negativeXEye']]
        else:loops=clientLoops['loops'][1:]
        projected=[project(vertices[l],camera)[0] for l in loops]
        order=np.argsort([x[:,0].mean() for x in projected]);loops=[loops[i] for i in order];projected=[projected[i] for i in order]
        eyes=[]
        for i,(loop,curve) in enumerate(zip(loops,projected)):
            if (project(vertices[loop],camera)[1]<=0).any():raise ValueError('Rim behind frozen camera')
            eyes.append({'imageEye':['left','right'][i],'boundaryVertexIds':loop,'projectedCurve':curve,
                'photoCurve':target[i],'photoCandidateAccepted':bool(observed['candidateEyeValidMask'][i]),
                'metrics':curve_metrics(curve,target[i],widths[i],originalScale),
                'skinOnlyVisibility':boundary_visibility(vertices,ff,loop,camera)})
            saved[f'{name}_eye{i}']=curve
        eyeRecords[name]=eyes;predictions[name]=projected
        if name!='uffFrozen':
            xyz=sampleClientLandmarks(vertices,binding);pixels,depth=project(xyz,camera)
            frozen=frozenValid&(depth>0);values={}
            for group,indices in [('eyes',range(36,48)),('mouth',range(48,68)),('nose',range(27,36))]:
                keep=frozen&np.isin(ids,list(indices));err=np.linalg.norm(pixels[keep]-target68[keep],axis=1)
                values[group]={'landmarkIds':ids[keep],'count':int(keep.sum()),'targetWorkingPixels':target68[keep],
                    'predictedWorkingPixels':pixels[keep],'meanWorkingPixels':float(err.mean()) if len(err) else None,
                    'meanOriginalPixels':float(err.mean()*originalScale) if len(err) else None,
                    'pointErrorsWorkingPixels':err,'meaning':'Fixed approximate client MP-to-iBUG binding; not shared UFF semantics'}
            pointsRecords[name]=values
    report={'caseId':case,'status':'completeReadOnlySemanticAudit','cameraRefit':False,'geometryFit':False,'detectorRerun':False,
        'referenceView':0,'workingImageSize':inputs[0]['workingSize'],'originalPixelsPerWorkingPixel':originalScale,
        'sourcePaths':{'reconstruction':str(source),'photoObservations':str(obs),'finalEyes':str(final),'uff':str(uffPath) if uffPath.exists() else None},
        'originalUffUnavailableReason':unavailable,'cachedInputsAndCamerasHashMatch':True,
        'uffIdentityProof':{'report':str(proofPath),'sha256':sha(proofPath),
            'directRegistrationReplayMaximumCanonicalDifference':proof['reproduction']['maximumCanonicalVertexDifference']} if uffPath.exists() else None,
        'method':'Arc-length-uniform symmetric distance between each own-topology skin-hole rim and the frozen photo eye contour; no cross-mesh vertex correspondence',
        'eyeSkinRimProxy':eyeRecords,'clientOnlyLandmarkProxies':pointsRecords,
        'sharedMouthNoseSemanticComparison':{'available':False,'reason':'UFF oral/nasal hole rims do not define the same semantics as client outer lip/nose MP-to-iBUG points; no validated common binding.'},
        'limitations':['Automatic photo STFR points are observations, not manual anatomical truth.',
            'Skin-hole rims are material-independent structural proxies, not the final visible aperture of eyes, inner corners and alpha cards.',
            'Comparing different rim definitions/topological discretizations does not isolate registration error by itself; combine with the frozen-target transfer audit.',
            'Skin-only visibility is reported, never used to drop difficult points or adapt curve masks.',
            'Front landmarks were also used by later fitting stages; improved error is in-sample, not independent generalization.',
            'No common audited UFF/client mouth or nose landmark binding exists; corresponding transfer-error scores are intentionally unavailable.'],
        'hashes':{str(p.relative_to(ROOT)):sha(p) for p in [source/'cameras.json',source/'inputs.json',source/'directClient/refinement/clientGeometry.npz',source/'directClient/geometry.npz',obs/'view00.npz',assetPath,final/'geometry.npz',Path(__file__)]}}
    if uffPath.exists():report['hashes'][str(uffPath.relative_to(ROOT))]=sha(uffPath)
    if 'uffFrozen' in eyeRecords:
        report['meanCurveDistanceByStage']={name:float(np.mean([x['metrics']['symmetricCurveMeanWorkingPixels'] for x in eyes])) for name,eyes in eyeRecords.items()}
        report['clientDirectMinusUffWorkingPixels']=report['meanCurveDistanceByStage']['clientDirect']-report['meanCurveDistanceByStage']['uffFrozen']
    else:report['meanCurveDistanceByStage']={name:float(np.mean([x['metrics']['symmetricCurveMeanWorkingPixels'] for x in eyes])) for name,eyes in eyeRecords.items()}
    write(out/'report.json',report);np.savez_compressed(out/'projectedCurves.npz',**saved,photoLeft=target[0],photoRight=target[1])
    photo=Image.open(obs/'view00WorkingPhoto.png').convert('RGB');overlay(photo,predictions,target,out,case)
    print(json.dumps({'case':case,'eyeMeanWorkingPixels':report['meanCurveDistanceByStage'],'uffMissing':unavailable is not None}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',nargs='+',default=['currentMale','highPitch4','cyanWall4','outdoorSideLight4'])
    p.add_argument('--output',type=Path,default=ROOT/'runs/registrationTransfer20261013/semantics')
    p.add_argument('--current-uff',type=Path,help='Only a verified exact original UFF cache; never another inference of the same photos')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=[run(c,a.output,a.current_uff) for c in a.cases]
    write(a.output/'summary.json',{'cases':[{k:r[k] for k in ('caseId','originalUffUnavailableReason','meanCurveDistanceByStage')} for r in rows],
        'comparisonScope':'Fixed same input camera; structural eye skin-rim proxy plus separately labeled client landmark proxies. Not a final appearance score.'})


if __name__=='__main__':main()
