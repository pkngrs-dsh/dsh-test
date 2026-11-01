"""Local fixed-camera client landmark postprocessing on frozen direct fits.

The zero-template-prior, hard-interface direct fit is the starting geometry.
Only the reference camera plane changes. Automatic photo points are proxies;
the experiment also retains raw proposals rejected by intersection backtracking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.camera import projectPoints
from faceAppearance.clientLandmarks import loadClientLandmarks, ibugClientLandmarks
from faceAppearance.frontLandmarkRefinement import refineFrontLandmarks, checkFrontGeometry
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.surfaceVisibility import ProjectedSurfaceVisibility
from auditDirectRegistrationSafety import broadPhase, intersectTriangles
from experimentDirectRegistration import inverse, makeGrid, serial, writeJson


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def sample(vertices, binding):
    return (vertices[binding['vertexIndices']] * binding['barycentricWeights'][..., None]).sum(1)


def project(vertices, camera):
    return projectPoints(vertices, camera['intrinsics'], camera['rotation'], camera['translation'])[0]


def visibility(vertices, faces, points, camera, shape):
    rotation, translation = np.asarray(camera['rotation']), np.asarray(camera['translation'])
    cv = vertices @ rotation.T + translation
    cp = points @ rotation.T + translation
    sampler = ProjectedSurfaceVisibility(cv, faces, np.asarray(camera['intrinsics']), shape)
    closest = sampler.closestDepth(cp)['depth']
    tolerance = float(np.ptp(vertices, axis=0).max()) * .002
    pixels = project(points, camera)
    valid = ((cp[:, 2] > 0) & np.isfinite(closest)
             & (np.abs(closest - cp[:, 2]) <= tolerance)
             & (pixels[:, 0] >= 0) & (pixels[:, 0] < shape[1])
             & (pixels[:, 1] >= 0) & (pixels[:, 1] < shape[0]))
    return valid, {'depthTolerance': tolerance, 'surfaceDepth': closest,
                   'pointDepth': cp[:, 2], 'visibleMask': valid}


def intersections(vertices, faces, epsilon=None):
    epsilon = float(np.linalg.norm(np.ptp(vertices, axis=0))) * 1e-9 if epsilon is None else epsilon
    pairs, _ = broadPhase(vertices, faces, epsilon)
    triangles = vertices[faces]
    hits = []
    for a, b in pairs:
        hit = intersectTriangles(triangles[a], triangles[b], epsilon)
        if hit and hit['kind'] in ('nonCoplanarSegment', 'coplanarAreaOverlap'):
            hits.append({'faceIds': [int(a), int(b)], **hit})
    return hits


def pairSet(hits):
    return {tuple(hit['faceIds']) for hit in hits}


def errors(vertices, binding, cameras, observations, masks):
    points = sample(vertices, binding)
    result = []
    for index, (camera, obs, keep) in enumerate(zip(cameras, observations, masks)):
        distances = np.linalg.norm(project(points, camera) - obs['pixels'], axis=1)
        record = {'viewIndex': index, 'groups': {}}
        for name, region in (('mouth', binding['ibugIds'] >= 48), ('eyes', binding['ibugIds'] < 48)):
            ids = keep & region
            record['groups'][name] = {'count': int(ids.sum()),
                'meanErrorPixels': float(distances[ids].mean()) if ids.any() else None,
                'maxErrorPixels': float(distances[ids].max()) if ids.any() else None}
        record['pointErrorsPixels'] = distances
        result.append(record)
    return result


def renderProjected(vertices, faces, camera, size=(512, 512)):
    """CPU perspective raster, common camera/light for scientific comparisons."""
    width, height = size
    cv = vertices @ np.asarray(camera['rotation']).T + camera['translation']
    pixels = project(vertices, camera)
    normals = np.zeros_like(cv)
    tri = cv[faces]
    cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], cross)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([-.25, -.4, -1.]); light /= np.linalg.norm(light)
    shade = .3 + .65*np.maximum(normals @ light, 0)
    image = np.full((height, width, 3), [28, 35, 39], np.uint8)
    depth = np.full((height, width), np.inf)
    for face in faces:
        if (cv[face, 2] <= 0).any():
            continue
        xy = pixels[face]
        lo = np.maximum(np.floor(xy.min(0)).astype(int), 0)
        hi = np.minimum(np.ceil(xy.max(0)).astype(int), [width-1, height-1])
        if (lo > hi).any():
            continue
        a,b,c = xy
        denominator = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(denominator) < 1e-10:
            continue
        yy,xx = np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        u = ((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/denominator
        v = ((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/denominator
        w = 1-u-v
        invz = u/cv[face[0],2]+v/cv[face[1],2]+w/cv[face[2],2]
        z = 1/np.maximum(invz,1e-12)
        window = depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        keep = (u>=0)&(v>=0)&(w>=0)&(z<window)
        window[keep] = z[keep]
        s = (u*shade[face[0]]/cv[face[0],2]+v*shade[face[1]]/cv[face[1],2]+w*shade[face[2]]/cv[face[2],2])*z
        rgb = np.clip(s[...,None]*[208,218,222],0,255).astype(np.uint8)
        image[lo[1]:hi[1]+1,lo[0]:hi[0]+1][keep] = rgb[keep]
    return Image.fromarray(image), np.isfinite(depth)


def images(directory, case, inputs, binding, cameras, observations, before, after, faces, fitMask):
    photo = ImageOps.exif_transpose(Image.open(case['sourceImages'][0]['path'])).convert('RGB')
    work = cv2.warpPerspective(np.asarray(photo), np.asarray(inputs[0]['inputToWorking']),
                               tuple(inputs[0]['workingSize']), flags=cv2.INTER_LINEAR)
    panel = Image.new('RGB', (512*4, 570), (28,35,39))
    titles = ['Photo / cyan: used; gray X: excluded', 'Before / orange: client points',
              'After / green: client points', 'Point overlay: target / before / after']
    sourcePoints = observations[0]['pixels']
    projected = [project(sample(mesh, binding), cameras[0]) for mesh in (before, after)]
    for col in range(4):
        pic = Image.fromarray(work.copy())
        if col in (1,2):
            gray, mask = renderProjected((before,after)[col-1], faces, cameras[0])
            mixed = np.asarray(pic).copy()
            mixed[mask] = (.30*mixed[mask]+.70*np.asarray(gray)[mask]).astype(np.uint8)
            pic = Image.fromarray(mixed)
        draw = ImageDraw.Draw(pic)
        series = [(sourcePoints,'cyan')]
        if col in (1,3): series.append((projected[0],'orange'))
        if col in (2,3): series.append((projected[1],'lime'))
        for points,color in series:
            for index,xy in enumerate(points):
                x,y = map(float,xy)
                if fitMask[index]:
                    draw.ellipse((x-2,y-2,x+2,y+2),outline=color,width=1)
                elif color=='cyan':
                    draw.line((x-2,y-2,x+2,y+2),fill='#aaaaaa')
                    draw.line((x-2,y+2,x+2,y-2),fill='#aaaaaa')
        panel.paste(pic,(col*512,58))
        ImageDraw.Draw(panel).text((col*512+10,32),titles[col],fill='white')
    ImageDraw.Draw(panel).text((10,9),case['caseId']+' | fixed saved photo camera | automatic landmarks; in-sample evaluation',fill='white')
    panel.save(directory/'frontOverlay.png')
    Image.fromarray(work).save(directory/'photo.png')
    panels=[]
    for view in range(1,min(3,len(cameras))):
        for mesh in (before,after):
            panels.append(renderProjected(mesh,faces,cameras[view])[0])
    side=Image.new('RGB',(512*len(panels),550),(28,35,39))
    for i,img in enumerate(panels):
        side.paste(img,(i*512,38))
        ImageDraw.Draw(side).text((i*512+10,12),f'View {i//2+1} / '+('before' if i%2==0 else 'after'),fill='white')
    side.save(directory/'side.png')


def runCase(case, output):
    start=perf_counter()
    caseId=case['caseId']; directory=output/caseId
    directory.mkdir(parents=True,exist_ok=True)
    for field,expected in [('geometryPath','geometrySha256'),('cameraPath','cameraSha256'),('inputsPath','inputsSha256')]:
        if digest(case[field]) != case[expected]: raise ValueError(f'{caseId}: changed {field}')
    inputs=json.loads(Path(case['inputsPath']).read_text())
    cameras=json.loads(Path(case['cameraPath']).read_text())
    if not all(c['valid'] for c in cameras): raise ValueError('All saved cameras must be accepted')
    cameras=[{k:np.asarray(c[k],float) for k in ('intrinsics','rotation','translation')} for c in cameras]
    oldRoot=ROOT/'runs/directRegistration20261012'/caseId
    old=archive(oldRoot/'meshes.npz')
    oldPreview=json.loads((oldRoot/'preview.json').read_text())
    gender=oldPreview['gender']; assets=AssemblyAssets(ROOT/f'assets/clientTemplates/legacy{gender.title()}V1')
    geometry=archive(case['geometryPath'])
    _,_,diag=assets._deform(geometry)
    transform=diag['templateToUff']
    rotation=np.asarray(transform['rotation']); translation=np.asarray(transform['translation']); scale=transform['scale']
    forward=lambda v: (v @ rotation)*scale+translation
    before=forward(old['noPriorHard'])
    faces=old['faces']; fixed=np.flatnonzero(assets.arrays['staticMask'])
    fullBinding=loadClientLandmarks(faces)
    binding=ibugClientLandmarks(fullBinding,groups=('eyes','mouth'))
    observations=[]; masks=[]; visibilityRecords=[]
    for view,(item,record,camera) in enumerate(zip(inputs,case['observationViews'],cameras)):
        if digest(record['npz']) != record['sha256']: raise ValueError('Observation changed')
        if digest(case['sourceImages'][view]['path']) != item['sha256']: raise ValueError('Photo identity changed')
        cache=archive(record['npz']); points=cache['faceLandmarks'][binding['ibugIds']]
        hom=np.column_stack([points,np.ones(len(points))]) @ np.asarray(item['inputToWorking']).T
        pixels=hom[:,:2]/hom[:,2:]
        confidence=cache['faceRawPeakScores'][binding['ibugIds']]
        visible,details=visibility(before,faces,sample(before,binding),camera,tuple(reversed(item['workingSize'])))
        keep=visible & binding['validMask'] & (confidence>=np.where(binding['ibugIds']>=48,.4,.5))
        keep &= (pixels>=0).all(1) & (pixels < np.asarray(item['workingSize'])).all(1)
        observations.append({'pixels':pixels,'confidence':confidence})
        masks.append(keep); visibilityRecords.append(details)
    beforeErrors=errors(before,binding,cameras,observations,masks)
    fitMask=masks[0] & (np.linalg.norm(project(sample(before,binding),cameras[0])-observations[0]['pixels'],axis=1)<=40.)
    collisionEpsilon=float(np.linalg.norm(np.ptp(old['noPriorHard'],axis=0)))*1e-9
    beforeHits=intersections(old['noPriorHard'],faces,collisionEpsilon); beforePairs=pairSet(beforeHits)
    meshes={'before':old['noPriorHard'],'rest':old['rest'],'faces':faces,'source':old['source'],
            'sourceFaces':old['sourceFaces'],'baseline':old['baseline']}
    branches=[b for b in oldPreview['branches'] if b['id'] in ('source','baseline')]
    branches.append({'id':'before','label':'Before: direct / zero template prior','vertices':old['noPriorHard'],
                     'faces':faces,'metrics':{'frontErrors':{},'collision':{'before':len(beforePairs),'after':len(beforePairs),'newPairs':0,'scale':0}}})
    reports={}; acceptedCanonical={}
    for variant,groups in [('mouthOnly',('mouth',)),('frontLandmarks',('mouth','eyes'))]:
        constraints=[]
        for group in groups:
            selected=fitMask & ((binding['ibugIds']>=48) if group=='mouth' else (binding['ibugIds']<48))
            if selected.sum() < (6 if group=='mouth' else 4): continue
            ids=binding['ibugIds'][selected]
            weights=np.where(np.isin(ids,[36,39,42,45,48,54]),4.,1.)
            pairs=[]
            for a,b in [(i,i+1) for i in range(36,67) if i not in (41,47,59)] + [(41,36),(47,42),(59,48),(67,60)]:
                if a in ids and b in ids: pairs.append([int(np.flatnonzero(ids==a)[0]),int(np.flatnonzero(ids==b)[0])])
            constraints.append({'name':group,'vertexIndices':binding['vertexIndices'][selected],
                'barycentric':binding['barycentricWeights'][selected], 'targetPixels':observations[0]['pixels'][selected],
                'weights':weights,'confidence':observations[0]['confidence'][selected],
                'segmentPairs':np.asarray(pairs,np.int64).reshape(-1,2)})
        fitted=refineFrontLandmarks(before,faces,cameras[0],constraints,fixedVertexIds=fixed,
            minConfidence=.4, supportRadius=float(np.ptp(before,axis=0).max())*.14,
            controlCount=32,maxDisplacement=float(np.ptp(before,axis=0).max())*.03,
            maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3,iterations=80)
        raw=fitted['vertices']; rawDisplay=inverse(raw,transform)
        # Eliminate inverse-transform rounding only at analytically fixed indices.
        rawDisplay[fixed]=old['noPriorHard'][fixed]
        attempts=[]; chosen=before.copy(); chosenDisplay=old['noPriorHard'].copy(); chosenHits=beforeHits; amount=0.
        for alpha in (1.,.5,.25,.125,.0625):
            candidate=before+alpha*(raw-before)
            local=checkFrontGeometry(before,candidate,faces,camera=cameras[0],fixedVertexIds=fixed,
                maxDisplacement=float(np.ptp(before,axis=0).max())*.03,
                maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3)
            display=inverse(candidate,transform); display[fixed]=old['noPriorHard'][fixed]
            hits=intersections(display,faces,collisionEpsilon); current=pairSet(hits); added=current-beforePairs
            attempts.append({'scale':alpha,'count':len(current),'newPairs':len(added),'newFacePairs':sorted(added),'localGeometry':local})
            if not added and local['accepted']:
                chosen,chosenDisplay,chosenHits,amount=candidate,display,hits,alpha
                break
        afterErrors=errors(chosen,binding,cameras,observations,masks)
        rawErrors=errors(raw,binding,cameras,observations,masks)
        change=np.linalg.norm(chosenDisplay-old['noPriorHard'],axis=1)
        frontErrors={}
        for name in ('mouth','eyes'):
            frontErrors[name+'Before']=beforeErrors[0]['groups'][name]['meanErrorPixels']
            frontErrors[name+'After']=afterErrors[0]['groups'][name]['meanErrorPixels']
        collision={'before':len(beforePairs),'after':len(pairSet(chosenHits)),'newPairs':0,'scale':amount,
                   'rawCount':attempts[0]['count'],'rawNewPairs':attempts[0]['newPairs']}
        changed=bool(np.max(change)>1e-10)
        metrics={'frontErrors':frontErrors,'fitStatus':'acceptedLocalCandidate' if changed else 'unchanged',
                 'collision':collision,'maxDisplacementMm':float(change.max()*10),
                 'fixedVerticesExactlyUnchanged':bool(np.array_equal(chosenDisplay[fixed],old['noPriorHard'][fixed])),
                 'referenceDepthChangeMax':float(np.max(np.abs((chosen-before)@cameras[0]['rotation'][2])))}
        branches.append({'id':variant,'label':variant,'vertices':chosenDisplay,'faces':faces,'metrics':metrics})
        meshes[variant]=chosenDisplay; acceptedCanonical[variant]=chosen
        if variant=='frontLandmarks':
            branches.append({'id':'rawFrontLandmarks','label':'Raw landmark proposal / diagnostic',
                'vertices':rawDisplay,'faces':faces,'metrics':{'fitStatus':'diagnosticOnly',
                'frontErrors':{name+stage:arr[0]['groups'][name]['meanErrorPixels'] for name in ('mouth','eyes')
                               for stage,arr in [('Before',beforeErrors),('After',rawErrors)]},
                'collision':{'before':len(beforePairs),'after':attempts[0]['count'],'newPairs':attempts[0]['newPairs'],'scale':1}}})
            meshes['rawFrontLandmarks']=rawDisplay
        reports[variant]={'metrics':metrics,'solver':fitted['diagnostics'],'before':beforeErrors,'after':afterErrors,
                         'rawAfter':rawErrors,'intersectionBacktracking':attempts,
                         'beforeIntersections':beforeHits,'afterIntersections':chosenHits}
    # Retain the simultaneous fit as a control. Separate eye passes prevent a
    # mouth collision from reducing an otherwise useful eye displacement.
    for branch in branches:
        if branch['id']=='frontLandmarks': branch['id']='jointFrontLandmarks'
    meshes['jointFrontLandmarks']=meshes.pop('frontLandmarks')
    reports['jointFrontLandmarks']=reports.pop('frontLandmarks')
    stages=[]; current=acceptedCanonical['mouthOnly'].copy()
    displacementLimit=float(np.ptp(before,axis=0).max())*.03
    for name,startId,endId in [('imageLeftEye',36,42),('imageRightEye',42,48)]:
        selected=fitMask & (binding['ibugIds']>=startId) & (binding['ibugIds']<endId)
        if selected.sum()<4:
            stages.append({'name':name,'status':'skippedInsufficientTargets','scale':0.}); continue
        ids=binding['ibugIds'][selected]
        constraint={'name':name,'vertexIndices':binding['vertexIndices'][selected],
            'barycentric':binding['barycentricWeights'][selected],
            'targetPixels':observations[0]['pixels'][selected],
            'weights':np.where(np.isin(ids,[36,39,42,45]),4.,1.),
            'confidence':observations[0]['confidence'][selected]}
        result=refineFrontLandmarks(current,faces,cameras[0],[constraint],fixedVertexIds=fixed,
            minConfidence=.5,supportRadius=float(np.ptp(before,axis=0).max())*.14,
            controlCount=12,maxDisplacement=displacementLimit,
            maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3,iterations=80)
        attempts=[]; amount=0.
        stageDisplay=inverse(current,transform);stageDisplay[fixed]=old['noPriorHard'][fixed]
        stagePairs=pairSet(intersections(stageDisplay,faces,collisionEpsilon))
        stageErrors=errors(current,binding,cameras,observations,masks)
        for alpha in (1.,.5,.25,.125,.0625,.03125,.015625):
            candidate=current+alpha*(result['vertices']-current)
            local=checkFrontGeometry(before,candidate,faces,camera=cameras[0],fixedVertexIds=fixed,
                maxDisplacement=displacementLimit,maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3)
            if not local['accepted']:
                attempts.append({'scale':alpha,'localGeometry':local,'newPairs':None}); continue
            display=inverse(candidate,transform); display[fixed]=old['noPriorHard'][fixed]
            hits=intersections(display,faces,collisionEpsilon); added=pairSet(hits)-stagePairs
            e=errors(candidate,binding,cameras,observations,masks)
            # Frozen side targets are only usable where the initial mesh was
            # visible. The fixed threshold is a regression gate, not validation
            # of those cameras or multi-view expression consistency.
            sideDeltas=[e[i]['groups']['eyes']['meanErrorPixels']-beforeErrors[i]['groups']['eyes']['meanErrorPixels']
                for i in range(1,len(e)) if e[i]['groups']['eyes']['count']>=4]
            sideSafe=not sideDeltas or max(sideDeltas)<=1.
            frontDeltas=[e[0]['groups'][region]['meanErrorPixels']-stageErrors[0]['groups'][region]['meanErrorPixels']
                for region in ('mouth','eyes') if e[0]['groups'][region]['count']>0]
            previousSafe=not frontDeltas or max(frontDeltas)<=.05
            attempts.append({'scale':alpha,'localGeometry':local,'newPairs':len(added),
                             'newFacePairs':sorted(added),'sideEyeMeanErrorChangesPixels':sideDeltas,'sideGate':sideSafe,
                             'frontMeanErrorChangesVsPreviousStage':frontDeltas,'previousStageGate':previousSafe})
            if not added and sideSafe and previousSafe:
                current=candidate; amount=alpha; break
        stages.append({'name':name,'scale':amount,'solver':result['diagnostics'],'attempts':attempts})
    finalDisplay=inverse(current,transform);finalDisplay[fixed]=old['noPriorHard'][fixed]
    finalHits=intersections(finalDisplay,faces,collisionEpsilon); finalErrors=errors(current,binding,cameras,observations,masks)
    finalLocal=checkFrontGeometry(before,current,faces,camera=cameras[0],fixedVertexIds=fixed,
        maxDisplacement=displacementLimit,maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3)
    if not finalLocal['accepted'] or pairSet(finalHits)-beforePairs:
        raise AssertionError('Final staged candidate failed its geometry checks')
    finalMetrics={'frontErrors':{name+stage:arr[0]['groups'][name]['meanErrorPixels'] for name in ('mouth','eyes')
                   for stage,arr in [('Before',beforeErrors),('After',finalErrors)]},
        'fitStatus':'acceptedLocalCandidate' if np.linalg.norm(current-before)>1e-10 else 'unchanged',
        'collision':{'before':len(beforePairs),'after':len(pairSet(finalHits)),'newPairs':0,
                     'stageScales':{'mouth':reports['mouthOnly']['metrics']['collision']['scale'],
                                    **{stage['name']:stage['scale'] for stage in stages}}},
        'maxDisplacementMm':float(np.linalg.norm(finalDisplay-old['noPriorHard'],axis=1).max()*10),
        'fixedVerticesExactlyUnchanged':bool(np.array_equal(finalDisplay[fixed],old['noPriorHard'][fixed])),
        'referenceDepthChangeMax':float(np.max(np.abs((current-before)@cameras[0]['rotation'][2])))}
    reports['frontLandmarks']={'metrics':finalMetrics,'before':beforeErrors,'after':finalErrors,'stages':stages,
        'localGeometry':finalLocal,'beforeIntersections':beforeHits,'afterIntersections':finalHits,
        'totalIntersectionLengthCm':{'before':sum(h.get('segmentLengthCm',0) for h in beforeHits),
                                     'after':sum(h.get('segmentLengthCm',0) for h in finalHits)},
        'finalVisibleEvaluationPoints':[int((visibility(current,faces,sample(current,binding),c,tuple(reversed(i['workingSize'])))[0]&mask).sum())
                                        for c,i,mask in zip(cameras,inputs,masks)],
        'method':'mouth correction followed by independent image-left and image-right eye passes'}
    meshes['frontLandmarks']=finalDisplay;acceptedCanonical['frontLandmarks']=current
    branches.append({'id':'frontLandmarks','label':'Staged mouth + eye refinement','vertices':finalDisplay,
                     'faces':faces,'metrics':finalMetrics})
    print(json.dumps({'case':caseId,'variant':'stagedFrontLandmarks',**finalMetrics},default=serial),flush=True)
    np.savez_compressed(directory/'meshes.npz',**meshes)
    np.savez_compressed(directory/'landmarks.npz',ibugIds=binding['ibugIds'],vertexIndices=binding['vertexIndices'],
        barycentricWeights=binding['barycentricWeights'],frontTargetPixels=observations[0]['pixels'],
        frontConfidence=observations[0]['confidence'],frontEligibleFitMask=fitMask,
        frontEvaluationMask=masks[0],beforeCanonical=before,
        afterCanonical=acceptedCanonical['frontLandmarks'])
    preview={'caseId':caseId,'gender':gender,'mouthCenter':oldPreview['mouthCenter'],'branches':branches}
    (directory/'preview.json').write_text(json.dumps(preview,default=serial,ensure_ascii=False,separators=(',',':')))
    mx,my,_=oldPreview['mouthCenter']; selected=[b for b in branches if b['id'] in ('before','mouthOnly','frontLandmarks','rawFrontLandmarks')]
    makeGrid(directory/'mouth.png',selected,(mx-4,mx+4,my-3,my+3),caseId+' | same camera; zero template shape prior',440)
    source=old['source']; ymin,ymax=source[:,1].min(),source[:,1].max()
    makeGrid(directory/'face.png',selected,(-8,8,ymin-.5,ymax+.5),caseId+' | same camera; geometry only',400)
    images(directory,case,inputs,binding,cameras,observations,before,acceptedCanonical['frontLandmarks'],faces,fitMask)
    # Visibility records have infinite background depths; encode them explicitly as null.
    for row in visibilityRecords:
        row['surfaceDepth']=[float(x) if np.isfinite(x) else None for x in row['surfaceDepth']]
    report={'caseId':caseId,'status':'localExperimentComplete','sourceHashes':{k:case[k] for k in ('geometrySha256','cameraSha256','inputsSha256')},
            'bindingSha256':fullBinding.get('sourceSha256'),'referenceViewIndex':0,'coordinateFrame':'512 working pixels / UFF canonical geometry',
            'templateShapePrior':0,'directBranch':'noPriorHard','camerasRefitted':False,'viewVisibility':visibilityRecords,
            'frozenEvaluationMasks':masks,'landmarkIds':binding['ibugIds'],'variants':reports,'wallSeconds':perf_counter()-start,
            'meshSha256':digest(directory/'meshes.npz'),'qualityApproved':False,
            'limits':['Automatic 68-point detections and MP-to-ibug binding are approximate, not ground truth.',
                      'Point scores are raw model heatmap peaks, not calibrated probabilities.',
                      'Reference depth preservation does not guarantee unchanged side appearance.',
                      'Fixed-camera errors are in-sample and include saved camera and binding errors.',
                      'No new nonadjacent positive intersection pairs at final pose; existing collisions persist.',
                      'Shared-vertex pairs and continuous motion collisions are not checked.',
                      'Texture reprojection and complete client accessories are not validated.']}
    writeJson(directory/'report.json',report)
    return {'id':caseId,'gender':gender,'previewUrl':f'/data/{caseId}/preview.json','branches':[{'id':b['id'],**b['metrics']} for b in branches],
            'reportUrl':f'/data/{caseId}/report.json','wallSeconds':report['wallSeconds']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,default=ROOT/'reports/frontLandmark20261012/sources.json')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/frontLandmark20261012')
    parser.add_argument('--case',action='append')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cases=json.loads(args.sources.read_text())['cases']
    report={'status':'localExperimentRunning','objective':'Zero template shape prior + fixed-camera local landmark postprocess',
            'remoteChanged':False,'cases':[]}
    for case in cases:
        if args.case and case['caseId'] not in args.case: continue
        report['cases'].append(runCase(case,args.output));writeJson(args.output/'experiment.json',report)
    report['status']='localExperimentCompleteQualityNotApproved';writeJson(args.output/'experiment.json',report)


if __name__=='__main__':
    main()
