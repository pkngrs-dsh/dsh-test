"""Local visible-lip-crease alignment on the frozen male client mesh.

Uses an automatic photo curve and a camera-depth crease, not the approximate
inner-lip landmark bindings. Experiments only; no production configuration is
changed and existing collisions are reported separately from new collisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.interpolate import UnivariateSpline

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from faceAppearance.camera import projectPoints
from faceAppearance.frontLandmarkRefinement import checkFrontGeometry
from faceAppearance.colmapTexture import prepareWorkspace, loadTexturedMesh, rebakeAtlas, sourceSupportFromMarkers, viewMarkerColors
from faceAppearance.headAssembly import AssemblyAssets, decodeColor
from experimentFrontLandmarks import intersections, pairSet, renderProjected
from prepareFrontLandmarkTextures import renderTexture, encodeSrgb
from runClientColmapExperiment import archive, digest, writeJson, clientColors, saveColor
from runColmapTextureExperiment import texturer
from runMaleMouthTexture import makeCandidate, addFilterFootprint
from diagnoseMaleLipTopology import traceVisibleLipCrease


def smooth(value):
    x=np.clip(value,0,1)
    return x*x*(3-2*x)


def curveFunction(xy,weights=None):
    xy=np.asarray(xy,float)
    if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<8 or not np.isfinite(xy).all():
        raise ValueError('Expected a finite densely sampled lip curve')
    order=np.argsort(xy[:,0]);xy=xy[order]
    if np.any(np.diff(xy[:,0])<=0):raise ValueError('Lip curve x must be strictly increasing')
    w=np.ones(len(xy)) if weights is None else np.asarray(weights,float)[order]
    if w.shape!=(len(xy),) or not np.isfinite(w).all() or np.any(w<0):raise ValueError('Invalid curve weights')
    spline=UnivariateSpline(xy[:,0],xy[:,1],w=np.sqrt(np.maximum(w,.001)),s=.04*max(w.sum(),1),k=3,ext=3)
    return spline,(float(xy[0,0]),float(xy[-1,0]))


def makeField(vertices,camera,sourceCurve,targetCurve,*,amount=1.,radius=45.,targetConfidence=None,mode='affine'):
    """One smooth vertical image-plane field shared by every nearby layer."""
    K=np.asarray(camera['intrinsics']);R=np.asarray(camera['rotation']);t=np.asarray(camera['translation'])
    cv=vertices@R.T+t;p=cv@K.T;xy=p[:,:2]/p[:,2:]
    source,sourceRange=curveFunction(sourceCurve);target,targetRange=curveFunction(targetCurve,targetConfidence)
    lo=max(sourceRange[0],targetRange[0]);hi=min(sourceRange[1],targetRange[1])
    if hi-lo<35:raise ValueError('Not enough common lip-curve support')
    center=source(xy[:,0]);shift=target(xy[:,0])-center
    xWeight=smooth((xy[:,0]-lo)/10)*smooth((hi-xy[:,0])/10)
    yWeight=smooth(1-np.abs(xy[:,1]-center)/radius)
    coefficients=None
    if mode=='affine':
        # A shared affine map in camera coordinates preserves the relative
        # arrangement of the inner mouth triangles. Taper only outside it.
        tx=targetCurve[:,0];difference=targetCurve[:,1]-source(tx)
        weight=np.ones(len(tx)) if targetConfidence is None else targetConfidence
        design=np.column_stack([tx-260.,np.ones(len(tx))])
        coefficients=np.linalg.lstsq(design*np.sqrt(weight[:,None]),difference*np.sqrt(weight),rcond=None)[0]
        shift=(xy[:,0]-260)*coefficients[0]+coefficients[1]
        xWeight=smooth((xy[:,0]-(lo-35))/20)*smooth(((hi+35)-xy[:,0])/20)
        yWeight=smooth((xy[:,1]-305)/40)*smooth((435-xy[:,1])/60)
    lip=(xy[:,0]>=lo)&(xy[:,0]<=hi)&(np.abs(xy[:,1]-center)<10)
    # The back of the head can project into the same rectangle. Restrict the
    # field to the front depth slab, with a continuous transition behind it.
    frontDepth=float(np.quantile(cv[lip,2],.2))
    extent=float(np.ptp(vertices,axis=0).max())
    zWeight=1-smooth((cv[:,2]-(frontDepth+.12*extent))/(.12*extent))
    displacementPixels=amount*shift*xWeight*yWeight*zWeight
    imageDelta=np.column_stack([np.zeros(len(vertices)),displacementPixels,np.zeros(len(vertices))])
    cameraDelta=(imageDelta@np.linalg.inv(K).T)*cv[:,2,None]
    delta=cameraDelta@R
    candidate=vertices+delta
    active=displacementPixels!=0
    candidate[~active]=vertices[~active]
    # Sample the continuous y derivative over the whole support. Positive is
    # necessary for a one-to-one vertical map; discrete mesh checks still run.
    gx,gy=np.meshgrid(np.linspace(lo,hi,301),np.linspace(float(sourceCurve[:,1].min())-radius,float(sourceCurve[:,1].max())+radius,301))
    rel=gy-source(gx);u=np.clip(1-np.abs(rel)/radius,0,1)
    derivative=6*u*(1-u)*(-np.sign(rel)/radius)
    xw=smooth((gx-lo)/10)*smooth((hi-gx)/10)
    jacobian=1+amount*(target(gx)-source(gx))*xw*derivative
    if mode=='affine':
        gx,gy=np.meshgrid(np.linspace(lo-35,hi+35,401),np.linspace(305,435,401))
        fieldShift=((gx-260)*coefficients[0]+coefficients[1])*smooth((gx-(lo-35))/20)*smooth(((hi+35)-gx)/20)
        mapped=gy+amount*fieldShift*smooth((gy-305)/40)*smooth((435-gy)/60)
        jacobian=np.gradient(mapped,gy[:,0],axis=0)
    return candidate,{'amount':amount,'supportRadiusPixels':radius,'commonCurveXRange':[lo,hi],
        'changedVertexCount':int(active.sum()),'maximumImageDisplacementPixels':float(np.abs(displacementPixels).max()),
        'minimumSampledVerticalJacobian':float(jacobian.min()),'frontDepthSlabStart':frontDepth+.12*extent,
        'fixedReferenceDepth':True,'horizontalShiftPixels':0,
        'mode':mode,'affineDyFromXMinus260':None if coefficients is None else coefficients.tolist(),
        'affinePlateau':None if mode!='affine' else {'x':[lo-15,hi+15],'y':[345,375],
            'xFadePixels':20,'topFadePixels':40,'bottomFadePixels':60},
        'method':'shared image-plane vertical displacement; reconstructed reference depth unchanged'}


def rebake(directory,geometry,cameras,delight):
    directory.mkdir(parents=True)
    executable=ROOT/'.runtimes/colmap/bin/colmap'
    quantized=np.rint(np.clip(delight['albedoImages'],0,1)*255).astype(np.uint8)
    workspaces={}
    workspaces['colorWorkspace']=prepareWorkspace(directory/'colorWorkspace',geometry,cameras,quantized)
    supportRgb=np.repeat((delight['validMasks'].astype(np.uint8)*255)[...,None],3,axis=-1)
    workspaces['supportWorkspace']=prepareWorkspace(directory/'supportWorkspace',geometry,cameras,supportRgb)
    markerRgb=np.zeros_like(quantized);active=workspaces['colorWorkspace']['activeViewIndices']
    for index,view in enumerate(active):markerRgb[view]=viewMarkerColors(len(active))[index]
    workspaces['markerWorkspace']=prepareWorkspace(directory/'markerWorkspace',geometry,cameras,markerRgb)
    records={}
    for workspace,result,support in [('colorWorkspace','colorOutput',False),('supportWorkspace','supportOutput',True),('markerWorkspace','markerOutput',True)]:
        records[result]=texturer(directory,None,workspace,result,support=support,executable=executable)
    names=('colorOutput','supportOutput','markerOutput')
    meshes=[loadTexturedMesh(directory/name/'mesh.ply',geometry) for name in names]
    for value in meshes[1:]:
        if not np.array_equal(value['faceUvs'],meshes[0]['faceUvs']) or not np.array_equal(value['validFaces'],meshes[0]['validFaces']):
            raise ValueError('COLMAP auxiliary layouts differ')
    atlases=[np.asarray(Image.open(directory/name/'texture.png').convert('RGB')) for name in names]
    if any(a.shape!=atlases[0].shape for a in atlases[1:]):raise ValueError('COLMAP atlas dimensions differ')
    support=rebakeAtlas(geometry,meshes[1],atlases[1],1024)
    markers=sourceSupportFromMarkers(geometry,meshes[2],atlases[2],cameras,delight['validMasks'],active,1024)
    observed=markers['coverageMask']&support['assignedMask']&(support['rawRebaked'].min(-1)>=.999)
    baked=rebakeAtlas(geometry,meshes[0],atlases[0],1024,{'coverageMask':observed,'sourceIndex':markers['sourceIndex']})
    baked['faceSourceIndex']=markers['faceSourceIndex']
    assets=AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleV1')
    originalReport=json.loads((ROOT/'runs/clientColmap20261012/maleExactRegression/report.json').read_text())
    gain=np.asarray(originalReport['templateFallbackGain'])
    fallback=np.clip(assets.materials[0]['baseColor'][...,:3]*gain,0,1)
    linear,_,_,_=clientColors(baked,fallback,assets.seamFeatherPixels)
    image=np.rint(encodeSrgb(linear)*255).astype(np.uint8)
    np.savez_compressed(directory/'bakedTexture.npz',**{k:v for k,v in baked.items() if isinstance(v,np.ndarray)})
    Image.fromarray(image).save(directory/'colmapTexture.png')
    writeJson(directory/'report.json',{'workspaces':workspaces,'colmap':records,'observedTexels':int(observed.sum()),
        'cameraRefit':False,'seconds':sum(value['seconds'] for value in records.values())})
    return image,baked


def creaseError(vertices,faces,camera,photo):
    trace=traceVisibleLipCrease(vertices,faces,camera,yRange=(336.,367.))
    curve=trace['pixels'][trace['validMask']]
    keep=photo['trustedMask']&(photo['x']>=curve[:,0].min())&(photo['x']<=curve[:,0].max())
    residual=np.interp(photo['x'][keep],curve[:,0],curve[:,1])-photo['y'][keep]
    return {'trustedPointCount':int(keep.sum()),'meanAbsoluteYErrorPixels':float(np.abs(residual).mean()),
        'medianSignedYErrorPixels':float(np.median(residual)),'maxAbsoluteYErrorPixels':float(np.abs(residual).max())},trace


def sideBindingErrors(vertices,state,cameras,inputs,case):
    pointIds=state['ibugIds']>=48
    positions=(vertices[state['vertexIndices'][pointIds]]*state['barycentricWeights'][pointIds,:,None]).sum(1)
    errors=[]
    for camera,item,observation in zip(cameras,inputs,case['observationViews']):
        cached=archive(observation['npz'])
        source=cached['faceLandmarks'][state['ibugIds'][pointIds]]
        p=np.column_stack([source,np.ones(len(source))])@np.asarray(item['inputToWorking']).T
        targets=p[:,:2]/p[:,2:]
        projected=projectPoints(positions,np.asarray(camera['intrinsics']),np.asarray(camera['rotation']),np.asarray(camera['translation']))[0]
        errors.append(float(np.linalg.norm(projected-targets,axis=1).mean()))
    return errors


def renderEvidence(output,geometries,textures,cameras,delight,inputs,case,state,sourceCurve,targetCurve):
    names=list(geometries);labels={'baseline':'Before / fixed texture','geometryOnly':'Aligned crease / SAME texture','reprojected':'Aligned crease / REBAKED texture'}
    box=(198,313,325,389);panels=[];sidePanels=[]
    for view,camera in enumerate(cameras):
        photo=ImageOps.exif_transpose(Image.open(case['sourceImages'][view]['path'])).convert('RGB')
        photo=Image.fromarray(cv2.warpPerspective(np.asarray(photo),np.asarray(inputs[view]['inputToWorking']),tuple(inputs[view]['workingSize'])))
        row=[photo]
        for name in names:
            g=geometries[name]
            base=renderTexture(g['vertices'],g['faces'],g['uvs'],g['uvFaces'],Image.fromarray(textures[name]),camera)
            lit=renderTexture(g['vertices'],g['faces'],g['uvs'],g['uvFaces'],Image.fromarray(textures[name]),camera,light=True)
            clay=renderProjected(g['vertices'],g['faces'],{k:np.asarray(camera[k]) for k in ('intrinsics','rotation','translation')})[0]
            if view==0:panels.append((name,base,lit,clay))
            row.extend([lit,clay])
        sidePanels.append(row)
        if view==0:
            photo.save(output/'photo.png')
            overview=Image.new('RGB',(512*4,550),(28,35,39));draw=ImageDraw.Draw(overview)
            for i,(label,picture) in enumerate([('Original photo',photo),*[(labels[n],p[1]) for n,p in zip(names,panels)]]):
                overview.paste(picture,(i*512,38));draw.text((i*512+8,12),label,fill='white')
            overview.save(output/'overview.png')
    width,height=384,268
    comparison=Image.new('RGB',(4*width,3*height),(28,35,39));draw=ImageDraw.Draw(comparison)
    photo=Image.open(output/'photo.png')
    for row,mode in enumerate(['Base color','Same light','Clay']):
        for col,(label,picture) in enumerate([('Original photo',photo),*[(labels[p[0]],p[row+1]) for p in panels]]):
            crop=picture.crop(box);scale=min(372/crop.width,212/crop.height)
            crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS)
            comparison.paste(crop,(col*width+(width-crop.width)//2,row*height+48))
            draw.text((col*width+8,row*height+13),mode+' | '+label,fill='white')
    comparison.save(output/'mouth.png')
    side=Image.new('RGB',(5*512,3*550),(28,35,39));draw=ImageDraw.Draw(side)
    for row,view in enumerate(range(1,len(cameras))):
        frames=sidePanels[view]
        for col,(label,picture) in enumerate(zip(['Photo','Before / light','Before / clay','After / light','After / clay'],[frames[0],frames[1],frames[2],frames[5],frames[6]])):
            side.paste(picture,(col*512,row*550+38));draw.text((col*512+8,row*550+12),f'View {view}: '+label,fill='white')
    side.save(output/'side.png')
    overlay=photo.copy();draw=ImageDraw.Draw(overlay)
    g=geometries['reprojected']
    afterTrace=traceVisibleLipCrease(g['vertices'],g['faces'],cameras[0],yRange=(336.,367.))
    for curve,color in [(sourceCurve,'orange'),(targetCurve,'cyan'),(afterTrace['pixels'][afterTrace['validMask']],'lime')]:
        draw.line([tuple(p) for p in curve],fill=color,width=1)
    zoom=overlay.crop(box).resize((762,456),Image.Resampling.NEAREST)
    diagnostic=Image.new('RGB',(762,496),(28,35,39));diagnostic.paste(zoom,(0,40))
    ImageDraw.Draw(diagnostic).text((12,13),'Orange: original depth crease | Cyan: photo proxy | Green: adjusted depth crease',fill='white')
    diagnostic.save(output/'diagnostic.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/maleLipGeometry20261012/affineSmooth')
    parser.add_argument('--mode',choices=('curve','affine'),default='affine')
    args=parser.parse_args();start=perf_counter()
    output=args.output/'maleExactRegression'
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    baselineSpec=json.loads((ROOT/'reports/clientColmapBaseline20261012.json').read_text())
    for record in baselineSpec['fixedReferenceArtifacts'].values():
        if digest(ROOT/record['path'])!=record['sha256']:raise ValueError('Frozen input changed: '+record['path'])
    base=ROOT/'runs/clientColmap20261012/maleExactRegression';source=ROOT/'reports/frontLandmark20261012/maleSource'
    geometry=archive(base/'geometry.npz');geometry['uvConvention']='topLeftPixelEdges'
    state=archive(ROOT/'runs/frontLandmark20261012/maleExactRegression/landmarks.npz')
    template=archive(ROOT/'assets/clientTemplates/legacyMaleV1/template.npz');fixed=np.flatnonzero(template['staticMask'])
    photoTrace=archive(ROOT/'runs/maleLipGeometry20261012/photoTrace/trace.npz')
    cameras=json.loads((source/'cameras.json').read_text());inputs=json.loads((source/'inputs.json').read_text())
    delight=archive(source/'delight.npz')
    case=next(c for c in json.loads((ROOT/'reports/frontLandmark20261012/sources.json').read_text())['cases'] if c['caseId']=='maleExactRegression')
    before=geometry['vertices'];faces=geometry['faces'];extent=float(np.ptp(before,axis=0).max())
    beforeError,trace=creaseError(before,faces,cameras[0],photoTrace)
    sourceCurve=trace['pixels'][trace['validMask']]
    beforeHits=intersections(before,faces);beforePairs=pairSet(beforeHits)
    bindingBefore=sideBindingErrors(before,state,cameras,inputs,case)
    attempts=[];chosen=before.copy();chosenAmount=0;chosenError=beforeError
    for amount in (1.,.75,.5,.25,.125,.0625):
        candidate,field=makeField(before,cameras[0],sourceCurve,photoTrace['xy'],amount=amount,targetConfidence=photoTrace['confidence'],mode=args.mode)
        local=checkFrontGeometry(before,candidate,faces,camera=cameras[0],fixedVertexIds=fixed,
            maxDisplacement=extent*.04,maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3)
        hits=intersections(candidate,faces);pairs=pairSet(hits)
        error,newTrace=creaseError(candidate,faces,cameras[0],photoTrace)
        sideErrors=sideBindingErrors(candidate,state,cameras,inputs,case)
        record={'field':field,'geometry':local,'intersectionPairs':len(pairs),'newIntersectionPairs':len(pairs-beforePairs),
            'newFacePairs':sorted(pairs-beforePairs),'removedIntersectionPairs':len(beforePairs-pairs),'creaseProxy':error,
            'approximateBindingMouthMeanErrors':sideErrors,'approximateBindingErrorChanges':(np.asarray(sideErrors)-bindingBefore).tolist()}
        attempts.append(record);print('geometry',json.dumps(record),flush=True)
        # Binding errors remain approximate; limit the worsening of either
        # lateral view without treating the unreliable binding as ground truth.
        sideSafe=max(np.asarray(sideErrors[1:3])-bindingBefore[1:3])<=1.
        accepted=(local['accepted'] and not pairs-beforePairs and field['minimumSampledVerticalJacobian']>.25
            and error['meanAbsoluteYErrorPixels']<beforeError['meanAbsoluteYErrorPixels'] and sideSafe)
        record['sideProxyGate']=bool(sideSafe);record['accepted']=bool(accepted)
        if accepted:
            chosen=candidate;chosenAmount=amount;chosenError=error
            break
    report={'caseId':'maleExactRegression','beforeCreaseProxy':beforeError,'afterCreaseProxy':chosenError,
        'chosenAmount':chosenAmount,'attempts':attempts,'beforeIntersectionPairs':len(beforePairs),'cameraRefit':False,
        'remoteChanged':False,'sourceCurve':'visible depth peak; not approximate iBUG inner binding',
        'targetCurve':'automatic photo ridge; confidence weighted, endpoints tapered; not manual ground truth',
        'approximateBindingMouthErrorsBefore':bindingBefore,
        'limits':['Depth and horizontal coordinates are preserved; deep fold and existing intersections are not repaired.',
            'Only nonadjacent positive-area/segment intersections are compared; shared-vertex and continuous-motion collisions are not certified.',
            'One male sample, fixed saved cameras, automatic photo proxy, in-sample comparison.']}
    writeJson(output/'geometryReport.json',report)
    if chosenAmount==0:raise RuntimeError('No geometric candidate passed; see geometryReport.json')
    after={**geometry,'vertices':chosen}
    np.savez_compressed(output/'geometry.npz',**{k:v for k,v in after.items() if isinstance(v,np.ndarray)})
    np.savez_compressed(output/'deformation.npz',before=before,after=chosen,delta=chosen-before,faces=faces,fixedVertexIds=fixed,sourceCurve=sourceCurve,targetCurve=photoTrace['xy'])
    textureRoot=ROOT/'runs/maleMouthSeam20261012/final/maleExactRegression'
    baselineTexture=np.asarray(Image.open(textureRoot/'branch3.png').convert('RGB'))
    colmap,baked=rebake(output/'rebake',after,cameras,delight)
    mouth,provenance,mouthReport=makeCandidate(after,colmap,baked,delight,cameras[0],state,expandSupport=True,minCosine=0)
    reprojected,provenance,filterReport=addFilterFootprint(after,mouth,provenance,delight,cameras[0],state)
    np.savez_compressed(output/'textureProvenance.npz',**provenance)
    geometries={'baseline':geometry,'geometryOnly':after,'reprojected':after}
    textures={'baseline':baselineTexture,'geometryOnly':baselineTexture,'reprojected':reprojected}
    oldPreview=json.loads((textureRoot/'preview.json').read_text());oldDisplay=np.asarray(oldPreview['branches'][0]['vertices'])
    affine=np.linalg.lstsq(np.column_stack([before,np.ones(len(before))]),oldDisplay,rcond=None)[0]
    if np.abs(np.column_stack([before,np.ones(len(before))])@affine-oldDisplay).max()>1e-8:raise ValueError('Display transform mismatch')
    display=oldDisplay+(chosen-before)@affine[:3]
    display[fixed]=oldDisplay[fixed]
    labels={'baseline':'当前基线 · 正面取色已修正','geometryOnly':'对齐几何唇缝 · 原贴图','reprojected':'对齐几何唇缝 · 重新烘焙'}
    branches=[];appearance={'schemaVersion':1,'caseId':'maleExactRegression','uvOrigin':'topLeftVDown','branches':{}}
    for i,(name,pixels) in enumerate(textures.items()):
        path=output/f'branch{i}.png'
        if name!='reprojected':path.write_bytes((textureRoot/'branch3.png').read_bytes())
        else:Image.fromarray(pixels).save(path)
        branches.append({'id':name,'label':labels[name],'vertices':(oldDisplay if name=='baseline' else display).tolist(),'faces':faces.tolist()})
        appearance['branches'][name]={'mesh':{'uvs':geometry['uvs'].tolist(),'uvFaces':geometry['uvFaces'].tolist()},
            'texture':{'url':f'/data/maleExactRegression/{path.name}','sha256':digest(path),'sizeBytes':path.stat().st_size,'colorSpace':'sRGB'},'provenanceLabel':labels[name]}
    writeJson(output/'preview.json',{**oldPreview,'branches':branches});writeJson(output/'appearance.json',appearance)
    renderEvidence(output,geometries,textures,cameras,delight,inputs,case,state,sourceCurve,photoTrace['xy'])
    report.update(status='localCandidateReady',mouthTexture=mouthReport,filterTexture=filterReport,
        maxDisplayDisplacementMm=float(np.linalg.norm(display-oldDisplay,axis=1).max()*10),
        baselineFilesUnchanged=all(digest(ROOT/r['path'])==r['sha256'] for r in baselineSpec['fixedReferenceArtifacts'].values()),
        seconds=perf_counter()-start)
    writeJson(output/'report.json',report)
    writeJson(args.output/'experiment.json',{'status':'succeeded','objective':'比较可见几何唇缝对齐及新几何取色。','cases':[{'id':'maleExactRegression','label':'男生扭头基准 · 几何唇缝对齐','previewUrl':'/data/maleExactRegression/preview.json'}]})
    print(json.dumps({k:report[k] for k in ('status','chosenAmount','beforeCreaseProxy','afterCreaseProxy','maxDisplayDisplacementMm','seconds')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
