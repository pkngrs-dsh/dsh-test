"""Frozen male-baseline mouth projection with explicit visibility and provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import distance_transform_edt

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from faceAppearance.colmapTexture import _destinationUvLookup
from faceAppearance.texture import samplePixels
from faceAppearance.headAssembly import decodeColor
from faceAppearance.surfaceVisibility import ProjectedSurfaceVisibility
from prepareFrontLandmarkTextures import renderTexture,encodeSrgb
from runClientColmapExperiment import archive,digest,writeJson


def smooth(value):
    x=np.clip(value,0,1);return x*x*(3-2*x)


def makeCandidate(geometry,baseline,baked,delight,camera,state,*,expandSupport=False,minCosine=.05):
    lookup,_=_destinationUvLookup(geometry,baseline.shape[0])
    valid=lookup['validMask'];uvIndices=np.column_stack(np.nonzero(valid))
    faceIds=lookup['faceIds'][valid]
    points=(geometry['vertices'][geometry['faces'][faceIds]]*lookup['barycentrics'][valid,:,None]).sum(1)
    K=np.asarray(camera['intrinsics']);R=np.asarray(camera['rotation']);t=np.asarray(camera['translation'])
    cv=geometry['vertices']@R.T+t;cp=points@R.T+t
    projected=cp@K.T;xy=projected[:,:2]/projected[:,2:]
    lip=state['frontTargetPixels'][(state['ibugIds']>=48)&(state['ibugIds']<60)]
    polygon=np.zeros((512,512),np.uint8);cv2.fillPoly(polygon,[np.rint(lip).astype(np.int32)],1)
    distance=distance_transform_edt(~polygon.astype(bool)).astype(np.float32)
    band=samplePixels(distance,xy)
    bounds=((xy>=0).all(1)&(xy<[511,511]).all(1)&(cp[:,2]>0))
    region=bounds&(band<12)
    selected=np.flatnonzero(region)
    triangles=cv[geometry['faces']]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-15)
    cos=np.sum(normals[faceIds[selected]]*(-cp[selected]/np.linalg.norm(cp[selected],axis=1,keepdims=True)),axis=1)
    mask=samplePixels(delight['validMasks'][0].astype(np.float32),xy[selected])>=.999
    visibility=ProjectedSurfaceVisibility(cv,geometry['faces'],K,(512,512))
    hit=visibility.closestDepth(cp[selected])
    tolerance=float(np.ptp(geometry['vertices'],axis=0).max())*1e-6
    depth=np.isfinite(hit['depth'])&(np.abs(hit['depth']-cp[selected,2])<=tolerance)
    frontGood=mask&depth&(cos>minCosine)
    if not expandSupport: frontGood &= baked['coverageMask'][valid][selected]
    accepted=selected[frontGood]
    alphaValues=1-smooth((band[accepted]-3)/9)
    alphaValues[alphaValues<1e-7]=0
    alpha=np.zeros(valid.shape,np.float32);alpha[tuple(uvIndices[accepted].T)]=alphaValues
    changed=alpha>0
    frontColors=samplePixels(decodeColor(delight['albedoImages'][0],'gamma2.2'),xy[accepted])
    linear=decodeColor(baseline/255.,'srgb')
    newLinear=linear.copy()
    index=tuple(uvIndices[accepted].T)
    newLinear[index]=linear[index]*(1-alphaValues[:,None])+frontColors*alphaValues[:,None]
    result=baseline.copy();result[changed]=np.rint(encodeSrgb(newLinear[changed])*255).astype(np.uint8)
    coverage=baked['coverageMask'].copy();coverage[changed]=True
    sourceIndex=baked['sourceIndex'].copy();sourceIndex[changed]=-2;sourceIndex[alpha==1]=0
    # Texture filtering gutter only; no observation is added outside the surface.
    distances,nearest=distance_transform_edt(~coverage,return_indices=True)
    gutter=(~valid)&(distances<=2)&changed[nearest[0],nearest[1]]
    result[gutter]=result[nearest[0][gutter],nearest[1][gutter]]
    outside=~(changed|gutter)
    report={'expandWithValidFrontPhoto':expandSupport,'changedTexels':int(changed.sum()),'fullyFrontTexels':int((alpha==1).sum()),
        'newObservedTexels':int((coverage&~baked['coverageMask']).sum()),'gutterTexelsNotObserved':int(gutter.sum()),
        'outsideLocalRegionByteIdentical':bool(np.array_equal(result[outside],baseline[outside])),
        'coreRadiusWorkingPixels':3,'outerRadiusWorkingPixels':12,'frontViewIndex':0,
        'localSurfaceTexels':int(region.sum()),'frontMaskRejectedTexels':int((~mask).sum()),
        'occlusionRejectedTexels':int((~depth).sum()),'normalRejectedTexels':int((cos<=minCosine).sum()),
        'depthTolerance':tolerance,'minCosine':minCosine,'visibility':'continuous perspective ray depth, two-sided occlusion; no camera refit',
        'colorOperation':'sample same-run OpenDelight linear RGB; blend in linear; encode sRGB once; exterior PNG bytes preserved'}
    return result,{'coverageMask':coverage,'frontAlpha':alpha,'sourceIndex':sourceIndex,'gutter':gutter},report


def addFilterFootprint(geometry,texture,provenance,delight,camera,state):
    """Pad only the bilinear footprint of actually front-visible lip surface.

    A texel center may be hidden although a visible thin triangle contributes
    to its filter kernel. Those values are filtering support, not newly observed
    center samples. Integrate visible source samples over the one-texel tent
    kernel; preserve every center that already has a front-photo observation.
    """
    size=texture.shape[0]
    K=np.asarray(camera['intrinsics']);R=np.asarray(camera['rotation']);t=np.asarray(camera['translation'])
    cv=geometry['vertices']@R.T+t
    visibility=ProjectedSurfaceVisibility(cv,geometry['faces'],K,(512,512))
    triangles=cv[geometry['faces']]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-15)
    lip=state['frontTargetPixels'][(state['ibugIds']>=48)&(state['ibugIds']<60)]
    polygon=np.zeros((512,512),np.uint8);cv2.fillPoly(polygon,[np.rint(lip).astype(np.int32)],1)
    distance=distance_transform_edt(~polygon.astype(bool)).astype(np.float32)
    tolerance=float(np.ptp(geometry['vertices'],axis=0).max())*1e-6
    rgbSum=np.zeros((*texture.shape[:2],3),np.float64)
    weightSum=np.zeros(texture.shape[:2],np.float64)
    alphaSum=np.zeros_like(weightSum)
    candidate=provenance['frontAlpha']==0
    centerLookup,_=_destinationUvLookup(geometry,size)
    owners={}
    # UV and geometric vertex identities must both agree to connect faces.
    corners=[set(zip(face.tolist(),uvFace.tolist())) for face,uvFace in zip(geometry['faces'],geometry['uvFaces'])]
    source=decodeColor(delight['albedoImages'][0],'gamma2.2')
    offsets=(-.75,-.25,.25,.75)
    totalWeight=sum((1-abs(x))*(1-abs(y)) for x in offsets for y in offsets)
    for dy in offsets:
        for dx in offsets:
            shifted={**geometry,'uvs':geometry['uvs']-np.array([dx,dy])/size}
            lookup,_=_destinationUvLookup(shifted,size)
            valid=lookup['validMask']&candidate
            indices=np.column_stack(np.nonzero(valid));faceIds=lookup['faceIds'][valid]
            points=(cv[geometry['faces'][faceIds]]*lookup['barycentrics'][valid,:,None]).sum(1)
            projected=points@K.T;xy=projected[:,:2]/projected[:,2:]
            bounds=(xy>=0).all(1)&(xy<[511,511]).all(1)&(points[:,2]>0)
            band=samplePixels(distance,xy)
            cos=np.sum(normals[faceIds]*(-points/np.linalg.norm(points,axis=1,keepdims=True)),axis=1)
            selected=np.flatnonzero(bounds&(band<12)&(cos>0))
            if not len(selected):continue
            hit=visibility.closestDepth(points[selected])
            good=np.isfinite(hit['depth'])&(np.abs(hit['depth']-points[selected,2])<=tolerance)
            good &= samplePixels(delight['validMasks'][0].astype(np.float32),xy[selected])>=.999
            accepted=selected[good]
            index=tuple(indices[accepted].T)
            for (row,column),faceId in zip(indices[accepted],faceIds[accepted]):
                owners.setdefault((int(row),int(column)),set()).add(int(faceId))
            weight=(1-abs(dx))*(1-abs(dy))
            rgbSum[index]+=weight*samplePixels(source,xy[accepted])
            alphaSum[index]+=weight*(1-smooth((band[accepted]-3)/9))
            weightSum[index]+=weight
    padding=weightSum>0
    rejected=0
    for pixel,faceSet in owners.items():
        if centerLookup['validMask'][pixel]:faceSet.add(int(centerLookup['faceIds'][pixel]))
        remaining=set(faceSet);connected={remaining.pop()}
        while remaining:
            neighbors={face for face in remaining if any(corners[face]&corners[other] for other in connected)}
            if not neighbors:break
            connected.update(neighbors);remaining.difference_update(neighbors)
        if remaining:
            padding[pixel]=False;weightSum[pixel]=0;rejected+=1
    linear=decodeColor(texture/255.,'srgb')
    alpha=alphaSum[padding]/weightSum[padding]
    linear[padding]=linear[padding]*(1-alpha[:,None])+(rgbSum[padding]/weightSum[padding,None])*alpha[:,None]
    result=texture.copy();result[padding]=np.rint(encodeSrgb(linear[padding])*255).astype(np.uint8)
    info={**provenance,'filterPadding':padding,'filterObservedKernelFraction':weightSum/totalWeight}
    report={'filterPaddingTexels':int(padding.sum()),'newObservedCenterTexels':0,
        'disconnectedUvGeometryFootprintsRejected':rejected,
        'observedCoverageUnchanged':bool(np.array_equal(info['coverageMask'],provenance['coverageMask'])),
        'outsidePaddingByteIdentical':bool(np.array_equal(result[~padding],texture[~padding])),
        'kernel':'bilinear tent, radius 1 texel, 4x4 subpositions',
        'source':'only exact front-visible positive-facing surface with a valid source-photo mask',
        'adjacency':'contributing faces and center owner must connect through shared (geometry vertex, UV vertex) pairs inside this footprint',
        'meaning':'filter padding for thin surface footprints; not a surface-center observation or geometric repair'}
    return result,info,report


def viewsAndMetrics(out,geometry,textures,cameras,delight,inputs,case,state,coverage):
    names=list(textures);images={name:Image.fromarray(value) for name,value in textures.items()}
    maskImage=Image.fromarray(np.repeat((coverage*255).astype(np.uint8)[...,None],3,axis=-1))
    frames=[];metrics=[]
    for view,camera in enumerate(cameras):
        item=inputs[view]
        photo=ImageOps.exif_transpose(Image.open(case['sourceImages'][view]['path'])).convert('RGB')
        photo=Image.fromarray(cv2.warpPerspective(np.asarray(photo),np.asarray(item['inputToWorking']),tuple(item['workingSize'])))
        target=np.rint(encodeSrgb(decodeColor(delight['albedoImages'][view],'gamma2.2'))*255).astype(np.uint8)
        renders={name:renderTexture(geometry['vertices'],geometry['faces'],geometry['uvs'],geometry['uvFaces'],image,camera) for name,image in images.items()}
        mask=np.asarray(renderTexture(geometry['vertices'],geometry['faces'],geometry['uvs'],geometry['uvFaces'],maskImage,camera)).min(-1)>=254
        # The same mouth ROI is projected from fixed landmarks in every branch.
        positions=(geometry['vertices'][state['vertexIndices'][state['ibugIds']>=48]]*state['barycentricWeights'][state['ibugIds']>=48,:,None]).sum(1)
        cv=positions@np.asarray(camera['rotation']).T+camera['translation'];p=cv@np.asarray(camera['intrinsics']).T;p=p[:,:2]/p[:,2:]
        if view==0:p=state['frontTargetPixels'][state['ibugIds']>=48]
        lo=np.maximum(np.floor(p.min(0)-18).astype(int),0);hi=np.minimum(np.ceil(p.max(0)+18).astype(int),512)
        region=np.zeros((512,512),bool);region[lo[1]:hi[1],lo[0]:hi[0]]=True
        common=mask&delight['validMasks'][view]
        record={'viewIndex':view,'mouthRoiXYXY':[*lo.tolist(),*hi.tolist()],'commonMouthPixels':int((common&region).sum()),'rgbMaeSrgb8':{}}
        for name,image in renders.items():
            error=np.abs(np.asarray(image).astype(float)-target).mean(-1)
            record['rgbMaeSrgb8'][name]={'fullSupported':float(error[common].mean()),'mouth':float(error[common&region].mean())}
        metrics.append(record);frames.append({'photo':photo,'input':Image.fromarray(target),**renders})
    display=['photo','input',*names]
    labels={'photo':'Original photo','input':'OpenDelight input','colmapBaseline':'Frozen COLMAP baseline',
            'frontMouth':'Front mouth / same coverage','frontMouthSupport':'Front mouth / valid support recovery',
            'frontMouthFiltered':'Front mouth / filter footprint'}
    overview=Image.new('RGB',(512*len(display),558),(28,35,39));draw=ImageDraw.Draw(overview)
    for i,name in enumerate(display):overview.paste(frames[0][name],(i*512,46));draw.text((i*512+12,15),labels[name],fill='white')
    overview.save(out/'overview.png')
    box=metrics[0]['mouthRoiXYXY'];panel=Image.new('RGB',(384*len(display),268),(28,35,39));draw=ImageDraw.Draw(panel)
    for i,name in enumerate(display):
        part=frames[0][name].crop(box);scale=min(372/part.width,212/part.height);part=part.resize((round(part.width*scale),round(part.height*scale)),Image.Resampling.LANCZOS)
        panel.paste(part,(i*384+(384-part.width)//2,48));draw.text((i*384+8,14),labels[name],fill='white')
    panel.save(out/'mouth.png')
    sideNames=['photo',*names];side=Image.new('RGB',(512*len(sideNames),550*3),(28,35,39));draw=ImageDraw.Draw(side)
    for row,view in enumerate(range(1,4)):
        for col,name in enumerate(sideNames):side.paste(frames[view][name],(col*512,row*550+36));draw.text((col*512+10,row*550+10),f'View {view}: '+labels[name],fill='white')
    side.save(out/'side.png')
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/maleMouthSeam20261012/final')
    parser.add_argument('--min-cosine',type=float,default=0)
    parser.add_argument('--filter-footprint',action=argparse.BooleanOptionalAction,default=True)
    args=parser.parse_args()
    if not 0<=args.min_cosine<1:parser.error('--min-cosine must be in [0, 1)')
    started=perf_counter();baselineSpec=json.loads((ROOT/'reports/clientColmapBaseline20261012.json').read_text())
    for record in baselineSpec['fixedReferenceArtifacts'].values():
        if digest(ROOT/record['path'])!=record['sha256']:raise ValueError('Pinned baseline changed: '+record['path'])
    out=args.output/'maleExactRegression'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    base=ROOT/'runs/clientColmap20261012/maleExactRegression';source=ROOT/'reports/frontLandmark20261012/maleSource'
    geometry=archive(base/'geometry.npz');geometry['uvConvention']='topLeftPixelEdges'
    baked=archive(base/'bakedTexture.npz');state=archive(ROOT/'runs/frontLandmark20261012/maleExactRegression/landmarks.npz')
    delight=archive(source/'delight.npz');cameras=json.loads((source/'cameras.json').read_text());inputs=json.loads((source/'inputs.json').read_text())
    baseline=np.asarray(Image.open(base/'colmapTexture.png').convert('RGB'))
    case=next(c for c in json.loads((ROOT/'reports/frontLandmark20261012/sources.json').read_text())['cases'] if c['caseId']=='maleExactRegression')
    images={'colmapBaseline':baseline};reports={};coverage=baked['coverageMask'].copy()
    provenance={}
    for name,expand in [('frontMouth',False),('frontMouthSupport',True)]:
        images[name],provenance[name],reports[name]=makeCandidate(geometry,baseline,baked,delight,cameras[0],state,expandSupport=expand,minCosine=args.min_cosine)
        np.savez_compressed(out/f'{name}.npz',**provenance[name]);coverage &= provenance[name]['coverageMask']
        print(name,reports[name],flush=True)
    if args.filter_footprint:
        name='frontMouthFiltered'
        images[name],provenance[name],reports[name]=addFilterFootprint(geometry,images['frontMouthSupport'],provenance['frontMouthSupport'],delight,cameras[0],state)
        np.savez_compressed(out/f'{name}.npz',**provenance[name]);print(name,reports[name],flush=True)
    oldPreview=json.loads((base/'preview.json').read_text());branch=oldPreview['branches'][0];branches=[];appearance={'schemaVersion':1,'caseId':'maleExactRegression','uvOrigin':'topLeftVDown','branches':{}}
    labels={'colmapBaseline':'固定 COLMAP 基线','frontMouth':'嘴部正面优先 · 原覆盖','frontMouthSupport':'嘴部正面优先 · 补回可见观测',
            'frontMouthFiltered':'嘴部正面优先 · 过滤边缘补色'}
    for i,(name,pixels) in enumerate(images.items()):
        filename=f'branch{i}.png';path=out/filename
        if i==0:path.write_bytes((base/'colmapTexture.png').read_bytes())
        else:Image.fromarray(pixels).save(path)
        branches.append({**branch,'id':name,'label':labels[name]})
        appearance['branches'][name]={'mesh':{'uvs':geometry['uvs'].tolist(),'uvFaces':geometry['uvFaces'].tolist()},
            'texture':{'url':f'/data/maleExactRegression/{filename}','sha256':digest(path),'sizeBytes':path.stat().st_size,'colorSpace':'sRGB'},'provenanceLabel':labels[name]}
    writeJson(out/'preview.json',{**oldPreview,'branches':branches});writeJson(out/'appearance.json',appearance)
    report={'caseId':'maleExactRegression','status':'succeeded','geometryAndCamerasFrozen':True,'variants':reports,
        'comparison':viewsAndMetrics(out,geometry,images,cameras,delight,inputs,case,state,coverage),
        'seconds':perf_counter()-started,'baselineSpecSha256':digest(ROOT/'reports/clientColmapBaseline20261012.json'),
        'baselineFilesUnchanged':all(digest(ROOT/r['path'])==r['sha256'] for r in baselineSpec['fixedReferenceArtifacts'].values()),
        'limits':['Fixed camera front projection optimizes an observed input, not independent shape or identity quality.',
                  'No geometry holes are filled and no points are moved. Any real aperture remains.',
                  'Additional support only uses front-photo mask, positive normal and exact continuous visibility.',
                  'Side views must be compared; exterior texture boundary remains unchanged.'], 'remoteChanged':False}
    writeJson(out/'report.json',report)
    writeJson(args.output/'experiment.json',{'status':'succeeded','objective':'固定男生几何，检查正面嘴部取色对断线的影响。','cases':[{'id':'maleExactRegression','label':'男生扭头基准 · 嘴缝对照','previewUrl':'/data/maleExactRegression/preview.json'}]})
    Image.fromarray((coverage*255).astype(np.uint8)).save(out/'coverage.png')
    Image.fromarray(np.full((4,4,3),235,np.uint8)).save(out/'white.png')
    audit=ROOT/'runs/maleMouthSeam20261012/audit'
    if (audit/'report.json').exists() and (audit/'mouthSeamAudit.png').exists():
        if json.loads((audit/'report.json').read_text())['geometrySha256']==digest(base/'geometry.npz'):
            (out/'diagnostic.png').write_bytes((audit/'mouthSeamAudit.png').read_bytes())
    print(json.dumps({'status':report['status'],'seconds':report['seconds'],'comparison':report['comparison']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
