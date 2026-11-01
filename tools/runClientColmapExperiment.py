"""Bake cached photographs on FINAL client geometry, bypassing legacy UV transfer.

No reconstruction, registration, landmark fit, or camera refit is performed.
COLMAP changes only the atlas; same-face barycentrics retain client UV seams.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.colmapTexture import (prepareWorkspace, loadTexturedMesh,
    rebakeAtlas, sourceSupportFromMarkers, viewMarkerColors)
from faceAppearance.headAssembly import AssemblyAssets, decodeColor
from faceAppearance.camera import projectPoints
from prepareFrontLandmarkTextures import renderTexture, encodeSrgb
from runColmapTextureExperiment import texturer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(path):
    with np.load(path, allow_pickle=False) as value:
        return dict(value)


def writeJson(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def saveColor(path, linear):
    Image.fromarray(np.rint(encodeSrgb(linear[..., :3]) * 255).astype(np.uint8)).save(path)


def clientColors(baked, fallback, seamPixels):
    """Keep observed color unblended; gutter is explicitly not observation."""
    observed=baked['coverageMask'];linear=decodeColor(baked['albedo'],'gamma2.2')
    legacyAlpha=np.minimum(distance_transform_edt(observed)/seamPixels,1.)
    feathered=linear*legacyAlpha[...,None]+fallback*(1-legacyAlpha[...,None])
    result=np.where(observed[...,None],linear,fallback)
    distance,nearest=distance_transform_edt(~observed,return_indices=True)
    gutter=(~baked['atlasMask']) & (distance<=2.)
    result[gutter]=linear[nearest[0][gutter],nearest[1][gutter]]
    return result,feathered,gutter,legacyAlpha


def diagnosticFigures(directory, geometry, old, candidate, coverage, delight, cameras, inputs, case, state):
    oldImage, newImage = Image.open(old).convert('RGB'), Image.open(candidate).convert('RGB')
    maskImage = Image.fromarray(np.repeat((coverage.astype(np.uint8)*255)[..., None], 3, axis=-1))
    records, views = [], []
    mouthPoints = state['frontTargetPixels'][state['ibugIds'] >= 48]
    lo = np.maximum(np.floor(mouthPoints.min(0)-18).astype(int), 0)
    hi = np.minimum(np.ceil(mouthPoints.max(0)+18).astype(int), [512, 512])
    for view, (camera, item) in enumerate(zip(cameras, inputs)):
        photo = ImageOps.exif_transpose(Image.open(case['sourceImages'][view]['path'])).convert('RGB')
        work = Image.fromarray(cv2.warpPerspective(np.asarray(photo), np.asarray(item['inputToWorking']), tuple(item['workingSize'])))
        inputSrgb = np.rint(encodeSrgb(decodeColor(delight['albedoImages'][view], 'gamma2.2'))*255).astype(np.uint8)
        textureImages=[oldImage,newImage]
        names=['legacyTexture','colmapDirect']
        if (directory/'colmapLegacyFeather.png').is_file():
            textureImages.append(Image.open(directory/'colmapLegacyFeather.png').convert('RGB'))
            names.append('colmapLegacyFeather')
        renders = [renderTexture(geometry['vertices'], geometry['faces'], geometry['uvs'], geometry['uvFaces'], texture, camera)
                   for texture in textureImages]
        validRender = np.asarray(renderTexture(geometry['vertices'], geometry['faces'], geometry['uvs'], geometry['uvFaces'], maskImage, camera))
        common = (validRender.min(-1) >= 254) & delight['validMasks'][view]
        metric = {'viewIndex': view, 'commonObservedPixelCount': int(common.sum()), 'rgbMaeSrgb8': {}}
        for name, result in zip(names, renders):
            error = np.abs(np.asarray(result).astype(float)-inputSrgb.astype(float)).mean(-1)
            metric['rgbMaeSrgb8'][name] = float(error[common].mean()) if common.any() else None
            if view == 0:
                region = np.zeros(common.shape, bool); region[lo[1]:hi[1],lo[0]:hi[0]] = True
                keep = region & common
                metric.setdefault('mouthRoi', {'bboxXYXY': [*lo.tolist(),*hi.tolist()], 'commonPixelCount': int(keep.sum())})[name] = float(error[keep].mean()) if keep.any() else None
        records.append(metric)
        views.append([work, Image.fromarray(inputSrgb), *renders[:2]])
    titles = ['Original photo', 'OpenDelight input', 'Final mesh / old UV transfer', 'Same final mesh / direct COLMAP']
    panel = Image.new('RGB',(2048,566),(28,35,39)); draw=ImageDraw.Draw(panel)
    draw.text((12,8),case['caseId']+' | identical final geometry and camera | base color',fill='white')
    for index,(picture,title) in enumerate(zip(views[0],titles)):
        panel.paste(picture,(index*512,54));draw.text((index*512+12,32),title,fill='white')
    panel.save(directory/'comparison.png')
    # This is a scientific image panel: identical photo-coordinate crop in all columns.
    cropBox = tuple(int(x) for x in [*lo,*hi])
    mouth = Image.new('RGB',(4*384,260),(28,35,39));draw=ImageDraw.Draw(mouth)
    for index,(picture,title) in enumerate(zip(views[0],titles)):
        crop=picture.crop(cropBox)
        scale=min(372/crop.width,206/crop.height)
        crop=crop.resize((round(crop.width*scale),round(crop.height*scale)),Image.Resampling.LANCZOS)
        mouth.paste(crop,(index*384+(384-crop.width)//2,48+(206-crop.height)//2))
        draw.text((index*384+9,15),title,fill='white')
    mouth.save(directory/'mouthComparison.png')
    side = Image.new('RGB',(3*512,3*550),(28,35,39));draw=ImageDraw.Draw(side)
    for row,view in enumerate(range(1,min(4,len(views)))):
        for col,picture in enumerate([views[view][0],views[view][2],views[view][3]]):
            side.paste(picture,(col*512,row*550+36))
            draw.text((col*512+12,row*550+10),f'View {view}: '+['photo','old UV transfer','direct COLMAP'][col],fill='white')
    side.save(directory/'sideComparison.png')
    return records


def runCase(case, output, executable):
    start=perf_counter(); caseId=case['caseId']; directory=output/caseId
    if directory.exists(): raise FileExistsError(f'Use a new case output: {directory}')
    directory.mkdir(parents=True)
    original=ROOT/'runs/frontLandmark20261012'/caseId
    source=Path(case['geometryPath']).parent
    expected=json.loads((original/'report.json').read_text())
    if digest(original/'meshes.npz') != expected['meshSha256']: raise ValueError('Frozen landmark mesh changed')
    for name,key in [('geometryPath','geometrySha256'),('cameraPath','cameraSha256'),('inputsPath','inputsSha256')]:
        if digest(case[name]) != case[key]: raise ValueError(f'Source changed: {name}')
    inputs=json.loads(Path(case['inputsPath']).read_text());cameras=json.loads(Path(case['cameraPath']).read_text())
    for photo,item in zip(case['sourceImages'],inputs):
        if digest(photo['path'])!=item['sha256']: raise ValueError('Photo identity mismatch')
    if len(cameras)!=4 or any(not camera['valid'] for camera in cameras): raise ValueError('Expected four accepted cached cameras')
    delight=archive(source/'delight.npz'); state=archive(original/'landmarks.npz'); meshes=archive(original/'meshes.npz')
    preview=json.loads((original/'preview.json').read_text())
    assets=AssemblyAssets(ROOT/f"assets/clientTemplates/legacy{preview['gender'].title()}V1")
    a=assets.arrays;faces=meshes['faces']
    if not np.array_equal(faces,a['nicpFaces']): raise ValueError('Client face order mismatch')
    lookup={tuple(face):index for index,face in enumerate(a['faces'])}
    fullIds=np.array([lookup[tuple(face)] for face in a['nicpToFullIds'][faces]])
    if not (a['materialIds'][fullIds]==0).all(): raise ValueError('Client subset must use head material')
    geometry={'vertices':state['afterCanonical'],'faces':faces,'uvs':a['uvs'],
              'uvFaces':a['uvFaces'][fullIds],'uvConvention':'topLeftPixelEdges'}
    if len(geometry['vertices'])!=2795: raise ValueError('Expected frozen client skin subset')
    np.savez_compressed(directory/'geometry.npz',**{k:v for k,v in geometry.items() if isinstance(v,np.ndarray)})
    report={'caseId':caseId,'status':'preparing','geometryStage':'directSurface zero template prior -> frontLandmarks',
            'geometrySource':str(original/'landmarks.npz'),'geometryKey':'afterCanonical',
            'sourceHashes':{str(p):digest(p) for p in [original/'landmarks.npz',original/'meshes.npz',original/'skinTexture.png',Path(case['cameraPath']),Path(case['inputsPath']),source/'delight.npz']},
            'geometryAndCamerasFrozen':True,'legacyAtlasSourceUvsUsed':False,
            'clientUvAndFaceIndicesPreserved':True,'destinationUvConvention':'topLeftPixelEdges',
            'inputColorEncoding':'OpenDelight native gamma2.2, quantized PNG; converted to sRGB only after baking',
            'sourceViewCount':len(cameras),'vertexCount':len(geometry['vertices']),'faceCount':len(faces),
            'limitations':['Existing geometry intersections and camera/landmark errors remain.',
                'COLMAP face selection/color correction does not itself use semantic masks; coverage is filtered afterwards.',
                'COLMAP uses affine projected face corners; large triangles can have perspective interpolation error.',
                'All views also supplied reconstruction and texture; appearance error is in-sample only.',
                'Observed texels use 100% COLMAP color with no template feather; unobserved surface texels use the preset template with the same old color gain.',
                'Two-pixel UV gutter copies colors only outside the client skin UV domain, never counted as observation.',
                'Unfeathered direct color is diagnostic; external photograph/template transition may be sharp.',
                'Skin only, no eyes or other accessories; not deployed.']}
    quantized=np.rint(delight['albedoImages']*255).astype(np.uint8)
    if quantized.shape!=(4,512,512,3) or delight['validMasks'].shape!=(4,512,512): raise ValueError('Unexpected cached image dimensions')
    report['colorWorkspace']=prepareWorkspace(directory/'colorWorkspace',geometry,cameras,quantized)
    supportRgb=np.repeat((delight['validMasks'].astype(np.uint8)*255)[...,None],3,axis=-1)
    report['supportWorkspace']=prepareWorkspace(directory/'supportWorkspace',geometry,cameras,supportRgb)
    markerRgb=np.zeros_like(quantized);active=report['colorWorkspace']['activeViewIndices']
    for index,view in enumerate(active): markerRgb[view]=viewMarkerColors(len(active))[index]
    report['markerWorkspace']=prepareWorkspace(directory/'markerWorkspace',geometry,cameras,markerRgb)
    report['status']='texturing';writeJson(directory/'report.json',report)
    for workspace,result,support in [('colorWorkspace','colorOutput',False),('supportWorkspace','supportOutput',True),('markerWorkspace','markerOutput',True)]:
        report[result]=texturer(directory,None,workspace,result,support=support,executable=executable)
    outputs=[loadTexturedMesh(directory/name/'mesh.ply',geometry) for name in ('colorOutput','supportOutput','markerOutput')]
    colorMesh,supportMesh,markerMesh=outputs
    for value in outputs[1:]:
        if not np.array_equal(value['faceUvs'],colorMesh['faceUvs']) or not np.array_equal(value['validFaces'],colorMesh['validFaces']):
            raise ValueError('COLMAP auxiliary assignments/layout differ')
    atlases=[np.asarray(Image.open(directory/name/'texture.png').convert('RGB')) for name in ('colorOutput','supportOutput','markerOutput')]
    if any(v.shape!=atlases[0].shape for v in atlases[1:]): raise ValueError('COLMAP auxiliary atlas dimensions differ')
    support=rebakeAtlas(geometry,supportMesh,atlases[1],1024)
    sourceSupport=sourceSupportFromMarkers(geometry,markerMesh,atlases[2],cameras,delight['validMasks'],active,1024)
    observed=sourceSupport['coverageMask'] & support['assignedMask'] & (support['rawRebaked'].min(-1)>=.999)
    if not observed.any(): raise ValueError('No verified photograph support')
    baked=rebakeAtlas(geometry,colorMesh,atlases[0],1024,{'coverageMask':observed,'sourceIndex':sourceSupport['sourceIndex']})
    baked['faceSourceIndex']=sourceSupport['faceSourceIndex']
    np.savez_compressed(directory/'bakedTexture.npz',**{k:v for k,v in baked.items() if isinstance(v,np.ndarray)})
    # Template fallback matches the OLD gain, with no source-UV transfer or old-photo fallback.
    if caseId=='maleExactRegression':
        with zipfile.ZipFile(ROOT/'runs/legacyPreview/preview-52b719539b164238903a9025bba9765a/asset.zip') as z:
            oldDiag=json.loads(z.read('legacyManifest.json'))['assemblyDiagnostics']
    else: oldDiag=json.loads((source/'assembly/diagnostics.json').read_text())
    gain=np.asarray(oldDiag['texture']['templateLinearColorGain'])
    fallback=np.clip(assets.materials[0]['baseColor'][...,:3]*gain,0,1)
    linear=decodeColor(baked['albedo'],'gamma2.2')
    result,feathered,gutter,legacyAlpha=clientColors(baked,fallback,assets.seamFeatherPixels)
    alpha=observed.astype(np.float32)
    saveColor(directory/'colmapLegacyFeather.png',feathered)
    saveColor(directory/'colmapTexture.png',result)
    saveColor(directory/'colmapObserved.png',np.where(observed[...,None],linear,.18))
    Image.fromarray((observed*255).astype(np.uint8)).save(directory/'coverage.png')
    (directory/'oldTexture.png').write_bytes((original/'skinTexture.png').read_bytes())
    np.savez_compressed(directory/'provenance.npz',coverage=observed,photoBlendAlpha=alpha,gutter=gutter,legacyFeatherAlpha=legacyAlpha.astype(np.float32),sourceIndex=baked['sourceIndex'])
    targetBranch=next(branch for branch in preview['branches'] if branch['id']=='frontLandmarks')
    branches=[{**targetBranch,'id':name,'label':label} for name,label in [('legacyTexture','最终几何 · 旧 UV 搬运'),('colmapDirect','相同最终几何 · COLMAP 直接烘焙')]]
    outputPreview={**preview,'branches':branches,'comparison':'Identical final geometry; old UV transfer versus direct final-mesh COLMAP'}
    writeJson(directory/'preview.json',outputPreview)
    appearance={'schemaVersion':1,'caseId':caseId,'uvOrigin':'topLeftVDown','branches':{}}
    for branch,filename,label in [('legacyTexture','oldTexture.png','旧融合贴图经旧 UV 表搬运'),('colmapDirect','colmapTexture.png','最终几何直接烘焙；有效照片处不混入模板')]:
        p=directory/filename
        appearance['branches'][branch]={'mesh':{'uvs':geometry['uvs'].tolist(),'uvFaces':geometry['uvFaces'].tolist()},
            'texture':{'url':f'/data/{caseId}/{filename}','sha256':digest(p),'sizeBytes':p.stat().st_size,'colorSpace':'sRGB'},'provenanceLabel':label}
    writeJson(directory/'appearance.json',appearance)
    print('Rendering fixed-camera texture comparison:',caseId,flush=True)
    report['appearanceComparison']=diagnosticFigures(directory,geometry,directory/'oldTexture.png',directory/'colmapTexture.png',observed,delight,cameras,inputs,case,state)
    report.update(colmapGeometry=colorMesh['diagnostics'],rebake=baked['diagnostics'],support=sourceSupport['diagnostics'],
                  auxiliaryAtlasLayoutIdentical=True,templateFallbackGain=gain.tolist(),
                  blendPolicy='100% baked color on supported texels; preset fallback elsewhere; two-pixel gutter outside UV domain only',
                  gutterTexelsNotObserved=int(gutter.sum()),legacyFeatherDiagnostic='colmapLegacyFeather.png',
                  coveredTexels=int(observed.sum()),skinAtlasTexels=int(baked['atlasMask'].sum()),
                  observedFractionOfClientSkinAtlas=float(observed.sum()/baked['atlasMask'].sum()),
                  identicalDisplayedGeometry=branches[0]['vertices']==branches[1]['vertices'] and branches[0]['faces']==branches[1]['faces'],
                  colmapSeconds=sum(report[name]['seconds'] for name in ('colorOutput','supportOutput','markerOutput')),
                  nativeColmap={'path':str(executable),'sha256':digest(executable)})
    report['sourceFilesUnchanged']=all(digest(p)==h for p,h in report['sourceHashes'].items())
    if not report['sourceFilesUnchanged']: raise ValueError('Original experiment changed')
    report['wallSeconds']=perf_counter()-start;report['status']='succeeded';writeJson(directory/'report.json',report)
    print(json.dumps({k:report[k] for k in ['caseId','status','colmapSeconds','wallSeconds','coveredTexels','observedFractionOfClientSkinAtlas']},ensure_ascii=False),flush=True)
    return {'id':caseId,'gender':preview['gender'],'label':{'maleExactRegression':'歪嘴回归 · 男性模板','orbitControl4':'环绕对照 · 女性模板'}.get(caseId,caseId),
            'previewUrl':f'/data/{caseId}/preview.json','status':'succeeded','qualityApproved':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/clientColmap20261012')
    parser.add_argument('--cases',nargs='+',default=['maleExactRegression'],
                        help='Default: current male turning-head baseline; additional historical cases must be selected explicitly')
    parser.add_argument('--colmap-executable',type=Path,default=ROOT/'.runtimes/colmap/bin/colmap')
    args=parser.parse_args();executable=args.colmap_executable.resolve(strict=True)
    sources=json.loads((ROOT/'reports/frontLandmark20261012/sources.json').read_text())['cases']
    caseMap={case['caseId']:case for case in sources}
    if any(case not in caseMap for case in args.cases): raise ValueError('Unknown case')
    args.output.mkdir(parents=True,exist_ok=True)
    helpResult=subprocess.run([str(executable),'mesh_texturer','--help'],capture_output=True,text=True,check=True,timeout=30)
    (args.output/'colmapHelp.txt').write_text(helpResult.stdout+helpResult.stderr)
    manifest={'status':'running','objective':'Photographs -> final registered and landmark-corrected client surface -> original client UV',
              'solver':'Frozen geometry and cameras; COLMAP mesh_texturer followed by same-face barycentric atlas rebake','cases':[]}
    writeJson(args.output/'experiment.json',manifest)
    for case in args.cases:
        manifest['cases'].append(runCase(caseMap[case],args.output,executable));writeJson(args.output/'experiment.json',manifest)
    manifest['status']='succeeded';writeJson(args.output/'experiment.json',manifest)


if __name__=='__main__': main()
