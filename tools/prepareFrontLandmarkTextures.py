"""Add same-texture A/B appearance to frozen frontal landmark experiments.

Client PNG bytes are copied from the exact existing assembly. No geometry or
texture projection is refitted. UFF native gamma-2.2 color is converted to sRGB
for the reference viewer; uncovered source texels retain their stored values.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.camera import projectPoints
from faceAppearance.headAssembly import decodeColor


def digest(value):
    return hashlib.sha256(value).hexdigest()


def loadArchive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def encodeSrgb(linear):
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= .0031308, 12.92*linear, 1.055*linear**(1/2.4)-.055)


def renderTexture(vertices, faces, uvs, uvFaces, texture, camera, *, light=False, size=512):
    """Fixed-camera CPU perspective raster; UVs index top-to-bottom PNG rows."""
    rotation, translation = np.asarray(camera['rotation']), np.asarray(camera['translation'])
    cv = vertices @ rotation.T + translation
    pixels, _ = projectPoints(vertices, np.asarray(camera['intrinsics']), rotation, translation)
    linear = decodeColor(np.asarray(texture.convert('RGB'),np.float32)/255.,'srgb')
    normals = np.zeros_like(cv)
    triangles = cv[faces]
    cross = np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0])
    for corner in range(3): np.add.at(normals,faces[:,corner],cross)
    normals /= np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    direction=np.array([-.25,-.4,-1.]);direction/=np.linalg.norm(direction)
    shade=.35+.8*np.maximum(normals@direction,0) if light else np.ones(len(vertices))
    output=np.full((size,size,3),[28,35,39],np.uint8)
    depth=np.full((size,size),np.inf)
    for face, uvFace in zip(faces,uvFaces):
        if (cv[face,2]<=0).any(): continue
        xy=pixels[face];lo=np.maximum(np.floor(xy.min(0)).astype(int),0)
        hi=np.minimum(np.ceil(xy.max(0)).astype(int),[size-1,size-1])
        if (lo>hi).any(): continue
        a,b,c=xy;den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-10: continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        v=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
        w=1-u-v;weights=np.stack([u,v,w],axis=-1)/cv[face,2]
        invz=weights.sum(-1);z=1/np.maximum(invz,1e-12);weights*=z[...,None]
        window=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        keep=(u>=0)&(v>=0)&(w>=0)&(z<window)
        if not keep.any(): continue
        sampleUv=weights@uvs[uvFace]
        # Normalized sampler coordinates address texel centers at (i+.5)/size.
        mx=(sampleUv[...,0]*texture.width-.5).astype(np.float32)
        my=(sampleUv[...,1]*texture.height-.5).astype(np.float32)
        color=cv2.remap(linear,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        color*= (weights@shade[face])[...,None]
        rgb=np.rint(255*encodeSrgb(color)).astype(np.uint8)
        output[lo[1]:hi[1]+1,lo[0]:hi[0]+1][keep]=rgb[keep];window[keep]=z[keep]
    return Image.fromarray(output)


def prepareCase(case, output):
    caseId=case['caseId'];directory=output/caseId
    preview=json.loads((directory/'preview.json').read_text())
    report=json.loads((directory/'report.json').read_text())
    if digest((directory/'meshes.npz').read_bytes())!=report['meshSha256']:
        raise ValueError('Landmark mesh differs from its report')
    geometry=loadArchive(case['geometryPath']);meshes=loadArchive(directory/'meshes.npz')
    bundle=ROOT/f"assets/clientTemplates/legacy{preview['gender'].title()}V1"
    manifest=json.loads((bundle/'manifest.json').read_text())
    if digest((bundle/'template.npz').read_bytes())!=manifest['files']['template.npz']:
        raise ValueError('Template UV source changed')
    arrays=loadArchive(bundle/'template.npz')
    if not np.array_equal(arrays['nicpFaces'],meshes['faces']): raise ValueError('Client face order differs')
    fullLookup={tuple(face):i for i,face in enumerate(arrays['faces'])}
    fullIds=np.array([fullLookup[tuple(face)] for face in arrays['nicpToFullIds'][arrays['nicpFaces']]])
    if not np.all(arrays['materialIds'][fullIds]==0): raise ValueError('Expected only head material')
    clientUvs=arrays['uvs'];clientUvFaces=arrays['uvFaces'][fullIds]
    if not np.array_equal(geometry['faces'],meshes['sourceFaces']): raise ValueError('UFF face order differs')

    if caseId=='maleExactRegression':
        source=ROOT/'runs/legacyPreview/preview-52b719539b164238903a9025bba9765a/asset.zip'
        with zipfile.ZipFile(source) as archive:
            textureBytes=archive.read('head_color.png');delivery=json.loads(archive.read('legacyManifest.json'))
        entry=delivery['files']['head_color.png'];sourceDescription=str(source)+'!head_color.png'
        nativeEncoding=delivery['assemblyDiagnostics']['texture']['sourceColorEncoding']
    else:
        assembly=Path(case['geometryPath']).parent/'assembly'
        delivery=json.loads((assembly/'manifest.json').read_text())
        source=assembly/'textures/baseColor_000.png';textureBytes=source.read_bytes()
        entry=delivery['files']['textures/baseColor_000.png'];sourceDescription=str(source)
        materials=json.loads((assembly/'materials.json').read_text())
        if materials['materials'][0]['baseColorTexture']['colorSpace']!='sRGB': raise ValueError('Expected sRGB assembly')
        nativeEncoding=json.loads((Path(case['geometryPath']).parent/'config.json').read_text())['assemblyColorEncoding']
    if entry['sha256']!=digest(textureBytes) or entry['bytes']!=len(textureBytes):
        raise ValueError('Delivered texture content changed')
    if delivery['templateId']!=manifest['templateId']: raise ValueError('Texture belongs to a different client template')
    skin=Image.open(io.BytesIO(textureBytes));skin.load()
    if skin.size!=(1024,1024): raise ValueError('Unexpected head texture resolution')
    (directory/'skinTexture.png').write_bytes(textureBytes)
    sourcePath=Path(case['geometryPath']).parent/'albedo.png'
    sourceNative=np.asarray(Image.open(sourcePath).convert('RGB'),np.float32)/255.
    sourceColor=Image.fromarray(np.rint(encodeSrgb(decodeColor(sourceNative,nativeEncoding))*255).astype(np.uint8))
    sourceColor.save(directory/'sourceTexture.png')

    branches={}
    for branch in preview['branches']:
        native=branch['id']=='source';filename='sourceTexture.png' if native else 'skinTexture.png'
        data=(directory/filename).read_bytes()
        branches[branch['id']]={
            'mesh':{'uvs':(geometry['uvs'] if native else clientUvs).tolist(),
                    'uvFaces':(geometry['faces'] if native else clientUvFaces).tolist()},
            'texture':{'url':f'/data/{caseId}/{filename}','sha256':digest(data),'sizeBytes':len(data),'colorSpace':'sRGB'},
            'provenanceLabel':'原始 UFF 融合贴图 · 色彩转换后显示' if native else '各客户端分支共用原融合贴图，未按新几何重投影'}
    appearance={'schemaVersion':1,'caseId':caseId,'uvOrigin':'topLeftVDown','branches':branches,
        'comparison':'Same client PNG bytes and fixed UVs before/after; no new texture reconstruction',
        'geometryPreviewSha256':digest((directory/'preview.json').read_bytes()),
        'limits':['Geometry correction can move existing texture features. New image-to-surface correspondence is not computed.',
                  'Skin only; eyeballs, teeth, brows, hair cards and other accessories are not added.',
                  'Unobserved source texels retain the original albedo PNG values; client fallback is the existing template.']}
    (directory/'appearance.json').write_text(json.dumps(appearance,ensure_ascii=False,separators=(',',':')))

    state=loadArchive(directory/'landmarks.npz')
    camera=json.loads(Path(case['cameraPath']).read_text())[0]
    inputs=json.loads(Path(case['inputsPath']).read_text())
    photo=ImageOps.exif_transpose(Image.open(case['sourceImages'][0]['path'])).convert('RGB')
    work=Image.fromarray(cv2.warpPerspective(np.asarray(photo),np.asarray(inputs[0]['inputToWorking']),
                                           tuple(inputs[0]['workingSize'])))
    renderInputs=[(geometry['vertices'],geometry['faces'],geometry['uvs'],geometry['faces'],sourceColor),
                  (state['beforeCanonical'],meshes['faces'],clientUvs,clientUvFaces,skin),
                  (state['afterCanonical'],meshes['faces'],clientUvs,clientUvFaces,skin)]
    for lit,filename in [(False,'texturedComparison.png'),(True,'texturedLitComparison.png')]:
        panels=[work]+[renderTexture(*args,camera,light=lit) for args in renderInputs]
        result=Image.new('RGB',(2048,566),(28,35,39));draw=ImageDraw.Draw(result)
        draw.text((12,8),caseId+' | fixed photo camera | '+('shared preview light' if lit else 'base color / no preview light'),fill='white')
        labels=['Original photo','UFF source','Before / same client texture','After / same client texture']
        for i,(panel,label) in enumerate(zip(panels,labels)):
            result.paste(panel,(i*512,54));draw.text((i*512+12,32),label,fill='white')
        result.save(directory/filename)
    return {'caseId':caseId,'clientTextureSource':sourceDescription,'clientTextureSha256':digest(textureBytes),
            'sourceTextureSource':str(sourcePath),'sourceTextureInputSha256':digest(sourcePath.read_bytes()),
            'sourceColorConversion':nativeEncoding+' -> linear -> sRGB','branchCount':len(branches),
            'clientTextureSameAcrossBranches':len({v['texture']['sha256'] for k,v in branches.items() if k!='source'})==1,
            'clientUvFaceCount':len(clientUvFaces),'clientAllFacesHeadMaterial':True,
            'originalGeometryHashUnchanged':digest((directory/'meshes.npz').read_bytes())==report['meshSha256'],
            'textureReprojected':False,'clientAccessoriesAdded':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,default=ROOT/'reports/frontLandmark20261012/sources.json')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/frontLandmark20261012')
    args=parser.parse_args();records=[]
    for case in json.loads(args.sources.read_text())['cases']:
        record=prepareCase(case,args.output);records.append(record)
        print(json.dumps({k:record[k] for k in ('caseId','branchCount','clientTextureSameAcrossBranches','originalGeometryHashUnchanged')}),flush=True)
    (args.output/'texturePreviewReport.json').write_text(json.dumps({'status':'complete','remoteChanged':False,
        'textureMode':'existing same-texture A/B, no reprojection','cases':records},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
