"""Render frozen client meshes for same-detector semantic correspondence evidence.

CPU pinhole rasterization preserves saved cameras. Original and working image
grids are independently rasterized; no photograph enters the rendered RGB.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.artifacts import writeJson


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key:data[key] for key in data.files}


def srgb_to_linear(value):
    return np.where(value <= .04045, value/12.92, ((value+.055)/1.055)**2.4)


def linear_to_srgb(value):
    return np.where(value <= .0031308, value*12.92, 1.055*np.maximum(value,0)**(1/2.4)-.055)


def sample(image, uv):
    """Original v-down UVs use pixel edges and bilinear border clamp."""
    coordinates = np.stack((uv[:,1]*image.shape[0]-.5, uv[:,0]*image.shape[1]-.5))
    return np.stack([map_coordinates(image[...,c], coordinates, order=1,
        mode='nearest', prefilter=False) for c in range(image.shape[2])],axis=-1)


def scene(case, vertices):
    assets = AssemblyAssets(case['templatePath'])
    a = assets.arrays
    transform = case['templateTransform']
    rotation, translation, scale = np.asarray(transform['rotation']),np.asarray(transform['translation']),transform['scale']
    client_cm = ((vertices-translation) @ rotation.T)/scale
    assembled, components = assets._assembleSkin(client_cm, eyePlacementMode='assetDefault')
    assembled['vertices'] = (assembled['vertices'] @ rotation)*scale+translation
    # Preserve the input geometry exactly; inverse/forward similarity is only
    # needed to initialize original accessory placement.
    assembled['vertices'][a['nicpToFullIds']] = vertices
    full_lookup = {tuple(face):i for i,face in enumerate(a['faces'])}
    skin_ids = np.array([full_lookup[tuple(face)] for face in a['nicpToFullIds'][a['nicpFaces']]],np.int64)
    full_to_skin = np.full(len(a['faces']),-1,np.int32)
    full_to_skin[skin_ids] = np.arange(len(skin_ids))
    materials, hair_policy, hidden_brows, hidden_lashes = assets._materialsFromHead(
        assets.materials[0]['baseColor'].copy(), [])
    maps = [material['baseColor'] for material in materials]
    for material, image in zip(materials,maps):
        if material['alphaMode']=='OPAQUE': image[...,3]=1
    result = {**assembled,'skinFaceIds':skin_ids,'fullToSkinFaceIndex':full_to_skin,
        'nicpToFullIds':a['nicpToFullIds'],'materials':materials,'maps':maps,
        'components':components,'provenance':{'templatePath':case['templatePath'],
            'templateManifestSha256':sha(case['templateManifestPath']),
            'templateArraySha256':sha(case['templateArrayPath']),
            'templateMaterialFiles':{m['texture']:sha(Path(case['templatePath'])/m['texture']) for m in assets.materials},
            'accessoryPlacement':'AssemblyAssets._assembleSkin(assetDefault); original components placed once against frozen skin',
            'templateTransform':transform,'hairPolicy':hair_policy,
            'hiddenBrowMaterials':hidden_brows,'hiddenEyelashMaterials':hidden_lashes}}
    if case['caseId']=='maleCurrent':
        path = ROOT/'runs/clientChinSmooth20261012/comparison/before/legacyAsset.zip'
        source = _load(path)
        material = source.materials['materials'][0]
        rgba = np.asarray(Image.open(io.BytesIO(source.payloads[material['baseColorTexture']['path']])).convert('RGBA'),np.float32)/255
        rgba[...,:3]=srgb_to_linear(rgba[...,:3])
        result['photoHead']=rgba
        result['provenance']['photoDerived']={'archivePath':str(path),'archiveSha256':sha(path),
            'materialPath':material['baseColorTexture']['path'],
            'meaning':'Existing SH-derived head RGB only, original UVs; no new baking. Original template accessory maps retained for controlled texture comparison.'}
    return result


def raster(scene, camera, shape):
    height,width=shape
    vertices,faces=scene['vertices'],scene['faces']
    camera_vertices=vertices@np.asarray(camera['rotation']).T+camera['translation']
    projected=camera_vertices@np.asarray(camera['intrinsics']).T
    pixels=projected[:,:2]/projected[:,2:]
    triangles=camera_vertices[faces]
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    front=(cross*(-triangles.mean(1))).sum(1)>0
    normals=np.zeros_like(camera_vertices)
    for i in range(3): np.add.at(normals,faces[:,i],cross)
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)
    depth=np.full((height,width),np.inf,np.float32)
    face_index=np.full((height,width),-1,np.int32)
    barycentric=np.zeros((height,width,3),np.float32)
    for face_id in np.flatnonzero(front):
        face=faces[face_id]
        z=camera_vertices[face,2]
        if (z<=0).any(): continue
        xy=pixels[face]
        lo=np.maximum(np.floor(xy.min(0)).astype(int),0)
        hi=np.minimum(np.ceil(xy.max(0)).astype(int),[width-1,height-1])
        if (lo>hi).any(): continue
        a,b,c=xy
        denominator=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(denominator)<1e-12: continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/denominator
        v=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/denominator
        w=1-u-v
        invz=u/z[0]+v/z[1]+w/z[2]
        distance=1/np.maximum(invz,1e-20)
        window=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        keep=(u>=-1e-10)&(v>=-1e-10)&(w>=-1e-10)&(distance<window)
        if not keep.any(): continue
        row,col=np.where(keep)
        weights=np.stack((u[keep]/z[0],v[keep]/z[1],w[keep]/z[2]),axis=-1)*distance[keep,None]
        material_id=scene['materialIds'][face_id]
        material=scene['materials'][material_id]
        if material['alphaMode']=='MASK':
            uv=weights@scene['uvs'][scene['uvFaces'][face_id]]
            valid=sample(scene['maps'][material_id],uv)[:,3]>=material.get('alphaCutoff',.5)
            row,col,weights=row[valid],col[valid],weights[valid]
        row,col=row+lo[1],col+lo[0]
        depth[row,col]=(weights*z).sum(-1)
        face_index[row,col]=face_id
        barycentric[row,col]=weights
    visible=face_index>=0
    ids=face_index[visible]
    bary=barycentric[visible]
    material_ids=scene['materialIds'][ids]
    uv=np.einsum('ni,nij->nj',bary,scene['uvs'][scene['uvFaces'][ids]])
    normals=np.einsum('ni,nij->nj',bary,normals[faces[ids]])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)
    colors=np.zeros((len(ids),3),np.float32)
    for i,image in enumerate(scene['maps']):
        mask=material_ids==i
        if mask.any(): colors[mask]=sample(image,uv[mask])[:,:3]
    skin=scene['fullToSkinFaceIndex'][ids]>=0
    gray=colors.copy(); gray[skin]=.42
    variants={}
    for name,base,direction in [('gray',gray,[-.5,-.5,-1]),('grayAlternate',gray,[.5,-.5,-1]),
                               ('template',colors,[-.5,-.5,-1])]:
        light=np.asarray(direction,float); light/=np.linalg.norm(light)
        shade=.35+.65*np.maximum(normals@light,0)
        values=np.clip(linear_to_srgb(base*shade[:,None]),0,1)
        rgb=np.full((height,width,3),[27,33,40],np.uint8)
        rgb[visible]=np.rint(values*255).astype(np.uint8)
        variants[name]=Image.fromarray(rgb)
    if 'photoHead' in scene:
        base=colors.copy(); mask=material_ids==0
        base[mask]=sample(scene['photoHead'],uv[mask])[:,:3]
        light=np.array([-.5,-.5,-1]); light/=np.linalg.norm(light)
        shade=.35+.65*np.maximum(normals@light,0)
        rgb=np.full((height,width,3),[27,33,40],np.uint8)
        rgb[visible]=np.rint(np.clip(linear_to_srgb(base*shade[:,None]),0,1)*255).astype(np.uint8)
        variants['photoDerived']=Image.fromarray(rgb)
    material_map=np.full((height,width),-1,np.int16); material_map[visible]=material_ids
    skin_map=np.zeros((height,width),bool); skin_map[visible]=skin
    return variants,{'faceIndex':face_index,'barycentric':barycentric,'depth':depth,
        'materialId':material_map,'skinMask':skin_map}, {'visiblePixels':int(visible.sum()),
        'skinPixels':int(skin.sum()),'frontFacingFaceCount':int(front.sum())}


def write_manifest(path,manifest):
    temporary=path.with_suffix('.tmp')
    writeJson(temporary,manifest)
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'runs/clientChinAdaptive20261012/cases.json')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/clientRenderedSemantics20261012/renderInputs')
    parser.add_argument('--case',action='append')
    parser.add_argument('--male-after',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    source=json.loads(args.manifest.read_text())
    manifest={'schemaVersion':1,'status':'running','sourceManifestPath':str(args.manifest),
        'sourceManifestSha256':sha(args.manifest),'rendererPath':str(Path(__file__).resolve()),
        'rendererSha256':sha(__file__),'cases':[],
        'variants':{'gray':'Neutral gray .42 linear RGB on skin, original texture on eye/brow/oral accessories; light [-.5,-.5,-1]',
            'grayAlternate':'Same as gray, opposing horizontal light [.5,-.5,-1]',
            'template':'Original template sRGB maps decoded linear, same main light',
            'photoDerived':'Male only: replace head RGB with previous SH-derived texture, same geometry/accessories/light'},
        'rendering':{'backgroundSrgb':[27,33,40],'shading':'Smooth area-weighted normals, Lambert .35 ambient + .65 diffuse; camera-relative light',
            'raster':'CPU triangle raster; front-face culling; integer pixel centers; perspective-correct interpolation; per-fragment alpha cutoff',
            'uv':'Original v-down UV; bilinear pixel-edge sampler; border clamp; no atlas resizing',
            'originalResolution':'Native oriented original width/height, independent raster, inverse(inputToWorking) @ saved working intrinsics',
            'surfaceFiles':'full assembly face IDs, perspective-correct triangle barycentrics, camera Z depth, material IDs, skin mask; background face=-1 depth=inf',
            'noPhotoCompositing':True},'remoteModified':False}
    start=perf_counter()
    for case in source['cases']:
        if args.case and case['caseId'] not in args.case: continue
        for key,hash_key in [('geometryPath','geometrySha256'),('facesPath','facesSha256'),('cameraPath','cameraSha256'),('inputsPath','inputsSha256')]:
            if sha(case[key])!=case[hash_key]: raise ValueError(f'Changed source {case["caseId"]} {key}')
        vertices=archive(case['geometryPath'])[case.get('verticesKey','vertices')]
        cameras=json.loads(Path(case['cameraPath']).read_text())
        inputs=json.loads(Path(case['inputsPath']).read_text())
        stages=[('before',vertices)]
        if args.male_after and case['caseId']=='maleCurrent':
            after_path=ROOT/'runs/clientChinAdaptive20261012/evaluationV1/maleCurrent/geometry.npz'
            stages.append(('after',archive(after_path)['after']))
        case_record={'caseId':case['caseId'],'caseManifest':case,'stages':[]}
        manifest['cases'].append(case_record)
        for stage,skin_vertices in stages:
            directory=args.output/case['caseId']/stage; directory.mkdir(parents=True,exist_ok=True)
            current=scene(case,skin_vertices)
            geometry_path=directory/'assemblyGeometry.npz'
            np.savez_compressed(geometry_path,**{key:current[key] for key in
                ('vertices','faces','uvs','uvFaces','materialIds','skinFaceIds','fullToSkinFaceIndex','nicpToFullIds')})
            stage_record={'stage':stage,'assemblyGeometryPath':str(geometry_path),
                'assemblyGeometrySha256':sha(geometry_path),'sourceProvenance':current['provenance'],'views':[]}
            if stage=='after': stage_record['afterGeometryPath']=str(after_path); stage_record['afterGeometrySha256']=sha(after_path)
            case_record['stages'].append(stage_record)
            for view_index,(camera,record,image_record) in enumerate(zip(cameras,inputs,case['sourceImages'])):
                photo_path=Path(case['observationPath']).parent/f'view{view_index}-photo.png'
                original_size=image_record['orientedSize']; working_size=record['workingSize']
                original_camera={**camera,'intrinsics':(np.linalg.inv(np.asarray(record['inputToWorking']))@np.asarray(camera['intrinsics'])).tolist()}
                view={'viewIndex':view_index,'photoPath':str(photo_path),'originalPhotoPath':image_record['path'],
                    'originalPhotoSha256':image_record['sha256'],'width':working_size[0],'height':working_size[1],
                    'originalWidth':original_size[0],'originalHeight':original_size[1],
                    'inputToWorking':record['inputToWorking'],'workingToInput':record['workingToInput'],
                    'cameraWorking':camera,'cameraOriginal':original_camera,'renders':{},'rendersOriginal':{}}
                for label,selected_camera,size,key in [('working',camera,working_size,'renders'),('original',original_camera,original_size,'rendersOriginal')]:
                    images,surface,stats=raster(current,selected_camera,(size[1],size[0]))
                    for variant,image in images.items():
                        path=directory/f'view{view_index}-{label}-{variant}.png'; image.save(path); view[key][variant]=str(path)
                    surface_path=directory/f'view{view_index}-{label}-surface.npz'
                    np.savez_compressed(surface_path,**surface)
                    view['surfacePath' if label=='working' else 'surfaceOriginalPath']=str(surface_path)
                    view[label+'RasterStats']=stats
                stage_record['views'].append(view)
                manifest['elapsedSeconds']=perf_counter()-start
                write_manifest(args.output/'manifest.json',manifest)
                print(json.dumps({'caseId':case['caseId'],'stage':stage,'viewIndex':view_index,'seconds':round(perf_counter()-start,2)}),flush=True)
    manifest['status']='complete'; manifest['elapsedSeconds']=perf_counter()-start
    write_manifest(args.output/'manifest.json',manifest)


if __name__=='__main__': main()
