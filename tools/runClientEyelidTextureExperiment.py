"""Local eyelid texture comparison on a frozen, delivered dense-eye mesh.

This changes two identical head-color payloads only. Source image warping is
explicitly estimated appearance, not an extra observation or geometry change.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter
import zipfile

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.colmapTexture import _destinationUvLookup
from faceAppearance.headAssembly import AssemblyAssets, decodeColor
from faceAppearance.legacyPreviewAsset import _load
from prepareFrontLandmarkTextures import encodeSrgb


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def archiveTexture(source, output, pixels, description):
    """Preserve every geometry/material/accessory payload byte; refresh hashes."""
    asset = _load(source)
    output.mkdir(parents=True, exist_ok=False)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format='PNG')
    png = buffer.getvalue()
    payloads = dict(asset.payloads)
    payloads['head_color.png'] = payloads['face_nicp.png'] = png
    manifest = asset.manifest
    manifest['files'] = {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
                         for name, data in payloads.items()}
    manifest['localEyelidTextureExperiment'] = description
    writeJson(output/'legacyManifest.json', manifest)
    for name, data in payloads.items():
        (output/name).write_bytes(data)
    with zipfile.ZipFile(output/'legacyAsset.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in [*payloads, 'legacyManifest.json']:
            z.write(output/name, name)
    checked = _load(output/'legacyAsset.zip')
    unchanged = {key: np.array_equal(asset.mesh[key], checked.mesh[key]) for key in asset.mesh}
    bytesUnchanged = {name: payloads[name] == asset.payloads[name]
                      for name in payloads if name not in ('head_color.png', 'face_nicp.png')}
    if not all(unchanged.values()) or not all(bytesUnchanged.values()):
        raise ValueError('Texture branch changed frozen scene payloads')
    info = {'sha256': sha(output/'legacyAsset.zip'), 'bytes': (output/'legacyAsset.zip').stat().st_size,
            'meshArraysExactlyEqual': unchanged, 'otherPayloadsByteIdentical': bytesUnchanged}
    writeJson(output/'archive.json', info)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, default=ROOT/'runs/denseClientEyes20261012/comparisonFinal')
    parser.add_argument('--source', type=Path, default=ROOT/'runs/liveJaw20261012/source/reconstruction')
    parser.add_argument('--delight-source', type=Path, default=ROOT/'runs/clientAppearanceFix20261012/deployedSource')
    parser.add_argument('--observations', type=Path, default=ROOT/'runs/denseClientEyes20261012/observations')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/eyelidTexture20261012/comparison')
    parser.add_argument('--render-only', action='store_true')
    parser.add_argument('--filter-footprint', action='store_true')
    args = parser.parse_args()
    if args.render_only:
        renderEyes(args)
        return
    from faceAppearance.eyelidTextureSampling import reprojectEyelidTexture
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Choose an empty local experiment directory')
    args.output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    inputs = json.loads((args.source/'inputs.json').read_text())
    oldInputs = json.loads((args.delight_source/'inputs.json').read_text())
    if len(inputs) != len(oldInputs) or any(a['sha256'] != b['sha256']
            or a['inputToWorking'] != b['inputToWorking'] or a['workingSize'] != b['workingSize']
            for a,b in zip(inputs,oldInputs)):
        raise ValueError('Cached delight must have exactly the same photos and crops')
    cache = json.loads((args.observations/'manifest.json').read_text())
    if cache['inputsSha256'] != sha(args.source/'inputs.json') or cache['camerasSha256'] != sha(args.source/'cameras.json'):
        raise ValueError('Dense observations belong to another reconstruction')
    sourceArchive = args.comparison/'after/legacyAsset.zip'
    geometryReport = json.loads((args.comparison/'report.json').read_text())
    denseApplied = geometryReport.get('status') == 'acceptedLocalExperiment'
    geometryLabel = '密集眼形' if denseApplied else '保留原几何'
    asset = _load(sourceArchive)
    similarity = asset.manifest['assemblyDiagnostics']['templateToUff']
    full = asset.mesh['vertices']*100 @ np.asarray(similarity['rotation'])*similarity['scale']+similarity['translation']
    assets = AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    skinIds = assets.arrays['nicpToFullIds']
    with np.load(args.comparison/'geometry.npz', allow_pickle=False) as z:
        if not np.allclose(full[skinIds], z['after'], rtol=0, atol=1e-12):
            raise ValueError('Input ZIP differs from final accepted dense geometry')
    with np.load(args.source/'directClient/geometry.npz', allow_pickle=False) as z:
        geometry = {k: z[k].copy() for k in z.files}
    geometry['vertices'] = full[skinIds]
    geometry['uvConvention'] = 'topLeftPixelEdges'
    with np.load(args.observations/'view00.npz', allow_pickle=False) as z:
        if not z['candidateEyeValidMask'].all():
            raise ValueError('Dense photo targets are not reliable')
        curves = z['eyelidCurvesWorking'].copy()
    binding = json.loads((args.comparison/'afterBinding.json').read_text())
    photoEyes = {name: {'lower': curves[i,0], 'upper': curves[i,1]}
                 for i,name in enumerate(('imageLeft','imageRight'))}
    modelEyes = {name: {part: np.asarray(binding['eyes'][name][part]['sourcePixels'])
                        for part in ('lower','upper')} for name in photoEyes}
    with np.load(args.delight_source/'delight.npz', allow_pickle=False) as z:
        image = decodeColor(z['albedoImages'][0], 'gamma2.2')
        valid = z['validMasks'][0].copy()
    original = np.asarray(Image.open(io.BytesIO(asset.payloads['head_color.png']))).copy()
    linear = decodeColor(original[...,:3]/255., 'srgb')
    size = original.shape[0]
    lookup, _ = _destinationUvLookup(geometry, size)
    skinMember = np.zeros(len(full), bool); skinMember[skinIds] = True
    accessoryFaces = (asset.mesh['materialIds'] == 0) & ~skinMember[asset.mesh['faces']].all(1)
    accessoryUv = np.zeros((size,size), bool)
    if accessoryFaces.any():
        excluded = {'vertices': full, 'faces': asset.mesh['faces'][accessoryFaces],
            'uvs': asset.mesh['uvs'], 'uvFaces': asset.mesh['uvFaces'][accessoryFaces],
            'uvConvention': 'topLeftPixelEdges'}
        accessoryUv = _destinationUvLookup(excluded,size)[0]['validMask']
    domain = binary_dilation(lookup['validMask'],iterations=1) if args.filter_footprint else lookup['validMask']
    writable = domain & ~binary_dilation(accessoryUv,iterations=2)
    camera = json.loads((args.source/'cameras.json').read_text())[0]
    # Masked brow/lash cards are excluded from conservative occlusion queries;
    # original alpha still applies during the final complete-scene rendering.
    visible = {'vertices': full, 'faces': asset.mesh['faces'][asset.mesh['materialIds'] != 3]}
    result = reprojectEyelidTexture(geometry,camera,image,valid,linear,photoEyes,modelEyes,
        writableMask=writable, visibleScene=visible, filterFootprint=args.filter_footprint)
    alpha = result['alpha']
    proposed = original.copy()
    filterPadding = result.get('filterPadding', np.zeros(alpha.shape, bool))
    modified = (alpha > 0) | filterPadding
    proposed[modified,:3] = np.rint(encodeSrgb(result['linearRGB'][modified])*255).astype(np.uint8)
    if not np.array_equal(proposed[~modified], original[~modified]):
        raise ValueError('Unrelated head texture pixels changed')
    if np.any(modified & ~writable):
        raise ValueError('Texture edits leaked to protected UVs')
    beforeDir = args.output/'before'; beforeDir.mkdir()
    shutil.copyfile(sourceArchive,beforeDir/'legacyAsset.zip')
    description = {'deployed': False, 'sourceArchiveSha256':sha(sourceArchive),
        'inputGeometryStatus':geometryReport.get('status'), 'denseGeometryApplied':denseApplied,
        'sourceDelightSha256':sha(args.delight_source/'delight.npz'),
        'sourceDelightScope':'Earlier cached inference of byte-identical photos and exact same working crops; not the latest remote inference cache',
        'method':'Dense contour guided frontal skin resampling; estimated warp/extrapolation marked separately',
        'geometryChanged':False, 'fullColmapRebake':False, 'filterFootprint':args.filter_footprint}
    identity = archiveTexture(sourceArchive,args.output/'after',proposed,description)
    arrays = {key:value for key,value in result.items() if isinstance(value,np.ndarray)}
    np.savez_compressed(args.output/'textureProvenance.npz', **arrays, writableMask=writable)
    Image.fromarray(original).save(args.output/'beforeTexture.png')
    Image.fromarray(proposed).save(args.output/'afterTexture.png')
    Image.fromarray(np.rint(alpha*255).astype(np.uint8)).save(args.output/'editAlpha.png')
    Image.fromarray(filterPadding).save(args.output/'filterPadding.png')
    Image.fromarray(np.rint(encodeSrgb(image)*255).astype(np.uint8)).save(args.output/'frontDelight.png')
    report = {**description,'status':'localTextureCandidate','seconds':perf_counter()-start,
        'diagnostics':result['diagnostics'],'archive':identity,
        'changedRgbTexels':int(np.any(proposed[...,:3]!=original[...,:3],axis=-1).sum()),
        'outsideEditPixelsByteIdentical':True,'protectedAccessoryUvsUnchanged':True,
        'sourceColorGain':[1.,1.,1.],
        'inputsSha256':sha(args.source/'inputs.json'),'camerasSha256':sha(args.source/'cameras.json'),
        'observationsSha256':sha(args.observations/'view00.npz'),
        'implementationSha256':{str(Path(name)):sha(ROOT/name) for name in
            ('tools/runClientEyelidTextureExperiment.py','src/faceAppearance/eyelidTextureSampling.py')},
        'limitations':['Single male sample, frontal appearance experiment.',
            'Dense curves are automatic observations, not manually annotated ground truth.',
            'Geometry, eye/corner/lash accessories, gaze and other maps remain unchanged.',
            'No all-view COLMAP rebake; warped colors are estimated appearance.',
            'MASK cards are excluded from texture visibility queries and retain original alpha during display.']}
    writeJson(args.output/'report.json',report)
    for name in report['implementationSha256']:
        destination=args.output/'implementation'/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,destination)
    writeJson(args.output/'config.json',{'title':'眼睑局部贴图 · 同几何对照','brand':'眼周贴图','badge':'本地实验 · 未部署',
        'subtitle':'两侧几何完全相同（'+geometryLabel+'）。右侧按正面眼睑轮廓重新取色，保留其他区域与附件。',
        'branches':[{'id':'before','label':geometryLabel+'＋旧贴图'},{'id':'after','label':'眼周局部重新取色'}],
        'defaultLeft':'before','defaultRight':'after','defaultViewerMode':'baseColor','defaultRenderMode':'baseColor',
        'notes':['冻结几何、相机、UV、眼球及附件；只修改头部贴图眼周局部。',
            '双线性取色支撑排除照片眼白和虹膜；轮廓重定位及安全肤色外扩属于估计外观。',
            '单个男生样本本地实验，未部署；内眼角几何棱角仍保留。']})
    print(json.dumps({'output':str(args.output),'seconds':report['seconds'],'changedRgbTexels':report['changedRgbTexels']}),flush=True)


def renderEyes(args):
    from renderClientAppearanceEvidence import EvidenceRenderer,evidenceGrid,solid
    import torch
    torch.set_num_threads(1)
    out = args.output/'eyes'; out.mkdir(exist_ok=False)
    source = _load(args.output/'before/legacyAsset.zip')
    similarity = source.manifest['assemblyDiagnostics']['templateToUff']
    camera = json.loads((args.source/'cameras.json').read_text())[0]
    with np.load(args.observations/'view00.npz',allow_pickle=False) as z:
        curves = z['eyelidCurvesWorking']
    boxes=[]
    for points in curves:
        points=points.reshape(-1,2)
        boxes.append(np.r_[np.floor(points.min(0)-[15,17]),np.ceil(points.max(0)+[15,17])].astype(int))
    roi=np.r_[np.min(np.asarray(boxes)[:,:2],0),np.max(np.asarray(boxes)[:,2:],0)]
    factor=4
    pixelMap=np.array([[factor,0,factor*(.5-roi[0])-.5],[0,factor,factor*(.5-roi[1])-.5],[0,0,1]])
    zoom={key:np.asarray(camera[key],float) for key in ('intrinsics','rotation','translation')}
    zoom['intrinsics']=pixelMap@zoom['intrinsics']
    width,height=(roi[2:]-roi[:2])*factor
    photo=Image.open(args.observations/'view00WorkingPhoto.png').convert('RGB').crop(tuple(roi)).resize((width,height),Image.Resampling.LANCZOS)
    images={}; records=[]
    for branch in ('before','after'):
        renderer=EvidenceRenderer(args.output/branch/'legacyAsset.zip',similarity,device='cpu')
        images[branch],record=renderer.variants(zoom,(int(height),int(width)))
        for mode,value in images[branch].items(): value.save(out/f'{branch}-{mode}.png')
        records.append({'branch':branch,**record})
    panels=[];labels=[]
    for i,box in enumerate(boxes):
        crop=np.r_[(box[:2]-roi[:2])*factor,(box[2:]-roi[:2])*factor]
        for mode,title in [('baseColor','贴图基色'),('lit','共同光照')]:
            panels += [photo.crop(tuple(crop)),solid(images['before'][mode]).crop(tuple(crop)),solid(images['after'][mode]).crop(tuple(crop))]
            labels += [f'图{"左" if i==0 else "右"}眼 · '+name for name in ('原照片','旧贴图 · '+title,'局部重新取色 · '+title)]
    state=json.loads((args.output/'report.json').read_text())
    geometryLabel='密集眼形' if state.get('denseGeometryApplied',True) else '保留原几何'
    evidenceGrid(panels,labels,columns=3,panelSize=380,subtitle='同一'+geometryLabel+' · 眼周贴图局部对照 · 未部署').save(out/'comparison.png')
    writeJson(out/'report.json',{'sourceArchives':{k:sha(args.output/k/'legacyAsset.zip') for k in images},
        'records':records,'geometryIdentical':True,'roiRasterScale':4})
    print(str(out/'comparison.png'),flush=True)


if __name__ == '__main__':
    main()
