"""Preserve SH facial appearance while adapting the client template locally."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, label

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.clientTextureBaking import bakeClientTexture
from faceAppearance.clientTextureBoundary import (blendClientTextureExterior, fillClientTextureSkinHoles,
    exteriorEdgeDiagnostics, _exteriorSilhouette)
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.legacyAsset import exportLegacyAsset
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.skinExtrapolation import insideDistance, smoothstep
from runClientShComparison import load, sha, display


def odEdgeLowFrequency(sh, od, provenance, *, sigma=12., strength=.7):
    """Transfer only color-aligned OD low frequencies in the existing SH rim.

    Original observed core/features remain bit-identical. Both smoothed inputs
    share donor support. Different connected UV domains never share samples.
    """
    observed = provenance['coverageMask']
    protected = (provenance['boundaryProtectedFeatures'] | provenance['filterPadding']
                 | provenance['gutter'] | (provenance['frontAlpha'] > 0))
    donors = (provenance['boundarySafeSkinDonor'] & provenance['openDelightBoundarySafeSkinDonor']
              & observed & ~protected & (provenance['boundaryDirectPhotoAlpha'] >= .999))
    logRatio = np.log(np.maximum(sh, .005)) - np.log(np.maximum(od, .005))
    gain = np.exp(np.median(logRatio[donors], axis=0))
    gain = np.clip(gain, .65, 1.8)
    _, silhouette = _exteriorSilhouette(observed, provenance['atlasMask'])
    depth = insideDistance(silhouette)
    edgeWeight = strength * (1-smoothstep(depth/24.)) * smoothstep(distance_transform_edt(~protected)/8.)
    edgeWeight *= observed & provenance['boundaryModified'] & ~protected
    eligible = observed & ~protected
    components, count = label(provenance['atlasMask'])
    result, modified = sh.copy(), np.zeros(observed.shape, bool)
    deltaField = np.zeros_like(sh)
    for component in range(1,count+1):
        chart = components == component
        mask = eligible & chart
        if mask.sum() < 32:
            continue
        weights = gaussian_filter(mask.astype(np.float32), sigma)
        # Low-frequency ratios retain SH's multiplicative texture detail.
        lowSh = gaussian_filter(sh*mask[...,None], (sigma,sigma,0)) / np.maximum(weights[...,None],1e-12)
        lowOd = gaussian_filter(od*gain*mask[...,None], (sigma,sigma,0)) / np.maximum(weights[...,None],1e-12)
        delta = np.clip(np.log(np.maximum(lowOd,.005)/np.maximum(lowSh,.005)), np.log(.8),np.log(1.25))
        active = chart & (edgeWeight > 0) & (weights > .01)
        proposed = sh * np.exp(edgeWeight[...,None]*delta)
        safe = (proposed >= 0).all(-1) & (proposed <= 1).all(-1) & np.isfinite(proposed).all(-1)
        active &= safe
        result[active] = proposed[active]
        deltaField[active] = delta[active]
        modified |= active
    weight = np.where(modified,edgeWeight,0).astype(np.float32)
    return {'linearRGB':result, 'maps':{'odEdgeLowFrequencyWeight':weight},
        'debug':{'odEdgeLogRgbDelta':deltaField},
        'diagnostics':{'method':'SH multiplied by color-aligned OD low-frequency ratio in existing exterior rim',
            'sigmaPixels':sigma,'maximumWeight':strength,'actualMaximumWeight':float(weight.max()),
            'alignedOdToShGain':gain.tolist(),'gainDonorTexels':int(donors.sum()),
            'modifiedObservedTexels':int(modified.sum()),'rawRatioRange':[.8,1.25],
            'coreExactlyUnchanged':bool(np.array_equal(result[~provenance['boundaryModified']],sh[~provenance['boundaryModified']])),
            'featuresExactlyUnchanged':bool(np.array_equal(result[protected],sh[protected]))}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'runs/clientShComparison20261012')
    parser.add_argument('--source',type=Path,default=ROOT/'runs/clientAppearanceFix20261012/deployedSource')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/clientShSeam20261012')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('New experiment requires an empty output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    started=perf_counter()
    from faceAppearance.clientShTemplateBlend import matchTemplateToSh
    geometry=load(args.source/'directClient/geometry.npz')
    geometry['uvConvention']='topLeftPixelEdges'
    refined=load(args.source/'directClient/refinement/clientGeometry.npz')
    transform={key:refined['templateToUff'+key.title()] for key in ('rotation','translation','scale')}
    mouth=load(args.source/'directClient/refinement/mouthState.npz')
    cameras=json.loads((args.source/'cameras.json').read_text())
    assets=AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    oldSh=load(args.input/'baking/shOnly/clientTexture.npz')
    od=load(args.input/'baking/openDelight/clientTexture.npz')
    inputImages=load(args.input/'shImages.npz')
    originalGain=np.asarray(json.loads((args.source/'directClient/baking/diagnostics.json').read_text())['templateLinearColorGain'])
    oldFallback=np.clip(assets.materials[0]['baseColor'][...,:3]*originalGain,0,1).astype(np.float32)
    provenance={k:v for k,v in oldSh.items() if k!='linearRGB'}
    provenance['openDelightBoundarySafeSkinDonor']=od['boundarySafeSkinDonor']
    provenance['openDelightDirectPhotoAlpha']=od['boundaryDirectPhotoAlpha']
    matchStart=perf_counter()
    matched=matchTemplateToSh(oldFallback,oldSh['linearRGB'],od['linearRGB'],provenance)
    matchSeconds=perf_counter()-matchStart
    writeJson(args.output/'templateMatch.json',matched['diagnostics'])
    np.savez_compressed(args.output/'templateMatchFallback.npz',linearRGB=matched['linearRGB'],**matched['maps'],**matched.get('debug',{}))
    display(oldFallback).save(args.output/'fallback-before.png')
    display(matched['linearRGB']).save(args.output/'fallback-matched.png')
    print(json.dumps({'stage':'templateMatch','seconds':matchSeconds,'diagnostics':matched['diagnostics']}),flush=True)
    # Recompute only the cached-photo bake to reconstruct its pre-SDF colors.
    # COLMAP settings, masks, geometry and the front-mouth procedure stay fixed.
    raw=bakeClientTexture(geometry,cameras,inputImages,matched['linearRGB'],args.output/'preBoundaryBake',
        colmapExecutable=ROOT/'.runtimes/colmap/bin/colmap',mouthState=mouth,fallbackGain=None,
        boundaryBlend=False,smallSkinHoles=False)
    frozenKeys=('coverageMask','colmapCoverageMask','sourceIndex','faceSourceIndex','frontAlpha',
                'atlasMask','assignedMask','gutter','filterPadding','filterObservedKernelFraction')
    for key in frozenKeys:
        if not np.array_equal(raw[key],oldSh[key]):
            raise ValueError('Fixed observation provenance changed: '+key)
    rawProvenance={k:v for k,v in raw.items() if k not in {'linearRGB','diagnostics'}}
    blendInput={**rawProvenance,**{k:v for k,v in provenance.items() if k.startswith('boundary') or k.startswith('openDelight')}}
    # The preserved deployed bake predates exterior SDF repair. Its colors
    # therefore do not contain the previous OD-matched template at the rim.
    odRawPath=args.source/'directClient/baking/clientTexture.npz'
    odRaw=load(odRawPath)
    if any(key.startswith('boundary') for key in odRaw):
        raise ValueError('OD auxiliary input must be a pre-boundary bake')
    for key in frozenKeys:
        if not np.array_equal(odRaw[key],oldSh[key]):
            raise ValueError('OD pre-boundary observation differs: '+key)
    edge=odEdgeLowFrequency(raw['linearRGB'],odRaw['linearRGB'],blendInput)
    writeJson(args.output/'odEdge.json',edge['diagnostics'])
    np.savez_compressed(args.output/'odEdge.npz',**edge['maps'],**edge['debug'])
    comparison=args.output/'comparison'; comparison.mkdir()
    branches=[{'id':'shOriginal','label':'SH 原版'}, {'id':'templateMatch','label':'SH＋模板肤色适配'},
              {'id':'hybridEdge','label':'SH＋模板适配＋OD 边界'}, {'id':'openDelight','label':'OpenDelight 参考'}]
    config={'title':'保留 SH 外观 · 模板接缝对照', 'subtitle':'保留面部核心、眼鼻嘴的 SH 像素，比较模板肤色适配与 OpenDelight 边界辅助。',
        'brand':'SH 边界实验','badge':'本地 · 同一几何与相机','branches':branches,
        'defaultLeft':'shOriginal','defaultRight':'templateMatch','defaultViewerMode':'baseColor','defaultRenderMode':'baseColor',
        'overviewDescription':'四张原照片与四个分支的基色、共同光照渲染。',
        'notes':['模板先按同一位置的可靠肤色适配 SH，再进行安全肤色外扩；黑发、眼口暗腔和其他材质单独保护。',
            'OD 辅助只作用于原有的外边界带，先对齐 SH 色阶，再引入低频颜色变化。',
            '脸部核心和眼鼻嘴的真实 SH 像素保持不变，鼻翼、眼下的明暗保留在颜色贴图中。',
            '模板补色是估计颜色，未增加照片观测覆盖。当前仅为这个男生样本的本地实验。']}
    writeJson(comparison/'config.json',config)
    for name,sourceName in [('shOriginal','shOnly'),('openDelight','openDelight')]:
        (comparison/name).mkdir()
        shutil.copyfile(args.input/'comparison'/sourceName/'legacyAsset.zip',comparison/name/'legacyAsset.zip')
    baseline=_load(comparison/'shOriginal/legacyAsset.zip')
    core=oldSh['coverageMask'] & ~oldSh['boundaryModified']
    features=oldSh['coverageMask'] & oldSh['boundaryProtectedFeatures']
    report={'status':'running','input':str(args.input),'source':str(args.source),
        'inputHashes':{name:sha(args.input/name) for name in ['baking/shOnly/clientTexture.npz','baking/openDelight/clientTexture.npz','shImages.npz']},
        'sourceHashes':{name:sha(args.source/name) for name in ['directClient/geometry.npz','cameras.json','directClient/refinement/mouthState.npz','directClient/baking/clientTexture.npz']},
        'templateMatch':matched['diagnostics'],'odEdge':edge['diagnostics'],
        'stageSeconds':{'templateMatch':matchSeconds,'cachedShBake':raw['diagnostics']['seconds']},
        'frozenCoreTexels':int(core.sum()),'frozenFeatureTexels':int(features.sum()),
        'geometryAndCamerasFrozen':True,'remoteModified':False,'branches':{}}
    for name,linear,extra in [('templateMatch',raw['linearRGB'],{}),('hybridEdge',edge['linearRGB'],edge['maps'])]:
        start=perf_counter()
        boundary=blendClientTextureExterior(geometry,linear,rawProvenance,matched['linearRGB'],cameras,
            mouthState=mouth,imageShape=inputImages['validMasks'].shape[1:])
        maps={**rawProvenance,**boundary['maps'],**matched['maps'],**extra}
        holes=fillClientTextureSkinHoles(boundary['linearRGB'],maps)
        colors=holes['linearRGB'];maps.update(holes['maps'])
        checks={'coreExactlyUnchanged':np.array_equal(colors[core],oldSh['linearRGB'][core]),
                'featuresExactlyUnchanged':np.array_equal(colors[features],oldSh['linearRGB'][features]),
                'mouthExactlyUnchanged':np.array_equal(colors[oldSh['frontAlpha']>0],oldSh['linearRGB'][oldSh['frontAlpha']>0])}
        if not all(checks.values()):
            raise ValueError('SH preservation guard failed: '+str(checks))
        for key in frozenKeys:
            if not np.array_equal(maps[key],oldSh[key]):raise ValueError('Provenance differs: '+key)
        assembly=assets.assembleFinalClientSurface(geometry,transform,colors,maps,eyePlacementMode='assetDefault',
            diagnostics={'experiment':'shTemplateSeam','branch':name,'templateMatch':matched['diagnostics'],
                         'odEdgeApplied':name=='hybridEdge'})
        exportLegacyAsset(comparison/name,assembly,gender='male')
        exported=_load(comparison/name/'legacyAsset.zip')
        for key in ('vertices','faces','uvs','uvFaces','materialIds'):
            if not np.array_equal(exported.mesh[key],baseline.mesh[key]):raise ValueError('Geometry changed: '+key)
        for texture in ('body_color.png','eye_color.png','brow_color.png','head_normals.png'):
            if exported.payloads[texture]!=baseline.payloads[texture]:raise ValueError('Accessory texture changed: '+texture)
        np.savez_compressed(args.output/(name+'.npz'),linearRGB=colors,**maps)
        display(colors).save(args.output/(name+'-atlas.png'))
        edgeReport=exteriorEdgeDiagnostics(oldSh['linearRGB'],colors,oldSh['coverageMask'],oldSh['atlasMask'],oldSh['boundaryProtectedFeatures'])
        report['branches'][name]={'seconds':perf_counter()-start,'preservation':checks,'observedTexels':int(maps['coverageMask'].sum()),
            'boundary':boundary['diagnostics'],'smallHoles':holes['diagnostics'],'fixedOriginalExterior':edgeReport,
            'zipSha256':sha(comparison/name/'legacyAsset.zip'),
            'headColorSha256':sha(comparison/name/'head_color.png')}
        writeJson(args.output/'report.json',report)
        print(json.dumps({'stage':'exported','branch':name,'preservation':checks,'edge':edgeReport}),flush=True)
    report.update(status='complete',seconds=perf_counter()-started,
        codeHashes={name:sha(ROOT/name) for name in ['tools/runClientShSeamExperiment.py','src/faceAppearance/clientShTemplateBlend.py',
            'src/faceAppearance/clientTextureBoundary.py','src/faceAppearance/clientTextureBaking.py']})
    writeJson(args.output/'report.json',report)
    print(json.dumps({'status':'complete','seconds':report['seconds'],'output':str(args.output)}),flush=True)


if __name__=='__main__':main()
