"""Local SH/OpenDelight comparison on the frozen, final client surface.

Only appearance is recomputed. No model inference, registration, camera fitting,
remote service changes, or dense UFF UV transfer is performed by this tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]

from faceAppearance.appearanceObservations import srgbToLinear, linearToGamma22
from faceAppearance.artifacts import writeJson
from faceAppearance.clientTextureBaking import bakeClientTexture
from faceAppearance.data import readImage
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.legacyAsset import exportLegacyAsset
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.shDelight import shBasis, solveShDelight
from faceAppearance.clientShDelight import (collectClientPhotoObservations as collect,
                                           renderClientPhotoNormals as photoNormals)
from faceAppearance.texture import samplePixels
from renderCropComparison import makeGrid


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replayComparisonPhoto(path, record):
    """Replay only when both geometry and color processing match the OD cache.

    Old metadata without color provenance is safe only for unprofiled images.
    Reject tagged legacy photos rather than mix newly ICC-corrected pixels with
    a historical OpenDelight cache whose input color processing is unknown.
    """
    working, replay = readImage(path, imageSize=record['workingSize'][0],
        cropPolicy=record['requestedCropPolicy'], cropMargin=record['cropMargin'],
        faceDetector=lambda _: record['faceDetection'])
    if replay['sha256'] != record['sha256']:
        raise ValueError('Original photo hash differs')
    for key in ('inputToWorking', 'cropBoxXyxy', 'paddingLeftTopRightBottom', 'workingSize'):
        if replay[key] != record[key]:
            raise ValueError('Preprocessing replay differs: ' + key)
    color = replay['colorManagement']
    if 'colorManagement' not in record:
        if color['sourceIccPresent']:
            raise ValueError('Cached input predates ICC color provenance; refusing to mix a tagged photo '
                             'with historical OpenDelight pixels. Use verified original working PNGs '
                             'in a separate cache-based experiment or rebuild all compared stages.')
    elif record['colorManagement'] != color:
        raise ValueError('Color preprocessing replay differs from the OpenDelight cache')
    return working, replay



def display(linear):
    value = np.clip(np.asarray(linear, np.float64), 0, 1)
    srgb = np.where(value <= .0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - .055)
    return Image.fromarray(np.rint(srgb * 255).astype(np.uint8))



def consistency(colors, weights):
    """Descriptive training-view consistency; not reflectance accuracy."""
    total = weights.sum(0)
    multi = (weights > 0).sum(0) >= 2
    shared = np.divide((colors * weights[..., None]).sum(0), total[..., None],
                       out=np.zeros_like(colors[0]), where=total[..., None] > 0)
    w = weights * multi[None]
    if not w.any():
        raise ValueError('No multi-view observations for consistency statistics')
    rmse = np.sqrt(np.sum(w[..., None] * (colors - shared[None]) ** 2) / (3 * w.sum()))
    mean = np.sum(w[..., None] * colors, axis=(0, 1, 2)) / w.sum()
    luminance = float(mean @ np.array([.2126, .7152, .0722]))
    return {'linearRgbRmse': float(rmse), 'relativeToMeanLuminance': float(rmse / max(luminance, 1e-8)),
            'meanLinearRgb': mean.tolist(), 'multiViewTexels': int(multi.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'runs/clientAppearanceFix20261012/deployedSource')
    parser.add_argument('--photos', type=Path, default=ROOT/'runs/legacyPreview/preview-fb21ed26fc4b41969d581365622054e5')
    parser.add_argument('--baseline', type=Path, default=ROOT/'runs/clientAppearanceFix20261012/comparison/combined/legacyAsset.zip')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/clientShComparison20261012')
    parser.add_argument('--fit-resolution', type=int, default=512)
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--colmap', type=Path, default=ROOT/'.runtimes/colmap/bin/colmap')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Experiment output must be absent or empty')
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    sourcePaths = {name: args.source/name for name in ['directClient/geometry.npz', 'directClient/refinement/clientGeometry.npz',
        'directClient/refinement/mouthState.npz', 'cameras.json', 'inputs.json', 'delight.npz', 'directClient/baking/diagnostics.json']}
    hashes = {key: sha(path) for key, path in sourcePaths.items()}
    geometry = load(sourcePaths['directClient/geometry.npz'])
    geometry['uvConvention'] = 'topLeftPixelEdges'
    refined = load(sourcePaths['directClient/refinement/clientGeometry.npz'])
    transform = {key: refined['templateToUff'+key.title()] for key in ('rotation', 'translation', 'scale')}
    mouth = load(sourcePaths['directClient/refinement/mouthState.npz'])
    cameras = json.loads(sourcePaths['cameras.json'].read_text())
    inputs = json.loads(sourcePaths['inputs.json'].read_text())
    od = load(sourcePaths['delight.npz'])
    photoImages, photoHashes = [], []
    for i, record in enumerate(inputs):
        path = args.photos/f'{i:02d}.image'
        working, replay = replayComparisonPhoto(path, record)
        photoImages.append(working)
        photoHashes.append(sha(path))
    raw = srgbToLinear(np.asarray(photoImages))
    masks = od['validMasks']
    prior = od['albedoImages'] ** 2.2
    sampleStart = perf_counter()
    observations = collect(geometry, cameras, raw, prior, masks, args.fit_resolution)
    sampleSeconds = perf_counter() - sampleStart
    writeJson(args.output/'observations.json', observations['diagnostics'])
    np.savez_compressed(args.output/'observations.npz', **{k:v for k,v in observations.items() if isinstance(v,np.ndarray)})
    print(json.dumps({'stage': 'sampling', 'seconds': sampleSeconds, **observations['diagnostics']}), flush=True)
    fitStart = perf_counter()
    fit = solveShDelight(observations['raw'], observations['weights'], observations['normals'],
                         mode='shOnly', priorAlbedo=None, iterations=args.iterations)
    fitSeconds = perf_counter() - fitStart
    writeJson(args.output/'shFit.json', fit['diagnostics'])
    np.savez_compressed(args.output/'shFit.npz', **{k:v for k,v in fit.items() if isinstance(v,np.ndarray)})
    applicationStart = perf_counter()
    corrected, correctionMasks, records = raw.copy(), [], []
    inputPanels, inputLabels = [], []
    for i, camera in enumerate(cameras):
        normal, validNormal = photoNormals(geometry, camera, masks.shape[1:])
        lighting = np.einsum('hwj,jc->hwc', shBasis(normal), fit['coefficients'][i])
        lightSafe = ((lighting >= .25).all(-1) & (lighting <= 4).all(-1) & np.isfinite(lighting).all(-1))
        candidate = np.divide(raw[i], lighting, out=np.zeros_like(raw[i]), where=lightSafe[..., None])
        rangeSafe = np.isfinite(candidate).all(-1) & (candidate >= 0).all(-1) & (candidate <= 1).all(-1)
        # Dark/saturated pixels do not fit the lighting. Applying a finite,
        # bounded estimated field to those colors is separately declared here.
        fitted = fit['diagnostics']['views'][i]['status'] == 'fitted'
        apply = validNormal & lightSafe & rangeSafe & fitted
        corrected[i, apply] = candidate[apply]
        correctionMasks.append(apply)
        support = masks[i]
        records.append({'viewIndex': i, 'fitSucceeded': fitted, 'validPhotoPixels': int(support.sum()),
            'correctedValidPhotoPixels': int((apply & support).sum()),
            'fallbackValidPhotoPixels': int((~apply & support).sum()),
            'missingNormalPixels': int((support & ~validNormal).sum()),
            'unsafeLightingPixels': int((support & validNormal & ~lightSafe).sum()),
            'outOfRangePixels': int((support & validNormal & lightSafe & ~rangeSafe).sum()),
            'rawFallbackFraction': float((~apply & support).sum() / support.sum())})
        display(raw[i]).save(args.output/f'view{i}-raw.png')
        display(prior[i]).save(args.output/f'view{i}-openDelight.png')
        display(corrected[i]).save(args.output/f'view{i}-shOnly.png')
        Image.fromarray((apply & support).astype(np.uint8)*255).save(args.output/f'view{i}-shCorrectionMask.png')
        for name, value in [('Raw / contains light', raw[i]), ('OpenDelight', prior[i]), ('SH / fallback outside correction mask', corrected[i])]:
            # Show the shared valid region, avoiding inference backgrounds.
            panel = display(np.where(support[..., None], value, .025))
            inputPanels.append(panel)
            inputLabels.append(f'Photo {i+1} / {name}')
    correctionMasks = np.asarray(correctionMasks)
    applicationSeconds = perf_counter() - applicationStart
    np.savez_compressed(args.output/'shImages.npz', albedoImages=linearToGamma22(corrected),
                        validMasks=masks, correctionMasks=correctionMasks)
    makeGrid(inputPanels, inputLabels, columns=3, panelSize=384,
        subtitle='Same 512 input / source masks | sRGB display of linear color | SH fallback retained and reported').save(args.output/'inputComparison.png')
    correctedSamples = np.zeros_like(observations['raw'])
    commonWeights = observations['weights'].copy()
    for i in range(len(cameras)):
        active = commonWeights[i] > 0
        xy = observations['pixels'][i, active]
        correctedSamples[i, active] = samplePixels(corrected[i], xy)
        commonWeights[i, active] *= samplePixels(correctionMasks[i].astype(np.float32), xy) >= .999
    metrics = {name: consistency(colors, commonWeights) for name,colors in [
        ('raw', observations['raw']), ('openDelight', observations['openDelight']), ('shOnly', correctedSamples)]}
    assets = AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    gain = np.asarray(json.loads(sourcePaths['directClient/baking/diagnostics.json'].read_text())['templateLinearColorGain'])
    fallback = assets.materials[0]['baseColor'][..., :3]
    baseline = _load(args.baseline)
    comparison = args.output/'comparison'
    comparison.mkdir()
    branches = [('openDelight', 'OpenDelight（当前）', od['albedoImages']),
                ('shOnly', 'SH 去光', linearToGamma22(corrected)),
                ('raw', '原照片（含光参考）', linearToGamma22(raw))]
    config = {'title': 'SH 与 OpenDelight · 本地对照', 'subtitle': '同一男生样本，固定当前客户端几何、眼球、相机与融合参数。切换分支查看去光方法的影响。',
        'branches': [{'id':name, 'label':label} for name,label,_ in branches],
        'defaultLeft': 'openDelight', 'defaultRight': 'shOnly',
        'defaultViewerMode': 'baseColor', 'defaultRenderMode': 'baseColor',
        'brand': '去光实验', 'badge': '本地 · 同一男生样本 · 三分支',
        'overviewDescription': '原照片、OpenDelight、SH 与含光参考：每个拍摄视角均提供基色和共同光照。',
        'notes': ['纯 SH 在最终客户端模型的法线上拟合每视角光照；输入与 OpenDelight 使用相同的 512 裁剪。',
            '三组使用相同的 COLMAP、正面嘴部处理、SDF 边界和眼球修正。原照片分支保留拍摄光照，只作参考。',
            'SH 在缺少可靠法线或校正超出安全范围时回退原色；回退比例保存在本地实验报告。',
            '肤色和光照未经物理标定；更清晰的纹理也可能包含残留阴影。本轮仅运行于本地。']}
    writeJson(comparison/'config.json', config)
    result = {'status': 'running', 'sourceHashes': hashes, 'photoHashes': photoHashes,
        'geometryFrozen': True, 'camerasFrozen': True, 'denseUvUsed': False, 'modelsRerun': False,
        'remoteModified': False, 'sourceBaselineSha256': sha(args.baseline),
        'stagesSeconds': {'sampling': sampleSeconds, 'shFit': fitSeconds, 'shPhotoApplication': applicationSeconds},
        'shApplication': records, 'trainingViewConsistency': metrics,
        'parameters': {'fitResolution': args.fit_resolution, 'iterations': args.iterations,
            'same512Bandwidth': True, 'templateGainFixed': gain.tolist(),
            'shPriorUsed': False, 'shGainRange': [.25, 4.0],
            'fitExcludesDarkAndSaturated': True, 'applicationMayCorrectDarkAndSaturatedIfBounded': True,
            'unsafeApplication': 'raw-photo fallback, not black or clipped correction',
            'sameInputValidityMasks': True, 'colmapColorCorrection': 'same algorithm, fitted independently per branch'},
        'branches': {}, 'limitations': config['notes'] + [
            'Cross-view RMSE uses participating fitting views, not held-out reflectance truth.',
            'OpenDelight native transfer is assumed gamma2.2; raw inputs use sRGB decoding.',
            'SDF skin donor gates and small-hole filling are color dependent despite identical parameters.']}
    writeJson(args.output/'report.json', result)
    reference = None
    frozenKeys = ('coverageMask', 'colmapCoverageMask', 'sourceIndex', 'faceSourceIndex',
                  'frontAlpha', 'atlasMask', 'assignedMask', 'gutter', 'filterPadding', 'filterObservedKernelFraction')
    for name, label, images in branches:
        print(json.dumps({'stage': 'baking', 'branch': name}), flush=True)
        baked = bakeClientTexture(geometry, cameras, {'albedoImages':images, 'validMasks':masks}, fallback,
            args.output/'baking'/name, colmapExecutable=args.colmap, mouthState=mouth, fallbackGain=gain)
        if reference is None:
            reference = baked
        same = {key: bool(np.array_equal(reference[key], baked[key])) for key in frozenKeys}
        if not all(same.values()):
            raise ValueError('Selected source or footprint changed: '+str(same))
        provenance = {key:value for key,value in baked.items() if key not in {'linearRGB', 'diagnostics'}}
        assembly = assets.assembleFinalClientSurface(geometry, transform, baked['linearRGB'], provenance,
            eyePlacementMode='assetDefault', diagnostics={'experiment': 'localShComparison', 'branch': name})
        exportLegacyAsset(comparison/name, assembly, gender='male')
        exported = _load(comparison/name/'legacyAsset.zip')
        for key in ('vertices', 'faces', 'uvs', 'uvFaces', 'materialIds'):
            if not np.array_equal(exported.mesh[key], baseline.mesh[key]):
                raise ValueError('Complete assembled geometry differs: '+key)
        for texture in ('body_color.png', 'eye_color.png', 'brow_color.png'):
            if exported.payloads[texture] != baseline.payloads[texture]:
                raise ValueError('Nonhead texture changed: '+texture)
        identical = {key: exported.payloads[key] == baseline.payloads[key] for key in baseline.payloads if key in exported.payloads}
        display(baked['linearRGB']).save(args.output/f'{name}-atlas.png')
        result['branches'][name] = {'seconds': baked['diagnostics']['seconds'], 'frozenProvenanceEqual': same,
            'zipSha256': sha(comparison/name/'legacyAsset.zip'), 'baselinePayloadEquality': identical,
            'headTextureSha256': hashlib.sha256(exported.payloads['head_color.png']).hexdigest(),
            'observedTexels': int(baked['coverageMask'].sum()), 'boundaryBlend': baked['diagnostics']['boundaryBlend'],
            'smallSkinHoles': baked['diagnostics']['smallSkinHoles']}
        writeJson(args.output/'report.json', result)
        print(json.dumps({'stage': 'baked', 'branch':name, 'seconds':baked['diagnostics']['seconds'],
                          'observedTexels':int(baked['coverageMask'].sum()), 'sameSelection':all(same.values())}), flush=True)
    result.update(status='complete', seconds=perf_counter()-started,
                  sourceFilesUnchanged=hashes == {key:sha(path) for key,path in sourcePaths.items()})
    writeJson(args.output/'report.json', result)
    print(json.dumps({'status':'complete', 'output':str(args.output), 'seconds':result['seconds'],
                      'shApplication':records, 'consistency':metrics}), flush=True)


if __name__ == '__main__':
    main()
