"""Replay SH appearance on a verified delivered head; CPU only, no service calls."""
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
from faceAppearance.clientShDelight import delightClientPhotos
from faceAppearance.clientTextureBaking import bakeClientTexture
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.legacyAppearanceVariant import exportLegacyAppearanceVariant
from runClientShComparison import consistency, display
from faceAppearance.texture import samplePixels


def loadArrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def run(caseDirectory, outputDirectory, referenceViewIndex, fitResolution=256,
        geometryReplayDirectory=None, fallbackPath=None):
    case, output = Path(caseDirectory).resolve(), Path(outputDirectory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new experiment output directory')
    output.mkdir(parents=True, exist_ok=True)
    root = case / 'reconstruction'
    sourceNames = ['directClient/geometry.npz', 'delight.npz', 'cameras.json', 'inputs.json',
                   'directClient/refinement/mouthState.npz']
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sourceNames}
    geometryPath = root / 'directClient/geometry.npz'
    mouthPath = root / 'directClient/refinement/mouthState.npz'
    baselineArchive = case / 'asset/legacyAsset.zip'
    geometryReplay = None
    if geometryReplayDirectory is not None:
        replay = Path(geometryReplayDirectory).resolve()
        replayReport = json.loads((replay / 'report.json').read_text())
        safety = replayReport['finalChainSafety']
        if (replayReport['case'] != case.name or not safety['accepted']
                or safety['newIntersectionPairsAgainstOriginalDirect'] != 0
                or replayReport['selection']['referenceViewIndex'] != referenceViewIndex):
            raise ValueError('Geometry replay must match this case/reference and pass its complete safety gate')
        geometryPath, mouthPath = replay / 'finalGeometry.npz', replay / 'mouthState.npz'
        for path in (geometryPath, mouthPath):
            if hashlib.sha256(path.read_bytes()).hexdigest() != replayReport['outputHashes'][path.name]:
                raise ValueError('Geometry replay output differs from its safety-checked report')
        baselineArchive = replay / 'diagnosticAsset/legacyAsset.zip'
        if hashlib.sha256(baselineArchive.read_bytes()).hexdigest() != replayReport['exported']['sha256']:
            raise ValueError('Geometry replay archive differs from its report')
        geometryReplay = {'directory': str(replay), 'finalChainSafety': safety,
            'sourceHashes': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (replay/'report.json', geometryPath, mouthPath, baselineArchive)}}
    geometry = loadArrays(geometryPath)
    geometry['uvConvention'] = 'topLeftPixelEdges'
    cameras = json.loads((root / 'cameras.json').read_text())
    inputs = json.loads((root / 'inputs.json').read_text())
    od = loadArrays(root / 'delight.npz')
    photos = []
    for index, record in enumerate(inputs):
        path = root / f'input{index:02d}.png'
        with Image.open(path) as source:
            if list(source.size) != record['workingSize']:
                raise ValueError('Saved photo size differs from input metadata')
            photos.append(np.asarray(source.convert('RGB'), np.float32) / 255)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    raw = srgbToLinear(np.asarray(photos))
    started = perf_counter()
    result = delightClientPhotos(geometry, cameras, raw, od['validMasks'].astype(bool),
                                 referenceViewIndex=referenceViewIndex, fitResolution=fitResolution)
    fitSeconds = perf_counter() - started
    writeJson(output / 'shDiagnostics.json', result['diagnostics'])
    np.savez_compressed(output / 'shImages.npz', linearImages=result['linearImages'],
                        validMasks=result['validMasks'], correctionMasks=result['correctionMasks'],
                        coefficients=result['coefficients'])
    observations = result['observations']
    correctedSamples = np.zeros_like(observations['raw'])
    commonWeights = observations['weights'].copy()
    for index in range(len(cameras)):
        active = commonWeights[index] > 0
        xy = observations['pixels'][index, active]
        correctedSamples[index, active] = samplePixels(result['linearImages'][index], xy)
        commonWeights[index, active] *= samplePixels(result['correctionMasks'][index].astype(np.float32), xy) >= .999
        display(result['linearImages'][index]).save(output / f'view{index:02d}Sh.png')
    metrics = {name: consistency(colors, commonWeights) for name, colors in
               [('raw', observations['raw']), ('sh', correctedSamples)]}
    print(json.dumps({'stage': 'fit', 'seconds': fitSeconds, 'metrics': metrics}), flush=True)
    assets = AssemblyAssets(ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1')
    mouth = loadArrays(mouthPath)
    fallback = assets.materials[0]['baseColor'][..., :3]
    fallbackGain = 'auto'
    fallbackSource = None
    if fallbackPath is not None:
        fallbackPath = Path(fallbackPath).resolve()
        fallback = loadArrays(fallbackPath)['linearRGB']
        fallbackGain = None
        fallbackSource = {'path': str(fallbackPath),
                          'sha256': hashlib.sha256(fallbackPath.read_bytes()).hexdigest(),
                          'purpose': 'Pre-adapted template for unobserved skin; new geometry is fully rebaked'}
    baked = bakeClientTexture(geometry, cameras,
        {'albedoImages': linearToGamma22(result['linearImages']), 'validMasks': result['validMasks']},
        fallback, output / 'baking',
        colmapExecutable=ROOT / '.runtimes/colmap/bin/colmap', mouthState=mouth,
        referenceView=referenceViewIndex, fallbackGain=fallbackGain, threads=2)
    shApplication = {**result['diagnostics']['applicationSummary'],
        'views': [{'viewIndex': item['viewIndex'], 'validPhotoPixels': item['validPhotoPixels'],
                   'fallbackValidPhotoPixels': item['fallbackValidPhotoPixels'],
                   'rawFallbackFraction': item['rawFallbackFraction']} for item in result['diagnostics']['application']],
        'correctionMaskArtifact': 'shImages.npz:correctionMasks',
        'rawFallbackMeaning': 'Uncorrected pixels retain photographic lighting and may remain photo-supported in the baked texture; coverage is not SH-success coverage'}
    variant = exportLegacyAppearanceVariant(baselineArchive, baked['linearRGB'],
        output / 'asset', diagnostics={'method': 'frontReferenceShPhotoDelight',
            'referenceViewIndex': referenceViewIndex, 'fallbackGain': fallbackGain,
            'sourceTaskId': case.name, 'shApplication': shApplication,
            'geometryReplay': geometryReplay, 'fallbackSource': fallbackSource})
    artifactIdentities = {name: {'sha256': hashlib.sha256((output / name).read_bytes()).hexdigest(),
                                'bytes': (output / name).stat().st_size}
        for name in ('shImages.npz', 'shDiagnostics.json', 'baking/clientTexture.npz')}
    report = {'status': 'complete', 'sourceTaskId': case.name, 'sourceHashes': hashes,
        'sourceArchiveSha256': hashlib.sha256((case / 'asset/legacyAsset.zip').read_bytes()).hexdigest(),
        'referenceViewIndex': referenceViewIndex, 'fitResolution': fitResolution,
        'geometryReplay': geometryReplay, 'fallbackSource': fallbackSource,
        'geometryBaselineArchiveSha256': hashlib.sha256(baselineArchive.read_bytes()).hexdigest(),
        'fitSeconds': fitSeconds, 'totalSeconds': perf_counter() - started,
        'trainingViewConsistency': metrics, 'variant': variant, 'baking': baked['diagnostics'],
        'artifactIdentities': artifactIdentities, 'shApplication': shApplication,
        'limitations': ['Uses frozen production working PNGs to isolate appearance; ICC fix is tested separately.',
            ('Uses a locally safety-checked geometry candidate and reprojects/rebakes photos on that surface; cameras stay fixed.'
             if geometryReplay else 'Uses final production geometry and cameras without modification, rather than a geometry-fix candidate.'),
            'Reference RGB gauge preserves photo color scale, not measured absolute reflectance.',
            ('Unobserved skin uses a previously adapted template; it is not additional photo evidence.' if fallbackSource
             else 'Template fallback gain is recomputed from the new bake instead of reusing OpenDelight-derived gains.'),
            'Unsafe or unsupported SH estimates retain raw illuminated photo RGB; photo coverage is not SH correction coverage.',
            'Consistency scores use fitted views and are not independent reflectance accuracy.']}
    writeJson(output / 'report.json', report)
    print(json.dumps({'stage': 'complete', 'seconds': report['totalSeconds'], 'archive': variant['archivePath']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-view', dest='referenceView', type=int, required=True)
    parser.add_argument('--fit-resolution', dest='fitResolution', type=int, default=256)
    parser.add_argument('--geometry-replay', dest='geometryReplay', type=Path)
    parser.add_argument('--fallback-texture', dest='fallbackTexture', type=Path)
    args = parser.parse_args()
    run(args.case, args.output, args.referenceView, args.fitResolution, args.geometryReplay, args.fallbackTexture)
