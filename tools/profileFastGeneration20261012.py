"""Opt-in local photo-to-ZIP timing with separately measured startup/warmup.

Runs real UFF, OpenDelight, camera/fusion and final client generation twice in
one process. The first complete request is warmup. An optional gate separates
warmup from measurement so other experiments can stop before timing begins.
No HTTP, OSS, callback, deployment or production default is changed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter, sleep
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--photos', type=Path, default=ROOT/'runs/legacyPreviewJaw20261012/preview-609d6c9b63154933b7e3ecd46bb37719')
    parser.add_argument('--gate', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    photos = [args.photos.resolve()/f'{index:02d}.image' for index in range(4)]
    if not all(path.is_file() for path in photos):
        raise FileNotFoundError('Four original .image inputs are required')
    report = {'status': 'startup', 'pid': os.getpid(),
        'scope': 'local four original photographs through real UFF/delight/fusion/direct client to ZIP',
        'includesHttpOrNetwork': False, 'includesQueueWait': False, 'deployed': False,
        'requestTargetSeconds': [30, 60], 'startupExcludedFromRequestTime': True,
        'sourceSha256': [hashlib.sha256(path.read_bytes()).hexdigest() for path in photos],
        'startup': {}, 'requests': []}
    write(output/'profile.json', report)
    import torch
    if not torch.cuda.is_available():
        report.update(status='blockedNoCuda', reason='CUDA is unavailable in this process; no timing fabricated')
        write(output/'profile.json', report)
        raise RuntimeError(report['reason'])
    report['gpu'] = {'name': torch.cuda.get_device_name(), 'torch': torch.__version__,
                     'totalBytes': torch.cuda.get_device_properties(0).total_memory}
    from faceAppearance.config import PipelineConfig
    from faceAppearance.pipeline import Pipeline
    from faceAppearance.directClientService import assembleDirectClient, checkDirectRuntime
    from faceAppearance.legacyAsset import exportLegacyAsset
    from faceAppearance import clientRenderedJaw, renderedJawRuntime, renderedJawRefinement
    from faceAppearance import colmapTexture, clientTextureBaking, clientTextureBoundary
    from faceAppearance.clientUvLookupCache import ClientUvLookupCache
    from profileFastClient20261012 import _front_photo_observer, _cached_intersection_checker, render_front_native
    config = replace(PipelineConfig.fromFile(ROOT/'configs/stableLegacyService.json', ROOT),
        assemblyAsset='assets/clientTemplates/legacyMaleDirectServiceV1',
        colmapExecutable=str(ROOT/'.runtimes/colmap/bin/colmap'))
    tick = perf_counter()
    pipeline = Pipeline(config).loadModels()
    assets = pipeline.assemblyAssets
    runtime = checkDirectRuntime(config, assets)
    torch.cuda.synchronize()
    report['startup']['modelAndAssetLoadSeconds'] = perf_counter()-tick
    report['runtime'] = runtime
    arrays = assets.arrays
    face_lookup = {tuple(face): index for index, face in enumerate(arrays['faces'])}
    full_ids = np.array([face_lookup[tuple(face)]
        for face in arrays['nicpToFullIds'][arrays['nicpFaces']]])
    geometry = {'vertices': arrays['vertices'][arrays['nicpToFullIds']],
        'faces': arrays['nicpFaces'], 'uvs': arrays['uvs'], 'uvFaces': arrays['uvFaces'][full_ids],
        'uvConvention': 'topLeftPixelEdges'}
    cache = ClientUvLookupCache(colmapTexture._destinationUvLookup)
    tick = perf_counter()
    cache.warmMouthGrids(geometry, assets.materials[0]['baseColor'].shape[0])
    report['startup']['uvGridWarmSeconds'] = perf_counter()-tick
    report['startup']['uvCache'] = cache.diagnostics()
    original_refine = clientRenderedJaw.refineRenderedClientJaw
    original_observer = renderedJawRuntime.observePhotos
    def fast_render(scene, vertices, cameras, inputs, observations, destination, *, projectRoot, viewIndices=None):
        if destination is not None:
            Path(destination).mkdir(parents=True, exist_ok=True)
        row, timing = render_front_native(scene, vertices, cameras, inputs, observations, projectRoot)
        timing.update(viewIndices=[0], workingDiagnosticRasterSkipped=True)
        return [row], timing
    def fast_refine(assets, vertices, faces, templateToUff, cameras, inputs, photoPaths,
                    *, projectRoot, outputDirectory):
        intersection_cache, counters = _cached_intersection_checker(vertices, faces)
        with patch.object(renderedJawRefinement, 'intersectionPairs', intersection_cache):
            result = original_refine(assets, vertices, faces, templateToUff, cameras, inputs, photoPaths,
                projectRoot=projectRoot, outputDirectory=outputDirectory)
        result['diagnostics']['fastExecution'] = {'frontOnly': True, 'nativeRasterOnly': True,
            'staticIntersectionReuse': counters, 'noSafetyThresholdRelaxation': True}
        return result
    def execute_request(name, *, measured):
        target = output/name
        target.mkdir()
        torch.cuda.synchronize()
        start = perf_counter()
        reconstruction = target/'reconstruction'
        reconstructed = pipeline.run(photos, reconstruction, stopAfter='fusion')
        geometry, texture = load(reconstruction/'geometry.npz'), load(reconstruction/'texture.npz')
        assembled, direct = assembleDirectClient(assets, geometry, texture, reconstruction, photos, config)
        tick = perf_counter()
        exportLegacyAsset(target/'asset', assembled, gender='male')
        export_seconds = perf_counter()-tick
        torch.cuda.synchronize()
        wall_seconds = perf_counter()-start
        archive = target/'asset/legacyAsset.zip'
        result = {'name': name, 'measuredRequest': measured, 'wallSeconds': wall_seconds,
            'uffRerun': True, 'delightRerun': True, 'inputBytesReadAgain': True,
            'geometryOrTextureResultCacheUsed': False,
            'stageSeconds': {**reconstructed['stageSeconds'], **direct['stageSeconds'],
                             'legacyZipExport': export_seconds},
            'pipelineInferenceSeconds': reconstructed.get('inferenceSeconds'),
            'directClientSeconds': direct['seconds'], 'jawStatus': direct['renderedJaw']['status'],
            'uvCache': cache.diagnostics(), 'zip': str(archive),
            'zipBytes': archive.stat().st_size, 'zipSha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
        write(target/'timing.json', result)
        report['requests'].append(result)
        write(output/'profile.json', report)
        print(json.dumps({'event': 'requestComplete', **result}, ensure_ascii=False), flush=True)
        return result
    with ExitStack() as contexts:
        for module in (colmapTexture, clientTextureBaking, clientTextureBoundary):
            contexts.enter_context(patch.object(module, '_destinationUvLookup', cache))
        contexts.enter_context(patch.object(renderedJawRuntime, 'observePhotos', _front_photo_observer(original_observer)))
        contexts.enter_context(patch.object(renderedJawRuntime, 'renderDetect', fast_render))
        contexts.enter_context(patch.object(clientRenderedJaw, 'refineRenderedClientJaw', fast_refine))
        report['status'] = 'warmingRealRequest'
        write(output/'profile.json', report)
        warmed = execute_request('warmup', measured=False)
        report['startup']['fullRequestWarmSeconds'] = warmed['wallSeconds']
        report['status'] = 'readyForMeasuredRequest'
        write(output/'profile.json', report)
        write(output/'ready.json', {'ready': True, 'pid': os.getpid(), 'gate': str(args.gate) if args.gate else None})
        print('READY_FOR_MEASURED_REQUEST', flush=True)
        if args.gate:
            while not args.gate.exists():
                sleep(.2)
        report['status'] = 'measuringRealRequest'
        write(output/'profile.json', report)
        measured = execute_request('measured', measured=True)
    report['status'] = 'completed'
    report['measuredRequestSeconds'] = measured['wallSeconds']
    report['withinSixtySeconds'] = measured['wallSeconds'] <= 60
    report['limitations'] = ['One male sample, local machine; not a server or HTTP SLA measurement.',
        'Includes existing fusion, current client geometry and jaw; new dense-eye quality work is not included.',
        'All original numerical geometry/intersection gates remain; nonfrontal semantic diagnostics are omitted.']
    write(output/'profile.json', report)


if __name__ == '__main__':
    main()
