"""Audit saved latency and benchmark front-only CPU rendering on cached geometry.

This is an isolated experiment, not a service configuration. UFF and OpenDelight
are never run. Saved complete HTTP timings and cached client timings are kept
separate. No timing in this report is a new end-to-end generation measurement.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
from pathlib import Path
import sys
import pstats
from time import perf_counter
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def report_rows():
    paths = {'currentMale': ROOT/'runs/liveJaw20261012/source/reconstruction'}
    paths.update({name: ROOT/'runs/eyelidTextureBatch20261012'/name/'baseline/reconstruction'
                  for name in ('highPitch4', 'cyanWall4', 'outdoorSideLight4')})
    result = []
    for name, source in paths.items():
        report = json.loads((source/'directClient/report.json').read_text())
        jaw = report['renderedJaw']
        colmap = report['baking']['colmap']
        result.append({'caseId': name, 'source': str(source),
            'measurementScope': 'saved remote delivered direct stage' if name == 'currentMale'
                else 'saved local replay of cached UFF/delight; excludes UFF/delight',
            'clientSeconds': report['seconds'], 'stageSeconds': report['stageSeconds'],
            'jawStageSeconds': jaw['stageSeconds'], 'jawStatus': jaw['status'],
            'colmapProcessSeconds': {key: value['seconds'] for key, value in colmap.items()},
            'bakeWrapperSeconds': report['baking']['seconds']})
    return result


def render_front_native(scene, vertices, cameras, inputs, observations, project_root):
    """Keep exact front FAN inputs, omit unused working-grid diagnostic raster."""
    from faceAppearance import renderedJawRuntime as runtime
    tick = perf_counter()
    detector = runtime.getDetector(project_root)
    current = {**scene, 'vertices': scene['vertices'].copy()}
    current['vertices'][scene['nicpToFullIds']] = vertices
    record, camera = inputs[0], cameras[0]
    original_size, working_size, mapping = runtime._inputCoordinates(record)
    box, rejected = runtime._savedBox(record, original_size)
    if box is None:
        raise ValueError(rejected)
    scale = float(observations['faceScaleWorkingPixels'][0])
    native_camera = {**camera, 'intrinsics': np.linalg.inv(mapping)@camera['intrinsics']}
    render_start = perf_counter()
    native, _, _ = runtime._raster(current, native_camera, original_size[::-1])
    render_seconds = perf_counter()-render_start
    row = {'viewIndex': 0, 'variants': {}}
    for variant in runtime.APPEARANCES:
        row['variants'][variant] = runtime._detect(detector, native[variant], box,
            mapping, working_size, scale)
    return row, {'wallSeconds': perf_counter()-tick, 'renderSeconds': render_seconds,
        'detectionSeconds': sum(item['seconds'] for item in row['variants'].values()),
        'photoViewCount': 1, 'rasterCount': 1, 'appearanceCount': 3,
        'cropVariantsPerAppearance': 3}


def benchmark_render(source, assets, *, all_views=False):
    from faceAppearance import renderedJawRuntime as runtime
    source = Path(source)
    jaw = source/'directClient/renderedJaw'
    cameras = json.loads((source/'cameras.json').read_text())
    inputs = json.loads((source/'inputs.json').read_text())
    if (jaw/'sourceState.npz').is_file():
        state = load(jaw/'sourceState.npz')
    else:
        state = {'vertices': load(jaw/'geometry.npz')['before']}
    if (jaw/'photoObservations.npz').is_file():
        observations = load(jaw/'photoObservations.npz')
    else:
        # renderDetect only consumes the saved face-box scale from observations.
        scales = []
        for record in inputs:
            box = np.asarray(record['faceDetection']['boxXyxy']).reshape(2, 2)
            projected = runtime._transform(box, record['inputToWorking'])
            scales.append(float(np.sqrt(np.prod(np.abs(projected[1]-projected[0])))))
        observations = {'faceScaleWorkingPixels': np.asarray(scales)}
    transform = json.loads((jaw/'templateToUff.json').read_text())
    tick = perf_counter()
    scene = runtime.buildScene(assets, state['vertices'], transform)
    scene_seconds = perf_counter()-tick
    tick = perf_counter()
    baseline, baseline_timing = runtime.renderDetect(scene, state['vertices'], cameras,
        inputs, observations, None, projectRoot=ROOT,
        viewIndices=None if all_views else [0])
    baseline_timing['wallSeconds'] = perf_counter()-tick
    fast, fast_timing = render_front_native(scene, state['vertices'], cameras,
        inputs, observations, ROOT)
    reference = next(row for row in baseline if row['viewIndex'] == 0)
    comparisons = {}
    for variant in runtime.APPEARANCES:
        comparisons[variant] = {}
        for key in ('points', 'variants', 'peaks', 'valid', 'spread'):
            old, new = reference['variants'][variant][key], fast['variants'][variant][key]
            comparisons[variant][key+'ExactlyEqual'] = bool(np.array_equal(old, new, equal_nan=True))
    if not all(all(row.values()) for row in comparisons.values()):
        raise RuntimeError('Front-only experiment changed front detector outputs')
    return {'scope': 'CPU renderer/FAN benchmark on saved geometry and observations; no UFF/delight',
        'sceneBuildSeconds': scene_seconds, 'baselineApi': baseline_timing,
        'nativeFrontOnly': fast_timing, 'frontDetectionsExactlyEqual': True,
        'equivalence': comparisons}


def benchmark_bake(source, destination, *, warm_uv=False):
    from contextlib import ExitStack
    from faceAppearance.headAssembly import AssemblyAssets
    from faceAppearance.clientTextureBaking import bakeClientTexture
    from faceAppearance import colmapTexture, clientTextureBaking, clientTextureBoundary
    from faceAppearance.clientUvLookupCache import ClientUvLookupCache
    source = Path(source)
    assets = AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    geometry = load(source/'directClient/geometry.npz')
    cameras = json.loads((source/'cameras.json').read_text())
    delight = load(source/'delight.npz')
    old = json.loads((source/'directClient/report.json').read_text())
    mouth_path = source/'directClient/mouthState.npz'
    mouth = load(mouth_path) if mouth_path.exists() else None
    profiler = cProfile.Profile()
    cache = ClientUvLookupCache(colmapTexture._destinationUvLookup)
    tick = perf_counter()
    if warm_uv:
        geometry['uvConvention'] = 'topLeftPixelEdges'
        cache.warmMouthGrids(geometry, assets.materials[0]['baseColor'].shape[0])
    warm_seconds = perf_counter()-tick
    tick = perf_counter()
    with ExitStack() as contexts:
        if warm_uv:
            for module in (colmapTexture, clientTextureBaking, clientTextureBoundary):
                contexts.enter_context(patch.object(module, '_destinationUvLookup', cache))
        result = profiler.runcall(bakeClientTexture, geometry, cameras, delight,
            assets.materials[0]['baseColor'][..., :3], destination,
            colmapExecutable=ROOT/'.runtimes/colmap/bin/colmap', mouthState=mouth,
            fallbackGain=old['baking']['templateLinearColorGain'])
    seconds = perf_counter()-tick
    profiler.dump_stats(str(destination/'baking.prof'))
    rows = []
    for (filename, line, name), (primitive, calls, self_seconds, cumulative, callers) in pstats.Stats(profiler).stats.items():
        rows.append({'file': filename, 'line': line, 'function': name, 'calls': calls,
                     'selfSeconds': self_seconds, 'cumulativeSeconds': cumulative})
    rows.sort(key=lambda row: row['cumulativeSeconds'], reverse=True)
    baseline = load(source/'directClient/baking/clientTexture.npz')
    equality = {key: bool(np.array_equal(value, baseline[key], equal_nan=True))
                for key, value in result.items() if isinstance(value, np.ndarray) and key in baseline}
    return {'scope': 'cached direct client texture rebake with profiler overhead; not full generation',
        'source': str(source), 'wallSeconds': seconds,
        'warmUvCache': warm_uv, 'startupUvWarmSeconds': warm_seconds,
        'uvCache': cache.diagnostics(), 'arrayEqualityWithSavedBake': equality,
        'colmap': {k: v['seconds'] for k, v in result['diagnostics']['colmap'].items()},
        'topFunctions': rows[:45], 'output': str(destination)}


def _front_photo_observer(original_observer):
    """Infer only explicitly designated first photo; preserve full view indexing."""
    def observe(project, photos, inputs):
        from PIL import Image, ImageOps
        from faceAppearance import renderedJawRuntime as runtime
        # Nonfrontal inputs still undergo identity/dimension checks, even though
        # their semantic arrays are deliberately unavailable in this experiment.
        for path, record in zip(photos[1:], inputs[1:]):
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('Nonfrontal photo identity mismatch')
            with Image.open(path) as opened:
                if list(ImageOps.exif_transpose(opened).size) != record['orientedSize']:
                    raise ValueError('Nonfrontal original photo dimensions mismatch')
        front = original_observer(project, photos[:1], inputs[:1])
        result = {}
        count = len(inputs)
        for key, value in front.items():
            if not isinstance(value, np.ndarray) or key == 'bboxScaleVariants':
                result[key] = value
                continue
            fill = False if value.dtype.kind == 'b' else (np.nan if value.dtype.kind == 'f' else 0)
            result[key] = np.full((count, *value.shape[1:]), fill, dtype=value.dtype)
            result[key][0] = value[0]
        result['workingImageShape'] = np.asarray([r['workingSize'][::-1] for r in inputs])
        for index, record in enumerate(inputs[1:], 1):
            box = np.asarray(record['faceDetection']['boxXyxy']).reshape(2, 2)
            box = runtime._transform(box, record['inputToWorking'])
            result['faceScaleWorkingPixels'][index] = np.sqrt(np.prod(np.abs(box[1]-box[0])))
        result['diagnostics'] = {**front['diagnostics'], 'method': 'frontOnlyCurrentPhotoCpuFan',
            'selectedReferenceViewIndex': 0, 'unobservedViewIndices': list(range(1, count)),
            'nonfrontalIdentityChecked': True, 'otherViewPolicy': 'No semantic inference; projection diagnostics only'}
        return result
    return observe


def _cached_intersection_checker(original, faces):
    """Reuse exactly unchanged face pairs, recompute every changed-face pair.

    Full global AABB broad phase is retained. The original narrow-phase routine
    and current-call epsilon are preserved; no contact threshold is relaxed.
    """
    from faceAppearance.clientGeometrySafety import broadPhase, intersectTriangles, intersectionPairs
    reference = np.asarray(original, np.float64).copy()
    topology = np.asarray(faces, np.int64).copy()
    references, memo = {}, {}
    counts = {'calls': 0, 'cacheHits': 0, 'fullBroadPhaseCandidates': 0, 'changedFaceCandidates': 0}
    def call(vertices, triangles, *, epsilon=None):
        if not np.array_equal(triangles, topology):
            raise ValueError('Intersection cache topology differs')
        vertices = np.asarray(vertices, np.float64)
        if epsilon is None:
            epsilon = max(float(np.linalg.norm(np.ptp(vertices, axis=0)))*1e-9, 1e-12)
        counts['calls'] += 1
        key = (float(epsilon), hashlib.sha256(vertices.tobytes()).digest())
        if key in memo:
            counts['cacheHits'] += 1
            return set(memo[key])
        if epsilon not in references:
            references[epsilon] = intersectionPairs(reference, topology, epsilon=epsilon)
        changed = np.any(vertices != reference, axis=1)[topology].any(axis=1)
        result = {pair for pair in references[epsilon] if not changed[list(pair)].any()}
        candidates, _ = broadPhase(vertices, topology, epsilon)
        counts['fullBroadPhaseCandidates'] += len(candidates)
        candidates = candidates[changed[candidates].any(axis=1)]
        counts['changedFaceCandidates'] += len(candidates)
        current = vertices[topology]
        for first, second in candidates:
            hit = intersectTriangles(current[first], current[second], epsilon)
            if hit and hit['kind'] in ('nonCoplanarSegment', 'coplanarAreaOverlap'):
                result.add((int(first), int(second)))
        memo[key] = frozenset(result)
        return result
    return call, counts


def fast_jaw(source, photos, destination, *, fast_intersections=False, audit_intersections=False):
    """Run the existing jaw solver with local, process-scoped observation hooks.

    Only this CLI process is patched. Production modules and defaults are never
    edited. The fixed reference is photo zero, as required by the API contract;
    equivalence with a previous run is explicitly reported, never assumed.
    """
    from contextlib import ExitStack
    from faceAppearance import renderedJawRuntime as runtime
    from faceAppearance import renderedJawRefinement as solver
    from faceAppearance.clientRenderedJaw import refineRenderedClientJaw
    from faceAppearance.headAssembly import AssemblyAssets
    source, photos = Path(source), Path(photos)
    jaw = source/'directClient/renderedJaw'
    old = json.loads((jaw/'report.json').read_text())
    geometry = load(jaw/'geometry.npz')
    cameras = json.loads((source/'cameras.json').read_text())
    inputs = json.loads((source/'inputs.json').read_text())
    transform = json.loads((jaw/'templateToUff.json').read_text())
    photo_paths = [photos/f'{index:02d}.image' for index in range(len(inputs))]
    assets = AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    runtime.getDetector(ROOT)  # startup load excluded from warm request timing
    observer = _front_photo_observer(runtime.observePhotos)
    def front_render(scene, vertices, cameras, inputs, observations, outputDirectory,
                     *, projectRoot, viewIndices=None):
        if outputDirectory is not None:
            Path(outputDirectory).mkdir(parents=True, exist_ok=True)
        row, timing = render_front_native(scene, vertices, cameras, inputs, observations, projectRoot)
        timing.update(viewIndices=[0], workingDiagnosticRasterSkipped=True)
        return [row], timing
    accelerated, counts = _cached_intersection_checker(geometry['before'], geometry['faces'])
    checker = accelerated
    if audit_intersections:
        from faceAppearance.clientGeometrySafety import intersectionPairs as full_checker
        def checker(vertices, faces, *, epsilon=None):
            fast = accelerated(vertices, faces, epsilon=epsilon)
            full = full_checker(vertices, faces, epsilon=epsilon)
            if fast != full:
                raise RuntimeError(f'Intersection cache mismatch: {sorted(fast^full)}')
            return fast
    tick = perf_counter()
    with ExitStack() as contexts:
        contexts.enter_context(patch.object(runtime, 'observePhotos', observer))
        contexts.enter_context(patch.object(runtime, 'renderDetect', front_render))
        if fast_intersections:
            contexts.enter_context(patch.object(solver, 'intersectionPairs', checker))
        result = refineRenderedClientJaw(assets, geometry['before'], geometry['faces'],
            transform, cameras, inputs, photo_paths, projectRoot=ROOT, outputDirectory=destination)
    wall_seconds = perf_counter()-tick
    new = result['diagnostics']
    projection_rows = []
    for index, camera in enumerate(cameras):
        before = geometry['before']@np.asarray(camera['rotation']).T+camera['translation']
        after = result['vertices']@np.asarray(camera['rotation']).T+camera['translation']
        a, b = before@np.asarray(camera['intrinsics']).T, after@np.asarray(camera['intrinsics']).T
        delta = np.linalg.norm(a[:, :2]/a[:, 2:]-b[:, :2]/b[:, 2:], axis=1)
        projection_rows.append({'viewIndex': index, 'allDepthsPositive': bool((after[:, 2]>0).all()),
            'maximumProjectionDisplacementPixels': float(delta.max()),
            'allProjectedCoordinatesFinite': bool(np.isfinite(delta).all()),
            'meaning': 'Geometric numeric diagnostic; not a semantic likeness or visibility certificate'})
    equality = bool(np.array_equal(result['vertices'], geometry['after']))
    summary = {'source': str(source), 'scope': 'warm local jaw stage only; current photos inferred, cached incoming geometry',
        'wallSeconds': wall_seconds, 'stageSeconds': new['stageSeconds'],
        'status': new['status'], 'selectedCandidateId': new['selectedCandidateId'],
        'referenceViewIndex': new['referenceViewIndex'], 'previousStatus': old['status'],
        'previousCandidateId': old['selectedCandidateId'], 'previousReferenceViewIndex': old['referenceViewIndex'],
        'sameDecision': new['status']==old['status'] and new['selectedCandidateId']==old['selectedCandidateId'],
        'verticesExactlyEqual': equality,
        'maximumVertexCoordinateDifference': float(np.max(np.abs(result['vertices']-geometry['after']))),
        'fastIntersections': fast_intersections, 'intersectionCounters': counts if fast_intersections else None,
        'allIntersectionCallsComparedToFull': audit_intersections,
        'otherViewProjectionDiagnostics': projection_rows, 'deployed': False}
    (destination/'fastProfile.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark-render', action='store_true')
    parser.add_argument('--all-views-reference', action='store_true')
    parser.add_argument('--benchmark-bake', choices=('highPitch4', 'cyanWall4', 'outdoorSideLight4'))
    parser.add_argument('--fast-jaw', choices=('all', 'currentMale', 'highPitch4', 'cyanWall4', 'outdoorSideLight4'))
    parser.add_argument('--fast-intersections', action='store_true')
    parser.add_argument('--warm-uv-cache', action='store_true')
    parser.add_argument('--audit-intersections', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output report')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {'scope': 'latency audit and isolated CPU postprocessing experiment',
        'fullGenerationMeasuredThisRun': False, 'deployed': False,
        'userTargetSeconds': [30, 60],
        'latestFullHttp': json.loads((ROOT/'reports/liveJaw20261012/summary.json').read_text()),
        'savedClientCases': report_rows(), 'renderBenchmarks': []}
    if args.benchmark_render:
        from faceAppearance.headAssembly import AssemblyAssets
        from faceAppearance.renderedJawRuntime import getDetector
        tick = perf_counter()
        assets = AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
        getDetector(ROOT)
        result['coldAssetAndFanLoadSeconds'] = perf_counter()-tick
        for row in result['savedClientCases']:
            measured = benchmark_render(row['source'], assets,
                all_views=args.all_views_reference)
            measured['caseId'] = row['caseId']
            result['renderBenchmarks'].append(measured)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps(measured, ensure_ascii=False), flush=True)
    if args.benchmark_bake:
        source = next(row['source'] for row in result['savedClientCases']
                      if row['caseId'] == args.benchmark_bake)
        result['bakeProfile'] = benchmark_bake(source, args.output.parent/(args.benchmark_bake+'BakeProfile'),
            warm_uv=args.warm_uv_cache)
    if args.fast_jaw:
        result['fastJaw'] = []
        for row in result['savedClientCases']:
            case = row['caseId']
            if args.fast_jaw not in ('all', case):
                continue
            photos = (ROOT/'runs/legacyPreviewJaw20261012/preview-609d6c9b63154933b7e3ecd46bb37719'
                      if case == 'currentMale' else ROOT/'runs/eyelidTextureBatch20261012'/case/'photos')
            measured = fast_jaw(row['source'], photos, args.output.parent/(case+'FastJaw'),
                fast_intersections=args.fast_intersections, audit_intersections=args.audit_intersections)
            measured['caseId'] = case
            result['fastJaw'].append(measured)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps(measured, ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(args.output, flush=True)


if __name__ == '__main__':
    main()
