"""Compare UFF, fairing, normal priors and normal+VGGT depth on saved cases.

View 1 is excluded from local fitting only. The initial UFF mesh has seen every
view. This is not an end-to-end held-out reconstruction or a ground-truth score.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from PIL import Image, ImageDraw, ImageFont

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / 'src'), str(projectRoot / 'tools')]
from faceAppearance.artifacts import exportMesh, writeJson
from faceAppearance.browObservations import sampleNormalObservations, fuseNormalObservations, fuseDepthObservations
from faceAppearance.camera import projectPoints
from faceAppearance.semanticProjection import visibleVertices
from faceAppearance.texture import vertexNormals
from compareClientAssembly import loadPhotoReference
from renderClientAssembly import ClientPreviewRenderer
from runBrowRefinementExperiment import readArrays, fileHash, summary, clayAsset, cropReference, makeRow


branchLabels = {'baseline': '当前 UFF', 'fairing': '几何平滑（权重 64）',
                'normal': '法线约束', 'normalDepth': '法线＋VGGT 深度'}


def normalScores(vertices, faces, observations, regionWeights):
    normals = vertexNormals(vertices, faces)
    angles = np.degrees(np.arccos(np.clip((observations['normalsByView'] * normals[None]).sum(2), -1, 1)))
    scores = []
    for index, weights in enumerate(observations['weightsByView']):
        weights = weights * regionWeights
        mask = weights > 0
        scores.append({'viewIndex': index, 'role': 'localRefinementHeldOut' if index == 1 else 'fitting',
            **summary(angles[index, mask]),
            'weightedMeanDegrees': float(np.average(angles[index, mask], weights=weights[mask])) if mask.any() else None})
    return scores


def saveArrays(path, result):
    np.savez_compressed(path, **{key: value for key, value in result.items() if isinstance(value, np.ndarray)})


def renderComparisons(output, caseId, geometries, references, semantics, panelWidth):
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 18)
    renderers = {name: ClientPreviewRenderer(clayAsset(mesh['vertices'], mesh['faces']), device='cpu', facesPerPixel=2)
                 for name, mesh in geometries.items()}
    rows, reports = {}, {}
    for viewIndex, viewName in ((0, 'front'), (1, 'side')):
        reference = cropReference(references[viewIndex], semantics[viewIndex]['landmarks'], panelWidth)
        panels = [reference['photo']]
        for name, renderer in renderers.items():
            camera, check = renderer.photoCamera(reference['camera'], reference['imageShape'])
            rgb, visible, stats = renderer.render(camera, reference['imageShape'], [-.5, .6, 1.], perspective=True)
            if not visible.any():
                raise ValueError('Empty comparison render')
            panel = Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
            panel.save(output / f'{viewName}_{name}.png')
            panels.append(panel)
            reports[f'{viewName}/{name}'] = {'projectionCheck': check, 'raster': stats}
        title = f"{caseId} | {'正面' if viewIndex == 0 else '侧面（未参与本次局部拟合）'} | 同相机/同灯光灰模"
        row = makeRow(panels, ['原照片'] + [branchLabels[name] for name in geometries], title, font)
        row.save(output / f'{viewName}Comparison.png')
        rows[viewName] = row
    writeJson(output / 'renderReport.json', reports)
    return rows


def runCase(caseDir, cacheDir, output, *, panelWidth=380):
    from faceAppearance.browGeometryRefinement import refineBrowGeometry
    started = perf_counter()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    sourcePath = caseDir / 'appearance/geometryRefined.npz'
    source = readArrays(sourcePath)
    vertices, faces = source['vertices'], source['faces']
    regionPath = projectRoot / 'runs/browRefinementV1' / caseDir.name / 'region.npz'
    previousReport = json.loads((regionPath.parent / 'report.json').read_text(encoding='utf-8'))
    if previousReport['sourceSha256'] != fileHash(sourcePath):
        raise ValueError('Saved region belongs to different baseline geometry')
    region = readArrays(regionPath)
    weights = region['regionWeights']
    references = [loadPhotoReference(caseDir / 'appearance', index) for index in range(4)]
    for stage in ('normals', 'trace'):
        cacheReport = json.loads((cacheDir / stage / 'cacheReport.json').read_text(encoding='utf-8'))
        cachedInputs = cacheReport['originalImages']
        if cacheReport['status'] != 'completed' or len(cachedInputs) != len(references):
            raise ValueError('Incomplete or mismatched prior cache')
        for cached, reference in zip(cachedInputs, references):
            metadata = reference['metadata']
            if (cached['sha256'] != metadata['sha256']
                    or not np.array_equal(cached['inputToWorking'], metadata['inputToWorking'])):
                raise ValueError('Prior cache uses different photo order or pixel transform')
    cameras = [item['camera'] for item in references]
    semantics = [readArrays(caseDir / 'semantics' / f'view{index:02d}.npz') for index in range(4)]
    predictions = readArrays(cacheDir / 'normals/normals.npz')
    visibility = visibleVertices(vertices, faces, cameras, [item['imageShape'] for item in references],
                                 minimumCosine=.15, visibilityMode='exact')
    observations = sampleNormalObservations(vertices, faces, cameras,
        [item['metadata']['inputToWorking'] for item in references], predictions['normals'], predictions['validMask'],
        [item['labelMap'] for item in semantics], [item['confidence'] for item in semantics], visibility)
    targets = fuseNormalObservations(observations['normalsByView'], observations['weightsByView'], [0, 2, 3])
    saveArrays(output / 'normalObservations.npz', {**observations, **targets})
    angles = np.degrees(np.arccos(np.clip((observations['normalsByView'] * vertexNormals(vertices, faces)[None]).sum(2), -1, 1)))
    calibration = [summary(angles[index, row > 0]) for index, row in enumerate(observations['weightsByView'])]
    # A coarse sign/axis sanity gate only, not proof that predictions are true.
    if any(item.get('mean', 180.) > 60. for item in calibration):
        writeJson(output / 'axisSanityFailure.json', calibration)
        raise ValueError('Canonical normal disagreement exceeds coarse axis sanity gate')
    cameraCenter = -cameras[0]['translation'] @ cameras[0]['rotation']
    directions = vertices.astype(float) - cameraCenter
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    partials = readArrays(cacheDir / 'trace/fusion.npz')
    if (not np.array_equal(partials['selectedViewIndices'], np.arange(4))
            or not np.array_equal(partials['faces'], faces) or not np.array_equal(partials['uvs'], source['uvs'])):
        raise ValueError('Trace view indices/topology/UVs differ from baseline')
    # Independent visible skin support applies to depth as well as normal data.
    skinDepthValid = partials['partialValid'] & (observations['weightsByView'] > 0)
    confidenceThresholds = []
    for index, scores in enumerate(partials['partialDepthConfidence']):
        supported = skinDepthValid[index] & np.isfinite(scores)
        threshold = float(np.quantile(scores[supported], .25)) if supported.any() else None
        skinDepthValid[index] &= supported
        if threshold is not None:
            skinDepthValid[index] &= scores >= threshold
        confidenceThresholds.append(threshold)
    depth = fuseDepthObservations(vertices, directions, partials['partialVertices'], skinDepthValid, [0, 2, 3])
    depth['diagnostics']['depthConfidenceGate'] = {'rule': 'within-view visible-skin lower-quartile threshold; ties retained',
        'thresholds': confidenceThresholds, 'meaning': 'rank filter only; raw VGGT scores are not calibrated probabilities'}
    saveArrays(output / 'depthObservations.npz', depth)
    if ((targets['normalWeights'] > 0) & (weights > 0)).sum() < 15:
        raise ValueError('Insufficient trusted multi-view brow normal observations')
    branches = {'baseline': {key: source[key] for key in ('vertices', 'faces', 'uvs')},
                'fairing': readArrays(projectRoot / 'runs/browRefinementV1' / caseDir.name / 'medium/geometry.npz')}
    reports = {}
    for name in ('normal', 'normalDepth'):
        options = {} if name == 'normal' else {'depthTargets': depth['depthTargets'],
            'depthWeights': depth['depthWeights'] * weights, 'depthLossWeight': .03}
        result = refineBrowGeometry(vertices, faces, directions, weights,
            targets['normalTargets'], targets['normalWeights'] * weights, **options)
        changed = result['vertices']
        newVisibility = visibleVertices(changed, faces, [cameras[0]], [references[0]['imageShape']],
                                        minimumCosine=0., visibilityMode='exact')[0]
        lostSeedCount = int((region['eyebrowSeedMask'] & ~newVisibility).sum())
        if lostSeedCount:
            raise ValueError(f'{name}: {lostSeedCount} trusted front eyebrow seeds lost visibility')
        branch = {**branches['baseline'], 'vertices': changed}
        branches[name] = branch
        directory = output / name
        directory.mkdir()
        saveArrays(directory / 'geometry.npz', branch)
        exportMesh(directory, branch, includeMaterial=False)
        writeJson(directory / 'solverReport.json', result['diagnostics'])
        if 'history' in result:
            writeJson(directory / 'history.json', result['history'])
        reports[name] = {'solver': result['diagnostics'], 'newlyOccludedEyebrowSeeds': lostSeedCount}
    for name, branch in branches.items():
        changed = branch['vertices']
        if not np.array_equal(changed[weights == 0], vertices[weights == 0]):
            raise AssertionError('Outside region moved')
        projectionChanges = []
        for index, camera in enumerate(cameras):
            before, initialDepth = projectPoints(vertices, **camera)
            after, finalDepth = projectPoints(changed, **camera)
            if np.any((initialDepth > 0) & (finalDepth <= 0)):
                raise AssertionError('Camera depth flipped')
            drift = np.linalg.norm(after[weights > 0] - before[weights > 0], axis=1)
            if index == 0 and drift.max() > .002:
                raise AssertionError('Reference camera projection moved')
            projectionChanges.append({'viewIndex': index, **summary(drift)})
        scalar = np.sum((changed.astype(float) - vertices) * directions, axis=1)
        depthMask = (depth['depthWeights'] > 0) & (weights > 0)
        reports.setdefault(name, {}).update(normalPredictionAngles=normalScores(changed, faces, observations, weights),
            depthPriorResidualCanonical=summary(np.abs(scalar[depthMask] - depth['depthTargets'][depthMask])),
            projectionDriftOriginalPixels=projectionChanges,
            displacementPercentHeadSpan=summary(100 * np.linalg.norm(changed.astype(float) - vertices, axis=1)[weights > 0] / np.ptp(vertices, axis=0).max()),
            outsideRegionExactlyUnchanged=True, topologyAndUvExactlyUnchanged=True)
    rows = renderComparisons(output, caseDir.name, branches, references, semantics, panelWidth)
    report = {'status': 'completed', 'caseId': caseDir.name, 'sourceGeometry': str(sourcePath),
        'sourceSha256': fileHash(sourcePath), 'normalCacheSha256': fileHash(cacheDir / 'normals/normals.npz'),
        'fusionCacheSha256': fileHash(cacheDir / 'trace/fusion.npz'), 'regionSha256': fileHash(regionPath),
        'normalSampling': observations['diagnostics'], 'normalConsensus': targets['diagnostics'],
        'depthConsensus': depth['diagnostics'], 'wholeSkinNormalSanityDegrees': calibration,
        'selectedVertices': int((weights > 0).sum()),
        'supportedBrowNormalVertices': int(((targets['normalWeights'] > 0) & (weights > 0)).sum()),
        'supportedBrowDepthVertices': int(((depth['depthWeights'] > 0) & (weights > 0)).sum()),
        'branches': reports, 'seconds': perf_counter() - started,
        'limits': ['Held-out view 1 is excluded only from this local refinement; initial UFF already used all four images.',
            'Normal predictions and VGGT depth are learned priors, not independent 3D ground truth.',
            'Angles use fixed baseline projections/masks. Side projection drift is displacement, not error.',
            'Gray illumination is preset and unmatched to the photograph. No production defaults or client assets were overwritten.']}
    writeJson(output / 'report.json', report)
    print(json.dumps({'case': caseDir.name, 'seconds': report['seconds'], 'normalSupport': report['supportedBrowNormalVertices'],
                      'depthSupport': report['supportedBrowDepthVertices']}), flush=True)
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, default=projectRoot / 'runs/clientBatchV1')
    parser.add_argument('--cache', type=Path, default=projectRoot / 'runs/browNormalRefinementV1')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['windowLight4', 'diffuseLight4', 'backlightGlasses4', 'overheadHair4'])
    parser.add_argument('--panelWidth', type=int, default=380)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports, rows = {}, {'front': [], 'side': []}
    for caseId in args.cases:
        report, caseRows = runCase(args.batch / caseId, args.cache / caseId, args.output / caseId, panelWidth=args.panelWidth)
        reports[caseId] = report
        for name, row in caseRows.items():
            rows[name].append(row)
        writeJson(args.output / 'report.json', {'status': 'running', 'cases': reports})
    for name, panels in rows.items():
        overview = Image.new('RGB', (max(panel.width for panel in panels), sum(panel.height for panel in panels)), '#eef0f4')
        top = 0
        for panel in panels:
            overview.paste(panel, (0, top))
            top += panel.height
        overview.save(args.output / f'{name}Overview.png')
    writeJson(args.output / 'report.json', {'status': 'completed', 'cases': reports})


if __name__ == '__main__':
    main()
