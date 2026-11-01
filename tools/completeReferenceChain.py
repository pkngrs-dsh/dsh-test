"""Bounded local replay experiment; no reconstruction, baking or service calls.

Combine a saved three-dimensional registration candidate with eye/mouth changes
freshly computed on that candidate. A fixed, short backtracking list enforces
the original cumulative shape limits. Replay rendered-jaw gates, then check the
complete chain against the original direct registration again.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.clientReferenceView import selectClientReference
from faceAppearance.clientRenderedJaw import refineRenderedClientJaw
from faceAppearance.frontLandmarkRefinement import checkFrontGeometry
from faceAppearance.headAssembly import AssemblyAssets
from renderClientAppearanceEvidence import evidenceGrid, solid
from renderRegistrationTransfer20261013 import SurfaceRenderer


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def run(case, replay, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = {}

    def source(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    reconstruction = case / 'reconstruction'
    cameras = json.loads(source(reconstruction / 'cameras.json').read_text())
    inputs = json.loads(source(reconstruction / 'inputs.json').read_text())
    manifest = json.loads(source(case / 'asset/legacyManifest.json').read_text())
    assets = AssemblyAssets(ROOT / 'assets/clientTemplates' / manifest['templateId'])
    if assets.manifest['templateHashes'] != manifest['templateHashes']:
        raise ValueError('Cached task template identity differs from local template')
    original = load(source(reconstruction / 'directClient/refinement/clientGeometry.npz'))
    refined = load(source(replay / 'selected/clientGeometry.npz'))
    upstream = json.loads(source(replay / 'selected/geometryRefinement.json').read_text())
    observationsPath = reconstruction / 'directClient/renderedJaw/photoObservations'
    observations = load(source(observationsPath.with_suffix('.npz')))
    observations['diagnostics'] = json.loads(source(observationsPath.with_suffix('.json')).read_text())
    selection = selectClientReference(cameras, inputs, observations)
    reference = selection['referenceViewIndex']
    if reference is None or upstream['referenceViewIndex'] != reference:
        raise ValueError('Selected reference must match the fresh downstream replay')
    faces = refined['faces']
    if not np.array_equal(faces, original['faces']) or not np.array_equal(faces, assets.arrays['nicpFaces']):
        raise ValueError('Cached stages differ in client topology')
    transform = {'rotation': refined['templateToUffRotation'],
                 'translation': refined['templateToUffTranslation'], 'scale': float(refined['templateToUffScale'])}
    for name in ('Rotation', 'Translation', 'Scale'):
        if not np.array_equal(original['templateToUff' + name], refined['templateToUff' + name]):
            raise ValueError('Cached stages differ in canonical similarity')
    toCanonical = lambda vertices: vertices @ transform['rotation'] * transform['scale'] + transform['translation']
    baseline = toCanonical(original['directTemplateVertices'])
    boundary = toCanonical(refined['directTemplateVertices'])
    fixed = np.flatnonzero(assets.arrays['staticMask'])
    camera = {key: np.asarray(cameras[reference][key], float) for key in ('intrinsics', 'rotation', 'translation')}
    extent = float(np.ptp(baseline, axis=0).max())
    boundaryDepthBudget = float(np.abs((boundary - baseline) @ camera['rotation'][2]).max()) + extent * 1e-9
    originalPairs = intersectionPairs(baseline, faces)

    def chainCheck(vertices):
        report = checkFrontGeometry(baseline, vertices, faces, camera=camera,
            fixedVertexIds=fixed, maxDisplacement=extent * .07, maxRelativeEdgeChange=.25,
            minAreaRatio=.4, minNormalCosine=.3, depthTolerance=boundaryDepthBudget)
        report['newIntersectionPairsAgainstOriginalDirect'] = len(intersectionPairs(vertices, faces) - originalPairs)
        report['accepted'] = report['accepted'] and report['newIntersectionPairsAgainstOriginalDirect'] == 0
        return report

    attempts, beforeJaw, chosenAmount = [], None, None
    for amount in (1., .5, .25, .125, 0.):
        candidate = boundary + amount * (refined['vertices'] - boundary)
        candidate[fixed] = baseline[fixed]
        check = chainCheck(candidate)
        frontOnly = checkFrontGeometry(boundary, candidate, faces, camera=camera,
            fixedVertexIds=fixed, maxDisplacement=extent * .07, maxRelativeEdgeChange=.25,
            minAreaRatio=.4, minNormalCosine=.3)
        attempts.append({'amount': amount, 'cumulative': check, 'eyeMouthDepthPreserved': frontOnly})
        if check['accepted'] and frontOnly['accepted']:
            beforeJaw, chosenAmount = candidate, amount
            break
    writeJson(output / 'backtracking.json', {'attempts': attempts, 'selectedAmount': chosenAmount,
        'depthBudgetMeaning': 'Only the already computed 3D registration boundary change; eye/mouth increments preserve camera depth',
        'depthBudgetCanonical': boundaryDepthBudget, 'policy': 'Fixed list 1, .5, .25, .125, 0; first complete safe chain'})
    if beforeJaw is None:
        raise RuntimeError('No safe combination; retain separate experiments')
    print(json.dumps({'selectedEyeMouthAmount': chosenAmount}), flush=True)
    jaw = refineRenderedClientJaw(assets, beforeJaw, faces, transform, cameras, inputs,
        [None] * len(inputs), projectRoot=ROOT, outputDirectory=output / 'renderedJaw',
        referenceViewIndex=reference, photoObservations=observations)
    final = jaw['vertices']
    jointJaw = chainCheck(final)
    rejectedJaw = not jointJaw['accepted']
    if rejectedJaw:
        final = beforeJaw.copy()
    finalSafety = chainCheck(final)
    if not finalSafety['accepted']:
        raise RuntimeError('Final chain failed safety after jaw fallback')
    lookup = {tuple(face): index for index, face in enumerate(assets.arrays['faces'])}
    faceIds = np.array([lookup[tuple(face)] for face in assets.arrays['nicpToFullIds'][faces]])
    geometry = {'vertices': final, 'faces': faces, 'uvs': assets.arrays['uvs'],
                'uvFaces': assets.arrays['uvFaces'][faceIds]}
    np.savez_compressed(output / 'finalGeometry.npz', **geometry)
    np.savez_compressed(output / 'templateToUff.npz', **{key: np.asarray(value) for key, value in transform.items()})
    np.savez_compressed(output / 'accessoryReference.npz', vertices=beforeJaw)
    mouth = load(source(replay / 'selected/mouthState.npz'))
    mouth['afterCanonical'] = final.copy()
    np.savez_compressed(output / 'mouthState.npz', **mouth)
    np.savez_compressed(output / 'stages.npz', baseline=baseline, boundary=boundary,
        beforeBacktracking=refined['vertices'], beforeJaw=beforeJaw, final=final, faces=faces)
    report = {'case': case.name, 'referenceSelection': selection, 'selectedEyeMouthAmount': chosenAmount,
        'upstreamFreshRefinement': upstream, 'backtracking': attempts,
        'jointJawCheck': jointJaw, 'jawRejectedByJointGate': rejectedJaw,
        'jaw': jaw['diagnostics'], 'finalSafety': finalSafety,
        'depthBudgetCanonical': boundaryDepthBudget,
        'templateToUff': {key: np.asarray(value).tolist() for key, value in transform.items()},
        'sourceHashes': sources, 'implementationSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations': ['Local experiment; no production default changed.',
            'Fresh eye/mouth refinement is computed on boundary candidate, not copied from production final geometry.',
            'Full-stage amount was reduced only by predeclared safety backtracking; no parameter sweep or photo optimization.',
            'Original cached STFR landmarks, cached ingestion pixels for lip trace; no original photo inference.',
            'No texture baking or ZIP export; final surface requires rebaking before delivery.']}
    writeJson(output / 'report.json', report)
    size = inputs[reference]['workingSize'][0]
    with Image.open(source(reconstruction / f'input{reference:02d}.png')) as image:
        photo = image.convert('RGB')
    panels, labels = [photo], [f'照片 view {reference}']
    comparison = [('productionEyeMouth', original['vertices'], '生产眼嘴后'),
                  ('boundary', boundary, '新边界纯配准'), ('final', final, '边界+统一参考+全链门控')]
    for name, vertices, label in comparison:
        variants, _ = SurfaceRenderer(vertices, faces, device='cpu').variants(camera, photo.size[::-1])
        variants['clay'].save(output / f'{name}.png')
        panels.append(solid(variants['clay']))
        labels.append(label)
    evidenceGrid(panels, labels, columns=4, panelSize=size,
        subtitle='原相机；灰模皮肤不含附件；3D边界+新refinement，最终累计安全门控').save(output / 'frontComparison.png')
    print(json.dumps({'finalSafetyAccepted': finalSafety['accepted'], 'jawStatus': jaw['diagnostics']['status'],
        'finalGeometry': str(output / 'finalGeometry.npz')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.case.resolve(), args.replay.resolve(), args.output.resolve())
