"""Frozen UFF eye-arc constraints added to the existing registration optimum.

The baseline's absolute smoothing/data objective, its fixed alignment and all
static anchors are retained. This diagnostic changes no production setting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.anatomicalBoundaries import validateBoundaryAsset
from faceAppearance.artifacts import writeJson
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.correspondence import sampleSurface
from faceAppearance.directTemplateRegistration import _laplacian
from faceAppearance.registrationBoundaryRefinement import buildEyeBoundaryTargets, solveBoundaryIncrement
from diagnoseRegistrationTransfer20261013 import load, summary, sha
from runRegistrationDifferentialExperiment import (ASSET, eyeRowWeights, triangleSafety,
                                                  projectLoop, loopMeasures)
from evaluateRegistrationApertures20261013 import components, locate_eye
from renderRegistrationTransfer20261013 import SurfaceRenderer
from renderClientAppearanceEvidence import evidenceGrid, solid



def run(case, previous, output, front, side, size):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    arrays, mapping = load(ASSET / 'template.npz'), load(ASSET / 'directRegistration.npz')
    cached = load(previous / 'meshes.npz')
    previousReport = json.loads((previous / 'report.json').read_text())
    if previousReport['baselineReplayMaximumCanonical'] > 1e-9:
        raise ValueError('Unverified direct registration replay')
    faces, baseline = cached['clientFaces'], cached['baselineVertices']
    n, scale = len(baseline), float(cached['templateToUffScale'])
    static = np.flatnonzero(arrays['staticMask'])
    valid = mapping['sampleWeights'] > 0
    triangleIds, bary = mapping['sampleNicpFaceIds'][valid], mapping['sampleNicpWeights'][valid]
    sampleCount = len(triangleIds)
    sample = sparse.csr_matrix((bary.ravel(),
        (np.repeat(np.arange(sampleCount), 3), faces[triangleIds].ravel())), shape=(sampleCount, n))
    anchors = sparse.csr_matrix((np.ones(len(static)), (np.arange(len(static)), static)), shape=(len(static), n))
    design = sparse.vstack([sample, anchors], format='csr')
    laplacian, _ = _laplacian(n, faces)
    options = previousReport['options']
    if options['solver'] != 'exact' or options['staticConstraint'] != 'hard':
        raise ValueError('Experiment requires verified hard-anchor exact baseline')
    operator = (options['dataWeight'] / (3 * (sampleCount + len(static))) * (design.T @ design)
                + options['smoothWeight'] / (3 * n) * (laplacian.T @ laplacian)
                + options['restWeight'] / (3 * n) * sparse.eye(n)).tocsc()
    _, clientLoops, topologyHash = eyeRowWeights(arrays['vertices'][arrays['nicpToFullIds']], faces)
    named = validateBoundaryAsset(arrays['sourceVertices'], arrays['sourceFaces'],
        json.loads((ROOT / 'assets/anatomical/uff6986/boundaries.json').read_text()))
    sourceLoops = [named[name] for name in ('negativeXEye', 'positiveXEye')]
    boundaryIds, targets, definitions = buildEyeBoundaryTargets(cached['neutralBaselineVertices'], clientLoops,
        arrays['sourceVertices'], sourceLoops, cached['uffVertices'])
    if np.intersect1d(static, boundaryIds).size:
        raise ValueError('Eye boundary must not include fixed interface vertices')
    pairs = intersectionPairs(baseline, faces)
    stages = dict(baseline=baseline)
    metrics = {}
    for weight in (100., 1000.):
        name = f'boundary{int(weight)}'
        vertices = solveBoundaryIncrement(baseline, operator, static, boundaryIds, targets, weight)
        safety = triangleSafety(baseline, vertices, faces, static, pairs, scale)
        stages[name] = vertices
        metrics[name] = dict(weight=weight, safety=safety,
            boundaryResidualNominalTemplateMm=summary(np.linalg.norm(vertices[boundaryIds] - targets, axis=1) / scale * 10))
        print(json.dumps({'case': case.name, 'candidate': name, 'maxMoveMm': safety['displacementNominalTemplateMm']['maximum'],
            'flips': safety['normalReversalCount'], 'newPairs': len(safety['newIntersectionPairs']),
            'edgeRange': [safety['edgeRatioMinimum'], safety['edgeRatioMaximum']]}), flush=True)
    raw = stages['boundary1000']
    trials = []
    for amount in (1., .5, .25, .125, .0625):
        candidate = baseline + amount * (raw - baseline)
        safety = (metrics['boundary1000']['safety'] if amount == 1 else
                  triangleSafety(baseline, candidate, faces, static, pairs, scale))
        passed = (not safety['newIntersectionPairs'] and safety['normalReversalCount'] == 0
                  and safety['edgeRatioMinimum'] >= .7 and safety['edgeRatioMaximum'] <= 1.3
                  and safety['displacementNominalTemplateMm']['maximum'] <= 2.)
        trials.append(dict(amount=amount, numericalGatePassed=passed, safety=safety))
        if passed:
            break
    amount = trials[-1]['amount'] if trials[-1]['numericalGatePassed'] else 0.
    stages['guardedBoundary'] = baseline + amount * (raw - baseline)
    for name, vertices in stages.items():
        metrics.setdefault(name, {})['boundaryResidualNominalTemplateMm'] = summary(
            np.linalg.norm(vertices[boundaryIds] - targets, axis=1) / scale * 10)
    np.savez_compressed(output / 'meshes.npz', **{name + 'Vertices': value for name, value in stages.items()},
        clientFaces=faces, uffVertices=cached['uffVertices'], uffFaces=cached['uffFaces'],
        boundaryVertexIds=boundaryIds, boundaryTargetPoints=targets,
        templateToUffRotation=cached['templateToUffRotation'], templateToUffScale=cached['templateToUffScale'],
        templateToUffTranslation=cached['templateToUffTranslation'])
    report = dict(caseId=case.name, objective='Baseline absolute quadratic plus boundaryWeight*mean((V[eyeIds]-sourceArcTargets)**2)',
        baselineReplayMaximumCanonical=previousReport['baselineReplayMaximumCanonical'],
        alignmentUnchanged=True, staticAnchorsExact=True, topologySha256=topologyHash,
        arcDefinitions=definitions, targetDepthPreserved=True, metrics=metrics,
        lineSearch=dict(chosenAmount=amount, trials=trials),
        sourceFiles={str(p): sha(p) for p in [previous / 'meshes.npz', previous / 'report.json', Path(__file__)]},
        limitations=['UFF eye boundaries are a reconstruction reference, not true photo anatomy.',
            'Client skin rim has 10 vertices per eye and UFF has 60; piecewise linear arc sampling cannot add topology detail.',
            'Skin-only aperture is not final visible eye opening after accessory assembly.',
            'Boundary meaning is inherited from reviewed topology assets; neutral reference fixes corner IDs, not a new per-photo detector.',
            'Pure registration only; postprocessing, texture and full accessory assembly not replayed.',
            'Gate uses baseline-relative intersection pair sets; existing intersections remain and shared-vertex faces are excluded.'])
    writeJson(output / 'report.json', report)
    reconstruction = case / 'reconstruction'
    cameras = json.loads((reconstruction / 'cameras.json').read_text())
    inputs = json.loads((reconstruction / 'inputs.json').read_text())
    stages = dict(uff=cached['uffVertices'], **stages)
    renderers = {name: SurfaceRenderer(v, cached['uffFaces'] if name == 'uff' else faces, device='cpu')
                 for name, v in stages.items()}
    report['renders'] = []
    for view in dict.fromkeys((front, side)):
        if not cameras[view].get('valid'):
            raise ValueError('Invalid original camera')
        width, height = inputs[view]['workingSize']
        sx, sy = size / width, size / height
        resize = np.array([[sx, 0, .5 * sx - .5], [0, sy, .5 * sy - .5], [0, 0, 1.]])
        camera = {k: np.asarray(cameras[view][k], float) for k in ('intrinsics', 'rotation', 'translation')}
        camera['intrinsics'] = resize @ camera['intrinsics']
        with Image.open(reconstruction / f'input{view:02d}.png') as image:
            photo = image.convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
        panels, labels = [photo], ['照片']
        polygons = projectLoop(cached['uffVertices'], sourceLoops, camera)
        record = dict(viewIndex=view, camera={k: v.tolist() for k, v in camera.items()}, stages={})
        for name, renderer in renderers.items():
            variants, stats = renderer.variants(camera, (size, size))
            clay = variants['clay']; clay.save(output / f'view{view:02d}_{name}.png')
            panels.append(solid(clay)); labels.append(name)
            projected = projectLoop(stages[name], sourceLoops if name == 'uff' else clientLoops, camera)
            stageRecord = dict(renderer=stats, structuralEyeLoops=loopMeasures(projected))
            if view == front:
                available, _ = components(np.asarray(clay)[..., 3]); opening = []
                for eye in range(2):
                    selected, diagnostic = locate_eye(available, polygons, eye)
                    opening.append(dict(selection=diagnostic, geometry=None if selected is None else
                        {key: value for key, value in selected.items() if key != 'mask'}))
                stageRecord['visibleSkinOpenings'] = opening
            record['stages'][name] = stageRecord
        evidenceGrid(panels, labels, columns=len(panels), panelSize=size,
            subtitle=f'{case.name[:8]} | 固定任务相机与原配准；UFF 三维边界约束').save(output / f'view{view:02d}Comparison.png')
        points = np.concatenate(polygons); lo, hi = points.min(0), points.max(0); span = hi - lo
        box = (int(lo[0] - .12 * span[0]), int(lo[1] - .7 * span[1]), int(hi[0] + .12 * span[0]), int(hi[1] + .7 * span[1]))
        evidenceGrid([panel.crop(box) for panel in panels], labels, columns=len(panels), panelSize=384,
            subtitle=f'{case.name[:8]} | 同一眼周裁剪；皮肤洞口不含眼球').save(output / f'view{view:02d}Eyes.png')
        report['renders'].append(record); writeJson(output / 'report.json', report)
        print(json.dumps({'case': case.name, 'view': view, 'chosenAmount': amount}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--front', type=int, required=True)
    parser.add_argument('--side', type=int, required=True)
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args()
    run(args.case.resolve(), args.previous.resolve(), args.output.resolve(), args.front, args.side, args.size)
