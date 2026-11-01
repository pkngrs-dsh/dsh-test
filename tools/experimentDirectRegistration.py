"""Compare direct client-surface registration using saved UFF reconstructions.

All branches share one UFF input and one display frame per case. Exact quadratic
solutions evaluate the old objective; they are not the original 600-step trace.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from faceAppearance.correspondence import sampleSurface
from faceAppearance.headAssembly import AssemblyAssets, _triangleQuality
from faceAppearance.directTemplateRegistration import registerDirectTemplate
from renderLegacyMouthAudit import render


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def writeJson(path, value):
    path.write_text(json.dumps(value, default=serial, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def inverse(points, transform):
    return ((points - transform['translation']) @ transform['rotation'].T) / transform['scale']


def geometryMetrics(rest, vertices, faces, mouthMask):
    before, after = rest[faces], vertices[faces]
    a = np.cross(before[:, 1] - before[:, 0], before[:, 2] - before[:, 0])
    b = np.cross(after[:, 1] - after[:, 0], after[:, 2] - after[:, 0])
    lengths = np.linalg.norm(b, axis=1)
    ratios = lengths / np.maximum(np.linalg.norm(a, axis=1), 1e-20)
    mouthFaces = mouthMask[faces].all(axis=1)
    result = _triangleQuality(rest, vertices, faces)
    result.update({'collapsedTriangleCount': int((lengths < 1e-12).sum()),
                   'areaRatioP01': float(np.percentile(ratios, 1)),
                   'areaRatioP99': float(np.percentile(ratios, 99)),
                   'mouthInvertedFaceCount': int(((np.sum(a*b, axis=1) <= 0) & mouthFaces).sum()),
                   'mouthTriangleCount': int(mouthFaces.sum()),
                   'scope': 'Orientation relative to aligned rest template; not a global self-intersection test'})
    return result


def makeGrid(path, branches, bounds, title, panelSize=440):
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    panels = [render(np.array(b['vertices']), np.array(b['faces']), bounds, panelSize) for b in branches]
    w, h = panels[0].size
    image = Image.new('RGB', (w*len(panels), h+70), (28, 35, 39))
    draw = ImageDraw.Draw(image)
    draw.text((12, 8), title, font=font, fill='white')
    for i, (panel, branch) in enumerate(zip(panels, branches)):
        image.paste(panel, (i*w, 70))
        draw.text((i*w+12, 40), branch['label'], font=font, fill='white')
    image.save(path)


def runCase(case, output, priors, solver='exact'):
    directory = output / case['id']
    directory.mkdir(parents=True, exist_ok=True)
    gender = case.get('gender') or case['suggestedComparisonTemplateGender']
    geometryPath = Path(case['geometryPath'])
    with np.load(geometryPath, allow_pickle=False) as stored:
        geometry = {key: stored[key] for key in ('vertices', 'faces', 'uvs')}
    base = AssemblyAssets(ROOT / f'assets/clientTemplates/legacy{gender.title()}V1')
    bundle = ROOT / f'assets/clientTemplates/legacy{gender.title()}DirectV1'
    with np.load(bundle / 'directRegistration.npz', allow_pickle=False) as saved:
        mapping = {key: saved[key] for key in saved.files}
    baseline, _, baselineDiag = base._deform(geometry)
    arrays = base.arrays
    ids, faces = arrays['nicpToFullIds'], arrays['nicpFaces']
    rest = arrays['vertices'][ids]
    transform = baselineDiag['templateToUff']
    source = inverse(geometry['vertices'], transform)
    sampleTargets = sampleSurface(geometry['vertices'], geometry['faces'],
            {'triangleIds': mapping['sourceTriangleIds'],
             'barycentricWeights': mapping['sourceBarycentricWeights']})
    mouthCenter = rest[[168, 1143]].mean(axis=0)
    mouthMask = ((np.abs(rest[:, 0] - mouthCenter[0]) < 3.5)
                 & (np.abs(rest[:, 1] - mouthCenter[1]) < 2.0)
                 & (rest[:, 2] > mouthCenter[2] - 2.0))
    branches = [
        {'id': 'source', 'label': 'UFF source', 'vertices': source, 'faces': geometry['faces'], 'metrics': {}},
        {'id': 'baseline', 'label': 'A: deployed UV transfer', 'vertices': baseline['vertices'][ids], 'faces': faces,
         'metrics': {'quality': geometryMetrics(rest, baseline['vertices'][ids], faces, mouthMask),
                     'guardedVertexCount': baselineDiag['guardedVertexCount']}}]
    arraysOut = {'source': source, 'sourceFaces': geometry['faces'], 'rest': rest,
                 'faces': faces, 'baseline': baseline['vertices'][ids]}
    records = []
    for name, prior, boundary, smooth in priors:
        started = perf_counter()
        result = registerDirectTemplate(rest, faces, mapping['sampleNicpFaceIds'],
                 mapping['sampleNicpWeights'], sampleTargets, mapping['sampleWeights'] > 0,
                 np.flatnonzero(arrays['staticMask']), restWeight=prior,
                 staticConstraint=boundary, smoothWeight=smooth, solver=solver)
        elapsed = perf_counter() - started
        vertices = inverse(result['rawVertices'], transform)
        alignedRest = inverse(result['alignedRest'], transform)
        quality = geometryMetrics(alignedRest, vertices, faces, mouthMask)
        staticDelta = np.linalg.norm(vertices - alignedRest, axis=1)[arrays['staticMask']]
        metrics = {'quality': quality, 'solver': result['diagnostics'], 'elapsedSeconds': elapsed,
                   'staticDriftMaxMm': float(staticDelta.max()*10),
                   'restWeight': prior, 'staticConstraint': boundary, 'smoothWeight': smooth,
                   'postSolveVertexGuardApplied': False}
        labels = {'legacyPrior': 'B: old prior 1000', 'weakPrior': 'C: weak prior 100', 'noPrior': 'D: no shape prior',
                  'noPriorHard': 'E: no prior, fixed boundary', 'noPriorHardSmooth300': 'F: E + lighter smoothing'}
        branches.append({'id': name, 'label': labels.get(name, name), 'vertices': vertices, 'faces': faces, 'metrics': metrics})
        arraysOut[name] = vertices
        writeJson(directory / f'{name}.json', {'metrics': metrics, 'templateToUff': result['transform']})
        records.append({'id': name, **metrics})
    np.savez_compressed(directory / 'meshes.npz', **arraysOut)
    preview = {'caseId': case['id'], 'gender': gender, 'coordinateFrame': 'shared baseline template centimeters',
               'sourceGeometrySha256': hashlib.sha256(geometryPath.read_bytes()).hexdigest(),
               'mouthCenter': mouthCenter, 'branches': branches,
               'displayNote': 'Same input, camera and light. Skin only; no textures or accessories. No candidate-specific display alignment.'}
    # Compact mesh arrays keep the interactive payload small.
    (directory / 'preview.json').write_text(json.dumps(preview, default=serial, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
    mx, my, _ = mouthCenter
    makeGrid(directory / 'mouth.png', branches, (mx-4, mx+4, my-3, my+3), case['id']+' | shared camera, geometry only')
    selected = [b for b in branches if b['id'] in ('source', 'baseline', 'noPriorHard', 'noPriorHardSmooth300')]
    makeGrid(directory / 'mouthSelected.png', selected, (mx-4, mx+4, my-3, my+3), case['id']+' | shared camera, geometry only', 550)
    midx = float((source[:, 0].min()+source[:, 0].max())*.5)
    ymin, ymax = float(source[:, 1].min()), float(source[:, 1].max())
    makeGrid(directory / 'face.png', branches, (midx-8, midx+8, ymin-.5, ymax+.5), case['id']+' | shared camera, geometry only')
    return {'id': case['id'], 'gender': gender, 'captureMode': case.get('captureMode'),
            'geometryPath': str(geometryPath), 'sourceGeometrySha256': preview['sourceGeometrySha256'],
            'previewUrl': f"/data/{case['id']}/preview.json", 'mouthImage': f"/data/{case['id']}/mouth.png",
            'branches': records, 'baseline': branches[1]['metrics']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT/'reports/directRegistration20261012/cases.json')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/directRegistration20261012')
    parser.add_argument('--case', action='append')
    parser.add_argument('--solver', choices=['exact', 'legacyAdamCpu'], default='exact')
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())['cases']
    if args.case:
        cases = [case for case in cases if case['id'] in args.case]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'local_experiment_not_approved_for_deployment',
              'objective': 'Legacy point/soft-anchor MSE + absolute uniform Laplacian + optional rest-shape MSE',
              'solver': ('Exact quadratic minimum, not reproduction of the original 600-step optimizer trajectory'
                         if args.solver == 'exact' else 'CPU float32 replay of the old 600-step algorithm, not CUDA bitwise identity'),
              'cases': []}
    for case in cases:
        print('Running '+case['id'], flush=True)
        priors = [('legacyPrior',1000.,'soft',3000.), ('weakPrior',100.,'soft',3000.),('noPrior',0.,'soft',3000.)]
        if args.solver == 'exact':
            priors += [('noPriorHard',0.,'hard',3000.),('noPriorHardSmooth300',0.,'hard',300.)]
        report['cases'].append(runCase(case, args.output, priors, args.solver))
        writeJson(args.output/'experiment.json', report)
        print(json.dumps({'case':case['id'], 'branches':[(b['id'],b['quality']['invertedFaceCount'],b['quality']['mouthInvertedFaceCount']) for b in report['cases'][-1]['branches']]}), flush=True)


if __name__ == '__main__':
    main()
