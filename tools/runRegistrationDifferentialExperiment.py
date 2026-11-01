"""CPU-only, frozen-camera experiment for direct client registration.

Compares absolute smoothing with full or eye-neighborhood differential targets.
No photograph fitting, camera alignment, neural inference, asset export or
production configuration change occurs. UFF is a transfer reference, not truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.anatomicalBoundaries import extractMeshBoundaryLoops, validateBoundaryAsset
from faceAppearance.artifacts import writeJson
from faceAppearance.camera import projectPoints
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.correspondence import sampleSurface
from faceAppearance.directTemplateRegistration import registerDirectTemplate, _laplacian
from diagnoseRegistrationTransfer20261013 import load, sha, summary, region_masks
from evaluateRegistrationApertures20261013 import components, locate_eye
from renderClientAppearanceEvidence import evidenceGrid, solid
from renderRegistrationTransfer20261013 import SurfaceRenderer

ASSET = ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1'


def eyeRowWeights(vertices, faces):
    """Audited legacy male eye holes; geodesic graph rings, no photo selection."""
    extracted = extractMeshBoundaryLoops(vertices, faces)
    if extracted['diagnostics']['loopVertexCounts'] != [40, 10, 10]:
        raise ValueError('Expected audited legacy male skin boundary topology')
    loops = extracted['loops'][1:]
    _, adjacency = _laplacian(len(vertices), faces)
    distance = dijkstra(adjacency, directed=False, indices=np.concatenate(loops),
                        unweighted=True, min_only=True)
    # Full support at the hole and its immediate ring; cosine decay to zero
    # by four edges. These are Laplacian ROW weights, not deformation masks.
    fraction = np.clip((distance - 1.) / 3., 0., 1.)
    weights = .5 * (1 + np.cos(np.pi * fraction))
    return weights, loops, extracted['topologySha256']


def triangleSafety(before, after, faces, staticIds, basePairs, scale):
    a, b = before[faces], after[faces]
    na = np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
    nb = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
    aa, ab = np.linalg.norm(na, axis=1), np.linalg.norm(nb, axis=1)
    cosine = np.sum(na * nb, axis=1) / np.maximum(aa * ab, 1e-30)
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                               faces[:, [2, 0]]]), axis=1), axis=0)
    ea = np.linalg.norm(before[edges[:, 0]] - before[edges[:, 1]], axis=1)
    eb = np.linalg.norm(after[edges[:, 0]] - after[edges[:, 1]], axis=1)
    ratios = eb / np.maximum(ea, 1e-30)
    pairs = intersectionPairs(after, faces)
    return dict(displacementNominalTemplateMm=summary(np.linalg.norm(after - before, axis=1) / scale * 10),
        staticMaximumAbsoluteCanonical=float(np.abs(after[staticIds] - before[staticIds]).max()),
        normalReversalCount=int(np.count_nonzero(cosine <= 0)), minimumNormalCosine=float(cosine.min()),
        areaRatioMinimum=float((ab / np.maximum(aa, 1e-30)).min()),
        edgeRatioMinimum=float(ratios.min()), edgeRatioMaximum=float(ratios.max()),
        intersectionPairCount=len(pairs), newIntersectionPairs=[list(x) for x in sorted(pairs - basePairs)],
        removedIntersectionPairs=[list(x) for x in sorted(basePairs - pairs)],
        sharedVertexPairsExcluded=True, collisionFreeCertification=False)


def projectLoop(vertices, loops, camera):
    projected = [projectPoints(vertices[ids], *(camera[k] for k in
                 ('intrinsics', 'rotation', 'translation')))[0] for ids in loops]
    return sorted(projected, key=lambda points: points[:, 0].mean())


def loopMeasures(polygons):
    return [dict(projectedWidth=float(np.ptp(p[:, 0])), projectedHeight=float(np.ptp(p[:, 1])),
                 projectedPolygonArea=float(abs(np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                                                      - p[:, 1] * np.roll(p[:, 0], -1))) / 2),
                 projectedPoints=p.tolist()) for p in polygons]


def run(case, output, front, side, size, render):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    reconstruction = case / 'reconstruction'
    arrays, mapping = load(ASSET / 'template.npz'), load(ASSET / 'directRegistration.npz')
    manifest = json.loads((ASSET / 'manifest.json').read_text())
    uff = load(reconstruction / 'geometry.npz')
    cached = load(reconstruction / 'directClient/refinement/clientGeometry.npz')
    if not np.array_equal(uff['faces'], arrays['sourceFaces']):
        raise ValueError('UFF topology differs from frozen mapping')
    rest, faces = arrays['vertices'][arrays['nicpToFullIds']], arrays['nicpFaces']
    static = np.flatnonzero(arrays['staticMask'])
    valid = mapping['sampleWeights'] > 0
    targetMap = dict(triangleIds=mapping['sourceTriangleIds'],
                     barycentricWeights=mapping['sourceBarycentricWeights'])
    clientMap = dict(triangleIds=mapping['sampleNicpFaceIds'],
                     barycentricWeights=mapping['sampleNicpWeights'])
    targets = sampleSurface(uff['vertices'].astype(float), uff['faces'], targetMap)
    neutralTargets = sampleSurface(arrays['sourceVertices'].astype(float), arrays['sourceFaces'], targetMap)
    options = {key: manifest['geometryTransfer'][key] for key in
               ('dataWeight', 'smoothWeight', 'restWeight', 'staticConstraint', 'solver')}

    def solve(observations, **extra):
        return registerDirectTemplate(rest, faces, mapping['sampleNicpFaceIds'], mapping['sampleNicpWeights'],
                                      observations, valid, static, **options, **extra)

    baseline = solve(targets)
    transform = baseline['transform']
    rotation, scale, translation = (transform[key] for key in ('rotation', 'scale', 'translation'))
    expected = (cached['directTemplateVertices'] @ cached['templateToUffRotation']
                * cached['templateToUffScale'] + cached['templateToUffTranslation'])
    replayError = float(np.abs(expected - baseline['rawVertices']).max())
    if replayError > 1e-9:
        raise ValueError(f'Baseline does not reproduce saved pure registration: {replayError}')
    rowWeights, eyeLoops, topologyHash = eyeRowWeights(rest, faces)
    solved = dict(baseline=baseline,
        differentialFull=solve(targets, smoothnessMode='displacement'),
        differentialEyes=solve(targets, smoothnessMode='displacement', differentialWeights=rowWeights))
    neutral = solve(neutralTargets)
    neutralDifferential = solve(neutralTargets, smoothnessMode='displacement')
    neutralTransform = neutral['transform']
    neutralDrift = ((neutral['rawVertices'] - neutral['alignedRest'])
                    @ neutralTransform['rotation'].T / neutralTransform['scale'])
    neutralDifferentialDrift = ((neutralDifferential['rawVertices'] - neutralDifferential['alignedRest'])
                    @ neutralTransform['rotation'].T / neutralTransform['scale'])
    personDrift = (baseline['rawVertices'] - baseline['alignedRest']) @ rotation.T / scale
    correlation = float(np.sum(personDrift * neutralDrift) /
                        max(np.linalg.norm(personDrift) * np.linalg.norm(neutralDrift), 1e-30))
    regions, regionDescriptions = region_masks(faces, mapping['sampleNicpFaceIds'])
    basePairs = intersectionPairs(baseline['rawVertices'], faces)
    metrics = {}
    for name, result in solved.items():
        vertices = result['rawVertices']
        residual = sampleSurface(vertices, faces, clientMap) - targets
        metrics[name] = dict(solver=result['diagnostics'],
            safety=triangleSafety(baseline['rawVertices'], vertices, faces, static, basePairs, scale),
            transferResidualNominalTemplateMm={region: summary(np.linalg.norm(residual[mask & valid], axis=1) / scale * 10)
                                               for region, mask in regions.items()})
        print(json.dumps({'case': case.name, 'candidate': name, 'safety': metrics[name]['safety']}), flush=True)

    # A line search changes a single global weight, never clips individual
    # vertices. Record all rejected raw candidates and the conservative trial.
    raw = solved['differentialEyes']['rawVertices']
    trials = []
    for amount in (1., .5, .25, .125):
        candidate = baseline['rawVertices'] + amount * (raw - baseline['rawVertices'])
        safety = (metrics['differentialEyes']['safety'] if amount == 1. else
                  triangleSafety(baseline['rawVertices'], candidate, faces, static, basePairs, scale))
        accepted = (not safety['newIntersectionPairs'] and safety['normalReversalCount'] == 0
                    and safety['edgeRatioMinimum'] >= .7 and safety['edgeRatioMaximum'] <= 1.3
                    and safety['displacementNominalTemplateMm']['maximum'] <= 2.)
        trials.append(dict(amount=amount, numericalGatePassed=accepted, safety=safety))
        if accepted:
            break
    chosen = trials[-1]['amount'] if trials[-1]['numericalGatePassed'] else 0.
    chosenVertices = baseline['rawVertices'] + chosen * (raw - baseline['rawVertices'])
    stages = {name: value['rawVertices'] for name, value in solved.items()}
    stages['guardedEyes'] = chosenVertices
    np.savez_compressed(output / 'meshes.npz', **{name + 'Vertices': v for name, v in stages.items()},
        uffVertices=uff['vertices'], uffFaces=uff['faces'], clientFaces=faces,
        alignedRest=baseline['alignedRest'], differentialEyeRowWeights=rowWeights,
        neutralBaselineVertices=neutral['rawVertices'], neutralAlignedRest=neutral['alignedRest'],
        neutralDifferentialVertices=neutralDifferential['rawVertices'],
        neutralDriftTemplateCm=neutralDrift, templateToUffRotation=rotation,
        templateToUffScale=np.asarray(scale), templateToUffTranslation=translation)
    report = dict(caseId=case.name, baselineReplayMaximumCanonical=replayError, options=options,
        topologySha256=topologyHash, vertexOrderAndFacesPreserved=True,
        neutral=dict(source='Frozen sourceVertices from template.npz, not an observed neutral person',
            absoluteDriftNominalTemplateMm=summary(np.linalg.norm(neutralDrift, axis=1) * 10),
            fullDifferentialDriftNominalTemplateMm=summary(np.linalg.norm(neutralDifferentialDrift, axis=1) * 10),
            personToNeutralDriftVectorCosine=correlation,
            personMinusNeutralDriftNominalTemplateMm=summary(np.linalg.norm(personDrift - neutralDrift, axis=1) * 10),
            interpretation='Nonzero neutral drift combines mapping/rest anatomy mismatch and smoothing. It is not person-specific identity; subtraction is not automatically anatomically correct.'),
        candidates=metrics, lineSearch=dict(chosenAmount=chosen, trials=trials,
            scope='Numerical regression gate only; not photo likeness, complete assembly or rig validation'),
        regions=regionDescriptions,
        sourceFiles={str(p): sha(p) for p in [ASSET / 'template.npz', ASSET / 'directRegistration.npz',
            ASSET / 'manifest.json', reconstruction / 'geometry.npz', reconstruction / 'cameras.json',
            reconstruction / 'directClient/refinement/clientGeometry.npz', Path(__file__),
            ROOT / 'src/faceAppearance/directTemplateRegistration.py']},
        limitations=['Source UFF is not true anatomy or a photo landmark target.',
            'Differential targets preserve client-template curvature, potentially retaining template identity bias.',
            'Candidate is pure registration only; later landmark, lip, jaw, accessory assembly and texture stages are not replayed.',
            'Intersection gate excludes shared-vertex neighbors and is not a complete collision certificate.',
            'Nominal template millimeters are normalized by the fitted similarity, not physical measurement of this person.'])
    writeJson(output / 'report.json', report)
    if render:
        cameras = json.loads((reconstruction / 'cameras.json').read_text())
        inputs = json.loads((reconstruction / 'inputs.json').read_text())
        sourceBoundaries = validateBoundaryAsset(uff['vertices'], uff['faces'],
            json.loads((ROOT / 'assets/anatomical/uff6986/boundaries.json').read_text()))
        sourceEyes = [sourceBoundaries[name] for name in ('positiveXEye', 'negativeXEye')]
        renderStages = dict(uff=uff['vertices'], **stages)
        renderers = {name: SurfaceRenderer(v, uff['faces'] if name == 'uff' else faces, device='cpu')
                     for name, v in renderStages.items()}
        report['renders'] = []
        for view in dict.fromkeys((front, side)):
            camera = cameras[view]
            if not camera.get('valid'):
                raise ValueError('Invalid saved camera')
            width, height = inputs[view]['workingSize']
            sx, sy = size / width, size / height
            pixelTransform = np.array([[sx, 0, .5 * sx - .5], [0, sy, .5 * sy - .5], [0, 0, 1.]])
            camera = {key: np.asarray(camera[key], float) for key in ('intrinsics', 'rotation', 'translation')}
            camera['intrinsics'] = pixelTransform @ camera['intrinsics']
            with Image.open(reconstruction / f'input{view:02d}.png') as image:
                photo = image.convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
            panels, labels = [photo], [f'照片 #{view + 1}']
            polygons = projectLoop(uff['vertices'], sourceEyes, camera)
            record = dict(viewIndex=view, camera={key: value.tolist() for key, value in camera.items()},
                          referenceEyePolygons='Projected UFF topological holes; not photo observations', stages={})
            for name, renderer in renderers.items():
                variants, stats = renderer.variants(camera, (size, size))
                clay = variants['clay']
                clay.save(output / f'view{view:02d}_{name}.png')
                panels.append(solid(clay)); labels.append(name)
                stageRecord = dict(renderer=stats, structuralEyeLoops=loopMeasures(projectLoop(
                    renderStages[name], sourceEyes if name == 'uff' else eyeLoops, camera)))
                if view == front:
                    available, _ = components(np.asarray(clay)[..., 3])
                    openings = []
                    for eye in range(2):
                        selected, diagnostic = locate_eye(available, polygons, eye)
                        openings.append(dict(selection=diagnostic, geometry=None if selected is None else
                                             {key: value for key, value in selected.items() if key != 'mask'}))
                    stageRecord['visibleSkinOpenings'] = openings
                record['stages'][name] = stageRecord
            evidenceGrid(panels, labels, columns=len(panels), panelSize=size,
                subtitle=f'{case.name[:8]} | 同任务相机/光照；纯配准皮肤，不含眼球；未重拟合相机').save(output / f'view{view:02d}Comparison.png')
            # The crop is identical across columns, preserving all registration
            # misalignment. It enlarges visibility only, never geometry.
            allPoints = np.concatenate(polygons)
            lo, hi = allPoints.min(0), allPoints.max(0)
            span = hi - lo
            box = (int(lo[0] - .12 * span[0]), int(lo[1] - .7 * span[1]),
                   int(hi[0] + .12 * span[0]), int(hi[1] + .7 * span[1]))
            evidenceGrid([image.crop(box) for image in panels], labels, columns=len(panels), panelSize=384,
                subtitle=f'{case.name[:8]} | 同一裁剪；灰模洞口不是完整装配眼裂').save(output / f'view{view:02d}Eyes.png')
            report['renders'].append(record)
            writeJson(output / 'report.json', report)
            print(json.dumps({'case': case.name, 'renderedView': view, 'chosenAmount': chosen}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--front', type=int, required=True)
    parser.add_argument('--side', type=int, required=True)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--skipRender', action='store_true')
    args = parser.parse_args()
    run(args.case.resolve(), args.output.resolve(), args.front, args.side, args.size, not args.skipRender)
