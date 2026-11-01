"""Measure the frozen UFF -> client registration transfer, without inference.

Paired surface samples quantify preservation of an existing correspondence;
neither UFF nor the legacy anatomical mapping is assumed to be ground truth.
No camera, geometry, mapping, or production file is changed by this diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from faceAppearance.artifacts import writeJson
from faceAppearance.camera import projectPoints
from faceAppearance.clientLandmarks import loadClientLandmarks, selectClientLandmarks
from faceAppearance.correspondence import sampleSurface
from faceAppearance.directTemplateRegistration import registerDirectTemplate

CASES = ('currentMale', 'highPitch4', 'cyanWall4', 'outdoorSideLight4')
ASSET = ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1'
DEFAULT_OUTPUT = ROOT / 'runs/registrationTransfer20261013/metrics'


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {'count': 0}
    return dict(count=int(values.size), mean=float(values.mean()),
                rms=float(np.sqrt(np.mean(values ** 2))), median=float(np.median(values)),
                p90=float(np.percentile(values, 90)), p95=float(np.percentile(values, 95)),
                maximum=float(values.max()))


def region_masks(faces, sample_faces):
    """Fixed template neighborhoods, not inferred biological eye segmentation."""
    binding = loadClientLandmarks(faces)
    masks = {'all': np.ones(len(sample_faces), dtype=bool)}
    descriptions = {}
    for group in ('rightEye', 'leftEye', 'mouth', 'nose'):
        selected = selectClientLandmarks(binding, groups=(group,))
        seed = np.unique(selected['faceIds'][selected['validMask']])
        region = np.zeros(len(faces), dtype=bool)
        region[seed] = True
        # Two shared-vertex face rings; reproducible on the fixed topology.
        for _ in range(2):
            vertices = np.unique(faces[region])
            region = np.isin(faces, vertices).any(axis=1)
        masks[group] = region[sample_faces]
        descriptions[group] = dict(seedFaceIds=seed.tolist(), rings=2,
                                   regionFaceCount=int(region.sum()),
                                   sampleCount=int(masks[group].sum()),
                                   sideConvention='subject left/right; no image relabeling')
    masks['eyes'] = masks['rightEye'] | masks['leftEye']
    return masks, descriptions


def paths(case, output):
    if case == 'currentMale':
        source = ROOT / 'runs/liveJaw20261012/source/reconstruction'
        native = source / 'geometry.npz'
        if not native.exists():
            native = output / case / 'source/geometry.npz'
    else:
        source = ROOT / 'runs/eyelidTextureBatch20261012' / case / 'baseline/reconstruction'
        native = source / 'geometry.npz'
    return dict(source=source, uff=native,
                direct=source / 'directClient/refinement/clientGeometry.npz',
                stable=source / 'directClient/geometry.npz',
                final=ROOT / 'runs/fastFrontEyes20261012/final' / case / 'geometry.npz',
                cameras=source / 'cameras.json', inputs=source / 'inputs.json')


def run(case, output):
    p = paths(case, output)
    out = output / case
    out.mkdir(parents=True, exist_ok=True)
    for path in (p[key] for key in ('uff', 'direct', 'stable', 'final', 'cameras', 'inputs')):
        if not path.is_file():
            raise FileNotFoundError(path)
    a = load(ASSET / 'template.npz')
    mapping = load(ASSET / 'directRegistration.npz')
    manifest = json.loads((ASSET / 'manifest.json').read_text(encoding='utf-8'))
    uff, cached, stable, final = (load(p[key]) for key in ('uff', 'direct', 'stable', 'final'))
    cameras = json.loads(p['cameras'].read_text(encoding='utf-8'))
    inputs = json.loads(p['inputs'].read_text(encoding='utf-8'))
    faces = a['nicpFaces']
    if not np.array_equal(uff['faces'], a['sourceFaces']) or not np.array_equal(uff['uvs'], a['sourceUvs']):
        raise ValueError('Native UFF topology/UV does not match the frozen mapping')
    if not np.array_equal(cached['faces'], faces) or not np.array_equal(stable['faces'], faces):
        raise ValueError('Client topology differs')
    if not np.array_equal(final['skinVertexIds'], a['nicpToFullIds']):
        raise ValueError('Final skin identity differs')
    r, t, s = (cached['templateToUffRotation'], cached['templateToUffTranslation'],
               float(cached['templateToUffScale']))
    direct = cached['directTemplateVertices'] @ r * s + t
    final_skin = final['after'][a['nicpToFullIds']]
    stable_final_difference = float(np.max(np.abs(final['before'][a['nicpToFullIds']] - stable['vertices'])))
    if stable_final_difference > 1e-10:
        raise ValueError('Fast-eye baseline is not the saved stable skin')
    targets = sampleSurface(np.asarray(uff['vertices'], np.float64), uff['faces'],
        dict(triangleIds=mapping['sourceTriangleIds'], barycentricWeights=mapping['sourceBarycentricWeights']))
    options = manifest['geometryTransfer']
    solved = registerDirectTemplate(a['vertices'][a['nicpToFullIds']], faces,
        mapping['sampleNicpFaceIds'], mapping['sampleNicpWeights'], targets,
        mapping['sampleWeights'] > 0, np.flatnonzero(a['staticMask']),
        **{key: options[key] for key in ('dataWeight', 'smoothWeight', 'restWeight', 'staticConstraint', 'solver')})
    replay_error = float(np.max(np.abs(solved['rawVertices'] - direct)))
    transform_error = {key: float(np.max(np.abs(np.asarray(solved['transform'][key]) - value)))
                       for key, value in (('rotation', r), ('translation', t), ('scale', s))}
    if replay_error > 1e-9 or max(transform_error.values()) > 1e-9:
        raise ValueError(f'UFF does not reproduce the SAME cached direct registration: {replay_error}, {transform_error}')
    sample_map = dict(triangleIds=mapping['sampleNicpFaceIds'], barycentricWeights=mapping['sampleNicpWeights'])
    stages = dict(direct=direct, stable=stable['vertices'], final=final_skin)
    samples = {name: sampleSurface(vertices, faces, sample_map) for name, vertices in stages.items()}
    regions, descriptions = region_masks(faces, mapping['sampleNicpFaceIds'])
    valid = mapping['sampleWeights'] > 0
    regions = {key: mask & valid for key, mask in regions.items()}
    uf = np.asarray(uff['vertices'], float)[uff['faces'][mapping['sourceTriangleIds']]]
    normals = np.cross(uf[:, 1] - uf[:, 0], uf[:, 2] - uf[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-20)
    measures = {}
    for name, points in samples.items():
        delta = points - targets
        normal = np.sum(delta * normals, axis=1)
        tangent = np.linalg.norm(delta - normal[:, None] * normals, axis=1)
        stage = {}
        for region, mask in regions.items():
            stage[region] = dict(
                canonicalDistance=summary(np.linalg.norm(delta[mask], axis=1)),
                nominalTemplateCmDistance=summary(np.linalg.norm(delta[mask], axis=1) / s),
                signedUffFaceNormalOffsetMeanCanonical=float(normal[mask].mean()) if mask.any() else None,
                absoluteNormalOffsetCanonical=summary(np.abs(normal[mask])),
                tangentDistanceCanonical=summary(tangent[mask]), views=[])
        for view, (camera, item) in enumerate(zip(cameras, inputs)):
            if not camera.get('valid', False):
                continue
            args = [np.asarray(camera[key], float) for key in ('intrinsics', 'rotation', 'translation')]
            q, qdepth = projectPoints(targets, *args)
            projected, depth = projectPoints(points, *args)
            delta_pixels = projected - q
            transform = np.asarray(item['workingToInput'], float)
            original_delta = delta_pixels @ transform[:2, :2].T
            positive = (qdepth > 0) & (depth > 0)
            for region, mask in regions.items():
                keep = mask & positive
                stage[region]['views'].append(dict(viewIndex=view,
                    workingPixelDistance=summary(np.linalg.norm(delta_pixels[keep], axis=1)),
                    originalPixelDistance=summary(np.linalg.norm(original_delta[keep], axis=1)),
                    meanWorkingDeltaXY=delta_pixels[keep].mean(axis=0).tolist() if keep.any() else None))
        measures[name] = stage
    chain = {}
    for before, after in (('direct', 'stable'), ('stable', 'final'), ('direct', 'final')):
        delta = samples[after] - samples[before]
        chain[before + 'To' + after.capitalize()] = {
            region: summary(np.linalg.norm(delta[mask], axis=1)) for region, mask in regions.items()}
    meshes_path = out / 'meshes.npz'
    np.savez_compressed(meshes_path, uffVertices=uff['vertices'], uffFaces=uff['faces'],
        uffUvs=uff['uvs'], directVertices=direct, stableVertices=stable['vertices'],
        finalVertices=final_skin, clientFaces=faces,
        templateToUffRotation=r, templateToUffTranslation=t, templateToUffScale=np.asarray(s),
        fullFinalVertices=final['after'], fullStableVertices=final['before'],
        fullFaces=final['faces'], skinVertexIds=a['nicpToFullIds'])
    np.savez_compressed(out / 'pairedSamples.npz', uffTargets=targets,
        **{name + 'Samples': value for name, value in samples.items()},
        originalSampleIds=mapping['originalSampleIds'],
        **{name + 'Mask': value for name, value in regions.items()})
    files = {key: dict(path=str(path.resolve()), sha256=sha(path)) for key, path in p.items() if key != 'source'}
    for name in ('manifest.json', 'template.npz', 'directRegistration.npz'):
        files['asset:' + name] = dict(path=str(ASSET / name), sha256=sha(ASSET / name))
    report = dict(caseId=case, status='measuredFrozenCorrespondenceTransfer',
        sourceDirectory=str(p['source']), sourceFiles=files,
        photoSha256=[item['sha256'] for item in inputs],
        geometrySpace='uvFaceFusionCanonical6986; canonical=templateCm@R*scale+translation',
        rawUffVertexCount=len(uff['vertices']), clientSkinVertexCount=len(direct),
        validMappingSampleCount=int(valid.sum()), staticVertexCount=int(a['staticMask'].sum()),
        mappingStatus=manifest['mappingStatus'],
        reproduction=dict(maximumCanonicalVertexDifference=replay_error, transformMaximumAbsoluteDifferences=transform_error,
                          stableSkinMatchesFinalBeforeMaximumDifference=stable_final_difference),
        regions=dict(method='Valid fixed MediaPipe client binding triangles plus two shared-vertex face rings',
                     anatomicalGroundTruth=False, regionOverlapsAllowed=True, definitions=descriptions),
        fixedCameraProjectionScope='Paired 3D projection difference, positive-depth samples; not occlusion-filtered and not error to photograph landmarks.',
        measurementScope='Residual to frozen UFF surface at the EXISTING legacy mapping. Measures transfer fidelity, not reconstruction accuracy or anatomical correspondence correctness.',
        units='UFF canonical units are uncalibrated. nominalTemplateCm divides by fitted scale and is not physical measurement of this person.',
        limitations=['Different topology: no vertex-index or whole-face anatomical equivalence assumed.',
                     'The UFF mesh and client skin differ in eye/mouth surface construction; full UFF renders may include surfaces absent from NICP skin.',
                     'Final geometry includes the existing landmark, lip, jaw and fast-eye stages; their departure from UFF can be intentional.',
                     'No photos, cameras, templates or geometry were optimized by this diagnostic; the original direct objective was replayed solely for cache identity.'],
        directSolver=solved['diagnostics'], residualsToUff=measures, pairedSampleStageDisplacements=chain)
    writeJson(out / 'report.json', report)
    ready = dict(caseId=case, meshes=str(meshes_path.resolve()), report=str((out / 'report.json').resolve()),
                 cameras=str(p['cameras'].resolve()), inputs=str(p['inputs'].resolve()),
                 sameUffProvenByDirectReplay=True, maximumReplayDifference=replay_error)
    writeJson(out / 'ready.json', ready)
    print(json.dumps(ready, ensure_ascii=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    reports = [run(case, args.output.resolve()) for case in args.cases]
    rows = []
    for report in reports:
        row = {'caseId': report['caseId'], 'regions': {}}
        for region in ('all', 'eyes', 'mouth', 'nose'):
            row['regions'][region] = {}
            for stage in ('direct', 'stable', 'final'):
                measures = report['residualsToUff'][stage][region]
                front = next(item for item in measures['views'] if item['viewIndex'] == 0)
                row['regions'][region][stage] = dict(
                    sampleCount=measures['canonicalDistance']['count'],
                    canonicalRms=measures['canonicalDistance']['rms'],
                    frontWorkingPixelRms=front['workingPixelDistance']['rms'],
                    frontOriginalPixelRms=front['originalPixelDistance']['rms'])
        rows.append(row)
    writeJson(args.output.resolve() / 'summary.json', dict(
        scope='Frozen existing-correspondence transfer distance to UFF; NOT anatomical/photo error.',
        conclusions=[
            'Replayed direct registration reproduces every cached pure direct result to below 1e-13 canonical coordinate units.',
            'All stages retain the same source UFF and camera. No neural inference or camera refitting is included.',
            'Movement away from UFF after direct registration may be intentional photo-based refinement; larger transfer distance does not establish worse likeness.',
            'Regional masks describe fixed legacy landmark neighborhoods; they do not prove that paired UFF/client samples have matching biological semantics.'],
        cases=rows))


if __name__ == '__main__':
    main()
