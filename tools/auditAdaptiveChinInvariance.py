"""Reproduce bounded order/unit/image-scale checks without changing the solver.

Landmark predictions are transformed mathematically; no detector, images,
reconstruction, textures, or cameras are refitted by this experiment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from faceAppearance.adaptiveChinRefinement import refineAdaptiveChin
from faceAppearance.artifacts import writeJson
from faceAppearance.headAssembly import AssemblyAssets


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected(result):
    candidate = result['diagnostics'].get('selectedCandidate', {})
    return {key: candidate.get(key) for key in ('smoothing', 'amplitudeFraction', 'amplitudePixels')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'runs/clientChinAdaptive20261012/cases.json')
    parser.add_argument('--evaluation', type=Path, default=ROOT/'runs/clientChinAdaptive20261012/evaluationV1')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/clientChinAdaptive20261012/invarianceReport.json')
    parser.add_argument('--case', action='append', dest='cases')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the existing invariance report; use a new output path')
    frozen = json.loads((args.evaluation/'summary.json').read_text())['solverSha256']
    solver = ROOT/'src/faceAppearance/adaptiveChinRefinement.py'
    if sha(solver) != frozen:
        raise RuntimeError('Current solver differs from the frozen evaluated solver')
    manifest = json.loads(args.manifest.read_text())
    case_ids = args.cases or ['maleCurrent', 'outdoorSideLight4']
    cases = [case for case in manifest['cases'] if case['caseId'] in case_ids]
    if len(cases) != len(set(case_ids)):
        raise ValueError('Unknown case ID')
    report = {'status': 'running', 'solverSha256': frozen, 'manifestSha256': sha(args.manifest),
              'method': 'mathematical input transformations, same frozen function and detector observations',
              'tolerancePolicy': {'maximumAbsoluteCanonicalCoordinateErrorPerOriginalExtent': 2e-8,
                                  'sameStatusAndMappedReferenceRequired': True},
              'cases': [], 'remoteModified': False,
              'limitations': ['These checks establish numerical invariance for the selected inputs, not face-reconstruction accuracy.',
                              'Resolution scaling transforms saved landmark predictions; it does not measure detector resolution behavior.']}
    for case in cases:
        geometry = archive(case['geometryPath'])
        vertices = geometry[case.get('verticesKey', 'vertices')]
        faces = archive(case['facesPath'])[case.get('facesKey', 'faces')]
        cameras = json.loads(Path(case['cameraPath']).read_text())
        n = len(cameras)
        observations = archive(case['observationPath'])
        inputs = json.loads(Path(case['inputsPath']).read_text())
        observations['workingImageShape'] = np.array(inputs[0]['workingSize'][::-1])
        binding = archive(case['protectedBindingsPath'])
        assets = AssemblyAssets(Path(case['templatePath']))
        fixed = np.flatnonzero(assets.arrays[case.get('fixedMaskKey', 'staticMask')])
        protected = np.unique(binding['vertexIndices'])
        options = dict(fixedVertexIds=fixed, protectedVertexIds=protected, landmarkBinding=binding)
        start = perf_counter()
        baseline = refineAdaptiveChin(vertices, faces, cameras, observations, **options)
        baseline_seconds = perf_counter()-start
        saved = archive(args.evaluation/case['caseId']/'geometry.npz')['after']
        baseline_error = float(np.max(np.abs(saved-baseline['vertices'])))
        extent = float(np.ptp(vertices, axis=0).max())
        tolerance = extent*2e-8
        baseline_reference = baseline['diagnostics'].get('referenceViewIndex')
        row = {'caseId': case['caseId'], 'viewCount': n,
               'baseline': {'seconds': baseline_seconds, 'status': baseline['diagnostics']['status'],
                            'referenceViewIndex': baseline_reference, 'selectedCandidate': selected(baseline),
                            'maximumAbsoluteErrorAgainstFrozenBatch': baseline_error},
               'originalCanonicalExtent': extent, 'absoluteCanonicalTolerance': tolerance,
               'experiments': []}
        report['cases'].append(row)
        for experiment in ('reverse_view_order', 'canonical_units_x10', 'working_pixels_x2'):
            current_vertices = np.array(vertices, copy=True)
            current_cameras = copy.deepcopy(cameras)
            current_observations = {name: np.array(value, copy=True) for name, value in observations.items()}
            ordering = np.arange(n)
            output_divisor = 1.
            if experiment == 'reverse_view_order':
                ordering = ordering[::-1]
                current_cameras = [current_cameras[index] for index in ordering]
                for name, value in current_observations.items():
                    if value.ndim and value.shape[0] == n and name != 'workingImageShape':
                        current_observations[name] = value[ordering].copy()
                description = 'Reverse cameras and every per-view observation array; semantic mesh bindings remain fixed.'
            elif experiment == 'canonical_units_x10':
                current_vertices *= 10.
                for camera in current_cameras:
                    camera['translation'] = (np.asarray(camera['translation'])*10.).tolist()
                output_divisor = 10.
                description = 'Multiply canonical vertex positions and all camera translations by10; compare returned vertices divided by10.'
            else:
                for camera in current_cameras:
                    intrinsics = np.asarray(camera['intrinsics'], float)
                    intrinsics[:2] *= 2.
                    camera['intrinsics'] = intrinsics.tolist()
                    for name in ('imageShape', 'imageSize'):
                        if name in camera:
                            camera[name] = (np.asarray(camera[name])*2).tolist()
                for name in ('faceLandmarksWorking', 'cropVariantLandmarksWorking',
                             'maximumCropVariationPixels', 'faceScaleWorkingPixels', 'workingImageShape'):
                    if name in current_observations:
                        current_observations[name] = current_observations[name]*2
                description = 'Multiply saved working landmark coordinates, crop variability, image shape and first two intrinsic rows by2; original-photo coordinates remain unchanged.'
            start = perf_counter()
            transformed = refineAdaptiveChin(current_vertices, faces, current_cameras, current_observations, **options)
            elapsed = perf_counter()-start
            expected_reference = (None if baseline_reference is None else int(np.flatnonzero(ordering == baseline_reference)[0]))
            returned_reference = transformed['diagnostics'].get('referenceViewIndex')
            compared = transformed['vertices']/output_divisor
            differences = compared-baseline['vertices']
            maximum_error = float(np.max(np.abs(differences)))
            maximum_vertex_error = float(np.linalg.norm(differences, axis=1).max())
            same_status = transformed['diagnostics']['status'] == baseline['diagnostics']['status']
            same_reference = returned_reference == expected_reference
            item = {'transformation': experiment, 'description': description, 'seconds': elapsed,
                    'inputPositionToOriginalViewIndex': ordering.tolist(),
                    'status': transformed['diagnostics']['status'], 'referenceViewIndex': returned_reference,
                    'expectedReferenceViewIndex': expected_reference,
                    'sameStatus': same_status, 'sameMappedReference': same_reference,
                    'selectedCandidate': selected(transformed),
                    'displayMetrics': transformed['diagnostics'].get('displayMetrics'),
                    'maximumAbsoluteCanonicalCoordinateError': maximum_error,
                    'maximumCanonicalVertexDistance': maximum_vertex_error,
                    'maximumAbsoluteErrorPerOriginalExtent': maximum_error/extent,
                    'passed': bool(same_status and same_reference and maximum_error <= tolerance)}
            row['experiments'].append(item)
            if sha(solver) != frozen:
                raise RuntimeError('Frozen solver changed during invariance audit')
            print(json.dumps({'caseId': case['caseId'], **item}), flush=True)
        row['passed'] = baseline_error <= tolerance and all(item['passed'] for item in row['experiments'])
    report['status'] = 'complete'
    report['allPassed'] = all(case['passed'] for case in report['cases'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writeJson(args.output, report)
    print(json.dumps({'output': str(args.output), 'allPassed': report['allPassed']}), flush=True)


if __name__ == '__main__':
    main()
