"""Freeze conservative render/photo jaw bindings for the six-case experiment."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.renderedJawCorrespondence import POLICY, prepareRenderedJawCorrespondence
from runAdaptiveChinBatch import archive, serial


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(serial(value), ensure_ascii=False, indent=2), encoding='utf-8')


def prepare_case(case, evaluation, render_report, directory):
    start = perf_counter()
    for key, hash_key in (("geometryPath", "geometrySha256"), ("facesPath", "facesSha256"),
            ("cameraPath", "cameraSha256"), ("inputsPath", "inputsSha256"), ("templateArrayPath", "templateSha256")):
        if case.get(hash_key) and sha(case[key]) != case[hash_key]:
            raise ValueError(f'Source changed: {case["caseId"]} {key}')
    for image in case['sourceImages']:
        if sha(image['path']) != image['sha256']: raise ValueError('Source photograph changed')
    vertices = archive(case['geometryPath'])[case['verticesKey']]
    faces = archive(case['facesPath'])[case['facesKey']]
    template = archive(case['templateArrayPath'])
    if not np.array_equal(faces, template['nicpFaces']): raise ValueError('Ordered template topology mismatch')
    rest = template['vertices'][template['nicpToFullIds']]
    photo = archive(case['observationPath'])
    cameras = json.loads(Path(case['cameraPath']).read_text())
    inputs = json.loads(Path(case['inputsPath']).read_text())
    image_shapes = [record['workingSize'][::-1] for record in inputs]
    fixed = np.flatnonzero(template[case['fixedMaskKey']])
    protected = np.unique(archive(case['protectedBindingsPath'])['vertexIndices'])
    renders, source_hashes = [], {}
    for view_index in range(len(cameras)):
        variants = {}
        for name in POLICY['requiredAppearances']:
            rows = [row for row in render_report['rows'] if row['caseId'] == case['caseId']
                    and row['stage'] == 'before' and row['viewIndex'] == view_index and row['variant'] == name]
            if len(rows) != 1: raise ValueError('Missing or duplicate frozen rendering')
            row = rows[0]
            if row['sourcePhotoObservationSha256'] != sha(case['observationPath']):
                raise ValueError('Photo observations changed after render experiment')
            if row['sourceRenderSha256'] != sha(row['sourceRenderPath']):
                raise ValueError('Rendered image changed after detection')
            path = evaluation/row['npzPath']
            data = archive(path)
            if not np.array_equal(data['photoPoints'], photo['faceLandmarksWorking'][view_index]):
                raise ValueError('Cached photo coordinates mismatch')
            if not np.array_equal(data['photoValid'], photo['landmarkCandidateValidMask'][view_index]):
                raise ValueError('Cached photo eligibility mismatch')
            variants[name] = data
            source_hashes[str(path.resolve())] = sha(path)
            source_hashes[row['sourceRenderPath']] = row['sourceRenderSha256']
        renders.append({'viewIndex': view_index, 'variants': variants})
    result = prepareRenderedJawCorrespondence(vertices, faces, rest, cameras, photo, renders,
        imageShapes=image_shapes, fixedVertexIds=fixed, protectedVertexIds=protected)
    result.update(caseId=case['caseId'], sourceHashes={**source_hashes,
        **{case[key]: sha(case[key]) for key in ('geometryPath', 'facesPath', 'cameraPath', 'inputsPath',
            'templateArrayPath', 'observationPath', 'protectedBindingsPath')}},
        implementationSha256=sha(ROOT/'src/faceAppearance/renderedJawCorrespondence.py'),
        seconds=perf_counter()-start, geometryModified=False, remoteModified=False)
    directory.mkdir(parents=True, exist_ok=True)
    write(directory/'correspondence.json', result)
    arrays = {key: value for key, value in result['domain'].items() if isinstance(value, np.ndarray)}
    for view in result['views']:
        arrays.update({f'view{view["viewIndex"]}_{key}': value for key, value in view.items() if isinstance(value, np.ndarray)})
    np.savez_compressed(directory/'correspondence.npz', **arrays)
    rejections = Counter(reason for view in result['diagnostics'] for point in view['landmarks'] for reason in point['reasons'])
    return {'caseId': case['caseId'], 'seconds': result['seconds'],
        'correspondencePath': str((directory/'correspondence.json').resolve()),
        'acceptedCounts': [len(view['semanticIds']) for view in result['views']],
        'acceptedSemanticIds': [view['semanticIds'] for view in result['views']],
        'eligibleViews': [view['viewIndex'] for view in result['views'] if view['eligible']],
        'anatomicalDomainVertexCount': int(result['domain']['vertexMask'].sum()),
        'rejectionReasonCounts': dict(rejections)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT/'runs/clientChinAdaptive20261012/cases.json')
    parser.add_argument('--evaluation', type=Path, default=ROOT/'runs/clientRenderedSemantics20261012/evaluation')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/clientRenderedJawFit20261012/correspondences')
    args = parser.parse_args()
    manifest = json.loads(args.cases.read_text())
    render_report = json.loads((args.evaluation/'report.json').read_text())
    if render_report['caseManifestSha256'] != sha(args.cases): raise ValueError('Case manifest changed since rendering')
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {'schemaVersion': 1, 'status': 'running', 'rows': [], 'policy': POLICY,
        'implementationSha256': sha(ROOT/'src/faceAppearance/renderedJawCorrespondence.py'),
        'runnerSha256': sha(__file__), 'caseManifestSha256': sha(args.cases),
        'renderReportSha256': sha(args.evaluation/'report.json'), 'geometryModified': False, 'remoteModified': False}
    write(args.output/'summary.json', summary)
    for case in manifest['cases']:
        row = prepare_case(case, args.evaluation, render_report, args.output/case['caseId'])
        summary['rows'].append(row)
        write(args.output/'summary.json', summary)
        print(json.dumps(serial(row), ensure_ascii=False), flush=True)
    summary['status'] = 'completed'
    summary['acceptedCount'] = sum(sum(row['acceptedCounts']) for row in summary['rows'])
    summary['candidateCount'] = len(manifest['cases'])*4*len(POLICY['semanticIds'])
    write(args.output/'summary.json', summary)


if __name__ == '__main__':
    main()
