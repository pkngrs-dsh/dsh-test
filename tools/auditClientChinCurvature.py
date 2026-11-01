"""Compare a frozen local edge set in experimental chin geometries.

The angle statistic describes this mesh's local crease change; it is neither an
anatomical ground truth nor a universal quality score. No geometry is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(vertices, camera):
    q = vertices @ np.asarray(camera['rotation']).T + np.asarray(camera['translation'])
    p = q @ np.asarray(camera['intrinsics']).T
    return p[:, :2] / p[:, 2:], q[:, 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('runs/clientChinRefine20261012/geometry/clientGeometry.npz'))
    parser.add_argument('--cameras', type=Path, default=Path('runs/clientAppearanceFix20261012/deployedSource/cameras.json'))
    parser.add_argument('--candidate', action='append', default=[], help='LABEL:NPZ (after, or vertices array)')
    parser.add_argument('--output', type=Path, default=Path('reports/clientChinCurvatureAudit20261012.json'))
    args = parser.parse_args()
    data = np.load(args.reference, allow_pickle=False)
    before, faces = data['before'].astype(float), data['faces']
    cameras = json.loads(args.cameras.read_text())
    incident = {}
    for fi, face in enumerate(faces):
        for a, b in zip(face, np.roll(face, -1)):
            incident.setdefault(tuple(sorted((int(a), int(b)))), []).append(fi)
    manifold = [(edge, face_ids) for edge, face_ids in sorted(incident.items()) if len(face_ids) == 2]
    edges = np.array([item[0] for item in manifold])
    adjacent = np.array([item[1] for item in manifold])
    midpoint = before[edges].mean(axis=1)
    front, depth = project(midpoint, cameras[0])
    # Frozen from the original mesh, so all candidates use exactly these edges.
    roi = (front[:, 0] >= 185) & (front[:, 0] <= 335) & (front[:, 1] >= 400) & (front[:, 1] <= 480) & (depth < 4.65)
    edges, adjacent, front = edges[roi], adjacent[roi], front[roi]
    lengths = np.linalg.norm(before[edges[:, 1]] - before[edges[:, 0]], axis=1)

    def measure(vertices):
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-30)
        angle = np.rad2deg(np.arccos(np.clip(np.sum(normals[adjacent[:, 0]] * normals[adjacent[:, 1]], axis=1), -1, 1)))
        up, _ = project(vertices[edges].mean(axis=1), cameras[3])
        return angle, up

    baseline, up_before = measure(before)
    report = {
        'method': 'frozenChinRoiAdjacentFaceUnsignedNormalAngle',
        'reference': str(args.reference), 'referenceSha256': sha(args.reference),
        'cameras': str(args.cameras), 'camerasSha256': sha(args.cameras),
        'facesArraySha256': hashlib.sha256(np.ascontiguousarray(faces).tobytes()).hexdigest(),
        'roi': {'selection': 'baseline edge midpoint in front image rectangle, positive near-face depth',
                'frontXyxy': [185, 400, 335, 480], 'maxCameraDepth': 4.65, 'edgeCount': len(edges),
                'edgeVertexIds': edges.tolist(), 'adjacentFaceIds': adjacent.tolist(),
                'beforeFrontPixels': front.tolist(), 'beforeUpwardPixels': up_before.tolist()},
        'limitations': [
            'ROI is chosen for this existing sample and is frozen in topology for comparison.',
            'No visibility filtering: lower chin and nearby underside edges can be included.',
            'Larger local angles can reflect intended anatomy as well as unwanted folds; inspect renders.',
            'The 10 degree increase count is descriptive, not a pass/fail threshold.',
            'Estimated cameras define only the ROI; face angles are measured in 3D.'],
        'candidates': {}}
    candidates = [('before', before, str(args.reference), 'before'), ('narrow', data['after'], str(args.reference), 'after')]
    for item in args.candidate:
        label, path = item.split(':', 1)
        candidate = np.load(path, allow_pickle=False)
        if 'faces' in candidate and not np.array_equal(faces, candidate['faces']):
            raise ValueError('Candidate topology differs: ' + path)
        key = 'after' if 'after' in candidate else 'vertices'
        candidates.append((label, candidate[key], path, key))
    for label, vertices, path, key in candidates:
        if vertices.shape != before.shape:
            raise ValueError('Vertex shape differs: ' + path)
        angle, up = measure(vertices)
        delta = angle - baseline
        top = np.argsort(delta)[::-1][:20]
        report['candidates'][label] = {
            'source': path, 'sha256': sha(path), 'arrayKey': key,
            'meanAngleDegrees': float(angle.mean()), 'lengthWeightedMeanAngleDegrees': float(np.average(angle, weights=lengths)),
            'p95AngleDegrees': float(np.percentile(angle, 95)), 'maxAngleDegrees': float(angle.max()),
            'p95PositiveIncreaseDegrees': float(np.percentile(np.maximum(delta, 0), 95)),
            'maxIncreaseDegrees': float(delta.max()), 'increasedOver10DegreesCount': int((delta > 10).sum()),
            'angleDegreesPerFrozenEdge': angle.tolist(),
            'topIncreases': [{'edgeVertexIds': edges[i].tolist(), 'adjacentFaceIds': adjacent[i].tolist(),
                'beforeAngleDegrees': float(baseline[i]), 'afterAngleDegrees': float(angle[i]),
                'increaseDegrees': float(delta[i]), 'beforeUpwardPixels': up_before[i].tolist(),
                'afterUpwardPixels': up[i].tolist()} for i in top]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'roiEdgeCount': len(edges), 'candidates': {
        label: {k: v for k, v in record.items() if k.endswith('Degrees') or k.endswith('Count')}
        for label, record in report['candidates'].items()}}, indent=2))


if __name__ == '__main__':
    main()
