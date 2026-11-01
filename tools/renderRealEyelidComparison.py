"""Render a local eyelid texture candidate against its exact frozen parent.

Uses saved subject cameras and independent high-resolution eye crops. No camera
fitting, geometry edits, or source-image resampling beyond display resizing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.legacyPreviewAsset import _load
from renderClientAppearanceEvidence import EvidenceRenderer, evidenceGrid, solid
from renderFeedbackAppearanceComparison import saveSrgb
from renderNoseRepairComparison import fileHash, run as renderComparison


def run(caseDirectory, candidateArchive, observationsPath, outputDirectory):
    case, candidate, observations, output = map(Path,
        (caseDirectory, candidateArchive, observationsPath, outputDirectory))
    shortId = case.name.split('-')[0]
    parentDirectory = ROOT / 'runs/feedbackFix20261024' / shortId / 'cavityGeneralizationCandidate'
    parent = parentDirectory / 'asset/legacyAsset.zip'
    parentReport = json.loads((parentDirectory / 'report.json').read_text())
    if fileHash(parent) != parentReport['variant']['sha256']:
        raise ValueError('Parent asset identity changed')
    before, after = _load(parent), _load(candidate)
    for key in before.mesh:
        np.testing.assert_array_equal(before.mesh[key], after.mesh[key])
    if before.manifest['binding'] != after.manifest['binding']:
        raise ValueError('Eyelid candidate changed binding')
    for name, data in before.payloads.items():
        if name not in ('head_color.png', 'face_nicp.png') and after.payloads[name] != data:
            raise ValueError(f'Protected asset payload changed: {name}')
    cameraPath = case / 'reconstruction/cameras.json'
    identity = fileHash(cameraPath)
    config = {'caseDirectory': str(case.resolve()), 'branches': [
        {'id': 'cavityBatch', 'label': '鼻腔版 · 原眼睑贴图',
         'archive': str(parent), 'archiveSha256': fileHash(parent),
         'cameraPath': str(cameraPath), 'cameraSha256': identity,
         'cachedRenderDirectory': str(ROOT / 'reports/noseGeneralization20261024' / shortId / 'render')},
        {'id': 'eyelidBatch', 'label': '眼睑局部取色 · 实验',
         'archive': str(candidate.resolve()), 'archiveSha256': fileHash(candidate),
         'cameraPath': str(cameraPath), 'cameraSha256': identity, 'appearanceFrozen': False}]}
    configPath = output.parent / (output.name + 'Config.json')
    if configPath.exists():
        raise FileExistsError(configPath)
    configPath.parent.mkdir(parents=True, exist_ok=True)
    writeJson(configPath, config)
    report = renderComparison(configPath, output)
    report['status'] = 'fullViewsCompleteEyeCropsPending'
    writeJson(output / 'report.json', report)
    with np.load(observations, allow_pickle=False) as arrays:
        curves = arrays['eyelidCurvesWorking'].copy()
    if curves.shape != (2, 2, 12, 2) or not np.isfinite(curves).all():
        raise ValueError('Expected two dense eye curves in working-photo coordinates')
    camera = json.loads(cameraPath.read_text())[0]
    boxes = [np.r_[np.floor(eye.reshape(-1, 2).min(0) - [16, 16]),
                   np.ceil(eye.reshape(-1, 2).max(0) + [16, 16])].astype(int) for eye in curves]
    boxes = np.asarray(boxes)
    roi = np.r_[boxes[:, :2].min(0), boxes[:, 2:].max(0)]
    factor = 4
    height, width = int((roi[3] - roi[1]) * factor), int((roi[2] - roi[0]) * factor)
    pixelTransform = np.array([[factor, 0, factor * (.5 - roi[0]) - .5],
                               [0, factor, factor * (.5 - roi[1]) - .5], [0, 0, 1]])
    zoom = {key: np.asarray(camera[key], float) for key in ('intrinsics', 'rotation', 'translation')}
    zoom['intrinsics'] = pixelTransform @ zoom['intrinsics']
    photoPath = case / 'reconstruction/input00.png'
    with Image.open(photoPath) as source:
        photo = source.convert('RGB').crop(tuple(roi)).resize((width, height), Image.Resampling.LANCZOS)
    similarity = before.manifest['assemblyDiagnostics']['templateToUff']
    images, records = {}, {}
    for branch, archive in [('cavityBatch', parent), ('eyelidBatch', candidate)]:
        renderer = EvidenceRenderer(archive, similarity, device='cpu', facesPerPixel=12)
        images[branch], records[branch] = renderer.variants(zoom, (height, width))
        for mode, value in images[branch].items():
            saveSrgb(solid(value), output / f'eyes_{branch}_{mode}.png')
    panels, labels = [], []
    for index, box in enumerate(boxes):
        crop = tuple(np.r_[(box[:2] - roi[:2]) * factor, (box[2:] - roi[:2]) * factor])
        for mode, modeLabel in [('baseColor', '基色'), ('lit', '共同检查光照')]:
            panels += [photo.crop(crop)] + [solid(images[key][mode]).crop(crop)
                                           for key in ('cavityBatch', 'eyelidBatch')]
            labels += [f'图{"左" if index == 0 else "右"}眼 · {name}' for name in
                       ('原照片', f'原贴图 · {modeLabel}', f'局部取色 · {modeLabel}')]
    saveSrgb(evidenceGrid(panels, labels, columns=3, panelSize=360,
        subtitle='原正面相机；几何和附件不变；局部取色属于估计外观'), output / 'eyeComparison.png')
    for row in config['branches']:
        if fileHash(row['archive']) != row['archiveSha256'] or fileHash(cameraPath) != identity:
            raise ValueError('Pinned render inputs changed during comparison')
    report.update(status='complete', geometryExactlyIdentical=True, referenceViewIndex=0,
                  qualityAccepted=False, finalCandidateAccepted=False,
                  eyeCrop={'roi': roi, 'scale': factor, 'boxes': boxes, 'camera': zoom,
                           'renderStats': records, 'observationsSha256': fileHash(observations),
                           'photoSha256': fileHash(photoPath)},
                  eyeComparisonSha256=fileHash(output / 'eyeComparison.png'))
    writeJson(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.case, args.candidate, args.observations, args.output)
