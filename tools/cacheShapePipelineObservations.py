"""Freeze one common CPU current-photo dense-eye protocol for all seven cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from faceAppearance.adapters.eyelidLandmarks import EyelidLandmarksAdapter
from faceAppearance.artifacts import writeJson
from faceAppearance.clientReferenceView import selectClientReference

POLICY = {
    'inferenceFrame': 'savedWorkingRgbPng',
    'referenceSelection': 'sharedReliablePhotoReferenceV1, automatic, no case override',
    'anchors': 'means FAN36:42 and FAN42:48, FAN30, FAN57 in working coordinates',
    'anchorOffsetsWorkingPixels': [[0, 0], [2, 0], [-2, 0], [0, 2], [0, -2]],
    'maximumCropVariationWorkingPixels': 3.,
    'quality': 'baseline adapter pointValidMask and both baseline eye quality.valid, plus maximum perturbation variation <=3px',
    'modelLoads': 1, 'cpuThreads': 1, 'rejectedEyes': 'Report; do not change anchors, view or quality thresholds',
    'sharedAcrossGeometryBranches': True,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def main():
    cv2.setNumThreads(1)
    splitPath = ROOT / 'runs/shapePreservation20261024/split.json'
    split = json.loads(splitPath.read_text())
    output = ROOT / 'runs/shapePipeline20261024'
    output.mkdir(parents=True, exist_ok=True)
    planPath = output / 'denseObservationPlan.json'
    if planPath.exists():
        raise FileExistsError('Preserve previously frozen dense observations')
    detector = EyelidLandmarksAdapter(ROOT)
    weightPath = detector.assetPath()
    commonHashes = {str(path): digest(path) for path in (splitPath, weightPath, Path(__file__),
        ROOT / 'src/faceAppearance/adapters/eyelidLandmarks.py', ROOT / 'src/faceAppearance/clientReferenceView.py')}
    entries = []
    for case in split['cases']:
        reconstruction, working = Path(case['reconstruction']), Path(case['workingPhotos'])
        fanPath = reconstruction / 'directClient/renderedJaw/photoObservations.npz'
        observations = arrays(fanPath)
        observations['diagnostics'] = json.loads(fanPath.with_suffix('.json').read_text())
        inputsPath, cameraPath = reconstruction / 'inputs.json', reconstruction / 'cameras.json'
        inputs, cameras = json.loads(inputsPath.read_text()), json.loads(cameraPath.read_text())
        selection = selectClientReference(cameras, inputs, observations)
        view = selection['referenceViewIndex']
        photoPath = None if view is None else working / f'input{view:02d}.png'
        sourcePaths = [fanPath, fanPath.with_suffix('.json'), inputsPath, cameraPath]
        if photoPath is not None:
            sourcePaths.append(photoPath)
        entries.append({'caseId': case['caseId'], 'referenceSelection': selection,
            'referenceViewIndex': view, 'photoPath': None if photoPath is None else str(photoPath),
            'fanPath': str(fanPath), 'sourceHashes': {str(path): digest(path) for path in sourcePaths}})
    writeJson(planPath, {'status': 'frozenBeforeDenseEyeInference', 'policy': POLICY,
        'commonHashes': commonHashes, 'cases': entries, 'candidateResultsInspected': False})
    detector.loadModels()
    reports = []
    for entry in entries:
        destination = output / entry['caseId'] / 'observations'
        destination.mkdir(parents=True, exist_ok=False)
        hashes = {**commonHashes, **entry['sourceHashes'], str(planPath): digest(planPath)}
        if any(digest(path) != expected for path, expected in hashes.items()):
            raise ValueError('Frozen observation input changed')
        writeJson(destination / 'referenceSelection.json', entry['referenceSelection'])
        view = entry['referenceViewIndex']
        if view is None:
            report = {'status': 'skippedNoReliableReference', 'sourceHashes': hashes, 'policy': POLICY}
            writeJson(destination / 'observations.json', report); reports.append(report)
            continue
        photo = np.asarray(Image.open(entry['photoPath']).convert('RGB')).copy()
        fan = arrays(entry['fanPath'])['faceLandmarksWorking'][view]
        anchors = np.stack([fan[36:42].mean(0), fan[42:48].mean(0), fan[30], fan[57]])
        predictions = [detector.infer(photo, faceAnchors=anchors + np.asarray(offset))
            for offset in POLICY['anchorOffsetsWorkingPixels']]
        first = predictions[0]
        curves = np.stack([prediction['eyelidCurves'] for prediction in predictions])
        variation = np.linalg.norm(curves - curves[:1], axis=-1).max(0)
        accepted = first['pointValidMask'].all(axis=(1, 2))
        accepted &= variation.max(axis=(1, 2)) <= POLICY['maximumCropVariationWorkingPixels']
        accepted &= np.asarray([eye['valid'] for eye in first['quality']])
        values = dict(eyelidCurvesWorking=curves[0], cropVariantCurvesWorking=curves,
            maximumVariationWorkingPixels=variation, rawPeakScores=first['rawPeakScores'],
            pointValidMask=first['pointValidMask'], acceptedEyeMask=accepted, faceAnchors=anchors,
            cropBoxes=first['cropBoxes'], faceLandmarksWorking=first['faceLandmarks'],
            faceRawPeakScores=first['faceRawPeakScores'])
        np.savez_compressed(destination / 'observations.npz', **values)
        report = {'caseId': entry['caseId'], 'status': 'accepted' if accepted.all() else 'rejectedPhotoContourQuality',
            'referenceViewIndex': view, 'sourcePhotoSha256': digest(entry['photoPath']),
            'sourceFanSha256': digest(entry['fanPath']), 'sourceHashes': hashes,
            'policy': POLICY, 'model': detector.modelMetadata, 'quality': first['quality'],
            'maximumCropVariationWorkingPixels': variation.max(axis=(1, 2)), 'acceptedEyeMask': accepted,
            'observationSha256': digest(destination / 'observations.npz'),
            'inferenceFrame': POLICY['inferenceFrame'], 'originalResolutionInference': False,
            'sharedAcrossGeometryBranches': True, 'geometryUsedForInference': False}
        if any(digest(path) != expected for path, expected in hashes.items()):
            raise ValueError('Frozen observation input changed during inference')
        writeJson(destination / 'observations.json', report)
        reports.append({'caseId': entry['caseId'], 'status': report['status'],
            'referenceViewIndex': view, 'maximumCropVariationWorkingPixels': variation.max(axis=(1, 2)),
            'report': str(destination / 'observations.json'), 'reportSha256': digest(destination / 'observations.json')})
        print(json.dumps({**reports[-1], 'maximumCropVariationWorkingPixels': variation.max(axis=(1, 2)).tolist()}), flush=True)
    writeJson(output / 'denseObservationReport.json', {'status': 'complete', 'policy': POLICY,
        'commonHashes': commonHashes, 'planSha256': digest(planPath), 'cases': reports})


if __name__ == '__main__':
    main()
