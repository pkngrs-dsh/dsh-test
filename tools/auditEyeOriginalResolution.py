"""Same-detector original-resolution diagnostic; never selects fitting targets."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter

for threadVariable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[threadVariable] = '1'

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from auditEyeObservationLocalization import digest, readJson, writeJson, loadArrays, transformPoints, cropForEye, font
from faceAppearance.adapters.eyelidLandmarks import EyelidLandmarksAdapter

PARENT = ROOT / 'reports/eyeObservation20261024/localization'
OUT = PARENT / 'originalResolution'
OFFSETS = np.asarray([[0, 0], [2, 0], [-2, 0], [0, 2], [0, -2]], float)


def renderComparison(photo, frozen, native, caseId, path):
    width, height, top, label = 360, 232, 76, 32
    canvas = Image.new('RGB', (width * 4, top + (height + label) * 2 + 58), '#192129')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), f'{caseId} · 同一检测器，不同输入采样', font=font(23), fill='#edf4fa')
    draw.text((12, 43), '原图及精确逆映射的相同 FAN 锚点；仅比较，不选择更好的结果，不修改几何目标。', font=font(16), fill='#c1ccd6')
    labels = ('原图 · 无标记', '冻结工作图预测 → 原图', '原分辨率重新预测', '两组自动预测叠加')
    colors = ('#50edc0', '#ffb449')
    for eye in range(2):
        crop = cropForEye(frozen[eye], photo.size)
        eyeImage = photo.crop(crop)
        zoom = min(width / eyeImage.width, height / eyeImage.height)
        size = tuple(np.rint(np.asarray(eyeImage.size) * zoom).astype(int))
        shown = eyeImage.resize(size, Image.Resampling.BICUBIC)
        xyScale = np.asarray(size) / np.asarray(eyeImage.size)
        for column in range(4):
            origin = np.array([column * width + (width - shown.width) // 2,
                top + eye * (height + label) + label + (height - shown.height) // 2])
            canvas.paste(shown, tuple(origin))
            draw.text((column * width + 10, top + eye * (height + label) + 3),
                ('左眼' if eye == 0 else '右眼') + ' · ' + labels[column], font=font(16), fill='#edf4fa')
            curves = [] if column == 0 else [(frozen[eye], colors[0])] if column == 1 else [(native[eye], colors[1])] if column == 2 else [(frozen[eye], colors[0]), (native[eye], colors[1])]
            for lids, color in curves:
                for line in lids:
                    screen = (line - np.array(crop[:2])) * xyScale + origin
                    draw.line([tuple(point) for point in screen], fill=color, width=2)
                    for point in screen:
                        draw.ellipse(tuple(point - 2) + tuple(point + 2), fill=color)
    draw.text((12, canvas.height - 48), '绿色：既有工作图预测；橙色：原图新预测。所有坐标均在同一原图眼区显示。', font=font(17), fill='#c1ccd6')
    draw.text((12, canvas.height - 25), '相同锚点仍会因裁剪整数舍入、插值及网络内部缩放产生差异；两者均无人工真值。', font=font(16), fill='#c1ccd6')
    canvas.save(path)


def main():
    cv2.setNumThreads(1)
    parentSummary = readJson(PARENT / 'summary.json')
    entries = []
    sourcePaths = [PARENT / 'summary.json', PARENT / 'auditPlan.json']
    for case in parentSummary['cases']:
        reportPath = PARENT / case['caseId'] / 'report.json'
        report = readJson(reportPath)
        sourcePaths.append(reportPath)
        context = readJson(ROOT / 'runs/boundaryField20261024' / case['caseId'] / 'inputContext.json')
        metadataPath = Path(context['entry']['reconstruction']) / 'inputs.json'
        metadata = readJson(metadataPath)[case['referenceViewIndex']]
        denseStem = ROOT / 'runs/shapePipeline20261024' / case['caseId'] / 'observations/observations'
        sourcePaths.extend([metadataPath, denseStem.with_suffix('.npz'), denseStem.with_suffix('.json')])
        available = report['originalStatus'] == 'exactBytesAndOrientedSizeVerified'
        if available:
            sourcePaths.append(Path(report['originalPath']))
        entries.append({'caseId': case['caseId'], 'referenceViewIndex': case['referenceViewIndex'],
            'available': available, 'metadataPath': str(metadataPath), 'denseStem': str(denseStem),
            'originalPath': report['originalPath'], 'originalExpectedSha256': metadata['sha256'],
            'workingToInput': metadata['workingToInput'], 'inputToWorking': metadata['inputToWorking'],
            'orientedSize': metadata['orientedSize']})
    attemptManifest = ROOT / 'reports/eyeObservation20261024/originals/manifest.json'
    if attemptManifest.is_file():
        sourcePaths.append(attemptManifest)
    codePaths = [Path(__file__).resolve(), ROOT / 'tools/auditEyeObservationLocalization.py',
        ROOT / 'src/faceAppearance/adapters/eyelidLandmarks.py']
    weightPath = ROOT / '04_STFR/weights/registration/mergefaneye_d2_ep40_sym_DM_ce_gray_noblend.onnx'
    sourcePaths.append(weightPath)
    OUT.mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for path in codePaths:
        target = OUT / 'implementation' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        snapshots[str(path)] = {'sha256': digest(path), 'snapshotPath': str(target)}
    sourceHashes = {str(path): digest(path) for path in sourcePaths}
    plan = {'status': 'frozenBeforeModelLoadAndInference', 'cases': entries, 'sourceHashes': sourceHashes,
        'implementation': snapshots, 'weightPath': str(weightPath), 'weightSha256': digest(weightPath),
        'sameDetector': True, 'backend': 'OpenCV DNN CPU', 'cpuThreads': 1, 'modelsLoaded': 1,
        'anchorPolicy': 'Exact saved float32 FAN faceAnchors plus original five working-pixel offsets; transform each by saved workingToInput, then existing adapter casts float32',
        'anchorOffsetsWorkingPixels': OFFSETS, 'newAnchorDetection': False,
        'comparison': 'Transform all original predictions back by saved inputToWorking; compare paired frozen working predictions at same offset and eye/curve/point IDs',
        'qualityPolicy': 'Report original adapter quality, accepted points, and max crop variation <=3 working pixels; no rejected point replacement',
        'caseParameters': 'Identical for every locally verified original; missing four remote sources remain missing',
        'manualTruthEstablished': False, 'chooseBestObservation': False, 'fittingTargetsChanged': False,
        'limitations': ['Native input still undergoes detector internal 128-pixel eye resize and 32-pixel heatmaps.',
            'Crop integer rounding and interpolation can change despite identical continuous anchors.',
            'Agreement or stability is not an accuracy certificate.']}
    writeJson(OUT / 'experimentPlan.json', plan)
    detector = EyelidLandmarksAdapter(ROOT).loadModels()
    results = []
    for entry in entries:
        caseId = entry['caseId']
        if not entry['available']:
            results.append({'caseId': caseId, 'status': 'missingExactLocalOriginal', 'newInferenceCount': 0})
            continue
        started = perf_counter()
        if digest(entry['originalPath']) != entry['originalExpectedSha256']:
            raise ValueError('Original bytes changed')
        photo = ImageOps.exif_transpose(Image.open(entry['originalPath'])).convert('RGB')
        if list(photo.size) != entry['orientedSize']:
            raise ValueError('Original oriented size changed')
        old = loadArrays(Path(entry['denseStem']).with_suffix('.npz'))
        oldReport = readJson(Path(entry['denseStem']).with_suffix('.json'))
        np.testing.assert_array_equal(OFFSETS, oldReport['policy']['anchorOffsetsWorkingPixels'])
        if oldReport['model']['weightSha256'] != detector.modelMetadata['weightSha256']:
            raise ValueError('Detector weight differs from frozen observation source')
        workingAnchors = old['faceAnchors'][None].astype(float) + OFFSETS[:, None]
        originalAnchors = transformPoints(workingAnchors, entry['workingToInput'])
        np.testing.assert_allclose(transformPoints(originalAnchors, entry['inputToWorking']), workingAnchors, rtol=0, atol=1e-10)
        predictions = [detector.infer(np.asarray(photo), faceAnchors=anchors) for anchors in originalAnchors]
        originalCurves = np.stack([item['eyelidCurves'] for item in predictions])
        workingCurves = transformPoints(originalCurves, entry['inputToWorking'])
        oldCurves = old['cropVariantCurvesWorking']
        pairedDistance = np.linalg.norm(workingCurves - oldCurves, axis=-1)
        variation = np.linalg.norm(workingCurves - workingCurves[:1], axis=-1)
        maximumVariation = variation.max(axis=(0, 2, 3))
        accepted = predictions[0]['pointValidMask'].all(axis=(1, 2)) & np.array([eye['valid'] for eye in predictions[0]['quality']]) & (maximumVariation <= 3)
        destination = OUT / caseId
        destination.mkdir()
        np.savez_compressed(destination / 'observations.npz', originalCurves=originalCurves,
            originalPredictionsWorking=workingCurves, frozenWorkingCurves=oldCurves,
            originalAnchors=originalAnchors, workingAnchors=workingAnchors,
            adapterUsedOriginalAnchors=np.stack([item['faceAnchors'] for item in predictions]),
            originalCropBoxes=np.stack([item['cropBoxes'] for item in predictions]),
            originalRawPeakScores=np.stack([item['rawPeakScores'] for item in predictions]),
            originalPointValidMask=np.stack([item['pointValidMask'] for item in predictions]),
            pairedDistanceWorkingPixels=pairedDistance, cropVariationWorkingPixels=variation,
            originalAcceptedEyeMask=accepted)
        renderComparison(photo, transformPoints(oldCurves[0], entry['workingToInput']), originalCurves[0], caseId, destination / 'comparison.png')
        report = {'caseId': caseId, 'status': 'diagnosticComplete', 'referenceViewIndex': entry['referenceViewIndex'],
            'newInferenceCount': 5, 'maximumCropVariationWorkingPixels': maximumVariation,
            'frozenMaximumCropVariationWorkingPixels': oldReport['maximumCropVariationWorkingPixels'],
            'acceptedOriginalEyeMaskUnderSameWorkingTolerance': accepted,
            'baselinePairedDifferenceWorkingPixels': {'mean': pairedDistance[0].mean(axis=(1, 2)), 'max': pairedDistance[0].max(axis=(1, 2))},
            'allPairedDifferenceWorkingPixels': {'mean': pairedDistance.mean(axis=(0, 2, 3)), 'max': pairedDistance.max(axis=(0, 2, 3))},
            'originalQualityByOffset': [item['quality'] for item in predictions],
            'originalSourceSha256': entry['originalExpectedSha256'], 'model': detector.modelMetadata,
            'sourceHashes': sourceHashes, 'manualTruthEstablished': False, 'selectedForFit': False,
            'elapsedSeconds': perf_counter() - started, 'planSha256': digest(OUT / 'experimentPlan.json'),
            'outputHashes': {path.name: digest(path) for path in destination.iterdir()}}
        writeJson(destination / 'report.json', report)
        results.append({'caseId': caseId, 'status': report['status'], 'newInferenceCount': 5,
            'baselinePairedDifferenceWorkingPixels': report['baselinePairedDifferenceWorkingPixels'],
            'maximumCropVariationWorkingPixels': maximumVariation,
            'acceptedOriginalEyeMaskUnderSameWorkingTolerance': accepted,
            'reportSha256': digest(destination / 'report.json')})
        print(json.dumps(results[-1], default=lambda value: value.tolist()), flush=True)
    if any(digest(path) != value for path, value in sourceHashes.items()):
        raise ValueError('Input changed during inference')
    if any(digest(path) != item['sha256'] for path, item in snapshots.items()):
        raise ValueError('Implementation changed during inference')
    writeJson(OUT / 'summary.json', {'status': 'complete', 'cases': results,
        'totalNewInferences': sum(item['newInferenceCount'] for item in results),
        'sourceFilesUnchanged': True, 'noSelectionOrFitting': True, 'manualTruthEstablished': False,
        'planSha256': digest(OUT / 'experimentPlan.json')})


if __name__ == '__main__':
    main()
