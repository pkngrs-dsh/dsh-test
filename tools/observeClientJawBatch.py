"""Full-face observations for frozen client cases, with scale-normalized gates.

The original photographs and saved detector boxes are the only observation
inputs. Geometry is deliberately not loaded and cannot select the detections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'thirdParty/ibug')]
from faceAppearance.artifacts import writeJson
from faceAppearance.data import readImage
from ibug.face_alignment import FANPredictor


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(points, matrix):
    homogeneous = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def expandedCrop(box, ratio):
    center = (box[:2] + box[2:]) / 2
    size = (box[2:] - box[:2]).mean() / ratio
    left = np.round(center - size / 2)
    return np.r_[left, np.round(left + size) + 1].astype(int)


def draw(photo, points, valid, label, detail):
    image = photo.copy()
    canvas = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    for index, (x, y) in enumerate(points):
        color = (78, 246, 130) if valid[index] else (255, 95, 70)
        if index >= 17:
            color = (110, 180, 255) if valid[index] else (255, 95, 70)
        radius = 3 if 5 <= index <= 11 else 2
        canvas.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)
        if index < 17:
            canvas.text((x+4, y-9), str(index), fill=color, font=font,
                        stroke_width=1, stroke_fill=(0, 0, 0))
    canvas.line([tuple(p) for p in points[:17]], fill=(255, 225, 74), width=1)
    canvas.rectangle((0, 0, image.width, 46), fill=(18, 24, 31))
    canvas.text((7, 4), label, font=font, fill='white')
    canvas.text((7, 24), detail, font=font, fill='white')
    return image


def observe(case, detector, *, minimumPeak, minimumVariantPeak, maximumSpreadFraction):
    output = Path(case['observationPath']).parent
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Observation output must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    inputsPath = Path(case['inputsPath'])
    if sha(inputsPath) != case['inputsSha256']:
        raise ValueError(f"Changed input metadata: {case['caseId']}")
    inputs = json.loads(inputsPath.read_text())
    if len(inputs) != len(case['sourceImages']):
        raise ValueError('Image and input count mismatch')
    scales = np.array([1., .95, 1.05])
    workingAll, originalAll, peaksAll, validAll, spreads, scaleAll = [], [], [], [], [], []
    gates = {'peak': [], 'variantPeak': [], 'stability': [], 'insideOriginal': [], 'insideWorking': []}
    records, panels = [], []
    started = perf_counter()
    for index, (record, source) in enumerate(zip(inputs, case['sourceImages'])):
        path = Path(source['path'])
        if sha(path) != source['sha256'] or source['sha256'] != record['sha256']:
            raise ValueError('Original photograph hash differs from saved input')
        original = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        if list(original.size) != record['orientedSize']:
            raise ValueError('EXIF-oriented dimensions differ from saved input')
        box = np.asarray(record['faceDetection']['boxXyxy'], np.float64)
        center, half = (box[:2] + box[2:]) / 2, (box[2:] - box[:2]) / 2
        boxes = np.asarray([np.r_[center-half*s, center+half*s] for s in scales])
        tick = perf_counter()
        points, peaks = detector(np.asarray(original), boxes, rgb=True)
        inferSeconds = perf_counter() - tick
        working = np.asarray([transform(p, record['inputToWorking']) for p in points])
        spread = np.linalg.norm(working-working[:1], axis=-1).max(0)
        boxWorking = transform(box.reshape(2, 2), record['inputToWorking'])
        faceScale = float(np.sqrt(np.prod(np.abs(boxWorking[1]-boxWorking[0]))))
        if not np.isfinite(faceScale) or faceScale <= 0:
            raise ValueError('Invalid saved face-box scale')
        normalizedSpread = spread / faceScale
        imageSize = np.asarray(record['workingSize'])
        insideOriginal = ((points[0] >= 0).all(-1) & (points[0] < original.size).all(-1))
        insideWorking = ((working[0] >= 0).all(-1) & (working[0] < imageSize).all(-1))
        viewGates = {'peak': peaks[0] >= minimumPeak,
                     'variantPeak': np.min(peaks, axis=0) >= minimumVariantPeak,
                     'stability': normalizedSpread <= maximumSpreadFraction,
                     'insideOriginal': insideOriginal, 'insideWorking': insideWorking}
        valid = np.isfinite(working).all(axis=(0, 2)) & np.isfinite(peaks).all(0)
        for key, value in viewGates.items():
            valid &= value
            gates[key].append(value)
        replayImage, replay = readImage(path, imageSize=record['workingSize'][0],
            cropPolicy=record['requestedCropPolicy'], cropMargin=record['cropMargin'],
            faceDetector=lambda _, record=record: record['faceDetection'])
        for key in ('inputToWorking', 'cropBoxXyxy', 'paddingLeftTopRightBottom'):
            if replay[key] != record[key]:
                raise ValueError(f'Working crop replay mismatch: {key}')
        photo = Image.fromarray(np.rint(replayImage*255).astype(np.uint8))
        detail = f'Chin peak {peaks[0,8]:.3f} | crop spread {normalizedSpread[8]:.3%} of face'
        panel = draw(photo, working[0], valid, f"{case['caseId']} | photo {index+1}", detail)
        photo.save(output / f'view{index}-photo.png')
        panel.save(output / f'view{index}-landmarks.png')
        # A separate original-frame overlay exposes low resolution and crop errors.
        originalScale = min(1., 1024. / max(original.size))
        originalPreview = original.resize(tuple(int(round(s*originalScale)) for s in original.size))
        draw(originalPreview, points[0]*originalScale, valid,
             f"{case['caseId']} | original {original.width} x {original.height}",
             'Green accepted / red rejected; source resolution unchanged in inference').save(output / f'view{index}-original-landmarks.png')
        panels.append(panel)
        workingAll.append(working); originalAll.append(points); peaksAll.append(peaks)
        validAll.append(valid); spreads.append(spread); scaleAll.append(faceScale)
        leftEye, rightEye = working[0,36:42].mean(0), working[0,42:48].mean(0)
        eyeVector = rightEye-leftEye
        interocular = float(np.linalg.norm(eyeVector))
        direction = eyeVector / max(interocular, 1e-12)
        noseOffset = float(np.dot(working[0,30]-(leftEye+rightEye)/2, direction) / max(interocular, 1e-12))
        failed = {str(j): [key for key, mask in viewGates.items() if not mask[j]] for j in range(5,12) if not valid[j]}
        r = {'viewIndex': index, 'photoPath': str(path), 'photoSha256': source['sha256'],
             'orientedSize': record['orientedSize'], 'inputToWorking': record['inputToWorking'],
             'workingSize': record['workingSize'], 'savedRetinaFaceBoxOriginal': box.tolist(),
             'originalFaceBoxWidthHeightPixels': (box[2:]-box[:2]).tolist(),
             'fanCropOriginal': expandedCrop(box, detector.config.crop_ratio).tolist(),
             'bboxScaleVariants': scales.tolist(), 'variantCropsOriginal': [expandedCrop(b, detector.config.crop_ratio).tolist() for b in boxes],
             'faceScaleWorkingPixels': faceScale, 'interocularWorkingPixels': interocular,
             'frontalAppearanceDiagnostics': {'eyeLineRollDegrees': float(np.degrees(np.arctan2(eyeVector[1],eyeVector[0]))),
                 'noseLateralOffsetPerInterocular': noseOffset,
                 'mouthOpeningPerFaceScale': float(np.linalg.norm(working[0,62]-working[0,66])/faceScale),
                 'meaning': 'Observation-only diagnostics; not calibrated pose or automatic proof of a neutral frontal photograph'},
             'chin8Working': working[0,8].tolist(), 'chin8RawPeakScore': float(peaks[0,8]),
             'chin8MaximumCropVariationPixels': float(spread[8]),
             'chin8MaximumCropVariationPerFaceScale': float(normalizedSpread[8]),
             'originalChinToLowerLipDistancePixels': float(np.linalg.norm(points[0,8]-points[0,57])),
             'central7to9Accepted': bool(valid[7:10].all()), 'jaw5to11Accepted': bool(valid[5:12].all()),
             'optionalSpline6and10Accepted': bool(valid[[6,10]].all()), 'rejectedJaw5to11': failed,
             'inferenceSeconds': inferSeconds}
        records.append(r)
        print(json.dumps({'caseId': case['caseId'], **r}), flush=True)
    workingAll, originalAll, peaksAll, validAll, spreads, scaleAll = map(np.asarray,
        (workingAll, originalAll, peaksAll, validAll, spreads, scaleAll))
    np.savez_compressed(output / 'landmarkObservations.npz',
        faceLandmarksWorking=workingAll[:,0], faceLandmarksOriginal=originalAll[:,0],
        faceRawPeakScores=peaksAll[:,0], landmarkCandidateValidMask=validAll,
        cropVariantLandmarksWorking=workingAll, cropVariantLandmarksOriginal=originalAll,
        cropVariantRawPeakScores=peaksAll, bboxScaleVariants=scales,
        maximumCropVariationPixels=spreads, faceScaleWorkingPixels=scaleAll,
        maximumCropVariationPerFaceScale=spreads/scaleAll[:,None],
        acceptedViewMask=validAll[:,7:10].all(1), acceptedWideJawViewMask=validAll[:,5:12].all(1),
        **{key+'GateMask': np.asarray(value) for key, value in gates.items()})
    width, height = panels[0].size
    board = Image.new('RGB', (width*2, height*((len(panels)+1)//2)), (20,24,30))
    for index, panel in enumerate(panels):
        board.paste(panel, ((index%2)*width, (index//2)*height))
    board.save(output / 'jawObservations.png')
    report = {'schemaVersion': 1, 'caseId': case['caseId'], 'status': 'complete',
        'method': 'Independent full-face ibug FAN 2dfan2_alt on EXIF-oriented original photos',
        'modelPath': str(ROOT/'assets/models/ibug/2dfan2_alt.pth'),
        'modelSha256': sha(ROOT/'assets/models/ibug/2dfan2_alt.pth'),
        'implementationSha256': sha(__file__), 'inputsSha256': case['inputsSha256'],
        'backend': 'CPU PyTorch, one thread, no JIT; one model reused for batch',
        'fanCropRatio': detector.config.crop_ratio, 'fanInputSize': detector.config.input_size,
        'selectedVariant': 'Saved full RetinaFace bbox scale 1.0; perturbations only gate stability',
        'candidateGate': {'rawPeakMinimum': minimumPeak, 'minimumAcrossVariantPeaks': minimumVariantPeak,
            'maximumCropVariationPerFaceScale': maximumSpreadFraction,
            'faceScale': 'sqrt(saved original face-box width * height) after exact inputToWorking transform',
            'insideOriginalAndWorkingImage': True, 'geometryUsedToSelectObservations': False},
        'acceptedCentral7to9ViewMask': validAll[:,7:10].all(1).tolist(),
        'acceptedJaw5to11ViewMask': validAll[:,5:12].all(1).tolist(),
        'views': records, 'seconds': perf_counter()-started,
        'limitations': ['Raw heatmap peaks are not calibrated confidence probabilities.',
            '2D silhouette landmarks are view dependent, not fixed 3D correspondences.',
            'Crop perturbation agreement does not prove anatomical accuracy or resolve beard/shadow boundaries.',
            'Upsampling low-resolution originals does not add measurement detail.',
            'Reference selection must be independent of the desired chin residual.']}
    writeJson(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'runs/clientChinAdaptive20261012/cases.json')
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--minimum-peak', type=float, default=.25)
    parser.add_argument('--minimum-variant-peak', type=float, default=.20)
    parser.add_argument('--maximum-spread-fraction', type=float, default=.015)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    started = perf_counter()
    model = FANPredictor.get_model('2dfan2_alt')
    model.weights = str(ROOT/'assets/models/ibug/2dfan2_alt.pth')
    detector = FANPredictor(device='cpu', model=model, config=FANPredictor.create_config(use_jit=False))
    manifest = json.loads(args.manifest.read_text())
    selected = [c for c in manifest['cases'] if not args.cases or c['caseId'] in args.cases]
    reports = [observe(case, detector, minimumPeak=args.minimum_peak,
        minimumVariantPeak=args.minimum_variant_peak, maximumSpreadFraction=args.maximum_spread_fraction) for case in selected]
    batch = {'schemaVersion': 1, 'status': 'complete', 'manifestPath': str(args.manifest),
             'manifestSha256': sha(args.manifest), 'seconds': perf_counter()-started,
             'caseCount': len(reports), 'viewCount': sum(len(r['views']) for r in reports),
             'cases': [{'caseId': r['caseId'], 'acceptedCentral7to9ViewMask': r['acceptedCentral7to9ViewMask'],
                        'acceptedJaw5to11ViewMask': r['acceptedJaw5to11ViewMask'], 'seconds': r['seconds']} for r in reports]}
    writeJson(args.manifest.parent / 'observationBatchReport.json', batch)
    print(json.dumps(batch), flush=True)


if __name__ == '__main__':
    main()
