"""Read frozen eye observations and aperture masks without fitting or rendering.

Every overlay is a diagnostic on the same photographs. Automatic curves,
landmarks and model boundaries are deliberately kept separate from truth.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

for threadVariable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[threadVariable] = '1'

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/boundaryField20261024'
DENSE = ROOT / 'runs/shapePipeline20261024'
OUT = ROOT / 'reports/eyeObservation20261024/localization'
EYES = ('imageLeft', 'imageRight')
COLORS = {'coarse': '#ffb449', 'dense': '#50edc0', 'model': '#f18ce6'}
FONT_PATH = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
POLICY = {
    'schemaVersion': 'frozenEyeLocalizationAuditV1',
    'caseSelection': 'All seven cases in boundaryField20261024; saved reference view unchanged',
    'coarseSource': 'Saved FAN68 photograph points; not projected coarse mesh bindings',
    'denseSource': 'Saved STFR 24 eyelid points per eye; no new inference',
    'modelSource': 'All external contours of the saved completed-baseline full visible aperture mask',
    'coordinateConvention': 'Exact saved working pixel coordinates; original mapping uses saved workingToInput',
    'difference': 'FAN upper/lower interior points versus dense polyline at the same frozen photo-eye-axis x; positive means FAN is inward',
    'interpolation': 'All polyline intersections; report ambiguity or outside domain, never extrapolate',
    'display': 'Each eye uses one crop for all four columns; separate unmarked photograph, FAN, dense, model panels',
    'originalAvailability': 'Use local original only if exact byte SHA and EXIF-transposed size match; missing source is explicit',
    'resolutionComparison': 'None in this cached audit; working scale alone does not establish localization accuracy',
    'manualTruthEstablished': False,
    'fitting': False,
    'rasterization': False,
    'productionDefaultsChanged': False,
}


def readJson(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def writeJson(path, value):
    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=encode))


def loadArrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def readEvidence(stem):
    stem = Path(stem)
    saved = loadArrays(stem.with_suffix('.npz'))
    used = set()
    def restore(value):
        if isinstance(value, dict) and 'arrayKey' in value:
            key = value['arrayKey']
            array = saved[key]
            if list(array.shape) != value['shape'] or str(array.dtype) != value['dtype']:
                raise ValueError('Saved evidence identity mismatch')
            used.add(key)
            return array
        if isinstance(value, dict):
            return {key: restore(item) for key, item in value.items()}
        if isinstance(value, list):
            return [restore(item) for item in value]
        return value
    result = restore(readJson(stem.with_suffix('.json')))
    if used != set(saved):
        raise ValueError('Unreferenced saved arrays')
    return result


def transformPoints(points, matrix):
    values = np.asarray(points, float)
    homogeneous = np.c_[values.reshape(-1, 2), np.ones(values.size // 2)] @ np.asarray(matrix, float).T
    if np.any(abs(homogeneous[:, 2]) < 1e-12):
        raise ValueError('Singular image mapping')
    return (homogeneous[:, :2] / homogeneous[:, 2:]).reshape(values.shape)


def curveIntersections(curve, x):
    """Keep multiple intersections explicit instead of sorting a folded curve."""
    result = []
    for first, second in zip(curve[:-1], curve[1:]):
        if abs(second[0] - first[0]) < 1e-10:
            if abs(x - first[0]) < 1e-10:
                result.extend([float(first[1]), float(second[1])])
            continue
        position = (x - first[0]) / (second[0] - first[0])
        if -1e-10 <= position <= 1 + 1e-10:
            result.append(float(first[1] + position * (second[1] - first[1])))
    return sorted({round(value, 10) for value in result})


def compareAtSameAxisX(coarse, dense):
    lower, upper = np.asarray(dense, float)
    corners = (lower[[0, -1]] + upper[[0, -1]]) / 2
    axis = corners[1] - corners[0]
    width = float(np.linalg.norm(axis))
    if width <= 0:
        raise ValueError('Invalid frozen photo eye width')
    axis /= width
    normal = np.array([-axis[1], axis[0]])
    frame = np.column_stack([axis, normal])
    coarseLocal = (coarse - corners[0]) @ frame
    rows = []
    for name, curve, ids in (('upper', upper, (1, 2)), ('lower', lower, (4, 5))):
        local = (curve - corners[0]) @ frame
        for localId in ids:
            x, y = coarseLocal[localId]
            intersections = curveIntersections(local, x)
            inward = None if len(intersections) != 1 else float((y - intersections[0]) * (1 if name == 'upper' else -1))
            rows.append({'lid': name, 'localFanId': localId, 'axisXPixels': float(x),
                'coarseNormalYPixels': float(y), 'denseIntersectionNormalYPixels': intersections,
                'inwardDifferencePixels': inward,
                'status': 'measured' if inward is not None else 'ambiguous' if intersections else 'outsideDenseDomain'})
    return {'origin': corners[0], 'axis': axis, 'normal': normal, 'eyeWidthPixels': width, 'points': rows}


def font(size):
    return ImageFont.truetype(str(FONT_PATH), size)


def cropForEye(dense, imageSize):
    values = np.asarray(dense).reshape(-1, 2)
    center = values.mean(0)
    width = np.linalg.norm(np.asarray(dense)[:, -1].mean(0) - np.asarray(dense)[:, 0].mean(0))
    half = np.array([width * .72, width * .44])
    minimum = np.maximum(np.floor(center - half), [0, 0]).astype(int)
    maximum = np.minimum(np.ceil(center + half), imageSize).astype(int)
    return tuple(np.r_[minimum, maximum])


def eyePanels(photo, eyeRows, title, subtitle, output):
    panelWidth, panelHeight = 360, 232
    headerHeight, labelHeight, footHeight = 76, 32, 56
    canvas = Image.new('RGB', (panelWidth * 4, headerHeight + (panelHeight + labelHeight) * 2 + footHeight), '#192129')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), title, font=font(23), fill='#edf4fa')
    draw.text((12, 43), subtitle, font=font(16), fill='#c1ccd6')
    labels = ('原照片 · 无标记', '旧 FAN 六点', '稠密睑缘预测', '模型完整可见眼裂')
    colors = (None, COLORS['coarse'], COLORS['dense'], COLORS['model'])
    for rowIndex, row in enumerate(eyeRows):
        crop = cropForEye(row['dense'], photo.size)
        cropImage = photo.crop(crop)
        zoom = min(panelWidth / cropImage.width, panelHeight / cropImage.height)
        shownSize = tuple(np.maximum(1, np.rint(np.asarray(cropImage.size) * zoom).astype(int)))
        # Bicubic is presentation only; all measurements use original saved coordinates.
        shown = cropImage.resize(shownSize, Image.Resampling.BICUBIC)
        xOffset, yOffset = (panelWidth - shown.width) // 2, (panelHeight - shown.height) // 2
        actualScale = np.array(shownSize) / np.array(cropImage.size)
        for column in range(4):
            origin = np.array([column * panelWidth + xOffset, headerHeight + rowIndex * (panelHeight + labelHeight) + labelHeight + yOffset])
            canvas.paste(shown, tuple(origin))
            draw.text((column * panelWidth + 10, headerHeight + rowIndex * (panelHeight + labelHeight) + 3),
                ('左眼' if rowIndex == 0 else '右眼') + ' · ' + labels[column], font=font(17), fill=colors[column] or '#edf4fa')
            if column == 0:
                continue
            if column == 1:
                lines = [np.asarray(row['coarse'])[[0, 1, 2, 3, 4, 5, 0]]]
            elif column == 2:
                lines = list(row['dense'])
            else:
                lines = [np.r_[points, points[:1]] for points in row['modelContours']]
            for points in lines:
                screen = (np.asarray(points) - np.array(crop[:2])) * actualScale + origin
                draw.line([tuple(point) for point in screen], fill=colors[column], width=2)
            if column in (1, 2):
                points = row['coarse'] if column == 1 else np.asarray(row['dense']).reshape(-1, 2)
                for point in (np.asarray(points) - np.array(crop[:2])) * actualScale + origin:
                    draw.ellipse(tuple(point - 2) + tuple(point + 2), fill=colors[column])
    draw.text((12, canvas.height - 48), '全部为自动观测；没有人工真值。图像放大不增加原始信息。模型轮廓含所有保存片段。', font=font(17), fill='#c1ccd6')
    draw.text((12, canvas.height - 25), '橙色：FAN；绿色：稠密预测；粉色：完整装配的可见眼裂。不同列使用完全相同的眼区和倍率。', font=font(16), fill='#c1ccd6')
    canvas.save(output)


def auditCase(caseId):
    caseRoot = RUN / caseId
    contextPath = caseRoot / 'inputContext.json'
    context = readJson(contextPath)
    view = context['referenceSelection']['referenceViewIndex']
    metadataPath = Path(context['entry']['reconstruction']) / 'inputs.json'
    metadata = readJson(metadataPath)[view]
    photoPath = Path(context['entry']['workingPhotos']) / f'input{view:02d}.png'
    denseStem = DENSE / caseId / 'observations/observations'
    denseReport = readJson(denseStem.with_suffix('.json'))
    denseArrays = loadArrays(denseStem.with_suffix('.npz'))
    legacy = readEvidence(caseRoot / 'legacyObservationState')
    evidence = readEvidence(caseRoot / 'baseline/evidence')
    effectPath = caseRoot / 'baseline/finalEffectGate.json'
    effect = readJson(effectPath)
    sourcePaths = [contextPath, metadataPath, photoPath, denseStem.with_suffix('.json'), denseStem.with_suffix('.npz'),
        caseRoot / 'legacyObservationState.json', caseRoot / 'legacyObservationState.npz',
        caseRoot / 'baseline/evidence.json', caseRoot / 'baseline/evidence.npz', effectPath]
    sourceHashes = {str(path): digest(path) for path in sourcePaths}
    if denseReport['referenceViewIndex'] != view or digest(photoPath) != denseReport['sourcePhotoSha256']:
        raise ValueError('Frozen reference photograph differs')
    if digest(denseStem.with_suffix('.npz')) != denseReport['observationSha256']:
        raise ValueError('Frozen dense observations differ')
    forward, inverse = np.asarray(metadata['inputToWorking'], float), np.asarray(metadata['workingToInput'], float)
    np.testing.assert_allclose(forward @ inverse, np.eye(3), rtol=0, atol=1e-10)
    photo = Image.open(photoPath).convert('RGB')
    if list(photo.size) != metadata['workingSize']:
        raise ValueError('Working photograph size differs')
    denseCurves = denseArrays['eyelidCurvesWorking']
    np.testing.assert_allclose(transformPoints(transformPoints(denseCurves, inverse), forward), denseCurves, rtol=0, atol=1e-10)
    originalPath = Path(metadata['path'])
    original = None
    originalStatus = 'missingLocalOriginal'
    if originalPath.is_file():
        if digest(originalPath) != metadata['sha256']:
            raise ValueError('Original photograph byte hash differs')
        sourceHashes[str(originalPath)] = digest(originalPath)
        original = ImageOps.exif_transpose(Image.open(originalPath)).convert('RGB')
        if list(original.size) != metadata['orientedSize']:
            raise ValueError('EXIF-transposed original photograph size differs')
        originalStatus = 'exactBytesAndOrientedSizeVerified'
    destination = OUT / caseId
    destination.mkdir(exist_ok=False)
    rows, originalRows, measures = [], [], {}
    for eyeIndex, name in enumerate(EYES):
        dense = denseCurves[eyeIndex].astype(float)
        coarse = legacy['observations']['points'][view, 36 + 6 * eyeIndex:42 + 6 * eyeIndex]
        saved = evidence['beforeContours']['eyes'][name]
        mask = saved['roiMasks']['eye'].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        model = [points[:, 0].astype(float) / saved['roi']['scale'] + saved['roi']['origin'] for points in contours if len(points) >= 3]
        row = {'dense': dense, 'coarse': coarse, 'modelContours': model}
        rows.append(row)
        originalRow = {'dense': transformPoints(dense, inverse), 'coarse': transformPoints(coarse, inverse),
            'modelContours': [transformPoints(points, inverse) for points in model]}
        originalRows.append(originalRow)
        measurement = compareAtSameAxisX(coarse, dense)
        originalMeasurement = compareAtSameAxisX(originalRow['coarse'], originalRow['dense'])
        for point in measurement['points']:
            point['ibugId'] = 36 + 6 * eyeIndex + point['localFanId']
        metrics = effect['baseline']['eyes'][name]
        measures[name] = {'working': measurement, 'originalCoordinateMapping': originalMeasurement,
            'coarsePointsWorking': coarse, 'denseCurvesWorking': dense, 'modelContoursWorking': model,
            'modelExtractionStatus': saved['status'], 'modelExtractionReasons': saved.get('reasons', []),
            'fullModelAreaToAutomaticPhotoArea': metrics['modelRasterArea'] / metrics['photoRasterArea'],
            'denseCropMaximumVariationWorkingPixels': denseReport['maximumCropVariationWorkingPixels'][eyeIndex],
            'denseAccepted': denseReport['acceptedEyeMask'][eyeIndex],
            'denseQuality': denseReport['quality'][eyeIndex]}
    scale = float(forward[0, 0])
    eyePanels(photo, rows, f'{caseId} · 保存参考视角 {view + 1} · 工作图观察对照',
        f'工作图 {photo.width}×{photo.height}；原图记录 {metadata["orientedSize"][0]}×{metadata["orientedSize"][1]}；工作缩放 {scale:.4f}×', destination / 'workingComparison.png')
    if original is not None:
        eyePanels(original, originalRows, f'{caseId} · 原始分辨率照片 + 已有观测逆映射',
            '仅把已有坐标映回已验证原图；未重新推理，不能视为新的检测结果', destination / 'originalComparison.png')
    if any(digest(path) != value for path, value in sourceHashes.items()):
        raise ValueError('Input changed during localization audit')
    report = {'caseId': caseId, 'referenceViewIndex': view, 'originalStatus': originalStatus,
        'originalPath': str(originalPath), 'originalExpectedSha256': metadata['sha256'],
        'workingSize': metadata['workingSize'], 'originalOrientedSize': metadata['orientedSize'],
        'inputToWorking': forward, 'workingToInput': inverse, 'workingScale': scale,
        'manualTruthEstablished': False, 'newInference': False, 'geometryChanged': False,
        'eyes': measures, 'sourceHashes': sourceHashes, 'sourceFilesUnchanged': True,
        'limitations': ['Crop stability does not establish absolute accuracy.',
            'Automatic FAN and STFR may encode different eyelid boundaries.',
            'A smaller visible model aperture does not uniquely identify geometry, accessory placement, or observation bias.',
            'Dense points are saved working-image predictions, even in the original-photograph panel.',
            'Previously inspected cases cannot establish unseen generalization.'],
        'images': {path.name: digest(path) for path in destination.glob('*.png')}}
    writeJson(destination / 'report.json', report)
    return {'caseId': caseId, 'referenceViewIndex': view, 'originalStatus': originalStatus, 'workingScale': scale,
        'originalEyeWidths': {name: measures[name]['originalCoordinateMapping']['eyeWidthPixels'] for name in EYES},
        'areaRatios': {name: measures[name]['fullModelAreaToAutomaticPhotoArea'] for name in EYES},
        'report': str(destination / 'report.json'), 'reportSha256': digest(destination / 'report.json')}


def main():
    cv2.setNumThreads(1)
    planPath = RUN / 'experimentPlan.json'
    cases = readJson(planPath)['cases']
    if len(cases) != 7 or len(set(cases)) != 7:
        raise ValueError('Expected the frozen seven-case split')
    OUT.mkdir(parents=True, exist_ok=False)
    codeHash = digest(__file__)
    writeJson(OUT / 'auditPlan.json', {'status': 'frozenBeforeCachedAudit', 'policy': POLICY, 'cases': cases,
        'parentPlanSha256': digest(planPath), 'implementationPath': str(Path(__file__).resolve()), 'implementationSha256': codeHash,
        'fontSha256': digest(FONT_PATH), 'originalInferencePlanned': False})
    results = []
    for case in cases:
        results.append(auditCase(case))
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    if digest(__file__) != codeHash:
        raise ValueError('Audit implementation changed while running')
    writeJson(OUT / 'summary.json', {'status': 'complete', 'policy': POLICY, 'cases': results,
        'implementationSha256': codeHash, 'auditPlanSha256': digest(OUT / 'auditPlan.json'), 'allSevenCases': True})
    options = ''.join(f'<option>{case}</option>' for case in cases)
    html = '''<!doctype html><meta charset="utf-8"><title>眼周观测核验</title>
<style>body{background:#192129;color:#edf4fa;font:17px sans-serif;margin:20px}p{line-height:1.7}img{max-width:100%;display:block;margin:20px 0}select{font:inherit}a{color:#8dd8ff}</style>
<h2>眼周定位：区分照片观测与模型可见眼裂</h2><p>每行从左至右：无标记照片、旧 FAN 六点、稠密眼睑预测、模型完整可见眼裂。同一眼使用同一裁剪和显示倍率。全部是自动结果，没有人工真值；裁剪扰动稳定不等于位置正确。这里没有新拟合、渲染或推理。</p>
<select id="case">OPTIONS</select><p id="state"></p><img id="working"><img id="original"><a id="report">查看数值及来源</a>
<script>const $=x=>document.getElementById(x);fetch('summary.json').then(r=>r.json()).then(data=>{const q=new URLSearchParams(location.search);if(data.cases.some(c=>c.caseId===q.get('case')))$('case').value=q.get('case');function show(){const c=data.cases.find(c=>c.caseId===$('case').value);history.replaceState(null,'','?case='+c.caseId);$('working').src=c.caseId+'/workingComparison.png';$('original').hidden=c.originalStatus!=='exactBytesAndOrientedSizeVerified';if(!$('original').hidden)$('original').src=c.caseId+'/originalComparison.png';$('report').href=c.caseId+'/report.json';$('state').textContent='原图映射眼宽：左 '+c.originalEyeWidths.imageLeft.toFixed(1)+' px，右 '+c.originalEyeWidths.imageRight.toFixed(1)+' px。'+($('original').hidden?'原始文件本地缺失：下方仅展示冻结工作图。':'原始文件 SHA 和朝向尺寸已核实；原图面板仅映回已有观测。')}show();$('case').onchange=show});</script>'''.replace('OPTIONS', options)
    (OUT / 'index.html').write_text(html)


if __name__ == '__main__':
    main()
