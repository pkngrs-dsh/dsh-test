"""Independently audit delivered before/after eye apertures against photo masks.

This local experiment includes the two render-confirmed inner-corner accessory
patches, so visible eye material is not confused with the whole eyelid opening.
No fitting, inference, geometry editing, contour completion, or deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

from faceAppearance.artifacts import writeJson
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.visibleEyeContours import (
    loadLegacyEyeBindingInputs, extractVisibleEyeContours, evaluateApertureMask)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=ROOT/'runs/liveJaw20261012/source/reconstruction')
    parser.add_argument('--observations', type=Path, default=ROOT/'runs/denseClientEyes20261012/observations')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'fullApertureComparison.json').exists():
        raise FileExistsError('Choose an output without an existing fullApertureComparison.json')
    inputsPath, camerasPath = args.source/'inputs.json', args.source/'cameras.json'
    inputs, cameras = json.loads(inputsPath.read_text()), json.loads(camerasPath.read_text())
    manifest = json.loads((args.observations/'manifest.json').read_text())
    if (manifest['inputsSha256'] != sha(inputsPath) or manifest['camerasSha256'] != sha(camerasPath)
            or manifest['views'][0]['arraySha256'] != sha(args.observations/'view00.npz')):
        raise ValueError('Photograph observations do not belong to the saved reconstruction')
    matrix = np.asarray(inputs[0]['inputToWorking'], float)
    if (not np.allclose(matrix[2], [0, 0, 1]) or not np.allclose(matrix[:2, :2], np.eye(2)*matrix[0, 0])):
        raise ValueError('Original-image pixel distances require an isotropic input crop')
    with np.load(args.observations/'view00.npz', allow_pickle=False) as z:
        curves = z['eyelidCurvesWorking'].copy()
        if not z['candidateEyeValidMask'].all():
            raise ValueError('Frozen photo eye observations did not pass their original quality gates')
    photoEyes = {name: {'upper': curves[i, 1], 'lower': curves[i, 0]}
                for i, name in enumerate(('imageLeft', 'imageRight'))}
    assets = AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    # Exact audited asset component identities. The names alone do not provide
    # anatomy; this opt-in follows the actual render overlay investigation.
    accessoryNames = ('male00SG.component001', 'male00SG.component002')
    accessories = [c for c in assets.components if c['name'] in accessoryNames]
    if len(accessories) != 2 or any(c.get('label') != 'headAccessory' or c['materialId'] != 0
            or len(c['vertexIndices']) != 12 or len(c['faceIndices']) != 16
            or np.isin(c['vertexIndices'], assets.arrays['nicpToFullIds']).any() for c in accessories):
        raise ValueError('The reviewed inner-corner accessory identities changed')
    apertureFaces = np.concatenate([c['faceIndices'] for c in accessories])
    photo = np.asarray(Image.open(args.observations/'view00WorkingPhoto.png').convert('RGB'))
    reports, panels = {}, []
    for branch in ('before', 'after'):
        archive = args.comparison/branch/'legacyAsset.zip'
        scene = loadLegacyEyeBindingInputs(archive, assets.arrays['nicpToFullIds'])
        if (not np.array_equal(scene['faces'], assets.arrays['faces'])
                or not np.array_equal(scene['materialIds'], assets.arrays['materialIds'])):
            raise ValueError('Delivered topology differs from the reviewed accessory identities')
        result = extractVisibleEyeContours(**scene, camera=cameras[0], photoEyes=photoEyes,
            imageShape=tuple(reversed(inputs[0]['workingSize'])),
            apertureAccessoryMode='innerCornerHeadAccessories', apertureAccessoryFaceIds=apertureFaces)
        report = evaluateApertureMask(result, photoEyes, originalPixelScale=float(matrix[0, 0]))
        report['archiveSha256'] = sha(archive)
        reports[branch] = report
        for name, eye in result['eyes'].items():
            roi, model = eye['roi'], eye['roiMasks']['eye']
            origin, scale = np.asarray(roi['origin']), roi['scale']
            target = photoEyes[name]
            polygon = (np.r_[target['upper'], target['lower'][::-1]]-origin)*scale
            targetMask = np.zeros(model.shape, np.uint8)
            cv2.fillPoly(targetMask, [np.rint(polygon*256).astype(np.int32)], 1, shift=8)
            targetMask = targetMask.astype(bool)
            transform = np.array([[scale, 0, -origin[0]*scale], [0, scale, -origin[1]*scale]])
            crop = cv2.warpAffine(photo, transform, (model.shape[1], model.shape[0]), flags=cv2.INTER_LINEAR)
            overlay = crop.astype(float)
            overlay[model & ~targetMask] = .35*overlay[model & ~targetMask]+.65*np.array([230, 75, 200])
            overlay[targetMask & ~model] = .35*overlay[targetMask & ~model]+.65*np.array([30, 230, 110])
            for mask, color in ((model, (235, 80, 220)), (targetMask, (20, 255, 120))):
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(overlay, contours, -1, color, 1)
            yy, xx = np.nonzero(model | targetMask)
            lo = np.maximum([xx.min()-20, yy.min()-20], 0)
            hi = np.minimum([xx.max()+21, yy.max()+21], model.shape[::-1])
            panel = Image.fromarray(np.rint(overlay).astype(np.uint8)).crop(tuple(np.r_[lo, hi]))
            panel.save(args.output/f'{branch}-{name}-fullApertureOverlay.png')
            Image.fromarray(model).save(args.output/f'{branch}-{name}-visibleApertureMask.png')
            Image.fromarray(targetMask).save(args.output/f'{branch}-{name}-photoApertureMask.png')
            panels.append((branch, name, panel, report['eyes'][name]))
    judgments = {}
    for name in photoEyes:
        before, after = reports['before']['eyes'][name], reports['after']['eyes'][name]
        judgments[name] = {
            'candidateVisibilityAccepted': after['extractionStatus'] == 'accepted',
            'componentCountDidNotIncrease': after['visibleComponentCount'] <= before['visibleComponentCount'],
            'intersectionOverUnionImproved': after['intersectionOverUnion'] > before['intersectionOverUnion'],
            'symmetricMeanImproved': after['symmetricBoundaryMeanOriginalPixels'] < before['symmetricBoundaryMeanOriginalPixels'],
            'boundaryP95DidNotIncrease': after['symmetricBoundaryP95OriginalPixels'] <= before['symmetricBoundaryP95OriginalPixels'],
            'boundaryMaximumDidNotIncrease': after['symmetricBoundaryMaximumOriginalPixels'] <= before['symmetricBoundaryMaximumOriginalPixels'],
        }
    report = {'before': reports['before'], 'after': reports['after'], 'independentChecks': judgments,
        'cameraRefitted': False, 'photoObservationsReestimated': False, 'geometryChanged': False,
        'fullApertureDefinition': 'Visible eye material plus the two render-confirmed inner-corner headAccessory patches; all raster components retained',
        'accessoryIdentities': [{k: c[k] for k in ('name', 'label', 'materialId', 'vertexIndices', 'faceIndices')} for c in accessories],
        'implementationSha256': {'tools/auditClientDenseEyeAperture.py': sha(Path(__file__)),
            'src/faceAppearance/visibleEyeContours.py': sha(ROOT/'src/faceAppearance/visibleEyeContours.py')},
        'sourceHashes': {'inputs': sha(inputsPath), 'cameras': sha(camerasPath),
            'photoObservations': sha(args.observations/'view00.npz')},
        'limitations': ['One male sample and one fixed front camera.',
            'Photo polygons are automatic dense-lid observations, not independent manual ground truth.',
            'Inner-corner accessories are identified by asset identity and actual render position, not anatomical manual annotation.',
            'A good mask score does not prove natural inner-corner geometry or iris/gaze correctness.',
            'Fragmented baseline is measured as it exists; no closing, hull, dilation, or invented continuous contour is applied.']}
    writeJson(args.output/'fullApertureComparison.json', report)
    width, height = 640, 320
    overview = Image.new('RGB', (width*2, height*2+45), (25, 30, 35))
    draw = ImageDraw.Draw(overview)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    draw.text((15, 12), 'Green: automatic photo opening | Magenta: complete delivered opening', font=font, fill=(230, 235, 240))
    for index, (branch, name, panel, metric) in enumerate(panels):
        x, y = (index%2)*width, (index//2)*height+45
        draw.text((x+15, y+10), f'{branch} {name} | IoU {metric["intersectionOverUnion"]:.3f} | mean {metric["symmetricBoundaryMeanOriginalPixels"]:.2f}px | components {metric["visibleComponentCount"]}',font=font,fill=(230,235,240))
        panel.thumbnail((width-30, height-55), Image.Resampling.LANCZOS)
        overview.paste(panel, (x+(width-panel.width)//2, y+45))
    overview.save(args.output/'fullApertureOverview.png')
    print(json.dumps({'before': reports['before']['meanSymmetricBoundaryDistanceOriginalPixels'],
        'after': reports['after']['meanSymmetricBoundaryDistanceOriginalPixels'],
        'checks': judgments, 'output': str(args.output/'fullApertureComparison.json')}, indent=2))


if __name__ == '__main__':
    main()
