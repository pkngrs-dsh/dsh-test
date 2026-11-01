"""Read frozen RGBA skin renders and measure reliably located eye openings.

The measured shape is a visible background hole in the SKIN-only raster. It
excludes occluded structural rim segments, but is not the complete eye after
eyeballs/accessories are added. STFR photograph polygons remain automatic
proxies, not manually established anatomical truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion, distance_transform_edt, label

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.artifacts import writeJson
from diagnoseRegistrationTransfer20261013 import summary

CASES = ('currentMale', 'highPitch4', 'cyanWall4', 'outdoorSideLight4')
STAGES = ('uff', 'direct', 'stable', 'final')
EYES = ('imageLeft', 'imageRight')
LABELS = dict(photo='Photo / STFR', uff='UFF skin', direct='Pure registration',
              stable='Existing refinement', final='Fast-eye final')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observation_paths(case):
    if case == 'currentMale':
        return (ROOT / 'runs/denseClientEyes20261012/observations',
                ROOT / 'runs/liveJaw20261012/source/reconstruction')
    base = ROOT / 'runs/eyelidTextureBatch20261012' / case
    return base / 'observations', base / 'baseline/reconstruction'


def outline(mask):
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)


def mask_geometry(mask):
    y, x = np.nonzero(mask)
    if not len(x):
        return {'areaWorkingPixelsSquared': 0}
    width = int(x.max() - x.min() + 1)
    columns = mask.sum(axis=0)
    columns = columns[columns > 0]
    return dict(areaWorkingPixelsSquared=int(mask.sum()),
                boundingBoxXyxyInclusive=[int(x.min()), int(y.min()), int(x.max()), int(y.max())],
                centroidWorkingPixels=[float(x.mean()), float(y.mean())],
                boundingWidthWorkingPixels=width,
                boundingHeightWorkingPixels=int(y.max() - y.min() + 1),
                meanVerticalThicknessWorkingPixels=float(mask.sum() / width),
                maximumVerticalThicknessWorkingPixels=int(columns.max()),
                medianNonemptyColumnThicknessWorkingPixels=float(np.median(columns)),
                meanVerticalThicknessToWidth=float(mask.sum() / width ** 2))


def mask_distance(model, photo, original_scale):
    model_edge, photo_edge = outline(model), outline(photo)
    to_photo = distance_transform_edt(~photo_edge)[model_edge]
    to_model = distance_transform_edt(~model_edge)[photo_edge]
    both = np.r_[to_photo, to_model]
    return dict(
        intersectionOverUnion=float((model & photo).sum() / (model | photo).sum()),
        photoAreaRatio=float(model.sum() / photo.sum()),
        symmetricBoundaryMeanWorkingPixels=float((to_photo.mean() + to_model.mean()) / 2),
        symmetricBoundaryMeanOriginalPixels=float((to_photo.mean() + to_model.mean()) / (2 * original_scale)),
        symmetricBoundaryP95WorkingPixels=float(np.percentile(both, 95)),
        symmetricBoundaryMaximumWorkingPixels=float(both.max()),
        modelToPhotoBoundaryDistanceWorkingPixels=summary(to_photo),
        photoToModelBoundaryDistanceWorkingPixels=summary(to_model))


def photo_polygon(curves, shape):
    # Saved order: lower then upper; both run across the eye in the same order.
    points = np.r_[curves[1], curves[0][::-1]]
    result = np.zeros(shape, np.uint8)
    cv2.fillPoly(result, [np.rint(points * 256).astype(np.int32)], 1, shift=8)
    return result.astype(bool), points


def components(alpha):
    labels, count = label(alpha < 128, structure=np.ones((3, 3)))
    exterior = set(np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]]).tolist())
    result = []
    for ident in range(1, count + 1):
        if ident in exterior:
            continue
        mask = labels == ident
        result.append(dict(id=ident, mask=mask, **mask_geometry(mask)))
    return result, sorted(exterior)


def locate_eye(available, polygons, eye):
    target = polygons[eye]
    other = polygons[1 - eye]
    lo, hi = target.min(axis=0), target.max(axis=0)
    center, other_center = (lo + hi) / 2, (other.min(axis=0) + other.max(axis=0)) / 2
    width = float(hi[0] - lo[0])
    candidates, evaluated = [], []
    for component in available:
        box = component['boundingBoxXyxyInclusive']
        c = np.asarray(component['centroidWorkingPixels'])
        center_distance = float(np.linalg.norm(c - center))
        horizontal_overlap = max(0., min(box[2] + 1, hi[0]) - max(box[0], lo[0])) / width
        reasons = []
        if component['areaWorkingPixelsSquared'] < 4:
            reasons.append('fewerThanFourPixels')
        if not .25 * width <= component['boundingWidthWorkingPixels'] <= 1.6 * width:
            reasons.append('implausibleWidth')
        if component['boundingHeightWorkingPixels'] > .8 * width:
            reasons.append('implausibleHeight')
        if center_distance > .65 * width:
            reasons.append('farFromPhotoEyeCenter')
        if horizontal_overlap < .3:
            reasons.append('insufficientHorizontalEyeOverlap')
        if center_distance >= np.linalg.norm(c - other_center):
            reasons.append('closerToOtherPhotoEye')
        if not (box[0] >= lo[0] - .6 * width and box[2] <= hi[0] + .6 * width
                and box[1] >= center[1] - .75 * width and box[3] <= center[1] + .75 * width):
            reasons.append('outsideLocalEyeRegion')
        evaluated.append(dict(componentId=component['id'], reasons=reasons,
                              centerDistanceWorkingPixels=center_distance,
                              horizontalPhotoOverlapFraction=horizontal_overlap))
        if not reasons:
            candidates.append(component)
    if len(candidates) != 1:
        return None, dict(status='rejected', reason='noUniquePlausibleInternalComponent',
                          plausibleComponentIds=[c['id'] for c in candidates], candidates=evaluated)
    return candidates[0], dict(status='acceptedUniqueInternalComponent',
                               componentId=candidates[0]['id'], candidates=evaluated)


def overlay_grid(case, photo, images, masks, target_masks, polygons, records, out):
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 15)
    small = ImageFont.truetype(font_path, 12)
    panel_width, panel_height = 320, 214
    canvas = Image.new('RGB', (panel_width * 5, panel_height * 2 + 65), (21, 27, 38))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 9), f'{case} | front visible SKIN-only openings; fixed camera', font=font, fill='white')
    draw.text((12, 34), 'Magenta = automatic photo polygon; cyan = measured visible hole. No eyeballs / corner accessories.', font=small, fill=(190, 205, 220))
    for eye, eye_name in enumerate(EYES):
        points = polygons[eye]
        center = (points.min(0) + points.max(0)) / 2
        span = points[:, 0].max() - points[:, 0].min()
        box = [int(center[0] - span * .8), int(center[1] - span * .5),
               int(center[0] + span * .8), int(center[1] + span * .5)]
        for column, stage in enumerate(('photo', *STAGES)):
            x, y = column * panel_width, 65 + eye * panel_height
            if stage == 'photo':
                pixels = np.asarray(photo.convert('RGB')).copy()
                footer = 'Automatic reference, not manual truth'
            else:
                rgba = np.asarray(images[stage])
                pixels = np.full(rgba.shape[:2] + (3,), 26, dtype=np.uint8)
                pixels[rgba[..., 3] >= 128] = rgba[..., :3][rgba[..., 3] >= 128]
                mask = masks.get((stage, eye_name))
                if mask is not None:
                    pixels[mask] = (31, 104, 120)
                    pixels[outline(mask)] = (80, 240, 245)
                    result = records[stage]['eyes'][eye_name]
                    g, d = result['geometry'], result['photoComparison']
                    footer = f"A={g['areaWorkingPixelsSquared']} maxH={g['maximumVerticalThicknessWorkingPixels']} W={g['boundingWidthWorkingPixels']} IoU={d['intersectionOverUnion']:.3f}"
                else:
                    footer = 'REJECTED: no reliable component'
            pixels[outline(target_masks[eye])] = (255, 86, 192)
            panel = Image.fromarray(pixels).crop(box)
            factor = min((panel_width - 12) / panel.width, (panel_height - 50) / panel.height)
            panel = panel.resize((round(panel.width * factor), round(panel.height * factor)), Image.Resampling.NEAREST)
            canvas.paste(panel, (x + (panel_width - panel.width) // 2, y + 25))
            draw.text((x + 8, y + 2), f'{eye_name} | {LABELS[stage]}', font=small, fill='white')
            draw.text((x + 8, y + panel_height - 20), footer, font=small, fill=(205, 220, 230))
    canvas.save(out / 'overlay.png')


def run(case, base):
    renders, out = base / 'renders' / case, base / 'apertures' / case
    out.mkdir(parents=True, exist_ok=True)
    obs_dir, source = observation_paths(case)
    observations = obs_dir / 'view00.npz'
    with np.load(observations, allow_pickle=False) as archive:
        curves = archive['eyelidCurvesWorking'].copy()
        valid = archive['candidateEyeValidMask'].copy()
    inputs = json.loads((source / 'inputs.json').read_text(encoding='utf-8'))
    render_report = json.loads((renders / 'renderReport.json').read_text(encoding='utf-8'))
    obs_manifest = json.loads((obs_dir / 'manifest.json').read_text(encoding='utf-8'))
    for key, path in (('camerasSha256', source / 'cameras.json'), ('inputsSha256', source / 'inputs.json')):
        if render_report[key] != sha(path) or obs_manifest[key] != sha(path):
            raise ValueError('Photo observations, camera and renders do not share the same frame')
    images = {stage: Image.open(renders / f'view0-{stage}.png').convert('RGBA') for stage in STAGES}
    shape = np.asarray(images['uff']).shape[:2]
    if any(np.asarray(image).shape[:2] != shape for image in images.values()):
        raise ValueError('Raster sizes differ')
    if list(reversed(shape)) != inputs[0]['workingSize']:
        raise ValueError('Render is not in saved photo working-pixel coordinates')
    target_masks, polygons = zip(*(photo_polygon(curves[i], shape) for i in range(2)))
    scale = float(np.asarray(inputs[0]['inputToWorking'])[0, 0])
    records, saved_masks = {}, {}
    for stage, image in images.items():
        alpha = np.asarray(image)[..., 3]
        available, exterior = components(alpha)
        row = dict(sourceRenderSha256=sha(renders / f'view0-{stage}.png'),
                   alphaValues=np.unique(alpha).tolist(), exteriorComponentIds=exterior,
                   internalComponents=[{k: v for k, v in c.items() if k != 'mask'} for c in available], eyes={})
        selected_ids = []
        for eye, name in enumerate(EYES):
            if not valid[eye]:
                row['eyes'][name] = dict(selection=dict(status='rejected', reason='photoEyeObservationRejected'))
                continue
            component, selection = locate_eye(available, polygons, eye)
            result = dict(selection=selection)
            if component is not None:
                mask = component['mask']
                selected_ids.append(component['id'])
                saved_masks[stage, name] = mask
                result.update(geometry=mask_geometry(mask), photoComparison=mask_distance(mask, target_masks[eye], scale))
                result['geometry']['areaOriginalPixelsSquared'] = float(mask.sum() / scale ** 2)
            row['eyes'][name] = result
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError('A component was assigned to both eyes')
        records[stage] = row
    stage_ratios = {}
    for name in EYES:
        stage_ratios[name] = {}
        for before, after in (('uff', 'direct'), ('direct', 'stable'), ('stable', 'final'), ('uff', 'final')):
            first, second = records[before]['eyes'][name], records[after]['eyes'][name]
            if 'geometry' not in first or 'geometry' not in second:
                continue
            ga, gb = first['geometry'], second['geometry']
            stage_ratios[name][before + 'To' + after.capitalize()] = {
                field: float(gb[field] / ga[field]) for field in (
                    'areaWorkingPixelsSquared', 'boundingWidthWorkingPixels',
                    'maximumVerticalThicknessWorkingPixels', 'meanVerticalThicknessWorkingPixels')}
    report = dict(caseId=case, status='measuredExistingRgbaSkinRenders',
        sourceObservation=str(observations), sourceObservationSha256=sha(observations),
        sourceRenderReportSha256=sha(renders / 'renderReport.json'), sourceRenderDirectory=str(renders),
        photoSha256=inputs[0]['sha256'], cameraRefitted=False, geometryFitted=False,
        workingResolution=list(reversed(shape)), originalToWorkingScale=scale,
        scope='Visible alpha-background components fully enclosed by skin-only renders. Eye balls, teeth, tongue, corner accessories and brow/lash cards are absent.',
        connectivity=8, alphaBackgroundThreshold=128,
        selectionRule='Exactly one plausible internal background component per valid automatic photo eye; no forced nearest component, no hole filling, no joining disconnected pieces.',
        photoReference='Saved STFR lower/upper 12-point automatic eye curves rasterized with 1/256 pixel fixed-point polygon coordinates; not manual ground truth.',
        referenceGeometry={name: mask_geometry(target_masks[i]) for i, name in enumerate(EYES)},
        branches=records, stageRatios=stage_ratios,
        limitations=['Measures actual projected skin coverage, not the full assembled visible eye aperture after eyeballs/accessories.',
                     '512 binary raster incurs roughly one-pixel boundary quantization; small thickness differences should not be overinterpreted.',
                     'Photo outlines are automatic proxies; IoU and distances quantify agreement with that proxy.',
                     'UFF and client topology differ. A change in enclosed visible hole can reflect rim location or occluding interior skin surfaces; it does not identify a unique deformation cause.',
                     'Image-left/right names are image positions, independent of subject-side anatomical labels.'])
    writeJson(out / 'report.json', report)
    np.savez_compressed(out / 'masks.npz', **{stage + '_' + name: mask for (stage, name), mask in saved_masks.items()},
                        **{name + '_photo': target_masks[i] for i, name in enumerate(EYES)})
    overlay_grid(case, Image.open(renders / 'view0-photo.png'), images, saved_masks, target_masks, polygons, records, out)
    print(json.dumps(dict(caseId=case, report=str(out / 'report.json'),
                         acceptedEyeMeasurements=len(saved_masks)), ensure_ascii=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--root', type=Path, default=ROOT / 'runs/registrationTransfer20261013')
    args = parser.parse_args()
    reports = [run(case, args.root.resolve()) for case in args.cases]
    writeJson(args.root.resolve() / 'apertures/summary.json', dict(
        scope='Actual visible skin-only eye holes from frozen front RGBA renders; not the complete assembled eye.',
        cases=[dict(caseId=r['caseId'], stageRatios=r['stageRatios']) for r in reports]))


if __name__ == '__main__':
    main()
