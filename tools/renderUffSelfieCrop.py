"""Render crop-study meshes using the fresh m135_raw camera in original-photo pixels.

Every variant carries its own crop affine. Only m135_raw's affine is used to
convert the frozen camera; a working 512-pixel camera is never shared directly
across different crops. No inference, fitting, texture or variant registration.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import re
import shutil
import sys
from time import perf_counter
import traceback

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from faceAppearance.anatomicalBoundaries import validateBoundaryAsset
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.camera import projectPoints
from renderUffBackgroundAblation import converted_camera, gray_pass, read_json, stack_images, variant_complete
from renderUffMultiCase import (as_image, box_from_points, digest, expand_eye_crop,
    grid, overlay, photo_projection_check, renderer_geometry)

BASELINE = 'm135_raw'
GROUPS = {'m110': ('m110_raw', 'm110_backgroundGray', 'm110_headGray'),
          'm135': ('m135_raw', 'm135_backgroundGray', 'm135_headGray'),
          'm165': ('m165_raw', 'm165_backgroundGray', 'm165_headGray'),
          'shared135': ('shared135_raw', 'shared135_headGray')}
VARIANTS = tuple(name for names in GROUPS.values() for name in names)
LABELS = {name: name.replace('_backgroundGray', ' / background gray').replace('_headGray', ' / head gray').replace('_raw', ' / original') for name in VARIANTS}
PANEL_FILES = ('fixedComparison', 'fixedEyes', 'fixedOverlays', 'inputContact')
PASSES = {'gray': ('clay', 'faceCropXyxy'), 'eyes': ('clay', 'eyeCropXyxy'), 'overlay': ('overlay', 'faceCropXyxy')}


def variant_entries(case):
    entries = {item['id']: item for item in case['variants']}
    return {name: entries[name] for name in VARIANTS if name in entries}


def source_hashes(case):
    paths = [Path(__file__), ROOT / 'assets/anatomical/uff6986/boundaries.json']
    for item in variant_entries(case).values():
        if variant_complete(item):
            source = Path(item['outputDirectory'])
            paths.extend(source / name for name in ('geometry.npz', 'cameras.json', 'inputs.json', 'result.json'))
            paths.extend(sorted(source.glob('input[0-9][0-9].png')))
    return {str(path.resolve()): digest(path) for path in paths}


def load_case(case):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', case['caseId']):
        raise ValueError('Unsafe case identifier')
    entries = variant_entries(case)
    if BASELINE not in entries or not variant_complete(entries[BASELINE]):
        raise ValueError('Fresh m135_raw baseline is not available')
    variants = {}
    for name, item in entries.items():
        if not variant_complete(item):
            continue
        source = Path(item['outputDirectory']).resolve()
        if read_json(source / 'result.json').get('status') != 'succeeded':
            raise ValueError(f'Variant {name} result did not succeed')
        with np.load(source / 'geometry.npz', allow_pickle=False) as arrays:
            geometry = {key: arrays[key] for key in ('vertices', 'faces', 'uvs')}
        v, f, uv = (geometry[key] for key in ('vertices', 'faces', 'uvs'))
        if (v.shape != (6986, 3) or not np.isfinite(v).all() or f.ndim != 2 or f.shape[1] != 3
                or f.size == 0 or f.dtype.kind not in 'iu' or f.min() < 0 or f.max() >= len(v)
                or uv.shape != (6986, 2) or not np.isfinite(uv).all()):
            raise ValueError(f'Invalid native UFF geometry: {name}')
        inputs, cameras = read_json(source / 'inputs.json'), read_json(source / 'cameras.json')
        if not inputs or len(cameras) != len(inputs):
            raise ValueError(f'Invalid camera/input count: {name}')
        variants[name] = {'source': source, 'geometry': geometry, 'inputs': inputs, 'cameras': cameras}
    baseline = variants[BASELINE]
    for name, data in variants.items():
        if any(not np.array_equal(data['geometry'][k], baseline['geometry'][k]) for k in ('faces', 'uvs')):
            raise ValueError(f'{name} topology/UV changed')
        if len(data['inputs']) != len(baseline['inputs']):
            raise ValueError(f'{name} photo count differs')
        for a, b in zip(baseline['inputs'], data['inputs']):
            if any(a[k] != b[k] for k in ('path', 'sha256', 'orientedSize')):
                raise ValueError(f'{name} original photo identity differs')
    return variants


def selected_views(case, cameras):
    front = int(case.get('referenceViewIndex', 0))
    if not 0 <= front < len(cameras):
        raise ValueError('Reference view outside camera list')
    centers = np.array([-np.asarray(c['rotation']).T @ np.asarray(c['translation']).reshape(3) for c in cameras])
    yaws = np.degrees(np.arctan2(centers[:, 0], centers[:, 2]))
    accepted = [i for i, c in enumerate(cameras) if c.get('valid', False)]
    if not accepted:
        raise ValueError('Baseline has no accepted PnP camera')
    side = max(accepted, key=lambda i: abs(yaws[i]))
    return front, side, yaws


def grouped_panels(case_id, index, photo, panels, framing, subtitle, output, *, eyes=False):
    paths = []
    for group, names in GROUPS.items():
        names = [name for name in names if name in panels]
        if not names:
            continue
        images = [photo.crop(framing)] + [as_image(panels[name]).crop(framing) for name in names]
        labels = ['Original photograph'] + [LABELS[name] for name in names]
        path = output.with_name(output.stem + '_' + group + '.png')
        grid(images, labels, 4, panel_size=360, panel_height=210 if eyes else 360,
             subtitle=f'{case_id} / view {index} | {group} | {subtitle}').save(path)
        paths.append(path)
    stack_images(paths, output)


def input_contact(case_id, index, variants, path):
    panels, labels = [], []
    for group, names in GROUPS.items():
        row = []
        for name in names:
            source = variants.get(name, {}).get('source')
            file = source / f'input{index:02d}.png' if source else None
            if file is not None and file.is_file():
                with Image.open(file) as image:
                    panel = image.convert('RGB')
                label = LABELS[name]
            else:
                panel = Image.new('RGB', (512, 512), (22, 27, 33))
                label = name + ' / unavailable'
            row.append((panel, label))
        while len(row) < 3:
            row.append((Image.new('RGB', (512, 512), (22, 27, 33)), ''))
        for panel, label in row:
            panels.append(panel)
            labels.append(label)
    grid(panels, labels, 3, panel_size=330,
         subtitle=f'{case_id} / view {index} | actual saved model inputs').save(path)


def resume_case(case, output, maximum_side):
    if not output.exists() or not any(output.iterdir()):
        return None
    path = output / 'diagnostics.json'
    if not path.is_file():
        raise FileExistsError(f'Incomplete preview preserved: {output}')
    report = read_json(path)
    if (report.get('status') != 'rendered' or report.get('sourceHashes') != source_hashes(case)
            or report.get('maximumPhotoSide') != maximum_side
            or report.get('selectedFrontView') != int(case.get('referenceViewIndex', 0))):
        raise ValueError(f'Existing preview inputs/settings changed: {output}')
    for relative, expected in report['artifactHashes'].items():
        path = (output / relative).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file() or digest(path) != expected:
            raise ValueError(f'Existing preview artifact changed: {path}')
    for view in report['views']:
        if digest(view['sourcePhoto']) != view['photoSha256']:
            raise ValueError('Source photograph changed')
    print(f"{case['caseId']} verified existing preview", flush=True)
    return report


def render_case(case, output, device, maximum_side):
    import torch
    from faceAppearance.pytorch3dRenderer import Pytorch3dFaceRenderer

    start = perf_counter()
    variants = load_case(case)
    baseline = variants[BASELINE]
    inputs, cameras = baseline['inputs'], baseline['cameras']
    front, side, yaws = selected_views(case, cameras)
    boundaries = validateBoundaryAsset(baseline['geometry']['vertices'], baseline['geometry']['faces'],
        read_json(ROOT / 'assets/anatomical/uff6986/boundaries.json'))
    loops = [ids for name, ids in boundaries.items() if 'eye' in name.lower()]
    if len(loops) != 2:
        raise ValueError('Expected two annotated UFF eye openings')
    eye_ids = np.concatenate(loops)
    geometry = {name: renderer_geometry(data['geometry']) for name, data in variants.items()}
    hashes = source_hashes(case)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output must be empty: {output}')
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for index in sorted({front, side}):
        item = inputs[index]
        photo_path = Path(item['path'])
        if digest(photo_path) != item['sha256']:
            raise ValueError('Source photo hash changed')
        with Image.open(photo_path) as image:
            original_photo = ImageOps.exif_transpose(image).convert('RGB')
        if list(original_photo.size) != item['orientedSize']:
            raise ValueError('Source photo dimensions changed')
        factor = min(1., maximum_side / max(original_photo.size))
        size = tuple(max(32, int(round(v * factor))) for v in original_photo.size)
        sx, sy = size[0] / original_photo.width, size[1] / original_photo.height
        resize = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]])
        photo = original_photo.resize(size, Image.Resampling.LANCZOS)
        pixels = np.asarray(photo, np.float32) / 255.
        fixed = converted_camera(cameras[index], item, resize)
        light = fixed['rotation'].T @ np.array([-.45, -.55, -1.])
        passes, camera_reports, projected, projected_eyes = {}, {}, [], []
        for name in variants:
            renderer = Pytorch3dFaceRenderer(geometry[name], device=device, imageSize=(size[1], size[0]), frameGeometry=False)
            renderer.setPerspectiveCamera(**fixed)
            error = photo_projection_check(renderer, geometry[name]['vertices'], fixed)
            rendered = gray_pass(renderer, light)
            rendered['overlay'] = overlay(pixels, rendered['clay'], rendered['visible'])
            passes[name] = rendered
            xy, depth = projectPoints(geometry[name]['vertices'], **fixed)
            projected.append(xy[depth > 0])
            projected_eyes.append(xy[eye_ids][depth[eye_ids] > 0])
            camera_reports[name] = {'maximumProjectionDifferencePixels': error,
                                   'visiblePixels': int(rendered['visible'].sum())}
            del renderer
            gc.collect()
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
        crop = box_from_points(np.concatenate(projected), size, square=True)
        eye_crop = expand_eye_crop(box_from_points(np.concatenate(projected_eyes), size, padding=.3), size)
        view_dir = output / f'view{index:02d}'
        view_dir.mkdir()
        photo.save(view_dir / 'photo.png')
        for name, rendered in passes.items():
            for pass_name in ('clay', 'overlay', 'visible'):
                writeImage(view_dir / f'{name}_fixed_{pass_name}.png', rendered[pass_name])
            for suffix, (pass_name, crop_name) in PASSES.items():
                framing = crop if crop_name == 'faceCropXyxy' else eye_crop
                as_image(rendered[pass_name]).crop(framing).save(view_dir / f'{name}_{suffix}.png')
        for suffix, framing in (('gray', crop), ('eyes', eye_crop), ('overlay', crop)):
            photo.crop(framing).save(view_dir / f'photo_{suffix}.png')
        for filename, pass_name, framing in (('fixedComparison', 'clay', crop), ('fixedEyes', 'clay', eye_crop), ('fixedOverlays', 'overlay', crop)):
            grouped_panels(case['caseId'], index, photo, {name: data[pass_name] for name, data in passes.items()}, framing,
                'FROZEN m135_raw camera in original-photo frame / shared light' + (' / overlay 50%' if pass_name == 'overlay' else ''),
                view_dir / f'{filename}.png', eyes=filename == 'fixedEyes')
        input_contact(case['caseId'], index, variants, view_dir / 'inputContact.png')
        for prefix, selected in (('front', front), ('side', side)):
            if index == selected:
                for name in PANEL_FILES:
                    shutil.copy2(view_dir / f'{name}.png', output / f'{prefix}{name[0].upper() + name[1:]}.png')
        reports.append({'viewIndex': index, 'sourcePhoto': str(photo_path), 'photoSha256': item['sha256'],
            'originalSize': original_photo.size, 'renderSize': size, 'originalToRender': resize,
            'baselineInputToWorking': item['inputToWorking'], 'fixedCamera': fixed,
            'fixedCameraAccepted': bool(cameras[index].get('valid')), 'faceCropXyxy': crop, 'eyeCropXyxy': eye_crop,
            'variants': camera_reports, 'fixedLightCanonical': light, 'freshACameraCenterYawDegrees': float(yaws[index])})
        print(f"{case['caseId']} view {index} rendered {len(variants)} variants", flush=True)
    report = {'caseId': case['caseId'], 'status': 'rendered', 'seconds': perf_counter() - start,
        'sourceHashes': hashes, 'maximumPhotoSide': maximum_side, 'selectedFrontView': front, 'selectedSideView': side,
        'views': reports, 'variantIds': list(variants), 'fixedCameraVariant': BASELINE,
        'fixedCameraFrame': 'Original photograph, then preview resize; m135_raw crop affine inverted',
        'topologyAndUvsUnchanged': True, 'cameraFittingRepeated': False, 'textureRead': False,
        'modelInferenceRepeated': False, 'perVariantRegistration': False,
        'limits': ['Input photographs are not independent geometry ground truth',
            'Only the front reference and largest baseline camera-center yaw are rendered',
            'Fixed camera includes canonical drift; no inter-variant alignment removes this',
            'Gray light is chosen rather than estimated original illumination',
            'Native UFF face surface excludes eyeballs, lashes, mouth cavity, and client head',
            'Overlay holes retain original-photo pixels; compare gray panels too']}
    report['artifactHashes'] = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob('*.png'))}
    writeJson(output / 'diagnostics.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'runs/uffSelfieCropStudy20261012/cropManifest.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/uffSelfieCropStudy20261012/preview')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--maximumPhotoSide', type=int, default=768)
    parser.add_argument('--case', action='append', default=[])
    args = parser.parse_args()
    if args.maximumPhotoSide < 256:
        parser.error('maximumPhotoSide must be at least 256')
    cases = [case for case in read_json(args.manifest)['cases'] if not args.case or case['caseId'] in args.case]
    if not cases or len({case['caseId'] for case in cases}) != len(cases):
        raise ValueError('Empty or duplicate cases')
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    snapshots = args.output / 'manifestSnapshots'
    snapshots.mkdir(exist_ok=True)
    snapshot = snapshots / f'{digest(args.manifest)}.json'
    if not snapshot.exists():
        shutil.copy2(args.manifest, snapshot)
    records = []
    for case in cases:
        try:
            if not variant_complete(variant_entries(case).get(BASELINE, {})):
                record = {'caseId': case['caseId'], 'status': 'notRendered', 'reason': 'Fresh m135_raw baseline unavailable'}
            else:
                output = args.output / case['caseId']
                record = resume_case(case, output, args.maximumPhotoSide) or render_case(case, output, args.device, args.maximumPhotoSide)
        except Exception as error:
            record = {'caseId': case['caseId'], 'status': 'failed', 'error': str(error), 'traceback': traceback.format_exc()}
            print(record['traceback'], flush=True)
        records.append(record)
        writeJson(args.output / 'progress.json', records)
    writeJson(args.output / 'diagnostics.json', {'status': 'completed' if all(r['status'] == 'rendered' for r in records) else 'partial',
        'manifestPath': str(args.manifest.resolve()), 'manifestSha256': digest(args.manifest), 'cases': records})
    if any(record['status'] == 'failed' for record in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
