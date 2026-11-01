"""Compare one frozen FAN detector on photographs and camera-matched renders.

This produces correspondence evidence only. It never changes fitted cameras,
geometry, texture, or production landmark bindings. Ray hits are diagnostics,
not anatomically validated constraints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools'), str(ROOT/'thirdParty/ibug')]
from faceAppearance.surfaceVisibility import ProjectedSurfaceVisibility
from runAdaptiveChinBatch import archive, serial
from observeClientJawBatch import transform
from ibug.face_alignment import FANPredictor

GROUPS = {'all': np.arange(68), 'jaw': np.arange(17), 'central': np.arange(7, 10),
          'eyebrows': np.arange(17, 27), 'nose': np.arange(27, 36),
          'eyes': np.arange(36, 48), 'mouth': np.arange(48, 68)}
SCALES = np.array([1., .95, 1.05])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(serial(value), ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def detect(detector, path, box, mapping, working_size, face_scale):
    image = Image.open(path).convert('RGB')
    center, half = (box[:2]+box[2:])/2, (box[2:]-box[:2])/2
    boxes = np.array([np.r_[center-half*s, center+half*s] for s in SCALES])
    tick = perf_counter()
    points, peaks = detector(np.asarray(image), boxes, rgb=True)
    working = np.array([transform(p, mapping) for p in points])
    spread = np.linalg.norm(working-working[:1], axis=-1).max(0)
    valid = (np.isfinite(working).all(axis=(0, 2)) & np.isfinite(peaks).all(0)
             & (peaks[0] >= .25) & (peaks.min(0) >= .20) & (spread <= .015*face_scale)
             & (points[0] >= 0).all(1) & (points[0] < image.size).all(1)
             & (working[0] >= 0).all(1) & (working[0] < working_size).all(1))
    return {'points': working[0], 'variants': working, 'peaks': peaks, 'valid': valid,
            'spread': spread, 'seconds': perf_counter()-tick, 'sourceSize': list(image.size)}


def overlay(image, points, valid, *, counterpart=None):
    out = image.copy().convert('RGB')
    draw = ImageDraw.Draw(out)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 10)
    if counterpart is not None:
        for a, b in zip(points, counterpart):
            if np.isfinite([a, b]).all():
                draw.line([tuple(a), tuple(b)], fill=(255, 193, 88), width=1)
                draw.ellipse((b[0]-1.5, b[1]-1.5, b[0]+1.5, b[1]+1.5), fill=(60, 219, 244))
    for i, (x, y) in enumerate(points):
        if not np.isfinite([x, y]).all():
            continue
        color = (77, 243, 138) if valid[i] else (255, 91, 91)
        draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
        if i < 17 or i in (48, 54, 57, 62, 66):
            draw.text((x+3, y-9), str(i), font=font, fill=color, stroke_width=1, stroke_fill='black')
    for ids in (range(17), range(17, 22), range(22, 27), range(27, 31), range(31, 36),
                [36, 37, 38, 39, 40, 41, 36], [42, 43, 44, 45, 46, 47, 42],
                list(range(48, 60))+[48], list(range(60, 68))+[60]):
        xy = points[list(ids)]
        if np.isfinite(xy).all():
            draw.line([tuple(p) for p in xy], fill=(255, 214, 70), width=1)
    return out


def geometric_diagnostics(vertices, faces, camera, shape, points):
    """Continuous skin-ray hits and visible transition edges; no anatomy claims."""
    rotation, translation, intrinsic = [np.asarray(camera[k], float) for k in ('rotation', 'translation', 'intrinsics')]
    cv = vertices@rotation.T+translation
    rays = np.c_[points, np.ones(len(points))]@np.linalg.inv(intrinsic).T
    visibility = ProjectedSurfaceVisibility(cv, faces, intrinsic, tuple(shape))
    hits = visibility.closestDepth(rays)
    valid = hits['faceIds'] >= 0
    positions = np.full((len(points), 3), np.nan)
    barycentrics = np.full_like(positions, np.nan)
    error = np.full(len(points), np.nan)
    if valid.any():
        positions[valid] = rays[valid]*(hits['depth'][valid]/rays[valid, 2])[:, None]
        triangle = cv[faces[hits['faceIds'][valid]]]
        a, b = triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0]
        p = positions[valid]-triangle[:, 0]
        aa, ab, bb, pa, pb = [np.einsum('ij,ij->i', x, y) for x, y in ((a,a),(a,b),(b,b),(p,a),(p,b))]
        denominator = aa*bb-ab*ab
        w1, w2 = (bb*pa-ab*pb)/denominator, (aa*pb-ab*pa)/denominator
        barycentrics[valid] = np.c_[1-w1-w2, w1, w2]
        rebound = np.einsum('ij,ijk->ik', barycentrics[valid], triangle)
        projected = rebound@intrinsic.T
        error[valid] = np.linalg.norm(projected[:, :2]/projected[:, 2:]-points[valid], axis=1)
    canonical = (positions-translation)@rotation
    tri = cv[faces]
    normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    front = np.einsum('ij,ij->i', normals, -tri.mean(1)) > 0
    raw = np.sort(np.concatenate([faces[:, [0,1]], faces[:, [1,2]], faces[:, [2,0]]]), axis=1)
    edges, inverse, counts = np.unique(raw, axis=0, return_inverse=True, return_counts=True)
    owners = np.tile(np.arange(len(faces)), 3)
    edges = edges[(counts == 2) & (np.bincount(inverse, weights=front[owners], minlength=len(edges)) == 1)]
    edges = edges[(cv[edges, 2] > 0).all(1)]
    projected = cv@intrinsic.T
    pixels = projected[:, :2]/projected[:, 2:]
    segments = pixels[edges]
    nearest = np.full((17, 2), np.nan)
    distance = np.full(17, np.nan)
    tolerance = np.ptp(vertices, axis=0).max()*1e-6
    for i, point in enumerate(points[:17]):
        vectors = segments[:, 1]-segments[:, 0]
        fraction = np.clip(np.einsum('ij,ij->i', point-segments[:, 0], vectors)/np.maximum((vectors*vectors).sum(1), 1e-16), 0, 1)
        on_edge = segments[:, 0]+fraction[:, None]*vectors
        distances = np.linalg.norm(on_edge-point, axis=1)
        # Enough candidates to expose whether nearest apparent edge is hidden.
        order = np.argsort(distances)[:40]
        weights = np.c_[1-fraction[order], fraction[order]]/cv[edges[order], 2]
        weights /= weights.sum(1, keepdims=True)
        samples = np.einsum('ij,ijk->ik', weights, cv[edges[order]])
        depth = visibility.closestDepth(samples)['depth']
        keep = np.isfinite(depth) & (abs(depth-samples[:, 2]) <= tolerance)
        if keep.any():
            index = order[np.flatnonzero(keep)[0]]
            nearest[i], distance[i] = on_edge[index], distances[index]
    return {'skinFaceIds': hits['faceIds'], 'skinBarycentric': barycentrics,
            'skinCanonicalPoints': canonical, 'skinRayHitMask': valid,
            'skinReprojectionErrorPixels': error, 'nearestVisibleEdgePixels': nearest,
            'nearestVisibleEdgeDistancePixels': distance}


def group_stats(observed, detected, geometry):
    residual = np.linalg.norm(observed['points']-detected['points'], axis=1)
    groups = {}
    for name, ids in GROUPS.items():
        accepted = detected['valid'][ids]
        common = accepted & observed['valid'][ids]
        boundary = ids[ids < 17]
        boundary = boundary[detected['valid'][boundary]]
        distances = geometry['nearestVisibleEdgeDistancePixels'][boundary]
        distances = distances[np.isfinite(distances)]
        groups[name] = {'count': int(accepted.sum()), 'total': len(ids),
            'photoCount': int(observed['valid'][ids].sum()), 'pairedCount': int(common.sum()),
            'meanPhotoResidualPixels': float(residual[ids][common].mean()) if common.any() else None,
            'medianCropSpreadPixels': float(np.median(detected['spread'][ids])),
            'maximumCropSpreadPixels': float(detected['spread'][ids].max()),
            'skinRayHitCount': int((accepted & geometry['skinRayHitMask'][ids]).sum()),
            'meanBoundaryDistancePixels': float(distances.mean()) if len(distances) else None}
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--renders', type=Path, default=ROOT/'runs/clientRenderedSemantics20261012/renderInputs/manifest.json')
    parser.add_argument('--cases', type=Path, default=ROOT/'runs/clientChinAdaptive20261012/cases.json')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/clientRenderedSemantics20261012/evaluation')
    parser.add_argument('--case', action='append')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    model = FANPredictor.get_model('2dfan2_alt')
    model.weights = str(ROOT/'assets/models/ibug/2dfan2_alt.pth')
    detector = FANPredictor(device='cpu', model=model, config=FANPredictor.create_config(use_jit=False))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.renders.read_text())
    cases = {c['caseId']: c for c in json.loads(args.cases.read_text())['cases']}
    report = {'schemaVersion': 1, 'status': 'running', 'rows': [], 'photoControls': [], 'variantComparisons': [],
        'renderManifestPath': str(args.renders.resolve()), 'renderManifestSha256': sha(args.renders),
        'caseManifestSha256': sha(args.cases), 'implementationSha256': sha(__file__),
        'modelPath': model.weights, 'modelSha256': sha(model.weights),
        'methods': {'detector': 'FAN 2dfan2_alt, same weights and full saved RetinaFace bbox as photos',
            'boxVariants': SCALES.tolist(), 'minimumPeak': .25, 'minimumAllVariantPeak': .20,
            'maximumSpreadFaceFraction': .015, 'primaryInference': 'Native original resolution renders; same original photo bbox',
            'coordinateUnits': 'All reported pixel distances in saved working-image coordinates',
            'photoControl': 'Working-photo reinference measures crop/resampling change against frozen original-photo observations',
            'geometry': 'Frozen before/after geometries; no fitting in this experiment'},
        'limitations': ['Automatic photo detections are not manual anatomical ground truth.',
            'Heatmap peaks and crop stability are heuristics, not calibrated confidence.',
            'Low photo/render landmark residual does not establish correct 3D geometry.',
            'Skin-only ray hits may land on neck or behind accessories; none are approved constraints.',
            'Nearest visible edge is geometric, not a certified anatomical jaw boundary.',
            'Photo-derived texture shares evidence with photo observations and may bias the detector.'],
        'remoteModified': False, 'geometryModified': False}
    started = perf_counter()
    comparison_data = {}
    for rendered_case in manifest['cases']:
        case_id = rendered_case['caseId']
        if args.case and case_id not in args.case:
            continue
        case = cases[case_id]
        cached = archive(case['observationPath'])
        inputs = json.loads(Path(case['inputsPath']).read_text())
        cameras = json.loads(Path(case['cameraPath']).read_text())
        base_vertices = archive(case['geometryPath'])[case['verticesKey']]
        faces = archive(case['facesPath'])[case['facesKey']]
        controls = set()
        for stage in rendered_case['stages']:
            stage_name = stage['stage']
            vertices = base_vertices if stage_name == 'before' else archive(ROOT/'runs/clientChinAdaptive20261012/evaluationV1'/case_id/'geometry.npz')['after']
            for view in stage['views']:
                index = view['viewIndex']
                record = inputs[index]
                mapping = np.asarray(record['inputToWorking'], float)
                box = np.asarray(record['faceDetection']['boxXyxy'], float)
                scale = float(cached['faceScaleWorkingPixels'][index])
                observed = {'points': cached['faceLandmarksWorking'][index],
                            'valid': cached['landmarkCandidateValidMask'][index]}
                photo_path = Path(case['observationPath']).parent/f'view{index}-photo.png'
                photo = Image.open(photo_path).convert('RGB')
                common_dir = args.output/case_id
                common_dir.mkdir(exist_ok=True)
                photo_name = f'{case_id}/view{index}-photo.png'
                photo_overlay_name = f'{case_id}/view{index}-photoOverlay.png'
                photo.save(args.output/photo_name)
                overlay(photo, observed['points'], observed['valid']).save(args.output/photo_overlay_name)
                if index not in controls:
                    control = detect(detector, photo_path, transform(box.reshape(2, 2), mapping).reshape(-1),
                                     np.eye(3), record['workingSize'], scale)
                    delta = np.linalg.norm(control['points']-observed['points'], axis=1)
                    report['photoControls'].append({'caseId': case_id, 'viewIndex': index,
                        'chinDifferencePixels': float(delta[8]), 'jawMeanDifferencePixels': float(delta[:17].mean()),
                        'allMeanDifferencePixels': float(delta.mean()), 'workingInferenceSeconds': control['seconds'],
                        'sourceOriginalSize': record['orientedSize'], 'workingSize': record['workingSize']})
                    controls.add(index)
                for variant, image_path in view['renders'].items():
                    original_path = view.get('rendersOriginal', {}).get(variant)
                    if original_path is None:
                        raise ValueError(f'Native render missing: {case_id} {stage_name} {index} {variant}')
                    result = detect(detector, original_path, box, mapping, record['workingSize'], scale)
                    if result['sourceSize'] != record['orientedSize']:
                        raise ValueError('Original render dimensions mismatch')
                    geometry = geometric_diagnostics(vertices, faces, cameras[index], photo.size[::-1], result['points'])
                    key = f'{stage_name}-view{index}-{variant}'
                    relative = f'{case_id}/{key}'
                    image = Image.open(image_path).convert('RGB')
                    image.save(args.output/f'{relative}.png')
                    rendered_overlay = overlay(image, result['points'], result['valid'], counterpart=observed['points'])
                    rendered_overlay.save(args.output/f'{relative}-overlay.png')
                    board = Image.new('RGB', (photo.width*2, photo.height+40), (20, 27, 35))
                    board.paste(Image.open(args.output/photo_overlay_name), (0, 40))
                    board.paste(rendered_overlay, (photo.width, 40))
                    ImageDraw.Draw(board).text((12, 12), f'{case_id} / {stage_name} / view {index+1} / {variant}', fill='white')
                    board.save(args.output/f'{relative}-comparison.png')
                    np.savez_compressed(args.output/f'{relative}.npz', **{k:v for k,v in result.items() if isinstance(v, np.ndarray)},
                        photoPoints=observed['points'], photoValid=observed['valid'], **geometry)
                    row = {'caseId': case_id, 'stage': stage_name, 'viewIndex': index, 'variant': variant,
                        'sourceRenderPath': original_path, 'sourceRenderSha256': sha(original_path),
                        'sourcePhotoObservationSha256': sha(case['observationPath']), 'faceScaleWorkingPixels': scale,
                        'groups': group_stats(observed, result, geometry),
                        'chin': {'photo': observed['points'][8], 'render': result['points'][8],
                            'residualY': float(result['points'][8, 1]-observed['points'][8, 1]),
                            'boundaryDistancePixels': float(geometry['nearestVisibleEdgeDistancePixels'][8]),
                            'renderValid': bool(result['valid'][8]), 'photoValid': bool(observed['valid'][8]),
                            'skinRayHit': bool(geometry['skinRayHitMask'][8]),
                            'nearestBoundaryPixel': geometry['nearestVisibleEdgePixels'][8]},
                        'display': {'photo': photo_name, 'photoOverlay': photo_overlay_name,
                            'render': f'{relative}.png', 'renderOverlay': f'{relative}-overlay.png',
                            'comparison': f'{relative}-comparison.png'}, 'npzPath': f'{relative}.npz',
                        'seconds': result['seconds'], 'allRayReprojectionMaxPixels': float(np.nanmax(geometry['skinReprojectionErrorPixels']))}
                    report['rows'].append(row)
                    comparison_data[(case_id, stage_name, index, variant)] = result
                    write(args.output/'report.json', report)
                    print(json.dumps(serial({k: row[k] for k in ('caseId','stage','viewIndex','variant','chin')})), flush=True)
    keys = sorted({key[:3] for key in comparison_data})
    for case_id, stage, view in keys:
        for first, second in [('gray','template'), ('gray','grayAlternate'), ('template','photoDerived')]:
            a, b = comparison_data.get((case_id,stage,view,first)), comparison_data.get((case_id,stage,view,second))
            if a is None or b is None:
                continue
            delta = np.linalg.norm(a['points']-b['points'], axis=1)
            common = a['valid'] & b['valid']
            report['variantComparisons'].append({'caseId':case_id,'stage':stage,'viewIndex':view,'first':first,'second':second,
                'chinDifferencePixels':float(delta[8]), 'chinPairedValid':bool(common[8]),
                'groups':{name:{'pairedCount':int(common[ids].sum()),
                    'meanDifferencePixels':float(delta[ids][common[ids]].mean()) if common[ids].any() else None,
                    'maximumDifferencePixels':float(delta[ids][common[ids]].max()) if common[ids].any() else None}
                    for name,ids in GROUPS.items()}})
    report.update(status='complete', seconds=perf_counter()-started,
        detectionSeconds=sum(row['seconds'] for row in report['rows']),
        photoControlSeconds=sum(row['workingInferenceSeconds'] for row in report['photoControls']))
    write(args.output/'report.json', report)
    print(json.dumps({'status':'complete','rows':len(report['rows']),'seconds':report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
