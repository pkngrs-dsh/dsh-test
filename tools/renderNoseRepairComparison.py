"""Render pinned camera, geometry and appearance experiments independently."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.legacyPreviewAsset import _load
from renderClientAppearanceEvidence import EvidenceRenderer, evidenceGrid, solid
from renderFeedbackAppearanceComparison import saveSrgb

RENDER_MODES = ('baseColor', 'clay', 'lit')


def fileHash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scaledViewCamera(camera, metadata, index, size):
    if not camera.get('valid') or camera.get('viewIndex', index) != index:
        raise ValueError('Invalid camera or view order')
    sx, sy = size / metadata['workingSize'][0], size / metadata['workingSize'][1]
    pixelTransform = np.array([[sx, 0, .5 * sx - .5], [0, sy, .5 * sy - .5], [0, 0, 1]])
    scaled = {key: np.asarray(camera[key], float) for key in ('intrinsics', 'rotation', 'translation')}
    for key, shape in (('intrinsics', (3, 3)), ('rotation', (3, 3)), ('translation', (3,))):
        if scaled[key].shape != shape or not np.isfinite(scaled[key]).all():
            raise ValueError(f'Invalid camera {key}')
    scaled['intrinsics'] = pixelTransform @ scaled['intrinsics']
    return scaled


def loadCachedBranch(row, cameras, inputs, *, size, sourceTaskId):
    """Load only a completed, matching render branch; never invoke a renderer.

    Older reports did not record PNG hashes. Their hashes are captured now;
    when a source report has image hashes, they must match as well. Opaque RGB
    PNGs are expanded to RGBA so another call to ``solid`` preserves pixels.
    """
    branchId = row['id']
    if re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]*', branchId) is None:
        raise ValueError('Unsafe cached branch ID')
    directory = Path(row['cachedRenderDirectory']).resolve()
    reportPath = directory / 'report.json'
    reportHash = fileHash(reportPath)
    report = json.loads(reportPath.read_text())
    if (report.get('status') != 'complete' or report.get('size') != size
            or report.get('sourceTaskId') != sourceTaskId or report.get('device') != 'cpu'
            or report.get('lightFrame') != 'cameraRelative'
            or report.get('photoIlluminationFitted') is not False
            or report.get('topologyUvMaterialOrderBindingIdentical') is not True
            or report.get('renderer', 'renderNoseRepairComparison') != 'renderNoseRepairComparison'):
        raise ValueError('Cached render report does not match this comparison')
    previous = report.get('branches', {}).get(branchId)
    if previous is None or any(previous.get(key) != row[key] for key in ('archiveSha256', 'cameraSha256')):
        raise ValueError('Cached branch archive/camera identity mismatch')
    if (fileHash(row['archive']) != row['archiveSha256']
            or fileHash(row['cameraPath']) != row['cameraSha256']):
        raise ValueError('Pinned cached archive/camera changed')
    if len(report.get('views', [])) != len(inputs) or len(cameras) != len(inputs):
        raise ValueError('Cached camera/view count mismatch')
    views, imageIdentities = [], []
    for index, (camera, metadata, view) in enumerate(zip(cameras, inputs, report['views'])):
        if view.get('viewIndex') != index:
            raise ValueError('Cached render view order mismatch')
        expectedCamera = scaledViewCamera(camera, metadata, index, size)
        previousCamera = view.get('cameras', {}).get(branchId, {})
        for key, value in expectedCamera.items():
            if not np.array_equal(value, np.asarray(previousCamera.get(key), float)):
                raise ValueError(f'Cached used camera mismatch in view {index}: {key}')
        if branchId not in view.get('stats', {}):
            raise ValueError('Cached render statistics are missing')
        images, identities = {}, {}
        for mode in RENDER_MODES:
            path = directory / f'view{index:02d}_{branchId}_{mode}.png'
            digest = fileHash(path)
            recordedHash = view.get('imageSha256', {}).get(branchId, {}).get(mode)
            if recordedHash is not None and recordedHash != digest:
                raise ValueError(f'Cached PNG hash mismatch: {path.name}')
            with Image.open(path) as picture:
                if picture.format != 'PNG' or picture.mode != 'RGB' or picture.size != (size, size):
                    raise ValueError(f'Cached PNG must be opaque RGB at the requested size: {path.name}')
                images[mode] = picture.convert('RGBA')
            identities[mode] = {'path': str(path), 'sha256': digest,
                                'verifiedAgainstSourceReport': recordedHash is not None}
        views.append({'images': images, 'stats': view['stats'][branchId]})
        imageIdentities.append({'viewIndex': index, 'images': identities})
    if fileHash(reportPath) != reportHash:
        raise ValueError('Cached report changed while loading')
    return {'views': views, 'provenance': {
        'directory': str(directory), 'reportPath': str(reportPath), 'reportSha256': reportHash,
        'branchId': branchId, 'size': size, 'cameraArraysExactlyMatched': True,
        'sourcePngMode': 'RGB', 'loadedAs': 'RGBA with opaque alpha', 'views': imageIdentities,
    }}


def noseGrid(panels, labels, crop, title):
    size, header = 256, 76
    result = Image.new('RGB', (size * len(panels), size + header), (22, 27, 33))
    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 15)
    draw.text((12, 10), title, font=font, fill='white')
    for index, (panel, label) in enumerate(zip(panels, labels)):
        draw.text((index * size + 10, 43), label, font=font, fill='white')
        picture = ImageOps.contain(panel.crop(crop).convert('RGB'), (size, size), Image.Resampling.LANCZOS)
        result.paste(picture, (index * size + (size - picture.width) // 2,
                              header + (size - picture.height) // 2))
    return result


def run(configPath, outputDirectory, size=512):
    configPath, output = Path(configPath), Path(outputDirectory)
    config = json.loads(configPath.read_text())
    case = Path(config['caseDirectory'])
    entries = config['branches']
    if not 2 <= len(entries) <= 5 or len({row['id'] for row in entries}) != len(entries):
        raise ValueError('Expected two to five unique comparison branches')
    if any(re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_]*', row['id']) is None for row in entries):
        raise ValueError('Unsafe branch ID')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh comparison directory')
    inputs = json.loads((case / 'reconstruction/inputs.json').read_text())
    sourceCameras = case / 'reconstruction/cameras.json'
    baseline = _load(entries[0]['archive'])
    similarity = baseline.manifest['assemblyDiagnostics']['templateToUff']
    renderers, cameras, identities, cachedBranches = {}, {}, {}, {}
    for row in entries:
        path = Path(row['archive'])
        cameraPath = Path(row.get('cameraPath', sourceCameras))
        if fileHash(path) != row['archiveSha256'] or fileHash(cameraPath) != row['cameraSha256']:
            raise ValueError('Pinned experiment archive/camera changed')
        asset = _load(path)
        for name in baseline.mesh:
            if name != 'vertices':
                np.testing.assert_array_equal(asset.mesh[name], baseline.mesh[name])
        if asset.manifest['binding'] != baseline.manifest['binding']:
            raise ValueError('Client binding changed')
        if asset.manifest['assemblyDiagnostics']['templateToUff'] != similarity:
            raise ValueError('Canonical assembly frame changed')
        if row.get('appearanceFrozen', True):
            for name, payload in baseline.payloads.items():
                if name != 'face_nicp.obj' and asset.payloads[name] != payload:
                    raise ValueError(f'Frozen appearance changed in {row["id"]}: {name}')
        cameras[row['id']] = json.loads(cameraPath.read_text())
        if len(cameras[row['id']]) != len(inputs):
            raise ValueError('Camera count mismatch')
        identities[row['id']] = {**row, 'cameraPath': str(cameraPath),
            'cameraChanged': fileHash(cameraPath) != fileHash(sourceCameras)}
        if row.get('cachedRenderDirectory'):
            cachedBranches[row['id']] = loadCachedBranch(
                identities[row['id']], cameras[row['id']], inputs, size=size, sourceTaskId=case.name)
            identities[row['id']]['renderCache'] = cachedBranches[row['id']]['provenance']
        else:
            key = row['archiveSha256']
            if key not in renderers:
                renderers[key] = EvidenceRenderer(path, similarity, device='cpu', facesPerPixel=12)
    cameraDescription = ('仅相机分支改变外参，其余使用原保存相机'
        if any(row['cameraChanged'] for row in identities.values()) else '共用原保存相机')
    output.mkdir(parents=True, exist_ok=True)
    with np.load(case / 'reconstruction/directClient/renderedJaw/photoObservations.npz', allow_pickle=False) as obs:
        photoPoints = obs['faceLandmarksWorking'].copy()
    views = []
    for index, metadata in enumerate(inputs):
        photoPath = case / 'reconstruction' / f'input{index:02d}.png'
        with Image.open(photoPath) as photoImage:
            if list(photoImage.size) != metadata['workingSize']:
                raise ValueError('Working image dimensions changed')
            photo = photoImage.convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
        sx, sy = size / metadata['workingSize'][0], size / metadata['workingSize'][1]
        images, stats, usedCameras, imageHashes = {}, {}, {}, {}
        for row in entries:
            camera = cameras[row['id']][index]
            scaled = scaledViewCamera(camera, metadata, index, size)
            if row['id'] in cachedBranches:
                cached = cachedBranches[row['id']]['views'][index]
                images[row['id']], stats[row['id']] = cached['images'], cached['stats']
            else:
                images[row['id']], stats[row['id']] = renderers[row['archiveSha256']].variants(scaled, (size, size))
            usedCameras[row['id']] = scaled
            imageHashes[row['id']] = {}
            for mode, image in images[row['id']].items():
                imagePath = output / f'view{index:02d}_{row["id"]}_{mode}.png'
                saveSrgb(solid(image), imagePath)
                imageHashes[row['id']][mode] = fileHash(imagePath)
        points = photoPoints[index, 27:36] * [sx, sy]
        center = (points.min(0) + points.max(0)) / 2
        half = max(float(np.ptp(points, axis=0).max()) * .7, 50. * sx)
        crop = tuple(np.r_[np.floor(center - half), np.ceil(center + half)].astype(int))
        fullPanels, fullLabels = [], []
        for mode, name in [('baseColor', '基色'), ('clay', '灰模'), ('lit', '检查光照')]:
            panels = [photo] + [solid(images[row['id']][mode]) for row in entries]
            labels = ['原工作照片'] + [row['label'] for row in entries]
            fullPanels.extend(panels)
            fullLabels.extend([f'{label} · {name}' if i else label for i, label in enumerate(labels)])
            saveSrgb(noseGrid(panels, labels, crop,
                f'视角{index + 1} · {name} · {cameraDescription}'),
                output / f'view{index:02d}_{mode}_nose.png')
        saveSrgb(evidenceGrid(fullPanels, fullLabels, columns=len(entries) + 1, panelSize=size,
            subtitle='局部修复实验；各分支状态见标签；灰模/光照为共用相机相对检查灯'),
            output / f'view{index:02d}Comparison.png')
        views.append({'viewIndex': index, 'photoSha256': fileHash(photoPath), 'noseCrop': crop,
                      'cameras': usedCameras, 'stats': stats, 'imageSha256': imageHashes})
        print(json.dumps({'renderedView': index}), flush=True)
    report = {'status': 'complete', 'sourceTaskId': case.name, 'branches': identities,
        'renderer': 'renderNoseRepairComparison', 'rendererSha256': fileHash(__file__),
        'cameraDescription': cameraDescription,
        'configSha256': fileHash(configPath), 'sourceCameraSha256': fileHash(sourceCameras),
        'topologyUvMaterialOrderBindingIdentical': True, 'views': views, 'size': size, 'device': 'cpu',
        'photoIlluminationFitted': False, 'lightFrame': 'cameraRelative',
        'limitations': ['Camera-only explicitly changes extrinsics; it is not a geometry correction.',
            'Branch acceptance is separate from rendering completion.',
            'Cached photo ICC handling and automatic correspondence uncertainty remain.']}
    writeJson(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args()
    run(args.config, args.output, args.size)
