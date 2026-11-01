"""Render a completed local geometry/appearance candidate on saved photo cameras."""
from __future__ import annotations

import argparse
import hashlib
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


def fileHash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(caseDirectory, candidateDirectory, matchedDirectory, outputDirectory, size=320):
    case, candidate, matched, output = map(Path,
        (caseDirectory, candidateDirectory, matchedDirectory, outputDirectory))
    reports = {key: json.loads((folder / 'report.json').read_text())
               for key, folder in [('geometry', candidate), ('matched', matched)]}
    archives = {'production': case / 'asset/legacyAsset.zip',
                'matched': matched / 'asset/legacyAsset.zip',
                'geometry': candidate / 'asset/legacyAsset.zip'}
    hashes = {key: fileHash(path) for key, path in archives.items()}
    for key, report in reports.items():
        if (report['status'] != 'complete' or report['sourceTaskId'] != case.name
                or report['sourceArchiveSha256'] != hashes['production']
                or report['variant']['sha256'] != hashes[key]):
            raise ValueError('Completed candidate source/package identity does not match')
    safety = reports['geometry']['geometryReplay']['finalChainSafety']
    if not safety['accepted'] or safety['newIntersectionPairsAgainstOriginalDirect']:
        raise ValueError('Geometry candidate did not pass the cumulative safety gate')
    assets = {key: _load(path) for key, path in archives.items()}
    for key in ('matched', 'geometry'):
        for field in ('faces', 'uvs', 'uvFaces', 'materialIds'):
            np.testing.assert_array_equal(assets[key].mesh[field], assets['production'].mesh[field])
        if assets[key].manifest['binding'] != assets['production'].manifest['binding']:
            raise ValueError('Client binding changed')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new rendering output directory')
    output.mkdir(parents=True, exist_ok=True)
    similarity = assets['production'].manifest['assemblyDiagnostics']['templateToUff']
    renderers = {key: EvidenceRenderer(path, similarity, device='cpu', facesPerPixel=12)
                 for key, path in archives.items()}
    cameras = json.loads((case / 'reconstruction/cameras.json').read_text())
    inputs = json.loads((case / 'reconstruction/inputs.json').read_text())
    views = []
    for index, (camera, metadata) in enumerate(zip(cameras, inputs)):
        if not camera.get('valid') or camera.get('viewIndex', index) != index:
            raise ValueError('Invalid saved camera or view order')
        photoPath = case / 'reconstruction' / f'input{index:02d}.png'
        with Image.open(photoPath) as image:
            if list(image.size) != metadata['workingSize']:
                raise ValueError('Input dimensions differ')
            photo = image.convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
        sx, sy = size / metadata['workingSize'][0], size / metadata['workingSize'][1]
        pixelTransform = np.array([[sx, 0, .5 * sx - .5], [0, sy, .5 * sy - .5], [0, 0, 1]])
        scaled = {key: np.asarray(camera[key], float) for key in ('intrinsics', 'rotation', 'translation')}
        scaled['intrinsics'] = pixelTransform @ scaled['intrinsics']
        images, stats = {}, {}
        for key, renderer in renderers.items():
            images[key], stats[key] = renderer.variants(scaled, (size, size))
        panels, labels = [], []
        for mode, name in [('baseColor', '无光照基色'), ('lit', '共同PBR光照')]:
            panels.append(photo); labels.append(f'原工作照片 · 视角{index + 1}')
            for key, label in [('production', '生产结果'), ('matched', '仅改肤色'), ('geometry', '肤色＋保守几何修正')]:
                panels.append(solid(images[key][mode])); labels.append(f'{label} · {name}')
        saveSrgb(evidenceGrid(panels, labels, columns=4, panelSize=size,
            subtitle='同一保存相机；新几何重新拟合SH并烘图；未重拟合相机/照片光照'), output / f'view{index:02d}Comparison.png')
        saveSrgb(evidenceGrid([photo] + [solid(images[key]['clay']) for key in archives],
            ['原工作照片', '生产灰模', '仅改肤色：灰模应相同', '修正后完整灰模'], columns=4, panelSize=size,
            subtitle='同一相机与光照；完整资产含眼球及口腔'), output / f'view{index:02d}Clay.png')
        np.testing.assert_array_equal(images['production']['clay'], images['matched']['clay'])
        views.append({'viewIndex': index, 'stats': stats, 'photoSha256': fileHash(photoPath),
            'comparison': f'view{index:02d}Comparison.png', 'clay': f'view{index:02d}Clay.png',
            'camera': {key: value.tolist() for key, value in scaled.items()}})
        print(json.dumps({'renderedView': index}), flush=True)
    writeJson(output / 'report.json', {'status': 'complete', 'sourceTaskId': case.name,
        'sourceArchiveSha256': hashes, 'views': views, 'size': size, 'device': 'cpu',
        'topologyUvMaterialOrderAndBindingUnchanged': True, 'finalChainSafety': safety,
        'cameraSourceSha256': fileHash(case / 'reconstruction/cameras.json'),
        'cameraRefitted': False, 'photoIlluminationFitted': False,
        'limitations': ['Cached working photos keep old ICC handling.',
            'Combined candidate includes geometry changes and SH/texture reprojection; this is not a geometry-only RGB metric.',
            'No independent reflectance or identity ground truth. Neck/ears retain estimated template content.']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('case', 'candidate', 'matched', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--size', type=int, default=320)
    args = parser.parse_args()
    run(args.case, args.candidate, args.matched, args.output, args.size)
