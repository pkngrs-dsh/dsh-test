"""Compare two client geometries with byte-identical appearance and fixed cameras."""
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


def renderComparison(caseDirectory, baselineArchive, candidateArchive, outputDirectory, size=384,
                     diagnosticOnly=False):
    case, baseline, candidate, output = map(Path,
        (caseDirectory, baselineArchive, candidateArchive, outputDirectory))
    before, after = _load(baseline), _load(candidate)
    for name in before.mesh:
        if name != 'vertices':
            np.testing.assert_array_equal(before.mesh[name], after.mesh[name])
    if before.manifest['binding'] != after.manifest['binding']:
        raise ValueError('Client binding changed')
    if set(before.payloads) != set(after.payloads):
        raise ValueError('Asset payload set changed')
    protected = [name for name in before.payloads if name != 'face_nicp.obj']
    for name in protected:
        if before.payloads[name] != after.payloads[name]:
            raise ValueError(f'Frozen appearance payload changed: {name}')
    similarity = before.manifest['assemblyDiagnostics']['templateToUff']
    if similarity != after.manifest['assemblyDiagnostics']['templateToUff']:
        raise ValueError('Assembly frame changed')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    renderers = {key: EvidenceRenderer(path, similarity, device='cpu', facesPerPixel=12)
                 for key, path in [('before', baseline), ('after', candidate)]}
    cameras = json.loads((case / 'reconstruction/cameras.json').read_text())
    inputs = json.loads((case / 'reconstruction/inputs.json').read_text())
    if len(cameras) != len(inputs):
        raise ValueError('Camera/input count mismatch')
    views = []
    for index, (camera, metadata) in enumerate(zip(cameras, inputs)):
        if not camera.get('valid') or camera.get('viewIndex', index) != index:
            raise ValueError('Invalid saved camera/view order')
        photoPath = case / 'reconstruction' / f'input{index:02d}.png'
        with Image.open(photoPath) as image:
            if list(image.size) != metadata['workingSize']:
                raise ValueError('Working photo dimensions differ')
            photo = image.convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
        sx, sy = size / metadata['workingSize'][0], size / metadata['workingSize'][1]
        pixelTransform = np.array([[sx, 0, .5 * sx - .5], [0, sy, .5 * sy - .5], [0, 0, 1]])
        scaled = {key: np.asarray(camera[key], float) for key in ('intrinsics', 'rotation', 'translation')}
        scaled['intrinsics'] = pixelTransform @ scaled['intrinsics']
        images, stats = {}, {}
        for key, renderer in renderers.items():
            images[key], stats[key] = renderer.variants(scaled, (size, size))
            for mode, rgba in images[key].items():
                saveSrgb(solid(rgba), output / f'view{index:02d}_{key}_{mode}.png')
        panels, labels = [], []
        for mode, name in [('baseColor', '基色'), ('lit', '共同光照'), ('clay', '灰模')]:
            panels.extend([photo, solid(images['before'][mode]), solid(images['after'][mode])])
            candidateLabel = '未采用轮廓实验' if diagnosticOnly else '轮廓微调实验'
            labels.extend([f'工作照片 · 视角{index + 1}', f'当前眼睑版 · {name}', f'{candidateLabel} · {name}'])
        saveSrgb(evidenceGrid(panels, labels, columns=3, panelSize=size,
            subtitle=('同相机、光照及贴图；未通过多视角验收，仅作诊断' if diagnosticOnly else
                      '同一保存相机、光照及贴图；只改变几何；照片不是反照率真值')),
            output / f'view{index:02d}Comparison.png')
        views.append({'viewIndex': index, 'stats': stats, 'photoSha256': fileHash(photoPath),
            'comparison': f'view{index:02d}Comparison.png',
            'camera': {key: value.tolist() for key, value in scaled.items()}})
        print(json.dumps({'renderedView': index}), flush=True)
    report = {'status': 'complete', 'sourceTaskId': case.name,
        'archives': {key: {'path': str(path), 'sha256': fileHash(path)}
                     for key, path in [('before', baseline), ('after', candidate)]},
        'protectedPayloadsByteIdentical': protected, 'topologyUvMaterialOrderAndBindingUnchanged': True,
        'cameraSourceSha256': fileHash(case / 'reconstruction/cameras.json'),
        'cameraRefitted': False, 'textureRebaked': False, 'photoIlluminationFitted': False,
        'views': views, 'size': size, 'device': 'cpu', 'diagnosticOnly': diagnosticOnly,
        'limitations': ['Geometry diagnostics with frozen texture; image samples are not reprojected to the new surface.',
            'Cached working photos retain historical ICC handling; no independent identity/reflectance ground truth.',
            'Geometry safety must be validated separately before promotion.']}
    writeJson(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('case', 'baseline', 'candidate', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--size', type=int, default=384)
    parser.add_argument('--diagnostic-only', action='store_true')
    args = parser.parse_args()
    renderComparison(args.case, args.baseline, args.candidate, args.output, args.size, args.diagnostic_only)
