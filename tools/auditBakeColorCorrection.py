"""Paired, diagnostic-only ablation of COLMAP atlas color correction.

Reuses saved geometry, photographs, cameras and all other texturing options.
Renders the raw rebake stage in both arms, before mouth, boundary or eyelid
postprocessing. Never overwrites parent experiments or exports client assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.colmapTexture import loadTexturedMesh, rebakeAtlas
from renderClientAppearanceEvidence import EvidenceRenderer, evidenceGrid, solid
from renderFeedbackAppearanceComparison import saveSrgb
from renderNoseRepairComparison import scaledViewCamera


def readJson(path):
    return json.loads(Path(path).read_text())


def fileHash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loadArrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def verifyHashes(hashes):
    for path, digest in hashes.items():
        if fileHash(path) != digest:
            raise ValueError(f'Changed input: {path}')


def runCase(entry, parent, output):
    caseId = entry['caseId']
    case = parent / caseId
    baking = case / 'appearance/pureDirect/baking'
    geometryRoot = case / 'geometry/pureDirect'
    geometryReport = readJson(geometryRoot / 'report.json')
    diagnostic = readJson(baking / 'diagnostics.json')
    if geometryReport['status'] != 'geometryDiagnosticComplete' or diagnostic['status'] != 'succeeded':
        raise ValueError('Expected completed geometry and native baking caches')
    if geometryReport['caseId'] != caseId or geometryReport['branch'] != 'pureDirect':
        raise ValueError('Geometry identity mismatch')
    sourceHashes = {str(geometryRoot / key): digest for key, digest in geometryReport['outputHashes'].items()}
    executable = Path(diagnostic['colmapExecutable']['path'])
    sourceHashes[str(executable)] = diagnostic['colmapExecutable']['sha256']
    workspace = baking / 'colorWorkspace'
    workspaceRecord = diagnostic['workspaces']['colorWorkspace']
    sourceHashes[str(workspace / 'mesh.ply')] = workspaceRecord['meshSha256']
    for row in workspaceRecord['cameras']:
        sourceHashes[row['imagePath']] = row['sha256']
    for name in ('sparse/cameras.txt', 'sparse/images.txt', 'sparse/points3D.txt'):
        sourceHashes[str(workspace / name)] = fileHash(workspace / name)
    prepared = readJson(entry['preparationReport'])
    sourceHashes[entry['preparationReport']] = entry['preparationReportSha256']
    for row in prepared['photos']:
        sourceHashes[row['workingPhotoPath']] = row['workingPhotoSha256']
    sourceDirectory = Path(entry['reconstruction'])
    for name in ('cameras.json', 'inputs.json'):
        path = str(sourceDirectory / name)
        sourceHashes[path] = prepared['sourceHashes'][path]
    commandPath = baking / 'colorOutput/command.json'
    for path in (commandPath, baking / 'diagnostics.json', baking / 'colorOutput/mesh.ply',
                 baking / 'colorOutput/texture.png', baking / 'clientTexture.npz'):
        sourceHashes[str(path)] = fileHash(path)
    verifyHashes(sourceHashes)
    output.mkdir()
    command = readJson(commandPath)['argv']
    if command[0] != str(executable) or command[1] != 'mesh_texturer':
        raise ValueError('Unexpected native executable/command')
    allowed = {'--output_path': str(output / 'nativeNoCorrection'),
               '--MeshTextureMapping.apply_color_correction': '0'}
    if command[command.index('--MeshTextureMapping.apply_color_correction') + 1] != '1':
        raise ValueError('Parent does not enable color correction')
    replay = command.copy()
    for key, value in allowed.items():
        if replay.count(key) != 1:
            raise ValueError(f'Expected unique native option: {key}')
        replay[replay.index(key) + 1] = value
    if replay[replay.index('--MeshTextureMapping.num_threads') + 1] != '2':
        raise ValueError('Preserve the two-thread native worker cap')
    (output / 'nativeNoCorrection').mkdir()
    writeJson(output / 'inputs.json', {'sourceHashes': sourceHashes, 'argv': replay})
    with (output / 'native.log').open('w') as log:
        subprocess.run(replay, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300)
    geometry = loadArrays(geometryRoot / 'finalGeometry.npz')
    geometry['uvConvention'] = 'topLeftPixelEdges'
    meshes = [loadTexturedMesh(path / 'mesh.ply', geometry)
              for path in (baking / 'colorOutput', output / 'nativeNoCorrection')]
    for key in ('vertices', 'faces', 'faceUvs', 'validFaces'):
        np.testing.assert_array_equal(meshes[0][key], meshes[1][key])
    cache = loadArrays(baking / 'clientTexture.npz')
    coverage = cache['colmapCoverageMask']
    atlases = []
    for mesh, directory in zip(meshes, (baking / 'colorOutput', output / 'nativeNoCorrection')):
        with Image.open(directory / 'texture.png') as image:
            rgb = np.asarray(image.convert('RGB'))
        rebaked = rebakeAtlas(geometry, mesh, rgb, coverage.shape[0], {'coverageMask': coverage})
        atlases.append(np.power(rebaked['albedo'], 2.2).astype(np.float32))
    delta = abs(atlases[0] - atlases[1])
    sourceArchive = Path(geometryReport['asset']['archivePath'])
    if fileHash(sourceArchive) != geometryReport['asset']['archiveSha256']:
        raise ValueError('Geometry asset changed')
    renderer = EvidenceRenderer(sourceArchive, geometryReport['templateToUff'], device='cpu', facesPerPixel=12)
    renderer.torch.set_num_threads(1)
    templateMap = renderer.maps[0].clone()
    maps = []
    for atlas in atlases:
        linear = templateMap.clone()
        values = renderer.torch.as_tensor(atlas.transpose(2, 0, 1).copy())
        supported = renderer.torch.as_tensor(coverage)
        linear[0, :3, supported] = values[:, supported]
        maps.append(linear)
    cameras = readJson(sourceDirectory / 'cameras.json')
    inputs = readJson(sourceDirectory / 'inputs.json')
    if len(cameras) != 4 or len(inputs) != 4:
        raise ValueError('Expected all four frozen cameras')
    views = []
    for index, metadata in enumerate(inputs):
        camera = scaledViewCamera(cameras[index], metadata, index, 512)
        photoPath = Path(entry['workingPhotos']) / f'input{index:02d}.png'
        with Image.open(photoPath) as image:
            photo = image.convert('RGB').copy()
        rendered = []
        for materialMap in maps:
            renderer.maps[0] = materialMap
            images, _ = renderer.variants(camera, (512, 512))
            rendered.append(images)
        np.testing.assert_array_equal(np.asarray(rendered[0]['clay']), np.asarray(rendered[1]['clay']))
        panels, labels = [], []
        for mode, modeLabel in (('baseColor', '基色'), ('lit', '共同光照')):
            panels.extend([photo, solid(rendered[0][mode]), solid(rendered[1][mode])])
            labels.extend(['原照片', f'现有接缝校色 · {modeLabel}', f'关闭接缝校色 · {modeLabel}'])
        name = f'view{index:02d}Comparison.png'
        saveSrgb(evidenceGrid(panels, labels, columns=3, panelSize=512,
            subtitle=f'{caseId} · 同几何/相机/SH照片，仅比较烘焙阶段；未接后处理，非交付资产'), output / name)
        views.append({'viewIndex': index, 'comparison': name, 'sha256': fileHash(output / name),
                      'clayExactlyIdentical': True})
    verifyHashes(sourceHashes)
    report = {'status': 'diagnosticComplete', 'caseId': caseId, 'split': entry['split'],
        'sourceHashes': sourceHashes, 'onlyChangedOption': allowed,
        'meshAndAtlasFaceLayoutExactlyIdentical': True,
        'coveredTexels': int(coverage.sum()), 'linearAbsoluteDeltaMean': float(delta[coverage].mean()),
        'linearAbsoluteDeltaP95': float(np.percentile(delta[coverage], 95)),
        'changedCoverageFraction': float((delta.max(-1)[coverage] > 1e-7).mean()),
        'views': views, 'clientAssetExported': False, 'acceptedAsFinalCandidate': False,
        'limitations': ['Raw atlas rebake only, without mouth/boundary/eyelid postprocessing.',
            'Unobserved texels use the identical geometry-asset head map in both arms.',
            'This does not validate geometry, true reflectance, semantic registration or client acceptance.',
            'Stock COLMAP omits explicit face-view IDs; identical atlas face layout checked, no claim of forced labels.']}
    writeJson(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT / 'runs/shapePipeline20261024')
    parser.add_argument('--split', type=Path, default=ROOT / 'runs/shapePreservation20261024/split.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh diagnostic directory')
    args.output.mkdir(parents=True)
    split = readJson(args.split)
    plan = {'status': 'frozenBeforeReplay', 'parent': str(args.parent), 'split': str(args.split),
        'splitSha256': fileHash(args.split), 'scriptSha256': fileHash(__file__),
        'caseIds': [entry['caseId'] for entry in split['cases']], 'branch': 'pureDirect',
        'onlyAlgorithmChange': {'MeshTextureMapping.apply_color_correction': 0},
        'sampleSpecificParameters': False, 'originalSafetyRulesChanged': False,
        'maxConcurrentHeavyJobs': 1, 'device': 'cpu', 'diagnosticOnly': True}
    writeJson(args.output / 'plan.json', plan)
    results = []
    for entry in split['cases']:
        try:
            report = runCase(entry, args.parent, args.output / entry['caseId'])
            results.append({key: report[key] for key in ('caseId', 'status', 'linearAbsoluteDeltaMean',
                'linearAbsoluteDeltaP95', 'changedCoverageFraction')})
        except Exception as error:
            results.append({'caseId': entry['caseId'], 'status': 'failed', 'error': str(error)})
        writeJson(args.output / 'summary.json', {'planSha256': fileHash(args.output / 'plan.json'), 'cases': results})
        print(json.dumps(results[-1]), flush=True)


if __name__ == '__main__':
    main()
