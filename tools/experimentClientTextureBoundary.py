"""Replay only the exterior atlas blend on preserved deployed caches."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.clientTextureBoundary import blendClientTextureExterior
from faceAppearance.headAssembly import AssemblyAssets
from prepareFrontLandmarkTextures import renderTexture, encodeSrgb


def archive(path):
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image(linear):
    # Match the production exporter: convert from stored float32 to float64
    # before the transfer function, avoiding rare one-level rounding changes.
    return Image.fromarray(np.rint(255 * encodeSrgb(linear.astype(np.float64))).astype(np.uint8))


def montage(panels, labels, path):
    width, height = panels[0].size
    canvas = Image.new('RGB', (width * len(panels), height + 30), (28, 35, 39))
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(panel, (index * width, 30))
        draw.text((index * width + 10, 8), label, fill='white')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inset', type=float, default=24.)
    parser.add_argument('--extension', type=float, default=48.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {'geometry': args.source / 'directClient/geometry.npz',
             'texture': args.source / 'directClient/baking/clientTexture.npz',
             'diagnostics': args.source / 'directClient/baking/diagnostics.json',
             'mouthState': args.source / 'directClient/refinement/mouthState.npz',
             'cameras': args.source / 'cameras.json', 'zip': args.zip}
    hashes = {key: sha(path) for key, path in paths.items()}
    geometry = archive(paths['geometry'])
    geometry['uvConvention'] = 'topLeftPixelEdges'
    before = archive(paths['texture'])
    diagnostics = json.loads(paths['diagnostics'].read_text())
    mouth = archive(paths['mouthState'])
    cameras = json.loads(paths['cameras'].read_text())
    assets = AssemblyAssets(args.template)
    gain = np.asarray(diagnostics['templateLinearColorGain'], np.float32)
    fallback = np.clip(assets.materials[0]['baseColor'][..., :3] * gain, 0, 1)
    started = time.perf_counter()
    result = blendClientTextureExterior(geometry, before['linearRGB'], before, fallback,
        cameras, mouthState=mouth, imageShape=(512, 512),
        insetPixels=args.inset, extensionPixels=args.extension)
    seconds = time.perf_counter() - started
    after = {**before, **result['maps'], 'linearRGB': result['linearRGB']}
    np.savez_compressed(args.output / 'clientTexture.npz', **after)
    original, candidate = image(before['linearRGB']), image(after['linearRGB'])
    original.save(args.output / 'before.png')
    candidate.save(args.output / 'after.png')
    montage([original, candidate], ['Deployed baseline', 'Exterior safe-skin SDF blend'], args.output / 'atlas.png')
    with zipfile.ZipFile(args.zip) as sourceZip:
        delivered = Image.open(io.BytesIO(sourceZip.read('head_color.png'))).convert('RGB')
    delta = np.abs(np.asarray(delivered).astype(int) - np.asarray(original).astype(int))
    mask = np.zeros((*before['coverageMask'].shape, 3), np.uint8)
    mask[before['coverageMask']] = [75, 75, 75]
    mask[after['boundarySafeSkinDonor']] = [40, 210, 70]
    mask[after['boundaryModified']] = [235, 155, 35]
    mask[after['boundaryProtectedFeatures']] = [65, 120, 245]
    mask[after['boundaryInternalHoles']] = [210, 45, 180]
    Image.fromarray(mask).save(args.output / 'regions.png')
    panels, labels = [], []
    for index, camera in enumerate(cameras):
        if not camera.get('valid', True):
            continue
        for name, texture in [('Before', original), ('After', candidate)]:
            panel = renderTexture(geometry['vertices'], geometry['faces'], geometry['uvs'],
                                  geometry['uvFaces'], texture, camera)
            panel.save(args.output / f'view{index}_{name.lower()}.png')
            panels.append(panel)
            labels.append(f'View {index} / {name}')
    montage(panels, labels, args.output / 'views.png')
    report = {'status': 'complete', 'sourcePaths': {k: str(p) for k, p in paths.items()},
        'sourceHashes': hashes, 'sourceFilesUnchanged': hashes == {k: sha(p) for k, p in paths.items()},
        'baselineMatchesDeliveredZipPixels': bool((delta == 0).all()),
        'baselineDeliveredZipMaxChannelDifference': int(delta.max()),
        'seconds': seconds, 'templateLinearColorGain': gain.tolist(),
        'parameters': {'insetPixels': args.inset, 'extensionPixels': args.extension},
        'coverageExactlyUnchanged': np.array_equal(before['coverageMask'], after['coverageMask']),
        'provenanceExactlyUnchanged': all(np.array_equal(before[k], after[k]) for k in before if k != 'linearRGB'),
        'geometryChanged': False, 'modelsRerun': False, 'colmapRerun': False,
        'boundaryBlend': result['diagnostics'],
        'limits': ['Atlas edge RGB jump is a local diagnostic, not a quality or identity score.',
                   'Estimated skin colors outside measured support do not expand observed coverage.',
                   'Feature exclusion plus relative skin color gate is heuristic; no new segmentation model was used.',
                   'Reference-camera views show the NICP skin subset; root supplies full-head accessory comparison.']}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
