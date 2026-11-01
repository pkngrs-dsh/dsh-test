"""Classify a residual forehead line and replay small nonfeature hole filling.

The experiment preserves real observations, all feature pixels and all source
provenance. Harmonic fills are explicitly estimated colors, not photo coverage.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import hashlib
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.colmapTexture import _destinationUvLookup
from faceAppearance.texture import samplePixels
from experimentClientTextureBoundary import archive, image, montage


def fillSmallHoles(data):
    from faceAppearance.clientTextureBoundary import fillClientTextureSkinHoles
    result = fillClientTextureSkinHoles(data['linearRGB'], data)
    return {**data, **result['maps'], 'linearRGB': result['linearRGB']}, result['diagnostics']['components']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--boundary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sourceHash = hashlib.sha256(args.boundary.read_bytes()).hexdigest()
    data = archive(args.boundary)
    geometry = archive(args.source / 'directClient/geometry.npz')
    geometry['uvConvention'] = 'topLeftPixelEdges'
    cameras = json.loads((args.source / 'cameras.json').read_text())
    delight = archive(args.source / 'delight.npz')
    boxes = []
    for bounds in [(530, 275, 645, 325), (540, 288, 640, 316), (560, 318, 610, 346)]:
        x0, y0, x1, y1 = bounds
        selection = np.zeros(data['coverageMask'].shape, bool)
        selection[y0:y1, x0:x1] = True
        indices, counts = np.unique(data['sourceIndex'][selection], return_counts=True)
        boxes.append({'boundsXyxyExclusive': bounds, 'texels': int(selection.sum()),
            'observed': int(data['coverageMask'][selection].sum()),
            'internalTemplateHole': int(data['boundaryInternalHoles'][selection].sum()),
            'protectedFeature': int(data['boundaryProtectedFeatures'][selection].sum()),
            'sourceIndexCounts': {int(i): int(n) for i, n in zip(indices, counts)}})
    coords = np.array([[515, 293], [535, 291], [555, 286], [573, 279], [588, 278], [598, 268]])
    lookup, _ = _destinationUvLookup(geometry, 1024)
    faces = lookup['faceIds'][coords[:, 1], coords[:, 0]]
    bary = lookup['barycentrics'][coords[:, 1], coords[:, 0]]
    points = (geometry['vertices'][geometry['faces'][faces]] * bary[:, :, None]).sum(1)
    camera = cameras[2]
    transformed = points @ np.asarray(camera['rotation']).T + np.asarray(camera['translation'])
    projected = transformed @ np.asarray(camera['intrinsics']).T
    xy = projected[:, :2] / projected[:, 2:]
    direct = samplePixels(delight['albedoImages'][2], xy) ** 2.2
    samples = []
    for pixel, face, photo, color in zip(coords, faces, xy, direct):
        x, y = pixel
        baked = data['linearRGB'][y, x]
        samples.append({'uvPixel': pixel.tolist(), 'face': int(face), 'observed': bool(data['coverageMask'][y, x]),
            'sourceIndex': int(data['sourceIndex'][y, x]), 'view2PhotoPixel': photo.tolist(),
            'directView2LinearRgb': color.tolist(), 'bakedLinearRgb': baked.tolist(),
            'meanAbsoluteRgbDifference': float(np.abs(color - baked).mean())})
    candidate, holes = fillSmallHoles(data)
    np.savez_compressed(args.output / 'clientTexture.npz', **candidate)
    original, after = image(data['linearRGB']), image(candidate['linearRGB'])
    after.save(args.output / 'after.png')
    montage([original, after], ['Exterior-only baseline', 'Plus qualified small internal holes'], args.output / 'atlas.png')
    crop = (480, 220, 680, 380)
    montage([im.crop(crop).resize((800, 640)) for im in [original, after]],
            ['Forehead exterior-only', 'Only small estimated holes changed'], args.output / 'forehead.png')
    photo = image(delight['albedoImages'][2].astype(np.float64)**2.2)
    draw = ImageDraw.Draw(photo)
    for index, (x, y) in enumerate(xy):
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline='red')
        draw.text((x + 3, y + 2), str(index), fill='red')
    photo.save(args.output / 'sourceView2.png')
    unchanged = np.array_equal(data['linearRGB'][data['coverageMask']], candidate['linearRGB'][data['coverageMask']])
    report = {'sourceBoundarySha256': sourceHash,
        'sourceBoundaryUnchanged': sourceHash == hashlib.sha256(args.boundary.read_bytes()).hexdigest(),
        'classification': 'Long curved line is already present inside valid OpenDelight view 2; small bright fragments include unobserved template holes.',
        'coordinateConvention': 'Top-left atlas PNG texel indices; atlas montages add a 30-pixel header.',
        'regions': boxes, 'sourceSamples': samples, 'holeCandidates': holes,
        'filledTexels': int(candidate['interiorSkinHoleFilled'].sum()),
        'observedColorsExactlyUnchanged': unchanged,
        'allPriorProvenanceExactlyUnchanged': all(np.array_equal(data[k], candidate[k]) for k in data if k != 'linearRGB'),
        'protectedFeaturesExactlyUnchanged': np.array_equal(data['linearRGB'][data['boundaryProtectedFeatures']],
                                                           candidate['linearRGB'][data['boundaryProtectedFeatures']]),
        'coverageExpanded': False, 'candidateUsesProductionHelper': True,
        'limits': ['Estimated harmonic fill is not a recovered observation and is not registered as photo coverage.',
                   'The long source-image forehead line remains intact.',
                   'The color gate is a heuristic, not a learned skin classifier or a quality score.']}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'filledTexels': report['filledTexels'], 'regions': boxes, 'holeCandidates': holes}, indent=2))


if __name__ == '__main__':
    main()
