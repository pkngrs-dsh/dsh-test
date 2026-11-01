"""Run one staged neural prior at a time, reusing audited batch crops and UVs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from PIL import Image

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / 'src'), str(projectRoot / 'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.data import readImage
from faceAppearance.runtime import configureRuntime
from runBrowRefinementExperiment import fileHash, readArrays


def loadAuditedImages(caseDir):
    directory = caseDir / 'reconstruction'
    metadata = json.loads((directory / 'inputs.json').read_text(encoding='utf-8'))
    images = []
    for index, record in enumerate(metadata):
        if fileHash(record['path']) != record['sha256']:
            raise ValueError('Source photo changed')
        image, actual = readImage(Path(record['path']), cropPolicy=record['requestedCropPolicy'],
            cropMargin=record.get('cropMargin', 1.35), faceDetector=lambda unused: record['faceDetection'])
        if not np.array_equal(actual['inputToWorking'], record['inputToWorking']):
            raise ValueError('Crop pixel transform differs from saved baseline')
        saved = np.asarray(Image.open(directory / f'input{index:02d}.png').convert('RGB')) / np.float32(255)
        if not np.array_equal(saved, image):
            raise ValueError('Saved working image differs from reconstructed original crop')
        images.append(image)
    return np.stack(images), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['normals', 'trace'], required=True)
    parser.add_argument('--batch', type=Path, default=projectRoot / 'runs/clientBatchV1')
    parser.add_argument('--output', type=Path, default=projectRoot / 'runs/browNormalRefinementV1')
    parser.add_argument('--cases', nargs='+', default=['windowLight4', 'diffuseLight4', 'backlightGlasses4', 'overheadHair4'])
    args = parser.parse_args()
    configureRuntime(projectRoot)
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Neural cache requires CUDA')
    torch.cuda.reset_peak_memory_stats()
    if args.stage == 'normals':
        from faceAppearance.adapters.normalPrior import NormalPriorAdapter
        adapter = NormalPriorAdapter(projectRoot, memoryPolicy='staged').loadModels()
    else:
        from faceAppearance.adapters.geometry import GeometryAdapter
        adapter = GeometryAdapter(projectRoot, memoryPolicy='staged').loadDiagnosticModels()
    for caseId in args.cases:
        started = perf_counter()
        caseDir, output = args.batch / caseId, args.output / caseId / args.stage
        if output.exists():
            raise FileExistsError(f'Refusing to overwrite cache: {output}')
        output.mkdir(parents=True)
        images, metadata = loadAuditedImages(caseDir)
        if args.stage == 'normals':
            result = adapter.predict(images)
            np.savez_compressed(output / 'normals.npz', normals=result['normals'], validMask=result['validMask'])
            writeJson(output / 'diagnostics.json', result['diagnostics'])
            for index, normal in enumerate(result['normals']):
                rgb = np.clip((normal + 1) * .5, 0, 1)
                Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(output / f'normal{index:02d}.png')
        else:
            geometry = readArrays(caseDir / 'reconstruction/geometry.npz')
            result = adapter.reconstructFromCachedUvs(images, geometry['uvMaps'], geometry['faceMasks'], diagnosticsDirectory=output)
            np.savez_compressed(output / 'reconstructed.npz', **{name: value for name, value in result.items() if isinstance(value, np.ndarray)})
            difference = np.linalg.norm(result['vertices'].astype(float) - geometry['vertices'], axis=1)
            writeJson(output / 'baselineReproduction.json', {'maximumVertexDistance': float(difference.max()),
                'meanVertexDistance': float(difference.mean()), 'baselineGeometrySha256': fileHash(caseDir / 'reconstruction/geometry.npz'),
                'reasonForPossibleDifference': 'GPU kernels/precision; cached UV and masks reused exactly'})
        writeJson(output / 'cacheReport.json', {'status': 'completed', 'stage': args.stage, 'caseId': caseId,
            'seconds': perf_counter() - started, 'originalImages': metadata,
            'workingCropsExactlyReproduced': True, 'peakAllocatedGpuBytes': torch.cuda.max_memory_allocated(),
            'peakReservedGpuBytes': torch.cuda.max_memory_reserved(), 'sequentialGpuStages': True})
        print(json.dumps({'case': caseId, 'stage': args.stage, 'seconds': perf_counter() - started,
                          'peakGpuGiB': torch.cuda.max_memory_allocated() / 2**30}), flush=True)


if __name__ == '__main__':
    main()
