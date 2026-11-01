"""Isolate safe frontal projection from contour-warp rejection on saved assets.

No reconstruction, rebake, geometry, camera, UV or source-image changes. The
disabled branch must reproduce the preceding delivered eye texture exactly.
"""
from __future__ import annotations
import argparse, io, json, shutil, sys
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.headAssembly import decodeColor
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.fastFrontEyeTexture import reprojectFastFrontEyeTexture
from runFastFrontEyeExperiment import npz, archiveCandidate, CASES
from prepareFrontLandmarkTextures import encodeSrgb
from runClientEyelidTextureExperiment import sha


def run(case, parent, output):
    source = parent/case
    out = output/case
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    prior = json.loads((source/'report.json').read_text())
    p = {k: Path(v) for k, v in prior['paths'].items()}
    original = _load(source/'combined/legacyAsset.zip')
    sim = original.manifest['assemblyDiagnostics']['templateToUff']
    full = original.mesh['vertices']*100 @ np.asarray(sim['rotation'])*float(sim['scale'])+sim['translation']
    saved = npz(source/'geometry.npz')
    geometry = npz(p['source']/'directClient/geometry.npz')
    geometry.update(vertices=full[saved['skinVertexIds']], uvConvention='topLeftPixelEdges')
    observations = npz(p['observations']/'view00.npz')['eyelidCurvesWorking']
    photo = {name: {'upper': observations[i, 1], 'lower': observations[i, 0]}
             for i, name in enumerate(('imageLeft', 'imageRight'))}
    binding = json.loads((source/'textureBinding.json').read_text())
    model = {name: {part: np.asarray(binding['eyes'][name][part]['sourcePixels'])
                   for part in ('upper', 'lower')} for name in photo}
    camera = json.loads((p['source']/'cameras.json').read_text())[0]
    delight = npz(p['delight']/'delight.npz')
    baseline_pixels = np.asarray(Image.open(source/'geometryTexture.png')).copy()
    prior_pixels = np.asarray(Image.open(io.BytesIO(original.payloads['head_color.png']))).copy()
    writable = npz(source/'textureProvenance.npz')['writableMask']
    bake_coverage = npz(source/'geometryBake/clientTexture.npz')['coverageMask']
    # Confine the new policy to the unobserved bake patches plus one atlas
    # texel at their edge. Do not retake the whole frontal appearance.
    direct_domain = binary_dilation(~bake_coverage, iterations=1) & writable
    kwargs = dict(writableMask=writable, visibleScene={'vertices': full,
        'faces': original.mesh['faces'][original.mesh['materialIds'] != 3]})
    args = (geometry, camera, decodeColor(delight['albedoImages'][0], 'gamma2.2'),
            delight['validMasks'][0], decodeColor(baseline_pixels[..., :3]/255., 'srgb'), photo, model)
    measured = []
    results = {}
    for enabled in (False, True):
        tick = perf_counter()
        result = reprojectFastFrontEyeTexture(*args, **kwargs, directProjectionFallback=enabled,
                                              directProjectionDomainMask=direct_domain)
        measured.append({'enabled': enabled, 'wallSeconds': perf_counter()-tick,
                         'kernelSeconds': result['diagnostics']['seconds']})
        pixels = baseline_pixels.copy()
        modified = (result['alpha'] > 0) | result['filterPadding']
        pixels[modified, :3] = np.rint(encodeSrgb(result['linearRGB'][modified])*255).astype(np.uint8)
        results[enabled] = (result, pixels)
    if not np.array_equal(results[False][1], prior_pixels):
        raise RuntimeError('Disabled control does not reproduce preceding final PNG')
    (out/'baseline').mkdir()
    shutil.copyfile(source/'combined/legacyAsset.zip', out/'baseline/legacyAsset.zip')
    result, pixels = results[True]
    before, after = results[False][0], result
    changed = np.any(pixels != prior_pixels, axis=2)
    if np.any(changed & ~writable):
        raise RuntimeError('Candidate changed protected UV')
    np.savez_compressed(out/'textureProvenance.npz', **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
                        writableMask=writable, directProjectionDomainMask=direct_domain,
                        originalBakeCoverage=bake_coverage)
    np.savez_compressed(out/'controlProvenance.npz', **{k: v for k, v in before.items() if isinstance(v, np.ndarray)})
    archive = archiveCandidate(source/'combined/legacyAsset.zip', out/'candidate', full, pixels,
        {'branch': 'safeDirectProjection', 'deployed': False, 'geometryChanged': False, 'texture': result['diagnostics']})
    if (out/'candidate/face_nicp.obj').read_bytes() != original.payloads['face_nicp.obj']:
        raise RuntimeError('Texture-only experiment changed OBJ bytes')
    Image.fromarray(pixels).save(out/'candidateTexture.png')
    Image.fromarray(prior_pixels).save(out/'baselineTexture.png')
    Image.fromarray(changed.astype(np.uint8)*255).save(out/'changedUv.png')
    writeJson(out/'config.json', {'branches': [{'id': 'baseline', 'label': '上一版'},
        {'id': 'candidate', 'label': '安全正面直投补充'}], 'defaultLeft': 'baseline', 'defaultRight': 'candidate'})
    report = {'caseId': case, 'status': 'completeLocalTextureExperiment', 'deployed': False,
        'sourceCase': str(source.resolve()), 'paths': prior['paths'],
        'sourceArchiveSha256': sha(source/'combined/legacyAsset.zip'),
        'controlReproducesPreviousPngExactly': True, 'geometryChanged': False,
        'changedTexels': int(changed.sum()), 'timing': measured, 'texture': result['diagnostics'], 'archive': archive,
        'directProjectionDomain': 'Original unobserved bake pixels dilated by one atlas texel, intersected with writable UVs',
        'fullGenerationMeasured': False,
        'scope': 'Cached final geometry, cameras, source albedo and photo semantics; texture kernel only.',
        'implementationSha256': {str(x.relative_to(ROOT)): sha(x) for x in
            (Path(__file__), ROOT/'src/faceAppearance/fastFrontEyeTexture.py')}}
    writeJson(out/'report.json', report)
    print(json.dumps({'case': case, 'changedTexels': report['changedTexels'], 'timing': measured}), flush=True)
    return report


def render(case_dir):
    import torch
    from renderClientAppearanceEvidence import EvidenceRenderer, solid, evidenceGrid
    torch.set_num_threads(1)
    report = json.loads((case_dir/'report.json').read_text())
    source = Path(report['sourceCase'])
    old = json.loads((source/'eyes/report.json').read_text())
    camera = json.loads((Path(report['paths']['source'])/'cameras.json').read_text())[0]
    zoom = {k: np.asarray(camera[k], float) for k in ('intrinsics','rotation','translation')}
    zoom['intrinsics'] = np.asarray(old['roiPixelMap']) @ zoom['intrinsics']
    photo = Image.open(source/'eyes/photo.png').convert('RGB')
    out = case_dir/'eyes'
    out.mkdir(exist_ok=False)
    photo.save(out/'photo.png')
    images, records = {}, []
    for name in ('baseline', 'candidate'):
        asset = _load(case_dir/name/'legacyAsset.zip')
        renderer = EvidenceRenderer(case_dir/name/'legacyAsset.zip', asset.manifest['assemblyDiagnostics']['templateToUff'])
        images[name], stats = renderer.variants(zoom, (photo.height, photo.width))
        for mode, value in images[name].items():
            value.save(out/f'{name}-{mode}.png')
        records.append({'branch': name, **stats})
    if not np.array_equal(np.asarray(images['baseline']['clay']), np.asarray(images['candidate']['clay'])):
        raise RuntimeError('Texture-only candidate changed clay render')
    panels, labels = [], []
    roi = np.asarray(old['roiXyxyWorking'])
    factor = old['roiRasterScale']
    for i, box in enumerate(np.asarray(old['eyeBoxesXyxyWorking'])):
        crop = tuple(np.r_[(box[:2]-roi[:2])*factor, (box[2:]-roi[:2])*factor])
        for mode, label in (('baseColor', '基色'), ('lit', '统一光照')):
            panels.extend([photo.crop(crop), *(solid(images[b][mode]).crop(crop) for b in ('baseline','candidate'))])
            labels.extend([f'图{["左","右"][i]}眼 · 原照片', f'上一版 · {label}', f'安全直投补充 · {label}'])
    evidenceGrid(panels, labels, columns=3, panelSize=420,
        subtitle='相同几何 / 相机 / 原图 · 仅比较眼周取色 · 本地候选').save(out/'comparison.png')
    writeJson(out/'report.json', {'geometryExactlySame': True, 'clayExactlySame': True, 'records': records,
        'cameraRefitted': False, 'roiPixelMap': old['roiPixelMap']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', choices=CASES)
    parser.add_argument('--parent', type=Path, default=ROOT/'runs/fastFrontEyes20261012/final')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/eyelidCleanup20261013/directProjectionV1')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    rows = []
    for case in args.case or CASES:
        rows.append(run(case, args.parent, args.output))
        if args.render:
            render(args.output/case)
    writeJson(args.output/'summary.json', rows)

if __name__ == '__main__':
    main()
