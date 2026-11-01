"""Render raw initial-registration NPZ results without exporting client assets.

The exact same frozen ZIP supplies both branches' appearance and accessories.
Only free skin vertices are replaced in memory. Rejected candidates remain
explicitly rejected; rendering is a diagnostic, never an acceptance operation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.legacyPreviewAsset import _load
from renderClientAppearanceEvidence import EvidenceRenderer, evidenceGrid, solid
from renderFeedbackAppearanceComparison import saveSrgb
from renderNoseRepairComparison import fileHash, noseGrid, scaledViewCamera

MODES = (('baseColor', '基色'), ('clay', '灰模'), ('lit', '共同检查光照'))
BRANCHES = ('pureDirect', 'rawPrior')
WEIGHTS = (.01, .03, .1, .25)


def loadArrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def verifyHashes(identities):
    for path, expected in identities.items():
        if fileHash(path) != expected:
            raise ValueError(f'Pinned diagnostic input changed: {path}')


def diagnosticBanner(picture, shortId, weight, gateLabel):
    """Keep limitations readable even on the narrower 768px nose sheets."""
    header = 96
    canvas = Image.new('RGB', (picture.width, picture.height + header), (22, 27, 33))
    canvas.paste(picture.convert('RGB'), (0, header))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 16)
    lines = [f'初始配准阶段 · {shortId} · 权重 {weight:g} · {gateLabel}',
             '两模型贴图相同，未重新烘焙；下游几何修正未重跑',
             '原始解仅作诊断，不是客户端资产；检查光照未拟合照片']
    for index, line in enumerate(lines):
        if draw.textlength(line, font=font) > canvas.width - 24:
            raise ValueError('Diagnostic limitations would be clipped at this image width')
        draw.text((12, 10 + index * 26), line, font=font, fill=(235, 239, 244))
    return canvas


def prepareRenderInputs(shortId, weight, experimentRoot):
    if re.fullmatch(r'[0-9a-f]{8}', shortId) is None or weight not in WEIGHTS:
        raise ValueError('Expected an eight-character subject ID and a frozen experiment weight')
    directory = Path(experimentRoot).resolve() / shortId
    prepared = directory / 'preparation'
    candidateDirectory = directory / 'candidates' / f'weight{weight:g}'
    preparationPath, candidateReportPath = prepared / 'report.json', candidateDirectory / 'report.json'
    identities = {str(path): fileHash(path) for path in (preparationPath, candidateReportPath)}
    preparation = json.loads(preparationPath.read_text())
    candidateReport = json.loads(candidateReportPath.read_text())
    caseId = preparation['caseId']
    if not caseId.startswith(shortId + '-') or candidateReport['weight'] != weight:
        raise ValueError('Subject or candidate weight identity mismatch')
    if preparation['status'] != 'originalDirectReplayAndRelativePriorPrepared':
        raise ValueError('Initial registration preparation is not complete')
    gate = candidateReport['geometryGate']
    if (type(gate['accepted']) is not bool or candidateReport['status'] not in
            ('rejected', 'passedFrozenGeometryAndMinimumEffectGates')
            or (candidateReport['status'] == 'rejected') == gate['accepted']
            or candidateReport['finalCandidateAccepted'] is not False):
        raise ValueError('Candidate gate status is inconsistent or is not a diagnostic result')
    identities.update(preparation['sourceHashes'])
    identities.update({str(prepared / name): digest for name, digest in preparation['preparedHashes'].items()})
    geometryPath, candidatePath = prepared / 'geometry.npz', candidateDirectory / 'geometry.npz'
    identities[str(candidatePath)] = candidateReport['geometrySha256']
    if str(geometryPath) not in identities:
        raise ValueError('Preparation did not pin its geometry NPZ')
    sourceArchive = Path(preparation['sourceArchive']).resolve()
    if str(sourceArchive) not in identities:
        raise ValueError('Preparation did not pin its appearance ZIP')
    case = ROOT / 'reports/feedbackAudit20261024/cases' / caseId
    cameraPath, inputsPath = case / 'reconstruction/cameras.json', case / 'reconstruction/inputs.json'
    if any(str(path) not in identities for path in (cameraPath, inputsPath)):
        raise ValueError('Preparation did not pin the subject cameras and image metadata')
    policyPath = directory / 'policy.json'
    identities[str(policyPath)] = fileHash(policyPath)
    rootReportPath = directory / 'report.json'
    if rootReportPath.is_file():
        identities[str(rootReportPath)] = fileHash(rootReportPath)
        if json.loads(rootReportPath.read_text())['caseId'] != caseId:
            raise ValueError('Experiment summary belongs to a different subject')
    observationsPath = case / 'reconstruction/directClient/renderedJaw/photoObservations.npz'
    identities[str(observationsPath)] = fileHash(observationsPath)
    for index in range(4):
        path = case / 'reconstruction' / f'input{index:02d}.png'
        identities[str(path)] = fileHash(path)
    for name in ('renderEarlyCavityComparison.py', 'renderNoseRepairComparison.py',
                 'renderClientAppearanceEvidence.py', 'renderClientAssembly.py',
                 'renderFeedbackAppearanceComparison.py'):
        path = ROOT / 'tools' / name
        identities[str(path)] = fileHash(path)
    verifyHashes(identities)
    raw, candidate = loadArrays(geometryPath), loadArrays(candidatePath)
    np.testing.assert_array_equal(raw['faces'], candidate['faces'])
    np.testing.assert_array_equal(raw['pureDirect'], candidate['pureDirect'])
    np.testing.assert_array_equal(raw['alignedRest'], candidate['alignedRest'])
    source = _load(sourceArchive)
    similarity = source.manifest['assemblyDiagnostics']['templateToUff']
    for key in ('rotation', 'translation', 'scale'):
        np.testing.assert_array_equal(raw[key], np.asarray(similarity[key]))
    sourceFull = source.mesh['vertices'] * 100 @ raw['rotation'] * float(raw['scale']) + raw['translation']
    np.testing.assert_array_equal(sourceFull, raw['fullAppearance'])
    np.testing.assert_array_equal(source.mesh['faces'], raw['fullFaces'])
    skinIds, staticIds = raw['skinIds'], raw['staticIds']
    for name, ids, bound in (('skinIds', skinIds, len(sourceFull)), ('staticIds', staticIds, len(skinIds))):
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids)
                or np.any(ids < 0) or np.any(ids >= bound)):
            raise ValueError(f'Invalid {name} indexing')
    baseline, proposed = raw['pureDirect'], candidate['vertices']
    for vertices in (baseline, proposed):
        if vertices.shape != (len(skinIds), 3) or not np.isfinite(vertices).all():
            raise ValueError('Invalid initial skin geometry')
    np.testing.assert_array_equal(proposed[staticIds], baseline[staticIds])
    np.testing.assert_array_equal(baseline[staticIds], raw['alignedRest'][staticIds])
    np.testing.assert_allclose(sourceFull[skinIds[staticIds]], baseline[staticIds], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(candidate['delta'], proposed - baseline)
    freeSkinIds = np.setdiff1d(np.arange(len(skinIds)), staticIds)
    replaceFullIds = skinIds[freeSkinIds]
    protectedFullIds = np.setdiff1d(np.arange(len(sourceFull)), replaceFullIds)
    fullBranches = {}
    for branch, skin in zip(BRANCHES, (baseline, proposed)):
        full = sourceFull.copy()
        full[replaceFullIds] = skin[freeSkinIds]
        np.testing.assert_array_equal(full[protectedFullIds], sourceFull[protectedFullIds])
        fullBranches[branch] = full
    cameras, inputs = json.loads(cameraPath.read_text()), json.loads(inputsPath.read_text())
    if len(cameras) != 4 or len(inputs) != 4:
        raise ValueError('Expected four matching original photographs and saved cameras')
    landmarks = loadArrays(observationsPath)['faceLandmarksWorking']
    if landmarks.shape[0] != 4 or landmarks.ndim != 3 or landmarks.shape[1] < 36 or landmarks.shape[2] != 2:
        raise ValueError('Invalid source-photo landmark dimensions for shared nose crops')
    if not np.isfinite(landmarks[:, 27:36]).all():
        raise ValueError('Invalid source-photo nose landmarks')
    verifyHashes(identities)
    return {'case': case, 'directory': directory, 'sourceArchive': sourceArchive, 'similarity': similarity,
            'preparationPath': preparationPath, 'candidateReportPath': candidateReportPath,
            'candidateReport': candidateReport, 'geometryPath': geometryPath, 'candidatePath': candidatePath,
            'policyPath': policyPath, 'source': source, 'raw': raw, 'fullBranches': fullBranches,
            'protectedFullIds': protectedFullIds, 'cameras': cameras, 'inputs': inputs,
            'landmarks': landmarks, 'identities': identities}


def run(shortId, weight, outputDirectory, size=512, experimentRoot=ROOT / 'runs/earlyCavity20261024'):
    if type(size) is not int or not 256 <= size <= 1024:
        raise ValueError('Diagnostic size must be an integer in [256,1024]')
    output = Path(outputDirectory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh diagnostic render directory')
    state = prepareRenderInputs(shortId, weight, experimentRoot)
    gate = state['candidateReport']['geometryGate']
    gateLabel = '门控已拒绝' if not gate['accepted'] else '几何门通过，质量未验收'
    labels = ['原工作照片', '初始 pureDirect', f'原始 prior 解 · {gateLabel}']
    renderer = EvidenceRenderer(state['sourceArchive'], state['similarity'], device='cpu', facesPerPixel=12)
    renderer.torch.set_num_threads(1)
    from pytorch3d.structures import Meshes

    originalVertices = renderer.vertices.clone()
    protectedIds = renderer.torch.as_tensor(state['protectedFullIds'], dtype=renderer.torch.int64)
    output.mkdir(parents=True, exist_ok=True)
    views = []
    for index, metadata in enumerate(state['inputs']):
        camera = scaledViewCamera(state['cameras'][index], metadata, index, size)
        photoPath = state['case'] / 'reconstruction' / f'input{index:02d}.png'
        with Image.open(photoPath) as sourcePhoto:
            if list(sourcePhoto.size) != metadata['workingSize']:
                raise ValueError('Source working-photo size changed')
            photo = sourcePhoto.convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
        saveSrgb(photo, output / f'view{index:02d}_photo.png')
        images, stats, imageHashes = {}, {}, {}
        for branch in BRANCHES:
            renderer.vertices = renderer.torch.as_tensor(state['fullBranches'][branch], dtype=renderer.torch.float32)
            if not renderer.torch.equal(renderer.vertices[protectedIds], originalVertices[protectedIds]):
                raise ValueError('Static or accessory vertices changed at render precision')
            # Rebuild Meshes so packed vertices and normals cannot use another
            # branch's cache. Faces, UVs, maps, materials and cameras stay fixed.
            renderer.mesh = Meshes(verts=[renderer.vertices], faces=[renderer.faces])
            images[branch], stats[branch] = renderer.variants(camera, (size, size))
            imageHashes[branch] = {}
            for mode, _ in MODES:
                path = output / f'view{index:02d}_{branch}_{mode}.png'
                saveSrgb(solid(images[branch][mode]), path)
                imageHashes[branch][mode] = fileHash(path)
        sx, sy = size / metadata['workingSize'][0], size / metadata['workingSize'][1]
        nosePoints = (state['landmarks'][index, 27:36] + .5) * [sx, sy] - .5
        center = (nosePoints.min(0) + nosePoints.max(0)) / 2
        half = max(float(np.ptp(nosePoints, axis=0).max()) * .7, 50. * sx)
        crop = tuple(np.r_[np.floor(center - half), np.ceil(center + half)].astype(int))
        fullPanels, fullLabels = [], []
        for mode, modeLabel in MODES:
            panels = [photo] + [solid(images[branch][mode]) for branch in BRANCHES]
            fullPanels.extend(panels)
            fullLabels.extend([labels[0]] + [f'{label} · {modeLabel}' for label in labels[1:]])
            cropSheet = noseGrid(panels, labels, crop, f'视角 {index + 1} · {modeLabel} · 共用原保存相机')
            saveSrgb(diagnosticBanner(cropSheet, shortId, weight, gateLabel), output / f'view{index:02d}_{mode}_nose.png')
        sheet = evidenceGrid(fullPanels, fullLabels, columns=3, panelSize=size,
            subtitle=f'视角 {index + 1} · 原工作照片 / 初始 pureDirect / 原始 prior 解 · 所有分支共用保存相机')
        saveSrgb(diagnosticBanner(sheet, shortId, weight, gateLabel), output / f'view{index:02d}Comparison.png')
        views.append({'viewIndex': index, 'camera': camera, 'photoSha256': fileHash(photoPath),
                      'noseCrop': crop, 'stats': stats, 'imageSha256': imageHashes})
        print(json.dumps({'case': shortId, 'weight': weight, 'renderedView': index, 'gateAccepted': gate['accepted']}), flush=True)
    verifyHashes(state['identities'])
    files = {path.name: fileHash(path) for path in sorted(output.glob('*.png'))}
    report = {'status': 'completeDiagnosticRendering', 'renderer': 'renderEarlyCavityComparison',
        'sourceTaskId': state['case'].name, 'weight': weight, 'size': size, 'device': 'cpu',
        'stage': 'initialGlobalDirectRegistration', 'branches': list(BRANCHES),
        'appearanceSourceArchive': str(state['sourceArchive']),
        'appearanceSourceArchiveSha256': state['identities'][str(state['sourceArchive'])],
        'preparationReportPath': str(state['preparationPath']), 'preparationGeometryPath': str(state['geometryPath']),
        'candidateGeometryPath': str(state['candidatePath']),
        'gateReportPath': str(state['candidateReportPath']),
        'gateReportSha256': state['identities'][str(state['candidateReportPath'])],
        'gateStatus': state['candidateReport']['status'], 'geometryGateAccepted': gate['accepted'],
        'gateRejectionReasons': gate['reasons'], 'policyPath': str(state['policyPath']),
        'inputHashesBeforeAndAfter': state['identities'], 'inputHashesUnchanged': True,
        'views': views, 'imageSha256': files, 'meshSource': 'rawNPZFreeSkinReplacementInMemory',
        'staticAndNonSkinSourceVerticesPreservedExactly': True,
        'topologyUvMaterialOrderBindingIdentical': True, 'appearanceIdenticalBetweenBranches': True,
        'textureRebakedForInitialGeometry': False, 'downstreamGeometryRerun': False,
        'cameraRefitted': False, 'photoIlluminationFitted': False, 'lightFrame': 'cameraRelative',
        'clientAssetExported': False, 'rawSolutionIsClientAsset': False,
        'qualityAccepted': False, 'finalCandidateAccepted': False,
        'limitations': [
            'These are initial global registration stages, not fully refined deliverable geometry.',
            'Gate rejection remains in effect even when a raw candidate is rendered.',
            'Both branches reuse identical existing appearance; texture has not been rebaked for either initial geometry.',
            'Downstream eye, mouth and other geometric refinement has not been rerun; existing accessories stay in their source positions.',
            'Inspection lighting is shared but not fitted to the photograph; saved estimated cameras may have residual error.',
            'Source working photographs are display-resized only; no new detection or anatomical measurement is performed.']}
    writeJson(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, help='Eight-character subject ID, for example 110bac9f')
    parser.add_argument('--weight', type=float, required=True, choices=WEIGHTS)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--experiment-root', type=Path, default=ROOT / 'runs/earlyCavity20261024')
    args = parser.parse_args()
    run(args.case, args.weight, args.output, args.size, args.experiment_root)
