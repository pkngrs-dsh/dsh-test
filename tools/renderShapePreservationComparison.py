"""Render frozen initial baseline and bounded global geometry in memory.

Subject lookup comes from an explicit frozen split. Every subject uses the same
baseline-template projection crop rule. Rendering does not export assets,
rebake appearance, rerun eye/mouth refinement, or grant client acceptance.
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

BRANCHES = ('pureDirect', 'bounded')
MODES = (('baseColor', '基色'), ('clay', '灰模'), ('lit', '共同检查光照'))


def loadArrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def pinHashes(identities, additional):
    """Reject conflicting provenance instead of overwriting an earlier pin."""
    for path, digest in additional.items():
        path = str(Path(path).resolve())
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError('Expected SHA-256 input identities')
        if path in identities and identities[path] != digest:
            raise ValueError(f'Conflicting frozen identities: {path}')
        identities[path] = digest


def verifyHashes(identities):
    for path, digest in identities.items():
        if fileHash(path) != digest:
            raise ValueError(f'Frozen diagnostic input changed: {path}')


def diagnosticBanner(picture, caseId, weight, gateLabel):
    header = 122
    canvas = Image.new('RGB', (picture.width, picture.height+header), (22, 27, 33))
    canvas.paste(picture.convert('RGB'), (0, header))
    draw = ImageDraw.Draw(canvas)
    lines = [f'初始配准阶段 · {caseId} · 权重 {weight:g}',
        f'{gateLabel}；通过仅指求解和几何门，尚未客户端验收',
        '两分支贴图相同且未重新烘焙；下游眼口修正未重跑',
        '裁剪来自初始模板投影，非照片标注；检查光照未拟合照片']
    for index, line in enumerate(lines):
        for fontSize in range(16, 11, -1):
            font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', fontSize)
            if draw.textlength(line, font=font) <= canvas.width-24:
                break
        else:
            raise ValueError('Diagnostic title would be clipped')
        draw.text((12, 10+index*26), line, font=font, fill=(235, 239, 244))
    return canvas


def baselinePatchCrop(vertices, patchIds, camera, size):
    """One fixed crop from baseline patch projection; no photo-derived fitting."""
    points = np.asarray(vertices)[patchIds]
    cameraPoints = points@camera['rotation'].T + camera['translation']
    homogeneous = cameraPoints@camera['intrinsics'].T
    if (not np.isfinite(cameraPoints).all() or not np.isfinite(homogeneous).all()
            or np.any(cameraPoints[:, 2] <= 0) or np.any(homogeneous[:, 2] <= 0)):
        raise ValueError('Baseline patch cannot be projected in front of the saved camera')
    pixels = homogeneous[:, :2]/homogeneous[:, 2:3]
    center = (pixels.min(0)+pixels.max(0))*.5
    half = max(float(np.ptp(pixels, axis=0).max())*.9, 40.*size/512.)
    crop = tuple(np.r_[np.floor(center-half), np.ceil(center+half)].astype(int))
    if crop[2] <= 0 or crop[3] <= 0 or crop[0] >= size or crop[1] >= size:
        raise ValueError('Baseline patch crop falls entirely outside the image')
    return crop, {'centerPixels': center, 'halfSizePixels': half, 'projectedPatchPixels': pixels,
                  'minimumCameraDepth': float(cameraPoints[:, 2].min()),
                  'rule': 'center of baseline patch projected bounds; half=max(maxSpan*0.9,40*size/512)',
                  'cropDerivedFromBaselinePatch': True, 'newPhotoAnnotation': False}


def prepareRenderInputs(splitPath, caseId, candidatePath, candidateReportPath):
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', caseId) is None:
        raise ValueError('Expected a safe case identifier from the frozen split')
    splitPath, candidatePath, candidateReportPath = map(lambda p: Path(p).resolve(),
        (splitPath, candidatePath, candidateReportPath))
    identities = {}
    pinHashes(identities, {str(splitPath): fileHash(splitPath), str(candidateReportPath): fileHash(candidateReportPath)})
    split = json.loads(splitPath.read_text())
    matches = [entry for entry in split['cases'] if entry['caseId'] == caseId]
    if len(matches) != 1:
        raise ValueError('Case must occur exactly once in the frozen split')
    entry = matches[0]
    if entry['split'] not in ('development', 'holdout', 'aiDomainControl') or caseId not in split[entry['split']]:
        raise ValueError('Case split membership is inconsistent')
    preparationPath, geometryPath = Path(entry['preparationReport']).resolve(), Path(entry['preparedGeometry']).resolve()
    pinHashes(identities, split['commonSourceHashes'])
    pinHashes(identities, {str(preparationPath): entry['preparationReportSha256'],
                          str(geometryPath): entry['preparedGeometrySha256']})
    verifyHashes(identities)
    preparation = json.loads(preparationPath.read_text())
    if preparation['status'] != 'inputIdentityAndOriginalDirectReplayVerified':
        raise ValueError('Initial preparation identity was not verified')
    for key in ('caseId', 'split', 'sourceDomain', 'reconstruction', 'workingPhotos', 'sourceArchive',
                'originalPhotoSetSha256', 'preparedGeometry', 'preparedGeometrySha256'):
        if preparation[key] != entry[key]:
            raise ValueError(f'Preparation disagrees with split: {key}')
    pinHashes(identities, preparation['sourceHashes'])
    candidateReport = json.loads(candidateReportPath.read_text())
    if (candidateReport['caseId'] != caseId or candidateReport.get('method') != 'boundedGlobalRelative3DPrior'
            or candidateReport.get('finalCandidateAccepted') is not False
            or candidateReport.get('initialRegistrationOnly') is not True
            or candidateReport.get('downstreamGeometryRerun') is not False
            or candidateReport.get('textureRebakedForInitialGeometry') is not False
            or candidateReport.get('exported') is not False):
        raise ValueError('Expected a matching raw bounded initial-registration diagnostic')
    gateAccepted = candidateReport['geometryGate']['accepted']
    solverAccepted = candidateReport['solverDiagnostics']['solverAccepted']
    if type(gateAccepted) is not bool or type(solverAccepted) is not bool:
        raise ValueError('Candidate solver/gate acceptance states must be boolean')
    accepted = gateAccepted and solverAccepted
    if (candidateReport['acceptedForInitialGeometryDiagnostic'] is not accepted
            or candidateReport['status'] != ('initialGeometryDiagnosticPassed' if accepted else 'boundedRejected')):
        raise ValueError('Candidate acceptance labels contradict its solver or geometry gate')
    weight = candidateReport['weight']
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight < 0:
        raise ValueError('Candidate weight must be a finite nonnegative number')
    candidateInputs = candidateReport['sourceHashes']
    for path, digest in ((splitPath, identities[str(splitPath)]),
                         (preparationPath, entry['preparationReportSha256']),
                         (geometryPath, entry['preparedGeometrySha256'])):
        if candidateInputs.get(str(path)) != digest:
            raise ValueError('Candidate was not generated from this exact frozen preparation and split')
    pinHashes(identities, candidateInputs)
    pinHashes(identities, {str(candidatePath): candidateReport['geometrySha256']})
    reconstruction, photoDirectory, sourceArchive = map(lambda p: Path(p).resolve(),
        (entry['reconstruction'], entry['workingPhotos'], entry['sourceArchive']))
    cameraPath, inputsPath = reconstruction/'cameras.json', reconstruction/'inputs.json'
    required = [sourceArchive, cameraPath, inputsPath] + [photoDirectory/f'input{i:02d}.png' for i in range(4)]
    if any(str(path) not in identities for path in required):
        raise ValueError('Preparation did not pin every rendering input')
    codePaths = [Path(__file__), ROOT/'tools/renderNoseRepairComparison.py',
        ROOT/'tools/renderClientAppearanceEvidence.py', ROOT/'tools/renderClientAssembly.py',
        ROOT/'tools/renderFeedbackAppearanceComparison.py', ROOT/'src/faceAppearance/legacyPreviewAsset.py',
        ROOT/'src/faceAppearance/pytorch3dRenderer.py', ROOT/'src/faceAppearance/artifacts.py']
    pinHashes(identities, {str(path): fileHash(path) for path in codePaths})
    verifyHashes(identities)
    raw, candidate = loadArrays(geometryPath), loadArrays(candidatePath)
    for key in ('pureDirect', 'alignedRest', 'faces'):
        np.testing.assert_array_equal(candidate[key], raw[key])
    source = _load(sourceArchive)
    similarity = source.manifest['assemblyDiagnostics']['templateToUff']
    for key in ('rotation', 'translation', 'scale'):
        np.testing.assert_array_equal(raw[key], np.asarray(similarity[key]))
    sourceFull = source.mesh['vertices']*100@raw['rotation']*float(raw['scale'])+raw['translation']
    np.testing.assert_array_equal(sourceFull, raw['fullAppearance'])
    np.testing.assert_array_equal(source.mesh['faces'], raw['fullFaces'])
    skinIds, staticIds = raw['skinIds'], raw['staticIds']
    for name, ids, bound in (('skinIds', skinIds, len(sourceFull)), ('staticIds', staticIds, len(skinIds))):
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids)
                or np.any(ids < 0) or np.any(ids >= bound)):
            raise ValueError(f'Invalid {name}')
    baseline, proposed = raw['pureDirect'], candidate['vertices']
    for vertices in (baseline, proposed):
        if vertices.shape != (len(skinIds), 3) or not np.isfinite(vertices).all():
            raise ValueError('Invalid initial skin vertices')
    np.testing.assert_array_equal(proposed[staticIds], baseline[staticIds])
    np.testing.assert_array_equal(baseline[staticIds], raw['alignedRest'][staticIds])
    np.testing.assert_allclose(sourceFull[skinIds[staticIds]], baseline[staticIds], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(candidate['delta'], proposed-baseline)
    free = np.setdiff1d(np.arange(len(skinIds)), staticIds)
    replaceFullIds = skinIds[free]
    protectedIds = np.setdiff1d(np.arange(len(sourceFull)), replaceFullIds)
    fullBranches = {}
    for branch, skin in zip(BRANCHES, (baseline, proposed)):
        full = sourceFull.copy(); full[replaceFullIds] = skin[free]
        np.testing.assert_array_equal(full[protectedIds], sourceFull[protectedIds])
        fullBranches[branch] = full
    regionNames = sorted(key[:-len('_interiorVertexIds')] for key in raw if key.endswith('_interiorVertexIds'))
    if not regionNames:
        raise ValueError('Prepared geometry has no template patch definitions')
    patchIds = np.unique(np.concatenate([raw[name+suffix] for name in regionNames
        for suffix in ('_interiorVertexIds', '_rimVertexIdsOrdered')]))
    if patchIds.dtype.kind not in 'iu' or not len(patchIds) or np.any(patchIds < 0) or np.any(patchIds >= len(baseline)):
        raise ValueError('Invalid baseline patch projection indices')
    inputs, cameras = json.loads(inputsPath.read_text()), json.loads(cameraPath.read_text())
    if len(inputs) != 4 or len(cameras) != 4 or len(preparation['photos']) != 4:
        raise ValueError('Expected four frozen original camera/photo pairs')
    for index, photo in enumerate(preparation['photos']):
        photoPath = photoDirectory/f'input{index:02d}.png'
        if (photo['viewIndex'] != index or Path(photo['workingPhotoPath']).resolve() != photoPath
                or photo['workingPhotoSha256'] != identities[str(photoPath)]
                or photo['workingSize'] != inputs[index]['workingSize']):
            raise ValueError('Prepared working-photo identity or view order differs')
    verifyHashes(identities)
    return {'entry': entry, 'preparationPath': preparationPath, 'geometryPath': geometryPath,
        'candidatePath': candidatePath, 'candidateReportPath': candidateReportPath, 'candidateReport': candidateReport,
        'sourceArchive': sourceArchive, 'similarity': similarity, 'photoDirectory': photoDirectory,
        'raw': raw, 'fullBranches': fullBranches, 'protectedIds': protectedIds, 'patchIds': patchIds,
        'cameras': cameras, 'inputs': inputs, 'identities': identities, 'splitPath': splitPath}


def run(splitPath, caseId, candidatePath, candidateReportPath, outputDirectory, size=512):
    if type(size) is not int or not 256 <= size <= 1024:
        raise ValueError('Diagnostic render size must be an integer in [256,1024]')
    output = Path(outputDirectory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh diagnostic output directory')
    state = prepareRenderInputs(splitPath, caseId, candidatePath, candidateReportPath)
    candidateReport = state['candidateReport']
    gateAccepted = candidateReport['geometryGate']['accepted']
    solverAccepted = candidateReport['solverDiagnostics']['solverAccepted']
    gateLabel = ('求解和几何门通过' if gateAccepted and solverAccepted else
                 '几何门拒绝' if not gateAccepted else '求解器未通过')
    labels = ['原工作照片', '初始 pureDirect', '有界全局解 · 诊断']
    renderer = EvidenceRenderer(state['sourceArchive'], state['similarity'], device='cpu', facesPerPixel=12)
    renderer.torch.set_num_threads(1)
    from pytorch3d.structures import Meshes

    originalVertices = renderer.vertices.clone()
    protected = renderer.torch.as_tensor(state['protectedIds'], dtype=renderer.torch.int64)
    output.mkdir(parents=True, exist_ok=True)
    views = []
    for index, metadata in enumerate(state['inputs']):
        camera = scaledViewCamera(state['cameras'][index], metadata, index, size)
        crop, cropReport = baselinePatchCrop(state['raw']['pureDirect'], state['patchIds'], camera, size)
        photoPath = state['photoDirectory']/f'input{index:02d}.png'
        with Image.open(photoPath) as sourcePhoto:
            if list(sourcePhoto.size) != metadata['workingSize']:
                raise ValueError('Working photograph dimensions differ from the saved camera frame')
            photo = sourcePhoto.convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
        saveSrgb(photo, output/f'view{index:02d}_photo.png')
        images, stats, imageHashes = {}, {}, {}
        for branch in BRANCHES:
            renderer.vertices = renderer.torch.as_tensor(state['fullBranches'][branch], dtype=renderer.torch.float32)
            if not renderer.torch.equal(renderer.vertices[protected], originalVertices[protected]):
                raise ValueError('Static or non-skin vertices changed at render precision')
            renderer.mesh = Meshes(verts=[renderer.vertices], faces=[renderer.faces])
            images[branch], stats[branch] = renderer.variants(camera, (size, size))
            imageHashes[branch] = {}
            for mode, _ in MODES:
                path = output/f'view{index:02d}_{branch}_{mode}.png'
                saveSrgb(solid(images[branch][mode]), path)
                imageHashes[branch][mode] = fileHash(path)
        np.testing.assert_array_equal(stats['pureDirect']['lightDirectionCanonical'], stats['bounded']['lightDirectionCanonical'])
        panels, panelLabels = [], []
        for mode, modeLabel in MODES:
            row = [photo]+[solid(images[branch][mode]) for branch in BRANCHES]
            panels.extend(row); panelLabels.extend([labels[0]]+[f'{label} · {modeLabel}' for label in labels[1:]])
            nose = noseGrid(row, labels, crop, f'视角 {index+1} · {modeLabel} · 共用保存相机和初始模板裁剪')
            saveSrgb(diagnosticBanner(nose, caseId, candidateReport['weight'], gateLabel), output/f'view{index:02d}_{mode}_nose.png')
        sheet = evidenceGrid(panels, panelLabels, columns=3, panelSize=size,
            subtitle=f'视角 {index+1} · 原工作照片 / 初始 pureDirect / 有界全局解 · 仅作初始几何诊断')
        saveSrgb(diagnosticBanner(sheet, caseId, candidateReport['weight'], gateLabel), output/f'view{index:02d}Comparison.png')
        views.append({'viewIndex': index, 'camera': camera, 'photoSha256': fileHash(photoPath),
            'noseCrop': crop, 'cropProjection': cropReport, 'stats': stats, 'imageSha256': imageHashes})
        print(json.dumps({'caseId': caseId, 'renderedView': index, 'geometryGateAccepted': gateAccepted,
                          'solverAccepted': solverAccepted}), flush=True)
    verifyHashes(state['identities'])
    report = {'status': 'completeDiagnosticRendering', 'renderer': 'renderShapePreservationComparison',
        'caseId': caseId, 'split': state['entry']['split'], 'sourceDomain': state['entry']['sourceDomain'],
        'domainEvidence': state['entry']['domainEvidence'], 'weight': candidateReport['weight'],
        'splitPath': str(state['splitPath']), 'preparationReportPath': str(state['preparationPath']),
        'preparedGeometryPath': str(state['geometryPath']), 'candidateGeometryPath': str(state['candidatePath']),
        'candidateReportPath': str(state['candidateReportPath']),
        'gateReportPath': str(state['candidateReportPath']), 'gateReportSha256': state['identities'][str(state['candidateReportPath'])],
        'candidateStatus': candidateReport['status'], 'geometryGateAccepted': gateAccepted,
        'solverAccepted': solverAccepted, 'gateRejectionReasons': candidateReport['geometryGate']['reasons'],
        'appearanceSourceArchive': str(state['sourceArchive']), 'inputHashesBeforeAndAfter': state['identities'],
        'inputHashesUnchanged': True, 'size': size, 'device': 'cpu', 'stage': 'initialGlobalDirectRegistration',
        'branches': list(BRANCHES), 'views': views,
        'imageSha256': {path.name: fileHash(path) for path in sorted(output.glob('*.png'))},
        'cropDerivedFromBaselinePatch': True, 'cropPatchVertexIds': state['patchIds'], 'newPhotoAnnotation': False,
        'staticAndNonSkinSourceVerticesPreservedExactly': True, 'topologyUvMaterialOrderBindingIdentical': True,
        'appearanceIdenticalBetweenBranches': True, 'textureRebakedForInitialGeometry': False,
        'downstreamEyeMouthGeometryRerun': False, 'cameraRefitted': False, 'photoIlluminationFitted': False,
        'lightFrame': 'cameraRelative', 'clientAssetExported': False, 'rawSolutionIsClientAsset': False,
        'qualityAccepted': False, 'finalCandidateAccepted': False,
        'limitations': ['A solver/geometry-gate pass is not client or photographic acceptance.',
            'Both branches reuse existing appearance without rebaking for either initial geometry.',
            'Downstream eye/mouth geometry has not been rerun; static and non-skin accessories remain at their source positions.',
            'Nose crops come from the original baseline template projection with one common rule, not newly annotated photograph contours.',
            'Saved estimated cameras and camera-relative inspection lighting are shared; neither was refitted to these photographs.',
            'Recorded source-domain provenance is retained; image appearance does not establish human or AI origin.']}
    writeJson(output/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--candidate-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args()
    run(args.split, args.case, args.candidate, args.candidate_report, args.output, args.size)
