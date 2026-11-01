"""CPU paired reference experiment using verified saved task evidence.

No reconstruction, photograph inference, registration, camera fitting, texture
baking, service submission or callback occurs. Real eye/mouth/lip refinement
runs from identical cached direct-registration vertices. Optional jaw replay
renders the current geometry and detects those renders with the CPU model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.artifacts import writeJson
import faceAppearance.clientGeometryRefinement as geometryModule
from faceAppearance.clientReferenceView import selectClientReference
from faceAppearance.headAssembly import AssemblyAssets
from renderClientAppearanceEvidence import evidenceGrid, solid
from renderRegistrationTransfer20261013 import SurfaceRenderer


def fileSha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def loadArrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def run(case, output, evidenceManifest, observationManifest, size, directGeometry=None, withJaw=False):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    evidence = json.loads(evidenceManifest.read_text())[case.name]
    observationRecords = json.loads(observationManifest.read_text())['files']
    evidence.update({row['relativeName']: row for row in observationRecords
                     if row['taskId'] == case.name and row.get('exists')})
    sourceFiles = {}

    def checked(relative):
        path = case / relative
        expected = evidence.get(relative, {}).get('sha256')
        actual = fileSha(path)
        if not expected or actual != expected:
            raise ValueError(f'Saved remote evidence hash mismatch: {relative}')
        sourceFiles[relative] = actual
        return path

    reconstruction = 'reconstruction/'
    refinement = reconstruction + 'directClient/refinement/'
    cameras = json.loads(checked(reconstruction + 'cameras.json').read_text())
    inputs = json.loads(checked(reconstruction + 'inputs.json').read_text())
    originalGeometry = loadArrays(checked(reconstruction + 'geometry.npz'))
    cachedGeometry = loadArrays(checked(refinement + 'clientGeometry.npz'))
    cachedDiagnostics = json.loads(checked(refinement + 'geometryRefinement.json').read_text())
    landmarks = loadArrays(checked(refinement + 'landmarkObservations.npz'))
    delight = loadArrays(checked(reconstruction + 'delight.npz'))
    robustPath = reconstruction + 'directClient/renderedJaw/photoObservations'
    robust = loadArrays(checked(robustPath + '.npz'))
    robust['diagnostics'] = json.loads(checked(robustPath + '.json').read_text())
    selection = selectClientReference(cameras, inputs, robust)
    if selection['referenceViewIndex'] is None:
        raise ValueError('Paired replay requires a reliable selected reference')
    if cachedDiagnostics['referenceViewIndex'] != 0:
        raise ValueError('This paired baseline requires saved reference zero geometry')
    manifest = json.loads(checked('asset/legacyManifest.json').read_text())
    assets = AssemblyAssets(ROOT / 'assets/clientTemplates' / manifest['templateId'])
    if assets.manifest['templateHashes'] != manifest['templateHashes']:
        raise ValueError('Local template identity differs from saved exported asset')
    if not np.array_equal(cachedGeometry['faces'], assets.arrays['nicpFaces']):
        raise ValueError('Cached direct-registration topology differs')
    if (landmarks['faceLandmarksWorking'].shape != (len(inputs), 68, 2)
            or landmarks['faceRawPeakScores'].shape != (len(inputs), 68)
            or len(cachedDiagnostics['observations']) != len(inputs)):
        raise ValueError('Cached landmark count differs from task inputs')
    observations, workingPaths = [], []
    records = deepcopy(cachedDiagnostics['observations'])
    for index, (metadata, record) in enumerate(zip(inputs, records)):
        if record['viewIndex'] != index or record.get('sourceSha256') != metadata['sha256']:
            raise ValueError('Cached STFR landmarks differ from original input identity/order')
        path = checked(reconstruction + f'input{index:02d}.png')
        workingPaths.append(path)
        with Image.open(path) as image:
            if list(image.size) != metadata['workingSize']:
                raise ValueError('Saved working image dimensions differ')
            workingRgb = np.array(image.convert('RGB'))
        accepted = bool(landmarks['acceptedViewMask'][index])
        observations.append({'pixels': landmarks['faceLandmarksWorking'][index].copy(),
                             'confidence': landmarks['faceRawPeakScores'][index].copy(),
                             'workingRgb': workingRgb} if accepted else None)
        record['replay'] = {'landmarks': 'Exact cached original-image STFR working-coordinate observations',
            'pixels': 'Saved ingestion working PNG; not guaranteed byte-identical to original-image cv2 warp',
            'workingPngSha256': fileSha(path), 'detectorRerun': False}
    transform = cachedDiagnostics['directRegistration']['templateToUff']
    rotation, translation, scale = (np.asarray(transform['rotation']),
                                   np.asarray(transform['translation']), float(transform['scale']))
    productionDirect = cachedGeometry['directTemplateVertices'] @ rotation * scale + translation
    directTemplate = cachedGeometry['directTemplateVertices'].copy()
    directOverride = None
    if directGeometry is not None:
        candidate = loadArrays(directGeometry)
        if not np.array_equal(candidate['clientFaces'], cachedGeometry['faces']):
            raise ValueError('Direct geometry override topology differs')
        for name in ('Rotation', 'Translation', 'Scale'):
            if not np.array_equal(candidate['templateToUff' + name], cachedGeometry['templateToUff' + name]):
                raise ValueError('Direct geometry override transform differs')
        if not np.allclose(candidate['baselineVertices'], productionDirect, atol=1e-13, rtol=0):
            raise ValueError('Direct geometry override has another registration baseline')
        candidateVertices = candidate['guardedBoundaryVertices']
        if not np.array_equal(candidateVertices[assets.arrays['staticMask']], productionDirect[assets.arrays['staticMask']]):
            raise ValueError('Direct geometry override moved the static interface')
        directTemplate = (candidateVertices - translation) @ rotation.T / scale
        directTemplate[assets.arrays['staticMask']] = cachedGeometry['directTemplateVertices'][assets.arrays['staticMask']]
        directOverride = {'path': str(directGeometry), 'sha256': fileSha(directGeometry),
                          'key': 'guardedBoundaryVertices', 'fixedInterfaceExactlyUnchanged': True}
    assembled = {'vertices': assets.arrays['vertices'].copy()}
    assembled['vertices'][assets.arrays['nicpToFullIds']] = directTemplate
    # Scoped substitutions supply verified cached inputs to the real refinement;
    # both branches execute the same implementation and all existing gates.
    with patch.object(assets, '_deform', return_value=(assembled, None, cachedDiagnostics['directRegistration'])), \
            patch.object(geometryModule, '_observations', return_value=(observations, records)):
        results = {}
        for name, reference in [('ref0', 0), ('selected', selection['referenceViewIndex'])]:
            result = geometryModule.refineClientGeometry(assets, originalGeometry, cameras, inputs,
                photoPaths=[None] * len(inputs), delight=delight, projectRoot=ROOT,
                outputDirectory=output / name, workingPhotoPaths=workingPaths,
                referenceViewIndex=reference)
            results[name] = result
            print(json.dumps({'branch': name, 'referenceViewIndex': reference,
                'status': result['diagnostics']['status'],
                'maxDisplacementCm': result['diagnostics']['maxDisplacementCm']}), flush=True)
        baseline, selected = results['ref0'], results['selected']
    jaw = None
    finalVertices = selected['vertices'].copy()
    if withJaw:
        from faceAppearance.clientRenderedJaw import refineRenderedClientJaw
        jaw = refineRenderedClientJaw(assets, finalVertices, selected['faces'], transform,
            cameras, inputs, [None] * len(inputs), projectRoot=ROOT,
            outputDirectory=output / 'renderedJaw',
            referenceViewIndex=selection['referenceViewIndex'], photoObservations=robust)
        finalVertices = jaw['vertices']
        print(json.dumps({'jawStatus': jaw['diagnostics']['status'],
                          'jawChangedVertexCount': jaw['diagnostics']['changedVertexCount']}), flush=True)
    from faceAppearance.clientGeometrySafety import intersectionPairs
    from faceAppearance.frontLandmarkRefinement import checkFrontGeometry
    added = intersectionPairs(finalVertices, selected['faces']) - intersectionPairs(productionDirect, selected['faces'])
    finalSafety = checkFrontGeometry(productionDirect, finalVertices, selected['faces'],
        camera={key: np.asarray(cameras[selection['referenceViewIndex']][key], float)
                for key in ('intrinsics', 'rotation', 'translation')},
        fixedVertexIds=np.flatnonzero(assets.arrays['staticMask']),
        maxDisplacement=float(np.ptp(productionDirect, axis=0).max()) * .07,
        maxRelativeEdgeChange=.25, minAreaRatio=.4, minNormalCosine=.3)
    finalSafety['newIntersectionPairsAgainstOriginalDirect'] = len(added)
    writeJson(output / 'finalChainSafety.json', finalSafety)
    if added or not finalSafety['accepted']:
        raise RuntimeError('Complete replay chain violates safety relative to original direct registration')
    arrays = assets.arrays
    lookup = {tuple(face): index for index, face in enumerate(arrays['faces'])}
    fullFaceIds = np.array([lookup[tuple(face)] for face in arrays['nicpToFullIds'][selected['faces']]])
    finalGeometry = {'vertices': finalVertices, 'faces': selected['faces'], 'uvs': arrays['uvs'],
                     'uvFaces': arrays['uvFaces'][fullFaceIds], 'uvConvention': 'topLeftPixelEdges'}
    np.savez_compressed(output / 'finalGeometry.npz', **{k: v for k, v in finalGeometry.items() if isinstance(v, np.ndarray)})
    np.savez_compressed(output / 'templateToUff.npz', **{key: np.asarray(value) for key, value in transform.items()})
    mouthState = dict(selected['mouthState'], afterCanonical=finalVertices)
    np.savez_compressed(output / 'mouthState.npz', **{key: value for key, value in mouthState.items()
        if isinstance(value, np.ndarray)}, referenceViewIndex=np.asarray(selection['referenceViewIndex']),
        mouthObservationReliable=np.asarray(mouthState['mouthObservationReliable']))
    exported = None
    if withJaw:
        from faceAppearance.legacyAsset import exportLegacyAsset
        oldTexture = loadArrays(checked('reconstruction/directClient/texture.npz'))
        provenance = {key: value for key, value in oldTexture.items() if key != 'linearAlbedo'}
        assembly = assets.assembleFinalClientSurface(finalGeometry, transform,
            oldTexture['linearAlbedo'], provenance,
            diagnostics={'referenceSelection': selection, 'geometry': selected['diagnostics'],
                'renderedJaw': jaw['diagnostics'], 'finalChainSafety': finalSafety,
                'appearanceLimitation': 'Unchanged production atlas, not rebaked for new geometry; geometry-only diagnostic'},
            accessoryReferenceCanonical=selected['vertices'])
        exported = exportLegacyAsset(output / 'asset', assembly, gender=manifest['gender'])
    equalBaseline = np.array_equal(baseline['vertices'], cachedGeometry['vertices'])
    delta = np.linalg.norm(selected['vertices'] - baseline['vertices'], axis=1)
    np.savez_compressed(output / 'pairedGeometry.npz', faces=baseline['faces'],
        productionEyeMouthVertices=cachedGeometry['vertices'], ref0Vertices=baseline['vertices'],
        selectedVertices=selected['vertices'], finalVertices=finalVertices, directTemplateVertices=directTemplate)
    renderers = {name: SurfaceRenderer(vertices, baseline['faces'], device='cpu')
                 for name, vertices in [('production', cachedGeometry['vertices']),
                     ('ref0', baseline['vertices']), ('selected', finalVertices)]}
    allPanels, allLabels, renderRecords = [], [], []
    for view, (camera, metadata, path) in enumerate(zip(cameras, inputs, workingPaths)):
        width, height = metadata['workingSize']
        sx, sy = size / width, size / height
        pixelTransform = np.array([[sx, 0, .5 * sx - .5], [0, sy, .5 * sy - .5], [0, 0, 1]])
        scaled = {key: np.asarray(camera[key], float) for key in ('intrinsics', 'rotation', 'translation')}
        scaled['intrinsics'] = pixelTransform @ scaled['intrinsics']
        with Image.open(path) as image:
            photo = image.convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
        panels, labels = [photo], [f'照片 view {view}']
        for name, label in [('production', '生产眼嘴后几何'), ('ref0', '同缓存 ref0 重放'),
                             ('selected', f"ref{selection['referenceViewIndex']} {'眼嘴+下颌' if withJaw else '眼嘴'}重放")]:
            variants, stats = renderers[name].variants(scaled, (size, size))
            variants['clay'].save(output / f'view{view:02d}_{name}.png')
            panels.append(solid(variants['clay']))
            labels.append(label)
            renderRecords.append({'viewIndex': view, 'branch': name, **stats})
        evidenceGrid(panels, labels, columns=4, panelSize=size,
            subtitle='同保存相机 / 相同灰模光照 / 无相机重拟合；皮肤不含眼球和牙齿').save(output / f'view{view:02d}Comparison.png')
        allPanels.extend(panels)
        allLabels.extend(labels)
        print(json.dumps({'renderedView': view, 'size': size}), flush=True)
    evidenceGrid(allPanels, allLabels, columns=4, panelSize=size,
        subtitle=f"{case.name[:8]} · 下游重放；原始 STFR 观测 + 缓存工作图；下颌{'重放' if withJaw else '未运行'}；未重烘贴图").save(output / 'overview.png')
    report = {'case': case.name, 'selection': selection,
        'baselineExactlyMatchesSavedProductionEyeMouthVertices': equalBaseline,
        'baselineMaximumDifferenceCanonical': float(np.abs(baseline['vertices'] - cachedGeometry['vertices']).max()),
        'selectedVsBaselineChangedVertexCount': int(np.count_nonzero(delta > 1e-12)),
        'selectedVsBaselineMaxDisplacementCanonical': float(delta.max()),
        'branches': {name: result['diagnostics'] for name, result in results.items()},
        'registrationReplayed': False, 'photoDetectorRerun': False, 'cameraRefitted': False,
        'directGeometryOverride': directOverride, 'finalChainSafety': finalSafety,
        'jawRefinementReplayed': withJaw, 'jaw': jaw['diagnostics'] if jaw else None,
        'exported': exported, 'textureBakingReplayed': False, 'device': 'cpu',
        'sourceFiles': sourceFiles, 'templateId': manifest['templateId'],
        'implementationHashes': {str(path.relative_to(ROOT)): fileSha(path) for path in
            [Path(__file__), ROOT / 'src/faceAppearance/clientGeometryRefinement.py',
             ROOT / 'src/faceAppearance/clientReferenceView.py', ROOT / 'src/faceAppearance/clientLipTrace.py']},
        'renderRecords': renderRecords, 'limitations': [
            'Landmark arrays are exact cached original-image STFR observations, verified against original input SHA metadata.',
            'Lip trace RGB uses saved ingestion PNG, which may differ from original-image cv2 warp; paired branches use identical pixels.',
            'No original full-resolution RGB is reloaded or inferred; no reconstruction/registration/texture is run.',
            'Optional rendered-jaw stage may render and infer FAN on the current mesh; cached photo observations remain fixed.',
            'Optional ZIP retains production texture; it is a geometry diagnostic pending rebaking on the new final surface.',
            'Skin-only gray render lacks eyes, teeth, texture and original photo illumination.',
            'Automatic landmark improvement is a proxy; visible or identity improvement requires image review.']}
    writeJson(output / 'report.json', report)
    print(json.dumps({key: report[key] for key in ('case', 'baselineExactlyMatchesSavedProductionEyeMouthVertices',
        'selectedVsBaselineChangedVertexCount', 'selectedVsBaselineMaxDisplacementCanonical')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidenceManifest', type=Path, required=True)
    parser.add_argument('--observationManifest', type=Path, required=True)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--directGeometry', type=Path)
    parser.add_argument('--withJaw', action='store_true')
    args = parser.parse_args()
    run(args.case.resolve(), args.output.resolve(), args.evidenceManifest.resolve(),
        args.observationManifest.resolve(), args.size,
        args.directGeometry.resolve() if args.directGeometry else None, args.withJaw)
