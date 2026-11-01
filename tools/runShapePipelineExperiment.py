"""Paired local replay from frozen initial geometry; never changes service defaults."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson, jsonValue
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.clientReferenceView import selectClientReference
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.frontLandmarkRefinement import checkFrontGeometry
from faceAppearance.legacyAsset import exportLegacyAsset
from faceAppearance.legacyPreviewAsset import _load

RUN = ROOT/'runs/shapePipeline20261024'
PRIOR = ROOT/'runs/shapePreservation20261024'
BRANCHES = ('pureDirect', 'bounded')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def read(path):
    return json.loads(Path(path).read_text())


def verify(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')


def fresh(path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Use a fresh output: {path}')
    path.mkdir(parents=True, exist_ok=True)


def inputsFor(caseId):
    plan = read(RUN/'experimentPlan.json')
    appearancePolicy = plan['appearance']
    if (plan['initialBranches'] != list(BRANCHES) or plan['boundedInitialWeight'] != .01
            or not plan['noParameterSelection'] or not plan['noPerCaseParameters']
            or appearancePolicy['shFitResolution'] != 256 or appearancePolicy['shIterations'] != 8
            or appearancePolicy['clientAtlasResolution'] != 1024
            or appearancePolicy['eyelidSampling'] != {'corePixels':12., 'outerPixels':24.,
                'sourceSafetyPixels':.75, 'maxExtensionPixels':4., 'boxPaddingPixels':14.,
                'boxFeatherPixels':4., 'filterFootprint':False}):
        raise ValueError('Runner settings differ from the frozen experiment plan')
    split = read(PRIOR/'split.json')
    entries = [case for case in split['cases'] if case['caseId'] == caseId]
    if len(entries) != 1:
        raise ValueError('Choose one case from the frozen split')
    entry = entries[0]
    parentPath = PRIOR/caseId/'boundedWeight001/report.json'
    parent = read(parentPath)
    verify(parent['sourceHashes'])
    if not parent['acceptedForInitialGeometryDiagnostic']:
        raise ValueError('Initial candidate did not pass its frozen gates')
    candidatePath = parentPath.with_name('geometry.npz')
    if sha(candidatePath) != parent['geometrySha256']:
        raise ValueError('Candidate geometry identity changed')
    prepared = arrays(entry['preparedGeometry'])
    candidate = arrays(candidatePath)
    for name in ('pureDirect', 'faces', 'alignedRest'):
        np.testing.assert_array_equal(candidate[name], prepared[name])
    reconstruction = Path(entry['reconstruction'])
    working = Path(entry['workingPhotos'])
    paths = [parentPath, candidatePath, reconstruction/'inputs.json', reconstruction/'cameras.json',
             reconstruction/'delight.npz', reconstruction/'directClient/renderedJaw/photoObservations.npz',
             reconstruction/'directClient/renderedJaw/photoObservations.json', RUN/'experimentPlan.json',
             RUN/'implementationHashes.json']
    paths += [working/f'input{i:02d}.png' for i in range(4)]
    hashes = {**parent['sourceHashes'], **{str(path): sha(path) for path in paths}}
    implementation = read(RUN/'implementationHashes.json')
    verify(implementation['sourceHashes'])
    hashes.update(implementation['sourceHashes'])
    metadata, cameras = read(reconstruction/'inputs.json'), read(reconstruction/'cameras.json')
    fan = arrays(reconstruction/'directClient/renderedJaw/photoObservations.npz')
    fan['diagnostics'] = read(reconstruction/'directClient/renderedJaw/photoObservations.json')
    selection = selectClientReference(cameras, metadata, fan)
    photos = [working/f'input{i:02d}.png' for i in range(len(metadata))]
    for path, item in zip(photos, metadata):
        with Image.open(path) as picture:
            if list(picture.size) != item['workingSize']:
                raise ValueError('Working image dimensions differ')
    delight = arrays(reconstruction/'delight.npz')
    if delight['validMasks'].dtype != np.bool_:
        raise ValueError('This replay requires boolean independent photo masks')
    return dict(entry=entry, raw=prepared, candidate=candidate, cameras=cameras, inputs=metadata,
                delight=delight, photoPaths=photos, fan=fan,
                selection=selection, hashes=hashes)


def validatedEyeObservations(context, directory):
    report = read(directory/'observations.json')
    verify(report['sourceHashes'])
    reference = context['selection']['referenceViewIndex']
    if (report['caseId'] != context['entry']['caseId'] or report['referenceViewIndex'] != reference
            or report['inferenceFrame'] != 'savedWorkingRgbPng'
            or report['geometryUsedForInference'] is not False
            or sha(directory/'observations.npz') != report['observationSha256']
            or report['sourcePhotoSha256'] != sha(context['photoPaths'][reference])):
        raise ValueError('Dense photo observation identity or frame mismatch')
    reconstruction = Path(context['entry']['reconstruction'])
    required = [reconstruction/'inputs.json', reconstruction/'cameras.json',
                reconstruction/'directClient/renderedJaw/photoObservations.npz']
    if any(report['sourceHashes'].get(str(path)) != sha(path) for path in required):
        raise ValueError('Dense observations use a different input/camera/FAN source')
    policy = report['policy']
    if (policy['anchorOffsetsWorkingPixels'] != [[0,0],[2,0],[-2,0],[0,2],[0,-2]]
            or policy['maximumCropVariationWorkingPixels'] != 3.):
        raise ValueError('Dense eye policy differs from the frozen experiment')
    values = arrays(directory/'observations.npz')
    variants = values['cropVariantCurvesWorking']
    variation = np.linalg.norm(variants-variants[:1], axis=-1).max(0)
    np.testing.assert_array_equal(values['eyelidCurvesWorking'], variants[0])
    np.testing.assert_allclose(values['maximumVariationWorkingPixels'], variation, rtol=0, atol=0)
    accepted = values['pointValidMask'].all(axis=(1,2)) & (variation.max(axis=(1,2)) <= 3.)
    accepted &= np.asarray([eye['valid'] for eye in report['quality']])
    np.testing.assert_array_equal(accepted, values['acceptedEyeMask'])
    if (variants.shape != (5,2,2,12,2) or not np.isfinite(variants).all()
            or report['status'] != ('accepted' if accepted.all() else 'rejected')):
        raise ValueError('Dense eye observation status or arrays are invalid')
    return report, values


def finalGeometry(assets, vertices):
    data = assets.arrays
    lookup = {tuple(face): i for i, face in enumerate(data['faces'])}
    fullFaces = np.asarray([lookup[tuple(face)] for face in data['nicpToFullIds'][data['nicpFaces']]])
    return {'vertices': vertices.copy(), 'faces': data['nicpFaces'].copy(), 'uvs': data['uvs'].copy(),
            'uvFaces': data['uvFaces'][fullFaces].copy(), 'uvConvention': 'topLeftPixelEdges'}


def saveArrays(path, values):
    np.savez_compressed(path, **{key: value for key, value in values.items() if isinstance(value, np.ndarray)})


def appearanceExportDiagnostics(branch, eyelid, shApplication, detailPath):
    """Keep dense debug measurements out of the bounded client manifest."""
    summaryKeys = ('method', 'status', 'policy', 'changedTexels', 'maximumAbsoluteLinearChange',
        'geometryChanged', 'uvChanged', 'cameraChanged', 'alphaPayloadChanged',
        'appearanceMeasurementStatus', 'strictSourceSupportChecks', 'sourceColorGain', 'reasons')
    payload = {'method': 'pairedShapePipelineShAndSafeEyelid', 'branch': branch,
        'diagnosticOnly': True, 'eyelid': {key: eyelid[key] for key in summaryKeys if key in eyelid},
        'shApplication': shApplication,
        'externalDebugEvidence': {'fileName': detailPath.name, 'sha256': sha(detailPath),
            'bytes': detailPath.stat().st_size, 'packaged': False}}
    encoded = json.dumps(payload, default=jsonValue, allow_nan=False)
    if len(encoded.encode('utf-8')) > 64 * 1024:
        raise ValueError('Appearance export summary exceeds its 64 KiB diagnostic budget')
    return json.loads(encoded)


def geometryStage(caseId):
    from faceAppearance.clientGeometryRefinement import refineRegisteredClientGeometry, observeClientGeometryPhotos
    from faceAppearance.geometryRefinement import _validateCameras
    from faceAppearance.clientRenderedJaw import refineRenderedClientJaw
    from faceAppearance.meshShapeConstraints import evaluateMeshShapeConstraints
    from experimentShapePreservation import buildShapeObjective, normalizedShapePolicy, buildGuardSpec

    context = inputsFor(caseId)
    root = RUN/caseId/'geometry'
    fresh(root)
    started = perf_counter()
    assets = AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    raw, camera, metadata = context['raw'], context['cameras'], context['inputs']
    transform = {name: raw[name].tolist() for name in ('rotation', 'translation', 'scale')}
    reference = context['selection']['referenceViewIndex']
    writeJson(root/'referenceSelection.json', context['selection'])
    checked = [_validateCameras([item])[0] if item.get('valid', True) else None for item in camera]
    shared = observeClientGeometryPhotos(ROOT, [None]*len(metadata), metadata, camera,
                                        workingPhotoPaths=context['photoPaths'])
    observations, records = shared['observations'], shared['reports']
    writeJson(root/'photoObservations.json', {'records': records, 'sharedAcrossBranches': True,
        'frame': 'fresh STFR inference on saved working PNG; original face anchors transformed by inputToWorking'})
    np.savez_compressed(root/'photoObservations.npz',
        points=np.stack([item['pixels'] if item else np.full((68, 2), np.nan) for item in observations]),
        confidence=np.stack([item['confidence'] if item else np.zeros(68) for item in observations]),
        accepted=np.asarray([item is not None for item in observations]))
    code = [Path(__file__), ROOT/'src/faceAppearance/clientGeometryRefinement.py',
            ROOT/'src/faceAppearance/clientRenderedJaw.py', ROOT/'src/faceAppearance/headAssembly.py',
            ROOT/'src/faceAppearance/renderedJawRuntime.py', ROOT/'src/faceAppearance/legacyAsset.py']
    hashes = {**context['hashes'], **{str(path): sha(path) for path in code}}
    objective = buildShapeObjective(raw, .01)
    guard = buildGuardSpec(objective, normalizedShapePolicy(read(PRIOR/'policy.json')))
    branchReports = {}
    for branch in BRANCHES:
        directory = root/branch
        directory.mkdir()
        initial = raw['pureDirect'] if branch == 'pureDirect' else context['candidate']['vertices']
        refined = refineRegisteredClientGeometry(assets, initial, raw['faces'], transform, camera, metadata,
            photoPaths=[None]*len(metadata), delight=context['delight'], projectRoot=ROOT,
            outputDirectory=directory/'refinement', workingPhotoPaths=context['photoPaths'],
            referenceViewIndex=reference, cachedObservations=shared,
            registrationDiagnostics={'method': 'frozenInitialShapeExperiment', 'branch': branch,
                'templateToUff': transform, 'initialRegistrationProvided': True})
        preJaw = refined['vertices'].copy()
        if reference is not None:
            jaw = refineRenderedClientJaw(assets, preJaw, raw['faces'], transform, camera, metadata,
                [None]*len(metadata), projectRoot=ROOT, outputDirectory=directory/'renderedJaw',
                referenceViewIndex=reference, photoObservations=context['fan'])
        else:
            jaw = {'vertices': preJaw.copy(), 'diagnostics': {'status': 'unchanged_no_shared_reference'}}
        final = jaw['vertices']
        fixed = raw['staticIds']
        np.testing.assert_array_equal(final[fixed], initial[fixed])
        newSkinPairs = sorted(intersectionPairs(final, raw['faces']) - intersectionPairs(initial, raw['faces']))
        check = checkFrontGeometry(initial, final, raw['faces'], camera=checked[reference],
            fixedVertexIds=fixed, maxDisplacement=float(np.ptp(initial, axis=0).max())*.07,
            maxRelativeEdgeChange=.25, minAreaRatio=.4, minNormalCosine=.3) if reference is not None else {'accepted': True}
        localValues = evaluateMeshShapeConstraints(final, guard, withJacobian=False)['values']
        localGuard = {'minimumConstraint': float(localValues.min()),
                      'violatedCount': int((localValues < -1e-8).sum()),
                      'accepted': bool((localValues >= -1e-8).all()),
                      'scope': 'Original common cavity patch guards, measured after all downstream deformation.'}
        geometry = finalGeometry(assets, final)
        mouth = dict(refined['mouthState'], afterCanonical=final.copy())
        mouth['referenceViewIndex'] = np.asarray(reference if reference is not None else -1)
        mouth['mouthObservationReliable'] = np.asarray(mouth['mouthObservationReliable'])
        saveArrays(directory/'finalGeometry.npz', geometry)
        saveArrays(directory/'mouthState.npz', mouth)
        np.savez_compressed(directory/'states.npz', initial=initial, preJaw=preJaw, final=final,
                            fixedVertexIds=fixed, **{key: np.asarray(value) for key, value in transform.items()})
        # Fresh accessory assembly for each branch; no old postprocessing delta.
        template = assets.materials[0]['baseColor'][..., :3]
        assembly = assets.assembleFinalClientSurface(geometry, transform, template,
            {'coverageMask': np.zeros(template.shape[:2], bool)},
            accessoryReferenceCanonical=preJaw,
            diagnostics={'method': 'pairedShapePipeline', 'branch': branch, 'diagnosticOnly': True,
                         'referenceSelection': context['selection']})
        exported = exportLegacyAsset(directory/'geometryAsset', assembly, gender='male')
        full = assembly['geometry']['vertices'] @ np.asarray(transform['rotation'])*transform['scale'] + transform['translation']
        np.savez_compressed(directory/'fullGeometry.npz', vertices=full, faces=assembly['geometry']['faces'])
        report = {'caseId': caseId, 'branch': branch, 'status': 'geometryDiagnosticComplete',
            'referenceViewIndex': reference, 'geometryRefinement': refined['diagnostics'],
            'renderedJaw': jaw['diagnostics'], 'cumulativeDownstreamGate': check,
            'newSkinPairsAgainstBranchInitial': newSkinPairs, 'finalCavityGuard': localGuard,
            'downstreamNumericalAccepted': bool(check['accepted'] and not newSkinPairs),
            'sourceHashes': hashes, 'templateToUff': transform,
            'initialToFinalMaximumMm': float(np.linalg.norm(final-initial, axis=1).max()/raw['unit']),
            'asset': exported, 'finalCandidateAccepted': False,
            'outputHashes': {name: sha(directory/name) for name in ('finalGeometry.npz', 'mouthState.npz', 'states.npz', 'fullGeometry.npz')},
            'limitations': ['Geometry-only template-colored diagnostic asset; not a final texture.',
                'Uses common production eye/mouth/jaw stages; historical AI eye-ring incremental optimization is not included.']}
        verify(hashes)
        writeJson(directory/'report.json', report)
        branchReports[branch] = {'path': str(directory/'report.json'), 'sha256': sha(directory/'report.json'),
                                'downstreamNumericalAccepted': report['downstreamNumericalAccepted'], 'finalCavityGuard': localGuard}
        print(json.dumps({'case': caseId, 'stage': 'geometry', 'branch': branch, **branchReports[branch]}), flush=True)
    verify(hashes)
    writeJson(root/'report.json', {'status': 'pairedGeometryComplete', 'caseId': caseId,
        'sourceHashes': hashes, 'branches': branchReports, 'seconds': perf_counter()-started,
        'referenceSelection': context['selection'], 'finalCandidateAccepted': False})


def appearanceStage(caseId, branch):
    from faceAppearance.appearanceObservations import srgbToLinear, linearToGamma22
    from faceAppearance.clientShDelight import delightClientPhotos
    from faceAppearance.clientTextureBaking import bakeClientTexture
    from faceAppearance.clientEyelidAppearance import refineClientEyelidAppearance
    from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs
    from faceAppearance.legacyAppearanceVariant import exportLegacyAppearanceVariant

    context = inputsFor(caseId)
    geometryRoot = RUN/caseId/'geometry'/branch
    geometryReport = read(geometryRoot/'report.json')
    verify(geometryReport['sourceHashes'])
    verify({str(geometryRoot/name): value for name, value in geometryReport['outputHashes'].items()})
    output = RUN/caseId/'appearance'/branch
    fresh(output)
    started = perf_counter()
    denseRoot = RUN/caseId/'observations'
    denseReport, dense = validatedEyeObservations(context, denseRoot)
    paths = [geometryRoot/'report.json', denseRoot/'observations.json', denseRoot/'observations.npz',
             Path(__file__), ROOT/'src/faceAppearance/clientShDelight.py', ROOT/'src/faceAppearance/clientTextureBaking.py',
             ROOT/'src/faceAppearance/clientEyelidAppearance.py', ROOT/'.runtimes/colmap/bin/colmap']
    paths += [geometryRoot/name for name in geometryReport['outputHashes']]
    paths += [Path(geometryReport['asset']['archivePath'])]
    if (sha(paths[-1]) != geometryReport['asset']['archiveSha256']
            or paths[-1].stat().st_size != geometryReport['asset']['sizeBytes']):
        raise ValueError('Geometry asset changed after its recorded export')
    hashes = {**geometryReport['sourceHashes'], **denseReport['sourceHashes'],
              **{str(path): sha(path) for path in paths}}
    geometry = arrays(geometryRoot/'finalGeometry.npz'); geometry['uvConvention'] = 'topLeftPixelEdges'
    mouth = arrays(geometryRoot/'mouthState.npz')
    reference = geometryReport['referenceViewIndex']
    photos = np.stack([np.asarray(Image.open(path).convert('RGB'), np.float32)/255 for path in context['photoPaths']])
    sh = delightClientPhotos(geometry, context['cameras'], srgbToLinear(photos),
        context['delight']['validMasks'].astype(bool), referenceViewIndex=reference, fitResolution=256, iterations=8)
    saveArrays(output/'shImages.npz', sh)
    writeJson(output/'shDiagnostics.json', sh['diagnostics'])
    assets = AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    baked = bakeClientTexture(geometry, context['cameras'],
        {'albedoImages': linearToGamma22(sh['linearImages']), 'validMasks': sh['validMasks']},
        assets.materials[0]['baseColor'][..., :3], output/'baking',
        colmapExecutable=ROOT/'.runtimes/colmap/bin/colmap', mouthState=mouth,
        referenceView=reference, fallbackGain='auto', threads=2)
    states = arrays(geometryRoot/'states.npz')
    provenance = {key: value for key, value in baked.items() if isinstance(value, np.ndarray) and key != 'linearRGB'}
    assembled = assets.assembleFinalClientSurface(geometry, geometryReport['templateToUff'], baked['linearRGB'],
        provenance, accessoryReferenceCanonical=states['preJaw'],
        diagnostics={'method': 'pairedShapePipelineShBake', 'branch': branch, 'diagnosticOnly': True,
                     'referenceSelection': context['selection'], 'shApplication': sh['diagnostics']['applicationSummary']})
    bakedArchive = exportLegacyAsset(output/'bakedAsset', assembled, gender='male')
    finalColor = baked['linearRGB']
    eyelid = {'status': 'skipped', 'reason': 'photoContoursRejectedOrReferenceUnavailable'}
    if reference is not None and denseReport['status'] == 'accepted':
        scene = loadLegacyEyeBindingInputs(bakedArchive['archivePath'], assets.arrays['nicpToFullIds'])
        curves = dense['eyelidCurvesWorking']
        eyes = {name: {part: curves[index, partIndex] for partIndex, part in enumerate(('lower', 'upper'))}
                for index, name in enumerate(('imageLeft', 'imageRight'))}
        edited = refineClientEyelidAppearance(geometry, scene, context['cameras'][reference],
            sh['linearImages'][reference], (sh['validMasks'][reference] >= .999) & sh['correctionMasks'][reference],
            baked['linearRGB'], eyes)
        finalColor, eyelid = edited['linearRGB'], edited['diagnostics']
        saveArrays(output/'eyelidMaps.npz', edited['maps'])
    writeJson(output/'eyelidDiagnostics.json', eyelid)
    np.savez_compressed(output/'finalTexture.npz', linearRGB=finalColor)
    exportDiagnostics = appearanceExportDiagnostics(branch, eyelid,
        sh['diagnostics']['applicationSummary'], output/'eyelidDiagnostics.json')
    finalAsset = exportLegacyAppearanceVariant(bakedArchive['archivePath'], finalColor, output/'asset',
        diagnostics=exportDiagnostics)
    verify(hashes)
    report = {'status': 'appearanceDiagnosticComplete', 'caseId': caseId, 'branch': branch,
        'sourceHashes': hashes, 'geometryReport': str(geometryRoot/'report.json'),
        'referenceViewIndex': reference, 'shApplication': sh['diagnostics']['applicationSummary'],
        'baking': baked['diagnostics'], 'eyelid': exportDiagnostics['eyelid'],
        'eyelidDetail': exportDiagnostics['externalDebugEvidence'], 'asset': finalAsset,
        'outputHashes': {str(path.relative_to(output)): sha(path) for path in output.rglob('*') if path.is_file()},
        'finalCandidateAccepted': False, 'deployed': False, 'seconds': perf_counter()-started,
        'limitations': ['SH raw fallback retains lighting; it is not successful albedo recovery.',
            'All appearance fitted to these photographs; image similarity is not independent relighting accuracy.',
            'Visible eyelid relocation is estimated appearance, not new geometry or measured skin texture.']}
    writeJson(output/'report.json', report)
    print(json.dumps({'case': caseId, 'stage': 'appearance', 'branch': branch,
                      'seconds': report['seconds'], 'asset': finalAsset['archivePath']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--stage', choices=('geometry', 'appearance'), required=True)
    parser.add_argument('--branch', choices=BRANCHES)
    args = parser.parse_args()
    if args.stage == 'geometry':
        geometryStage(args.case)
    else:
        if args.branch is None:
            parser.error('--branch is required for appearance')
        appearanceStage(args.case, args.branch)
