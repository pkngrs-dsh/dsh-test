"""Independent real-subject eyelid color sampling on frozen cavity geometry.

The source is each subject's own OpenDelight cache. A single CPU STFR instance
detects five crop variants per subject. Original and resampled eye apertures
both protect every bilinear source contributor. No geometry is fitted here.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.adapters.eyelidLandmarks import EyelidLandmarksAdapter
from faceAppearance.artifacts import writeJson
from faceAppearance.camera import projectPoints
from faceAppearance.colmapTexture import _destinationUvLookup
from faceAppearance.eyelidTextureSampling import reprojectEyelidTexture, _outsidePolygon
from faceAppearance.headAssembly import AssemblyAssets, decodeColor
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.texture import interpolateSurface
from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs, bindVisibleEyeContours, extractVisibleEyeContours
from prepareFrontLandmarkTextures import encodeSrgb
from runClientDenseEyeExperiment import compactBinding
from runClientEyelidTextureExperiment import archiveTexture

POLICY = {
    'referenceViewIndex': 0, 'coreWorkingPixels': 12., 'outerWorkingPixels': 24.,
    'sourceSafetyWorkingPixels': .75, 'maximumExtensionWorkingPixels': 4.,
    'photoBoundsPaddingWorkingPixels': 14., 'destinationBoxFeatherWorkingPixels': 4.,
    'maximumCropVariationWorkingPixels': 3., 'pointsPerLid': 24,
    'anchorOffsetsWorkingPixels': [[0, 0], [2, 0], [-2, 0], [0, 2], [0, -2]],
    'filterFootprint': False, 'sourceColorGain': [1., 1., 1.],
    'sourceColorEncoding': 'OpenDelight native gamma2.2 decoded into linear RGB',
    'baselineColorEncoding': 'Existing head color sRGB decoded into linear RGB',
    'outputColorEncoding': 'sRGB encoded PNG',
}


def loadArrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verifyHashes(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen eyelid input changed: {path}')


def detectPhotoEyes(detector, photo, fan, photoPath, output):
    anchors = np.stack([fan[36:42].mean(0), fan[42:48].mean(0), fan[30], fan[57]])
    offsets = np.asarray(POLICY['anchorOffsetsWorkingPixels'])
    predictions = [detector.infer(photo, faceAnchors=anchors + offset) for offset in offsets]
    first = predictions[0]
    curves = np.stack([prediction['eyelidCurves'] for prediction in predictions])
    variation = np.linalg.norm(curves - curves[:1], axis=-1).max(0)
    accepted = first['pointValidMask'].all(axis=(1, 2))
    accepted &= variation.max(axis=(1, 2)) <= POLICY['maximumCropVariationWorkingPixels']
    accepted &= np.asarray([eye['valid'] for eye in first['quality']])
    observation = {
        'status': 'accepted' if accepted.all() else 'rejected',
        'inferenceFrame': 'savedWorkingRgbPng', 'sourcePhotoSha256': sha(photoPath),
        'originalResolutionInference': False, 'referenceViewIndex': POLICY['referenceViewIndex'],
        'model': detector.modelMetadata, 'quality': first['quality'],
        'maximumCropVariationWorkingPixels': variation.max(axis=(1, 2)),
        'acceptedEyeMask': accepted, 'anchorOffsetsWorkingPixels': offsets,
        'maximumAllowedVariationWorkingPixels': POLICY['maximumCropVariationWorkingPixels'],
    }
    writeJson(output / 'observations.json', observation)
    np.savez_compressed(output / 'observations.npz', eyelidCurvesWorking=curves[0],
        cropVariantCurvesWorking=curves, maximumVariationWorkingPixels=variation,
        rawPeakScores=first['rawPeakScores'], pointValidMask=first['pointValidMask'],
        acceptedEyeMask=accepted, faceAnchors=anchors, cropBoxes=first['cropBoxes'])
    photoEyes = {name: {'lower': curves[0, index, 0], 'upper': curves[0, index, 1]}
                 for index, name in enumerate(('imageLeft', 'imageRight'))}
    return photoEyes, observation, bool(accepted.all())


def reusePhotoEyes(directory, photoPath, fanPath, camerasPath, photo, output, detector):
    """Reuse separately audited inference only after checking source identity."""
    observation = json.loads((directory / 'observations.json').read_text())
    values = loadArrays(directory / 'observations.npz')
    if (observation['sourcePhotoSha256'] != sha(photoPath)
            or observation['sourceFanSha256'] != sha(fanPath)
            or observation['sourceCameraSha256'] != sha(camerasPath)
            or observation['model']['weightSha256'] != sha(detector.assetPath())
            or observation['referenceViewIndex'] != POLICY['referenceViewIndex']
            or observation['anchorOffsetsWorkingPixels'] != POLICY['anchorOffsetsWorkingPixels']
            or observation['maximumAllowedVariationWorkingPixels'] != POLICY['maximumCropVariationWorkingPixels']):
        raise ValueError('Independent photo observation cache identity/policy differs')
    variants = values['cropVariantCurvesWorking']
    variation = np.linalg.norm(variants - variants[:1], axis=-1).max(0)
    np.testing.assert_array_equal(values['eyelidCurvesWorking'], variants[0])
    np.testing.assert_array_equal(values['maximumVariationWorkingPixels'], variation)
    valid = (values['pointValidMask'].all(axis=(1, 2))
        & (values['rawPeakScores'] >= detector.minimumPeakScore).all(axis=(1, 2))
        & (variation.max(axis=(1, 2)) <= POLICY['maximumCropVariationWorkingPixels'])
        & np.asarray([eye['valid'] for eye in observation['quality']]))
    np.testing.assert_array_equal(valid, values['acceptedEyeMask'])
    if (variants.shape != (5, 2, 2, 12, 2) or not np.isfinite(variants).all()
            or (variants < 0).any() or (variants >= np.asarray(photo.shape[1::-1])).any()):
        raise ValueError('Independent photo curves are malformed or out of bounds')
    shutil.copyfile(directory / 'observations.npz', output / 'observations.npz')
    writeJson(output / 'observations.json', observation)
    curves = values['eyelidCurvesWorking']
    photoEyes = {name: {'lower': curves[index, 0], 'upper': curves[index, 1]}
                 for index, name in enumerate(('imageLeft', 'imageRight'))}
    return photoEyes, observation, bool(valid.all())


def localSamplingDomain(geometry, camera, lookup, sourceValid, photoEyes, originalPhotoEyes,
                        protected):
    skinDomain = lookup['validMask']
    surface = interpolateSurface(geometry['vertices'], geometry['faces'], lookup)
    projected, depth = projectPoints(surface[skinDomain], np.asarray(camera['intrinsics']),
        np.asarray(camera['rotation']), np.asarray(camera['translation']))
    rows, columns = np.indices(sourceValid.shape)
    sourceBand = np.zeros(sourceValid.shape, bool)
    destinationBand = np.zeros(len(projected), bool)
    featherValues = np.zeros(len(projected), np.float64)
    boxes = []
    for eye in photoEyes.values():
        points = np.concatenate([eye['upper'], eye['lower']])
        minimum = points.min(0) - POLICY['photoBoundsPaddingWorkingPixels']
        maximum = points.max(0) + POLICY['photoBoundsPaddingWorkingPixels']
        boxes.append(np.r_[minimum, maximum].tolist())
        sourceBand |= ((columns >= minimum[0]) & (columns <= maximum[0])
                       & (rows >= minimum[1]) & (rows <= maximum[1]))
        destinationBand |= ((projected >= minimum).all(1) & (projected <= maximum).all(1)
                            & (depth > 0))
        distance = np.minimum(projected - minimum, maximum - projected).min(1)
        fraction = np.clip(distance / POLICY['destinationBoxFeatherWorkingPixels'], 0, 1)
        featherValues = np.maximum(featherValues, fraction * fraction * (3 - 2 * fraction))
    localDomain = np.zeros(skinDomain.shape, bool)
    localDomain[skinDomain] = destinationBand
    writable = skinDomain & localDomain & ~protected
    feather = np.zeros(skinDomain.shape, np.float32)
    feather[skinDomain] = featherValues
    originalValid = sourceValid.copy()
    sourceValid &= sourceBand
    sourceRows, sourceColumns = np.nonzero(sourceValid)
    coordinates = np.column_stack([sourceColumns, sourceRows])
    safeOriginal = np.ones(len(coordinates), bool)
    for eye in originalPhotoEyes.values():
        polygon = np.concatenate([eye['upper'], eye['lower'][::-1]])
        safeOriginal &= _outsidePolygon(coordinates, polygon, POLICY['sourceSafetyWorkingPixels'])
    sourceValid[sourceRows, sourceColumns] &= safeOriginal
    return writable, feather, sourceValid, originalValid, boxes


def validateSourceFootprints(result, sourceValid, photoEyes, originalPhotoEyes):
    queryMask = (result['alpha'] > 0) | result['filterPadding']
    query = result['sourceXY'][queryMask]
    if not np.isfinite(query).all():
        raise ValueError('Modified eyelid texel has a nonfinite source query')
    floors = np.floor(query).astype(int)
    support = np.stack([floors, floors + [0, 1], floors + [1, 0], floors + [1, 1]], axis=1)
    inBounds = ((support >= 0) & (support < np.asarray(sourceValid.shape[::-1]))).all(-1)
    supportValid = np.zeros(inBounds.shape, bool)
    supportValid[inBounds] = sourceValid[support[inBounds, 1], support[inBounds, 0]]
    outside = np.ones(inBounds.shape, bool)
    for eye in list(photoEyes.values()) + list(originalPhotoEyes.values()):
        polygon = np.concatenate([eye['upper'], eye['lower'][::-1]])
        outside &= _outsidePolygon(support.reshape(-1, 2), polygon,
            POLICY['sourceSafetyWorkingPixels']).reshape(inBounds.shape)
    invalid = ~(inBounds & supportValid & outside).all(1)
    checks = {
        'sourceQueriesIncludingFilterPadding': int(queryMask.sum()),
        'allFourSourcePixelsInBounds': bool(inBounds.all()),
        'allFourSourcePixelsValid': bool(supportValid.all()),
        'allFourSourcePixelsOutsideBothPhotoOpeningsWithMargin': bool(outside.all()),
        'originalDenseAndResampledContoursBothChecked': True,
        'invalidSourceSupportQueryCount': int(invalid.sum()),
    }
    return checks, queryMask


def runCase(shortId, output, detector, assets, observationsRoot=None):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh real eyelid experiment directory')
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    parent = ROOT / 'runs/feedbackFix20261024' / shortId / 'cavityGeneralizationCandidate'
    parentReportPath = parent / 'report.json'
    parentReport = json.loads(parentReportPath.read_text())
    archive = parent / 'asset/legacyAsset.zip'
    case = ROOT / 'reports/feedbackAudit20261024/cases' / parentReport['sourceTaskId']
    reconstruction = case / 'reconstruction'
    view = POLICY['referenceViewIndex']
    photoPath = reconstruction / f'input{view:02d}.png'
    geometryPath = reconstruction / 'directClient/geometry.npz'
    cavityPath = ROOT / 'runs/noseGeneralization20261024' / shortId / 'cavityOnly/geometry.npz'
    cavityReportPath = cavityPath.with_name('report.json')
    fanPath = reconstruction / 'directClient/renderedJaw/photoObservations.npz'
    paths = [parentReportPath, archive, photoPath, geometryPath, cavityPath, cavityReportPath,
        fanPath, reconstruction / 'cameras.json', reconstruction / 'inputs.json',
        reconstruction / 'delight.npz', detector.assetPath(),
        ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1/template.npz',
        ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1/manifest.json',
        ROOT / 'src/faceAppearance/eyelidTextureSampling.py',
        ROOT / 'src/faceAppearance/visibleEyeContours.py',
        ROOT / 'src/faceAppearance/adapters/eyelidLandmarks.py',
        ROOT / 'tools/runClientEyelidTextureExperiment.py', Path(__file__)]
    if observationsRoot is not None:
        paths.extend([observationsRoot / shortId / 'observations.npz',
                      observationsRoot / shortId / 'observations.json'])
    sourceHashes = {str(path): sha(path) for path in paths}
    if (parentReport['status'] != 'complete' or parentReport['variant']['sha256'] != sha(archive)
            or not parentReport['sourceTaskId'].startswith(shortId)):
        raise ValueError('Cavity parent identity differs')
    cavityReport = json.loads(cavityReportPath.read_text())
    if (cavityReport['variant']['sha256'] != sha(archive)
            or not cavityReport['baselinePassedCumulativeGates']):
        raise ValueError('Parent cavity archive does not match the accepted geometry diagnostic')
    report = {
        'status': 'started', 'diagnosticOnly': True, 'finalCandidateAccepted': False,
        'defaultPipelineChanged': False, 'deployed': False, 'caseId': case.name,
        'method': 'realSubjectOpenDelightSafeRelocatedEyelidColor', 'policy': POLICY,
        'sourceArchiveSha256': sha(archive), 'sourceArchive': str(archive),
        'sourceDelightSha256': sha(reconstruction / 'delight.npz'),
        'sourceHashes': sourceHashes, 'geometryChanged': False, 'cameraChanged': False,
        'sourceColorGain': [1., 1., 1.], 'device': 'cpu', 'threadCount': 1,
        'fullFaceRebake': False, 'sourceShApplied': False,
        'limitations': [
            'Working photo inference has lower resolution than the original photo.',
            'Contour warps and bounded exterior extension are estimated appearance.',
            'OpenDelight albedo can lose detail and retain color bias; no extra color calibration is applied.',
            'Eye aperture, lid geometry, eyeball position and gaze remain unchanged.',
            'Automatic dense curves are not manually annotated anatomical ground truth.',
            'Masked brow/lash cards are excluded from conservative occlusion queries but retain their original display alpha.'],
    }
    writeJson(output / 'policy.json', POLICY)
    writeJson(output / 'report.json', report)
    cameras = json.loads((reconstruction / 'cameras.json').read_text())
    camera = cameras[view]
    if not camera['valid'] or camera['viewIndex'] != view:
        raise ValueError('Reference camera is not valid')
    photo = np.asarray(Image.open(photoPath).convert('RGB')).copy()
    fan = loadArrays(fanPath)['faceLandmarksWorking'][view]
    if observationsRoot is None:
        originalPhotoEyes, observation, accepted = detectPhotoEyes(detector, photo, fan, photoPath, output)
    else:
        originalPhotoEyes, observation, accepted = reusePhotoEyes(observationsRoot / shortId,
            photoPath, fanPath, reconstruction / 'cameras.json', photo, output, detector)
        report['observationCacheSource'] = str(observationsRoot / shortId)
    report['observation'] = observation
    if not accepted:
        report.update(status='rejectedPhotoContourQuality', seconds=perf_counter() - started)
        verifyHashes(sourceHashes)
        writeJson(output / 'report.json', report)
        return report
    scene = loadLegacyEyeBindingInputs(archive, assets.arrays['nicpToFullIds'])
    binding = bindVisibleEyeContours(**scene, camera=camera, photoEyes=originalPhotoEyes,
        imageShape=photo.shape[:2], pointsPerLid=POLICY['pointsPerLid'])
    writeJson(output / 'binding.json', compactBinding(binding))
    measured = extractVisibleEyeContours(**scene, camera=camera, photoEyes=originalPhotoEyes,
        imageShape=photo.shape[:2], pointsPerLid=POLICY['pointsPerLid'])
    writeJson(output / 'measurementContours.json', compactBinding(measured))
    report['geometryBindingStatus'] = {name: binding['eyes'][name]['status'] for name in originalPhotoEyes}
    report['appearanceMeasurementStatus'] = {name: measured['eyes'][name]['status'] for name in originalPhotoEyes}
    report['curveUse'] = 'Appearance-only visible sourcePixels/targetPixels; no NICP barycentric fitting or deformation'
    if any(measured['eyes'][name]['status'] != 'accepted' for name in originalPhotoEyes):
        report.update(status='rejectedVisibleAppearanceMeasurement', seconds=perf_counter() - started)
        verifyHashes(sourceHashes)
        writeJson(output / 'report.json', report)
        return report
    # Use the matched arc-length resampling stored by the binding; raw detector
    # points remain a second safety polygon, not mismatched correspondence IDs.
    modelEyes = {name: {part: np.asarray(measured['eyes'][name][part]['sourcePixels'])
        for part in ('upper', 'lower')} for name in originalPhotoEyes}
    photoEyes = {name: {part: np.asarray(measured['eyes'][name][part]['targetPixels'])
        for part in ('upper', 'lower')} for name in originalPhotoEyes}
    source = _load(archive)
    geometry = loadArrays(geometryPath)
    geometry['vertices'] = loadArrays(cavityPath)['candidate']
    geometry['uvConvention'] = 'topLeftPixelEdges'
    skinIds = assets.arrays['nicpToFullIds']
    difference = float(np.abs(scene['vertices'][skinIds] - geometry['vertices']).max())
    if difference > 1e-12:
        raise ValueError('Cavity ZIP skin differs from the recorded candidate geometry')
    faceLookup = {tuple(face): index for index, face in enumerate(scene['faces'])}
    fullFaceIds = np.asarray([faceLookup[tuple(face)] for face in skinIds[geometry['faces']]])
    np.testing.assert_array_equal(geometry['uvs'], scene['uvs'])
    np.testing.assert_array_equal(geometry['uvFaces'], scene['uvFaces'][fullFaceIds])
    original = np.asarray(Image.open(io.BytesIO(source.payloads['head_color.png']))).copy()
    originalAlias = np.asarray(Image.open(io.BytesIO(source.payloads['face_nicp.png']))).copy()
    np.testing.assert_array_equal(original, originalAlias)
    if original.shape[0] != original.shape[1] or original.shape[-1] not in (3, 4):
        raise ValueError('Expected square RGB or RGBA legacy head atlas')
    size = original.shape[0]
    member = np.zeros(len(scene['vertices']), bool)
    member[skinIds] = True
    accessoryFaces = (scene['materialIds'] == 0) & ~member[scene['faces']].all(1)
    accessory = {'vertices': scene['vertices'], 'faces': scene['faces'][accessoryFaces],
        'uvs': scene['uvs'], 'uvFaces': scene['uvFaces'][accessoryFaces],
        'uvConvention': 'topLeftPixelEdges'}
    protected = binary_dilation(_destinationUvLookup(accessory, size)[0]['validMask'], iterations=2)
    lookup = _destinationUvLookup(geometry, size)[0]
    delight = loadArrays(reconstruction / 'delight.npz')
    images, masks = delight['albedoImages'], delight['validMasks']
    if (images.shape != (len(cameras), *photo.shape) or masks.shape != images.shape[:-1]
            or not np.isfinite(images).all() or not np.isfinite(masks).all()
            or images.min() < 0 or images.max() > 1 or masks.min() < 0 or masks.max() > 1):
        raise ValueError('OpenDelight cache must match the saved working-camera image frame')
    sourceLinear = decodeColor(images[view], 'gamma2.2')
    sourceValid = masks[view] >= .999
    writable, feather, sourceValid, originalValid, boxes = localSamplingDomain(
        geometry, camera, lookup, sourceValid, photoEyes, originalPhotoEyes, protected)
    baselineLinear = decodeColor(original[..., :3] / 255., 'srgb')
    visible = {'vertices': scene['vertices'], 'faces': scene['faces'][scene['materialIds'] != 3]}
    result = reprojectEyelidTexture(geometry, camera, sourceLinear, sourceValid,
        baselineLinear, photoEyes, modelEyes, writableMask=writable, visibleScene=visible,
        corePixels=POLICY['coreWorkingPixels'], outerPixels=POLICY['outerWorkingPixels'],
        sourceSafetyPixels=POLICY['sourceSafetyWorkingPixels'],
        maxExtensionPixels=POLICY['maximumExtensionWorkingPixels'], filterFootprint=False)
    result['linearRGB'] = (baselineLinear + (result['linearRGB'] - baselineLinear)
        * feather[..., None]).astype(baselineLinear.dtype)
    result['alpha'] *= feather
    kept = result['alpha'] > 0
    for key in ('observedMask', 'estimatedMask', 'warpMask', 'extensionMask'):
        result[key] &= kept
    checks, queryMask = validateSourceFootprints(result, sourceValid, photoEyes, originalPhotoEyes)
    report['strictSourceSupportChecks'] = checks
    np.savez_compressed(output / 'textureProvenance.npz',
        **{key: value for key, value in result.items() if isinstance(value, np.ndarray)},
        writableMask=writable, protectedAccessoryUvs=protected,
        sourceValidMask=sourceValid, originalOpenDelightValidMask=originalValid,
        destinationFeather=feather, skinDomainMask=lookup['validMask'],
        sourceLinearRGB=sourceLinear,
        rawPhotoCurves=np.asarray([[originalPhotoEyes[name][part] for part in ('lower', 'upper')]
                                  for name in ('imageLeft', 'imageRight')]),
        resampledPhotoCurves=np.asarray([[photoEyes[name][part] for part in ('lower', 'upper')]
                                        for name in ('imageLeft', 'imageRight')]),
        modelCurves=np.asarray([[modelEyes[name][part] for part in ('lower', 'upper')]
                              for name in ('imageLeft', 'imageRight')]))
    if checks['invalidSourceSupportQueryCount']:
        report.update(status='rejectedIndependentSourceSupport', seconds=perf_counter() - started)
        verifyHashes(sourceHashes)
        writeJson(output / 'report.json', report)
        return report
    modified = queryMask
    pixels = original.copy()
    pixels[modified, :3] = np.rint(encodeSrgb(result['linearRGB'][modified]) * 255).astype(np.uint8)
    if (modified & ~writable).any() or not np.array_equal(pixels[~modified], original[~modified]):
        raise ValueError('Eyelid color edit escaped the permitted local UV domain')
    if not np.array_equal(pixels[protected], original[protected]):
        raise ValueError('Protected accessory UV color changed')
    if original.shape[-1] == 4 and not np.array_equal(pixels[..., 3], original[..., 3]):
        raise ValueError('Head alpha changed')
    changed = np.any(pixels[..., :3] != original[..., :3], axis=-1)
    colorDelta = pixels[..., :3].astype(float) - original[..., :3].astype(float)
    description = {key: report[key] for key in (
        'method', 'diagnosticOnly', 'finalCandidateAccepted', 'deployed', 'caseId',
        'sourceArchiveSha256', 'sourceDelightSha256', 'geometryChanged', 'cameraChanged',
        'sourceColorGain', 'fullFaceRebake', 'sourceShApplied')}
    description.update(policy=POLICY, strictSourceSupportChecks=checks,
        sourceValidMask='Original OpenDelight validMasks AND finite photo eyelid band AND original dense aperture exclusion; no support expansion',
        sourceAndDestinationBoxesWorking=boxes, textureDiagnostics=result['diagnostics'])
    Image.fromarray(original).save(output / 'beforeTexture.png')
    Image.fromarray(pixels).save(output / 'afterTexture.png')
    Image.fromarray(np.rint(result['alpha'] * 255).astype(np.uint8)).save(output / 'editAlpha.png')
    Image.fromarray(np.rint(encodeSrgb(sourceLinear) * 255).astype(np.uint8)).save(output / 'frontDelight.png')
    verifyHashes(sourceHashes)
    if not changed.any():
        report.update(status='noSafeVisibleColorChange', seconds=perf_counter() - started)
        writeJson(output / 'report.json', report)
        return report
    exported = archiveTexture(archive, output / 'asset', pixels, description)
    checked = _load(output / 'asset/legacyAsset.zip')
    if checked.manifest['binding'] != source.manifest['binding']:
        raise ValueError('Texture export changed binding metadata')
    verifyHashes(sourceHashes)
    report.update(description, status='completeLocalTextureCandidate', seconds=perf_counter() - started,
        archive=exported, archivePath=str((output / 'asset/legacyAsset.zip').resolve()),
        maximumGeometryCanonicalRoundTripError=difference, changedRgbTexels=int(changed.sum()),
        changedRgbFraction=float(changed.mean()),
        meanAbsoluteChangedChannelDifference8Bit=float(np.abs(colorDelta[changed]).mean()),
        maximumChangedChannelDifference8Bit=float(np.abs(colorDelta[changed]).max()),
        meanSignedChangedRgbDifference8Bit=colorDelta[changed].mean(0),
        outsideEditPixelsExactlyIdentical=True, protectedAccessoryUvsExactlyIdentical=True,
        headAlphaExactlyIdentical=True, bindingMetadataExactlyIdentical=True,
        observationsSha256=sha(output / 'observations.npz'),
        bindingSha256=sha(output / 'binding.json'), provenanceSha256=sha(output / 'textureProvenance.npz'))
    report['measurementContoursSha256'] = sha(output / 'measurementContours.json')
    report['outputHashes'] = {path.name: sha(path) for path in output.iterdir() if path.is_file() and path.name != 'report.json'}
    writeJson(output / 'report.json', report)
    print(json.dumps({'case': shortId, 'status': report['status'], 'changedRgbTexels': int(changed.sum()),
        'maximumCropVariation': np.asarray(observation['maximumCropVariationWorkingPixels']).tolist(),
        'seconds': report['seconds']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs/realRefinement20261024')
    parser.add_argument('--cases', nargs='+', default=['110bac9f', '06732111', '10840519'])
    parser.add_argument('--observations-root', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    detector = EyelidLandmarksAdapter(ROOT)
    if args.observations_root is None:
        detector.loadModels()
    assets = AssemblyAssets(ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1')
    reports = []
    for shortId in args.cases:
        reports.append(runCase(shortId, args.output_root.resolve() / shortId / 'eyelidOnly', detector, assets,
            args.observations_root.resolve() if args.observations_root else None))
    writeJson(args.output_root.resolve() / 'eyelidBatchReport.json', {
        'status': 'completeLocalEyelidTextureBatch', 'modelLoadCountInThisProcess': int(detector.isLoaded),
        'observationCacheReused': bool(args.observations_root),
        'cpuThreads': 1, 'policy': POLICY, 'finalCandidateAccepted': False,
        'cases': [{key: report.get(key) for key in ('caseId', 'status', 'changedRgbTexels',
            'archivePath', 'meanAbsoluteChangedChannelDifference8Bit', 'seconds')} for report in reports]})
