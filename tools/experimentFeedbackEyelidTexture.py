"""Local fixed-geometry eyelid appearance replay using safe photographic skin.

All old assets stay untouched. Fresh STFR contours are inferred on the cached
working photograph, not original-resolution input. Geometry, cameras, UVs,
accessories and all pixels outside the accepted local edit remain identical.
"""
from __future__ import annotations
import argparse, hashlib, io, json, sys
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.adapters.eyelidLandmarks import EyelidLandmarksAdapter
from faceAppearance.artifacts import writeJson
from faceAppearance.colmapTexture import _destinationUvLookup
from faceAppearance.fastFrontEyeTexture import reprojectFastFrontEyeTexture
from faceAppearance.eyelidTextureSampling import reprojectEyelidTexture, _outsidePolygon
from faceAppearance.camera import projectPoints
from faceAppearance.texture import interpolateSurface
from faceAppearance.headAssembly import AssemblyAssets, decodeColor
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs, bindVisibleEyeContours
from runClientDenseEyeExperiment import compactBinding
from runClientEyelidTextureExperiment import archiveTexture
from prepareFrontLandmarkTextures import encodeSrgb


def loadArrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(caseDirectory, appearanceDirectory, geometryPath, outputDirectory, referenceView=3, method="bounded"):
    case, appearance, output = map(Path, (caseDirectory, appearanceDirectory, outputDirectory))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new eyelid experiment directory')
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    prior = json.loads((appearance/'report.json').read_text())
    archive = appearance/'asset/legacyAsset.zip'
    if prior['status'] != 'complete' or prior['sourceTaskId'] != case.name or prior['variant']['sha256'] != sha(archive):
        raise ValueError('Completed source appearance identity differs')
    root = case/'reconstruction'
    for relative in ('shImages.npz', 'baking/clientTexture.npz'):
        if sha(appearance/relative) != prior['artifactIdentities'][relative]['sha256']:
            raise ValueError('Appearance cache differs from its recorded artifact identity')
    for relative in ('cameras.json', f'input{referenceView:02d}.png'):
        if sha(root/relative) != prior['sourceHashes'][relative]:
            raise ValueError('Photo/camera cache differs from the appearance source')
    expectedGeometryHash = prior['geometryReplay']['sourceHashes'].get(str(Path(geometryPath).resolve()))
    if expectedGeometryHash != sha(geometryPath):
        raise ValueError('Geometry differs from appearance replay source')
    cameras = json.loads((root/'cameras.json').read_text())
    camera = cameras[referenceView]
    if camera.get('viewIndex') != referenceView or not camera.get('valid'):
        raise ValueError('Reference camera is not valid')
    photoPath = root/f'input{referenceView:02d}.png'
    photo = np.asarray(Image.open(photoPath).convert('RGB')).copy()
    fan = loadArrays(root/'directClient/renderedJaw/photoObservations.npz')['faceLandmarksWorking'][referenceView]
    anchors = np.stack([fan[36:42].mean(0), fan[42:48].mean(0), fan[30], fan[57]])
    cv2.setNumThreads(1)
    detector = EyelidLandmarksAdapter(ROOT).loadModels()
    offsets = np.array([[0, 0], [2, 0], [-2, 0], [0, 2], [0, -2]])
    predictions = [detector.infer(photo, faceAnchors=anchors+offset) for offset in offsets]
    first = predictions[0]
    curves = np.stack([prediction['eyelidCurves'] for prediction in predictions])
    variation = np.linalg.norm(curves-curves[:1], axis=-1).max(0)
    accepted = first['pointValidMask'].all(axis=(1, 2)) & (variation.max(axis=(1, 2)) <= 3.)
    accepted &= np.array([eye['valid'] for eye in first['quality']])
    observation = {'status': 'accepted' if accepted.all() else 'rejected',
        'inferenceFrame': 'savedWorkingRgbPng', 'sourcePhotoSha256': sha(photoPath),
        'originalResolutionInference': False, 'referenceViewIndex': referenceView,
        'model': detector.modelMetadata, 'quality': first['quality'],
        'maximumCropVariationWorkingPixels': variation.max(axis=(1, 2)),
        'acceptedEyeMask': accepted, 'anchorOffsetsWorkingPixels': offsets,
        'maximumAllowedVariationWorkingPixels': 3.}
    writeJson(output/'observations.json', observation)
    np.savez_compressed(output/'observations.npz', eyelidCurvesWorking=curves[0],
        cropVariantCurvesWorking=curves, maximumVariationWorkingPixels=variation,
        rawPeakScores=first['rawPeakScores'], acceptedEyeMask=accepted)
    if not accepted.all():
        raise ValueError('Dense photographic contours rejected; no candidate exported')
    photoEyes = {name: {'lower': curves[0, index, 0], 'upper': curves[0, index, 1]}
                 for index, name in enumerate(('imageLeft', 'imageRight'))}
    originalPhotoEyes = photoEyes
    assets = AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    scene = loadLegacyEyeBindingInputs(archive, assets.arrays['nicpToFullIds'])
    binding = bindVisibleEyeContours(**scene, camera=camera, photoEyes=photoEyes,
        imageShape=photo.shape[:2], pointsPerLid=24)
    writeJson(output/'binding.json', compactBinding(binding))
    if any(binding['eyes'][name]['status'] != 'accepted' for name in photoEyes):
        raise ValueError('Visible model eye binding rejected; no candidate exported')
    # Binding stores matching arc-length target indices; do not pair these with
    # the detector raw nonuniform point indices merely because lengths agree.
    modelEyes = {name: {part: np.asarray(binding['eyes'][name][part]['sourcePixels'])
                 for part in ('upper', 'lower')} for name in photoEyes}
    photoEyes = {name: {part: np.asarray(binding['eyes'][name][part]['targetPixels'])
                 for part in ('upper', 'lower')} for name in photoEyes}
    source = _load(archive)
    geometry = loadArrays(geometryPath)
    geometry['uvConvention'] = 'topLeftPixelEdges'
    skinIds = assets.arrays['nicpToFullIds']
    if not np.allclose(scene['vertices'][skinIds], geometry['vertices'], rtol=0, atol=1e-12):
        raise ValueError('Source ZIP skin differs from recorded final geometry')
    fullFaceLookup = {tuple(face): index for index, face in enumerate(scene['faces'])}
    fullFaceIds = np.array([fullFaceLookup[tuple(face)] for face in skinIds[geometry['faces']]])
    if (not np.array_equal(geometry['uvs'], scene['uvs'])
            or not np.array_equal(geometry['uvFaces'], scene['uvFaces'][fullFaceIds])):
        raise ValueError('Source ZIP skin UV/corner identity differs from final geometry')
    original = np.asarray(Image.open(io.BytesIO(source.payloads['head_color.png']))).copy()
    size = original.shape[0]
    member = np.zeros(len(scene['vertices']), bool); member[skinIds] = True
    accessoryFaces = (scene['materialIds'] == 0) & ~member[scene['faces']].all(1)
    accessory = {'vertices': scene['vertices'], 'faces': scene['faces'][accessoryFaces],
        'uvs': scene['uvs'], 'uvFaces': scene['uvFaces'][accessoryFaces], 'uvConvention': 'topLeftPixelEdges'}
    protected = binary_dilation(_destinationUvLookup(accessory, size)[0]['validMask'], iterations=2)
    lookup = _destinationUvLookup(geometry, size)[0]
    skinDomain = lookup['validMask']
    writable = binary_dilation(skinDomain, iterations=1) & ~protected
    bake = loadArrays(appearance/'baking/clientTexture.npz')
    directDomain = binary_dilation(~bake['coverageMask'], iterations=1) & writable
    sh = loadArrays(appearance/'shImages.npz')
    visible = {'vertices': scene['vertices'], 'faces': scene['faces'][scene['materialIds'] != 3]}
    sourceValid = sh['validMasks'][referenceView].astype(bool)
    sourcePolicy = 'Original source mask; both photo eye openings also excluded'
    policy = {}
    destinationFeather = np.ones(writable.shape, np.float32)
    if method == 'relocated':
        # A separate explicitly estimated experiment, never a changed fast
        # solver threshold. Keep old source validity and require successful SH.
        sourceValid &= sh['correctionMasks'][referenceView].astype(bool)
        sourceBand = np.zeros(sourceValid.shape, bool)
        rows, columns = np.indices(sourceValid.shape)
        surface = interpolateSurface(geometry['vertices'], geometry['faces'], lookup)
        projected, depth = projectPoints(surface[skinDomain], np.asarray(camera['intrinsics']),
            np.asarray(camera['rotation']), np.asarray(camera['translation']))
        destinationBand = np.zeros(len(projected), bool)
        featherValues = np.zeros(len(projected), np.float64)
        boxes = []
        for eye in photoEyes.values():
            points = np.concatenate([eye['upper'], eye['lower']])
            minimum, maximum = points.min(0)-[14., 14.], points.max(0)+[14., 14.]
            boxes.append(np.r_[minimum, maximum].tolist())
            sourceBand |= ((columns >= minimum[0]) & (columns <= maximum[0])
                           & (rows >= minimum[1]) & (rows <= maximum[1]))
            destinationBand |= ((projected >= minimum).all(1) & (projected <= maximum).all(1)
                                & (depth > 0))
            distanceToBox = np.minimum(projected-minimum, maximum-projected).min(1)
            fraction = np.clip(distanceToBox/4., 0, 1)
            featherValues = np.maximum(featherValues, fraction*fraction*(3-2*fraction))
        localWritable = np.zeros(writable.shape, bool)
        localWritable[skinDomain] = destinationBand
        writable &= localWritable
        destinationFeather[:] = 0
        destinationFeather[skinDomain] = featherValues
        sourceValid &= sourceBand
        # The module uses arc-resampled correspondence curves. Preserve the
        # original detector polygon as an additional independent exclusion so
        # interpolation chords cannot expose a sliver of photographic eye.
        sourceRows, sourceColumns = np.nonzero(sourceValid)
        sourceCoordinates = np.column_stack([sourceColumns, sourceRows])
        safeOriginal = np.ones(len(sourceCoordinates), bool)
        for eye in originalPhotoEyes.values():
            polygon = np.concatenate([eye['upper'], eye['lower'][::-1]])
            safeOriginal &= _outsidePolygon(sourceCoordinates, polygon, .75)
        sourceValid[sourceRows, sourceColumns] &= safeOriginal
        sourcePolicy = 'Original source-valid mask AND successful SH correction AND finite photographic eye band; no raw fallback or expanded OD support'
        policy = {'methodBranch': method, 'coreWorkingPixels': 12., 'outerWorkingPixels': 24.,
            'sourceSafetyWorkingPixels': .75, 'maximumExtensionWorkingPixels': 4.,
            'sourceAndDestinationBoxesWorking': boxes, 'destinationBoxFeatherWorkingPixels': 4., 'expandedOldMaskSourcePixels': 0,
            'browScope': 'Photo lid bounds plus 14 working pixels; confirmed below brow in overlay'}
        result = reprojectEyelidTexture(geometry, camera, sh['linearImages'][referenceView], sourceValid,
            decodeColor(original[..., :3]/255., 'srgb'), photoEyes, modelEyes, writableMask=writable,
            visibleScene=visible, corePixels=12., outerPixels=24., sourceSafetyPixels=.75,
            maxExtensionPixels=4., filterFootprint=False)
    else:
        result = reprojectFastFrontEyeTexture(geometry, camera, sh['linearImages'][referenceView], sourceValid,
            decodeColor(original[..., :3]/255., 'srgb'), photoEyes, modelEyes, writableMask=writable,
            visibleScene=visible, directProjectionFallback=True, directProjectionDomainMask=directDomain)
    if method == 'relocated':
        # A binary region-of-interest gate must not introduce a new visible
        # texture seam. Fade its contribution to zero before the hard edge;
        # this only reduces accepted sample weight, never introduces a query.
        baseline = decodeColor(original[..., :3]/255., 'srgb')
        result['linearRGB'] = (baseline+(result['linearRGB']-baseline)*destinationFeather[..., None]).astype(baseline.dtype)
        result['alpha'] *= destinationFeather
        kept = result['alpha'] > 0
        for key in ('observedMask', 'estimatedMask', 'warpMask', 'extensionMask'):
            result[key] &= kept
        result['diagnostics']['destinationBoxFeatherWorkingPixels'] = 4.
        result['diagnostics']['changedTexelsAfterBoxFeather'] = int(np.any(result['linearRGB'] != baseline, axis=2).sum())
    # Independently inspect every bilinear contributor, not only module flags.
    queryMask = (result['alpha'] > 0) | result['filterPadding']
    query = result['sourceXY'][queryMask]
    floors = np.floor(query).astype(int)
    support = np.stack([floors, floors+[0, 1], floors+[1, 0], floors+[1, 1]], axis=1)
    inBounds = ((support >= 0) & (support < np.array(sourceValid.shape[::-1]))).all(-1)
    supportValid = np.zeros(inBounds.shape, bool)
    supportValid[inBounds] = sourceValid[support[inBounds, 1], support[inBounds, 0]]
    outside = np.ones(inBounds.shape, bool)
    for eye in list(photoEyes.values())+list(originalPhotoEyes.values()):
        polygon = np.concatenate([eye['upper'], eye['lower'][::-1]])
        outside &= _outsidePolygon(support.reshape(-1, 2), polygon,
            result['diagnostics']['sourceSafetyWorkingPixels']).reshape(inBounds.shape)
    checks = {'sourceQueriesIncludingFilterPadding': int(queryMask.sum()),
        'allFourSourcePixelsInBounds': bool(inBounds.all()),
        'allFourSourcePixelsValid': bool(supportValid.all()),
        'allFourSourcePixelsOutsideBothPhotoOpeningsWithMargin': bool(outside.all()),
        'originalDenseAndResampledContoursBothChecked': True,
        'invalidSourceSupportQueryCount': int((~(inBounds & supportValid & outside).all(1)).sum())}
    if checks['invalidSourceSupportQueryCount']:
        raise RuntimeError('Independent source footprint validation rejected a query')
    if method == 'relocated':
        delta = result['sourceXY'][queryMask]-result['projectedXY'][queryMask]
        policy['maximumTotalSamplingDisplacementWorkingPixels'] = float(np.linalg.norm(delta, axis=1).max(initial=0))
        policy['allModifiedCentersEstimatedAppearance'] = bool(result['estimatedMask'][queryMask].all())
    modified = (result['alpha'] > 0) | result['filterPadding']
    pixels = original.copy()
    pixels[modified, :3] = np.rint(encodeSrgb(result['linearRGB'][modified])*255).astype(np.uint8)
    if not np.array_equal(pixels[~modified], original[~modified]) or (modified & ~writable).any():
        raise RuntimeError('Eyelid edit escaped permitted UV')
    if not np.array_equal(pixels[protected], original[protected]):
        raise RuntimeError('Accessory UV changed')
    if original.shape[-1] == 4 and not np.array_equal(pixels[..., 3], original[..., 3]):
        raise RuntimeError('Head alpha changed')
    description = {'deployed': False, 'method': 'safeBoundedFrontalEyelidResampling',
        'sourceArchiveSha256': sha(archive), 'sourceTaskId': case.name,
        'referenceViewIndex': referenceView, 'geometryChanged': False,
        'denseObservationFrame': 'savedWorkingRgbPng', 'sourceShIdentity': sha(appearance/'shImages.npz'),
        'sourceValidMask': sourcePolicy, 'policy': policy, 'strictSourceSupportChecks': checks,
        'textureDiagnostics': result['diagnostics']}
    exported = archiveTexture(archive, output/'asset', pixels, description)
    Image.fromarray(original).save(output/'beforeTexture.png')
    Image.fromarray(pixels).save(output/'afterTexture.png')
    Image.fromarray(np.rint(result['alpha']*255).astype(np.uint8)).save(output/'editAlpha.png')
    np.savez_compressed(output/'textureProvenance.npz',
        **{key: value for key, value in result.items() if isinstance(value, np.ndarray)},
        writableMask=writable, protectedAccessoryUvs=protected, directProjectionDomainMask=directDomain, sourceValidMask=sourceValid, destinationFeather=destinationFeather)
    report = {**description, 'status': 'completeLocalTextureCandidate', 'seconds': perf_counter()-started,
        'observation': observation, 'archive': exported, 'archivePath': str((output/'asset/legacyAsset.zip').resolve()),
        'changedRgbTexels': int(np.any(pixels[..., :3] != original[..., :3], axis=-1).sum()),
        'outsideEditPixelsExactlyIdentical': True, 'headAlphaExactlyIdentical': True,
        'sourceHashes': {str(path): sha(path) for path in [archive, appearance/'shImages.npz',
            appearance/'baking/clientTexture.npz', geometryPath, photoPath, root/'cameras.json']},
        'limitations': ['Working photo inference is lower resolution than original source photos.',
            'Contour warps, bounded exterior extension and filter padding are estimated appearance.',
            'No eye aperture, lid curvature, eyeball placement or gaze has been corrected.',
            'Large curve residuals can retain baseline color when no safe source support exists.',
            'No production service change and no full GPU reconstruction.']}
    writeJson(output/'report.json', report)
    print(json.dumps({'archive': report['archivePath'], 'changedRgbTexels': report['changedRgbTexels'],
                      'seconds': report['seconds'], 'diagnostics': result['diagnostics']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('case', 'appearance', 'geometry', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--method', choices=('bounded', 'relocated'), default='bounded')
    parser.add_argument('--reference-view', dest='referenceView', type=int, default=3)
    arguments = parser.parse_args()
    run(arguments.case, arguments.appearance, arguments.geometry, arguments.output, arguments.referenceView, arguments.method)
