"""Local nose-only direct reprojection on an accepted, fixed legacy asset.

No camera fit, semantic warp, geometry edit or new side-view blending. Input
hashes, complete-scene visibility and protected eye/attachment pixels are
checked before exporting a texture-only comparison candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src')]
from faceAppearance.artifacts import writeJson
from faceAppearance.camera import projectPoints
from faceAppearance.colmapTexture import _destinationUvLookup
from faceAppearance.headAssembly import AssemblyAssets, decodeColor
from faceAppearance.legacyAppearanceVariant import exportLegacyAppearanceVariant
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.noseTextureRefinement import refineNoseTexture, encodeNoseTextureEdit, evaluateNoseViewQuality
from faceAppearance.texture import interpolateSurface, samplePixels
from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(caseDirectory, appearanceDirectory, baselineArchive, geometryPath, maskPath,
        maskReportPath, outputDirectory, *, referenceViewIndex=3,
        templateAssets=ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1'):
    started = perf_counter()
    case, appearance, baselinePath, geometryPath, maskPath, maskReportPath, output = map(Path,
        (caseDirectory, appearanceDirectory, baselineArchive, geometryPath, maskPath, maskReportPath, outputDirectory))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Use a new nose texture experiment output')
    parent = json.loads((appearance/'report.json').read_text())
    audit = json.loads(maskReportPath.read_text())
    reconstruction = case/'reconstruction'
    if parent['status'] != 'complete' or parent['sourceTaskId'] != case.name:
        raise ValueError('SH source belongs to another or incomplete reconstruction')
    for name in ('shImages.npz', 'baking/clientTexture.npz'):
        if digest(appearance/name) != parent['artifactIdentities'][name]['sha256']:
            raise ValueError('SH appearance artifact differs from its recorded hash')
    for name in ('cameras.json', f'input{referenceViewIndex:02d}.png'):
        if digest(reconstruction/name) != parent['sourceHashes'][name]:
            raise ValueError('Camera/photo differs from saved SH source')
    source = _load(baselinePath)
    localGeometry = source.manifest['assemblyDiagnostics'].get('localGeometryVariant', {}).get('diagnostics', {})
    if localGeometry.get('diagnosticOnly') or localGeometry.get('liveAcceptancePassed') is False:
        raise ValueError('Rejected/diagnostic geometry cannot become this accepted baseline')
    geometry = arrays(geometryPath)
    geometry['uvConvention'] = 'topLeftPixelEdges'
    assets = AssemblyAssets(templateAssets)
    scene = loadLegacyEyeBindingInputs(baselinePath, assets.arrays['nicpToFullIds'])
    skinIds = assets.arrays['nicpToFullIds']
    if not np.allclose(scene['vertices'][skinIds], geometry['vertices'], rtol=0, atol=1e-12):
        raise ValueError('Accepted full ZIP and supplied skin geometry differ')
    faceLookup = {tuple(face): index for index, face in enumerate(scene['faces'])}
    fullFaceIds = np.array([faceLookup[tuple(face)] for face in skinIds[geometry['faces']]])
    if (not np.array_equal(geometry['uvs'], scene['uvs'])
            or not np.array_equal(geometry['uvFaces'], scene['uvFaces'][fullFaceIds])):
        raise ValueError('Skin topology/UV order differs from the complete ZIP')
    originalGeometryPath = Path(parent['geometryReplay']['directory'])/'finalGeometry.npz'
    if digest(originalGeometryPath) != parent['geometryReplay']['sourceHashes'][str(originalGeometryPath)]:
        raise ValueError('Original SH-normal geometry identity changed')
    originalGeometry = arrays(originalGeometryPath)
    sameNormalGeometry = np.array_equal(geometry['vertices'], originalGeometry['vertices'])
    if not sameNormalGeometry:
        raise ValueError('This fixed-geometry experiment cannot accept a new geometry')
    if audit['baselineSha256'] != digest(baselinePath):
        raise ValueError('Baseline differs from the audited eye-protected asset')
    for name in ('sh', 'eyeProvenance', 'photoLandmarks'):
        path = Path(audit['sourcePaths'][name])
        if digest(path) != audit['sourceHashes'][str(path)]:
            raise ValueError('Audited mask support source changed')
    masks = arrays(maskPath)
    eye = arrays(audit['sourcePaths']['eyeProvenance'])
    with Image.open(io.BytesIO(source.payloads['head_color.png'])) as image:
        original = np.asarray(image).copy()
    shape = original.shape[:2]
    if shape[0] != shape[1]:
        raise ValueError('This legacy atlas experiment requires a square head texture')
    eyeGuard = binary_dilation((eye['alpha'] > 0) | eye['filterPadding'], iterations=2)
    if not np.array_equal(eyeGuard, masks['protectedEyeUvMask']):
        raise ValueError('Eye protection differs from actual prior edited UV provenance')
    if not np.array_equal(eye['protectedAccessoryUvs'], masks['protectedAccessoryUvMask']):
        raise ValueError('Accessory protection differs from actual prior edited UV provenance')
    protected = eyeGuard | masks['protectedAccessoryUvMask']
    if protected.shape != shape or masks['noseSemanticUvMask'].shape != shape:
        raise ValueError('Nose/eye masks do not match the original atlas')
    cameras = json.loads((reconstruction/'cameras.json').read_text())
    camera = cameras[referenceViewIndex]
    if camera.get('viewIndex') != referenceViewIndex or not camera.get('valid'):
        raise ValueError('Requested front camera is invalid')
    observation = arrays(audit['sourcePaths']['photoLandmarks'])
    photoLandmarks = observation['faceLandmarksWorking'][referenceViewIndex]
    if not observation['landmarkCandidateValidMask'][referenceViewIndex, 27:36].all():
        raise ValueError('Reference nose landmarks are not reliable enough for the local ROI')
    sh = arrays(appearance/'shImages.npz')
    sourceLinear = sh['linearImages'][referenceViewIndex]
    sourceValid = sh['validMasks'][referenceViewIndex].copy()
    correction = sh['correctionMasks'][referenceViewIndex].copy()
    correctionPolicy = {'method': 'reuseCorrectedImagesOnIdenticalGeometry', 'geometryMatchesSavedSHNormals': sameNormalGeometry}
    photoMask = np.zeros(sourceValid.shape, np.uint8)
    cv2.fillConvexPoly(photoMask, cv2.convexHull(np.rint(photoLandmarks[27:36]).astype(np.int32)), 1)
    photoMask = binary_dilation(photoMask, iterations=10)
    # This extra source exclusion only restricts old support; it never relaxes
    # original masks to borrow an eye/mouth pixel near a misregistered feature.
    excluded = np.zeros(sourceValid.shape, np.uint8)
    for ids in (range(36, 42), range(42, 48), range(48, 60)):
        cv2.fillPoly(excluded, [np.rint(photoLandmarks[list(ids)]).astype(np.int32)], 1)
    sourceValid &= ~binary_dilation(excluded.astype(bool), iterations=1)
    lookup, _ = _destinationUvLookup(geometry, shape[0])
    surface = interpolateSurface(geometry['vertices'], geometry['faces'], lookup)
    uvValid = lookup['validMask']
    xy, depth = projectPoints(surface[uvValid], np.asarray(camera['intrinsics']),
        np.asarray(camera['rotation']), np.asarray(camera['translation']))
    projectedBand = np.zeros(shape, bool)
    projectedBand[uvValid] = (samplePixels(photoMask.astype(float), xy) > .999) & (depth > 0)
    domain = masks['noseSemanticUvMask'] & projectedBand & uvValid & ~protected
    visibleScene = {'vertices': scene['vertices'], 'faces': scene['faces'][scene['materialIds'] != 3]}
    quality = evaluateNoseViewQuality(geometry, cameras, sh['validMasks'] & sh['correctionMasks'], domain,
        referenceViewIndex=referenceViewIndex, visibleScene=visibleScene,
        minimumFacingCosine=.5, fullFacingCosine=.65, minimumRelativeFacing=.85, fullRelativeFacing=.95)
    result = refineNoseTexture(geometry, camera, sourceLinear, sourceValid, correction,
        decodeColor(original[..., :3]/255., 'srgb'), domain, protected,
        visibleScene=visibleScene, minimumFacingCosine=.5, sourceQuality=quality['sourceQuality'])
    pixels = encodeNoseTextureEdit(original, result['linearRGB'], result['alpha'], protected)
    modified = result['alpha'] > 0
    checks = {'outsideEditPixelsExactlyIdentical': bool(np.array_equal(pixels[~modified], original[~modified])),
        'protectedEyePixelsExactlyIdentical': bool(np.array_equal(pixels[eyeGuard], original[eyeGuard])),
        'protectedAccessoryPixelsExactlyIdentical': bool(np.array_equal(pixels[protected], original[protected])),
        'headAlphaExactlyIdentical': bool(np.array_equal(pixels[..., 3], original[..., 3])) if original.shape[-1] == 4 else True}
    if not all(checks.values()):
        raise RuntimeError('Nose texture edit changed a protected pixel')
    if not modified.any():
        raise ValueError('No safe local frontal source pixels; no empty candidate exported')
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output/'textureProvenance.npz',
        **{name: value for name, value in result.items() if isinstance(value, np.ndarray)},
        **{name: value for name, value in quality.items() if isinstance(value, np.ndarray)},
        noseUvMask=domain, protectedEyeUvMask=eyeGuard, protectedUvMask=protected,
        sourceValidMask=sourceValid, sourceCorrectionMask=correction)
    Image.fromarray(np.rint(result['alpha']*255).astype(np.uint8)).save(output/'editAlpha.png')
    description = {'method': 'noseOnlyDirectFrontReprojection', 'sourceTaskId': case.name,
        'referenceViewIndex': referenceViewIndex, 'geometryChanged': False, 'cameraChanged': False,
        'correctionPolicy': correctionPolicy, 'textureDiagnostics': result['diagnostics'],
        'viewQualityDiagnostics': quality['diagnostics'], **checks}
    exported = exportLegacyAppearanceVariant(baselinePath, decodeColor(pixels[..., :3]/255., 'srgb'),
        output/'asset', diagnostics=description)
    verified = _load(exported['archivePath'])
    with Image.open(io.BytesIO(verified.payloads['head_color.png'])) as image:
        np.testing.assert_array_equal(np.asarray(image), pixels)
    sourceNames = [baselinePath, geometryPath, maskPath, maskReportPath, appearance/'shImages.npz',
        appearance/'baking/clientTexture.npz', reconstruction/'cameras.json', Path(audit['sourcePaths']['photoLandmarks']),
        Path(audit['sourcePaths']['eyeProvenance']), reconstruction/f'input{referenceViewIndex:02d}.png']
    report = {'status': 'complete', 'sourceTaskId': case.name, 'sourceArchiveSha256': digest(baselinePath),
        'parentArchiveSha256': digest(baselinePath), 'variant': exported, **description,
        'seconds': perf_counter()-started, 'changedRgbTexels': int(np.any(pixels[..., :3] != original[..., :3], axis=-1).sum()),
        'sourceHashes': {str(path.resolve()): digest(path) for path in sourceNames},
        'textureProvenanceSha256': digest(output/'textureProvenance.npz'),
        'implementationHashes': {str(path.relative_to(ROOT)): digest(path) for path in
            (Path(__file__), ROOT/'src/faceAppearance/noseTextureRefinement.py')},
        'qualityStatus': 'pendingMultiViewVisualReview',
        'previousExperiment': {'path': str(output.parent/'textureConditioned'), 'qualityAccepted': False,
            'failure': 'The incidence-conditioned contribution left a tip seam because feathering used the requested nose domain, not final accepted support.',
            'change': 'Feather final accepted support and its internal rejection holes; identical incidence thresholds, source queries and support, without any expansion.'},
        'visibilityScope': 'All opaque assembled triangles; MASK brow/lash cards excluded; their head UV footprint remains protected',
        'limitations': ['This branch changes only nasal color sampling on the supplied baseline; it is not a nasal geometry repair.',
            'Direct reprojection support is not new true anatomical coverage or independent albedo truth.',
            'Existing side-view or template values remain wherever frontal support is unavailable.',
            'SH can retain specular and self-shadow components, especially around nostrils.',
            'The UV ROI uses approximate template semantics and cached FAN observations.']}
    writeJson(output/'report.json', report)
    print(json.dumps({'archive': exported['archivePath'], 'changedRgbTexels': report['changedRgbTexels'],
                      'seconds': report['seconds'], 'sourceSupportTexels': result['diagnostics']['directReprojectionSupportedTexels']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('case', 'appearance', 'baseline', 'geometry', 'masks', 'mask-report', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--reference-view', dest='referenceView', type=int, default=3)
    args = parser.parse_args()
    run(args.case, args.appearance, args.baseline, args.geometry, args.masks, args.mask_report,
        args.output, referenceViewIndex=args.referenceView)
