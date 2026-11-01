"""Real warm photo-to-ZIP benchmark including fresh dense-eye postprocessing.

Reuses the earlier benchmark's startup and real request orchestration. Only the
local direct-client assembly callback is replaced; no service default changes.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeImage, writeJson
from faceAppearance.colmapTexture import _destinationUvLookup as uncachedUvLookup
from faceAppearance.headAssembly import decodeColor

DETECTOR = None
UV_MASKS = {}
EYE_REQUESTS = []


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def project(points, matrix):
    original = np.asarray(points)
    flat = original.reshape(-1, 2)
    mapped = np.c_[flat, np.ones(len(flat))]@np.asarray(matrix).T
    return (mapped[:, :2]/mapped[:, 2:]).reshape(original.shape)


def observe_front(photo, record, camera, output):
    """Five current-image STFR predictions with the existing frozen gates."""
    tick = perf_counter()
    payload = Path(photo).read_bytes()
    if hashlib.sha256(payload).hexdigest() != record['sha256']:
        raise ValueError('Fresh front photograph identity mismatch')
    with Image.open(io.BytesIO(payload)) as opened:
        image = ImageOps.exif_transpose(opened).convert('RGB')
    if list(image.size) != record['orientedSize']:
        raise ValueError('Fresh front oriented dimensions mismatch')
    anchors = np.asarray(record['faceDetection']['landmarks'])
    offsets = np.array([[0., 0.], [2., 0.], [-2., 0.], [0., 2.], [0., -2.]])
    predictions = [DETECTOR.infer(np.asarray(image), faceAnchors=anchors+offset) for offset in offsets]
    first = predictions[0]
    variants = np.stack([project(item['eyelidCurves'], record['inputToWorking']) for item in predictions])
    curves = variants[0]
    variation = np.linalg.norm(variants-variants[:1], axis=-1).max(0)
    point_valid = first['pointValidMask'] & np.isfinite(curves).all(-1)
    point_valid &= ((curves >= 0) & (curves < record['workingSize'])).all(-1) & (variation <= 3.)
    eye_valid = point_valid.all(axis=(1, 2)) & np.asarray([item['valid'] for item in first['quality']])
    eye_valid &= bool(camera.get('valid', False))
    np.savez_compressed(output/'freshFrontEyeObservations.npz', eyelidCurvesWorking=curves,
        candidatePointValidMask=point_valid, candidateEyeValidMask=eye_valid,
        cropVariantEyelidCurvesWorking=variants, maximumCropVariationWorkingPixels=variation,
        rawPeakScores=first['rawPeakScores'])
    diagnostics = {'status': 'accepted' if eye_valid.all() else 'unreliablePhotoEyes',
        'freshInference': True, 'inferenceCount': len(predictions), 'referenceViewIndex': 0,
        'photoSha256': record['sha256'], 'model': DETECTOR.modelMetadata,
        'candidateEyeValidMask': eye_valid, 'maximumCropVariationWorkingPixels': variation.max(axis=(1, 2)),
        'quality': first['quality'], 'seconds': perf_counter()-tick,
        'gates': {'minimumRawPeak': .2, 'maximumCropVariationWorkingPixels': 3., 'anchorOffsetOriginalPixels': 2.}}
    writeJson(output/'freshFrontEyeObservations.json', diagnostics)
    eyes = {name: {'lower': curves[index, 0], 'upper': curves[index, 1]}
            for index, name in enumerate(('imageLeft', 'imageRight'))}
    return eyes, diagnostics


def uv_masks(assets, geometry, size):
    key = (assets.manifestSha256, size)
    if key not in UV_MASKS:
        arrays = assets.arrays
        skin = np.zeros(len(arrays['vertices']), bool)
        skin[arrays['nicpToFullIds']] = True
        others = (arrays['materialIds'] == 0) & ~skin[arrays['faces']].all(1)
        other = {'vertices': arrays['vertices'], 'faces': arrays['faces'][others],
            'uvs': arrays['uvs'], 'uvFaces': arrays['uvFaces'][others], 'uvConvention': 'topLeftPixelEdges'}
        protected = binary_dilation(uncachedUvLookup(other, size)[0]['validMask'], iterations=2)
        domain = uncachedUvLookup(geometry, size)[0]['validMask']
        writable = binary_dilation(domain, iterations=1)&~protected
        UV_MASKS[key] = (protected, writable)
    return UV_MASKS[key]


def assemble_with_eyes(assets, geometry, texture, reconstructionDirectory, photoPaths, config):
    """Same request: stable direct/jaw, bounded full-eye fit, one bake, eye RGB."""
    from faceAppearance.clientGeometryRefinement import refineClientGeometry
    from faceAppearance.clientRenderedJaw import refineRenderedClientJaw
    from faceAppearance.renderedJawRuntime import buildScene
    from faceAppearance.clientTextureBaking import bakeClientTexture
    from faceAppearance.clientLandmarks import ibugClientLandmarks, loadClientLandmarks
    from faceAppearance.fastFrontEyeGeometry import refineFastFrontEyes
    from faceAppearance.fastFrontEyeTexture import reprojectFastFrontEyeTexture
    start = perf_counter()
    root = Path(reconstructionDirectory)
    output = root/'directClient'
    output.mkdir(exist_ok=False)
    cameras = json.loads((root/'cameras.json').read_text(encoding='utf-8'))
    inputs = json.loads((root/'inputs.json').read_text(encoding='utf-8'))
    delight = load(root/'delight.npz')
    stage = {}
    tick = perf_counter()
    refined = refineClientGeometry(assets, geometry, cameras, inputs,
        photoPaths=photoPaths, delight=delight, projectRoot=config.projectRoot,
        outputDirectory=output/'refinement', referenceViewIndex=0)
    stage['directRegistrationAndRefinement'] = perf_counter()-tick
    reference = np.asarray(refined['vertices']).copy()
    transform = refined['templateToUff']
    tick = perf_counter()
    jaw = refineRenderedClientJaw(assets, reference, refined['faces'], transform,
        cameras, inputs, photoPaths, projectRoot=config.projectRoot, outputDirectory=output/'renderedJaw')
    stage['renderedFrontalJaw'] = perf_counter()-tick
    arrays = assets.arrays
    skin = arrays['nicpToFullIds']
    if not np.array_equal(refined['faces'], arrays['nicpFaces']):
        raise ValueError('Refined skin topology differs from client template')
    face_lookup = {tuple(face): index for index, face in enumerate(arrays['faces'])}
    full_ids = np.asarray([face_lookup[tuple(face)] for face in skin[arrays['nicpFaces']]])
    if not np.all(arrays['materialIds'][full_ids] == 0):
        raise ValueError('Client skin must use head material')
    final_geometry = {'vertices': jaw['vertices'].copy(), 'faces': arrays['nicpFaces'],
        'uvs': arrays['uvs'], 'uvFaces': arrays['uvFaces'][full_ids], 'uvConvention': 'topLeftPixelEdges'}
    camera = cameras[0]
    if not camera.get('valid') or camera.get('viewIndex', 0) != 0:
        raise ValueError('Fast eyes require a valid current frontal camera zero')
    photo_eyes, observation = observe_front(photoPaths[0], inputs[0], camera, output)
    stage['frontDenseEyeObservation'] = observation['seconds']
    tick = perf_counter()
    # Accessories are placed once from exactly the same pre-jaw reference that
    # final assembleFinalClientSurface receives below. Only skin becomes jaw-after.
    scene = buildScene(assets, reference, transform)
    scene['vertices'][skin] = jaw['vertices']
    before_full = scene['vertices'].copy()
    corner_faces = np.concatenate([c['faceIndices'] for c in assets.components
                                   if c['label'] == 'headAccessory']).astype(np.int64)
    corner_ids = np.unique(scene['faces'][corner_faces])
    mouth = ibugClientLandmarks(loadClientLandmarks(arrays['nicpFaces']), groups=('mouth',))
    fixed_skin = np.unique(np.r_[np.flatnonzero(arrays['staticMask']), mouth['vertexIndices'].ravel(),
        np.flatnonzero(np.any(reference != jaw['vertices'], axis=1))])
    alpha = {index: material['baseColor'][..., 3] for index, material in enumerate(scene['materials'])
             if material['alphaMode'] == 'MASK'}
    if np.asarray(observation['candidateEyeValidMask']).all():
        fit = refineFastFrontEyes(scene['vertices'], scene['faces'], scene['materialIds'], camera, photo_eyes,
            skinVertexIds=skin, cornerFaceIds=corner_faces, fixedVertexIds=skin[fixed_skin],
            imageShape=tuple(reversed(inputs[0]['workingSize'])), uvs=scene['uvs'], uvFaces=scene['uvFaces'],
            materials=scene['materials'], alphaTextures=alpha,
            originalPixelScale=float(np.asarray(inputs[0]['inputToWorking'])[0, 0]))
        fit['diagnostics']['photoTargets'] = 'Fresh current-photo STFR five crop variants; automatic proxy, not manual truth'
        final_full = fit['vertices']
    else:
        fit = {'diagnostics': {'status': 'skippedUnreliableFreshPhotoEyes'}, 'textureContours': None}
        final_full = before_full.copy()
    final_geometry['vertices'] = final_full[skin].copy()
    stage['frontEyeGeometryAndScene'] = perf_counter()-tick
    if not np.array_equal(final_geometry['vertices'][fixed_skin], jaw['vertices'][fixed_skin]):
        raise RuntimeError('Eyes changed fixed mouth/jaw/static skin')
    np.savez_compressed(output/'fullEyeGeometry.npz', before=before_full, after=final_full,
        faces=scene['faces'], skinVertexIds=skin, cornerVertexIds=corner_ids, fixedVertexIds=skin[fixed_skin])
    np.savez_compressed(output/'geometry.npz', **{key: value for key, value in final_geometry.items() if isinstance(value, np.ndarray)})
    mouth_state = refined.get('mouthState')
    if mouth_state is not None:
        mouth_state['afterCanonical'] = final_geometry['vertices'].copy()
        np.savez_compressed(output/'mouthState.npz', **mouth_state)
    encoding = 'linear' if 'linearAlbedo' in texture else config.assemblyColorEncoding
    _, _, prior_texture = assets._transferTexture(texture, encoding)
    gain = np.asarray(prior_texture['templateLinearColorGain'])
    tick = perf_counter()
    protected, writable = uv_masks(assets, final_geometry, assets.materials[0]['baseColor'].shape[0])
    stage['frontEyeUvProtection'] = perf_counter()-tick
    tick = perf_counter()
    baked = bakeClientTexture(final_geometry, cameras, delight, assets.materials[0]['baseColor'][..., :3],
        output/'baking', colmapExecutable=ROOT/'.runtimes/colmap-eyelid/bin/colmap-eyelid',
        mouthState=mouth_state, referenceView=0, fallbackGain=gain, skinMask=~protected, freezeViewLabels=True)
    stage['clientColmapAndMouthTexture'] = perf_counter()-tick
    protected_fallback = np.clip(assets.materials[0]['baseColor'][..., :3]*gain, 0, 1).astype(np.float32)
    if not np.array_equal(baked['linearRGB'][protected], protected_fallback[protected]):
        raise RuntimeError('Baking changed protected non-NICP template color')
    tick = perf_counter()
    binding = fit['textureContours']
    eye_texture = {'status': 'skippedInvalidTextureProxy'}
    if binding is not None and all(binding['eyes'][name]['status'] == 'accepted' for name in photo_eyes):
        model_eyes = {name: {part: np.asarray(binding['eyes'][name][part]['sourcePixels'])
                            for part in ('lower', 'upper')} for name in photo_eyes}
        if not all(np.isfinite(curve).all() for parts in model_eyes.values() for curve in parts.values()):
            raise ValueError('Accepted texture proxy has nonfinite coordinates')
        before_rgb = baked['linearRGB'].copy()
        changed = reprojectFastFrontEyeTexture(final_geometry, camera,
            decodeColor(delight['albedoImages'][0], 'gamma2.2'), delight['validMasks'][0],
            before_rgb, photo_eyes, model_eyes, writableMask=writable,
            visibleScene={'vertices': final_full, 'faces': scene['faces'][scene['materialIds'] != 3]})
        modified = (changed['alpha'] > 0)|changed['filterPadding']
        if np.any(modified&~writable) or not np.array_equal(changed['linearRGB'][~modified], before_rgb[~modified]):
            raise RuntimeError('Eye texture escaped local editable support')
        if not np.array_equal(changed['linearRGB'][protected], before_rgb[protected]):
            raise RuntimeError('Eye texture changed accessory UV')
        baked['linearRGB'] = changed['linearRGB']
        baked['incomingBakedCoverageMask'] = baked['coverageMask'].copy()
        baked['incomingBakedSourceIndex'] = baked['sourceIndex'].copy()
        baked['incomingBakedColmapCoverageMask'] = baked['colmapCoverageMask'].copy()
        baked['incomingBakedFaceSourceIndex'] = baked.pop('faceSourceIndex')
        baked['coverageMask'] = baked['coverageMask'].copy()
        # Current RGB with warped/extended/filter contributions is estimated,
        # even where the old bake had a valid sample. Never add new coverage.
        baked['coverageMask'][changed['estimatedMask']] = False
        if 'sourceCoverage' in baked:
            baked['sourceCoverage'] = baked['coverageMask'].copy()
        baked['colmapCoverageMask'] = baked['colmapCoverageMask'].copy()
        baked['colmapCoverageMask'][modified] = False
        baked['sourceIndex'] = baked['sourceIndex'].copy()
        baked['sourceIndex'][modified] = -2
        baked['sourceIndex'][modified & (changed['alpha'] == 1) & changed['observedMask']] = 0
        baked['sourceIndex'][~baked['coverageMask']] = -1
        for key in ('confidence', 'sourceConfidence'):
            if key in baked:
                baked[key] = baked[key].copy()
                baked[key][modified] = 0
        baked['fastEyeAppearanceAlpha'] = changed['alpha']
        baked['fastEyeFilterPadding'] = changed['filterPadding']
        for name in ('estimatedMask', 'observedMask', 'warpMask', 'extensionMask'):
            baked['fastEye'+name[0].upper()+name[1:]] = changed[name]
        np.savez_compressed(output/'eyeTextureProvenance.npz', **{
            key: value for key, value in changed.items() if isinstance(value, np.ndarray)}, writableMask=writable)
        eye_texture = {**changed['diagnostics'], 'protectedAccessoryUvExact': True,
            'freshPhotoSemanticTargets': True, 'coverageMask': 'Incoming bake coverage minus estimated eye pixels; never increases',
            'sourceIndexMeaning': '-1 unobserved/estimated final appearance; -2 mixture of observed sources; 0 fully frontal directly sampled color; remaining nonnegative values original selected view',
            'incomingBakedMaps': 'Historical bake evidence only, not final RGB provenance',
            'finalTextureArtifact': 'directClient/texture.npz; baking/clientTexture.npz is the pre-eye bake',
            'textureContourScope': 'Measured eye-material proxy; complete aperture quality remains separately reported'}
    stage['frontEyeTexture'] = perf_counter()-tick
    all_maps = {key: value for key, value in baked.items() if isinstance(value, np.ndarray) and key != 'linearRGB'}
    # Historical integer index arrays are evidence in this NPZ, not scalar
    # blend masks accepted by the client's final assembly contract.
    np.savez_compressed(output/'texture.npz', linearAlbedo=baked['linearRGB'], **all_maps)
    historical_indices = {'incomingBakedSourceIndex', 'incomingBakedFaceSourceIndex'}
    provenance = {key: value for key, value in all_maps.items() if key not in historical_indices}
    rgb = np.clip(baked['linearRGB'], 0, 1)
    writeImage(output/'headColor.png', np.where(rgb <= .0031308, rgb*12.92, 1.055*rgb**(1/2.4)-.055))
    diagnostics = {'pipeline': 'localWarmFastFrontFullEyesExperiment', 'stageSeconds': stage,
        'geometry': refined['diagnostics'], 'renderedJaw': jaw['diagnostics'], 'baking': baked['diagnostics'],
        'freshEyeObservation': observation, 'frontEyeGeometry': fit['diagnostics'], 'frontEyeTexture': eye_texture,
        'bakeCallCount': 1, 'actualBakeRuntime': baked['diagnostics']['colmapExecutable'],
        'baseRuntimeScope': 'Base preflight only; actualBakeRuntime identifies the executable used by this opt-in eye experiment',
        'deployed': False}
    tick = perf_counter()
    assembled = assets.assembleFinalClientSurface(final_geometry, transform, baked['linearRGB'], provenance,
        diagnostics=diagnostics, accessoryReferenceCanonical=reference)
    rotation, translation, scale = np.asarray(transform['rotation']), np.asarray(transform['translation']), float(transform['scale'])
    actual = assembled['geometry']['vertices']@rotation*scale+translation
    others = np.ones(len(final_full), bool); others[np.r_[skin, corner_ids]] = False
    if not np.allclose(actual[others], before_full[others], rtol=0, atol=1e-12):
        raise RuntimeError('Final assembly moved a frozen non-corner accessory')
    if not np.allclose(actual[corner_ids], before_full[corner_ids], rtol=0, atol=1e-12):
        raise RuntimeError('Corner placement reference differs before explicit override')
    changed_corners = corner_ids[np.any(final_full[corner_ids] != before_full[corner_ids], axis=1)]
    assembled['geometry']['vertices'][changed_corners] = ((final_full[changed_corners]-translation)@rotation.T)/scale
    corner_metadata = []
    for component in assembled['components']:
        ids = np.asarray(component['vertexIndices'], np.int64)
        if component['label'] != 'headAccessory' or not np.intersect1d(ids, changed_corners).size:
            continue
        component['initialOnlyTransformRow4x4'] = component.pop('transformRow4x4')
        component['transformAppliesToComponent'] = False
        component['placement'] = 'preJawRigidPlacementThenExplicitFrontEyeNonrigidCorrection'
        component['pivotAfterCm'] = assembled['geometry']['vertices'][ids].mean(0)
        component['frontEyeNonrigidCorrection'] = True
        corner_metadata.append({'name': component.get('name'), 'vertexIndices': ids,
            'initialOnlyTransformRow4x4': component['initialOnlyTransformRow4x4'],
            'finalPivotCm': component['pivotAfterCm'], 'transformAppliesToComponent': False})
    final_canonical = assembled['geometry']['vertices']@rotation*scale+translation
    if not np.allclose(final_canonical, final_full, rtol=0, atol=1e-12):
        raise RuntimeError('Final assembly differs from eye-validated full surface')
    assembly_check = {'frozenNonSkinNonCornerAccessoriesExactWithin1e12': True,
        'finalFullCanonicalMaximumDifference': float(np.max(np.abs(final_canonical-final_full))),
        'explicitChangedCornerVertexCount': len(changed_corners), 'cornerComponents': corner_metadata,
        'fixedMouthJawStaticExactlyUnchanged': True}
    assembled['diagnostics']['frontEyeExplicitAssembly'] = assembly_check
    assembled['diagnostics']['geometryTransfer']['accessoryPlacement'] += '; explicit audited inner-corner vertices may receive the separately recorded nonrigid frontal eye correction'
    stage['finalClientAssembly'] = perf_counter()-tick
    diagnostics.update(seconds=perf_counter()-start, fullAssemblyCheck=assembly_check)
    writeJson(output/'report.json', diagnostics)
    EYE_REQUESTS.append({'source': str(root), 'stageSeconds': stage,
        'eyeObservation': observation, 'eyeGeometry': fit['diagnostics'], 'eyeTexture': eye_texture,
        'assembly': assembly_check, 'bakeCallCount': 1,
        'actualBakeRuntime': baked['diagnostics']['colmapExecutable']})
    return assembled, diagnostics


def main():
    global DETECTOR
    import profileFastGeneration20261012 as base
    from faceAppearance import directClientService
    from faceAppearance.adapters.eyelidLandmarks import EyelidLandmarksAdapter
    cv2.setNumThreads(1)
    tick = perf_counter()
    DETECTOR = EyelidLandmarksAdapter(ROOT).loadModels()
    model_load_seconds = perf_counter()-tick
    original_write = base.write
    def write_with_eye_scope(path, data):
        if path.name == 'profile.json':
            data['includesFreshFrontEyeModules'] = True
            data['scope'] = 'Four current photos through fresh UFF/delight/fusion/direct/jaw, fresh dense eye detection, geometry, one bake, eye texture, final ZIP'
            data['startup']['additionalDenseEyeModelLoadSeconds'] = model_load_seconds
            data['eyeRequests'] = EYE_REQUESTS
            if 'limitations' in data:
                data['limitations'] = ['One male sample and local RTX3090, not server or HTTP SLA.',
                    'Includes fresh five-variant dense front detection, full eye geometry and eye texture.',
                    'Automatic contour proxies are not manual ground truth; semantic failure retains safe incoming geometry/bake.']
        # Existing orchestration writes plain values; rich eye diagnostics use
        # the project's explicit NumPy-aware UTF-8 serializer.
        if path.name == 'profile.json':
            writeJson(path, data)
        else:
            original_write(path, data)
    with patch.object(directClientService, 'assembleDirectClient', assemble_with_eyes), patch.object(base, 'write', write_with_eye_scope):
        base.main()


if __name__ == '__main__':
    main()
