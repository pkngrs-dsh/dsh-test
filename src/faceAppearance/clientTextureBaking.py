"""Bake photographs on final client geometry while retaining client UVs.

COLMAP's selected-view support, front-mouth center observations, and texture
filter padding are separate provenance channels. The returned color remains
linear RGB until the caller writes its final client texture. No experiment
paths, template identities, camera fits, or geometric edits are used here.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from .artifacts import writeJson
from .colmapTexture import (_destinationUvLookup, loadTexturedMesh, prepareWorkspace,
                            rebakeAtlas, sourceSupportFromMarkers, viewMarkerColors)
from .surfaceVisibility import ProjectedSurfaceVisibility
from .texture import samplePixels, validateTriangles


def _smooth(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def _nativeLinear(value):
    """OpenDelight's native gamma2.2 RGB -> linear RGB, without PNG round trips."""
    return np.power(np.asarray(value, np.float64), 2.2).astype(np.float32)


def _digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _texturer(executable, workspace, destination, *, support, threads, timeout,
              selectionOptions=None):
    destination.mkdir()
    options = {'texture_scale_factor': 4, 'atlas_patch_padding': 50,
               'min_cos_normal_angle': .1, 'min_visible_vertices': 3,
               'view_selection_smoothing_iterations': 3, 'inpaint_radius': 5,
               'apply_color_correction': 0 if support else 1,
               'color_correction_regularization': .1, 'num_threads': int(threads)}
    command = [str(executable), 'mesh_texturer', '--input_path', str(workspace / 'mesh.ply'),
               '--workspace_path', str(workspace), '--output_path', str(destination)]
    for name, value in options.items():
        command.extend([f'--MeshTextureMapping.{name}', str(value)])
    for name, value in (selectionOptions or {}).items():
        if name not in ('eyelid_prior_path', 'face_view_ids_input',
                        'face_view_ids_output', 'eyelid_smoothing_weight',
                        'eyelid_smoothing_iterations'):
            raise ValueError(f'Unknown experimental view selection option: {name}')
        command.extend([f'--{name}', str(value)])
    writeJson(destination / 'command.json', {'argv': command, 'options': options})
    started = perf_counter()
    logPath = destination / 'run.log'
    try:
        with logPath.open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f'COLMAP texture pass timed out; see {logPath}') from error
    if completed.returncode:
        raise RuntimeError(f'COLMAP texture pass exited {completed.returncode}; see {logPath}')
    return {'seconds': perf_counter() - started, 'options': options,
            'log': str(logPath), 'argv': command}


def _validateInputs(geometry, cameras, delight, fallbackLinear, skinMask):
    vertices, faces = validateTriangles(geometry['vertices'], geometry['faces'], 3)
    if float(np.ptp(vertices, axis=0).max()) <= 0:
        raise ValueError('Client texture geometry must have nonzero extent')
    if 'uvFaces' not in geometry:
        raise ValueError('Client baking requires explicit original per-corner uvFaces')
    uvs, uvFaces = validateTriangles(geometry['uvs'], geometry['uvFaces'], 2)
    if faces.shape != uvFaces.shape:
        raise ValueError('UV triangles must retain client geometry face order')
    if (uvs < -1e-6).any() or (uvs > 1 + 1e-6).any():
        raise ValueError('Client UVs must lie inside the normalized texture atlas')
    if geometry.get('uvConvention', 'topLeftPixelEdges') != 'topLeftPixelEdges':
        raise ValueError('Client texture baking requires topLeftPixelEdges UV coordinates')
    prepared = {'vertices': vertices, 'faces': faces, 'uvs': uvs, 'uvFaces': uvFaces,
                'uvConvention': 'topLeftPixelEdges'}
    images = np.asarray(delight['albedoImages'])
    masks = np.asarray(delight['validMasks'])
    if (images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] < 1
            or min(images.shape[1:3]) < 2 or not np.isfinite(images).all()
            or (images < 0).any() or (images > 1).any()):
        raise ValueError('Expected finite native gamma2.2 albedoImages shaped NxHxWx3 in [0,1]')
    if (masks.shape != images.shape[:3] or masks.dtype.kind not in 'bifu'
            or not np.isfinite(masks).all() or (masks < 0).any() or (masks > 1).any()):
        raise ValueError('Expected independent per-view validMasks shaped NxHxW in [0,1]')
    fallback = np.asarray(fallbackLinear, np.float32)
    if (fallback.ndim != 3 or fallback.shape[2] != 3 or fallback.shape[0] != fallback.shape[1]
            or fallback.shape[0] < 2 or not np.isfinite(fallback).all()
            or (fallback < 0).any() or (fallback > 1).any()):
        raise ValueError('fallbackLinear must be a square finite linear RGB atlas in [0,1]')
    if skinMask is None:
        writable = np.ones(fallback.shape[:2], bool)
    else:
        writable = np.asarray(skinMask)
        if writable.shape != fallback.shape[:2] or writable.dtype != np.bool_:
            raise ValueError('skinMask must be a boolean writable atlas domain matching fallbackLinear')
        writable = writable.copy()
    cameras = list(cameras)
    if not cameras:
        raise ValueError('At least one accepted photograph camera is required')
    return prepared, cameras, images, masks, fallback.copy(), writable


def _mouthObservation(state, referenceView, imageShape, minimumConfidence):
    """Reject the whole override if any outer-lip point is unreliable."""
    reasons = []
    if state is None:
        return None, {'status': 'skipped', 'reasons': ['missingMouthObservation']}
    if state.get('referenceViewIndex', referenceView) != referenceView:
        reasons.append('mouthObservationReferenceViewMismatch')
    if not state.get('mouthObservationReliable', True):
        reasons.extend(str(value) for value in state.get('mouthObservationRejectionReasons', []))
        reasons.append('mouthObservationMarkedUnreliable')
    if 'outerLipPixels' in state:
        lip = np.asarray(state['outerLipPixels'], np.float64)
        confidence = np.asarray(state.get('outerLipConfidence', []), np.float64)
    else:
        ids = np.asarray(state.get('ibugIds', []))
        points = np.asarray(state.get('frontTargetPixels', []), np.float64)
        scores = np.asarray(state.get('frontConfidence', []), np.float64)
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or points.shape != (len(ids), 2)
                or scores.shape != (len(ids),)):
            return None, {'status': 'skipped', 'reasons': reasons + ['missingOrderedOuterLipPointsOrConfidence']}
        indices = []
        for landmark in range(48, 60):
            matches = np.flatnonzero(ids == landmark)
            if len(matches) != 1:
                return None, {'status': 'skipped', 'reasons': reasons + ['outerLipLandmarkMissingOrDuplicate']}
            indices.append(int(matches[0]))
        lip, confidence = points[indices], scores[indices]
    if lip.shape != (12, 2) or confidence.shape != (12,):
        return None, {'status': 'skipped', 'reasons': reasons + ['expectedTwelveOrderedOuterLipPointsAndScores']}
    height, width = imageShape
    if not np.isfinite(lip).all() or (lip < 0).any() or (lip >= [width - 1, height - 1]).any():
        reasons.append('outerLipOutsidePhotographBilinearDomain')
    if not np.isfinite(confidence).all() or (confidence < minimumConfidence).any():
        reasons.append('outerLipConfidenceBelowThreshold')
    if np.isfinite(lip).all():
        area = abs(float(cv2.contourArea(lip.astype(np.float32))))
        if area < 1.0:
            reasons.append('outerLipPolygonDegenerate')
    else:
        area = 0.0
    diagnostics = {'status': 'skipped' if reasons else 'accepted', 'reasons': reasons,
                   'referenceViewIndex': int(referenceView), 'minimumRequiredConfidence': minimumConfidence,
                   'minimumOuterLipConfidence': float(confidence.min()) if np.isfinite(confidence).all() else None,
                   'outerLipAreaWorkingPixels': area}
    return (None if reasons else lip), diagnostics


def _mouthContext(geometry, camera, source, mask, lip, lookup, *, corePixels, outerPixels):
    height, width = source.shape[:2]
    K, R, t = (np.asarray(camera[name], np.float64) for name in ('intrinsics', 'rotation', 'translation'))
    cameraVertices = geometry['vertices'] @ R.T + t
    triangles = cameraVertices[geometry['faces']]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
    polygon = np.zeros((height, width), np.uint8)
    cv2.fillPoly(polygon, [np.rint(lip).astype(np.int32)], 1)
    return {'geometry': geometry, 'intrinsics': K, 'cameraVertices': cameraVertices,
            'normals': normals, 'source': _nativeLinear(source), 'mask': mask.astype(np.float32),
            'imageShape': (height, width), 'lookup': lookup,
            'distance': distance_transform_edt(~polygon.astype(bool)).astype(np.float32),
            'visibility': ProjectedSurfaceVisibility(cameraVertices, geometry['faces'], K, (height, width)),
            'depthTolerance': float(np.ptp(geometry['vertices'], axis=0).max()) * 1e-6,
            'corePixels': corePixels, 'outerPixels': outerPixels}


def _visibleMouthSamples(context, lookup, eligible):
    """Actual surface points and true perspective projection at UV samples."""
    indices = np.column_stack(np.nonzero(lookup['validMask'] & eligible))
    selectedPixels = tuple(indices.T)
    faceIds = lookup['faceIds'][selectedPixels]
    points = (context['cameraVertices'][context['geometry']['faces'][faceIds]]
              * lookup['barycentrics'][selectedPixels][:, :, None]).sum(1)
    projected = points @ context['intrinsics'].T
    xy = np.divide(projected[:, :2], projected[:, 2:], out=np.full((len(points), 2), -1.),
                   where=projected[:, 2:] > 1e-8)
    height, width = context['imageShape']
    bounds = ((xy >= 0).all(1) & (xy < [width - 1, height - 1]).all(1) & (points[:, 2] > 1e-8))
    band = samplePixels(context['distance'], xy)
    cos = np.sum(context['normals'][faceIds]
                 * (-points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-15)), axis=1)
    local = bounds & (band < context['outerPixels'])
    selected = np.flatnonzero(local & (cos > 0))
    if len(selected):
        hit = context['visibility'].closestDepth(points[selected])
        depth = np.isfinite(hit['depth']) & (np.abs(hit['depth'] - points[selected, 2]) <= context['depthTolerance'])
        supported = samplePixels(context['mask'], xy[selected]) >= .999
        accepted = selected[depth & supported]
    else:
        depth = supported = np.zeros(0, bool)
        accepted = selected
    alpha = 1 - _smooth((band[accepted] - context['corePixels'])
                        / (context['outerPixels'] - context['corePixels']))
    alpha[alpha < 1e-7] = 0
    positive = alpha > 0
    accepted, alpha = accepted[positive], alpha[positive]
    return {'indices': indices[accepted], 'faceIds': faceIds[accepted], 'alpha': alpha,
            'color': samplePixels(context['source'], xy[accepted]),
            'diagnostics': {'localSurfaceTexels': int(local.sum()),
                'normalRejectedLocalTexels': int((local & (cos <= 0)).sum()),
                'occlusionRejectedFrontFacingTexels': int((~depth).sum()),
                'sourceMaskRejectedFrontFacingTexels': int((~supported).sum())}}


def _projectMouth(linear, provenance, context, writable, referenceView):
    samples = _visibleMouthSamples(context, context['lookup'], writable)
    index = tuple(samples['indices'].T)
    alpha = np.zeros(linear.shape[:2], np.float32)
    alpha[index] = samples['alpha']
    result = linear.copy()
    result[index] = linear[index] * (1 - samples['alpha'][:, None]) + samples['color'] * samples['alpha'][:, None]
    changed = alpha > 0
    coverage = provenance['coverageMask'].copy()
    coverage[changed] = True
    sourceIndex = provenance['sourceIndex'].copy()
    sourceIndex[changed] = -2
    sourceIndex[alpha == 1] = referenceView
    distance, nearest = distance_transform_edt(~coverage, return_indices=True)
    gutter = (~context['lookup']['validMask']) & writable & (distance <= 2.) & changed[nearest[0], nearest[1]]
    result[gutter] = result[nearest[0][gutter], nearest[1][gutter]]
    updated = {**provenance, 'coverageMask': coverage, 'sourceIndex': sourceIndex,
               'frontAlpha': alpha, 'gutter': provenance['gutter'] | gutter}
    report = {**samples['diagnostics'], 'status': 'applied', 'changedTexels': int(changed.sum()),
              'fullyFrontTexels': int((alpha == 1).sum()),
              'newObservedTexels': int((coverage & ~provenance['coverageMask']).sum()),
              'mouthGutterTexelsNotObserved': int(gutter.sum()), 'referenceViewIndex': int(referenceView),
              'depthTolerance': context['depthTolerance'], 'minimumCosineExclusive': 0,
              'coreRadiusWorkingPixels': context['corePixels'], 'outerRadiusWorkingPixels': context['outerPixels'],
              'visibility': 'continuous perspective ray depth with two-sided occlusion',
              'outsideLocalRegionUnchanged': bool(np.array_equal(result[~(changed | gutter)], linear[~(changed | gutter)]))}
    return result, updated, report


def _filterMouth(linear, provenance, context, writable):
    """One-texel tent support from visible subpositions, separate from coverage."""
    geometry, size = context['geometry'], linear.shape[0]
    rgbSum = np.zeros(linear.shape, np.float64)
    weightSum = np.zeros(linear.shape[:2], np.float64)
    alphaSum = np.zeros_like(weightSum)
    candidate = (provenance['frontAlpha'] == 0) & writable
    owners = {}
    corners = [set(zip(face.tolist(), uvFace.tolist()))
               for face, uvFace in zip(geometry['faces'], geometry['uvFaces'])]
    offsets = (-.75, -.25, .25, .75)
    totalWeight = sum((1 - abs(x)) * (1 - abs(y)) for x in offsets for y in offsets)
    for dy in offsets:
        for dx in offsets:
            shifted = {**geometry, 'uvs': geometry['uvs'] - np.array([dx, dy]) / size}
            lookup, _ = _destinationUvLookup(shifted, size)
            samples = _visibleMouthSamples(context, lookup, candidate)
            index = tuple(samples['indices'].T)
            for (row, column), faceId in zip(samples['indices'], samples['faceIds']):
                owners.setdefault((int(row), int(column)), set()).add(int(faceId))
            weight = (1 - abs(dx)) * (1 - abs(dy))
            rgbSum[index] += weight * samples['color']
            alphaSum[index] += weight * samples['alpha']
            weightSum[index] += weight
    padding = weightSum > 0
    rejected = 0
    centerLookup = context['lookup']
    for pixel, contributors in owners.items():
        faces = set(contributors)
        if centerLookup['validMask'][pixel]:
            faces.add(int(centerLookup['faceIds'][pixel]))
        remaining = set(faces)
        connected = {remaining.pop()}
        while remaining:
            neighbors = {face for face in remaining if any(corners[face] & corners[other] for other in connected)}
            if not neighbors:
                break
            connected.update(neighbors)
            remaining.difference_update(neighbors)
        if remaining:
            padding[pixel] = False
            weightSum[pixel] = 0
            rejected += 1
    result = linear.copy()
    alpha = alphaSum[padding] / weightSum[padding]
    result[padding] = (linear[padding] * (1 - alpha[:, None])
                       + (rgbSum[padding] / weightSum[padding, None]) * alpha[:, None])
    updated = {**provenance, 'filterPadding': padding,
               'filterObservedKernelFraction': (weightSum / totalWeight).astype(np.float32)}
    report = {'status': 'applied', 'filterPaddingTexels': int(padding.sum()), 'newObservedCenterTexels': 0,
              'disconnectedUvGeometryFootprintsRejected': rejected,
              'observedCoverageUnchanged': bool(np.array_equal(updated['coverageMask'], provenance['coverageMask'])),
              'outsidePaddingUnchanged': bool(np.array_equal(result[~padding], linear[~padding])),
              'kernel': 'bilinear tent, radius one texel, 4x4 subpositions',
              'adjacency': 'contributors and center owner connect through shared (geometry vertex, UV vertex) within footprint',
              'source': 'only exact visible, positive-facing surface with source-mask support',
              'meaning': 'filter support only; not a new observed surface center or geometry repair'}
    return result, updated, report


def bakeClientTexture(geometry, cameras, delight, fallbackLinear, outputDirectory, *,
                      colmapExecutable, mouthState=None, referenceView=0, skinMask=None,
                      fallbackGain=None, mouthEnabled=True, filterFootprint=True,
                      mouthCorePixels=3., mouthOuterPixels=12., minimumMouthConfidence=.4,
                      boundaryBlend=True, boundaryInsetPixels=24., boundaryExtensionPixels=48.,
                      smallSkinHoles=True,
                      viewSelectionPrior=None, freezeViewLabels=False,
                      viewSelectionSmoothingWeight=.35,
                      viewSelectionSmoothingIterations=10,
                      timeoutSeconds=900, threads=4):
    """Bake final canonical client geometry and return linear RGB plus provenance.

    ``geometry`` contains ordered vertices/faces and ORIGINAL client uvs/uvFaces
    in top-left normalized pixel-edge coordinates. ``cameras`` project that
    exact geometry into ``delight['albedoImages']`` (native gamma2.2 NxHxWx3);
    ``validMasks`` are independent same-frame source support. Camera viewIndex
    values address the original image array and need not be contiguous.

    ``fallbackLinear`` is an RGB atlas, already linear; pass unscaled template
    RGB with ``fallbackGain='auto'`` to estimate the old median-overlap gain.
    None preserves the supplied fallback; an RGB gain applies that exact gain.
    ``skinMask`` optionally limits writable atlas texels, including any desired
    gutters. Omit it to use all original UV surface plus unused outside gutters.

    ``mouthState`` supplies ordered outerLipPixels/outerLipConfidence (12 rows),
    or ibugIds/frontTargetPixels/frontConfidence containing IDs 48..59. All are
    in the reference camera's working image. Missing, low-confidence, out-of-
    bounds or explicitly unreliable observations skip the whole mouth override.
    The multi-view COLMAP bake remains available and the reason is recorded.

    The output directory must be absent or empty. It receives the three COLMAP
    workspaces/logs, a linear clientTexture.npz, and diagnostics.json. The return
    dictionary has linearRGB, coverageMask (surface CENTER observations),
    colmapCoverageMask, atlasMask, assignedMask, sourceIndex (-1 unknown, -2
    mixed, otherwise original viewIndex), frontAlpha, gutter, filterPadding,
    filterObservedKernelFraction, and diagnostics. Gutter/filter padding never
    becomes observed coverage. The caller encodes linearRGB to sRGB once.
    Boundary blending extends safe interior skin across only the external photo
    silhouette; feature/core/filter pixels and source coverage stay unchanged.
    Small nonfeature internal holes can separately receive estimated skin color
    from their observed rim. Both stages add explicit contribution maps.
    ``viewSelectionPrior`` and ``freezeViewLabels`` require the experimental
    COLMAP runner. They are opt-in: color selects once, then support and marker
    reuse its exact face labels. Existing production invocations are unchanged.
    ``referenceView=None`` keeps multi-view COLMAP observations and skips mouth
    and exterior-boundary overrides, without silently choosing the first image.
    """
    started = perf_counter()
    if referenceView is not None and (isinstance(referenceView, (bool, np.bool_))
            or not isinstance(referenceView, (int, np.integer)) or referenceView < 0):
        raise ValueError('referenceView must be None or a nonnegative original view index')
    if (isinstance(threads, (bool, np.bool_)) or not isinstance(threads, (int, np.integer)) or threads < 1
            or not np.isfinite(timeoutSeconds) or timeoutSeconds <= 0):
        raise ValueError('Expected positive worker count and timeout')
    if (not np.isfinite([mouthCorePixels, mouthOuterPixels, minimumMouthConfidence]).all()
            or not 0 <= mouthCorePixels < mouthOuterPixels or not 0 <= minimumMouthConfidence <= 1):
        raise ValueError('Invalid mouth support radii or confidence threshold')
    if (not np.isfinite([boundaryInsetPixels, boundaryExtensionPixels]).all()
            or not 0 < boundaryInsetPixels <= 256 or not 0 < boundaryExtensionPixels <= 256):
        raise ValueError('Boundary distances must be finite atlas pixels in (0,256]')
    if not np.isfinite(viewSelectionSmoothingWeight) or viewSelectionSmoothingWeight < 0:
        raise ValueError('View selection smoothing weight must be finite and nonnegative')
    if (isinstance(viewSelectionSmoothingIterations, (bool, np.bool_))
            or not isinstance(viewSelectionSmoothingIterations, (int, np.integer))
            or not 0 <= viewSelectionSmoothingIterations <= 1000):
        raise ValueError('View selection smoothing iterations must be an integer in [0,1000]')
    explicitGain = None
    if isinstance(fallbackGain, str):
        if fallbackGain != 'auto':
            raise ValueError("fallbackGain must be None, 'auto', or an RGB gain")
    elif fallbackGain is not None:
        explicitGain = np.asarray(fallbackGain, np.float64)
        if explicitGain.shape != (3,) or not np.isfinite(explicitGain).all() or (explicitGain <= 0).any():
            raise ValueError('Explicit fallbackGain must be finite positive RGB')
    prepared, cameras, images, masks, fallback, writable = _validateInputs(
        geometry, cameras, delight, fallbackLinear, skinMask)
    if referenceView is not None and referenceView >= len(images):
        raise ValueError('referenceView does not address an input image')
    executable = Path(colmapExecutable).expanduser()
    if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError('colmapExecutable must be an absolute executable file path')
    executable = executable.resolve()
    output = Path(outputDirectory).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Client texture output directory must be absent or empty')
    output.mkdir(parents=True, exist_ok=True)
    if viewSelectionPrior is not None:
        viewSelectionPrior = Path(viewSelectionPrior).resolve()
        if not viewSelectionPrior.is_file():
            raise FileNotFoundError(viewSelectionPrior)
        freezeViewLabels = True
    resolution = fallback.shape[0]
    workspaces = {}
    quantized = np.rint(images * 255).astype(np.uint8)
    workspaces['colorWorkspace'] = prepareWorkspace(output / 'colorWorkspace', prepared, cameras, quantized)
    supportImages = np.repeat(np.rint(masks * 255).astype(np.uint8)[..., None], 3, axis=-1)
    workspaces['supportWorkspace'] = prepareWorkspace(output / 'supportWorkspace', prepared, cameras, supportImages)
    active = workspaces['colorWorkspace']['activeViewIndices']
    markerImages = np.zeros_like(quantized)
    for markerIndex, view in enumerate(active):
        markerImages[view] = viewMarkerColors(len(active))[markerIndex]
    workspaces['markerWorkspace'] = prepareWorkspace(output / 'markerWorkspace', prepared, cameras, markerImages)
    diagnostics = {'status': 'texturing', 'schemaVersion': 1, 'workspaces': workspaces,
        'colmapExecutable': {'path': str(executable), 'sha256': _digest(executable)},
        'destinationUvConvention': 'topLeftPixelEdges', 'geometryAndCamerasUnchanged': True,
        'inputColorEncoding': 'OpenDelight native gamma2.2', 'outputColorEncoding': 'linear RGB',
        'colorProcessing': 'PNG quantization only for native COLMAP inputs/outputs; subsequent projection and blending stay linear',
        'referenceViewIndex': int(referenceView) if referenceView is not None else None, 'colmap': {}}
    writeJson(output / 'diagnostics.json', diagnostics)
    frozenLabels = output / 'colorOutput/faceViewIds.txt'
    for workspace, result, support in [('colorWorkspace', 'colorOutput', False),
                                        ('supportWorkspace', 'supportOutput', True),
                                        ('markerWorkspace', 'markerOutput', True)]:
        selectionOptions = {}
        if freezeViewLabels:
            selectionOptions['eyelid_smoothing_weight'] = viewSelectionSmoothingWeight
            selectionOptions['eyelid_smoothing_iterations'] = viewSelectionSmoothingIterations
            selectionOptions['face_view_ids_output'] = output / result / 'faceViewIds.txt'
            if support:
                selectionOptions['face_view_ids_input'] = frozenLabels
            elif viewSelectionPrior is not None:
                selectionOptions['eyelid_prior_path'] = viewSelectionPrior
        diagnostics['colmap'][result] = _texturer(executable, output / workspace, output / result,
            support=support, threads=threads, timeout=timeoutSeconds,
            selectionOptions=selectionOptions)
    if freezeViewLabels:
        labelBytes = frozenLabels.read_bytes()
        if any((output / name / 'faceViewIds.txt').read_bytes() != labelBytes
               for name in ('supportOutput', 'markerOutput')):
            raise ValueError('Frozen COLMAP labels differ across color/support/marker')
        diagnostics['viewSelection'] = {
            'sharedLabelsSha256': _digest(frozenLabels),
            'labelsIdenticalAcrossThreePasses': True,
            'smoothingWeight': float(viewSelectionSmoothingWeight),
            'smoothingIterations': int(viewSelectionSmoothingIterations),
            'priorPath': str(viewSelectionPrior) if viewSelectionPrior is not None else None,
            'priorSha256': _digest(viewSelectionPrior) if viewSelectionPrior is not None else None}
    names = ('colorOutput', 'supportOutput', 'markerOutput')
    meshes = [loadTexturedMesh(output / name / 'mesh.ply', prepared) for name in names]
    for value in meshes[1:]:
        if (not np.array_equal(value['faceUvs'], meshes[0]['faceUvs'])
                or not np.array_equal(value['validFaces'], meshes[0]['validFaces'])):
            raise ValueError('COLMAP color/support/marker assignments or atlas layout differ')
    atlases = []
    for name in names:
        with Image.open(output / name / 'texture.png') as image:
            atlases.append(np.asarray(image.convert('RGB')).copy())
    if any(atlas.shape != atlases[0].shape for atlas in atlases[1:]):
        raise ValueError('COLMAP color/support/marker atlas dimensions differ')
    support = rebakeAtlas(prepared, meshes[1], atlases[1], resolution)
    markers = sourceSupportFromMarkers(prepared, meshes[2], atlases[2], cameras, masks, active, resolution)
    observed = (markers['coverageMask'] & support['assignedMask']
                & (support['rawRebaked'].min(-1) >= .999) & writable)
    if not observed.any():
        raise ValueError('No verified photograph support on the client UV atlas')
    baked = rebakeAtlas(prepared, meshes[0], atlases[0], resolution,
                         {'coverageMask': observed, 'sourceIndex': markers['sourceIndex']})
    bakedLinear = _nativeLinear(baked['albedo'])
    gain = np.ones(3, np.float64)
    gainMask = observed & (fallback.min(-1) > .08) & (bakedLinear.min(-1) > .03)
    if isinstance(fallbackGain, str):
        if gainMask.sum() >= 128:
            gain = np.clip(np.median(bakedLinear[gainMask], axis=0)
                           / np.maximum(np.median(fallback[gainMask], axis=0), .01), .65, 1.5)
    elif explicitGain is not None:
        gain = explicitGain
    fallback = np.clip(fallback * gain, 0, 1).astype(np.float32)
    linear = np.where(observed[..., None], bakedLinear, fallback)
    distance, nearest = distance_transform_edt(~observed, return_indices=True)
    gutter = (~baked['atlasMask']) & writable & (distance <= 2.)
    linear[gutter] = bakedLinear[nearest[0][gutter], nearest[1][gutter]]
    provenance = {'coverageMask': observed.copy(), 'colmapCoverageMask': observed.copy(),
        'atlasMask': baked['atlasMask'], 'assignedMask': baked['assignedMask'],
        'sourceIndex': baked['sourceIndex'].copy(), 'faceSourceIndex': markers['faceSourceIndex'],
        'frontAlpha': np.zeros(observed.shape, np.float32), 'gutter': gutter,
        'filterPadding': np.zeros(observed.shape, bool),
        'filterObservedKernelFraction': np.zeros(observed.shape, np.float32)}
    mouthReport = {'status': 'skipped', 'reasons': ['noReliableReference' if referenceView is None else 'disabled']}
    filterReport = {'status': 'skipped', 'reasons': ['mouthOverrideNotApplied']}
    if mouthEnabled and referenceView is not None:
        lip, mouthReport = _mouthObservation(mouthState, referenceView, images.shape[1:3], minimumMouthConfidence)
        cameraByView = {camera.get('viewIndex', index): camera for index, camera in enumerate(cameras)
                        if camera.get('valid', True)}
        if referenceView not in cameraByView:
            lip = None
            mouthReport = {**mouthReport, 'status': 'skipped',
                           'reasons': mouthReport.get('reasons', []) + ['referenceCameraRejected']}
        if lip is not None:
            lookup, _ = _destinationUvLookup(prepared, resolution)
            context = _mouthContext(prepared, cameraByView[referenceView], images[referenceView], masks[referenceView],
                                    lip, lookup, corePixels=mouthCorePixels, outerPixels=mouthOuterPixels)
            linear, provenance, projectedReport = _projectMouth(linear, provenance, context, writable, referenceView)
            mouthReport = {**projectedReport, 'observation': mouthReport}
            if filterFootprint:
                linear, provenance, filterReport = _filterMouth(linear, provenance, context, writable)
            else:
                filterReport = {'status': 'skipped', 'reasons': ['disabled']}
    boundaryReport = {'status': 'skipped', 'reasons': ['noReliableReference' if referenceView is None else 'disabled']}
    if boundaryBlend and referenceView is not None:
        from .clientTextureBoundary import blendClientTextureExterior
        boundary = blendClientTextureExterior(prepared, linear, provenance, fallback, cameras,
            mouthState=mouthState, referenceView=referenceView, imageShape=images.shape[1:3],
            writableMask=writable, insetPixels=boundaryInsetPixels, extensionPixels=boundaryExtensionPixels)
        linear = boundary['linearRGB']
        provenance.update(boundary['maps'])
        boundaryReport = boundary['diagnostics']
    smallHoleReport = {'status': 'skipped', 'reasons': ['disabled']}
    if smallSkinHoles:
        from .clientTextureBoundary import fillClientTextureSkinHoles
        holes = fillClientTextureSkinHoles(linear, provenance)
        linear = holes['linearRGB']
        provenance.update(holes['maps'])
        smallHoleReport = holes['diagnostics']
    diagnostics.update(status='succeeded', resolution=int(resolution),
        auxiliaryAtlasLayoutIdentical=True, colmapGeometry=meshes[0]['diagnostics'],
        rebake=baked['diagnostics'], support=markers['diagnostics'],
        templateLinearColorGain=gain.tolist(), fallbackGainSampleCount=int(gainMask.sum()),
        fallbackGainMode='preserveInput' if fallbackGain is None else ('auto' if isinstance(fallbackGain, str) else 'explicit'),
        mouth=mouthReport, filterFootprint=filterReport, boundaryBlend=boundaryReport,
        smallSkinHoles=smallHoleReport,
        colmapObservedTexels=int(observed.sum()), observedCenterTexels=int(provenance['coverageMask'].sum()),
        gutterTexelsNotObserved=int(provenance['gutter'].sum()), seconds=perf_counter() - started,
        limitations=['COLMAP view selection uses geometry; original source masks validate selected observations afterwards.',
            'COLMAP atlas patches retain its affine projected-corner interpolation; mouth override uses true perspective.',
            'Unobserved center texels retain template fallback unless separately marked as texture-filter support or estimated exterior skin.',
            'No mesh positions, topology, UVs or camera parameters are changed by texture baking.'])
    result = {'linearRGB': np.asarray(linear, np.float32), **provenance, 'diagnostics': diagnostics}
    np.savez_compressed(output / 'clientTexture.npz', **{key: value for key, value in result.items() if isinstance(value, np.ndarray)})
    writeJson(output / 'diagnostics.json', diagnostics)
    return result
