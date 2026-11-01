"""Conservative exterior-only SDF skin extension on the final client atlas.

Photographic coverage, estimated skin, and texture-filter padding retain
different meanings. Feature masks use the current observations and the audited
client surface binding; no subject-specific raster mask is stored here.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt, label
from scipy.sparse import coo_matrix, lil_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from .clientLandmarks import loadClientLandmarks, selectClientLandmarks
from .colmapTexture import _destinationUvLookup
from .skinExtrapolation import blendSkinExterior, insideDistance, smoothstep, _skinColorGate
from .texture import samplePixels


def _paintHull(shape, points, radius):
    canvas = np.zeros(shape, np.uint8)
    points = np.asarray(points, np.float64)
    if len(points) < 3 or not np.isfinite(points).all():
        return canvas.astype(bool)
    cv2.fillConvexPoly(canvas, cv2.convexHull(np.rint(points).astype(np.int32)), 1)
    if radius > 0:
        size = int(np.ceil(radius)) * 2 + 1
        canvas = cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return canvas.astype(bool)


def clientFeatureProtection(geometry, cameras, mouthState, referenceView, imageShape,
                            resolution, *, lookup=None):
    """Return union of fixed client semantic areas and current-photo features.

    Semantic binding gives a conservative target-surface protection even if
    photographs do not provide landmarks. Current image polygons additionally
    protect photographic feature positions when binding and image differ.
    These masks are exclusions, not a positive learned skin segmentation.
    """
    shape = (resolution, resolution)
    protected = np.zeros(shape, bool)
    records = []
    atlasScale = resolution / 1024.
    try:
        binding = loadClientLandmarks(geometry['faces'])
        uvFaces = geometry['uvFaces']
        a, b = uvFaces.ravel(), np.roll(uvFaces, 1, axis=1).ravel()
        _, uvChart = connected_components(coo_matrix((np.ones(len(a)), (a, b)),
            shape=(len(geometry['uvs']), len(geometry['uvs']))), directed=False)
        for name, radius in [('rightEye', 9), ('leftEye', 9), ('rightBrow', 9),
                             ('leftBrow', 9), ('mouth', 14), ('nose', 6)]:
            selected = selectClientLandmarks(binding, name)
            good = selected['validMask']
            uv = (geometry['uvs'][geometry['uvFaces'][selected['faceIds'][good]]]
                  * selected['barycentricWeights'][good, :, None]).sum(1)
            # Eyelid bindings can straddle a UV seam into an internal eye
            # chart. A single hull would draw a diagonal across unrelated skin.
            charts = uvChart[uvFaces[selected['faceIds'][good], 0]]
            region = np.zeros(shape, bool)
            for chart in np.unique(charts):
                chartPoints = uv[charts == chart] * resolution - .5
                if len(chartPoints) >= 3:
                    region |= _paintHull(shape, chartPoints, radius * atlasScale)
                else:
                    isolated = np.zeros(shape, np.uint8)
                    for point in chartPoints:
                        cv2.circle(isolated, tuple(np.rint(point).astype(int)),
                                   max(1, int(np.ceil(radius * atlasScale))), 1, -1)
                    region |= isolated.astype(bool)
            protected |= region
            records.append({'source': 'clientSurfaceBinding', 'group': name, 'texels': int(region.sum())})
    except (ValueError, FileNotFoundError) as error:
        records.append({'source': 'clientSurfaceBinding', 'status': 'unavailable', 'reason': str(error)})
    state = {} if mouthState is None else mouthState
    detected = np.asarray(state.get('faceLandmarksWorking', []), np.float64)
    cameraByView = {camera.get('viewIndex', i): camera for i, camera in enumerate(cameras)
                    if camera.get('valid', True)}
    referenceMatches = int(state.get('referenceViewIndex', referenceView)) == referenceView
    if (detected.shape == (68, 2) and np.isfinite(detected).all() and referenceMatches
            and referenceView in cameraByView):
        height, width = imageShape
        imageScale = min(imageShape) / 512.
        imageMask = np.zeros(imageShape, bool)
        for name, ids, radius in [('imageLeftEye', range(36, 42), 6), ('imageRightEye', range(42, 48), 6),
                                  ('imageLeftBrow', range(17, 22), 6), ('imageRightBrow', range(22, 27), 6),
                                  ('mouth', range(48, 60), 10), ('noseBase', range(31, 36), 5)]:
            points = detected[list(ids)]
            if (points < 0).any() or (points >= [width - 1, height - 1]).any():
                records.append({'source': 'currentPhoto', 'group': name, 'status': 'outsideImage'})
                continue
            imageMask |= _paintHull(imageShape, points, radius * imageScale)
        if lookup is None:
            lookup, _ = _destinationUvLookup(geometry, resolution)
        valid = lookup['validMask']
        faceIds = lookup['faceIds'][valid]
        points = (geometry['vertices'][geometry['faces'][faceIds]]
                  * lookup['barycentrics'][valid, :, None]).sum(1)
        camera = cameraByView[referenceView]
        R, t, K = (np.asarray(camera[key], np.float64) for key in ('rotation', 'translation', 'intrinsics'))
        cameraPoints = points @ R.T + t
        projected = cameraPoints @ K.T
        xy = np.divide(projected[:, :2], projected[:, 2:], out=np.full((len(points), 2), -1.),
                       where=projected[:, 2:] > 1e-8)
        inside = (cameraPoints[:, 2] > 1e-8) & (xy >= 0).all(1) & (xy < [width - 1, height - 1]).all(1)
        # Protection may conservatively include hidden surfaces at the same ray;
        # it never creates donors or claims visibility/photographic support.
        mapped = inside & (samplePixels(imageMask.astype(np.float32), xy) > 0)
        photoProtection = np.zeros(shape, bool)
        photoProtection[valid] = mapped
        protected |= photoProtection
        records.append({'source': 'currentPhoto', 'group': 'allFeatures', 'texels': int(photoProtection.sum())})
    else:
        records.append({'source': 'currentPhoto', 'status': 'unavailable'})
    return protected, {'method': 'currentPhotoPolygonsAndClientSemanticBindings', 'groups': records,
                       'protectedTexels': int(protected.sum()),
                       'positiveSkinSegmentation': False}


def _exteriorSilhouette(observed, domain):
    """Match skinExtrapolation's silhouette; retain filled holes for exclusion."""
    charts, count = label(domain)
    silhouette = np.zeros(domain.shape, bool)
    main = np.zeros(domain.shape, bool)
    for chart in range(1, count + 1):
        component, _ = label(observed & (charts == chart))
        sizes = np.bincount(component.ravel())
        eligible = np.flatnonzero(sizes >= 128)
        eligible = eligible[eligible != 0]
        if len(eligible):
            seed = np.isin(component, eligible)
            main |= seed
            silhouette |= binary_fill_holes(seed)
    return main, silhouette


def exteriorEdgeDiagnostics(before, after, observed, domain, protected):
    """Compare the SAME original exterior pixel pairs, excluding inner holes.

    These one-pixel RGB discontinuities are a seam diagnostic, not an image
    quality score; neither exposure accuracy nor identity follows from them.
    """
    main, silhouette = _exteriorSilhouette(observed, domain)
    jumpsBefore, jumpsAfter = [], []
    for axis in (0, 1):
        a = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        b = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        edge = ((main[a] & ~silhouette[b]) | (main[b] & ~silhouette[a]))
        edge &= domain[a] & domain[b] & ~protected[a] & ~protected[b]
        jumpsBefore.extend(np.abs(before[a] - before[b]).mean(-1)[edge].tolist())
        jumpsAfter.extend(np.abs(after[a] - after[b]).mean(-1)[edge].tolist())
    return {'method': 'mean absolute linear RGB jump over fixed original exterior-neighbor pixel pairs',
            'edgePairCount': len(jumpsBefore),
            'meanBefore': float(np.mean(jumpsBefore)) if jumpsBefore else None,
            'meanAfter': float(np.mean(jumpsAfter)) if jumpsAfter else None,
            'p95Before': float(np.percentile(jumpsBefore, 95)) if jumpsBefore else None,
            'p95After': float(np.percentile(jumpsAfter, 95)) if jumpsAfter else None,
            'qualityScore': False}


def blendClientTextureExterior(geometry, linearRGB, provenance, fallbackLinear, cameras, *,
                               mouthState=None, referenceView=0, imageShape=(512, 512),
                               writableMask=None, insetPixels=24., extensionPixels=48.,
                               donorBandPixels=36., seamFeatherPixels=24.):
    """Extend safe skin at the outer photo silhouette, preserving inner detail.

    The existing SDF routine's legacy inward feather is intentionally *not*
    adopted. Only its explicit outer-repair/skin-extension contribution is
    applied to this already-baked texture. Mouth/filter/gutter pixels, feature
    exclusions, internal UV/photo holes and the atlas core remain unchanged.
    Coverage/source indices are never enlarged or rewritten by this stage.
    """
    distances = (insetPixels, extensionPixels, donorBandPixels, seamFeatherPixels)
    if any(not np.isfinite(value) or not 0 < value <= 256 for value in distances):
        raise ValueError('Client texture boundary distances must be finite atlas pixels in (0,256]')
    if insetPixels < seamFeatherPixels:
        raise ValueError('Client texture boundary requires insetPixels >= seamFeatherPixels '
                         'so the skin repair band contains the complete feather band')
    linear = np.asarray(linearRGB, np.float32)
    fallback = np.asarray(fallbackLinear, np.float32)
    observed = np.asarray(provenance['coverageMask'])
    atlas = np.asarray(provenance['atlasMask'])
    shape = linear.shape[:2]
    if (linear.shape != (*shape, 3) or fallback.shape != linear.shape or shape[0] != shape[1]
            or observed.shape != shape or atlas.shape != shape or observed.dtype != np.bool_ or atlas.dtype != np.bool_):
        raise ValueError('Expected RGB client atlases and independent boolean coverage/atlas masks')
    writable = np.ones(shape, bool) if writableMask is None else np.asarray(writableMask)
    if writable.shape != shape or writable.dtype != np.bool_:
        raise ValueError('writableMask must be a boolean atlas domain')
    domain = atlas & writable
    protected, protectionReport = clientFeatureProtection(geometry, cameras, mouthState, referenceView,
                                                           imageShape, shape[0])
    padding = np.asarray(provenance.get('filterPadding', np.zeros(shape, bool)), bool)
    gutter = np.asarray(provenance.get('gutter', np.zeros(shape, bool)), bool)
    front = np.asarray(provenance.get('frontAlpha', np.zeros(shape))) > 0
    # Protect the filter footprint and mouth override exactly; neither defines
    # an exterior silhouette and neither supplies extrapolated skin colors.
    protected |= binary_dilation(front | padding, iterations=max(1, int(round(shape[0] / 512.)))) | gutter
    safe = observed & domain & ~protected & ~padding & ~gutter
    if not protectionReport['protectedTexels']:
        empty = np.zeros(shape, np.float32)
        return {'linearRGB': linear.copy(), 'maps': {'boundaryEstimatedSkinAlpha': empty},
                'diagnostics': {'method': 'sdfSafeSkinExterior', 'status': 'skipped',
                                'reason': 'noReliableFeatureProtection', 'coverageExpanded': False}}
    result = blendSkinExterior(linear, observed, fallback, domain,
        safeSkinMask=safe, protectedMask=protected, seamFeatherPixels=seamFeatherPixels,
        insetPixels=insetPixels, extensionPixels=extensionPixels, donorBandPixels=donorBandPixels)
    estimated = result['maps']['estimatedSkinBlendAlpha']
    repair = result['maps']['skinRepairWeight']
    modified = ((estimated > 0) | (repair > 0)) & domain & ~protected
    main, silhouette = _exteriorSilhouette(observed, domain)
    internalHoles = silhouette & domain & ~observed
    modified &= ~internalHoles
    color = linear.copy()
    color[modified] = result['color'][modified]
    directAlpha = observed.astype(np.float32)
    directAlpha[modified] = result['maps']['directPhotoBlendAlpha'][modified]
    estimateAlpha = np.where(modified, estimated, 0).astype(np.float32)
    # The template barrier deliberately forbids growing skin into hair. At
    # that barrier the *observed outer band* must still fade to the template,
    # otherwise an unsupported skin island retains a hard rim on black hair.
    # Use the filled exterior SDF, never distance to an eye/lip/internal hole.
    interior = insideDistance(silhouette)
    outerObserved = main & domain & ~protected & (interior < max(insetPixels, seamFeatherPixels))
    repairPotential = 1 - smoothstep(interior / insetPixels)
    targetGate = np.divide(repair, repairPotential, out=np.zeros_like(repair),
                           where=repairPotential > 1e-6)
    targetGate = np.clip(targetGate, 0, 1)
    exteriorAlpha = smoothstep(interior / seamFeatherPixels)
    exteriorAlpha += (1 - exteriorAlpha) * targetGate
    repaired = linear * (1 - repair[..., None]) + result['debug']['extendedSkinLinear'] * repair[..., None]
    blended = repaired * exteriorAlpha[..., None] + fallback * (1 - exteriorAlpha[..., None])
    color[outerObserved] = blended[outerObserved]
    directAlpha[outerObserved] = (exteriorAlpha * (1 - repair))[outerObserved]
    estimateAlpha[outerObserved] = (exteriorAlpha * repair)[outerObserved]
    modified |= outerObserved
    maps = {'boundarySafeSkinDonor': result['maps']['safeSkinDonor'],
            'boundaryProtectedFeatures': protected, 'boundaryInternalHoles': internalHoles,
            'boundaryModified': modified, 'boundaryExtrapolated': result['maps']['skinExtrapolated'] & modified,
            'boundaryRepairWeight': np.where(modified, repair, 0).astype(np.float32),
            'boundaryExteriorFeather': outerObserved & (targetGate < 1),
            'boundaryEstimatedSkinAlpha': estimateAlpha, 'boundaryDirectPhotoAlpha': directAlpha,
            'boundarySdfPreview': result['maps']['skinSdfPreview']}
    diagnostics = {**result['diagnostics'], 'method': 'sdfSafeSkinExteriorOnly',
        'skinDonorPolicy': 'featureExcludedInteriorAndRelativeColor',
        'status': 'applied' if modified.any() else 'noEligibleExterior',
        'featureProtection': protectionReport,
        'safeDonorMeaning': 'observed interior pixels excluding current-photo/client features, then relative chroma/luminance gate',
        'positiveSkinSegmentation': False, 'legacyInternalFeatherApplied': False,
        'templateBarrierBoundaryPolicy': 'fade observed exterior band to template where safe-skin extension is blocked',
        'modifiedTexels': int(modified.sum()), 'modifiedObservedTexels': int((modified & observed).sum()),
        'estimatedExteriorTexels': int((modified & ~observed).sum()),
        'protectedFeaturesExactlyUnchanged': bool(np.array_equal(color[protected], linear[protected])),
        'internalHolesExactlyUnchanged': bool(np.array_equal(color[internalHoles], linear[internalHoles])),
        'mouthOverrideExactlyUnchanged': bool(np.array_equal(color[front], linear[front])),
        'filterPaddingExactlyUnchanged': bool(np.array_equal(color[padding], linear[padding])),
        'outsideBoundaryExactlyUnchanged': bool(np.array_equal(color[~modified], linear[~modified])),
        'sourceCoverageExactlyUnchanged': True,
        'uvChartLimitation': 'Exterior SDF propagation uses connected raster atlas domains; touching UV charts may need explicit topology labels',
        'exteriorEdgeDiagnostic': exteriorEdgeDiagnostics(linear, color, observed, domain, protected)}
    return {'linearRGB': color, 'maps': maps, 'diagnostics': diagnostics}


def fillClientTextureSkinHoles(linearRGB, provenance, *, maximumHoleTexels=128,
                               featureMarginPixels=4.):
    """Estimate only small enclosed skin holes from the unchanged observed rim.

    All sizes scale from a 1024px atlas. Protected feature/cavity holes, large
    holes, holes lacking a fully observed rim, and unreliable color donors are
    rejected. The discrete harmonic estimate is a separate provenance channel;
    no observed color or photographic coverage is changed by this operation.
    """
    before = np.asarray(linearRGB, np.float32)
    coverage = np.asarray(provenance['coverageMask'])
    if (not np.isfinite([maximumHoleTexels, featureMarginPixels]).all()
            or maximumHoleTexels < 1 or featureMarginPixels < 0):
        raise ValueError('Expected positive hole area and nonnegative feature margin')
    if not {'boundaryProtectedFeatures', 'boundaryInternalHoles'} <= provenance.keys():
        return {'linearRGB': before.copy(), 'maps': {},
                'diagnostics': {'status': 'skipped', 'reason': 'missingInternalHoleClassification'}}
    protected = (provenance['boundaryProtectedFeatures']
                 | provenance.get('filterPadding', np.zeros(coverage.shape, bool))
                 | provenance.get('gutter', np.zeros(coverage.shape, bool)))
    labels, count = label(provenance['boundaryInternalHoles'])
    nearProtected = distance_transform_edt(~protected)
    result, filled = before.copy(), np.zeros(coverage.shape, bool)
    scale = coverage.shape[0] / 1024.
    records = []
    for index in range(1, count + 1):
        hole = labels == index
        y, x = np.where(hole)
        record = {'component': index, 'texels': len(x),
                  'boundsInclusive': [int(x.min()), int(y.min()), int(x.max()), int(y.max())]}
        ring = binary_dilation(hole) & ~hole
        wideRing = binary_dilation(hole, iterations=max(1, int(round(4 * scale)))) & ~hole
        donor = wideRing & coverage & ~protected
        rejection = []
        if len(x) > maximumHoleTexels * scale**2:
            rejection.append('tooLargeForSmallHoleFill')
        if x.min() == 0 or y.min() == 0 or x.max() == coverage.shape[1]-1 or y.max() == coverage.shape[0]-1:
            rejection.append('touchesAtlasFrame')
        if protected[hole].any() or nearProtected[hole].min() < featureMarginPixels * scale:
            rejection.append('nearProtectedFeature')
        if not coverage[ring].all():
            rejection.append('notFullyEnclosedByPhotoObservations')
        if donor.sum() < 16:
            rejection.append('insufficientLocalObservedDonors')
        if not rejection:
            skinGate = _skinColorGate(before, donor)
            ratio = float(skinGate[donor].mean())
            record['relativeColorGateDonorFraction'] = ratio
            if ratio < .8 or np.median(before[donor] @ [.2126, .7152, .0722]) < .035:
                rejection.append('localDonorsFailRelativeSkinColorGate')
        record['rejectionReasons'] = rejection
        if rejection:
            record['status'] = 'rejected'
            records.append(record)
            continue
        # Four-neighbor Laplace equation; only the hole's unknown pixels enter
        # the solve. The surrounding observed skin remains an exact boundary.
        indexMap = np.full(coverage.shape, -1, np.int32)
        indexMap[y, x] = np.arange(len(x))
        matrix = lil_matrix((len(x), len(x)))
        rhs = np.zeros((len(x), 3), np.float64)
        for row, (yy, xx) in enumerate(zip(y, x)):
            matrix[row, row] = 4
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = indexMap[yy + dy, xx + dx]
                if neighbor >= 0:
                    matrix[row, neighbor] = -1
                else:
                    rhs[row] += before[yy + dy, xx + dx]
        result[y, x] = np.clip(spsolve(matrix.tocsr(), rhs), 0, 1)
        filled |= hole
        record['status'] = 'estimatedFromLocalObservedBoundary'
        records.append(record)
    return {'linearRGB': result,
            'maps': {'estimatedInteriorSkinAlpha': filled.astype(np.float32), 'interiorSkinHoleFilled': filled},
            'diagnostics': {'status': 'applied' if filled.any() else 'noQualifiedSmallHoles',
                'method': 'harmonicLinearRgbFromObservedRim', 'maximumHoleTexelsAt1024': maximumHoleTexels,
                'featureMarginPixelsAt1024': featureMarginPixels, 'filledTexels': int(filled.sum()),
                'components': records, 'coverageExpanded': False,
                'observedColorsExactlyUnchanged': bool(np.array_equal(before[coverage], result[coverage])),
                'protectedFeaturesExactlyUnchanged': bool(np.array_equal(before[protected], result[protected])),
                'outsideFilledHolesExactlyUnchanged': bool(np.array_equal(before[~filled], result[~filled])),
                'meaning': 'Estimated colors within small nonfeature holes; not new photo observations'}}
