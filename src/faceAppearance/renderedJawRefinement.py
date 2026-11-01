"""Propose conservative chin changes from render-detected semantic bindings.

The proposals in this module are not accepted reconstructions.  A binding
attached to an initial render is only a local prediction of detector motion.
Every proposed surface must be rendered again and passed through the same
detector before a caller can accept it.  The unchanged input is always offered.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from .adaptiveChinRefinement import (
    _LIMITS as _GEOMETRY_LIMITS, _binding, _dihedral, _normals, _prepare_views,
    _project, _smooth, _topology,
)
from .clientChinRefinement import _ids, _json
from .clientGeometrySafety import intersectionPairs
from .frontLandmarkRefinement import checkFrontGeometry
from .geometryRefinement import _validateCameras, _validateMesh


_LIMITS = dict(_GEOMETRY_LIMITS,
    minimumProposalImprovementFaceFraction=.004,
    minimumProposalRelativeImprovement=.08,
    minimumReferenceCentralPointCount=2,
    requiredReferenceSemanticId=8,
    minimumPointsPerComparableView=3,
    maximumSemanticTargetDistanceFaceFraction=.18,
    maximumSourceReprojectionErrorFaceFraction=.006,
    robustResidualFaceFraction=.03,
    minimumPositiveAmplitudeFaceFraction=.006,
    maximumCrossViewIncreaseFaceFraction=.006,
    maximumCrossViewPointIncreaseFaceFraction=.016,
)


_SAFE_STEP_SEARCH = {
    'maximumHalvings': 8,
    'maximumBracketRefinements': 6,
    'candidateFractions': (.35, .6, .85, 1.),
}


def _search_safe_amplitude(fitted_amplitude, cap, evaluate):
    """Sample a bounded feasible step without inferring unsampled safety.

    ``evaluate`` contains only original-relative geometry checks. In particular,
    a safe sample with no useful semantic motion may still bound the search.
    Bracket refinement only chooses additional sample locations; it does not
    establish monotonic feasibility or a maximal safe amplitude.
    """
    start = min(float(fitted_amplitude), float(cap))
    report = {
        'method': 'bounded halving followed by sampled feasible/infeasible bracket refinement',
        'fittedAmplitudePixels': float(fitted_amplitude),
        'maximumAllowedAmplitudePixels': float(cap),
        'initialAmplitudePixels': start,
        'limits': dict(_SAFE_STEP_SEARCH),
        'maximumSearchEvaluations': 1+_SAFE_STEP_SEARCH['maximumHalvings']
            +_SAFE_STEP_SEARCH['maximumBracketRefinements'],
        'maximumCandidateRechecks': len(_SAFE_STEP_SEARCH['candidateFractions']),
        'predicate': 'original-relative geometry, smoothness, depth, fixed supports and no new intersections only',
        'semanticGatesUsedInSearch': False,
        'assumesMonotoneSafety': False,
        'claimsMaximumSafeAmplitude': False,
        'trials': [],
    }

    def sample(amplitude, phase):
        _, safety = evaluate(amplitude, include_intersections=True)
        safe = safety['status'] == 'geometry_safe'
        report['trials'].append({'amplitudePixels': float(amplitude), 'phase': phase,
                                 'feasible': safe, **safety})
        return safe

    feasible, infeasible = None, None
    for step in range(_SAFE_STEP_SEARCH['maximumHalvings']+1):
        amplitude = start/(2.**step)
        if sample(amplitude, 'initial' if step == 0 else 'halving'):
            feasible = amplitude
            break
        infeasible = amplitude
    report['initialWitnessedBracketPixels'] = {'feasible': feasible, 'infeasible': infeasible}
    if feasible is None:
        report.update(status='no_feasible_sample_within_search_budget',
                      verifiedFeasibleAmplitudePixels=None, searchEvaluationCount=len(report['trials']))
        return None, report
    if infeasible is not None:
        for _ in range(_SAFE_STEP_SEARCH['maximumBracketRefinements']):
            midpoint = .5*(feasible+infeasible)
            if midpoint == feasible or midpoint == infeasible:
                break
            if sample(midpoint, 'bracket_refinement'):
                feasible = midpoint
            else:
                infeasible = midpoint
    report.update(status='verified_feasible_sample_found',
        verifiedFeasibleAmplitudePixels=float(feasible),
        finalWitnessedBracketPixels={'feasible': feasible, 'infeasible': infeasible},
        searchEvaluationCount=len(report['trials']),
        interpretation='Only evaluated amplitudes have safety evidence; smaller steps and candidate fractions require their own full checks')
    return feasible, report


def _points(vertices, view):
    return np.sum(vertices[view['indices']]*view['weights'][..., None], axis=1)


def _prepare_semantic_views(vertices, cameras, correspondence):
    """Validate frozen rendered correspondences; never silently rebind them."""
    views, reports, seen = [], [], set()
    for supplied in correspondence.get('views', []):
        index = int(supplied['viewIndex'])
        if index in seen or not 0 <= index < len(cameras):
            raise ValueError('Duplicate or out-of-range semantic view index')
        seen.add(index)
        ids = np.asarray(supplied.get('semanticIds', supplied.get('ids')), int)
        indices = np.asarray(supplied['vertexIndices'])
        barycentric = np.asarray(supplied.get('barycentricWeights', supplied.get('barycentric')), float)
        target = np.asarray(supplied['targetPixels'], float)
        source = np.asarray(supplied['sourcePixels'], float)
        confidence = np.asarray(supplied.get('confidence', np.ones(len(ids))), float)
        reliability = (np.asarray(supplied['frozenReliabilityWeights'], float)
                       if 'frozenReliabilityWeights' in supplied else None)
        valid = np.asarray(supplied.get('acceptedMask', np.ones(len(ids), bool)), bool)
        scale = float(supplied['faceScalePixels'])
        n = len(ids)
        # JSON cannot preserve the trailing dimensions of empty arrays.
        if not n:
            reports.append({'viewIndex': index, 'status': 'no_accepted_bindings', 'count': 0})
            continue
        if (ids.shape != (n,) or len(np.unique(ids)) != n or ((ids < 0) | (ids > 16)).any()
                or indices.shape != (n, 3) or indices.dtype.kind not in 'iu'
                or barycentric.shape != indices.shape or target.shape != (n, 2)
                or source.shape != (n, 2) or confidence.shape != (n,) or valid.shape != (n,)
                or not np.isfinite(scale) or scale <= 0):
            raise ValueError('Malformed rendered semantic correspondence')
        if (indices.min() < 0 or indices.max() >= len(vertices)
                or not np.isfinite(barycentric).all() or (barycentric < -1e-7).any()
                or not np.allclose(barycentric.sum(1), 1., atol=1e-6)):
            raise ValueError('Invalid triangle binding in rendered semantic correspondence')
        if reliability is not None and (reliability.shape != (n,)
                or not np.isfinite(reliability).all() or (reliability <= 0).any()
                or not np.isfinite(reliability.sum()) or reliability.sum() <= 0):
            raise ValueError('Frozen semantic reliability weights must be finite and positive')
        view = {'index': index, 'ids': ids, 'indices': indices, 'weights': barycentric,
                'target': target, 'source': source, 'confidence': confidence,
                'scale': scale, 'camera': cameras[index]}
        pixels, depth = _project(_points(vertices, view), cameras[index])
        projection_error = np.linalg.norm(pixels-source, axis=1)
        residual = np.linalg.norm(pixels-target, axis=1)
        keep = (valid & np.isfinite(target).all(1) & np.isfinite(source).all(1)
                & np.isfinite(confidence) & (confidence > 0) & (depth > 0)
                & (projection_error <= _LIMITS['maximumSourceReprojectionErrorFaceFraction']*scale)
                & (residual <= _LIMITS['maximumSemanticTargetDistanceFaceFraction']*scale))
        eligible = supplied.get('eligible', True)
        row = {'viewIndex': index, 'inputCount': n, 'count': int(keep.sum()),
               'keptSemanticIds': ids[keep], 'rejectedSemanticIds': ids[~keep],
               'sourceProjectionErrorPixels': projection_error,
               'status': 'comparable' if eligible and keep.sum() >= _LIMITS['minimumPointsPerComparableView'] else 'insufficient_bindings'}
        if reliability is not None:
            row.update(frozenReliabilityInputWeights=reliability,
                frozenReliabilityRemovedSemanticIds=ids[~keep],
                frozenReliabilityRemovedMass=float(reliability[~keep].sum()/reliability.sum()),
                frozenReliabilityRenormalizedAfterFiltering=False)
            # A failed frozen binding invalidates this view. Do not improve
            # its effective evidence by dropping IDs and renormalizing.
            if not keep.all():
                row['status'] = 'invalid_frozen_reliability_binding'
        reports.append(row)
        if row['status'] != 'comparable':
            continue
        for key in ('ids', 'indices', 'weights', 'target', 'source', 'confidence'):
            view[key] = view[key][keep]
        view['confidence'] = np.minimum(view['confidence'], 1.)
        if reliability is not None:
            view['reliabilityWeights'] = reliability[keep]/reliability[keep].sum()
            row['frozenReliabilityWeights'] = view['reliabilityWeights']
        view['beforePixels'], _ = _project(_points(vertices, view), view['camera'])
        view['beforeError'] = np.linalg.norm(view['beforePixels']-view['target'], axis=1)
        views.append(view)
    return views, reports


def _metrics(vertices, views):
    rows = []
    for view in views:
        pixels, depth = _project(_points(vertices, view), view['camera'])
        error = np.linalg.norm(pixels-view['target'], axis=1)
        before = view['beforeError']
        increase = (error-before)/view['scale']
        reliability = view.get('reliabilityWeights')
        before_mean = float(before.mean()) if reliability is None else float(np.dot(reliability, before))
        after_mean = float(error.mean()) if reliability is None else float(np.dot(reliability, error))
        increase_mean = float(increase.mean()) if reliability is None else float(np.dot(reliability, increase))
        row = {'viewIndex': view['index'], 'semanticIds': view['ids'],
            'beforeErrorPixels': before_mean, 'afterErrorPixels': after_mean,
            'beforePointErrorsPixels': before, 'afterPointErrorsPixels': error,
            'beforePixels': view['beforePixels'], 'afterPixels': pixels, 'targetPixels': view['target'],
            'faceScalePixels': view['scale'], 'meanIncreaseFaceFraction': increase_mean,
            'maximumPointIncreaseFaceFraction': float(increase.max()),
            'accepted': bool((depth > 0).all() and np.isfinite(error).all()
                and increase_mean <= _LIMITS['maximumCrossViewIncreaseFaceFraction']
                and increase.max() <= _LIMITS['maximumCrossViewPointIncreaseFaceFraction']),
            'meaning': 'Frozen render-binding prediction; live detector not yet evaluated'}
        if reliability is not None:
            row.update(frozenReliabilityWeights=reliability,
                unweightedBeforeErrorPixels=float(before.mean()), unweightedAfterErrorPixels=float(error.mean()),
                unweightedMeanIncreaseFaceFraction=float(increase.mean()),
                weighting='frozen input reliability; every frozen ID retained')
        rows.append(row)
    return rows


def _fit_amplitude(vertices, direction, views):
    """Robust scalar projection fit, with equal total weight per view."""
    jacobians, residuals, weights = [], [], []
    for view in views:
        point = _points(vertices, view)
        delta = _points(direction, view)
        camera, scale = view['camera'], view['scale']
        c = point@camera['rotation'].T+camera['translation']
        dc = delta@camera['rotation'].T
        h, dh = c@camera['intrinsics'].T, dc@camera['intrinsics'].T
        jacobian = (dh[:, :2]*h[:, 2:]-h[:, :2]*dh[:, 2:])/(h[:, 2:]**2)/scale
        residual = (h[:, :2]/h[:, 2:]-view['target'])/scale
        weight = view['confidence']/view['confidence'].sum()
        jacobians.append(jacobian)
        residuals.append(residual)
        weights.append(weight)
    j, r, w = np.concatenate(jacobians), np.concatenate(residuals), np.concatenate(weights)
    amplitude = 0.
    for _ in range(4):
        length = np.linalg.norm(r+amplitude*j, axis=1)
        robust = np.minimum(1., _LIMITS['robustResidualFaceFraction']/np.maximum(length, 1e-12))
        effective = w*robust
        denominator = float(np.sum(effective[:, None]*j*j))
        if denominator <= 1e-16:
            return 0., {'reason': 'insufficient_semantic_motion_support'}
        amplitude = -float(np.sum(effective[:, None]*j*r))/denominator
    return amplitude, {'method': 'four robust scalar projection iterations; views weighted equally',
                       'amplitudePixels': amplitude, 'linearizedObjective': float(np.sum(w[:, None]*(r+amplitude*j)**2)),
                       'baselineObjective': float(np.sum(w[:, None]*r*r))}


def proposeRenderedJawCandidates(vertices, faces, cameras, photoObservations, correspondence, *,
                                 fixedVertexIds=(), protectedVertexIds=(), landmarkBinding=None,
                                 maximumCandidates=3, allSafeCandidates=False, frontPrimary=False,
                                 safeStepSearch=False, referenceViewIndex=None):
    """Return no-op and safe provisional semantic candidates.

    By default return at most three candidates, one per smoothing family.
    ``allSafeCandidates`` exposes every safe member of the same finite grid
    for a bounded exhaustive live-render experiment; it changes no gates.
    ``frontPrimary`` fits and ranks using only the reliable frontal reference.
    Other comparable semantic views are reported as diagnostics and do not
    veto a proposal; original-relative geometric constraints still apply.
    ``safeStepSearch`` first samples a geometrically feasible amplitude per
    smoothing family with bounded halving and bracket refinement, then fully
    rechecks four fractions of that witnessed amplitude. The search uses no
    semantic acceptance gate and establishes no global maximum or monotonicity.
    The default retains the existing fitted-amplitude grid.

    ``referenceViewIndex`` optionally pins the caller's shared photo reference.
    It never bypasses the jaw support gates or falls back to another image.

    ``correspondence`` is produced by ``prepareRenderedJawCorrespondence``.
    Its views carry filtered semanticIds/vertexIndices/barycentricWeights,
    sourcePixels/targetPixels, confidence, and faceScalePixels. Optional positive
    ``frozenReliabilityWeights`` set the semantic metric means; they are frozen
    by the caller and normalized after source-binding validation. ``confidence``
    continues to supply the amplitude-fit weights, so a caller enabling frozen
    reliability should also supply its corresponding fitting confidences.
    Cameras and all
    photo coordinates are in their saved working-image system. A source view's
    image evidence is never transferred to another view through an assumed
    fixed anatomical vertex identity.

    The full photo observations and optional fixed eye/mouth surface bindings
    select a frontal reference before shape residuals are measured. A broad,
    smoothed lower-face field moves only along that camera's image vertical;
    its amplitude is estimated from all accepted semantic views by default,
    or only the frontal reference when ``frontPrimary`` is enabled. This bounded
    prototype only proposes shortening and cannot repair depth, jaw width,
    camera calibration, or expressions. Original-relative geometric checks
    cannot establish semantic correctness: rerender and redetect every view.
    """
    unchanged = np.array(vertices, copy=True)
    result = {'originalVertices': unchanged, 'allHardFixedVertexIds': np.empty(0, int),
              'candidates': [{'candidateId': 'noop', 'vertices': unchanged.copy(),
                              'diagnostics': {'status': 'unchanged_fallback', 'requiresLiveValidation': False}}]}
    diagnostics = {'method': 'renderedSemanticBroadChinProposals', 'status': 'unchanged',
        'thresholds': dict(_LIMITS), 'candidates': [], 'requiresLiveValidation': True,
        'frontPrimary': bool(frontPrimary), 'fitViewIndices': [], 'diagnosticViewIndices': [],
        'safeStepSearch': bool(safeStepSearch), 'safetySearches': [],
        'semanticSelectionPolicy': ('Fit, rank and gate semantic predictions on the frontal reference only; '
            'other views are observation diagnostics; retain every original-relative geometry constraint'
            if frontPrimary else 'Fit all comparable views with equal total weight; gate every comparable view'),
        'productionQualityApproved': False, 'usesTemplateShapeAttraction': False,
        'limitations': ['Frozen surface bindings only predict detector motion and are not final acceptance evidence.',
            'This prototype proposes broad vertical shortening with frontal camera depth preserved.',
            'Missing or rejected semantic views cannot support quantitative claims.',
            'Render lighting, material and detector crop bias remain possible even for stable bindings.']}
    result['diagnostics'] = diagnostics
    try:
        if not isinstance(maximumCandidates, int) or not 1 <= maximumCandidates <= 3:
            raise ValueError('maximumCandidates must be an integer from one through three')
        original, triangles = _validateMesh(vertices, faces)
        raw_cameras = list(cameras)
        if referenceViewIndex is not None and (isinstance(referenceViewIndex, bool)
                or not isinstance(referenceViewIndex, int) or not 0 <= referenceViewIndex < len(raw_cameras)):
            raise ValueError('Invalid fixed reference view index')
        checked = _validateCameras(raw_cameras)
        fixed = _ids(fixedVertexIds, len(original), 'fixedVertexIds')
        protected = _ids(protectedVertexIds, len(original), 'protectedVertexIds')
        domain = correspondence.get('domain', {})
        fixed = np.union1d(fixed, _ids(domain.get('fixedVertexIds', ()), len(original), 'domainFixed')).astype(int)
        protected = np.union1d(protected, _ids(domain.get('protectedVertexIds', ()), len(original), 'domainProtected')).astype(int)
        if landmarkBinding is None:
            landmarkBinding = domain.get('landmarkBinding')
        binding = _binding(original, landmarkBinding)
        if binding is not None:
            protected = np.union1d(protected, binding['indices'].ravel()).astype(int)
        views, view_reports = _prepare_semantic_views(original, checked, correspondence)
        photo_views, photo_reports = _prepare_views(original, raw_cameras, checked, photoObservations, binding)
        by_index = {view['index']: view for view in views}
        eligible = [v for v in photo_views if v['report']['referenceEligible'] and v['index'] in by_index
                    and _LIMITS['requiredReferenceSemanticId'] in by_index[v['index']]['ids']
                    and np.isin(by_index[v['index']]['ids'], np.arange(6, 11)).sum()
                        >= _LIMITS['minimumReferenceCentralPointCount']]
        if referenceViewIndex is not None:
            eligible = [view for view in eligible if view['index'] == referenceViewIndex]
        diagnostics['requestedReferenceViewIndex'] = referenceViewIndex
        diagnostics.update(semanticViews=view_reports, photoViews=photo_reports,
                           comparableViewIndices=[v['index'] for v in views],
                           weightedSemanticViewIndices=[v['index'] for v in views if 'reliabilityWeights' in v])
        if not eligible or (not frontPrimary and len(checked) > 1 and len(views) < 2):
            diagnostics.update(status='unchanged_insufficient_reliable_semantics',
                reason=('Need a reliable frontal photo with chin ID 8, a second central binding and at least three eligible bindings'
                    if frontPrimary else 'Need a reliable frontal photo with chin ID 8, a second central binding and a second comparable view'))
            if referenceViewIndex is not None:
                diagnostics['reason'] = 'Shared reference lacks reliable jaw support; no other photograph may replace it'
            result['diagnostics'] = _json(diagnostics)
            return result
        reference = max(eligible, key=lambda v: (v['report']['referenceSelectionScore'], -v['index']))
        ref = by_index[reference['index']]
        fit_views = [ref] if frontPrimary else views
        fit_indices = {v['index'] for v in fit_views}
        scale, camera, local = reference['scale'], reference['camera'], reference['points']
        diagnostics.update(referenceViewIndex=reference['index'],
            fitViewIndices=[v['index'] for v in fit_views],
            diagnosticViewIndices=[v['index'] for v in views if v['index'] not in fit_indices],
            referenceSelectionPolicy='Photo reliability/symmetry and eye-mouth pose/registration, independent of jaw residual; center 8 and at least two ids6..10 among at least three eligible bindings')
        pixels, depth = _project(original, camera)
        _, bound_depth = _project(_points(original, ref), camera)
        jaw_depth = float(np.median(bound_depth))
        world_scale = scale*jaw_depth/camera['intrinsics'][1, 1]
        mouth_y = float(local[[48, 54, 57], 1].max())
        if binding is not None:
            mouth = binding['points'][(binding['ids'] >= 48) & (binding['ids'] <= 67)]
            if len(mouth):
                mouth_y = max(mouth_y, float(_project(mouth, camera)[0][:, 1].max()))
        mouth_y += .01*scale
        chin_x, chin_y = local[8]
        if chin_y-mouth_y <= .1*scale:
            raise ValueError('Insufficient lower-face space below protected mouth')
        extra_fixed = np.flatnonzero((pixels[:, 1] <= mouth_y) | (depth >= jaw_depth+.95*world_scale) | (depth <= 0))
        hard_fixed = np.union1d(np.union1d(fixed, protected), extra_fixed).astype(int)
        result['allHardFixedVertexIds'] = hard_fixed
        free = np.setdiff1d(np.arange(len(original)), hard_fixed)
        field = _smooth((pixels[:, 1]-mouth_y)/(chin_y-mouth_y))
        field *= 1-_smooth((pixels[:, 1]-chin_y-.14*scale)/(.45*scale))
        field *= 1-_smooth((np.abs(pixels[:, 0]-chin_x)-.27*scale)/(.42*scale))
        field *= 1-_smooth((depth-jaw_depth-.4*world_scale)/(.55*world_scale))
        field[hard_fixed] = 0.
        if not len(free) or field.max(initial=0) <= 0:
            raise ValueError('No free lower-face vertices')
        laplacian, face_pairs, paired_edges = _topology(original, triangles)
        bending = laplacian.T@laplacian
        original_normals = _normals(original, triangles)
        original_dihedral = _dihedral(original, original_normals, face_pairs, paired_edges)
        baseline_pairs = intersectionPairs(original, triangles)
        region = ((np.abs(pixels[:, 0]-chin_x) < .36*scale) & (pixels[:, 1] > mouth_y)
                  & (pixels[:, 1] < chin_y+.16*scale) & (depth > jaw_depth-.2*world_scale)
                  & (depth < jaw_depth+.4*world_scale))
        region_ids = np.flatnonzero(region)
        nearby = cKDTree(pixels[region_ids]).query_pairs(.04*scale, output_type='ndarray')
        layer_pairs = region_ids[nearby] if len(nearby) else np.empty((0, 2), int)
        if len(layer_pairs):
            differences = np.abs(depth[layer_pairs[:, 0]]-depth[layer_pairs[:, 1]])
            layer_pairs = layer_pairs[(differences > .03*world_scale) & (differences < .4*world_scale)]
        diagnostics.update(faceScalePixels=scale, faceScaleCanonical=world_scale,
            hardFixedVertexCount=len(hard_fixed), protectedVertexCount=len(protected),
            baselineIntersectionPairs=len(baseline_pairs), selfIntersectionChecked=True,
            deformationRegion={'mouthBoundaryPixels': mouth_y, 'observedChinPixels': [chin_x, chin_y],
                'boundJawDepth': jaw_depth, 'coherencePairCount': len(layer_pairs),
                'semanticDomainUsedFor': 'source-binding validation; broad smooth deformation may extend beyond jaw band'},
            candidatePolicy=('Shared smoothing 4/16/64; robust frontal amplitude times .35/.6/.85/1; preserve no-op'
                if frontPrimary else 'Shared smoothing 4/16/64; robust multiview amplitude times .35/.6/.85/1; preserve no-op'))
        if safeStepSearch:
            diagnostics['candidatePolicy'] = ('Shared smoothing 4/16/64; bounded geometry-only search below robust '
                'amplitude and existing cap; fully recheck .35/.6/.85/1 of a witnessed feasible sample; preserve no-op')
        options = dict(fixedVertexIds=hard_fixed, camera=camera,
            maxDisplacement=_LIMITS['maximumDisplacementFaceFraction']*world_scale,
            maxRelativeEdgeChange=_LIMITS['maximumRelativeEdgeChange'],
            minAreaRatio=_LIMITS['minimumAreaRatio'], minNormalCosine=_LIMITS['minimumNormalCosine'])
        accepted = []
        baseline_mean = float(np.mean([
            (v['beforeError'].mean() if 'reliabilityWeights' not in v
             else np.dot(v['reliabilityWeights'], v['beforeError']))/v['scale'] for v in fit_views]))
        for smoothing in (4., 16., 64.):
            system = diags(np.ones(len(original)))+smoothing*bending
            broad = np.zeros(len(original))
            broad[free] = spsolve(system[free][:, free], field[free])
            if not np.isfinite(broad).all():
                continue
            # Projection of this direction is exactly one broad-field pixel
            # per amplitude unit in the de-rolled reference camera.
            direction = (-broad*depth/camera['intrinsics'][1, 1])[:, None]*camera['rotation'][1]
            direction[hard_fixed] = 0.
            fitted_amplitude, fit_report = _fit_amplitude(original, direction, fit_views)
            if fitted_amplitude <= _LIMITS['minimumPositiveAmplitudeFaceFraction']*scale:
                diagnostics['candidates'].append({'smoothing': smoothing, 'status': 'rejected_no_shortening_signal', 'fit': fit_report})
                continue

            def assess_geometry(amplitude, *, include_intersections=False):
                """Original-relative safety only, shared by search and proposals."""
                proposed = original+amplitude*direction
                proposed[hard_fixed] = original[hard_fixed]
                geometry = checkFrontGeometry(original, proposed, triangles, **options)
                evidence = {'status': 'rejected_geometry', 'geometry': geometry}
                if not geometry['accepted']:
                    return proposed, evidence
                normals = _normals(proposed, triangles)
                normal_rotation = np.arccos(np.clip(np.sum(original_normals*normals, axis=1), -1, 1))
                angles = _dihedral(proposed, normals, face_pairs, paired_edges)-original_dihedral
                angles = np.abs(np.arctan2(np.sin(angles), np.cos(angles)))
                active_faces = np.any(proposed != original, axis=1)[triangles].any(1)
                active_edges = active_faces[face_pairs].any(1)
                movement = -amplitude*broad
                layer_gap = float(np.abs(movement[layer_pairs[:, 0]]-movement[layer_pairs[:, 1]]).max(initial=0))
                smoothness = {'maximumNormalRotationRadians': float(normal_rotation.max()),
                    'maximumDihedralChangeRadians': float(angles.max(initial=0)),
                    'globalP95DihedralChangeRadians': float(np.quantile(angles, .95)) if len(angles) else 0.,
                    'activeP95DihedralChangeRadians': float(np.quantile(angles[active_edges], .95)) if active_edges.any() else 0.,
                    'maximumLayerMovementDifferencePixels': layer_gap,
                    'layerPairCount': len(layer_pairs), 'reference': 'original input mesh'}
                smoothness['accepted'] = bool(
                    smoothness['maximumNormalRotationRadians'] <= _LIMITS['maximumNormalRotationRadians']
                    and smoothness['maximumDihedralChangeRadians'] <= _LIMITS['maximumDihedralChangeRadians']
                    and smoothness['globalP95DihedralChangeRadians'] <= _LIMITS['maximumDihedralChangeP95Radians']
                    and layer_gap <= max(_LIMITS['maximumLayerMovementDifferenceFaceFraction']*scale,
                                         _LIMITS['maximumRelativeLayerMovementDifference']*amplitude))
                evidence['smoothness'] = smoothness
                if not smoothness['accepted']:
                    evidence['status'] = 'rejected_smoothness'
                    return proposed, evidence
                if include_intersections:
                    pairs = intersectionPairs(proposed, triangles)
                    added = sorted(pairs-baseline_pairs)
                    evidence.update(newIntersectionPairs=len(added), newFacePairs=added,
                                    finalIntersectionPairs=len(pairs))
                    if added:
                        evidence['status'] = 'rejected_intersections'
                        return proposed, evidence
                evidence['status'] = 'geometry_safe'
                return proposed, evidence

            verified_amplitude = None
            if safeStepSearch:
                verified_amplitude, search = _search_safe_amplitude(fitted_amplitude,
                    _LIMITS['maximumAmplitudeFaceFraction']*scale, assess_geometry)
                search.update(smoothing=smoothing, fit=fit_report)
                diagnostics['safetySearches'].append(search)
                if verified_amplitude is None:
                    diagnostics['candidates'].append({'smoothing': smoothing,
                        'status': 'rejected_no_feasible_geometry_sample', 'fit': fit_report})
                    continue
            for fraction in (.35, .6, .85, 1.):
                amplitude = (verified_amplitude*fraction if safeStepSearch else
                    min(fitted_amplitude*fraction, _LIMITS['maximumAmplitudeFaceFraction']*scale))
                proposed, safety = assess_geometry(amplitude, include_intersections=safeStepSearch)
                row = {'smoothing': smoothing, 'amplitudeFraction': fraction, 'amplitudePixels': amplitude,
                       'fit': fit_report, **safety}
                if safeStepSearch:
                    row.update(amplitudeFractionReference='witnessed feasible search sample',
                        verifiedFeasibleAmplitudePixels=verified_amplitude,
                        fullGeometryRecheckedForCandidate=True)
                diagnostics['candidates'].append(row)
                if safety['status'] != 'geometry_safe':
                    continue
                geometry = safety['geometry']
                layer_gap = safety['smoothness']['maximumLayerMovementDifferencePixels']
                cross = _metrics(proposed, views)
                for metric in cross:
                    metric['usedForSemanticAcceptance'] = metric['viewIndex'] in fit_indices
                row['predictedSemanticViews'] = cross
                fit_metrics = [v for v in cross if v['viewIndex'] in fit_indices]
                if not all(v['accepted'] for v in fit_metrics):
                    row['status'] = ('rejected_frontal_binding_prediction' if frontPrimary
                        else 'rejected_multiview_binding_prediction')
                    continue
                after_mean = float(np.mean([v['afterErrorPixels']/v['faceScalePixels'] for v in fit_metrics]))
                reference_metric = next(v for v in cross if v['viewIndex'] == reference['index'])
                reference_improvement = (reference_metric['beforeErrorPixels']-reference_metric['afterErrorPixels'])/ref['scale']
                improvement = baseline_mean-after_mean
                row.update(predictedMeanBeforeFaceFraction=baseline_mean, predictedMeanAfterFaceFraction=after_mean,
                           predictedImprovementFaceFraction=improvement, predictedReferenceImprovementFaceFraction=reference_improvement)
                if (improvement < _LIMITS['minimumProposalImprovementFaceFraction']
                        or improvement < _LIMITS['minimumProposalRelativeImprovement']*baseline_mean
                        or reference_improvement < _LIMITS['minimumProposalImprovementFaceFraction']):
                    row['status'] = 'rejected_insufficient_binding_prediction_improvement'
                    continue
                if not safeStepSearch:
                    pairs = intersectionPairs(proposed, triangles)
                    added = sorted(pairs-baseline_pairs)
                    row.update(newIntersectionPairs=len(added), newFacePairs=added, finalIntersectionPairs=len(pairs))
                    if added:
                        row['status'] = 'rejected_intersections'
                        continue
                strain = geometry['metrics']['maximumRelativeEdgeChange']
                cost = after_mean**2+.004*strain**2+.2*(layer_gap/scale)**2
                row.update(status='safe_proposal_pending_live_detection', selectionCost=float(cost),
                           requiresLiveValidation=True, changedVertexCount=int(np.any(proposed != original, axis=1).sum()),
                           fixedVerticesExactlyUnchanged=bool(np.array_equal(proposed[hard_fixed], original[hard_fixed])),
                           eyesMouthSupportExactlyUnchanged=bool(np.array_equal(proposed[protected], original[protected])))
                accepted.append((cost, row, proposed))
        # Include the best safe candidate from each smoothing family so the
        # expensive live detector sees shape variation, not near-identical
        # amplitude steps. No detector outputs influence this proposal policy.
        chosen = list(accepted) if allSafeCandidates else []
        if not allSafeCandidates:
            for smoothing in (4., 16., 64.):
                group = [item for item in accepted if item[1]['smoothing'] == smoothing]
                if group:
                    chosen.append(min(group, key=lambda item: item[0]))
        chosen.sort(key=lambda item: item[0])
        selection = chosen if allSafeCandidates else chosen[:maximumCandidates]
        for number, (_, row, proposed) in enumerate(selection, 1):
            candidate_id = f'proposal{number}'
            row['candidateId'] = candidate_id
            result['candidates'].append({'candidateId': candidate_id, 'vertices': proposed,
                                         'diagnostics': _json(row)})
        diagnostics.update(status='proposals_ready_pending_live_validation' if chosen else 'unchanged_no_safe_semantic_proposal',
            safeProposalCount=len(accepted), returnedProposalCount=len(result['candidates'])-1,
            liveProposalSelection=(('all safe members of sampled finite grid' if safeStepSearch else
                'all safe members of existing finite grid') if allSafeCandidates else 'best safe member per smoothing family'),
            selectedCandidateIds=[v['candidateId'] for v in result['candidates'][1:]])
    except (ValueError, KeyError, TypeError, IndexError, np.linalg.LinAlgError) as error:
        diagnostics.update(status='unchanged_invalid_or_unsupported_input', reason=str(error))
    result['diagnostics'] = _json(diagnostics)
    return result
