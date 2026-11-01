"""Contact-aware residual fitting after the completed legacy coarse alignment.

This opt-in module never runs, replaces or restricts the earlier coarse eye
stages. Its input is their complete assembled result. Cameras, topology and
observations stay fixed. Non-skin vertices are fixed by default; an explicit
frozen accessory mapping can make whitelisted rows depend on skin motion.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib

import numpy as np

from .camera import projectPoints
from .clientGeometrySafety import intersectionPairs
from .clientVisibleEyeRefinement import (
    _indices, _policy, _scene, _strictCandidateChecks, buildClientVisibleEyeSupport,
)
from .eyelidTextureSampling import _curves
from .fastFrontEyeGeometry import refineFastFrontEyes
from .geometryRefinement import _validateCameras
from .visibleEyeContours import extractVisibleEyeContours, evaluateApertureMask
from .surfaceContactConstraints import (
    evaluateSurfaceContacts, evaluateSurfaceContactResidual, evaluateSurfaceContactSafety,
)


EYE_NAMES = ('imageLeftEye', 'imageRightEye')


def _identity(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, np.ndarray):
            digest.update(b'array'+item.dtype.str.encode()+repr(item.shape).encode()+item.tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict')
            for key in sorted(item, key=str):
                visit(key); visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode()+str(len(item)).encode())
            for child in item: visit(child)
        elif item is None or isinstance(item, (str, int, float, bool, np.generic)):
            digest.update(type(item).__name__.encode()+repr(item).encode())
        else:
            raise ValueError('Preparation contains unsupported mutable state')
    visit(value)
    return digest.hexdigest()


def _coarseConstraints(constraints, scene):
    if not isinstance(constraints, (list, tuple)) or len(constraints) != 2:
        raise ValueError('Original corresponding coarse constraints are required for both eyes')
    rows = {}
    faces = {tuple(row) for row in np.sort(scene['faces'], axis=1)}
    skin = np.zeros(len(scene['vertices']), bool)
    skin[scene['nicpToFullIds']] = True
    for item in constraints:
        name = item.get('name')
        if name not in EYE_NAMES or name in rows:
            raise ValueError('Expected distinct original image-left and image-right coarse eye groups')
        indices = np.asarray(item['vertexIndices'])
        bary = np.asarray(item['barycentric'], float)
        target = np.asarray(item['targetPixels'], float)
        ids = np.asarray(item['ibugIds'])
        count = len(ids)
        allowed = np.arange(36, 42) if name == EYE_NAMES[0] else np.arange(42, 48)
        weights = np.asarray(item.get('weights', np.ones(count)), float)
        confidence = np.asarray(item.get('confidence', np.ones(count)), float)
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not 4 <= count <= 6
                or len(np.unique(ids)) != count or not np.isin(ids, allowed).all()
                or indices.shape != (count, 3) or indices.dtype.kind not in 'iu'
                or indices.min() < 0 or indices.max() >= len(skin)
                or not skin[indices].all() or bary.shape != indices.shape
                or not np.isfinite(bary).all() or (bary < -1e-7).any()
                or (bary > 1+1e-7).any() or not np.allclose(bary.sum(1), 1., atol=1e-7, rtol=0)
                or target.shape != (count, 2) or not np.isfinite(target).all()
                or weights.shape != (count,) or not np.isfinite(weights).all() or (weights <= 0).any()
                or confidence.shape != (count,) or not np.isfinite(confidence).all()
                or (confidence < 0).any() or (confidence > 1).any()):
            raise ValueError('Invalid original coarse eye binding or target arrays')
        if any(tuple(row) not in faces for row in np.sort(indices, axis=1)):
            raise ValueError('Coarse eye bindings must address actual original skin triangles')
        rows[name] = dict(name=name, vertexIndices=indices.astype(np.int64, copy=True),
            barycentric=bary.copy(), targetPixels=target.copy(), ibugIds=ids.astype(np.int64, copy=True),
            weights=weights.copy(), confidence=confidence.copy())
    return [rows[name] for name in EYE_NAMES]


def _coarseScores(vertices, constraints, camera):
    scores = {}
    for item in constraints:
        points = (vertices[item['vertexIndices']]*item['barycentric'][..., None]).sum(1)
        pixels, depth = projectPoints(points, camera['intrinsics'], camera['rotation'], camera['translation'])
        if not np.isfinite(pixels).all() or not (depth > 0).all():
            raise ValueError('Coarse eye bindings must remain in front of the frozen camera')
        errors = np.linalg.norm(pixels-item['targetPixels'], axis=1)
        scores[item['name']] = dict(count=len(errors), ibugIds=item['ibugIds'].copy(),
            errorsPixels=errors, meanErrorPixels=float(errors.mean()), maximumErrorPixels=float(errors.max()))
    return scores


def evaluateResidualCoarseAlignment(prepared, vertices):
    """Guard the same original coarse targets against the completed baseline."""
    value = np.asarray(vertices, float)
    original = prepared['scene']['vertices']
    if value.shape != original.shape or not np.isfinite(value).all():
        return dict(accepted=False, reasons=['nonFiniteOrWrongShapeCandidate'])
    try:
        scores = _coarseScores(value, prepared['coarseEyeConstraints'], prepared['camera'])
    except ValueError:
        return dict(accepted=False, reasons=['coarseEyeProjectionInvalid'])
    tolerance = .05*min(prepared['imageShape'])/512.
    eyes = {}
    for name, after in scores.items():
        before = prepared['beforeCoarseAlignment'][name]
        accepted = (after['meanErrorPixels'] <= before['meanErrorPixels']+tolerance
            and after['maximumErrorPixels'] <= before['maximumErrorPixels']+tolerance)
        eyes[name] = dict(accepted=bool(accepted), before=deepcopy(before), after=after)
    accepted = all(row['accepted'] for row in eyes.values())
    return dict(accepted=accepted, reasons=[] if accepted else ['completedCoarseEyeAlignmentRegressed'],
        eyes=eyes, tolerancePixels=tolerance,
        reference='same original eligible targets measured on the completed coarse-aligned input',
        metric='per-eye unweighted mean and maximum; neither may regress beyond the original tolerance')


def prepareClientResidualEyes(scene, camera, photoEyes, *, coarseEyeConstraints,
                              imageShape, cornerFaceIds, fixedVertexIds=(),
                              protectedVertexIds=(), originalPixelScale=1., policy=None):
    """Freeze baseline aperture support without changing any coarse geometry.

    ``coarseEyeConstraints`` are the original eligible triangle bindings mapped
    to full-scene vertex IDs, with original iBUG IDs and photo targets. Every
    non-NICP vertex, including the explicit canthi, is fixed. The returned
    preparation can be used to construct a contact specification from the
    same baseline and movable support before calling ``refineClientResidualEyes``.
    """
    effective = _policy(policy)
    if (len(imageShape) != 2 or any(type(value) not in (int, np.int32, np.int64) or value < 2 for value in imageShape)
            or isinstance(originalPixelScale, (bool, np.bool_)) or not np.isfinite(originalPixelScale) or originalPixelScale <= 0):
        raise ValueError('Expected a positive image shape and original pixel scale')
    if camera.get('valid', True) is not True:
        raise ValueError('Cannot refine with a rejected camera')
    checked, corners = _scene(scene, cornerFaceIds)
    camera = _validateCameras([camera])[0]
    photos = {name: {part: value.copy() for part, value in eye.items()}
        for name, eye in _curves(photoEyes, imageShape, 'photoEyes').items()}
    if any(len(curve) < 4 for eye in photos.values() for curve in eye.values()):
        raise ValueError('Complete visible apertures need at least four points per lid')
    coarse = _coarseConstraints(coarseEyeConstraints, checked)
    for item in coarse:
        if (item['targetPixels'] < 0).any() or (item['targetPixels'] >= np.asarray(imageShape[::-1])).any():
            raise ValueError('Coarse eye targets must lie in the original working photo')
    fixed = _indices(fixedVertexIds, len(checked['vertices']), 'fixedVertexIds')
    protected = _indices(protectedVertexIds, len(checked['vertices']), 'protectedVertexIds')
    nonSkin = np.ones(len(checked['vertices']), bool)
    nonSkin[checked['nicpToFullIds']] = False
    fixed = np.union1d(fixed, np.flatnonzero(nonSkin))
    baseline = extractVisibleEyeContours(**checked, camera=camera, photoEyes=photos, imageShape=imageShape,
        apertureAccessoryMode='innerCornerHeadAccessories', apertureAccessoryFaceIds=corners,
        minimumEyeRasterWidth=256, maximumRoiSize=768)
    support = buildClientVisibleEyeSupport(checked, camera, photos, baseline, cornerFaceIds=corners,
        fixedVertexIds=fixed, protectedVertexIds=protected, policy=effective)
    if support['movableMask'][nonSkin].any():
        raise RuntimeError('Residual refinement must fix every non-skin accessory')
    measurementFailure = None
    try:
        apertureMetrics = evaluateApertureMask(baseline, photos, originalPixelScale=originalPixelScale)
    except ValueError as error:
        apertureMetrics = None
        measurementFailure = str(error)
    prepared = dict(schemaVersion='completedCoarseEyeResidualV1', scene=checked, camera=camera,
        photoEyes=photos, imageShape=tuple(imageShape), cornerFaceIds=corners,
        originalPixelScale=float(originalPixelScale), policy=effective, support=support,
        coarseEyeConstraints=coarse, beforeCoarseAlignment=_coarseScores(checked['vertices'], coarse, camera),
        beforeContours=baseline, beforeApertureMetrics=apertureMetrics,
        baselineVerticesSha256=hashlib.sha256(checked['vertices'].tobytes()).hexdigest(),
        diagnostics=dict(coarseStagesRerun=False, coarseStagesSkipped=False, coarseStagesRestricted=False,
            allNonSkinFixed=True, supportFrozenAtCompletedBaseline=True,
            apertureMeasurementFailure=measurementFailure,
            productionDefaultsChanged=False))
    prepared['preparationSha256'] = _identity(prepared)
    return prepared


def _validateContactSpec(prepared, contactSpec):
    scene, support = prepared['scene'], prepared['support']
    if (not np.array_equal(contactSpec['baselineVertices'], scene['vertices'])
            or not np.array_equal(contactSpec['faces'], scene['faces'])):
        raise ValueError('Contact spec must use the exact completed full-scene baseline')
    if not np.isfinite(contactSpec['weight']) or contactSpec['weight'] <= 0:
        raise ValueError('Contact optimization requires a positive residual weight')
    skin = np.zeros(len(scene['vertices']), bool); skin[scene['nicpToFullIds']] = True
    for group in contactSpec['contactGroups']:
        source = _indices(group['sourceVertexIds'], len(skin), 'contact source vertices', nonempty=True)
        targets = _indices(group['targetFaceIds'], len(scene['faces']), 'contact target faces', nonempty=True)
        corners = scene['faces'][targets]
        if not (skin[source] & support['movableMask'][source]).all():
            raise ValueError('Contact sources must use the frozen movable skin domain')
        if skin[corners].any() or not support['fixedMask'][corners].all():
            raise ValueError('Contact targets must be the unchanged non-skin accessories')
        if any(scene['materials'][int(mid)].get('alphaMode', 'OPAQUE') != 'OPAQUE'
                for mid in np.unique(scene['materialIds'][targets])):
            raise ValueError('Alpha cards are not oriented solid contact targets')
    baseline = evaluateSurfaceContactSafety(scene['vertices'], contactSpec)
    if not baseline['accepted']:
        raise ValueError('The completed baseline must satisfy its frozen contact envelope')
    return baseline


def _sameRois(before, after):
    return (set(before['eyes']) == set(after['eyes'])
        and all('roi' in before['eyes'][name] and 'roi' in after['eyes'][name]
            and before['eyes'][name]['roi'] == after['eyes'][name]['roi']
            and before['eyes'][name]['roiMasks']['eye'].shape == after['eyes'][name]['roiMasks']['eye'].shape
            for name in before['eyes']))


def refineClientResidualEyes(prepared, contactSpec=None, *, candidateGate=None,
                             deformationMode='compact', smoothingRadiusFraction=.1,
                             captureSolverEvidence=False, accessoryFollowingSpec=None):
    """Fit only the visible-aperture residual, with contact inside optimization.

    ``None`` is an explicit no-contact ablation. A supplied data-only contact
    spec must have been built from this preparation's exact baseline and skin
    support; no correspondences or orientation are reestimated while fitting.
    Raw proposals are retained even when strict gates return the baseline.
    An optional external ``candidateGate(fullVertices)`` adds the caller's
    original side-view/mouth/cumulative checks at every outer proposal and the
    final result. It receives a read-only copy and cannot waive internal gates.
    ``accessoryFollowingSpec`` optionally releases only its frozen whitelist
    from the assembled fixed domain. The original skin support, local fixed
    rows, observations and solver basis are retained; follower motion is fully
    determined by skin displacement relative to this completed baseline.
    """
    if (prepared.get('schemaVersion') != 'completedCoarseEyeResidualV1'
            or prepared.get('preparationSha256') != _identity({key: value for key, value in prepared.items() if key != 'preparationSha256'})):
        raise ValueError('Prepared baseline, camera, targets or support changed')
    if candidateGate is not None and not callable(candidateGate):
        raise ValueError('candidateGate must be callable or None')
    if deformationMode not in ('compact', 'fixedBoundaryBiharmonic') or type(captureSolverEvidence) is not bool:
        raise ValueError('Invalid residual deformation mode or solver evidence flag')
    if (not isinstance(smoothingRadiusFraction, (int, float, np.integer, np.floating))
            or isinstance(smoothingRadiusFraction, (bool, np.bool_))
            or not np.isfinite(smoothingRadiusFraction) or not 0 < smoothingRadiusFraction <= 1):
        raise ValueError('smoothingRadiusFraction must be a real scalar in (0, 1]')
    scene, support, camera = prepared['scene'], prepared['support'], prepared['camera']
    original, baseline = scene['vertices'], prepared['beforeContours']
    following = deepcopy(accessoryFollowingSpec) if accessoryFollowingSpec is not None else None
    strictSupport = support
    if following is not None:
        from .eyeAccessoryFollowing import validateEyeAccessoryFollowingSpec
        validateEyeAccessoryFollowingSpec(following, baselineVertices=original,
            faces=scene['faces'], skinVertexIds=scene['nicpToFullIds'],
            originalFixedVertexIds=support['fixedVertexIds'])
        if not np.array_equal(np.sort(following['movableSkinVertexIds']),
                np.flatnonzero(support['movableMask'])):
            raise ValueError('Accessory following must use the exact frozen movable skin support')
        # The optimizer still receives the unmodified support. Only physical
        # full-scene safety recognizes these deterministically driven rows.
        strictSupport = deepcopy(support)
        strictSupport['fixedVertexIds'] = np.asarray(following['effectiveFixedVertexIds'], np.int64).copy()
        strictSupport['fixedMask'] = np.zeros(len(original), bool)
        strictSupport['fixedMask'][strictSupport['fixedVertexIds']] = True
        strictSupport['movableMask'] = ~strictSupport['fixedMask']
    contact = deepcopy(contactSpec) if contactSpec is not None else None
    initialContact = _validateContactSpec(prepared, contact) if contact is not None else None
    skip = None
    if support['diagnostics']['status'] != 'supported' or not support['movableMask'].any():
        skip = 'skippedInsufficientVisibleSupport'
    elif prepared['beforeApertureMetrics'] is None:
        skip = 'skippedUnmeasurableCompleteAperture'
    elif contact is not None and initialContact['noActiveContacts']:
        skip = 'skipNoActiveContactRows'
    if skip is not None:
        output = dict(vertices=original.copy(), rawCandidateVertices=original.copy(), solverVertices=original.copy(),
            candidateProposals=[], support=deepcopy(support), beforeContours=baseline, afterContours=baseline,
            rawCandidateContours=baseline, solverContours=baseline, textureContours=None, contactSpec=contact,
            coarseEyeConstraints=deepcopy(prepared['coarseEyeConstraints']),
            beforeApertureMetrics=prepared['beforeApertureMetrics'], solverApertureMetrics=prepared['beforeApertureMetrics'],
            contactBefore=evaluateSurfaceContacts(original, contact, False) if contact is not None else None,
            contactSolver=evaluateSurfaceContacts(original, contact, False) if contact is not None else None,
            diagnostics=dict(status=skip, accepted=False, geometryChanged=False, fastSolverInvoked=False,
                contactOptimizationEnabled=contact is not None, initialContact=initialContact,
                contactEvaluationCalls=dict(residual=0, jacobian=0, safety=0), proposalCount=0,
                externalCandidateGateUsed=candidateGate is not None, rawCandidateChanged=False,
                completeApertureRoisExactlyPreserved=True, fixedDomainExactlyPreserved=True,
                coarseStagesRerun=False, coarseStagesSkipped=False, coarseStagesRestricted=False,
                apertureMeasurementFailure=prepared['diagnostics']['apertureMeasurementFailure'],
                allNonSkinExactlyFixed=True, clientQualityApproved=False))
        if captureSolverEvidence:
            output['solverEvidence'] = []
        if deformationMode != 'compact':
            output['diagnostics']['deformationMode'] = deformationMode
        if following is not None:
            output['accessoryFollowingSpec'] = following
            output['diagnostics'].update(accessoryFollowingEnabled=True,
                accessoryFollowingSpecSha256=following['specSha256'], accessoryDegreesOfFreedom=0,
                effectiveFixedVertexIds=strictSupport['fixedVertexIds'].copy(),
                effectiveFixedDomainExactlyPreserved=True, frozenSolverSupportPreserved=True)
        return output
    baselinePairs = intersectionPairs(original, scene['faces'])
    proposals = []
    calls = dict(residual=0, jacobian=0, safety=0)
    def extraResidual(vertices, withJacobian):
        calls['residual'] += 1
        calls['jacobian'] += int(withJacobian)
        return evaluateSurfaceContactResidual(vertices, contact, withJacobian=withJacobian)
    def extraSafety(vertices):
        calls['safety'] += 1
        coarse = evaluateResidualCoarseAlignment(prepared, vertices)
        contactGate = evaluateSurfaceContactSafety(vertices, contact) if contact is not None else None
        return dict(accepted=bool(coarse['accepted'] and (contactGate is None or contactGate['accepted'])),
            coarseAlignment=coarse, contact=contactGate)
    def strictChecks(vertices):
        strict = _strictCandidateChecks(original, vertices, scene, camera, strictSupport, baselinePairs)
        extra = extraSafety(vertices)
        strict['coarseAlignment'] = extra['coarseAlignment']
        strict['contact'] = extra['contact']
        if not extra['coarseAlignment']['accepted']:
            strict['reasons'].append('completedCoarseEyeAlignmentRegressed')
        if extra['contact'] is not None and not extra['contact']['accepted']:
            strict['reasons'].append('frozenContactEnvelopeRegressed')
        strict['accepted'] = bool(strict['accepted'] and extra['accepted'])
        if candidateGate is not None:
            snapshot = np.asarray(vertices).copy()
            snapshot.setflags(write=False)
            external = candidateGate(snapshot)
            if not isinstance(external, dict) or type(external.get('accepted')) not in (bool, np.bool_):
                raise ValueError('candidateGate must return an explicit boolean accepted field')
            strict['externalCandidateGate'] = deepcopy(external)
            strict['accepted'] = bool(strict['accepted'] and external['accepted'])
            if not external['accepted']:
                strict['reasons'].append('externalCandidateGateRejected')
        return strict
    def gateProposal(vertices, context):
        strict = strictChecks(vertices)
        proposals.append(dict(vertices=np.asarray(vertices).copy(), round=context['round'], scale=context['scale'],
            internalGeometry=deepcopy(context['geometry']), strict=deepcopy(strict),
            contactEvaluation=evaluateSurfaceContacts(vertices, contact, False) if contact is not None else None))
        return strict
    options = dict(additionalSafety=extraSafety, omitFullyFixedBoundaryConstraints=True,
        deformationMode=deformationMode, smoothingRadiusFraction=smoothingRadiusFraction,
        captureSolverEvidence=captureSolverEvidence)
    if contact is not None:
        options['additionalResidual'] = extraResidual
    if following is not None:
        options['accessoryFollowingSpec'] = following
    fitted = refineFastFrontEyes(original, scene['faces'], scene['materialIds'], camera, prepared['photoEyes'],
        skinVertexIds=scene['nicpToFullIds'], cornerFaceIds=prepared['cornerFaceIds'],
        fixedVertexIds=support['fixedVertexIds'], imageShape=prepared['imageShape'],
        uvs=scene['uvs'], uvFaces=scene['uvFaces'], materials=scene['materials'], alphaTextures=scene['alphaTextures'],
        originalPixelScale=prepared['originalPixelScale'], rounds=prepared['policy']['rounds'],
        iterations=prepared['policy']['iterationsPerRound'], candidateGate=gateProposal, **options)
    solver = np.asarray(fitted['vertices']).copy()
    raw = proposals[-1]['vertices'].copy() if proposals else solver.copy()
    strict = strictChecks(solver)
    after = fitted['afterContours']
    visibleAccepted = (set(after.get('eyes', {})) == set(prepared['photoEyes'])
        and all(eye.get('status') == 'accepted' for eye in after['eyes'].values()))
    sameRois = _sameRois(baseline, after)
    if not visibleAccepted:
        strict['reasons'].append('unreliableCompleteCandidateContours')
    if not sameRois:
        strict['reasons'].append('completeApertureRoiChanged')
    strict['accepted'] = bool(strict['accepted'] and visibleAccepted and sameRois)
    changed = not np.array_equal(solver, original)
    accepted = bool(strict['accepted'] and changed and fitted['diagnostics']['status'] == 'acceptedLocalExperiment')
    result = solver.copy() if accepted else original.copy()
    for proposal in proposals:
        stage = next((row for row in fitted['diagnostics'].get('stages', []) if row['round'] == proposal['round']), {})
        proposal['solverAttempt'] = deepcopy(next((row for row in stage.get('attempts', []) if row['scale'] == proposal['scale']), {}))
        proposal['selectedByFastRound'] = stage.get('status') == 'accepted' and stage.get('selectedScale') == proposal['scale']
    nonSkin = np.ones(len(original), bool); nonSkin[scene['nicpToFullIds']] = False
    output = dict(vertices=result, rawCandidateVertices=raw, solverVertices=solver,
        candidateProposals=proposals, support=deepcopy(support), contactSpec=contact,
        coarseEyeConstraints=deepcopy(prepared['coarseEyeConstraints']),
        beforeContours=baseline, afterContours=after if accepted else baseline, solverContours=after,
        rawCandidateContours=after if np.array_equal(raw, solver) else None,
        textureContours=fitted.get('textureContours') if accepted else None,
        beforeApertureMetrics=prepared['beforeApertureMetrics'], solverApertureMetrics=fitted['diagnostics']['after'],
        contactBefore=evaluateSurfaceContacts(original, contact, False) if contact is not None else None,
        contactSolver=evaluateSurfaceContacts(solver, contact, False) if contact is not None else None,
        diagnostics=dict(status='acceptedResidualExperiment' if accepted else ('rejectedStrictCandidate' if changed else 'noImprovingSafeCandidate'),
            accepted=accepted, geometryChanged=accepted, fastSolverInvoked=True, policy=prepared['policy'],
            preparationSha256=prepared['preparationSha256'], fast=fitted['diagnostics'], strict=strict,
            contactOptimizationEnabled=contact is not None, contactEvaluationCalls=calls,
            externalCandidateGateUsed=candidateGate is not None,
            completeApertureRoisExactlyPreserved=sameRois, proposalCount=len(proposals),
            rawCandidateChanged=not np.array_equal(raw, original),
            allNonSkinExactlyFixed=bool(np.array_equal(result[nonSkin], original[nonSkin])),
            fixedDomainExactlyPreserved=bool(np.array_equal(result[support['fixedMask']], original[support['fixedMask']])),
            coarseStagesRerun=False, coarseStagesSkipped=False, coarseStagesRestricted=False,
            cameraRefitted=False, eyeBallsRefitted=False, gazeRefitted=False, accessoriesReassembled=False,
            topologyOrUvsChanged=False, photoObservationsReestimated=False, clientQualityApproved=False,
            limitations=['Automatic photo curves and six-point targets are proxies, not anatomical ground truth.',
                'The contact objective uses frozen oriented local features and is not a full collision certificate.',
                'Every old geometry, complete-aperture and strict full-scene collision gate remains mandatory.',
                'Other-view support and upstream whole-pipeline geometry gates still require independent validation.']))
    if captureSolverEvidence:
        output['solverEvidence'] = fitted.get('solverEvidence', [])
    if deformationMode != 'compact':
        output['diagnostics']['deformationMode'] = deformationMode
        output['diagnostics']['smoothingRadiusFraction'] = float(smoothingRadiusFraction)
    if following is not None:
        output['accessoryFollowingSpec'] = following
        output['diagnostics'].update(accessoryFollowingEnabled=True,
            accessoryFollowingSpecSha256=following['specSha256'], accessoryDegreesOfFreedom=0,
            effectiveFixedVertexIds=strictSupport['fixedVertexIds'].copy(),
            effectiveFixedDomainExactlyPreserved=bool(np.array_equal(
                result[strictSupport['fixedMask']], original[strictSupport['fixedMask']])),
            frozenSolverSupportPreserved=True,
            originalAllNonSkinFixedPolicyActive=False,
            fullAssemblyRelativeToGlobalCompletedBaseline=True)
    return output
