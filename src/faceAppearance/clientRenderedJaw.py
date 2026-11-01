"""Per-request frontal rendered-semantic jaw correction before texture baking.

The frozen frontWeightedSafeV1 policy is applied to the current request's
photographs, cameras and already refined client surface. Original semantic
IDs and reliability weights remain fixed during candidate validation. Other
angles are saved as observations and never veto or rank a semantic candidate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .clientLandmarks import ibugClientLandmarks, loadClientLandmarks
from .frontalSemanticReliability import freezeFrontalReliability
from .geometryRefinement import _validateCameras, _validateMesh
from .renderedJawCorrespondence import prepareRenderedJawCorrespondence, POLICY as BINDING_POLICY
from .renderedJawRefinement import proposeRenderedJawCandidates
from .renderedJawValidation import live_check, live_policy


METHOD = 'frontWeightedSafeV1'


def _serial(value):
    """Diagnostic absence is JSON null; finite numeric evidence stays exact."""
    if isinstance(value, np.ndarray):
        return _serial(value.tolist())
    if isinstance(value, np.generic):
        return _serial(value.item())
    if isinstance(value, dict):
        return {str(k): _serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serial(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write(path, value):
    Path(path).write_text(json.dumps(_serial(value), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def _inputQuality(cameras, inputs):
    """Reject unprocessable numerical observations before renderer invocation."""
    reports = []
    for index, (camera, record) in enumerate(zip(cameras, inputs)):
        reasons = []
        try:
            _validateCameras([camera])
            if not bool(camera.get('valid', True)):
                reasons.append('camera_not_accepted_by_reconstruction')
            matrix = np.asarray(record['inputToWorking'], float)
            if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                    or abs(np.linalg.det(matrix)) < 1e-12):
                reasons.append('invalid_input_to_working_transform')
            for field in ('orientedSize', 'workingSize'):
                size = np.asarray(record[field], float)
                if (size.shape != (2,) or not np.isfinite(size).all()
                        or (size <= 0).any() or not np.equal(size, np.round(size)).all()):
                    reasons.append(f'invalid_{field}')
            box = np.asarray(record['faceDetection']['boxXyxy'], float)
            if (box.shape != (4,) or not np.isfinite(box).all()
                    or (box[2:] <= box[:2]).any()):
                reasons.append('invalid_saved_photo_detection_box')
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
            reasons.append(f'unprocessable_camera_or_input: {exc}')
        reports.append({'viewIndex': index, 'valid': not reasons, 'reasons': reasons})
    return reports


def refineRenderedClientJaw(assets, vertices, faces, templateToUff, cameras, inputs,
                           photoPaths, *, projectRoot, outputDirectory,
                           referenceViewIndex=None, photoObservations=None):
    """Return ``vertices`` and diagnostics using the qualified frontal policy.

    Malformed topology, asset identity, source identity and implementation
    errors propagate. Unreliable numerical input or semantic evidence returns
    an exact copy of the incoming geometry with an explicit fallback reason.
    All geometry safety limits use this stage's original input, not the prior
    candidate. This stage never changes texture, camera, UV or accessory data.
    A supplied ``referenceViewIndex`` pins the caller's shared photo reference;
    insufficient jaw support returns unchanged geometry instead of switching.
    ``photoObservations`` may reuse identity-checked current-request FAN data.
    """
    start = perf_counter()
    output = Path(outputDirectory)
    output.mkdir(parents=True, exist_ok=False)
    project = Path(projectRoot).resolve()
    original = np.array(vertices, copy=True)
    validated, triangles = _validateMesh(original, faces)
    arrays = assets.arrays
    expected = np.asarray(arrays['nicpFaces'])
    full_ids = np.asarray(arrays['nicpToFullIds'])
    if not np.array_equal(np.asarray(faces), expected):
        raise ValueError('Rendered jaw correction requires the exact client NICP topology')
    if (full_ids.ndim != 1 or full_ids.dtype.kind not in 'iu'
            or len(full_ids) != len(original) or len(np.unique(full_ids)) != len(full_ids)
            or (full_ids < 0).any() or (full_ids >= len(arrays['vertices'])).any()):
        raise ValueError('Client NICP-to-full identity mapping is invalid')
    template_vertices = np.asarray(arrays['vertices'])[full_ids].copy()
    _validateMesh(template_vertices, expected)
    static = np.asarray(arrays['staticMask'])
    if static.shape != (len(original),) or static.dtype.kind != 'b':
        raise ValueError('Client static mask must be a boolean value per ordered NICP vertex')
    fixed = np.flatnonzero(static)
    full_binding = loadClientLandmarks(triangles, path=project / 'assets/clientLandmarks'
        / 'legacyMediapipe468/template_tri_lms_468_bicentroid.json')
    if not full_binding['isOriginalBinding']:
        raise ValueError('Rendered jaw correction requires the audited landmark source')
    binding = ibugClientLandmarks(full_binding, groups=('eyes', 'mouth'))
    protected = np.unique(binding['vertexIndices'])
    rotation = np.asarray(templateToUff['rotation'], float)
    translation = np.asarray(templateToUff['translation'], float)
    transform_scale = float(templateToUff['scale'])
    if (rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.isfinite(transform_scale) or transform_scale <= 0
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
            or np.linalg.det(rotation) <= 0):
        raise ValueError('Rendered jaw correction requires a proper template-to-UFF similarity')
    count = len(inputs)
    if not count or len(cameras) != count or len(photoPaths) != count:
        raise ValueError('Photographs, saved input records and cameras must have identical view order')
    if referenceViewIndex is not None and (isinstance(referenceViewIndex, bool)
            or not isinstance(referenceViewIndex, int) or not 0 <= referenceViewIndex < count):
        raise ValueError('Invalid fixed reference view index')

    stage_seconds = {}
    report = {'method': METHOD, 'status': 'running', 'selectedCandidateId': 'noop',
        'frontPrimary': True, 'reliabilityWeighted': True, 'safeStepSearch': True,
        'maximumCandidates': 3, 'livePolicy': live_policy(True, True),
        'bindingPolicy': dict(BINDING_POLICY), 'referenceViewIndex': None,
        'fitViewIndices': [], 'diagnosticViewIndices': list(range(count)),
        'viewCount': count, 'stageSeconds': stage_seconds, 'candidateChecks': [],
        'requestedReferenceViewIndex': referenceViewIndex,
        'sourceLandmarkSha256': full_binding['sourceSha256'],
        'artifactDirectory': str(output.resolve()),
        'limitations': [
            'Broad vertical shortening only; this does not recover jaw depth, width or expression.',
            'Automatic landmarks and anatomical exclusion are proxies, not ground truth.',
            'Only frontal semantics affect candidate acceptance; other angles are observational.',
            'Self-intersection checks exclude adjacent/shared-vertex triangles and do not certify the deformation path.',
            'Full accessory collision and animation behavior are not qualified by this stage.']}
    implementation = ('clientRenderedJaw.py', 'renderedJawRuntime.py', 'renderedJawValidation.py',
        'renderedJawCorrespondence.py', 'renderedJawRefinement.py', 'frontalSemanticReliability.py',
        'adaptiveChinRefinement.py', 'clientGeometrySafety.py', 'frontLandmarkRefinement.py',
        'surfaceVisibility.py')
    report['implementationHashes'] = {name: hashlib.sha256(
        Path(__file__).with_name(name).read_bytes()).hexdigest() for name in implementation}
    _write(output / 'inputs.json', inputs)
    _write(output / 'cameras.json', cameras)
    _write(output / 'templateToUff.json', templateToUff)
    np.savez_compressed(output / 'sourceState.npz', vertices=original, faces=triangles,
        templateVertices=template_vertices, nicpToFullIds=full_ids,
        fixedVertexIds=fixed, protectedVertexIds=protected,
        **{f'landmark_{k}': v for k, v in binding.items() if isinstance(v, np.ndarray)})

    def finish(chosen, *, hard_fixed=(), selected_check=None):
        all_fixed = np.union1d(np.union1d(fixed, protected), np.asarray(hard_fixed, int))
        if not np.array_equal(original[all_fixed], chosen[all_fixed]):
            raise RuntimeError('Rendered jaw correction changed fixed eye, mouth or interface geometry')
        report.update(changedVertexCount=int(np.any(chosen != original, axis=1).sum()),
            protectedVerticesExactlyUnchanged=True,
            noopExactlyUnchanged=bool(np.array_equal(chosen, original)) if report['selectedCandidateId'] == 'noop' else None,
            selectedLiveCheck=selected_check, seconds=perf_counter() - start)
        if report['selectedCandidateId'] == 'noop' and not np.array_equal(chosen, original):
            raise RuntimeError('Rendered jaw fallback must retain the exact original geometry')
        np.savez_compressed(output / 'geometry.npz', before=original, after=chosen,
            faces=triangles, fixedVertexIds=fixed, protectedVertexIds=protected,
            allHardFixedVertexIds=all_fixed)
        report['stageSeconds'] = stage_seconds
        _write(output / 'report.json', report)
        return {'vertices': np.array(chosen, copy=True), 'diagnostics': _serial(report)}

    report['inputQuality'] = _inputQuality(cameras, inputs)
    if count > 16 or any(not row['valid'] for row in report['inputQuality']):
        report.update(status='unchanged_unprocessable_observations',
            reason='At most 16 numerically valid camera/input records are supported without dropping or reordering views')
        return finish(original.copy())

    # Import the model-dependent renderer only after identity/numerical checks.
    from .renderedJawRuntime import observePhotos, buildScene, renderDetect
    tick = perf_counter()
    if photoObservations is None:
        observations = observePhotos(project, photoPaths, inputs)
    else:
        from .clientReferenceView import validateReferenceObservationIdentity
        validateReferenceObservationIdentity(photoObservations, inputs)
        observations = photoObservations
    report['photoObservationsReused'] = photoObservations is not None
    stage_seconds['photoObservation'] = perf_counter() - tick
    report['photoObservations'] = observations.get('diagnostics', {})
    np.savez_compressed(output / 'photoObservations.npz',
        **{k: v for k, v in observations.items() if isinstance(v, np.ndarray)})
    _write(output / 'photoObservations.json', report['photoObservations'])
    base_scene = buildScene(assets, validated, templateToUff)
    report['renderScene'] = base_scene.get('provenance', {})
    _write(output / 'renderScene.json', report['renderScene'])
    np.savez_compressed(output / 'renderScene.npz',
        **{k: v for k, v in base_scene.items() if isinstance(v, np.ndarray)})
    tick = perf_counter()
    before, baseline_timing = renderDetect(base_scene, validated, cameras, inputs,
        observations, output / 'baseline', projectRoot=project)
    stage_seconds['baselineRenderAndDetection'] = perf_counter() - tick
    report['baselineTiming'] = baseline_timing

    def correspond(current, detections, domain=None):
        # Missing render observations cannot supply a ray. Keep an explicit
        # empty view instead of sending NaNs to the surface index or dropping
        # required frontal IDs from live validation.
        usable, unavailable = [], []
        semantic_ids = np.asarray(BINDING_POLICY['semanticIds'], int)
        for row in detections:
            index = int(row['viewIndex'])
            scale = float(observations['faceScaleWorkingPixels'][index])
            finite = np.isfinite(scale) and scale > 0
            for name in BINDING_POLICY['requiredAppearances']:
                points = np.asarray(row['variants'][name]['points'])
                if points.shape != (68, 2):
                    raise ValueError('Rendered FAN observations must have 68 ordered 2D points')
                finite = finite and np.isfinite(points[semantic_ids]).all()
            (usable if finite else unavailable).append(row)
        result = prepareRenderedJawCorrespondence(current, triangles, template_vertices,
            cameras, observations, usable,
            imageShapes=[record['workingSize'][::-1] for record in inputs],
            fixedVertexIds=fixed, protectedVertexIds=protected, domain=domain)
        for row in unavailable:
            index = int(row['viewIndex'])
            result['views'].append({'viewIndex': index, 'eligible': False,
                'semanticIds': np.empty(0, int), 'vertexIndices': np.empty((0, 3), int),
                'barycentricWeights': np.empty((0, 3)), 'targetPixels': np.empty((0, 2)),
                'sourcePixels': np.empty((0, 2)), 'confidence': np.empty(0),
                'faceScalePixels': float(observations['faceScaleWorkingPixels'][index]),
                'allSemanticIds': semantic_ids, 'allAcceptedMask': np.zeros(len(semantic_ids), bool)})
            result['diagnostics'].append({'viewIndex': index, 'eligible': False,
                'acceptedCount': 0, 'acceptedSemanticIds': [],
                'landmarks': [{'semanticId': int(sid), 'accepted': False,
                    'reasons': ['unavailable_or_nonfinite_photo_or_render_observation'],
                    'appearanceDriftFaceFraction': np.inf,
                    'lightingDriftFaceFraction': np.inf} for sid in semantic_ids]})
        result['views'].sort(key=lambda v: v['viewIndex'])
        result['diagnostics'].sort(key=lambda v: v['viewIndex'])
        return result

    bindings = freezeFrontalReliability(correspond(validated, before), observations, before)
    _write(output / 'baselineCorrespondence.json', bindings)
    tick = perf_counter()
    proposals = proposeRenderedJawCandidates(validated, triangles, cameras, observations, bindings,
        fixedVertexIds=fixed, protectedVertexIds=protected, landmarkBinding=binding,
        maximumCandidates=3, allSafeCandidates=False, frontPrimary=True, safeStepSearch=True,
        referenceViewIndex=referenceViewIndex)
    stage_seconds['candidateProposal'] = perf_counter() - tick
    solver = proposals['diagnostics']
    if solver['status'] == 'unchanged_invalid_or_unsupported_input':
        # Numerical observation problems were handled explicitly above. A
        # caught solver schema/programming error must fail the job visibly.
        unsupported_shape = {'Insufficient lower-face space below protected mouth',
                             'No free lower-face vertices'}
        if solver.get('reason') not in unsupported_shape:
            raise RuntimeError(f"Rendered jaw solver rejected validated input: {solver.get('reason')}")
    _write(output / 'proposalReport.json', solver)
    np.savez_compressed(output / 'proposals.npz',
        **{c['candidateId']: c['vertices'] for c in proposals['candidates']})
    reference = solver.get('referenceViewIndex')
    comparable = solver.get('comparableViewIndices', [])
    report.update(solver=solver, referenceViewIndex=reference,
        comparableViewIndices=comparable, fitViewIndices=solver.get('fitViewIndices', []),
        diagnosticViewIndices=[i for i in range(count) if i != reference],
        frontReliability=next((v.get('reliability') for v in bindings['views']
                               if v['viewIndex'] == reference), None))
    survivors = []
    tick = perf_counter()
    for candidate in proposals['candidates']:
        name = candidate['candidateId']
        if name == 'noop':
            continue
        current_dir = output / 'candidates' / name
        rendered, timing = renderDetect(base_scene, candidate['vertices'], cameras, inputs,
            observations, current_dir, projectRoot=project)
        after = correspond(candidate['vertices'], rendered, domain=bindings['domain'])
        _write(current_dir / 'correspondence.json', after)
        check = live_check(before, rendered, bindings, after, observations, inputs,
            reference, comparable, front_primary=True, reliability_weighted=True)
        check.update(candidateId=name, proposal=candidate['diagnostics'], **timing)
        _write(current_dir / 'liveCheck.json', check)
        report['candidateChecks'].append(check)
        if check['accepted']:
            survivors.append((check['selectionCost'], candidate, check))
        _write(output / 'candidateChecks.json', report['candidateChecks'])
    stage_seconds['candidateRenderAndAcceptance'] = perf_counter() - tick
    if survivors:
        _, winner, check = min(survivors, key=lambda item: item[0])
        report.update(status='accepted_live_semantics', selectedCandidateId=winner['candidateId'])
        return finish(winner['vertices'], hard_fixed=proposals['allHardFixedVertexIds'], selected_check=check)
    report.update(status='unchanged_no_live_candidate', reason=solver.get('reason',
        'No proposed geometry passed the frozen frontal semantic and geometry requirements'))
    return finish(original.copy(), hard_fixed=proposals['allHardFixedVertexIds'])
