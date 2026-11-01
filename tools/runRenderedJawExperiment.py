"""Fit frozen client jaws with rendered semantics, then rerender to accept.

Fixed barycentric loss only proposes shapes. Final decisions use a frozen set
of semantic IDs redetected on three appearances and the original photo targets.
This is a local experiment and does not alter production assets or services.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools'), str(ROOT/'thirdParty/ibug')]
from runAdaptiveChinBatch import archive, serial, sha, write
from renderClientSemanticInputs import scene, raster
from observeRenderedClientSemantics import detect, overlay
from faceAppearance.renderedJawCorrespondence import prepareRenderedJawCorrespondence, POLICY as BINDING_POLICY
from faceAppearance.renderedJawRefinement import proposeRenderedJawCandidates
from faceAppearance.frontalSemanticReliability import (
    freezeFrontalReliability, checkFrontalReliability, POLICY as RELIABILITY_POLICY,
)
from ibug.face_alignment import FANPredictor

APPEARANCES = ('template', 'gray', 'grayAlternate')
LIVE_POLICY = {
    'primaryAppearance': 'template', 'controlAppearances': ['gray', 'grayAlternate'],
    'minimumReferenceImprovementFaceFraction': .0075,
    'minimumReferenceImprovementOriginalPixels': 1.,
    'minimumRelativeReferenceImprovement': .15,
    'grayMinimumImprovementFractionOfRequiredGain': .5,
    'maximumPerViewMeanIncreaseFaceFraction': .005,
    'maximumPerPointIncreaseFaceFraction': .015,
    'allowRequiredPointDropout': False,
    'requireRepeatedAnatomicalAndAppearanceGates': True,
    'measurementScale': 'sqrt(saved photo bbox area) in working pixels',
    'fixedIdentitySet': 'Baseline solver-comparable views and their original accepted semantic IDs, all three appearances',
    'selection': 'minimum mean normalized live template error over fixed views plus normalized reference error',
}


def live_policy(front_primary=False, reliability_weighted=False):
    if not front_primary:
        return dict(LIVE_POLICY)
    policy = dict(LIVE_POLICY,
        frontPrimary=True,
        fixedIdentitySet='Frontal reference only; frozen baseline IDs and original photo targets in all three appearances',
        selection='minimum normalized live template error in the frontal reference only',
        otherViews='Diagnostic observations only; no fit, semantic veto, minimum-view requirement, or ranking weight',
        geometrySafety='Original-relative displacement, strain, smoothness, fixed supports and new-intersection gates remain mandatory')
    if reliability_weighted:
        policy.update(reliabilityWeighted=True, reliability=dict(RELIABILITY_POLICY),
            selection='minimum baseline-frozen reliability weighted template error in frontal reference only',
            requireRepeatedAnatomicalAndAppearanceGates='All detection/anatomical gates mandatory; appearance/light use frozen-weight RMS and every-point limits')
    return policy


def detector_model():
    import torch
    torch.set_num_threads(1)
    model = FANPredictor.get_model('2dfan2_alt')
    model.weights = str(ROOT/'assets/models/ibug/2dfan2_alt.pth')
    return FANPredictor(device='cpu', model=model, config=FANPredictor.create_config(use_jit=False))


def baseline_detections(case, prior):
    rows = [r for r in prior['rows'] if r['caseId'] == case['caseId'] and r['stage'] == 'before']
    output = []
    for i in sorted({row['viewIndex'] for row in rows}):
        variants = {}
        for variant in APPEARANCES:
            row = next(r for r in rows if r['viewIndex'] == i and r['variant'] == variant)
            variants[variant] = archive(ROOT/'runs/clientRenderedSemantics20261012/evaluation'/row['npzPath'])
        output.append({'viewIndex': i, 'variants': variants})
    return output


def correspondence(vertices, faces, template_vertices, cameras, observations, detections,
                   inputs, fixed, protected, domain=None):
    return prepareRenderedJawCorrespondence(vertices, faces, template_vertices, cameras,
        observations, detections, imageShapes=[record['workingSize'][::-1] for record in inputs],
        fixedVertexIds=fixed, protectedVertexIds=protected, domain=domain)


def live_check(baseline, after, original_correspondence, new_correspondence, observations,
               inputs, reference_index, comparable_indices, *, front_primary=False, reliability_weighted=False):
    if reliability_weighted and not front_primary:
        raise ValueError('Reliability weighting requires frontal-only acceptance')
    policy = live_policy(front_primary, reliability_weighted)
    before_map = {row['viewIndex']: row['variants'] for row in baseline}
    after_map = {row['viewIndex']: row['variants'] for row in after}
    original_map = {v['viewIndex']: v for v in original_correspondence['views']}
    new_map = {v['viewIndex']: v for v in new_correspondence['views']}
    required_indices = [reference_index] if front_primary else comparable_indices
    if reference_index is None or reference_index not in comparable_indices:
        raise ValueError('Live acceptance requires an available frozen reference view')
    diagnostics = {'accepted': True, 'reasons': [], 'views': [], 'diagnosticViews': [],
                   'referenceViewIndex': reference_index, 'frontPrimary': front_primary,
                   'reliabilityWeighted': reliability_weighted,
                   'policy': policy, 'requiredViewIndices': required_indices}
    main_errors = []
    for index in comparable_indices:
        required = index in required_indices
        reasons = []
        ids = np.asarray(original_map[index]['semanticIds'], int)
        after_ids = np.asarray(new_map[index]['semanticIds'], int)
        missing = sorted(set(ids.tolist())-set(after_ids.tolist()))
        weighted = reliability_weighted and required
        reliability = (checkFrontalReliability(original_correspondence, new_correspondence, index)
                       if weighted else None)
        if weighted:
            weights = np.asarray(original_map[index]['frozenReliabilityWeights'], float)
            reasons.extend(reliability['reasons'])
        else:
            weights = np.ones(len(ids))/len(ids)
        if missing and not weighted:
            reasons.append(f'view{index}: required anatomical/appearance gates lost IDs {missing}')
        scale = float(observations['faceScaleWorkingPixels'][index])
        target = observations['faceLandmarksWorking'][index, ids]
        minimum_gain = max(policy['minimumReferenceImprovementFaceFraction']*scale,
            policy['minimumReferenceImprovementOriginalPixels']*
            np.linalg.svd(np.asarray(inputs[index]['inputToWorking'])[:2, :2], compute_uv=False)[0])
        view = {'viewIndex': index, 'semanticIds': ids,
                'baselineSetPreserved': not reliability['hardFailures'] if weighted else not missing,
                'missingSemanticIds': missing, 'faceScaleWorkingPixels': scale, 'appearances': [],
                'role': 'fit' if required else 'diagnostic', 'affectsAcceptance': required,
                'semanticWeights': weights, 'reliabilityCheck': reliability}
        for appearance in APPEARANCES:
            a, b = before_map[index][appearance], after_map[index][appearance]
            before_error = np.linalg.norm(a['points'][ids]-target, axis=1)
            after_error = np.linalg.norm(b['points'][ids]-target, axis=1)
            dropout = ids[~np.asarray(b['valid'])[ids]]
            increase = after_error-before_error
            before_mean = float(weights@before_error) if weighted else float(before_error.mean())
            after_mean = float(weights@after_error) if weighted else float(after_error.mean())
            mean_increase = after_mean-before_mean
            accepted = (not len(dropout) and np.isfinite(after_error).all()
                        and mean_increase <= policy['maximumPerViewMeanIncreaseFaceFraction']*scale
                        and increase.max() <= policy['maximumPerPointIncreaseFaceFraction']*scale)
            if not accepted:
                reasons.append(f'view{index}/{appearance}: invalid or regressed fixed semantic set')
            metric = {'variant': appearance, 'accepted': bool(accepted), 'dropoutIds': dropout,
                'beforeMeanPixels': before_mean, 'afterMeanPixels': after_mean,
                'reliabilityWeighted': weighted,
                'unweightedBeforeMeanPixels': float(before_error.mean()),
                'unweightedAfterMeanPixels': float(after_error.mean()),
                'beforePointErrorsPixels': before_error, 'afterPointErrorsPixels': after_error,
                'maximumPointIncreasePixels': float(increase.max()), 'meanIncreasePixels': mean_increase,
                'beforePixels': a['points'][ids], 'afterPixels': b['points'][ids], 'targetPixels': target}
            if index == reference_index:
                gain = before_mean-after_mean
                required_gain = (max(minimum_gain, policy['minimumRelativeReferenceImprovement']*before_mean)
                    if appearance == 'template' else minimum_gain*policy['grayMinimumImprovementFractionOfRequiredGain'])
                metric.update(improvementPixels=gain, requiredImprovementPixels=required_gain)
                if gain < required_gain:
                    reasons.append(f'reference/{appearance}: insufficient live improvement')
                if (weighted and appearance == 'template'
                        and increase.mean() > RELIABILITY_POLICY['unweightedNonRegressionNumericalTolerancePixels']):
                    reasons.append('reference/template: unweighted original semantic set regressed')
            if appearance == 'template':
                if required:
                    main_errors.append(after_mean/scale)
                if index == reference_index:
                    diagnostics['referenceMetric'] = metric
            view['appearances'].append(metric)
        view['observations'] = reasons
        if required:
            diagnostics['views'].append(view)
            diagnostics['reasons'].extend(reasons)
        else:
            diagnostics['diagnosticViews'].append(view)
    diagnostics['accepted'] = not diagnostics['reasons']
    diagnostics['selectionCost'] = (float(np.mean(main_errors)) if front_primary else
        float(np.mean(main_errors))+
        diagnostics['referenceMetric']['afterMeanPixels']/float(observations['faceScaleWorkingPixels'][reference_index]))
    return diagnostics


def render_detect(case, vertices, base_scene, cameras, inputs, observations, directory, detector):
    directory.mkdir(parents=True)
    current = {**base_scene, 'vertices': base_scene['vertices'].copy()}
    # Isolate skin geometry. All accessory geometry and all UV/material payloads
    # are copied from the original scene for every candidate, never refitted.
    current['vertices'][current['nicpToFullIds']] = vertices
    views = []
    render_seconds = detect_seconds = 0.
    for index, (camera, record) in enumerate(zip(cameras, inputs)):
        mapping = np.asarray(record['inputToWorking'])
        native_camera = {**camera, 'intrinsics': np.linalg.inv(mapping)@np.asarray(camera['intrinsics'])}
        tick = perf_counter()
        native, _, _ = raster(current, native_camera, tuple(record['orientedSize'][::-1]))
        working, _, _ = raster(current, camera, tuple(record['workingSize'][::-1]))
        render_seconds += perf_counter()-tick
        row = {'viewIndex': index, 'variants': {}}
        for variant in APPEARANCES:
            path = directory/f'view{index}-{variant}-original.png'
            native[variant].save(path)
            working[variant].save(directory/f'view{index}-{variant}.png')
            prediction = detect(detector, path, np.asarray(record['faceDetection']['boxXyxy']), mapping,
                                record['workingSize'], observations['faceScaleWorkingPixels'][index])
            row['variants'][variant] = prediction
            detect_seconds += prediction['seconds']
            np.savez_compressed(directory/f'view{index}-{variant}.npz',
                **{k:v for k,v in prediction.items() if isinstance(v, np.ndarray)})
            overlay(working[variant], prediction['points'], prediction['valid'],
                counterpart=observations['faceLandmarksWorking'][index]).save(directory/f'view{index}-{variant}-landmarks.png')
        # Only visual evidence, never enters fitting/selection.
        if 'photoDerived' in working:
            working['photoDerived'].save(directory/f'view{index}-photoDerived.png')
        views.append(row)
    return views, {'renderSeconds': render_seconds, 'detectionSeconds': detect_seconds}


def evidence(case, directory, selected, before, after, prior_rows, observations, inputs,
             comparable_indices, reference_index, *, front_primary=False):
    output = directory/'evidence'
    output.mkdir(exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    rows = []
    for index, record in enumerate(inputs):
        photo = Image.open(Path(case['observationPath']).parent/f'view{index}-photo.png').convert('RGB')
        width, height = photo.size
        photo.save(output/f'view{index}-photo.png')
        photo_overlay = overlay(photo, observations['faceLandmarksWorking'][index],
                                observations['landmarkCandidateValidMask'][index])
        photo_overlay.save(output/f'view{index}-photo-landmarks.png')
        appearances = {}
        for variant in APPEARANCES:
            old_row = next(r for r in prior_rows if r['viewIndex'] == index and r['variant'] == variant)
            source_root = ROOT/'runs/clientRenderedSemantics20261012/evaluation'
            old_image = Image.open(source_root/old_row['display']['render']).convert('RGB')
            new_image = (old_image.copy() if selected == 'noop' else
                Image.open(directory/'candidates'/selected/f'view{index}-{variant}.png').convert('RGB'))
            old_overlay = overlay(old_image, before[index]['variants'][variant]['points'], before[index]['variants'][variant]['valid'],
                                  counterpart=observations['faceLandmarksWorking'][index])
            new_overlay = overlay(new_image, after[index]['variants'][variant]['points'], after[index]['variants'][variant]['valid'],
                                  counterpart=observations['faceLandmarksWorking'][index])
            names = {}
            for label, im in [('before',old_image),('after',new_image),('before-landmarks',old_overlay),('after-landmarks',new_overlay)]:
                name = f'view{index}-{variant}-{label}.png'
                im.save(output/name); names[label] = f'evidence/{name}'
            board = Image.new('RGB', (width*3, height+52), (20, 27, 35))
            draw = ImageDraw.Draw(board)
            for col, (im, title) in enumerate(zip((photo, old_image, new_image), ('Original photo','Before','Semantic fit / unchanged'))):
                draw.text((col*width+8, 8), f'{case["caseId"]} | view {index+1} | {title}', font=font, fill='white')
                draw.text((col*width+8, 29), variant, font=font, fill=(170,194,204))
                board.paste(im, (col*width, 52))
            board.save(output/f'view{index}-{variant}-comparison.png')
            names['comparison'] = f'evidence/view{index}-{variant}-comparison.png'
            appearances[variant] = names
        participated = index == reference_index if front_primary else index in comparable_indices
        rows.append({'viewIndex':index, 'reference': index == reference_index,
            'participated': participated,
            'role': 'fit' if participated else ('diagnostic' if index in comparable_indices else 'unavailable'),
            'photo':f'evidence/view{index}-photo.png',
            'photoLandmarks':f'evidence/view{index}-photo-landmarks.png', 'appearances':appearances})
    return rows


def run_case(case, directory, prior, detector, frozen_hashes, *, all_safe=False, front_primary=False,
             reliability_weighted=False, safe_step_search=False):
    if directory.exists():
        raise FileExistsError(f'Use a new experiment output: {directory}')
    directory.mkdir(parents=True)
    start = perf_counter()
    for path_key, hash_key in [('geometryPath','geometrySha256'),('facesPath','facesSha256'),
            ('cameraPath','cameraSha256'),('inputsPath','inputsSha256'),('templateArrayPath','templateSha256')]:
        if sha(case[path_key]) != case[hash_key]: raise ValueError(f'Changed source {path_key}')
    vertices = archive(case['geometryPath'])[case['verticesKey']]
    faces = archive(case['facesPath'])[case['facesKey']]
    template = archive(case['templateArrayPath'])
    template_vertices = template['vertices'][template['nicpToFullIds']]
    fixed = np.flatnonzero(template['staticMask'])
    landmark_binding = archive(case['protectedBindingsPath'])
    protected = np.unique(landmark_binding['vertexIndices'])
    cameras = json.loads(Path(case['cameraPath']).read_text())
    inputs = json.loads(Path(case['inputsPath']).read_text())
    observations = archive(case['observationPath'])
    observations['workingImageShape'] = np.asarray([r['workingSize'][::-1] for r in inputs])
    before = baseline_detections(case, prior)
    bindings = correspondence(vertices, faces, template_vertices, cameras, observations, before,
                              inputs, fixed, protected)
    if reliability_weighted:
        bindings = freezeFrontalReliability(bindings, observations, before)
    write(directory/'baselineCorrespondence.json', bindings)
    tick = perf_counter()
    proposals = proposeRenderedJawCandidates(vertices, faces, cameras, observations, bindings,
        fixedVertexIds=fixed, protectedVertexIds=protected, landmarkBinding=landmark_binding,
        allSafeCandidates=all_safe, frontPrimary=front_primary, safeStepSearch=safe_step_search)
    proposal_seconds = perf_counter()-tick
    write(directory/'proposalReport.json', proposals['diagnostics'])
    np.savez_compressed(directory/'proposals.npz', **{c['candidateId']:c['vertices'] for c in proposals['candidates']})
    reference = proposals['diagnostics'].get('referenceViewIndex')
    comparable = proposals['diagnostics'].get('comparableViewIndices', [])
    fit_indices = proposals['diagnostics'].get('fitViewIndices', comparable)
    diagnostic_indices = [i for i in range(len(cameras)) if i not in fit_indices]
    base_scene = scene(case, vertices)
    checks, survivors = [], []
    selected, chosen_vertices, selected_detection = 'noop', vertices.copy(), before
    for candidate in proposals['candidates']:
        if candidate['candidateId'] == 'noop': continue
        name = candidate['candidateId']
        current_dir = directory/'candidates'/name
        rendered, timing = render_detect(case, candidate['vertices'], base_scene, cameras, inputs,
                                        observations, current_dir, detector)
        after_bindings = correspondence(candidate['vertices'], faces, template_vertices, cameras, observations,
            rendered, inputs, fixed, protected, domain=bindings['domain'])
        write(current_dir/'correspondence.json', after_bindings)
        check = live_check(before, rendered, bindings, after_bindings, observations, inputs, reference, comparable,
                           front_primary=front_primary, reliability_weighted=reliability_weighted)
        check.update(candidateId=name, proposal=candidate['diagnostics'], **timing)
        write(current_dir/'liveCheck.json', check)
        checks.append(check)
        if check['accepted']:
            survivors.append((check['selectionCost'], candidate, rendered, check))
        write(directory/'candidateChecks.json', checks)
        print(json.dumps(serial({'caseId':case['caseId'],'candidateId':name,'liveAccepted':check['accepted'],
            'reference':check.get('referenceMetric'),'reasons':check['reasons']})),flush=True)
    winner = None
    if survivors:
        _, candidate, selected_detection, winner = min(survivors, key=lambda item:item[0])
        selected, chosen_vertices = candidate['candidateId'], candidate['vertices']
    protected_all = np.union1d(fixed, protected)
    if not np.array_equal(vertices[protected_all], chosen_vertices[protected_all]):
        raise ValueError('Protected geometry changed')
    for relative, digest in frozen_hashes.items():
        if sha(ROOT/relative) != digest: raise RuntimeError('Implementation changed during frozen batch')
    np.savez_compressed(directory/'geometry.npz', before=vertices, after=chosen_vertices, faces=faces,
        fixedVertexIds=fixed, protectedVertexIds=protected, allHardFixedVertexIds=proposals['allHardFixedVertexIds'])
    prior_rows = [r for r in prior['rows'] if r['caseId']==case['caseId'] and r['stage']=='before']
    pictures = evidence(case, directory, selected, before, selected_detection, prior_rows, observations,
                        inputs, comparable, reference, front_primary=front_primary)
    # A simple canonical mesh export supports later independent inspection.
    write(directory/'mesh.json', {'verticesBefore': vertices, 'verticesAfter': chosen_vertices, 'faces': faces})
    status = 'accepted_live_semantics' if selected != 'noop' else 'unchanged_no_live_candidate'
    report = {'caseId':case['caseId'],'status':status,'selectedCandidateId':selected,
        'referenceViewIndex':reference, 'comparableViewIndices':comparable,
        'frontPrimary':front_primary, 'fitViewIndices':fit_indices, 'diagnosticViewIndices':diagnostic_indices,
        'reliabilityWeighted':reliability_weighted, 'safeStepSearch':safe_step_search,
        'frontReliability':next((v.get('reliability') for v in bindings['views'] if v['viewIndex']==reference),None),
        'excludedViewIndices':[i for i in range(len(cameras)) if i not in comparable],
        'proposalSeconds':proposal_seconds, 'seconds':perf_counter()-start,
        'livePolicy':live_policy(front_primary,reliability_weighted),'bindingPolicy':BINDING_POLICY,'solver':proposals['diagnostics'],
        'candidateChecks':checks,'selectedLiveCheck':winner,'views':pictures,
        'changedVertexCount':int(np.any(chosen_vertices!=vertices,axis=1).sum()),
        'protectedVerticesExactlyUnchanged':bool(np.array_equal(chosen_vertices[protected_all],vertices[protected_all])),
        'noopExactlyUnchanged':bool(np.array_equal(chosen_vertices,vertices)) if selected=='noop' else None,
        'sourceCase':case, 'implementationHashes':frozen_hashes,'remoteModified':False,
        'accessories':'Original assembled accessory geometry and textures held constant for controlled before/after comparison.',
        'limitations':['This is a conservative vertical shortening prototype, not general jaw shape or depth recovery.',
            'FAN observations and template semantic exclusion domain are automatic proxies, not annotated ground truth.',
            ('Only the frontal fixed semantic set is checked for acceptance; other angles are observations, not semantic guarantees.'
             if front_primary else 'Only baseline-comparable semantic views are checked for acceptance; all views are displayed.'),
            'No texture rebake, full accessory collision validation, animation validation, or deployment.',
            'Intersection checks exclude adjacent/shared-vertex triangle pairs and do not certify the deformation path.']}
    write(directory/'report.json',report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'runs/clientChinAdaptive20261012/cases.json')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/clientRenderedJawFit20261012/evaluationV1')
    parser.add_argument('--case',action='append')
    parser.add_argument('--all-safe-proposals',action='store_true',
        help='Live-evaluate every safe member of the unchanged finite 3 by 4 proposal grid')
    parser.add_argument('--front-primary',action='store_true',
        help='Fit, rank and accept from frontal semantics only; other angles remain diagnostic')
    parser.add_argument('--reliability-weighted',action='store_true',
        help='Freeze baseline frontal reliability weights and use bounded weighted appearance stability')
    parser.add_argument('--safe-step-search',action='store_true',
        help='Find a witnessed geometrically feasible amplitude before proposing correction fractions')
    args=parser.parse_args()
    if args.reliability_weighted and not args.front_primary:
        parser.error('--reliability-weighted requires --front-primary')
    args.output.mkdir(parents=True,exist_ok=True)
    paths=['src/faceAppearance/renderedJawCorrespondence.py','src/faceAppearance/renderedJawRefinement.py',
           'tools/runRenderedJawExperiment.py','tools/renderClientSemanticInputs.py','tools/observeRenderedClientSemantics.py',
           'src/faceAppearance/adaptiveChinRefinement.py','src/faceAppearance/clientGeometrySafety.py',
           'src/faceAppearance/frontLandmarkRefinement.py','src/faceAppearance/surfaceVisibility.py',
           'src/faceAppearance/frontalSemanticReliability.py']
    hashes={p:sha(ROOT/p) for p in paths}
    implementation=args.output/'implementation';implementation.mkdir(exist_ok=True)
    for relative in paths:shutil.copyfile(ROOT/relative,implementation/Path(relative).name)
    policy={'live':live_policy(args.front_primary,args.reliability_weighted),'binding':BINDING_POLICY,'implementationHashes':hashes,
        'frozenBeforeCandidateRun':True,'manifestSha256':sha(args.manifest),
        'allSafeProposals':args.all_safe_proposals,
        'frontPrimary':args.front_primary,
        'reliabilityWeighted':args.reliability_weighted,'safeStepSearch':args.safe_step_search,
        'developmentScope':'Policy frozen before this batch; prior development experiments are not held-out validation.'}
    write(args.output/'policy.json',policy)
    cases=json.loads(args.manifest.read_text())['cases']
    prior=json.loads((ROOT/'runs/clientRenderedSemantics20261012/evaluation/report.json').read_text())
    detector=detector_model()
    started=perf_counter()
    summary={'status':'running','cases':[],'policy':policy,'remoteModified':False}
    for case in cases:
        if args.case and case['caseId'] not in args.case:continue
        report=run_case(case,args.output/case['caseId'],prior,detector,hashes,
                        all_safe=args.all_safe_proposals,front_primary=args.front_primary,
                        reliability_weighted=args.reliability_weighted,safe_step_search=args.safe_step_search)
        metric=report['selectedLiveCheck']['referenceMetric'] if report['selectedLiveCheck'] else None
        row={key:report[key] for key in ('caseId','status','selectedCandidateId','referenceViewIndex',
            'comparableViewIndices','excludedViewIndices','frontPrimary','fitViewIndices',
            'diagnosticViewIndices','reliabilityWeighted','safeStepSearch','changedVertexCount','seconds')}
        row.update(reportPath=f'{case["caseId"]}/report.json',referenceMetric=metric)
        summary['cases'].append(row);write(args.output/'summary.json',summary)
        print(json.dumps(serial(row)),flush=True)
    summary.update(status='complete',seconds=perf_counter()-started)
    write(args.output/'summary.json',summary)


if __name__=='__main__':main()
