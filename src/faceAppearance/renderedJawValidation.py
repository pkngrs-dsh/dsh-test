"""Frozen live acceptance policy for rendered jaw correction.

Extracted unchanged from the qualified frontWeightedSafeV1 experiment. This
module has no model initialization, project discovery or experiment I/O.
"""
from __future__ import annotations

import numpy as np
from .frontalSemanticReliability import checkFrontalReliability, POLICY as RELIABILITY_POLICY

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
