"""Baseline-frozen reliability for frontal rendered semantic observations.

Weights are engineering heuristics, not calibrated probabilities. This module
never adds a previously invalid landmark or drops a difficult candidate point.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from .renderedJawCorrespondence import POLICY as BINDING_POLICY


POLICY = {
    'minimumReliabilityMultiplier': .05,
    'minimumEffectivePointCount': 2.5,
    'maximumWeightedDriftFaceFraction': .02,
    'maximumWeightedDriftIncreaseFaceFraction': .005,
    'maximumIndividualDriftFaceFraction': .03,
    'requireUnweightedPrimaryNonRegression': True,
    'unweightedNonRegressionNumericalTolerancePixels': 1e-9,
    'softenedReasons': ['appearance_drift', 'lighting_drift'],
    'formula': 'confidence * max(0.05, 1 - max(normalized appearance, lighting and crop instability)^2)',
    'identity': 'Only original accepted IDs; weights fixed before candidate generation; never renormalized using candidate validity',
    'meaning': 'Engineering reliability heuristic, not a calibrated probability',
}


def freezeFrontalReliability(correspondence, observations, renderViews):
    """Copy correspondence with baseline-only weights; leave its IDs intact.

    Compute independent per-view weights before reference selection. Only the
    solver-selected frontal reference uses them; side views remain diagnostic.
    """
    output = deepcopy(correspondence)
    detections = {int(v['viewIndex']): v['variants'] for v in renderViews}
    reports = {int(v['viewIndex']): {int(p['semanticId']): p for p in v['landmarks']}
               for v in correspondence['diagnostics']}
    summaries = []
    for view in output['views']:
        index = int(view['viewIndex'])
        ids = np.asarray(view['semanticIds'], int)
        if not len(ids):
            continue
        scale = float(view['faceScalePixels'])
        confidence = np.asarray(view['confidence'], float)
        app = np.array([reports[index][int(i)]['appearanceDriftFaceFraction'] for i in ids])
        light = np.array([reports[index][int(i)]['lightingDriftFaceFraction'] for i in ids])
        crop = np.asarray(observations['maximumCropVariationPixels'])[index, ids]/scale
        for name in BINDING_POLICY['requiredAppearances']:
            crop = np.maximum(crop, np.asarray(detections[index][name]['spread'])[ids]/scale)
        q = np.maximum.reduce([app/BINDING_POLICY['maximumAppearanceDriftFaceFraction'],
            light/BINDING_POLICY['maximumGrayLightingDriftFaceFraction'],
            crop/BINDING_POLICY['maximumCropVariationFaceFraction']])
        if not np.isfinite(q).all() or not np.isfinite(confidence).all() or (confidence <= 0).any():
            raise ValueError('Invalid accepted baseline reliability evidence')
        multiplier = np.maximum(POLICY['minimumReliabilityMultiplier'], 1-q*q)
        raw = confidence*multiplier
        weights = raw/raw.sum()
        effective = float(1/np.sum(weights*weights))
        summary = {'viewIndex': index, 'semanticIds': ids.tolist(), 'weights': weights.tolist(),
            'originalConfidence': confidence.tolist(), 'instabilityRatio': q.tolist(),
            'reliabilityMultiplier': multiplier.tolist(), 'rawWeights': raw.tolist(),
            'rawWeightRetention': float(raw.sum()/confidence.sum()),
            'nearThresholdPointCount': int(np.sum(q >= .9)),
            'effectivePointCount': effective,
            'eligible': bool(view.get('eligible', True) and effective >= POLICY['minimumEffectivePointCount'])}
        view['confidence'] = raw
        view['frozenReliabilityWeights'] = weights
        view['eligible'] = summary['eligible']
        view['reliability'] = summary
        summaries.append(summary)
    output['frozenReliability'] = {'policy': dict(POLICY), 'views': summaries}
    return output


def checkFrontalReliability(originalCorrespondence, candidateCorrespondence, index):
    """Validate candidate evidence on all frozen IDs, including weak points."""
    baseline = next(v for v in originalCorrespondence['views'] if v['viewIndex'] == index)
    ids = np.asarray(baseline['semanticIds'], int)
    weights = np.asarray(baseline['frozenReliabilityWeights'], float)
    def points(source):
        row = next(v for v in source['diagnostics'] if v['viewIndex'] == index)
        return {int(p['semanticId']): p for p in row['landmarks']}
    before, after = points(originalCorrespondence), points(candidateCorrespondence)
    reasons, hard_failures = [], []
    softened = set(POLICY['softenedReasons'])
    for sid in ids:
        p = after.get(int(sid))
        hard = ['missing_candidate_semantic_record'] if p is None else [r for r in p['reasons'] if r not in softened]
        if hard:
            hard_failures.append({'semanticId': int(sid), 'reasons': hard})
    if hard_failures:
        reasons.append(f'view{index}: frozen point detection or anatomical checks failed')
    stability = []
    for field in ('appearanceDriftFaceFraction', 'lightingDriftFaceFraction'):
        a = np.array([before[int(sid)][field] for sid in ids], float)
        b = np.array([after.get(int(sid), {}).get(field, np.inf) for sid in ids], float)
        old, new = float(np.sqrt(weights@(a*a))), float(np.sqrt(weights@(b*b)))
        accepted = bool(np.isfinite(b).all()
            and new <= POLICY['maximumWeightedDriftFaceFraction']
            and new-old <= POLICY['maximumWeightedDriftIncreaseFaceFraction']
            and b.max() <= POLICY['maximumIndividualDriftFaceFraction'])
        stability.append({'field': field, 'beforeWeightedRms': old, 'afterWeightedRms': new,
            'beforePointDrifts': a.tolist(), 'afterPointDrifts': b.tolist(),
            'afterMaximum': float(b.max()), 'accepted': accepted})
        if not accepted:
            reasons.append(f'view{index}: frozen-weight {field} limit exceeded')
    return {'accepted': not reasons, 'reasons': reasons, 'semanticIds': ids.tolist(),
            'weights': weights.tolist(), 'hardFailures': hard_failures, 'stability': stability,
            'policy': dict(POLICY)}
