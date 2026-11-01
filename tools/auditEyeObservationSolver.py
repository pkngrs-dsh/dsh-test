"""Attribute the unified-observation experiment using saved evidence only.

No fitting, projection, rasterization, collision detection or offline original-
coarse evaluation is performed here. The independent validator owns those
recomputations. This reader reports saved decisions and their stage, without
equating an inner accepted step with an accepted complete head or likeness.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/eyeObservation20261024'
OUTPUT = ROOT/'reports/eyeObservation20261024/solverAudit/summary.json'
BRANCHES = ('baseline', 'boundaryReplay', 'unifiedObservationContact')
PAIR_CLASSES = ('skin-skin', 'skin-canthus', 'skin-eye', 'skin-alpha', 'skin-otherOpaque', 'other-nonSkinPair')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def accepted(value, key='accepted'):
    """A missing decision is unknown, never an implicit pass."""
    result = value.get(key) if isinstance(value, dict) else None
    if result is not None and type(result) is not bool:
        raise ValueError(f'Expected explicit saved boolean {key}')
    return result


def recordFlags(counter, flags):
    for name, value in flags.items():
        counter[name+('Accepted' if value is True else 'Rejected' if value is False else 'Unavailable')] += 1


def savedSafety(trial):
    safety = trial.get('safety') or {}
    extra = safety.get('additionalSafety') or {}
    flags = dict(geometry=accepted(safety, 'geometricSafetyAccepted'),
        contact=accepted(extra.get('contact')), activeCoarse=accepted(extra.get('coarseAlignment')),
        combinedSafety=accepted(trial, 'safetyAccepted'))
    parts = [name+'Rejected' if value is False else name+'Unavailable'
        for name, value in flags.items() if name != 'combinedSafety' and value is not True]
    return flags, '+'.join(parts) or 'geometryContactActiveCoarsePass'


def displacement(before, after, unit):
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape or before.ndim != 2 or before.shape[1] != 3:
        raise ValueError('Saved vertex arrays must have the same Nx3 identity')
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError('Saved vertices contain nonfinite values')
    distances = np.linalg.norm(after-before, axis=1)/unit
    return dict(changedVertexCount=int(np.any(before != after, axis=1).sum()),
        exactlyUnchanged=bool(np.array_equal(before, after)), maximumMm=float(distances.max(initial=0)),
        rmsMm=float(np.sqrt(np.mean(distances*distances))) if len(distances) else 0.)


def arrayReference(saved, item):
    if not isinstance(item, dict) or set(item) != {'arrayKey', 'shape', 'dtype'}:
        raise ValueError('Expected JSON-indexed evidence array')
    value = saved[item['arrayKey']]
    if list(value.shape) != item['shape'] or str(value.dtype) != item['dtype']:
        raise ValueError('Saved evidence array identity differs from index')
    return value


def summarizeRound(roundEvidence, stage, saved, unit):
    item = roundEvidence['evidence']; counters = Counter(); combinations = Counter()
    decisions = Counter(); acceptedNorms = []; acceptedMeans = []
    for trial in item['trials']:
        flags, combination = savedSafety(trial)
        recordFlags(counters, flags); combinations[combination] += 1
        decisions[trial['decision']] += 1
        if trial['decision'] == 'acceptedStep':
            before = arrayReference(saved, trial['beforeCoefficients'])
            after = arrayReference(saved, trial['trialCoefficients'])
            acceptedNorms.append(float(np.linalg.norm(after-before)))
            acceptedMeans.append(float(trial['evaluation']['meanErrorPixels']))
    castCounts = Counter(); castRows = []
    for trial in item['castTrials']:
        flags, combination = savedSafety(trial); recordFlags(castCounts, flags)
        castRows.append(dict(scale=trial['scale'], flags=flags, exclusiveSafetyCombination=combination))
    before = arrayReference(saved, item['referenceVertices'])
    after = arrayReference(saved, item['outputVertices'])
    finalCoefficients = arrayReference(saved, item['finalCoefficients'])
    basis = arrayReference(saved, item['basis'])
    if not np.isfinite(basis).all(): raise ValueError('Saved basis is nonfinite')
    singular = np.linalg.svd(basis, compute_uv=False)
    threshold = max(basis.shape)*np.finfo(float).eps*singular[0] if len(singular) else 0.
    active = np.any(basis != 0, axis=1)
    faces = arrayReference(saved, item['faces'])
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    solver = stage['solver']
    return dict(round=roundEvidence['round'], trialCount=len(item['trials']), decisionCounts=dict(decisions),
        safetyCounts=dict(counters), exclusiveSafetyCombinations=dict(combinations),
        acceptedStepCount=len(acceptedNorms), zeroAcceptedCoefficientStepCount=sum(value == 0 for value in acceptedNorms),
        minimumAcceptedCoefficientStepNorm=min(acceptedNorms) if acceptedNorms else None,
        maximumAcceptedCoefficientStepNorm=max(acceptedNorms) if acceptedNorms else None,
        bestAcceptedPointMeanPixels=min(acceptedMeans) if acceptedMeans else None,
        initialPointMeanPixels=solver['initialMeanErrorPixels'], finalPointMeanPixels=solver['finalMeanErrorPixels'],
        bestIteration=solver['bestIteration'], iterationsCompleted=solver['iterationsCompleted'],
        initialObjective=item['initialObjective'], initialObjectiveComponents=item['initialObjectiveComponents'],
        finalEvaluation=item['finalEvaluation'], castTrialCount=len(castRows), castSafetyCounts=dict(castCounts),
        castTrials=castRows, castRejected=item['castRejected'],
        savedSolverDisplacement=displacement(before, after, unit),
        finalCoefficientsExactlyZero=bool(np.count_nonzero(finalCoefficients) == 0),
        basis=dict(controlCount=basis.shape[1], activeVertexCount=int(active.sum()),
            activeEdgeCount=int(active[edges].any(1).sum()), numericalRank=int((singular > threshold).sum()),
            rankTolerance=float(threshold), normalization=float(item['normalization'])),
        originalCoarseOnSameGnTrial='Not evaluated here; independent observation validator owns that comparison',
        acceptedStepMeaning='Inner GN point/objective decision only; outer/full-head and complete-mask gates remain separate')


def compactCoarse(gate):
    if not isinstance(gate, dict): return None
    eyes = {}
    for name, eye in gate.get('eyes', {}).items():
        before, after = eye.get('before', {}), eye.get('after', {})
        eyes[name] = dict(accepted=accepted(eye), beforeMeanPixels=before.get('meanErrorPixels'),
            afterMeanPixels=after.get('meanErrorPixels'), beforeMaximumPixels=before.get('maximumErrorPixels'),
            afterMaximumPixels=after.get('maximumErrorPixels'))
    return dict(accepted=accepted(gate), reasons=gate.get('reasons'), tolerancePixels=gate.get('tolerancePixels'), eyes=eyes)


def compactReference(gate):
    if not isinstance(gate, dict): return None
    return {key: gate.get(key) for key in ('accepted', 'frontMeanChanges',
        'sideMeanChangesAgainstOriginalInitial', 'frontTolerance', 'sideTolerance')}


def summarizeProposal(proposal, saved, baseline, unit, branch, classify):
    strict = proposal['strict']; external = strict.get('externalCandidateGate') or {}
    activeCoarse, activeReference = strict.get('coarseAlignment'), external.get('legacyGate')
    original = external.get('originalPolicy')
    if branch != 'unifiedObservationContact':
        originalCoarse, originalReference = activeCoarse, activeReference
        originalSource = 'Active original policy in the unchanged-observation arm'
    else:
        originalCoarse = (original or {}).get('coarseAlignment')
        originalReference = (original or {}).get('legacyGate')
        originalSource = 'Separately saved original-policy diagnostic at this same outer proposal'
    assembly = strict.get('fullAssembly'); pairs = None if assembly is None else assembly.get('newPairs')
    classes = Counter({name: 0 for name in PAIR_CLASSES})
    if pairs is not None: classes.update(classify(int(a), int(b)) for a, b in pairs)
    attempt = proposal.get('solverAttempt') or {}
    measured = 'apertureMetrics' in attempt
    aperture = attempt.get('apertureGate')
    # The fast wrapper stores this particular gate as a bool, whereas the
    # geometry/contact/coarse gates are dictionaries with an accepted field.
    apertureAccepted = aperture if type(aperture) is bool else accepted(aperture)
    flags = dict(strict=accepted(strict), geometry=accepted(strict.get('geometry')),
        fullAssembly=None if pairs is None else not bool(pairs), contact=accepted(strict.get('contact')),
        activeCoarse=accepted(activeCoarse), originalCoarse=accepted(originalCoarse),
        activeReference=accepted(activeReference), originalReference=accepted(originalReference),
        completeMask=apertureAccepted)
    return dict(round=proposal['round'], scale=proposal['scale'], selectedByFastRound=proposal['selectedByFastRound'],
        flags=flags, savedStrictReasons=strict.get('reasons'),
        savedDisplacement=displacement(baseline, arrayReference(saved, proposal['vertices']), unit),
        activeCoarse=compactCoarse(activeCoarse), originalCoarse=compactCoarse(originalCoarse),
        activeReference=compactReference(activeReference), originalReference=compactReference(originalReference),
        originalPolicySource=originalSource, completeMaskMeasured=measured, completeMaskGate=aperture,
        stoppedBeforeCompleteMask=bool(not measured and (flags['strict'] is False or flags['geometry'] is False)),
        newFullPairCount=None if pairs is None else len(pairs), savedNewFullFacePairs=pairs,
        pairRecordClasses=dict(classes), excludedMaterialIds=None if assembly is None else assembly.get('excludedMaterialIds'),
        pairClassificationMeaning='Counts include repeats across proposals; face-pair changes do not measure penetration depth')


def aggregateBranches(rows, branch):
    selected = [row['branches'][branch] for row in rows]
    decisions = Counter(); safety = Counter(); combinations = Counter(); outerFlags = Counter()
    reasons = Counter(); pairs = Counter({name: 0 for name in PAIR_CLASSES}); cast = Counter()
    for value in selected:
        for roundRow in value['solverRounds']:
            decisions.update(roundRow['decisionCounts']); safety.update(roundRow['safetyCounts'])
            combinations.update(roundRow['exclusiveSafetyCombinations']); cast.update(roundRow['castSafetyCounts'])
        for proposal in value['outerProposals']:
            recordFlags(outerFlags, proposal['flags']); reasons.update(proposal['savedStrictReasons'] or [])
            pairs.update(proposal['pairRecordClasses'])
    rounds = [item for value in selected for item in value['solverRounds']]
    outer = [item for value in selected for item in value['outerProposals']]
    return dict(caseCount=len(selected), solverRoundCount=len(rounds), totalGnTrials=sum(item['trialCount'] for item in rounds),
        decisionCounts=dict(decisions), safetyCounts=dict(safety), exclusiveSafetyCombinations=dict(combinations),
        castTrialCount=sum(item['castTrialCount'] for item in rounds), castSafetyCounts=dict(cast),
        bestIterationZeroRounds=sum(item['bestIteration'] == 0 for item in rounds),
        nonzeroSolverOutputRounds=sum(not item['savedSolverDisplacement']['exactlyUnchanged'] for item in rounds),
        outerProposalCount=len(outer), outerGateCounts=dict(outerFlags), outerFailureReasons=dict(reasons),
        outerProposalsStoppedBeforeCompleteMask=sum(item['stoppedBeforeCompleteMask'] for item in outer),
        outerProposalsWithCompleteMaskMeasurement=sum(item['completeMaskMeasured'] for item in outer),
        pairRecordClasses=dict(pairs), pairRecordCount=sum(pairs.values()), pairCountsRepeatAcrossScalesAndRounds=True,
        rawRefinementAcceptedCases=sum(value['rawRefinementAccepted'] is True for value in selected),
        changedRawRefinementCases=sum(not value['rawRefinementDisplacement']['exactlyUnchanged'] for value in selected),
        changedDeliveredCases=sum(not value['deliveredDisplacement']['exactlyUnchanged'] for value in selected),
        savedImprovementAcceptedCases=sum(value['savedImprovementAccepted'] is True for value in selected),
        savedChangingAbsoluteEligibleCases=sum(value['savedChangingAbsoluteEligible'] is True for value in selected),
        finalEffectActivelyRejectedChangedAcceptedCases=sum(value['finalEffectActivelyRejectedChangedAcceptedRaw'] for value in selected))


def audit(output):
    hashes = {}
    def pin(path):
        path = Path(path).resolve(); value = sha(path)
        if str(path) in hashes and hashes[str(path)] != value: raise ValueError('Input changed during read')
        hashes[str(path)] = value; return path
    def read(path): return json.loads(pin(path).read_text())
    def archive(path): return np.load(pin(path), allow_pickle=False)
    codeHash = sha(__file__)
    plan = read(RUN/'experimentPlan.json'); manifest = read(RUN/'implementationHashes.json')
    if sha(RUN/'experimentPlan.json') != manifest['planSha256']: raise ValueError('Frozen plan identity lost')
    if tuple(plan['branches']) != BRANCHES or len(plan['cases']) != 7 or len(set(plan['cases'])) != 7:
        raise ValueError('Expected the unchanged seven-case, three-arm experiment')
    template = ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1'
    components = read(template/'components.json'); materials = read(template/'manifest.json')['materials']
    with archive(template/'template.npz') as data: faces, materialIds = data['faces'], data['materialIds']
    labels = {}
    for component in components:
        for face in component['faceIndices']:
            if face in labels: raise ValueError('Template component membership overlaps')
            labels[face] = component['label']
    if set(labels) != set(range(len(faces))): raise ValueError('Template components do not cover every face')
    def classify(first, second):
        a, b = labels[first], labels[second]
        if a == b == 'head': return 'skin-skin'
        if a != 'head' and b != 'head': return 'other-nonSkinPair'
        other = second if a == 'head' else first; label = labels[other]
        if materials[int(materialIds[other])]['alphaMode'] == 'MASK': return 'skin-alpha'
        if label in ('leftEye', 'rightEye'): return 'skin-eye'
        return 'skin-canthus' if label == 'headAccessory' else 'skin-otherOpaque'
    rows = []
    for case in plan['cases']:
        folder = RUN/case; complete = read(folder/'report.json'); context = read(folder/'inputContext.json')
        if complete['status'] != 'complete' or complete['planSha256'] != manifest['planSha256']:
            raise ValueError('Case is unfinished or belongs to a different plan')
        alignment = read(folder/'observationAlignment.json')
        with archive(context['entry']['preparedGeometry']) as data: unit = float(data['unit'])
        if not np.isfinite(unit) or unit <= 0: raise ValueError('Expected finite positive units per millimeter')
        with archive(folder/'baseline/fullGeometry.npz') as data:
            baseline = data['vertices']; np.testing.assert_array_equal(data['faces'], faces)
        branches = {}
        for branch in BRANCHES:
            directory = folder/branch; report = read(directory/'report.json')
            raw = read(directory/'rawRefinement.json'); evidence = read(directory/'solverEvidence.json')
            stages = raw['diagnostics'].get('fast', {}).get('stages', [])
            byRound = {stage['round']: stage for stage in stages}
            if len(byRound) != len(stages): raise ValueError('Duplicate saved solver round')
            with archive(directory/'solverEvidence.npz') as saved:
                rounds = [summarizeRound(value, byRound[value['round']], saved, unit) for value in evidence]
            with archive(directory/'rawRefinement.npz') as saved:
                rawDisplacement = displacement(baseline, arrayReference(saved, raw['vertices']), unit)
                solverDisplacement = displacement(baseline, arrayReference(saved, raw['solverVertices']), unit)
                lastProposalDisplacement = displacement(baseline, arrayReference(saved, raw['rawCandidateVertices']), unit)
                proposals = [summarizeProposal(value, saved, baseline, unit, branch, classify) for value in raw['candidateProposals']]
            with archive(directory/'fullGeometry.npz') as saved:
                np.testing.assert_array_equal(saved['faces'], faces)
                delivered = displacement(baseline, saved['vertices'], unit)
            rawAccepted = accepted(raw['diagnostics']); effect = report['finalEffectGate']
            activeEffectRejection = bool(rawAccepted is True and not rawDisplacement['exactlyUnchanged']
                and accepted(effect, 'fallbackToExactBaseline') is True)
            if rawAccepted is not True and rawDisplacement['exactlyUnchanged']:
                effectMeaning = 'Earlier solver/proposal safety or improvement policy already returned baseline; final mask gate confirms this output'
            elif activeEffectRejection:
                effectMeaning = 'Final complete-mask gate actively returned a changed accepted raw refinement to baseline'
            else:
                effectMeaning = 'Read the saved raw acceptance, displacement and final complete-mask checks together'
            branches[branch] = dict(solverRounds=rounds, outerProposals=proposals,
                rawStatus=raw['diagnostics']['status'], deliveredStatus=report['diagnostics']['status'],
                rawRefinementAccepted=rawAccepted, rawRefinementDisplacement=rawDisplacement,
                savedSolverDisplacement=solverDisplacement, rawLastProposalDisplacement=lastProposalDisplacement,
                deliveredDisplacement=delivered, finalEffectGate=effect, finalEffectInterpretation=effectMeaning,
                finalEffectActivelyRejectedChangedAcceptedRaw=activeEffectRejection,
                savedIncrementalSafe=accepted(report, 'incrementalSafe'), savedAbsoluteEligible=accepted(report, 'absoluteEligible'),
                savedImprovementAccepted=accepted(report, 'improvementAccepted'),
                savedChangingAbsoluteEligible=accepted(report, 'changingAbsoluteEligible'),
                savedOriginalObservationPolicyAccepted=accepted(report, 'originalObservationPolicyAccepted'),
                savedActiveObservationPolicyAccepted=accepted(report, 'activeObservationPolicyAccepted'),
                replayMatch=report.get('replayMatch'), observationTableSha256=report.get('observationTableSha256'))
        rows.append(dict(caseId=case, observationAlignmentValid=accepted(alignment, 'valid'),
            observationAlignmentError=alignment.get('error'), branches=branches))
    aggregates = {branch: aggregateBranches(rows, branch) for branch in BRANCHES}
    commonValid = [row for row in rows if row['observationAlignmentValid'] is True]
    result = dict(schemaVersion='unifiedEyeObservationSavedSolverAttributionV1', planSha256=manifest['planSha256'], cases=rows,
        aggregate=dict(caseCount=len(rows), validMappingCaseCount=len(commonValid),
            invalidMappingCaseIds=[row['caseId'] for row in rows if row['observationAlignmentValid'] is not True],
            branches=aggregates, commonValidCaseIds=[row['caseId'] for row in commonValid],
            branchesOnCommonValidCases={branch: aggregateBranches(commonValid, branch) for branch in BRANCHES}),
        method=dict(savedEvidenceOnly=True, geometryRefitted=False, projectionsRecomputed=False,
            rasterRecomputed=False, intersectionsRecomputed=False, originalCoarseOnGnTrialsRecomputed=False,
            verticesReconstructedFromCoefficients=False, sourceObservationsReestimated=False,
            rank='SVD of saved B; threshold=max(B.shape)*float64 epsilon*largest singular value',
            counters='Individual gate failure counts overlap; exclusive GN safety combinations also supplied',
            unknownPolicy='Missing saved decisions remain null/unavailable, never passed', numpyVersion=np.__version__),
        limitations=[
            'All dense and sparse observations are automatic estimates; unifying their definitions does not establish anatomical truth.',
            'This is a saved-record attribution reader, not the independent solver/gate validator. It never evaluates original coarse gates on GN trial geometry.',
            'In the unified arm the original coarse/reference gates are retained as separate diagnostics; legacy absolute eligibility still requires them. An active-policy improvement alone does not pass the original policy.',
            'Both fitting arms retain the same field and collision/contact thresholds. This experiment does not lower or exempt any collision gate or material.',
            'Saved full-face pair changes identify a gate condition, not signed penetration depth, visible severity or a continuous collision certificate.',
            'Accepted inner GN steps and reduced point error are not complete-mask or photographic likeness improvements.',
            'Absent complete-mask measurement after an earlier rejection is not itself a failed complete-mask test.',
            'All-case arm counters contain skipped invalid mappings. Compare the separately supplied common-valid subset before comparing counts; even matched counts do not imply causal geometric benefit.',
            'The seven cases are previously inspected regressions, not unseen evidence of generalization.'
        ], inputHashes=hashes, code=dict(path=str(Path(__file__).resolve()), sha256=codeHash), allReadInputsUnchanged=True)
    for path, expected in hashes.items():
        if sha(path) != expected: raise ValueError('Input changed before audit completion: '+path)
    if sha(__file__) != codeHash: raise ValueError('Audit code changed during read')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream: json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(report=str(output.resolve()), sha256=sha(output), inputCount=len(hashes), aggregate=result['aggregate']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, default=OUTPUT)
    audit(parser.parse_args().output)
