"""Summarize only completed, independently validated fixed-boundary evidence.

No fitting, rendering, threshold changes or candidate promotion occurs here.
Incomplete evidence fails closed. ``--status`` only lists missing completion
artifacts; the default command writes the new experiment's result and viewer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson

RUN = ROOT/'runs/boundaryField20261024'
OUT = ROOT/'reports/boundaryField20261024'
BRANCHES = ('baseline', 'legacyResidualReplay', 'boundaryFieldContact')
LABELS = {'baseline': '完成粗对齐的基线', 'legacyResidualReplay': '旧共用半径（精确重放）',
          'boundaryFieldContact': '固定边界平滑场＋接触'}
MODES = ('clay', 'baseColor', 'lit')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def verify(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError(f'Frozen evidence changed: {path}')


def readEvidence(stem):
    """Decode the existing plain JSON/NPZ evidence without loading pickle."""
    stem = Path(stem)
    metadata, saved = read(stem.with_suffix('.json')), arrays(stem.with_suffix('.npz'))
    used = set()
    def restore(value):
        if isinstance(value, dict) and 'arrayKey' in value:
            key = value['arrayKey']; item = saved[key]
            if list(item.shape) != value['shape'] or str(item.dtype) != value['dtype']:
                raise ValueError(f'Evidence array shape or type changed: {stem}/{key}')
            used.add(key)
            return item
        if isinstance(value, dict): return {key: restore(item) for key, item in value.items()}
        if isinstance(value, list): return [restore(item) for item in value]
        return value
    value = restore(metadata)
    if used != set(saved): raise ValueError(f'Unreferenced evidence arrays: {stem}')
    return value


def flag(mapping, key):
    value = mapping[key]
    if type(value) is not bool: raise ValueError(f'Expected an explicit boolean: {key}')
    return value


def vertexChange(vertices, baseline, unit):
    value = np.asarray(vertices)
    if value.shape != baseline.shape or not np.isfinite(value).all() or not np.isfinite(unit) or unit <= 0:
        raise ValueError('Invalid ordered geometry or nominal template length scale')
    displacement = np.linalg.norm(value-baseline, axis=1)
    changed = np.any(value != baseline, axis=1)
    return dict(geometryChanged=bool(changed.any()), changedVertexCount=int(changed.sum()),
        maximumDisplacementNominalMm=float(displacement.max(initial=0.)/unit),
        rmsDisplacementNominalMm=float(np.sqrt(np.mean(displacement**2))/unit))


def apertureSummary(metrics):
    if metrics is None: return dict(defined=False)
    eyes = metrics.get('eyes', {})
    if set(eyes) != {'imageLeft', 'imageRight'}: return dict(defined=False)
    return dict(defined=True,
        meanIntersectionOverUnion=float(np.mean([eye['intersectionOverUnion'] for eye in eyes.values()])),
        meanBoundaryDistanceOriginalPixels=float(metrics['meanSymmetricBoundaryDistanceOriginalPixels']),
        eyes={name: {key: eye[key] for key in ('intersectionOverUnion',
            'symmetricBoundaryMeanOriginalPixels', 'symmetricBoundaryP95OriginalPixels',
            'visibleComponentCount', 'secondaryComponentAreaFraction')} for name, eye in eyes.items()})


def stageSummaries(captures, diagnostics, unit):
    stages = {stage['round']: stage for stage in diagnostics.get('fast', {}).get('stages', [])}
    summaries = []
    for captured in captures:
        evidence = captured['evidence']; stage = stages[captured['round']]
        basis = np.asarray(evidence['basis']); fixed = np.asarray(evidence['fixedVertexIds'], int)
        active = np.any(basis != 0, axis=1)
        faces = np.asarray(evidence['faces'])
        edges = np.unique(np.sort(np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1), axis=0)
        activeEdges = active[edges].any(1)
        freeMask = np.ones(len(basis), bool); freeMask[fixed] = False
        singular = np.linalg.svd(basis, compute_uv=False) if basis.size else np.empty(0)
        tolerance = max(basis.shape)*np.finfo(float).eps*singular[0] if len(singular) else 0.
        rowNorms = np.linalg.norm(basis, axis=1)
        meaningful = rowNorms > rowNorms.max(initial=0.)*1e-8
        decisions = Counter(); safety = Counter()
        for trial in evidence['trials']:
            decisions[trial['decision']] += 1
            quality = trial['safety']
            if not quality['geometricSafetyAccepted']: safety['ordinaryGeometry'] += 1
            extra = quality.get('additionalSafety')
            if extra is not None:
                if not extra['accepted']: safety['anyAdditionalSafety'] += 1
                if not extra['coarseAlignment']['accepted']: safety['completedCoarseAlignment'] += 1
                if extra['contact'] is not None and not extra['contact']['accepted']: safety['contactEnvelope'] += 1
        solver = stage['solver']; boundary = solver['fixedBoundary']
        displacement = vertexChange(evidence['outputVertices'], evidence['referenceVertices'], unit)
        summaries.append(dict(round=captured['round'], status=stage['status'],
            selectedScale=stage.get('selectedScale'), activeVertexCount=int(active.sum()),
            activeEdgeCount=int(activeEdges.sum()),
            allowedMovableVertexCount=int(freeMask.sum()),
            activeFractionOfAllowed=float(active[freeMask].mean()) if freeMask.any() else 0.,
            meaningfulActiveVertexCount=int(meaningful.sum()), meaningfulRelativeThreshold=1e-8,
            controlCount=basis.shape[1], numericalRank=int((singular > tolerance).sum()),
            singularValues=singular.tolist(), numericalRankTolerance=float(tolerance),
            normalization=float(evidence['normalization']),
            maximumAbsoluteRowSum=float(np.abs(basis).sum(1).max(initial=0.)),
            fixedBasisRowsExactlyZero=bool(not basis[fixed].any()),
            requestedRadiusNominalMm=float(boundary['requestedSupportRadius']/unit),
            effectiveRadiusNominalMm=float(boundary['effectiveSupportRadius']/unit),
            initialPointErrorPixels=solver['initialMeanErrorPixels'], finalPointErrorPixels=solver['finalMeanErrorPixels'],
            solverOutputChange=displacement, solverOutputSafety=solver['safety'],
            reportedRejections=solver['rejections'], trialCount=len(evidence['trials']),
            castTrialCount=len(evidence['castTrials']), castRejected=bool(evidence['castRejected']),
            trialDecisions=dict(decisions), overlappingTrialSafetyRejections=dict(safety),
            rejectionCountMeaning='Independent conditions can reject the same trial; these counts are not additive.',
            regularization=solver['regularization'],
            regularizationNormalization=dict(displacementVertexDenominator=max(int(active.sum()), 1),
                strainEdgeDenominator=max(int(activeEdges.sum()), 1),
                coefficientDenominator=max(basis.shape[1], 1)),
            fieldConstructionPresent=evidence.get('fieldConstruction') is not None,
            fieldConstructionDiagnostics=solver.get('deformationBasis')))
    return summaries


def inheritedFailureSummary(report):
    """Keep original anchors visible even when local updates add no failures."""
    return dict(initialAccepted=flag(report, 'initialAccepted'),
        cumulativeDownstreamAccepted=flag(report['cumulativeDownstreamGate'], 'accepted'),
        cumulativeDownstreamReasons=report['cumulativeDownstreamGate'].get('reasons',
            report['cumulativeDownstreamGate'].get('rejectionReasons', [])),
        newSkinPairsAgainstBranchInitial=len(report['newSkinPairsAgainstBranchInitial']),
        cavityGuardAccepted=flag(report['finalCavityGuard'], 'accepted'),
        cavityBenefitAccepted=flag(report['finalCavityBenefit'], 'accepted'),
        originalPureDirectFullPairCount=len(report['newFullPairsAgainstOriginalPureDirect']),
        currentBoundedFullPairCount=len(report['newFullPairsAgainstCurrentBounded']),
        pairedBaselineFullPairCount=len(report['newFullPairsAgainstPairedBaseline']))


def savedSolverAttribution(row):
    """Separate rejection stages without interpreting pair IDs as depth."""
    rounds, attempts = row['solverRounds'], row['outerAttempts']
    changedAttempts = [item for item in attempts if item['maximumDisplacementMm'] > 0]
    collisionBeforeMask = bool(changedAttempts and all(
        'newFullAssemblyIntersections' in item['reasons'] and not item['completeMaskMeasured']
        for item in changedAttempts))
    noEffectiveInnerChange = bool(rounds and all(item['bestIteration'] == 0
        and item['savedSolverOutputExactlyReference'] for item in rounds))
    return dict(nonzeroInnerSolution=any(not item['savedSolverOutputExactlyReference'] for item in rounds),
        noEffectiveInnerChange=noEffectiveInnerChange,
        allChangedOuterProposalsRejectedByFullPairsBeforeMask=collisionBeforeMask,
        changedOuterAttemptCount=len(changedAttempts),
        fullPairRejectedOuterAttemptCount=sum('newFullAssemblyIntersections' in item['reasons'] for item in attempts),
        completeMaskMeasuredOuterAttemptCount=sum(item['completeMaskMeasured'] for item in attempts),
        maximumAttemptDisplacementNominalMm=max((item['maximumDisplacementMm'] for item in attempts), default=0.),
        rawStatus=row['rawStatus'], outputStatus=row['outputStatus'],
        rawRefinementAccepted=row['rawRefinementAccepted'], geometryChanged=row['geometryChanged'],
        gnCounts=row['gnCounts'], exclusiveSafetyCombinations=row['exclusiveSafetyCombinations'],
        pairRecordClasses=row['pairRecordClasses'],
        pairMeaning='Saved changed face-pair records may repeat across scales and do not establish signed depth or visible penetration severity.',
        savedEvidenceOnly=True, independentGateRecomputationRequired=True)


def expectedImages():
    names = set()
    for view in range(4):
        for mode in MODES:
            names.update((f'view{view:02d}_{mode}.png', f'view{view:02d}_{mode}_eyes.png'))
            names.update(f'view{view:02d}_{branch}_{mode}.png' for branch in BRANCHES)
    return names


def completionStatus():
    if not (RUN/'experimentPlan.json').is_file():
        return dict(status='notFrozen', missing=[str(RUN/'experimentPlan.json')])
    config = read(RUN/'experimentPlan.json'); missing = []
    for path in (RUN/'implementationHashes.json', RUN/'finalUnitTests.json', OUT/'independentAudit/auditImplementation.json',
                 OUT/'solverAudit/summary.json'):
        if not path.is_file(): missing.append(str(path))
    for case in config['cases']:
        for path in (RUN/case/'report.json', OUT/case/'report.json', OUT/'independentAudit'/f'{case}.json'):
            if not path.is_file(): missing.append(str(path))
        for branch in BRANCHES:
            for name in ('report.json', 'fullGeometry.npz', 'rawRefinement.json', 'rawRefinement.npz',
                         'solverEvidence.json', 'solverEvidence.npz', 'finalEffectGate.json'):
                path = RUN/case/branch/name
                if not path.is_file(): missing.append(str(path))
    return dict(status='readyForVerification' if not missing else 'incomplete',
        caseCount=len(config['cases']), missing=missing)


def summarize():
    readiness = completionStatus()
    if readiness['status'] != 'readyForVerification':
        raise ValueError(f"Experiment is incomplete; no final summary written ({len(readiness['missing'])} missing artifacts)")
    config = read(RUN/'experimentPlan.json'); implementation = read(RUN/'implementationHashes.json')
    planHash = sha(RUN/'experimentPlan.json')
    if implementation['planSha256'] != planHash or tuple(config['branches']) != BRANCHES or len(config['cases']) != 7:
        raise ValueError('Plan identity, branch order or seven-case matrix changed')
    verify(implementation['sourceHashes']); verify(config['historicalHashes'])
    auditPath = OUT/'independentAudit/auditImplementation.json'; auditImplementation = read(auditPath)
    verify(auditImplementation['hashes'])
    validatorPath = ROOT/'tools/validateBoundaryFieldExperiment.py'
    validatorHash = auditImplementation['hashes'][str(validatorPath)]
    if auditImplementation['planSha256'] != planHash:
        raise ValueError('Independent audit was frozen for a different plan')
    proof = {str(path): sha(path) for path in (RUN/'experimentPlan.json', RUN/'implementationHashes.json', auditPath)}
    solverAuditPath = OUT/'solverAudit/summary.json'; solverAudit = read(solverAuditPath)
    verify(solverAudit['inputHashes'])
    verify({solverAudit['code']['path']: solverAudit['code']['sha256']})
    if not flag(solverAudit, 'allReadInputsUnchanged'):
        raise ValueError('Saved solver attribution did not preserve input identities')
    solverCases = {row['caseId']: row for row in solverAudit['cases']}
    if len(solverCases) != len(solverAudit['cases']) or set(solverCases) != set(config['cases']):
        raise ValueError('Saved solver attribution case matrix differs')
    proof[str(solverAuditPath)] = sha(solverAuditPath)
    proof[solverAudit['code']['path']] = solverAudit['code']['sha256']
    cases = {}; imageCount = 0
    for case in config['cases']:
        directory = RUN/case; contextPath = directory/'inputContext.json'; context = read(contextPath)
        verify(context['sourceHashes']); proof[str(contextPath)] = sha(contextPath)
        caseReport = read(directory/'report.json')
        if caseReport['status'] != 'complete' or caseReport['planSha256'] != planHash:
            raise ValueError(f'Incomplete case report: {case}')
        verify(caseReport['preparationHashes'])
        auditFile = OUT/'independentAudit'/f'{case}.json'; audit = read(auditFile)
        if not flag(audit, 'validationPassed') or audit['validatorSha256'] != validatorHash:
            raise ValueError(f'Independent validation missing or failed: {case}')
        proof[str(auditFile)] = sha(auditFile)
        renderFile = OUT/case/'report.json'; rendered = read(renderFile)
        verify(rendered['inputHashes']); verify(rendered['renderImplementationHashes'])
        if set(rendered['images']) != expectedImages(): raise ValueError(f'Incomplete four-camera render matrix: {case}')
        for name, expected in rendered['images'].items():
            if sha(OUT/case/name) != expected: raise ValueError(f'Rendered image changed: {case}/{name}')
        for key in ('identicalMaterials', 'allFourCameras', 'templateColorsOnly', 'nonSkinAndStaticExactAcrossArms', 'vertexOrderUvBindingUnchanged'):
            if not flag(rendered, key): raise ValueError(f'Render identity failed: {case}/{key}')
        imageCount += len(rendered['images']); proof[str(renderFile)] = sha(renderFile)
        baseline = arrays(directory/'baseline/fullGeometry.npz')['vertices']
        unit = float(arrays(context['entry']['preparedGeometry'])['unit'])
        branches = {}
        for branch in BRANCHES:
            branchDir = directory/branch; reportPath = branchDir/'report.json'; report = read(reportPath)
            if report['caseId'] != case or report['branch'] != branch: raise ValueError('Branch identity changed')
            branchAudit = audit['branches'][branch]
            if (not flag(branchAudit, 'validationPassed') or not flag(branchAudit, 'evidenceValidated')
                    or branchAudit['reportSha256'] != sha(reportPath)):
                raise ValueError(f'Independent branch evidence does not authenticate this report: {case}/{branch}')
            verify({str(branchDir/name): expected for name, expected in report['outputHashes'].items()})
            raw = readEvidence(branchDir/'rawRefinement'); captured = readEvidence(branchDir/'solverEvidence')
            delivered = arrays(branchDir/'fullGeometry.npz')['vertices']
            change = vertexChange(delivered, baseline, unit)
            incremental, absolute = flag(report, 'incrementalSafe'), flag(report, 'absoluteEligible')
            outputAccepted = flag(report, 'residualCandidateAccepted')
            rawAccepted = flag(report, 'rawRefinementAccepted')
            effect = report['finalEffectGate']
            if effect != read(branchDir/'finalEffectGate.json'): raise ValueError('Final effect report identity changed')
            effectAccepted = flag(effect, 'accepted')
            improvement = bool(branch != 'baseline' and change['geometryChanged'] and outputAccepted and incremental and effectAccepted)
            changingAbsolute = bool(improvement and absolute)
            if change['geometryChanged'] != outputAccepted: raise ValueError('Delivered vertex changes differ from wrapper output acceptance')
            if report.get('improvementAccepted') != improvement:
                raise ValueError('Producer improvement definition differs from changed + selected + incremental + final effect')
            if flag(report, 'changingAbsoluteEligible') != changingAbsolute:
                raise ValueError('Producer absolute improvement classification differs')
            classification = branchAudit['classification']
            independentlyExpected = dict(producerAccepted=outputAccepted,
                geometryChanged=change['geometryChanged'], incrementalSafe=incremental,
                absoluteEligible=absolute, requestedEffectAccepted=effectAccepted,
                improvementAccepted=improvement, changingAbsoluteEligible=changingAbsolute, clientReady=False)
            if any(flag(classification, key) != value for key, value in independentlyExpected.items()):
                raise ValueError(f'Independent candidate classification differs: {case}/{branch}')
            fallback = flag(effect, 'fallbackToExactBaseline')
            if fallback and not np.array_equal(delivered, baseline): raise ValueError('Rejected final effect did not return exact baseline')
            replay = report['replayMatch']
            if branch == 'legacyResidualReplay':
                if replay != read(branchDir/'replayMatch.json') or not flag(replay, 'accepted'):
                    raise ValueError('Old compact refinement did not replay exactly')
            rawChange = vertexChange(raw['vertices'], baseline, unit)
            proposalChanges = [vertexChange(proposal['vertices'], baseline, unit) for proposal in raw['candidateProposals']]
            entry = dict(status=report['diagnostics']['status'], fieldMode=report['fieldMode'],
                delivered=change, rawRefinement=rawChange,
                rawLastProposal=vertexChange(raw['rawCandidateVertices'], baseline, unit),
                rawFastSolver=vertexChange(raw['solverVertices'], baseline, unit),
                proposalChanges=dict(attemptCount=len(proposalChanges),
                    changedAttemptCount=sum(item['geometryChanged'] for item in proposalChanges),
                    maximumChangedVertexCount=max((item['changedVertexCount'] for item in proposalChanges), default=0),
                    maximumDisplacementNominalMm=max((item['maximumDisplacementNominalMm'] for item in proposalChanges), default=0.)),
                rawRefinementAccepted=rawAccepted, deliveredResidualAccepted=outputAccepted,
                incrementalSafe=incremental, absoluteEligible=absolute, requestedEffectAccepted=effectAccepted,
                improvementAccepted=improvement, changingAbsoluteEligible=changingAbsolute,
                independentEvidenceVerified=True, clientReady=False,
                independentClassification=classification,
                independentRawFinalEffect=branchAudit['finalEffectGate'],
                independentDeliveredEffect=branchAudit['finalOutputEffect'],
                finalEffectGate=effect, fallbackToExactBaseline=fallback,
                finalEffectActivelyRejectedAcceptedRawRefinement=bool(fallback and rawAccepted and rawChange['geometryChanged']),
                alreadyUnchangedBeforeFinalEffect=bool(not rawAccepted and not rawChange['geometryChanged']),
                apertureBefore=apertureSummary(effect['baseline']), rawApertureAfter=apertureSummary(effect['candidate']),
                deliveredApertureAfter=apertureSummary(effect['baseline'] if fallback else effect['candidate']),
                legacyReplayMatched=None if replay is None else replay['accepted'],
                replayMatch=replay, originalAnchorGates=inheritedFailureSummary(report),
                stages=stageSummaries(captured, raw['diagnostics'], unit),
                allOutsideSupportExactlyFixed=flag(report, 'allOutsideSupportExactlyFixed'),
                strictLegacyBindingAndPayloadPreserved=True)
            if not entry['allOutsideSupportExactlyFixed']: raise ValueError('Fixed domain changed')
            branches[branch] = entry; proof[str(reportPath)] = sha(reportPath)
        cases[case] = dict(branches=branches, renderedImages=len(rendered['images']),
            referenceViewIndex=context['referenceSelection']['referenceViewIndex'],
            inheritedBaselineFailures=branches['baseline']['originalAnchorGates'],
            newFieldSolverAttribution=savedSolverAttribution(solverCases[case]),
            independentEvidenceVerified=True)
        attribution = cases[case]['newFieldSolverAttribution']; candidate = branches['boundaryFieldContact']
        if (attribution['rawRefinementAccepted'] != candidate['rawRefinementAccepted']
                or attribution['geometryChanged'] != candidate['delivered']['geometryChanged']):
            raise ValueError('Saved solver attribution disagrees with actual branch outputs')
    if imageCount != 420: raise ValueError('Expected 420 rendered PNGs')
    unitTestsPath = RUN/'finalUnitTests.json'; unitTests = read(unitTestsPath)
    if unitTests['status'] != 'passed' or int(unitTests['passed']) < 1: raise ValueError('Source tests did not pass')
    if sha(unitTests['logPath']) != unitTests['logSha256']: raise ValueError('Source test log changed')
    proof[str(unitTestsPath)] = sha(unitTestsPath)
    candidateRows = [row['branches']['boundaryFieldContact'] for row in cases.values()]
    oldRows = [row['branches']['legacyResidualReplay'] for row in cases.values()]
    improvedCases = [case for case, row in cases.items() if row['branches']['boundaryFieldContact']['improvementAccepted']]
    result = dict(schemaVersion='fixedBoundaryFieldSummaryV1', status='complete', cases=cases,
        caseCount=7, branchCount=21, cameraStates=84, imageCount=imageCount,
        candidateImprovedCases=improvedCases, candidateImprovedCaseCount=len(improvedCases),
        candidateChangingAbsoluteEligibleCount=sum(row['changingAbsoluteEligible'] for row in candidateRows),
        candidateFinalEffectFallbackCount=sum(row['fallbackToExactBaseline'] for row in candidateRows),
        candidateFinalEffectActiveRejectionCount=sum(row['finalEffectActivelyRejectedAcceptedRawRefinement'] for row in candidateRows),
        candidateAlreadyUnchangedBeforeFinalEffectCount=sum(row['alreadyUnchangedBeforeFinalEffect'] for row in candidateRows),
        candidateBeforeMaskCollisionRejectedCaseCount=sum(row['newFieldSolverAttribution']['allChangedOuterProposalsRejectedByFullPairsBeforeMask'] for row in cases.values()),
        candidateNoEffectiveInnerChangeCaseCount=sum(row['newFieldSolverAttribution']['noEffectiveInnerChange'] for row in cases.values()),
        candidateRawRefinementAcceptedCount=sum(row['rawRefinementAccepted'] for row in candidateRows),
        legacyReplayMatchedCaseCount=sum(row['legacyReplayMatched'] is True for row in oldRows),
        promotedCandidates=0, productionDefaultsChanged=False, realLipsRefined=False, textureRebaked=False,
        finalAdaptiveAssemblyValidated=False, planSha256=planHash,
        implementationFileCount=len(implementation['sourceHashes']), unitTests=unitTests,
        independentAuditTests=auditImplementation.get('tests'),
        testCountInterpretation='Source and independent-audit test suites may overlap; counts are separate, never summed as unique tests.',
        independentValidatorSha256=validatorHash, evidenceHashes=proof,
        savedSolverAudit=dict(path=str(solverAuditPath), relativeUrl='solverAudit/summary.json',
            sha256=sha(solverAuditPath), code=solverAudit['code'], aggregate=solverAudit['aggregate'],
            method=solverAudit['method'], limitations=solverAudit['limitations'], inputHashCount=len(solverAudit['inputHashes'])),
        summaryImplementationSha256=sha(__file__), viewerUrl='http://127.0.0.1:18100/',
        previousViewerUrl='http://127.0.0.1:18099/',
        conclusions=[
            f'The new field has {len(improvedCases)} of seven independently verified local improvements against the completed baseline.',
            f"{sum(row['changingAbsoluteEligible'] for row in candidateRows)} changed new-field outputs also satisfy the original whole-pipeline gates.",
            'Exact old compact replay, raw refinement, last rejected proposal and delivered baseline fallback are reported separately.',
            f"{sum(row['newFieldSolverAttribution']['allChangedOuterProposalsRejectedByFullPairsBeforeMask'] for row in cases.values())} new-field cases lose their nonzero proposals at full-scene collision gates before mask measurement; {sum(row['newFieldSolverAttribution']['noEffectiveInnerChange'] for row in cases.values())} have no selected effective inner change.",
            f"The final effect gate actively rejects {sum(row['finalEffectActivelyRejectedAcceptedRawRefinement'] for row in candidateRows)} already accepted nonzero raw refinements; it separately confirms the lack of improvement in {sum(row['alreadyUnchangedBeforeFinalEffect'] for row in candidateRows)} already unchanged results.",
            'Inherited upstream geometry and complete-assembly failures remain visible; incremental safety does not erase them.',
            'The basis uses original triangle adjacency and exact fixed rows; it does not certify anatomical shape or collision freedom.'],
        limitations=['The same template materials isolate geometry; eyebrow and nostril photo texture quality is not evaluated.',
            'Automatic photo contours are proxies; nominal millimeters use the template scale rather than a metric scan.',
            'Previously inspected regression cases are not independent unseen validation.',
            'A higher active count or rank demonstrates representation capacity, not photographic resemblance.',
            'Coefficient regularization is basis-dependent even with the original weight unchanged.',
            'Displacement and strain regularization retain their formulas and weights, but divide by active vertex/edge counts; the wider active domain changes these denominators. Outcomes cannot be uniquely attributed to the field representation alone.'])
    verify(proof); verify(solverAudit['inputHashes'])
    writeJson(OUT/'result.json', result)
    writeViewer(result)
    result['viewerSha256'] = sha(OUT/'index.html')
    writeJson(OUT/'result.json', result)
    return result


def writeViewer(result):
    """All plots remain delivered outputs; raw rejected states are labeled data."""
    writeJson(OUT/'viewer.json', dict(cases=list(result['cases']), complete=list(result['cases']),
        labels=LABELS, branches=list(BRANCHES), result='result.json',
        geometrySafetyAccepted={case: {branch: item['absoluteEligible'] for branch, item in row['branches'].items()}
            for case, row in result['cases'].items()}))
    (OUT/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>固定边界平滑场 · 七组完整对照</title>
<style>body{background:#171c23;color:#e8eef6;font:16px system-ui,sans-serif;margin:20px;line-height:1.65}select{font:inherit;padding:6px;background:#283240;color:white;border:1px solid #61738a;margin:4px}img{display:block;max-width:100%;margin:16px 0}a{color:#7bc8ff}table{border-collapse:collapse;max-width:100%;font-size:14px}td,th{padding:7px 12px;border:1px solid #495463;text-align:left}section{padding:12px;background:#252c35;margin:16px 0}.muted{color:#b8c3d1}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}</style>
<h2>固定边界平滑形变场 · 统一七组试验</h2>
<p>从左到右：原照片、完成粗对齐的共同基线、旧共用半径精确重放、固定边界平滑场加接触。全部保留原相机、顶点顺序、绑定和固定附件。七组采用统一参数及原安全门。</p>
<section id="experimentOutcome">正在加载已验证结果…</section>
<p class="muted">图中展示最终输出。下方区分“尝试过的提案”“通过前段安全与有效改善检查的精修”和“最终输出”；前段已经返回基线时，最终效果门只确认没有改善，不代表它主动否决了明显变好的模型。三个模型使用共同模板材质，本页不评价眉毛、鼻翼的照片贴图，也不能证明客户端最终资产质量。</p>
<p class="muted">本轮保留原正则公式和权重，但位移项按活动顶点数、应变项按活动边数归一；新场的活动域增大时分母也会改变。控制系数惩罚同样依赖基函数，因此效果变化不能只归因于形变场本身。</p>
<label>样本<select id="case"></select></label>
<label>视角<select id="view"><option value="0">1</option><option value="1">2</option><option value="2">3</option><option value="3">4</option></select></label>
<label>显示<select id="mode"><option value="clay">灰模</option><option value="lit">共同光照</option><option value="baseColor">共同模板基色</option></select></label>
<a id="previous" target="_blank" rel="noopener">返回上一轮公共半径对照</a>
<section id="caseOutcome"></section><img id="main" alt="原照片与三分支最终输出"><img id="eyes" alt="相同基线区域的眼周放大">
<details><summary>当前样本：原始精修、输出、形变容量及继承失败的完整数据</summary><pre id="details"></pre></details>
<script>
const $=id=>document.getElementById(id),qs=new URLSearchParams(location.search);
const labels={baseline:'完成粗对齐基线',legacyResidualReplay:'旧共用半径精确重放',boundaryFieldContact:'固定边界平滑场＋接触'};
const yn=v=>v?'通过':'未通过',num=v=>Number.isFinite(v)?v.toFixed(4):'无有效测量';
fetch('result.json').then(r=>{if(!r.ok)throw Error('结果读取失败');return r.json()}).then(data=>{
  $('experimentOutcome').textContent=`完整矩阵：7 组 × 3 分支 × 4 视角，420 张图片。新场相对共同基线的已验证局部改善：${data.candidateImprovedCaseCount}/7；其中改变了几何且通过原整体验收：${data.candidateChangingAbsoluteEligibleCount}/7。旧公共半径精确重放：${data.legacyReplayMatchedCaseCount}/7。未修改生产默认配置。`;
  const link=document.createElement('a');link.href='result.json';link.textContent='完整结果 JSON';$('experimentOutcome').append(document.createElement('br'),link);
  const cause=document.createElement('p');cause.textContent=`拒绝发生的位置：${data.candidateBeforeMaskCollisionRejectedCaseCount} 组产生了非零内部解，但所有改变后的外层提案在完整面片相交门被拒绝，尚未测量其完整眼孔效果；${data.candidateNoEffectiveInnerChangeCaseCount} 组没有选出达到原改善门槛的有效内部形变。最终效果门主动否决已通过前段检查的非零精修：${data.candidateFinalEffectActiveRejectionCount} 组；对前段已经不变的结果作确认：${data.candidateAlreadyUnchangedBeforeFinalEffectCount} 组。`;$('experimentOutcome').append(cause);
  const auditLink=document.createElement('a');auditLink.href=data.savedSolverAudit.relativeUrl;auditLink.textContent='求解与拒绝阶段的保存证据';$('experimentOutcome').append(auditLink);
  for(const c of Object.keys(data.cases))$('case').add(new Option(c,c));
  for(const k of ['case','view','mode'])if(qs.has(k)&&Array.from($(k).options).some(o=>o.value===qs.get(k)))$(k).value=qs.get(k);
  function update(){
    const c=$('case').value,v=$('view').value,m=$('mode').value,row=data.cases[c];
    history.replaceState(null,'','?'+new URLSearchParams({case:c,view:v,mode:m}));
    $('main').src=c+'/view'+v.padStart(2,'0')+'_'+m+'.png';$('eyes').src=c+'/view'+v.padStart(2,'0')+'_'+m+'_eyes.png';
    $('previous').href='http://127.0.0.1:18099/?'+new URLSearchParams({case:c,view:v,mode:m});
    $('caseOutcome').replaceChildren();
    const note=document.createElement('p');note.textContent='局部改善、增量安全与原整体验收分别判断。未变化的基线即使通过几何门，也不记作优化成功。';$('caseOutcome').append(note);
    const table=document.createElement('table'),head=document.createElement('tr');
    for(const label of ['分支','输出改变顶点','完整眼孔效果','局部改善','增量安全','原整体验收','输出平均 IoU','最大位移（名义 mm）']){const cell=document.createElement('th');cell.textContent=label;head.append(cell)}table.append(head);
    for(const [branch,b] of Object.entries(row.branches)){
      const tr=document.createElement('tr');const values=[labels[branch],b.delivered.changedVertexCount,branch==='baseline'?'基准':yn(b.requestedEffectAccepted),b.improvementAccepted?'已验证':'未成立',yn(b.incrementalSafe),yn(b.absoluteEligible),num(b.deliveredApertureAfter.meanIntersectionOverUnion),num(b.delivered.maximumDisplacementNominalMm)];
      for(const value of values){const cell=document.createElement('td');cell.textContent=value;tr.append(cell)}table.append(tr);
    }$('caseOutcome').append(table);
    const b=row.branches.boundaryFieldContact,a=row.newFieldSolverAttribution,p=document.createElement('p');
    let decision=b.finalEffectActivelyRejectedAcceptedRawRefinement?'前段已接受非零精修，但最终完整眼孔效果门未通过，因此退回基线。':!b.rawRefinementAccepted?'前段安全或有效改善检查已返回基线；最终效果门只确认没有新增改善。':'前段精修通过，最终效果门未触发额外回退。';
    if(a.allChangedOuterProposalsRejectedByFullPairsBeforeMask)decision+=' 本样本的改变提案均被完整面片相交门拒绝，尚未进行这些提案的完整眼孔测量。';
    else if(a.noEffectiveInnerChange)decision+=' 内部虽可能接受极小步骤，但未选出达到原最小改善要求的形变（bestIteration=0）。';
    p.textContent=`新场：尝试的提案最多改变 ${b.proposalChanges.maximumChangedVertexCount} 个顶点；最后一次提案改变 ${b.rawLastProposal.changedVertexCount} 个；前段检查后精修改变 ${b.rawRefinement.changedVertexCount} 个，最终输出改变 ${b.delivered.changedVertexCount} 个。${decision}形变域参与量（各轮）: ${b.stages.map(s=>`${s.activeVertexCount}/${s.allowedMovableVertexCount}，秩 ${s.numericalRank}/${s.controlCount}`).join('；')||'未执行求解'}。`;
    $('caseOutcome').append(p);$('details').textContent=JSON.stringify(row,null,2);
    const caution=document.createElement('p');caution.className='muted';caution.textContent='相交面片对的变化说明触发了现有安全门，不能直接解释为可见穿透或穿透加深；不同回退尺度会重复记录同一面片对。眼球、眼角附件与透明面片的安全门均保留。';$('caseOutcome').append(caution);
  }
  for(const k of ['case','view','mode'])$(k).onchange=update;update();
}).catch(error=>{$('experimentOutcome').textContent='结果尚不可核验：'+error.message});
</script></html>''', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true', help='Read completion state without writing summary or viewer')
    arguments = parser.parse_args()
    if arguments.status:
        print(json.dumps(completionStatus(), ensure_ascii=False))
    else:
        result = summarize()
        print(json.dumps({key: result[key] for key in ('status', 'caseCount', 'branchCount', 'cameraStates',
            'imageCount', 'candidateImprovedCaseCount', 'candidateChangingAbsoluteEligibleCount', 'legacyReplayMatchedCaseCount')}))
