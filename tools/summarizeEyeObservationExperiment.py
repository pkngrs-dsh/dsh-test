"""Summarize the completed observation-definition ablation and independent audit.

This tool never fits, renders or promotes geometry. It refuses to produce a
final result until all seven independent audits and 420 render images exist.
Invalid observation mappings remain explicit, unevaluated candidate policies.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.artifacts import writeJson
from summarizeBoundaryFieldExperiment import (
    read, sha, arrays, verify, readEvidence, flag, vertexChange, apertureSummary, inheritedFailureSummary,
)

RUN = ROOT / 'runs/eyeObservation20261024'
OUT = ROOT / 'reports/eyeObservation20261024'
BRANCHES = ('baseline', 'boundaryReplay', 'unifiedObservationContact')
LABELS = {'baseline': '完成粗对齐基线', 'boundaryReplay': '原观测约束精确重放',
    'unifiedObservationContact': '统一眼睑观测诊断'}
MODES = ('clay', 'baseColor', 'lit')


def expectedImages():
    names = set()
    for view in range(4):
        for mode in MODES:
            names.update((f'view{view:02d}_{mode}.png', f'view{view:02d}_{mode}_eyes.png'))
            names.update(f'view{view:02d}_{branch}_{mode}.png' for branch in BRANCHES)
    return names


def completionStatus():
    planPath = RUN / 'experimentPlan.json'
    if not planPath.is_file():
        return dict(status='notFrozen', missing=[str(planPath)])
    config = read(planPath)
    required = [RUN / 'implementationHashes.json', RUN / 'finalUnitTests.json',
        OUT / 'independentAudit/auditImplementation.json', OUT / 'localization/summary.json',
        OUT / 'localization/originalResolution/summary.json']
    for case in config['cases']:
        required.extend([RUN / case / 'report.json', RUN / case / 'observationAlignment.json',
            RUN / case / 'observationAlignment.npz', OUT / case / 'report.json',
            OUT / 'independentAudit' / f'{case}.json', OUT / 'localization' / case / 'report.json'])
        required.extend(OUT / case / name for name in expectedImages())
        for branch in BRANCHES:
            required.extend(RUN / case / branch / name for name in ('report.json', 'fullGeometry.npz',
                'rawRefinement.json', 'rawRefinement.npz', 'solverEvidence.json', 'solverEvidence.npz', 'finalEffectGate.json'))
    missing = [str(path) for path in required if not path.is_file()]
    return dict(status='readyForVerification' if not missing else 'incomplete',
        caseCount=len(config['cases']), missing=missing)


def solverTrace(captures, rawDiagnostics, unit):
    """Read saved GN decisions without inventing a new independent solver audit."""
    stages = {stage['round']: stage for stage in rawDiagnostics.get('fast', {}).get('stages', [])}
    result = []
    for captured in captures:
        evidence = captured['evidence']
        stage = stages[captured['round']]
        solver = stage['solver']
        decisions, rejected = Counter(), Counter()
        for trial in evidence['trials']:
            decisions[trial['decision']] += 1
            safety = trial['safety']
            if not safety['geometricSafetyAccepted']:
                rejected['ordinaryGeometry'] += 1
            extra = safety.get('additionalSafety')
            if extra is not None:
                if not extra['coarseAlignment']['accepted']:
                    rejected['activeCoarseObservation'] += 1
                if extra.get('contact') is not None and not extra['contact']['accepted']:
                    rejected['contactEnvelope'] += 1
        change = vertexChange(evidence['outputVertices'], evidence['referenceVertices'], unit)
        result.append(dict(round=captured['round'], status=stage['status'], selectedScale=stage.get('selectedScale'),
            innerOutputChange=change, bestIteration=solver['bestIteration'],
            initialPointErrorWorkingPixels=solver['initialMeanErrorPixels'],
            finalPointErrorWorkingPixels=solver['finalMeanErrorPixels'],
            activeVertexCount=solver['activeVertexCount'], controlCount=solver['controlCount'],
            trialCount=len(evidence['trials']), trialDecisions=dict(decisions),
            overlappingSafetyRejections=dict(rejected), castTrialCount=len(evidence['castTrials']),
            castRejected=bool(evidence['castRejected']),
            meaning='Saved solver trace; original-policy counterfactual recomputation is retained in the independent audit. Rejection categories overlap.'))
    return result


def outerTrace(proposals, baseline, unit):
    rows = []
    for proposal in proposals:
        strict = proposal['strict']
        external = strict['externalCandidateGate']
        attempt = proposal['solverAttempt']
        original = external.get('originalPolicy')
        rows.append(dict(round=proposal['round'], scale=proposal['scale'],
            change=vertexChange(proposal['vertices'], baseline, unit),
            selectedByFastRound=bool(proposal['selectedByFastRound']),
            internalGeometry=proposal['internalGeometry'], strictAccepted=flag(strict, 'accepted'),
            reasons=strict['reasons'], newFullPairCount=len(strict['fullAssembly']['newPairs']),
            newFullPairs=strict['fullAssembly']['newPairs'],
            activeCoarseAccepted=flag(strict['coarseAlignment'], 'accepted'),
            activePhotoGate=external['legacyGate'], originalPolicyAudit=original,
            contactAccepted=None if strict['contact'] is None else flag(strict['contact'], 'accepted'),
            completeApertureMeasured='apertureMetrics' in attempt,
            completeApertureAccepted=attempt.get('apertureGate'),
            completeAperture=apertureSummary(attempt.get('apertureMetrics')),
            sphereContact=attempt.get('eyeContact'), subsequentIntersection=attempt.get('intersection'),
            gateOrder='Internal/strict geometry, full pairs and external gates precede complete-mask measurement; sphere and subsequent intersection gates follow mask benefit.'))
    return rows


def alignmentSummary(saved):
    valid = flag(saved, 'valid')
    if not valid:
        if saved['table'] is not None or saved['coarseAlignment'] is not None:
            raise ValueError('Invalid mapping must not contain a usable target table')
        return dict(valid=False, error=saved['error'], tableSha256=None, candidatePolicyStatus='notEvaluated')
    table = saved['table']
    change = np.linalg.norm(table['targetPixels'] - table['originalTargetPixels'], axis=1)
    return dict(valid=True, error=None, tableSha256=table['tableSha256'],
        ibugIds=table['ibugIds'], targetDisplacementsWorkingPixels=change,
        meanTargetDisplacementWorkingPixels=float(change.mean()), maximumTargetDisplacementWorkingPixels=float(change.max()),
        originalTargetPixels=table['originalTargetPixels'], mappedTargetPixels=table['targetPixels'],
        evidence=table['evidence'], manualTruthEstablished=False,
        policy='Interior points preserve dense eye-axis x; corners explicitly move to endpoint midpoints; no repairs or extrapolation.')


def localizationEvidence(cases, proof):
    root = OUT / 'localization'
    summaryPath = root / 'summary.json'
    summary = read(summaryPath)
    if summary['status'] != 'complete' or {row['caseId'] for row in summary['cases']} != set(cases):
        raise ValueError('Cached localization audit is incomplete')
    nativeRoot = root / 'originalResolution'
    nativeSummaryPath, nativePlanPath = nativeRoot / 'summary.json', nativeRoot / 'experimentPlan.json'
    nativeSummary, nativePlan = read(nativeSummaryPath), read(nativePlanPath)
    if nativeSummary['status'] != 'complete' or nativeSummary['planSha256'] != sha(nativePlanPath):
        raise ValueError('Native-resolution diagnostic is incomplete')
    verify(nativePlan['sourceHashes'])
    for source, snapshot in nativePlan['implementation'].items():
        if sha(snapshot['snapshotPath']) != snapshot['sha256']:
            raise ValueError('Native-resolution implementation snapshot changed')
        proof[snapshot['snapshotPath']] = snapshot['sha256']
    nativeRows = {row['caseId']: row for row in nativeSummary['cases']}
    if set(nativeRows) != set(cases):
        raise ValueError('Native-resolution availability matrix differs')
    proof.update({str(path): sha(path) for path in (summaryPath, nativeSummaryPath, nativePlanPath)})
    rows = {}
    for item in summary['cases']:
        case = item['caseId']
        reportPath = root / case / 'report.json'
        report = read(reportPath)
        if sha(reportPath) != item['reportSha256'] or report['manualTruthEstablished'] is not False:
            raise ValueError('Cached localization evidence changed or overclaims truth')
        verify(report['sourceHashes'])
        verify({str(root / case / name): expected for name, expected in report['images'].items()})
        proof[str(reportPath)] = sha(reportPath)
        native = nativeRows[case]
        available = native['status'] == 'diagnosticComplete'
        nativeReport = None
        if available:
            nativeReportPath = nativeRoot / case / 'report.json'
            nativeReport = read(nativeReportPath)
            if sha(nativeReportPath) != native['reportSha256'] or nativeReport['selectedForFit'] is not False:
                raise ValueError('Native observations were changed or selected for fitting')
            verify({str(nativeRoot / case / name): expected for name, expected in nativeReport['outputHashes'].items()})
            proof[str(nativeReportPath)] = sha(nativeReportPath)
        elif native['status'] != 'missingExactLocalOriginal' or native['newInferenceCount'] != 0:
            raise ValueError('Missing native original was not explicitly retained')
        rows[case] = dict(workingOverlay=f'localization/{case}/workingComparison.png',
            cachedOriginalOverlay=f'localization/{case}/originalComparison.png' if report['originalStatus'] == 'exactBytesAndOrientedSizeVerified' else None,
            originalResolutionOverlay=f'localization/originalResolution/{case}/comparison.png' if available else None,
            cachedReport=f'localization/{case}/report.json', nativeReport=f'localization/originalResolution/{case}/report.json' if available else None,
            originalStatus=report['originalStatus'], workingScale=report['workingScale'],
            originalOrientedSize=report['originalOrientedSize'], nativeInferenceAvailable=available,
            nativeDiagnostic=nativeReport, cachedEyes=report['eyes'], manualTruthEstablished=False)
    return rows


def summarize():
    ready = completionStatus()
    if ready['status'] != 'readyForVerification':
        raise ValueError(f'Experiment incomplete; no final result written ({len(ready["missing"])} missing artifacts)')
    planPath, implementationPath = RUN / 'experimentPlan.json', RUN / 'implementationHashes.json'
    config, implementation = read(planPath), read(implementationPath)
    planHash = sha(planPath)
    if implementation['planSha256'] != planHash or tuple(config['branches']) != BRANCHES or len(config['cases']) != 7:
        raise ValueError('Plan, branch matrix or implementation identity differs')
    verify(implementation['sourceHashes'])
    verify(config['historicalHashes'])
    auditManifestPath = OUT / 'independentAudit/auditImplementation.json'
    auditManifest = read(auditManifestPath)
    if auditManifest['planSha256'] != planHash:
        raise ValueError('Independent audit belongs to a different plan')
    verify(auditManifest['hashes'])
    validatorPath = ROOT / 'tools/validateEyeObservationExperiment.py'
    validatorHash = auditManifest['hashes'][str(validatorPath)]
    proof = {str(path): sha(path) for path in (planPath, implementationPath, auditManifestPath,
        ROOT / 'tools/summarizeBoundaryFieldExperiment.py')}
    localization = localizationEvidence(config['cases'], proof)
    diagnosticRoot = OUT / 'rejectedApertureDiagnostic'
    diagnosticPlan = read(diagnosticRoot / 'diagnosticPlan.json')
    rejectedDiagnostic = read(diagnosticRoot / 'summary.json')
    if (diagnosticPlan['sourceExperimentPlanSha256'] != planHash
            or diagnosticPlan['cases'] != config['cases']
            or rejectedDiagnostic['diagnosticPlanSha256'] != sha(diagnosticRoot / 'diagnosticPlan.json')
            or rejectedDiagnostic['noFittingOrOutputModification'] is not True
            or rejectedDiagnostic['allFrozenInputsUnchanged'] is not True):
        raise ValueError('Rejected-proposal diagnostic identity differs')
    verify(diagnosticPlan['inputHashes'])
    verify(diagnosticPlan['sourceHashes'])
    diagnosticOutputs = {str(diagnosticRoot / key): value for key, value in read(diagnosticRoot / 'outputHashes.json').items()}
    verify(diagnosticOutputs)
    proof.update(diagnosticOutputs)
    cases, imageCount = {}, 0
    for case in config['cases']:
        directory = RUN / case
        contextPath = directory / 'inputContext.json'
        context = read(contextPath)
        verify(context['sourceHashes'])
        proof[str(contextPath)] = sha(contextPath)
        caseReportPath = directory / 'report.json'
        caseReport = read(caseReportPath)
        if caseReport['status'] != 'complete' or caseReport['planSha256'] != planHash:
            raise ValueError('Case is incomplete')
        verify(caseReport['preparationHashes'])
        proof[str(caseReportPath)] = sha(caseReportPath)
        alignment = alignmentSummary(readEvidence(directory / 'observationAlignment'))
        auditPath = OUT / 'independentAudit' / f'{case}.json'
        audit = read(auditPath)
        if not flag(audit, 'validationPassed') or audit['validatorSha256'] != validatorHash:
            raise ValueError(f'Independent validation missing or failed: {case}')
        proof[str(auditPath)] = sha(auditPath)
        renderPath = OUT / case / 'report.json'
        render = read(renderPath)
        verify(render['inputHashes'])
        verify(render['renderImplementationHashes'])
        if set(render['images']) != expectedImages() or len(render['views']) != 4:
            raise ValueError('Render matrix incomplete')
        verify({str(OUT / case / name): expected for name, expected in render['images'].items()})
        for key in ('identicalMaterials', 'allFourCameras', 'templateColorsOnly', 'nonSkinAndStaticExactAcrossArms', 'vertexOrderUvBindingUnchanged'):
            if not flag(render, key):
                raise ValueError(f'Render identity failed: {case}/{key}')
        proof[str(renderPath)] = sha(renderPath)
        imageCount += len(render['images'])
        baseline = arrays(directory / 'baseline/fullGeometry.npz')['vertices']
        unit = float(arrays(context['entry']['preparedGeometry'])['unit'])
        branches = {}
        for branch in BRANCHES:
            branchDir = directory / branch
            reportPath = branchDir / 'report.json'
            report = read(reportPath)
            branchAudit = audit['branches'][branch]
            if (report['caseId'] != case or report['branch'] != branch or not flag(branchAudit, 'validationPassed')
                    or not flag(branchAudit, 'evidenceValidated') or branchAudit['reportSha256'] != sha(reportPath)):
                raise ValueError('Independent branch report identity differs')
            verify({str(branchDir / name): expected for name, expected in report['outputHashes'].items()})
            raw = readEvidence(branchDir / 'rawRefinement')
            captures = readEvidence(branchDir / 'solverEvidence')
            delivered = arrays(branchDir / 'fullGeometry.npz')['vertices']
            change = vertexChange(delivered, baseline, unit)
            invalid = branch == 'unifiedObservationContact' and not alignment['valid']
            if report['observationAlignmentValid'] != alignment['valid']:
                raise ValueError('Branch mapping validity differs from saved observation table')
            active = report['activeObservationPolicyAccepted']
            if invalid:
                if (active is not None or captures or raw['candidateProposals'] or change['geometryChanged']
                        or not np.array_equal(raw['vertices'], baseline) or report['absoluteEligible']
                        or report['observationPolicyChangedForDiagnostic']):
                    raise ValueError('Invalid observation mapping was used or relabeled as passed')
                candidatePolicyStatus = 'notEvaluatedInvalidObservationMapping'
            else:
                if type(active) is not bool:
                    raise ValueError('Evaluated policy must have an explicit boolean decision')
                candidatePolicyStatus = 'passed' if active else 'failed'
            effect = report['finalEffectGate']
            if effect != read(branchDir / 'finalEffectGate.json'):
                raise ValueError('Final effect identity differs')
            effectAccepted = flag(effect, 'accepted')
            accepted = flag(report, 'residualCandidateAccepted')
            incremental, absolute = flag(report, 'incrementalSafe'), flag(report, 'absoluteEligible')
            original = flag(report, 'originalObservationPolicyAccepted')
            improvement = bool(not invalid and branch != 'baseline' and change['geometryChanged'] and effectAccepted and incremental and accepted)
            changingAbsolute = improvement and absolute
            expected = dict(producerAccepted=accepted, geometryChanged=change['geometryChanged'], incrementalSafe=incremental,
                absoluteEligible=absolute, requestedEffectAccepted=effectAccepted, improvementAccepted=improvement,
                changingAbsoluteEligible=changingAbsolute, clientReady=False)
            if any(flag(branchAudit['classification'], key) != value for key, value in expected.items()):
                raise ValueError('Independent classification differs from saved result')
            if (report['improvementAccepted'] != improvement or report['changingAbsoluteEligible'] != changingAbsolute
                    or report['geometryChanged'] != change['geometryChanged'] or accepted != change['geometryChanged']):
                raise ValueError('Producer acceptance or geometry identity differs')
            if (branchAudit['classification']['originalObservationPolicyAccepted'] != original
                    or branchAudit['classification']['activeObservationPolicyAccepted'] != active):
                raise ValueError('Independent original/active observation policies differ')
            fallback = flag(effect, 'fallbackToExactBaseline')
            if fallback and change['geometryChanged']:
                raise ValueError('Fallback was not exact completed baseline')
            replay = report['replayMatch']
            if branch == 'boundaryReplay':
                if replay != read(branchDir / 'replayMatch.json') or not flag(replay, 'accepted'):
                    raise ValueError('Previous fixed-boundary solve did not replay exactly')
            stages = solverTrace(captures, raw['diagnostics'], unit)
            proposals = outerTrace(raw['candidateProposals'], baseline, unit)
            rawChange = vertexChange(raw['vertices'], baseline, unit)
            entry = dict(status=report['diagnostics']['status'], invalidObservationMapping=invalid,
                candidatePolicyStatus=candidatePolicyStatus, observationAlignmentError=report['observationAlignmentError'],
                originalObservationPolicyAccepted=original, activeObservationPolicyAccepted=active,
                observationPolicyChangedForDiagnostic=report['observationPolicyChangedForDiagnostic'],
                originalCoarseAlignment=report['originalCoarseAlignment'], activeCoarseAlignment=None if invalid else report['activeCoarseAlignment'],
                originalPhotoGate=report['legacyObservationGate'], activePhotoGate=None if invalid else report['activeLegacyObservationGate'],
                contactSafety=report['contactSafety'], inheritedAndOriginalAnchorGates=inheritedFailureSummary(report),
                delivered=change, rawRefinement=rawChange,
                rawFastSolver=vertexChange(raw['solverVertices'], baseline, unit),
                lastOuterProposal=vertexChange(raw['rawCandidateVertices'], baseline, unit),
                solverStages=stages, outerProposals=proposals,
                innerSolverChanged=any(stage['innerOutputChange']['geometryChanged'] for stage in stages),
                changedOuterProposalCount=sum(item['change']['geometryChanged'] for item in proposals),
                fullPairRejectedOuterProposalCount=sum(item['newFullPairCount'] > 0 for item in proposals),
                maskMeasuredOuterProposalCount=sum(item['completeApertureMeasured'] for item in proposals),
                rawRefinementAccepted=flag(report, 'rawRefinementAccepted'), deliveredResidualAccepted=accepted,
                incrementalSafe=incremental, absoluteEligible=absolute, requestedEffectAccepted=effectAccepted,
                diagnosticImprovementAccepted=improvement, changingAbsoluteEligible=changingAbsolute,
                fallbackToExactBaseline=fallback,
                finalEffectActivelyRejectedAcceptedRaw=bool(fallback and report['rawRefinementAccepted'] and rawChange['geometryChanged']),
                apertureBefore=apertureSummary(effect['baseline']), rawApertureAfter=apertureSummary(effect['candidate']),
                deliveredApertureAfter=apertureSummary(effect['baseline'] if fallback else effect['candidate']),
                finalEffectGate=effect, independentClassification=branchAudit['classification'],
                independentRawEffect=branchAudit['finalEffectGate'], independentDeliveredEffect=branchAudit['finalOutputEffect'],
                independentAuditRelativeUrl=f'independentAudit/{case}.json',
                replayMatched=None if replay is None else replay['accepted'],
                allOutsideSupportExactlyFixed=flag(report, 'allOutsideSupportExactlyFixed'),
                strictLegacyBindingAndPayloadPreserved=True, clientReady=False)
            if not entry['allOutsideSupportExactlyFixed']:
                raise ValueError('Fixed support or attachment changed')
            branches[branch] = entry
            proof[str(reportPath)] = sha(reportPath)
        cases[case] = dict(branches=branches, observationAlignment=alignment, localization=localization[case],
            independentEvidenceVerified=True, renderedImages=len(render['images']),
            referenceViewIndex=context['referenceSelection']['referenceViewIndex'])
    if imageCount != 420:
        raise ValueError('Expected 420 render images')
    testPath = RUN / 'finalUnitTests.json'
    tests = read(testPath)
    if tests['status'] != 'passed' or int(tests['passed']) < 1 or sha(tests['logPath']) != tests['logSha256']:
        raise ValueError('Implementation tests missing or changed')
    proof[str(testPath)] = sha(testPath)
    candidates = [case['branches']['unifiedObservationContact'] for case in cases.values()]
    result = dict(schemaVersion='unifiedEyeObservationSummaryV1', status='complete', cases=cases,
        caseCount=7, branchCount=21, cameraStates=84, imageCount=imageCount,
        invalidObservationMappingCount=sum(row['invalidObservationMapping'] for row in candidates),
        innerSolverChangedCaseCount=sum(row['innerSolverChanged'] for row in candidates),
        deliveredChangedCaseCount=sum(row['delivered']['geometryChanged'] for row in candidates),
        diagnosticImprovedCaseCount=sum(row['diagnosticImprovementAccepted'] for row in candidates),
        changingAbsoluteEligibleCaseCount=sum(row['changingAbsoluteEligible'] for row in candidates),
        activePolicyPassedCaseCount=sum(row['activeObservationPolicyAccepted'] is True for row in candidates),
        originalPolicyPassedCaseCount=sum(row['originalObservationPolicyAccepted'] for row in candidates),
        exactReplayCaseCount=sum(case['branches']['boundaryReplay']['replayMatched'] is True for case in cases.values()),
        candidateImprovedCases=[case for case, row in cases.items() if row['branches']['unifiedObservationContact']['diagnosticImprovementAccepted']],
        promotedCandidates=0, productionDefaultsChanged=False, textureRebaked=False, realLipsRefined=False,
        finalAdaptiveAssemblyValidated=False, manualTruthEstablished=False, planSha256=planHash,
        rejectedApertureDiagnostic=dict(aggregate=rejectedDiagnostic['aggregate'], report='rejectedApertureDiagnostic/summary.json',
            viewer='rejectedApertureDiagnostic/index.html', diagnosticOnly=True, affectsCandidateAcceptance=False),
        unitTests=tests, independentAuditTests=auditManifest.get('tests'), independentValidatorSha256=validatorHash,
        testCountInterpretation='Suites may overlap; source and independent test counts are separate.',
        evidenceHashes=proof, summaryImplementationSha256=sha(__file__),
        previousViewerUrl='http://127.0.0.1:18100/',
        conclusions=['Only reference-eye observation definitions differ between fitting arms; original guards remain a separate audit.',
            'Inner solver changes, outer safety checks, full-aperture improvement and delivered geometry are distinct stages.',
            'Invalid or ambiguous observation mapping skips the candidate policy, never counts as a passed observation gate.',
            'No automatic contour or crop stability check establishes anatomical ground truth.'],
        limitations=['Original-resolution inference remains a diagnostic and never selects fitting targets.',
            'The same template materials isolate geometry; eyebrow and nostril appearance are not evaluated.',
            'Complete face-pair intersections are conservative geometric gates, not a measured depth or visual severity scale.',
            'Previously inspected cases do not establish unseen generalization; nominal millimeters are template-derived.'])
    verify(proof)
    writeJson(OUT / 'result.json', result)
    writeViewer(result)
    result['viewerSha256'] = sha(OUT / 'index.html')
    writeJson(OUT / 'result.json', result)
    return result


def writeViewer(result):
    writeJson(OUT / 'viewer.json', dict(cases=list(result['cases']), complete=list(result['cases']), labels=LABELS,
        branches=list(BRANCHES), result='result.json'))
    (OUT / 'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>统一眼睑观测 · 七组核验</title>
<style>body{background:#192129;color:#edf4fa;font:16px system-ui,sans-serif;margin:20px;line-height:1.7}a{color:#8dd8ff}select{font:inherit;padding:6px;background:#293440;color:white;margin:4px}img{display:block;max-width:100%;margin:18px 0}section{padding:14px;background:#252f39;margin:18px 0}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #53606d;padding:7px 10px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}.muted{color:#bcc8d3}summary{cursor:pointer}.scroll{overflow-x:auto}</style>
<h2>统一眼睑观测 · 同一七组、四视角</h2>
<p>从左到右：照片、完成粗对齐基线、原观测约束精确重放、统一眼睑观测诊断。两条精修分支使用相同形变场、相机、完整眼裂目标、接触及相交约束。候选统一了粗点和参考眼部约束中的观测定义；原门仍独立复核。几何门通过与眼裂形状改善分别显示。</p>
<section id="overall">等待已核验结果…</section>
<label>样本<select id="case"></select></label><label>视角<select id="view"><option value="0">1</option><option value="1">2</option><option value="2">3</option><option value="3">4</option></select></label><label>显示<select id="mode"><option value="clay">灰模</option><option value="lit">共同光照</option><option value="baseColor">共同模板基色</option></select></label>
<p><a id="previous">上一轮形变场对照</a> · <a id="localizationLink">定位核验数据</a> · <a id="audit">独立复核</a> · <a href="result.json">完整结果</a> · <a href="rejectedApertureDiagnostic/index.html">被拒绝中间形状的离线诊断（不用于交付）</a> · <a href="solverAudit/summary.json">求解阻挡原因统计</a></p>
<section id="state"></section><img id="main" alt="照片和三分支最终输出"><img id="eyes" alt="固定眼区的最终输出对照">
<details open><summary>照片定位核验：粗点、稠密睑缘和模型可见眼裂</summary><p id="localizationState"></p><img id="localization" alt="相同眼区上的照片与三种轮廓"><img id="original" alt="原图上的冻结预测逆映射"></details>
<details><summary>同一检测器的原分辨率诊断</summary><p id="nativeState"></p><img id="native" alt="相同锚点的工作图预测与原图预测"></details>
<details><summary>求解阶段、外层提案、旧门和新门的完整数据</summary><pre id="data"></pre></details>
<p class="muted">全部观测是自动代理，没有人工真值。原图诊断没有用于择优或改写拟合目标。主图展示最终输出；被拒绝的中间解另列为离线诊断，不能作为已通过检查的资产。共同模板贴图不评价眉毛或鼻翼的照片贴图质量；本轮不代表完成客户端交付。</p>
<script>
const $=id=>document.getElementById(id),qs=new URLSearchParams(location.search),labels={baseline:'完成粗对齐基线',boundaryReplay:'原观测精确重放',unifiedObservationContact:'统一眼睑观测诊断'};
const yn=v=>v===null?'未评估':v?'通过':'未通过',n=v=>Number.isFinite(v)?v.toFixed(4):'无测量';
fetch('result.json').then(r=>{if(!r.ok)throw Error('结果尚未生成');return r.json()}).then(result=>{
 $('overall').textContent=`已核验：7 组 × 3 分支 × 4 视角，420 张图。统一观测的诊断改善 ${result.diagnosticImprovedCaseCount}/7；其中改变几何且通过原整体验收 ${result.changingAbsoluteEligibleCaseCount}/7。原分支精确重放 ${result.exactReplayCaseCount}/7；观测映射无效而跳过 ${result.invalidObservationMappingCount}/7。未修改生产默认配置。离线测量中，${result.rejectedApertureDiagnostic.aggregate.changedProposalCount} 个改变形状的中间提案里，${result.rejectedApertureDiagnostic.aggregate.physicallyRejectedWithHypotheticalMaskEffectCount} 个满足自动眼裂指标，但仍被原相交门拒绝。`;
 for(const c of Object.keys(result.cases))$('case').add(new Option(c,c));for(const key of ['case','view','mode'])if(qs.has(key)&&Array.from($(key).options).some(o=>o.value===qs.get(key)))$(key).value=qs.get(key);
 function show(){const c=$('case').value,v=$('view').value,m=$('mode').value,row=result.cases[c],b=row.branches.unifiedObservationContact,l=row.localization;history.replaceState(null,'','?'+new URLSearchParams({case:c,view:v,mode:m}));$('main').src=c+'/view'+v.padStart(2,'0')+'_'+m+'.png';$('eyes').src=c+'/view'+v.padStart(2,'0')+'_'+m+'_eyes.png';$('previous').href=result.previousViewerUrl+'?'+new URLSearchParams({case:c,view:v,mode:m});$('localizationLink').href=l.cachedReport;$('audit').href='independentAudit/'+c+'.json';$('state').replaceChildren();
 const note=document.createElement('p');note.textContent=b.invalidObservationMapping?'观测映射无效，统一分支未求解，原样返回基线：'+b.observationAlignmentError:'原观测门与统一观测门分别判断；通过新观测门不意味着通过旧门，未变化的基线也不记为改善。';$('state').append(note);
 const scroll=document.createElement('div');scroll.className='scroll';const t=document.createElement('table'),head=document.createElement('tr');for(const title of ['分支','原观测门','当前观测门','内部解改变','输出改变顶点','完整眼裂改善','原整体验收','输出 IoU']){const cell=document.createElement('th');cell.textContent=title;head.append(cell)}t.append(head);for(const [branch,item] of Object.entries(row.branches)){const tr=document.createElement('tr');for(const value of [labels[branch],yn(item.originalObservationPolicyAccepted),item.invalidObservationMapping?'映射无效，未评估':yn(item.activeObservationPolicyAccepted),item.innerSolverChanged?'是':'否',item.delivered.changedVertexCount,branch==='baseline'?'基准':item.diagnosticImprovementAccepted?'已验证':'未成立',yn(item.absoluteEligible),n(item.deliveredApertureAfter.meanIntersectionOverUnion)]){const cell=document.createElement('td');cell.textContent=value;tr.append(cell)}t.append(tr)}scroll.append(t);$('state').append(scroll);
 const p=document.createElement('p');p.textContent=`统一分支：内部解${b.innerSolverChanged?'产生改变':'未产生选定改变'}；改变的外层提案 ${b.changedOuterProposalCount} 次，其中完整相交门拒绝 ${b.fullPairRejectedOuterProposalCount} 次，完成眼裂测量 ${b.maskMeasuredOuterProposalCount} 次。前段精修改变 ${b.rawRefinement.changedVertexCount} 个顶点，最终输出改变 ${b.delivered.changedVertexCount} 个。统计可能重叠；相交记录不能直接解释为可见穿透严重度。`;$('state').append(p);
 const gate=document.createElement('p'),g=b.originalPhotoGate;gate.textContent='最终输出原照片门：参考嘴部误差变化 '+n(g.frontMeanChanges.mouth)+' 工作像素；参考眼部 '+n(g.frontMeanChanges.eyes)+'；侧面眼部相对原始输入变化 '+g.sideMeanChangesAgainstOriginalInitial.map(x=>'视角'+(x.viewIndex+1)+' '+n(x.changePixels)).join('，')+'。完整装配新增相交对（相对共同基线） '+b.inheritedAndOriginalAnchorGates.pairedBaselineFullPairCount+'。';$('state').append(gate);
 $('localization').src=l.workingOverlay;$('localizationState').textContent=`以下定位图固定使用该样本参考视角 ${row.referenceViewIndex+1}，不随上方展示视角变化。原图映射眼宽：左 ${n(l.cachedEyes.imageLeft.originalCoordinateMapping.eyeWidthPixels)}、右 ${n(l.cachedEyes.imageRight.originalCoordinateMapping.eyeWidthPixels)} 像素；工作缩放 ${n(l.workingScale)} 倍。预测稳定不等于准确。`;$('original').hidden=!l.cachedOriginalOverlay;if(l.cachedOriginalOverlay)$('original').src=l.cachedOriginalOverlay;$('native').hidden=!l.nativeInferenceAvailable;if(l.nativeInferenceAvailable){$('native').src=l.originalResolutionOverlay;const d=l.nativeDiagnostic.baselinePairedDifferenceWorkingPixels;$('nativeState').textContent='同一权重、精确逆映射的相同 FAN 锚点。新原图预测与冻结预测点差均值（工作像素）：左 '+n(d.mean[0])+'、右 '+n(d.mean[1])+'。这不是精度排名，也没有用于替换目标。'}else $('nativeState').textContent='精确原图字节本地不可用，原分辨率重新推理未执行；没有以放大的工作图冒充原图。';$('data').textContent=JSON.stringify(row,null,2);
 }for(const key of ['case','view','mode'])$(key).onchange=show;show();
}).catch(error=>{$('overall').textContent='结果未通过完整核验：'+error.message});
</script></html>''', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true', help='Read readiness without writing a final result')
    args = parser.parse_args()
    if args.status:
        print(json.dumps(completionStatus(), ensure_ascii=False))
    else:
        result = summarize()
        print(json.dumps({key: result[key] for key in ('status', 'caseCount', 'imageCount',
            'invalidObservationMappingCount', 'diagnosticImprovedCaseCount', 'changingAbsoluteEligibleCaseCount')}))
