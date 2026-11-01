"""Summarize the frozen seven-case accessory-following ablation.

This reads saved evidence only. It requires all 21 branch audits and all 420
rendered images before writing summary.json. The candidate changes the fixed
accessory domain; it must not be described as passing the old all-accessories-
fixed policy merely because every other threshold remains unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from summarizeBoundaryFieldExperiment import (
    read, sha, arrays, verify, readEvidence, flag, vertexChange, apertureSummary, inheritedFailureSummary,
)
from summarizeEyeObservationExperiment import solverTrace, outerTrace, alignmentSummary

RUN = ROOT/'runs/accessoryFollowing20261024'
OUT = ROOT/'reports/accessoryFollowing20261024'
BRANCHES = ('baseline', 'fixedAccessoriesReplay', 'followingAccessories')
MODES = ('clay', 'baseColor', 'lit')


def expectedImages():
    return {name for view in range(4) for mode in MODES for name in (
        f'view{view:02d}_{mode}.png', f'view{view:02d}_{mode}_eyes.png',
        *(f'view{view:02d}_{branch}_{mode}.png' for branch in BRANCHES))}


def completionStatus():
    planPath = RUN/'experimentPlan.json'
    if not planPath.is_file(): return dict(status='notFrozen', missing=[str(planPath)])
    plan = read(planPath)
    required = [RUN/'implementationHashes.json', RUN/'finalUnitTests.json',
        RUN/'templateAccessoryBinding.json', RUN/'templateAccessoryBinding.npz',
        OUT/'independentAudit/auditImplementation.json', OUT/'independentAudit/validationSummary.json']
    for case in plan['cases']:
        required.extend(RUN/case/name for name in ('report.json', 'inputContext.json',
            'observationAlignment.json', 'observationAlignment.npz', 'observationReplayMatch.json',
            'accessoryFollowingSpec.json', 'accessoryFollowingSpec.npz', 'accessoryFollowingOperator.npz'))
        required.extend((OUT/case/'report.json', OUT/'independentAudit'/f'{case}.json'))
        required.extend(OUT/case/name for name in sorted(expectedImages()))
        for branch in BRANCHES:
            required.extend(RUN/case/branch/name for name in ('report.json', 'fullGeometry.npz',
                'rawRefinement.json', 'rawRefinement.npz', 'solverEvidence.json', 'solverEvidence.npz',
                'finalEffectGate.json', 'allowedMovement.npz'))
    missing = [str(path) for path in required if not path.is_file()]
    return dict(status='readyForVerification' if not missing else 'incomplete',
        caseCount=len(plan['cases']), missing=missing)


def componentFaceTypes(templateBinding):
    """Coarse pair categories, without treating them as penetration severity."""
    assetPath = ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1/components.json'
    result = {}
    followerNames = {part['componentName']: part['role'] for part in templateBinding['components']}
    for part in read(assetPath):
        label = part['label']
        kind = ('skin' if label == 'head' else 'innerCorner' if followerNames.get(part['name']) == 'innerCorner'
            else 'eyelash' if part['name'] in followerNames else 'eye' if label in ('leftEye', 'rightEye')
            else 'eyebrow' if label == 'eyebrow' else label)
        for faceId in part['faceIndices']:
            if int(faceId) in result: raise ValueError('Overlapping component face definitions')
            result[int(faceId)] = kind
    return result


def rejectionSummary(proposals, stages, faceTypes):
    reasons, pairs, exclusive = Counter(), Counter(), Counter()
    for proposal in proposals:
        failed = list(proposal['reasons']); reasons.update(failed)
        exclusive[' + '.join(sorted(failed)) if failed else 'noStrictRejection'] += 1
        for first, second in proposal['newFullPairs']:
            pairs[' / '.join(sorted((faceTypes[int(first)], faceTypes[int(second)])))] += 1
    gn = Counter()
    for stage in stages: gn.update(stage['overlappingSafetyRejections'])
    return dict(outerProposalCount=len(proposals), overlappingOuterReasons=dict(reasons),
        exclusiveOuterReasonCombinations=dict(exclusive), repeatedNewFacePairCategories=dict(pairs),
        overlappingGnSafetyReasons=dict(gn),
        changedOuterProposalCount=sum(row['change']['geometryChanged'] for row in proposals),
        fullPairRejectedOuterProposalCount=sum(row['newFullPairCount'] > 0 for row in proposals),
        completeMaskMeasuredOuterProposalCount=sum(row['completeApertureMeasured'] for row in proposals),
        interpretation='Reasons can overlap. Face-pair records repeat across scales and rounds; categories do not measure penetration depth or visible severity.')


def _verifyImplementation(implementation, planHash):
    if implementation['planSha256'] != planHash: raise ValueError('Implementation plan identity differs')
    verify(implementation['sourceHashes'])
    for source, snapshot in implementation['snapshots'].items():
        if sha(snapshot['snapshotPath']) != snapshot['sha256'] or snapshot['sha256'] != implementation['sourceHashes'][source]:
            raise ValueError('Frozen implementation snapshot differs')


def auditProvenance(manifestPath, planHash, cases):
    """Bind each unchanged report to the implementation that actually audited it.

    Historical validators may differ from the current workspace source after a
    narrowly scoped audit-tool repair. Their immutable snapshot bytes must
    match the original manifest; never assign a new signature to old reports.
    """
    manifest = read(manifestPath)
    if manifest['planSha256'] != planHash:
        raise ValueError('Independent audit belongs to a different experiment plan')
    proof = {str(manifestPath): sha(manifestPath)}
    validatorPath = str(ROOT/'tools/validateAccessoryFollowingExperiment.py')
    mixed = manifest.get('schemaVersion') == 'mixedIndependentAuditManifestV1'
    if mixed:
        if set(manifest['caseImplementations']) != set(cases) or manifest['unresolvedToolExceptions'] != []:
            raise ValueError('Mixed audit must cover all seven cases without unresolved exceptions')
        entries = manifest['caseImplementations']
    else:
        if manifest.get('candidateSolvesPerformed') is not False:
            raise ValueError('Independent audit must not perform candidate solves')
        entries = {case: dict(manifestPath=str(manifestPath), manifestSha256=sha(manifestPath),
            validatorSha256=manifest['hashes'][validatorPath], reportSha256=None) for case in cases}
    checkedManifests, caseVersions = {}, {}
    for case in cases:
        entry = entries[case]; path = Path(entry['manifestPath'])
        if not path.is_absolute(): raise ValueError('Audit implementation manifest path must be absolute')
        if sha(path) != entry['manifestSha256']:
            raise ValueError('Per-case original audit manifest bytes changed')
        if str(path) not in checkedManifests:
            original = read(path)
            if original['planSha256'] != planHash or original.get('candidateSolvesPerformed') is not False:
                raise ValueError('Per-case audit manifest has wrong plan or scope')
            if set(original['snapshots']) != set(original['hashes']):
                raise ValueError('Every original audit dependency needs an immutable snapshot')
            for source, snapshot in original['snapshots'].items():
                if (snapshot['sha256'] != original['hashes'][source]
                        or sha(snapshot['snapshotPath']) != snapshot['sha256']):
                    raise ValueError('Original audit implementation snapshot changed')
                proof[snapshot['snapshotPath']] = snapshot['sha256']
            if original.get('validatorSha256', original['hashes'][validatorPath]) != original['hashes'][validatorPath]:
                raise ValueError('Original validator identity differs inside its manifest')
            checkedManifests[str(path)] = original; proof[str(path)] = sha(path)
        original = checkedManifests[str(path)]
        if entry['validatorSha256'] != original['hashes'][validatorPath]:
            raise ValueError('A case was relabelled with a validator that did not audit it')
        if mixed and (not isinstance(entry['reportSha256'], str) or len(entry['reportSha256']) != 64):
            raise ValueError('Mixed audit needs the original exact report identity for every case')
        caseVersions[case] = dict(entry)
    for execution in manifest.get('executionRuns', []):
        for stem in ('validationSummary', 'auditLog', 'testEvidence'):
            path = execution[stem+'Path']; expected = execution[stem+'Sha256']
            if sha(path) != expected: raise ValueError('Historical audit execution record changed')
            proof[path] = expected
        testEvidence = read(execution['testEvidencePath'])
        if testEvidence['exitCode'] != 0 or testEvidence['testCount'] < 1:
            raise ValueError('Audit implementation tests did not pass')
        # Archived evidence keeps its original stdoutPath. Resolve the same
        # basename inside the explicitly named immutable execution directory;
        # the canonical path may now hold the repaired run's different log.
        testLog = Path(execution['directory'])/Path(testEvidence['stdoutPath']).name
        if sha(testLog) != testEvidence['stdoutSha256']:
            raise ValueError('Original archived audit test output changed')
        proof[str(testLog)] = testEvidence['stdoutSha256']
        original = read(Path(execution['directory'])/'auditImplementation.json')
        if (testEvidence['validatorSha256'] != original['hashes'][validatorPath]
                or testEvidence['testSourceSha256'] != original['hashes'][str(ROOT/'tests/testAccessoryFollowingValidation.py')]):
            raise ValueError('Audit tests do not correspond to their original implementation')
    validationPath = Path(manifestPath).parent/'validationSummary.json'
    validation = read(validationPath)
    if (validation['planSha256'] != planHash or not flag(validation, 'validationPassed')
            or validation['caseCount'] != len(cases) or validation['toolExceptionCount'] != 0
            or validation.get('failures', []) != []):
        raise ValueError('Final aggregate audit has missing cases or unresolved tool failures')
    validatedCases = {row['caseId']: row for row in validation['cases']}
    if set(validatedCases) != set(cases) or len(validation['cases']) != len(cases):
        raise ValueError('Final independent audit case matrix differs')
    for case, row in validatedCases.items():
        if not flag(row, 'validationPassed'):
            raise ValueError('Final independent case audit did not pass')
        expected = caseVersions[case]['reportSha256']
        if expected is not None and row['reportSha256'] != expected:
            raise ValueError('Final merged audit report was re-signed or replaced')
        caseVersions[case]['reportSha256'] = row['reportSha256']
    proof[str(validationPath)] = sha(validationPath)
    return manifest, caseVersions, proof


def summarize():
    ready = completionStatus()
    if ready['status'] != 'readyForVerification':
        raise ValueError(f'Experiment incomplete: {len(ready["missing"])} missing artifacts; no summary written')
    planPath, implementationPath = RUN/'experimentPlan.json', RUN/'implementationHashes.json'
    config, implementation = read(planPath), read(implementationPath); planHash = sha(planPath)
    if tuple(config['branches']) != BRANCHES or len(config['cases']) != 7 or len(set(config['cases'])) != 7:
        raise ValueError('Expected the frozen seven cases and three ordered branches')
    _verifyImplementation(implementation, planHash)
    verify(config['historicalHashes']); verify(config['accessoryFollowing']['inputHashes'])
    templateBinding = readEvidence(RUN/'templateAccessoryBinding')
    if (templateBinding['bindingSha256'] != config['accessoryFollowing']['bindingSha256']
            or len(templateBinding['followerVertexIds']) != config['accessoryFollowing']['followerVertexCount']):
        raise ValueError('Frozen template binding differs')
    faceTypes = componentFaceTypes(templateBinding)
    auditManifestPath = OUT/'independentAudit/auditImplementation.json'
    auditManifest, auditVersions, auditProof = auditProvenance(auditManifestPath, planHash, config['cases'])
    proof = {str(path): sha(path) for path in (planPath, implementationPath, auditManifestPath,
        ROOT/'tools/summarizeBoundaryFieldExperiment.py', ROOT/'tools/summarizeEyeObservationExperiment.py',
        ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1/components.json')}
    proof.update(auditProof)
    cases, imageCount = {}, 0
    for case in config['cases']:
        directory = RUN/case; contextPath = directory/'inputContext.json'; context = read(contextPath)
        verify(context['sourceHashes']); proof[str(contextPath)] = sha(contextPath)
        casePath = directory/'report.json'; caseReport = read(casePath)
        if caseReport['status'] != 'complete' or caseReport['planSha256'] != planHash:
            raise ValueError(f'Incomplete case: {case}')
        verify(caseReport['preparationHashes']); proof[str(casePath)] = sha(casePath)
        observationReplay = read(directory/'observationReplayMatch.json')
        if not flag(observationReplay, 'accepted') or not flag(observationReplay, 'exactAllFields'):
            raise ValueError('Observation targets differ from the previous experiment')
        alignment = alignmentSummary(readEvidence(directory/'observationAlignment'))
        following = readEvidence(directory/'accessoryFollowingSpec')
        if following['binding']['bindingSha256'] != templateBinding['bindingSha256']:
            raise ValueError('Case uses a different template binding')
        auditPath = OUT/'independentAudit'/f'{case}.json'; audit = read(auditPath)
        validatorHash = auditVersions[case]['validatorSha256']
        if (sha(auditPath) != auditVersions[case]['reportSha256']
                or not flag(audit, 'validationPassed') or audit['validatorSha256'] != validatorHash
                or audit['planSha256'] != planHash or set(audit['branches']) != set(BRANCHES)):
            raise ValueError(f'Independent audit incomplete or failed: {case}')
        for key in ('accessoryTemplateBinding', 'accessoryFollowingSpecification'):
            if not flag(audit[key], 'validationPassed'): raise ValueError(f'Independent accessory audit failed: {case}/{key}')
        proof[str(auditPath)] = sha(auditPath)
        renderPath = OUT/case/'report.json'; render = read(renderPath)
        verify(render['inputHashes']); verify(render['renderImplementationHashes'])
        if set(render['images']) != expectedImages() or len(render['views']) != 4 or render['branches'] != list(BRANCHES):
            raise ValueError('Four-camera render matrix differs')
        for key in ('identicalMaterials', 'allFourCameras', 'templateColorsOnly', 'vertexOrderUvBindingUnchanged',
                'nonWhitelistAndStaticExactAcrossArms', 'frozenFollowingRelationVerified',
                'originalAllNonSkinFixedPolicyChangedOnlyInCandidate', 'allFrozenInputsUnchanged'):
            if not flag(render, key): raise ValueError(f'Render identity check failed: {case}/{key}')
        images = {str(OUT/case/name): expected for name, expected in render['images'].items()}
        verify(images); proof.update(images); proof[str(renderPath)] = sha(renderPath); imageCount += len(images)
        baseline = arrays(directory/'baseline/fullGeometry.npz')['vertices']
        unit = float(arrays(context['entry']['preparedGeometry'])['unit'])
        if not np.array_equal(following['baselineVertices'], baseline): raise ValueError('Following rest state differs')
        branches = {}
        for branch in BRANCHES:
            branchDir = directory/branch; reportPath = branchDir/'report.json'; report = read(reportPath)
            audited = audit['branches'][branch]
            if (report['caseId'] != case or report['branch'] != branch or not flag(audited, 'validationPassed')
                    or not flag(audited, 'evidenceValidated') or audited['reportSha256'] != sha(reportPath)):
                raise ValueError('Independent branch report identity differs')
            verify({str(branchDir/name): expected for name, expected in report['outputHashes'].items()})
            raw = readEvidence(branchDir/'rawRefinement'); captures = raw['solverEvidence']
            # The independent audit verifies the separately saved solver sidecar;
            # reuse its identical embedded arrays here instead of doubling RAM.
            full = arrays(branchDir/'fullGeometry.npz')['vertices']; change = vertexChange(full, baseline, unit)
            allowed = arrays(branchDir/'allowedMovement.npz'); isFollowing = branch == 'followingAccessories'
            expectedAllowed = np.zeros(len(full), bool); expectedAllowed[following['movableSkinVertexIds']] = True
            if isFollowing: expectedAllowed[following['followerVertexIds']] = True
            if (not np.array_equal(allowed['movableMask'], expectedAllowed)
                    or not np.array_equal(full[~expectedAllowed], baseline[~expectedAllowed])
                    or not flag(report, 'allOutsideAllowedMovementExactlyFixed')
                    or flag(report, 'accessoryFollowingEnabled') != isFollowing
                    or flag(report, 'originalAllNonSkinFixedPolicyActive') == isFollowing
                    or report['accessoryDegreesOfFreedom'] != 0):
                raise ValueError('Declared physical fixed domain or zero accessory freedom differs')
            if isFollowing and (report['accessoryFollowingSpecSha256'] != following['specSha256']
                    or not flag(report, 'followerRelationExactlySatisfied')
                    or not flag(audited['followerRelationship'], 'validationPassed')):
                raise ValueError('Final accessory mapping identity differs')
            changedFollowers = int(np.any(full[following['followerVertexIds']] != baseline[following['followerVertexIds']], axis=1).sum())
            if changedFollowers != report['changedFollowerVertexCount']: raise ValueError('Follower movement count differs')
            invalid = branch != 'baseline' and not alignment['valid']
            active = report['activeObservationPolicyAccepted']; original = flag(report, 'originalObservationPolicyAccepted')
            if invalid:
                if (active is not None or captures or raw['candidateProposals'] or change['geometryChanged']
                        or report['absoluteEligible'] or report['observationPolicyChangedForDiagnostic']):
                    raise ValueError('Invalid observation policy was solved or labelled as passing')
            elif type(active) is not bool: raise ValueError('Evaluated observation policy needs an explicit decision')
            if report['observationAlignmentValid'] != alignment['valid']: raise ValueError('Observation validity differs')
            effect = report['finalEffectGate']
            if effect != read(branchDir/'finalEffectGate.json'): raise ValueError('Final effect report identity differs')
            accepted = flag(report, 'residualCandidateAccepted'); incremental = flag(report, 'incrementalSafe')
            absolute = flag(report, 'absoluteEligible'); effectAccepted = flag(effect, 'accepted')
            improvement = bool(not invalid and branch != 'baseline' and change['geometryChanged'] and effectAccepted and incremental and accepted)
            expected = dict(producerAccepted=accepted, geometryChanged=change['geometryChanged'], incrementalSafe=incremental,
                absoluteEligible=absolute, requestedEffectAccepted=effectAccepted, improvementAccepted=improvement,
                changingAbsoluteEligible=improvement and absolute, clientReady=False)
            if any(flag(audited['classification'], key) != value for key, value in expected.items()):
                raise ValueError('Independent improvement or safety classification differs')
            if (report['improvementAccepted'] != improvement or report['geometryChanged'] != change['geometryChanged']
                    or report['changingAbsoluteEligible'] != (improvement and absolute)
                    or audited['classification']['originalObservationPolicyAccepted'] != original
                    or audited['classification']['activeObservationPolicyAccepted'] != active):
                raise ValueError('Producer and independent observation decisions differ')
            fallback = flag(effect, 'fallbackToExactBaseline')
            if fallback and change['geometryChanged']: raise ValueError('Fallback did not return exact baseline')
            replay = report['replayMatch']
            if branch == 'fixedAccessoriesReplay':
                if replay != read(branchDir/'replayMatch.json') or not flag(replay, 'accepted'):
                    raise ValueError('Fixed-accessory arm did not exactly replay its previous solve')
            stages = solverTrace(captures, raw['diagnostics'], unit)
            proposals = outerTrace(raw['candidateProposals'], baseline, unit)
            rejection = rejectionSummary(proposals, stages, faceTypes)
            oldFixed = flag(report, 'allOutsideSupportExactlyFixed')
            legacyAbsolute = bool(absolute and oldFixed)
            if (flag(report, 'legacyAbsoluteEligible') != legacyAbsolute
                    or flag(audited['classification'], 'legacyAbsoluteEligible') != legacyAbsolute
                    or flag(audited['classification'], 'changingLegacyAbsoluteEligible') != bool(improvement and legacyAbsolute)):
                raise ValueError('Original all-accessories-fixed eligibility was not retained honestly')
            rawChange = vertexChange(raw['vertices'], baseline, unit)
            branches[branch] = dict(status=report['diagnostics']['status'], invalidObservationMapping=invalid,
                candidatePolicyStatus='notEvaluatedInvalidObservationMapping' if invalid else 'passed' if active else 'failed',
                originalObservationPolicyAccepted=original, activeObservationPolicyAccepted=active,
                originalPhotoGate=report['legacyObservationGate'], activePhotoGate=None if invalid else report['activeLegacyObservationGate'],
                originalCoarseAlignment=report['originalCoarseAlignment'], activeCoarseAlignment=None if invalid else report['activeCoarseAlignment'],
                inheritedAndOriginalAnchorGates=inheritedFailureSummary(report), contactSafety=report['contactSafety'],
                delivered=change, rawRefinement=rawChange, rawFastSolver=vertexChange(raw['solverVertices'], baseline, unit),
                lastOuterProposal=vertexChange(raw['rawCandidateVertices'], baseline, unit),
                solverStages=stages, outerProposals=proposals, rejectionSummary=rejection,
                innerSolverChanged=any(row['innerOutputChange']['geometryChanged'] for row in stages),
                rawRefinementAccepted=flag(report, 'rawRefinementAccepted'), deliveredResidualAccepted=accepted,
                incrementalSafe=incremental, declaredPolicyAbsoluteEligible=absolute,
                originalAllFixedRuleSatisfiedByOutput=oldFixed,
                originalAllFixedPolicyAbsoluteEligible=legacyAbsolute,
                changingOriginalAllFixedPolicyAbsoluteEligible=bool(improvement and legacyAbsolute),
                diagnosticImprovementAccepted=improvement, changingDeclaredPolicyAbsoluteEligible=bool(improvement and absolute),
                fallbackToExactBaseline=fallback,
                finalEffectActivelyRejectedAcceptedRaw=bool(fallback and report['rawRefinementAccepted'] and rawChange['geometryChanged']),
                apertureBefore=apertureSummary(effect['baseline']), rawApertureAfter=apertureSummary(effect['candidate']),
                deliveredApertureAfter=apertureSummary(effect['baseline'] if fallback else effect['candidate']),
                accessoryFollowingEnabled=isFollowing, followingAppliedToSolver=report['accessoryFollowingAppliedToSolver'],
                changedFollowerVertexCount=changedFollowers, allOutsideAllowedMovementExactlyFixed=True,
                followerRelationExactlySatisfied=report['followerRelationExactlySatisfied'],
                fixedPolicy='frozenSkinSupportPlus264DeterministicFollowers' if isFollowing else 'originalSkinSupportAllAccessoriesFixed',
                replayMatched=None if replay is None else replay['accepted'],
                independentClassification=audited['classification'], independentRawEffect=audited['finalEffectGate'],
                independentDeliveredEffect=audited['finalOutputEffect'], independentAuditRelativeUrl=f'independentAudit/{case}.json',
                strictLegacyBindingAndPayloadPreserved=True, clientReady=False)
            proof[str(reportPath)] = sha(reportPath)
        cases[case] = dict(branches=branches, observationAlignment=alignment,
            accessoryFollowingSpecSha256=following['specSha256'], independentEvidenceVerified=True,
            independentAuditImplementation=auditVersions[case],
            renderedImages=len(render['images']), referenceViewIndex=context['referenceSelection']['referenceViewIndex'])
    if imageCount != 420: raise ValueError('Expected exactly 420 PNG images')
    testPath = RUN/'finalUnitTests.json'; tests = read(testPath)
    if tests['status'] != 'passed' or int(tests['passed']) < 1 or sha(tests['logPath']) != tests['logSha256']:
        raise ValueError('Implementation tests missing or changed')
    proof[str(testPath)] = sha(testPath)
    candidates = [row['branches']['followingAccessories'] for row in cases.values()]
    combinedOuter, combinedPairs, combinedGn = Counter(), Counter(), Counter()
    for row in candidates:
        combinedOuter.update(row['rejectionSummary']['overlappingOuterReasons'])
        combinedPairs.update(row['rejectionSummary']['repeatedNewFacePairCategories'])
        combinedGn.update(row['rejectionSummary']['overlappingGnSafetyReasons'])
    result = dict(schemaVersion='deterministicAccessoryFollowingSummaryV1', status='complete', cases=cases,
        caseCount=7, branchCount=21, cameraStates=84, imageCount=imageCount,
        invalidObservationMappingCount=sum(row['invalidObservationMapping'] for row in candidates),
        innerSolverChangedCaseCount=sum(row['innerSolverChanged'] for row in candidates),
        rawRefinementChangedCaseCount=sum(row['rawRefinement']['geometryChanged'] for row in candidates),
        deliveredChangedCaseCount=sum(row['delivered']['geometryChanged'] for row in candidates),
        diagnosticImprovedCaseCount=sum(row['diagnosticImprovementAccepted'] for row in candidates),
        changingDeclaredPolicyAbsoluteEligibleCaseCount=sum(row['changingDeclaredPolicyAbsoluteEligible'] for row in candidates),
        changingOriginalAllFixedPolicyAbsoluteEligibleCaseCount=sum(row['changingOriginalAllFixedPolicyAbsoluteEligible'] for row in candidates),
        exactReplayCaseCount=sum(row['branches']['fixedAccessoriesReplay']['replayMatched'] is True for row in cases.values()),
        candidateImprovedCases=[case for case, row in cases.items() if row['branches']['followingAccessories']['diagnosticImprovementAccepted']],
        candidateOuterRejections=dict(combinedOuter), candidateRepeatedFacePairCategories=dict(combinedPairs),
        candidateGnSafetyRejections=dict(combinedGn),
        fixedDomainPolicyChanged=True, followerVertexCount=len(templateBinding['followerVertexIds']),
        templateBindingSha256=templateBinding['bindingSha256'], bindingPolicy=templateBinding['policy'],
        accessoryDegreesOfFreedom=0, originalSkinSupportAndParametersUnchanged=True,
        geometricThresholdsAndGateOrderUnchanged=True, originalObservationGatesSeparatelyAudited=True,
        fixedRuleMeaning='Only the candidate permits the frozen 264 canthus/eyelash vertices to follow. Absolute eligibility here uses that declared fixed domain plus the original observation, shape, contact and collision thresholds; it is not unconditional compliance with the previous all-accessories-fixed rule.',
        promotedCandidates=0, productionDefaultsChanged=False, textureRebaked=False, realLipsRefined=False,
        finalAdaptiveAssemblyValidated=False, manualTruthEstablished=False, planSha256=planHash,
        unitTests=tests, independentAuditTests=auditManifest.get('tests'),
        independentValidatorSha256s=sorted({entry['validatorSha256'] for entry in auditVersions.values()}),
        independentAuditImplementationByCase=auditVersions, unresolvedAuditToolExceptions=[],
        testCountInterpretation='Suites may overlap; counts are not additive.', evidenceHashes=proof,
        summaryImplementationSha256=sha(__file__), previousExperimentViewerUrl='http://127.0.0.1:18101/',
        previousLocalizationViewerUrl='http://127.0.0.1:18101/localization/index.html',
        conclusions=['Both fitting branches use the same previously frozen unified observation table.',
            'Only deterministic accessory following and its necessary derivative chain differ; no added accessory degrees of freedom.',
            'Inner solver movement, outer rejection, raw acceptance, final aperture benefit and delivered change are reported separately.',
            'An unchanged baseline or inherited failure cannot count as a successful improvement.'],
        limitations=['Template-visible material boundaries and automatic photo curves are anatomical proxies, not manual truth.',
            'Template binding introduces explicit shared domain/distance/tie parameters, frozen once for every case.',
            'Fixed-domain policy is deliberately changed for the candidate; all other numeric thresholds and gate positions stay unchanged.',
            'Repeated triangle-pair records are conservative intersection evidence, not a penetration severity score.',
            'No texture rebake, brow/nose texture comparison, mouth repair or client-ready certification is included.',
            'Previously inspected cases do not establish unseen generalization; nominal millimeters are template-derived.'])
    verify(proof)
    writeJson(OUT/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    result = completionStatus() if args.status else summarize()
    if args.status: print(json.dumps(result, ensure_ascii=False, indent=2))
    else: print(json.dumps({key: result[key] for key in ('status', 'caseCount', 'branchCount', 'imageCount',
        'diagnosticImprovedCaseCount', 'changingDeclaredPolicyAbsoluteEligibleCaseCount', 'exactReplayCaseCount')}, ensure_ascii=False))
