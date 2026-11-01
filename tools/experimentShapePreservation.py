"""Shared-config global relative-3D shape-prior ablation and objective export.

All subject inputs come from the previously frozen split. This runner does not
choose per-case hyperparameters or use held-out results for parameter selection.
Exported H/b use the convention E(x)=x.T H x-2*b.T x+constant, vertex-major XYZ.
The optional bounded optimizer can consume exactly that objective separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.directTemplateRegistration import registerDirectTemplate, _laplacian
from faceAppearance.unsupportedShapePrior import buildUnsupportedShapePrior
from experimentNoseSurfaceRegistration import arrays, digest
from experimentEarlyCavityRegistration import assembleInitial, safetyMetrics, triangleMeasures

REGIONS = ('frontImageLeft', 'frontImageRight')
PRIOR_CONFIG = {'normalTargetFraction': .35, 'tangentTargetFraction': 1.,
    'normalPenalty': 1., 'tangentPenalty': 1., 'lengthScaleMm': 2., 'supportThreshold': .15}
ABLATION_WEIGHTS = [.01, .03, .1, .25]
BOUNDED_WEIGHT = .01
OPTIONS = {'dataWeight': 10000., 'smoothWeight': 3000., 'restWeight': 0.,
           'staticConstraint': 'hard', 'solver': 'exact'}


def verifyHashes(hashes):
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError(f'Frozen shape-preservation input changed: {path}')


def buildShapeObjective(raw, weight):
    """Build the original global objective plus one shared normalized 3D prior.

    No optimization or candidate metric is evaluated here. The returned scalar
    normal operator includes original joint sample/static MSE denominators.
    Hard static coordinates are subsequently eliminated by the chosen solver.
    """
    working = dict(raw)
    faces, rest, before = raw['faces'], raw['alignedRest'], raw['pureDirect']
    valid, sampleFaces, bary = raw['validMask'], raw['sampleFaceIds'], raw['sampleBarycentric']
    n, static = len(before), raw['staticIds']
    sample = sparse.csr_matrix((bary[valid].ravel(),
        (np.repeat(np.arange(valid.sum()), 3), faces[sampleFaces[valid]].ravel())),
        shape=(int(valid.sum()), n))
    anchors = sparse.csr_matrix((np.ones(len(static)), (np.arange(len(static)), static)), shape=(len(static), n))
    design = sparse.vstack([sample, anchors], format='csr')
    target = np.concatenate([raw['targetPoints'][valid], rest[static]])
    laplacian, _ = _laplacian(n, faces)
    coefficients = {'data': OPTIONS['dataWeight'] / (3 * (int(valid.sum()) + len(static))),
        'smooth': OPTIONS['smoothWeight'] / (3 * n), 'rest': OPTIONS['restWeight'] / (3 * n)}
    scalarH = (coefficients['data'] * (design.T @ design)
        + coefficients['smooth'] * (laplacian.T @ laplacian)
        + coefficients['rest'] * sparse.eye(n)).tocsc()
    scalarB = coefficients['data'] * (design.T @ target) + coefficients['rest'] * rest
    total = sum(len(raw[name + '_interiorVertexIds']) for name in REGIONS)
    operators, targets, priors, depthOperators = [], [], {}, {}
    for name in REGIONS:
        normal = raw[name + '_inwardDirection']
        prior = buildUnsupportedShapePrior(rest, faces, raw[name + '_rimVertexIdsOrdered'],
            raw[name + '_interiorVertexIds'], faces[sampleFaces], bary, valid.astype(float),
            normalDirection=normal, lengthScale=PRIOR_CONFIG['lengthScaleMm'] * float(raw['unit']),
            supportThreshold=PRIOR_CONFIG['supportThreshold'],
            normalTargetFraction=PRIOR_CONFIG['normalTargetFraction'],
            tangentTargetFraction=PRIOR_CONFIG['tangentTargetFraction'],
            normalPenalty=PRIOR_CONFIG['normalPenalty'], tangentPenalty=PRIOR_CONFIG['tangentPenalty'])
        fraction = np.sqrt(len(prior['interiorVertexIds']) / total)
        operators.append(prior['weightedOperator'] * fraction)
        targets.append(prior['weightedTargets'] * fraction)
        unitNormal = prior['normalDirection']
        depthOperators[name] = sparse.kron(prior['relativeOperator'], unitNormal.reshape(1, 3), format='csr')
        working[name + '_restRelativeDepths'] = prior['restRelativeOffsets'] @ unitNormal
        working[name + '_targetDepths'] = prior['targetOffsets'] @ unitNormal
        working[name + '_priorWeights'] = prior['priorWeights']
        working[name + '_interiorSupportRatio'] = prior['interiorSupportRatio']
        working[name + '_targetOffsets'] = prior['targetOffsets']
        working[name + '_restRelativeOffsets'] = prior['restRelativeOffsets']
        priors[name] = prior
    priorOperator = sparse.vstack(operators, format='csr')
    priorTargets = np.concatenate(targets)
    globalH = (sparse.kron(scalarH, sparse.eye(3), format='csc') + weight * (priorOperator.T @ priorOperator)).tocsc()
    globalB = scalarB.ravel() + weight * (priorOperator.T @ priorTargets)
    constant = (coefficients['data'] * float(np.sum(target ** 2))
        + coefficients['rest'] * float(np.sum(rest ** 2)) + weight * float(priorTargets @ priorTargets))
    return {'raw': working, 'H': globalH, 'b': globalB, 'constant': constant,
        'priorOperator': priorOperator, 'priorTargets': priorTargets,
        'scalarOriginalH': scalarH, 'scalarOriginalB': scalarB,
        'depthOperators': depthOperators, 'priors': priors, 'coefficients': coefficients,
        'weight': weight, 'hardStaticCoordinates': (3 * static[:, None] + np.arange(3)).ravel()}


def saveObjective(objective, directory):
    """Preserve everything needed to independently solve or audit the quadratic."""
    directory.mkdir(parents=True, exist_ok=False)
    sparse.save_npz(directory / 'normalMatrixH.npz', objective['H'])
    sparse.save_npz(directory / 'priorOperator.npz', objective['priorOperator'])
    sparse.save_npz(directory / 'originalScalarH.npz', objective['scalarOriginalH'])
    np.savez_compressed(directory / 'normalRhs.npz', b=objective['b'], constant=objective['constant'],
        weightedPriorTargets=objective['priorTargets'], originalScalarB=objective['scalarOriginalB'],
        hardStaticCoordinates=objective['hardStaticCoordinates'],
        hardStaticValues=objective['raw']['pureDirect'].ravel()[objective['hardStaticCoordinates']])
    np.savez_compressed(directory / 'safetyGeometry.npz', **objective['raw'])
    for name, operator in objective['depthOperators'].items():
        sparse.save_npz(directory / f'{name}DepthOperator.npz', operator)
    writeJson(directory / 'objective.json', {'formula': 'E(x)=x.T@H@x-2*b.T@x+constant',
        'gradient': '2*(H@x-b)', 'coordinateOrder': 'vertexMajorXYZ',
        'hardStaticTreatment': 'Original static coordinates must be eliminated/fixed exactly; H includes original static-data denominator',
        'weight': objective['weight'], 'commonPriorConfig': PRIOR_CONFIG,
        'originalObjectiveOptions': OPTIONS, 'originalEffectiveCoefficients': objective['coefficients'],
        'priorDiagnostics': {name: prior['diagnostics'] for name, prior in objective['priors'].items()},
        'shape': list(objective['H'].shape), 'nonzeroCoefficientCount': objective['H'].nnz})


def normalizedShapePolicy(frozen):
    """Translate unchanged numeric shape gates; signed-depth logic is below."""
    solve, outside, cavity, numeric = (frozen[name] for name in
        ('solvePolicy', 'outside', 'cavity', 'numericTolerance'))
    expected = (PRIOR_CONFIG['normalTargetFraction'], PRIOR_CONFIG['tangentTargetFraction'],
        PRIOR_CONFIG['normalPenalty'], PRIOR_CONFIG['tangentPenalty'], PRIOR_CONFIG['lengthScaleMm'],
        PRIOR_CONFIG['supportThreshold'], ABLATION_WEIGHTS, BOUNDED_WEIGHT)
    actual = tuple(solve[name] for name in ('normalDepthFraction', 'tangentialShapeFraction',
        'normalEnergyWeight', 'tangentialEnergyWeight', 'depthScaleMm', 'supportThreshold',
        'unconstrainedAblationWeights', 'constrainedWeight'))
    if actual != expected or frozen['schemaVersion'] != 'shapePreservationSafetyV1':
        raise ValueError('Frozen policy does not match the shared prior configuration')
    return {'frozen': frozen,
        'exterior': {'maximumDisplacementMm': outside['maximumNonInteriorDisplacementMm'],
            'eyeMouthMaximumDisplacementMm': outside['maximumEyeMouthSupportDisplacementMm'],
            'maximumRelativeEdgeChange': outside['maximumEdgeRatioToBaseline'] - 1.,
            'minimumAreaRatio': outside['minimumAreaRatioToBaseline'],
            'maximumAreaRatio': outside['maximumAreaRatioToBaseline'],
            'minimumNormalCosine': outside['minimumNormalCosineToBaseline']},
        'cavity': {'maximumDisplacementMm': cavity['maximumInteriorDisplacementMm'],
            'minimumRestEdgeRatio': cavity['edgeReferenceLowerRatio'],
            'maximumRestEdgeRatio': cavity['edgeReferenceUpperRatio'],
            'minimumRestAreaRatio': cavity['areaReferenceLowerRatio'],
            'maximumRestAreaRatio': cavity['areaReferenceUpperRatio'],
            'minimumRestQualityRatio': cavity['minimumQualityFractionOfRest'],
            'minimumBaselineNormalCosine': cavity['minimumNormalCosineToBaseline'],
            'positiveTargetMinimumRestDepthMm': 1.},
        'maximumDataMseRatio': frozen['dataAndBenefit']['maximumValidSampleDataMseRatioToBaseline'],
        'minimumWeightedMissingDepthRmsImprovementFraction': frozen['dataAndBenefit']['minimumWeightedUnsupportedMissingDepthRmsImprovementFraction'],
        'numericalTolerance': {'lengthAndDepthMm': numeric['lengthOrDepthMm'],
            'areaMmSquared': numeric['areaMm2'], 'normalCosine': numeric['normalDot'],
            'dimensionless': numeric['dimensionless'], 'dataMseRatio': numeric['dataMseRatio'],
            'missingDepthImprovementFraction': numeric['missingDepthImprovementFraction']}}


def shapeSafetyMetrics(raw, candidate, depthOperators, policy, baselineSkinPairs, baselineFullPairs):
    """Reuse unchanged mesh gates, then apply the newly frozen signed-depth rules.

    Vector-prior confidence covers all weakly supported interior points. Depth
    benefit has its own positive-rest subset, never an implicit prior mask.
    """
    benefitRaw = dict(raw)
    for name in REGIONS:
        benefitRaw[name + '_priorWeights'] = raw[name + '_priorWeights'] * (raw[name + '_restRelativeDepths'] > 0)
    result = safetyMetrics(benefitRaw, candidate, depthOperators, policy, baselineSkinPairs, baselineFullPairs)
    before, unit = raw['pureDirect'], float(raw['unit'])
    depthToleranceMm = policy['numericalTolerance']['lengthAndDepthMm']
    for name in REGIONS:
        operator, ids = depthOperators[name], raw[name + '_interiorVertexIds']
        d0, dn = operator @ before.ravel() / unit, operator @ candidate.ravel() / unit
        dr = raw[name + '_restRelativeDepths'] / unit
        confidence = raw[name + '_priorWeights']
        lower, upper = np.minimum(np.minimum(d0, dr), 0.), np.maximum(np.maximum(d0, dr), 0.)
        active = (confidence > 0) & (dr >= 1.)
        activeLower, activeUpper = np.minimum(d0, 0.), np.maximum(d0, dr)
        masks = {'belowBaselineRestZeroHull': dn < lower - depthToleranceMm,
            'aboveBaselineRestZeroHull': dn > upper + depthToleranceMm,
            'activeDeepBelowBaselineOrZero': active & (dn < activeLower - depthToleranceMm),
            'activeDeepAboveBaselineOrRest': active & (dn > activeUpper + depthToleranceMm)}
        result['checks'][name + 'DepthIntervals'] = not any(mask.any() for mask in masks.values())
        def absoluteDepthResidual(kind):
            old = np.maximum(-d0, 0.) if kind == 'negative' else np.maximum(d0 - dr, 0.)
            new = np.maximum(-dn, 0.) if kind == 'negative' else np.maximum(dn - dr, 0.)
            selected = np.flatnonzero(active & ((old > depthToleranceMm) | (new > depthToleranceMm)))
            return {'beforeCount': int((active & (old > depthToleranceMm)).sum()),
                'candidateCount': int((active & (new > depthToleranceMm)).sum()),
                'beforeMaximumMm': float(old[active].max(initial=0.)),
                'candidateMaximumMm': float(new[active].max(initial=0.)),
                'beforeSummedMm': float(old[active].sum()), 'candidateSummedMm': float(new[active].sum()),
                'records': [{'vertexId': int(ids[i]), 'beforeSeverityMm': float(old[i]),
                    'candidateSeverityMm': float(new[i]),
                    'change': 'worsened' if new[i] - old[i] > depthToleranceMm else
                        'improved' if old[i] - new[i] > depthToleranceMm else 'unchanged'} for i in selected]}
        result['depthMetrics'][name] = {
            'interiorVertexIds': ids, 'beforeMm': d0, 'candidateMm': dn, 'alignedRestMm': dr,
            'lowerHullMm': lower, 'upperHullMm': upper, 'supportConfidence': confidence,
            'benefitWeights': benefitRaw[name + '_priorWeights'],
            'violationsOriginalVertexIds': {key: ids[mask] for key, mask in masks.items()},
            'activeDeepPointCount': int(active.sum()), 'activeDeepPointIds': ids[active],
            'inheritedNegativeDepth': absoluteDepthResidual('negative'),
            'inheritedOverRestDepth': absoluteDepthResidual('overRest'),
            'perPointMissingDepth': [{'vertexId': int(ids[i]), 'beforeMm': float(max(dr[i] - d0[i], 0.)),
                'candidateMm': float(max(dr[i] - dn[i], 0.)),
                'benefitWeight': float(benefitRaw[name + '_priorWeights'][i]),
                'changeMm': float(max(dr[i] - dn[i], 0.) - max(dr[i] - d0[i], 0.))} for i in range(len(ids))]}
    effectKey = 'minimumWeightedMissingDepthImprovement'
    benefitMass = sum(float(benefitRaw[name + '_priorWeights'].sum()) for name in REGIONS)
    effectDefined = bool(benefitMass > 0 and np.isfinite(result['weightedMissingDepthRmsMm']['before'])
        and result['weightedMissingDepthRmsMm']['before'] > 0)
    result['weightedMissingDepthRmsMm']['benefitWeightMass'] = benefitMass
    result['weightedMissingDepthRmsMm']['effectDefined'] = effectDefined
    result['weightedMissingDepthRmsMm']['undefinedReason'] = None if effectDefined else 'No positive benefit mass or no baseline missing depth'
    if not effectDefined:
        result['weightedMissingDepthRmsMm']['improvementFraction'] = None
    result['checks'][effectKey] = effectDefined and (result['weightedMissingDepthRmsMm']['improvementFraction'] >=
        policy['minimumWeightedMissingDepthRmsImprovementFraction'] -
        policy['numericalTolerance']['missingDepthImprovementFraction'])
    result['minimumEffectAccepted'] = bool(result['checks'][effectKey])
    result['safetyAccepted'] = all(value for key, value in result['checks'].items() if key != effectKey)
    result['accepted'] = bool(all(result['checks'].values()))
    result['reasons'] = [key for key, value in result['checks'].items() if not value]
    return result


def baselinePreflight(objective, policy):
    """Validate references and baseline-inclusive safety before any new solve."""
    raw, depth = objective['raw'], objective['depthOperators']
    before, rest, faces, edges = (raw[key] for key in ('pureDirect', 'alignedRest', 'faces', 'edges'))
    referenceChecks = {'positiveFiniteUnit': bool(np.isfinite(raw['unit']) and raw['unit'] > 0),
        'allReferenceCoordinatesFinite': bool(np.isfinite(before).all() and np.isfinite(rest).all()),
        'staticFrameExact': bool(np.array_equal(before[raw['staticIds']], rest[raw['staticIds']])),
        'priorFinite': bool(np.isfinite(objective['priorOperator'].data).all() and np.isfinite(objective['priorTargets']).all())}
    for label, vertices in (('baseline', before), ('rest', rest)):
        lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
        areas, normals = triangleMeasures(vertices, faces)
        referenceChecks[label + 'PositiveFiniteEdges'] = bool(np.isfinite(lengths).all() and np.all(lengths > 0))
        referenceChecks[label + 'PositiveFiniteAreasAndNormals'] = bool(np.isfinite(areas).all() and np.all(areas > 0) and np.isfinite(normals).all())
    interior = np.concatenate([raw[name + '_interiorVertexIds'] for name in REGIONS])
    referenceChecks['exactInteriorDomain'] = bool(np.array_equal(np.sort(interior), np.flatnonzero(raw['interiorMask'])))
    referenceChecks['exactCavityFaceDomain'] = bool(np.array_equal(raw['patchFaceMask'], raw['interiorMask'][faces].any(1)))
    referenceChecks['exactCavityEdgeDomain'] = bool(np.array_equal(raw['interiorEdgeMask'], raw['interiorMask'][edges].any(1)))
    referenceChecks['depthOperatorCount'] = all(depth[name].shape == (len(raw[name + '_interiorVertexIds']), before.size) for name in REGIONS)
    if not all(referenceChecks.values()):
        raise ValueError(f'Invalid baseline references: {referenceChecks}')
    skinPairs = intersectionPairs(before, faces)
    full = assembleInitial(raw['fullAppearance'], raw['skinIds'], raw['staticIds'], before)
    fullPairs = intersectionPairs(full, raw['fullFaces'])
    gate = shapeSafetyMetrics(raw, before, depth, policy, skinPairs, fullPairs)
    gate['referenceChecks'] = referenceChecks
    gate['baselineSafetyPreflightAccepted'] = bool(gate['safetyAccepted'] and all(referenceChecks.values()))
    if not gate['baselineSafetyPreflightAccepted']:
        raise ValueError(f'Baseline-inclusive safety preflight failed: {gate["reasons"]}')
    return gate, skinPairs, fullPairs


def freezeExecution(splitPath, policyPath, settingsPath):
    output = splitPath.parent
    paths = {'policy.json': policyPath, 'solverSettings.json': settingsPath}
    for name, source in paths.items():
        target = output / name
        if target.exists():
            if digest(target) != digest(source):
                raise ValueError(f'Frozen execution evidence differs: {name}')
        else:
            shutil.copyfile(source, target)
    plan = {'status': 'frozenBeforeNewShapeCandidateSolves', 'splitSha256': digest(splitPath),
        'policySha256': digest(policyPath), 'solverSettingsSha256': digest(settingsPath),
        'supersedesUnexecutedSplitFields': ['parameterSelection', 'holdoutExecutionRule'],
        'inputSplitRemainsUnchanged': True,
        'unconstrained': {'purpose': 'development-only ablation, never fallback assets or weight selection',
            'weights': ABLATION_WEIGHTS, 'export': False},
        'bounded': {'weight': BOUNDED_WEIGHT, 'commonConfig': PRIOR_CONFIG,
            'selection': 'No selection: .01 is fixed before any candidate',
            'holdoutRule': 'Run fixed bounded configuration only after all three dev cases pass safety and minimum effect'},
        'noPerCaseParameters': True, 'noProductionChanges': True}
    planPath = output / 'experimentPlan.json'
    if planPath.exists():
        if json.loads(planPath.read_text()) != plan:
            raise ValueError('Existing execution plan differs; do not overwrite')
    else:
        writeJson(planPath, plan)
    return planPath


def runAblation(case, policy, frozenHashes):
    verifyHashes(frozenHashes)
    output = Path(case['preparedGeometry']).parent.parent / 'unconstrainedAblation'
    output.mkdir(exist_ok=False)
    raw = arrays(case['preparedGeometry'])
    firstObjective = buildShapeObjective(raw, ABLATION_WEIGHTS[0])
    baseline, skinPairs, fullPairs = baselinePreflight(firstObjective, policy)
    writeJson(output / 'baselineMetrics.json', baseline)
    summaries, started = [], perf_counter()
    for weight in ABLATION_WEIGHTS:
        objective = firstObjective if weight == ABLATION_WEIGHTS[0] else buildShapeObjective(raw, weight)
        candidateDirectory = output / f'weight{weight:g}'
        candidateDirectory.mkdir()
        saveObjective(objective, candidateDirectory / 'objective')
        result = registerDirectTemplate(raw['rawRest'], raw['faces'], raw['sampleFaceIds'],
            raw['sampleBarycentric'], raw['targetPoints'], raw['validMask'], raw['staticIds'],
            **OPTIONS, quadraticPrior={'operator': objective['priorOperator'],
                'targets': objective['priorTargets'], 'weight': weight})
        candidate = result['rawVertices']
        np.testing.assert_array_equal(result['alignedRest'], raw['alignedRest'])
        np.testing.assert_array_equal(candidate[raw['staticIds']], raw['pureDirect'][raw['staticIds']])
        np.savez_compressed(candidateDirectory / 'geometry.npz', vertices=candidate,
            pureDirect=raw['pureDirect'], alignedRest=raw['alignedRest'], faces=raw['faces'], delta=candidate - raw['pureDirect'])
        gate = shapeSafetyMetrics(objective['raw'], candidate, objective['depthOperators'], policy, skinPairs, fullPairs)
        residual = objective['H'] @ candidate.ravel() - objective['b']
        free = np.setdiff1d(np.arange(candidate.size), objective['hardStaticCoordinates'])
        stationarity = {'freeMaximumAbsoluteGradient': float(2 * np.abs(residual[free]).max()),
            'freeGradientRms': float(2 * np.sqrt(np.mean(residual[free] ** 2))),
            'objectiveValue': float(candidate.ravel() @ (objective['H'] @ candidate.ravel()) - 2 * objective['b'] @ candidate.ravel() + objective['constant'])}
        item = {'caseId': case['caseId'], 'weight': weight, 'method': 'unconstrainedVectorPriorAblation',
            'geometryGate': gate, 'registrationDiagnostics': result['diagnostics'],
            'exportedObjectiveStationarity': stationarity, 'finalCandidateAccepted': False,
            'status': 'ablationPassNotSelected' if gate['accepted'] else 'ablationRejected',
            'geometrySha256': digest(candidateDirectory / 'geometry.npz'),
            'sourceHashes': frozenHashes, 'initialRegistrationOnly': True,
            'downstreamGeometryRerun': False, 'textureRebakedForInitialGeometry': False}
        writeJson(candidateDirectory / 'report.json', item)
        brief = {'caseId': case['caseId'], 'weight': weight, 'accepted': gate['accepted'],
            'safetyAccepted': gate['safetyAccepted'], 'reasons': gate['reasons'],
            'benefit': gate['weightedMissingDepthRmsMm']['improvementFraction'],
            'dataMseRatio': gate['sampleDataMseRatio'], 'maxDisplacementMm': gate['maximumDisplacementMm'],
            'report': str(candidateDirectory / 'report.json')}
        summaries.append(brief)
        print(json.dumps(brief), flush=True)
        verifyHashes(frozenHashes)
    final = {'caseId': case['caseId'], 'status': 'ablationCompleteNoSelection',
        'finalCandidateAccepted': False, 'exported': False, 'seconds': perf_counter() - started,
        'candidates': summaries, 'sourceHashes': frozenHashes}
    writeJson(output / 'report.json', final)
    return final


def buildGuardSpec(objective, policy):
    """Create uniform topology-derived guards in the same global coordinates."""
    from faceAppearance.meshShapeConstraints import buildMeshShapeGuardSpec
    raw = objective['raw']
    depth = sparse.vstack([objective['depthOperators'][name] for name in REGIONS], format='csr')
    beforeDepth = depth @ raw['pureDirect'].ravel()
    restDepth = depth @ raw['alignedRest'].ravel()
    lower = np.minimum(np.minimum(beforeDepth, restDepth), 0.)
    upper = np.maximum(np.maximum(beforeDepth, restDepth), 0.)
    cavity = policy['cavity']
    guardPolicy = {key: cavity[key] for key in ('minimumRestEdgeRatio', 'maximumRestEdgeRatio',
        'minimumRestAreaRatio', 'maximumRestAreaRatio', 'minimumRestQualityRatio', 'minimumBaselineNormalCosine')}
    guardPolicy.update(restNormalCosineFloor=0., interiorDisplacementLimit=cavity['maximumDisplacementMm'],
        rimDisplacementLimit=policy['exterior']['maximumDisplacementMm'])
    return buildMeshShapeGuardSpec(raw['pureDirect'], raw['alignedRest'], raw['faces'],
        np.concatenate([raw[name + '_interiorVertexIds'] for name in REGIONS]),
        np.concatenate([raw[name + '_rimVertexIdsOrdered'] for name in REGIONS]),
        lengthScale=float(raw['unit']), depthOperator=depth, depthLower=lower, depthUpper=upper,
        policy=guardPolicy)


def runBounded(case, policy, settings, frozenHashes):
    """One frozen bounded configuration; failure never triggers another solve."""
    from faceAppearance.guardedMeshQuadratic import solveGuardedMeshQuadratic
    from faceAppearance.meshShapeConstraints import evaluateMeshShapeConstraints
    verifyHashes(frozenHashes)
    output = Path(case['preparedGeometry']).parent.parent / 'boundedWeight001'
    output.mkdir(exist_ok=False)
    raw = arrays(case['preparedGeometry'])
    objective = buildShapeObjective(raw, BOUNDED_WEIGHT)
    saveObjective(objective, output / 'objective')
    baseline, skinPairs, fullPairs = baselinePreflight(objective, policy)
    writeJson(output / 'baselineMetrics.json', baseline)
    spec = buildGuardSpec(objective, policy)
    sparse.save_npz(output / 'guardDepthOperator.npz', spec['depthOperator'])
    np.savez_compressed(output / 'guardSpec.npz', **{key: value for key, value in spec.items()
        if isinstance(value, np.ndarray)})
    writeJson(output / 'guardSpec.json', {'policy': spec['policy'], 'diagnostics': spec['diagnostics'],
        'lengthScale': spec['lengthScale'], 'baselineNormalCosineLower': spec['baselineNormalCosineLower']})
    solverConfig = {key: value for key, value in settings['commonToAllCases'].items() if key != 'weight'}
    started = perf_counter()
    try:
        result = solveGuardedMeshQuadratic(raw['pureDirect'], objective['H'], objective['b'],
            spec['controlVertexIds'], raw['staticIds'], lambda vertices: evaluateMeshShapeConstraints(vertices, spec),
            lengthScale=float(raw['unit']), **solverConfig)
    except Exception as error:
        failure = {'caseId': case['caseId'], 'status': 'boundedSolverException',
            'errorType': type(error).__name__, 'error': str(error), 'seconds': perf_counter() - started,
            'solverDiagnostics': {'solverAccepted': False}, 'finalCandidateAccepted': False,
            'acceptedForInitialGeometryDiagnostic': False, 'sourceHashes': frozenHashes}
        writeJson(output / 'report.json', failure)
        print(json.dumps(failure), flush=True)
        return failure
    candidate = result['rawVertices']
    np.savez_compressed(output / 'geometry.npz', vertices=candidate, pureDirect=raw['pureDirect'],
        alignedRest=raw['alignedRest'], faces=raw['faces'], delta=candidate - raw['pureDirect'])
    gate = shapeSafetyMetrics(objective['raw'], candidate, objective['depthOperators'], policy, skinPairs, fullPairs)
    values = evaluateMeshShapeConstraints(candidate, spec, withJacobian=False)
    residual = objective['H'] @ candidate.ravel() - objective['b']
    exteriorVertices = np.setdiff1d(np.arange(len(candidate)), np.union1d(spec['controlVertexIds'], raw['staticIds']))
    exteriorCoordinates = (exteriorVertices[:, None] * 3 + np.arange(3)).ravel()
    accepted = bool(result['diagnostics']['solverAccepted'] and gate['accepted'])
    item = {'caseId': case['caseId'], 'weight': BOUNDED_WEIGHT, 'method': 'boundedGlobalRelative3DPrior',
        'geometryGate': gate, 'solverDiagnostics': result['diagnostics'],
        'explicitGuardMinimum': float(values['values'].min()),
        'exportedObjectiveStationarity': {
            'outerMaximumAbsoluteGradient': float(2 * np.abs(residual[exteriorCoordinates]).max()),
            'outerGradientRms': float(2 * np.sqrt(np.mean(residual[exteriorCoordinates] ** 2))),
            'meaning': 'Only eliminated outer-coordinate stationarity is required; retained gradients include active inequality forces'},
        'status': 'initialGeometryDiagnosticPassed' if accepted else 'boundedRejected',
        'acceptedForInitialGeometryDiagnostic': accepted, 'finalCandidateAccepted': False,
        'geometrySha256': digest(output / 'geometry.npz'), 'sourceHashes': frozenHashes,
        'initialRegistrationOnly': True, 'downstreamGeometryRerun': False,
        'textureRebakedForInitialGeometry': False, 'exported': False, 'seconds': perf_counter() - started}
    writeJson(output / 'report.json', item)
    verifyHashes(frozenHashes)
    print(json.dumps({'caseId': case['caseId'], 'boundedAccepted': accepted,
        'solverAccepted': result['diagnostics']['solverAccepted'], 'reasons': gate['reasons'],
        'benefit': gate['weightedMissingDepthRmsMm']['improvementFraction'],
        'dataMseRatio': gate['sampleDataMseRatio'], 'report': str(output / 'report.json')}), flush=True)
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', type=Path, default=ROOT / 'runs/shapePreservation20261024/split.json')
    parser.add_argument('--policy', type=Path, default=ROOT / 'reports/shapePreservation20261024/independentAudit/proposedPolicy.json')
    parser.add_argument('--settings', type=Path, default=ROOT / 'reports/shapePreservation20261024/independentAudit/solverSettingsFrozen.json')
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--bounded', action='store_true')
    parser.add_argument('--validation', action='store_true', help='Fixed bounded holdout/control only after all dev reports pass')
    args = parser.parse_args()
    if args.ablation == args.bounded or (args.validation and not args.bounded):
        parser.error('Choose exactly one of --ablation or --bounded; --validation requires --bounded')
    split = json.loads(args.split.read_text())
    policy = normalizedShapePolicy(json.loads(args.policy.read_text()))
    settings = json.loads(args.settings.read_text())
    if settings['policySha256'] != digest(args.policy) or settings['commonToAllCases']['weight'] != BOUNDED_WEIGHT:
        raise ValueError('Frozen solver settings or policy mismatch')
    verifyHashes(split['commonSourceHashes'])
    plan = freezeExecution(args.split, args.policy, args.settings)
    mode = 'ablation' if args.ablation else 'boundedValidation' if args.validation else 'boundedDevelopment'
    implementation = args.split.parent / (mode + 'Implementation')
    implementation.mkdir(exist_ok=False)
    sources = [Path(__file__), ROOT / 'tools/experimentEarlyCavityRegistration.py',
        ROOT / 'src/faceAppearance/directTemplateRegistration.py',
        ROOT / 'src/faceAppearance/unsupportedShapePrior.py', ROOT / 'src/faceAppearance/clientGeometrySafety.py']
    if args.bounded:
        sources.extend([ROOT / 'src/faceAppearance/guardedMeshQuadratic.py', ROOT / 'src/faceAppearance/meshShapeConstraints.py'])
    hashes = {str(path): digest(path) for path in sources + [args.split, args.policy, args.settings, plan]}
    for path in sources:
        shutil.copyfile(path, implementation / path.name)
    frozen = {**split['commonSourceHashes'], **hashes}
    reports = []
    if args.validation:
        for dev in split['development']:
            report = json.loads((args.split.parent / dev / 'boundedWeight001/report.json').read_text())
            if not report['acceptedForInitialGeometryDiagnostic']:
                raise ValueError('All development cases must pass the fixed bounded policy before validation')
    for case in split['cases']:
        if (case['split'] == 'development') == args.validation:
            continue
        caseHashes = {case['preparedGeometry']: case['preparedGeometrySha256'],
            case['preparationReport']: case['preparationReportSha256']}
        verifyHashes(caseHashes)
        prepared = json.loads(Path(case['preparationReport']).read_text())
        inputs = {**frozen, **caseHashes, **prepared['sourceHashes']}
        reports.append(runAblation(case, policy, inputs) if args.ablation else runBounded(case, policy, settings, inputs))
    writeJson(args.split.parent / (mode + 'Report.json'), {'status': mode + 'Complete',
        'selectionPerformed': False, 'holdoutRun': args.validation, 'sourceHashes': frozen, 'cases': reports})


if __name__ == '__main__':
    main()
