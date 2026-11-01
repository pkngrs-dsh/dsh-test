"""Independent prospective gates for fixed-boundary eyelid deformation fields.

No candidate fitting, reassembly or production mutation occurs here. The
complete-aperture comparison to the completed coarse baseline is mandatory
for a claimed improvement, independently of the fitter's per-round decision.
Basis construction is audited separately once its prospective formula is
frozen; merely zeroing fixed rows cannot prove a continuous field.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from validateResidualContactExperiment import fullApertureRasterAudit, sameEvidence
from validateFeatureGeometryExperiment import _contourMasksAgree


def classifyBoundaryCandidate(*, producerAccepted, geometryChanged, incrementalSafe,
                              absoluteEligible, finalBaselineEffect, evidenceValidated):
    """Keep producer selection, requested improvement and deployment distinct."""
    flags = (producerAccepted, geometryChanged, incrementalSafe, absoluteEligible, evidenceValidated)
    if any(type(value) not in (bool, np.bool_) for value in flags):
        raise ValueError('Candidate states must be explicit booleans')
    defined = finalBaselineEffect.get('effectDefined') is True
    finalMaskAccepted = finalBaselineEffect.get('accepted') is True
    improved = bool(evidenceValidated and producerAccepted and geometryChanged and incrementalSafe and defined and finalMaskAccepted)
    return dict(producerAccepted=bool(producerAccepted), geometryChanged=bool(geometryChanged),
        incrementalSafe=bool(incrementalSafe), absoluteEligible=bool(absoluteEligible),
        requestedEffectAccepted=bool(defined and finalMaskAccepted), improvementAccepted=improved,
        changingAbsoluteEligible=bool(improved and absoluteEligible), clientReady=False,
        reason=None if improved else ('unchangedGeometry' if not geometryChanged else
            'finalBaselineMaskEffectUndefined' if not defined else
            'finalBaselineMaskEffectRejected' if not finalMaskAccepted else
            'independentEvidenceRejected' if not evidenceValidated else
            'incrementalSafetyRejected' if not incrementalSafe else 'producerRejected'),
        reference='The original completed coarse baseline, never a preceding residual round.')


def fieldRepresentationAudit(referenceVertices, basis, axes, *, fixedVertexIds,
                             maxDisplacement, cameraRotation, reportedActiveMask=None):
    """Validate the saved scalar field and its exact hard-boundary interface.

    This tests representation, rank and support as diagnostics. It does not
    assign arbitrary minimum support/rank or certify a construction formula.
    A completely inactive field is reported as ineffective, never repaired by
    enlarging the support or silently falling back to another field.
    """
    reference = np.asarray(referenceVertices); matrix = basis.toarray() if sparse.issparse(basis) else np.asarray(basis)
    axes = np.asarray(axes); fixed = np.asarray(fixedVertexIds); rotation = np.asarray(cameraRotation)
    if (reference.ndim != 2 or reference.shape[1] != 3 or matrix.ndim != 2 or len(matrix) != len(reference)
            or axes.shape != (3, 2) or rotation.shape != (3, 3)
            or fixed.ndim != 1 or fixed.dtype.kind not in 'iu'
            or fixed.size and (fixed.min() < 0 or fixed.max() >= len(reference))
            or not all(np.isfinite(value).all() for value in (reference, matrix, axes, rotation))
            or not np.isfinite(maxDisplacement) or maxDisplacement <= 0):
        raise ValueError('Expected finite ordered vertices, N by K field, camera-plane axes and fixed integer IDs')
    fixed = np.unique(fixed); active = np.any(matrix != 0, axis=1)
    checks = dict(fixedBasisRowsExactlyZero=np.array_equal(matrix[fixed], np.zeros_like(matrix[fixed])),
        cameraPlaneAxes=np.array_equal(axes, rotation[:2].T),
        originalCameraRotationValidity=np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) and np.linalg.det(rotation) >= .999)
    if reportedActiveMask is not None:
        value = np.asarray(reportedActiveMask)
        checks['actualActiveMask'] = value.dtype.kind == 'b' and value.shape == active.shape and np.array_equal(active, value)
    singular = np.linalg.svd(matrix, compute_uv=False) if matrix.size else np.empty(0)
    rankThreshold = np.finfo(float).eps * max(matrix.shape, default=0) * (singular[0] if len(singular) else 0)
    rank = int((singular > rankThreshold).sum())
    return dict(validationPassed=bool(all(checks.values())), checks=checks,
        activeVertexMask=active, activeVertexCount=int(active.sum()),
        columnSupportCounts=np.count_nonzero(matrix, axis=0), rowAbsoluteSums=np.abs(matrix).sum(1),
        controlCount=matrix.shape[1], numericalRank=rank, numericalRankThreshold=rankThreshold,
        singularValues=singular, ineffectiveField=not active.any(),
        maximumDepthAxisCoupling=float(np.max(np.abs(rotation[2] @ axes), initial=0)),
        depthSafety='Original raw-mesh camera-depth gate remains mandatory; no tighter camera orthogonality rule is introduced here.',
        fixedBoundaryProof='All saved scalar basis rows at fixed vertices are exactly zero; no post-solve clipping.',
        constructionFormulaAudited=False)


def reconstructFieldTrial(referenceVertices, basis, axes, maxDisplacement, coefficients):
    """Rebuild a raw trial directly from saved coefficients, before any gates."""
    reference, axes, coefficients = (np.asarray(value) for value in (referenceVertices, axes, coefficients))
    matrix = sparse.csr_matrix(basis) if sparse.issparse(basis) else np.asarray(basis)
    if (reference.ndim != 2 or reference.shape[1] != 3 or matrix.ndim != 2 or matrix.shape[0] != len(reference)
            or axes.shape != (3, 2) or coefficients.shape != (matrix.shape[1], 2)
            or not np.isfinite(reference).all() or not np.isfinite(axes).all() or not np.isfinite(coefficients).all()
            or not np.isfinite(matrix.data if sparse.issparse(matrix) else matrix).all()
            or not np.isfinite(maxDisplacement) or maxDisplacement <= 0):
        raise ValueError('Field and coefficient shapes/values must agree with the original camera-plane parameterization')
    return reference + maxDisplacement * ((matrix @ coefficients) @ axes.T)


def trialFieldEvidenceAudit(referenceVertices, basis, axes, *, maxDisplacement,
                            coefficientsBefore, coefficientStep, stepScale,
                            trialCoefficients, rawVertices, fixedVertexIds):
    """Check the actual coefficient step and reject post-solve fixed-row clipping."""
    before, step, proposed = (np.asarray(value) for value in (coefficientsBefore, coefficientStep, trialCoefficients))
    if (before.shape != step.shape or before.shape != proposed.shape or not np.isfinite(stepScale)
            or not 0 < stepScale <= 1 or not all(np.isfinite(value).all() for value in (before, step, proposed))):
        raise ValueError('A trial needs finite equal-shape coefficients and a line-search scale in (0,1]')
    expectedCoefficients = before + stepScale * step
    expectedVertices = reconstructFieldTrial(referenceVertices, basis, axes, maxDisplacement, proposed)
    raw, reference, fixed = np.asarray(rawVertices), np.asarray(referenceVertices), np.asarray(fixedVertexIds)
    if raw.shape != reference.shape or not np.isfinite(raw).all(): raise ValueError('Raw trial geometry must retain the original finite shape')
    if fixed.ndim != 1 or fixed.dtype.kind not in 'iu' or fixed.size and (fixed.min() < 0 or fixed.max() >= len(reference)):
        raise ValueError('Fixed vertex indices are invalid')
    matrix = basis.toarray() if sparse.issparse(basis) else np.asarray(basis)
    # Equality here is intentional: these are the original persisted arithmetic
    # operands, not independently rounded OBJ or reference surface projections.
    checks = dict(coefficientStepExactlyReproduced=np.array_equal(expectedCoefficients, proposed),
        rawGeometryExactlyFromField=np.array_equal(expectedVertices, raw),
        hardBoundaryFromBasis=np.array_equal(matrix[fixed], np.zeros_like(matrix[fixed])),
        fixedRawVerticesExact=np.array_equal(raw[fixed], reference[fixed]))
    return dict(validationPassed=bool(all(checks.values())), checks=checks,
        maximumRawReconstructionError=float(np.abs(expectedVertices - raw).max(initial=0)),
        changedVertexIds=np.flatnonzero(np.any(reference != raw, axis=1)),
        clippedRawGeometryDetected=bool(checks['fixedRawVerticesExact'] and not checks['rawGeometryExactlyFromField']))


def trialGateTraceAudit(gates, *, requiredGateNames, accepted, rejectionReasons):
    """Do not turn a skipped GN gate into an evaluated pass.

    Each gate is {evaluated: bool, accepted: bool | None, reasons: list[str]}.
    Rejected trials may short-circuit later gates only when at least one prior
    evaluated gate explicitly failed; all such failures must remain named.
    This schema is independent of outer full-mask/collision acceptance.
    """
    if type(accepted) not in (bool, np.bool_) or not isinstance(gates, dict):
        raise ValueError('Trial acceptance and gate mappings must be explicit')
    expected = tuple(requiredGateNames)
    if len(set(expected)) != len(expected) or set(gates) != set(expected):
        raise ValueError('Every required gate needs one explicit trace entry')
    evaluatedFailures, skipped, malformed = [], [], []
    for name in expected:
        row = gates[name]
        if type(row.get('evaluated')) not in (bool, np.bool_): malformed.append(name); continue
        if row['evaluated']:
            if type(row.get('accepted')) not in (bool, np.bool_): malformed.append(name)
            elif not row['accepted']: evaluatedFailures.append(name)
        else:
            skipped.append(name)
            if row.get('accepted') is not None: malformed.append(name)
    named = set(rejectionReasons)
    checks = dict(noInventedGatePass=not malformed,
        acceptedOnlyAfterAllGatesPass=not accepted or not (evaluatedFailures or skipped or malformed),
        rejectedTrialHasActualFailedGate=bool(accepted or evaluatedFailures),
        rejectionReasonsComplete=set(evaluatedFailures) <= named,
        noUnnamedOrFabricatedRejectionReasons=named <= set(evaluatedFailures))
    return dict(validationPassed=bool(all(checks.values())), checks=checks, failedGates=evaluatedFailures,
        notEvaluatedGates=skipped, malformedGates=malformed,
        scope='Inner GN trial trace only. Original outer full-scene, mask and historical gates remain separate.')


def replayGeometryEvidenceAudit(oldEvidence, newEvidence):
    """Compare a default-path replay with immutable prior fitted coordinates."""
    checks = {key + 'Exact': np.array_equal(oldEvidence[key], newEvidence[key])
        for key in ('vertices', 'rawCandidateVertices', 'solverVertices')}
    for key in ('beforeContours', 'afterContours', 'solverContours'):
        checks[key + 'MasksAndStatusExact'] = _contourMasksAgree(oldEvidence[key], newEvidence[key])
        checks[key + 'RoiExact'] = all(sameEvidence(oldEvidence[key]['eyes'][name].get('roi'), newEvidence[key]['eyes'][name].get('roi'))
            for name in oldEvidence[key]['eyes'])
    oldProposals, newProposals = oldEvidence['candidateProposals'], newEvidence['candidateProposals']
    checks['sameProposalCount'] = len(oldProposals) == len(newProposals)
    checks['allRawProposalsExact'] = len(oldProposals) == len(newProposals) and all(
        np.array_equal(a['vertices'], b['vertices']) and a['round'] == b['round'] and a['scale'] == b['scale']
        and a['selectedByFastRound'] == b['selectedByFastRound'] for a, b in zip(oldProposals, newProposals))
    oldStages=oldEvidence['diagnostics'].get('fast',{}).get('stages',[])
    newStages=newEvidence['diagnostics'].get('fast',{}).get('stages',[])
    checks['sameStagesAndBindings']=len(oldStages)==len(newStages) and all(sameEvidence(a.get('binding'),b.get('binding')) and all(a.get(k)==b.get(k) for k in ('round','status','selectedScale')) for a,b in zip(oldStages,newStages))
    keys=('inputTargetGate','fixedBoundary','initialMeanErrorPixels','finalMeanErrorPixels','initialGroups','finalGroups','bestIteration','iterationsCompleted','controlCount','activeVertexCount','maximumReferenceDepthChange','supportRadius','maxDisplacement')
    checks['sameLegacySolverState']=len(oldStages)==len(newStages) and all(all(sameEvidence(a.get('solver',{}).get(k),b.get('solver',{}).get(k)) for k in keys) for a,b in zip(oldStages,newStages))
    for key in ('accepted', 'status', 'geometryChanged', 'contactOptimizationEnabled'):
        checks[key + 'Unchanged'] = oldEvidence['diagnostics'].get(key) == newEvidence['diagnostics'].get(key)
    return dict(validationPassed=bool(all(checks.values())), checks=checks,
        ignoredFields='Runtime durations and newly added diagnostic trace fields only; no coordinate, selection or mask tolerance.')


def independentBoundaryFieldSystem(vertices, faces, controlPoints, *, fixedVertexIds,
                                   supportRadius, controlCount=32, smoothingRadiusFraction=.1):
    """Reconstruct the frozen FEM objective with independent barycentric gradients.

    No linear system is solved here. In particular Q is formed on the entire
    original mesh BEFORE free-row restriction. A fixed neighboring row remains
    in the curvature energy; truncating K first defines a different field.
    """
    vertices, points = np.asarray(vertices, np.float64), np.asarray(controlPoints, np.float64)
    faces, fixed = np.asarray(faces), np.asarray(fixedVertexIds)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices)
            or points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(vertices).all() or not np.isfinite(points).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu' or not len(faces)
            or faces.min() < 0 or faces.max() >= len(vertices)
            or fixed.ndim != 1 or fixed.dtype.kind not in 'iu' or fixed.size and (fixed.min() < 0 or fixed.max() >= len(vertices))
            or type(controlCount) not in (int, np.int32, np.int64) or controlCount <= 0
            or not np.isfinite(supportRadius) or supportRadius <= 0
            or not np.isfinite(smoothingRadiusFraction) or smoothingRadiusFraction <= 0):
        raise ValueError('Expected valid original triangle mesh, original control samples, fixed IDs and positive frozen scales')
    fixed = np.unique(fixed); center = vertices.mean(0); normalized = (vertices - center) / supportRadius
    triangles = normalized[faces]; cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    doubleAreas = np.linalg.norm(cross, axis=1)
    if (doubleAreas <= 0).any() or not np.isfinite(doubleAreas).all(): raise ValueError('FEM triangles must be nondegenerate')
    normals = cross / doubleAreas[:, None]
    opposite = np.stack((triangles[:, 2] - triangles[:, 1], triangles[:, 0] - triangles[:, 2], triangles[:, 1] - triangles[:, 0]), axis=1)
    gradients = np.cross(normals[:, None, :], opposite) / doubleAreas[:, None, None]
    localStiffness = (doubleAreas / 2)[:, None, None] * np.einsum('tix,tjx->tij', gradients, gradients)
    rows = np.broadcast_to(faces[:, :, None], localStiffness.shape).ravel()
    columns = np.broadcast_to(faces[:, None, :], localStiffness.shape).ravel()
    stiffness = sparse.csr_matrix((localStiffness.ravel(), (rows, columns)), shape=(len(vertices), len(vertices)))
    mass = np.bincount(faces.ravel(), weights=np.repeat(doubleAreas / 6, 3), minlength=len(vertices))
    free = np.setdiff1d(np.arange(len(vertices)), fixed)
    if (mass[free] <= 0).any() or not np.isfinite(mass).all(): raise ValueError('Every free field vertex needs positive lumped mass')
    inverseMass = np.zeros_like(mass); np.divide(1., mass, out=inverseMass, where=mass > 0)
    curvature = stiffness.T @ sparse.diags(inverseMass) @ stiffness
    unique = np.unique(points, axis=0)
    chosen = []
    if len(unique):
        chosen = [int(np.argmax(np.linalg.norm(unique - unique.mean(0), axis=1)))]
        distances = np.sum((unique - unique[chosen[0]]) ** 2, axis=1)
        for _ in range(1, min(controlCount, len(unique))):
            chosen.append(int(distances.argmax()))
            distances = np.minimum(distances, np.sum((unique - unique[chosen[-1]]) ** 2, axis=1))
    controls = unique[chosen]
    r = np.linalg.norm(vertices[:, None, :] - controls[None, :, :], axis=2) / supportRadius
    compact = np.maximum(1 - r, 0) ** 4 * (4 * np.minimum(r, 1) + 1)
    system = sparse.diags(mass[free]) + smoothingRadiusFraction ** 4 * curvature[free][:, free]
    rhs = mass[free, None] * compact[free]
    return dict(referenceVertices=vertices, faces=faces, center=center, normalizedVertices=normalized,
        selectedControlPoints=controls, compactRaw=compact, stiffness=stiffness, lumpedMass=mass,
        fullCurvature=curvature, freeVertexIds=free, fixedVertexIds=fixed, operator=system.tocsr(), rhs=rhs,
        smoothingRadiusFraction=float(smoothingRadiusFraction), requestedSupportRadius=float(supportRadius),
        controlCountRequested=int(controlCount), negativeCotangentsClamped=False, systemSolved=False)


def boundaryFieldConstructionAudit(system, savedBasis, savedControlPoints, normalization, *, equationTolerance=1e-10):
    """Verify full-domain FEM minimization and the common absolute-row-sum scale."""
    basis = np.asarray(savedBasis); controls = np.asarray(savedControlPoints)
    expectedShape = system['compactRaw'].shape
    if (basis.shape != expectedShape or not np.isfinite(basis).all()
            or not np.isfinite(normalization) or normalization < 1
            or not np.isfinite(equationTolerance) or equationTolerance <= 0):
        raise ValueError('Saved basis/normalization must be finite and match the frozen selected centers')
    raw = basis * normalization; free, fixed = system['freeVertexIds'], system['fixedVertexIds']
    expectedScale = max(1., float(np.abs(raw).sum(1).max(initial=0)))
    residual = system['operator'] @ raw[free] - system['rhs']
    operatorNorm = float(np.asarray(np.abs(system['operator']).sum(1)).max(initial=0))
    rawNorm = float(np.abs(raw[free]).sum(1).max(initial=0))
    rhsNorm = float(np.abs(system['rhs']).sum(1).max(initial=0))
    residualNorm = float(np.abs(residual).sum(1).max(initial=0))
    denominator = max(operatorNorm * rawNorm, rhsNorm, np.finfo(float).tiny)
    backwardError = residualNorm / denominator
    scaleTolerance = 64 * np.finfo(float).eps * max(1., expectedScale, normalization)
    checks = dict(originalFarthestCenters=np.array_equal(controls, system['selectedControlPoints']),
        fixedRowsExactlyZero=np.array_equal(basis[fixed], np.zeros_like(basis[fixed])),
        freeFullFEMEquation=backwardError <= equationTolerance,
        sharedAbsoluteRowNormalization=abs(normalization - expectedScale) <= scaleTolerance,
        noRowwiseRenormalization=float(np.abs(basis).sum(1).max(initial=0)) <= 1 + 64 * np.finfo(float).eps)
    return dict(validationPassed=bool(all(checks.values())), checks=checks,
        equationBackwardError=backwardError, equationTolerance=equationTolerance,
        normalizationExpected=expectedScale, normalizationSaved=float(normalization), normalizationTolerance=scaleTolerance,
        freeVertexCount=len(free), fixedVertexCount=len(fixed), selectedControlCount=basis.shape[1],
        activeVertexCount=int(np.any(basis != 0, axis=1).sum()),
        independentMethod='Triangle barycentric gradient stiffness; full K^T M^-1 K before restriction; verify saved field without re-solving.',
        coordinateTranslationAndRadiusNormalization=True, noRidge=True, noCotangentClamp=True)


def _csrFromEvidence(value):
    shape = tuple(np.asarray(value['shape'], int))
    result = sparse.csr_matrix((np.asarray(value['data']), np.asarray(value['indices']), np.asarray(value['indptr'])), shape=shape)
    if len(shape) != 2 or not np.isfinite(result.data).all(): raise ValueError('Saved FEM CSR is invalid')
    return result


def _gramMatrixAgreement(expected, actual, relativeTolerance=1e-10):
    """Use PSD diagonal scales for each differing entry, not one global maximum."""
    if expected.shape != actual.shape: return dict(accepted=False, reason='shapeChanged')
    difference = (expected - actual).tocoo()
    diagonal = np.maximum(expected.diagonal(), 0)
    scale = np.sqrt(diagonal[difference.row] * diagonal[difference.col])
    tolerance = relativeTolerance * np.maximum(scale, np.finfo(float).tiny)
    bad = np.abs(difference.data) > tolerance
    return dict(accepted=bool(not bad.any()), differingEntryCount=difference.nnz,
        rejectedEntryCount=int(bad.sum()), maximumAbsoluteDifference=float(np.abs(difference.data).max(initial=0)),
        relativeTolerance=relativeTolerance, comparisonScale='sqrt(expected diagonal_i * diagonal_j), independently for every matrix entry')


def savedBoundaryFieldAudit(evidence, basis, *, requestedControlCount=32, supportRadius,
                            smoothingRadiusFraction=.1, expectedFixedVertexIds):
    """Check the actual construction evidence against the prospectively frozen formula."""
    system = independentBoundaryFieldSystem(evidence['referenceVertices'], evidence['referenceFaces'], evidence['inputControlPoints'],
        fixedVertexIds=expectedFixedVertexIds, supportRadius=supportRadius, controlCount=requestedControlCount,
        smoothingRadiusFraction=smoothingRadiusFraction)
    solution = boundaryFieldConstructionAudit(system, basis, evidence['controlPoints'], evidence['normalization'])
    checks = dict(savedRadius=evidence['supportRadius'] == supportRadius,
        savedSmoothing=evidence['smoothingRadiusFraction'] == smoothingRadiusFraction,
        exactFixedIds=np.array_equal(evidence['fixedVertexIds'], system['fixedVertexIds']),
        exactFreeIds=np.array_equal(evidence['freeVertexIds'], system['freeVertexIds']),
        exactCoordinateCenter=np.array_equal(evidence['coordinateCenter'], system['center']),
        exactNormalizedVertices=np.array_equal(evidence['normalizedVertices'], system['normalizedVertices']),
        positiveMass=np.allclose(evidence['lumpedMass'], system['lumpedMass'], atol=0, rtol=1e-12),
        ordinaryUnnormalizedWendlandDrivers=np.allclose(evidence['drivingBasis'], system['compactRaw'], atol=64*np.finfo(float).eps, rtol=64*np.finfo(float).eps),
        rightHandSide=np.allclose(evidence['rightHandSide'], system['rhs'], atol=64*np.finfo(float).eps, rtol=64*np.finfo(float).eps),
        rawBasisNormalization=np.allclose(np.asarray(evidence['rawBasis']) / evidence['normalization'], basis, atol=0, rtol=0),
        linearSystemSolution=solution['validationPassed'])
    matrixReports = {}
    for saved, independent in (('stiffness', 'stiffness'), ('bending', 'fullCurvature'), ('systemMatrix', 'operator')):
        matrixReports[saved] = _gramMatrixAgreement(system[independent], _csrFromEvidence(evidence[saved]))
        checks[saved + 'IndependentFEM'] = matrixReports[saved]['accepted']
    return dict(validationPassed=bool(all(checks.values())), checks=checks, linearSystem=solution, matrixComparisons=matrixReports,
        continuousFieldMeaning='Piecewise linear FEM interpolation of the solved displacement values; no claim of a globally C2 surface field.',
        independentMatrixConstruction='Barycentric triangle gradients, without the producer cotangent assembler.')


def independentFieldRegularizer(vertices, faces, basis, activeMask, *, maxDisplacement,
                                displacementWeight, strainWeight, controlWeight):
    """Rebuild the unchanged solver regularizer in the new coefficient basis."""
    vertices, faces, basis, active = (np.asarray(value) for value in (vertices, faces, basis, activeMask))
    count = basis.shape[1]
    if (active.shape != (len(vertices),) or active.dtype.kind != 'b' or basis.shape[0] != len(vertices)
            or not np.isfinite([maxDisplacement, displacementWeight, strainWeight, controlWeight]).all()
            or maxDisplacement <= 0 or min(displacementWeight, strainWeight, controlWeight) < 0):
        raise ValueError('Invalid original regularization dimensions or weights')
    edges = np.unique(np.sort(np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    if (lengths <= 0).any(): raise ValueError('Original solver edges must have positive lengths')
    regularizer = controlWeight / max(count, 1) * np.eye(2 * count)
    if active.any():
        small = displacementWeight / int(active.sum()) * (basis[active].T @ basis[active])
        local = active[edges].any(1)
        if local.any():
            difference = (basis[edges[local, 1]] - basis[edges[local, 0]]) * (maxDisplacement / lengths[local])[:, None]
            small += strainWeight / int(local.sum()) * (difference.T @ difference)
        regularizer += np.kron(small, np.eye(2))
    return regularizer


def independentSolverObjective(evidence, coefficients, camera, *, regularizer,
                               pixelScale, curveWeight, referenceWeight=1., additionalEnergy=0., actualVertices=None):
    """Evaluate photo/curve/regularization terms without calling the solver."""
    reference, basis, axes = (np.asarray(evidence[key]) for key in ('referenceVertices', 'basis', 'axes'))
    coefficients = np.asarray(coefficients); prepared = evidence['preparedConstraints']
    if not np.isfinite([pixelScale, curveWeight, referenceWeight, additionalEnergy]).all() or pixelScale <= 0 or min(curveWeight, referenceWeight, additionalEnergy) < 0:
        raise ValueError('Invalid frozen objective scales')
    # The wrapper provides one original front camera. Other-view source gates
    # remain independent and are not mixed into this local fit objective.
    if any(row['viewIndex'] != 0 for row in prepared): raise ValueError('Expected unchanged single-front-camera residual solver')
    totalWeight = sum(float(np.asarray(row['weights']).sum()) for row in prepared)
    multiplier = referenceWeight / max(referenceWeight, 1.)
    flat = coefficients.ravel(); regularization = .5 * float(flat @ regularizer @ flat)
    photo = meanError = weightedViewError = 0.; projectionValid = True; bindingsValid = True
    def robust(error):
        radius = np.linalg.norm(error, axis=1) / pixelScale
        return np.where(radius <= 1, .5 * radius ** 2, radius - .5)
    for row in prepared:
        ids, bary = np.asarray(row['vertexIndices']), np.asarray(row['barycentric'])
        originalPoints = (reference[ids] * bary[..., None]).sum(1)
        sampleBasis = (basis[ids] * bary[..., None]).sum(1)
        weights = np.asarray(row['weights']); normalized = weights * multiplier / totalWeight
        bindingsValid &= np.array_equal(originalPoints, row['originalPoints']) and np.array_equal(sampleBasis, row['basis'])
        bindingsValid &= np.allclose(normalized, row['normalizedWeights'], atol=1e-15, rtol=1e-14)
        point = (originalPoints + evidence['maxDisplacement'] * ((sampleBasis @ coefficients) @ axes.T) if actualVertices is None else (np.asarray(actualVertices)[ids] * bary[..., None]).sum(1))
        cameraPoints = point @ np.asarray(camera['rotation']).T + camera['translation']
        homogeneous = cameraPoints @ np.asarray(camera['intrinsics']).T
        pixels = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-12)
        projectionValid &= bool(np.isfinite(pixels).all() and (cameraPoints[:, 2] > 1e-6).all())
        error = pixels - row['targetPixels']; norms = np.linalg.norm(error, axis=1)
        meanError += float(normalized @ norms); weightedViewError += float(weights @ norms)
        directional = np.einsum('nij,nj->ni', row['transform'], error)
        photo += float(normalized @ robust(directional))
        pairs = np.asarray(row['segmentPairs'])
        if len(pairs) and curveWeight:
            first, second = pairs.T
            difference = (pixels[second] - pixels[first]) - (row['targetPixels'][second] - row['targetPixels'][first])
            photo += curveWeight * float((.5 * (normalized[first] + normalized[second])) @ robust(difference))
    return dict(objective=regularization + additionalEnergy + photo,
        objectiveComponents=dict(regularization=regularization, additionalResidual=additionalEnergy, photoAndCurve=photo),
        meanErrorPixels=meanError, viewErrors={0: weightedViewError / totalWeight} if totalWeight else {},
        projectionValid=projectionValid, preparedBindingsValidated=bool(bindingsValid))


def _close(a, b, *, atol=1e-10, rtol=1e-9):
    if isinstance(a, dict) or isinstance(b, dict):
        if not isinstance(a, dict) or not isinstance(b, dict): return False
        a, b = {str(k): v for k, v in a.items()}, {str(k): v for k, v in b.items()}
        return a.keys() == b.keys() and all(_close(a[k], b[k], atol=atol, rtol=rtol) for k in a)
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)): return a == b
    return np.shape(a) == np.shape(b) and np.allclose(a, b, atol=atol, rtol=rtol)


def localTrialSafety(reference, candidate, faces, limit):
    edges = np.unique(np.sort(np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1), axis=0)
    oldCross = np.cross(reference[faces[:, 1]]-reference[faces[:, 0]], reference[faces[:, 2]]-reference[faces[:, 0]])
    newCross = np.cross(candidate[faces[:, 1]]-candidate[faces[:, 0]], candidate[faces[:, 2]]-candidate[faces[:, 0]])
    oldArea, newArea = np.linalg.norm(oldCross, axis=1), np.linalg.norm(newCross, axis=1)
    oldLength = np.linalg.norm(reference[edges[:, 1]]-reference[edges[:, 0]], axis=1)
    edge = np.abs(np.linalg.norm(candidate[edges[:, 1]]-candidate[edges[:, 0]], axis=1)/oldLength-1)
    metrics = dict(minimumAreaRatio=float((newArea/oldArea).min()),
        minimumNormalCosine=float((np.sum(newCross*oldCross, axis=1)/np.maximum(newArea*oldArea, 1e-25)).min()),
        maximumRelativeEdgeChange=float(edge.max()), maximumDisplacement=float(np.linalg.norm(candidate-reference, axis=1).max()))
    limits = dict(area=metrics['minimumAreaRatio'] >= .4, normal=metrics['minimumNormalCosine'] >= .3,
        edge=metrics['maximumRelativeEdgeChange'] <= .25, displacement=metrics['maximumDisplacement'] <= limit*(1+1e-8), finite=np.isfinite(candidate).all())
    return dict(accepted=bool(all(limits.values())), metrics=metrics, gates=limits)


def independentFrontFiltering(constraints, reference, camera):
    """Original per-observation eligibility; no targets or confidences invented."""
    kept, reports, prepared = [], [], []
    for index, row in enumerate(constraints):
        ids, bary, target = (np.asarray(row[k]) for k in ('vertexIndices', 'barycentric', 'targetPixels'))
        weights = np.asarray(row.get('weights', np.ones(len(ids))), float)
        confidence = np.asarray(row.get('confidence', np.ones(len(ids))), float)
        points = (reference[ids]*bary[..., None]).sum(1)
        view = points @ np.asarray(camera['rotation']).T + camera['translation']
        homogeneous = view @ np.asarray(camera['intrinsics']).T
        pixels = homogeneous[:, :2]/np.maximum(homogeneous[:, 2:], 1e-12)
        distance = np.linalg.norm(pixels-target, axis=1); reasons = np.full(len(ids), 'kept', object)
        for mask, reason in (((weights <= 0)|(weights < .05), 'lowWeight'), (confidence <= 0, 'lowConfidence'),
            ((view[:, 2] <= 1e-6)|~np.isfinite(pixels).all(1), 'invalidProjection'),
            (~np.isfinite(distance)|(distance > 40.), 'targetTooDistant')):
            reasons[mask & (reasons == 'kept')] = reason
        selected = reasons == 'kept'
        reports.append(dict(groupIndex=index, name=row.get('name', 'frontLandmarks'), inputCount=len(ids), keptCount=int(selected.sum()),
            keptIndices=np.flatnonzero(selected).tolist(), rejectedIndices=np.flatnonzero(~selected).tolist(), reasons=reasons.tolist(),
            initialDistancePixels=[float(v) if np.isfinite(v) else None for v in distance]))
        if not selected.any(): continue
        if 'segmentPairs' in row: pairs = np.asarray(row['segmentPairs'], int)
        elif row.get('orderedCurve', False):
            first = np.flatnonzero(np.diff(row.get('sampleIndices', np.arange(len(ids)))) == 1); pairs = np.c_[first, first+1]
        else: pairs = np.empty((0, 2), int)
        pairs = pairs[selected[pairs].all(1)]; remap = np.full(len(ids), -1, int); remap[selected] = np.arange(selected.sum())
        item = dict(viewIndex=0, name=row.get('name', 'frontLandmarks'), vertexIndices=ids[selected].copy(), barycentric=bary[selected].copy(),
            targetPixels=target[selected].copy(), weights=(weights[selected]*confidence[selected]).copy(),
            tangentWeight=float(row.get('tangentWeight', 1.)), segmentPairs=remap[pairs])
        transform = np.tile(np.eye(2), (selected.sum(), 1, 1))
        if 'targetTangents' in row:
            tangent = np.asarray(row['targetTangents'], float)[selected]; item['targetTangents'] = tangent.copy()
            tangent /= np.linalg.norm(tangent, axis=1)[:, None]
            if item['tangentWeight'] != 1:
                transform[:, 0] = np.c_[-tangent[:, 1], tangent[:, 0]]; transform[:, 1] = np.sqrt(item['tangentWeight'])*tangent
        kept.append(item); prepared.append(dict(item, transform=transform))
    points = np.concatenate([(reference[r['vertexIndices']]*r['barycentric'][..., None]).sum(1) for r in kept]) if kept else np.empty((0, 3))
    return kept, reports, prepared, points


def independentCompactField(reference, points, radius, count=32):
    unique = np.unique(points, axis=0)
    if not len(unique): return np.zeros((len(reference), 0)), unique, 1.
    selected = [int(np.linalg.norm(unique-unique.mean(0), axis=1).argmax())]
    distance = np.sum((unique-unique[selected[0]])**2, axis=1)
    for _ in range(1, min(count, len(unique))):
        selected.append(int(distance.argmax())); distance = np.minimum(distance, np.sum((unique-unique[selected[-1]])**2, axis=1))
    centers = unique[selected]; radiusValue = np.linalg.norm(reference[:, None]-centers[None], axis=2)/radius
    clipped = np.minimum(radiusValue, 1); basis = np.where(radiusValue < 1, (1-clipped)**4*(4*clipped+1), 0.)
    normalization = max(1., float(basis.sum(1).max(initial=0)))
    return basis/normalization, centers, normalization


def independentGaussNewtonSystem(evidence, coefficients, camera, regularizer, *, fullReference, fullVertexIds, contactOperator, contactSpec):
    """Rebuild IRLS normal equations and gradient; never solve a candidate."""
    B, axes = evidence['basis'], evidence['axes']; limit = evidence['maxDisplacement']; flat = coefficients.ravel()
    normal, gradient = regularizer.copy(), regularizer @ flat
    for row in evidence['preparedConstraints']:
        points = row['originalPoints'] + limit*((row['basis'] @ coefficients) @ axes.T)
        view = points @ np.asarray(camera['rotation']).T + camera['translation']; K = np.asarray(camera['intrinsics']); R = np.asarray(camera['rotation'])
        h = view @ K.T; z = np.maximum(h[:, 2:], 1e-12); pixels = h[:, :2]/z
        derivative = ((K[:2][None]*z[:, :, None]-h[:, :2, None]*K[2])/z[:, :, None]**2) @ R @ axes
        jacobian = (derivative[:, :, None, :]*row['basis'][:, None, :, None]*limit).reshape(len(points), 2, -1)
        errors = np.einsum('nij,nj->ni', row['transform'], pixels-row['targetPixels'])
        jacobian = np.einsum('nij,njk->nik', row['transform'], jacobian)
        radius = np.linalg.norm(errors, axis=1)/4.
        irls = np.sqrt(row['normalizedWeights']/np.maximum(radius, 1.))/4.
        J = (jacobian*irls[:, None, None]).reshape(-1, len(flat)); residual = (errors*irls[:, None]).ravel()
        normal += J.T @ J; gradient += J.T @ residual
    full = fullReference.copy(); full[fullVertexIds] = reconstructFieldTrial(evidence['referenceVertices'], B, axes, limit, coefficients)
    gap = contactOperator @ full.ravel(); count = len(gap)
    if count:
        columns = (fullVertexIds[:, None]*3+np.arange(3)).ravel()
        chained = (contactOperator[:, columns] @ (limit*sparse.kron(B, axes, format='csr'))).toarray()
        scale = np.sqrt(contactSpec['weight']/count)/contactSpec['lengthScale']
        J = scale*(gap < contactSpec['minimumGap'])[:, None]*chained
        residual = scale*np.minimum(gap-contactSpec['minimumGap'], 0)
        normal += J.T @ J; gradient += J.T @ residual
    return normal, gradient


def auditSolverFields(branch, prepared, rawEvidence, camera, baseline, scene, support, constraints, imageScale, spec, operator, measure, config):
    """Independently reconstruct fields, inner trial gates and outer field steps."""
    from faceAppearance.fastFrontEyeGeometry import _constraints
    records = rawEvidence.get('solverEvidence', []); stages = rawEvidence['diagnostics'].get('fast', {}).get('stages', [])
    allChecks = dict(everySolverCaptured=[r['round'] for r in records] == [s['round'] for s in stages if 'solver' in s])
    mode = config['fieldPolicy']['legacyMode' if branch == 'legacyResidualReplay' else 'candidateMode']
    ids = np.unique(np.r_[scene['nicpToFullIds'], scene['faces'][prepared['cornerFaceIds']].ravel()])
    inverse = np.full(len(baseline), -1, int); inverse[ids] = np.arange(len(ids))
    faceIds = np.flatnonzero((inverse[scene['faces']] >= 0).all(1)); faces = inverse[scene['faces'][faceIds]]
    localFixed = inverse[support['fixedVertexIds']]; localFixed = localFixed[localFixed >= 0]
    fixedFull = np.ones(len(baseline), bool); fixedFull[ids] = False; fixedFull[support['fixedVertexIds']] = True
    extent = float(np.ptp(baseline[scene['nicpToFullIds']], axis=0).max()); limit, radius = extent*.03, extent*.12
    current = baseline.copy(); reports = []
    for record in records:
        e = record['evidence']; roundId = record['round']; stage = next(row for row in stages if row['round'] == roundId)
        checks = dict(fullVertexIds=np.array_equal(record['fullVertexIds'], ids), fullFaceIds=np.array_equal(record['fullFaceIds'], faceIds),
            fullFixed=np.array_equal(record['fullFixedVertexIds'], np.flatnonzero(fixedFull)), originalRoundReference=np.array_equal(record['fullReferenceVertices'], current),
            localOriginal=np.array_equal(e['referenceVertices'], current[ids]), localFaces=np.array_equal(e['faces'], faces),
            localFixed=np.array_equal(e['fixedVertexIds'], localFixed), mode=e['deformationMode'] == mode,
            radius=e['requestedSupportRadius'] == radius, maxDisplacement=e['maxDisplacement'] == limit)
        originalConstraints, _ = _constraints(current, scene['faces'], scene['materialIds'], camera, prepared['photoEyes'], measure(current), ids,
            prepared['cornerFaceIds'], scene['uvs'], scene['uvFaces'], scene['materials'], scene['alphaTextures'], fixedVertexIds=support['fixedVertexIds'])
        kept, inputReports, preparedRows, points = independentFrontFiltering(originalConstraints, current[ids], camera)
        checks['originalFilteredConstraints'] = sameEvidence(kept, e['keptConstraints'])
        checks['originalEligibility'] = all(sameEvidence({k:v for k,v in a.items() if k != 'initialDistancePixels'}, {k:v for k,v in b.items() if k != 'initialDistancePixels'}) and _close(a['initialDistancePixels'], b['initialDistancePixels']) for a,b in zip(inputReports, e['inputReports'])) and len(inputReports) == len(e['inputReports'])
        checks['inputControlPoints'] = np.array_equal(points, e['inputControlPoints'])
        checks['preparedConstraintCount'] = len(preparedRows) == len(e['preparedConstraints'])
        for index, (a,b) in enumerate(zip(preparedRows, e['preparedConstraints'])):
            checks['prepared'+str(index)] = all(sameEvidence(a[k], b[k]) for k in ('viewIndex','name','vertexIndices','barycentric','targetPixels','weights','segmentPairs','transform'))
        field = fieldRepresentationAudit(e['referenceVertices'], e['basis'], e['axes'], fixedVertexIds=localFixed, maxDisplacement=limit,
            cameraRotation=np.asarray(camera['rotation']), reportedActiveMask=e['activeVertexMask'])
        checks['fieldRepresentation'] = field['validationPassed']
        if mode == 'fixedBoundaryBiharmonic':
            construction = savedBoundaryFieldAudit(e['fieldConstruction'], e['basis'], requestedControlCount=32,
                supportRadius=radius, smoothingRadiusFraction=.1, expectedFixedVertexIds=localFixed)
            checks['fieldConstruction'] = construction['validationPassed']
            checks['fieldActualConstructionIdentity'] = np.array_equal(e['referenceVertices'], e['fieldConstruction']['referenceVertices']) and np.array_equal(faces, e['fieldConstruction']['referenceFaces']) and np.array_equal(points, e['fieldConstruction']['inputControlPoints'])
            checks['normalizationIdentity'] = e['normalization'] == e['fieldConstruction']['normalization']
            checks['unshrunkRequestedRadius'] = e['effectiveSupportRadius'] == radius
        else:
            _, centers, _ = independentCompactField(current[ids], points, radius)
            nearest = np.linalg.norm(centers[:, None]-current[ids][localFixed][None], axis=2).min() if len(centers) and len(localFixed) else radius/(1-1e-12)
            effective = min(radius, float(nearest)*(1-1e-12)); blocked = effective <= float(np.ptp(current[ids],axis=0).max())*1e-12
            expectedB, expectedCenters, norm = independentCompactField(current[ids], np.empty((0,3)) if blocked else points, radius if blocked else effective)
            checks['compactFormula'] = np.array_equal(e['basis'], expectedB) and np.array_equal(e['controlPoints'], expectedCenters) and e['normalization'] == norm
            checks['oldRadiusCap'] = e['effectiveSupportRadius'] == effective
            checks['noNewFieldOnReplay'] = e['fieldConstruction'] is None
            construction = dict(validationPassed=checks['compactFormula'] and checks['oldRadiusCap'], compactRadius=effective, requestedRadius=radius, blocked=bool(blocked))
        reg = independentFieldRegularizer(e['referenceVertices'], faces, e['basis'], e['activeVertexMask'], maxDisplacement=limit, displacementWeight=.002, strainWeight=.03, controlWeight=.0001)
        checks['unchangedPhysicalAndCoefficientRegularizer'] = _close(reg, e['regularizer'], atol=1e-12, rtol=1e-10)
        def fullFrom(local):
            result = current.copy(); result[ids] = local; return result
        def energy(local):
            gaps = operator @ fullFrom(local).ravel()
            residual = np.sqrt(spec['weight']/len(gaps))*np.minimum((gaps-spec['minimumGap'])/spec['lengthScale'],0) if len(gaps) else np.empty(0)
            return .5*float(residual@residual)
        def evaluate(coeff, actual=None):
            local = reconstructFieldTrial(e['referenceVertices'], e['basis'], e['axes'], limit, coeff) if actual is None else actual
            return independentSolverObjective(e, coeff, camera, regularizer=reg, pixelScale=4., curveWeight=0., additionalEnergy=energy(local), actualVertices=actual)
        def safety(local, saved):
            mesh = localTrialSafety(e['referenceVertices'],local,faces,limit); full = fullFrom(local)
            coarse = coarseEyePreservationAudit(baseline, full, constraints, camera, imageScale=imageScale)
            contact = frozenContactGapAudit(baseline,full,spec['sourceVertexIds'],spec['targetVertexIds'],spec['targetBarycentric'],spec['normals'],spec['minimumGap'],lengthScale=spec['lengthScale'],tolerance=1e-8)
            safe = mesh['accepted'] and coarse['accepted'] and contact['accepted']
            detail = dict(mesh=mesh, coarse=coarse, contact=contact)
            consistent = saved['geometricSafetyAccepted'] == mesh['accepted'] and _close(mesh['metrics'], {k:saved[k] for k in mesh['metrics']})
            extra = saved['additionalSafety']; consistent &= extra['accepted'] == (coarse['accepted'] and contact['accepted']) and extra['coarseAlignment']['accepted'] == coarse['accepted'] and extra['contact']['accepted'] == contact['accepted']
            return bool(safe), bool(consistent), detail
        coefficients = np.zeros_like(e['finalCoefficients']); initial = evaluate(coefficients); best = coefficients.copy(); bestObjective = initial['objective']; bestIteration = 0
        checks['initialObjective'] = _close(initial['objective'], e['initialObjective']) and _close(initial['objectiveComponents'], e['initialObjectiveComponents']) and initial['preparedBindingsValidated']
        trials = []; lastIteration = 0; lastBacktrack = -1; lastAccepted = False; previousDamping = .001
        for trial in e['trials']:
            proposed = trial['trialCoefficients']; raw = reconstructFieldTrial(e['referenceVertices'],e['basis'],e['axes'],limit,proposed)
            proof = trialFieldEvidenceAudit(e['referenceVertices'],e['basis'],e['axes'],maxDisplacement=limit,
                coefficientsBefore=trial['beforeCoefficients'],coefficientStep=trial['coefficientStep'],stepScale=trial['stepScale'],trialCoefficients=proposed,rawVertices=raw,fixedVertexIds=localFixed)
            pc = dict(coefficientArithmetic=proof['validationPassed'], selectedCoefficientChain=np.array_equal(coefficients,trial['beforeCoefficients']),
                boundedBacktrack=trial['stepScale']==.5**trial['backtrack'] and 0<=trial['backtrack']<14,
                boundedIteration=1<=trial['iteration']<=20,
                trialOrder=(trial['iteration']>lastIteration and trial['backtrack']==0) or trial['iteration']==lastIteration and trial['backtrack']==lastBacktrack+1 and not lastAccepted)
            if trial['iteration']==lastIteration: pc['sameDampingWithinIteration'] = trial['dampingBeforeTrial'] == previousDamping
            elif lastIteration == 0 and trial['iteration']==1: pc['initialDamping'] = trial['dampingBeforeTrial'] == .001
            elif trial['iteration']==lastIteration+1: pc['nextDamping'] = trial['dampingBeforeTrial'] == (max(previousDamping*.5,1e-8) if lastAccepted else previousDamping*10)
            now=evaluate(coefficients);pc['currentObjective']=_close(now['objective'],trial['currentObjective']) and _close(now['objectiveComponents'],trial['currentObjectiveComponents'])
            H,g=independentGaussNewtonSystem(e,coefficients,camera,reg,fullReference=current,fullVertexIds=ids,contactOperator=operator,contactSpec=spec)
            lhs=H+trial['dampingBeforeTrial']*np.diag(np.maximum(np.diag(H),1e-8)); step=trial['coefficientStep'].ravel()
            denominator=max(float(np.linalg.norm(lhs,np.inf)*np.linalg.norm(step,np.inf)),float(np.linalg.norm(g,np.inf)),np.finfo(float).tiny)
            stepError=float(np.linalg.norm(lhs@step+g,np.inf)/denominator);pc['gaussNewtonStepEquation']=stepError<=1e-9
            safe,consistent,detail=safety(raw,trial['safety']);pc['safetyEvidence']=consistent and safe==trial['safetyAccepted']
            result=evaluate(proposed) if safe else None
            if not safe: decision='safetyRejected'
            elif not result['projectionValid']: decision='projectionRejected'
            elif any(v>initial['viewErrors'][k]+.5 for k,v in result['viewErrors'].items()):decision='viewDegradationRejected'
            elif not np.isfinite(result['objective']) or result['objective']>=now['objective']-1e-13:decision='objectiveRejected'
            else:decision='acceptedStep'
            pc['decision']=decision==trial['decision'];pc['explicitSkippedEvaluation']=trial['evaluation'] is None if not safe else trial['evaluation'] is not None
            if safe and trial['evaluation'] is not None:
                pc['evaluatedObjective']=all(_close(result[k],trial['evaluation'][k]) for k in ('objective','objectiveComponents','meanErrorPixels','viewErrors','projectionValid')) and result['preparedBindingsValidated']
            if decision=='acceptedStep':
                coefficients=proposed.copy()
                if result['objective']<bestObjective and result['meanErrorPixels']<initial['meanErrorPixels']-.001:
                    best=proposed.copy();bestObjective=result['objective'];bestIteration=trial['iteration']
            trials.append(dict(iteration=trial['iteration'],backtrack=trial['backtrack'],scale=trial['stepScale'],decision=decision,
                validationPassed=bool(all(pc.values())),checks=pc,normalEquationRelativeResidual=stepError,independentSafety=detail,evaluation=result,
                rawCoordinatesReconstructedFromSavedCoefficients=True,fullSceneCollisionEvaluated=False))
            lastIteration,lastBacktrack,lastAccepted,previousDamping=trial['iteration'],trial['backtrack'],decision=='acceptedStep',trial['dampingBeforeTrial']
        checks['unroundedBestSelection']=np.array_equal(best,e['unroundedBestCoefficients'])
        casts=[]; finalBeforeFallback=None; castSafe=False; castScale=None; finalCoeff=None
        for index,cast in enumerate(e['castTrials']):
            expectedScale=1. if index==0 else max(0.,1.-1e-4*2**(index-1)); cc=best*cast['scale']
            local=reconstructFieldTrial(e['referenceVertices'],e['basis'],e['axes'],limit,cc).astype(np.dtype(cast['outputDtype']))
            # Original solver explicitly copies inactive values after dtype cast.
            local[~e['activeVertexMask']]=e['referenceVertices'][~e['activeVertexMask']].astype(local.dtype)
            castSafe,consistent,detail=safety(local.astype(float),cast['safety'])
            pc=dict(scale=cast['scale']==expectedScale,coefficients=np.array_equal(cc,cast['coefficients']),safety=consistent and castSafe==cast['safetyAccepted'],
                noTrialAfterSafe=index==0 or not casts[-1]['safe'],outputDtype=cast['outputDtype']==e['outputDtype'])
            casts.append(dict(index=index,scale=cast['scale'],safe=castSafe,validationPassed=bool(all(pc.values())),checks=pc,independentSafety=detail))
            finalBeforeFallback=local;castScale=cast['scale'];finalCoeff=cc
        checks['castTracePresent']=bool(casts) and len(casts)<=17
        checks['castTraceTerminatedBySafetyOrBudget']=bool(casts) and (casts[-1]['safe'] or len(casts)==17)
        evaluated=evaluate(finalCoeff,finalBeforeFallback.astype(float))
        rejected=not castSafe or castScale==0 or (bestIteration>0 and (evaluated['meanErrorPixels']>=initial['meanErrorPixels']-.001 or any(v>initial['viewErrors'][k]+.5 for k,v in evaluated['viewErrors'].items())))
        output=e['referenceVertices'].astype(np.dtype(e['outputDtype'])) if rejected else finalBeforeFallback
        finalCoeff=np.zeros_like(finalCoeff) if rejected else finalCoeff
        evaluated=evaluate(finalCoeff,output.astype(float))
        checks['castRejectionDecision']=rejected==e['castRejected'];checks['finalCoefficients']=np.array_equal(finalCoeff,e['finalCoefficients'])
        checks['finalOutputFromFieldAndCast']=np.array_equal(output,e['outputVertices'])
        checks['finalEvaluation']=all(_close(evaluated[k],e['finalEvaluation'][k]) for k in ('objective','objectiveComponents','meanErrorPixels','viewErrors','projectionValid'))
        outer=[]
        for proposal in (row for row in rawEvidence['candidateProposals'] if row['round']==roundId):
            expected=current.copy();expected[ids]=current[ids]+proposal['scale']*(output-current[ids]);expected[fixedFull]=baseline[fixedFull]
            outer.append(dict(scale=proposal['scale'],coordinatesExactlyFromSolverField=np.array_equal(expected,proposal['vertices']),selected=proposal['selectedByFastRound']))
        checks['outerProposalsFromActualField']=all(row['coordinatesExactlyFromSolverField'] for row in outer)
        checks['allGnTrials']=all(row['validationPassed'] for row in trials);checks['allCastTrials']=all(row['validationPassed'] for row in casts)
        reports.append(dict(round=roundId,validationPassed=bool(all(checks.values())),checks=checks,fieldRepresentation=field,fieldConstruction=construction,
            trials=trials,castTrials=casts,outerFieldProposals=outer,finalEvaluation=evaluated))
        selected=[row for row in rawEvidence['candidateProposals'] if row['round']==roundId and row['selectedByFastRound']]
        if selected: current=selected[-1]['vertices'].copy()
    allChecks['allRounds']=all(row['validationPassed'] for row in reports)
    return dict(validationPassed=bool(all(allChecks.values())),checks=allChecks,rounds=reports,scope='Independent FEM/compact basis, IRLS equation, GN decision trace and cast; complete-scene collisions independently checked for every OUTER proposal.')


# Shared old audit primitives are pinned separately; no old audit or candidate is modified.
import argparse
import json
import shutil
import traceback
from validateResidualContactExperiment import (
    EYES, COARSE_EYES, coarseEyePreservationAudit, frozenAssemblyAudit, legacyResidualObservationAudit,
    historicalCollisionAudit, frozenContactGapAudit, contactSpecAudit, independentEyeSupport,
    radialWindingAudit, exportIdentityAudit, auditResidualEvidence, _scoreFieldsAgree,
)
from validateFeatureGeometryExperiment import (readJson, readArrays, readEvidence, sha256, fileIdentityAudit,
    loadOriginalAudit, downstreamGeometryAudit, initialGeometryAudit)
from validateProtectedFeatureExperiment import originalObservationMasks, originalObservationErrors
from faceAppearance.artifacts import writeJson
from faceAppearance.clientGeometrySafety import intersectionPairs
PLAN_SHA = 'afd8ba41054a7583852bdabb71d42c949268d884cfa25cc8cad88d80cf0381d1'
BRANCHES = ('baseline','legacyResidualReplay','boundaryFieldContact')
def auditCase(case, runRoot, outputRoot):
    from faceAppearance.headAssembly import AssemblyAssets
    from faceAppearance.legacyPreviewAsset import _load
    from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs, extractVisibleEyeContours
    from faceAppearance.clientLandmarks import loadClientLandmarks, ibugClientLandmarks
    import hashlib
    root = runRoot / case; planPath = runRoot / 'experimentPlan.json'; config = readJson(planPath)
    if sha256(planPath) != PLAN_SHA or case not in config['cases']: raise ValueError('Case or prospective plan identity changed')
    context = readJson(root / 'inputContext.json'); complete = readJson(root / 'report.json')
    originalAudit = loadOriginalAudit(); system = originalAudit.priorAudit.independentSystem(case); raw = system['a']
    parentRoot, oldRoot = Path(context['completedBaselineRoot']), Path(context['oldPipelineRoot'])
    parentReport = readJson(parentRoot / 'report.json'); parentFull = readArrays(parentRoot / 'fullGeometry.npz')
    parentStates = readArrays(parentRoot / 'states.npz'); initial = parentStates['initial']; base = parentFull['vertices']
    skinIds = raw['skinIds']; reference = context['referenceSelection']['referenceViewIndex']
    cameras = readJson(Path(context['entry']['reconstruction']) / 'cameras.json'); camera = cameras[reference]
    inputs = readJson(Path(context['entry']['reconstruction']) / 'inputs.json'); imageShape = tuple(inputs[reference]['workingSize'][::-1])
    pixelScale = float(np.asarray(inputs[reference]['inputToWorking'])[0, 0]); imageScale = min(imageShape) / 512.
    photo = readArrays(oldRoot / 'observations/observations.npz')['eyelidCurvesWorking']
    photoEyes = {name: {part: photo[i, j] for j, part in enumerate(('lower', 'upper'))} for i, name in enumerate(EYES)}
    prepared = readEvidence(root / 'preparation'); spec = readEvidence(root / 'contactSpec'); legacy = readEvidence(root / 'legacyObservationState')
    assets = AssemblyAssets(ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1')
    sourceAsset = _load(parentReport['asset']['archivePath'])
    scene = loadLegacyEyeBindingInputs(parentReport['asset']['archivePath'], skinIds); scene['vertices'] = base
    corners = np.concatenate([c['faceIndices'] for c in assets.components if c['name'] in ('male00SG.component001', 'male00SG.component002')])
    caseChecks = dict(planIdentity=context['planSha256'] == PLAN_SHA and complete['planSha256'] == PLAN_SHA,
        exactCompletedSkin=np.array_equal(base[skinIds], parentStates['final']),
        originalInitial=np.array_equal(initial, readArrays(ROOT / 'runs/shapePreservation20261024' / case / 'boundedWeight001/geometry.npz')['vertices']),
        exactScene=sameEvidence(scene, prepared['scene']), exactCamera=sameEvidence({k: camera[k] for k in ('intrinsics', 'rotation', 'translation')}, prepared['camera']),
        originalPhotoEyes=sameEvidence(photoEyes, prepared['photoEyes']), imageShape=tuple(prepared['imageShape']) == imageShape,
        originalPixelScale=prepared['originalPixelScale'] == pixelScale, policy=prepared['policy'] == config['residualPolicy'],
        cornerFaceIds=np.array_equal(prepared['cornerFaceIds'], corners),
        baselineVerticesHash=prepared['baselineVerticesSha256'] == hashlib.sha256(base.tobytes()).hexdigest())
    legacyRoot = Path(context['legacyReplayRoot'])
    previousPrepared = readEvidence(legacyRoot / 'preparation')
    previousSpec = readEvidence(legacyRoot / 'contactSpec')
    caseChecks['legacyPreparationExact'] = all(sameEvidence(prepared[k],previousPrepared[k]) for k in ('scene','camera','photoEyes','imageShape','cornerFaceIds','originalPixelScale','policy','support','coarseEyeConstraints','beforeCoarseAlignment','beforeContours','beforeApertureMetrics','baselineVerticesSha256'))
    caseChecks['legacyContactExact'] = sameEvidence(spec, previousSpec)
    previousOperator = sparse.load_npz(legacyRoot / 'contactOperator.npz'); currentOperator = sparse.load_npz(root / 'contactOperator.npz')
    caseChecks['legacyContactOperatorExact'] = previousOperator.shape == currentOperator.shape and (previousOperator != currentOperator).nnz == 0
    caseChecks['casePreparationHashes'] = fileIdentityAudit(complete['preparationHashes'])['accepted']
    caseChecks['parentOutputHashes'] = fileIdentityAudit({str(parentRoot / k): v for k, v in parentReport['outputHashes'].items()})['accepted']
    identities = fileIdentityAudit(context['sourceHashes']); caseChecks['inputHashes'] = identities['accepted']
    constraints = []
    for stage in parentReport['geometryRefinement']['stagedLandmarks']['stages']:
        if stage['name'] not in COARSE_EYES: continue
        e = stage['stageEvidence']; keep = np.asarray(stage['solver']['inputTargetGate']['groups'][0]['keptIndices'], int)
        constraints.append(dict(name=stage['name'], vertexIndices=skinIds[np.asarray(e['vertexIndices'], int)[keep]],
            barycentric=np.asarray(e['barycentricWeights'])[keep], **{k: np.asarray(e[k])[keep] for k in ('targetPixels', 'ibugIds', 'weights', 'confidence')}))
    caseChecks['exactOriginalSolverKeptCoarseConstraints'] = sameEvidence(constraints, prepared['coarseEyeConstraints'])
    binding = ibugClientLandmarks(loadClientLandmarks(raw['faces'], path=ROOT / 'assets/clientLandmarks/legacyMediapipe468/template_tri_lms_468_bicentroid.json'), groups=('eyes', 'mouth'))
    observations = readArrays(oldRoot / 'geometry/photoObservations.npz')
    masks = originalObservationMasks(initial, raw['faces'], binding, cameras, inputs, observations)
    caseChecks['legacyInputIdentity'] = sameEvidence(legacy['binding'], binding) and sameEvidence(legacy['observations'], observations) and sameEvidence(legacy['masks'], masks)
    caseChecks['legacyInitialScores'] = sameEvidence(legacy['initialErrors'], originalObservationErrors(initial, binding, cameras, observations, masks))
    caseChecks['legacyCompletedScores'] = sameEvidence(legacy['completedBaselineErrors'], originalObservationErrors(base[skinIds], binding, cameras, observations, masks))
    measurementCache = {}; pairCache = {}
    def measure(vertices):
        key = hashlib.sha256(vertices.tobytes()).hexdigest()
        if key not in measurementCache:
            measurementCache[key] = extractVisibleEyeContours(**{**scene, 'vertices': vertices}, camera=camera,
                photoEyes=photoEyes, imageShape=imageShape, apertureAccessoryMode='innerCornerHeadAccessories',
                apertureAccessoryFaceIds=corners, minimumEyeRasterWidth=256, maximumRoiSize=768)
        return measurementCache[key]
    def pairs(vertices):
        key = hashlib.sha256(vertices.tobytes()).hexdigest()
        if key not in pairCache: pairCache[key] = intersectionPairs(vertices, scene['faces'])
        return pairCache[key]
    baselineMeasurement = measure(base)
    caseChecks['completeBaselineRaster'] = _contourMasksAgree(baselineMeasurement, prepared['beforeContours'])
    support = independentEyeSupport(scene, camera, photoEyes, baselineMeasurement, skinIds[raw['staticIds']], config['residualPolicy']['geodesicBandEyeWidthFraction'])
    caseChecks['frozenSupportMask'] = np.array_equal(support['movableMask'], prepared['support']['movableMask'])
    caseChecks['frozenFixedIds'] = np.array_equal(support['fixedVertexIds'], prepared['support']['fixedVertexIds'])
    for name in EYES:
        for key, value in support['eyes'][name].items():
            caseChecks[name + 'Support' + key] = sameEvidence(value, prepared['support']['eyes'][name].get(key))
    orientation = []; groups = []; savedOrientation = readJson(root / 'orientation.json')
    for component in assets.components:
        if component['materialId'] != config['orientation']['semanticMaterialId']: continue
        ids = np.asarray(component['faceIndices'], int)
        template = radialWindingAudit(assets.arrays['vertices'], assets.arrays['faces'], ids, config['orientation'])
        assembled = radialWindingAudit(base, scene['faces'], ids, config['orientation'])
        provenance = 'legacyMaleDirectServiceV1 ' + component['name'] + '; template and fixed assembled eye radial winding verified by frozen .02/.95 rule; open sphere patch; no skin-gap sign inference'
        if support['movableMask'].any(): groups.append(dict(name=component['name'], sourceVertexIds=np.flatnonzero(support['movableMask']), targetFaceIds=ids, normalSign=1, orientationProvenance=provenance))
        saved = next(row for row in savedOrientation if row['name'] == component['name'])
        valid = template['accepted'] and assembled['accepted'] and saved['provenance'] == provenance and np.array_equal(ids, saved['faceIds'])
        for key in ('template', 'assembled'):
            value = template if key == 'template' else assembled
            valid &= saved[key]['accepted'] == value['accepted'] and saved[key]['normalSign'] == 1
            for field in ('radius', 'residualRatio', 'minimumRadialFaceCosine'):
                valid &= abs(saved[key][field] - value[field]) <= 1e-10
        orientation.append(dict(name=component['name'], template=template, assembled=assembled, validationPassed=bool(valid)))
    caseChecks['radialWindingProvenance'] = all(row['validationPassed'] for row in orientation)
    contactPolicy = {key: float(raw['unit']) * config['contactPolicy'][key + 'Mm'] for key in ('lengthScale', 'searchRadius', 'activationDistance', 'minimumGap')}
    contactPolicy['weight'] = config['contactPolicy']['weight']
    contactProof = contactSpecAudit(spec, base, scene['faces'], groups, contactPolicy, savedOperator=sparse.load_npz(root / 'contactOperator.npz'))
    operator = contactProof.pop('reconstructedOperator'); caseChecks['contactSpec'] = contactProof['validationPassed']
    def contactGap(vertices):
        return frozenContactGapAudit(base, vertices, spec['sourceVertexIds'], spec['targetVertexIds'], spec['targetBarycentric'], spec['normals'],
            spec['minimumGap'], lengthScale=spec['lengthScale'], tolerance=config['contactPolicy']['tolerance'])
    def legacyGate(vertices):
        return legacyResidualObservationAudit(initial, base[skinIds], vertices[skinIds], raw['faces'], binding, cameras, inputs, observations, reference, masks=masks)
    pairAnchors = {'completedCoarseBaseline': pairs(base)}
    for name, arm in (('originalPureDirect', 'pureDirect'), ('currentBounded', 'bounded')):
        anchor = readArrays(oldRoot / 'geometry' / arm / 'fullGeometry.npz')
        caseChecks[name + 'FullTopology'] = np.array_equal(anchor['faces'], scene['faces'])
        pairAnchors[name] = pairs(anchor['vertices'])
    initialGate = initialGeometryAudit(system, initial)
    caseChecks['inheritedInitialDecision'] = initialGate['accepted'] == parentReport['initialAccepted']
    branches = {}
    for branch in BRANCHES:
        directory = root / branch; report = readJson(directory / 'report.json')
        full = readArrays(directory / 'fullGeometry.npz'); states = readArrays(directory / 'states.npz'); evidence = readEvidence(directory / 'evidence')
        rawEvidence = readEvidence(directory / 'rawRefinement')
        final = full['vertices']; outputs = fileIdentityAudit({str(directory / k): v for k, v in report['outputHashes'].items()})
        checks = dict(caseAndBranch=report['caseId'] == case and report['branch'] == branch,
            allOutputHashes=outputs['accepted'], sameInputHashes=report['sourceHashes'] == context['sourceHashes'],
            referenceView=report['referenceViewIndex'] == reference, fullTopology=np.array_equal(full['faces'], scene['faces']),
            initialState=np.array_equal(states['initial'], initial), completedState=np.array_equal(states['completedCoarseBaseline'], base[skinIds]),
            baselineState=np.array_equal(states['baseline'], base[skinIds]), finalState=np.array_equal(states['final'], final[skinIds]),
            evidenceFinal=np.array_equal(evidence['vertices'], final), sameFrozenSupport=sameEvidence(evidence['support'], prepared['support']),
            noClientPromotion=report['finalCandidateAccepted'] is False and report['finalAdaptiveAssemblyValidated'] is False and report['textureRebaked'] is False)
        fixed = frozenAssemblyAudit(scene, {**scene, 'vertices': final}, support['movableMask'], staticFullIds=skinIds[raw['staticIds']])
        checks['fixedAssembly'] = fixed['accepted']
        mouthFullIds = skinIds[np.unique(binding['vertexIndices'][binding['ibugIds'] >= 48])]
        checks['realMouthExactlyFixed'] = np.array_equal(final[mouthFullIds], base[mouthFullIds])
        archivePath = Path(report['asset']['archivePath']); archive = _load(archivePath)
        export = exportIdentityAudit(sourceAsset, archive, readArrays(directory / 'exportIdentity.npz'), base, final, support['movableMask'], raw, report['asset'])
        checks['exportIdentity'] = export['validationPassed']; checks['archiveHash'] = sha256(archivePath) == report['asset']['sha256']
        downstream = downstreamGeometryAudit(system, initial, final[skinIds], camera)
        coarse = coarseEyePreservationAudit(base, final, constraints, camera, imageScale=imageScale)
        contacts = contactGap(final); legacyReport = legacyGate(final)
        collision = historicalCollisionAudit(pairAnchors['completedCoarseBaseline'], pairs(final), pairAnchors, scene['materialIds'])
        for name, suffix in (('originalPureDirect', 'OriginalPureDirect'), ('currentBounded', 'CurrentBounded'), ('completedCoarseBaseline', 'PairedBaseline')):
            checks[name + 'NewPairs'] = collision['anchors'][name]['newPairs'] == [tuple(v) for v in report['newFullPairsAgainst' + suffix]]
        checks['skinNewPairs'] = downstream['cumulativeDownstream']['collisions']['newPairs'] == [tuple(v) for v in report['newSkinPairsAgainstBranchInitial']]
        checks['downstreamDecision'] = downstream['cumulativeDownstream']['accepted'] == bool(report['cumulativeDownstreamGate']['accepted'] and not report['newSkinPairsAgainstBranchInitial'])
        checks['cavityDecision'] = downstream['originalCavity']['accepted'] == bool(report['finalCavityGuard']['accepted'] and report['finalCavityBenefit']['accepted'])
        checks['coarseDecision'] = coarse['accepted'] == report['pairedSemanticAlignment']['accepted']
        checks['contactDecision'] = contacts['accepted'] == report['contactSafety']['accepted']
        checks['legacyDecision'] = legacyReport['accepted'] == report['legacyObservationGate']['accepted']
        incremental = bool(fixed['accepted'] and coarse['accepted'] and contacts['accepted'] and collision['incrementalNoNewPairs'])
        absolute = bool(initialGate['accepted'] and incremental and downstream['accepted'] and legacyReport['accepted'] and collision['allHistoricalAnchorsAccepted'])
        checks['incrementalDecision'] = incremental == report['incrementalSafe']
        checks['absoluteDecision'] = absolute == report['absoluteEligible'] == report['geometrySafetyAccepted']
        effect = fullApertureRasterAudit(baselineMeasurement, measure(final), photoEyes, originalPixelScale=pixelScale)
        eyeProof = auditResidualEvidence('baseline' if branch == 'baseline' else 'residualWithContact', prepared, rawEvidence, base, rawEvidence['vertices'], scene, support, constraints, camera, imageScale,
            baselineMeasurement, photoEyes, pixelScale, measure, pairs, contactGap, legacyGate, spec, operator)
        checks['residualEvidence'] = eyeProof['validationPassed']
        checks['rawProducerDecision'] = eyeProof['residualCandidateAccepted'] == report['rawRefinementAccepted']
        effectRaw = fullApertureRasterAudit(baselineMeasurement, measure(rawEvidence['vertices']), photoEyes, originalPixelScale=pixelScale)
        savedEffect = readJson(directory / 'finalEffectGate.json')
        checks['finalEffectReportIdentity'] = sameEvidence(savedEffect, report['finalEffectGate'])
        checks['finalEffectDecision'] = effectRaw['accepted'] == savedEffect['accepted']
        checks['finalEffectRawMetrics'] = (not effectRaw['effectDefined'] or _scoreFieldsAgree(effectRaw['metrics']['baseline'], savedEffect['baseline']) and _scoreFieldsAgree(effectRaw['metrics']['candidate'], savedEffect['candidate']))
        fallback = branch == 'boundaryFieldContact' and not effectRaw['accepted']
        checks['fallbackPolicy'] = fallback == savedEffect['fallbackToExactBaseline']
        checks['finalOutputPolicy'] = np.array_equal(final, base if fallback else rawEvidence['vertices'])
        checks['effectOnlyFallbackLeavesRawAndSolver'] = all(np.array_equal(evidence[k],rawEvidence[k]) for k in ('rawCandidateVertices','solverVertices')) and sameEvidence(evidence['candidateProposals'],rawEvidence['candidateProposals'])
        producerAccepted = bool(eyeProof['residualCandidateAccepted'] and not fallback)
        checks['outputProducerDecision'] = producerAccepted == report['residualCandidateAccepted']
        changed = not np.array_equal(final,base)
        checks['changedDecision'] = changed == report['geometryChanged']
        fieldProof = auditSolverFields(branch,prepared,rawEvidence,camera,base,scene,support,constraints,imageScale,spec,operator,measure,config) if branch != 'baseline' else dict(validationPassed=True,checks={'noSolver':not rawEvidence['solverEvidence']},rounds=[])
        checks['solverFieldsAndTrials'] = fieldProof['validationPassed']
        checks['solverSidecarIdentity'] = sameEvidence(readEvidence(directory / 'solverEvidence'),rawEvidence['solverEvidence'])
        replay = None
        if branch == 'legacyResidualReplay':
            oldEvidence = readEvidence(Path(context['legacyReplayRoot']) / 'residualWithContact/evidence')
            replay = replayGeometryEvidenceAudit(oldEvidence,rawEvidence)
            replaySaved = readJson(directory / 'replayMatch.json')
            checks['legacyReplayExact'] = replay['validationPassed'] and replaySaved['accepted']
            checks['legacyReplayReportIdentity'] = sameEvidence(replaySaved,report['replayMatch'])
        classification = classifyBoundaryCandidate(producerAccepted=producerAccepted,geometryChanged=changed,
            incrementalSafe=incremental,absoluteEligible=absolute,finalBaselineEffect=effectRaw,evidenceValidated=bool(all(checks.values())))
        checks['improvementDecision'] = classification['improvementAccepted'] == report['improvementAccepted']
        checks['changingAbsoluteDecision'] = classification['changingAbsoluteEligible'] == report['changingAbsoluteEligible']
        branches[branch] = dict(validationPassed=bool(all(checks.values())), checks=checks,
            incrementalSafe=incremental, absoluteEligible=absolute, producerAccepted=producerAccepted,
            residualCandidateAccepted=producerAccepted, rawRefinementAccepted=eyeProof['residualCandidateAccepted'],
            requestedEffectAccepted=classification['requestedEffectAccepted'], improvementAccepted=classification['improvementAccepted'],
            changingAbsoluteEligible=classification['changingAbsoluteEligible'], evidenceValidated=bool(all(checks.values())),
            classification=classification, finalEffectGate=effectRaw, finalOutputEffect=effect,
            gnEvidence=fieldProof, replayEvidence=replay,
            unchangedRealMouth=np.array_equal(final[skinIds[np.unique(binding['vertexIndices'][binding['ibugIds'] >= 48])]], base[skinIds[np.unique(binding['vertexIndices'][binding['ibugIds'] >= 48])]]),
            fixedAssembly=fixed, exportIdentity=export, downstream=downstream, originalCoarseAlignment=coarse,
            contactEnvelope=contacts, legacyPhotoGate=legacyReport, fullCollisionAnchors=collision,
            fullApertureEffect=effect, residualEvidence=eyeProof, producerOutputs=outputs,
            reportSha256=sha256(directory / 'report.json'), archiveSha256=sha256(archivePath), clientReady=False)
    caseChecks['inputHashesAfterAudit'] = fileIdentityAudit(context['sourceHashes'])['accepted']
    result = dict(caseId=case, validationPassed=bool(all(caseChecks.values()) and all(v['validationPassed'] for v in branches.values())),
        checks=caseChecks, initialGate=initialGate, contactSpecification=contactProof, orientation=orientation,
        originalInputIdentities=identities, branches=branches, planSha256=PLAN_SHA, validatorSha256=sha256(__file__),
        candidateSolvesPerformed=False, clientReady=False,
        scope='Same completed coarse baseline and fixed attachments. Contact proxy and automatic photo curves are not anatomical ground truth; inherited failures stay failures.')
    outputRoot.mkdir(parents=True, exist_ok=True); writeJson(outputRoot / f'{case}.json', result)
    return result



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',nargs='+',required=True)
    parser.add_argument('--run-root',type=Path,default=ROOT/'runs/boundaryField20261024')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/boundaryField20261024/independentAudit')
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),ROOT/'tests/testBoundaryFieldValidation.py',ROOT/'tools/validateResidualContactExperiment.py',
        ROOT/'tools/validateFeatureGeometryExperiment.py',ROOT/'tools/validateProtectedFeatureExperiment.py']
    snapshots={}
    for path in files:
        target=args.output/'implementation'/sha256(path)/path.name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): shutil.copyfile(path,target)
        snapshots[str(path)]=dict(snapshotPath=str(target),sha256=sha256(path))
    writeJson(args.output/'auditImplementation.json',dict(planSha256=PLAN_SHA,validatorSha256=sha256(__file__),snapshots=snapshots,
        hashes={str(p):sha256(p) for p in files},candidateSolvesPerformed=False,
        scope='Independent coordinate/basis/IRLS/GN/cast/contact/mask/complete scene audit; existing numerical visibility and triangle intersection predicates reused.'))
    results=[]; failures=[]
    for case in args.case:
        try:
            result=auditCase(case,args.run_root,args.output);results.append(result)
            print(json.dumps(dict(caseId=case,validationPassed=result['validationPassed'],improvements=[k for k,v in result['branches'].items() if v['improvementAccepted']])),flush=True)
        except Exception as error:
            failure=dict(caseId=case,validationPassed=False,toolException=True,exceptionType=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            failures.append(failure);writeJson(args.output/(case+'.toolException.json'),failure);print(json.dumps(failure),flush=True)
    counts={branch:{key:sum(row['branches'][branch][key] for row in results) for key in
        ('validationPassed','incrementalSafe','absoluteEligible','rawRefinementAccepted','requestedEffectAccepted','improvementAccepted','changingAbsoluteEligible')} for branch in BRANCHES}
    summary=dict(validationPassed=not failures and all(r['validationPassed'] for r in results),caseCount=len(results),counts=counts,
        toolExceptionCount=len(failures),failures=failures,cases=[dict(caseId=r['caseId'],validationPassed=r['validationPassed'],
            report=str(args.output/(r['caseId']+'.json')),reportSha256=sha256(args.output/(r['caseId']+'.json'))) for r in results],
        validatorSha256=sha256(__file__),planSha256=PLAN_SHA,clientReady=False,expectedRejectionsAreToolFailures=False)
    writeJson(args.output/'validationSummary.json',summary);print(json.dumps(summary,indent=2),flush=True)
    if not summary['validationPassed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
