"""Bounded CPU experiment fitting the complete frontal visible eye aperture.

The original full assembly is the shape prior. No camera, topology, UV or eye
ball changes are made. Explicit inner-corner skin accessories may follow the
same smooth image-plane field as the eyelids. Every visible aperture fragment
is measured; rejected contour ordering does not invent a completed contour.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from time import perf_counter
import numpy as np

from .camera import projectPoints
from .clientGeometrySafety import broadPhase, intersectTriangles
from .frontLandmarkRefinement import refineFrontLandmarks, checkFrontGeometry
from .visibleEyeContours import extractVisibleEyeContours, evaluateApertureMask, _raster, _resample, _components, _sphere


def _constraints(vertices, faces, materialIds, camera, photoEyes, measurement,
                 movableIds, cornerFaceIds, uvs, uvFaces, materials, alphaTextures,
                 *, fixedVertexIds=None):
    """Bind actual visible boundary samples, including disconnected fragments."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    from scipy.spatial import cKDTree
    inverse = np.full(len(vertices), -1, np.int64)
    inverse[movableIds] = np.arange(len(movableIds))
    groups, stats = [], {}
    for name, eye in measurement['eyes'].items():
        if 'roiMasks' not in eye:
            stats[name] = {'status':'noVisibleRoi'}
            continue
        roi, masks = eye['roi'], eye['roiMasks']
        aperture = masks['eye']
        boundary = aperture & ~binary_erosion(aperture, structure=np.ones((3,3)))
        yy, xx = np.nonzero(boundary)
        # Spatially distributed samples retain secondary fragments; this is a
        # bounded sampling policy, not an interpolation over missing segments.
        if len(xx) > 128:
            ids = np.linspace(0, len(xx)-1, 128).astype(int)
            yy, xx = yy[ids], xx[ids]
        hits, bary, _, _ = _raster(vertices, faces, materialIds, camera, roi,
                                  uvs, uvFaces, materials, alphaTextures)
        distance, nearest = distance_transform_edt(~masks['eligibleBindableSkin'], return_indices=True)
        by, bx = nearest[:, yy, xx]
        bindDistance = distance[yy, xx]
        # Some isolated inner-corner fragment edges have no adjacent eyelid.
        # Their explicitly audited accessory surface itself supplies binding.
        accessory = np.isin(hits[yy, xx], cornerFaceIds) & (bindDistance > 2.6)
        by[accessory], bx[accessory] = yy[accessory], xx[accessory]
        faceIds = hits[by, bx]
        valid = ((bindDistance <= 2.6) | accessory) & (faceIds >= 0)
        valid &= (inverse[faces[np.maximum(faceIds, 0)]] >= 0).all(1)
        removedFixed = 0
        if fixedVertexIds is not None:
            fixedMask = np.zeros(len(vertices), bool)
            fixedMask[np.asarray(fixedVertexIds, np.int64)] = True
            fullyFixed = fixedMask[faces[np.maximum(faceIds, 0)]].all(1)
            removedFixed = int((valid & fullyFixed).sum())
            valid &= ~fullyFixed
        yy, xx, by, bx, faceIds, accessory = [a[valid] for a in (yy, xx, by, bx, faceIds, accessory)]
        if len(faceIds) < 12:
            stats[name] = {'status':'insufficientLocalBoundarySamples', 'kept':len(faceIds)}
            if fixedVertexIds is not None:
                stats[name]['fullyFixedBoundarySamplesRemoved'] = removedFixed
            continue
        weights = bary[by, bx]
        points = (vertices[faces[faceIds]]*weights[...,None]).sum(1)
        projected, _ = projectPoints(points, np.asarray(camera['intrinsics']),
                                     np.asarray(camera['rotation']), np.asarray(camera['translation']))
        boundaryPixels = (np.c_[xx,yy]+np.c_[bx,by])*.5/roi['scale']+roi['origin']
        boundaryPixels[accessory] = np.c_[xx,yy][accessory]/roi['scale']+roi['origin']
        target = np.r_[_resample(photoEyes[name]['upper'],256),
                       _resample(photoEyes[name]['lower'],256)]
        distance, nearestTarget = cKDTree(target).query(boundaryPixels)
        targetPixels = target[nearestTarget] - (boundaryPixels-projected)
        # The observed edge receives a normal-like closest-boundary target.
        # Robust fitting below limits large unreliable attraction independently.
        groups.append({'name':name+'_completeVisibleBoundary',
                       'vertexIndices':inverse[faces[faceIds]], 'barycentric':weights,
                       'targetPixels':targetPixels, 'weights':np.ones(len(faceIds))})
        stats[name] = {'status':'bound', 'kept':len(faceIds),
                       'accessoryBoundarySamples':int(accessory.sum()),
                       'meanInitialWorkingPixelDistance':float(distance.mean()),
                       'visibleComponentCount':eye['diagnostics'].get('visibleContourCount')}
        if fixedVertexIds is not None:
            stats[name]['fullyFixedBoundarySamplesRemoved'] = removedFixed
    return groups, stats


def _newIntersections(reference, candidate, faces, activeVertexMask, epsilon,
                      materialIds=None, materials=None):
    """Check every changed-face pair against the whole assembly.

    Unchanged pairs cannot be new. Baseline intersections are queried for the
    same candidate pair, avoiding a second whole-scene narrow phase. Shared
    topology neighbors use the established clientGeometrySafety exclusion.
    """
    pairs, _ = broadPhase(candidate, faces, epsilon)
    changedFaces = activeVertexMask[faces].any(1)
    pairs = pairs[changedFaces[pairs].any(1)]
    # Eye balls already overlap eyelid skin to define the visible eye opening;
    # a new triangle pair along that same contact is not new self-penetration.
    # Use separate sphere-depth gates below. Alpha cards intentionally cross
    # the skin and do not represent a solid collision volume.
    excludedEye, excludedAlpha = 0, 0
    if materialIds is not None:
        eyeContact = (materialIds[pairs] == 2).any(1)
        alphaIds = [i for i,m in enumerate(materials or []) if m['alphaMode']=='MASK']
        alphaContact = np.isin(materialIds[pairs],alphaIds).any(1)
        excludedEye,excludedAlpha = int(eyeContact.sum()),int(alphaContact.sum())
        pairs = pairs[~eyeContact & ~alphaContact]
    triangles, original = candidate[faces], reference[faces]
    added = []
    checked = 0
    kinds = ('nonCoplanarSegment','coplanarAreaOverlap')
    for first, second in pairs:
        checked += 1
        hit = intersectTriangles(triangles[first], triangles[second], epsilon)
        if not hit or hit['kind'] not in kinds:
            continue
        old = intersectTriangles(original[first], original[second], epsilon)
        if not old or old['kind'] not in kinds:
            added.append([int(first),int(second)])
    return {'newPairCount':len(added), 'newPairs':added, 'narrowPhasePairCount':checked,
            'eyeContactPairsCheckedSeparately':excludedEye,'excludedAlphaCardPairs':excludedAlpha,
            'scope':'changed opaque non-eye faces against assembly, excluding shared-index neighbors; eye depth separately'}


def _eyeContactSafety(reference,candidate,faces,materialIds,movable,epsilon):
    """Bound existing lid/eye interpenetration without locking triangle IDs."""
    reports=[]
    for component in _components(faces,materialIds,(2,)):
        center,radius,residual=_sphere(reference[np.unique(faces[component])])
        before=np.linalg.norm(reference[movable]-center,axis=1)-radius
        after=np.linalg.norm(candidate[movable]-center,axis=1)-radius
        initialDepth=np.maximum(-before,0.)
        finalDepth=np.maximum(-after,0.)
        baselineContactBand=before<radius*.30
        safe=(finalDepth.max(initial=0)<=initialDepth.max(initial=0)+epsilon
              and np.square(finalDepth).sum()<=np.square(initialDepth).sum()*1.01+epsilon**2
              and not np.any((finalDepth>epsilon)&~baselineContactBand))
        reports.append({'accepted':bool(safe),'sphereRadius':radius,'sphereResidual':residual,
            'baselineMaximumPenetration':float(initialDepth.max(initial=0)),
            'candidateMaximumPenetration':float(finalDepth.max(initial=0)),
            'baselineSquaredPenetration':float(np.square(initialDepth).sum()),
            'candidateSquaredPenetration':float(np.square(finalDepth).sum()),
            'newContactOutsideBaselineBand':int(((finalDepth>epsilon)&~baselineContactBand).sum())})
    return {'accepted':all(r['accepted'] for r in reports),'eyes':reports,
            'policy':'fixed sphere maximum penetration cannot increase; summed squared penetration <= baseline*1.01; no remote new contacts',
            'scope':'spherical eye component regression gate, not continuous collision certification'}


def refineFastFrontEyes(vertices, faces, materialIds, camera, photoEyes, *,
                       skinVertexIds, cornerFaceIds, fixedVertexIds=(), imageShape,
                       uvs=None, uvFaces=None, materials=None, alphaTextures=None,
                       originalPixelScale=1., rounds=2, iterations=20, candidateGate=None,
                       additionalResidual=None, additionalSafety=None,
                       omitFullyFixedBoundaryConstraints=False, deformationMode='compact',
                       smoothingRadiusFraction=.1, captureSolverEvidence=False,
                       accessoryFollowingSpec=None):
    """Fit full frontal apertures with <= 2 rebindings and <= 40 solver steps.

    All IDs refer to the full assembly. ``cornerFaceIds`` must be the audited
    two non-NICP material-zero inner-corner components. ``fixedVertexIds``
    should include the stable pipeline's protected mouth, jaw and static skin.
    This is a local experiment; runtime includes measurement, fitting and
    full-assembly new-intersection gates, but excludes neural photo detection.
    An optional ``candidateGate(vertices, context)`` adds a caller's guard to
    every backtracking proposal. It receives a read-only copy, round/scale and
    the existing geometry check, and returns a dict with a boolean ``accepted``.
    It cannot waive any existing gate. Omission retains the original behavior.
    Optional additional residual/safety callbacks receive full-assembly vertex
    arrays during the inner optimization; their Jacobian uses full XYZ order.
    They are mapped to the original local solver IDs without changing the
    contact reference. Fixed-only boundary observations can optionally be
    omitted from fitting; their visible pixels remain in every aperture score.
    Optional ``accessoryFollowingSpec`` assembles whitelisted accessories from
    total skin displacement relative to the completed input. It adds no solver
    variables or boundary samples. Contact derivatives include this mapping;
    every full-scene gate and raster sees the assembled candidate. Solver-fixed
    accessory rows remain fixed in the parameterization, while the physical
    attachments can move as deterministic dependents of the skin.
    """
    started = perf_counter()
    original = np.asarray(vertices, float).copy()
    faces = np.asarray(faces, np.int64)
    materialIds = np.asarray(materialIds, np.int64)
    skin = np.asarray(skinVertexIds, np.int64)
    cornerFaces = np.asarray(cornerFaceIds, np.int64)
    fixed = np.asarray(fixedVertexIds, np.int64)
    if rounds not in (1,2) or not 1 <= iterations <= 20:
        raise ValueError('Bounded fast fitting requires 1-2 rounds and <=20 iterations per round')
    if candidateGate is not None and not callable(candidateGate):
        raise ValueError('candidateGate must be callable or None')
    if any(value is not None and not callable(value) for value in (additionalResidual, additionalSafety)):
        raise ValueError('Additional optimization callbacks must be callable or None')
    if type(omitFullyFixedBoundaryConstraints) is not bool:
        raise ValueError('omitFullyFixedBoundaryConstraints must be boolean')
    if deformationMode not in ('compact', 'fixedBoundaryBiharmonic') or type(captureSolverEvidence) is not bool:
        raise ValueError('Invalid deformation mode or solver evidence flag')
    if (not isinstance(smoothingRadiusFraction, (int, float, np.integer, np.floating))
            or isinstance(smoothingRadiusFraction, (bool, np.bool_))
            or not np.isfinite(smoothingRadiusFraction) or not 0 < smoothingRadiusFraction <= 1):
        raise ValueError('smoothingRadiusFraction must be a real scalar in (0, 1]')
    if fixed.ndim != 1 or (len(fixed) and (fixed.min()<0 or fixed.max()>=len(original))):
        raise ValueError('Invalid full-assembly fixed vertex IDs')
    following = deepcopy(accessoryFollowingSpec) if accessoryFollowingSpec is not None else None
    if following is not None:
        from .eyeAccessoryFollowing import (
            applyEyeAccessoryFollowing, eyeAccessoryDisplacementOperator,
            validateEyeAccessoryFollowingSpec,
        )
        from scipy import sparse
        validateEyeAccessoryFollowingSpec(following, baselineVertices=original,
            faces=faces, skinVertexIds=skin, originalFixedVertexIds=fixed)
    corners = np.unique(faces[cornerFaces])
    movable = np.unique(np.r_[skin,corners])
    inverse = np.full(len(original),-1,np.int64)
    inverse[movable] = np.arange(len(movable))
    fitFaceMask = (inverse[faces]>=0).all(1)
    fitFaces = inverse[faces[fitFaceMask]]
    localFixed = inverse[fixed]
    localFixed = localFixed[localFixed>=0]
    unchanged = np.ones(len(original), bool)
    unchanged[movable] = False
    unchanged[fixed] = True
    physicalFixed = fixed
    physicalUnchanged = unchanged
    if following is not None:
        # These are only safety/assembly domains. Keep localFixed and the
        # original binding mask unchanged, so followers add no degrees of freedom.
        physicalFixed = np.asarray(following['effectiveFixedVertexIds'], np.int64)
        physicalUnchanged = unchanged.copy()
        physicalUnchanged[np.asarray(following['followerVertexIds'], np.int64)] = False
        physicalLocalFixed = inverse[physicalFixed]
        physicalLocalFixed = physicalLocalFixed[physicalLocalFixed >= 0]
        fullFromLocal = sparse.kron(eyeAccessoryDisplacementOperator(following)[:, movable],
            sparse.eye(3, format='csr'), format='csr')
    else:
        physicalLocalFixed = localFixed
    extent = float(np.ptp(original[skin],axis=0).max())
    limit = extent*.03
    epsilon = max(float(np.linalg.norm(np.ptp(original,axis=0)))*1e-9,1e-12)
    def measure(value):
        return extractVisibleEyeContours(value,faces,materialIds,skin,camera,photoEyes,
            imageShape=imageShape,uvs=uvs,uvFaces=uvFaces,materials=materials,alphaTextures=alphaTextures,
            apertureAccessoryMode='innerCornerHeadAccessories',apertureAccessoryFaceIds=cornerFaces,
            minimumEyeRasterWidth=256,maximumRoiSize=768)
    tick = perf_counter()
    baseline = measure(original)
    baselineScore = evaluateApertureMask(baseline,photoEyes,originalPixelScale=originalPixelScale)
    measureSeconds = perf_counter()-tick
    current, currentMeasure, currentScore = original.copy(), baseline, baselineScore
    stages = []
    solverEvidence = []
    for roundIndex in range(rounds):
        stage = {'round':roundIndex+1, 'attempts':[]}
        stages.append(stage)
        tick = perf_counter()
        bindingOptions = {'fixedVertexIds':fixed} if omitFullyFixedBoundaryConstraints else {}
        constraints, bindingStats = _constraints(current,faces,materialIds,camera,photoEyes,currentMeasure,
            movable,cornerFaces,uvs,uvFaces,materials,alphaTextures,**bindingOptions)
        stage['binding'] = bindingStats
        stage['bindingSeconds'] = perf_counter()-tick
        if not constraints:
            stage['status'] = 'noUsableBoundaryConstraints'
            break
        tick = perf_counter()
        solverOptions = {}
        def fullSnapshot(local):
            full = current.copy()
            full[movable] = local
            if following is not None:
                full = applyEyeAccessoryFollowing(following, full)
            full.setflags(write=False)
            return full
        if additionalResidual is not None:
            from scipy import sparse
            columns = (movable[:,None]*3+np.arange(3)).ravel()
            def localResidual(local, withJacobian):
                evaluated = additionalResidual(fullSnapshot(local), withJacobian)
                if not isinstance(evaluated,dict) or 'residual' not in evaluated:
                    raise ValueError('Additional residual must return a residual mapping')
                result = dict(evaluated)
                residual = np.asarray(result['residual'],float)
                if residual.ndim!=1 or not np.isfinite(residual).all():
                    raise ValueError('Additional residual must be a finite vector')
                result['residual'] = residual.copy()
                if withJacobian:
                    matrix = result.get('jacobian')
                    if not sparse.issparse(matrix) or matrix.shape!=(len(residual),3*len(original)):
                        raise ValueError('Additional residual Jacobian must address full XYZ coordinates')
                    result['jacobian'] = (matrix.tocsr()[:,columns].copy() if following is None
                        else (matrix.tocsr() @ fullFromLocal).tocsr())
                return result
            solverOptions['additionalResidual'] = localResidual
        if additionalSafety is not None:
            def localSafety(local):
                snapshot = fullSnapshot(local)
                followingEvidence = None
                if following is not None:
                    # Record the actual array passed to the callback, not a
                    # later reconstruction. This is evidence only, never a
                    # replacement for or extension of the active safety gate.
                    followingEvidence = dict(specSha256=following['specSha256'],
                        fullVerticesSha256=hashlib.sha256(snapshot.tobytes()).hexdigest(),
                        changedFollowerVertexCount=int(np.any(snapshot[following['followerVertexIds']]
                            != original[following['followerVertexIds']], axis=1).sum()),
                        effectiveFixedExact=bool(np.array_equal(snapshot[physicalFixed], original[physicalFixed])))
                result = additionalSafety(snapshot)
                if not isinstance(result,dict) or type(result.get('accepted')) not in (bool,np.bool_):
                    raise ValueError('Additional safety must return an explicit boolean accepted')
                if followingEvidence is not None:
                    result = dict(result, accessoryFollowingSnapshot=followingEvidence)
                return result
            solverOptions['additionalSafety'] = localSafety
        fitted = refineFrontLandmarks(current[movable],fitFaces,camera,constraints,
            fixedVertexIds=localFixed,supportRadius=extent*.12,controlCount=32,
            maxDisplacement=limit,minConfidence=0.,maxTargetDistancePixels=40.,
            iterations=iterations,maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3,
            curveWeight=0.,deformationMode=deformationMode,smoothingRadiusFraction=smoothingRadiusFraction,
            captureSolverEvidence=captureSolverEvidence,**solverOptions)
        if captureSolverEvidence:
            solverEvidence.append(dict(round=roundIndex+1, fullVertexIds=movable.copy(),
                fullFaceIds=np.flatnonzero(fitFaceMask), fullReferenceVertices=current.copy(),
                fullFixedVertexIds=np.flatnonzero(unchanged), evidence=fitted['solverEvidence']))
            if following is not None:
                solverEvidence[-1].update(accessoryFollowingSpecSha256=following['specSha256'],
                    physicalFixedVertexIds=physicalFixed.copy(),
                    fullAssemblyFrom='global completed baseline plus mapped total skin displacement',
                    followerVertexIds=np.asarray(following['followerVertexIds']).copy())
        stage['solverSeconds'] = perf_counter()-tick
        stage['solver'] = fitted['diagnostics']
        accepted = False
        # At most three evaluations keeps fallback bounded. All candidates are
        # checked against the original input, including after a second binding.
        for scale in (1.,.5,.25):
            proposal = current.copy()
            proposal[movable] = current[movable]+scale*(fitted['vertices']-current[movable])
            proposal[unchanged] = original[unchanged]
            if following is not None:
                proposal = applyEyeAccessoryFollowing(following, proposal)
            report = {'scale':scale}
            stage['attempts'].append(report)
            check = checkFrontGeometry(original[movable],proposal[movable],fitFaces,camera=camera,
                fixedVertexIds=physicalLocalFixed,maxDisplacement=limit,maxRelativeEdgeChange=.25,
                minAreaRatio=.4,minNormalCosine=.3)
            report['geometry'] = check
            extra = None
            if candidateGate is not None:
                snapshot = proposal.copy()
                snapshot.setflags(write=False)
                extra = candidateGate(snapshot, {'round':roundIndex+1,'scale':scale,'geometry':deepcopy(check)})
                if not isinstance(extra,dict) or type(extra.get('accepted')) not in (bool,np.bool_):
                    raise ValueError('candidateGate must return a dict with boolean accepted')
                report['externalCandidateGate'] = extra
            if not check['accepted']:
                continue
            if extra is not None and not extra['accepted']:
                continue
            tick = perf_counter()
            measured = measure(proposal)
            score = evaluateApertureMask(measured,photoEyes,originalPixelScale=originalPixelScale)
            report['measurementSeconds'] = perf_counter()-tick
            report['apertureMetrics'] = score
            tolerance = .15/originalPixelScale
            improves = score['meanSymmetricBoundaryDistanceOriginalPixels'] < currentScore['meanSymmetricBoundaryDistanceOriginalPixels']-.03/originalPixelScale
            for name in photoEyes:
                old, new = currentScore['eyes'][name], score['eyes'][name]
                improves &= new['symmetricBoundaryMeanOriginalPixels'] <= old['symmetricBoundaryMeanOriginalPixels']+tolerance
                improves &= new['symmetricBoundaryP95OriginalPixels'] <= old['symmetricBoundaryP95OriginalPixels']+2*tolerance
                improves &= new['intersectionOverUnion'] >= old['intersectionOverUnion']-.005
                improves &= new['visibleComponentCount'] <= old['visibleComponentCount']
                improves &= new['secondaryComponentAreaFraction'] <= old['secondaryComponentAreaFraction']+.005
            report['apertureGate'] = bool(improves)
            if not improves:
                continue
            eyeContact=_eyeContactSafety(original,proposal,faces,materialIds,movable,epsilon)
            report['eyeContact']=eyeContact
            if not eyeContact['accepted']:
                continue
            tick = perf_counter()
            collision = _newIntersections(original,proposal,faces,np.any(proposal!=original,axis=1),epsilon,
                                           materialIds,materials)
            report['intersectionSeconds'] = perf_counter()-tick
            report['intersection'] = collision
            if collision['newPairCount']:
                continue
            current,currentMeasure,currentScore = proposal,measured,score
            accepted = True
            stage['status'] = 'accepted'
            stage['selectedScale'] = scale
            break
        if not accepted:
            stage['status'] = 'noImprovingSafeCandidate'
            break
    changed = np.any(current!=original,axis=1)
    # Texture warping requires ordered upper/lower curves. A fragmented full
    # aperture cannot provide them; retain its complete-mask rejection and use
    # a separately measured eye-material boundary only for texture skin bands.
    textureMeasure=extractVisibleEyeContours(current,faces,materialIds,skin,camera,photoEyes,
        imageShape=imageShape,uvs=uvs,uvFaces=uvFaces,materials=materials,alphaTextures=alphaTextures,
        minimumEyeRasterWidth=256,maximumRoiSize=768)
    textureScope='eye-material boundary proxy only; complete aperture status remains separately reported'
    output = {'vertices':current,'beforeContours':baseline,'afterContours':currentMeasure,
        'textureContours':textureMeasure,'afterTextureContours':textureMeasure,
        'diagnostics':{'status':'acceptedLocalExperiment' if changed.any() else 'noImprovingSafeCandidate',
            'method':'boundedFullApertureDistanceFieldFrontGeometry',
            'seconds':perf_counter()-started,'initialMeasurementSeconds':measureSeconds,
            'before':baselineScore,'after':currentScore,'stages':stages,
            'changedVertexCount':int(changed.sum()),'changedCornerVertexCount':int(changed[corners].sum()),
            'maximumReferenceDepthChange':float(np.abs((current-original)@np.asarray(camera['rotation'])[2]).max()),
            'fixedVerticesExactlyUnchanged':bool(np.array_equal(current[physicalUnchanged],original[physicalUnchanged])),
            'newOpaqueNonEyeIntersectionPairs':0,'cameraRefitted':False,'textureRebaked':False,
            'eyeContactPolicy':'fixed eye spheres; maximum and total squared penetration bounded relative to original input',
            'photoTargets':'cached automatic dense eye curves, not manual ground truth',
            'rasterEyeWidth':256,'maximumRoiSize':768,'roundsRequested':rounds,
            'textureContourScope':textureScope,
            'maximumSolverIterations':rounds*iterations,'prior':'original full input assembly'}}
    if captureSolverEvidence:
        output['solverEvidence'] = solverEvidence
    if following is not None:
        output['diagnostics'].update(accessoryFollowingEnabled=True,
            accessoryFollowingSpecSha256=following['specSha256'],
            changedFollowerVertexCount=int(changed[following['followerVertexIds']].sum()),
            originalSolverFixedDomainPreserved=True, accessoryDegreesOfFreedom=0,
            effectiveFixedVertexIds=physicalFixed.copy(),
            fullAssemblyRelativeToGlobalCompletedBaseline=True)
    return output
