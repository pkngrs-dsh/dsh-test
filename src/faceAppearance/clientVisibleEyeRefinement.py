"""Opt-in complete-eye refinement with a frozen local skin support domain.

The existing fast frontal aperture solver supplies proposals. This adapter
protects distant skin and all non-canthus accessories, and applies the original
strict full-assembly intersection rule without excluding eyes or alpha cards.
It performs no observation inference, asset assembly, texture editing or IO.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy import sparse
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.sparse.csgraph import dijkstra

from .clientGeometrySafety import intersectionPairs
from .eyelidTextureSampling import _curves
from .fastFrontEyeGeometry import refineFastFrontEyes
from .frontLandmarkRefinement import checkFrontGeometry
from .geometryRefinement import _meshEdges, _validateCameras, _validateMesh
from .visibleEyeContours import extractVisibleEyeContours


DEFAULT_POLICY = {
    'geodesicBandEyeWidthFraction': .30,
    'rounds': 2,
    'iterationsPerRound': 20,
}


def _policy(value):
    result = DEFAULT_POLICY.copy()
    if value is not None:
        if not isinstance(value, dict) or set(value)-set(result):
            raise ValueError('Unknown visible-eye refinement policy entries')
        result.update(value)
    fraction = result['geodesicBandEyeWidthFraction']
    if (isinstance(fraction, (bool, np.bool_)) or not isinstance(fraction, (int, float, np.number))
            or not np.isfinite(fraction) or not 0 < fraction <= .5):
        raise ValueError('Eye support fraction must be in (0, .5]')
    if type(result['rounds']) is not int or result['rounds'] not in (1, 2):
        raise ValueError('Visible-eye refinement supports one or two rounds')
    if type(result['iterationsPerRound']) is not int or not 1 <= result['iterationsPerRound'] <= 20:
        raise ValueError('Visible-eye iterations must be an integer from 1 to 20')
    result['geodesicBandEyeWidthFraction'] = float(fraction)
    return result


def _indices(values, size, name, *, nonempty=False):
    ids = np.asarray(values)
    if ids.size == 0 and not nonempty:
        return np.empty(0, np.int64)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids)
            or ids.min() < 0 or ids.max() >= size or len(np.unique(ids)) != len(ids)):
        raise ValueError(f'{name} must contain distinct in-range integer indices')
    return ids.astype(np.int64, copy=True)


def _scene(scene, cornerFaceIds):
    vertices, faces = _validateMesh(scene['vertices'], scene['faces'])
    materials = deepcopy(scene['materials'])
    materialIds = np.asarray(scene['materialIds'])
    skin = _indices(scene['nicpToFullIds'], len(vertices), 'nicpToFullIds', nonempty=True)
    corners = _indices(cornerFaceIds, len(faces), 'cornerFaceIds', nonempty=True)
    uv, uvFaces = np.asarray(scene['uvs']), np.asarray(scene['uvFaces'])
    if (materialIds.shape != (len(faces),) or materialIds.dtype.kind not in 'iu'
            or materialIds.min() < 0 or materialIds.max() >= len(materials)
            or uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all()
            or uvFaces.shape != faces.shape or uvFaces.dtype.kind not in 'iu'
            or uvFaces.min() < 0 or uvFaces.max() >= len(uv)):
        raise ValueError('Expected original full material and per-corner UV identities')
    member = np.zeros(len(vertices), bool)
    member[skin] = True
    if (not np.all(materialIds[corners] == 0) or member[faces[corners]].any()
            or not np.any(materialIds == 2)):
        raise ValueError('Canthi must be explicit non-NICP head faces; eye material is required')
    skinFaces = member[faces].all(1)
    if not skinFaces.any() or not np.all(materialIds[skinFaces] == 0):
        raise ValueError('The skin mapping must identify original head-material faces')
    return dict(vertices=vertices.copy(), faces=faces.copy(), materialIds=materialIds.copy(),
        nicpToFullIds=skin, uvs=uv.copy(), uvFaces=uvFaces.copy(), materials=materials,
        alphaTextures={key:np.asarray(value).copy() for key,value in scene.get('alphaTextures', {}).items()}), corners


def buildClientVisibleEyeSupport(scene, camera, photoEyes, measurement, *, cornerFaceIds,
                                 fixedVertexIds=(), protectedVertexIds=(), policy=None):
    """Freeze a geodesic eyelid band from actual baseline-visible skin.

    All IDs address the original full mesh. Seeds are the existing skin faces
    adjacent to every visible aperture fragment, never nearest 3D points from
    another surface. Geodesics traverse skin topology only. Physical eye width
    is the frozen photo canthus chord backprojected at local seed median depth.
    Explicit canthus vertices remain eligible; caller protections always win.
    The support must be computed once before any proposal is fitted.
    """
    effective = _policy(policy)
    scene, corners = _scene(scene,cornerFaceIds)
    vertices, faces = scene['vertices'], scene['faces']
    skin = _indices(scene['nicpToFullIds'], len(vertices), 'nicpToFullIds', nonempty=True)
    fixed = _indices(fixedVertexIds, len(vertices), 'fixedVertexIds')
    protected = _indices(protectedVertexIds, len(vertices), 'protectedVertexIds')
    if camera.get('valid', True) is not True:
        raise ValueError('Cannot refine with a rejected photo camera')
    camera = _validateCameras([camera])[0]
    member = np.zeros(len(vertices), bool); member[skin] = True
    skinFaceMask = member[faces].all(1)
    edges = _meshEdges(faces[skinFaceMask])
    lengths = np.linalg.norm(vertices[edges[:,1]]-vertices[edges[:,0]], axis=1)
    if not len(edges) or (lengths <= 0).any():
        raise ValueError('Skin topology needs positive-length edges')
    graph = sparse.csr_matrix((np.r_[lengths,lengths],
        (np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(vertices),len(vertices)))
    movable = np.zeros(len(vertices), bool)
    reports = {}
    depth = vertices@camera['rotation'][2]+camera['translation'][2]
    for name, photo in photoEyes.items():
        eye = measurement['eyes'].get(name, {})
        masks = eye.get('roiMasks')
        if masks is None:
            reports[name] = {'status':'noVisibleRoi', 'seedVertexIds':np.empty(0,np.int64)}
            continue
        aperture = np.asarray(masks['eye'], bool)
        bindable = np.asarray(masks['eligibleBindableSkin'], bool)
        hitFaces = np.asarray(masks['faceIds'])
        if (aperture.ndim != 2 or aperture.shape != bindable.shape or hitFaces.shape != aperture.shape
                or hitFaces.dtype.kind not in 'iu' or not bindable.any()):
            reports[name] = {'status':'noBindableVisibleSkin', 'seedVertexIds':np.empty(0,np.int64)}
            continue
        boundary = aperture & ~binary_erosion(aperture, structure=np.ones((3,3)), border_value=0)
        yy, xx = np.nonzero(boundary)
        distances, nearest = distance_transform_edt(~bindable, return_indices=True)
        by, bx = nearest[:,yy,xx]
        adjacent = distances[yy,xx] <= 2.6  # Existing fast contour-binding support rule.
        faceIds = hitFaces[by[adjacent],bx[adjacent]]
        valid = (faceIds >= 0) & (faceIds < len(faces))
        faceIds = faceIds[valid]
        faceIds = faceIds[skinFaceMask[faceIds]]
        seeds = np.unique(faces[faceIds])
        if len(seeds) < 3 or not np.all(depth[seeds] > 0):
            reports[name] = {'status':'insufficientVisibleSkinSeeds', 'seedVertexIds':seeds}
            continue
        upper, lower = np.asarray(photo['upper']), np.asarray(photo['lower'])
        canthi = np.asarray([(upper[0]+lower[0])/2, (upper[-1]+lower[-1])/2])
        rays = np.linalg.solve(camera['intrinsics'], np.c_[canthi,np.ones(2)].T).T
        rays /= rays[:,2,None]
        localDepth = float(np.median(depth[seeds]))
        eyeWidth = float(np.linalg.norm(rays[1]-rays[0])*localDepth)
        if not np.isfinite(eyeWidth) or eyeWidth <= 0:
            raise ValueError('Cannot infer a finite positive local eye width')
        radius = eyeWidth*effective['geodesicBandEyeWidthFraction']
        distance = dijkstra(graph, directed=False, indices=seeds, min_only=True, limit=radius)
        selected = member & (distance <= radius)
        movable |= selected
        reports[name] = dict(status='supported', seedVertexIds=seeds, skinFaceIds=np.unique(faceIds),
            distance=distance, movableSkinVertexIds=np.flatnonzero(selected), localDepth=localDepth,
            eyeWidth=eyeWidth, geodesicRadius=radius,
            source='all baseline visible aperture fragments adjacent to NICP skin')
    supported = len(reports) == 2 and all(row['status']=='supported' for row in reports.values())
    if supported:
        movable[np.unique(faces[corners])] = True
    else:
        movable[:] = False
    movable[np.r_[fixed,protected]] = False
    return dict(movableMask=movable, fixedMask=~movable, fixedVertexIds=np.flatnonzero(~movable),
        requestedFixedVertexIds=fixed, protectedVertexIds=protected,
        cornerVertexIds=np.unique(faces[corners]), eyes=reports,
        diagnostics=dict(status='supported' if supported else 'insufficientVisibleSupport',
            movableVertexCount=int(movable.sum()), fixedVertexCount=int((~movable).sum()),
            policy=effective, rule='baseline frozen skin-topology geodesic band; explicit canthi; hard fixed/protected override',
            candidateDependentSupport=False))


def _strictCandidateChecks(original, candidate, scene, camera, support, baselinePairs):
    finite = candidate.shape == original.shape and np.isfinite(candidate).all()
    if not finite:
        return {'accepted':False, 'reasons':['nonFiniteOrWrongShapeCandidate']}
    extent = float(np.ptp(original[scene['nicpToFullIds']], axis=0).max())
    geometry = checkFrontGeometry(original,candidate,scene['faces'],camera=camera,
        fixedVertexIds=support['fixedVertexIds'],maxDisplacement=extent*.03,
        maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3)
    if not geometry['accepted']:
        return dict(accepted=False,reasons=list(geometry['reasons']),geometry=geometry,
            fullAssembly={'status':'notEvaluatedBecauseGeometryRejected','excludedMaterialIds':[]})
    pairs = intersectionPairs(candidate,scene['faces'])
    added = sorted(pairs-baselinePairs)
    reasons = list(geometry['reasons'])
    if added: reasons.append('newFullAssemblyIntersections')
    return dict(accepted=not reasons, reasons=reasons, geometry=geometry,
        fullAssembly=dict(baselinePairCount=len(baselinePairs),candidatePairCount=len(pairs),
            newPairs=added,excludedMaterialIds=[],
            scope='all original full faces including eye balls and alpha cards; existing shared-index neighbor rule retained'))


def refineClientVisibleEyes(scene, camera, photoEyes, *, imageShape, cornerFaceIds,
                           fixedVertexIds=(), protectedVertexIds=(), originalPixelScale=1., policy=None):
    """Propose local aperture correction, retaining raw rejected geometry.

    ``scene`` has the fields returned by ``loadLegacyEyeBindingInputs``. The
    caller authenticates camera/photo/cache identity and accepted observations.
    This function validates their numerical contract, never infers landmarks.
    Eye balls, gaze, alpha cards, mouth accessories, topology and UVs are fixed.
    ``vertices`` is the accepted result or an exact original copy; inspect
    ``rawCandidateVertices`` and ``candidateProposals`` retain the last and all
    raw backtracking proposals. ``solverVertices`` records the solver's final
    accepted iterate separately; it can differ from the last rejected proposal.
    A result is still experimental and does not imply client-ready quality.
    """
    effective = _policy(policy)
    if (len(imageShape)!=2 or any(type(value) not in (int,np.int64,np.int32) or value<2 for value in imageShape)
            or isinstance(originalPixelScale,(bool,np.bool_)) or not np.isfinite(originalPixelScale) or originalPixelScale<=0):
        raise ValueError('Expected a positive image shape and original pixel scale')
    if camera.get('valid', True) is not True:
        raise ValueError('Cannot refine with a rejected photo camera')
    checked, corners = _scene(scene,cornerFaceIds)
    camera = _validateCameras([camera])[0]
    photoEyes = {name:{part:value.copy() for part,value in eye.items()}
        for name,eye in _curves(photoEyes,imageShape,'photoEyes').items()}
    if any(len(curve)<4 for eye in photoEyes.values() for curve in eye.values()):
        raise ValueError('Visible contour measurement needs at least four points per lid')
    original = checked['vertices'].copy()
    baseline = extractVisibleEyeContours(**checked,camera=camera,photoEyes=photoEyes,imageShape=imageShape,
        apertureAccessoryMode='innerCornerHeadAccessories',apertureAccessoryFaceIds=corners,
        minimumEyeRasterWidth=256,maximumRoiSize=768)
    support = buildClientVisibleEyeSupport(checked,camera,photoEyes,baseline,cornerFaceIds=corners,
        fixedVertexIds=fixedVertexIds,protectedVertexIds=protectedVertexIds,policy=effective)
    if support['diagnostics']['status'] != 'supported' or not support['movableMask'].any():
        return dict(vertices=original.copy(),rawCandidateVertices=original.copy(),support=support,
            solverVertices=original.copy(),candidateProposals=[],
            beforeContours=baseline,afterContours=baseline,rawCandidateContours=baseline,textureContours=None,
            diagnostics=dict(status='skippedInsufficientVisibleSupport',accepted=False,policy=effective,
                geometryChanged=False,clientQualityApproved=False,fastSolverInvoked=False))
    baselinePairs = intersectionPairs(original,checked['faces'])
    proposals = []
    def candidateGate(candidate, context):
        strict = _strictCandidateChecks(original,candidate,checked,camera,support,baselinePairs)
        proposals.append(dict(vertices=np.asarray(candidate).copy(),round=context['round'],scale=context['scale'],
            internalGeometry=deepcopy(context['geometry']),strict=deepcopy(strict)))
        return strict
    fitted = refineFastFrontEyes(original,checked['faces'],checked['materialIds'],camera,photoEyes,
        skinVertexIds=checked['nicpToFullIds'],cornerFaceIds=corners,fixedVertexIds=support['fixedVertexIds'],
        imageShape=imageShape,uvs=checked['uvs'],uvFaces=checked['uvFaces'],materials=checked['materials'],
        alphaTextures=checked['alphaTextures'],originalPixelScale=originalPixelScale,
        rounds=effective['rounds'],iterations=effective['iterationsPerRound'],candidateGate=candidateGate)
    solverVertices = np.asarray(fitted['vertices']).copy()
    raw = proposals[-1]['vertices'].copy() if proposals else solverVertices.copy()
    strict = _strictCandidateChecks(original,solverVertices,checked,camera,support,baselinePairs)
    changed = solverVertices.shape==original.shape and not np.array_equal(solverVertices,original)
    visibleEyes = fitted['afterContours'].get('eyes', {})
    visibleAccepted = (set(visibleEyes) == set(photoEyes)
        and all(eye.get('status')=='accepted' for eye in visibleEyes.values()))
    if not visibleAccepted:
        strict['accepted']=False
        strict['reasons'].append('unreliableCompleteCandidateContours')
    accepted = strict['accepted'] and changed and fitted['diagnostics']['status']=='acceptedLocalExperiment'
    result = solverVertices.copy() if accepted else original.copy()
    for proposal in proposals:
        stage = next((stage for stage in fitted['diagnostics'].get('stages',[]) if stage['round']==proposal['round']),{})
        proposal['solverAttempt'] = deepcopy(next((attempt for attempt in stage.get('attempts',[]) if attempt['scale']==proposal['scale']),{}))
        proposal['selectedByFastRound'] = stage.get('status')=='accepted' and stage.get('selectedScale')==proposal['scale']
    return dict(vertices=result,rawCandidateVertices=raw,solverVertices=solverVertices,candidateProposals=proposals,support=support,
        beforeContours=baseline,afterContours=fitted['afterContours'] if accepted else baseline,
        rawCandidateContours=fitted['afterContours'] if np.array_equal(raw,solverVertices) else None,
        solverContours=fitted['afterContours'],textureContours=fitted.get('textureContours') if accepted else None,
        diagnostics=dict(status='acceptedLocalExperiment' if accepted else ('rejectedStrictCandidate' if changed else 'noImprovingSafeCandidate'),
            accepted=bool(accepted),policy=effective,fastSolverInvoked=True,fast=fitted['diagnostics'],strict=strict,
            geometryChanged=bool(accepted),rawCandidateChanged=not np.array_equal(raw,original),
            rawCandidateEqualsSelectedResult=bool(np.array_equal(raw,result)),proposalCount=len(proposals),
            fixedVerticesExactlyPreserved=bool(np.array_equal(result[support['fixedMask']],original[support['fixedMask']])),
            cameraRefitted=False,eyeBallsRefitted=False,gazeRefitted=False,accessoriesReassembled=False,
            topologyOrUvsChanged=False,photoObservationsReestimated=False,clientQualityApproved=False,
            limitations=['Photo contours are automatic observations, not manually verified anatomy.',
                'Strict full-mesh intersections include eye and alpha-card contact; rejection is retained, never waived.',
                'A smaller field can limit fitting; no fallback widens the support or moves protected skin.',
                'Client appearance, other-view support and earlier nose/mouth gates need independent whole-pipeline validation.']))
