"""Frozen skin-topology support for the existing six-point eye alignment.

This does not replace semantic observations, align cameras, or place eyes and
other accessories. It supplies hard fixed vertices to the existing solver.
Its input geometry must stay the original reference for both eye stages.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from .camera import projectPoints
from .geometryRefinement import _meshEdges, _validateCameras, _validateMesh


DEFAULT_POLICY={'radiusEyeWidthFraction':.5}


def normalizeSemanticEyeSupportPolicy(policy=None):
    result=DEFAULT_POLICY.copy()
    if policy is not None:
        if not isinstance(policy,dict) or set(policy)-set(result):
            raise ValueError('Unknown semantic eye support policy')
        result.update(policy)
    value=result['radiusEyeWidthFraction']
    if (isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number))
            or not np.isfinite(value) or not 0<value<=1.):
        raise ValueError('Semantic eye support radius must be a finite fraction in (0,1]')
    result['radiusEyeWidthFraction']=float(value)
    return result


def _fixedIds(values,count):
    ids=np.asarray(values)
    if not ids.size:return np.empty(0,np.int64)
    if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or ids.min()<0 or ids.max()>=count
            or len(np.unique(ids))!=len(ids)):
        raise ValueError('Fixed vertices must be distinct original skin indices')
    return ids.astype(np.int64,copy=True)


def _binding(binding,count):
    ids=np.asarray(binding['ibugIds']);vertices=np.asarray(binding['vertexIndices'])
    bary=np.asarray(binding['barycentricWeights'],float)
    if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids) or ids.min()<0 or ids.max()>=68
            or len(np.unique(ids))!=len(ids) or vertices.shape!=(len(ids),3)
            or vertices.dtype.kind not in 'iu' or vertices.min()<0 or vertices.max()>=count
            or bary.shape!=vertices.shape or not np.isfinite(bary).all()):
        raise ValueError('Expected original iBUG semantic triangle bindings')
    return ids,vertices,bary


def semanticEyeFitMetrics(vertices,binding,observation,camera,eligibleMask):
    """Report unchanged semantic point errors, without silently adding targets."""
    vertices=np.asarray(vertices,float)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():
        raise ValueError('Expected finite canonical skin vertices')
    ids,indices,bary=_binding(binding,len(vertices))
    eligible=np.asarray(eligibleMask)
    if eligible.shape!=ids.shape or eligible.dtype.kind!='b' or not eligible.any():
        raise ValueError('A nonempty original semantic evaluation mask is required')
    points=np.asarray(observation['pixels'],float)
    if points.shape!=(68,2) or not np.isfinite(points).all():raise ValueError('Expected the original finite 68 photo points')
    camera=_validateCameras([camera])[0]
    samples=(vertices[indices[eligible]]*bary[eligible,:,None]).sum(1)
    projected,depth=projectPoints(samples,camera['intrinsics'],camera['rotation'],camera['translation'])
    if not (depth>0).all():raise ValueError('Semantic eye support must be in front of the camera')
    errors=np.linalg.norm(projected-points[ids[eligible]],axis=1)
    return dict(count=len(errors),ibugIds=ids[eligible].copy(),errorsPixels=errors,
        meanErrorPixels=float(errors.mean()),maximumErrorPixels=float(errors.max()))


def buildSemanticEyeSupport(vertices,faces,binding,observation,camera,*,eyeIbugIds,
                            eligibleMask,fixedVertexIds=(),policy=None):
    """Hard-pin outside a common eye-width geodesic band around semantic seeds.

    All selected original binding-triangle vertices seed the skin graph. The
    photo canthus chord is backprojected at the median selected binding depth;
    this gives a physical radius in the current canonical frame. Disconnected
    surfaces never become movable merely because they are spatially nearby.
    Missing reliable canthi or insufficient eligible targets explicitly skip
    this optional stage. No fallback expands the support.
    """
    vertices,faces=_validateMesh(vertices,faces)
    if camera.get('valid',True) is not True:raise ValueError('Rejected semantic eye camera')
    camera=_validateCameras([camera])[0];effective=normalizeSemanticEyeSupportPolicy(policy)
    ids,indices,bary=_binding(binding,len(vertices))
    eye=np.asarray(eyeIbugIds)
    if (eye.ndim!=1 or eye.dtype.kind not in 'iu' or len(eye)!=6
            or set(eye.tolist()) not in (set(range(36,42)),set(range(42,48)))):
        raise ValueError('Expected one original six-point iBUG eye group')
    eligible=np.asarray(eligibleMask)
    if eligible.shape!=ids.shape or eligible.dtype.kind!='b':raise ValueError('Eligible mask must address the original semantic bindings')
    selected=eligible & np.isin(ids,eye)
    fixed=_fixedIds(fixedVertexIds,len(vertices))
    pixels=np.asarray(observation['pixels'],float);confidence=np.asarray(observation['confidence'],float)
    if (pixels.shape!=(68,2) or confidence.shape!=(68,) or not np.isfinite(pixels).all()
            or not np.isfinite(confidence).all()):raise ValueError('Expected original finite photo observations')
    reasons=[]
    if selected.sum()<4:reasons.append('insufficientEligibleEyeTargets')
    canthi=np.array([int(eye.min()),int(eye.min())+3])
    if (confidence[canthi]<.5).any():reasons.append('unreliablePhotoCanthusWidth')
    if 'workingRgb' in observation:
        shape=np.asarray(observation['workingRgb']).shape
        if len(shape)!=3 or shape[2]!=3:raise ValueError('Expected an RGB observation image')
        if np.any(pixels[canthi]<0) or np.any(pixels[canthi]>=np.array(shape[1::-1])):
            reasons.append('photoCanthusOutsideWorkingImage')
    if (np.any(bary[selected]<-1e-7) or np.any(bary[selected]>1+1e-7)
            or not np.allclose(bary[selected].sum(1),1.,atol=1e-7,rtol=0)):
        raise ValueError('Selected semantic bindings need valid triangle barycentric weights')
    triangleSet={tuple(row) for row in np.sort(faces,axis=1)}
    if any(tuple(row) not in triangleSet for row in np.sort(indices[selected],axis=1)):
        raise ValueError('Selected semantic bindings must refer to actual original skin faces')
    samples=(vertices[indices[selected]]*bary[selected,:,None]).sum(1)
    depths=samples@camera['rotation'][2]+camera['translation'][2]
    if len(depths) and not (depths>0).all():reasons.append('semanticSeedsBehindCamera')
    seeds=np.unique(indices[selected]);distance=np.full(len(vertices),np.inf)
    movable=np.zeros(len(vertices),bool);width=None;radius=None;localDepth=None
    if not reasons:
        rays=np.linalg.solve(camera['intrinsics'],np.c_[pixels[canthi],np.ones(2)].T).T
        rays/=rays[:,2,None];localDepth=float(np.median(depths))
        width=float(np.linalg.norm(rays[1]-rays[0])*localDepth)
        if not np.isfinite(width) or width<=0:reasons.append('degeneratePhotoCanthusWidth')
        else:
            radius=width*effective['radiusEyeWidthFraction']
            edges=_meshEdges(faces);lengths=np.linalg.norm(vertices[edges[:,1]]-vertices[edges[:,0]],axis=1)
            graph=sparse.csr_matrix((np.r_[lengths,lengths],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),
                shape=(len(vertices),len(vertices)))
            # Relative numerical allowance prevents equivalent units or graph
            # summation order from flipping a vertex exactly on the boundary.
            limit=radius*(1+1e-12)
            distance=dijkstra(graph,directed=False,indices=seeds,min_only=True,limit=limit)
            movable=distance<=limit;movable[fixed]=False
            if not movable.any():reasons.append('allLocalSupportHardFixed')
    if reasons:movable[:]=False
    measured=np.isfinite(distance)
    return dict(movableMask=movable,fixedMask=~movable,fixedVertexIds=np.flatnonzero(~movable),
        originalFixedVertexIds=fixed,selectedBindingMask=selected,seedVertexIds=seeds,
        geodesicDistance=np.where(measured,distance,0.),geodesicDistanceMeasuredMask=measured,
        photoCanthusIbugIds=canthi,photoCanthusPixels=pixels[canthi].copy(),
        diagnostics=dict(status='supported' if not reasons else 'unsupported',reasons=reasons,policy=effective,
            eyeIbugIds=sorted(eye.tolist()),eligibleTargetCount=int(selected.sum()),seedVertexCount=len(seeds),
            movableVertexCount=int(movable.sum()),fixedVertexCount=int((~movable).sum()),
            localCameraDepth=localDepth,photoEyeWidthCanonical=width,geodesicRadiusCanonical=radius,
            relativeBoundaryRoundingAllowance=1e-12,
            distanceEncoding='Only truncated-Dijkstra finite distances are measured; outside the measured mask the stored zero is a placeholder, not zero distance.',
            reference='original input to staged landmarks, frozen before mouth and either eye',
            graph='original skin topology only',candidateDependentSupport=False,
            attachmentPlacementChanged=False,fallback='retain exact stage input; never expand the support'))
