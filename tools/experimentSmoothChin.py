"""Compare broad, smooth lower-face shortening fields on the original mesh."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from runClientChinExperiment import ROOT, context, load, project, sha
from faceAppearance.artifacts import writeJson
from faceAppearance.clientChinRefinement import visibleLowerJawConstraints
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.frontLandmarkRefinement import checkFrontGeometry


def smoothstep(value):
    value = np.clip(value, 0., 1.)
    return value*value*(3.-2.*value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'runs/clientAppearanceFix20261012/deployedSource')
    parser.add_argument('--output', type=Path, default=ROOT/'runs/clientChinSmooth20261012')
    args = parser.parse_args()
    output = args.output/'geometry'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    geometry, _, mouth, cameras, assets = context(args)
    old = load(ROOT/'runs/clientChinRefine20261012/geometry/clientGeometry.npz')
    priorReport = json.loads((ROOT/'runs/clientChinRefine20261012/geometry/report.json').read_text())
    vertices, faces = geometry['vertices'], geometry['faces']
    camera = cameras[0]
    rotation, intrinsics = np.asarray(camera['rotation']), np.asarray(camera['intrinsics'])
    cameraVertices = vertices@rotation.T+camera['translation']
    pixels = project(vertices, camera)
    fixed = np.flatnonzero(assets.arrays['staticMask'])
    protected = np.unique(mouth['vertexIndices'])
    # Extra fixed boundaries keep all deformation below the mouth, in front
    # of the back of the head. Fixed values are imposed in the linear solve.
    extraFixed = np.flatnonzero((pixels[:,1] <= 372.) | (cameraVertices[:,2] >= 5.25))
    hardFixed = np.union1d(np.union1d(fixed, protected), extraFixed)
    free = np.setdiff1d(np.arange(len(vertices)), hardFixed)
    extent = float(np.ptp(vertices, axis=0).max())
    maximum = extent*.035
    target = old['observedPixels'][0]
    observations = load(ROOT/'runs/clientChinRefine20261012/observations/landmarkObservations.npz')
    from scipy.interpolate import PchipInterpolator
    support = observations['faceLandmarksWorking'][0,6:11]
    spline = PchipInterpolator(support[:,0], support[:,1])
    tangents = np.column_stack([np.ones(len(target)), spline.derivative()(target[:,0])])
    contourArgs = dict(targetPixels=target, confidence=old['rawPeakScores'][0],
        landmarkIds=old['jawIds'], minimumYPixels=375.62147708732346, targetTangents=tangents)
    visibleBefore = visibleLowerJawConstraints(vertices, faces, camera, **contourArgs)
    baselinePairs = intersectionPairs(vertices, faces)
    edges = np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    weights = 1/np.maximum(np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1), extent*1e-5)
    adjacency = coo_matrix((np.r_[weights,weights], (np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(vertices),len(vertices))).tocsr()
    degree = np.asarray(adjacency.sum(1)).ravel()
    laplacian = diags(np.ones(len(vertices)))-diags(1/degree)@adjacency
    bending = laplacian.T@laplacian
    # Broad projected field has the same desired motion through chin depth
    # layers; it is not centered on whichever edge is currently the contour.
    y = pixels[:,1]
    field = smoothstep((y-372.)/76.)*(1-smoothstep((y-482.)/105.))
    field *= 1-smoothstep((np.abs(pixels[:,0]-258.)-65.)/100.)
    field *= 1-smoothstep((cameraVertices[:,2]-4.65)/.6)
    field[hardFixed] = 0.
    firstLayer = np.unique([row['vertexIndices'] for row in priorReport['visibleContourBefore']['selectedEdges']])
    nextLayer = np.unique([row['vertexIndices'] for row in priorReport['visibleContourAfter']['selectedEdges']])
    candidates, accepted = [], []
    options = dict(fixedVertexIds=hardFixed, camera=camera, maxDisplacement=maximum,
        maxRelativeEdgeChange=.25, minAreaRatio=.4, minNormalCosine=.3)
    for smoothing in (1.,4.,16.,64.):
        system = diags(np.ones(len(vertices)))+smoothing*bending
        broad = np.zeros(len(vertices))
        broad[free] = spsolve(system[free][:,free], field[free])
        # A broad scalar field moves only in the reference camera vertical
        # plane; depth stays exactly fixed up to floating-point roundoff.
        pixelToWorld = cameraVertices[:,2]/intrinsics[1,1]
        for amplitude in (10.,14.,18.,22.,26.):
            deltaY = -amplitude*broad*pixelToWorld
            candidate = vertices+deltaY[:,None]*rotation[1]
            check = checkFrontGeometry(vertices,candidate,faces,**options)
            record = {'smoothing':smoothing,'requestedPixels':amplitude,'geometry':check}
            candidates.append(record)
            if not check['accepted']:
                record['status']='rejected_geometry'; continue
            pairs = intersectionPairs(candidate,faces)
            added = pairs-baselinePairs
            record.update(newIntersectionPairs=len(added),finalIntersectionPairs=len(pairs))
            if added:
                record['status']='rejected_intersections'; continue
            try:
                contour = visibleLowerJawConstraints(candidate,faces,camera,**contourArgs)
            except ValueError as error:
                record.update(status='rejected_contour',reason=str(error)); continue
            projectedDelta = project(candidate,camera)[:,1]-pixels[:,1]
            layerGap = abs(float(projectedDelta[firstLayer].mean()-projectedDelta[nextLayer].mean()))
            error = contour['diagnostics']['normalMeanAbsoluteErrorPixels']
            record.update(status='accepted',visibleContour=contour['diagnostics'],
                contourNormalErrorPixels=error,
                originalContourLayerMeanMovementPixels=float(projectedDelta[firstLayer].mean()),
                neighboringFrontLayerMeanMovementPixels=float(projectedDelta[nextLayer].mean()),
                layerMovementDifferencePixels=layerGap,
                changedVertexCount=int((np.linalg.norm(candidate-vertices,axis=1)>1e-12).sum()))
            # Prefer a visibly useful correction with strain margin and coherent
            # depth-layer motion. No reward for forcing sub-pixel front fitting.
            strain = check['metrics']['maximumRelativeEdgeChange']
            record['selectionCost'] = max(error-8.,0.)**2 + 2.*layerGap**2 + 15.*strain**2
            accepted.append((record['selectionCost'],record,candidate,contour))
    if not accepted:
        writeJson(output/'rejected.json',{'candidates':candidates})
        raise RuntimeError('No safe broad candidate')
    _, selected, after, visibleAfter = min(accepted,key=lambda item:item[0])
    binding=visibleBefore['constraint']
    np.savez_compressed(output/'clientGeometry.npz',before=vertices,after=after,
        faces=faces,uvs=geometry['uvs'],uvFaces=geometry['uvFaces'],
        jawIds=old['jawIds'],jawVertexIndices=binding['vertexIndices'],jawBarycentricWeights=binding['barycentric'],
        observedPixels=old['observedPixels'],rawPeakScores=old['rawPeakScores'],protectedVertexIds=protected,
        fixedVertexIds=fixed,allHardFixedVertexIds=hardFixed)
    report={'status':'complete','method':'broadLowerFaceScreenedBiharmonicVerticalField',
        'source':str(args.source),'seconds':perf_counter()-start,'referenceViewIndex':0,
        'rationale':'Original mesh receives a broad camera-vertical field shared across chin depth layers; screened mesh bending smooths transitions to exact fixed mouth and neck boundaries.',
        'selectionPolicy':'Moderate front correction, coherent chin-depth-layer movement, and strain margin; all candidates checked cumulatively against original geometry.',
        'selectedCandidate':selected,'candidates':candidates,
        'geometry':{'finalGeometryCheck':selected['geometry'],'changedVertexCount':selected['changedVertexCount'],
            'baselineIntersectionPairs':len(baselinePairs),'finalIntersectionPairs':selected['finalIntersectionPairs'],
            'newIntersectionPairs':selected['newIntersectionPairs'],'selfIntersectionChecked':True,
            'fixedVerticesExactlyUnchanged':bool(np.array_equal(after[hardFixed],vertices[hardFixed]))},
        'eyesMouthSupportExactlyUnchanged':bool(np.array_equal(after[protected],vertices[protected])),
        'visibleContourBefore':visibleBefore['diagnostics'],'visibleContourAfter':visibleAfter['diagnostics'],
        'sourceHashes':{name:sha(args.source/name) for name in ['cameras.json','inputs.json','directClient/geometry.npz']},
        'remoteModified':False,'limitations':['One male subject and estimated fixed cameras.',
            'Automatic front landmarks are observed proxies; side appearance requires render inspection.',
            'Two-layer displacement coherence is a diagnostic, not proof that rendered shading is artifact free.']}
    writeJson(output/'report.json',report)
    print(json.dumps({'status':'complete','output':str(output),'seconds':report['seconds'],
        'selected':{k:v for k,v in selected.items() if k not in {'visibleContour','geometry'}},
        'geometry':selected['geometry']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
