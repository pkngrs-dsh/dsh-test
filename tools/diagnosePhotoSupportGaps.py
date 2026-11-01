"""Read-only CPU audit of saved visible-but-unsupported mouth pixels.

No renderer, model, new reconstruction, or filled texture is created. Continuous
triangle intersections audit the exact same frozen projection used by the saved
renderer, and distinguish low-resolution nearest-depth rejection from occlusion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

projectRoot = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(projectRoot / 'src'), str(projectRoot / 'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.camera import projectPoints
from faceAppearance.texture import (buildUvLookup, interpolateSurface, rasterizeDepth,
                                     samplePixels, vertexNormals)
from renderCropComparison import fileHash, photoCamera
from runReferenceExperiment import readArrays


def intersectPhotoRays(queryPixels, projectedVertices, vertexDepths, faces, frontFaces=None):
    """Perspective-correct nearest intersections at arbitrary photo XY points.

    This deliberately small diagnostic batches at most 64 query rays. Integer
    XY denotes original image pixel centers. Optional face culling matches the
    renderer; two-sided intersections match collectAppearanceObservations.
    """
    first, second, third = projectedVertices[faces[:, 0]], projectedVertices[faces[:, 1]], projectedVertices[faces[:, 2]]
    denominator = ((second[:, 1]-third[:, 1])*(first[:, 0]-third[:, 0])
                   + (third[:, 0]-second[:, 0])*(first[:, 1]-third[:, 1]))
    denominator = np.where(abs(denominator) < 1e-12, np.nan, denominator)
    allDepth, allFaces, allBarycentric = [], [], []
    for start in range(0, len(queryPixels), 64):
        query = queryPixels[start:start+64]
        with np.errstate(invalid='ignore', divide='ignore'):
            firstWeight = ((second[:, 1]-third[:, 1])*(query[:, None, 0]-third[:, 0])
                           + (third[:, 0]-second[:, 0])*(query[:, None, 1]-third[:, 1]))/denominator
            secondWeight = ((third[:, 1]-first[:, 1])*(query[:, None, 0]-third[:, 0])
                            + (first[:, 0]-third[:, 0])*(query[:, None, 1]-third[:, 1]))/denominator
            screenWeights = np.stack([firstWeight, secondWeight, 1-firstWeight-secondWeight], -1)
            weightedInverseDepth = screenWeights/vertexDepths[faces][None]
            depth = 1/weightedInverseDepth.sum(-1)
            barycentric = weightedInverseDepth/weightedInverseDepth.sum(-1, keepdims=True)
        inside = ((screenWeights.min(-1) >= -1e-7) & np.isfinite(screenWeights).all(-1)
                  & (depth > 0) & (vertexDepths[faces] > 0).all(1)[None])
        if frontFaces is not None:
            inside &= frontFaces[None]
        depth = np.where(inside, depth, np.inf)
        indices = depth.argmin(1)
        selectedDepth = depth[np.arange(len(query)), indices]
        if not np.isfinite(selectedDepth).all():
            raise ValueError('Audit ray lacks a mesh intersection; do not infer a texture cause')
        allDepth.append(selectedDepth)
        allFaces.append(indices)
        allBarycentric.append(barycentric[np.arange(len(query)), indices])
    return np.concatenate(allDepth), np.concatenate(allFaces), np.concatenate(allBarycentric)


def auditSupportGaps(diagnostic, output, viewIndex=0):
    diagnostic, output = Path(diagnostic).resolve(), Path(output).resolve()
    report = json.loads((diagnostic/'diagnosticReport.json').read_text())
    source = Path(report['source'])
    sourceReport = json.loads((source/'experimentReport.json').read_text())
    for name in ('geometry.npz', 'cameras.json', 'inputs.json', 'semanticProtectedObservations.npz'):
        if fileHash(source/name) != report['sourceHashes'][name]:
            raise ValueError('Saved support diagnostic no longer matches geometry/observations')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty audit directory')
    output.mkdir(parents=True)
    geometry = readArrays(source/'geometry.npz')
    observations = readArrays(source/'semanticProtectedObservations.npz')
    semantics = readArrays(Path(sourceReport['semanticCache'])/f'view{viewIndex:02d}.npz')
    inputs = json.loads((source/'inputs.json').read_text())
    camera = json.loads((source/'cameras.json').read_text())[viewIndex]
    photo = photoCamera(camera, inputs[viewIndex], 'fitted')
    referenceRun = Path(sourceReport['source'])
    referenceReport = json.loads((referenceRun/'experimentReport.json').read_text())
    delight = readArrays(Path(referenceReport['source'])/'delight.npz')
    directory = diagnostic/f'view{viewIndex:02d}'
    visible = np.asarray(Image.open(directory/'meshVisible.png')) > 0
    supported = np.asarray(Image.open(directory/'ownRawSupport.png')) > 0
    ys, xs = np.where(np.isin(semantics['labelMap'], [11, 12, 13]))
    box = (max(int(xs.min())-20, 0), max(int(ys.min())-20, 0),
           min(int(xs.max())+21, visible.shape[1]), min(int(ys.max())+21, visible.shape[0]))
    region = np.zeros_like(visible)
    region[box[1]:box[3], box[0]:box[2]] = True
    ys, xs = np.where(region & visible & ~supported)
    query = np.stack([xs, ys], 1)
    if not len(query):
        raise ValueError('No visible unsupported mouth pixels in this saved view')
    vertices, faces, uvs = geometry['vertices'].astype(float), geometry['faces'], geometry['uvs']
    rotation, translation = np.asarray(camera['rotation']), np.asarray(camera['translation'])
    cameraVertices = vertices@rotation.T + translation
    faceNormals = np.cross(cameraVertices[faces[:, 1]]-cameraVertices[faces[:, 0]],
                           cameraVertices[faces[:, 2]]-cameraVertices[faces[:, 0]])
    frontFaces = (faceNormals*cameraVertices[faces].mean(1)).sum(1) < 0
    projected, depths = projectPoints(vertices, photo['intrinsics'], rotation, translation)
    _, hitFaces, hitBarycentric = intersectPhotoRays(query, projected, depths, faces, frontFaces)
    queryUvs = (uvs[faces[hitFaces]]*hitBarycentric[..., None]).sum(1)
    resolution = observations['weights'].shape[-1]
    atlasPositions = queryUvs*(resolution-1)
    validAtlas = observations['weights'][viewIndex] > 0
    sampledCoverage = samplePixels(validAtlas.astype(float), atlasPositions)
    if not (sampledCoverage < .999).all():
        raise ValueError('Independent intersections did not reproduce every saved support gap')
    bases = np.floor(atlasPositions).astype(int)
    missingNeighbors = []
    for offset in ([0, 0], [1, 0], [0, 1], [1, 1]):
        coordinates = bases+offset
        valid = validAtlas[coordinates[:, 1], coordinates[:, 0]]
        interpolationWeight = np.prod(1-abs(atlasPositions-coordinates), axis=1)
        missingNeighbors.extend(coordinates[(~valid) & (interpolationWeight > 1e-8)].tolist())
    missingNeighbors = np.unique(missingNeighbors, axis=0)
    lookup = buildUvLookup(uvs, faces, resolution)
    normals = interpolateSurface(vertexNormals(vertices, faces), faces, lookup)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-12)
    surfaces = interpolateSurface(vertices, faces, lookup).astype(float)
    x, y = missingNeighbors[:, 0], missingNeighbors[:, 1]
    cameraSurface = surfaces[y, x]@rotation.T+translation
    directions = -cameraSurface/np.linalg.norm(cameraSurface, axis=1, keepdims=True)
    cosine = np.maximum(np.sum((normals[y, x]@rotation.T)*directions, axis=1), 0)
    workProjected = cameraSurface@np.asarray(camera['intrinsics']).T
    workPixels = workProjected[:, :2]/workProjected[:, 2:]
    height, width = delight['validMasks'][viewIndex].shape
    depthMap = rasterizeDepth(cameraVertices, faces, np.asarray(camera['intrinsics']), height, width)
    nearest = np.rint(workPixels).astype(int)
    sampledDepth = depthMap[nearest[:, 1], nearest[:, 0]]
    nearestResidual = abs(sampledDepth-cameraSurface[:, 2])
    tolerance = float(np.ptp(vertices, axis=0).max())*.005
    originalMaskCoverage = samplePixels(delight['validMasks'][viewIndex].astype(float), workPixels)
    workVertexPixels, workVertexDepths = projectPoints(vertices, np.asarray(camera['intrinsics']), rotation, translation)
    exactDepth, exactFaces, _ = intersectPhotoRays(workPixels, workVertexPixels, workVertexDepths, faces)
    exactResidual = abs(exactDepth-cameraSurface[:, 2])
    recoverable = ((nearestResidual > tolerance) & (exactResidual <= tolerance) & (cosine > .1)
                   & (originalMaskCoverage >= .999) & lookup['validMask'][y, x])
    # Mask-only what-if: this creates neither colors nor a new rendered result.
    exactMask = validAtlas.copy()
    exactMask[y[recoverable], x[recoverable]] = True
    potentialCoverage = samplePixels(exactMask.astype(float), atlasPositions)
    supportComparison = {}
    for name in ('ownRawSupport', 'ownShSupport', 'frontShSupport', 'fusedShSupport'):
        mask = np.asarray(Image.open(directory/f'{name}.png')) > 0
        supportComparison[name] = {'unsupportedPixelsInMouthRegion': int((region & ~mask).sum()),
                                   'originalGapPixelsNowSupported': int(mask[ys, xs].sum())}
    outputReport = {'status': 'complete', 'viewIndex': viewIndex, 'mouthRegionXyxy': box,
        'regionPixels': int(region.sum()), 'nonVisiblePixels': int((region & ~visible).sum()),
        'visibleUnsupportedPixels': len(query), 'partialBilinearSupportPixels': int((sampledCoverage > 1e-8).sum()),
        'noBilinearNeighborSupportPixels': int((sampledCoverage <= 1e-8).sum()),
        'supportComparison': supportComparison, 'missingAtlasNeighborTexels': len(missingNeighbors),
        'missingTexelGates': {'outsideUvLookup': int((~lookup['validMask'][y, x]).sum()),
            'normalCosineAtMostPointOne': int((cosine <= .1).sum()),
            'nearestDepthRejected': int((nearestResidual > tolerance).sum()),
            'noRasterDepthSupport': int((~np.isfinite(sampledDepth)).sum()),
            'savedOpenDelightMaskRejected': int((originalMaskCoverage < .999).sum()),
            'exactContinuousDepthRejected': int((exactResidual > tolerance).sum()),
            'nearestDepthRejectedButExactWithinTolerance': int(((nearestResidual > tolerance) & (exactResidual <= tolerance)).sum()),
            'alsoPassNormalAndMaskAfterExactDepth': int(recoverable.sum())},
        'depthTolerance': tolerance, 'observationDepthImageShape': [height, width],
        'maskOnlyWhatIf': {'fullSupportAmongOriginalGaps': int((potentialCoverage >= .999).sum()),
                          'anySupportAmongOriginalGaps': int((potentialCoverage > 1e-8).sum()),
                          'meaning': 'No colors generated; effect of exact depth on these atlas gates only'},
        'source': str(source), 'savedDiagnostic': str(diagnostic), 'savedDiagnosticSha256': fileHash(diagnostic/'diagnosticReport.json'),
        'implementationSha256': fileHash(Path(__file__)),
        'limits': ['This is a saved-result support audit, not a reconstructed or corrected image',
                   'The saved fused atlas may already fill gaps using another photo and introduce a separate alignment issue',
                   'Partial-neighbor normalization cannot fill pixels with no valid neighbors',
                   'Continuous depth retains the original tolerance; low-angle and truly occluded texels remain invalid']}
    np.savez_compressed(output/'supportSamples.npz', pixels=query, faceIds=hitFaces, samplingUvs=queryUvs,
        bilinearCoverage=sampledCoverage, missingUvPixels=missingNeighbors, missingUvCosine=cosine,
        missingUvNearestDepthResidual=nearestResidual, missingUvExactDepthResidual=exactResidual,
        missingUvExactOccluderFaceIds=exactFaces, recoverableTexels=recoverable,
        maskOnlyPotentialCoverage=potentialCoverage)
    writeJson(output/'report.json', outputReport)
    names = ('original', 'ownRaw', 'meshVisible', 'ownRawSupport', 'fusedShSupport')
    tileWidth, tileHeight = box[2]-box[0], box[3]-box[1]
    canvas = Image.new('RGB', (tileWidth*len(names), tileHeight+25))
    for index, name in enumerate(names):
        canvas.paste(Image.open(directory/f'{name}.png').convert('RGB').crop(box), (index*tileWidth, 25))
        ImageDraw.Draw(canvas).text((index*tileWidth+3, 3), name, fill='white')
    canvas.resize((canvas.width*2, canvas.height*2)).save(output/'supportComparison.png')
    print(json.dumps(outputReport, ensure_ascii=False, indent=2))
    return outputReport


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=projectRoot/'runs/ghostingSourceDiagnostic')
    parser.add_argument('--output', type=Path, default=projectRoot/'runs/mouthSupportGapDiagnostic')
    parser.add_argument('--viewIndex', type=int, default=0)
    args = parser.parse_args()
    auditSupportGaps(args.input, args.output, args.viewIndex)
