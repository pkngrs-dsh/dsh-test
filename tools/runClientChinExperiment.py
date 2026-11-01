"""Local lower-face geometry experiment with frozen photos, cameras and SH input.

This does not modify the production pipeline or remote deployment. The geometry
stage consumes separately inspected full-face landmark observations. The bake
stage reprojects the existing SH photographs onto the corrected client surface.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.clientLandmarks import loadClientLandmarks, ibugClientLandmarks, sampleClientLandmarks
from faceAppearance.clientTextureBaking import bakeClientTexture
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.legacyAsset import exportLegacyAsset
from faceAppearance.legacyPreviewAsset import _load
from runClientShComparison import load, sha, display


def project(vertices, camera):
    K, R, t = [np.asarray(camera[k], np.float64) for k in ('intrinsics', 'rotation', 'translation')]
    q = (vertices @ R.T + t) @ K.T
    return q[:, :2] / q[:, 2:]


def context(args):
    source = args.source
    geometry = load(source / 'directClient/geometry.npz')
    geometry['uvConvention'] = 'topLeftPixelEdges'
    refined = load(source / 'directClient/refinement/clientGeometry.npz')
    mouth = load(source / 'directClient/refinement/mouthState.npz')
    if not np.array_equal(geometry['vertices'], mouth['afterCanonical']):
        raise ValueError('Saved final surface and mouth state disagree')
    similarity = {k: refined['templateToUff' + k.title()] for k in ('rotation', 'translation', 'scale')}
    cameras = json.loads((source / 'cameras.json').read_text())
    assets = AssemblyAssets(ROOT / 'assets/clientTemplates/legacyMaleDirectServiceV1')
    return geometry, similarity, mouth, cameras, assets


def geometryStage(args):
    from faceAppearance.clientChinRefinement import refineClientChin, visibleLowerJawConstraints

    output = args.output / 'geometry'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new experiment output for geometry')
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    geometry, similarity, mouth, cameras, assets = context(args)
    observations = load(args.observations)
    points, scores = observations['faceLandmarksWorking'], observations['faceRawPeakScores']
    if points.shape != (4, 68, 2) or scores.shape != (4, 68):
        raise ValueError('Expected four ordered iBUG68 observations')
    if 'landmarkCandidateValidMask' in observations:
        scores = np.where(observations['landmarkCandidateValidMask'], scores, 0.)
    # Only the central visible chin has a reliable front transition chain in
    # this source mesh. Seven curve samples interpolate three actual central
    # photo anchors (7,8,9); they are not seven independent detected landmarks.
    targetByView, scoreByView, tangentByView, supportIdsByView = [], [], [], []
    for view in range(4):
        reliable = (scores[view] >= args.minConfidence) & (scores[view] > 0)
        if not reliable[7:10].all():
            raise ValueError(f'View {view} needs reliable central chin anchors 7, 8, 9')
        supportIds = np.arange(6, 11)[reliable[6:11]]
        supportIdsByView.append(supportIds.tolist())
        support = points[view, supportIds]
        if (np.diff(support[:,0]) <= 0).any():
            raise ValueError('Central photo contour is not ordered in image X')
        spline = PchipInterpolator(support[:,0], support[:,1])
        xs = np.r_[np.linspace(points[view,7,0], points[view,8,0],4),
                   np.linspace(points[view,8,0], points[view,9,0],4)[1:]]
        targetByView.append(np.column_stack([xs,spline(xs)]))
        scoreByView.append(np.full(7,float(scores[view,7:10].min())))
        tangentByView.append(np.column_stack([np.ones(7),spline.derivative()(xs)]))
    targetByView, scoreByView, tangentByView = map(np.asarray,(targetByView,scoreByView,tangentByView))
    requestedIds = np.arange(700,707,dtype=np.int64)
    observedTangents = tangentByView[0]
    lowerLipY = float(project(geometry['vertices'][np.unique(mouth['vertexIndices'][mouth['ibugIds']>=48])], cameras[0])[:,1].max())
    contourArguments = dict(targetPixels=targetByView[0], confidence=np.clip(scoreByView[0],0,1),
        landmarkIds=requestedIds, minimumYPixels=lowerLipY+5., targetTangents=observedTangents)
    visibleBefore = visibleLowerJawConstraints(geometry['vertices'], geometry['faces'], cameras[0], **contourArguments)
    constraint = visibleBefore['constraint']
    ids = np.asarray(constraint['landmarkIds'], np.int64)
    binding = {'vertexIndices':constraint['vertexIndices'], 'barycentricWeights':constraint['barycentric']}
    extent = float(np.ptp(geometry['vertices'], axis=0).max())
    fixed = np.flatnonzero(assets.arrays['staticMask'])
    protected = np.unique(mouth['vertexIndices'])
    cross = []
    for view in range(1, 4):
        cross.append({**constraint, 'viewIndex': view, 'targetPixels': targetByView[view],
            'targetTangents': tangentByView[view],
            'confidence': np.clip(scoreByView[view], 0, 1), 'minConfidence': args.minConfidence})
    result = refineClientChin(geometry['vertices'], geometry['faces'], cameras, constraint,
        fixedVertexIds=fixed, protectedVertexIds=protected,
        referenceViewIndex=0, supportRadius=extent*.16, maxDisplacement=extent*.035,
        minConfidence=args.minConfidence, minimumPoints=5, maxTargetDistancePixels=35.,
        crossViewConstraints=cross)
    after = result['vertices']
    if np.array_equal(after, geometry['vertices']):
        writeJson(output/'rejected.json', result['diagnostics'])
        raise RuntimeError('No accepted lower-face modification; inspect rejected.json')
    if not np.array_equal(after[protected], geometry['vertices'][protected]):
        raise ValueError('Existing eye/mouth geometry moved')
    visibleAfter = visibleLowerJawConstraints(after, geometry['faces'], cameras[0], **contourArguments)
    np.savez_compressed(output/'clientGeometry.npz', before=geometry['vertices'], after=after,
        faces=geometry['faces'], uvs=geometry['uvs'], uvFaces=geometry['uvFaces'],
        jawIds=ids, jawVertexIndices=binding['vertexIndices'], jawBarycentricWeights=binding['barycentricWeights'],
        observedPixels=targetByView, rawPeakScores=scoreByView, protectedVertexIds=protected,
        fixedVertexIds=fixed)
    # Preserve explicit jaw coordinates in all views to inspect errors without
    # treating fixed template bindings as true dynamic silhouette correspondences.
    sampledBefore = (geometry['vertices'][binding['vertexIndices']] * binding['barycentricWeights'][...,None]).sum(1)
    sampledAfter = (after[binding['vertexIndices']] * binding['barycentricWeights'][...,None]).sum(1)
    projections = {'before': np.stack([project(sampledBefore, c) for c in cameras]),
        'after': np.stack([project(sampledAfter, c) for c in cameras]), 'observed': targetByView, 'ids': ids}
    np.savez_compressed(output/'projections.npz', **projections)
    report = {'status': 'complete', 'source': str(args.source), 'observations': str(args.observations),
        'observationSha256': sha(args.observations), 'seconds': perf_counter()-start,
        'referenceViewIndex': 0, 'minRawPeak': args.minConfidence,
        'photoTargetPolicy':{'originalCentralAnchorIds':[7,8,9],'pchipSupportIds':supportIdsByView[0],
            'pchipSupportIdsByView':supportIdsByView,
            'pchipSupportPolicy':'only crop-valid observations above minRawPeak; central anchors 7,8,9 required',
            'interpolatedSampleIds':requestedIds.tolist(),'independentLandmarkCount':3,
            'reason':'lateral jaw transition components do not form a reliable shared visible chain in this source mesh'},
        'geometry': result['diagnostics'], 'eyesMouthSupportExactlyUnchanged': True,
        'visibleContourBefore':visibleBefore['diagnostics'],'visibleContourAfter':visibleAfter['diagnostics'],
        'sourceHashes': {name: sha(args.source/name) for name in ['cameras.json','inputs.json',
            'directClient/geometry.npz','directClient/refinement/mouthState.npz']},
        'remoteModified': False,
        'limitations': ['Automatic image landmarks are inspected proxies, not manual anatomical ground truth.',
            'Visible front/back triangle edges are selected once for fitting and extracted again for evaluation.',
            'Other views retain the front-selected edge identities only as correspondence diagnostics.']}
    writeJson(output/'report.json', report)
    print(json.dumps({'stage':'geometry','seconds':report['seconds'],'diagnostics':result['diagnostics']}), flush=True)


def bakeStage(args):
    from faceAppearance.clientShTemplateBlend import matchTemplateDetail
    start = perf_counter()
    comparison = args.output/'comparison'
    if comparison.exists() and any(comparison.iterdir()):
        raise FileExistsError('Use an empty comparison output')
    geometry, similarity, mouth, cameras, assets = context(args)
    corrected = load(args.output/'geometry/clientGeometry.npz')
    before = geometry['vertices'].copy()
    geometry['vertices'] = corrected['after']
    if not np.array_equal(before, corrected['before']):
        raise ValueError('Geometry-stage source differs')
    # Reuse the original SH correction to isolate geometry/UV reprojection.
    # A later end-to-end route would refit SH using the new normals.
    shRoot = ROOT/'runs/clientShComparison20261012'
    seamRoot = ROOT/'runs/clientShSeam20261012'
    oldSh = load(shRoot/'baking/shOnly/clientTexture.npz')
    od = load(shRoot/'baking/openDelight/clientTexture.npz')
    fallback = load(seamRoot/'templateMatchFallback.npz')
    donorMaps = {k:v for k,v in oldSh.items() if k != 'linearRGB'}
    donorMaps.update(openDelightBoundarySafeSkinDonor=od['boundarySafeSkinDonor'],
        openDelightDirectPhotoAlpha=od['boundaryDirectPhotoAlpha'])
    detail = matchTemplateDetail(fallback['linearRGB'],oldSh['linearRGB'],donorMaps,
        fallback['templateMatchSkinWeight'])
    # Use the same color-matched fallback/SDF as the source experiment, then
    # inject the detail delta through the freshly recorded template weight.
    updatedMouth = {**mouth, 'beforeCanonical': before, 'afterCanonical': geometry['vertices']}
    baked = bakeClientTexture(geometry,cameras,load(shRoot/'shImages.npz'),fallback['linearRGB'],
        args.output/'baking',colmapExecutable=ROOT/'.runtimes/colmap/bin/colmap',
        mouthState=updatedMouth,fallbackGain=None,boundaryBlend=True,smallSkinHoles=True)
    maps = {k:v for k,v in baked.items() if k not in {'linearRGB','diagnostics'}}
    templateWeight = np.clip(1-maps['boundaryDirectPhotoAlpha']-maps['boundaryEstimatedSkinAlpha'],0,1)
    templateWeight *= 1-maps.get('estimatedInteriorSkinAlpha',np.zeros_like(templateWeight))
    preserve = ((maps['coverageMask'] & ~maps['boundaryModified'])
        | (maps['coverageMask'] & maps['boundaryProtectedFeatures'])
        | (maps['frontAlpha']>0) | maps['filterPadding'] | maps['gutter'])
    templateWeight[preserve] = 0
    colors = np.clip(baked['linearRGB']+(detail['linearRGB']-fallback['linearRGB'])*templateWeight[...,None],0,1).astype(np.float32)
    maps.update(detail['maps']);maps['templateDetailContribution']=templateWeight.astype(np.float32)
    old = load(seamRoot/'templateDetail.npz')
    comparison.mkdir(parents=True)
    (comparison/'before').mkdir()
    shutil.copyfile(seamRoot/'comparison/templateDetail/legacyAsset.zip',comparison/'before/legacyAsset.zip')
    baseline = _load(comparison/'before/legacyAsset.zip')
    branches = {}
    for name,texture,provenance in [('geometryOnly',old['linearRGB'],{k:v for k,v in old.items() if k!='linearRGB'}),
                                     ('rebaked',colors,maps)]:
        assembly = assets.assembleFinalClientSurface(geometry,similarity,texture,provenance,
            eyePlacementMode='assetDefault',diagnostics={'experiment':'localChinRefinement',
                'branch':name,'cameraRefitted':False,'cachedShRetained':True})
        exportLegacyAsset(comparison/name,assembly,gender='male')
        exported = _load(comparison/name/'legacyAsset.zip')
        for key in ('faces','uvs','uvFaces','materialIds'):
            if not np.array_equal(exported.mesh[key],baseline.mesh[key]):
                raise ValueError('Client topology/UV changed: '+key)
        for key in ('body_color.png','eye_color.png','brow_color.png','head_normals.png'):
            if exported.payloads[key] != baseline.payloads[key]:
                raise ValueError('Accessory material changed: '+key)
        if name == 'geometryOnly' and exported.payloads['head_color.png'] != baseline.payloads['head_color.png']:
            raise ValueError('Frozen-color geometry branch changed head colors')
        branches[name] = {'zipSha256':sha(comparison/name/'legacyAsset.zip'),
            'vertexDeltaMaxMeters':float(np.linalg.norm(exported.mesh['vertices']-baseline.mesh['vertices'],axis=1).max())}
    np.savez_compressed(args.output/'rebaked.npz',linearRGB=colors,**maps)
    display(colors).save(args.output/'rebaked-atlas.png')
    writeJson(comparison/'config.json', {'title':'下巴与下颌修正对照','brand':'下半脸局部实验',
        'badge':'本地 · 同一照片与相机','subtitle':'保留眼嘴对齐，比较下巴修正与重新投影贴图。',
        'branches':[{'id':'before','label':'修正前 · SH 纹理适配'},
            {'id':'geometryOnly','label':'仅修几何 · 原贴图'},
            {'id':'rebaked','label':'修正几何＋重新烘焙'}],
        'defaultLeft':'before','defaultRight':'rebaked','defaultViewerMode':'baseColor','defaultRenderMode':'baseColor',
        'overviewDescription':'四张原照片与三个分支的同视角基色、共同光照渲染。',
        'notes':['眼睛与嘴部对应三角形的顶点保持不变，局部调整下巴和下颌。',
            '照片与相机保持一致；第三组将同一批 SH 去光照片重新烘焙到修正后的几何。',
            '灰模用于观察形状，基色用于检查贴图；可以叠加原照片查看轮廓。',
            '这是一组男生样本的本地实验，自动轮廓与固定模板对应仍有误差。']})
    writeJson(args.output/'bakeReport.json',{'status':'complete','seconds':perf_counter()-start,
        'geometryReport':str(args.output/'geometry/report.json'),'branches':branches,
        'cachedShRetained':True,'cameraRefitted':False,'remoteModified':False,
        'baking':baked['diagnostics'],'observedTexelsBefore':int(old['coverageMask'].sum()),
        'observedTexelsAfter':int(maps['coverageMask'].sum()),'templateDetail':detail['diagnostics']})
    print(json.dumps({'stage':'bake','status':'complete','seconds':perf_counter()-start}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('geometry','bake','all'),default='all')
    parser.add_argument('--source',type=Path,default=ROOT/'runs/clientAppearanceFix20261012/deployedSource')
    parser.add_argument('--observations',type=Path,default=ROOT/'runs/clientChinRefine20261012/observations/landmarkObservations.npz')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/clientChinRefine20261012')
    parser.add_argument('--minConfidence',type=float,default=.25)
    args=parser.parse_args()
    if args.stage in ('geometry','all'):geometryStage(args)
    if args.stage in ('bake','all'):bakeStage(args)


if __name__=='__main__':main()
