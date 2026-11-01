"""Audit frozen registration metrics, semantic proxies and render provenance.

No inference, fitting or rendering. Independent arithmetic checks saved surface
samples, normal equations, projections and original-photo replay. UFF is a
fixed reconstruction reference, never anatomical ground truth.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.clientLandmarks import loadClientLandmarks, selectClientLandmarks, ibugClientLandmarks
from faceAppearance.data import readImage
from runFastFrontEyeExperiment import paths as case_paths

CASES = ('currentMale', 'highPitch4', 'cyanWall4', 'outdoorSideLight4')


def require(ok, message):
    if not bool(ok): raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as z: return {key: z[key].copy() for key in z.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(vertices, camera):
    cv = vertices@np.asarray(camera['rotation']).T+camera['translation']
    hom = cv@np.asarray(camera['intrinsics']).T
    return hom[:, :2]/np.maximum(hom[:, 2:], 1e-9), cv[:, 2]


def sample(vertices, faces, ids, bary):
    return np.sum(np.asarray(vertices, np.float64)[faces[ids]]*bary[..., None], axis=1)


def stats(values):
    values = np.asarray(values)
    if not values.size: return {'count': 0}
    return dict(count=len(values), mean=float(values.mean()), rms=float(np.sqrt((values**2).mean())),
        median=float(np.median(values)), p90=float(np.percentile(values, 90)),
        p95=float(np.percentile(values, 95)), maximum=float(values.max()))


def equal_stats(actual, expected, label):
    require(actual.keys() == expected.keys(), label+' summary fields differ')
    for key in actual:
        require(np.isclose(actual[key], expected[key], rtol=1e-11, atol=2e-11), label+' '+key+' differs')


def topology(vertices, faces):
    edges = np.sort(np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    both = np.r_[edges, edges[:, ::-1]]
    graph = sparse.coo_matrix((np.ones(len(both)), both.T), shape=(len(vertices), len(vertices))).tocsr()
    components, _ = connected_components(graph, directed=False)
    boundary = edges[counts == 1]; ids = np.unique(boundary)
    remap = np.full(len(vertices), -1); remap[ids] = np.arange(len(ids))
    e = remap[np.r_[boundary, boundary[:, ::-1]]]
    n, labels = connected_components(sparse.coo_matrix((np.ones(len(e)), e.T), shape=(len(ids), len(ids))), directed=False)
    return dict(vertices=len(vertices), faces=len(faces), connectedComponents=components,
        boundaryEdges=len(boundary), boundaryComponentVertexCounts=sorted(int((labels == i).sum()) for i in range(n)),
        nonManifoldEdges=int((counts > 2).sum()),
        interpretation='No separate eyeball/teeth/tongue components in this UFF topology; connected inner-lid or inner-lip surface is not excluded by this count.')


def audit_metrics(root, case):
    directory = root/'metrics'/case; report = read(directory/'report.json')
    require(report['status'] == 'measuredFrozenCorrespondenceTransfer', 'Incomplete metric report')
    source = report['sourceFiles']
    for key, record in source.items(): require(sha(record['path']) == record['sha256'], 'Changed source '+key)
    asset, mapping, uff, direct_cache, stable_cache, final_cache = [arrays(source[key]['path']) for key in
        ('asset:template.npz', 'asset:directRegistration.npz', 'uff', 'direct', 'stable', 'final')]
    meshes, paired = arrays(directory/'meshes.npz'), arrays(directory/'pairedSamples.npz')
    cameras, inputs = read(source['cameras']['path']), read(source['inputs']['path'])
    require(len(cameras) == len(inputs) == 4, 'Unexpected camera/input count')
    require(np.array_equal(meshes['uffVertices'], uff['vertices']) and np.array_equal(meshes['uffFaces'], uff['faces'])
            and np.array_equal(meshes['uffUvs'], uff['uvs']), 'Native UFF changed')
    require(np.array_equal(uff['faces'], asset['sourceFaces']) and np.array_equal(uff['uvs'], asset['sourceUvs']), 'UFF topology identity differs')
    faces = asset['nicpFaces']; skin = asset['nicpToFullIds']
    for f in (meshes['clientFaces'], direct_cache['faces'], stable_cache['faces']):
        require(np.array_equal(f, faces), 'Client topology changed')
    R, t, s = direct_cache['templateToUffRotation'], direct_cache['templateToUffTranslation'], float(direct_cache['templateToUffScale'])
    direct = direct_cache['directTemplateVertices']@R*s+t
    require(np.array_equal(meshes['directVertices'], direct), 'Pure direct stage is not directTemplateVertices transformed back')
    require(np.array_equal(meshes['stableVertices'], stable_cache['vertices']), 'Stable stage differs')
    require(np.array_equal(meshes['finalVertices'], final_cache['after'][skin]), 'Final stage differs')
    require(np.allclose(final_cache['before'][skin], stable_cache['vertices'], rtol=0, atol=1e-10), 'Final eye stage starts from another stable mesh')
    mouth_candidates = [Path(source['direct']['path']).parent/'mouthState.npz',
                        Path(source['direct']['path']).parent.parent/'mouthState.npz']
    mouth_path = next((p for p in mouth_candidates if p.is_file()), None)
    if mouth_path is not None:
        mouth_before = arrays(mouth_path)['beforeCanonical']
        require(np.array_equal(direct, mouth_before), 'Extracted direct differs from pre-landmark canonical mouthState')
    targets = sample(uff['vertices'], uff['faces'], mapping['sourceTriangleIds'], mapping['sourceBarycentricWeights'])
    require(np.array_equal(targets, paired['uffTargets']), 'UFF paired targets differ')
    valid = mapping['sampleWeights'] > 0
    require(np.array_equal(valid, paired['allMask']), 'Eligible sample set changed')
    client_binding = loadClientLandmarks(faces)
    for group in ('rightEye', 'leftEye', 'mouth', 'nose'):
        chosen = selectClientLandmarks(client_binding, groups=(group,))
        region = np.zeros(len(faces), bool)
        region[np.unique(chosen['faceIds'][chosen['validMask']])] = True
        for _ in range(2): region = np.isin(faces, np.unique(faces[region])).any(1)
        require(np.array_equal(paired[group+'Mask'], region[mapping['sampleNicpFaceIds']]&valid),
                'Template region selection differs from two frozen face rings')
    require(np.array_equal(paired['eyesMask'], paired['leftEyeMask']|paired['rightEyeMask']), 'Combined eye region differs')
    require(np.array_equal(mapping['originalSampleIds'], paired['originalSampleIds']), 'Legacy sample indexing changed')
    samples = {}
    for stage in ('direct', 'stable', 'final'):
        samples[stage] = sample(meshes[stage+'Vertices'], faces, mapping['sampleNicpFaceIds'], mapping['sampleNicpWeights'])
        require(np.array_equal(samples[stage], paired[stage+'Samples']), stage+' paired samples differ')
    # Independently evaluate the frozen exact objective gradient on free vertices.
    manifest = read(source['asset:manifest.json']['path']); config = manifest['geometryTransfer']
    require(config['restWeight'] == 0 and config['staticConstraint'] == 'hard', 'Unexpected direct objective')
    N = len(direct); M = int(valid.sum()); fixed = asset['staticMask']; ids = mapping['sampleNicpFaceIds'][valid]
    bary = mapping['sampleNicpWeights'][valid]
    C = sparse.coo_matrix((bary.ravel(), (np.repeat(np.arange(M), 3), faces[ids].ravel())), shape=(M, N)).tocsr()
    edges = np.unique(np.sort(np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1), axis=0)
    e = np.r_[edges, edges[:, ::-1]]
    adjacency = sparse.coo_matrix((np.ones(len(e)), e.T), shape=(N, N)).tocsr()
    laplacian = sparse.diags(np.asarray(adjacency.sum(1)).ravel())-adjacency
    cd = config['dataWeight']/(3*(M+int(fixed.sum()))); cs = config['smoothWeight']/(3*N)
    half_gradient = cd*C.T@(C@direct-targets[valid])+cs*laplacian.T@(laplacian@direct)
    gradient_error = float(np.max(abs(half_gradient[~fixed])))
    require(gradient_error < 1e-10, 'Cached pure direct does not satisfy same UFF objective on free vertices')
    aligned = asset['vertices'][skin]@R*s+t
    require(np.array_equal(direct[fixed], aligned[fixed]), 'Hard interface not fixed to aligned rest')
    triangle = np.asarray(uff['vertices'], float)[uff['faces'][mapping['sourceTriangleIds']]]
    normal = np.cross(triangle[:, 1]-triangle[:, 0], triangle[:, 2]-triangle[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-20)
    checked = 0
    for stage, points in samples.items():
        delta = points-targets; signed = (delta*normal).sum(1); tangent = np.linalg.norm(delta-signed[:, None]*normal, axis=1)
        for region, recorded in report['residualsToUff'][stage].items():
            mask = paired[region+'Mask']; require(mask.dtype == np.bool_ and not (mask&~valid).any(), 'Invalid region mask')
            quantities = {'canonicalDistance': np.linalg.norm(delta[mask], axis=1),
                'nominalTemplateCmDistance': np.linalg.norm(delta[mask], axis=1)/s,
                'absoluteNormalOffsetCanonical': abs(signed[mask]), 'tangentDistanceCanonical': tangent[mask]}
            for name, value in quantities.items(): equal_stats(recorded[name], stats(value), stage+'/'+region+'/'+name); checked += 1
            require(np.isclose(recorded['signedUffFaceNormalOffsetMeanCanonical'], signed[mask].mean(), rtol=1e-11, atol=2e-11), 'Signed normal mean differs')
            for view in recorded['views']:
                i = view['viewIndex']; q, qd = project(targets, cameras[i]); p, pd = project(points, cameras[i])
                keep = mask&(qd > 0)&(pd > 0); dp = p[keep]-q[keep]
                transform = np.asarray(inputs[i]['workingToInput'])
                require(np.array_equal(transform[2], [0, 0, 1]), 'Original pixel displacement needs affine crop')
                for key, value in (('workingPixelDistance', np.linalg.norm(dp, axis=1)),
                                   ('originalPixelDistance', np.linalg.norm(dp@transform[:2, :2].T, axis=1))):
                    equal_stats(view[key], stats(value), stage+'/'+region+'/'+key); checked += 1
                require(np.allclose(view['meanWorkingDeltaXY'], dp.mean(0), rtol=1e-11, atol=2e-11), 'Projected signed offset differs')
    for before, after in (('direct', 'stable'), ('stable', 'final'), ('direct', 'final')):
        for region, recorded in report['pairedSampleStageDisplacements'][before+'To'+after.capitalize()].items():
            mask = paired[region+'Mask']
            equal_stats(recorded, stats(np.linalg.norm(samples[after][mask]-samples[before][mask], axis=1)), 'Stage difference'); checked += 1
    require('not occlusion-filtered' in report['fixedCameraProjectionScope'], 'Projection scope overclaims visibility')
    require(report['regions']['anatomicalGroundTruth'] is False, 'Region mask is mislabeled as truth')
    return dict(caseId=case, status='passed', allSourceHashesExact=True, rawUffAndStageArraysExact=True,
        pureDirectExtractionUsesSavedPreRefinementTemplate=True,
        additionalMouthStateCrosscheck=None if mouth_path is None else str(mouth_path),
        independentFreeHalfGradientMaximum=gradient_error,
        validPairedSampleCount=M, independentlyRecomputedSummaries=checked,
        sourceUffSha256=source['uff']['sha256'], uffTopology=topology(uff['vertices'], uff['faces']))


def audit_renders(root, case):
    directory = root/'renders'/case; report = read(directory/'renderReport.json')
    paths = case_paths(case); meshes = root/'metrics'/case/'meshes.npz'
    require(report['sourceMeshesSha256'] == sha(meshes), 'Renderer used different stage meshes')
    require(report['camerasSha256'] == sha(paths['source']/'cameras.json')
            and report['inputsSha256'] == sha(paths['source']/'inputs.json'), 'Renderer used different cameras/crops')
    require(report['cameraRefitted'] is False and report['geometryFitted'] is False and report['texturesUsed'] is False,
            'Renderer changed diagnostic controls')
    inputs, cameras = read(paths['source']/'inputs.json'), read(paths['source']/'cameras.json')
    require(len(report['views']) == 4 and len(report['renderRecords']) == 16, 'Missing view or branch')
    for view in report['views']:
        i = view['id']; record = inputs[i]; photo = paths['photos']/f'{i:02d}.image'
        require(sha(photo) == record['sha256'], 'Original input bytes differ')
        pixels, replay = readImage(photo, imageSize=record['workingSize'][0], cropPolicy=record['requestedCropPolicy'],
            cropMargin=record['cropMargin'], faceDetector=lambda _, saved=record: saved['faceDetection'])
        for key in ('inputToWorking', 'orientedSize', 'workingSize', 'cropBoxXyxy', 'paddingLeftTopRightBottom'):
            require(replay[key] == record[key], 'Photo replay '+key+' differs')
        expected = np.rint(pixels*255).astype(np.uint8)
        require(np.array_equal(expected, np.asarray(Image.open(directory/view['photo']))), 'Rendered source photograph differs')
        require(set(view['branches']) == {'uff', 'direct', 'stable', 'final'}, 'Missing branch')
        light = np.asarray(cameras[i]['rotation']).T@np.array([-.5, -.5, -1.])
        for branch, file in view['branches'].items():
            rgba = np.asarray(Image.open(directory/file)); rows = [x for x in report['renderRecords'] if x['view'] == i and x['branch'] == branch]
            require(len(rows) == 1, 'Ambiguous render record'); row = rows[0]
            require(rgba.shape == (*expected.shape[:2], 4), 'Rendered image size differs')
            require(np.isin(rgba[..., 3], [0, 255]).all() and int((rgba[..., 3] > 0).sum()) == row['visiblePixels'], 'Rendered visible mask differs')
            require(np.array_equal(rgba[..., 0], rgba[..., 1]) and np.array_equal(rgba[..., 0], rgba[..., 2]), 'Clay render contains nonuniform material color')
            require(np.allclose(row['lightDirectionCanonical'], light, rtol=0, atol=1e-12), 'Branch lighting differs')
            require(row['unresolvedPixels'] == 0 and row['alphaRejectedCandidates'] == 0 and row['maximumSelectedDepthLayer'] == 0, 'Unexpected accessory/alpha handling')
        for box in view['crops'].values():
            require(0 <= box[0] < box[2] <= expected.shape[1] and 0 <= box[1] < box[3] <= expected.shape[0], 'Invalid display crop')
    return dict(caseId=case, status='passed', cameraAndInputsHashesExact=True, originalPhotoReplayPixelsExact=True,
        allFourStagesAllFourViewsPresent=True, uniformGrayMaterials=True, sameInspectionLightingPerView=True,
        fixedCameraNoRefit=True, skinOnlyNoClientAccessories=True)


def arc_samples(poly):
    edge = np.roll(poly, -1, axis=0)-poly; length = np.linalg.norm(edge, axis=1)
    cumulative = np.r_[0., np.cumsum(length)]; total = cumulative[-1]
    q = np.arange(max(16, int(np.ceil(total/.25))))*total/max(16, int(np.ceil(total/.25)))
    index = np.minimum(np.searchsorted(cumulative, q, side='right')-1, len(poly)-1)
    return poly[index]+edge[index]*((q-cumulative[index])/np.maximum(length[index], 1e-12))[:, None]


def point_curve_distance(points, polygon):
    edge = np.roll(polygon, -1, axis=0)-polygon
    offset = points[:, None]-polygon
    factor = np.clip((offset*edge).sum(2)/np.maximum((edge*edge).sum(1), 1e-20), 0, 1)
    return np.sqrt(((offset-factor[..., None]*edge)**2).sum(2)).min(1)


def polygon_area_center(poly):
    nxt = np.roll(poly, -1, axis=0); cross = poly[:, 0]*nxt[:, 1]-poly[:, 1]*nxt[:, 0]
    signed = cross.sum()/2
    return (abs(float(signed)), ((poly+nxt)*cross[:, None]).sum(0)/(6*signed)) if abs(signed) >= 1e-8 else (0., poly.mean(0))


def audit_semantics(root, case):
    directory = root/'semantics'/case; report = read(directory/'report.json')
    require(report['status'] == 'completeReadOnlySemanticAudit' and not report['cameraRefit']
            and not report['geometryFit'] and not report['detectorRerun'], 'Semantic audit changed controls')
    for name, wanted in report['hashes'].items(): require(sha(ROOT/name) == wanted, 'Changed semantic source '+name)
    metrics = read(root/'metrics'/case/'report.json'); m = arrays(root/'metrics'/case/'meshes.npz')
    require(sha(report['sourcePaths']['uff']) == metrics['sourceFiles']['uff']['sha256'], 'Semantics used another UFF')
    source = Path(report['sourcePaths']['reconstruction']); obs_dir = Path(report['sourcePaths']['photoObservations'])
    camera = read(source/'cameras.json')[0]; inputs = read(source/'inputs.json')
    observation = arrays(obs_dir/'view00.npz'); curves = arrays(directory/'projectedCurves.npz')
    metadata = read(obs_dir/'view00.json'); identity = read(obs_dir/'manifest.json')
    require(identity['camerasSha256'] == sha(source/'cameras.json') and identity['inputsSha256'] == sha(source/'inputs.json'), 'Semantic coordinate identity differs')
    require(metadata['inputSha256'] == inputs[0]['sha256'] and np.array_equal(observation['inputToWorking'], inputs[0]['inputToWorking']), 'Photo semantics use another photograph/crop')
    c = arrays(metrics['sourceFiles']['direct']['path'])
    stages = {'uffFrozen': (m['uffVertices'], m['uffFaces']), 'clientDirect': (m['directVertices'], m['clientFaces']),
        'clientPostLandmarks': (c['vertices'], m['clientFaces']), 'clientAfterJaw': (m['stableVertices'], m['clientFaces']),
        'clientFinalEyes': (m['finalVertices'], m['clientFaces'])}
    require(set(stages) == set(report['eyeSkinRimProxy']), 'Semantic stage missing')
    inv = np.asarray(inputs[0]['workingToInput']); scale = float(np.linalg.svd(inv[:2, :2], compute_uv=False).mean())
    require(np.isclose(scale, report['originalPixelsPerWorkingPixel'], rtol=0, atol=1e-12), 'Original pixel units differ')
    widths = np.linalg.norm(observation['cornersWorking'][:, 1]-observation['cornersWorking'][:, 0], axis=1)
    visibility = []
    for stage, (vertices, faces) in stages.items():
        edges, counts = np.unique(np.sort(np.r_[faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1), axis=0, return_counts=True)
        boundary = {tuple(x) for x in edges[counts == 1]}
        means = []
        for i, eye in enumerate(report['eyeSkinRimProxy'][stage]):
            ids = np.asarray(eye['boundaryVertexIds']); model, depth = project(vertices[ids], camera)
            require((depth > 0).all(), 'Semantic eye rim behind camera')
            require(all(tuple(sorted((int(a), int(b)))) in boundary for a, b in zip(ids, np.roll(ids, -1))), 'Semantic curve is not an original boundary cycle')
            require(np.allclose(model, eye['projectedCurve'], rtol=0, atol=1e-11)
                    and np.allclose(model, curves[f'{stage}_eye{i}'], rtol=0, atol=1e-11), 'Semantic projection differs from saved stage')
            target = np.r_[observation['eyelidCurvesWorking'][i, 1], observation['eyelidCurvesWorking'][i, 0, ::-1]]
            require(np.array_equal(target, eye['photoCurve']), 'Stage changed frozen photo target')
            require(eye['photoCandidateAccepted'] == bool(observation['candidateEyeValidMask'][i]), 'Photo confidence gate mislabeled')
            d1, d2 = point_curve_distance(arc_samples(model), target), point_curve_distance(arc_samples(target), model)
            mean = (d1.mean()+d2.mean())/2; means.append(mean)
            values = {'symmetricCurveMeanWorkingPixels': mean, 'symmetricCurveMeanOriginalPixels': mean*scale,
                'symmetricCurveMeanEyeWidths': mean/widths[i], 'modelToPhotoMeanWorkingPixels': d1.mean(),
                'photoToModelMeanWorkingPixels': d2.mean(), 'bidirectionalP95WorkingPixels': np.percentile(np.r_[d1, d2], 95)}
            area, center = polygon_area_center(model); target_area, target_center = polygon_area_center(target)
            axis = target[len(target)//2-1]-target[0]; axis /= np.linalg.norm(axis); normal = [-axis[1], axis[0]]
            height, target_height = np.ptp(model@normal), np.ptp(target@normal)
            values.update(polygonAreaWorkingPixelsSquared=area, photoPolygonAreaWorkingPixelsSquared=target_area,
                analyticPolygonAreaOverPhotoEye=area/target_area, polygonCenterWorkingPixels=center,
                photoPolygonCenterWorkingPixels=target_center, centerOffsetWorkingPixels=center-target_center,
                centerErrorWorkingPixels=np.linalg.norm(center-target_center), photoAxisHeightWorkingPixels=height,
                photoAxisHeightOriginalPixels=height*scale, photoAxisTargetHeightWorkingPixels=target_height,
                photoAxisHeightOverPhoto=height/target_height)
            for key, value in values.items(): require(np.allclose(eye['metrics'][key], value, rtol=1e-10, atol=1e-10), 'Semantic metric '+key+' differs')
            visibility.append(eye['skinOnlyVisibility']['visibleSkinBoundaryFraction'])
        require(np.isclose(report['meanCurveDistanceByStage'][stage], np.mean(means), rtol=1e-10, atol=1e-10), 'Stage aggregate semantic score differs')
    require(report['sharedMouthNoseSemanticComparison']['available'] is False, 'Unsupported common UFF mouth/nose semantics claimed')
    return dict(caseId=case, status='passed', sourceHashesExact=True, sameFrozenPhotoTargets=True,
        allOwnTopologyBoundaryCyclesVerified=True, actualSavedStageProjectionsExact=True,
        curveDistancesPhotoFixedAxisHeightsAreasCentersIndependentlyRecomputed=True,
        semanticLimits='Own-topology structural skin-hole rims, not the final visible eye aperture or anatomical ground truth; later photo fitting is in-sample.',
        minimumReportedSkinRimVisibility=min(visibility), maximumReportedSkinRimVisibility=max(visibility),
        sharedUffMouthNoseComparisonCorrectlyUnavailable=True)


def mask_geometry(mask):
    y, x = np.nonzero(mask); width = int(x.max()-x.min()+1); columns = mask.sum(0); columns = columns[columns > 0]
    return dict(areaWorkingPixelsSquared=int(len(x)), boundingBoxXyxyInclusive=[int(x.min()), int(y.min()), int(x.max()), int(y.max())],
        centroidWorkingPixels=[float(x.mean()), float(y.mean())], boundingWidthWorkingPixels=width,
        boundingHeightWorkingPixels=int(y.max()-y.min()+1), meanVerticalThicknessWorkingPixels=len(x)/width,
        maximumVerticalThicknessWorkingPixels=int(columns.max()), medianNonemptyColumnThicknessWorkingPixels=float(np.median(columns)),
        meanVerticalThicknessToWidth=len(x)/width**2)


def audit_apertures(root, case):
    import cv2
    cv2.setNumThreads(1)
    directory = root/'apertures'/case; report = read(directory/'report.json'); masks = arrays(directory/'masks.npz')
    renders = root/'renders'/case; rr = read(renders/'renderReport.json')
    require(report['sourceRenderReportSha256'] == sha(renders/'renderReport.json')
            and report['sourceObservationSha256'] == sha(report['sourceObservation']), 'Aperture source hash differs')
    observed = arrays(report['sourceObservation']); inputs = read(case_paths(case)['source']/'inputs.json')
    require(report['photoSha256'] == inputs[0]['sha256'] and report['connectivity'] == 8
            and report['alphaBackgroundThreshold'] == 128, 'Aperture frame or component rule differs')
    scale = float(np.asarray(inputs[0]['inputToWorking'])[0, 0])
    require(np.isclose(scale, report['originalToWorkingScale'], rtol=0, atol=1e-12), 'Aperture original pixel scale differs')
    polys = [np.r_[observed['eyelidCurvesWorking'][i, 1], observed['eyelidCurvesWorking'][i, 0, ::-1]] for i in range(2)]
    targets = []
    for i, name in enumerate(('imageLeft', 'imageRight')):
        expected = np.zeros(tuple(inputs[0]['workingSize'][::-1]), np.uint8)
        cv2.fillPoly(expected, [np.rint(polys[i]*256).astype(np.int32)], 1, shift=8)
        expected = expected.astype(bool); targets.append(expected)
        require(np.array_equal(expected, masks[name+'_photo']), 'Aperture photo polygon raster differs')
        require(report['referenceGeometry'][name] == mask_geometry(expected), 'Photo mask measurements differ')
    ratios = {}; accepted_count = 0
    for stage, row in report['branches'].items():
        image_path = renders/f'view0-{stage}.png'; rgba = np.asarray(Image.open(image_path))
        require(row['sourceRenderSha256'] == sha(image_path), 'Aperture uses another render')
        background = (rgba[..., 3] < 128).astype(np.uint8)
        number, labels, cc_stats, centers = cv2.connectedComponentsWithStats(background, connectivity=8)
        exterior = set(np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        components = [(labels == ident, mask_geometry(labels == ident)) for ident in range(1, number) if ident not in exterior]
        require(len(components) == len(row['internalComponents']), 'An internal background component was omitted')
        require(sorted((x[1]['areaWorkingPixelsSquared'], tuple(x[1]['boundingBoxXyxyInclusive'])) for x in components)
                == sorted((x['areaWorkingPixelsSquared'], tuple(x['boundingBoxXyxyInclusive'])) for x in row['internalComponents']),
                'Internal component geometry differs')
        chosen_masks = []
        for i, name in enumerate(('imageLeft', 'imageRight')):
            eye = row['eyes'][name]; target = polys[i]; other = polys[1-i]
            lo, hi = target.min(0), target.max(0); center = (lo+hi)/2
            other_center = (other.min(0)+other.max(0))/2; width = hi[0]-lo[0]
            plausible = []
            for mask, geometry in components:
                box = geometry['boundingBoxXyxyInclusive']; c = np.asarray(geometry['centroidWorkingPixels'])
                distance = np.linalg.norm(c-center)
                overlap = max(0., min(box[2]+1, hi[0])-max(box[0], lo[0]))/width
                if (geometry['areaWorkingPixelsSquared'] >= 4 and .25*width <= geometry['boundingWidthWorkingPixels'] <= 1.6*width
                    and geometry['boundingHeightWorkingPixels'] <= .8*width and distance <= .65*width and overlap >= .3
                    and distance < np.linalg.norm(c-other_center) and box[0] >= lo[0]-.6*width and box[2] <= hi[0]+.6*width
                    and box[1] >= center[1]-.75*width and box[3] <= center[1]+.75*width): plausible.append(mask)
            if not observed['candidateEyeValidMask'][i] or len(plausible) != 1:
                require(eye['selection']['status'] == 'rejected' and stage+'_'+name not in masks, 'Ambiguous aperture was forced to a result')
                continue
            require(eye['selection']['status'] == 'acceptedUniqueInternalComponent', 'Reliable aperture unexpectedly rejected')
            mask = masks[stage+'_'+name]
            require(np.array_equal(mask, plausible[0]), 'Selected aperture is not exactly one enclosed background component')
            require(not any(np.any(mask&prior) for prior in chosen_masks), 'Same component assigned to both eyes')
            chosen_masks.append(mask); accepted_count += 1
            geometry = mask_geometry(mask)
            for key, value in geometry.items(): require(np.allclose(eye['geometry'][key], value, rtol=0, atol=1e-12), 'Aperture geometry '+key+' differs')
            require(np.isclose(eye['geometry']['areaOriginalPixelsSquared'], mask.sum()/scale**2), 'Aperture area original-unit conversion differs')
            border = mask&~binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
            photo = targets[i]; photo_border = photo&~binary_erosion(photo, structure=np.ones((3, 3)), border_value=0)
            a, b = distance_transform_edt(~photo_border)[border], distance_transform_edt(~border)[photo_border]
            expected = {'intersectionOverUnion': (mask&photo).sum()/(mask|photo).sum(), 'photoAreaRatio': mask.sum()/photo.sum(),
                'symmetricBoundaryMeanWorkingPixels': (a.mean()+b.mean())/2,
                'symmetricBoundaryMeanOriginalPixels': (a.mean()+b.mean())/(2*scale),
                'symmetricBoundaryP95WorkingPixels': np.percentile(np.r_[a, b], 95),
                'symmetricBoundaryMaximumWorkingPixels': np.r_[a, b].max()}
            for key, value in expected.items(): require(np.isclose(eye['photoComparison'][key], value, rtol=1e-11, atol=1e-11), 'Aperture distance '+key+' differs')
            equal_stats(eye['photoComparison']['modelToPhotoBoundaryDistanceWorkingPixels'], stats(a), 'Aperture model-to-photo')
            equal_stats(eye['photoComparison']['photoToModelBoundaryDistanceWorkingPixels'], stats(b), 'Aperture photo-to-model')
    for name, transitions in report['stageRatios'].items():
        for transition, values in transitions.items():
            before, after = transition.split('To'); after = after.lower()
            a, b = report['branches'][before]['eyes'][name]['geometry'], report['branches'][after]['eyes'][name]['geometry']
            for key, value in values.items(): require(np.isclose(value, b[key]/a[key], rtol=0, atol=1e-12), 'Aperture stage ratio differs')
        ratios[name] = transitions.get('uffToDirect')
    return dict(caseId=case, status='passed', independentOpenCvComponentExtraction=True,
        originalRgbaAndPhotoHashesExact=True, exactEnclosedAlphaComponentsNoFillingOrJoining=True,
        uniquePlausibleEyeAssignmentVerified=True, acceptedEyeCount=accepted_count,
        allAreaWidthThicknessAndBoundaryDistancesRecomputed=True, uffToDirectRatios=ratios,
        scope='Visible openings in skin-only frozen RGBA; not the final eye with eyeballs/corners/alpha cards, and not a unique causal attribution to the solver.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'runs/registrationTransfer20261013')
    args = parser.parse_args()
    rows = [{'metrics': audit_metrics(args.root, case), 'renders': audit_renders(args.root, case),
             'semantics': audit_semantics(args.root, case), 'visibleSkinApertures': audit_apertures(args.root, case)} for case in CASES]
    result = dict(status='passedIndependentRegistrationTransferAudit', cases=rows,
        currentMaleSourceIdentity=read(args.root/'metrics/currentMale/source/sourceIdentity.json'),
        semanticAudit='Passed frozen sources, own-boundary projections, curve distances, photo-axis height/area/center arithmetic',
        scope='Frozen UFF transfer fidelity and controlled visualization; not calibrated 3-D truth or proof of anatomical mapping.',
        caveats=['Original camera is fitted to UFF; equal-camera rendering isolates transfer under that estimate, not camera-estimation error.',
                 'Paired projection residuals include positive-depth hidden samples; they are not observed photograph landmark errors.',
                 'UFF and client skin differ in topology and interior rim construction; skin holes are not assembled visible eye apertures.',
                 'Nominal template centimetres are not calibrated physical person measurements.'])
    writeJson(args.root/'audit.json', result)
    print(json.dumps({'status': result['status'], 'cases': list(CASES)}, ensure_ascii=False))


if __name__ == '__main__': main()
