"""Local, front-photo dense-eyelid experiment on an actual delivered client ZIP.

The input photographs, camera, texture bytes and accessories remain fixed.
Dense detector curves are observations, not independent anatomical ground truth.
Only the rendered visible eye/skin boundary supplies skin surface bindings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.clientGeometrySafety import intersectionPairs
from faceAppearance.clientLandmarks import loadClientLandmarks, ibugClientLandmarks
from faceAppearance.frontLandmarkRefinement import refineFrontLandmarks, checkFrontGeometry
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.legacyAsset import exportLegacyAsset, LEGACY_FILES
from faceAppearance.legacyPreviewAsset import _load
from buildDeployedJawComparison import retainDeliveredPayloads


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def project(points, camera):
    cameraPoints = points@np.asarray(camera['rotation']).T + camera['translation']
    homogeneous = cameraPoints@np.asarray(camera['intrinsics']).T
    return homogeneous[:, :2]/homogeneous[:, 2:]


def densePolyline(points, samples=256):
    points = np.asarray(points, float)
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if lengths[-1] <= 1e-8:
        raise ValueError('Degenerate contour')
    positions = np.linspace(0, lengths[-1], samples)
    return np.column_stack([np.interp(positions, lengths, points[:, axis]) for axis in range(2)])


def curveMetrics(binding, photoEyes, originalScale):
    records = {}
    for name, target in photoEyes.items():
        eye = binding['eyes'][name]
        if eye['status'] != 'accepted':
            records[name] = {'status': eye['status'], 'diagnostics': eye.get('diagnostics'), 'reasons':eye.get('reasons')}
            continue
        errors = []
        for part in ('upper', 'lower'):
            current = densePolyline(eye[part]['sourcePixels'])
            wanted = densePolyline(target[part])
            symmetric = np.r_[cKDTree(current).query(wanted)[0], cKDTree(wanted).query(current)[0]]
            errors.append(symmetric)
        corners = .5*(eye['upper']['sourcePixels'][[0,-1]]+eye['lower']['sourcePixels'][[0,-1]])
        wantedCorners = .5*(target['upper'][[0,-1]]+target['lower'][[0,-1]])
        currentAngle = np.arctan2(*(corners[1]-corners[0])[::-1])
        targetAngle = np.arctan2(*(wantedCorners[1]-wantedCorners[0])[::-1])
        angleError = abs((np.degrees(currentAngle-targetAngle)+180)%360-180)
        records[name] = {
            'status': 'accepted',
            'meanContourDistanceOriginalPixels': float(np.concatenate(errors).mean()/originalScale),
            'p95ContourDistanceOriginalPixels': float(np.quantile(np.concatenate(errors), .95)/originalScale),
            'upperMeanDistanceOriginalPixels': float(errors[0].mean()/originalScale),
            'lowerMeanDistanceOriginalPixels': float(errors[1].mean()/originalScale),
            'cornerErrorsOriginalPixels': (np.linalg.norm(corners-wantedCorners, axis=1)/originalScale).tolist(),
            'cornerDirectionErrorDegrees': float(angleError),
            'scope': 'Freshly rendered visible boundary against fixed automatic photo curves; not manual ground truth',
        }
    accepted = [row['meanContourDistanceOriginalPixels'] for row in records.values() if row['status']=='accepted']
    return {'eyes': records, 'meanContourDistanceOriginalPixels': float(np.mean(accepted)) if len(accepted)==2 else None}


def compactBinding(binding):
    result = {}
    for key, value in binding.items():
        if key in ('roiMasks', 'raster', 'rasterization'):
            continue
        if isinstance(value, dict):
            result[key] = compactBinding(value)
        elif isinstance(value, np.ndarray) and value.ndim > 2:
            continue
        else:
            result[key] = value
    return result


def drawContours(path, photo, photoEyes, baseline, after):
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 18)
    small = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 15)
    tileW, tileH, scale = 430, 255, 5.
    canvas = Image.new('RGB', (tileW*3, tileH*2+75), '#151d25')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), '正面眼睑 · 黄线：原照片检测目标；青线：重新渲染的实际可见眼裂', fill='white', font=font)
    for row, name in enumerate(('imageLeft', 'imageRight')):
        target = photoEyes[name]
        points = np.concatenate([target['upper'], target['lower']])
        center = points.mean(0)
        crop = (center[0]-tileW/(2*scale), center[1]-(tileH-40)/(2*scale),
                center[0]+tileW/(2*scale), center[1]+(tileH-40)/(2*scale))
        for col, (label, bound) in enumerate((('照片检测线', None), ('当前服务 · 可见轮廓', baseline), ('密集眼睑拟合 · 可见轮廓', after))):
            tile = photo.crop(crop).resize((tileW, tileH-40), Image.Resampling.LANCZOS)
            td = ImageDraw.Draw(tile)
            def xy(values):
                return [tuple(p) for p in (np.asarray(values)-crop[:2])*scale]
            for part in ('upper', 'lower'):
                td.line(xy(target[part]), fill='#ffcf43', width=2)
                for x,y in xy(target[part]):
                    td.ellipse((x-2,y-2,x+2,y+2), fill='#ffcf43')
                if bound is not None and bound['eyes'][name]['status']=='accepted':
                    td.line(xy(bound['eyes'][name][part]['sourcePixels']), fill='#1ae6f0', width=2)
            x,y = col*tileW, 45+row*tileH
            draw.text((x+10,y+8), ('图像左眼' if row==0 else '图像右眼')+' · '+label, font=small, fill='white')
            canvas.paste(tile,(x,y+40))
    draw.text((12, canvas.height-26), '局部放大用于看轮廓；插值放大不增加照片信息。检测目标不是人工真值。', font=small, fill='#b9c8d1')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--include-inner-corner-accessories', action='store_true',
                        help='Treat the audited two inner-eye head accessories as part of the visible aperture')
    args = parser.parse_args()
    from faceAppearance.visibleEyeContours import bindVisibleEyeContours, extractVisibleEyeContours, loadLegacyEyeBindingInputs
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    implementation = {name:(ROOT/name).read_bytes() for name in ('tools/runClientDenseEyeExperiment.py',
        'src/faceAppearance/visibleEyeContours.py','src/faceAppearance/frontLandmarkRefinement.py',
        'src/faceAppearance/eyeRefinement.py')}
    for name,data in implementation.items():
        target = output/'implementation'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
    start = perf_counter()
    inputs = json.loads((source/'inputs.json').read_text())
    savedCameras = json.loads((source/'cameras.json').read_text())
    camera = {key: np.asarray(savedCameras[0][key], float) for key in ('intrinsics','rotation','translation')}
    if not savedCameras[0]['valid']:
        raise ValueError('Front camera is invalid')
    assets = AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    arrays = assets.arrays
    apertureOptions = {}
    if args.include_inner_corner_accessories:
        components = [item for item in assets.components if item['label']=='headAccessory']
        if len(components)!=2:
            raise ValueError('The audited male template must have two inner-corner head accessories')
        apertureOptions = {'apertureAccessoryMode':'innerCornerHeadAccessories',
            'apertureAccessoryFaceIds':np.concatenate([item['faceIndices'] for item in components]).astype(np.int64)}
    original = load(source/'directClient/geometry.npz')
    before = original['vertices']
    jaw = load(source/'directClient/renderedJaw/geometry.npz')
    transform = json.loads((source/'directClient/renderedJaw/templateToUff.json').read_text())
    faces = original['faces']
    if not np.array_equal(before, jaw['after']) or not np.array_equal(faces, arrays['nicpFaces']):
        raise ValueError('Source final geometry or topology mismatch')
    observation = load(args.observations/'view00.npz')
    metadata = json.loads((args.observations/'view00.json').read_text())
    manifest = json.loads((args.observations/'manifest.json').read_text())
    # The cache supplies recorded input identity and validates actual photo bytes.
    if (manifest.get('status') != 'complete' or manifest['inputsSha256'] != sha(source/'inputs.json')
            or manifest['camerasSha256'] != sha(source/'cameras.json')
            or manifest['views'][0]['arraySha256'] != sha(args.observations/'view00.npz')
            or manifest['views'][0]['metadataSha256'] != sha(args.observations/'view00.json')
            or metadata['inputSha256'] != inputs[0]['sha256']):
        raise ValueError('Observation cache differs from this reconstruction')
    if not observation['candidateEyeValidMask'].all():
        raise ValueError('Front dense photo eye curves failed the frozen observation gates')
    photoEyes = {name:{'upper':observation['eyelidCurvesWorking'][i,1],
                      'lower':observation['eyelidCurvesWorking'][i,0]}
                for i,name in enumerate(('imageLeft','imageRight'))}
    scene = loadLegacyEyeBindingInputs(args.archive, arrays['nicpToFullIds'])
    shape = tuple(reversed(inputs[0]['workingSize']))
    def bind(skin, *, measurementOnly=False):
        full = scene['vertices'].copy()
        full[arrays['nicpToFullIds']] = skin
        operation = extractVisibleEyeContours if measurementOnly else bindVisibleEyeContours
        return operation(full, scene['faces'], scene['materialIds'], arrays['nicpToFullIds'],
            camera, photoEyes, imageShape=shape, uvs=scene['uvs'], uvFaces=scene['uvFaces'],
            materials=scene['materials'], alphaTextures=scene['alphaTextures'], **apertureOptions)
    if not np.allclose(scene['vertices'][arrays['nicpToFullIds']], before, rtol=0, atol=1e-12):
        raise ValueError('Delivered skin and experiment source differ')
    print('Extracting baseline perspective-visible eye contours', flush=True)
    baseline = bind(before)
    writeJson(output/'baselineBinding.json', compactBinding(baseline))
    constraints = []
    for eyeIndex, name in enumerate(('imageLeft','imageRight')):
        eye = baseline['eyes'][name]
        if eye['status'] != 'accepted':
            raise ValueError(f'Baseline visible eye rejected: {name}: {eye.get("diagnostics")}')
        for part, partIndex in (('lower',0),('upper',1)):
            item = eye[part]
            valid = item['acceptedMask'] & observation['candidatePointValidMask'][eyeIndex,partIndex]
            ids = np.flatnonzero(valid)
            if len(ids)<8 or not valid[[0,-1]].all():
                raise ValueError('Insufficient complete upper/lower visible eye support')
            pairs = np.array([[i,i+1] for i in range(len(ids)-1) if ids[i+1]==ids[i]+1], np.int64).reshape(-1,2)
            weights = np.ones(12); weights[[0,-1]] = 4.
            constraints.append({'name':name+'_'+part,
                'vertexIndices':item['vertexIndices'][valid],
                'barycentric':item['barycentricWeights'][valid],
                'targetPixels':photoEyes[name][part][valid]-item['pixelOffsets'][valid],
                'weights':weights[valid],
                'confidence':observation['curveConfidence'][eyeIndex,partIndex,valid],
                'segmentPairs':pairs})
    mouth = ibugClientLandmarks(loadClientLandmarks(faces), groups=('mouth',))
    fixed = np.unique(np.r_[np.flatnonzero(arrays['staticMask']), mouth['vertexIndices'].ravel(),
                            np.flatnonzero(np.any(jaw['before']!=jaw['after'], axis=1))])
    extent = float(np.ptp(before,axis=0).max())
    originalScale = float(np.asarray(inputs[0]['inputToWorking'])[0,0])
    baselineMeasurement = bind(before, measurementOnly=True)
    baselineMetrics = curveMetrics(baselineMeasurement, photoEyes, originalScale)
    writeJson(output/'constraints.json', constraints)
    print('Fitting four ordered dense eyelid curves in the fixed front image plane', flush=True)
    fitted = refineFrontLandmarks(before, faces, camera, constraints, fixedVertexIds=fixed,
        supportRadius=extent*.12, controlCount=40, maxDisplacement=extent*.03,
        minConfidence=.2, maxTargetDistancePixels=40., iterations=args.iterations,
        maxRelativeEdgeChange=.25, minAreaRatio=.4, minNormalCosine=.3, curveWeight=.25)
    writeJson(output/'solver.json', fitted['diagnostics'])
    intersectionEpsilon = max(float(np.linalg.norm(np.ptp(before,axis=0)))*1e-9,1e-12)
    initialPairs = intersectionPairs(before,faces,epsilon=intersectionEpsilon)
    attempts = []; selected = before.copy(); selectedBinding = baselineMeasurement; selectedScale = 0.
    selectedMetrics = baselineMetrics
    for scale in (1., .5, .25, .125, .0625):
        candidate = before+scale*(fitted['vertices']-before)
        geometry = checkFrontGeometry(before, candidate, faces, camera=camera, fixedVertexIds=fixed,
            maxDisplacement=extent*.03, maxRelativeEdgeChange=.25, minAreaRatio=.4, minNormalCosine=.3)
        record = {'scale':scale,'geometry':geometry}
        attempts.append(record)
        if not geometry['accepted']:
            continue
        added = intersectionPairs(candidate,faces,epsilon=intersectionEpsilon)-initialPairs
        record['newIntersectionPairs'] = len(added)
        if added:
            continue
        rendered = bind(candidate, measurementOnly=True)
        metrics = curveMetrics(rendered, photoEyes, originalScale)
        record['visibleContourMetrics'] = metrics
        if metrics['meanContourDistanceOriginalPixels'] is None:
            continue
        # Check each lid and corner as well as the mean, accounting for the
        # supersampled raster's finite spatial precision (all values original px).
        componentSafe = True
        for name in photoEyes:
            initial, current = baselineMetrics['eyes'][name], metrics['eyes'][name]
            for key,tolerance in (('meanContourDistanceOriginalPixels',.1),
                    ('upperMeanDistanceOriginalPixels',.1),('lowerMeanDistanceOriginalPixels',.1),
                    ('p95ContourDistanceOriginalPixels',.25)):
                componentSafe &= current[key] <= initial[key]+tolerance
            componentSafe &= bool(np.all(np.asarray(current['cornerErrorsOriginalPixels'])
                                        <= np.asarray(initial['cornerErrorsOriginalPixels'])+.5))
        minimumImprovement = .25/originalScale
        accepted = componentSafe and metrics['meanContourDistanceOriginalPixels'] < baselineMetrics['meanContourDistanceOriginalPixels']-minimumImprovement
        record.update(componentGate=bool(componentSafe), minimumImprovementOriginalPixels=minimumImprovement)
        record['accepted'] = accepted
        if accepted:
            selected, selectedBinding, selectedMetrics, selectedScale = candidate, rendered, metrics, scale
            break
    np.savez_compressed(output/'geometry.npz', before=before, after=selected, faces=faces,
                        fixedVertexIds=fixed, proposed=fitted['vertices'])
    writeJson(output/'afterBinding.json', compactBinding(selectedBinding))
    writeJson(output/'attempts.json', attempts)
    finalGeometryCheck = checkFrontGeometry(before,selected,faces,camera=camera,fixedVertexIds=fixed,
        maxDisplacement=extent*.03,maxRelativeEdgeChange=.25,minAreaRatio=.4,minNormalCosine=.3)
    finalPairs = intersectionPairs(selected,faces,epsilon=intersectionEpsilon)
    depthDelta = float(np.abs((selected-before)@camera['rotation'][2]).max())
    if not finalGeometryCheck['accepted'] or finalPairs-initialPairs:
        raise RuntimeError('Final selected geometry failed cumulative checks')
    delivered = _load(args.archive)
    textures = load(source/'directClient/texture.npz'); linear = textures.pop('linearAlbedo')
    assembly = assets.assembleFinalClientSurface({**original,'vertices':selected}, transform, linear, textures,
        accessoryReferenceCanonical=jaw['before'],
        diagnostics={'comparison':'Front visible dense eyelid experiment', 'textureRebaked':False,
                     'cameraRefitted':False, 'denseEyelidCorrectionScale':selectedScale})
    nonSkin = np.ones(len(arrays['vertices']),bool); nonSkin[arrays['nicpToFullIds']]=False
    if not np.allclose(assembly['geometry']['vertices'][nonSkin]*.01, delivered.mesh['vertices'][nonSkin], rtol=0,atol=1e-13):
        raise ValueError('Accessory placement changed before export')
    (output/'before').mkdir(); shutil.copyfile(args.archive,output/'before/legacyAsset.zip')
    exported = exportLegacyAsset(output/'after',assembly,gender='male')
    retainDeliveredPayloads(output/'after',delivered,exported,nonSkin)
    actual = _load(output/'after/legacyAsset.zip')
    exportedCanonical = (actual.mesh['vertices'][arrays['nicpToFullIds']]*100)@np.asarray(transform['rotation'])*transform['scale']+transform['translation']
    exportError = float(np.abs(exportedCanonical-selected).max())
    if exportError>1e-12:
        raise ValueError('Exported skin differs from the accepted candidate')
    if not np.array_equal(actual.mesh['vertices'][nonSkin], delivered.mesh['vertices'][nonSkin]):
        raise ValueError('Exported accessories changed')
    for key in ('faces','uvFaces','materialIds'):
        if not np.array_equal(actual.mesh[key],delivered.mesh[key]):
            raise ValueError('Export changed fixed '+key)
    if not np.allclose(actual.mesh['uvs'],delivered.mesh['uvs'],rtol=0,atol=1e-15):
        raise ValueError('Export changed UVs')
    if any(actual.payloads[name]!=delivered.payloads[name] for name in LEGACY_FILES if name.endswith('.png')):
        raise ValueError('Export changed texture payload')
    config = {'title':'正面密集眼睑 · 局部几何实验','brand':'本地眼形对照','badge':'同一任务 · 眼睑轮廓',
        'subtitle':'左侧为已上线结果，右侧为密集眼睑局部拟合。照片、相机、贴图和眼球附件相同。',
        'branches':[{'id':'before','label':'当前服务结果'},{'id':'after','label':'密集眼睑拟合'}],
        'defaultLeft':'before','defaultRight':'after','defaultViewerMode':'baseColor','defaultRenderMode':'baseColor',
        'notes':['上、下眼睑各12个检测点，只以正面照片拟合；其他视角仅供观察。',
                 '绑定来自完整模型同相机下的可见眼/皮肤交界；新轮廓重新渲染后评估。',
                 '贴图保持原始字节，尚未重烘焙；眼球、虹膜朝向和附件位置保持相同。',
                 '自动检测轮廓不是人工真值，本实验尚未部署。']}
    writeJson(output/'config.json',config)
    photo = Image.open(args.observations/'view00WorkingPhoto.png').convert('RGB')
    drawContours(output/'contours.png',photo,photoEyes,baselineMeasurement,selectedBinding)
    report = {'status':'acceptedLocalExperiment' if selectedScale else 'noImprovingSafeCandidate',
        'apertureIncludesAuditedInnerCornerAccessories':args.include_inner_corner_accessories,
        'selectedScale':selectedScale,'changedVertexCount':int(np.any(selected!=before,axis=1).sum()),
        'seconds':perf_counter()-start,'before':baselineMetrics,'after':selectedMetrics,
        'cameraRefitted':False,'textureRebaked':False,'textureBytesExactlyEqual':True,
        'accessoryVerticesExactlyEqual':True,'fixedVerticesExactlyUnchanged':bool(np.array_equal(selected[fixed],before[fixed])),
        'protectedJawVerticesExactlyUnchanged':bool(np.array_equal(selected[np.any(jaw['before']!=jaw['after'],axis=1)], before[np.any(jaw['before']!=jaw['after'],axis=1)])),
        'topologyAndUvUnchanged':True,'maximumFrontCameraDepthChange':depthDelta,
        'finalGeometryCheck':finalGeometryCheck,'baselineIntersectionPairCount':len(initialPairs),
        'finalIntersectionPairCount':len(finalPairs),'newIntersectionPairCount':len(finalPairs-initialPairs),
        'intersectionEpsilon':intersectionEpsilon,'exportMaximumCanonicalError':exportError,
        'implementationSha256':{name:hashlib.sha256(data).hexdigest() for name,data in implementation.items()},
        'sourceArchiveSha256':sha(args.archive),'observationManifestSha256':sha(args.observations/'manifest.json'),
        'scope':'One male sample. Front dense automatic photo targets; other views for visual inspection only; no production changes'}
    writeJson(output/'report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
