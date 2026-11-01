"""Inspect fixed legacy eyelid semantics independently of rendered eye edges.

Read the unchanged seven completed baselines and four saved cameras. This tool
does not fit, rebind, infer landmarks, change geometry or create client assets.
Lines joining semantic samples illustrate order, not a traced mesh-edge path.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.camera import projectPoints
from faceAppearance.semanticEyelidBinding import buildSemanticEyelidBinding, sampleSemanticEyelids
from faceAppearance.visibleEyeContours import _raster
from runAccessoryFollowingExperiment import RUN, OLD, arrays, read, readEvidence, sha, verify, plan
from runProtectedFeatureExperiment import saveEvidence
from renderNoseRepairComparison import scaledViewCamera

OUT = ROOT/'reports/semanticEyelid20261024'
EYES = ('subjectRight', 'subjectLeft')
COLORS = dict(upper='#5ed9ff', lower='#ffb35c', visible='#ef8cd2', observed='#83e2a7')
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def font(size):
    return ImageFont.truetype(FONT, size)


def nearestPolyline(points, curve):
    first, delta = curve[:-1], np.diff(curve, axis=0)
    squared = (delta*delta).sum(1)
    usable = squared > 1e-20
    if not usable.any():
        raise ValueError('Collapsed diagnostic polyline')
    parameter = np.clip(((points[:, None]-first)*delta).sum(2)/np.maximum(squared, 1e-20), 0, 1)
    closest = first+parameter[..., None]*delta
    distance = np.linalg.norm(points[:, None]-closest, axis=2)
    distance[:, ~usable] = np.inf
    chosen = distance.argmin(1)
    arc = np.r_[0., np.cumsum(np.sqrt(squared))]
    return dict(distancePixels=distance[np.arange(len(points)), chosen],
        closestPixels=closest[np.arange(len(points)), chosen],
        targetArcFractions=(arc[chosen]+parameter[np.arange(len(points)), chosen]*np.sqrt(squared[chosen]))/arc[-1])


def raster(scene, camera, roi):
    return _raster(scene['vertices'], scene['faces'], scene['materialIds'], camera, roi,
        scene['uvs'], scene['uvFaces'], scene['materials'], scene['alphaTextures'])


def exactVisibility(scene, camera, points, pixels, depths, unit):
    """Use the same alpha/backface semantics at each exact subpixel query."""
    tolerance = max(float(np.ptp(scene['vertices'], axis=0).max())*1e-8, 1e-10)
    rows = []
    for point, pixel, depth in zip(points, pixels, depths):
        if depth <= 0 or (pixel < 0).any() or (pixel > 511).any():
            rows.append(dict(status='outsidePhotoOrBehindCamera', frontFaceId=None, gapMm=None))
            continue
        hit, _, z, _ = raster(scene, camera, dict(origin=pixel, scale=1., shape=(1, 1)))
        gap = float(depth-z[0, 0])
        if hit[0, 0] < 0:
            status = 'noFrontSurfaceHit'
        elif gap > tolerance:
            status = 'occluded'
        elif gap < -tolerance:
            status = 'queryInFrontOfRasterSurface'
        else:
            status = 'frontSurface'
        rows.append(dict(status=status, frontFaceId=int(hit[0, 0]),
            frontMaterialId=int(scene['materialIds'][hit[0, 0]]) if hit[0, 0] >= 0 else None,
            gapMm=gap/unit if np.isfinite(gap) else None))
    return dict(rows=rows, toleranceCanonical=tolerance,
        method='Exact query pixel alpha-aware front-facing raster; no nearest pixel rounding',
        gapDefinition='Camera-Z depth difference in millimeters; not shortest surface or penetration distance',
        visibleCount=sum(row['status'] == 'frontSurface' for row in rows))


def observations(fan, dense, eyeIndex, isReference):
    start = 36+eyeIndex*6
    upper = fan[start+np.array([0, 1, 2, 3])]
    lower = fan[start+np.array([0, 5, 4, 3])]
    result = dict(upper=upper, lower=lower)
    if isReference:
        result = dict(upper=dense[eyeIndex, 1], lower=dense[eyeIndex, 0])
    # Match explicit subject-side semantics; never sort using candidate pixels.
    return {key: value[::-1].copy() if eyeIndex == 0 else value.copy() for key, value in result.items()}


def drawLine(draw, coords, color, width=2):
    if len(coords) > 1:
        draw.line([tuple(point) for point in coords], fill=color, width=width)


def drawPanel(base, crop, row, kind, bindingEye):
    width, height = 340, 240
    piece = base.crop(crop)
    factor = min(width/piece.width, height/piece.height)
    size = tuple(np.maximum(1, np.rint(np.array(piece.size)*factor).astype(int)))
    resized = piece.resize(size, Image.Resampling.BICUBIC)
    panel = Image.new('RGB', (width, height), '#172029')
    offset = (np.array([width, height])-size)//2
    panel.paste(resized, tuple(offset))
    draw = ImageDraw.Draw(panel)
    scale = np.array(size)/np.array(piece.size)
    def xy(points):
        return (np.asarray(points)-np.array(crop[:2]))*scale+offset
    if kind == 'observed':
        for curve in row['observations'].values():
            drawLine(draw, xy(curve), COLORS['observed'])
    if kind in ('semantic', 'model'):
        if kind == 'model':
            for contour in row['visibleContours']:
                drawLine(draw, xy(np.r_[contour, contour[:1]]), COLORS['visible'], 1)
        for lid in ('upper', 'lower'):
            coords = xy(row['projected'][lid])
            drawLine(draw, coords, COLORS[lid], 1)
            for index, ((x, y), status, mpId) in enumerate(zip(coords,
                    row['visibility'][lid]['rows'], bindingEye[lid]['mpIds'])):
                if status['status'] == 'frontSurface':
                    draw.ellipse((x-3, y-3, x+3, y+3), fill=COLORS[lid])
                else:
                    draw.line((x-3, y-3, x+3, y+3), fill=COLORS[lid], width=2)
                    draw.line((x-3, y+3, x+3, y-3), fill=COLORS[lid], width=2)
                if lid == 'upper' and index in (0, len(coords)-1):
                    draw.text((x+4, y-18 if lid == 'upper' else y+4), str(mpId), font=font(11), fill='#ffffff')
    return panel


def drawGrid(case, view, reference, photo, clay, rows, binding, destination):
    picture = Image.new('RGB', (1360, 650), '#172029')
    draw = ImageDraw.Draw(picture)
    draw.text((12, 9), f'{case} · 视角 {view+1} · 固定语义点与可见轮廓分开核验', font=font(23), fill='white')
    draw.text((12, 45), '青：上睑语义候选  橙：下睑语义候选  粉：可见眼裂  绿：照片自动观测  ×：遮挡或无可见表面支持', font=font(16), fill='#c4d5df')
    titles = ['原照片', '照片观测 · '+('稠密 STFR' if view == reference else '稀疏 FAN68'),
        '固定 MP 语义候选投影', '原灰模 · 语义候选＋可见眼裂']
    for eyeIndex, eye in enumerate(EYES):
        row = rows[eye]
        sources = [*row['projected'].values(), *row['observations'].values(), *row['visibleContours']]
        points = np.concatenate([value for value in sources if len(value)])
        low, high = points.min(0), points.max(0)
        margin = max(8., .14*float(np.ptp(points, axis=0).max()))
        crop = tuple(np.r_[np.floor(low-margin), np.ceil(high+margin)].astype(int))
        row['displayCrop'] = crop
        for column, (title, kind) in enumerate(zip(titles, ('plain', 'observed', 'semantic', 'model'))):
            y = 85+eyeIndex*272
            if kind == 'observed' and not row['observationUsableForDiagnostics']:
                title = '观测未通过检查 · 仅显示记录'
            draw.text((column*340+8, y), ('右眼' if eyeIndex == 0 else '左眼')+' · '+title, font=font(15), fill='#e4ecf0')
            picture.paste(drawPanel(clay if kind == 'model' else photo, crop, row, kind, binding['eyes'][eye]),
                (column*340, y+28))
    draw.text((12, 628), '连接线仅表示点的语义顺序，并非已标注的网格睑缘路径。所有几何、相机、观测与原图保持不变。', font=font(16), fill='#bccdd8')
    picture.save(destination)


def runCase(case):
    config = plan()
    if case not in config['cases']:
        raise ValueError('Case is outside the frozen seven-case list')
    contextPath = RUN/case/'inputContext.json'
    context = read(contextPath)
    verify(context['sourceHashes'])
    entry = context['entry']
    prepared = readEvidence(RUN/case/'preparation')
    scene = prepared['scene']
    # JSON evidence serializes material IDs as keys; raster lookup uses integers.
    scene['alphaTextures'] = {int(key): value for key, value in scene['alphaTextures'].items()}
    raw = arrays(entry['preparedGeometry'])
    binding = buildSemanticEyelidBinding(raw['faces'])
    sampled = sampleSemanticEyelids(scene['vertices'][raw['skinIds']], binding)
    cameraPath, metaPath = [Path(entry['reconstruction'])/name for name in ('cameras.json', 'inputs.json')]
    cameras, metadata = read(cameraPath), read(metaPath)
    if len(cameras) != 4 or len(metadata) != 4:
        raise ValueError('Expected all four saved views')
    densePath = OLD/case/'observations/observations.npz'
    fanPath = OLD/case/'geometry/photoObservations.npz'
    dense, fan = arrays(densePath), arrays(fanPath)
    alignment = readEvidence(RUN/case/'observationAlignment')
    reference = context['referenceSelection']['referenceViewIndex']
    componentsPath = ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1/components.json'
    components = read(componentsPath)
    faceGroups = {}
    for eye, label, cornerName in (('subjectRight', 'rightEye', 'male00SG.component001'),
                                  ('subjectLeft', 'leftEye', 'male00SG.component002')):
        faceGroups[eye] = np.concatenate([np.asarray(part['faceIndices']) for part in components
            if part['label'] == label or part['name'] == cornerName])
    renderRoot = ROOT/'reports/accessoryFollowing20261024'/case
    renderReport = read(renderRoot/'report.json')
    verify(renderReport['inputHashes'])
    paths = [contextPath, cameraPath, metaPath, densePath, fanPath, componentsPath,
        RUN/case/'preparation.json', RUN/case/'preparation.npz', RUN/case/'observationAlignment.json',
        RUN/case/'observationAlignment.npz', renderRoot/'report.json', Path(binding['sourcePath'])]
    hashes = {str(path): sha(path) for path in paths}
    directory = OUT/case
    directory.mkdir(parents=True, exist_ok=False)
    saveEvidence(directory/'semanticBinding', binding)
    saveEvidence(directory/'semanticPositions', sampled)
    views = []
    unit = float(raw['unit'])
    for view in range(4):
        photoPath = Path(entry['workingPhotos'])/f'input{view:02d}.png'
        clayPath = renderRoot/f'view{view:02d}_baseline_clay.png'
        if sha(clayPath) != renderReport['images'][clayPath.name]:
            raise ValueError('Cached baseline rendering changed')
        hashes.update({str(path): sha(path) for path in (photoPath, clayPath)})
        photo, clay = Image.open(photoPath).convert('RGB'), Image.open(clayPath).convert('RGB')
        if photo.size != (512, 512) or clay.size != (512, 512):
            raise ValueError('Expected original saved 512px working images')
        camera = scaledViewCamera(cameras[view], metadata[view], view, 512)
        for key in camera:
            np.testing.assert_array_equal(camera[key], renderReport['views'][view]['camera'][key])
        hits, _, _, rasterStats = raster(scene, camera, dict(origin=[0., 0.], scale=1., shape=(512, 512)))
        rows = {}
        for eyeIndex, eye in enumerate(EYES):
            target = observations(fan['points'][view], dense['eyelidCurvesWorking'], eyeIndex, view == reference)
            eyeConfidence = fan['confidence'][view, 36+eyeIndex*6:42+eyeIndex*6]
            completeFANConfidence = bool(np.isfinite(eyeConfidence).all() and (eyeConfidence >= .5).all())
            observationUsable = (bool(alignment['valid']) if view == reference
                else bool(fan['accepted'][view]) and completeFANConfidence)
            observationUsable &= all(np.isfinite(curve).all() for curve in target.values())
            mask = (hits >= 0) & np.isin(hits, faceGroups[eye])
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [contour[:, 0].astype(float) for contour in contours]
            projected, visibility, distances = {}, {}, {}
            for lid in ('upper', 'lower'):
                pixels, depths = projectPoints(sampled[eye][lid], camera['intrinsics'], camera['rotation'], camera['translation'])
                projected[lid] = pixels
                visibility[lid] = exactVisibility(scene, camera, sampled[eye][lid], pixels, depths, unit)
                if observationUsable:
                    comparison = nearestPolyline(pixels, target[lid])
                    distances[lid] = dict(**comparison,
                        interiorMeanPixels=float(comparison['distancePixels'][1:-1].mean()),
                        cornerDistancesPixels=np.linalg.norm(pixels[[0, -1]]-target[lid][[0, -1]], axis=1),
                        nearestParameterReversalCount=int((np.diff(comparison['targetArcFractions']) < -1e-10).sum()))
            row = dict(projected=projected, observations=target, visibleContours=contours, visibility=visibility,
                distancesToSameLidObservation=distances,
                observationSource='cachedSTFR' if view == reference else 'cachedFAN68',
                observationUsableForDiagnostics=observationUsable,
                fanViewDetected=bool(fan['accepted'][view]), fanEyeConfidence=eyeConfidence,
                fanEyeConfidenceMask=eyeConfidence >= .5,
                completeFANPolylineConfidenceAccepted=completeFANConfidence,
                confidencePolicy='For this diagnostic only, a FAN polyline is compared when all six eye points have confidence >= .5; this is not the fitting acceptance gate.',
                observationInvalidReason=alignment['error'] if view == reference else
                    (None if observationUsable else 'Incomplete FAN eye confidence or detection'),
                semanticReviewStatus=binding['reviewStatus'], visibleEyePixelCount=int(mask.sum()),
                role='Read-only discrepancy; not an accuracy score, fitting target or visibility-based point selection')
            # Invalid observations are displayed as recorded but not graded.
            if not observationUsable:
                row['distancesToSameLidObservation'] = None
            rows[eye] = row
        drawGrid(case, view, reference, photo, clay, rows, binding, directory/f'view{view:02d}.png')
        views.append(dict(viewIndex=view, camera=camera, eyes=rows, raster=rasterStats))
        print(f'{case}: semantic diagnostics view {view}', flush=True)
    verify(hashes)
    writeJson(directory/'report.json', dict(caseId=case, referenceViewIndex=reference, views=views,
        status='completeReadOnlyDiagnostic', sourceHashes=hashes,
        bindingSha256=binding['bindingSha256'], semanticReviewStatus=binding['reviewStatus'],
        geometryChanged=False, camerasChanged=False, newInference=False, candidateSolves=0,
        images={path.name:sha(path) for path in sorted(directory.glob('*.png'))}))


def finalize():
    cases = plan()['cases']
    reports = {case:read(OUT/case/'report.json') for case in cases}
    referenceSummary = {}
    uniqueStatusCounts = {}
    diagnosticEyeCount = 0
    for case, report in reports.items():
        verify(report['sourceHashes'])
        verify({str(OUT/case/name):value for name,value in report['images'].items()})
        if len(report['views']) != 4 or len(report['images']) != 4 or report['candidateSolves'] != 0:
            raise ValueError('Incomplete four-view diagnostic')
        binding = readEvidence(OUT/case/'semanticBinding')
        for view in report['views']:
            perEye = {}
            for eye, row in view['eyes'].items():
                uniquePoints = {}
                for lid in ('upper', 'lower'):
                    for mpId, point in zip(binding['eyes'][eye][lid]['mpIds'], row['visibility'][lid]['rows']):
                        key = int(mpId)
                        if key in uniquePoints and uniquePoints[key] != point:
                            raise ValueError('Shared corner visibility changed between paths')
                        uniquePoints[key] = point
                if len(uniquePoints) != 16:
                    raise ValueError('Expected exactly sixteen unique semantic points per eye')
                counts = {}
                for point in uniquePoints.values():
                    status = point['status']
                    counts[status] = counts.get(status, 0)+1
                    uniqueStatusCounts[status] = uniqueStatusCounts.get(status, 0)+1
                diagnosticEyeCount += int(row['observationUsableForDiagnostics'])
                perEye[eye] = dict(uniquePointStatusCounts=counts,
                    observationUsableForDiagnostics=row['observationUsableForDiagnostics'],
                    meanInteriorDiscrepancyPixels={lid:value['interiorMeanPixels']
                        for lid,value in (row['distancesToSameLidObservation'] or {}).items()})
            if view['viewIndex'] == report['referenceViewIndex']:
                referenceSummary[case] = perEye
    templatePath = OUT/'templateAudit.json'
    template = read(templatePath)
    verify(template['sourceHashes'])
    verify({template['originalBindingSource']:template['originalBindingSha256']})
    summary = dict(status='completeReadOnlyDiagnostic', cases=cases, caseCount=7, viewCount=28,
        geometryChanged=False, newInference=False, candidateSolves=0, manualTruthEstablished=False,
        semanticReviewStatus='unreviewedLegacySemanticCandidate',
        uniquePointCountPerView=32, uniquePointViewStatusCounts=uniqueStatusCounts,
        diagnosticEyeViewCount=diagnosticEyeCount, totalEyeViewCount=56,
        referenceSummary=referenceSummary,
        originalTemplateFrontOccludedPointCount=sum(eye['exactRayOccludedPointCount']
            for eye in template['views']['front']['eyes'].values()),
        reports={case:dict(path=f'{case}/report.json',sha256=sha(OUT/case/'report.json'),
            referenceViewIndex=report['referenceViewIndex']) for case,report in reports.items()},
        templateAudit=dict(path='templateAudit.json',sha256=sha(templatePath)) if templatePath.exists() else None,
        implementationHashes={str(path):sha(path) for path in (Path(__file__),
            ROOT/'src/faceAppearance/semanticEyelidBinding.py',ROOT/'src/faceAppearance/clientLandmarks.py',
            ROOT/'src/faceAppearance/visibleEyeContours.py')})
    writeJson(OUT/'summary.json',summary)
    writeIndex(cases)
    writeTemplateIndex()


def writeTemplateIndex():
    figures = ''.join(f'<h3>{label}</h3><img src="template_{name}.png" alt="{label}原模板核验图">'
        for name,label in [('front','正面'),('leftOblique','左斜侧'),('rightOblique','右斜侧'),('profile','侧面')])
    (OUT/'template.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>原模板睑缘核验</title>
<style>body{background:#172029;color:#e6edf2;font:17px sans-serif;margin:24px}p{max-width:1200px;line-height:1.8}a{color:#84cfff}img{max-width:100%;display:block}h3{margin-top:35px}</style>
<h2>原模板中，部分旧眼部语义点位于眼睑折返面</h2>
<p>原始男性模板正面有 12／32 个旧 MP 语义点被主皮肤遮挡：右眼 8 个、左眼 4 个。右上睑全部 7 个内部点的绑定面朝后，射线深度差约 0.91–4.09 mm。二维位置接近可见边界，并不能保证它们位于同一层表面。</p>
<p>这说明旧语义点不能直接当作可见睑缘。被遮挡本身不等于标错；眼角与眼球材质边界也可能合理地不同。后续需要一次模板级人工核验，确认上下睑与内外眼角，再进行分角色、保序的照片拟合。</p>
<p>图例：绿色实心点＝可见旧 MP 点，红色空心点＝被遮挡的旧 MP 点，红线＝渲染眼球材质边界，蓝色＝目标眼球，紫色填充＝另一侧眼球。紫色虚线为后方拓扑孔环；语义点连线只是投影顺序，均未当作解剖真值。</p>
<p>此模板检查使用厘米坐标、正交投影、双面不透明皮肤／眼球，并排除透明睫毛卡片；七组样本页使用原透视相机及原背面剔除／透明度规则。二者可见点数不能直接相减作为改善量。</p>
<p><a href="index.html?case=outdoorSideLight4&view=0">返回七组四视角</a> · <a href="eyelidTemplateReview.zip">下载一次性模板标注包</a> · <a href="reviewPackage/README.md">标注说明</a> · <a href="templateAudit.json">原模板审计数据</a></p>'''+figures+'</html>',encoding='utf-8')


def writeIndex(cases):
    writeJson(OUT/'viewer.json',dict(cases=cases))
    (OUT/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>语义睑缘独立核验</title>
<style>body{background:#172029;color:#e6edf2;font:16px sans-serif;margin:22px}p{line-height:1.7;max-width:1200px}a{color:#84cfff}select{background:#293642;color:white;padding:7px;font:inherit;margin:8px}img{display:block;max-width:100%;margin:16px 0}</style>
<h2>固定语义睑缘 · 七组四视角核验</h2>
<p>青色为旧客户端上睑语义候选，橙色为下睑，粉色为模型可见眼裂，绿色为照片自动检测。圆点可见，× 表示被遮挡或无可见表面支持。内外眼角保留 MP 编号。连接线只展示语义顺序，不是已确认的网格睑缘路径。</p>
<p>这轮没有修改模型或重新拟合。参考视角显示原稠密 STFR 观测，其余视角显示原 FAN68 稀疏观测；它们不是人工真值。旧语义绑定仍需模板级核验。</p>
<p><a href="template.html">查看原始模板的正面与侧面检查</a> · <a href="summary.json">完整诊断数据</a> · <a href="http://127.0.0.1:18102/">上一轮随动实验</a></p>
<label>样本<select id="case"></select></label><label>视角<select id="view"><option value="0">1</option><option value="1">2</option><option value="2">3</option><option value="3">4</option></select></label><p id="status"></p><img id="comparison" alt="照片、观测、固定语义投影和模型可见轮廓的四列对照">
<script>const $=id=>document.getElementById(id),q=new URLSearchParams(location.search);fetch('viewer.json').then(r=>r.json()).then(d=>{for(const c of d.cases)$('case').add(new Option(c,c));for(const k of ['case','view'])if(q.has(k)&&Array.from($(k).options).some(o=>o.value===q.get(k)))$(k).value=q.get(k);async function show(){const c=$('case').value,v=$('view').value;history.replaceState(null,'','?'+new URLSearchParams({case:c,view:v}));$('comparison').src=c+'/view'+v.padStart(2,'0')+'.png';const r=await(await fetch(c+'/report.json')).json(),e=r.views[Number(v)].eyes;const invalid=Object.values(e).some(x=>!x.observationUsableForDiagnostics);$('status').textContent='原模型保持不变；'+(invalid?'本视角含未通过曲线或置信度检查的眼部观测；保留原始记录，对该眼不计算对应差异。':'本视角用于对应差异检查，尚不代表解剖正确性。')}for(const k of ['case','view'])$(k).onchange=show;show()});</script></html>''',encoding='utf-8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case')
    parser.add_argument('--finalize',action='store_true')
    args=parser.parse_args()
    if args.case:runCase(args.case)
    elif args.finalize:finalize()
    else:parser.error('Choose --case or --finalize')
