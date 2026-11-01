"""Three-arm unified-observation inspection using identical template materials.

Render only frozen outputs of runEyeObservationExperiment. No fit, texture
bake, geometry export or candidate promotion takes place in this tool.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.clientLandmarks import loadClientLandmarks, ibugClientLandmarks
from renderClientAppearanceEvidence import EvidenceRenderer, evidenceGrid, solid
from renderFeedbackAppearanceComparison import saveSrgb
from renderNoseRepairComparison import scaledViewCamera
from renderShapePreservationComparison import baselinePatchCrop
from runEyeObservationExperiment import RUN, OLD, PRIOR, plan, read, sha, arrays, verify

OUT = ROOT/'reports/eyeObservation20261024'
BRANCHES = ('baseline', 'boundaryReplay', 'unifiedObservationContact')
LABELS = {'baseline': '完成粗对齐的基线', 'boundaryReplay': '原观测约束（重放）',
    'unifiedObservationContact': '统一眼睑观测＋接触'}
SIZE = 512
MODES = ('clay', 'baseColor', 'lit')


def renderCase(caseId):
    """Four saved cameras, three modes, fixed baseline-derived eye crops."""
    from pytorch3d.structures import Meshes

    config = plan()
    if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', caseId) is None or caseId not in config['cases']:
        raise ValueError('Case must belong to the frozen seven-case plan')
    if tuple(config['branches']) != BRANCHES:
        raise ValueError('Expected the frozen baseline/boundaryReplay/unifiedObservationContact order')
    codePaths = [Path(__file__), ROOT/'tools/runEyeObservationExperiment.py',
        ROOT/'tools/renderClientAppearanceEvidence.py', ROOT/'tools/renderClientAssembly.py',
        ROOT/'tools/renderFeedbackAppearanceComparison.py', ROOT/'tools/renderNoseRepairComparison.py',
        ROOT/'tools/renderShapePreservationComparison.py']
    codeHashes = {str(path): sha(path) for path in codePaths}
    contextPath = RUN/caseId/'inputContext.json'
    context = read(contextPath)
    verify(context['sourceHashes'])
    entry = context['entry']
    if entry['caseId'] != caseId:
        raise ValueError('Context case identity mismatch')
    reportPaths = {branch: RUN/caseId/branch/'report.json' for branch in BRANCHES}
    reports = {branch: read(path) for branch, path in reportPaths.items()}
    geometryPaths = {branch: RUN/caseId/branch/'fullGeometry.npz' for branch in BRANCHES}
    archivePaths = {branch: Path(reports[branch]['asset']['archivePath']) for branch in BRANCHES}
    cameraPath = Path(entry['reconstruction'])/'cameras.json'
    metadataPath = Path(entry['reconstruction'])/'inputs.json'
    photos = [Path(entry['workingPhotos'])/f'input{view:02d}.png' for view in range(4)]
    frozenInputs = [RUN/'experimentPlan.json', contextPath, Path(entry['preparedGeometry']), cameraPath, metadataPath,
        *reportPaths.values(), *geometryPaths.values(), *archivePaths.values(), *photos]
    inputHashes = {str(path): sha(path) for path in frozenInputs}
    raw = arrays(entry['preparedGeometry'])
    transform = {key: raw[key].tolist() for key in ('rotation', 'translation', 'scale')}
    cameras, metadata = read(cameraPath), read(metadataPath)
    if len(cameras) != 4 or len(metadata) != 4:
        raise ValueError('This frozen experiment requires all four saved cameras/photos')
    for branch in BRANCHES:
        if reports[branch].get('caseId') != caseId:
            raise ValueError(f'Branch report case identity mismatch: {branch}')
        if not isinstance(reports[branch]['geometrySafetyAccepted'], bool):
            raise ValueError('Branch report must contain an explicit geometrySafetyAccepted boolean')

    renderers = {branch: EvidenceRenderer(archivePaths[branch], transform, device='cpu', facesPerPixel=12)
        for branch in BRANCHES}
    renderers['baseline'].torch.set_num_threads(1)
    baselineAsset = renderers['baseline'].source
    expected = {branch: arrays(path)['vertices'] for branch, path in geometryPaths.items()}
    nonSkinIds = np.setdiff1d(np.arange(len(expected['baseline'])), raw['skinIds'])
    staticFullIds = np.asarray(raw['skinIds'])[raw['staticIds']]
    for branch, renderer in renderers.items():
        source = renderer.source
        for key in ('faces', 'uvs', 'uvFaces', 'materialIds'):
            np.testing.assert_array_equal(source.mesh[key], baselineAsset.mesh[key])
        if source.manifest['binding'] != baselineAsset.manifest['binding']:
            raise ValueError('Binding changed across arms')
        if set(source.payloads) != set(baselineAsset.payloads):
            raise ValueError('Branch payload membership changed')
        for name, payload in baselineAsset.payloads.items():
            if name != 'face_nicp.obj' and source.payloads[name] != payload:
                raise ValueError(f'Template materials/other payloads changed: {branch}/{name}')
        canonical = source.mesh['vertices']*100 @ np.asarray(transform['rotation'])*transform['scale']+transform['translation']
        np.testing.assert_allclose(canonical, expected[branch], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(expected[branch][nonSkinIds], expected['baseline'][nonSkinIds])
        np.testing.assert_array_equal(expected[branch][staticFullIds], expected['baseline'][staticFullIds])
        # Use exactly the NPZ state checked by the gates, then recreate Meshes so
        # face/vertex normals cannot remain cached for a previous geometry.
        renderer.vertices = renderer.torch.as_tensor(expected[branch], dtype=renderer.torch.float32)
        renderer.mesh = Meshes(verts=[renderer.vertices], faces=[renderer.faces])

    output = OUT/caseId
    output.mkdir(parents=True, exist_ok=False)
    binding = ibugClientLandmarks(loadClientLandmarks(raw['faces']), groups=('eyes',))
    eyeIds = np.unique(binding['vertexIndices'][binding['validMask']])
    views = []
    gateLabels = {branch: ('观测映射无效 · 未求解' if branch == 'unifiedObservationContact' and not reports[branch]['observationAlignmentValid']
        else '原验收门通过' if reports[branch]['geometrySafetyAccepted'] else '原门未通过/继承问题')
        for branch in BRANCHES}
    observationStates = {branch: {
        'observationAlignmentValid': reports[branch]['observationAlignmentValid'],
        'observationAlignmentError': reports[branch]['observationAlignmentError'],
        'observationPolicyChangedForDiagnostic': reports[branch]['observationPolicyChangedForDiagnostic'],
        'originalObservationPolicyAccepted': reports[branch]['originalObservationPolicyAccepted'],
        'activeObservationPolicyAccepted': reports[branch]['activeObservationPolicyAccepted'],
        'completeApertureEffectAccepted': reports[branch]['finalEffectGate']['accepted'],
        'improvementAccepted': reports[branch]['improvementAccepted'],
        'geometryChanged': reports[branch]['geometryChanged'],
    } for branch in BRANCHES}
    for view, information in enumerate(metadata):
        camera = scaledViewCamera(cameras[view], information, view, SIZE)
        with Image.open(photos[view]) as opened:
            photo = opened.convert('RGB').resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        # All arms use the original pureDirect template patch, never candidate
        # selection, hand-picked photo landmarks or branch-dependent crops.
        crop, cropReport = baselinePatchCrop(raw['pureDirect'], eyeIds, camera, SIZE)
        rendered, stats = {}, {}
        for branch, renderer in renderers.items():
            rendered[branch], stats[branch] = renderer.variants(camera, (SIZE, SIZE))
        for mode in MODES:
            images = [photo]+[solid(rendered[branch][mode]) for branch in BRANCHES]
            labels = ['原工作照片']+[LABELS[branch]+' · '+gateLabels[branch] for branch in BRANCHES]
            title = (f'{caseId} · 视角 {view+1} · {mode} · 统一眼睑观测诊断 / 保留原验收结论 · '
                '同模板贴图未重烘焙，不评价眉鼻纹理；固定保存相机')
            picture = evidenceGrid(images, labels, columns=4, panelSize=SIZE, subtitle=title)
            saveSrgb(picture, output/f'view{view:02d}_{mode}.png')
            eyes = evidenceGrid([image.crop(crop) for image in images], labels, columns=4,
                panelSize=256, subtitle=f'{caseId} · 视角 {view+1} · {mode} · 同一基线区域放大 / 局部几何诊断')
            saveSrgb(eyes, output/f'view{view:02d}_{mode}_eyes.png')
            for branch in BRANCHES:
                saveSrgb(solid(rendered[branch][mode]), output/f'view{view:02d}_{branch}_{mode}.png')
        views.append({'viewIndex': view, 'camera': camera, 'stats': stats, 'eyeCrop': crop, 'eyeCropReport': cropReport})
        print(f'{caseId}: rendered all three branches at saved view {view}', flush=True)

    verify(context['sourceHashes']); verify(inputHashes); verify(codeHashes)
    images = {path.name: sha(path) for path in sorted(output.glob('*.png'))}
    if len(images) != 60:
        raise RuntimeError('Expected exactly 4 views x 3 modes x (full grid + eye grid + 3 arms) = 60 PNG files')
    writeJson(output/'report.json', {'caseId': caseId, 'branches': list(BRANCHES), 'views': views,
        'geometrySafetyAccepted': {branch: reports[branch]['geometrySafetyAccepted'] for branch in BRANCHES},
        'observationStates': observationStates,
        'originalObservationGatesRemainAudit': True,
        'candidateObservationPolicyChanged': reports['unifiedObservationContact']['observationPolicyChangedForDiagnostic'],
        'gateLabels': gateLabels, 'gateReportPaths': {branch: str(path) for branch, path in reportPaths.items()},
        'identicalMaterials': True, 'allFourCameras': True, 'templateColorsOnly': True,
        'textureRebaked': False, 'noGeometryPromotion': True, 'localEyeExperimentOnly': True,
        'inheritedGateFailuresRetained': True, 'mouthActivelyRefined': False,
        'nonSkinAndStaticExactAcrossArms': True, 'vertexOrderUvBindingUnchanged': True,
        'cropDerivedFromBaselinePatch': True, 'renderSize': SIZE, 'device': 'cpu',
        'photoUsedForNewFit': False, 'inputHashes': inputHashes,
        'upstreamSourceHashes': context['sourceHashes'], 'renderImplementationHashes': codeHashes,
        'allFrozenInputsUnchanged': True, 'images': images,
        'interpretation': 'Unified reference-eye observation comparison against exact completed baseline and previous boundary-field replay. Original observation gates are still independently audited; passing an alternative observation policy cannot establish legacy eligibility. Identical template maps; inherited failures remain. Geometry gates do not establish aperture benefit, photographic likeness, or complete-avatar delivery.'})


def writeIndex():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = plan()['cases']
    complete = [case for case in cases if (OUT/case/'report.json').is_file()]
    states = {case: read(OUT/case/'report.json')['geometrySafetyAccepted'] for case in complete}
    observationStates = {case: read(OUT/case/'report.json')['observationStates'] for case in complete}
    writeJson(OUT/'viewer.json', {'cases': cases, 'complete': complete, 'labels': LABELS,
        'branches': list(BRANCHES), 'geometrySafetyAccepted': states, 'observationStates': observationStates})
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>统一眼睑观测与原约束对照</title>
<style>body{background:#171c23;color:#e8eef6;font:16px sans-serif;margin:20px}select{font:inherit;padding:6px;background:#283240;color:white;border:1px solid #61738a;margin:4px}img{display:block;max-width:100%;margin:16px 0}a{color:#7bc8ff}p{line-height:1.7}</style>
<h2>统一眼睑观测 · 同一七组局部试验</h2>
<p><a id="localization" href="localization/index.html">查看七组照片的眼睑定位叠图</a> · <a href="localization/originalResolution/summary.json">查看同一检测器的原分辨率核验数据</a></p>
<p>从左到右：原照片、完成粗对齐的共同基线、原观测约束下的平滑形变重放、统一参考视角眼睑观测后的诊断。两条分支保留相同形变场、稠密拟合目标、完整眼裂评价、接触与相交门；候选仅统一粗点和参考眼部约束的观测定义。原观测门仍单独复核，不能把替代观测策略通过称为原规范通过。模型使用相同模板贴图、固定非皮肤附件和保存相机；继承失败继续保留。</p>
<p>这是眼周局部几何诊断。通过几何门不等于相对共同基线的完整眼孔形状已有改善；效果须另看完整眼孔测量。未重烘焙贴图，不评价眉毛和鼻翼纹理，也不代表可交付资产。真实样本口唇保持原状，本轮不能声称已修复口唇穿插。所有四个视角均展示，局部放大范围来自同一个基线模板。</p>
<label>样本<select id="case"></select></label><label>视角<select id="view"><option value="0">1</option><option value="1">2</option><option value="2">3</option><option value="3">4</option></select></label><label>显示<select id="mode"><option value="clay">灰模</option><option value="lit">共同光照</option><option value="baseColor">共同模板基色</option></select></label><p id="state"></p><img id="main"><img id="eyes">
<script>const qs=new URLSearchParams(location.search),$=id=>document.getElementById(id);fetch('viewer.json').then(r=>r.json()).then(data=>{for(const c of data.cases){$('case').add(new Option(c+(data.complete.includes(c)?'':'（等待渲染）'),c))}for(const k of ['case','view','mode'])if(qs.has(k)&&Array.from($(k).options).some(o=>o.value===qs.get(k)))$(k).value=qs.get(k);function update(){const c=$('case').value,v=$('view').value,m=$('mode').value;history.replaceState(null,'','?'+new URLSearchParams({case:c,view:v,mode:m}));$('main').src=c+'/view'+v.padStart(2,'0')+'_'+m+'.png';$('eyes').src=c+'/view'+v.padStart(2,'0')+'_'+m+'_eyes.png';$('localization').href='localization/index.html?case='+c;$('state').textContent=data.complete.includes(c)?data.branches.map(b=>{const o=data.observationStates[c][b];return data.labels[b]+'：'+(data.geometrySafetyAccepted[c][b]?'原验收门通过（仍为诊断）':'原验收门未通过，含继承问题')+'；原观测约束'+(o.originalObservationPolicyAccepted?'通过':'未通过')+'，当前观测约束'+(o.activeObservationPolicyAccepted===null?'映射无效，未执行':o.activeObservationPolicyAccepted?'通过':'未通过')+'，完整眼裂改善'+(o.improvementAccepted?'通过':'未通过')}).join(' / '):'此样本尚未完成渲染'}for(const k of ['case','view','mode'])$(k).onchange=update;update()});</script>''', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.case:
        renderCase(args.case)
    writeIndex()
