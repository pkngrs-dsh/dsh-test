"""Offline full-aperture measurement of EVERY saved unified-eye proposal.

Freeze this diagnostic and its inputs first. No solver, collision decision,
delivered geometry or parameter is changed. Rejected proposals remain rejected;
any reported mask benefit is hypothetical and cannot override physical gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from faceAppearance.visibleEyeContours import extractVisibleEyeContours, evaluateApertureMask

RUN = ROOT/'runs/eyeObservation20261024'
OUT = ROOT/'reports/eyeObservation20261024/rejectedApertureDiagnostic'
FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
BRANCH = 'unifiedObservationContact'
PLAN_SHA = 'fffd0854fbbbaf4e0cc3cd18fee75c4cd8eba22a9cd75fc1569ccdcdc5605c95'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
def verify(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected: raise ValueError('Frozen diagnostic input changed: '+path)


def readEvidence(stem):
    stem = Path(stem); metadata = read(stem.with_suffix('.json')); used = set()
    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as saved:
        def restore(item):
            if isinstance(item, dict) and 'arrayKey' in item:
                value = saved[item['arrayKey']].copy(); used.add(item['arrayKey'])
                if list(value.shape) != item['shape'] or str(value.dtype) != item['dtype']:
                    raise ValueError('Saved evidence index differs from array')
                return value
            if isinstance(item, dict): return {key: restore(value) for key, value in item.items()}
            if isinstance(item, list): return [restore(value) for value in item]
            return item
        result = restore(metadata)
        if used != set(saved): raise ValueError('Evidence archive contains unreferenced arrays')
    return result


def saveEvidence(stem, value):
    saved = {}
    def convert(item, key='root'):
        if isinstance(item, np.ndarray):
            saved[key] = item
            return dict(arrayKey=key, shape=list(item.shape), dtype=str(item.dtype))
        if isinstance(item, dict): return {str(k): convert(v, key+'/'+str(k)) for k, v in item.items()}
        if isinstance(item, (tuple, list)): return [convert(v, key+'/'+str(i)) for i, v in enumerate(item)]
        if isinstance(item, np.generic): return item.item()
        return item
    metadata = convert(value)
    if stem.with_suffix('.npz').exists(): raise FileExistsError(stem)
    np.savez_compressed(stem.with_suffix('.npz'), **saved)
    write(stem.with_suffix('.json'), metadata)


def freeze():
    original = read(RUN/'experimentPlan.json'); manifest = read(RUN/'implementationHashes.json')
    if sha(RUN/'experimentPlan.json') != PLAN_SHA or manifest['planSha256'] != PLAN_SHA:
        raise ValueError('Unexpected original seven-case experiment')
    verify(manifest['sourceHashes'])
    cases = original['cases']
    if len(cases) != 7 or len(set(cases)) != 7: raise ValueError('Expected every original case once')
    paths = {RUN/'experimentPlan.json', RUN/'implementationHashes.json', FONT}
    caseInputs = {}
    for case in cases:
        root = RUN/case; context = read(root/'inputContext.json'); complete = read(root/'report.json')
        if complete['status'] != 'complete': raise ValueError('All fits must finish before diagnostic freeze')
        report = read(root/BRANCH/'report.json'); valid = report['observationAlignmentValid']
        if type(valid) is not bool: raise ValueError('Mapping validity must be explicit')
        stem = 'unifiedPreparation' if valid else 'preparation'
        reference = context['referenceSelection']['referenceViewIndex']
        photo = Path(context['entry']['workingPhotos'])/f'input{reference:02d}.png'
        entries = [root/'inputContext.json', root/'report.json', root/'observationAlignment.json', root/'observationAlignment.npz',
            root/BRANCH/'report.json', root/BRANCH/'rawRefinement.json', root/BRANCH/'rawRefinement.npz',
            root/BRANCH/'fullGeometry.npz', root/'baseline/fullGeometry.npz',
            root/(stem+'.json'), root/(stem+'.npz'), Path(context['entry']['preparedGeometry']), photo]
        paths.update(entries)
        caseInputs[case] = dict(preparation=str(root/stem), rawRefinement=str(root/BRANCH/'rawRefinement'),
            report=str(root/BRANCH/'report.json'), baseline=str(root/'baseline/fullGeometry.npz'),
            delivered=str(root/BRANCH/'fullGeometry.npz'), preparedGeometry=context['entry']['preparedGeometry'],
            photo=str(photo), referenceViewIndex=reference, mappingValid=valid)
    sourceHashes = dict(manifest['sourceHashes']); sourceHashes[str(Path(__file__).resolve())] = sha(__file__)
    inputHashes = {str(path.resolve()): sha(path) for path in sorted(paths)}
    config = dict(schemaVersion='rejectedEyeApertureDiagnosticPlanV1', frozenAt=datetime.now(timezone.utc).isoformat(),
        sourceExperimentPlanSha256=PLAN_SHA, cases=cases, branch=BRANCH, caseInputs=caseInputs,
        policy=dict(proposals='Every saved outer proposal, including unchanged/zero and rejected proposals, in original order',
            invalidMapping='notEvaluated; no fabricated target or geometry', referenceViewOnly=True,
            measurement=dict(apertureAccessoryMode='innerCornerHeadAccessories', minimumEyeRasterWidth=256, maximumRoiSize=768),
            finalEffectPolicy=original['finalEffectPolicy'], baseline='Exact saved completed baseline; same camera/photo curves/scene and complete aperture method',
            cache='Exact per-case vertex bytes; all repeated proposals still retain their own report row',
            overlay='Original working photo, all visible mask contours, fixed baseline ROI; target cyan, baseline gold, proposal orange',
            cameraOrGeometryRefit=False, safetyGateChanged=False, candidateSelectionChanged=False,
            chooseBestProposal=False, completeFourViewAsset=False, observationsAreManualTruth=False),
        sourceHashes=sourceHashes, inputHashes=inputHashes,
        dependencies=dict(numpy=np.__version__, opencv=cv2.__version__))
    OUT.mkdir(parents=True, exist_ok=False)
    for source in sourceHashes:
        path = Path(source); target = OUT/'implementation'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    write(OUT/'diagnosticPlan.json', config)
    write(OUT/'freeze.json', dict(planSha256=sha(OUT/'diagnosticPlan.json'),
        sourceHashes=sourceHashes, inputHashes=inputHashes))
    print(json.dumps(dict(status='frozenBeforeOfflineRaster', planSha256=sha(OUT/'diagnosticPlan.json'),
        cases=cases, sourceCount=len(sourceHashes), inputCount=len(inputHashes))), flush=True)


def sameRoi(first, second):
    return (first['origin'] == second['origin'] and first['scale'] == second['scale']
        and tuple(first['shape']) == tuple(second['shape'])
        and first['sourceImageShape'] == second['sourceImageShape'])


def hypotheticalEffect(before, after, beforeMetrics, afterMetrics, scale, policy):
    """Same numerical final-baseline mask rule, with no physical acceptance."""
    checks = {}; differences = {}
    if beforeMetrics is None or afterMetrics is None:
        return dict(hypotheticalMaskEffectAccepted=False, checks=dict(metricsAvailable=False), differences=None)
    checks['sameTwoEyes'] = set(before['eyes']) == set(after['eyes']) == {'imageLeft', 'imageRight'}
    oldMean = beforeMetrics['meanSymmetricBoundaryDistanceOriginalPixels']
    newMean = afterMetrics['meanSymmetricBoundaryDistanceOriginalPixels']
    differences['meanSymmetricBoundaryDistanceOriginalPixels'] = newMean-oldMean
    differences['meanIntersectionOverUnion'] = afterMetrics['meanIntersectionOverUnion']-beforeMetrics['meanIntersectionOverUnion']
    checks['globalBoundaryBenefit'] = bool(np.isfinite([oldMean, newMean]).all() and newMean < oldMean-policy['globalBoundaryImprovement']/scale)
    for name in ('imageLeft', 'imageRight'):
        a, b = before['eyes'].get(name, {}), after['eyes'].get(name, {})
        checks[name+'SameRoi'] = 'roi' in a and 'roi' in b and sameRoi(a['roi'], b['roi'])
        checks[name+'CompleteFinalExtraction'] = b.get('status') == 'accepted'
        values = {}; differences[name] = values
        for metric, limit, direction in (
            ('symmetricBoundaryMeanOriginalPixels', policy['perEyeMeanTolerance']/scale, 1),
            ('symmetricBoundaryP95OriginalPixels', policy['perEyeP95Tolerance']/scale, 1),
            ('intersectionOverUnion', -policy['minimumIoUChange'], -1),
            ('visibleComponentCount', policy['maximumComponentCountIncrease'], 1),
            ('secondaryComponentAreaFraction', policy['maximumSecondaryAreaFractionIncrease'], 1)):
            left, right = float(beforeMetrics['eyes'][name][metric]), float(afterMetrics['eyes'][name][metric])
            values[metric] = right-left
            checks[name+metric] = bool(np.isfinite([left, right]).all() and direction*(right-left) <= limit)
    return dict(hypotheticalMaskEffectAccepted=bool(checks and all(checks.values())), checks=checks,
        differences=differences, physicalSafetyEstablished=False, acceptedForDelivery=False,
        meaning='Mask effect ONLY if physical rejection were ignored; actual rejection and delivered output stay unchanged')


def overlayImage(photo, before, after, photoEyes, title, path):
    font = ImageFont.truetype(str(FONT), 15); small = ImageFont.truetype(str(FONT), 12)
    width, height = 320, 215
    image = Image.new('RGB', (width*4, 78+height*2), '#121922'); draw = ImageDraw.Draw(image)
    draw.text((12, 8), title, fill='white', font=font)
    draw.text((12, 31), 'REFERENCE VIEW ONLY | rejected proposals stay rejected | cyan: automatic photo; gold: baseline; orange: proposal', fill='#f9b6a4', font=small)
    for column, label in enumerate(('Original working photo', 'Baseline + automatic target', 'Proposal + automatic target', 'All three contours')):
        draw.text((column*width+10, 57), label, fill='white', font=small)
    colors = dict(target=(45, 230, 235), baseline=(255, 216, 70), proposal=(255, 98, 62))
    for row, name in enumerate(('imageLeft', 'imageRight')):
        baselineEye = before['eyes'][name]; roi = baselineEye.get('roi')
        if roi is None:
            draw.text((12, 90+row*height), name+': baseline ROI unavailable', fill='white', font=font); continue
        rasterHeight, rasterWidth = roi['shape']; scale = roi['scale']; origin = np.asarray(roi['origin'])
        crop = photo.transform((rasterWidth, rasterHeight), Image.Transform.AFFINE,
            (1/scale, 0, origin[0], 0, 1/scale, origin[1]), resample=Image.Resampling.BICUBIC)
        ratio = min((width-16)/rasterWidth, (height-24)/rasterHeight)
        size = (max(1, round(rasterWidth*ratio)), max(1, round(rasterHeight*ratio)))
        def line(panel, points, color):
            points = (np.asarray(points)-origin)*scale
            if len(points) >= 2: ImageDraw.Draw(panel).line([tuple(p) for p in points], fill=color, width=max(1, round(1/ratio)))
        def contours(panel, eye, color):
            if 'roiMasks' not in eye: return
            paths, _ = cv2.findContours(np.asarray(eye['roiMasks']['eye'], np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            r = eye['roi']
            for contour in paths:
                points = contour[:, 0].astype(float)/r['scale']+np.asarray(r['origin'])
                if len(points): line(panel, np.r_[points, points[:1]], color)
        for column in range(4):
            panel = crop.copy()
            if column:
                for lid in ('upper', 'lower'): line(panel, photoEyes[name][lid], colors['target'])
            if column in (1, 3): contours(panel, baselineEye, colors['baseline'])
            if column in (2, 3): contours(panel, after['eyes'].get(name, {}), colors['proposal'])
            x = column*width+(width-size[0])//2; y = 78+row*height+20
            image.paste(panel.resize(size, Image.Resampling.LANCZOS), (x, y))
            draw.text((column*width+10, 80+row*height), name, fill='white', font=small)
    image.save(path)


def run():
    config = read(OUT/'diagnosticPlan.json'); frozen = read(OUT/'freeze.json')
    if sha(OUT/'diagnosticPlan.json') != frozen['planSha256']: raise ValueError('Diagnostic policy changed')
    verify(config['sourceHashes']); verify(config['inputHashes'])
    rows = []
    for case in config['cases']:
        item = config['caseInputs'][case]; folder = OUT/case; folder.mkdir(exist_ok=False)
        report = read(item['report']); raw = readEvidence(item['rawRefinement'])
        if not item['mappingValid']:
            if raw['candidateProposals']: raise ValueError('Invalid mapping unexpectedly contains proposals')
            row = dict(caseId=case, status='notEvaluated', reason=report['observationAlignmentError'],
                referenceViewIndex=item['referenceViewIndex'], proposals=[], mappingValid=False)
            write(folder/'report.json', row); rows.append(row); print(case+': notEvaluated (invalid mapping)', flush=True); continue
        prepared = readEvidence(item['preparation']); scene = prepared['scene']
        # JSON object keys were strings; this restores the original material-ID
        # identity without modifying any alpha array or material classification.
        scene['alphaTextures'] = {int(key): value for key, value in scene['alphaTextures'].items()}
        with np.load(item['baseline'], allow_pickle=False) as saved:
            baseline = saved['vertices']; np.testing.assert_array_equal(saved['faces'], scene['faces'])
        np.testing.assert_array_equal(scene['vertices'], baseline)
        with np.load(item['delivered'], allow_pickle=False) as saved:
            deliveredExactlyBaseline = bool(np.array_equal(saved['vertices'], baseline))
        with np.load(item['preparedGeometry'], allow_pickle=False) as saved: unit = float(saved['unit'])
        photo = Image.open(item['photo']).convert('RGB')
        if photo.size != tuple(prepared['imageShape'][::-1]): raise ValueError('Photo is not at the saved working resolution')
        cache = {}
        def measure(vertices):
            fingerprint = hashlib.sha256(np.ascontiguousarray(vertices).tobytes()).hexdigest()
            if fingerprint not in cache:
                value = extractVisibleEyeContours(**{**scene, 'vertices': vertices}, camera=prepared['camera'],
                    photoEyes=prepared['photoEyes'], imageShape=prepared['imageShape'],
                    apertureAccessoryFaceIds=prepared['cornerFaceIds'], **config['policy']['measurement'])
                error = None
                try: metrics = evaluateApertureMask(value, prepared['photoEyes'], originalPixelScale=prepared['originalPixelScale'])
                except ValueError as exception: metrics = None; error = str(exception)
                stem = folder/('measurement-'+fingerprint)
                saveEvidence(stem, dict(verticesSha256=fingerprint, completeContours=value, metrics=metrics, metricError=error))
                cache[fingerprint] = (value, metrics, error, str(stem.relative_to(OUT)))
            return cache[fingerprint]
        before, beforeMetrics, baselineError, baselineStem = measure(baseline)
        baselineChecks = {}
        for name in ('imageLeft', 'imageRight'):
            old, new = prepared['beforeContours']['eyes'][name], before['eyes'][name]
            baselineChecks[name] = (sameRoi(old['roi'], new['roi']) and np.array_equal(old['roiMasks']['eye'], new['roiMasks']['eye']))
        if not all(baselineChecks.values()): raise ValueError('Offline baseline mask differs from frozen saved baseline')
        overlayImage(photo, before, before, prepared['photoEyes'], case+' | common baseline | view '+str(item['referenceViewIndex']), folder/'baseline.png')
        proposals = []
        for index, proposal in enumerate(raw['candidateProposals']):
            vertices = proposal['vertices']; after, metrics, error, stem = measure(vertices)
            effect = hypotheticalEffect(before, after, beforeMetrics, metrics, prepared['originalPixelScale'], config['policy']['finalEffectPolicy'])
            delta = np.linalg.norm(vertices-baseline, axis=1)/unit
            imageName = f'proposal{index:02d}.png'
            rejected = not proposal['strict']['accepted']
            title = f'{case} | proposal {index+1} | round {proposal["round"]}, scale {proposal["scale"]} | '+('PHYSICALLY REJECTED' if rejected else 'outer strict passed; not final acceptance')
            overlayImage(photo, before, after, prepared['photoEyes'], title, folder/imageName)
            proposals.append(dict(index=index, round=proposal['round'], scale=proposal['scale'],
                savedStrictAccepted=proposal['strict']['accepted'], originallyRejectedByStrictGate=rejected,
                originalStrictReasons=proposal['strict']['reasons'], selectedByFastRound=proposal['selectedByFastRound'],
                savedOriginalCompleteMaskGate=proposal['solverAttempt'].get('apertureGate'),
                savedOriginalMaskMeasured='apertureMetrics' in proposal['solverAttempt'],
                changedVertexCount=int(np.any(vertices != baseline, axis=1).sum()), maximumDisplacementMm=float(delta.max(initial=0)),
                verticesSha256=hashlib.sha256(np.ascontiguousarray(vertices).tobytes()).hexdigest(),
                completeMetrics=metrics, metricError=error, hypotheticalMaskEffect=effect,
                originalCoarseAndReferencePolicy=proposal['strict']['externalCandidateGate'].get('originalPolicy'),
                measurementEvidence=stem+'.json', overlay=case+'/'+imageName,
                acceptedForDelivery=False, diagnosticOnly=True))
            print(f'{case}: proposal {index+1}/{len(raw["candidateProposals"])} measured; hypothetical mask effect={effect["hypotheticalMaskEffectAccepted"]}; strict rejected={rejected}', flush=True)
        row = dict(caseId=case, status='complete', mappingValid=True, referenceViewIndex=item['referenceViewIndex'],
            baselineMetrics=beforeMetrics, baselineMetricError=baselineError, baselineReplayChecks=baselineChecks,
            baselineEvidence=baselineStem+'.json', baselineOverlay=case+'/baseline.png',
            deliveredExactlyBaseline=deliveredExactlyBaseline, uniqueGeometryMeasurementCount=len(cache), proposals=proposals)
        # The diagnostic stores only scalar original-policy summaries in its
        # main JSON; full numerical evidence remains in the frozen source.
        for proposal in row['proposals']:
            original = proposal['originalCoarseAndReferencePolicy']
            if original is not None:
                proposal['originalCoarseAndReferencePolicy'] = dict(coarseAccepted=original['coarseAlignment']['accepted'], referenceAccepted=original['legacyGate']['accepted'])
        write(folder/'report.json', row); rows.append(row)
    verify(config['sourceHashes']); verify(config['inputHashes'])
    proposals = [proposal for row in rows for proposal in row['proposals']]
    summary = dict(schemaVersion='rejectedEyeApertureDiagnosticV1', diagnosticPlanSha256=frozen['planSha256'], cases=rows,
        aggregate=dict(caseCount=len(rows), evaluatedCaseCount=sum(row['status'] == 'complete' for row in rows), proposalCount=len(proposals),
            changedProposalCount=sum(row['changedVertexCount'] > 0 for row in proposals),
            hypotheticalMaskEffectPassCount=sum(row['hypotheticalMaskEffect']['hypotheticalMaskEffectAccepted'] for row in proposals),
            physicallyRejectedWithHypotheticalMaskEffectCount=sum(row['originallyRejectedByStrictGate'] and row['hypotheticalMaskEffect']['hypotheticalMaskEffectAccepted'] for row in proposals)),
        allFrozenInputsUnchanged=True, noFittingOrOutputModification=True, referenceViewOnly=True,
        interpretation='Every saved proposal measured without selecting a best proposal. Hypothetical mask benefit cannot override any physical rejection, restore original-policy eligibility, establish manual truth, or create a delivered four-view asset.')
    write(OUT/'summary.json', summary)
    writeIndex()
    outputs = {str(path.relative_to(OUT)): sha(path) for path in OUT.rglob('*') if path.is_file() and 'implementation' not in path.parts}
    write(OUT/'outputHashes.json', outputs)
    print(json.dumps(dict(status='complete', summary=str(OUT/'summary.json'), aggregate=summary['aggregate'])), flush=True)


def writeIndex():
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>被拒候选的离线完整眼裂诊断</title>
<style>body{background:#151c25;color:#e7edf5;font:16px sans-serif;margin:24px;line-height:1.65}select{font:inherit;padding:5px;margin:8px}img{display:block;max-width:100%}.reject{color:#ff9c85;font-weight:bold}a{color:#7acaff}pre{white-space:pre-wrap;font-size:13px}</style>
<h2>被拒候选的完整眼裂方向核验</h2><p>仅参考视角，按固定七例与全部 proposal 原顺序展示，不选最佳。照片青色线为自动检测；黄色为完成粗对齐基线；橙色为候选完整可见眼裂。所有碎片和孔洞边界保留。这里只补做离线 mask 测量，不重拟合、不降低碰撞门、不更改交付资产。即使假设性 mask 效果通过，原物理拒绝仍有效。</p>
<p><a href="../index.html">返回四视角资产对照</a> · <a href="summary.json">全部逐候选数值</a> · <a href="diagnosticPlan.json">运行前冻结的输入与规则</a></p>
<label>样本<select id="case"></select></label><label>保存候选<select id="proposal"></select></label><p id="status" class="reject"></p><img id="overlay"><pre id="metrics"></pre>
<script>const $=id=>document.getElementById(id);fetch('summary.json').then(r=>r.json()).then(s=>{for(const c of s.cases)$('case').add(new Option(c.caseId,c.caseId));let rows;function render(){const c=s.cases.find(x=>x.caseId===$('case').value),p=c.proposals[+$('proposal').value];if(!p){$('status').textContent='未评价：'+c.reason;$('overlay').removeAttribute('src');$('metrics').textContent='';return}$('status').textContent=(p.originallyRejectedByStrictGate?'原物理门拒绝：'+p.originalStrictReasons.join(', '):'原 strict 门通过；不等于最终通过')+'；最大位移 '+p.maximumDisplacementMm.toFixed(4)+' mm；假设性完整 mask 效果 '+(p.hypotheticalMaskEffect.hypotheticalMaskEffectAccepted?'通过（仍不得交付）':'未通过');$('overlay').src=p.overlay;$('metrics').textContent=JSON.stringify({referenceView:c.referenceViewIndex,baseline:c.baselineMetrics,proposal:p.completeMetrics,differences:p.hypotheticalMaskEffect.differences,checks:p.hypotheticalMaskEffect.checks},null,2)}function change(){const c=s.cases.find(x=>x.caseId===$('case').value);$('proposal').replaceChildren();for(const p of c.proposals)$('proposal').add(new Option('proposal '+(p.index+1)+' / round '+p.round+' / scale '+p.scale,p.index));render()}$('case').onchange=change;$('proposal').onchange=render;change()});</script>''', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true'); parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.freeze == args.run: parser.error('Choose exactly one of --freeze or --run')
    freeze() if args.freeze else run()
