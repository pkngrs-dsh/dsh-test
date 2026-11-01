"""Summarize the frozen seven-case replay without selecting or changing assets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/shapePipeline20261024'
RUNS = ROOT / 'runs/shapePipeline20261024'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>七组统一流程对照</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:16px/1.6 system-ui,sans-serif;background:#171b22;color:#e5eaf0;margin:0;padding:24px}
main{max-width:1400px;margin:auto}h1{font-size:25px;margin:0 0 8px}p{color:#bec7d2;margin:8px 0}
nav{display:flex;flex-wrap:wrap;gap:16px;margin:20px 0}label{display:flex;gap:8px;align-items:center}
select{font:inherit;background:#252d39;color:#fff;border:1px solid #657083;border-radius:6px;padding:5px}
img{display:block;width:100%;height:auto;margin-top:16px;background:#222}a{color:#8dc4ff}
#status{padding:10px 14px;background:#28303c;border-radius:6px}small{color:#bbc3ce}</style>
<main><h1>七组统一流程对照</h1>
<p>七组使用同一组参数尝试完整重放：六组完成，另一组保留原安全检查拒绝记录。</p>
<p>历史资产仅供参照；“统一流程基线”和“统一流程新解”是配对实验。当前均为本地诊断产物。</p>
<p>眼睑坐标精度缺陷已修复：13 份缓存的模块重放通过，原 12 份成功纹理逐像素不变。
<a href="independentAudit/exactCoordinateReplay/report.json">修复验证</a> · <a href="precisionFixTests.json">99 项回归测试</a></p>
<nav><label>样本<select id="case"></select></label><label>视角<select id="view"></select></label>
<label>对照<select id="kind"><option value="Comparison">完整对照</option>
<option value="_baseColor_eyes">眼部基色</option><option value="_clay_nose">鼻部灰模</option>
<option value="_baseColor_nose">鼻部基色</option><option value="_lit_nose">鼻部共同光照</option>
<option value="_baseColor_factorialNose">鼻部几何 × 贴图</option>
<option value="_baseColor_factorial">全脸几何 × 贴图</option></select></label></nav>
<div id="status">读取报告…</div><a id="imageLink" target="_blank"><img id="comparison" alt="相同保存相机下的统一流程对照"></a>
<p><a href="overview_baseColor.png">七组全脸总览</a> · <a href="overview_clay_nose.png">鼻部灰模总览</a> ·
<a href="overview_baseColor_eyes.png">眼部总览</a> · <a href="result.json">完整数值报告</a></p>
<small>几何与贴图交叉组合仅用于定位问题，不是交付资产。数值门通过不能证明照片相似度或重光照准确性。</small></main>
<script>
const caseSelect=document.getElementById('case'),viewSelect=document.getElementById('view'),kindSelect=document.getElementById('kind');
const query=new URLSearchParams(location.search);let data;
function refresh(){const row=data.cases.find(x=>x.caseId===caseSelect.value);const name='view'+viewSelect.value.padStart(2,'0')+kindSelect.value+'.png';
if(row.status!=='complete'){document.getElementById('comparison').style.display='none';document.getElementById('imageLink').removeAttribute('href');document.getElementById('status').textContent=row.caseId+' · 本轮在 '+row.failedStage+' 被安全检查拒绝，未生成完整对照；精度缺陷修复重放单独记录。';return;}
document.getElementById('comparison').style.display='block';
const url=encodeURIComponent(row.caseId)+'/render/'+name;document.getElementById('comparison').src=url;document.getElementById('imageLink').href=url;
document.getElementById('status').textContent=row.caseId+' · '+(row.pipelineNumericalAccepted?'数值门通过；仍待视觉与客户端评估':'含未通过数值门；仅供诊断')+' · 新增完整装配相交对数：'+row.newFullIntersectionPairs.length;
history.replaceState(null,'','?case='+encodeURIComponent(row.caseId)+'&view='+viewSelect.value+'&kind='+encodeURIComponent(kindSelect.value));}
fetch('result.json').then(r=>{if(!r.ok)throw Error('报告尚未完成');return r.json()}).then(x=>{data=x;
for(const row of data.cases){const o=document.createElement('option');o.value=row.caseId;o.textContent=row.caseId;caseSelect.append(o)}
for(let i=0;i<4;i++){const o=document.createElement('option');o.value=i;o.textContent='视角 '+(i+1);viewSelect.append(o)}
if(data.cases.some(x=>x.caseId===query.get('case')))caseSelect.value=query.get('case');
const ref=data.cases.find(x=>x.caseId===caseSelect.value).referenceViewIndex;viewSelect.value=/^[0-3]$/.test(query.get('view'))?query.get('view'):ref;
if([...kindSelect.options].some(x=>x.value===query.get('kind')))kindSelect.value=query.get('kind');
caseSelect.onchange=()=>{viewSelect.value=data.cases.find(x=>x.caseId===caseSelect.value).referenceViewIndex;refresh()};viewSelect.onchange=kindSelect.onchange=refresh;refresh();
}).catch(e=>{document.getElementById('status').textContent=e.message});</script></html>'''


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def makeOverview(rows, mode, cropName=None):
    panelWidth, panelHeight = 256, 256 if cropName is None else 150
    rowHeight = panelHeight + 50
    canvas = Image.new('RGB', (panelWidth * 4, 106 + len(rows) * rowHeight), (23, 27, 34))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 15)
    titleFont = ImageFont.truetype(FONT, 21)
    title = {'baseColor': '重新烘焙基色', 'clay': '灰模几何', 'lit': '共同检查光照'}[mode]
    if cropName:
        title += ' · ' + {'eyes': '眼部', 'nose': '鼻部'}[cropName]
    draw.text((12, 8), '七组统一流程对照 · ' + title, font=titleFont, fill='white')
    draw.text((12, 40), '同一配置、保存相机；历史资产仅作参照。新解均为诊断产物，未替换客户端资产。', font=font, fill='#c9d1db')
    labels = ['原工作照片', '历史资产', '统一流程基线', '统一流程新解']
    for index, label in enumerate(labels):
        draw.text((index * panelWidth + 12, 74), label, font=font, fill='white')
    for rowIndex, row in enumerate(rows):
        directory = REPORTS / row['caseId'] / 'render'
        reference = row['referenceViewIndex']
        y = 106 + rowIndex * rowHeight
        if row['status'] != 'complete':
            draw.text((12, y + 5), f"{row['caseId']} · 安全检查拒绝，未生成完整配对资产", font=font, fill='#f0cc88')
            draw.text((panelWidth + 12, y + 75), '原失败保留；精度修复的模块重放单独记录。', font=font, fill='#c9d1db')
            with Image.open(row['referencePhoto']) as source:
                picture = ImageOps.contain(source.convert('RGB'), (panelWidth - 8, panelHeight))
                canvas.paste(picture, ((panelWidth - picture.width) // 2, y + 38))
            continue
        view = row['renderViews'][reference]
        prefix = f'view{reference:02d}'
        names = [prefix + '_photo.png'] + [prefix + f'_{branch}_{mode}.png' for branch in ('historical', 'pureDirect', 'bounded')]
        status = '数值门通过' if row['pipelineNumericalAccepted'] else '含未通过数值门'
        draw.text((12, y + 5), f"{row['caseId']} · 参考视角 {reference + 1} · {status}", font=font, fill='#f0cc88')
        for index, name in enumerate(names):
            path = directory / name
            if digest(path) != row['imageHashes'][name]:
                raise ValueError(f'Render changed: {path}')
            with Image.open(path) as source:
                picture = source.convert('RGB')
                if cropName:
                    picture = picture.crop(view['crops'][cropName])
                picture = ImageOps.contain(picture, (panelWidth - 8, panelHeight))
                canvas.paste(picture, (index * panelWidth + (panelWidth - picture.width) // 2, y + 38))
    name = 'overview_' + mode + ('_' + cropName if cropName else '') + '.png'
    canvas.save(REPORTS / name)
    return str(REPORTS / name)


def main():
    split = read(ROOT / 'runs/shapePreservation20261024/split.json')
    batch = read(RUNS / 'batchStatus.json')
    if batch['status'] not in ('complete', 'completeWithFailures'):
        raise ValueError('Wait for every frozen case attempt to finish')
    rows, sources = [], {}
    for entry in split['cases']:
        caseId = entry['caseId']
        attempt = batch['cases'][caseId]
        if attempt['status'] != 'complete':
            auditPath = REPORTS / 'independentAudit' / caseId / 'geometryValidation.json'
            audit = read(auditPath)
            reference = read(RUNS / caseId / 'geometry/report.json')['referenceSelection']['referenceViewIndex']
            sources[str(auditPath)] = digest(auditPath)
            rows.append(dict(caseId=caseId, split=entry['split'], sourceDomain=entry['sourceDomain'],
                status='rejectedBeforeCompleteAsset', failedStage=attempt['failedStage'],
                referenceViewIndex=reference, referencePhoto=str(Path(entry['workingPhotos']) / f'input{reference:02d}.png'),
                validationPassed=False, geometryValidationPassed=audit['validationPassed'],
                pipelineNumericalAccepted=False, geometry=audit['geometry'],
                newFullIntersectionPairs=audit['geometry']['fullCollisions']['newPairs']))
            continue
        auditPath = REPORTS / 'independentAudit' / caseId / 'pipelineValidation.json'
        renderPath = REPORTS / caseId / 'render/report.json'
        audit, render = read(auditPath), read(renderPath)
        if audit['status'] != 'completeIndependentPipelineAudit' or render['status'] != 'completeDiagnosticRendering':
            raise ValueError('Only completed full pipeline audits and renders may be summarized')
        sources.update({str(path): digest(path) for path in (auditPath, renderPath)})
        geometry = audit['geometry']
        appearances = {branch: read(RUNS / caseId / 'appearance' / branch / 'report.json') for branch in ('pureDirect', 'bounded')}
        rows.append(dict(caseId=caseId, split=entry['split'], sourceDomain=entry['sourceDomain'], status='complete',
            referenceViewIndex=appearances['bounded']['referenceViewIndex'],
            validationPassed=audit['validationPassed'], pipelineNumericalAccepted=audit['pipelineNumericalAccepted'],
            finalCavity=geometry['branches']['bounded']['finalCavity'],
            baselineCavity=geometry['branches']['pureDirect']['finalCavity'],
            newFullIntersectionPairs=geometry['fullCollisions']['newPairs'],
            downstream={key: value['downstream'] for key, value in geometry['branches'].items()},
            appearance=audit['appearance'],
            archives={branch: value['asset'] for branch, value in appearances.items()},
            renderViews=render['views'], imageHashes=render['imageSha256']))
    images = [makeOverview(rows, 'baseColor'), makeOverview(rows, 'clay', 'nose'), makeOverview(rows, 'baseColor', 'eyes')]
    result = dict(status='completeSevenCaseDiagnostic', caseCount=len(rows),
        completePairedCaseCount=sum(row['status']=='complete' for row in rows),
        rejectedCaseCount=sum(row['status']!='complete' for row in rows),
        independentlyValidated=sum(row['validationPassed'] for row in rows),
        pipelineNumericalAcceptedCount=sum(row['pipelineNumericalAccepted'] for row in rows),
        sourceHashes=sources, overviewImages={path: digest(Path(path)) for path in images},
        cases=[{key: value for key, value in row.items() if key not in ('renderViews', 'imageHashes')} for row in rows],
        finalCandidateAccepted=False, productionDefaultsChanged=False, deployed=False,
        testedCodeSnapshot=str(REPORTS / 'frozenImplementation/index.json'),
        limitations=['All seven use the same template; this does not establish generalization to other templates.',
            'Template-relative cavity depth benefit is not photographic likeness.',
            'Historical AI asset used an additional eye-ring experiment excluded from both paired branches.',
            'Visual review is recorded separately; numerical validation is not client acceptance.'])
    precisionPath = REPORTS / 'independentAudit/exactCoordinateReplay/report.json'
    precision = read(precisionPath)
    result['precisionFixReplay'] = {key: value for key, value in precision.items() if key != 'cases'}
    result['precisionFixReplay'].update(report=str(precisionPath), sha256=digest(precisionPath))
    result['independentFinalReview'] = str(REPORTS / 'independentAudit/finalSummary.json')
    result['visualReview'] = str(REPORTS / 'visualQa/summary.json')
    result['regressionTests'] = read(REPORTS / 'precisionFixTests.json')
    (REPORTS / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (REPORTS / 'index.html').write_text(PAGE)
    print(json.dumps({key: result[key] for key in ('status', 'caseCount', 'independentlyValidated', 'pipelineNumericalAcceptedCount')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
