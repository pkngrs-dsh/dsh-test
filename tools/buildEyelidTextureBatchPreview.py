#!/usr/bin/env python3
"""Build a static, local multi-case eye-texture comparison from explicit evidence.

Only named PNG evidence is copied. Missing comparisons remain missing; failed
cases can still show their original detector overlay. No reconstruction, image
processing, remote calls or server startup is performed by this command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>眼睑贴图 · 多样本对照</title>
<style>
:root{color-scheme:dark;font:16px/1.6 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;background:#10131a;color:#e8ebf2}
*{box-sizing:border-box}body{margin:0}main{max-width:1500px;margin:auto;padding:32px 24px 60px}
h1{font-size:29px;margin:0 0 8px;letter-spacing:.02em}h2{font-size:20px;margin:0 0 8px}p{margin:0 0 12px}
.muted{color:#a9b2c6}.intro{max-width:900px;margin-bottom:25px}.badge{display:inline-block;font-size:12px;border:1px solid #576276;padding:3px 9px;border-radius:20px;vertical-align:middle;margin-left:10px}
.toolbar{display:flex;flex-wrap:wrap;align-items:end;gap:14px;padding:18px;background:#1b202b;border:1px solid #30394b;border-radius:14px;margin-bottom:18px}
label{display:grid;gap:5px;font-size:13px;color:#bcc5d6}select,button{font:inherit;color:inherit;background:#151a25;border:1px solid #526079;border-radius:7px;padding:9px 12px;cursor:pointer}select{min-width:180px}button[aria-pressed=true]{background:#27619b;border-color:#7abaff;color:#fff}button:disabled{opacity:.4;cursor:default}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #95c9ff;outline-offset:3px}.tabs{display:flex;gap:8px;flex-wrap:wrap}a{color:#9fceff}.case-info{background:#181e28;padding:17px 20px;border-left:3px solid #729bca;border-radius:5px;margin-bottom:18px}.case-info p{margin:3px 0;white-space:pre-line}.status{font-size:13px;color:#b7c8dc}.image-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}figure{margin:0;background:#1b202b;border:1px solid #30394b;border-radius:10px;overflow:hidden}figcaption{padding:10px 14px;font-size:14px;border-bottom:1px solid #30394b}figure img{display:block;width:100%;height:auto;object-fit:contain;background:#252a33}.missing{padding:55px 20px;text-align:center;color:#a9b2c6}.wide-image{max-width:1200px;margin:auto}.wide-image img{width:100%}.foot{font-size:13px;margin-top:18px}.empty{border:1px dashed #576276;border-radius:9px;padding:24px;color:#bcc5d6;margin-bottom:20px}.hidden{display:none!important}#error{color:#ffb7b7}#detailLink{margin-top:10px}#evidenceNote{margin-top:14px}#caseNote{overflow-wrap:anywhere}
@media(max-width:750px){main{padding:20px 12px}.image-grid{grid-template-columns:1fr}.toolbar{gap:10px}.toolbar label{flex:1}select{width:100%;min-width:130px}h1{font-size:23px}.badge{margin-left:0;display:block;width:max-content;margin-top:7px}}
</style></head><body><main>
<header><h1>眼睑贴图 · 多样本对照 <span class="badge">本地实验 · 未部署</span></h1>
<p class="intro muted">按样本查看原照片、清理前与局部修复后的效果。眼区放大用于观察睑缘和残留色块；灰模用于核对本次贴图处理前后的几何。</p></header>
<section class="toolbar" aria-label="对照选项">
<label>样本<select id="caseSelect"></select></label>
<label>照片视角<select id="viewSelect"><option value="0">正面</option><option value="1">视角 2</option><option value="2">视角 3</option><option value="3">视角 4</option></select></label>
<label>渲染方式<select id="modeSelect"><option value="baseColor">无光照贴图</option><option value="lit">统一光照</option><option value="clay">灰模</option></select></label>
<div class="tabs" role="group" aria-label="显示内容"><button id="viewsButton" data-panel="views" aria-pressed="true">四视角对照</button><button id="eyesButton" data-panel="eyes" aria-pressed="false">眼区放大</button><button id="detectionButton" data-panel="detection" aria-pressed="false">检测线</button></div>
</section>
<section class="case-info" aria-live="polite"><h2 id="caseTitle"></h2><p id="caseStatus" class="status"></p><p id="caseNote"></p><p id="detailLink" class="hidden"><a id="detailAnchor" target="_blank" rel="noopener">打开此样本的交互预览 ↗</a></p></section>
<p id="error" class="hidden"></p><div id="empty" class="empty hidden"></div>
<section id="viewsPanel" class="image-grid" aria-label="照片与模型对照">
<figure><figcaption>原照片 · 同一视角</figcaption><img id="photoImage" alt="原照片"><div id="photoMissing" class="missing hidden">没有该视角的原照片证据</div></figure>
<figure><figcaption>贴图清理前</figcaption><img id="beforeImage" alt="清理前模型渲染"><div id="beforeMissing" class="missing hidden">没有该视角、模式的清理前渲染</div></figure>
<figure><figcaption>局部修复后</figcaption><img id="afterImage" alt="局部修复后模型渲染"><div id="afterMissing" class="missing hidden">没有该视角、模式的修复后渲染</div></figure>
</section>
<section id="eyesPanel" class="wide-image hidden" aria-label="眼区放大对照"><figure><figcaption>眼区放大 · 原照片 / 清理前 / 局部修复后</figcaption><img id="eyesImage" alt="眼区放大对照"></figure></section>
<section id="detectionPanel" class="wide-image hidden" aria-label="原照片眼睑检测"><figure><figcaption>原照片的自动眼睑检测 · 不代表手工标注真值</figcaption><img id="detectionImage" alt="原照片眼睑自动检测与编号"></figure></section>
<p id="evidenceNote" class="muted foot"></p>
<footer class="muted foot">多个采集目录不等于已核验的多个身份。拍摄模式沿用用户对这一批“基本扭头”的描述，未逐例确认。缺失或被拒绝的实验不生成替代结果。</footer>
</main><script id="evidenceData" type="application/json">__DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('evidenceData').textContent);
const $=id=>document.getElementById(id);let panel='views';
const state={caseIndex:0,view:0,mode:'baseColor'};
function image(id,missingId,path){const img=$(id);img.classList.toggle('hidden',!path);if(path)img.src=path;else img.removeAttribute('src');if(missingId)$(missingId).classList.toggle('hidden',!!path);}
function show(){
 const c=data.cases[state.caseIndex];if(!c)return;
 $('caseTitle').textContent=c.label||c.caseId;$('caseStatus').textContent=c.status||'';$('caseNote').textContent=c.note||'';
 $('detailLink').classList.toggle('hidden',!c.previewUrl);if(c.previewUrl)$('detailAnchor').href=c.previewUrl;
 const files=c.files;const prefix=`view${state.view}`;
 image('photoImage','photoMissing',files[`${prefix}-photo.png`]);image('beforeImage','beforeMissing',files[`${prefix}-before-${state.mode}.png`]);image('afterImage','afterMissing',files[`${prefix}-after-${state.mode}.png`]);
 image('eyesImage',null,files['eyes-comparison.png']);image('detectionImage',null,files['detection.png']);
 const hasViews=Object.keys(files).some(k=>/^view\d-(before|after)-/.test(k));
 $('eyesButton').disabled=!files['eyes-comparison.png'];$('detectionButton').disabled=!files['detection.png'];$('viewsButton').disabled=!hasViews;
 if((panel==='eyes'&&!files['eyes-comparison.png'])||(panel==='detection'&&!files['detection.png'])||(panel==='views'&&!hasViews))panel=hasViews?'views':files['detection.png']?'detection':files['eyes-comparison.png']?'eyes':'empty';
 for(const name of ['views','eyes','detection']){$(`${name}Panel`).classList.toggle('hidden',panel!==name);$(`${name}Button`).setAttribute('aria-pressed',String(panel===name));}
 $('viewSelect').disabled=panel!=='views';$('modeSelect').disabled=panel!=='views';
 const incomplete=!hasViews;$('empty').classList.toggle('hidden',!incomplete);$('empty').textContent='此样本未生成可用的贴图前后对照。保留检测证据和原因，避免把失败样本当作成功结果。';
 $('evidenceNote').textContent=panel==='views'?'所有图像均来自同一采集照片及保存的相机；未用其他样本补充缺失视图。':panel==='eyes'?'这里显示已保存的眼区局部对照。建议同时检查四视角，避免只看正面的局部改善。':'检测线仅用于判断算法是否正确跟随眼裂；低清晰度、睫毛和遮挡都可能影响自动检测。';
 const url=new URL(location.href);url.searchParams.set('case',c.caseId);url.searchParams.set('view',String(state.view));url.searchParams.set('mode',state.mode);try{history.replaceState(null,'',url);}catch(e){}
}
for(let i=0;i<data.cases.length;i++){const c=data.cases[i];const option=document.createElement('option');option.value=String(i);option.textContent=c.label||c.caseId;$('caseSelect').append(option);}
const initial=new URLSearchParams(location.search);const wanted=data.cases.findIndex(c=>c.caseId===initial.get('case'));if(wanted>=0)state.caseIndex=wanted;
if(['0','1','2','3'].includes(initial.get('view')))state.view=Number(initial.get('view'));if(['baseColor','lit','clay'].includes(initial.get('mode')))state.mode=initial.get('mode');
$('caseSelect').value=String(state.caseIndex);$('viewSelect').value=String(state.view);$('modeSelect').value=state.mode;
$('caseSelect').addEventListener('change',e=>{state.caseIndex=Number(e.target.value);show();});$('viewSelect').addEventListener('change',e=>{state.view=Number(e.target.value);show();});$('modeSelect').addEventListener('change',e=>{state.mode=e.target.value;show();});
for(const b of document.querySelectorAll('[data-panel]'))b.addEventListener('click',()=>{panel=b.dataset.panel;show();});
if(!data.cases.length){$('error').classList.remove('hidden');$('error').textContent='没有可显示的样本。';}else show();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.summary.read_text(encoding='utf-8'))
    rows = source.get('cases')
    if not isinstance(rows, list) or not rows:
        raise ValueError('summary.cases must be a non-empty list')
    seen = set()
    for row in rows:
        case_id = row['caseId']
        if not isinstance(case_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', case_id):
            raise ValueError(f'Unsafe caseId: {case_id!r}')
        if case_id in seen:
            raise ValueError(f'Duplicate caseId: {case_id}')
        seen.add(case_id)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {'title': '眼睑贴图 · 多样本对照', 'cases': []}
    identities = []
    for row in rows:
        case_id = row['caseId']
        directory = args.output / case_id
        directory.mkdir()
        item = {key: row.get(key, '') for key in ('caseId', 'label', 'status', 'note')}
        preview_url = row.get('previewUrl')
        if preview_url and not re.match(r'^https?://', preview_url):
            raise ValueError(f'Only HTTP(S) previewUrl is supported: {case_id}')
        item['previewUrl'] = preview_url
        item['files'] = {}
        expected = []
        if row.get('comparison'):
            comparison = Path(row['comparison'])
            for view in range(4):
                expected.append((comparison / 'evidence' / f'view{view}-photo.png', f'view{view}-photo.png'))
                for branch in ('before', 'after'):
                    for mode in ('baseColor', 'lit', 'clay'):
                        filename = f'view{view}-{branch}-{mode}.png'
                        expected.append((comparison / 'evidence' / filename, filename))
            expected.append((comparison / 'eyes/comparison.png', 'eyes-comparison.png'))
        if row.get('observations'):
            expected.append((Path(row['observations']) / 'view00EyeDetail.png', 'detection.png'))
        for original, filename in expected:
            if not original.is_file():
                continue
            with original.open('rb') as handle:
                if handle.read(8) != b'\x89PNG\r\n\x1a\n':
                    raise ValueError(f'Named evidence is not a PNG: {original}')
            target = directory / filename
            shutil.copyfile(original, target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            item['files'][filename] = f'{case_id}/{filename}'
            identities.append({'source': str(original.resolve()), 'copy': f'{case_id}/{filename}', 'sha256': digest})
        result['cases'].append(item)
    data = json.dumps(result, ensure_ascii=False, indent=2)
    (args.output / 'data.json').write_text(data + '\n', encoding='utf-8')
    # Escape '<' so a source label cannot close the inert JSON script block.
    (args.output / 'index.html').write_text(PAGE.replace('__DATA__', data.replace('<', '\\u003c')), encoding='utf-8')
    manifest = {'summary': str(args.summary.resolve()), 'summarySha256': hashlib.sha256(args.summary.read_bytes()).hexdigest(),
                'copiedPngCount': len(identities), 'files': identities, 'sourceFilesUnmodified': True}
    (args.output / 'buildManifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output.resolve()), 'caseCount': len(rows), 'copiedPngCount': len(identities)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
