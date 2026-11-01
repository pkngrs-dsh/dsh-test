"""Read-only localhost viewer for the selfie crop experiment (port 18091)."""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import re
import sys
import threading

import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from renderUffSelfieCrop import BASELINE, GROUPS, PANEL_FILES, PASSES, VARIANTS, digest, read_json, variant_complete, variant_entries
from serveUffBackgroundAblation import preview_payload

LABELS = {name: ('固定框 1.35' if name.startswith('shared') else '逐帧框 ' + {'m110': '1.10', 'm135': '1.35', 'm165': '1.65'}[name.split('_')[0]]) + ' · ' + {'raw': '原背景', 'backgroundGray': '去背景灰底', 'headGray': '头部灰底'}[name.split('_')[1]] for name in VARIANTS}
COMPARISONS = {f'{prefix}{name[0].upper() + name[1:]}' for prefix in ('front', 'side') for name in PANEL_FILES}

HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>自拍裁剪 · UFF 对照</title><link rel="icon" href="data:,"><script type="importmap">{"imports":{"three":"/vendor/three.module.js"}}</script><style>
:root{font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif;color:#e7eee8;background:#101914;color-scheme:dark}*{box-sizing:border-box}body{margin:0}header{padding:18px 4vw;border-bottom:1px solid #ffffff1c;display:flex;justify-content:space-between;gap:20px;font-size:13px}main{max-width:1900px;margin:auto;padding:25px 4vw}h1{font-size:28px;font-weight:550;margin:0 0 9px}h2{font-size:14px;font-weight:500;margin:0 0 10px}p{margin:0}.muted,.note{font-size:12px;color:#a0b5a6;line-height:1.8}.heading{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:22px}select,button,a{font:inherit}select{background:#1c3023;border:1px solid #56715c;border-radius:6px;padding:10px;max-width:100%;font-size:12px;color:#e5eee7}button{cursor:pointer;background:none;border:1px solid transparent;color:#b0c2b5;border-radius:5px;padding:8px 10px;font-size:12px}button:hover,button[aria-pressed=true]{background:#bbd8bd16;border-color:#b7d2bc55;color:#eff6ee}button:disabled,select:disabled{opacity:.45;cursor:default}button:focus-visible,select:focus-visible,a:focus-visible{outline:2px solid #b1dcb8;outline-offset:3px}.layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(230px,.72fr);gap:18px}.card{background:#17261c;border:1px solid #ffffff1a;border-radius:10px;overflow:hidden}.toolbar{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;padding:12px 15px;border-bottom:1px solid #ffffff13}.group{display:flex;align-items:center;gap:5px;flex-wrap:wrap}.stage{position:relative;height:520px;max-height:65vh;min-height:350px;background:radial-gradient(ellipse at 50% 40%,#334a37,#1c3021 68%,#132018)}#viewport{position:absolute;inset:0}#loading{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);background:#14261ee8;border:1px solid #b4d0b755;border-radius:7px;padding:15px;max-width:90%;font-size:12px}.views{position:absolute;left:13px;bottom:13px;background:#102017ed;border-radius:6px}.caption{font-size:12px;color:#a0b5a5;line-height:1.8;padding:14px 17px}.photo{padding:17px}.photo a{display:block;background:#0e1912;aspect-ratio:1;border-radius:7px;overflow:hidden}.photo img{width:100%;height:100%;object-fit:contain}.photo .note{padding-top:13px}.comparisons{margin-top:20px}.scroll{overflow:auto;background:#101914;min-height:200px}.comparison-image{width:100%;min-width:960px;display:block}.empty{padding:70px 20px;text-align:center;color:#9bb3a1;font-size:13px}.legend{padding:14px 17px;color:#a5b9aa;font-size:12px;line-height:1.8;border-top:1px solid #ffffff15}.pair{display:grid;grid-template-columns:1fr 1fr;min-width:640px}.pair figure{margin:0;border-right:1px solid #ffffff18}.pair img{width:100%;display:block}.pair figcaption{padding:12px 15px;font-size:12px;color:#c1d1c3}.pair-controls{display:flex;gap:14px;flex-wrap:wrap;padding:12px 15px}.pair-controls label{font-size:12px;color:#9eb4a4;display:flex;gap:10px;align-items:center}a{color:#c0d9c5;text-decoration:none}footer{font-size:11px;color:#91aa97;line-height:1.9;margin-top:20px}[hidden]{display:none!important}@media(max-width:850px){.layout{grid-template-columns:1fr}.heading{flex-direction:column;align-items:flex-start}.stage{height:430px}.photo a{max-height:400px}.comparison-image{min-width:800px}main{padding:20px 3vw}}
</style></head><body><header><span>FaceAppearance · 自拍裁剪实验</span><span id="batch" class="muted">读取实验</span></header><main><div class="heading"><div><h1>哪种裁剪更适合扭头自拍</h1><p class="muted">三个裁剪范围 × 三种背景处理，另对照整段固定裁剪框</p></div><select id="case-select" aria-label="选择样本"></select></div><div class="layout"><section class="card"><div class="toolbar"><select id="variant-select" aria-label="选择三维版本"></select><div class="group"><button data-mode="clay" aria-pressed="true">灰模</button><button data-mode="wireframe">线框</button></div></div><div class="stage"><div id="viewport"></div><div id="loading">读取模型</div><div class="views group"><button data-view="front">正面</button><button data-view="left">左侧</button><button data-view="right">右侧</button><button id="eyes">眼部</button><button id="reset">重置</button></div></div><div class="caption" id="caption">切换版本保留观察角度。</div></section><aside class="card photo"><h2 id="photo-title">原始照片</h2><a id="photo-link" target="_blank" rel="noopener"><img id="photo" alt="参与重建的原始照片"></a><p class="note" id="photo-note"></p><p class="note" id="status"></p></aside></div><section class="card comparisons"><div class="toolbar"><div class="group"><select id="view-select" aria-label="选择对照视角"><option value="front">正面</option><option value="side">最大转角侧面</option></select><button data-panel="gray" aria-pressed="true">全脸灰模</button><button data-panel="eyes">眼部灰模</button><button data-panel="overlay">50% 叠图</button><button data-panel="inputs">输入裁剪</button></div><div class="group"><button data-layout="grid" aria-pressed="true">全部方案</button><button data-layout="pair">A/B 对照</button><a id="full-image" target="_blank" rel="noopener" class="muted">打开大图 ↗</a></div></div><div id="pair-controls" class="pair-controls" hidden><label>A<select id="pair-a" aria-label="左侧版本"></select></label><label>B<select id="pair-b" aria-label="右侧版本"></select></label></div><div class="scroll"><img id="comparison" class="comparison-image" alt="相同原照相机下的裁剪方案对照" hidden><div id="pair" class="pair" hidden><figure><figcaption id="label-a"></figcaption><img id="image-a" alt="A 版本"></figure><figure><figcaption id="label-b"></figcaption><img id="image-b" alt="B 版本"></figure></div><div id="empty" class="empty">读取对比图</div></div><div class="legend" id="legend"></div><div class="legend"><span id="details"></span> <a id="report" target="_blank" rel="noopener">实验记录 ↗</a></div></section><footer>灰模对照固定采用本次「逐帧框 1.35 · 原背景」的 PnP 相机，先还原到原照坐标再渲染；各版使用同一光照、同一裁切。输入图都参与了重建，并非独立真值。仅显示 UFF 面部曲面；不包含眼球、睫毛、完整头部和新纹理。叠图中的孔洞保留原照，请结合独立灰模判断。</footer></main><script type="module">
import * as THREE from 'three';import {FaceViewer} from '/scene.js';
const $=s=>document.querySelector(s),baseline='m135_raw',cache=new Map(),keys={gray:'fixedComparison',eyes:'fixedEyes',overlay:'fixedOverlays',inputs:'inputContact'};let viewer,manifest,selected,variant=baseline,mode='clay',panel='gray',view='front',layout='grid',pairA=baseline,pairB='m135_headGray',busy=false,preview=null;
async function request(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw new Error(`${r.status} ${await r.text()}`);return r;}
function fail(e){$('#loading').hidden=false;$('#loading').textContent=e.message||String(e);}
function entry(){return manifest?.cases.find(c=>c.caseId===selected);}
function sync(){for(const b of document.querySelectorAll('[data-mode]'))b.setAttribute('aria-pressed',String(b.dataset.mode===mode));$('#case-select').disabled=busy;$('#variant-select').disabled=busy;}
function save(){return{position:viewer.camera.position.clone(),quaternion:viewer.camera.quaternion.clone(),up:viewer.camera.up.clone(),target:viewer.controls.target.clone(),center:viewer.center.clone(),radius:viewer.radius,fitDistance:viewer.fitDistance,lightPosition:viewer.lightRig.position.clone(),lightScale:viewer.lightRig.scale.clone(),near:viewer.camera.near,far:viewer.camera.far,min:viewer.controls.minDistance,max:viewer.controls.maxDistance};}
function restore(s){viewer.camera.position.copy(s.position);viewer.camera.quaternion.copy(s.quaternion);viewer.camera.up.copy(s.up);viewer.controls.target.copy(s.target);viewer.center.copy(s.center);viewer.radius=s.radius;viewer.fitDistance=s.fitDistance;viewer.lightRig.position.copy(s.lightPosition);viewer.lightRig.scale.copy(s.lightScale);viewer.camera.near=s.near;viewer.camera.far=s.far;viewer.controls.minDistance=s.min;viewer.controls.maxDistance=s.max;viewer.camera.updateProjectionMatrix();viewer.controls.update();}
function showPhoto(){const e=entry(),index=view==='side'?e.sideViewIndex:e.frontViewIndex,p=e.photos.find(p=>p.index===index)||e.photos[0];if(!p){$('#photo').removeAttribute('src');$('#photo-link').removeAttribute('href');return;}$('#photo').src=p.url;$('#photo-link').href=p.url;$('#photo-title').textContent=`${view==='side'?'最大转角侧面':'正面'}原照 · 视图 ${p.index}`;$('#photo-note').textContent=`${p.size.join(' × ')} · ${e.label}`;}
function showPanel(){const e=entry();if(!e)return;const contact=panel==='inputs',showPair=layout==='pair'&&!contact,key=`${view}${keys[panel][0].toUpperCase()+keys[panel].slice(1)}`,url=e.comparisons[key];for(const b of document.querySelectorAll('[data-panel]'))b.setAttribute('aria-pressed',String(b.dataset.panel===panel));for(const b of document.querySelectorAll('[data-layout]')){b.setAttribute('aria-pressed',String(b.dataset.layout===layout));b.disabled=contact&&b.dataset.layout==='pair';}$('#pair-controls').hidden=!showPair;$('#pair').hidden=!showPair||!e.rendered;$('#comparison').hidden=showPair||!url;$('#empty').hidden=showPair?e.rendered:!!url;$('#empty').textContent='这个样本的固定相机图尚未生成；可先旋转上方已完成的灰模。';$('#full-image').hidden=showPair||!url;if(url){$('#comparison').src=url;$('#full-image').href=url;}if(showPair&&e.rendered){const index=view==='side'?e.sideViewIndex:e.frontViewIndex;for(const [slot,id] of [['a',pairA],['b',pairB]]){$(`#image-${slot}`).src=`/cases/${e.caseId}/single/${index}/${id}/${panel}.png`;$(`#label-${slot}`).textContent=id==='photo'?'原始照片':e.variants.find(v=>v.id===id)?.label||id;}}$('#legend').textContent=contact?'每行依次是原背景、去环境背景灰底、仅保留头部灰底。前三行为逐帧框 1.10 / 1.35 / 1.65；最后一行是整段固定框。输入灰底预览为 PNG，模型实际使用 float32 0.5。':showPair?'A/B 使用同一原照相机、同一光照和同一裁切。切换只改变重建几何；坐标漂移也会反映在叠图上。':'每行是同一裁剪范围：原照、原背景、去环境背景灰底、头部灰底。最后一行为整段固定框的两个版本。基准是逐帧框 1.35 · 原背景。';showPhoto();}
function options(select,entries,value){select.replaceChildren();for(const v of entries){const o=document.createElement('option');o.value=v.id;o.textContent=v.label;o.disabled=v.available===false;select.append(o);}select.value=entries.some(v=>v.id===value&&v.available!==false)?value:entries.find(v=>v.available!==false)?.id||'';return select.value;}
async function choose(id,kind=variant){if(busy)return;const e=manifest.cases.find(c=>c.caseId===id);if(!e)return;const saved=id===selected&&preview?save():null;selected=id;$('#case-select').value=id;const available=e.variants.filter(v=>v.available);variant=options($('#variant-select'),e.variants,kind);pairA=options($('#pair-a'),[{id:'photo',label:'原始照片'},...available],pairA);pairB=options($('#pair-b'),[{id:'photo',label:'原始照片'},...available],pairB);$('#status').textContent=`${available.length}/${e.variants.length} 个展示版本已完成${e.rendered?' · 对照已渲染':' · 对照待渲染'}。`;$('#details').textContent='固定原照坐标，避免不同裁剪框的 512 像素相机混用。';$('#report').href=`/cases/${id}/report`;showPanel();const u=new URL(location.href);u.searchParams.set('case',id);u.searchParams.set('variant',variant);history.replaceState(null,'',u);if(!available.length){preview=null;if(viewer.asset?.mesh)viewer.asset.mesh.visible=false;$('#loading').hidden=false;$('#loading').textContent='本组重建尚未完成';sync();return;}busy=true;$('#loading').hidden=false;$('#loading').textContent='读取模型';sync();try{const key=`${id}/${variant}`;if(!cache.has(key))cache.set(key,await(await request(`/cases/${key}/preview`)).json());preview=cache.get(key);await viewer.load(preview,async url=>(await request(url)).blob());if(saved)restore(saved);viewer.setMode(mode);$('#caption').textContent=`${e.label} · ${e.variants.find(v=>v.id===variant).label} · 6,986 顶点。切换版本保留观察角度；拖动旋转、滚轮缩放。`;$('#loading').hidden=true;}catch(error){preview=null;if(viewer.asset?.mesh)viewer.asset.mesh.visible=false;fail(error);}finally{busy=false;sync();}}
function eyeFocus(){if(!preview?.eyeFocus)return;viewer.setView('front');const c=new THREE.Vector3().fromArray(preview.eyeFocus.center),halfV=THREE.MathUtils.degToRad(viewer.camera.fov/2),halfH=Math.atan(Math.tan(halfV)*viewer.camera.aspect),distance=preview.eyeFocus.width/(2*Math.tan(Math.min(halfV,halfH)))*1.22;viewer.controls.target.copy(c);viewer.camera.position.copy(c).add(new THREE.Vector3(0,0,distance));viewer.controls.minDistance=Math.min(viewer.radius*.1,distance*.25);viewer.controls.update();}
try{viewer=new FaceViewer($('#viewport'),{onError:fail});viewer.setMode(mode);manifest=await(await request('/comparison')).json();$('#batch').textContent=`${manifest.cases.length} 组 · ${manifest.status||'运行中'} · 独立预览`;options($('#case-select'),manifest.cases.map(c=>({id:c.caseId,label:c.label})),null);$('#case-select').addEventListener('change',e=>choose(e.target.value));$('#variant-select').addEventListener('change',e=>choose(selected,e.target.value));$('#view-select').addEventListener('change',e=>{view=e.target.value;showPanel();});$('#pair-a').addEventListener('change',e=>{pairA=e.target.value;showPanel();});$('#pair-b').addEventListener('change',e=>{pairB=e.target.value;showPanel();});for(const b of document.querySelectorAll('[data-mode]'))b.addEventListener('click',()=>{mode=b.dataset.mode;viewer.setMode(mode);sync();});for(const b of document.querySelectorAll('[data-view]'))b.addEventListener('click',()=>viewer.setView(b.dataset.view));for(const b of document.querySelectorAll('[data-panel]'))b.addEventListener('click',()=>{panel=b.dataset.panel;showPanel();});for(const b of document.querySelectorAll('[data-layout]'))b.addEventListener('click',()=>{layout=b.dataset.layout;showPanel();});$('#eyes').addEventListener('click',eyeFocus);$('#reset').addEventListener('click',()=>viewer.reset());window.addEventListener('pagehide',()=>viewer.dispose(),{once:true});const q=new URL(location.href).searchParams;await choose(q.get('case')||manifest.cases[0]?.caseId,q.get('variant')||baseline);}catch(error){fail(error);}
</script></body></html>'''


def createApp(experiment: Path) -> FastAPI:
    experiment = Path(experiment).resolve()
    web = ROOT / 'src/faceAppearance/web'
    scripts = {'scene.js': (web / 'scene.js').read_bytes()}
    scripts.update({name: (web / 'vendor' / name).read_bytes() for name in ('three.module.js', 'three.core.js', 'OrbitControls.js')})
    cache, photos, lock = {}, {}, threading.RLock()
    app = FastAPI(title='UFF selfie crop comparison', docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1'])

    @app.middleware('http')
    async def headers(request, callNext):
        response = await callNext(request)
        response.headers.update({'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'no-store', 'Referrer-Policy': 'same-origin',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"})
        return response

    def inside(path):
        path = Path(path).resolve()
        if not path.is_relative_to(experiment):
            raise HTTPException(409, 'Artifact path outside experiment')
        return path

    def manifest():
        try:
            value = read_json(experiment / 'cropManifest.json')
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(503, 'Experiment manifest unavailable') from error
        ids = []
        for case in value['cases']:
            if not re.fullmatch(r'[A-Za-z0-9_-]+', case.get('caseId', '')):
                raise HTTPException(409, 'Unsafe case identifier')
            ids.append(case['caseId'])
            names = []
            for item in case['variants']:
                names.append(item['id'])
                if item.get('outputDirectory'):
                    inside(item['outputDirectory'])
            if len(names) != len(set(names)):
                raise HTTPException(409, 'Duplicate variant')
        if len(ids) != len(set(ids)):
            raise HTTPException(409, 'Duplicate case')
        return value

    def selected(case_id):
        entry = next((item for item in manifest()['cases'] if item['caseId'] == case_id), None)
        if entry is None:
            raise HTTPException(404, 'Unknown case')
        return entry

    def preview_directory(value=None):
        value = manifest() if value is None else value
        return inside(value.get('previewDirectory', experiment / 'preview'))

    def photo_inputs(entry):
        baseline = variant_entries(entry).get(BASELINE)
        path = Path(baseline['outputDirectory']) / 'inputs.json' if baseline and baseline.get('outputDirectory') else None
        if path is not None and path.is_file():
            return read_json(path)
        if entry.get('sourceDirectory'):
            path = Path(entry['sourceDirectory']).resolve() / 'inputs.json'
            if path.is_relative_to(ROOT / 'runs') and path.is_file():
                return read_json(path)
        return []

    def variants(entry):
        entries = variant_entries(entry)
        return [{'id': name, 'label': LABELS[name], 'status': entries.get(name, {}).get('status', 'pending'),
                 'available': name in entries and variant_complete(entries[name])} for name in VARIANTS]

    def render_report(case_id):
        path = preview_directory() / case_id / 'diagnostics.json'
        return read_json(path) if path.is_file() else {}

    def loaded(case_id, name):
        entry = selected(case_id)
        item = variant_entries(entry).get(name)
        if item is None or not variant_complete(item):
            raise HTTPException(404, 'Variant unavailable')
        source = inside(item['outputDirectory'])
        gp, rp = source / 'geometry.npz', source / 'result.json'
        key = (case_id, name, gp.stat().st_mtime_ns, rp.stat().st_mtime_ns)
        with lock:
            if key not in cache:
                if read_json(rp).get('status') != 'succeeded':
                    raise HTTPException(409, 'Variant did not succeed')
                with np.load(gp, allow_pickle=False) as arrays:
                    geometry = {k: arrays[k] for k in ('vertices', 'faces', 'uvs')}
                baseline = variant_entries(entry).get(BASELINE)
                bp = Path(baseline['outputDirectory']) / 'geometry.npz' if baseline else None
                if name != BASELINE and bp is not None and bp.is_file():
                    with np.load(bp, allow_pickle=False) as arrays:
                        if any(not np.array_equal(geometry[k], arrays[k]) for k in ('faces', 'uvs')):
                            raise HTTPException(409, 'Variant topology or UV changed')
                try:
                    cache[key] = preview_payload(geometry, case_id, name)
                except ValueError as error:
                    raise HTTPException(409, str(error)) from error
            return cache[key]

    @app.get('/', response_class=HTMLResponse)
    def index():
        return HTMLResponse(HTML)

    @app.get('/comparison')
    def comparison():
        value, entries = manifest(), []
        preview_root = preview_directory(value)
        for entry in value['cases']:
            case_id, inputs = entry['caseId'], photo_inputs(entry)
            output = preview_root / case_id
            report = render_report(case_id)
            compared = {name: f'/cases/{case_id}/comparisons/{name}.png' for name in COMPARISONS if (output / f'{name}.png').is_file()}
            entries.append({'caseId': case_id, 'label': entry.get('label', case_id), 'status': entry.get('status'),
                'frontViewIndex': int(entry.get('referenceViewIndex', 0)), 'sideViewIndex': report.get('selectedSideView'),
                'rendered': report.get('status') == 'rendered', 'variants': variants(entry), 'comparisons': compared,
                'photos': [{'index': i, 'url': f'/cases/{case_id}/photos/{i}', 'size': item['orientedSize']} for i, item in enumerate(inputs)]})
        return {'status': value.get('status'), 'cases': entries}

    @app.get('/scene.js')
    def scene():
        return Response(scripts['scene.js'], media_type='text/javascript')

    @app.get('/vendor/{filename}')
    def vendor(filename: str):
        if filename not in scripts or filename == 'scene.js':
            raise HTTPException(404, 'Unknown vendor script')
        return Response(scripts[filename], media_type='text/javascript')

    @app.get('/cases/{case_id}/{variant}/preview')
    def preview(case_id: str, variant: str):
        return loaded(case_id, variant)

    @app.get('/cases/{case_id}/photos/{index}')
    def photo(case_id: str, index: int):
        inputs = photo_inputs(selected(case_id))
        if not 0 <= index < len(inputs):
            raise HTTPException(404, 'Unknown photograph')
        item = inputs[index]
        with lock:
            if item['sha256'] not in photos:
                if digest(item['path']) != item['sha256']:
                    raise HTTPException(409, 'Photo changed')
                with Image.open(item['path']) as image:
                    image = ImageOps.exif_transpose(image).convert('RGB')
                output = io.BytesIO()
                image.save(output, format='PNG')
                photos[item['sha256']] = output.getvalue()
            return Response(photos[item['sha256']], media_type='image/png')

    def png(path):
        path = inside(path)
        if not path.is_file():
            raise HTTPException(404, 'Image not rendered')
        return Response(path.read_bytes(), media_type='image/png')

    @app.get('/cases/{case_id}/comparisons/{filename}')
    def comparison_image(case_id: str, filename: str):
        selected(case_id)
        if filename not in {name + '.png' for name in COMPARISONS}:
            raise HTTPException(404, 'Unknown comparison')
        return png(preview_directory() / case_id / filename)

    @app.get('/cases/{case_id}/single/{index}/{variant}/{mode}.png')
    def single(case_id: str, index: int, variant: str, mode: str):
        selected(case_id)
        report = render_report(case_id)
        if (mode not in PASSES or variant not in (*report.get('variantIds', []), 'photo')
                or index not in [v['viewIndex'] for v in report.get('views', [])]):
            raise HTTPException(404, 'Unknown pair panel')
        return png(preview_directory() / case_id / f'view{index:02d}' / f'{variant}_{mode}.png')

    @app.get('/cases/{case_id}/report')
    def report(case_id: str):
        return {'case': selected(case_id), 'preview': render_report(case_id)}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=ROOT / 'runs/uffSelfieCropStudy20261012')
    parser.add_argument('--host', choices=['127.0.0.1'], default='127.0.0.1')
    parser.add_argument('--port', type=int, default=18091)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be 1..65535')
    uvicorn.run(createApp(args.experiment), host=args.host, port=args.port, access_log=False)


if __name__ == '__main__':
    main()
