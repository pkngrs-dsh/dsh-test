"""Read-only localhost background-ablation viewer, default port 18090.

Only freshly reconstructed native UFF meshes are shown. Variant switches keep
the interactive camera. Saved image panels distinguish frozen-A cameras from
each variant's own PnP cameras; no texture or old geometry is substituted.
"""

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
sys.path.insert(0, str(ROOT / "src"))
from renderUffBackgroundAblation import (PANEL_FILES, VARIANTS, digest, read_json,
    contact_paths, variant_complete, variant_entries)
from serveUffMultiCase import nativeEyeFocus


LABELS = {"original": "A 原图", "backgroundGray": "B 背景灰",
          "headGray": "C 头部外灰", "headBlack": "D 头部外黑"}
COMPARISONS = {f"{prefix}{name[0].upper() + name[1:]}" for prefix in ("front", "side", "all") for name in PANEL_FILES}
COMPARISONS.update(("inputContact", "maskContact"))


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UFF 背景消融对照</title><link rel="icon" href="data:,"><script type="importmap">{"imports":{"three":"/vendor/three.module.js"}}</script><style>
:root{font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif;color:#e5eee9;background:#101916;color-scheme:dark}*{box-sizing:border-box}body{margin:0}header{padding:19px 4vw;border-bottom:1px solid #ffffff19;display:flex;justify-content:space-between;gap:20px;font-size:14px}main{max-width:1840px;margin:auto;padding:24px 4vw}h1{font-size:28px;font-weight:550;margin:0 0 8px}h2{font-size:15px;font-weight:500;margin:0 0 10px}p{margin:0}.muted,.note{font-size:12px;color:#9db3a7;line-height:1.7}.heading{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:22px}select,button,a{font:inherit}select{padding:11px;background:#1d3026;border:1px solid #56785c;border-radius:6px;min-width:230px}button{cursor:pointer;background:none;border:1px solid transparent;color:#a8bcb0;border-radius:5px;padding:8px 10px;font-size:12px}button:hover,button[aria-pressed=true]{color:#e3f4e7;border-color:#b5d4b844;background:#c1dbbf10}button:disabled{opacity:.35;cursor:default}button:focus-visible,a:focus-visible,select:focus-visible{outline:2px solid #b2dfc1;outline-offset:3px}.layout{display:grid;grid-template-columns:minmax(0,2fr) minmax(245px,.85fr);gap:18px}.card{background:#17251e;border:1px solid #ffffff19;border-radius:11px;overflow:hidden}.toolbar{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;padding:12px 15px;border-bottom:1px solid #ffffff13}.group{display:flex;gap:4px;flex-wrap:wrap}.stage{position:relative;height:540px;max-height:62vh;min-height:360px;background:radial-gradient(ellipse at 50% 40%,#354d3c,#1b3023 67%,#14241b)}#viewport{position:absolute;inset:0}#loading{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);background:#12221dea;border:1px solid #bedac133;border-radius:7px;padding:15px 20px;font-size:13px;max-width:85%;text-align:center}.views{position:absolute;left:14px;bottom:14px;background:#112419e8;border-radius:6px}.caption{padding:14px 17px;font-size:12px;line-height:1.7;color:#a6bbac}.photo{padding:17px}.photo>a{display:block;background:#0d1812;border-radius:6px;overflow:hidden;aspect-ratio:1}.photo>a img{width:100%;height:100%;object-fit:contain}.thumbs{display:flex;gap:8px;margin:11px 0}.thumbs button{padding:2px;width:23%;aspect-ratio:1;overflow:hidden}.thumbs img{width:100%;height:100%;object-fit:cover}.comparisons{margin-top:20px}.scroll{overflow-x:auto;background:#101916;min-height:180px}.comparison-image{display:block;width:100%;min-width:1080px;height:auto}.empty{padding:70px 20px;text-align:center;color:#99b5a3;font-size:13px}.legend,.details{padding:15px 18px;font-size:12px;line-height:1.8;color:#a8beaf;border-top:1px solid #ffffff10}.details{display:flex;gap:20px;align-items:center;justify-content:space-between;flex-wrap:wrap}.details a,.toolbar a{color:#c4ddcb;text-decoration:none}.legend.own{background:#5b452526;color:#e2c491}.view-selector{min-width:140px;padding:8px;font-size:12px}.statuses{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:9px}.statuses span{font-size:11px;color:#a8beaf;border:1px solid #ffffff19;border-radius:4px;padding:4px 8px}.statuses .missing{color:#d2b88f}footer{padding:18px 0;color:#88a190;font-size:11px;line-height:1.9}@media(max-width:900px){.layout{grid-template-columns:1fr}.heading{align-items:flex-start;flex-direction:column}.stage{height:480px}.photo>a{max-height:400px}.photo>a img{max-height:400px}.thumbs{max-width:400px}.comparison-image{min-width:900px}main{padding:20px 3vw}}
</style></head><body><header><span>FaceAppearance · UFF 背景实验</span><span id="batch" class="muted">读取实验</span></header><main><div class="heading"><div><h1>背景怎样影响重建</h1><p class="muted">四种输入处理独立重建 · 同一套 6,986 顶点 · 灰模比较</p></div><select id="case-select" aria-label="选择实验组"></select></div><div class="layout"><section class="card"><div class="toolbar"><div class="group" id="variants"></div><div class="group"><button data-mode="clay" aria-pressed="true">灰模</button><button data-mode="wireframe" aria-pressed="false">线框</button></div></div><div class="stage"><div id="viewport"></div><div id="loading">读取模型中</div><div class="views group"><button data-view="front">正面</button><button data-view="left">左侧</button><button data-view="right">右侧</button><button id="eyes">眼部</button><button id="reset">重置</button></div></div><div class="caption" id="caption">切换 A/B/C/D 保留观察角度。</div></section><aside class="card photo"><h2 id="photo-title">原始照片</h2><a id="photo-link" target="_blank" rel="noopener"><img id="photo" alt="当前实验组原始照片"></a><div class="thumbs" id="thumbs"></div><div class="statuses" id="statuses"></div><p class="note" id="photo-note"></p></aside></div><section class="card comparisons"><div class="toolbar"><div class="group"><select id="view-select" class="view-selector" aria-label="选择照片视图"><option value="front">正面 · 视图 0</option><option value="side">最大转角侧面</option><option value="all">全部视图</option></select><button data-panel="fixedEyes" aria-pressed="true">固定 A · 眼部灰模</button><button data-panel="fixedComparison">固定 A · 全脸灰模</button><button data-panel="fixedOverlays">固定 A · 50% 叠图</button><button data-panel="ownOverlays">各自 PnP · 50% 叠图</button><button data-panel="ownEyes">各自 PnP · 眼部叠图</button><button data-panel="inputContact">输入处理</button><button data-panel="maskContact">掩码</button></div><a id="full-image" class="muted" target="_blank" rel="noopener">打开大图 ↗</a></div><div class="scroll"><img id="comparison" class="comparison-image" alt="背景消融几何对照" hidden><div class="empty" id="comparison-empty">读取对比图</div></div><div class="legend" id="legend"></div><div class="details"><span id="details"></span><a id="report" target="_blank" rel="noopener">实验记录 ↗</a></div></section><footer>固定 A 比较使用本次重新推理的 A 几何与 PnP 相机，所有版本使用相同光照和裁切。每张原照都参与过重建。叠图中的眼部孔洞会保留原照，请结合独立灰模判断。仅显示 UFF 面部曲面；不包含眼球、睫毛、完整头部或新纹理。</footer></main><script type="module">
import * as THREE from 'three';import {FaceViewer} from '/scene.js';
const $=selector=>document.querySelector(selector);let viewer,manifest,selected,variant='original',mode='clay',panel='fixedEyes',view='front',busy=false,preview=null;const cache=new Map();
async function request(url){const result=await fetch(url,{cache:'no-store'});if(!result.ok)throw new Error(`${result.status} ${await result.text()}`);return result;}
function fail(error){$('#loading').hidden=false;$('#loading').textContent=error.message||String(error);}
function sync(){document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));document.querySelectorAll('[data-variant]').forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.variant===variant));b.disabled=busy||b.dataset.available!=='true';});$('#case-select').disabled=busy;}
function showPhoto(entry,index){const p=entry.photos.find(p=>p.index===index)||entry.photos[0];if(!p){$('#photo').removeAttribute('src');$('#photo-link').removeAttribute('href');$('#photo-note').textContent='原照元数据尚未生成。';return;}$('#photo').src=p.url;$('#photo-link').href=p.url;$('#photo-title').textContent=`${p.index===entry.frontViewIndex?'正面':'输入'}原照 · 视图 ${p.index}`;$('#photo-note').textContent=`${p.size.join(' × ')} · ${entry.label}`;}
function showPanel(){const entry=manifest.cases.find(c=>c.caseId===selected),contact=panel.endsWith('Contact'),key=contact?panel:`${view}${panel[0].toUpperCase()+panel.slice(1)}`,url=entry?.comparisons?.[key];document.querySelectorAll('[data-panel]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.panel===panel)));$('#view-select').disabled=contact;$('#comparison').hidden=!url;$('#comparison-empty').hidden=!!url;$('#comparison-empty').textContent='该对比图尚未生成；可先旋转上方已完成的灰模。';$('#full-image').hidden=!url;if(url){$('#comparison').src=url;$('#full-image').href=url;}const own=panel.startsWith('own');$('#legend').classList.toggle('own',own);$('#legend').textContent=contact?(panel==='maskContact'?'语义标签、保护面部区域、前景与头部 alpha；保护区域来自冻结的原始采样掩码。':'保存的输入处理：A 原图，B 背景置灰，C 头部外置灰，D 头部外置黑。'):own?'各版本使用自己的 PnP 相机。此处叠图同时包含几何变化与相机变化；评价几何时请结合「固定 A」比较。':'A 原图 · B 背景灰 · C 头部外灰 · D 头部外黑。四个版本固定使用本次 A 的 PnP 相机、同一光照和同一裁切。';}
function state(){return{position:viewer.camera.position.clone(),quaternion:viewer.camera.quaternion.clone(),up:viewer.camera.up.clone(),target:viewer.controls.target.clone(),center:viewer.center.clone(),radius:viewer.radius,fitDistance:viewer.fitDistance,lightPosition:viewer.lightRig.position.clone(),lightScale:viewer.lightRig.scale.clone(),near:viewer.camera.near,far:viewer.camera.far,min:viewer.controls.minDistance,max:viewer.controls.maxDistance};}
function restore(s){viewer.camera.position.copy(s.position);viewer.camera.quaternion.copy(s.quaternion);viewer.camera.up.copy(s.up);viewer.controls.target.copy(s.target);viewer.center.copy(s.center);viewer.radius=s.radius;viewer.fitDistance=s.fitDistance;viewer.lightRig.position.copy(s.lightPosition);viewer.lightRig.scale.copy(s.lightScale);viewer.camera.near=s.near;viewer.camera.far=s.far;viewer.controls.minDistance=s.min;viewer.controls.maxDistance=s.max;viewer.camera.updateProjectionMatrix();viewer.controls.update();}
async function choose(id,kind=variant){if(busy)return;const entry=manifest.cases.find(c=>c.caseId===id);if(!entry)return;const same=id===selected&&!!preview,saved=same?state():null;selected=id;const available=entry.variants.filter(v=>v.available);variant=available.some(v=>v.id===kind)?kind:available[0]?.id||'original';$('#case-select').value=id;$('#variants').replaceChildren();$('#statuses').replaceChildren();for(const v of entry.variants){const b=document.createElement('button');b.textContent=v.label;b.dataset.variant=v.id;b.dataset.available=String(v.available);b.addEventListener('click',()=>choose(id,v.id));$('#variants').append(b);const span=document.createElement('span');span.textContent=`${v.label} · ${v.available?'已完成':v.status||'待运行'}`;span.classList.toggle('missing',!v.available);$('#statuses').append(span);}$('#thumbs').replaceChildren();for(const p of entry.photos){const b=document.createElement('button'),img=document.createElement('img');img.src=p.url;img.alt=`原照视图 ${p.index}`;b.setAttribute('aria-label',img.alt);b.append(img);b.addEventListener('click',()=>{showPhoto(entry,p.index);view=`view${String(p.index).padStart(2,'0')}`;$('#view-select').value=view;showPanel();});$('#thumbs').append(b);}$('#view-select').replaceChildren();for(const [value,label] of [['front',`正面 · 视图 ${entry.frontViewIndex}`],['side',`最大转角侧面${entry.sideViewIndex===null?'':` · 视图 ${entry.sideViewIndex}`}`],['all','全部视图'],...entry.photos.map(p=>[`view${String(p.index).padStart(2,'0')}`,`视图 ${p.index}`])]){const o=document.createElement('option');o.value=value;o.textContent=label;$('#view-select').append(o);}if(![...$('#view-select').options].some(o=>o.value===view))view='front';$('#view-select').value=view;showPhoto(entry,entry.frontViewIndex);showPanel();$('#details').textContent=`${available.length}/4 个可见版本已完成${entry.rendered?' · 固定相机对照已生成':' · 对照渲染待完成'}。A/B/C/D 均来自本次独立重建。`;$('#report').href=`/cases/${id}/report`;const u=new URL(location.href);u.searchParams.set('case',id);u.searchParams.set('variant',variant);history.replaceState(null,'',u);if(!available.length){preview=null;if(viewer.asset?.mesh)viewer.asset.mesh.visible=false;$('#loading').hidden=false;$('#loading').textContent='本组重建尚未完成';$('#caption').textContent='等待本组原生 UFF 几何。';sync();return;}busy=true;$('#loading').hidden=false;$('#loading').textContent='读取模型中';sync();try{const key=`${id}/${variant}`;if(!cache.has(key))cache.set(key,await(await request(`/cases/${id}/${variant}/preview`)).json());preview=cache.get(key);await viewer.load(preview,async url=>(await request(url)).blob());if(saved)restore(saved);viewer.asset.mesh.userData.units='uncalibratedCanonical';viewer.setMode(mode);$('#caption').textContent=`${entry.label} · ${entry.variants.find(v=>v.id===variant).label} · ${preview.counts.sourceVertices.toLocaleString()} 顶点。切换版本保留观察角度；拖动旋转、滚轮缩放。`;$('#loading').hidden=true;}catch(e){preview=null;if(viewer.asset?.mesh)viewer.asset.mesh.visible=false;fail(e);}finally{busy=false;sync();}}
function eyeFocus(){if(!preview?.eyeFocus)return;viewer.setView('front');const c=new THREE.Vector3().fromArray(preview.eyeFocus.center),width=preview.eyeFocus.width,halfV=THREE.MathUtils.degToRad(viewer.camera.fov/2),halfH=Math.atan(Math.tan(halfV)*viewer.camera.aspect),distance=width/(2*Math.tan(Math.min(halfV,halfH)))*1.22;viewer.controls.target.copy(c);viewer.camera.position.copy(c).add(new THREE.Vector3(0,0,distance));viewer.controls.minDistance=Math.min(viewer.radius*.1,distance*.25);viewer.controls.update();}
try{viewer=new FaceViewer($('#viewport'),{onError:fail});viewer.setMode(mode);manifest=await(await request('/comparison')).json();$('#batch').textContent=`${manifest.cases.length} 组 · ${manifest.status||'运行中'} · 独立预览`;for(const c of manifest.cases){const o=document.createElement('option');o.value=c.caseId;o.textContent=c.label;$('#case-select').append(o);}$('#case-select').addEventListener('change',e=>choose(e.target.value));$('#view-select').addEventListener('change',e=>{view=e.target.value;showPanel();});document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.mode;viewer.setMode(mode);sync();}));document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>viewer.setView(b.dataset.view)));document.querySelectorAll('[data-panel]').forEach(b=>b.addEventListener('click',()=>{panel=b.dataset.panel;showPanel();}));$('#eyes').addEventListener('click',eyeFocus);$('#reset').addEventListener('click',()=>viewer.reset());window.addEventListener('pagehide',()=>viewer.dispose(),{once:true});const q=new URL(location.href).searchParams;await choose(q.get('case')||manifest.cases[0]?.caseId,q.get('variant')||'original');if(!manifest.cases.length)fail(new Error('实验清单中没有组。'));}catch(e){fail(e);}
</script></body></html>'''


def preview_payload(geometry, case_id, variant):
    vertices, faces, uvs = (np.asarray(geometry[key]) for key in ("vertices", "faces", "uvs"))
    if (vertices.shape != (6986, 3) or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0
            or faces.dtype.kind not in "iu" or faces.min() < 0 or faces.max() >= len(vertices)
            or uvs.shape != (6986, 2) or not np.isfinite(uvs).all()):
        raise ValueError("Invalid native UFF 6986 mesh")
    normals = np.zeros_like(vertices)
    face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    np.divide(normals, lengths[:, None], out=normals, where=lengths[:, None] > 0)
    return {"schemaVersion": 1, "jobId": case_id, "templateId": "nativeUvFaceFusionCanonical6986",
        "counts": {"sourceVertices": len(vertices), "renderVertices": len(vertices), "faces": len(faces), "materials": 1},
        "coordinates": {"lengthUnit": "uncalibratedCanonical", "uvOrigin": "vDown", "axisTransform": "identity"},
        "mesh": {"positions": vertices.ravel().tolist(), "normals": normals.ravel().tolist(), "uvs": uvs.ravel().tolist(),
                 "indices": faces.ravel().tolist(), "sourceVertexIds": list(range(len(vertices))),
                 "groups": [{"start": 0, "count": int(faces.size), "materialIndex": 0}]},
        "materials": [{"index": 0, "name": variant, "roughnessFactor": .85, "metallicFactor": 0,
                       "alphaMode": "OPAQUE", "alphaCutoff": .5, "baseColorFactor": [.5, .5, .5, 1]}],
        "eyeFocus": nativeEyeFocus(vertices, faces)}


def createApp(experiment: Path) -> FastAPI:
    experiment = Path(experiment).resolve()
    web = ROOT / "src/faceAppearance/web"
    scripts = {"scene.js": (web / "scene.js").read_bytes()}
    scripts.update({name: (web / "vendor" / name).read_bytes() for name in ("three.module.js", "three.core.js", "OrbitControls.js")})
    cache, photos, lock = {}, {}, threading.RLock()
    app = FastAPI(title="UFF background ablation", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

    @app.middleware("http")
    async def headers(request, callNext):
        response = await callNext(request)
        response.headers.update({"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store", "Referrer-Policy": "same-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"})
        return response

    def inside(path, root=experiment):
        path = Path(path).resolve()
        if not path.is_relative_to(root):
            raise HTTPException(409, "Artifact path is outside this experiment")
        return path

    def manifest():
        try:
            value = read_json(experiment / "backgroundManifest.json")
        except (FileNotFoundError, ValueError) as error:
            raise HTTPException(503, "Experiment manifest unavailable") from error
        ids = []
        for case in value["cases"]:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", case.get("caseId", "")):
                raise HTTPException(409, "Invalid case identifier")
            ids.append(case["caseId"])
            names = []
            for variant in case["variants"]:
                names.append(variant["id"])
                inside(variant["outputDirectory"])
            if len(names) != len(set(names)):
                raise HTTPException(409, "Duplicate variant identifier")
            for contact in contact_paths(case).values():
                inside(contact)
        if len(ids) != len(set(ids)):
            raise HTTPException(409, "Duplicate case identifier")
        return value

    def selected(case_id):
        entry = next((case for case in manifest()["cases"] if case["caseId"] == case_id), None)
        if entry is None:
            raise HTTPException(404, "Unknown case")
        return entry

    def preview_directory(value=None):
        value = manifest() if value is None else value
        return inside(value.get("previewDirectory", experiment / "preview"))

    def photo_inputs(entry):
        original = variant_entries(entry).get("original")
        path = Path(original["outputDirectory"]) / "inputs.json" if original else None
        if path is not None and path.is_file():
            return read_json(path)
        if entry.get("sourceDirectory"):
            path = Path(entry["sourceDirectory"]).resolve() / "inputs.json"
            if path.is_relative_to(ROOT / "runs") and path.is_file():
                return read_json(path)
        return []

    def variants(entry):
        entries = variant_entries(entry)
        return [{"id": name, "label": LABELS[name], "status": entries.get(name, {}).get("status", "pending"),
                 "available": name in entries and variant_complete(entries[name])} for name in VARIANTS]

    def loaded(case_id, name):
        entry = selected(case_id)
        item = variant_entries(entry).get(name)
        if item is None or not variant_complete(item):
            raise HTTPException(404, "Variant is not available")
        source = inside(item["outputDirectory"])
        geometry_path, result_path = source / "geometry.npz", source / "result.json"
        key = (case_id, name, geometry_path.stat().st_mtime_ns, result_path.stat().st_mtime_ns)
        with lock:
            if key not in cache:
                report = read_json(result_path)
                if report.get("status") != "succeeded":
                    raise HTTPException(409, "Variant result is not successful")
                with np.load(geometry_path, allow_pickle=False) as arrays:
                    geometry = {field: arrays[field] for field in ("vertices", "faces", "uvs")}
                original_item = variant_entries(entry).get("original")
                original_path = Path(original_item["outputDirectory"]) / "geometry.npz" if original_item else None
                if name != "original" and original_path is not None and original_path.is_file():
                    with np.load(original_path, allow_pickle=False) as original:
                        if any(not np.array_equal(geometry[field], original[field]) for field in ("faces", "uvs")):
                            raise HTTPException(409, "Variant topology or UVs changed")
                try:
                    cache[key] = preview_payload(geometry, case_id, name)
                except ValueError as error:
                    raise HTTPException(409, str(error)) from error
            return cache[key]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(HTML)

    @app.get("/comparison")
    def comparison():
        value, entries = manifest(), []
        preview_root = preview_directory(value)
        for entry in value["cases"]:
            case_id = entry["caseId"]
            inputs = photo_inputs(entry)
            output = preview_root / case_id
            report_path = output / "diagnostics.json"
            report = read_json(report_path) if report_path.is_file() else {}
            compared = {name: f"/cases/{case_id}/comparisons/{name}.png" for name in COMPARISONS
                        if (output / f"{name}.png").is_file()}
            for index in range(len(inputs)):
                view = f"view{index:02d}"
                compared.update({f"{view}{name[0].upper() + name[1:]}": f"/cases/{case_id}/views/{index}/{name}.png"
                                 for name in PANEL_FILES if (output / view / f"{name}.png").is_file()})
            for name in contact_paths(entry):
                if name not in compared:
                    compared[name] = f"/cases/{case_id}/prepared/{name}"
            entries.append({"caseId": case_id, "label": entry.get("label", case_id), "status": entry.get("status"),
                "frontViewIndex": int(entry.get("referenceViewIndex", 0)), "sideViewIndex": report.get("selectedSideView"),
                "rendered": report.get("status") == "rendered", "variants": variants(entry), "comparisons": compared,
                "photos": [{"index": index, "url": f"/cases/{case_id}/photos/{index}", "size": item["orientedSize"]}
                           for index, item in enumerate(inputs)]})
        return {"status": value.get("status"), "cases": entries}

    @app.get("/scene.js")
    def scene():
        return Response(scripts["scene.js"], media_type="text/javascript")

    @app.get("/vendor/{filename}")
    def vendor(filename: str):
        if filename not in scripts or filename == "scene.js":
            raise HTTPException(404, "Unknown vendor script")
        return Response(scripts[filename], media_type="text/javascript")

    @app.get("/cases/{case_id}/{variant}/preview")
    def preview(case_id: str, variant: str):
        return loaded(case_id, variant)

    @app.get("/cases/{case_id}/photos/{index}")
    def photo(case_id: str, index: int):
        inputs = photo_inputs(selected(case_id))
        if not 0 <= index < len(inputs):
            raise HTTPException(404, "Unknown photograph")
        item = inputs[index]
        with lock:
            if item["sha256"] not in photos:
                if digest(item["path"]) != item["sha256"]:
                    raise HTTPException(409, "Source photograph changed")
                with Image.open(item["path"]) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                output = io.BytesIO()
                image.save(output, format="PNG")
                photos[item["sha256"]] = output.getvalue()
            return Response(photos[item["sha256"]], media_type="image/png")

    def png(path):
        path = inside(path)
        if not path.is_file():
            raise HTTPException(404, "Comparison is not rendered")
        return Response(path.read_bytes(), media_type="image/png")

    @app.get("/cases/{case_id}/comparisons/{filename}")
    def comparison_image(case_id: str, filename: str):
        selected(case_id)
        if filename not in {name + ".png" for name in COMPARISONS}:
            raise HTTPException(404, "Unknown comparison")
        return png(preview_directory() / case_id / filename)

    @app.get("/cases/{case_id}/views/{index}/{filename}")
    def view_image(case_id: str, index: int, filename: str):
        entry = selected(case_id)
        if filename not in {name + ".png" for name in PANEL_FILES} or not 0 <= index < len(photo_inputs(entry)):
            raise HTTPException(404, "Unknown view comparison")
        return png(preview_directory() / case_id / f"view{index:02d}" / filename)

    @app.get("/cases/{case_id}/prepared/{name}")
    def input_contact(case_id: str, name: str):
        entry = selected(case_id)
        contacts = contact_paths(entry)
        if name not in contacts:
            raise HTTPException(404, "Input contact sheet is not available")
        path = inside(contacts[name])
        with Image.open(path) as image:
            output = io.BytesIO()
            image.convert("RGB").save(output, format="PNG")
        return Response(output.getvalue(), media_type="image/png")

    @app.get("/cases/{case_id}/report")
    def report(case_id: str):
        entry = selected(case_id)
        path = preview_directory() / case_id / "diagnostics.json"
        return {"case": entry, "preview": read_json(path) if path.is_file() else None}

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/uffBackgroundAblation20261012")
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18090)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be 1..65535")
    uvicorn.run(createApp(args.experiment), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
