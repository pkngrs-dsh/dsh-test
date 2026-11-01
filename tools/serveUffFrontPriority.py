"""Read-only localhost A/B native-UFF refinement viewer, independent of port 18088.

Interactive meshes share coordinates and retain the camera across variants.
Offline photo comparisons use the exact same saved camera for A/B/C/D.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
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
from faceAppearance.headAssembly import decodeColor
from renderUffFrontPriority import digest, read_json
from serveUffMultiCase import nativeEyeFocus


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UFF 正面精修对照</title><link rel="icon" href="data:,"><script type="importmap">{"imports":{"three":"/vendor/three.module.js"}}</script><style>
:root{font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif;color:#e4ede9;background:#111916;color-scheme:dark}*{box-sizing:border-box}body{margin:0}header{padding:19px 4vw;border-bottom:1px solid #ffffff19;display:flex;justify-content:space-between;gap:20px;font-size:14px}main{max-width:1800px;margin:auto;padding:24px 4vw}h1{font-size:28px;font-weight:550;margin:0 0 8px}h2{font-size:15px;font-weight:500;margin:0 0 10px}.muted,.note{font-size:12px;color:#9db3a7;line-height:1.7}.heading{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:22px}select,button,a{font:inherit}select{padding:11px;background:#1d3026;border:1px solid #56785c;border-radius:6px;min-width:240px}button{cursor:pointer;background:none;border:1px solid transparent;color:#a7bfb0;border-radius:5px;padding:8px 10px;font-size:12px}button:hover,button[aria-pressed=true]{color:#e4f4e6;background:#c4e2bd14;border-color:#a1cfaa50}button:disabled{opacity:.3;cursor:default}button:focus-visible,a:focus-visible,select:focus-visible{outline:2px solid #c6e6c2;outline-offset:3px}.layout{display:grid;grid-template-columns:minmax(0,1.8fr) minmax(250px,.75fr);gap:18px}.card{border:1px solid #ffffff1c;background:#17251e;border-radius:10px;overflow:hidden}.toolbar{padding:10px 14px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:10px;border-bottom:1px solid #ffffff14}.group{display:flex;flex-wrap:wrap;gap:4px}.stage{height:560px;max-height:65vh;min-height:360px;position:relative;background:radial-gradient(ellipse at 50% 42%,#3b4d40,#1b3024 70%)}#viewport{position:absolute;inset:0}#loading{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);padding:15px;background:#102319e8;border-radius:8px;font-size:12px}.views{position:absolute;bottom:14px;left:14px;background:#12251bdd;border-radius:7px}.caption{padding:14px 17px;font-size:12px;line-height:1.8;color:#a8c0ae}.photo{padding:16px}.photo a{display:block;aspect-ratio:1;max-height:470px;background:#0c1711;border-radius:6px;overflow:hidden}.photo img{width:100%;height:100%;object-fit:contain}.thumbs{display:flex;gap:6px;margin-top:8px}.thumbs button{padding:4px}.thumbs img{height:55px;width:44px;object-fit:cover;border-radius:3px}.comparisons{margin-top:22px}.comparison-image{width:100%;display:block;min-width:700px;background:#111b15}.scroll{overflow-x:auto}.empty{padding:30px;color:#adc2b4;font-size:13px}.details{padding:14px 17px;font-size:12px;color:#a8c0ae;line-height:1.8}.details strong{color:#dfebe2;font-weight:500}a{color:#cee5d0;text-decoration:none}a:hover{text-decoration:underline}.legend{padding:12px 17px;font-size:12px;color:#9ab5a2;line-height:1.8}footer{margin-top:15px;font-size:11px;color:#86a490;line-height:1.8}@media(max-width:850px){.layout{grid-template-columns:1fr}.heading{display:block}select{margin-top:15px;width:100%}.photo a{max-height:360px}.stage{height:480px}header .muted{display:none}}@media(max-width:500px){main{padding:20px 3vw}.toolbar{padding:8px}.stage{min-height:320px}.group button{padding:8px 6px}h1{font-size:24px}}
</style></head><body><header><span>FaceAppearance · UFF 正面精修</span><span class="muted" id="batch"></span></header><main><section class="heading"><div><h1>先把正面轮廓对齐</h1><p class="muted">同一组照片、同一套 UFF 拓扑，对比精修前后的眼睑和面部。</p></div><select id="case-select" aria-label="选择照片组"></select></section><div class="layout"><section class="card"><div class="toolbar"><div class="group" id="variants"></div><div class="group"><button data-mode="clay" aria-pressed="true">灰模</button><button data-mode="unlit">基色</button><button data-mode="textured">照明</button><button data-mode="wireframe">线框</button></div></div><div class="stage"><div id="viewport"></div><div id="loading">读取模型中</div><div class="views group"><button data-view="front">正面</button><button data-view="left">左侧</button><button data-view="right">右侧</button><button id="eyes">眼部</button><button id="reset">重置</button></div></div><div class="caption" id="caption">切换 A/B 会保留观察角度。</div></section><aside class="card photo"><h2 id="photo-title">正面原照</h2><a id="photo-link" target="_blank" rel="noopener"><img id="photo" alt="当前组原始照片"></a><div class="thumbs" id="thumbs"></div><p class="note" id="photo-note"></p></aside></div><section class="card comparisons"><div class="toolbar"><div class="group"><button data-panel="frontEyes" aria-pressed="true">正面眼部</button><button data-panel="frontComparison">正面全脸</button><button data-panel="sideComparison">侧面全脸</button><button data-panel="frontOverlays">正面叠图</button></div><a id="full-image" class="muted" target="_blank" rel="noopener">打开大图 ↗</a></div><div class="scroll"><img id="comparison" class="comparison-image" alt="固定原照片相机的精修前后对比" hidden><div class="empty" id="comparison-empty">正在读取对比图</div></div><div class="legend">A：原始 UFF　B：精修几何 + 原贴图　C：精修几何 + 重算融合贴图。D 为精修几何 + 正面优先取色，可能出现双重睑缘，仅供诊断。灰模只比较几何。</div><div class="details" id="details"></div></section><footer>上方可自由旋转；下方对比图固定使用原照片相机和相同光照。原照同时参与拟合，只能用于观察拟合效果。模型不包含眼球、睫毛和客户端头部。</footer></main><script type="module">
import * as THREE from 'three';import {FaceViewer} from '/scene.js';const $=s=>document.querySelector(s);let viewer,manifest,selected,variant='original',mode='clay',panel='frontEyes',busy=false,preview=null;const cache=new Map();
async function request(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw new Error(`${r.status} ${await r.text()}`);return r;}function fail(e){$('#loading').hidden=false;$('#loading').textContent=e.message||String(e);}
function sync(){document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===mode)));document.querySelectorAll('[data-variant]').forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.variant===variant));b.disabled=busy;});$('#case-select').disabled=busy;}
function applyMode(){viewer.setMode(mode==='unlit'?'textured':mode);if(mode==='unlit'&&viewer.asset){const a=viewer.asset;if(!a.modes.unlit){a.modes.unlit=a.modes.textured.map(m=>new THREE.MeshBasicMaterial({color:m.color.clone(),map:m.map,side:m.side,toneMapped:false}));a.materials.push(...a.modes.unlit);}a.mesh.material=a.modes.unlit;}sync();}
function showPhoto(entry,index){const p=entry.photos.find(p=>p.index===index)||entry.photos[0];if(!p)return;$('#photo').src=p.url;$('#photo-link').href=p.url;$('#photo-title').textContent=`${index===entry.frontViewIndex?'正面':'输入'}原照 · 视图 ${p.index+1}`;$('#photo-note').textContent=`${p.size.join(' × ')} · ${entry.label}`;}
function showPanel(){const entry=manifest.cases.find(c=>c.caseId===selected),url=entry?.comparisons?.[panel];document.querySelectorAll('[data-panel]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.panel===panel)));$('#comparison').hidden=!url;$('#comparison-empty').hidden=!!url;$('#comparison-empty').textContent=entry?.status==='skipped'?'本组未应用精修，请查看原始模型与照片。':'该对比图尚未生成。';$('#full-image').hidden=!url;if(url){$('#comparison').src=url;$('#full-image').href=url;}}
function state(){return{position:viewer.camera.position.clone(),quaternion:viewer.camera.quaternion.clone(),up:viewer.camera.up.clone(),target:viewer.controls.target.clone(),center:viewer.center.clone(),radius:viewer.radius,fitDistance:viewer.fitDistance,lightPosition:viewer.lightRig.position.clone(),lightScale:viewer.lightRig.scale.clone(),near:viewer.camera.near,far:viewer.camera.far,min:viewer.controls.minDistance,max:viewer.controls.maxDistance};}
function restore(s){viewer.camera.position.copy(s.position);viewer.camera.quaternion.copy(s.quaternion);viewer.camera.up.copy(s.up);viewer.controls.target.copy(s.target);viewer.center.copy(s.center);viewer.radius=s.radius;viewer.fitDistance=s.fitDistance;viewer.lightRig.position.copy(s.lightPosition);viewer.lightRig.scale.copy(s.lightScale);viewer.camera.near=s.near;viewer.camera.far=s.far;viewer.controls.minDistance=s.min;viewer.controls.maxDistance=s.max;viewer.camera.updateProjectionMatrix();viewer.controls.update();}
async function choose(id,kind=variant){if(busy)return;const entry=manifest.cases.find(c=>c.caseId===id);if(!entry)return;const same=id===selected&&!!preview,saved=same?state():null;selected=id;variant=entry.variants.some(v=>v.id===kind)?kind:'original';$('#case-select').value=id;$('#variants').replaceChildren();for(const v of entry.variants){const b=document.createElement('button');b.textContent=v.label;b.dataset.variant=v.id;b.addEventListener('click',()=>choose(id,v.id));$('#variants').append(b);}$('#thumbs').replaceChildren();for(const p of entry.photos){const b=document.createElement('button'),img=document.createElement('img');img.src=p.url;img.alt=`视图 ${p.index+1}`;b.append(img);b.addEventListener('click',()=>showPhoto(entry,p.index));$('#thumbs').append(b);}showPhoto(entry,entry.frontViewIndex);showPanel();$('#details').replaceChildren();const strong=document.createElement('strong');strong.textContent=entry.status==='skipped'?'本组未应用精修，保留原始几何。':entry.status==='failed'?'本组精修失败，原始结果仍可查看。':'本组精修候选已生成。';$('#details').append(strong);if(entry.summary){const p=document.createElement('p');p.textContent=entry.summary;$('#details').append(p);}if(entry.reportUrl){const a=document.createElement('a');a.href=entry.reportUrl;a.target='_blank';a.textContent='查看实验记录 ↗';$('#details').append(a);}const u=new URL(location.href);u.searchParams.set('case',id);u.searchParams.set('variant',variant);history.replaceState(null,'',u);busy=true;$('#loading').hidden=false;$('#loading').textContent='读取模型中';sync();try{const key=`${id}/${variant}`;if(!cache.has(key))cache.set(key,await(await request(`/cases/${id}/${variant}/preview`)).json());preview=cache.get(key);await viewer.load(preview,async url=>(await request(url)).blob());if(saved)restore(saved);viewer.asset.mesh.userData.units='uncalibratedCanonical';applyMode();$('#caption').textContent=`${entry.label} · ${entry.variants.find(v=>v.id===variant).label} · ${preview.counts.sourceVertices.toLocaleString()} 顶点。切换版本保留观察角度；C/D 和 B 共用精修几何。`;$('#loading').hidden=true;}catch(e){fail(e);}finally{busy=false;sync();}}
function eyeFocus(){if(!preview?.eyeFocus)return;viewer.setView('front');const c=new THREE.Vector3().fromArray(preview.eyeFocus.center),width=preview.eyeFocus.width,halfV=THREE.MathUtils.degToRad(viewer.camera.fov/2),halfH=Math.atan(Math.tan(halfV)*viewer.camera.aspect),distance=width/(2*Math.tan(Math.min(halfV,halfH)))*1.22;viewer.controls.target.copy(c);viewer.camera.position.copy(c).add(new THREE.Vector3(0,0,distance));viewer.controls.minDistance=Math.min(viewer.radius*.1,distance*.25);viewer.controls.update();}
try{viewer=new FaceViewer($('#viewport'),{onError:fail});manifest=await(await request('/comparison')).json();$('#batch').textContent=`${manifest.cases.length} 组 · 原生 UFF 前后对照`;for(const c of manifest.cases){const o=document.createElement('option');o.value=c.caseId;o.textContent=c.label+(c.status==='skipped'?' · 未应用精修':c.status==='failed'?' · 失败':'');$('#case-select').append(o);}$('#case-select').addEventListener('change',e=>choose(e.target.value));document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.mode;applyMode();}));document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>viewer.setView(b.dataset.view)));document.querySelectorAll('[data-panel]').forEach(b=>b.addEventListener('click',()=>{panel=b.dataset.panel;showPanel();}));$('#eyes').addEventListener('click',eyeFocus);$('#reset').addEventListener('click',()=>viewer.reset());window.addEventListener('pagehide',()=>viewer.dispose(),{once:true});const q=new URL(location.href).searchParams;await choose(q.get('case')||manifest.cases[0]?.caseId,q.get('variant')||'original');}catch(e){fail(e);}
</script></body></html>'''


def preview_payload(vertices, faces, uvs, texture_path, report, case_id, variant):
    vertices, faces, uvs = np.asarray(vertices), np.asarray(faces), np.asarray(uvs)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(vertices)
            or uvs.shape != (len(vertices), 2) or not np.isfinite(uvs).all()):
        raise ValueError("Invalid native UFF mesh")
    normals = np.zeros_like(vertices)
    face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    np.divide(normals, lengths[:, None], out=normals, where=lengths[:, None] > 0)
    with np.load(texture_path, allow_pickle=False) as arrays:
        key = "linearAlbedo" if "linearAlbedo" in arrays else "albedo"
        encoding = "linear" if key == "linearAlbedo" else report["config"]["assemblyColorEncoding"]
        linear = decodeColor(arrays[key], encoding)
        coverage = arrays["coverageMask"].astype(bool)
    srgb = np.where(linear <= .0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - .055)
    srgb[~coverage] = .32
    output = io.BytesIO()
    Image.fromarray(np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8)).save(output, format="PNG")
    texture = output.getvalue()
    return {"schemaVersion": 1, "jobId": case_id, "templateId": "nativeUvFaceFusionCanonical6986",
        "counts": {"sourceVertices": len(vertices), "renderVertices": len(vertices), "faces": len(faces), "materials": 1},
        "coordinates": {"lengthUnit": "uncalibratedCanonical", "uvOrigin": "vDown", "axisTransform": "identity"},
        "mesh": {"positions": vertices.ravel().tolist(), "normals": normals.ravel().tolist(), "uvs": uvs.ravel().tolist(),
                 "indices": faces.ravel().tolist(), "sourceVertexIds": list(range(len(vertices))),
                 "groups": [{"start": 0, "count": int(faces.size), "materialIndex": 0}]},
        "materials": [{"index": 0, "name": variant, "roughnessFactor": .8, "metallicFactor": 0,
                       "alphaMode": "OPAQUE", "alphaCutoff": .5, "baseColorFactor": [1, 1, 1, 1],
                       "texture": {"url": f"/cases/{case_id}/{variant}/texture", "sha256": hashlib.sha256(texture).hexdigest(),
                                   "sizeBytes": len(texture), "colorSpace": "sRGB"}}],
        "eyeFocus": nativeEyeFocus(vertices, faces)}, texture


def summary_text(entry):
    summary = entry.get("summary", entry.get("reason", entry.get("error", "")))
    if isinstance(summary, str):
        return summary
    if isinstance(summary, dict):
        summary = summary.get("workingSummary", summary)
        index = str(entry.get("frontViewIndex", 0))
        before = summary.get("before", {}).get(index, {}).get("meanContourErrorPixels")
        after = summary.get("after", {}).get(index, {}).get("meanContourErrorPixels")
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            return f"正面眼睑轮廓平均误差：{before:.2f} → {after:.2f} 像素（拟合工作图坐标，照片检测轮廓为参照）。"
    return json.dumps(summary, ensure_ascii=False, separators=(", ", ": "))


def createApp(experiment: Path) -> FastAPI:
    experiment = Path(experiment).resolve()
    web = ROOT / "src/faceAppearance/web"
    scripts = {"scene.js": (web / "scene.js").read_bytes()}
    scripts.update({name: (web / "vendor" / name).read_bytes() for name in ("three.module.js", "three.core.js", "OrbitControls.js")})
    cache, photos, lock = {}, {}, threading.RLock()
    app = FastAPI(title="UFF front-priority comparison", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

    @app.middleware("http")
    async def headers(request, callNext):
        response = await callNext(request)
        response.headers.update({"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store", "Referrer-Policy": "same-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"})
        return response

    def manifest():
        try:
            value = read_json(experiment / "frontManifest.json")
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise HTTPException(503, "Experiment manifest unavailable") from error
        ids = []
        for case in value["cases"]:
            if not re.fullmatch(r"[A-Za-z0-9_-]+", case.get("caseId", "")):
                raise HTTPException(409, "Invalid case id")
            ids.append(case["caseId"])
            source = Path(case["sourceDirectory"]).resolve()
            if not source.is_relative_to(ROOT / "runs"):
                raise HTTPException(409, "Source must be a saved local run")
            for field in ("geometryPath", "texturePath", "texturePriorityPath", "resultPath"):
                if case.get(field) and not Path(case[field]).resolve().is_relative_to(experiment):
                    raise HTTPException(409, "Candidate path must be inside this experiment")
        if len(ids) != len(set(ids)):
            raise HTTPException(409, "Duplicate case id")
        return value

    def selected(case_id):
        entry = next((c for c in manifest()["cases"] if c["caseId"] == case_id), None)
        if entry is None:
            raise HTTPException(404, "Unknown case")
        return entry

    def preview_directory(value=None):
        value = manifest() if value is None else value
        directory = Path(value.get("previewDirectory", experiment / "preview")).resolve()
        if not directory.is_relative_to(experiment):
            raise HTTPException(409, "Preview directory must be inside this experiment")
        return directory

    def variants(entry):
        values = [{"id": "original", "label": "A 原始 UFF"}]
        if entry.get("geometryPath") and Path(entry["geometryPath"]).is_file():
            values.append({"id": "refined", "label": "B 精修几何"})
            if entry.get("texturePath") and Path(entry["texturePath"]).is_file():
                values.append({"id": "refused", "label": "C 重算融合"})
            if entry.get("texturePriorityPath") and Path(entry["texturePriorityPath"]).is_file():
                values.append({"id": "priority", "label": "D 正面取色（诊断）"})
        return values

    def loaded(case_id, variant):
        entry = selected(case_id)
        if variant not in [v["id"] for v in variants(entry)]:
            raise HTTPException(404, "Unavailable variant")
        source = Path(entry["sourceDirectory"])
        geometry_path = source / "geometry.npz" if variant == "original" else Path(entry["geometryPath"])
        texture_path = Path(entry["texturePath"]) if variant == "refused" else Path(entry["texturePriorityPath"]) if variant == "priority" else source / "texture.npz"
        key = (case_id, variant, geometry_path.stat().st_mtime_ns, texture_path.stat().st_mtime_ns)
        with lock:
            if key not in cache:
                with np.load(source / "geometry.npz", allow_pickle=False) as arrays:
                    original = dict(arrays)
                with np.load(geometry_path, allow_pickle=False) as arrays:
                    geometry = dict(arrays)
                for name in ("faces", "uvs"):
                    if name in geometry and not np.array_equal(geometry[name], original[name]):
                        raise HTTPException(409, "Candidate topology or UVs changed")
                report = read_json(source / "result.json")
                if report.get("geometrySpace") != "uvFaceFusionCanonical6986" or "v down" not in report.get("arrayUvConvention", ""):
                    raise HTTPException(409, "Unknown geometry or UV convention")
                cache[key] = preview_payload(geometry["vertices"], original["faces"], original["uvs"], texture_path, report, case_id, variant)
            return cache[key]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(HTML)

    @app.get("/comparison")
    def comparison():
        value, entries = manifest(), []
        preview_root = preview_directory(value)
        for entry in value["cases"]:
            source = Path(entry["sourceDirectory"])
            inputs = read_json(source / "inputs.json")
            case_id = entry["caseId"]
            compared = {name: f"/cases/{case_id}/comparisons/{name}.png" for name in ("frontEyes", "frontComparison", "sideComparison", "frontOverlays")
                        if (preview_root / case_id / f"{name}.png").is_file()}
            entries.append({"caseId": case_id, "label": entry.get("label", case_id), "status": entry.get("status"),
                "frontViewIndex": entry.get("frontViewIndex", 0), "variants": variants(entry), "comparisons": compared,
                "summary": summary_text(entry),
                "reportUrl": f"/cases/{case_id}/report" if entry.get("resultPath") and Path(entry["resultPath"]).is_file() else None,
                "photos": [{"index": i, "url": f"/cases/{case_id}/photos/{i}", "size": item["orientedSize"]} for i, item in enumerate(inputs)]})
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
        return loaded(case_id, variant)[0]

    @app.get("/cases/{case_id}/{variant}/texture")
    def texture(case_id: str, variant: str):
        return Response(loaded(case_id, variant)[1], media_type="image/png")

    @app.get("/cases/{case_id}/photos/{index}")
    def photo(case_id: str, index: int):
        entry = selected(case_id)
        inputs = read_json(Path(entry["sourceDirectory"]) / "inputs.json")
        if not 0 <= index < len(inputs):
            raise HTTPException(404, "Unknown photo")
        item = inputs[index]
        with lock:
            if item["sha256"] not in photos:
                if digest(item["path"]) != item["sha256"]:
                    raise HTTPException(409, "Source photograph changed")
                with Image.open(item["path"]) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                out = io.BytesIO(); image.save(out, format="PNG")
                photos[item["sha256"]] = out.getvalue()
            return Response(photos[item["sha256"]], media_type="image/png")

    @app.get("/cases/{case_id}/comparisons/{filename}")
    def comparison_image(case_id: str, filename: str):
        selected(case_id)
        if filename not in {name + ".png" for name in ("frontEyes", "frontComparison", "sideComparison", "frontOverlays")}:
            raise HTTPException(404, "Unknown comparison")
        path = preview_directory() / case_id / filename
        if not path.is_file():
            raise HTTPException(404, "Comparison not rendered")
        return Response(path.read_bytes(), media_type="image/png")

    @app.get("/cases/{case_id}/report")
    def report(case_id: str):
        entry = selected(case_id)
        if not entry.get("resultPath") or not Path(entry["resultPath"]).is_file():
            raise HTTPException(404, "No refinement report")
        return Response(Path(entry["resultPath"]).read_bytes(), media_type="application/json")

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/uffFrontPriority20261011")
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18089)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be 1..65535")
    uvicorn.run(createApp(args.experiment), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
