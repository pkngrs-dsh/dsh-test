"""Read-only, localhost viewer for the saved UFF multi-case experiment.

The original UFF positions/UVs and packaged client mesh are kept in their own
coordinate systems. The camera frames each independently; this is not a metric
registration. No inference, fitting, file extraction or output mutation occurs.
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
from PIL import Image
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faceAppearance.clientImport import loadClientAsset
from faceAppearance.headAssembly import decodeColor
from faceAppearance.webPreview import buildPreview


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>UFF 多组实验 · FaceAppearance</title><link rel="icon" href="data:,">
<style>
:root{font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif;color:#e7eeeb;background:#111a18;color-scheme:dark}*{box-sizing:border-box}body{margin:0}button,a,select{font:inherit}button{cursor:pointer}button:disabled{opacity:.35;cursor:default}button:focus-visible,a:focus-visible,select:focus-visible{outline:2px solid #bfe6c9;outline-offset:3px}header{border-bottom:1px solid #ffffff17;padding:20px 4vw;display:flex;justify-content:space-between;gap:20px}.brand{font-size:17px;font-weight:600}.muted{color:#9eb4a9;font-size:12px;line-height:1.7}main{max-width:1680px;padding:25px 4vw;margin:auto}.heading{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:22px}h1{font-size:27px;margin:0 0 8px;font-weight:550}p{margin:0}select{background:#1c3027;color:#e2f1e5;padding:12px;border:1px solid #78968255;border-radius:7px;min-width:240px}.layout{display:grid;grid-template-columns:minmax(0,1.9fr) minmax(250px,1fr);gap:18px}.studio,.photos,.facts{border:1px solid #ffffff17;border-radius:12px;background:#17251f;overflow:hidden}.toolbar{padding:12px 16px;display:flex;gap:16px;flex-wrap:wrap;justify-content:space-between;border-bottom:1px solid #ffffff12}.group{display:flex;gap:3px;flex-wrap:wrap}.group button{border:1px solid transparent;border-radius:5px;background:transparent;color:#9fb5a8;padding:8px 10px;font-size:12px}.group button[aria-pressed=true],.group button:hover{background:#c5e0c711;color:#d8efdc;border-color:#b9dab333}.stage{height:580px;max-height:64vh;min-height:360px;position:relative;background:radial-gradient(ellipse at 50% 42%,#3b4e41,#203528 65%,#16271e)}#viewport{position:absolute;inset:0}#loading{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);background:#101f18ed;border:1px solid #abcbb33a;padding:17px 22px;border-radius:8px;max-width:85%;font-size:13px}.stage-note{position:absolute;top:15px;left:18px;font-size:11px;color:#acbfaf;pointer-events:none}.views{position:absolute;left:15px;bottom:15px;background:#13251bdd;border-radius:7px}.hint{position:absolute;right:17px;bottom:23px;font-size:10px;color:#95ac9b;pointer-events:none}.asset-info{padding:15px 18px;font-size:11px;line-height:1.7;color:#a6baae}.asset-info strong{display:block;color:#d9e8dc;font-weight:500;font-size:13px;margin-bottom:3px}.photos{padding:16px}.photos h2,.facts h2{font-size:14px;font-weight:500;margin:0 0 8px}.photo-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:15px}.photo-grid figure{margin:0;min-width:0}.photo-grid a{display:block;aspect-ratio:1;overflow:hidden;border-radius:6px;background:#0e1812}.photo-grid img{display:block;width:100%;height:100%;object-fit:contain;transition:transform .2s}.photo-grid.zoom img{transform:scale(2.35)}.photo-grid figcaption{margin-top:6px;font-size:11px;color:#aabfaf;line-height:1.5}.downloads{display:flex;flex-wrap:wrap;gap:8px;margin-top:13px}.downloads a{font-size:11px;text-decoration:none;color:#cee5d0;border:1px solid #ffffff25;border-radius:5px;padding:7px 10px}.downloads a[hidden]{display:none}.facts{margin-top:20px;padding:18px;overflow-x:auto}.metrics{display:flex;gap:35px;flex-wrap:wrap;margin:16px 0 18px}.metric span{display:block;font-size:11px;color:#90a89a}.metric strong{font-size:21px;font-weight:500;color:#d8eadd;display:block;margin-top:5px}.timing{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:17px;font-size:11px;color:#adc2b1}table{border-collapse:collapse;width:100%;min-width:650px;font-size:12px;text-align:left}th{font-weight:500;color:#97b09e;font-size:11px}td,th{padding:11px 9px;border-top:1px solid #ffffff10}td button{border:0;background:none;color:#dbedde;padding:0;cursor:pointer}.active-row{background:#bad8bd09}.warn{color:#eacb96}.note{margin-top:14px;font-size:11px;color:#8da997;line-height:1.8}#camera-table{margin-top:17px;min-width:580px}footer{padding:15px 0;color:#789a86;font-size:11px;line-height:1.8}@media(max-width:900px){.layout{grid-template-columns:1fr}.stage{height:520px;max-height:65vh}.heading{display:block}select{margin-top:15px;width:100%}.photos .photo-grid{grid-template-columns:repeat(4,minmax(0,1fr))}header>.muted{display:none}.hint{display:none}}@media(max-width:500px){.photos .photo-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.metrics{gap:20px}.stage{min-height:340px}.toolbar{gap:8px}main{padding:20px 4vw}}
</style><script type="importmap">{"imports":{"three":"/vendor/three.module.js"}}</script></head><body>
<header><div class="brand">FaceAppearance · UFF 多组实验</div><div class="muted" id="batch-status">正在读取批次</div></header>
<main><section class="heading"><div><h1>从不同照片组观察重建表现</h1><p class="muted">先看原生 UFF 几何，再查看映射到客户端模板后的结果。</p></div><select id="case-select" aria-label="选择实验样本"></select></section>
<div class="layout"><section class="studio"><div class="toolbar"><div class="group" aria-label="网格来源"><button data-asset="uff" aria-pressed="true">原生 UFF</button><button data-asset="client" aria-pressed="false">客户端模板</button></div><div class="group" aria-label="显示模式"><button data-mode="unlit" aria-pressed="true">贴图</button><button data-mode="textured" aria-pressed="false">照明</button><button data-mode="clay" aria-pressed="false">灰模</button><button data-mode="wireframe" aria-pressed="false">线框</button></div></div><div class="stage"><div id="viewport"></div><div class="stage-note" id="stage-note">原生 UFF · 未标定单位</div><div id="loading" role="status">等待实验结果…</div><div class="group views"><button data-view="front">正面</button><button data-view="left">左侧</button><button data-view="right">右侧</button><button id="eye-focus">眼部放大</button><button id="reset-view">全脸</button></div><span class="hint">拖动旋转 · 滚轮缩放 · 右键平移</span></div><div class="asset-info"><strong id="asset-title">正在读取</strong><span id="asset-note"></span><div class="downloads" id="downloads"></div></div></section>
<aside class="photos"><h2>输入照片</h2><p class="muted" id="photo-note">模型实际使用的裁剪图。点击可打开大图。</p><div class="group" style="margin-top:8px"><button id="photo-zoom" aria-pressed="false">照片眼部放大</button></div><div class="photo-grid" id="photos"></div></aside></div>
<section class="facts"><h2 id="fact-title">本组结果</h2><div class="metrics"><div class="metric"><span>总推理时间 · 不含首次模型加载</span><strong id="total-time">—</strong></div><div class="metric"><span>UFF 几何推理</span><strong id="geometry-time">—</strong></div><div class="metric"><span>相机通过</span><strong id="camera-count">—</strong></div><div class="metric"><span>UV 有观测覆盖</span><strong id="coverage">—</strong></div></div><div class="timing" id="stage-times"></div><p class="muted" id="case-error"></p><table id="camera-table"><thead><tr><th>输入视图</th><th>相机状态</th><th>拟合中位误差 / px</th><th>留出中位误差 / px</th><th>留出 P90 / px</th></tr></thead><tbody></tbody></table><p class="note">误差来自模型预测对应点在工作图像上的相机重投影；它不是人工标注误差，也不能单独证明眼睑或三维形状准确。</p></section>
<section class="facts"><h2>本批样本</h2><table><thead><tr><th>样本</th><th>状态</th><th>视图数</th><th>UFF 几何</th><th>总推理</th><th>相机通过</th><th>留出中位误差范围 / px</th></tr></thead><tbody id="batch-table"></tbody></table></section>
<footer>两种网格各自自动取景，未做等比例或度量对齐。原生 UFF 保留未标定的 canonical 坐标；客户端网格保留导出的米制模板坐标。贴图采用本次配置声明的颜色解释，显示结果不代表经过物理标定的反照率。<div class="downloads"><a id="rotation-diagnostics" href="/rotationDiagnostics" target="_blank" rel="noopener" hidden>查看相机旋转诊断 JSON</a></div></footer></main>
<script type="module">
import * as THREE from 'three';
import { FaceViewer } from '/scene.js';
const $ = q => document.querySelector(q), fmt = (v,n=2) => Number.isFinite(v) ? v.toFixed(n) : '—';
let viewer, manifest, selected = new URL(location.href).searchParams.get('case'), assetKind = 'uff', mode = 'unlit', busy = false, preview = null, photosZoomed = false;
const previews = new Map(), statusNames = {succeeded:'完成',failed:'失败',running:'运行中',pending:'待运行',queued:'待运行'};
async function fetchOK(url){const r=await fetch(url);if(!r.ok){let s='';try{s=(await r.json()).detail||'';}catch{}throw new Error(`${r.status} ${typeof s==='string'?s:'资源读取失败'}`);}return r;}
function cell(row,value){const c=document.createElement('td');c.textContent=value;row.append(c);return c;}
function failure(error){$('#loading').hidden=false;$('#loading').textContent=`无法载入：${error.message}`;}
function sync(){document.querySelectorAll('[data-asset],[data-mode],[data-view],#eye-focus,#reset-view').forEach(b=>b.disabled=busy||!preview);$('#case-select').disabled=busy;document.querySelectorAll('[data-asset]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.asset===assetKind)));document.querySelectorAll('[data-mode]').forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.mode===mode));if(preview?.geometryOnly&&['unlit','textured'].includes(b.dataset.mode))b.disabled=true;});}
function applyMode(){viewer.setMode(mode==='unlit'?'textured':mode);if(mode==='unlit'&&viewer.asset){const a=viewer.asset;if(!a.modes.unlit){a.modes.unlit=a.modes.textured.map(m=>new THREE.MeshBasicMaterial({color:m.color.clone(),map:m.map,opacity:m.opacity,alphaTest:m.alphaTest,side:m.side,transparent:false,toneMapped:false}));a.materials.push(...a.modes.unlit);}a.mesh.material=a.modes.unlit;}sync();}
function showFacts(entry){$('#fact-title').textContent=entry.label+' · 结果';$('#total-time').textContent=fmt(entry.inferenceSeconds)+' s';$('#geometry-time').textContent=fmt(entry.stageSeconds?.geometry)+' s';$('#camera-count').textContent=`${entry.acceptedCameraCount??'—'} / ${entry.viewCount??'—'}`;$('#coverage').textContent=Number.isFinite(entry.atlasCoverage)?(entry.atlasCoverage*100).toFixed(1)+'%':'—';$('#case-error').textContent=entry.error||'';$('#stage-times').replaceChildren();const names={inputPreparation:'输入准备',geometry:'UFF 几何',delight:'OpenDelight',cameraAndFusion:'相机与贴图融合',assemblyAndExport:'客户端装配'};for(const [k,v]of Object.entries(entry.stageSeconds||{})){const s=document.createElement('span');s.textContent=`${names[k]||k} ${fmt(v)} s`;$('#stage-times').append(s);}const body=$('#camera-table tbody');body.replaceChildren();for(const c of entry.cameras||[]){const row=document.createElement('tr');cell(row,`视图 ${c.viewIndex+1}`);cell(row,c.valid?'通过':`未通过 · ${c.reason||'未报告'}`);for(const k of ['medianErrorPixels','heldOutMedianErrorPixels','heldOutP90ErrorPixels'])cell(row,fmt(c[k]));body.append(row);}}
function showPhotos(entry){$('#photo-note').textContent=entry.photosAreSource?'原始输入照片（已归正方向）。本组在裁剪前失败，未生成模型输入图。':'模型实际使用的裁剪图。点击可打开大图。';$('#photo-zoom').disabled=!!entry.photosAreSource;$('#photos').replaceChildren();for(const p of entry.photos||[]){const figure=document.createElement('figure'),link=document.createElement('a'),img=document.createElement('img'),caption=document.createElement('figcaption');link.href=p.url;link.target='_blank';link.rel='noopener';img.src=p.url;img.alt=`${entry.label} 输入视图 ${p.index+1}`;img.loading='lazy';if(p.eyeCenter)img.style.transformOrigin=`${p.eyeCenter[0]*100}% ${p.eyeCenter[1]*100}%`;else img.style.transformOrigin='50% 40%';caption.textContent=`视图 ${p.index+1}${p.name?' · '+p.name:''}${p.sourceSize?' · 原图 '+p.sourceSize.join(' × '):''}`;link.append(img);figure.append(link,caption);$('#photos').append(figure);}$('#photos').classList.toggle('zoom',photosZoomed&&!entry.photosAreSource);}
function downloads(entry){$('#downloads').replaceChildren();for(const item of entry.downloads||[]){const a=document.createElement('a');a.href=item.url;a.textContent=item.label;a.download='';$('#downloads').append(a);}}
async function choose(caseId,kind=assetKind){if(busy)return;const entry=manifest.cases.find(c=>c.caseId===caseId);if(!entry)return;selected=caseId;assetKind=kind;$('#case-select').value=selected;const url=new URL(location.href);url.searchParams.set('case',selected);history.replaceState(null,'',url);showFacts(entry);showPhotos(entry);downloads(entry);$('#loading').hidden=false;preview=null;busy=true;viewer.controls.enabled=false;sync();$('#asset-title').textContent=entry.label;$('#asset-note').textContent='';if(entry.status!=='succeeded'&&!(kind==='uff'&&entry.nativeGeometryAvailable)){if(viewer.asset)viewer.asset.mesh.visible=false;$('#loading').textContent=entry.status==='failed'?`实验失败：${entry.error||'查看日志'}`:'本组尚未完成，页面会自动更新。';busy=false;viewer.controls.enabled=true;sync();return;}
try{const key=`${caseId}/${kind}`;if(!previews.has(key))previews.set(key,await(await fetchOK(`/cases/${caseId}/${kind}/preview`)).json());const data=previews.get(key);await viewer.load(data,async u=>(await fetchOK(u)).blob());preview=data;if(data.geometryOnly)mode='clay';viewer.asset.mesh.userData.units=data.coordinates.lengthUnit;viewer.asset.mesh.name=kind==='uff'?'Native UFF canonical face':'Packaged client template';applyMode();$('#stage-note').textContent=kind==='uff'?'原生 UFF · canonical 坐标，单位未标定':'客户端模板 · 导出坐标单位为米';$('#asset-title').textContent=`${entry.label} · ${kind==='uff'?'原生 UFF':'客户端模板'} · ${data.counts.sourceVertices.toLocaleString()} 顶点 / ${data.counts.faces.toLocaleString()} 面`;$('#asset-note').textContent=data.displayNote;$('#loading').hidden=true;}catch(e){if(viewer.asset)viewer.asset.mesh.visible=false;failure(e);}finally{busy=false;viewer.controls.enabled=true;sync();}}
function eyeFocus(){if(!preview?.eyeFocus)return;viewer.setView('front');const c=new THREE.Vector3().fromArray(preview.eyeFocus.center),width=preview.eyeFocus.width;const halfV=THREE.MathUtils.degToRad(viewer.camera.fov/2),halfH=Math.atan(Math.tan(halfV)*viewer.camera.aspect),distance=width/(2*Math.tan(Math.min(halfV,halfH)))*1.22;viewer.controls.target.copy(c);viewer.camera.position.copy(c).add(new THREE.Vector3(0,0,distance));viewer.controls.minDistance=Math.min(viewer.radius*.1,distance*.25);viewer.controls.update();}
async function refresh(){const fresh=await(await fetchOK('/comparison')).json();manifest=fresh;$('#rotation-diagnostics').hidden=!fresh.rotationDiagnosticsUrl;$('#batch-status').textContent=`${fresh.cases.filter(c=>c.status==='succeeded').length} / ${fresh.cases.length} 组完成 · ${fresh.status==='complete'?'本批运行结束':'运行中'}`;const select=$('#case-select');select.replaceChildren();const body=$('#batch-table');body.replaceChildren();for(const c of fresh.cases){const o=document.createElement('option');o.value=c.caseId;o.textContent=`${c.label} · ${statusNames[c.status]||c.status}`;select.append(o);const row=document.createElement('tr');row.className=c.caseId===selected?'active-row':'';const b=document.createElement('button');b.textContent=c.label;b.addEventListener('click',()=>choose(c.caseId));cell(row,'').append(b);cell(row,statusNames[c.status]||c.status);cell(row,String(c.viewCount??'—'));cell(row,fmt(c.stageSeconds?.geometry)+' s');cell(row,fmt(c.inferenceSeconds)+' s');cell(row,String(c.acceptedCameraCount??'—'));const errors=(c.cameras||[]).filter(v=>v.valid&&Number.isFinite(v.heldOutMedianErrorPixels)).map(v=>v.heldOutMedianErrorPixels);cell(row,errors.length?`${fmt(Math.min(...errors))} – ${fmt(Math.max(...errors))}`:'—');body.append(row);}if(!selected||!fresh.cases.some(c=>c.caseId===selected))selected=fresh.cases.find(c=>c.status==='succeeded')?.caseId||fresh.cases[0]?.caseId;select.value=selected||'';if(selected&&!busy&&!preview)await choose(selected);if(fresh.status!=='complete')setTimeout(()=>refresh().catch(failure),5000);}
try{viewer=new FaceViewer($('#viewport'),{onError:failure});$('#case-select').addEventListener('change',e=>choose(e.target.value));document.querySelectorAll('[data-asset]').forEach(b=>b.addEventListener('click',()=>choose(selected,b.dataset.asset)));document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{mode=b.dataset.mode;applyMode();}));document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>viewer.setView(b.dataset.view)));$('#eye-focus').addEventListener('click',eyeFocus);$('#reset-view').addEventListener('click',()=>viewer.reset());$('#photo-zoom').addEventListener('click',()=>{photosZoomed=!photosZoomed;$('#photos').classList.toggle('zoom',photosZoomed);$('#photo-zoom').setAttribute('aria-pressed',String(photosZoomed));});window.addEventListener('pagehide',()=>viewer.dispose(),{once:true});await refresh();}catch(e){failure(e);}
</script></body></html>'''


def readJson(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def focusForPoints(points: np.ndarray) -> dict:
    low, high = points.min(axis=0), points.max(axis=0)
    return {"center": ((low + high) / 2).tolist(),
            "width": float(max(high[0] - low[0], high[1] - low[1]) * 1.25)}


def nativeEyeFocus(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """Use the two 60-vertex UFF eye openings only for camera framing."""
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges, counts = np.unique(edges, axis=0, return_counts=True)
    adjacency = {}
    for a, b in edges[counts == 1]:
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))
    remaining, loops = set(adjacency), []
    while remaining:
        stack, component = [next(iter(remaining))], set()
        while stack:
            value = stack.pop()
            if value in component:
                continue
            component.add(value)
            stack.extend(adjacency[value] - component)
        remaining -= component
        if len(component) == 60 and all(len(adjacency[v]) == 2 for v in component):
            loops.append(sorted(component))
    if len(loops) == 2:
        return focusForPoints(vertices[np.asarray(loops).ravel()])
    # A framing fallback never claims an anatomical correspondence.
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = (low + high) / 2
    center[1] += (high[1] - low[1]) * .12
    return {"center": center.tolist(), "width": float((high[0] - low[0]) * .7)}


def nativePreview(directory: Path, caseId: str, report: dict):
    with np.load(directory / "geometry.npz", allow_pickle=False) as arrays:
        vertices = np.asarray(arrays["vertices"], np.float64)
        faces = np.asarray(arrays["faces"], np.int64)
        uvs = np.asarray(arrays["uvs"], np.float64)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or uvs.shape != (len(vertices), 2)
            or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0
            or faces.min() < 0 or faces.max() >= len(vertices)
            or not np.isfinite(vertices).all() or not np.isfinite(uvs).all()):
        raise ValueError("Invalid saved native UFF mesh")
    if report.get("geometrySpace") != "uvFaceFusionCanonical6986" or "v down" not in report.get("arrayUvConvention", ""):
        raise ValueError("Unsupported native UFF coordinate or UV convention")
    normals = np.zeros_like(vertices)
    faceNormals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                           vertices[faces[:, 2]] - vertices[faces[:, 0]])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], faceNormals)
    lengths = np.linalg.norm(normals, axis=1)
    np.divide(normals, lengths[:, None], out=normals, where=lengths[:, None] > 0)
    texture, encoding = None, None
    if (directory / "texture.npz").is_file():
        with np.load(directory / "texture.npz", allow_pickle=False) as arrays:
            key = "linearAlbedo" if "linearAlbedo" in arrays else "albedo"
            encoding = "linear" if key == "linearAlbedo" else report["config"]["assemblyColorEncoding"]
            linear = decodeColor(arrays[key], encoding)
        # scene.js expects sRGB PNG bytes and V-down UVs with texture.flipY=false.
        srgb = np.where(linear <= .0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - .055)
        output = io.BytesIO()
        Image.fromarray(np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8)).save(output, format="PNG")
        texture = output.getvalue()
    preview = {
        "schemaVersion": 1, "jobId": caseId, "templateId": "nativeUvFaceFusionCanonical6986",
        "counts": {"sourceVertices": len(vertices), "renderVertices": len(vertices), "faces": len(faces), "materials": 1},
        "coordinates": {"lengthUnit": "uncalibratedCanonical", "uvOrigin": "vDown", "axisTransform": "identity"},
        "mesh": {"positions": vertices.ravel().tolist(), "normals": normals.ravel().tolist(),
                 "uvs": uvs.ravel().tolist(), "indices": faces.ravel().tolist(),
                 "sourceVertexIds": list(range(len(vertices))),
                 "groups": [{"start": 0, "count": int(faces.size), "materialIndex": 0}]},
        "materials": [{"index": 0, "name": "UFF observed albedo", "roughnessFactor": .8,
                       "metallicFactor": 0, "alphaMode": "OPAQUE", "alphaCutoff": .5,
                       "baseColorFactor": [1, 1, 1, 1], "texture": None if texture is None else {
                           "url": f"/cases/{caseId}/uff/texture", "sha256": hashlib.sha256(texture).hexdigest(),
                           "sizeBytes": len(texture), "colorSpace": "sRGB"}}],
        "eyeFocus": nativeEyeFocus(vertices, faces),
        "geometryOnly": texture is None,
        "displayNote": f"原生 UFF 面部网格，眼孔内没有模板眼球。使用观测 albedo（{encoding} → sRGB），未观测区域保留原值；照明粗糙度为预设。",
    }
    if texture is None:
        preview["displayNote"] = "仅几何；融合未完成。本视图只展示已保存的 UFF 网格，未生成可用贴图或客户端资产。"
    return preview, texture


def createApp(experiment: Path) -> FastAPI:
    experiment = Path(experiment).resolve()
    if not experiment.is_dir():
        raise ValueError("Experiment directory does not exist")
    web = ROOT / "src/faceAppearance/web"
    scripts = {"scene.js": (web / "scene.js").read_bytes()}
    scripts.update({name: (web / "vendor" / name).read_bytes()
                    for name in ("three.module.js", "three.core.js", "OrbitControls.js")})
    cache, lock = {}, threading.RLock()
    app = FastAPI(title="UFF multi-case experiment", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.middleware("http")
    async def localHeaders(request, callNext):
        response = await callNext(request)
        response.headers.update({"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store",
            "Referrer-Policy": "same-origin", "Content-Security-Policy":
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"})
        return response

    def manifest():
        try:
            value = readJson(experiment / "batchManifest.json")
        except (FileNotFoundError, json.JSONDecodeError) as error:
            raise HTTPException(503, "批次清单尚未准备好，请稍后刷新") from error
        if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
            raise HTTPException(409, "Invalid batch manifest")
        identities = []
        for entry in value["cases"]:
            caseId = entry.get("caseId", "")
            if not isinstance(caseId, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", caseId):
                raise HTTPException(409, "Invalid case identifier")
            identities.append(caseId)
            expected = (experiment / "cases" / caseId / "reconstruction").resolve()
            if (not expected.is_relative_to(experiment) or
                    Path(entry.get("outputDirectory", "")).resolve() != expected):
                raise HTTPException(409, "Case output is outside its declared batch directory")
        if len(identities) != len(set(identities)):
            raise HTTPException(409, "Duplicate case identifier")
        return value

    def selected(caseId):
        entry = next((v for v in manifest()["cases"] if v["caseId"] == caseId), None)
        if entry is None:
            raise HTTPException(404, "Unknown case")
        return entry, Path(entry["outputDirectory"]).resolve()

    def localFile(directory, name):
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise HTTPException(404, "Artifact unavailable")
        return path

    def loaded(caseId):
        entry, directory = selected(caseId)
        if entry.get("status") == "failed" and (directory / "geometry.npz").is_file():
            with lock:
                if caseId not in cache:
                    localFile(directory, "geometry.npz")
                    if (directory / "texture.npz").is_file():
                        localFile(directory, "texture.npz")
                    # The saved pipeline contract supplies these conventions;
                    # no successful result.json or success status is invented.
                    report = {"geometrySpace": "uvFaceFusionCanonical6986",
                              "arrayUvConvention": "u right, v down",
                              "config": readJson(localFile(directory, "config.json"))}
                    native, texture = nativePreview(directory, caseId, report)
                    native["displayNote"] = "本组流程失败，保留的 UFF 输出仅用于诊断。" + native["displayNote"]
                    cache[caseId] = {"uff": native, "uffTexture": texture}
                return cache[caseId]
        if entry.get("status") != "succeeded":
            raise HTTPException(409, "Case has not succeeded")
        with lock:
            if caseId not in cache:
                report = readJson(localFile(directory, "result.json"))
                if report.get("status") != "succeeded":
                    raise HTTPException(409, "Reconstruction is incomplete")
                for name in ("geometry.npz", "texture.npz"):
                    localFile(directory, name)
                native, texture = nativePreview(directory, caseId, report)
                archive = readJson(localFile(directory, "assembly/archive.json"))
                expected = report["clientAssetManifest"]
                asset = loadClientAsset(localFile(directory, "assembly/clientAsset.zip"),
                    expectedTemplateId=expected["templateId"], expectedTemplateHashes=expected["templateHashes"],
                    expectedArchiveSha256=archive["sha256"], expectedArchiveBytes=archive["bytes"])
                payload = asset.archivePath.read_bytes()
                if len(payload) != asset.archiveBytes or hashlib.sha256(payload).hexdigest() != asset.archiveSha256:
                    raise HTTPException(409, "Client archive changed while reading")
                client = buildPreview(asset, caseId)
                for material in client["materials"]:
                    material["texture"]["url"] = f"/cases/{caseId}/client/textures/{material['index']}"
                eyeIndices = [index for c in asset.components["components"] if c.get("materialId") == 2
                              for index in c["vertexIndices"]]
                if eyeIndices:
                    client["eyeFocus"] = focusForPoints(asset.mesh["vertices"][eyeIndices])
                client["displayNote"] = "UFF 面部形变映射到固定客户端模板；后脑、眼球与口腔来自模板。保留导出坐标与材质；骨骼和眼睑解剖对应仍需独立验证。"
                cache[caseId] = {"uff": native, "uffTexture": texture, "client": client,
                                 "asset": asset, "archive": payload}
            return cache[caseId]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(HTML)

    @app.get("/comparison")
    def comparison():
        value = manifest()
        entries = []
        cameraKeys = ("viewIndex", "valid", "reason", "medianErrorPixels",
                      "heldOutMedianErrorPixels", "heldOutP90ErrorPixels")
        for entry in value["cases"]:
            caseId, directory = entry["caseId"], Path(entry["outputDirectory"]).resolve()
            info = {k: entry[k] for k in ("caseId", "label", "status", "error") if k in entry}
            info.setdefault("label", caseId)
            info["nativeGeometryAvailable"] = entry.get("status") in ("succeeded", "failed") and (directory / "geometry.npz").is_file()
            info["viewCount"] = entry.get("sourceImageCount")
            if entry.get("status") == "succeeded":
                report = readJson(localFile(directory, "result.json"))
                inputs = readJson(localFile(directory, "inputs.json"))
                info.update({"stageSeconds": report.get("stageSeconds", {}),
                             "inferenceSeconds": report.get("inferenceSeconds"),
                             "viewCount": len(inputs), "acceptedCameraCount": report.get("acceptedCameraCount"),
                             "atlasCoverage": report.get("textureDiagnostics", {}).get("atlasCoverage"),
                             "cameras": [{k: c[k] for k in cameraKeys if k in c}
                                         for c in report.get("cameraDiagnostics", [])]})
                photos = []
                for i, item in enumerate(inputs):
                    filename = f"input{i:02d}.png"
                    localFile(directory, filename)
                    photo = {"index": i, "url": f"/cases/{caseId}/photos/{i}",
                             "name": item.get("name", ""), "sourceSize": item.get("orientedSize")}
                    points = np.asarray(item.get("faceLandmarksWorking", []), dtype=float)
                    size = np.asarray(item.get("workingSize", [512, 512]), dtype=float)
                    if points.shape == (5, 2) and size.shape == (2,) and np.all(size > 0):
                        photo["eyeCenter"] = (points[:2].mean(axis=0) / size).tolist()
                    photos.append(photo)
                info["photos"] = photos
                info["downloads"] = [{"label": "客户端 ZIP", "url": f"/cases/{caseId}/client/download"}]
                for filename, label in (("face.obj", "UFF OBJ"), ("face.mtl", "UFF MTL"),
                                         ("albedoPreview.png", "OBJ 配套贴图"), ("albedo.png", "观测贴图")):
                    if (directory / filename).is_file():
                        info["downloads"].append({"label": label, "url": f"/cases/{caseId}/downloads/{filename}"})
            elif entry.get("status") == "failed":
                if (directory / "inputs.json").is_file():
                    inputs = readJson(localFile(directory, "inputs.json"))
                    info["photos"] = [{"index": i, "url": f"/cases/{caseId}/photos/{i}",
                                       "name": item.get("name", ""), "sourceSize": item.get("orientedSize")}
                                      for i, item in enumerate(inputs) if (directory / f"input{i:02d}.png").is_file()]
                if not info.get("photos"):
                    info["photosAreSource"] = True
                    info["photos"] = [{"index": i, "url": f"/cases/{caseId}/sourcePhotos/{i}",
                                       "name": item.get("filename", ""), "sourceSize": item.get("orientedSize")}
                                      for i, item in enumerate(entry.get("files", []))
                                      if (directory.parent / "sourcePhotos" / f"view{i:02d}.png").is_file()]
                if (directory / "cameras.json").is_file():
                    cameras = readJson(localFile(directory, "cameras.json"))
                    info["cameras"] = [{k: c[k] for k in cameraKeys if k in c} for c in cameras]
                    info["acceptedCameraCount"] = sum(bool(c.get("valid")) for c in cameras)
            entries.append(info)
        return {"status": value.get("status", "running"), "cases": entries,
                "rotationDiagnosticsUrl": "/rotationDiagnostics" if (experiment / "rotationDiagnostics.json").is_file() else None}

    @app.get("/rotationDiagnostics")
    def rotationDiagnostics():
        return Response(localFile(experiment, "rotationDiagnostics.json").read_bytes(), media_type="application/json")

    @app.get("/scene.js")
    def scene():
        return Response(scripts["scene.js"], media_type="text/javascript")

    @app.get("/vendor/{filename}")
    def vendor(filename: str):
        if filename not in scripts or filename == "scene.js":
            raise HTTPException(404, "Unknown bundled script")
        return Response(scripts[filename], media_type="text/javascript")

    @app.get("/cases/{caseId}/{kind}/preview")
    def preview(caseId: str, kind: str):
        if kind not in ("uff", "client"):
            raise HTTPException(404, "Unknown mesh variant")
        values = loaded(caseId)
        if kind not in values:
            raise HTTPException(409, "本组客户端资产未完成")
        return values[kind]

    @app.get("/cases/{caseId}/uff/texture")
    def nativeTexture(caseId: str):
        texture = loaded(caseId)["uffTexture"]
        if texture is None:
            raise HTTPException(404, "本组仅有几何输出")
        return Response(texture, media_type="image/png")

    @app.get("/cases/{caseId}/client/textures/{materialIndex}")
    def clientTexture(caseId: str, materialIndex: int):
        values = loaded(caseId)
        if "asset" not in values:
            raise HTTPException(409, "本组客户端资产未完成")
        asset = values["asset"]
        materials = asset.materials["materials"]
        if not 0 <= materialIndex < len(materials):
            raise HTTPException(404, "Unknown material")
        name = materials[materialIndex]["baseColorTexture"]["path"]
        return Response(asset.payloads[name], media_type="image/png")

    @app.get("/cases/{caseId}/photos/{index}")
    def photo(caseId: str, index: int):
        entry, directory = selected(caseId)
        if entry.get("status") not in ("succeeded", "failed") or not 0 <= index < 32:
            raise HTTPException(404, "Unknown input view")
        inputs = readJson(localFile(directory, "inputs.json"))
        if index >= len(inputs):
            raise HTTPException(404, "Unknown input view")
        return Response(localFile(directory, f"input{index:02d}.png").read_bytes(), media_type="image/png")

    @app.get("/cases/{caseId}/client/download")
    def download(caseId: str):
        values = loaded(caseId)
        if "archive" not in values:
            raise HTTPException(409, "本组客户端资产未完成")
        return Response(values["archive"], media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{caseId}-clientAsset.zip"'})

    @app.get("/cases/{caseId}/sourcePhotos/{index}")
    def sourcePhoto(caseId: str, index: int):
        entry, directory = selected(caseId)
        if entry.get("status") != "failed" or not 0 <= index < len(entry.get("files", [])):
            raise HTTPException(404, "Unknown frozen source photo")
        name = f"sourcePhotos/view{index:02d}.png"
        return Response(localFile(directory.parent, name).read_bytes(), media_type="image/png")

    @app.get("/cases/{caseId}/downloads/{filename}")
    def nativeDownload(caseId: str, filename: str):
        types = {"face.obj": "text/plain", "face.mtl": "text/plain", "albedo.png": "image/png", "albedoPreview.png": "image/png"}
        if filename not in types:
            raise HTTPException(404, "Unknown artifact")
        entry, directory = selected(caseId)
        if entry.get("status") != "succeeded":
            raise HTTPException(409, "Case has not succeeded")
        return Response(localFile(directory, filename).read_bytes(), media_type=types[filename],
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/uffMultiCase20261011")
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18088)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    uvicorn.run(createApp(args.experiment), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
