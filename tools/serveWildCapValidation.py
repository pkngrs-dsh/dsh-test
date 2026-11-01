"""Serve five verified WildCap usage candidates on localhost, without models."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

from presentWildCapValidation import (NAMES, ROOT, checkMesh, loadVerifiedAsset,
                                      readArrays, readJson, validateExperiment,
                                      variantIdentity)
from faceAppearance.webPreview import buildPreview


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#162522"><title>WildCap 用法验证 · FaceAppearance</title><link rel="icon" href="data:,">
<style>
:root{font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif;color:#e8efea;background:#101a18;color-scheme:dark;font-synthesis:none}*{box-sizing:border-box}body{margin:0}button,a{font:inherit;touch-action:manipulation}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}button:focus-visible,a:focus-visible{outline:2px solid #bcebc8;outline-offset:3px}
header{padding:23px 5vw;border-bottom:1px solid #ffffff12;display:flex;justify-content:space-between;align-items:center}.brand{font-size:17px;font-weight:650}.brand span{color:#a8d7b5;font-size:12px;font-weight:400}.badge{font-size:11px;color:#a8d7b5;border:1px solid #a8d7b52b;border-radius:20px;padding:7px 11px}main{max-width:1560px;margin:auto;padding:26px 5vw}.heading{display:flex;justify-content:space-between;align-items:end;gap:22px;margin-bottom:20px}.eyebrow{font-size:10px;letter-spacing:2px;color:#8caca0}h1{font-size:27px;font-weight:550;letter-spacing:-1px;margin:8px 0}p{font-size:12px;line-height:1.8;color:#9aaea5;margin:0}.timing{font-size:11px;color:#b9d2bd;white-space:nowrap}.choices{display:flex;flex-wrap:wrap;gap:5px;padding:5px;border-radius:10px;background:#0b1411;border:1px solid #ffffff15;margin-bottom:18px}.choices button{padding:11px 15px;border:0;border-radius:6px;background:none;color:#9aaeaa;font-size:12px}.choices button[aria-pressed=true]{background:#d4eccf;color:#193424}
.studio{border:1px solid #ffffff18;border-radius:14px;overflow:hidden;background:#14211d;box-shadow:0 22px 80px #0002}.toolbar{padding:14px 20px;display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;border-bottom:1px solid #ffffff0c}.label{font-size:14px;font-weight:600}.label small{font-size:10px;color:#8da399;font-weight:400;margin-left:10px}.controls,.modes,.lights,.angles{display:flex;gap:4px;flex-wrap:wrap}.controls{gap:14px}.modes button,.lights button,.angles button{color:#99b0a4;border:1px solid transparent;border-radius:6px;padding:7px 10px;background:transparent;font-size:11px}.modes button[aria-pressed=true],.lights button[aria-pressed=true],.angles button:hover{border-color:#aac5aa35;color:#d4eccf;background:#d4eccf09}.stage{position:relative;height:min(62vh,720px);min-height:370px;background:radial-gradient(ellipse at 50% 39%,#394c40 0,#263c30 39%,#18281f 75%,#14221c 100%)}#viewport{position:absolute;inset:0}.stage-top{position:absolute;top:20px;left:22px;pointer-events:none;color:#96ada0;font-size:10px;letter-spacing:1px}.dot{display:inline-block;width:6px;height:6px;margin-right:8px;background:#bbdfb9;border-radius:100%}.angles{position:absolute;bottom:18px;left:18px;background:#122319bb;padding:3px;border-radius:7px}.help{position:absolute;right:22px;bottom:27px;color:#94aa9c;font-size:10px;pointer-events:none}#loading{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);border:1px solid #ffffff21;border-radius:10px;background:#0e1e18ed;padding:20px 28px;font-size:13px;max-width:90%;pointer-events:none}.bottom{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:19px 22px}.summary{font-size:11px;color:#8ea69a}.summary strong{display:block;font-size:13px;font-weight:500;color:#d9e6dc;margin-bottom:8px}.download{font-size:11px;color:#203b25;text-decoration:none;background:#c5e3be;padding:11px 15px;border-radius:6px;white-space:nowrap}.footnote{margin-top:14px;display:flex;justify-content:space-between;gap:24px}.footnote p{font-size:11px;color:#7f998b;max-width:990px}.status{font-size:11px;color:#a8d7b5;text-align:right;min-width:160px}
@media(max-width:850px){.heading{display:block}.timing{display:block;margin-top:10px}.badge{display:none}main{padding-top:23px}.toolbar{padding:14px}.label small{display:none}.controls{gap:8px}.stage{min-height:430px;height:56vh}.help{bottom:61px;right:15px}.bottom{padding:17px;align-items:start;flex-direction:column}.footnote{display:block}.status{text-align:left;margin-top:8px}h1{font-size:24px}}
</style><script type="importmap">{"imports":{"three":"/vendor/three.module.js"}}</script></head><body>
<header><div class="brand">FaceAppearance <span>· 材质验证</span></div><span class="badge">候选配置 · 固定几何与输入</span></header>
<main><section class="heading"><div><span class="eyebrow">WILDCAP USAGE VALIDATION</span><h1>光照拟合与回传边界对照</h1><p>同一份 SH + OpenDelight 先验输入，分开观察光照调度、局部阴影和边界过渡的影响。</p></div><span class="timing" id="timing"></span></section>
<section class="choices" id="choices" role="group" aria-label="WildCap 验证候选"></section>
<section class="studio" aria-label="三维材质对照"><div class="toolbar"><span class="label" id="branch-title">正在载入候选</span><div class="controls"><div class="modes" role="group" aria-label="显示模式"><button data-mode="unlit" aria-pressed="true">无光照基色</button><button data-mode="textured" aria-pressed="false">统一照明</button><button data-mode="clay" aria-pressed="false">灰模</button></div><div class="lights" role="group" aria-label="照明方向"><button data-light="front" aria-pressed="true">正面光</button><button data-light="left" aria-pressed="false">左侧光</button><button data-light="right" aria-pressed="false">右侧光</button></div></div></div>
<div class="stage"><div id="viewport"></div><div class="stage-top"><span class="dot"></span><span id="stage-label">无光照基色 · 观察贴图本身的明暗</span></div><div id="loading" role="status" aria-live="polite">正在读取已校验资产…</div><div class="angles" role="group" aria-label="观察角度"><button data-view="front">正面</button><button data-view="threequarter">四分之三</button><button data-view="left">左侧</button><button data-view="right">右侧</button></div><span class="help">拖动旋转 · 滚轮缩放 · 右键平移</span></div>
<div class="bottom"><div class="summary"><strong id="method-note">固定客户端网格与材质参数</strong><span id="asset-counts">正在校验 ZIP 资产</span></div><a class="download" id="download" href="/assets/input/download">下载当前 ZIP ↓</a></div></section>
<div class="footnote"><p>五组共用照片覆盖区、网格法线和预设粗糙度，仅比较漫反射颜色。局部阴影 + 边界过渡复用同一份 WildCap 推理结果。优先切换无光照和左右光，观察鼻侧、眼下与脸颊边界。所有版本均为验证候选。</p><span id="status" class="status" role="status" aria-live="polite">正在加载</span></div></main><noscript>请启用 JavaScript 以查看三维对照。</noscript>
<script type="module">
import * as THREE from 'three';
import { FaceViewer } from '/scene.js';
const $ = value => document.querySelector(value);
const cache = new Map();
let viewer, metadata, busy = false, selected = 'input', mode = 'unlit', light = 'front';
function fail(error) { $('#loading').hidden = false; $('#loading').textContent = `无法载入三维预览：${error.message}`; $('#status').textContent = '加载失败'; }
async function checkedFetch(url) { const response = await fetch(url, {credentials: 'same-origin'}); if (!response.ok) throw new Error(`资源读取失败 (${response.status})`); return response; }
function syncButtons() {
  document.querySelectorAll('button').forEach(button => { button.disabled = busy; });
  document.querySelectorAll('[data-variant]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.variant === selected)));
  document.querySelectorAll('[data-mode]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.mode === mode)));
  document.querySelectorAll('[data-light]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.light === light)));
}
function applyMode() {
  viewer.setMode(mode === 'unlit' ? 'textured' : mode);
  if (mode === 'unlit' && viewer.asset) {
    const asset = viewer.asset;
    if (!asset.modes.unlit) {
      asset.modes.unlit = asset.modes.textured.map(material => new THREE.MeshBasicMaterial({
        name: `${material.name}:unlit`, color: material.color.clone(), map: material.map,
        opacity: material.opacity, alphaTest: material.alphaTest, side: material.side,
        transparent: false, toneMapped: false,
      }));
      asset.materials.push(...asset.modes.unlit);
    }
    asset.mesh.material = asset.modes.unlit;
  }
  $('#stage-label').textContent = mode === 'unlit' ? '无光照基色 · 观察贴图本身的明暗' : mode === 'clay' ? '灰模 · 五组几何完全相同' : '统一照明 · 相同网格法线、粗糙度与白光';
  syncButtons();
}
function applyLight() {
  const positions = {front: [0, 1, 4], left: [-3, 1, 3], right: [3, 1, 3]};
  const directionals = viewer.lightRig.children.filter(item => item.isDirectionalLight);
  directionals.forEach((item, index) => { item.color.set(0xffffff); item.intensity = index === 0 ? 2.5 : 0; });
  directionals[0].position.fromArray(positions[light]);
  viewer.scene.children.filter(item => item.isHemisphereLight).forEach(item => {
    item.color.set(0xffffff); item.groundColor.set(0xffffff); item.intensity = 0.35;
  });
  syncButtons();
}
async function loadVariant(name) {
  if (busy) return;
  const entry = metadata.entries.find(item => item.name === name);
  if (!entry) return;
  busy = true; syncButtons();
  // Finish residual orbit motion once, then freeze input while textures load.
  // FaceViewer.load resets framing internally; restore this exact inspection
  // camera before the next animation frame can display the new material.
  const controlState = {enabled: viewer.controls.enabled, damping: viewer.controls.enableDamping,
                        autoRotate: viewer.controls.autoRotate};
  viewer.controls.enableDamping = false; viewer.controls.autoRotate = false;
  viewer.controls.update(); viewer.controls.enabled = false;
  const old = viewer.hasBounds ? {position: viewer.camera.position.clone(), target: viewer.controls.target.clone(), up: viewer.camera.up.clone(), zoom: viewer.camera.zoom} : null;
  $('#loading').hidden = false; $('#loading').textContent = `正在载入${entry.label}…`; $('#status').textContent = '正在加载';
  try {
    if (!cache.has(name)) cache.set(name, await (await checkedFetch(`/assets/${name}/preview`)).json());
    const preview = cache.get(name);
    await viewer.load(preview, async url => (await checkedFetch(url)).blob());
    if (old) {
      viewer.camera.position.copy(old.position); viewer.controls.target.copy(old.target); viewer.camera.up.copy(old.up);
      viewer.camera.zoom = old.zoom; viewer.camera.updateProjectionMatrix();
      viewer.controls.update();
    }
    selected = name; applyMode(); applyLight();
    $('#branch-title').replaceChildren(document.createTextNode(entry.label));
    const subtitle = document.createElement('small'); subtitle.textContent = '候选验证 · 固定观察角度'; $('#branch-title').append(subtitle);
    $('#method-note').textContent = entry.description;
    $('#asset-counts').textContent = `${preview.counts.sourceVertices.toLocaleString()} 个源顶点　${preview.counts.faces.toLocaleString()} 个三角面　共同覆盖 ${entry.commonCoverageTexels.toLocaleString()} 个 UV 像素`;
    $('#timing').textContent = Number.isFinite(entry.seconds) ? `记录处理耗时 ${entry.seconds.toFixed(1)} s${name === 'localTglFeather' ? ' · 复用局部阴影推理' : ''}` : name === 'input' ? '复用固定输入' : '';
    $('#download').href = `/assets/${name}/download`; $('#download').textContent = `下载${entry.label} ZIP ↓`;
    $('#loading').hidden = true; $('#status').textContent = '五组资产校验通过 · 已就绪';
  } catch (error) { fail(error); }
  finally {
    viewer.controls.enabled = controlState.enabled; viewer.controls.enableDamping = controlState.damping;
    viewer.controls.autoRotate = controlState.autoRotate; busy = false; syncButtons();
  }
}
try {
  metadata = await (await checkedFetch('/comparison')).json();
  viewer = new FaceViewer($('#viewport'), {onError: fail});
  for (const entry of metadata.entries) {
    const button = document.createElement('button'); button.dataset.variant = entry.name; button.textContent = entry.label;
    button.setAttribute('aria-pressed', String(entry.name === selected)); button.addEventListener('click', () => loadVariant(entry.name));
    $('#choices').append(button);
  }
  document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => { mode = button.dataset.mode; applyMode(); }));
  document.querySelectorAll('[data-light]').forEach(button => button.addEventListener('click', () => { light = button.dataset.light; if (mode === 'unlit') mode = 'textured'; applyMode(); applyLight(); }));
  document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => {
    if (button.dataset.view !== 'threequarter') { viewer.setView(button.dataset.view); return; }
    viewer.setView('front'); const offset = viewer.fitDistance / Math.sqrt(2);
    viewer.camera.position.set(viewer.center.x + offset, viewer.center.y, viewer.center.z + offset); viewer.controls.update();
  }));
  window.addEventListener('pagehide', () => viewer.dispose(), {once: true});
  await loadVariant('input');
} catch (error) { fail(error); }
</script></body></html>'''


def createApp(experiment: Path) -> FastAPI:
    data = validateExperiment(experiment)
    experiment, source = data["experiment"], data["source"]
    manifest = readJson(experiment / "clientComparison.json")
    if manifest.get("status") != "complete" or manifest.get("inputIdentity") != data["identity"]:
        raise ValueError("A complete client export matching the current validation is required")
    entries = manifest["entries"]
    if [entry["name"] for entry in entries] != list(NAMES):
        raise ValueError("Client export must include the five ordered validation candidates")
    templateManifest = readJson(source / "assembly/manifest.json")
    if any(manifest.get(key) != templateManifest[key] for key in ("templateId", "templateHashes")):
        raise ValueError("Comparison and source template identities differ")
    sourceMesh = readArrays(source / "assembly/clientMesh.npz")
    assets, previews, archives = {}, {}, {}
    for entry in entries:
        name = entry["name"]
        if entry["directory"] != f"clientAssets/{name}" or entry["inputIdentity"] != variantIdentity(data, name):
            raise ValueError(f"Client candidate input identity differs: {name}")
        asset, metadata = loadVerifiedAsset(experiment / entry["directory"], templateManifest, entry["inputIdentity"])
        if entry["archive"] != metadata["archive"]:
            raise ValueError(f"Client candidate archive identity differs: {name}")
        checkMesh(sourceMesh, asset.mesh)
        payload = asset.archivePath.read_bytes()
        if hashlib.sha256(payload).hexdigest() != asset.archiveSha256:
            raise ValueError(f"Client ZIP changed during startup: {name}")
        preview = buildPreview(asset, name)
        for material in preview["materials"]:
            material["texture"]["url"] = f"/assets/{name}/textures/{material['index']}"
        assets[name], previews[name], archives[name] = asset, preview, payload
    summary = {key: manifest[key] for key in ("status", "commonCoverageTexels", "comparison")}
    summary["entries"] = [{key: entry[key] for key in ("name", "label", "description", "seconds",
                                                       "commonCoverageTexels", "candidateOnly")} for entry in entries]
    web = ROOT / "src/faceAppearance/web"
    scripts = {"/scene.js": (web / "scene.js").read_bytes()}
    scripts.update({f"/vendor/{name}": (web / "vendor" / name).read_bytes()
                    for name in ("three.module.js", "three.core.js", "OrbitControls.js")})
    app = FastAPI(title="Local WildCap usage validation", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.middleware("http")
    async def localHeaders(request, callNext):
        response = await callNext(request)
        response.headers.update({"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store",
                                 "Referrer-Policy": "same-origin", "Content-Security-Policy":
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"})
        return response

    def selected(name):
        if name not in assets:
            raise HTTPException(404, "Unknown comparison asset")
        return assets[name]

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTMLResponse(HTML)

    @app.get("/comparison")
    async def comparison():
        return summary

    @app.get("/scene.js")
    async def scene():
        return Response(scripts["/scene.js"], media_type="text/javascript")

    @app.get("/vendor/{filename}")
    async def vendor(filename: str):
        path = f"/vendor/{filename}"
        if path not in scripts:
            raise HTTPException(404, "Unknown bundled script")
        return Response(scripts[path], media_type="text/javascript")

    @app.get("/assets/{name}/preview")
    async def preview(name: str):
        selected(name)
        return previews[name]

    @app.get("/assets/{name}/textures/{materialIndex}")
    async def texture(name: str, materialIndex: int):
        asset = selected(name)
        materials = asset.materials["materials"]
        if not 0 <= materialIndex < len(materials):
            raise HTTPException(404, "Unknown material")
        texturePath = materials[materialIndex]["baseColorTexture"]["path"]
        return Response(asset.payloads[texturePath], media_type="image/png")

    @app.get("/assets/{name}/download")
    async def download(name: str):
        selected(name)
        return Response(archives[name], media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{name}-clientAsset.zip"'})

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "runs/wildCapUsageValidation")
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18086)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    uvicorn.run(createApp(args.experiment), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
