"""Serve verified SH/OpenDelight client assets before/after WildCap on localhost.

Reads only completed clientComparison.json exports. No model or GPU startup.
Example: .venv/bin/python tools/serveShClientComparison.py --experiment runs/shWildCapClientComparison
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faceAppearance.webPreview import buildPreview
from exportShClientComparison import (NAMES, checkMesh, digest, loadVerifiedAsset, readArrays,
                                       readJson, validateInputs)


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#162522"><title>SH 去光 × WildCap · FaceAppearance</title><link rel="icon" href="data:,">
<style>
:root{font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif;color:#e8efea;background:#101a18;color-scheme:dark;font-synthesis:none}*{box-sizing:border-box}body{margin:0}button,a{font:inherit;touch-action:manipulation}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}button:focus-visible,a:focus-visible{outline:2px solid #bcebc8;outline-offset:3px}
header{height:72px;padding:0 5vw;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #ffffff12}.brand{font-size:17px;font-weight:650}.brand span{color:#a8d7b5;font-size:12px;font-weight:400}.badge{font-size:11px;color:#a8d7b5;border:1px solid #a8d7b52b;border-radius:20px;padding:7px 11px}
main{max-width:1560px;margin:auto;padding:28px 5vw}.heading{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:20px}.eyebrow{font-size:10px;letter-spacing:2.3px;color:#8caca0}h1{font-size:27px;font-weight:550;letter-spacing:-1px;margin:8px 0}p{font-size:12px;line-height:1.8;color:#9aaea5;margin:0}.method-hint{max-width:660px}
.choices{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-bottom:18px}.group-label{font-size:11px;color:#8ea89a;margin-right:2px}.switch,.modes,.angles,.lights{display:flex;gap:4px;flex-wrap:wrap}.switch{background:#0b1411;border:1px solid #ffffff15;padding:4px;border-radius:9px}.switch button{padding:10px 16px;border-radius:6px;background:none;border:0;color:#9aaeaa;font-size:12px}.switch button[aria-pressed=true]{background:#d4eccf;color:#193424}.divider{width:1px;height:25px;background:#ffffff1b;margin:0 4px}
.studio{border:1px solid #ffffff18;border-radius:14px;overflow:hidden;background:#14211d;box-shadow:0 22px 80px #0002}.toolbar{display:flex;gap:15px;align-items:center;justify-content:space-between;padding:15px 22px;border-bottom:1px solid #ffffff0c;flex-wrap:wrap}.label{font-size:14px;font-weight:600}.label small{font-size:10px;color:#8da399;font-weight:400;margin-left:10px}.control-row{display:flex;gap:15px;align-items:center;flex-wrap:wrap}.modes button,.angles button,.lights button{color:#99b0a4;border:1px solid transparent;border-radius:6px;padding:7px 10px;background:transparent;font-size:11px}.modes button[aria-pressed=true],.lights button[aria-pressed=true],.angles button:hover{border-color:#aac5aa35;color:#d4eccf;background:#d4eccf09}
.stage{position:relative;height:min(62vh,720px);min-height:360px;background:radial-gradient(ellipse at 50% 39%,#394c40 0,#263c30 39%,#18281f 75%,#14221c 100%)}#viewport{position:absolute;inset:0}.stage-top{position:absolute;top:20px;left:22px;pointer-events:none;color:#96ada0;font-size:10px;letter-spacing:1px}.dot{display:inline-block;width:6px;height:6px;margin-right:8px;background:#bbdfb9;border-radius:100%}.angles{position:absolute;bottom:18px;left:18px;background:#122319bb;padding:3px;border-radius:7px;backdrop-filter:blur(8px)}.help{position:absolute;right:22px;bottom:27px;color:#94aa9c;font-size:10px;pointer-events:none}#loading{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);border:1px solid #ffffff21;border-radius:10px;background:#0e1e18ed;padding:20px 28px;font-size:13px;text-align:center;max-width:90%;pointer-events:none}
.bottom{display:flex;align-items:center;justify-content:space-between;gap:24px;padding:19px 22px}.summary{font-size:11px;color:#8ea69a}.summary strong{display:block;font-size:14px;font-weight:500;color:#d9e6dc;margin-bottom:7px}.download{font-size:11px;color:#203b25;text-decoration:none;background:#c5e3be;padding:11px 15px;border-radius:6px;white-space:nowrap}.footnote{margin-top:14px;display:flex;justify-content:space-between;gap:24px}.footnote p{font-size:11px;color:#7f998b;max-width:990px}.status{font-size:11px;color:#a8d7b5;text-align:right;min-width:140px}.timing{color:#b9d2bd;font-size:11px;white-space:nowrap}
@media(max-width:850px){header{height:62px}.heading{display:block}main{padding-top:23px}.choices{gap:8px}.switch button{padding:9px 12px}.divider{display:none}.toolbar{padding:14px}.label small{display:none}.control-row{gap:8px}.stage{min-height:430px;height:56vh}.help{bottom:61px;font-size:9px;right:15px}.bottom{padding:17px;align-items:start;flex-direction:column;gap:15px}.footnote{display:block}.status{text-align:left;margin-top:8px}h1{font-size:24px}}
</style><script type="importmap">{"imports":{"three":"/vendor/three.module.js"}}</script></head><body>
<header><div class="brand">FaceAppearance <span>· 去光实验室</span></div><span class="badge">同一组照片 · 固定几何</span></header>
<main><section class="heading"><div><span class="eyebrow">DELIGHTING × MATERIAL REFINEMENT</span><h1>SH 去光与 WildCap 对照</h1><p id="sampling" class="method-hint">正在读取实验设置…</p></div><span class="timing" id="timing"></span></section>
<section class="choices" aria-label="对照方法"><span class="group-label">去光</span><div class="switch" role="group" aria-label="去光方式"><button data-method="openDelight" aria-pressed="true">OpenDelight</button><button data-method="shOnly" aria-pressed="false">SH 去光</button><button data-method="shPrior" aria-pressed="false">SH + 先验</button></div><span class="divider"></span><span class="group-label">WildCap</span><div class="switch" role="group" aria-label="WildCap 处理"><button data-wildcap="false" aria-pressed="true">处理前</button><button data-wildcap="true" aria-pressed="false">处理后</button></div></section>
<section class="studio" aria-label="三维材质对照"><div class="toolbar"><span class="label" id="branch-title">OpenDelight<small>固定网格 · 固定观察角度</small></span><div class="control-row"><div class="modes" role="group" aria-label="显示模式"><button data-mode="unlit" aria-pressed="true">无光照基色</button><button data-mode="textured" aria-pressed="false">统一照明</button><button data-mode="clay" aria-pressed="false">灰模</button></div><div class="lights" role="group" aria-label="照明方向"><button data-light="front" aria-pressed="true">正面光</button><button data-light="left" aria-pressed="false">左侧光</button><button data-light="right" aria-pressed="false">右侧光</button></div></div></div>
<div class="stage"><div id="viewport"></div><div class="stage-top"><span class="dot"></span><span id="stage-label">无光照基色 · 保留贴图本身的明暗</span></div><div id="loading" role="status" aria-live="polite">正在读取资产…</div><div class="angles" role="group" aria-label="观察角度"><button data-view="front">正面</button><button data-view="threequarter">四分之三</button><button data-view="left">左侧</button><button data-view="right">右侧</button></div><span class="help">拖动旋转 · 滚轮缩放 · 右键平移</span></div>
<div class="bottom"><div class="summary"><strong id="method-note">正在载入对照</strong><span id="asset-counts">客户端 ZIP 正在校验</span></div><a class="download" id="download" href="/assets/openDelight/download">下载当前 ZIP ↓</a></div></section>
<div class="footnote"><p id="comparison-note">照片覆盖区、融合规则与模板补全规则相同。WildCap 对照只替换合格区域的漫反射颜色，统一使用网格法线与预设粗糙度。优先观察眉毛、唇线、鼻翼、眼下，以及换光后仍固定在脸上的阴影。</p><span id="status" class="status" role="status" aria-live="polite">正在加载</span></div></main><noscript>请启用 JavaScript 以查看三维对照。</noscript>
<script type="module">
import * as THREE from 'three';
import { FaceViewer } from '/scene.js';
const $ = value => document.querySelector(value);
const names = {openDelight: 'OpenDelight', shOnly: 'SH 去光', shPrior: 'SH + OpenDelight 先验'};
const notes = {openDelight: '缓存的逐视图去光结果，以共同支持区重新融合。', shOnly: '从照片与固定网格法线拟合每视图 SH 光照。', shPrior: '用 OpenDelight 约束低频颜色，再以 SH 保留照片细节。'};
const cache = new Map();
let viewer, metadata, busy = false, method = 'openDelight', after = false, mode = 'unlit', light = 'front';
function fail(error) { $('#loading').hidden = false; $('#loading').textContent = `无法载入三维预览：${error.message}`; $('#status').textContent = '加载失败'; }
async function checkedFetch(url) { const response = await fetch(url, {credentials: 'same-origin'}); if (!response.ok) throw new Error(`资源读取失败 (${response.status})`); return response; }
function syncButtons() {
  document.querySelectorAll('button').forEach(button => { button.disabled = busy; });
  document.querySelectorAll('[data-method]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.method === method)));
  document.querySelectorAll('[data-wildcap]').forEach(button => {
    const value = button.dataset.wildcap === 'true';
    button.setAttribute('aria-pressed', String(value === after));
    button.disabled = busy || (value && !metadata?.entries.some(entry => entry.method === method && entry.wildCap));
  });
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
  $('#stage-label').textContent = mode === 'unlit' ? '无光照基色 · 保留贴图本身的明暗' : mode === 'clay' ? '灰模 · 相同几何与材质参数' : '统一照明 · 相同网格法线、粗糙度与白光';
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
async function loadBranch(nextMethod, nextAfter) {
  if (busy) return;
  const entry = metadata.entries.find(item => item.method === nextMethod && item.wildCap === nextAfter);
  if (!entry) return;
  busy = true; syncButtons();
  const old = viewer.hasBounds ? {position: viewer.camera.position.clone(), target: viewer.controls.target.clone(), up: viewer.camera.up.clone()} : null;
  $('#loading').hidden = false; $('#loading').textContent = `正在载入${entry.label}…`; $('#status').textContent = '正在加载';
  try {
    if (!cache.has(entry.name)) cache.set(entry.name, await (await checkedFetch(`/assets/${entry.name}/preview`)).json());
    const preview = cache.get(entry.name);
    await viewer.load(preview, async url => (await checkedFetch(url)).blob());
    if (old) {
      const damping = viewer.controls.enableDamping; viewer.controls.enableDamping = false;
      viewer.camera.position.copy(old.position); viewer.controls.target.copy(old.target); viewer.camera.up.copy(old.up);
      viewer.controls.update(); viewer.controls.enableDamping = damping;
    }
    method = nextMethod; after = nextAfter; applyMode(); applyLight();
    $('#branch-title').replaceChildren(document.createTextNode(entry.label));
    const subtitle = document.createElement('small'); subtitle.textContent = '固定网格 · 固定观察角度'; $('#branch-title').append(subtitle);
    const support = entry.wildCapValidTexels ? ` WildCap 替换共同覆盖区的 ${(100 * entry.wildCapValidTexels / entry.commonCoverageTexels).toFixed(1)}%。` : '';
    $('#method-note').textContent = notes[method] + support;
    $('#asset-counts').textContent = `${preview.counts.sourceVertices.toLocaleString()} 个源顶点　${preview.counts.faces.toLocaleString()} 个三角面　共同覆盖 ${entry.commonCoverageTexels.toLocaleString()} 个 UV 像素`;
    const seconds = [];
    if (Number.isFinite(entry.frontEndSeconds)) seconds.push(`${method === 'openDelight' ? 'OpenDelight 原任务' : 'SH 拟合'} ${entry.frontEndSeconds.toFixed(2)} s`);
    if (Number.isFinite(entry.wildCapSeconds)) seconds.push(`WildCap ${entry.wildCapSeconds.toFixed(1)} s`);
    $('#timing').textContent = seconds.join('　+　');
    $('#download').href = `/assets/${entry.name}/download`;
    $('#download').textContent = `下载${entry.label} ZIP ↓`;
    $('#loading').hidden = true; $('#status').textContent = '资产校验通过 · 已就绪';
  } catch (error) { fail(error); }
  finally { busy = false; syncButtons(); }
}
try {
  metadata = await (await checkedFetch('/comparison')).json();
  const parameters = metadata.parameters;
  const working = parameters.rawSampling === 'working' || parameters.originalResolutionRawSampling === false;
  const sizes = parameters.workingImageSizesWh || [];
  const size = sizes.length && sizes.every(item => item[0] === sizes[0][0] && item[1] === sizes[0][1]) ? `${sizes[0][0]} × ${sizes[0][1]}` : '相同工作分辨率';
  $('#sampling').textContent = working ? `SH 使用 ${size} 工作照片；OpenDelight 使用缓存输出。固定相机、UV 对应与共同覆盖区。` : 'SH 使用原始分辨率照片；OpenDelight 使用缓存输出。固定相机、UV 对应与共同覆盖区。';
  if (!metadata.entries.some(entry => entry.wildCap)) $('#comparison-note').textContent = '当前已导出三种去光结果，WildCap 处理后资产尚未加入此预览。统一使用网格法线与预设粗糙度。优先观察眉毛、唇线、鼻翼和眼下阴影。';
  viewer = new FaceViewer($('#viewport'), {onError: fail});
  document.querySelectorAll('[data-method]').forEach(button => button.addEventListener('click', () => loadBranch(button.dataset.method, after)));
  document.querySelectorAll('[data-wildcap]').forEach(button => button.addEventListener('click', () => loadBranch(method, button.dataset.wildcap === 'true')));
  document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => { mode = button.dataset.mode; applyMode(); }));
  document.querySelectorAll('[data-light]').forEach(button => button.addEventListener('click', () => { light = button.dataset.light; if (mode === 'unlit') mode = 'textured'; applyMode(); applyLight(); }));
  document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => {
    if (button.dataset.view !== 'threequarter') { viewer.setView(button.dataset.view); return; }
    viewer.setView('front'); const offset = viewer.fitDistance / Math.sqrt(2);
    viewer.camera.position.set(viewer.center.x + offset, viewer.center.y, viewer.center.z + offset); viewer.controls.update();
  }));
  window.addEventListener('pagehide', () => viewer.dispose(), {once: true});
  await loadBranch('openDelight', false);
} catch (error) { fail(error); }
</script></body></html>'''


def createApp(experiment: Path) -> FastAPI:
    experiment = Path(experiment).resolve()
    manifest = readJson(experiment / "clientComparison.json")
    if manifest.get("status") not in {"complete", "frontEndsOnly"}:
        raise ValueError("A committed client comparison export is required")
    source = Path(manifest["sourceRun"]).resolve()
    report, _, sourceHashes = validateInputs(source, experiment, manifest["status"] == "complete")
    if manifest["parameters"] != report["parameters"] or manifest["sourceHashes"] != sourceHashes:
        raise ValueError("Client comparison metadata differs from the experiment")
    templateManifest = readJson(source / "assembly/manifest.json")
    sourceMesh = readArrays(source / "assembly/clientMesh.npz")
    expectedNames = {name + suffix for name in NAMES for suffix in
                     (("", "WildCap") if manifest["status"] == "complete" else ("",))}
    if {entry["name"] for entry in manifest["entries"]} != expectedNames or len(manifest["entries"]) != len(expectedNames):
        raise ValueError("Client comparison must contain exactly its declared three or six variants")
    assets, previews, archives = {}, {}, {}
    for entry in manifest["entries"]:
        name = entry["name"]
        if entry["directory"] != f"clientAssets/{name}":
            raise ValueError("Client asset directory must be its fixed comparison path")
        method = entry["method"]
        after = entry["wildCap"]
        if method not in NAMES or type(after) is not bool or name != method + ("WildCap" if after else ""):
            raise ValueError("Client variant method/WildCap identity differs")
        identity = entry["inputIdentity"]
        expectedFields = {"method": method, "wildCap": after, "sourceRun": str(source),
                          "sourceHashes": sourceHashes, "parameters": report["parameters"],
                          "textureSha256": digest(experiment / method / "texture.npz"),
                          "viewTransformsSha256": digest(source / "viewTransforms.json"),
                          "commonSupportSha256": digest(experiment / "commonSupport.npz"),
                          "templateManifestSha256": readJson(source / "assembly/diagnostics.json")["assetManifestSha256"]}
        if after:
            expectedFields["wildCapMaterialsSha256"] = digest(experiment / method / "wildcapUff/materials.npz")
            expectedFields["wildCapDiagnosticsSha256"] = digest(experiment / method / "wildcapIct/diagnostics.json")
        if any(identity.get(key) != value for key, value in expectedFields.items()):
            raise ValueError(f"Client export inputs changed: {name}")
        asset, metadata = loadVerifiedAsset(experiment / entry["directory"], templateManifest, entry["inputIdentity"])
        if entry["archive"] != metadata["archive"]:
            raise ValueError("Client comparison archive identity differs")
        checkMesh(sourceMesh, asset.mesh)
        payload = asset.archivePath.read_bytes()
        if hashlib.sha256(payload).hexdigest() != asset.archiveSha256:
            raise ValueError("Client ZIP changed during comparison startup")
        preview = buildPreview(asset, name)
        for material in preview["materials"]:
            material["texture"]["url"] = f"/assets/{name}/textures/{material['index']}"
        assets[name], previews[name], archives[name] = asset, preview, payload
    # Serve only client-facing summary fields, fixed script names and verified
    # in-memory ZIP payloads; never expose arbitrary workspace paths or files.
    summary = {key: manifest[key] for key in ("status", "parameters", "commonCoverageTexels", "comparison")}
    summary["entries"] = [{key: entry[key] for key in ("name", "method", "wildCap", "label", "frontEndSeconds",
                          "wildCapSeconds", "wildCapValidTexels", "commonCoverageTexels")} for entry in manifest["entries"]]
    web = ROOT / "src/faceAppearance/web"
    scripts = {"/scene.js": (web / "scene.js").read_bytes()}
    scripts.update({f"/vendor/{name}": (web / "vendor" / name).read_bytes()
                    for name in ("three.module.js", "three.core.js", "OrbitControls.js")})
    app = FastAPI(title="Local SH and WildCap comparison", docs_url=None, redoc_url=None, openapi_url=None)
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
    def index():
        return HTMLResponse(HTML)

    @app.get("/comparison")
    def comparison():
        return summary

    @app.get("/scene.js")
    def scene():
        return Response(scripts["/scene.js"], media_type="text/javascript")

    @app.get("/vendor/{filename}")
    def vendor(filename: str):
        path = f"/vendor/{filename}"
        if path not in scripts:
            raise HTTPException(404, "Unknown bundled script")
        return Response(scripts[path], media_type="text/javascript")

    @app.get("/assets/{name}/preview")
    def preview(name: str):
        selected(name)
        return previews[name]

    @app.get("/assets/{name}/textures/{materialIndex}")
    def texture(name: str, materialIndex: int):
        asset = selected(name)
        materials = asset.materials["materials"]
        if not 0 <= materialIndex < len(materials):
            raise HTTPException(404, "Unknown material")
        texturePath = materials[materialIndex]["baseColorTexture"]["path"]
        return Response(asset.payloads[texturePath], media_type="image/png")

    @app.get("/assets/{name}/download")
    def download(name: str):
        selected(name)
        return Response(archives[name], media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{name}-clientAsset.zip"'})

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18085)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    uvicorn.run(createApp(args.experiment), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
