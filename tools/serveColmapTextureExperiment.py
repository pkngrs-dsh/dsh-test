"""Local, read-only comparison of two verified client ZIPs; no model startup.

Example:
    .venv/bin/python tools/serveColmapTextureExperiment.py \
        --source runs/service/<job>/output --experiment runs/colmapTextureExperiment
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from faceAppearance.clientImport import loadClientAsset
from faceAppearance.webPreview import buildPreview


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#162522"><title>贴图融合对照 · FaceAppearance</title>
<link rel="icon" href="data:,">
<style>
:root{font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif;color:#e8efea;background:#101a18;font-synthesis:none;color-scheme:dark}
*{box-sizing:border-box}body{margin:0}button,a{font:inherit}button{cursor:pointer}button:disabled{cursor:wait;opacity:.45}
button,a{touch-action:manipulation}button:focus-visible,a:focus-visible{outline:2px solid #bcebc8;outline-offset:4px}
header{height:82px;padding:0 5vw;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #ffffff12}
.brand{font-size:17px;font-weight:700;letter-spacing:-.4px}.brand span{color:#a8d7b5}.eyebrow{font-size:10px;letter-spacing:2.4px;color:#8caca0}
.badge{font-size:11px;color:#a8d7b5;border:1px solid #a8d7b52b;border-radius:20px;padding:7px 11px}
main{max-width:1560px;margin:auto;padding:34px 5vw 28px}.heading{display:flex;justify-content:space-between;gap:24px;align-items:end;margin-bottom:24px}
h1{font-size:27px;font-weight:550;letter-spacing:-1px;margin:8px 0 10px}p{font-size:13px;line-height:1.8;color:#9aaea5;margin:0}
.switch,.modes,.angles{display:flex;gap:5px;flex-wrap:wrap}.switch{background:#0b1411;border:1px solid #ffffff15;padding:5px;border-radius:11px}
.switch button{padding:11px 19px;border-radius:7px;background:none;border:0;color:#9aaeaa;font-size:12px}
.switch button[aria-pressed=true]{background:#d4eccf;color:#193424;box-shadow:0 3px 12px #0002}
.studio{border:1px solid #ffffff18;border-radius:15px;overflow:hidden;background:#14211d;box-shadow:0 22px 80px #0002}
.toolbar{display:flex;gap:20px;align-items:center;justify-content:space-between;padding:18px 24px;border-bottom:1px solid #ffffff0c}
.label{font-size:14px;font-weight:600}.label small{font-size:11px;color:#8da399;font-weight:400;margin-left:12px}
.modes button,.angles button{color:#99b0a4;border:1px solid transparent;border-radius:6px;padding:7px 11px;background:transparent;font-size:11px}
.modes button[aria-pressed=true],.angles button:hover{border-color:#aac5aa35;color:#d4eccf;background:#d4eccf09}
.stage{position:relative;height:min(65vh,760px);min-height:360px;background:radial-gradient(ellipse at 50% 39%,#394c40 0,#263c30 39%,#18281f 75%,#14221c 100%)}
#viewport{position:absolute;inset:0}.stage-top{position:absolute;top:22px;left:24px;pointer-events:none;color:#96ada0;font-size:10px;letter-spacing:1.3px}
.dot{display:inline-block;width:6px;height:6px;margin-right:8px;background:#bbdfb9;border-radius:100%;box-shadow:0 0 12px #bbdfb970}
.angles{position:absolute;bottom:20px;left:22px;background:#122319bb;padding:3px;border-radius:7px;backdrop-filter:blur(8px)}
.help{position:absolute;right:24px;bottom:30px;color:#94aa9c;font-size:10px;pointer-events:none}
#loading{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);border:1px solid #ffffff21;border-radius:10px;background:#0e1e18ed;padding:20px 28px;font-size:13px;text-align:center;max-width:90%;pointer-events:none}
.bottom{display:flex;align-items:center;justify-content:space-between;gap:24px;padding:22px 24px}.summary{font-size:12px;color:#8ea69a}.summary strong{display:block;font-size:15px;font-weight:500;color:#d9e6dc;margin-bottom:8px}
.downloads{display:flex;gap:10px;align-items:center}.downloads a{font-size:11px;color:#b7cebe;text-decoration:none;border:1px solid #b7cebe38;padding:10px 14px;border-radius:6px}.downloads a:last-child{background:#c5e3be;color:#203b25;border-color:transparent}
.footnote{margin-top:17px;display:flex;justify-content:space-between;gap:20px}.footnote p{font-size:11px;color:#7f998b}.status{font-size:11px;color:#a8d7b5;text-align:right;min-width:130px}
@media(max-width:760px){header{height:65px}.heading{display:block;margin-bottom:19px}.switch{margin-top:18px;width:fit-content}main{padding-top:24px}.toolbar{padding:15px;gap:10px;flex-wrap:wrap}.label small{display:none}.stage{min-height:430px;height:58vh}.bottom{padding:19px;align-items:start;flex-direction:column}.help{bottom:64px;font-size:9px;right:16px}.angles{bottom:18px;left:15px}.footnote{display:block}.status{text-align:left;margin-top:8px}h1{font-size:25px}}
</style>
<script type="importmap">{"imports":{"three":"/vendor/three.module.js"}}</script>
</head><body>
<header><div class="brand">FaceAppearance<span> · </span><span style="font-size:12px;font-weight:400">贴图实验室</span></div><span class="badge">本地对照预览</span></header>
<main><section class="heading"><div><span class="eyebrow">TEXTURE FUSION COMPARISON</span><h1>同一张脸，两种融合方式</h1><p>切换贴图，保留当前观察角度。查看眉眼、嘴唇与侧脸的细节差异。</p></div>
<div class="switch" role="group" aria-label="融合结果"><button data-branch="baseline" aria-pressed="false">当前融合</button><button data-branch="colmap" aria-pressed="true">COLMAP 实验</button></div></section>
<section class="studio" aria-label="三维资产对照"><div class="toolbar"><span class="label" id="branch-title">COLMAP 实验<small>同一网格 · 同一灯光</small></span>
<div class="modes" role="group" aria-label="显示模式"><button data-mode="textured" aria-pressed="true">纹理</button><button data-mode="clay" aria-pressed="false">灰模</button><button data-mode="wireframe" aria-pressed="false">线框</button></div></div>
<div class="stage"><div id="viewport"></div><div class="stage-top"><span class="dot"></span>THREE.JS · LOCAL PREVIEW</div><div id="loading" role="status" aria-live="polite">正在读取资产…</div>
<div class="angles" role="group" aria-label="观察角度"><button data-view="front">正面</button><button data-view="threequarter">四分之三</button><button data-view="left">左侧</button><button data-view="right">右侧</button></div><span class="help">拖动旋转 · 滚轮缩放 · 右键平移</span></div>
<div class="bottom"><div class="summary"><strong id="asset-summary">等待资产载入</strong><span id="asset-counts">贴图来自经过校验的客户端资产包</span></div><div class="downloads"><a href="/assets/baseline/download">下载当前 ZIP</a><a href="/assets/colmap/download">下载 COLMAP ZIP ↓</a></div></div></section>
<div class="footnote"><p>预览使用相同的工作室灯光；照片未覆盖的区域由模板补全。可切换灰模确认几何一致。</p><span id="status" class="status" role="status" aria-live="polite">正在加载</span></div></main>
<noscript>请启用 JavaScript 以查看三维对照。</noscript>
<script type="module">
import { FaceViewer } from '/scene.js';
const $ = (value) => document.querySelector(value);
const names = { baseline: '当前融合', colmap: 'COLMAP 实验' };
const previewCache = new Map();
let viewer, busy = false;
function fail(error) {
  $('#loading').hidden = false;
  $('#loading').textContent = `无法载入三维预览：${error.message}`;
  $('#status').textContent = '加载失败';
}
async function checkedFetch(url) {
  const response = await fetch(url, { credentials: 'same-origin' });
  if (!response.ok) throw new Error(`资源读取失败 (${response.status})`);
  return response;
}
async function loadBranch(branch) {
  if (busy) return;
  busy = true;
  const old = viewer.hasBounds ? { position: viewer.camera.position.clone(), target: viewer.controls.target.clone() } : null;
  document.querySelectorAll('button').forEach(button => { button.disabled = true; });
  $('#loading').hidden = false;
  $('#loading').textContent = `正在载入${names[branch]}…`;
  $('#status').textContent = '正在加载';
  try {
    if (!previewCache.has(branch)) previewCache.set(branch, await (await checkedFetch(`/assets/${branch}/preview`)).json());
    const preview = previewCache.get(branch);
    await viewer.load(preview, async url => (await checkedFetch(url)).blob());
    if (old) {
      const damping = viewer.controls.enableDamping;
      viewer.controls.enableDamping = false;
      viewer.camera.position.copy(old.position);
      viewer.controls.target.copy(old.target);
      viewer.controls.update();
      viewer.controls.enableDamping = damping;
    }
    document.querySelectorAll('[data-branch]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.branch === branch)));
    $('#branch-title').replaceChildren(document.createTextNode(names[branch]));
    const subtitle = document.createElement('small'); subtitle.textContent = '同一网格 · 同一灯光'; $('#branch-title').append(subtitle);
    $('#asset-summary').textContent = `${names[branch]} · ${preview.templateId}`;
    $('#asset-counts').textContent = `${preview.counts.sourceVertices.toLocaleString()} 个源顶点　${preview.counts.faces.toLocaleString()} 个三角面　${preview.counts.materials} 个材质`;
    $('#loading').hidden = true;
    $('#status').textContent = '资产校验通过 · 已就绪';
  } catch (error) { fail(error); }
  finally { busy = false; document.querySelectorAll('button').forEach(button => { button.disabled = false; }); }
}
try {
  viewer = new FaceViewer($('#viewport'), { onError: fail });
  document.querySelectorAll('[data-branch]').forEach(button => button.addEventListener('click', () => loadBranch(button.dataset.branch)));
  document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
    viewer.setMode(button.dataset.mode);
    document.querySelectorAll('[data-mode]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
  }));
  document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => {
    if (button.dataset.view !== 'threequarter') { viewer.setView(button.dataset.view); return; }
    viewer.setView('front');
    const offset = viewer.fitDistance / Math.sqrt(2);
    viewer.camera.position.set(viewer.center.x + offset, viewer.center.y, viewer.center.z + offset);
    viewer.controls.update();
  }));
  window.addEventListener('pagehide', () => viewer.dispose(), { once: true });
  await loadBranch('colmap');
} catch (error) { fail(error); }
</script></body></html>'''


def createApp(source: Path, experiment: Path) -> FastAPI:
    source, experiment = Path(source).resolve(), Path(experiment).resolve()
    baselineManifest = json.loads((source / "assembly/manifest.json").read_text())
    assets, previews, archives = {}, {}, {}
    for name, root in (("baseline", source), ("colmap", experiment)):
        assembly = root / "assembly"
        identity = json.loads((assembly / "archive.json").read_text())
        asset = loadClientAsset(assembly / "clientAsset.zip",
                               expectedTemplateId=baselineManifest["templateId"],
                               expectedTemplateHashes=baselineManifest["templateHashes"],
                               expectedArchiveSha256=identity["sha256"],
                               expectedArchiveBytes=identity["bytes"])
        payload = asset.archivePath.read_bytes()
        if hashlib.sha256(payload).hexdigest() != asset.archiveSha256:
            raise ValueError("Archive changed while starting comparison viewer")
        preview = buildPreview(asset, name)
        for material in preview["materials"]:
            material["texture"]["url"] = f"/assets/{name}/textures/{material['index']}"
        assets[name], previews[name], archives[name] = asset, preview, payload
    for name, values in assets["baseline"].mesh.items():
        if not np.array_equal(values, assets["colmap"].mesh[name]):
            raise ValueError(f"Comparison requires identical client geometry and UVs: {name}")

    app = FastAPI(title="Local COLMAP texture comparison", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    web = ROOT / "src/faceAppearance/web"
    scripts = {"/scene.js": (web / "scene.js").read_bytes()}
    for filename in ("three.module.js", "three.core.js", "OrbitControls.js"):
        scripts[f"/vendor/{filename}"] = (web / "vendor" / filename).read_bytes()

    @app.middleware("http")
    async def localHeaders(request, callNext):
        response = await callNext(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        return response

    def selected(name):
        if name not in assets:
            raise HTTPException(404, "Unknown comparison asset")
        return assets[name]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(HTML)

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
    parser.add_argument("--source", type=Path, required=True, help="Cached source output containing assembly/")
    parser.add_argument("--experiment", type=Path, required=True, help="COLMAP experiment containing assembly/")
    parser.add_argument("--host", choices=["127.0.0.1"], default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18084)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    uvicorn.run(createApp(args.source, args.experiment), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
