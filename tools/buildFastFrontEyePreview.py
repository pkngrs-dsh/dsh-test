#!/usr/bin/env python3
"""Build a local four-case, three-branch fast frontal eye evidence gallery.

Only explicit PNG evidence is copied; missing evidence stays missing. This does
not reconstruct faces, modify assets, run rendering, or start a web server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

CASES = (("currentMale", "当前男生"), ("highPitch4", "大幅抬头组"),
         ("cyanWall4", "浅蓝背景低清组"), ("outdoorSideLight4", "室外侧光低清组"))
BRANCHES = (("baseline", "原稳定几何 · 重烘焙"), ("geometry", "快速眼形 · 重烘焙"), ("combined", "快速眼形 + 正面取色"))

PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>快速正面眼部 · 对照实验</title>
<style>
:root{color-scheme:dark;font:16px/1.6 system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;background:#10141b;color:#e8edf4}*{box-sizing:border-box}body{margin:0}main{max-width:1760px;margin:auto;padding:30px 24px 56px}h1{font-size:29px;margin:0 0 10px}h2{font-size:20px;margin:0 0 6px}p{margin:5px 0 12px}.muted{color:#aab8ca}.badge{display:inline-block;vertical-align:middle;font-size:12px;border:1px solid #65758c;border-radius:20px;padding:3px 10px;margin-left:10px}.toolbar{display:flex;flex-wrap:wrap;gap:14px;align-items:end;background:#1a2230;padding:18px;border:1px solid #35445b;border-radius:12px;margin:23px 0 18px}label{display:grid;gap:4px;font-size:13px;color:#bbc8d9}select,button{font:inherit;border:1px solid #5a6d87;border-radius:6px;padding:9px 12px;background:#141c29;color:#e8edf4;cursor:pointer}button:disabled,select:disabled{opacity:.45;cursor:default}button[aria-pressed=true]{background:#255d91;border-color:#8fcaff}.buttons{display:flex;gap:7px;flex-wrap:wrap}:focus-visible{outline:3px solid #9dceff;outline-offset:3px}.info{padding:16px 20px;border-left:3px solid #80a8d2;background:#18212e;margin-bottom:20px;border-radius:4px}.info p{white-space:pre-line}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}figure{margin:0;border:1px solid #35445b;border-radius:9px;overflow:hidden;background:#1b2532}figcaption{padding:10px 13px;font-size:14px;border-bottom:1px solid #35445b}img{display:block;width:100%;height:auto;background:#1b2128}.missing{min-height:220px;display:grid;place-content:center;padding:20px;color:#aab8ca;text-align:center}.hidden{display:none!important}.wide{max-width:1520px;margin:auto}.foot{font-size:13px;margin-top:18px}details{margin-top:18px}summary{cursor:pointer;color:#a4ceff}a{color:#a4ceff}.diagnostic{margin-bottom:14px;max-width:1200px}.diagnostic img{object-fit:contain;max-height:780px}.diagnostic p{padding:10px 14px}.status{font-size:13px;color:#bdcbe0}@media(max-width:950px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}main{padding:23px 14px}}@media(max-width:560px){.grid{grid-template-columns:1fr}h1{font-size:24px}.badge{display:block;width:max-content;margin:7px 0}.toolbar{gap:9px}.toolbar label{flex:1}select{width:100%}}
</style></head><body><main>
<h1>快速正面眼部 · 对照实验 <span class="badge">本地实验 · 未部署</span></h1>
<p class="muted">以原稳定版本为统一基底，共用相机、UV 和贴图烘焙策略。依次比较原几何、快速眼形修正、再加正面眼周取色。眼形只调整相机平面的局部顶点，保留 UFF 深度。此页面的对照渲染是离线证据，不计入生成时间。</p>
<section class="toolbar" aria-label="对照选项"><label>样本<select id="case"></select></label><label>照片视角<select id="view"><option value="0">正面</option><option value="1">视角 2</option><option value="2">视角 3</option><option value="3">视角 4</option></select></label><label>渲染方式<select id="mode"><option value="baseColor">无光照贴图</option><option value="lit">统一光照</option><option value="clay">灰模</option></select></label><div class="buttons"><button data-panel="views" aria-pressed="true">四视角对照</button><button data-panel="eyes" aria-pressed="false">正面眼区放大</button><button data-panel="diagnostics" aria-pressed="false">局部取色权重</button></div></section>
<section class="info" aria-live="polite"><h2 id="title"></h2><p id="status" class="status"></p><p id="note"></p></section>
<section id="grid" class="grid" aria-label="原照片与三组优化结果"></section>
<section id="diagnostics" class="hidden"></section>
<p id="evidenceNote" class="muted foot"></p>
<details id="contactSheet"><summary>展开左右眼近景总览</summary><figure class="wide"><figcaption>图左眼与图右眼 · 原照片 / 原稳定几何 / 快速眼形 / 正面取色</figcaption><img id="sheet" alt="两眼局部对照总览"></figure></details>
<p class="muted foot">统一光照未拟合照片的光照。眼区采用 4 倍像素栅格渲染，原照片仅放大显示，不增加细节。多个采集目录不代表已核验的多个身份；缺失输出不会用其他结果替代。</p>
</main><script type="application/json" id="data">__DATA__</script><script>
'use strict';const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);let state={case:0,view:0,mode:'baseColor',panel:'views'};
function figure(label,path){const f=document.createElement('figure'),cap=document.createElement('figcaption');cap.textContent=label;f.append(cap);if(path){const img=document.createElement('img');img.alt=label;img.src=path;f.append(img);}else{const miss=document.createElement('div');miss.className='missing';miss.textContent='尚无此项证据';f.append(miss);}return f;}
function show(){const c=data.cases[state.case];if(!c)return;$('title').textContent=c.label;$('status').textContent=c.status||'';$('note').textContent=c.note||'';const f=c.files;
const hasEyes=!!f['eyes-photo.png'],hasDiagnostics=c.diagnostics.length>0;for(const b of document.querySelectorAll('[data-panel]')){b.disabled=(b.dataset.panel==='eyes'&&!hasEyes)||(b.dataset.panel==='diagnostics'&&!hasDiagnostics);}
if((state.panel==='eyes'&&!hasEyes)||(state.panel==='diagnostics'&&!hasDiagnostics))state.panel='views';
for(const b of document.querySelectorAll('[data-panel]'))b.setAttribute('aria-pressed',String(b.dataset.panel===state.panel));$('view').disabled=state.panel!=='views';$('mode').disabled=state.panel==='diagnostics';
$('grid').replaceChildren();const eye=state.panel==='eyes',prefix=eye?'eyes':`view${state.view}`;
$('grid').append(figure(eye?'原照片 · 正面眼区':'原照片 · 同一视角',f[`${prefix}-photo.png`]));for(const b of data.branches)$('grid').append(figure(b.label,f[`${prefix}-${b.id}-${state.mode}.png`]));
$('grid').classList.toggle('hidden',state.panel==='diagnostics');$('diagnostics').classList.toggle('hidden',state.panel!=='diagnostics');$('diagnostics').replaceChildren();for(const d of c.diagnostics){const fig=figure(d.label,d.path);fig.className='diagnostic';if(d.note){const p=document.createElement('p');p.textContent=d.note;fig.append(p);}$('diagnostics').append(fig);}
$('contactSheet').classList.toggle('hidden',!f['eyes-comparison.png']||state.panel==='diagnostics');if(f['eyes-comparison.png'])$('sheet').src=f['eyes-comparison.png'];else $('sheet').removeAttribute('src');
$('evidenceNote').textContent=state.panel==='eyes'?'同一正面相机和取景范围。切换灰模检查眼形；快速眼形与正面取色两组几何相同。':state.panel==='diagnostics'?'权重仅解释局部取色；重采样和肤色外扩不属于新增真实观测。':'每个视角沿用该次重建保存的相机和原图裁剪；所有分支共用相机，没有为对照重新对齐。';
const url=new URL(location.href);for(const [k,v] of Object.entries({case:c.caseId,view:state.view,mode:state.mode,panel:state.panel}))url.searchParams.set(k,String(v));history.replaceState(null,'',url);}
for(let i=0;i<data.cases.length;i++){const o=document.createElement('option');o.value=String(i);o.textContent=data.cases[i].label;$('case').append(o);}const q=new URLSearchParams(location.search),c=data.cases.findIndex(x=>x.caseId===q.get('case'));if(c>=0)state.case=c;if(['0','1','2','3'].includes(q.get('view')))state.view=Number(q.get('view'));if(['baseColor','lit','clay'].includes(q.get('mode')))state.mode=q.get('mode');if(['views','eyes','diagnostics'].includes(q.get('panel')))state.panel=q.get('panel');for(const k of ['case','view','mode']){$(k).value=String(state[k]);$(k).addEventListener('change',e=>{state[k]=k==='mode'?e.target.value:Number(e.target.value);show();});}for(const b of document.querySelectorAll('[data-panel]'))b.addEventListener('click',()=>{state.panel=b.dataset.panel;show();});show();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Preview output must be empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"title": "快速正面眼部 · 对照实验", "branches": [{"id": b, "label": label} for b, label in BRANCHES], "cases": []}
    identities, reports = [], []
    for case_id, default_label in CASES:
        case_dir = args.root / case_id
        if not case_dir.is_dir():
            continue
        report_path = case_dir / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
        if report_path.is_file():
            reports.append({"path": str(report_path.resolve()), "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()})
        item = {"caseId": case_id, "label": str(report.get("label", default_label)),
                "status": str(report.get("previewStatus", report.get("status", "输出生成中"))),
                "note": str(report.get("previewNote", report.get("note", "同相机的几何与贴图独立对照。"))),
                "files": {}, "diagnostics": []}
        directory = args.output / case_id
        directory.mkdir()

        def copy_png(original, filename):
            if not original.is_file():
                return None
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.png", filename):
                raise ValueError(f"Unsafe destination filename: {filename}")
            with original.open("rb") as handle:
                if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                    raise ValueError(f"Evidence is not a PNG: {original}")
            target = directory / filename
            if target.exists():
                raise ValueError(f"Duplicate destination: {target}")
            shutil.copyfile(original, target)
            relative = f"{case_id}/{filename}"
            identities.append({"source": str(original.resolve()), "copy": relative,
                               "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
            item["files"][filename] = relative
            return relative

        for view in range(4):
            filename = f"view{view}-photo.png"
            copy_png(case_dir / "evidence" / filename, filename)
            for branch, _ in BRANCHES:
                for mode in ("baseColor", "lit", "clay"):
                    filename = f"view{view}-{branch}-{mode}.png"
                    copy_png(case_dir / "evidence" / filename, filename)
        for filename in ("photo.png", "comparison.png", *(f"{branch}-{mode}.png" for branch, _ in BRANCHES for mode in ("baseColor", "lit", "clay"))):
            copy_png(case_dir / "eyes" / filename, f"eyes-{filename}")
        # Diagnostics are explicitly named in report.json, never recursively
        # copied from a run that may also contain request payloads or secrets.
        for i, diagnostic in enumerate(report.get("previewDiagnostics", [])):
            original = Path(diagnostic["path"])
            if not original.is_absolute():
                original = case_dir / original
            original = original.resolve()
            if not original.is_relative_to(case_dir.resolve()):
                raise ValueError(f"Diagnostic must be inside its case directory: {original}")
            path = copy_png(original, f"diagnostic-{i}.png")
            if path:
                item["diagnostics"].append({"label": str(diagnostic["label"]), "path": path,
                                            "note": str(diagnostic.get("note", ""))})
        result["cases"].append(item)
    if not result["cases"]:
        raise ValueError("No expected case directories found")
    data = json.dumps(result, ensure_ascii=False, indent=2)
    (args.output / "data.json").write_text(data + "\n", encoding="utf-8")
    (args.output / "index.html").write_text(PAGE.replace("__DATA__", data.replace("<", "\\u003c")), encoding="utf-8")
    (args.output / "buildManifest.json").write_text(json.dumps({
        "sourceRoot": str(args.root.resolve()), "sourceReports": reports,
        "sourceFilesUnmodified": True, "copiedPngCount": len(identities), "files": identities,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "caseCount": len(result["cases"]),
                      "copiedPngCount": len(identities)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
