"""Build a dependency-free read-only preview for rendered FAN observations."""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>照片与渲染语义点对照</title>
<style>
:root{color-scheme:dark;--bg:#10171d;--panel:#19232c;--line:#31414e;--ink:#e8eef3;--muted:#a6b9c7;--accent:#79d6cd;--warn:#f0c98b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;font-size:14px;line-height:1.65}main{max-width:1460px;margin:auto;padding:28px 28px 52px}header{display:flex;justify-content:space-between;gap:20px;align-items:flex-start}h1{font-size:26px;line-height:1.3;margin:0 0 10px;font-weight:650}h2{font-size:17px;margin:0 0 12px}p{margin:7px 0;color:var(--muted)}a{color:var(--accent);text-underline-offset:3px}.status{white-space:nowrap;display:inline-block;padding:5px 12px;border:1px solid var(--line);border-radius:20px;color:var(--accent);font-size:12px}.notice{margin:20px 0;padding:13px 16px;border-left:3px solid var(--accent);background:#192930;color:#cee0e8}.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:end;margin:20px 0}label{display:flex;flex-direction:column;gap:6px;color:var(--muted);font-size:12px;min-width:145px}select,button{font:inherit;color:var(--ink);background:var(--panel);border:1px solid #486070;border-radius:7px;padding:9px 12px;min-height:40px}select{font-size:14px;max-width:300px}button{cursor:pointer;font-size:13px}button:hover,select:hover{border-color:var(--accent)}button:focus-visible,select:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:3px}.toolbar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:10px 0 18px}.toolbar p{flex:1;min-width:250px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{border:1px solid var(--line);background:var(--panel);border-radius:10px;overflow:hidden}.panel-head{display:flex;align-items:center;justify-content:space-between;padding:13px 16px;border-bottom:1px solid var(--line);gap:12px}.panel-head h2{font-size:15px;margin:0}.image-box{display:flex;align-items:center;justify-content:center;min-height:300px;background:#111a22}.image-box img{width:100%;height:auto;max-height:720px;object-fit:contain;display:block}.missing{padding:28px;color:var(--muted)}.card-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.card{border:1px solid var(--line);border-radius:9px;background:var(--panel);padding:14px 16px}.card-name{color:var(--muted);font-size:12px}.card-main{font-size:24px;letter-spacing:-.4px;line-height:1.4;margin:4px 0}.card-sub{font-size:12px;color:var(--muted)}.details{margin:22px 0;padding:18px 20px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;color:var(--muted);font-weight:500;white-space:nowrap}td,th{padding:9px 12px;border-bottom:1px solid var(--line)}td:first-child,th:first-child{padding-left:0}tbody tr:last-child td{border-bottom:0}.definitions{display:grid;grid-template-columns:1fr 1fr;gap:14px 24px}.definition{font-size:13px}.definition strong{color:var(--ink);font-weight:550}.legend{font-size:12px;color:var(--muted);margin-top:12px}.good{color:var(--accent)}.bad{color:var(--warn)}.error{padding:20px;border:1px solid #b98459;border-radius:8px;color:var(--warn)}ul{padding-left:22px;color:var(--muted)}li{margin:5px 0}details summary{cursor:pointer;color:var(--ink)}pre{max-height:400px;overflow:auto;background:#111a22;padding:14px;border-radius:6px;font-size:12px;white-space:pre-wrap;overflow-wrap:anywhere}.empty{display:none}#loading{margin:30px 0;color:var(--muted)}.pair-note{padding:10px 0;color:var(--muted);font-size:13px}.footer{font-size:12px;color:var(--muted);margin-top:25px}
@media(max-width:760px){main{padding:20px 14px 35px}header{display:block}header .status{margin-top:12px}h1{font-size:22px}.grid,.definitions{grid-template-columns:1fr}.card-grid{grid-template-columns:1fr 1fr}.controls{gap:10px}label{min-width:calc(50% - 8px);flex:1}select{max-width:100%;width:100%}.image-box{min-height:200px}.details{padding:14px}.toolbar{gap:10px}}
</style>
</head>
<body><main>
<header><div><h1>照片与渲染语义点对照</h1><p>同一 FAN 人脸关键点算法，观察真实照片与对应视角的模型渲染是否获得一致语义。</p></div><span class="status" id="status">读取报告</span></header>
<div class="notice"><strong>本实验保持几何与相机不变。</strong> 检测点更接近照片，不代表几何已经改善；贴图、光照和输入分辨率都可能影响检测结果。</div>
<div id="loading">正在读取 report.json…</div><div id="error" class="error empty" role="alert"></div>
<div id="content" class="empty">
<section class="controls" aria-label="选择对照"><label>样本<select id="case"></select></label><label>照片视角<select id="view"></select></label><label>几何阶段<select id="stage"></select></label><label>渲染外观<select id="variant"></select></label><button id="refresh">刷新报告</button></section>
<div class="toolbar"><p id="selection-note"></p><a id="comparison-link" target="_blank" rel="noopener">打开拼图</a><a href="report.json" target="_blank" rel="noopener">完整报告 JSON</a></div>
<section class="grid" aria-label="关键点叠加对照"><article class="panel"><div class="panel-head"><h2>真实照片 · 68 点检测</h2><a id="photo-link" target="_blank" rel="noopener">原图</a></div><div id="photo-box" class="image-box"></div></article><article class="panel"><div class="panel-head"><h2>模型渲染 · 68 点检测</h2><a id="render-link" target="_blank" rel="noopener">渲染图</a></div><div id="render-box" class="image-box"></div></article></section>
<div class="legend">绿色：通过门槛的检测点；红色：未通过门槛；黄色：语义连线。渲染图中的青色点为照片对应点，橙色短线连接两者。点击图片可查看原尺寸。</div>
<section class="card-grid" id="cards" aria-label="当前对照摘要"></section>
<section class="details"><h2>语义分组</h2><div class="table-wrap"><table><thead><tr><th>区域</th><th>渲染有效点数</th><th>照片与渲染共同有效</th><th>照片与渲染平均距离</th><th>框扰动中位漂移</th><th>距几何边界平均距离</th></tr></thead><tbody id="groups"></tbody></table></div></section>
<section class="details"><h2>如何解读</h2><div class="definitions"><div class="definition"><strong>照片与渲染距离</strong><p>仅统计照片与渲染共同有效的同编号语义点。反映两张图的检测差异，包含几何、相机、外观与检测误差。</p></div><div class="definition"><strong>框扰动漂移</strong><p>改变检测框尺寸后的点位变化。稳定性可筛除部分失败检测，不能证明解剖位置正确。</p></div><div class="definition"><strong>距几何边界距离</strong><p>渲染检测点与可见几何轮廓边的距离。仰头时，语义下颌线可位于下巴与脖子之间的表面内部；距离大不等于检测失败。</p></div><div class="definition"><strong>下巴点 8 的纵向差</strong><p>渲染 y 减照片 y，正值表示检测出的渲染下巴更低。该数值并非直接测得的下巴几何长度。</p></div></div><p class="legend" id="units-note">所有像素距离均使用保存的工作图坐标；放大显示不增加原照片的测量细节。边界距离仅统计组内轮廓点；缺失值显示为“—”。</p></section>
<section class="details"><h2>下巴点与表面命中</h2><div id="chin-detail"></div><p>射线命中皮肤不等于命中下颌。下巴外侧的点也可能命中后面的脖子；命中结果只作为诊断。</p></section>
<section class="details"><h2>实验方法与限制</h2><div id="limitations"></div><details><summary>当前条目原始数据</summary><pre id="row-json"></pre></details></section>
<section class="details" id="pairwise-section"><h2>外观与阶段对照</h2><div class="pair-note">相同几何在不同外观下的检测漂移，有助于识别贴图与光照影响。完整对照数据见报告。</div><details><summary>展开对照数据</summary><pre id="pairwise-json"></pre></details></section>
</div><div class="footer">只读实验预览 · 本页不修改模型、不提交任务、不连接远程服务。</div>
</main>
<script>
'use strict';
const $=id=>document.getElementById(id);
const state={report:null,rows:[],selected:null};
const groupNames={all:'全部语义点',jaw:'下颌轮廓 0–16',central:'中央下巴 7–9',mouth:'嘴部',eyes:'眼部',brows:'眉毛',eyebrows:'眉毛',nose:'鼻部'};
const stageNames={before:'修正前',after:'已有自适应修正后',baseline:'基准几何'};
const variantNames={template:'模板原始贴图',clay:'灰模',gray:'灰模 · 主光照',grayAlternate:'灰模 · 对侧光照',photoDerived:'照片 SH 贴图',baked:'照片烘焙贴图',textured:'贴图渲染',baseColor:'基础颜色',lit:'固定光照'};
const number=(v,d=2)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(d):'—';
const bool=v=>v===true?'是':v===false?'否':'—';
const pretty=v=>JSON.stringify(v,null,2);
const rowsFrom=report=>Array.isArray(report.rows)?report.rows:Array.isArray(report.results)?report.results:[];
function path(value){if(typeof value!=='string'||!value)return null;try{const u=new URL(value,location.href);const base=new URL('.',location.href);return u.origin===base.origin&&u.pathname.startsWith(base.pathname)?u.href:null;}catch{return null;}}
function setLink(id,value){const node=$(id),url=path(value);node.hidden=!url;if(url)node.href=url;else node.removeAttribute('href');}
function setImage(id,value,alt){const box=$(id);box.replaceChildren();const url=path(value);if(!url){const note=document.createElement('div');note.className='missing';note.textContent='该图片尚未生成';box.append(note);return;}const link=document.createElement('a');link.href=url;link.target='_blank';link.rel='noopener';const img=document.createElement('img');img.src=url;img.alt=alt;img.loading='eager';img.onerror=()=>{box.replaceChildren();const note=document.createElement('div');note.className='missing';note.textContent='图片暂不可用，生成完成后可刷新报告';box.append(note);};link.append(img);box.append(link);}
function options(id,values,desired,label){const node=$(id),old=desired??node.value;node.replaceChildren();for(const value of values){const option=document.createElement('option');option.value=String(value);option.textContent=label(value);node.append(option);}node.value=values.map(String).includes(String(old))?String(old):String(values[0]??'');}
const unique=(rows,key)=>[...new Set(rows.map(row=>String(row[key]??'')))];
function sync(level=0){let rows=state.rows;const params=new URLSearchParams(location.search);if(level===0)options('case',unique(rows,'caseId'),$('case').value||params.get('case')||'maleCurrent',v=>v);rows=rows.filter(r=>String(r.caseId)===$('case').value);if(level<=1)options('view',unique(rows,'viewIndex').sort((a,b)=>Number(a)-Number(b)),$('view').value||params.get('view')||'0',v=>`视角 ${Number(v)+1} · 原始索引 ${v}`);rows=rows.filter(r=>String(r.viewIndex)===$('view').value);if(level<=2)options('stage',unique(rows,'stage'),$('stage').value||params.get('stage')||'before',v=>stageNames[v]||v);rows=rows.filter(r=>String(r.stage??'')===$('stage').value);if(level<=3)options('variant',unique(rows,'variant'),$('variant').value||params.get('variant')||'template',v=>variantNames[v]||v);state.selected=rows.find(r=>String(r.variant??'')===$('variant').value)||null;render();}
function cell(row,text){const node=document.createElement('td');node.textContent=text;row.append(node);}
function card(name,value,sub){const div=document.createElement('article');div.className='card';for(const [cls,text] of [['card-name',name],['card-main',value],['card-sub',sub]]){const node=document.createElement('div');node.className=cls;node.textContent=text;div.append(node);}return div;}
function render(){const row=state.selected;if(!row)return;const display=row.display||{},groups=row.groups||{},chin=row.chin||{};$('selection-note').textContent=`${row.caseId} · ${stageNames[row.stage]||row.stage||''} · ${variantNames[row.variant]||row.variant||''} · 推理 ${number(row.seconds)} 秒`;setImage('photo-box',display.photoOverlay||display.photo,'照片语义点叠加');setImage('render-box',display.renderOverlay||display.render,'渲染语义点叠加');setLink('photo-link',display.photo);setLink('render-link',display.render);setLink('comparison-link',display.comparison);$('cards').replaceChildren();for(const key of ['jaw','central','mouth']){const item=groups[key]||{};$('cards').append(card(groupNames[key],`${item.count??'—'} / ${item.total??'—'}`,`渲染有效 · 共同有效 ${item.pairedCount??'—'} · 平均距离 ${number(item.meanPhotoResidualPixels)} px`));}$('cards').append(card('下巴点 8 · 渲染 y − 照片 y',`${number(chin.residualY)} px`,`照片有效：${bool(chin.photoValid)} · 渲染有效：${bool(chin.renderValid)}`));$('groups').replaceChildren();for(const [key,item] of Object.entries(groups)){if(!item||typeof item!=='object')continue;const tr=document.createElement('tr');cell(tr,groupNames[key]||key);cell(tr,`${item.count??'—'} / ${item.total??'—'}`);cell(tr,String(item.pairedCount??'—'));cell(tr,`${number(item.meanPhotoResidualPixels)} px`);cell(tr,`${number(item.medianCropSpreadPixels)} px`);cell(tr,`${number(item.meanBoundaryDistancePixels)} px`);$('groups').append(tr);}$('chin-detail').textContent=`照片位置 ${Array.isArray(chin.photo)?chin.photo.map(v=>number(v)).join(', '):'—'}；渲染位置 ${Array.isArray(chin.render)?chin.render.map(v=>number(v)).join(', '):'—'}；距几何边界 ${number(chin.boundaryDistancePixels)} px；皮肤射线命中：${bool(chin.skinRayHit)}。`;$('row-json').textContent=pretty(row);const params=new URLSearchParams({case:row.caseId,view:String(row.viewIndex),stage:row.stage??'',variant:row.variant??''});history.replaceState(null,'',`${location.pathname}?${params}`);}
function addProse(parent,value){if(Array.isArray(value)){const ul=document.createElement('ul');for(const item of value){const li=document.createElement('li');li.textContent=typeof item==='string'?item:pretty(item);ul.append(li);}parent.append(ul);}else if(value&&typeof value==='object'){const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='方法参数';const pre=document.createElement('pre');pre.textContent=pretty(value);details.append(summary,pre);parent.append(details);}else if(value){const p=document.createElement('p');p.textContent=String(value);parent.append(p);}}
async function load(){const previous=state.rows.length;$('refresh').disabled=true;$('error').classList.add('empty');try{const response=await fetch(`report.json?_=${Date.now()}`,{cache:'no-store'});if(!response.ok)throw Error(`HTTP ${response.status}`);const report=await response.json();state.report=report;state.rows=rowsFrom(report);$('status').textContent=`${report.status==='complete'?'已完成':report.status==='running'?'生成中':report.status||'报告已加载'} · ${state.rows.length} 条对照`;$('loading').textContent=state.rows.length?'':'报告已加载，尚无可预览条目。可稍后点击刷新报告。';$('loading').classList.toggle('empty',state.rows.length>0);$('content').classList.toggle('empty',state.rows.length===0);$('limitations').replaceChildren();addProse($('limitations'),report.method||report.methods);addProse($('limitations'),report.limitations||['自动检测不是人工语义真值。轮廓语义点随视角变化；贴图上的照片特征可能影响检测。']);const pairs=report.variantComparisons??report.pairwiseComparisons??report.pairwise??report.comparisons;$('pairwise-section').classList.toggle('empty',!pairs);$('pairwise-json').textContent=pairs?pretty(pairs):'';if(state.rows.length)sync(0);}catch(error){$('error').textContent=`无法读取实验报告：${error.message}。请通过本地 HTTP 服务打开本页，并确认 report.json 已生成。`;$('error').classList.remove('empty');$('status').textContent='报告暂不可用';if(!previous)$('loading').classList.add('empty');}finally{$('refresh').disabled=false;}}
for(const [id,level] of [['case',1],['view',2],['stage',3],['variant',4]])$(id).addEventListener('change',()=>sync(level));$('refresh').addEventListener('click',load);load();
</script></body></html>
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path,
                        default=ROOT / 'runs/clientRenderedSemantics20261012/evaluation')
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    path = args.directory / 'index.html'
    path.write_text(HTML, encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
