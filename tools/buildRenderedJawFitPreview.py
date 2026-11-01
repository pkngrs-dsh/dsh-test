"""Build a dependency-free static preview of the frozen rendered-jaw batch."""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>渲染语义约束 · 下巴实验</title>
<style>
:root{color-scheme:dark;--bg:#10161c;--panel:#18222b;--line:#2b3a46;--text:#e9f0f4;--muted:#a9bbc8;--good:#79dab1;--warn:#e9bb77;--accent:#80c9f5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"Noto Sans CJK SC",sans-serif;font-size:14px;line-height:1.6}main{max-width:1720px;margin:auto;padding:26px 24px 48px}h1{font-size:25px;line-height:1.25;margin:0 0 9px;font-weight:650}h2{font-size:17px;margin:0 0 10px}.eyebrow{font-size:11px;letter-spacing:2px;color:var(--accent);margin-bottom:7px}.intro{margin:0;color:var(--muted);max-width:1050px}.toolbar{display:flex;flex-wrap:wrap;align-items:end;gap:13px;margin:23px 0 16px;padding:16px;background:var(--panel);border:1px solid var(--line);border-radius:10px}label{display:flex;flex-direction:column;gap:5px;color:var(--muted);font-size:12px}select,button{font:inherit;border:1px solid #425562;border-radius:6px;color:var(--text);background:#101a22;padding:9px 12px;min-height:40px}select{min-width:150px}button{cursor:pointer}button:hover{border-color:var(--accent)}label.toggle{flex-direction:row;align-items:center;gap:8px;padding:8px 0;font-size:14px;color:var(--text)}input[type=checkbox]{accent-color:var(--accent);width:17px;height:17px}.toolbar .spacer{flex:1}.status{display:flex;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:12px}.badge{display:inline-block;padding:4px 10px;border-radius:20px;font-size:13px;font-weight:650}.pass{background:#18382d;color:var(--good)}.fallback{background:#3a2e1c;color:var(--warn)}.neutral{background:#25333e;color:#bed0dc}.statusnote{color:var(--muted)}.metricrow{display:flex;flex-wrap:wrap;gap:16px;margin:0 0 16px}.metric{padding:10px 14px;border:1px solid var(--line);border-radius:7px;background:#142029;min-width:200px}.metric strong{display:block;font-size:19px;font-variant-numeric:tabular-nums}.metric small{color:var(--muted)}a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.frame{margin:0;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#18212a}.frame figcaption{padding:11px 13px;min-height:65px;border-bottom:1px solid var(--line)}.frame figcaption strong{display:block;font-size:15px}.frame figcaption span{font-size:12px;color:var(--muted)}.frame a{display:block}.frame img{display:block;width:100%;height:auto;min-height:170px;background:#19222b}.notes{margin:12px 0 24px;color:var(--muted);font-size:12px}.legend{display:inline-flex;align-items:center;gap:5px;margin-right:14px}.dot{width:8px;height:8px;border-radius:50%;display:inline-block}.lower{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(320px,1fr);gap:18px}.panel{border:1px solid var(--line);border-radius:8px;padding:17px;background:var(--panel)}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:12px}th{text-align:left;font-weight:500;color:var(--muted)}th,td{padding:9px 8px;border-bottom:1px solid var(--line);white-space:nowrap}tbody tr:last-child td{border-bottom:0}tr.current{background:#223441}td.nums{font-variant-numeric:tabular-nums}td button{background:transparent;border:0;padding:0;min-height:unset;color:var(--accent);text-align:left}details{margin-top:14px}summary{cursor:pointer;color:var(--accent)}ul{padding-left:20px;color:var(--muted)}.plain{color:var(--muted);margin:5px 0 0}.error{background:#402b2b;color:#ffb6ad;padding:16px;border-radius:8px}.hidden{display:none}.small{font-size:12px}#viewScope{font-weight:500}#reasons{max-height:300px;overflow:auto}#loading{padding:30px;color:var(--muted)}.rule{border:0;border-top:1px solid var(--line);margin:18px 0}footer{margin-top:18px;color:var(--muted);font-size:12px}
@media(max-width:900px){main{padding:18px 12px}.lower{grid-template-columns:1fr}.metric{min-width:unset;flex:1}.toolbar .spacer{display:none}.images{gap:6px}.frame figcaption{padding:8px;font-size:12px}select{min-width:120px}}
@media(max-width:560px){.images{grid-template-columns:1fr}.frame img{max-height:650px;object-fit:contain}.frame figcaption{min-height:unset}.metricrow{gap:8px}.metric strong{font-size:16px}h1{font-size:22px}}
</style>
</head>
<body><main>
<div class="eyebrow">LOCAL EXPERIMENT / RENDERED SEMANTICS</div>
<h1 id="pageTitle">用渲染人脸的语义，约束下巴几何</h1>
<p id="pageIntro" class="intro">照片与模型渲染使用同一 FAN 检测器。候选形状重新渲染后，再检查模板外观和两种灰模光照的语义误差。“修改前”统一使用本轮开始时、未做下巴修正的已配准模型。</p>
<p id="modePolicy" class="intro hidden"></p>
<div class="toolbar">
<label>样本<select id="caseSelect" aria-label="样本"></select></label>
<label>拍摄视角<select id="viewSelect" aria-label="拍摄视角"></select></label>
<label>渲染外观<select id="appearanceSelect" aria-label="渲染外观"><option value="template">模板贴图</option><option value="gray">灰模 · 光照 A</option><option value="grayAlternate">灰模 · 光照 B</option></select></label>
<label>查看结果<select id="candidateSelect" aria-label="查看结果"><option value="final">最终输出</option></select></label>
<label class="toggle"><input type="checkbox" id="landmarkToggle">显示关键点</label>
<div class="spacer"></div><a id="reportLink" target="_blank" rel="noopener">当前样本报告 ↗</a><button id="reload">刷新结果</button>
</div>
<div id="loading">正在读取本地实验结果…</div><div id="error" class="error hidden"></div>
<section id="content" class="hidden">
<div class="status"><span id="resultBadge" class="badge"></span><span id="viewBadge" class="badge neutral"></span><span id="viewScope" class="statusnote"></span></div>
<p id="selectedReason" class="notes"></p>
<div class="metricrow">
<div class="metric"><small id="metricLabel">当前视角的语义平均误差</small><strong id="metricValue">—</strong><small id="metricDetail"></small></div>
<div id="unweightedMetric" class="metric hidden"><small>原有效点等权误差</small><strong id="unweightedValue">—</strong><small id="unweightedDetail">保留全部原有效点，用等权指标并列观察</small></div>
<div class="metric"><small id="changedLabel">最终输出</small><strong id="changedValue">—</strong><small id="changedDetail"></small></div>
<div class="metric"><small id="participationLabel">参与验收的照片视角</small><strong id="participationValue">—</strong><small id="participationDetail">其余视角仅提供图像对照</small></div>
</div>
<div class="images">
<figure class="frame"><figcaption><strong>原始照片</strong><span>同一照片目标，固定裁剪与坐标</span></figcaption><a id="photoLink" target="_blank" rel="noopener"><img id="photoImage" alt="当前视角的原始照片"></a></figure>
<figure class="frame"><figcaption><strong>修改前</strong><span>本轮输入网格 · 原始相机</span></figcaption><a id="beforeLink" target="_blank" rel="noopener"><img id="beforeImage" alt="下巴语义修正前的模型渲染"></a></figure>
<figure class="frame"><figcaption><strong id="afterTitle">最终输出</strong><span id="afterSubtitle"></span></figcaption><a id="afterLink" target="_blank" rel="noopener"><img id="afterImage" alt="最终保留模型的渲染"></a></figure>
</div>
<div class="notes"><span class="legend"><i class="dot" style="background:#4df38a"></i>检测有效</span><span class="legend"><i class="dot" style="background:#ff5b5b"></i>检测未通过门槛</span><span class="legend"><i class="dot" style="background:#3cdbf4"></i>照片目标位置</span>关键点叠图显示完整人脸；误差只统计本轮预先固定、通过筛选的下颌语义点。点击图片可查看原尺寸。</div>
<div class="lower">
<section class="panel"><h2>六组样本结果</h2><p id="batchState" class="plain small"></p><div class="scroll"><table><thead><tr><th>样本</th><th>结果</th><th id="referenceMetricHeading">参考视角误差 / px</th><th>变化顶点</th></tr></thead><tbody id="caseTable"></tbody></table></div></section>
<section class="panel"><h2 id="viewTableHeading">当前样本的验收范围</h2><div class="scroll"><table><thead><tr><th>视角</th><th>下颌点</th><th id="viewMetricHeading">当前外观误差 / px</th></tr></thead><tbody id="viewTable"></tbody></table></div><p class="plain small">单位为保存的工作图像像素。自动关键点是观测，不是人工标注真值；低分辨率图像放大不会增加证据。</p><details><summary>候选验收记录与回退原因</summary><div id="reasons"></div></details></section>
</div>
<footer id="footerNote">本地实验：仅调整下巴附近皮肤几何。相机、眼嘴支撑点、模板接口及原装配附件保持固定；未重烘焙贴图、未部署服务器。所有视角都可查看，但定量结论只覆盖参与验收的固定语义点集合。</footer>
</section></main>
<script>
'use strict';
const $=id=>document.getElementById(id);
const names={maleCurrent:'男生基准',orbitControl4:'环绕拍摄对照',highPitch4:'高俯仰样本',cyanWall4:'青墙样本',outdoorSideLight4:'户外侧光样本',backlitCap4:'逆光戴帽样本'};
const appearances={template:'模板贴图',gray:'灰模 · 光照 A',grayAlternate:'灰模 · 光照 B'};
const query=new URLSearchParams(location.search);
let chosenCase=query.get('case')||'maleCurrent',chosenView=Number(query.get('view')||0),chosenAppearance=query.get('appearance')||query.get('variant')||'template',chosenCandidate=query.get('candidate')||'final';
if(!(chosenAppearance in appearances))chosenAppearance='template';
let summary=null,current=null,baseline=null,loadVersion=0;
const reports=new Map(),bindings=new Map();
const text=(id,value)=>{$(id).textContent=value;};
const number=value=>Number.isFinite(value)?value.toFixed(2):'—';
const accepted=report=>report?.status==='accepted_live_semantics'&&report?.selectedCandidateId!=='noop';
const reliabilityWeighted=report=>report?.reliabilityWeighted??summary?.reliabilityWeighted??false;
const frontPrimary=report=>report?.frontPrimary??summary?.frontPrimary??false;
function viewRole(view){
  if(!frontPrimary(current))return view.participated?'fit':'diagnostic';
  if(view.role)return view.role;
  if(current.fitViewIndices?.includes(view.viewIndex))return 'fit';
  if(current.diagnosticViewIndices?.includes(view.viewIndex))return 'diagnostic';
  return view.participated?'fit':'unavailable';
}
const participates=view=>viewRole(view)==='fit';
const viewRoleLabel=view=>frontPrimary(current)?(participates(view)?'正面语义拟合':viewRole(view)==='diagnostic'?'仅观察':'未用于拟合'):(view.participated?'参与验收':'视觉对照');
const caseName=id=>names[id]||id;
const viewName=index=>'视角 '+(index+1);
const option=(value,label)=>{const e=document.createElement('option');e.value=value;e.textContent=label;return e;};
const cell=(value,cls)=>{const e=document.createElement('td');e.textContent=value;if(cls)e.className=cls;return e;};
function reasonText(reason){
  let m=reason.match(/^view(\d+): required anatomical\/appearance gates lost IDs (.+)$/);
  if(m)return viewName(Number(m[1]))+'：原固定关键点 '+m[2]+' 未通过语义 / 外观有效性筛选';
  m=reason.match(/^view(\d+)\/([^:]+): invalid or regressed fixed semantic set$/);
  if(m)return viewName(Number(m[1]))+' / '+(appearances[m[2]]||m[2])+'：有效点缺失，或误差增加超限';
  m=reason.match(/^reference\/([^:]+): insufficient live improvement$/);
  if(m)return '参考视角 / '+(appearances[m[1]]||m[1])+'：重新检测后的改善不足';
  m=reason.match(/^view(\d+): frozen point detection or anatomical checks failed$/);
  if(m)return viewName(Number(m[1]))+'：原固定点的检测有效性或解剖域检查未通过';
  m=reason.match(/^view(\d+): frozen-weight (appearanceDriftFaceFraction|lightingDriftFaceFraction) limit exceeded$/);
  if(m)return viewName(Number(m[1]))+'：'+(m[2]==='appearanceDriftFaceFraction'?'外观':'光照')+'稳定性超过冻结权重 RMS、增幅或单点上限';
  return reason;
}
async function read(path){const response=await fetch(path,{cache:'no-store'});if(!response.ok)throw new Error(path+'：HTTP '+response.status);return response.json();}
async function reportFor(row){if(!reports.has(row.caseId))reports.set(row.caseId,await read(row.reportPath));return reports.get(row.caseId);}
function baselineMetric(viewIndex,variant){
  const v=baseline?.views?.find(v=>v.viewIndex===viewIndex);
  if(!v||!v.semanticIds?.length)return null;
  // The serialized correspondence source is the template appearance only.
  // Do not present it as a gray-render measurement.
  if(variant!=='template')return null;
  const values=v.targetPixels.map((p,i)=>Math.hypot(p[0]-v.sourcePixels[i][0],p[1]-v.sourcePixels[i][1]));
  const unweighted=values.reduce((a,b)=>a+b,0)/values.length;
  const weights=v.semanticWeights??v.frozenReliabilityWeights,validWeights=weights?.length===values.length&&weights.every(w=>Number.isFinite(w)&&w>=0)&&weights.some(w=>w>0);
  const before=reliabilityWeighted(current)?(validWeights?values.reduce((sum,value,i)=>sum+value*weights[i],0)/weights.reduce((a,b)=>a+b,0):null):unweighted;
  return {beforeMeanPixels:before,afterMeanPixels:before,unweightedBeforeMeanPixels:unweighted,unweightedAfterMeanPixels:unweighted,semanticIds:v.semanticIds,semanticWeights:weights,baselineOnly:true};
}
function currentMetric(index,variant){
  if(chosenCandidate!=='final'){
    const check=current?.candidateChecks?.find(c=>c.candidateId===chosenCandidate),v=check?.views?.find(v=>v.viewIndex===index),m=v?.appearances?.find(a=>a.variant===variant);
    return m?{...m,semanticIds:v.semanticIds,semanticWeights:v.semanticWeights}:null;
  }
  const selected=current?.selectedLiveCheck?.views?.find(v=>v.viewIndex===index);
  if(selected){const m=selected.appearances.find(a=>a.variant===variant);return m?{...m,semanticIds:selected.semanticIds,semanticWeights:selected.semanticWeights}:null;}
  // Rejected live candidates still contain the same fixed baseline metrics.
  // The final fallback is exactly the original mesh, so after equals before.
  const checked=current?.candidateChecks?.find(c=>c.views?.some(v=>v.viewIndex===index));
  if(checked){const v=checked.views.find(v=>v.viewIndex===index),m=v.appearances.find(a=>a.variant===variant);if(m)return {...m,afterMeanPixels:m.beforeMeanPixels,unweightedAfterMeanPixels:m.unweightedBeforeMeanPixels,semanticIds:v.semanticIds,semanticWeights:v.semanticWeights,baselineOnly:true};}
  return baselineMetric(index,variant);
}
function setImage(prefix,path){$(prefix+'Image').src=path;$(prefix+'Link').href=path;}
function renderCaseTable(){
  $('caseTable').replaceChildren();
  for(const row of summary.cases){
    const report=reports.get(row.caseId),pass=accepted(report||row),tr=document.createElement('tr');
    if(row.caseId===chosenCase)tr.className='current';
    const button=document.createElement('button');button.textContent=caseName(row.caseId);button.addEventListener('click',()=>{chosenCase=row.caseId;chosenView=row.referenceViewIndex??0;chosenCandidate='final';loadCase();});
    const first=cell('');first.append(button);tr.append(first,cell(pass?'通过本轮门槛':'保持原样'));
    const metric=report?.selectedLiveCheck?.referenceMetric||row.referenceMetric;
    tr.append(cell(metric?number(metric.beforeMeanPixels)+' → '+number(metric.afterMeanPixels):'—','nums'),cell(String(row.changedVertexCount??report?.changedVertexCount??0),'nums'));
    $('caseTable').append(tr);
  }
}
function renderViewTable(){
  $('viewTable').replaceChildren();
  for(const view of current.views){
    const tr=document.createElement('tr');if(view.viewIndex===chosenView)tr.className='current';
    const button=document.createElement('button');button.textContent=viewName(view.viewIndex)+(frontPrimary(current)?' · '+viewRoleLabel(view):(view.reference?' · 参考':''));button.addEventListener('click',()=>{chosenView=view.viewIndex;render();});
    const first=cell('');first.append(button);tr.append(first);
    const metric=currentMetric(view.viewIndex,chosenAppearance),bound=baseline?.views?.find(v=>v.viewIndex===view.viewIndex);
    tr.append(cell(participates(view)?(metric?.semanticIds||bound?.semanticIds||[]).join('、'):'未参与'));
    tr.append(cell(participates(view)&&metric?number(metric.beforeMeanPixels)+' → '+number(metric.afterMeanPixels):'仅图像对照','nums'));
    $('viewTable').append(tr);
  }
}
function renderReasons(){
  $('reasons').replaceChildren();
  const p=document.createElement('p');p.className='plain small';p.textContent=accepted(current)?(frontPrimary(current)?'所选候选通过正面的固定语义集合、模板贴图与两种灰模光照验收，以及几何安全门槛；其他角度不参与拟合或关键点验收。':'所选候选通过固定语义集合、三种外观及几何安全门槛。'):'最终输出与本轮输入网格完全一致。候选未达到所有门槛时，保留原始几何。';$('reasons').append(p);
  if(!current.candidateChecks?.length){const p=document.createElement('p');p.className='plain small';p.textContent='未进入重新渲染验收：'+(current.solver?.reason||current.solver?.status||'没有安全候选');$('reasons').append(p);}
  for(const check of current.candidateChecks||[]){
    const h=document.createElement('p');h.textContent=check.candidateId+' · '+(check.accepted?'通过本轮门槛':'未通过')+(check.candidateId===current.selectedCandidateId?' · 最终采用':'');$('reasons').append(h);
    const metric=check.referenceMetric;
    if(metric){const p=document.createElement('p');p.className='plain small';p.textContent=(reliabilityWeighted(current)?'加权正面语义误差':frontPrimary(current)?'正面拟合视角':'参考视角')+' / 模板外观：'+number(metric.beforeMeanPixels)+' → '+number(metric.afterMeanPixels)+' px；所需改善 ≥ '+number(metric.requiredImprovementPixels)+' px。'+(reliabilityWeighted(current)?' 原有效点等权误差：'+number(metric.unweightedBeforeMeanPixels)+' → '+number(metric.unweightedAfterMeanPixels)+' px。':'');$('reasons').append(p);}
    if(check.reasons?.length){const list=document.createElement('ul');for(const reason of check.reasons){const li=document.createElement('li');li.textContent=reasonText(reason);list.append(li);} $('reasons').append(list);}
  }
}
function render(){
  if(!current)return;
  let view=current.views.find(v=>v.viewIndex===chosenView)||current.views[0];chosenView=view.viewIndex;
  const front=frontPrimary(current),weighted=reliabilityWeighted(current),fit=participates(view);
  text('pageTitle',weighted?'正面可靠点加权 · 下巴几何后处理':front?'正面语义拟合 · 下巴几何后处理':'用渲染人脸的语义，约束下巴几何');
  text('pageIntro',weighted?'照片与模型渲染使用同一 FAN 检测器。只使用选定正面视角的有效语义点，以修改几何前冻结的可靠性权重驱动拟合和验收，保留全部原有效点。候选重新渲染后，检查正面的模板贴图与两种灰模光照。“修改前”统一使用本轮开始时、未做下巴修正的已配准模型。':front?'照片与模型渲染使用同一 FAN 检测器。只使用选定正面视角的有效语义点驱动拟合和关键点验收；候选重新渲染后，检查正面的模板贴图与两种灰模光照。有效点筛选门槛保持不变。“修改前”统一使用本轮开始时、未做下巴修正的已配准模型。':'照片与模型渲染使用同一 FAN 检测器。候选形状重新渲染后，再检查模板外观和两种灰模光照的语义误差。“修改前”统一使用本轮开始时、未做下巴修正的已配准模型。');
  $('modePolicy').classList.toggle('hidden',!front);
  text('modePolicy',(weighted?'检测有效性、解剖域与几何安全仍需通过。外观与光照稳定性改用冻结权重的 RMS ≤ 脸部尺度的 2%，RMS 增幅 ≤ 0.5%，每点偏移 ≤ 3%；模板外观的原有效点等权误差也不得增加。可靠性权重是工程估计，不是概率。':'几何安全仍是硬门槛。')+'其他视角仅观察，不参与拟合或关键点验收；正面的指标变化不能作为多视角改善的定量结论。');
  text('participationLabel',front?'正面语义拟合视角':'参与验收的照片视角');
  text('participationDetail',front?'其他视角仅观察，不参与拟合或关键点验收':'其余视角仅提供图像对照');
  text('referenceMetricHeading',weighted?'加权正面语义误差 / px':front?'正面拟合误差 / px':'参考视角误差 / px');
  text('viewMetricHeading',weighted?'当前外观加权误差 / px':'当前外观误差 / px');
  text('viewTableHeading',front?'正面拟合与其他视角观察':'当前样本的验收范围');
  text('batchState',summary.status==='complete'?'本轮批次已完成。误差列为'+(front?'选定的正面拟合视角':'自动选择的参考视角')+'，使用模板外观'+(weighted?'和修改几何前冻结的权重':'')+'。':'实验仍在运行；当前已完成 '+summary.cases.length+' 组，可刷新读取更多结果。');
  text('footerNote','本地实验：仅调整下巴附近皮肤几何。相机、眼嘴支撑点、模板接口及原装配附件保持固定；未重烘焙贴图、未部署服务器。'+(front?'几何安全保持硬门槛；其他视角仅观察，不参与拟合或关键点验收，本页不提供多视角改善的定量结论。':'所有视角都可查看，但定量结论只覆盖参与验收的固定语义点集合。'));
  const pass=accepted(current),landmarks=$('landmarkToggle').checked,preview=current.candidateChecks?.find(c=>c.candidateId===chosenCandidate);
  if(chosenCandidate!=='final'&&!preview)chosenCandidate='final';
  $('caseSelect').value=chosenCase;$('viewSelect').value=String(chosenView);$('appearanceSelect').value=chosenAppearance;$('candidateSelect').value=chosenCandidate;
  const badge=$('resultBadge');badge.className='badge '+((preview?preview.accepted:pass)?'pass':'fallback');badge.textContent=preview?(preview.candidateId===current.selectedCandidateId?'已采用候选 · '+preview.candidateId:(preview.accepted?'通过本轮门槛 · 未选为最终输出':'未采用候选 · 未通过全部门槛')):(pass?'通过本轮门槛 · 采用修正':'保持原样 · 回退输入');
  text('selectedReason',preview?(preview.accepted?'当前展示该实验候选；最终结果以“最终输出”为准。':(preview.reasons||[]).map(reasonText).join('；')):(pass?'最终输出通过本轮预先固定的验收门槛。':(current.candidateChecks?.length?'存在实验候选，但未通过全部验收。可在“查看结果”中比较未采用的候选。':'未生成同时满足语义与几何条件的候选，最终保留原始网格。')));
  text('viewBadge',viewName(chosenView)+(front?' · '+viewRoleLabel(view):(view.reference?' · 自动选择的参考视角':'')));
  text('viewScope',front?(fit?'正面语义拟合：该视角参与固定语义点拟合与验收':'仅观察，不参与拟合或关键点验收'):(fit?'该视角参与固定语义点验收':'该视角仅供视觉对照，未用于定量验收'));
  const metric=fit?currentMetric(chosenView,chosenAppearance):null;
  text('metricLabel',appearances[chosenAppearance]+(weighted?' · 加权正面语义误差':front?' · 正面语义拟合误差':' · 当前视角语义平均误差'));
  text('metricValue',metric?number(metric.beforeMeanPixels)+' → '+number(metric.afterMeanPixels)+' px':'未作定量验收');
  text('metricDetail',metric?(metric.semanticIds.length+' 个固定下颌点'+(preview||pass?' · 重新渲染后检测':' · 输出保持原样')):(fit?'此外观暂无可用基线指标':front?'可切换正面拟合视角查看指标':'可切换其他参与视角查看指标'));
  $('unweightedMetric').classList.toggle('hidden',!weighted);
  text('unweightedValue',metric?number(metric.unweightedBeforeMeanPixels)+' → '+number(metric.unweightedAfterMeanPixels)+' px':'未作定量验收');
  const reliability=current.frontReliability,weightNotes=['全部原有效点保留；权重在修改几何前冻结'];
  if(Number.isFinite(reliability?.effectivePointCount))weightNotes.push('权重等效点数 '+number(reliability.effectivePointCount));
  if(Number.isFinite(reliability?.rawWeightRetention))weightNotes.push('可靠性权重保留率 '+number(reliability.rawWeightRetention*100)+'%');
  text('unweightedDetail',fit?weightNotes.join(' · '):'其他角度仅观察，不参与拟合或关键点验收');
  text('changedLabel',preview?'当前查看的候选网格':'最终输出');
  text('changedValue',preview?(preview.proposal?.changedVertexCount??'—')+' 个顶点改变':(pass?current.changedVertexCount+' 个顶点改变':'0 个顶点改变'));
  text('changedDetail',preview?('最终输出：'+(pass?current.selectedCandidateId:'保持原样')):(pass?'通过全部门槛的 '+current.selectedCandidateId:'与本轮输入网格完全相同'));
  text('participationValue',(front?(current.fitViewIndices??current.views.filter(participates).map(v=>v.viewIndex)):(current.comparableViewIndices||[])).map(i=>i+1).join('、')||'无');
  text('afterTitle',preview?(preview.candidateId===current.selectedCandidateId?'已采用候选 · ':'未采用候选 · ')+preview.candidateId:(pass?'语义修正后':'最终输出 · 保持原样'));
  text('afterSubtitle',preview?('实验候选预览 · 最终'+(pass?'采用 '+current.selectedCandidateId:'保留输入网格')):(pass?'同一相机与外观 · 重新渲染验收':'未采用候选形状 · 输入网格原样保留'));
  const dir=chosenCase+'/',assets=view.appearances[chosenAppearance];
  setImage('photo',dir+(landmarks?view.photoLandmarks:view.photo));
  setImage('before',dir+assets[landmarks?'before-landmarks':'before']);
  setImage('after',preview?dir+'candidates/'+preview.candidateId+'/view'+chosenView+'-'+chosenAppearance+(landmarks?'-landmarks':'')+'.png':dir+assets[landmarks?'after-landmarks':'after']);
  $('afterImage').alt=preview&&preview.candidateId!==current.selectedCandidateId?'未采用候选的模型渲染':'最终保留模型的渲染';
  $('reportLink').href=dir+'report.json';
  renderCaseTable();renderViewTable();renderReasons();
  const params=new URLSearchParams({case:chosenCase,view:String(chosenView),appearance:chosenAppearance});
  if(chosenCandidate!=='final')params.set('candidate',chosenCandidate);
  if(landmarks)params.set('landmarks','1');history.replaceState(null,'','?'+params);
}
async function loadCase(){
  const version=++loadVersion;try{
    const row=summary.cases.find(c=>c.caseId===chosenCase)||summary.cases[0];chosenCase=row.caseId;
    const report=await reportFor(row);
    if(!bindings.has(chosenCase))bindings.set(chosenCase,await read(chosenCase+'/baselineCorrespondence.json'));
    if(version!==loadVersion)return;
    current=report;baseline=bindings.get(chosenCase);
    $('viewSelect').replaceChildren(...current.views.map(v=>option(v.viewIndex,viewName(v.viewIndex)+(frontPrimary(current)?' · '+viewRoleLabel(v):(v.reference?' · 参考':'')+(!v.participated?' · 视觉对照':'')))));
    $('candidateSelect').replaceChildren(option('final','最终输出'),...(current.candidateChecks||[]).map(c=>option(c.candidateId,c.candidateId+' · '+(c.candidateId===current.selectedCandidateId?'最终采用':c.accepted?'通过本轮门槛 / 未选':'拒绝 / 未采用'))));
    $('loading').classList.add('hidden');$('content').classList.remove('hidden');$('error').classList.add('hidden');render();
  }catch(error){if(version===loadVersion){text('error','结果尚不可用：'+error.message+'。实验运行期间可稍后点击“刷新结果”。');$('error').classList.remove('hidden');$('loading').classList.add('hidden');}}
}
async function load(){
  try{summary=await read('summary.json');if(!summary.cases?.length)throw new Error('首个样本仍在运行');
    $('caseSelect').replaceChildren(...summary.cases.map(c=>option(c.caseId,caseName(c.caseId))));
    text('batchState',summary.status==='complete'?'本轮批次已完成。误差列为自动选择的参考视角，使用模板外观。':'实验仍在运行；当前已完成 '+summary.cases.length+' 组，可刷新读取更多结果。');
    await loadCase();
  }catch(error){text('error','无法读取实验结果：'+error.message+'。首组完成后点击“刷新结果”。');$('error').classList.remove('hidden');$('loading').classList.add('hidden');}
}
$('caseSelect').addEventListener('change',e=>{chosenCase=e.target.value;chosenView=0;chosenCandidate='final';loadCase();});
$('viewSelect').addEventListener('change',e=>{chosenView=Number(e.target.value);render();});
$('appearanceSelect').addEventListener('change',e=>{chosenAppearance=e.target.value;render();});
$('candidateSelect').addEventListener('change',e=>{chosenCandidate=e.target.value;render();});
$('landmarkToggle').checked=query.get('landmarks')==='1';$('landmarkToggle').addEventListener('change',render);
$('reload').addEventListener('click',()=>{reports.clear();bindings.clear();load();});
load();
</script></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', '--directory', dest='output', type=Path, default=ROOT/'runs/clientRenderedJawFit20261012/evaluationV1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output/'index.html'
    destination.write_text(HTML, encoding='utf-8')
    print(destination.resolve())


if __name__ == '__main__':
    main()
