import * as THREE from 'three';
import { FaceViewer } from '/viewer-common/scene.js';
import { createPhotoEvidence } from './evidence.js';

const $ = (id) => document.getElementById(id), sides = ['left', 'right'];
const viewers = {}, payloads = new Map(), blobs = new Map();
const allowedBranches = new Set();
let syncing = false, busy = false, viewerReady = false, mode = 'textured', frame;
const selectedBranches = {left:'baseline',right:'combined'};
const status = (text, error=false) => { $('status').textContent=text; $('status').classList.toggle('error',error); };
async function json(path) {
  const response = await fetch(path,{cache:'no-store'});
  if (!response.ok) throw new Error(`资产尚未就绪（HTTP ${response.status}），请稍后刷新。`);
  return response.json();
}
const branchPromise = json('/api/branches').then(branches=>{
  if(!Array.isArray(branches)||branches.length<2)throw new Error('对照分支清单无效。');
  for(const branch of branches){
    if(!/^[A-Za-z][A-Za-z0-9_-]{0,63}$/.test(branch.id)||allowedBranches.has(branch.id)||typeof branch.label!=='string')throw new Error('对照分支信息无效。');
    allowedBranches.add(branch.id);
  }
  return branches;
});
const configPromise = (async()=>{
  const response=await fetch('/api/config',{cache:'no-store'});
  // Older local preview processes can continue serving their original comparison.
  if(response.status===404)return {};
  if(!response.ok)throw new Error(`对照设置读取失败（HTTP ${response.status}）。`);
  return response.json();
})();
const settingsPromise=Promise.all([branchPromise,configPromise]).then(([branches,config])=>{
  selectedBranches.left=allowedBranches.has(config.defaultLeft)?config.defaultLeft:branches[0].id;
  selectedBranches.right=allowedBranches.has(config.defaultRight)?config.defaultRight:(allowedBranches.has('combined')?'combined':branches[1].id);
  mode=['textured','baseColor','clay'].includes(config.defaultViewerMode)?config.defaultViewerMode:'textured';
  for(const [key,id] of Object.entries({title:'page-title',subtitle:'page-subtitle',brand:'comparison-brand',badge:'comparison-badge',photoDescription:'photo-description',overviewDescription:'overview-description'})){
    if(typeof config[key]==='string')$(id).textContent=key==='brand'?`／ ${config[key]}`:config[key];
  }
  if(typeof config.title==='string')document.title=config.title;
  if(Array.isArray(config.notes)){
    const notes=config.notes.map(note=>{
      const paragraph=document.createElement('p');
      if(typeof note==='string')paragraph.textContent=note;
      else{const title=document.createElement('strong');title.textContent=`${note.title}：`;paragraph.append(title,document.createTextNode(note.text));}
      return paragraph;
    });
    $('comparison-notes').replaceChildren(...notes);$('comparison-notes').hidden=!notes.length;
  }
  document.querySelectorAll('[data-mode]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.mode===mode)));
  return {...config,defaultLeft:selectedBranches.left,defaultRight:selectedBranches.right};
});
const photoEvidence = createPhotoEvidence({branches:branchPromise,config:settingsPromise,onBranchChange:chooseBranch});
function setBusy(value) {
  busy=value;sides.forEach(side=>$(side+'-branch').disabled=value);photoEvidence.setBusy(value);
}
async function chooseBranch(side,id) {
  if(busy||!allowedBranches.has(id))return;
  if(!viewerReady){selectedBranches[side]=id;photoEvidence.setBranch(side,id);return;}
  const previous=selectedBranches[side];setBusy(true);status('正在切换分支…');
  try{await load(side,id);$(side+'-branch').value=id;status('同一重建结果 · 旋转与缩放同步');}
  catch(error){$(side+'-branch').value=previous;photoEvidence.setBranch(side,previous);status(error.message,true);}
  finally{setBusy(false);}
}
function pose(viewer) { return {position:viewer.camera.position.clone(),target:viewer.controls.target.clone(),up:viewer.camera.up.clone(),center:viewer.center.clone(),radius:viewer.radius,distance:viewer.fitDistance}; }
function restore(viewer,p) {
  viewer.camera.position.copy(p.position);viewer.camera.up.copy(p.up);viewer.controls.target.copy(p.target);
  viewer.center.copy(p.center);viewer.radius=p.radius;viewer.fitDistance=p.distance;viewer.camera.lookAt(p.target);viewer.controls.update();
}
function sync(source,target) {
  if(syncing||busy)return; syncing=true;
  try{restore(target,pose(source));}finally{syncing=false;}
}
function applyMode(viewer) {
  viewer.setMode(mode==='baseColor'?'textured':mode);
  if(mode!=='baseColor')return;
  const asset=viewer.asset;
  if(!asset.modes.baseColor){
    asset.modes.baseColor=asset.modes.textured.map(m=>new THREE.MeshBasicMaterial({map:m.map,color:m.color.clone(),alphaTest:m.alphaTest,side:m.side,opacity:m.opacity,transparent:false,toneMapped:false}));
    asset.materials.push(...asset.modes.baseColor);
  }
  asset.mesh.material=asset.modes.baseColor;
}
async function textureLoader(preview,url) {
  const descriptor=preview.materials.find(m=>m.texture.url===url)?.texture;
  const match=/^\/api\/jobs\/([A-Za-z][A-Za-z0-9_-]{0,63})\/textures\/[0-3]$/.exec(url);
  if(!descriptor||!match||!allowedBranches.has(match[1]))throw new Error('贴图不在当前资产清单内。');
  const key=descriptor.sha256;
  if(!blobs.has(key))blobs.set(key,(async()=>{
    const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw new Error('贴图读取失败。');
    const bytes=await r.arrayBuffer();if(bytes.byteLength!==descriptor.sizeBytes)throw new Error('贴图大小不匹配。');
    const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),x=>x.toString(16).padStart(2,'0')).join('');
    if(hash!==descriptor.sha256)throw new Error('贴图哈希不匹配。');return new Blob([bytes],{type:'image/png'});
  })());
  return blobs.get(key);
}
async function load(side,id,preserve=true) {
  const viewer=viewers[side],saved=preserve&&viewer.asset?pose(viewer):null;
  if(!payloads.has(id))payloads.set(id,await json(`/api/jobs/${id}/preview`));
  const preview=payloads.get(id);
  await viewer.load(preview,url=>textureLoader(preview,url));applyMode(viewer);
  if(saved)restore(viewer,saved);
  $(''+side+'-download').href=`/api/jobs/${id}/asset.zip`;
  $(''+side+'-caption').textContent=`${preview.counts.sourceVertices} 顶点 · ${preview.counts.faces} 三角面 · 4 个材质`;
  selectedBranches[side]=id;photoEvidence.setBranch(side,id);
}
function focus(part) {
  syncing=true;
  try{
    const v=viewers.left;
    v.center.copy(part==='eyes'?frame.eyes:frame.center);v.radius=frame.radius*(part==='eyes'?.31:1);
    v.fitDistance=v._fitDistance();v.setView('front');restore(viewers.right,pose(v));
  }finally{syncing=false;}
}
try {
  const [branches]=await Promise.all([branchPromise,settingsPromise]);
  for(const side of sides){viewers[side]=new FaceViewer($(side+'-viewport'),{onError:e=>status(e.message,true)});viewers[side].controls.enableDamping=false;}
  viewers.left.controls.addEventListener('change',()=>sync(viewers.left,viewers.right));
  viewers.right.controls.addEventListener('change',()=>sync(viewers.right,viewers.left));
  for(const side of sides){
    const select=$(side+'-branch');
    branches.forEach(b=>select.add(new Option(b.label,b.id)));
    select.value=selectedBranches[side];
  }
  setBusy(true);await load('left',selectedBranches.left,false);await load('right',selectedBranches.right,false);
  const mesh=payloads.get(selectedBranches.left).mesh,eye=new THREE.Box3();
  for(const group of mesh.groups.filter(g=>g.materialIndex===2))for(let i=group.start;i<group.start+group.count;i++){const n=mesh.indices[i]*3;eye.expandByPoint(new THREE.Vector3(...mesh.positions.slice(n,n+3)));}
  frame={center:viewers.left.center.clone(),radius:viewers.left.radius,eyes:eye.isEmpty()?viewers.left.center.clone():eye.getCenter(new THREE.Vector3())};
  viewerReady=true;setBusy(false);focus('head');
  for(const side of sides){const select=$(side+'-branch');select.addEventListener('change',()=>chooseBranch(side,select.value));}
  document.querySelectorAll('[data-mode]').forEach(button=>button.addEventListener('click',()=>{if(busy)return;mode=button.dataset.mode;Object.values(viewers).forEach(applyMode);document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));}));
  document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{if(busy)return;viewers.left.setView(button.dataset.view);sync(viewers.left,viewers.right);}));
  document.querySelectorAll('[data-focus]').forEach(button=>button.addEventListener('click',()=>{if(!busy)focus(button.dataset.focus);}));
  status('同一重建结果 · 旋转与缩放同步');
}catch(error){status(error.message,true);setBusy(false);}
window.addEventListener('pagehide',()=>Object.values(viewers).forEach(v=>v.dispose()));
