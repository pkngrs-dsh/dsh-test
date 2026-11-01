const $ = (id) => document.getElementById(id);
const modes = {lit:'共同光照',baseColor:'贴图基色',clay:'灰模'};
const sides = ['left','right'];

function imageUrl(filename) {
  if(typeof filename!=='string'||!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.png$/.test(filename))throw new Error('渲染图地址无效。');
  return `/evidence/${filename}`;
}
function decodeImage(url) {
  const image=new Image();image.src=url;return image.decode().then(()=>url);
}

export function createPhotoEvidence({branches,config,onBranchChange}) {
  const state={manifest:null,view:null,mode:'lit',layout:'separate',opacity:.5,branches:{left:null,right:null},labels:{},generation:0,loading:false,busy:false,ready:false,configured:false};
  const status=(text,error=false)=>{ $('evidence-status').textContent=text;$('evidence-status').classList.toggle('error',error); };
  function controls() {
    for(const side of sides)$(`photo-${side}-branch`).disabled=state.busy||!state.ready;
  }
  function layout() {
    const overlay=state.layout==='overlay';
    $('photo-grid').classList.toggle('with-overlay',overlay);
    $('photo-grid').style.setProperty('--model-opacity',String(overlay?state.opacity:1));
    $('opacity-control').hidden=!overlay;
    for(const side of sides)$(`photo-${side}-background`).hidden=!overlay||!state.view;
    document.querySelectorAll('[data-photo-layout]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.photoLayout===state.layout)));
  }
  async function render() {
    if(!state.manifest||!state.view)return;
    const generation=++state.generation,view=state.view;
    const photo=imageUrl(view.photo);
    const renderUrls=sides.map(side=>imageUrl(view.renders[state.branches[side]][state.mode]));
    $('photo-grid').setAttribute('aria-busy','true');status('正在读取这一视角的渲染图…');
    try{
      await Promise.all([photo,...renderUrls].map(decodeImage));
      if(generation!==state.generation)return;
      $('source-photo').src=photo;$('source-photo').hidden=false;$('photo-full').href=photo;
      $('photo-label').textContent=`${view.label||view.id} · 原照片`;
      $('original-photo-link').href=imageUrl(view.original||view.photo);
      for(const [index,side] of sides.entries()){
        const branch=state.branches[side],label=state.labels[branch]||branch;
        const image=$(`photo-${side}-render`);image.src=renderUrls[index];image.hidden=false;
        image.alt=`${view.label||view.id} · ${label} · ${modes[state.mode]}`;
        $(`photo-${side}-background`).src=photo;
        $(`photo-${side}-full`).href=renderUrls[index];
        $(`photo-${side}-caption`).textContent=`${label} · ${modes[state.mode]}`;
      }
      const comparison=$('view-comparison-link');comparison.hidden=!view.comparison;
      if(view.comparison)comparison.href=imageUrl(view.comparison);
      $('view-description').textContent=`${view.description||'沿用本次重建估计的相机与原照片裁剪。'} 共同光照便于看形状，贴图基色便于看颜色；灰模叠加便于看轮廓。`;
      document.querySelectorAll('[data-evidence-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.evidenceView===view.id)));
      document.querySelectorAll('[data-render-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.renderMode===state.mode)));
      layout();status(`${view.label||view.id} · 原照与渲染使用同一视角，亮度差异不等同于形状误差。`);
    }catch(error){if(generation===state.generation)status(`渲染图读取失败：${error.message}`,true);}
    finally{if(generation===state.generation)$('photo-grid').setAttribute('aria-busy','false');}
  }
  async function initialize() {
    if(state.loading)return;state.loading=true;$('reload-evidence').disabled=true;
    try{
      const [options,settings,response]=await Promise.all([branches,config,fetch('/api/evidence',{cache:'no-store'})]);
      if(!state.configured){
        state.branches.left=state.branches.left||settings.defaultLeft;state.branches.right=state.branches.right||settings.defaultRight;
        if(Object.hasOwn(modes,settings.defaultRenderMode))state.mode=settings.defaultRenderMode;
        state.configured=true;
      }
      if(!response.ok){const body=await response.json().catch(()=>({}));throw new Error(typeof body.detail==='string'?body.detail:`HTTP ${response.status}`);}
      const manifest=await response.json();
      if(manifest.schemaVersion!==1||!Array.isArray(manifest.views)||!manifest.views.length)throw new Error('照片对照清单无效。');
      for(const view of manifest.views){
        imageUrl(view.photo);
        for(const option of options)for(const mode of Object.keys(modes))imageUrl(view.renders?.[option.id]?.[mode]);
      }
      state.manifest=manifest;state.labels=Object.fromEntries(options.map(b=>[b.id,b.label]));
      const selectedId=state.view?.id;state.view=manifest.views.find(v=>v.id===selectedId)||manifest.views[0];
      for(const side of sides){const select=$(`photo-${side}-branch`);select.replaceChildren();options.forEach(b=>select.add(new Option(b.label,b.id)));select.value=state.branches[side];}
      $('photo-views').replaceChildren();
      for(const view of manifest.views){
        const button=document.createElement('button'),thumbnail=document.createElement('img'),label=document.createElement('span');
        button.type='button';button.dataset.evidenceView=view.id;button.setAttribute('aria-pressed',String(view.id===state.view.id));
        thumbnail.src=imageUrl(view.photo);thumbnail.alt='';thumbnail.loading='lazy';label.textContent=view.label||view.id;
        button.append(thumbnail,label);button.addEventListener('click',()=>{state.view=view;render();});$('photo-views').append(button);
      }
      $('render-overview').hidden=!manifest.overview;
      if(manifest.overview){const url=imageUrl(manifest.overview);$('overview-image').src=url;$('overview-full').href=url;$('overview-image-link').href=url;}
      state.ready=true;controls();await render();
    }catch(error){status(error.message,true);}
    finally{state.loading=false;$('reload-evidence').disabled=false;}
  }
  for(const side of sides)$(`photo-${side}-branch`).addEventListener('change',()=>onBranchChange(side,$(`photo-${side}-branch`).value));
  document.querySelectorAll('[data-render-mode]').forEach(button=>button.addEventListener('click',()=>{state.mode=button.dataset.renderMode;render();}));
  document.querySelectorAll('[data-photo-layout]').forEach(button=>button.addEventListener('click',()=>{state.layout=button.dataset.photoLayout;layout();}));
  $('photo-opacity').addEventListener('input',()=>{state.opacity=Number($('photo-opacity').value)/100;$('opacity-value').textContent=`${Math.round(state.opacity*100)}%`;layout();});
  $('reload-evidence').addEventListener('click',initialize);initialize();
  return {
    setBranch(side,id){state.branches[side]=id;$(`photo-${side}-branch`).value=id;render();},
    setBusy(value){state.busy=value;controls();},
  };
}
