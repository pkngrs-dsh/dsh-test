"""Fixed-camera four-arm geometry inspection with identical template materials."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.clientLandmarks import loadClientLandmarks,ibugClientLandmarks
from renderClientAppearanceEvidence import EvidenceRenderer,evidenceGrid,solid
from renderFeedbackAppearanceComparison import saveSrgb
from renderNoseRepairComparison import scaledViewCamera
from renderShapePreservationComparison import baselinePatchCrop
from runProtectedFeatureExperiment import RUN,OLD,PRIOR,plan,read,sha,arrays,verify
OUT=ROOT/'reports/protectedFeature20261024'
LABELS={'baseline':'共同基线','sourceProtected':'源形状＋支撑保护','localEyes':'局部六点眼拟合','combined':'两者组合'}

def renderCase(caseId):
    from pytorch3d.structures import Meshes
    code=[Path(__file__),ROOT/'tools/renderClientAppearanceEvidence.py',ROOT/'tools/renderClientAssembly.py',
        ROOT/'tools/renderNoseRepairComparison.py',ROOT/'tools/renderShapePreservationComparison.py']
    codeHashes={str(p):sha(p) for p in code}
    context=read(RUN/caseId/'inputContext.json');verify(context['sourceHashes']);entry=context['entry']
    branches=list(plan()['branches']);reports={b:read(RUN/caseId/b/'report.json') for b in branches}
    raw=arrays(entry['preparedGeometry']);transform={k:raw[k].tolist() for k in ('rotation','translation','scale')}
    cameras=read(Path(entry['reconstruction'])/'cameras.json');inputs=read(Path(entry['reconstruction'])/'inputs.json')
    renderers={b:EvidenceRenderer(Path(reports[b]['asset']['archivePath']),transform,device='cpu',facesPerPixel=12) for b in branches}
    renderers['baseline'].torch.set_num_threads(1);base=renderers['baseline'].source
    for b,renderer in renderers.items():
        source=renderer.source
        for key in ('faces','uvs','uvFaces','materialIds'):np.testing.assert_array_equal(source.mesh[key],base.mesh[key])
        if source.manifest['binding']!=base.manifest['binding']:raise ValueError('Binding changed')
        for name,payload in base.payloads.items():
            if name!='face_nicp.obj' and source.payloads[name]!=payload:raise ValueError('Materials not identical')
        expected=arrays(RUN/caseId/b/'fullGeometry.npz')['vertices']
        canonical=source.mesh['vertices']*100@np.asarray(transform['rotation'])*transform['scale']+transform['translation']
        np.testing.assert_allclose(canonical,expected,rtol=0,atol=1e-12)
        # Rasterize the same numerical state used by the safety checker.
        renderer.vertices=renderer.torch.as_tensor(expected,dtype=renderer.torch.float32)
        renderer.mesh=Meshes(verts=[renderer.vertices],faces=[renderer.faces])
    output=OUT/caseId;output.mkdir(parents=True,exist_ok=False)
    binding=ibugClientLandmarks(loadClientLandmarks(raw['faces']),groups=('eyes',))
    eyeIds=np.unique(binding['vertexIndices'][binding['validMask']]);views=[]
    for view,metadata in enumerate(inputs):
        camera=scaledViewCamera(cameras[view],metadata,view,512)
        photo=Image.open(Path(entry['workingPhotos'])/f'input{view:02d}.png').convert('RGB').resize((512,512),Image.Resampling.LANCZOS)
        crop,cropReport=baselinePatchCrop(raw['pureDirect'],eyeIds,camera,512)
        rendered={};stats={}
        for b,renderer in renderers.items():
            rendered[b],stats[b]=renderer.variants(camera,(512,512))
        for mode in ('clay','baseColor','lit'):
            images=[photo]+[solid(rendered[b][mode]) for b in branches]
            labels=['原工作照片']+[LABELS[b]+(' · 数值通过' if reports[b]['geometrySafetyAccepted'] else ' · 未通过/诊断') for b in branches]
            title=f'{caseId} · 视角 {view+1} · {mode} · 共同模板贴图，仅比较几何；固定保存相机'
            picture=evidenceGrid(images,labels,columns=5,panelSize=512,subtitle=title)
            saveSrgb(picture,output/f'view{view:02d}_{mode}.png')
            eyes=evidenceGrid([im.crop(crop) for im in images],labels,columns=5,panelSize=256,subtitle=title+' · 同一区域放大')
            saveSrgb(eyes,output/f'view{view:02d}_{mode}_eyes.png')
            for b in branches:saveSrgb(solid(rendered[b][mode]),output/f'view{view:02d}_{b}_{mode}.png')
        views.append({'viewIndex':view,'camera':camera,'stats':stats,'eyeCrop':crop,'eyeCropReport':cropReport})
        print(f'{caseId}: rendered view {view}',flush=True)
    verify(context['sourceHashes']);verify(codeHashes)
    writeJson(output/'report.json',{'caseId':caseId,'views':views,'identicalMaterials':True,'allFourCameras':True,
        'templateColorsOnly':True,'textureRebaked':False,'noGeometryPromotion':True,'inputHashes':context['sourceHashes'],
        'renderImplementationHashes':codeHashes,
        'images':{p.name:sha(p) for p in sorted(output.glob('*.png'))}})

def writeIndex():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=plan()['cases'];complete=[c for c in cases if (OUT/c/'report.json').exists()]
    writeJson(OUT/'viewer.json',{'cases':cases,'complete':complete,'labels':LABELS})
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>几何四分支对照</title>
<style>body{background:#171c23;color:#e8eef6;font:16px sans-serif;margin:20px}select{font:inherit;padding:6px;background:#283240;color:white;border:1px solid #61738a;margin:4px}img{display:block;max-width:100%;margin:16px 0}a{color:#7bc8ff}p{line-height:1.7}</style>
<h2>支撑区域保护与局部眼部形变 · 统一七组测试</h2><p>四列模型使用相同模板贴图和固定的基线附件。这是皮肤形变隔离实验，尚不代表最终自适应装配可交付。此页只评估几何，眉毛、鼻翼贴图尚未重新烘焙。未通过安全门的分支仅供诊断；照片对照使用原保存相机。</p>
<label>样本<select id="case"></select></label><label>视角<select id="view"><option value="0">1</option><option value="1">2</option><option value="2">3</option><option value="3">4</option></select></label><label>显示<select id="mode"><option value="clay">灰模</option><option value="lit">共同光照</option><option value="baseColor">共同模板基色</option></select></label><p id="state"></p><img id="main"><img id="eyes">
<script>const qs=new URLSearchParams(location.search), $=id=>document.getElementById(id);fetch('viewer.json').then(r=>r.json()).then(data=>{for(const c of data.cases){const o=new Option(c+(data.complete.includes(c)?'':'（等待渲染）'),c);$('case').add(o)}for(const k of ['case','view','mode'])if(qs.has(k))$(k).value=qs.get(k);function update(){const c=$('case').value,v=$('view').value,m=$('mode').value;history.replaceState(null,'','?'+new URLSearchParams({case:c,view:v,mode:m}));$('main').src=c+'/view'+v.padStart(2,'0')+'_'+m+'.png';$('eyes').src=c+'/view'+v.padStart(2,'0')+'_'+m+'_eyes.png';$('state').textContent=data.complete.includes(c)?'从左到右：原照片 / 共同基线 / 源形状＋支撑保护 / 局部六点眼拟合 / 两者组合':'此样本尚未完成渲染'}for(const k of ['case','view','mode'])$(k).onchange=update;update()});</script>''',encoding='utf-8')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case');args=parser.parse_args()
    if args.case:renderCase(args.case)
    writeIndex()
