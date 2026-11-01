"""Publish only explicit diagnostic PNGs and their metadata to a local preview."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
CASES = [('currentMale', '当前男生'), ('highPitch4', '大幅抬头'),
         ('cyanWall4', '浅蓝背景低清'), ('outdoorSideLight4', '室外侧光低清')]

PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>UFF → 客户端配准 · 误差诊断</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#11171d;color:#e4ebf1}
body{margin:0;padding:24px}h1{font-size:25px;margin:0 0 10px}p{color:#adbbc8;line-height:1.7;margin:8px 0}
.bar{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:20px 0;background:#1b242d;padding:14px;border-radius:10px}
select,button{font:inherit;padding:8px;background:#263542;color:inherit;border:1px solid #526678;border-radius:6px}
label{display:flex;align-items:center;gap:7px}input[type=range]{width:130px}
.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px}
article{min-width:0;background:#1b242d;border-radius:8px;padding:8px}h2{font-size:14px;margin:7px 3px 10px}
canvas{width:100%;height:auto;background:#1b2128;display:block;border-radius:4px}
.notes{border-left:3px solid #638ec4;padding-left:16px;margin:20px 0;max-width:1200px}
.badge{font-size:12px;color:#91c5e8;letter-spacing:.05em;margin-bottom:12px}
details{margin-top:20px}a{color:#9fd4ff}li{margin:8px 0;color:#bdc9d5;line-height:1.6}
.assembly{max-width:360px}#status{font-size:13px;color:#9eafbe}
@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}body{padding:14px}}
</style>
<div class="badge">本地诊断 · 同一次 UFF / 同一相机 · 未进行新的优化</div>
<h1>UFF → 客户端配准，究竟改变了哪里？</h1>
<p>横向依次比较原照片、原始 UFF、纯配准、稳定版后处理和最新快速眼形后处理。先比较 UFF 与纯配准，再看后续步骤的影响。</p>
<div class="bar">
<label>样本 <select id="case"></select></label>
<label>视角 <select id="view"><option value="0">照片 1 · 正面</option><option value="1">照片 2</option><option value="2">照片 3</option><option value="3">照片 4</option></select></label>
<label>区域 <select id="region"><option value="all">面部</option><option value="eyes">双眼</option><option value="mouth">嘴部</option><option value="nose">鼻部</option><option value="whole">整个裁剪</option></select></label>
<label><input id="overlay" type="checkbox">叠加原照片</label>
<label>灰模不透明度 <input id="opacity" type="range" min="0" max="1" value="0.65" step="0.01"></label>
</div>
<div id="status"></div><div class="grid" id="grid"></div>
<div class="notes"><p><strong>如何解读：</strong>所有皮肤灰模共用已保存的照片相机及同一灯光，没有为分支重新对齐。UFF 与客户端的网格密度和拓扑不同；眼洞边界不等于含眼球、眼角和睫毛后的最终可见眼裂。照片与灰模光照也不同。</p>
<p id="finding"></p><p>图像在原 512×512 工作图上渲染；放大只便于查看。配准与 UFF 的距离衡量信息传递偏差，不能当作真实人脸的三维误差。自动照片眼裂检测也不是人工真值。</p></div>
<details><summary>正面可见皮肤眼孔：面积、厚度与照片重叠</summary><p>青色为灰模渲染中实际可见的皮肤开口，粉色为自动照片眼裂。已考虑皮肤自身遮挡，尚未加入眼球和附件。</p><img id="apertures" style="width:100%;max-width:1600px" alt="可见皮肤眼孔与照片轮廓"></details>
<details><summary>照片眼裂与结构眼孔边界叠加</summary><p>绿色为缓存的照片眼裂，红色为网格自身的眼孔边界投影。UFF 部分边界被自身皮肤遮挡，因此此图解释结构偏差，不代表最终可见眼裂。</p><img id="curves" style="width:100%;max-width:1536px" alt="眼孔结构边界与自动照片轮廓对照"></details>
<details><summary>查看最终客户端完整灰模（仅作装配参照）</summary><p>含眼球、口内、眉睫与眼角附件；与上面的皮肤表面图不作等价眼裂比较。</p><div class="assembly"><canvas id="assembly"></canvas></div></details>
<details><summary>实验记录与复核入口</summary><ul id="links"></ul></details>
<script>
const DATA=__DATA__, $=id=>document.getElementById(id), cache=new Map();
for(const item of DATA.cases){const option=document.createElement('option');option.value=item.id;option.textContent=item.label;$('case').append(option)}
const params=new URLSearchParams(location.search);for(const key of ['case','view','region'])if(params.has(key))$(key).value=params.get(key);
if(!$('case').value)$('case').value=DATA.cases[0].id;
function img(path){if(!cache.has(path)){const image=new Image();cache.set(path,new Promise((resolve,reject)=>{image.onload=()=>resolve(image);image.onerror=reject;image.src=path}))}return cache.get(path)}
function draw(canvas,image,box,photo,overlay){const [x,y,r,b]=box,w=r-x,h=b-y;canvas.width=Math.round(720);canvas.height=Math.round(720*h/w);const c=canvas.getContext('2d');c.fillStyle='#1b2128';c.fillRect(0,0,canvas.width,canvas.height);if(overlay)c.drawImage(photo,x,y,w,h,0,0,canvas.width,canvas.height);c.globalAlpha=overlay?Number($('opacity').value):1;c.drawImage(image,x,y,w,h,0,0,canvas.width,canvas.height);c.globalAlpha=1}
let request=0;async function update(){const token=++request,item=DATA.cases.find(c=>c.id===$('case').value),view=item.views[Number($('view').value)],region=$('region').value,box=view.crops[region];$('status').textContent='读取固定相机渲染…';const photo=await img(item.id+'/'+view.photo);const panels=[{label:'原照片',image:photo},...await Promise.all(item.branches.map(async b=>({label:b.label,image:await img(item.id+'/'+view.branches[b.id])})))];if(token!==request)return;$('grid').replaceChildren();for(const [i,panel]of panels.entries()){const card=document.createElement('article'),title=document.createElement('h2'),canvas=document.createElement('canvas');title.textContent=panel.label;draw(canvas,panel.image,box,photo,i>0&&$('overlay').checked);card.append(title,canvas);$('grid').append(card)}$('status').textContent=`${item.label} · 工作图区域 [${box.join(', ')}] · 保存相机，未重拟合`;$('finding').textContent=item.finding||'';$('curves').src=item.id+'/eyeSkinRims.png';$('links').replaceChildren();for(const link of [{label:'渲染记录',path:item.id+'/renderReport.json'},...(item.reportLinks||[])]){const li=document.createElement('li'),a=document.createElement('a');a.textContent=link.label;a.href=link.path;a.target='_blank';li.append(a);$('links').append(li)}if(view.fullAssembly){const full=await img(item.id+'/'+view.fullAssembly);if(token===request)draw($('assembly'),full,box,photo,false)}history.replaceState(null,'','?'+new URLSearchParams({case:item.id,view:$('view').value,region}));}
for(const id of ['case','view','region','overlay','opacity'])$(id).addEventListener('input',()=>update().catch(e=>$('status').textContent=String(e)));update().catch(e=>$('status').textContent=String(e));
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'runs/registrationTransfer20261013')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/registrationTransfer20261013/preview')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    data, files = {'cases': []}, []
    for case, label in CASES:
        source = args.root / 'renders' / case
        report = json.loads((source / 'renderReport.json').read_text())
        output = args.output / case
        output.mkdir()
        def copy(path, filename):
            target = output / filename
            shutil.copyfile(path, target)
            files.append({'source': str(path.resolve()), 'copy': str(target.relative_to(args.output)),
                          'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        for path in source.glob('*.png'):
            copy(path, path.name)
        copy(source / 'renderReport.json', 'renderReport.json')
        for view in report['views']:
            path = ROOT / f'runs/fastFrontEyes20261012/final/{case}/evidence/view{view["id"]}-combined-clay.png'
            if path.is_file():
                filename = f'view{view["id"]}-fullAssembly.png'
                copy(path, filename)
                view['fullAssembly'] = filename
        reportLinks = []
        labels = {'metrics': '三维传递指标', 'semantics': '照片语义诊断', 'apertures': '可见皮肤眼孔诊断'}
        for name, directory in [('transferMetrics', 'metrics'), ('semanticMetrics', 'semantics'), ('apertureMetrics', 'apertures')]:
            for basename in ('report.json', 'metrics.json'):
                path = args.root / directory / case / basename
                if path.is_file():
                    filename = name + '.json'
                    copy(path, filename)
                    reportLinks.append({'label': labels[directory],
                                        'path': case + '/' + filename})
                    break
        copy(args.root / 'semantics' / case / 'eyeSkinRims.png', 'eyeSkinRims.png')
        data['cases'].append({'id': case, 'label': label, 'branches': report['branches'],
                              'views': report['views'], 'reportLinks': reportLinks})
    for item in data['cases']:
        case = item['id']
        path = args.root / 'apertures' / case / 'report.json'
        if path.is_file():
            aperture = json.loads(path.read_text())
            percentages = [100 * aperture['stageRatios'][eye]['uffToDirect']['areaWorkingPixelsSquared']
                           for eye in ('imageLeft', 'imageRight')]
            item['finding'] = ('正面实测：纯配准后的可见皮肤眼孔面积，图左 / 图右分别为原 UFF 的 '
                + ' / '.join(f'{value:.0f}%' for value in percentages) + '。这是皮肤开口的比较，不是最终眼球装配后的可见眼裂。')
            target = args.output / case / 'apertures.png'
            source = args.root / 'apertures' / case / 'overlay.png'
            shutil.copyfile(source, target)
            files.append({'source': str(source.resolve()), 'copy': str(target.relative_to(args.output)),
                          'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
    encoded = json.dumps(data, ensure_ascii=False, indent=2)
    (args.output / 'data.json').write_text(encoded + '\n')
    page = PAGE.replace("$('curves').src=item.id+'/eyeSkinRims.png';", "$('curves').src=item.id+'/eyeSkinRims.png';$('apertures').src=item.id+'/apertures.png';")
    (args.output / 'index.html').write_text(page.replace('__DATA__', encoded.replace('<', '\\u003c')))
    (args.output / 'buildManifest.json').write_text(json.dumps(files, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'copiedFiles': len(files)}))


if __name__ == '__main__':
    main()
