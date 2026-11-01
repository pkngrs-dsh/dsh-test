"""Build an explicit local PNG-only eyelid texture correction comparison."""
from pathlib import Path
import argparse, hashlib, json, shutil
from buildFastFrontEyePreview import PAGE, CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    data = {'branches': [{'id':'baseline','label':'上一版'}, {'id':'candidate','label':'安全正面直投补充'}], 'cases':[]}
    manifest = []
    for case, label in CASES:
        directory = args.root/case
        report = json.loads((directory/'report.json').read_text())
        item = {'caseId':case, 'label':label, 'status':'同几何的纹理修复 · 本地候选',
                'note':report.get('visualReview', '仅补充安全正面取色；不扩大几何偏差容限。') +
                    '\n纹理模块：%.2f → %.2f 秒（不含完整生成）。' % tuple(x['kernelSeconds'] for x in report['timing']),
                'files':{}, 'diagnostics':[]}
        (args.output/case).mkdir()
        def copy(path, filename):
            if not path.is_file(): return None
            if path.read_bytes()[:8] != b'\x89PNG\r\n\x1a\n': raise ValueError(path)
            target = args.output/case/filename
            shutil.copyfile(path,target)
            relative = f'{case}/{filename}'
            item['files'][filename] = relative
            manifest.append({'source':str(path.resolve()),'copy':relative,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
            return relative
        for filename in ('photo.png','comparison.png',*(f'{b}-{m}.png' for b in ('baseline','candidate') for m in ('baseColor','lit','clay'))):
            copy(directory/'eyes'/filename, 'eyes-'+filename)
        for view in range(4):
            copy(directory/'evidence'/f'view{view}-photo.png',f'view{view}-photo.png')
            for b in ('baseline','candidate'):
                for m in ('baseColor','lit','clay'):
                    filename=f'view{view}-{b}-{m}.png'
                    copy(directory/'evidence'/filename,filename)
        path=copy(directory/'changedUv.png','changedUv.png')
        item['diagnostics'].append({'label':'本轮改变的 UV 像素','path':path,
            'note':'白色表示实际导出像素发生改变。几何、相机、UV、眼球和附件贴图保留。'})
        data['cases'].append(item)
    page = PAGE.replace('快速正面眼部 · 对照实验','眼睑模板残留 · 修复对照')
    page = page.replace('repeat(4,minmax(0,1fr))','repeat(3,minmax(0,1fr))')
    page = page.replace('以原稳定版本为统一基底，共用相机、UV 和贴图烘焙策略。依次比较原几何、快速眼形修正、再加正面眼周取色。眼形只调整相机平面的局部顶点，保留 UFF 深度。此页面的对照渲染是离线证据，不计入生成时间。',
        '修复前后共用完全相同的几何、相机、UV 和去光照片。只补回被轮廓偏差误拦的安全正面皮肤取色；源像素必须有效且位于照片眼开口之外。仍保留几何错位和独立内眼角残留。离线渲染不计入生成时间。')
    page = page.replace('原照片与三组优化结果','原照片与修复前后结果').replace('局部取色权重','改动范围')
    page = page.replace('图左眼与图右眼 · 原照片 / 原稳定几何 / 快速眼形 / 正面取色','图左眼与图右眼 · 原照片 / 上一版 / 安全直投补充')
    page = page.replace('切换灰模检查眼形；快速眼形与正面取色两组几何相同。','切换灰模可核对修复前后几何完全相同。')
    encoded=json.dumps(data,ensure_ascii=False,indent=2)
    (args.output/'data.json').write_text(encoded+'\n',encoding='utf-8')
    (args.output/'index.html').write_text(page.replace('__DATA__',encoded.replace('<','\\u003c')),encoding='utf-8')
    (args.output/'buildManifest.json').write_text(json.dumps({'sourceRoot':str(args.root.resolve()),'files':manifest},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'output':str(args.output.resolve()),'copiedPngCount':len(manifest)}))

if __name__ == '__main__': main()
