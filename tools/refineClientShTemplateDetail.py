"""Reduce excess template detail only in its declared contribution to SH assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.clientShTemplateBlend import matchTemplateDetail
from faceAppearance.clientTextureBoundary import exteriorEdgeDiagnostics
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.legacyAsset import exportLegacyAsset
from faceAppearance.legacyPreviewAsset import _load
from runClientShComparison import load,sha,display


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,default=ROOT/'runs/clientShSeam20261012')
    args=parser.parse_args();root=args.experiment
    report=json.loads((root/'report.json').read_text())
    parent,source=Path(report['input']),Path(report['source'])
    target=root/'comparison/templateDetail'
    if target.exists():raise FileExistsError('templateDetail already exists')
    started=perf_counter()
    before=load(root/'templateMatch.npz')
    # Recompute the float fallback to avoid a PNG quantization round trip.
    from faceAppearance.clientShTemplateBlend import matchTemplateToSh
    old=load(parent/'baking/shOnly/clientTexture.npz')
    od=load(parent/'baking/openDelight/clientTexture.npz')
    assets=AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1')
    gain=np.asarray(json.loads((source/'directClient/baking/diagnostics.json').read_text())['templateLinearColorGain'])
    template=np.clip(assets.materials[0]['baseColor'][...,:3]*gain,0,1).astype(np.float32)
    matchingProvenance={**old,'openDelightBoundarySafeSkinDonor':od['boundarySafeSkinDonor'],
        'openDelightDirectPhotoAlpha':od['boundaryDirectPhotoAlpha']}
    matchingProvenance.pop('linearRGB')
    fallback=matchTemplateToSh(template,old['linearRGB'],od['linearRGB'],matchingProvenance)
    np.savez_compressed(root/'templateMatchFallback.npz',linearRGB=fallback['linearRGB'],**fallback['maps'],**fallback.get('debug',{}))
    detail=matchTemplateDetail(fallback['linearRGB'],old['linearRGB'],matchingProvenance,
                               fallback['maps']['templateMatchSkinWeight'])
    templateWeight=np.clip(1-before['boundaryDirectPhotoAlpha']-before['boundaryEstimatedSkinAlpha'],0,1)
    templateWeight*=1-before.get('estimatedInteriorSkinAlpha',np.zeros_like(templateWeight))
    core=old['coverageMask'] & ~old['boundaryModified']
    features=old['coverageMask'] & old['boundaryProtectedFeatures']
    mouth=old['frontAlpha']>0
    protected=core|features|mouth|old['filterPadding']|old['gutter']
    templateWeight[protected]=0
    colors=before['linearRGB']+(detail['linearRGB']-fallback['linearRGB'])*templateWeight[...,None]
    colors=np.clip(colors,0,1).astype(np.float32)
    checks={key:bool(np.array_equal(colors[mask],old['linearRGB'][mask])) for key,mask in
        [('coreExactlyUnchanged',core),('featuresExactlyUnchanged',features),('mouthExactlyUnchanged',mouth)]}
    if not all(checks.values()):raise ValueError('SH preservation changed: '+str(checks))
    maps={k:v for k,v in before.items() if k!='linearRGB'}
    maps.update(detail['maps']);maps['templateDetailContribution']=templateWeight.astype(np.float32)
    geometry=load(source/'directClient/geometry.npz');geometry['uvConvention']='topLeftPixelEdges'
    refined=load(source/'directClient/refinement/clientGeometry.npz')
    transform={key:refined['templateToUff'+key.title()] for key in ('rotation','translation','scale')}
    assembly=assets.assembleFinalClientSurface(geometry,transform,colors,maps,eyePlacementMode='assetDefault',
        diagnostics={'experiment':'shTemplateSeam','branch':'templateDetail','templateDetail':detail['diagnostics']})
    exportLegacyAsset(target,assembly,gender='male')
    base=_load(root/'comparison/shOriginal/legacyAsset.zip');exported=_load(target/'legacyAsset.zip')
    for key in ('vertices','faces','uvs','uvFaces','materialIds'):
        if not np.array_equal(base.mesh[key],exported.mesh[key]):raise ValueError('Geometry changed: '+key)
    for key in ('body_color.png','eye_color.png','brow_color.png','head_normals.png'):
        if base.payloads[key]!=exported.payloads[key]:raise ValueError('Other material changed: '+key)
    for key in ('coverageMask','sourceIndex','faceSourceIndex','frontAlpha'):
        if not np.array_equal(maps[key],old[key]):raise ValueError('Coverage/source changed: '+key)
    np.savez_compressed(root/'templateDetail.npz',linearRGB=colors,**maps)
    display(colors).save(root/'templateDetail-atlas.png')
    display(detail['linearRGB']).save(root/'fallback-detail-matched.png')
    branchReport={'seconds':perf_counter()-started,'preservation':checks,'templateDetail':detail['diagnostics'],
        'fixedOriginalExterior':exteriorEdgeDiagnostics(old['linearRGB'],colors,old['coverageMask'],old['atlasMask'],old['boundaryProtectedFeatures']),
        'modifiedTexels':int(np.any(colors!=before['linearRGB'],axis=-1).sum()),'observedTexels':int(old['coverageMask'].sum()),
        'zipSha256':sha(target/'legacyAsset.zip'),'headColorSha256':sha(target/'head_color.png'),
        'sourceColorAtlasSha256':sha(root/'templateMatch.npz'),
        'codeHashes':{name:sha(ROOT/name) for name in ['tools/refineClientShTemplateDetail.py','src/faceAppearance/clientShTemplateBlend.py']}}
    report['branches']['templateDetail']=branchReport
    writeJson(root/'report.json',report)
    config=json.loads((root/'comparison/config.json').read_text())
    config['branches'].insert(2,{'id':'templateDetail','label':'SH＋模板肤色与纹理适配'})
    config['defaultRight']='templateDetail'
    config['subtitle']='保留面部核心、眼鼻嘴的 SH 像素，让模板肤色与纹理反差适配 SH。'
    config['overviewDescription']='四张原照片与五个分支的基色、共同光照渲染。'
    config['notes'].insert(1,'纹理适配只降低模板贡献区域中过强的细小纹理反差；SH 面部颜色与鼻翼、眼下明暗保留。')
    writeJson(root/'comparison/config.json',config)
    print(json.dumps(branchReport,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
