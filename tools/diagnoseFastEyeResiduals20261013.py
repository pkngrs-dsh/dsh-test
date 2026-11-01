"""Read-only CPU attribution of visible final eye patches to mesh and UV data."""
from pathlib import Path
import hashlib,io,json,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import map_coordinates,binary_dilation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]
from faceAppearance.artifacts import writeJson
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs,_raster
from faceAppearance.colmapTexture import _destinationUvLookup

CASES=('currentMale','highPitch4','cyanWall4','outdoorSideLight4')
OUT=ROOT/'runs/fastEyeResidualAttribution20261013'

def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def sample(values,uv,order=1):
    xy=uv*np.array(values.shape[:2][::-1])-.5
    if values.ndim==2:return map_coordinates(values.astype(float),[xy[:,1],xy[:,0]],order=order,mode='nearest',prefilter=False)
    return np.stack([map_coordinates(values[...,c].astype(float),[xy[:,1],xy[:,0]],order=order,mode='nearest',prefilter=False) for c in range(values.shape[-1])],-1)

def main():
    OUT.mkdir(exist_ok=False)
    assets=AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1');a=assets.arrays
    skinMember=np.zeros(len(a['vertices']),bool);skinMember[a['nicpToFullIds']]=True
    skinFaces=skinMember[a['faces']].all(1)
    cornerFaces=np.concatenate([c['faceIndices'] for c in assets.components if c['label']=='headAccessory']).astype(int)
    component=np.full(len(a['faces']),-1,int)
    for i,c in enumerate(assets.components):component[c['faceIndices']]=i
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',15)
    rows=[]
    for case in CASES:
        root=ROOT/'runs/fastFrontEyes20261012/final'/case;out=OUT/case;out.mkdir()
        report=json.loads((root/'report.json').read_text(encoding='utf-8'))
        camera=json.loads((Path(report['paths']['source'])/'cameras.json').read_text(encoding='utf-8'))[0]
        obs=npz(Path(report['paths']['observations'])/'view00.npz')
        archive=root/'combined/legacyAsset.zip';zipdata=_load(archive)
        scene=loadLegacyEyeBindingInputs(archive,a['nicpToFullIds'])
        baked=npz(root/'geometryBake/clientTexture.npz');estimated=npz(root/'textureProvenance.npz')
        textures={i:np.asarray(Image.open(io.BytesIO(zipdata.payloads[mat['baseColorTexture']['path']])).convert('RGBA'))/255 for i,mat in enumerate(scene['materials'])}
        original=_load(Path(report['paths']['archive']))
        originalHead=np.asarray(Image.open(io.BytesIO(original.payloads['head_color.png'])).convert('RGBA'))/255
        template=assets.materials[0]['baseColor'][...,:3]
        fallbackGain=np.asarray(json.loads((root/'geometryBake/diagnostics.json').read_text(encoding='utf-8'))['templateLinearColorGain'])
        fallback=np.clip(template*fallbackGain,0,1)
        fallback=np.where(fallback<=.0031308,fallback*12.92,1.055*fallback**(1/2.4)-.055)
        eyeReport={};panels=[]
        for ei,name in enumerate(('imageLeft','imageRight')):
            upper=obs['eyelidCurvesWorking'][ei,1];lower=obs['eyelidCurvesWorking'][ei,0]
            points=np.r_[upper,lower];width=np.ptp(points[:,0]);low=np.floor(points.min(0)-[width*.35,width*.60]);hi=np.ceil(points.max(0)+[width*.35,width*.35]);scale=384/(hi[0]-low[0]);shape=tuple((np.ceil((hi-low)*scale).astype(int)+1)[::-1]);roi={'origin':low,'scale':scale,'shape':shape}
            hit,bary,depth,rasterStats=_raster(scene['vertices'],scene['faces'],scene['materialIds'],camera,roi,scene['uvs'],scene['uvFaces'],scene['materials'],scene['alphaTextures'])
            valid=hit>=0;faceId=hit[valid];uv=(scene['uvs'][scene['uvFaces'][faceId]]*bary[valid,...,None]).sum(1)
            mat=scene['materialIds'][faceId]
            rgb=np.zeros((*shape,3),float)
            color=np.zeros((len(faceId),3),float)
            for mid in np.unique(mat):color[mat==mid]=sample(textures[int(mid)][...,:3],uv[mat==mid])
            rgb[valid]=color
            fullSkin=np.zeros(shape,bool);fullSkin[valid]=skinFaces[faceId]
            accessory=np.zeros(shape,bool);accessory[valid]=np.isin(faceId,cornerFaces)
            eyeball=np.zeros(shape,bool);eyeball[valid]=mat==2
            coverage=np.zeros(shape,float);coverage[valid]=sample(baked['coverageMask'],uv)
            protected=np.zeros(shape,bool);protected[valid]=sample(estimated['writableMask'],uv,order=0)<.5
            alpha=np.zeros(shape,float);alpha[valid]=sample(estimated['alpha'],uv)
            padding=np.zeros(shape,float);padding[valid]=sample(estimated['filterPadding'],uv)
            sources=np.full(shape,-9,int);sources[valid]=sample(baked['sourceIndex'],uv,order=0).astype(int)
            originalError=np.full(shape,np.nan);originalError[valid]=np.max(np.abs(color-sample(originalHead[...,:3],uv)),1)
            fallbackError=np.full(shape,np.nan);fallbackError[valid]=np.max(np.abs(color-sample(fallback,uv)),1)
            componentRGB=np.zeros((*shape,3),np.uint8);componentRGB[fullSkin]=[45,190,150];componentRGB[accessory]=[245,55,190];componentRGB[eyeball]=[60,115,255]
            alphaCard=np.zeros(shape,bool);alphaCard[valid]=mat==3;componentRGB[alphaCard]=[255,165,30]
            provenanceRGB=np.asarray(np.clip(rgb*.25,0,1)*255,np.uint8)
            provenanceRGB[fullSkin&(coverage>=.999)]=[45,190,150]
            provenanceRGB[fullSkin&(coverage<.999)]=[135,70,235]
            provenanceRGB[fullSkin&protected]=[245,55,190]
            provenanceRGB[fullSkin&(alpha>.01)]=[255,185,30]
            provenanceRGB[accessory]=[255,90,200]
            yy,xx=np.indices(shape);screen=np.stack([xx/scale+low[0],yy/scale+low[1]],-1)
            loX,hiX=points[:,0].min(),points[:,0].max()
            uY=np.interp(screen[...,0],upper[:,0],upper[:,1]);lY=np.interp(screen[...,0],lower[:,0],lower[:,1])
            band=(screen[...,0]>=loX)&(screen[...,0]<=hiX)&(screen[...,1]<=uY+width*.08)&(screen[...,1]>=uY-width*.40)
            inner=points[np.argmax(points[:,0]) if ei==0 else np.argmin(points[:,0])]
            innerRegion=np.linalg.norm(screen-inner,axis=-1)<=width*.22
            summaries={}
            for regionName,region in [('upperLidBand',band),('innerCorner',innerRegion)]:
                selection=region&valid
                selectedIds=hit[selection];ids,counts=np.unique(selectedIds,return_counts=True);order=np.argsort(counts)[::-1]
                summaries[regionName]={'pixels':int(selection.sum()),'visibleFaceIds':ids.tolist(),
                    'componentCounts':{assets.components[int(i)]['name']+':'+assets.components[int(i)]['label']:int(n) for i,n in zip(*np.unique(component[selectedIds],return_counts=True))},
                    'topVisibleFaces':[{'faceId':int(ids[i]),'pixels':int(counts[i]),'material':int(scene['materialIds'][ids[i]]),'component':assets.components[component[ids[i]]]['label']} for i in order[:12]],
                    'nicpSkinPixels':int((selection&fullSkin).sum()),'nonNicpCornerPixels':int((selection&accessory).sum()),'eyeMaterialPixels':int((selection&eyeball).sum()),'alphaCardPixels':int((selection&alphaCard).sum()),
                    'nicpSkinUnobservedPixels':int((selection&fullSkin&(coverage<.999)).sum()),'nicpSkinProtectedPixels':int((selection&fullSkin&protected).sum()),'nicpSkinEyeTextureEditedPixels':int((selection&fullSkin&(alpha>.01)).sum()),
                    'nicpSkinUnobservedUneditedPixels':int((selection&fullSkin&(coverage<.999)&(alpha<.01)&(padding<.5)).sum()),
                    'originalHeadErrorMeanProtectedSkin':float(np.nanmean(originalError[selection&fullSkin&protected])) if np.any(selection&fullSkin&protected) else None,
                    'fallbackRgbErrorMeanUnobservedUnedited':float(np.nanmean(fallbackError[selection&fullSkin&(coverage<.999)&(alpha<.01)&(padding<.5)])) if np.any(selection&fullSkin&(coverage<.999)&(alpha<.01)&(padding<.5)) else None}
            uvMap=np.full((*shape,2),np.nan);uvMap[valid]=uv
            np.savez_compressed(out/(name+'Raster.npz'),faceIds=hit,barycentric=bary,uvs=uvMap,valid=valid,fullSkin=fullSkin,accessory=accessory,eyeball=eyeball,coverage=coverage,protected=protected,alpha=alpha,sources=sources,originalHeadError=originalError,fallbackError=fallbackError,upperBand=band,innerRegion=innerRegion,origin=low,scale=scale)
            eyeReport[name]={'roi':{'origin':low,'scale':scale,'shape':shape},'raster':rasterStats,'regions':summaries}
            colorImage=Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype(np.uint8));meshImage=Image.fromarray(componentRGB);provImage=Image.fromarray(provenanceRGB)
            for label,img in [('finalColor',colorImage),('component',meshImage),('provenance',provImage)]:img.save(out/(name+'-'+label+'.png'))
            strip=Image.new('RGB',(384*3,shape[0]+55),'#151a22');d=ImageDraw.Draw(strip)
            for i,(text,img) in enumerate([('最终颜色',colorImage),('几何归属：绿皮肤/粉眼角/蓝眼球',meshImage),('来源：紫无覆盖/粉保护/黄已取色',provImage)]):strip.paste(img,(i*384,55));d.text((i*384+5,12),name+' '+text,font=font,fill='white')
            panels.append(strip)
        grid=Image.new('RGB',(1152,sum(i.height for i in panels)),'#151a22');offset=0
        for panel in panels:grid.paste(panel,(0,offset));offset+=panel.height
        grid.save(out/'attribution.png')
        item={'case':case,'archiveSha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'eyes':eyeReport,
              'meaning':'First visible perspective-correct face after source alpha test; face IDs are zero-based full template faces. Estimated eye alpha is separate from original bake support.'}
        writeJson(out/'report.json',item);rows.append(item)
        print(case,{n:e['regions']['upperLidBand'] for n,e in eyeReport.items()},flush=True)
    writeJson(OUT/'summary.json',rows)

if __name__=='__main__':main()
