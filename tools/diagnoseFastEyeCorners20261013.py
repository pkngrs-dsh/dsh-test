"""Read-only corner photo-projection and UV-support audit on saved CPU rasters."""
from pathlib import Path
import json,sys
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.ndimage import map_coordinates
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from faceAppearance.artifacts import writeJson
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.headAssembly import AssemblyAssets
from faceAppearance.colmapTexture import _destinationUvLookup

def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def main():
    out=ROOT/'runs/fastEyeResidualAttribution20261013';assets=AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1');a=assets.arrays
    corners=[c for c in assets.components if c['label']=='headAccessory'];cornerFaces=np.concatenate([c['faceIndices'] for c in corners]).astype(int)
    skinFaces=np.array([tuple(f) for f in a['nicpToFullIds'][a['nicpFaces']]])
    other={'vertices':a['vertices'],'faces':a['faces'][cornerFaces],'uvs':a['uvs'],'uvFaces':a['uvFaces'][cornerFaces],'uvConvention':'topLeftPixelEdges'}
    uvCoverage=_destinationUvLookup(other,1024)[0]['validMask']
    lookup={tuple(f):i for i,f in enumerate(a['faces'])};skinFullIds=np.array([lookup[tuple(f)] for f in skinFaces])
    skinUV=_destinationUvLookup({**other,'faces':a['faces'][skinFullIds],'uvFaces':a['uvFaces'][skinFullIds]},1024)[0]['validMask']
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',14)
    results={'cornerComponents':[{'name':c['name'],'fullFaceIds':c['faceIndices'],'vertexIds':c['vertexIndices']} for c in corners],
        'templateCornerUvRasterTexels':int(uvCoverage.sum()),'templateCornerUvOverlapNicpSkinTexels':int((uvCoverage&skinUV).sum()),
        'caveat':'Photo eyelid polygon is an automatic eye-opening proxy. Pixels inside it are not classified as skin; no dedicated tear-caruncle semantic detector is present.', 'cases':[]}
    for case in ('currentMale','highPitch4','cyanWall4','outdoorSideLight4'):
        base=ROOT/'runs/fastFrontEyes20261012/final'/case;record=json.loads((base/'report.json').read_text(encoding='utf-8'));photoPath=Path(record['paths']['observations'])/'view00WorkingPhoto.png'
        photo=np.asarray(Image.open(photoPath).convert('RGB'))/255;obs=npz(Path(record['paths']['observations'])/'view00.npz');prov=npz(base/'textureProvenance.npz');mesh=_load(base/'combined/legacyAsset.zip').mesh
        row={'case':case,'eyes':{}};panels=[]
        for ei,name in enumerate(('imageLeft','imageRight')):
            p=out/case/(name+'Raster.npz');r=npz(p);shape=r['faceIds'].shape;lo=r['origin'];s=float(r['scale']);upper=obs['eyelidCurvesWorking'][ei,1];lower=obs['eyelidCurvesWorking'][ei,0]
            if upper[0,0]>upper[-1,0]:upper=upper[::-1]
            if lower[0,0]>lower[-1,0]:lower=lower[::-1]
            polygon=np.r_[upper,lower[::-1]];eyeMask=np.zeros(shape,np.uint8);cv2.fillPoly(eyeMask,[np.rint((polygon-lo)*s*256).astype(np.int32)],1,shift=8)
            yy,xx=np.indices(shape);xy=np.stack([xx/s+lo[0],yy/s+lo[1]],-1)
            photoColors=np.stack([map_coordinates(photo[...,c],[xy[...,1],xy[...,0]],order=1,mode='nearest') for c in range(3)],-1)
            corner=r['accessory'];inside=corner&eyeMask.astype(bool)
            safe=map_coordinates(prov['sourceSafePixelMask'].astype(float),[xy[...,1],xy[...,0]],order=0,mode='constant',cval=0)>.5
            valid=r['valid'];ids=r['faceIds'][valid];uv=(mesh['uvs'][mesh['uvFaces'][ids]]*r['barycentric'][valid,...,None]).sum(1)
            corrected=np.full((*shape,2),np.nan);corrected[valid]=uv;r['uvs']=corrected;np.savez_compressed(p,**r)
            ids,counts=np.unique(r['faceIds'][corner],return_counts=True)
            byFace=[]
            for faceId,count in zip(ids,counts):
                mask=r['faceIds']==faceId
                byFace.append({'faceId':int(faceId),'visiblePixels':int(count),'insidePhotoEyePolygonFraction':float(eyeMask[mask].mean()),'validSafeSkinFraction':float(safe[mask].mean()),'screenCentroidWorking':xy[mask].mean(0)})
            photoCorners=.5*(upper[[0,-1]]+lower[[0,-1]])
            nasal=photoCorners[1 if ei==0 else 0];eyeWidth=np.linalg.norm(photoCorners[1]-photoCorners[0])
            originalError=r['originalHeadError'][corner]
            row['eyes'][name]={'visibleCornerPixels':int(corner.sum()),'insidePhotoEyePolygonPixels':int(inside.sum()),'insidePhotoEyePolygonFraction':float(inside.sum()/max(1,corner.sum())),
                'safePhotoSkinPixels':int((corner&safe).sum()),'protectedPixels':int((corner&r['protected']).sum()),'newTextureEditedPixels':int((corner&(r['alpha']>.01)).sum()),
                'preservedIncomingHeadColorMaximumError':float(np.nanmax(originalError)),
                'medianDistanceToPhotoInnerCanthusInEyeWidths':float(np.median(np.linalg.norm(xy[corner]-nasal,axis=1))/eyeWidth),'faces':byFace}
            annotated=photoColors.copy();annotated[corner]=.35*annotated[corner]+.65*np.array([1.,.05,.7]);annotated[inside]=.25*annotated[inside]+.75*np.array([1.,.3,0.])
            img=Image.fromarray(np.rint(np.clip(annotated,0,1)*255).astype(np.uint8));d=ImageDraw.Draw(img)
            for curve in (upper,lower):d.line([tuple(v) for v in (curve-lo)*s],fill='#4ffaff',width=2)
            for face in sorted(byFace,key=lambda x:x['visiblePixels'],reverse=True)[:3]:
                x,y=(np.asarray(face['screenCentroidWorking'])-lo)*s;d.text((x+3,y),str(face['faceId']),font=font,fill='white')
            img.save(out/case/(name+'-cornerPhotoProjection.png'));panels.append(img)
        results['cases'].append(row)
        grid=Image.new('RGB',(max(i.width for i in panels),sum(i.height for i in panels)+70),'#131b24');d=ImageDraw.Draw(grid);d.text((5,5),'青线：照片眼开口；粉：模型眼角附件投影；橙：投到照片眼内',font=font,fill='white');off=70
        for im in panels:grid.paste(im,(0,off));off+=im.height
        grid.save(out/case/'cornerPhotoProjection.png')
        print(case,{n:{k:v for k,v in e.items() if k!='faces'} for n,e in row['eyes'].items()},flush=True)
    writeJson(out/'cornerProjectionAudit.json',results)

if __name__=='__main__':main()
