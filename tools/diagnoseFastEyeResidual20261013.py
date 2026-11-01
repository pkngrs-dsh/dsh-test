"""Read-only eye UV/source attribution on saved final meshes (CPU only)."""
import argparse,io,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from faceAppearance.visibleEyeContours import loadLegacyEyeBindingInputs,_raster
from faceAppearance.headAssembly import AssemblyAssets,decodeColor
from faceAppearance.texture import samplePixels
from faceAppearance.eyelidTextureSampling import _nearestSegments,_querySupport,_outsidePolygon
from renderClientAppearanceEvidence import evidenceGrid


def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sample(a,uv):
 xy=uv*np.array(a.shape[1::-1])-.5
 if a.ndim==2:return samplePixels(a[...,None].astype(float),xy.reshape(-1,2)).reshape(uv.shape[:2])
 return samplePixels(a,xy.reshape(-1,2)).reshape((*uv.shape[:2],a.shape[-1]))
def run(case,out):
 folder=ROOT/'runs/fastFrontEyes20261012/final'/case;report=json.loads((folder/'report.json').read_text());out.mkdir(parents=True,exist_ok=False)
 assets=AssemblyAssets(ROOT/'assets/clientTemplates/legacyMaleDirectServiceV1');skin=assets.arrays['nicpToFullIds']
 scene=loadLegacyEyeBindingInputs(folder/'combined/legacyAsset.zip',skin);scene.pop('nicpToFullIds')
 render=json.loads((folder/'eyes/report.json').read_text());roi=np.asarray(render['roiXyxyWorking']);factor=4
 shape=tuple(((roi[2:]-roi[:2])*factor).astype(int)[::-1]);origin=roi[:2]-.5+.5/factor
 camera=json.loads((Path(report['paths']['source'])/'cameras.json').read_text())[0]
 hits,weights,depth,_=_raster(scene['vertices'],scene['faces'],scene['materialIds'],camera,{'shape':shape,'origin':origin,'scale':factor},scene['uvs'],scene['uvFaces'],scene['materials'],scene['alphaTextures'])
 visible=hits>=0;face=hits.clip(0);uv=np.sum(scene['uvs'][scene['uvFaces'][face]]*weights[...,None],axis=2);mat=scene['materialIds'][face]
 member=np.zeros(len(scene['vertices']),bool);member[skin]=True;skinFace=member[scene['faces']].all(1);nicp=skinFace[face]&visible&(mat==0);corner=~skinFace[face]&visible&(mat==0)
 curves=load(Path(report['paths']['observations'])/'view00.npz')['eyelidCurvesWorking'];binding=json.loads((folder/'textureBinding.json').read_text())
 bake=load(folder/'geometryBake/clientTexture.npz');tex=load(folder/'textureProvenance.npz')
 yy,xx=np.indices(shape);xy=np.stack([origin[0]+xx/factor,origin[1]+yy/factor],-1)
 coverage=sample(bake['coverageMask'],uv);alpha=sample(tex['alpha'],uv);padding=sample(tex['filterPadding'],uv)
 bakeRgb=sample(bake['linearRGB'],uv)
 diag=report['branches'][-1]['texture'];gain=np.asarray(report['branches'][1]['texture']['templateLinearColorGain'])
 fallback=np.clip(assets.materials[0]['baseColor'][...,:3]*gain,0,1).astype(np.float32);templateRgb=sample(fallback,uv)
 fallbackAtlas=np.max(abs(bake['linearRGB']-fallback),axis=-1)<1e-7
 fallbackFraction=sample(fallbackAtlas,uv)
 safety=_querySupport(xy.reshape(-1,2),tex['sourceSafeCellMask']).reshape(shape)
 photoInside=np.zeros(shape,bool)
 for eye in curves:
  polygon=np.r_[eye[1],eye[0][::-1]]
  photoInside|=~_outsidePolygon(xy.reshape(-1,2),polygon,0).reshape(shape)
 mstart,mend,pstart,pend,widths,parts,which=[],[],[],[],[],[],[]
 for ei,name in enumerate(('imageLeft','imageRight')):
  width=np.linalg.norm(curves[ei,1,-1]-curves[ei,1,0])
  for pi,part in enumerate(('upper','lower')):
   m=np.asarray(binding['eyes'][name][part]['sourcePixels']);p=curves[ei,1 if pi==0 else 0]
   mstart.extend(m[:-1]);mend.extend(m[1:]);pstart.extend(p[:-1]);pend.extend(p[1:]);widths.extend([width]*11);parts.extend([pi]*11);which.extend([ei]*11)
 mstart,mend,pstart,pend=map(np.asarray,(mstart,mend,pstart,pend));widths=np.asarray(widths);parts=np.asarray(parts);which=np.asarray(which)
 distance,segment,fraction=_nearestSegments(xy.reshape(-1,2),mstart,mend)
 residual=np.linalg.norm((pstart[segment]+fraction[:,None]*(pend[segment]-pstart[segment]))-(mstart[segment]+fraction[:,None]*(mend[segment]-mstart[segment])),axis=1)
 residualFraction=(residual/widths[segment]).reshape(shape);part=parts[segment].reshape(shape);eyeId=which[segment].reshape(shape)
 inBand=(distance<widths[segment]*np.where(parts[segment]==0,.42,.28)).reshape(shape)
 highResidual=residualFraction>=.10
 palette=np.zeros((*shape,3),np.uint8);palette[visible]=[45,55,68];palette[corner]=[200,40,210];palette[nicp]=[85,85,85]
 palette[nicp&(coverage>=.999)]=[30,180,80];palette[nicp&(fallbackFraction>=.999)]=[245,190,30];palette[nicp&(alpha>.1)]=[30,155,235]
 safePic=np.zeros_like(palette);safePic[nicp]=[100,100,100];safePic[nicp&safety]=[20,200,95];safePic[nicp&photoInside]=[230,40,50];safePic[corner]=[200,40,210]
 residualPic=np.zeros_like(palette);residualPic[nicp&inBand&~highResidual]=[20,200,95];residualPic[nicp&inBand&highResidual]=[235,80,40];residualPic[corner]=[200,40,210]
 for name,pixels in [('sourceAttribution',palette),('directSafety',safePic),('residualGate',residualPic)]:Image.fromarray(pixels).save(out/(name+'.png'))
 rows=[]
 for ei,name in enumerate(('imageLeft','imageRight')):
  upper=nicp&inBand&(eyeId==ei)&(part==0)
  unedited=upper&(alpha<.01)&(padding<.01)
  template=upper&(fallbackFraction>.999)&(alpha<.01)&(padding<.01)
  row={'eye':name,'upperBandVisibleNicpPixels':int(upper.sum()),'upperBandUneditedPixels':int(unedited.sum()),'unmodifiedPureTemplatePixels':int(template.sum()),'unmodifiedTemplateSafeDirectPixels':int((template&safety).sum()),'unmodifiedTemplateInsidePhotoEyePixels':int((template&photoInside).sum()),'unmodifiedTemplateResidualRejectedPixels':int((template&highResidual).sum()),'residualRejectedSafeDirectPixels':int((unedited&highResidual&safety).sum()),'residualRejectedInsidePhotoEyePixels':int((unedited&highResidual&photoInside).sum()),'dominantFaceIds':[]}
  for fid in np.unique(hits[template]):
   domain=template&(hits==fid);row['dominantFaceIds'].append({'fullFaceId':int(fid),'vertices':scene['faces'][fid].tolist(),'rasterPixels':int(domain.sum()),'safeDirectPixels':int((domain&safety).sum()),'photoInteriorPixels':int((domain&photoInside).sum()),'meanResidualEyeWidths':float(residualFraction[domain].mean()),'meanAlpha':float(alpha[domain].mean())})
  row['dominantFaceIds']=sorted(row['dominantFaceIds'],key=lambda r:-r['rasterPixels'])[:12]
  rows.append(row)
 witness={}
 for label,mask in [('safeTemplate',nicp&inBand&highResidual&safety&(fallbackFraction>.999)&(alpha<.01)),('unsafeTemplate',nicp&inBand&photoInside&(fallbackFraction>.999)&(alpha<.01)),('corner',corner)]:
  pts=np.argwhere(mask)
  if len(pts):
   y,x=pts[len(pts)//2];witness[label]={'roiPixelXY':[int(x),int(y)],'workingXY':xy[y,x].tolist(),'faceId':int(hits[y,x]),'uv':uv[y,x].tolist(),'coverage':float(coverage[y,x]),'templateFraction':float(fallbackFraction[y,x]),'eyeAlpha':float(alpha[y,x]),'residualEyeWidths':float(residualFraction[y,x]),'directSourceSafe':bool(safety[y,x]),'photoInsideEye':bool(photoInside[y,x])}
 np.savez_compressed(out/'diagnosticMaps.npz',faceIds=hits,materialIds=mat,uvs=uv,nicp=nicp,corner=corner,workingXY=xy,bakeCoverage=coverage,fallbackFraction=fallbackFraction,eyeAlpha=alpha,directSafe=safety,insidePhotoEye=photoInside,residualEyeWidths=residualFraction,inBand=inBand,nearestEye=eyeId,nearestPart=part)
 panels=[Image.open(folder/'eyes/combined-baseColor.png'),Image.fromarray(palette),Image.fromarray(safePic),Image.fromarray(residualPic)]
 evidenceGrid(panels,['Combined base color','Green=bake photo / yellow=template / blue=eye override / purple=corner','Green=safe direct / red=photo eye interior / purple=corner','Green=warp residual valid / red=residual >10% / purple=corner'],columns=2,panelSize=650,subtitle=case+' | read-only attribution').save(out/'diagnosis.png')
 result={'caseId':case,'roiWorking':roi.tolist(),'rasterScale':4,'countsAreFourTimesDisplayPixelsNotIndependentOriginalPixels':True,'eyes':rows,'witnessPixels':witness,'cornerVisiblePixels':int(corner.sum()),'alphaCardVisiblePixels':int(((mat==3)&visible).sum()),'hypothesis':'Direct frontal projection may recover rejected pure-template skin only where its complete bilinear source support is valid and outside the photo eyes; eye-interior projections must remain rejected.'}
 (out/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'case':case,'eyes':[{k:v for k,v in x.items() if k!='dominantFaceIds'} for x in rows]}),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--case',action='append');args=parser.parse_args()
 for case in args.case or ['currentMale','highPitch4','outdoorSideLight4','cyanWall4']:run(case,ROOT/'runs/fastEyeResidualDiagnosis20261013'/case)
