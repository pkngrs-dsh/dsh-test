"""Read-only bounded SE(3) camera audit with independently held-out nose landmarks."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from faceAppearance.artifacts import writeJson
from faceAppearance.surfaceVisibility import ProjectedSurfaceVisibility
from faceAppearance.camera import projectPoints
from faceAppearance.cameraPoseRefinement import solveBoundedCameraPose, validateCameraPoseCandidate
from faceAppearance.legacyPreviewAsset import _load


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def metrics(before,after,target,mask,ids):
 keep=mask&np.isin(np.arange(68),ids)
 if not keep.any():return {'count':0,'ids':[]}
 b,a=before[keep]-target[keep],after[keep]-target[keep]
 return {'count':int(keep.sum()),'ids':np.flatnonzero(keep).tolist(),'meanBefore':float(np.linalg.norm(b,axis=1).mean()),'meanAfter':float(np.linalg.norm(a,axis=1).mean()),'rmseBefore':float(np.sqrt((b*b).sum(1).mean())),'rmseAfter':float(np.sqrt((a*a).sum(1).mean())),'maxBefore':float(np.linalg.norm(b,axis=1).max()),'maxAfter':float(np.linalg.norm(a,axis=1).max()),'deltaXYBefore':b.mean(0).tolist(),'deltaXYAfter':a.mean(0).tolist()}


def run(outputDirectory):
 output=Path(outputDirectory)
 if output.exists() and any(output.iterdir()):raise FileExistsError('Use empty camera audit output')
 output.mkdir(parents=True,exist_ok=True)
 source=ROOT/'reports/feedbackAudit20261024/cases/28696732-c03c-4855-bde2-fe06f8ccaee7/reconstruction'
 replay=ROOT/'reports/feedbackAudit20261024/referenceGeometryReplay/ai286BoundarySafeChain'
 archive=ROOT/'runs/feedbackFix20261024/28696732/eyelidCandidate/asset/legacyAsset.zip'
 cameraPath=source/'cameras.json';cameraHash=sha(cameraPath);cameras=json.loads(cameraPath.read_text());photo=dict(np.load(source/'directClient/renderedJaw/photoObservations.npz'));geom=dict(np.load(replay/'finalGeometry.npz'));v,f=geom['vertices'],geom['faces'];state=np.load(replay/'renderedJaw/sourceState.npz');np.testing.assert_array_equal(v,state['vertices'])
 asset=_load(archive);sim=asset.manifest['assemblyDiagnostics']['templateToUff'];full=asset.mesh['vertices']*100@np.asarray(sim['rotation'])*sim['scale']+sim['translation'];np.testing.assert_allclose(full[state['nicpToFullIds']],v,atol=1e-12,rtol=0)
 trainIds=[36,39,42,45,48,51,54,57];testIds=list(range(27,36))+[49,50,52,53,55,56,58,59];domainIds=trainIds+testIds
 pivot=v.mean(0);scale=float(sim['scale']);views=[];newCameras=[]
 for index,camera in enumerate(cameras):
  k,r,t=(np.asarray(camera[n],float) for n in ['intrinsics','rotation','translation']);cv=v@r.T+t
  detections=[dict(np.load(replay/f'renderedJaw/baseline/view{index}-{name}.npz')) for name in ['template','gray','grayAlternate']];renderPoints=np.median(np.stack([d['points'] for d in detections]),axis=0);appearanceDrift=np.max(np.linalg.norm(np.stack([d['points'] for d in detections])-renderPoints[None],axis=2),axis=0)
  imageShape=tuple(map(int,photo['workingImageShape'][index]));faceScale=float(photo['faceScaleWorkingPixels'][index]);valid=photo['landmarkCandidateValidMask'][index].copy();valid &= (photo['faceRawPeakScores'][index]>=.25)&(photo['maximumCropVariationPixels'][index]<=.015*faceScale)
  for d in detections:valid &= d['valid'] & (d['peaks'].min(0)>=.2) & (d['spread']<=.015*faceScale)
  valid &= appearanceDrift<=.015*faceScale;valid &= np.isin(np.arange(68),domainIds)
  rays=np.c_[renderPoints,np.ones(68)]@np.linalg.inv(k).T;hits=ProjectedSurfaceVisibility(cv,f,k,imageShape).closestDepth(rays);hasHit=hits['faceIds']>=0;pointsCv=np.full((68,3),np.nan);pointsCv[hasHit]=rays[hasHit]*(hits['depth'][hasHit]/rays[hasHit,2])[:,None];points=(pointsCv-t)@r
  facing=np.zeros(68);bary=np.full((68,3),np.nan)
  for lid in np.flatnonzero(hasHit):
   tri=cv[f[hits['faceIds'][lid]]];norm=np.cross(tri[1]-tri[0],tri[2]-tri[0]);facing[lid]=norm@(-pointsCv[lid])/(np.linalg.norm(norm)*np.linalg.norm(pointsCv[lid]));uv=np.linalg.lstsq(np.stack([tri[1]-tri[0],tri[2]-tri[0]],axis=1),pointsCv[lid]-tri[0],rcond=None)[0];bary[lid]=[1-uv.sum(),*uv]
  valid &= hasHit & (facing>.05)&np.isfinite(points).all(1);valid &= np.nan_to_num(bary,nan=-1).min(1)>-1e-6
  before=np.full((68,2),np.nan);before[hasHit]=projectPoints(points[hasHit],k,r,t)[0];target=photo['faceLandmarksWorking'][index];train=valid&np.isin(np.arange(68),trainIds);heldout=valid&np.isin(np.arange(68),testIds)
  # The reusable module receives only finite, gated correspondences. Their order
  # and fitting subset are identical to the original standalone audit.
  candidate=solveBoundedCameraPose(points[valid],target[valid],k,r,t,train[valid],pivot=pivot)
  validation=validateCameraPoseCandidate(candidate,target[valid],heldout[valid],validationGroups={'nose':np.isin(np.flatnonzero(valid),np.arange(27,36))})
  parameters=candidate['normalizedParameters'];newR,newT=candidate['rotation'],candidate['translation'];after=np.full_like(before,np.nan);after[hasHit]=projectPoints(points[hasHit],k,newR,newT)[0]
  rotationDegrees=candidate['rotationDeltaDegrees'];deltaCenter=(-newR.T@newT)-(-r.T@t);translationMm=candidate['translationAtPivot']/scale*10;touches=validation['anyBoundTouched']
  row={'viewIndex':index,'status':candidate['status'],'training':metrics(before,after,target,valid,trainIds),'heldout':metrics(before,after,target,valid,testIds),'heldoutNose':metrics(before,after,target,valid,range(27,36)),'heldoutLip':metrics(before,after,target,valid,[49,50,52,53,55,56,58,59]),'all':metrics(before,after,target,valid,domainIds),'trainingIds':np.flatnonzero(train),'heldoutIds':np.flatnonzero(heldout),'rejectedDomainIds':np.flatnonzero(~valid&np.isin(np.arange(68),domainIds)),'rotationDeltaDegrees':rotationDegrees,'deltaParametersNormalized':parameters,'translationAboutFacePivotTemplateMm':translationMm,'opticalCenterShiftTemplateMm':float(np.linalg.norm(deltaCenter)/scale*10),'anyBoundTouched':touches,'before':{n:camera[n] for n in ['intrinsics','rotation','translation']},'after':{'intrinsics':k,'rotation':newR,'translation':newT},'perPoint':[{'id':int(lid),'accepted':bool(valid[lid]),'sourceMedianRenderPixel':renderPoints[lid],'photoPixel':target[lid],'beforeProjection':before[lid] if hasHit[lid] else None,'afterProjection':after[lid] if hasHit[lid] else None,'faceId':int(hits['faceIds'][lid]),'viewNormalCosine':float(facing[lid]),'appearanceDriftPixels':float(appearanceDrift[lid])} for lid in domainIds]}
  nose=row['heldoutNose'];held=row['heldout'];row['cameraOnlyDiagnosticAccepted']=validation['accepted'];row['cameraValidation']=validation
  newCameras.append({**camera,'intrinsics':k.tolist(),'rotation':newR.tolist(),'translation':newT.tolist(),'localCameraAuditOnly':True});views.append(row)
  np.savez_compressed(output/f'view{index:02d}Correspondence.npz',canonicalPoints=points,sourceRenderPixels=renderPoints,targetPhotoPixels=target,beforeProjection=before,afterProjection=after,acceptedMask=valid,trainingMask=train,heldoutMask=heldout,faceIds=hits['faceIds'],barycentric=bary,appearanceDrift=appearanceDrift)
  print(json.dumps({'viewIndex':index,'train':row['training'],'heldoutNose':nose,'heldout':held,'rotationDegrees':rotationDegrees,'bound':touches,'accepted':row['cameraOnlyDiagnosticAccepted']}),flush=True)
 if sha(cameraPath)!=cameraHash:raise RuntimeError('Source camera file changed')
 writeJson(output/'camerasBefore.json',cameras);writeJson(output/'camerasAfter.json',newCameras)
 writeJson(output/'cameraCandidates.json',newCameras)
 writeJson(output/'acceptedCameras.json',[new if row['cameraOnlyDiagnosticAccepted'] else old for old,new,row in zip(cameras,newCameras,views)])
 report={'status':'completeReadOnlyCameraAudit','sourceHashes':{str(p):sha(p) for p in [cameraPath,archive,replay/'finalGeometry.npz',source/'directClient/renderedJaw/photoObservations.npz']},'originalCameraFileUnchanged':True,'geometryAndIntrinsicsFrozen':True,'noPerPointImageWarp':True,'method':'Bounded six-DOF camera extrinsics around fixed face centroid; original mesh ray intersections at median of three rendered FAN appearances. Fit eye corners + mouth; all nasal points withheld.','trainingIdsRequested':trainIds,'heldoutIdsRequested':testIds,'bounds':{'rotationVectorComponentDegrees':2.,'translationAtFacePivotCanonical':[.05,.05,.12],'regularizerResidual':'.2 * normalized parameters; soft_l1 2px'},'views':views,'limitations':['Rendered/photo FAN are semantic detector proxies, not manually matched material points.','Correspondence is view-specific and frozen; camera fit can compensate some shape or detector bias.','Near-profile eye corners may be uncertain even when stability gates pass.','No new camera is applied to service, source reconstruction, package or texture bake.','All accepted camera variants require whole-scene renders and other-feature checks before adoption.']}
 report['correspondenceCacheHashes']={str(path):sha(path) for path in [replay/'renderedJaw/sourceState.npz',replay/'renderedJaw/renderScene.npz',replay/'renderedJaw/renderScene.json',source/'directClient/renderedJaw/photoObservations.json',*[replay/f'renderedJaw/baseline/view{view}-{name}.npz' for view in range(4) for name in ('template','gray','grayAlternate')]]}
 report['implementationSha256']=sha(Path(__file__))
 report['moduleImplementationSha256']=sha(ROOT/'src/faceAppearance/cameraPoseRefinement.py')
 report['candidateCameraPath']=str((output/'cameraCandidates.json').resolve())
 report['acceptedCameraPath']=str((output/'acceptedCameras.json').resolve())
 writeJson(output/'report.json',report)
 return report

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'reports/noseRepair20261024/cameraAudit');a=p.parse_args();run(a.output)
