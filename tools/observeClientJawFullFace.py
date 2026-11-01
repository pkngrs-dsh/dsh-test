"""Local full-face FAN observations for chin refinement; no geometry changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'thirdParty/ibug')]
from faceAppearance.artifacts import writeJson
from faceAppearance.data import readImage
from ibug.face_alignment import FANPredictor


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expandedCrop(box, ratio):
    box = np.asarray(box, np.float64)
    center = (box[:2]+box[2:])/2
    size = (box[2:]-box[:2]).mean()/ratio
    left = np.round(center-size/2)
    right = np.round(left+size)+1
    return np.r_[left,right].astype(int)


def toWorking(points, matrix):
    homogeneous = np.c_[points,np.ones(len(points))]@matrix.T
    return homogeneous[:,:2]/homogeneous[:,2:]


def drawObservation(photo, points, scores, stability, *, view):
    image = photo.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    for j,(x,y) in enumerate(points):
        if not (0<=x<image.width and 0<=y<image.height):
            continue
        central=5<=j<=11
        color=(65,255,118) if central else (255,225,74) if j<17 else (110,180,255)
        if scores[j]<.25:color=(255,105,65)
        r=3 if central else 2
        draw.ellipse((x-r,y-r,x+r,y+r),fill=color)
        if j<17:
            draw.text((x+4,y-9),str(j),font=font,fill=color,stroke_width=1,stroke_fill=(0,0,0))
    jaw=points[:17]
    draw.line([tuple(p)for p in jaw],fill=(255,225,74),width=1)
    draw.rectangle((0,0,511,49),fill=(18,24,31))
    draw.text((8,5),f'Photo {view+1} | Full-face FAN, original-image inference',font=font,fill='white')
    draw.text((8,25),f'Chin 8: ({points[8,0]:.1f}, {points[8,1]:.1f}) | peak {scores[8]:.3f} | crop spread {stability[8]:.2f}px',font=font,fill='white')
    return image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'runs/clientAppearanceFix20261012/deployedSource')
    parser.add_argument('--photos',type=Path,default=ROOT/'runs/legacyPreview/preview-fb21ed26fc4b41969d581365622054e5')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/clientChinRefine20261012/observations')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Observation output must be empty')
    args.output.mkdir(parents=True,exist_ok=True)
    started=perf_counter()
    import torch
    torch.set_num_threads(1)
    model=FANPredictor.get_model('2dfan2_alt')
    model.weights=str(ROOT/'assets/models/ibug/2dfan2_alt.pth')
    detector=FANPredictor(device='cpu',model=model,config=FANPredictor.create_config(use_jit=False))
    inputs=json.loads((args.source/'inputs.json').read_text())
    arrays,scores,allVariants,allScores,records,panels=[],[],[],[],[],[]
    scales=[1.,.95,1.05]
    for i,record in enumerate(inputs):
        path=args.photos/f'{i:02d}.image'
        if sha(path)!=record['sha256']:raise ValueError('Photo hash differs')
        image=np.asarray(ImageOps.exif_transpose(Image.open(path)).convert('RGB'))
        box=np.asarray(record['faceDetection']['boxXyxy'],np.float64)
        center=(box[:2]+box[2:])/2
        half=(box[2:]-box[:2])/2
        boxes=np.asarray([np.r_[center-half*scale,center+half*scale]for scale in scales])
        tick=perf_counter()
        points,peaks=detector(image,boxes,rgb=True)
        workingPoints=np.asarray([toWorking(p,np.asarray(record['inputToWorking']))for p in points])
        spread=np.linalg.norm(workingPoints-workingPoints[0],axis=-1).max(0)
        working,replay=readImage(path,imageSize=record['workingSize'][0],cropPolicy=record['requestedCropPolicy'],
            cropMargin=record['cropMargin'],faceDetector=lambda _,record=record:record['faceDetection'])
        for key in ['inputToWorking','cropBoxXyxy','paddingLeftTopRightBottom']:
            if replay[key]!=record[key]:raise ValueError('Working crop mismatch')
        photo=Image.fromarray(np.rint(working*255).astype(np.uint8))
        panel=drawObservation(photo,workingPoints[0],peaks[0],spread,view=i)
        photo.save(args.output/f'view{i}-photo.png')
        panel.save(args.output/f'view{i}-landmarks.png')
        panels.append(panel)
        arrays.append(workingPoints[0]);scores.append(peaks[0]);allVariants.append(workingPoints);allScores.append(peaks)
        fullCrop=expandedCrop(box,detector.config.crop_ratio)
        cropWorking=toWorking(fullCrop.reshape(2,2),np.asarray(record['inputToWorking']))
        view={'viewIndex':i,'photoSha256':record['sha256'],'sourceImageSize':record['orientedSize'],
            'savedRetinaFaceBoxOriginal':box.tolist(),'fanCropOriginal':fullCrop.tolist(),'fanCropWorking':cropWorking.tolist(),
            'cropPadding':'constant zero outside original image, upstream FANPredictor policy',
            'bboxScaleVariants':scales,'variantCropsOriginal':[expandedCrop(b,detector.config.crop_ratio).tolist()for b in boxes],
            'chin8Working':workingPoints[0,8].tolist(),'chin8RawPeakScore':float(peaks[0,8]),
            'chin8MaximumCropVariationPixels':float(spread[8]),'jaw5to11RawPeakScores':peaks[0,5:12].tolist(),
            'jaw5to11MaximumCropVariationPixels':spread[5:12].tolist(),
            'allJawRawPeakScores':peaks[0,:17].tolist(),'seconds':perf_counter()-tick}
        records.append(view)
        print(json.dumps(view),flush=True)
    arrays,scores,allVariants,allScores=map(np.asarray,(arrays,scores,allVariants,allScores))
    spread=np.linalg.norm(allVariants-allVariants[:,:1],axis=-1).max(1)
    valid=(scores>=.25)&np.isfinite(arrays).all(-1)&(arrays>=0).all(-1)&(arrays<512).all(-1)&(spread<=5)
    np.savez_compressed(args.output/'landmarkObservations.npz',faceLandmarksWorking=arrays,
        faceRawPeakScores=scores,landmarkCandidateValidMask=valid,
        cropVariantLandmarksWorking=allVariants,cropVariantRawPeakScores=allScores,
        maximumCropVariationPixels=spread,acceptedViewMask=valid[:,5:12].all(1))
    board=Image.new('RGB',(1024,1024),(20,24,30))
    for i,panel in enumerate(panels):board.paste(panel,((i%2)*512,(i//2)*512))
    board.save(args.output/'jawObservations.png')
    report={'status':'complete','method':'Independent full-face ibug FAN 2dfan2_alt on original photographs',
        'backend':'PyTorch CPU, one thread, no JIT','modelPath':model.weights,'modelSha256':sha(model.weights),
        'implementationSha256':sha(Path(__file__)),'predictorSha256':sha(ROOT/'thirdParty/ibug/ibug/face_alignment/fan/fan_predictor.py'),
        'fanCropRatio':detector.config.crop_ratio,'fanInputSize':detector.config.input_size,'workingSize':[512,512],
        'selectedVariant':'saved complete RetinaFace box, scale 1.0; no selection by desired geometry result',
        'cropVariants':'0.95 and 1.05 are sensitivity diagnostics; output uses baseline detector predictions',
        'candidateGate':{'rawPeakMinimum':.25,'maximumCropVariationPixels':5,'insideWorkingImage':True},
        'acceptedCentralJawViewMask':valid[:,5:12].all(1).tolist(),'views':records,'seconds':perf_counter()-started,
        'limitations':['Raw heatmap peaks are not calibrated confidence probabilities.',
            'Contour landmarks are view dependent 2D silhouettes, not fixed 3D material correspondences.',
            'Small crop perturbation stability does not establish anatomical accuracy.',
            'Independent manual labels were not supplied; visually inspect overlays before geometry fitting.',
            'Existing eyelid detector and service code are unchanged; these are local experimental observations.']}
    writeJson(args.output/'report.json',report)
    print(json.dumps({'status':'complete','seconds':report['seconds'],'acceptedCentralJawViewMask':report['acceptedCentralJawViewMask']}),flush=True)


if __name__=='__main__':main()
