"""Trace a bounded closed-lip appearance proxy in the frozen male photographs.

This is an automatic image-feature observation, never hand-labelled truth.
Only experiment artifacts are written; source photographs and geometry stay fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]
from faceAppearance.headAssembly import decodeColor
from faceAppearance.texture import samplePixels


def encodeSrgb(linear):
    value=np.clip(linear,0,1)
    return np.where(value<=.0031308,value*12.92,1.055*value**(1/2.4)-.055)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def continuousPath(scores,allowed,y,stepPenalty=.08,maxStep=.75):
    """Exact first-order dynamic program with a bounded vertical slope."""
    n,m=scores.shape;cost=np.where(allowed[0],-scores[0],np.inf)
    previous=np.full((n,m),-1,np.int32)
    maxIndex=int(round(maxStep/(y[1]-y[0])))
    offsets=np.arange(-maxIndex,maxIndex+1)
    for x in range(1,n):
        candidates=np.full((len(offsets),m),np.inf)
        for k,delta in enumerate(offsets):
            here=np.arange(max(0,-delta),min(m,m-delta));before=here+delta
            candidates[k,here]=cost[before]+stepPenalty*((y[1]-y[0])*delta)**2
        best=candidates.argmin(0);previous[x]=np.arange(m)+offsets[best]
        cost=np.where(allowed[x],candidates[best,np.arange(m)]-scores[x],np.inf)
    if not np.isfinite(cost).any():raise ValueError('No continuous path through the bounded lip corridor')
    indices=np.empty(n,np.int32);indices[-1]=cost.argmin()
    for x in range(n-1,0,-1):indices[x-1]=previous[x,indices[x]]
    return indices


def imageScores(image,x,y,allowed):
    # Photo intensity retains its source encoding. Local contrast is normalized
    # separately in each image, making overall skin-tone changes irrelevant.
    intensity=np.asarray(image,dtype=np.float32)/255@np.array([.2126,.7152,.0722])
    xx,yy=np.meshgrid(x,y,indexing='ij');positions=np.stack([xx,yy],axis=-1)
    center=samplePixels(intensity,positions.reshape(-1,2)).reshape(xx.shape)
    above=samplePixels(intensity,(positions+[0,-2.0]).reshape(-1,2)).reshape(xx.shape)
    below=samplePixels(intensity,(positions+[0,2.0]).reshape(-1,2)).reshape(xx.shape)
    ridge=(above+below)*.5-center
    # The ridge is compared only inside the landmark-defined oral strip.
    low=np.min(np.where(allowed,center,np.inf),axis=1,keepdims=True)
    high=np.max(np.where(allowed,center,-np.inf),axis=1,keepdims=True)
    darkness=np.clip((high-center)/np.maximum(high-low,.01),0,1)
    contrastScale=max(float(np.percentile(np.maximum(ridge[allowed],0),90)),.005)
    scores=.8*np.clip(ridge/contrastScale,-2,2)+.2*darkness
    scores=np.where(allowed,scores,-np.inf)
    return scores,ridge,center,contrastScale


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/maleLipGeometry20261012/photoTrace')
    args=parser.parse_args();out=args.output
    if out.exists():raise FileExistsError(f'Keep prior tracing evidence: {out}')
    out.mkdir(parents=True)
    baseline=json.loads((ROOT/'reports/clientColmapBaseline20261012.json').read_text())
    photoPath=Path(baseline['sourceImages'][0]['path'])
    paths=[photoPath,ROOT/'reports/frontLandmark20261012/maleSource/inputs.json',
        ROOT/'reports/frontLandmark20261012/maleSource/delight.npz',
        ROOT/'runs/frontLandmark20261012/maleExactRegression/landmarks.npz']
    before={str(path):digest(path) for path in paths}
    if before[str(photoPath)]!=baseline['sourceImages'][0]['sha256']:raise ValueError('Original front photo changed')
    inputs=json.loads(paths[1].read_text());item=inputs[0]
    photo=ImageOps.exif_transpose(Image.open(photoPath)).convert('RGB')
    raw=cv2.warpPerspective(np.asarray(photo),np.asarray(item['inputToWorking']),tuple(item['workingSize']))
    with np.load(paths[2]) as data:delight=np.rint(encodeSrgb(decodeColor(data['albedoImages'][0],'gamma2.2'))*255).astype(np.uint8)
    with np.load(paths[3]) as data:landmarks={int(i):p for i,p in zip(data['ibugIds'],data['frontTargetPixels'])}
    upper=np.array([landmarks[i] for i in [60,61,62,63,64]])
    lower=np.array([landmarks[i] for i in [60,67,66,65,64]])
    x=np.arange(np.ceil(upper[0,0]-2),np.floor(upper[-1,0]+2)+1,dtype=float)
    y=np.arange(np.floor(min(upper[:,1].min(),lower[:,1].min())-4)*4,
                np.ceil(max(upper[:,1].max(),lower[:,1].max())+4)*4+1)/4
    upperY=np.interp(x,upper[:,0],upper[:,1]);lowerY=np.interp(x,lower[:,0],lower[:,1])
    outerTop=np.array([landmarks[i] for i in range(48,55)])
    outerBottom=np.array([landmarks[i] for i in [48,59,58,57,56,55,54]])
    lo=np.maximum(np.minimum(upperY,lowerY)-3,np.interp(x,outerTop[:,0],outerTop[:,1])+1)
    hi=np.minimum(np.maximum(upperY,lowerY)+2,np.interp(x,outerBottom[:,0],outerBottom[:,1])-1)
    allowed=(y[None,:]>=lo[:,None])&(y[None,:]<=hi[:,None])
    if not allowed.any(1).all():raise ValueError('Oral landmark corridor closes before the requested trace endpoints')
    rawScores,rawRidge,_,rawScale=imageScores(raw,x,y,allowed)
    delightScores,delightRidge,_,delightScale=imageScores(delight,x,y,allowed)
    rawIndex=continuousPath(rawScores,allowed,y);delightIndex=continuousPath(delightScores,allowed,y)
    combined=(rawScores+delightScores)/2
    selected=continuousPath(combined,allowed,y);curve=y[selected]
    rows=np.arange(len(x));jointRidge=np.minimum(rawRidge[rows,selected]/rawScale,delightRidge[rows,selected]/delightScale)
    agreement=np.exp(-abs(y[rawIndex]-y[delightIndex])/1.0)
    support=np.clip(jointRidge,0,1)*agreement
    endpointWeight=np.clip(np.minimum(x-upper[0,0],upper[-1,0]-x)/6,0,1)
    confidence=support*endpointWeight
    competing=combined.copy();competing[abs(y[None,:]-curve[:,None])<1.0]=-np.inf
    margin=combined[rows,selected]-np.max(competing,axis=1)
    unregularized=y[np.argmax(combined,axis=1)]
    arrays={'x':x,'y':curve,'xy':np.column_stack([x,curve]),'confidence':confidence,'rawPhotoY':y[rawIndex],
        'openDelightY':y[delightIndex],'unregularizedColumnPeakY':unregularized,'rawVsDelightDisagreementPixels':abs(y[rawIndex]-y[delightIndex]),
        'ridgeSupport':np.clip(jointRidge,0,1),'endpointWeight':endpointWeight,'alternativeRidgeScoreMargin':margin,
        'trustedMask':(confidence>=.35)&(endpointWeight>=.99),'searchLowerY':lo,'searchUpperY':hi,
        'innerUpperLandmarks':upper,'innerLowerLandmarks':lower}
    np.savez_compressed(out/'trace.npz',**arrays)
    comparisons=[]
    for i in range(60,68):
        point=landmarks[i];target=float(np.interp(point[0],x,curve))
        comparisons.append({'ibugId':i,'photoPoint':point.tolist(),'traceYAtPointX':target,'pointMinusTraceY':float(point[1]-target)})
    report={'caseId':'maleExactRegression','viewIndex':0,'coordinateFrame':'original photo transformed by exact cached inputToWorking into 512x512 pixels',
        'purpose':'automatic visible closed-lip appearance proxy; not manual truth or anatomical correspondence',
        'method':{'search':'inner upper/lower iBUG envelope plus [-3,+2] pixels, bounded inside outer lip contour by 1 pixel',
            'imageScore':'.8 locally normalized vertical dark-ridge contrast at +/-2 pixels + .2 in-corridor darkness',
            'combination':'equal separately normalized raw photograph and same-run OpenDelight scores',
            'continuity':'dynamic programming, 0.25 pixel y grid, at most 0.75 y pixel per x pixel; quadratic step cost 0.08',
            'confidence':'heuristic minimum per-image ridge support times exp(-independent trace disagreement/1px) times endpoint taper; not a probability',
            'trustedMask':'confidence >= .35 and at least 6px inside both inner corner x coordinates'},
        'xRange':x[[0,-1]].tolist(),'yRange':[float(curve.min()),float(curve.max())],
        'trustedXRange':x[arrays['trustedMask']][[0,-1]].tolist() if arrays['trustedMask'].any() else None,
        'rawVsDelightTraceDifferencePixels':{'median':float(np.median(abs(y[rawIndex]-y[delightIndex]))),'maximum':float(np.max(abs(y[rawIndex]-y[delightIndex])))},
        'endpointUncertainty':{'leftXRange':[float(x[0]),float(upper[0,0]+6)],'rightXRange':[float(upper[-1,0]-6),float(x[-1])],
            'reason':'mouth corner visibility and inner/outer lip detectors do not define a unique line endpoint; endpoint weights taper to zero at inner-corner x and outside'},
        'landmarkComparison':comparisons,'sourceHashes':before,'sourceFilesUnchanged':all(digest(p)==h for p,h in before.items()),
        'limitations':['The photographs used here also supplied reconstruction: this is not independent validation.',
            'Inner lip landmarks bracket a finite dark oral band; neither arc alone is the visible closed-lip ridge.',
            'The trace is an automatic image appearance proxy, affected by illumination, compression, OpenDelight processing and landmark localization.',
            'Confidence and the trusted mask are heuristic, not calibrated certainty; ambiguous endpoints should not become hard constraints.',
            'A smooth trace cannot establish actual 3D lip topology, anatomical closure or eliminate mesh intersections.']}
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    bbox=(208,326,313,377);scale=5;w=(bbox[2]-bbox[0])*scale;h=(bbox[3]-bbox[1])*scale
    panel=Image.new('RGB',(w*2,(h+44)*2),(28,35,39));draw=ImageDraw.Draw(panel)
    for i,(name,pixels,index) in enumerate([('Original photo / independent ridge',raw,rawIndex),('OpenDelight / independent ridge',delight,delightIndex),
                                           ('Joint curve + iBUG inner upper/lower',raw,selected),('Joint curve / confidence endpoints',delight,selected)]):
        tile=Image.fromarray(pixels).crop(bbox).resize((w,h),Image.Resampling.LANCZOS);d=ImageDraw.Draw(tile)
        points=[((xx-bbox[0])*scale,(yy-bbox[1])*scale) for xx,yy in zip(x,y[index])]
        d.line(points,fill=(60,235,240),width=2)
        if i>=2:
            for xx,l,hv in zip(x,lo,hi):
                d.point(((xx-bbox[0])*scale,(l-bbox[1])*scale),fill=(120,210,90));d.point(((xx-bbox[0])*scale,(hv-bbox[1])*scale),fill=(120,210,90))
            for lid in range(60,68):
                px,py=(landmarks[lid]-bbox[:2])*scale
                color=(255,205,40) if lid<=64 else (230,90,220)
                d.ellipse((px-3,py-3,px+3,py+3),outline=color,width=2);d.text((px+4,py+4),str(lid),fill=color)
            for j in np.where(confidence<.35)[0]:
                px,py=points[j];d.ellipse((px-1,py-1,px+1,py+1),fill=(245,100,60))
        ox=(i%2)*w;oy=(i//2)*(h+44);panel.paste(tile,(ox,oy+44));draw.text((ox+8,oy+10),name,fill='white')
    panel.save(out/'overlay.png')
    print(json.dumps({k:report[k] for k in ['xRange','yRange','trustedXRange','rawVsDelightTraceDifferencePixels','landmarkComparison','sourceFilesUnchanged']},indent=2))


if __name__=='__main__':main()
