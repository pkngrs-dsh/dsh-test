"""Current-request observation and rendering for frontal semantic refinement.

``getDetector(projectRoot)`` returns one cached local CPU FAN 2dfan2_alt model.
``observePhotos(projectRoot, photoPaths, inputs)`` returns batch-compatible
observation arrays plus ``diagnostics``; photographs must match the current
reconstruction input hashes and EXIF-oriented sizes. Invalid detections become
invalid views, while missing files, identity mismatches and backend errors raise.
``buildScene(assets, vertices, templateToUff)`` places the original accessories
once using assetDefault and retains only original template material maps.
``renderDetect(baseScene, vertices, cameras, inputs, observations,
outputDirectory, *, projectRoot, viewIndices=None)`` replaces only skin vertices,
independently rasterizes native and working grids, and returns render detections
and timing. ``outputDirectory=None`` keeps intermediate images in memory.

Rasterization and FAN settings match the local rendered semantic experiments.
There are no case IDs, cached observations, photo-derived maps or tool imports.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import io
from pathlib import Path
import sys
from threading import RLock
from time import perf_counter

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from .data import readImage
from .faceCrop import validateFaceDetection
from .imageColor import readSrgbImage

APPEARANCES = ('template', 'gray', 'grayAlternate')
SCALES = np.array([1., .95, 1.05])
_DETECTOR_LOCK = RLock()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def _cachedDetector(root):
    weights = Path(root)/'assets/models/ibug/2dfan2_alt.pth'
    if not weights.is_file():
        raise FileNotFoundError(f'Missing local FAN weights: {weights}')
    vendor = Path(root)/'thirdParty/ibug'
    if not (vendor/'ibug/face_alignment').is_dir():
        raise FileNotFoundError(f'Missing local FAN implementation: {vendor}')
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    import torch
    from ibug.face_alignment import FANPredictor

    torch.set_num_threads(1)
    model = FANPredictor.get_model('2dfan2_alt')
    model.weights = str(weights)
    digest = _sha(weights)
    detector = FANPredictor(device='cpu', model=model,
                            config=FANPredictor.create_config(use_jit=False))
    if _sha(weights) != digest:
        raise ValueError('FAN weight file changed while loading the model')
    detector.faceAppearanceWeightPath = str(weights)
    detector.faceAppearanceWeightSha256 = digest
    return detector


def getDetector(projectRoot):
    """Return a FANPredictor, cached by resolved project root; never download."""
    with _DETECTOR_LOCK:
        return _cachedDetector(str(Path(projectRoot).resolve()))


def _transform(points, matrix):
    homogeneous = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    return homogeneous[:, :2]/homogeneous[:, 2:]


def _expandedCrop(box, ratio):
    center = (box[:2]+box[2:])/2
    size = (box[2:]-box[:2]).mean()/ratio
    left = np.round(center-size/2)
    return np.r_[left, np.round(left+size)+1].astype(int)


def _size(value, label):
    size = np.asarray(value)
    if size.shape != (2,) or size.dtype.kind not in 'iu' or (size < 2).any():
        raise ValueError(f'{label} must contain two positive integer image dimensions')
    return tuple(int(v) for v in size)


def _inputCoordinates(record):
    """Coordinate metadata is integrity-critical and never silently rejected."""
    originalSize = _size(record['orientedSize'], 'orientedSize')
    workingSize = _size(record['workingSize'], 'workingSize')
    mapping = np.asarray(record['inputToWorking'], np.float64)
    inverse = np.asarray(record['workingToInput'], np.float64)
    if (mapping.shape != (3, 3) or inverse.shape != (3, 3)
            or not np.isfinite(mapping).all() or not np.isfinite(inverse).all()
            or not np.allclose(mapping@inverse, np.eye(3), rtol=1e-10, atol=1e-10)
            or not np.allclose(mapping[2], [0., 0., 1.], rtol=0, atol=1e-12)):
        raise ValueError('Invalid current input coordinate transform')
    return originalSize, workingSize, mapping


def _savedBox(record, originalSize):
    """A missing/unusable saved face detection is an observation failure."""
    try:
        box, _, _ = validateFaceDetection(record.get('faceDetection'))
    except ValueError as exc:
        return None, 'unusableSavedFaceDetection: '+str(exc)
    width, height = originalSize
    if box[2] <= 0 or box[3] <= 0 or box[0] >= width or box[1] >= height:
        return None, 'savedFaceBoxOutsideOriginalImage'
    # The ingestion contract already limits crops. Recheck grossly invalid
    # boxes before FAN constructs its padded image.
    if (box[2:]-box[:2]).max() > 4*max(originalSize):
        return None, 'savedFaceBoxExceedsImageExtent'
    return box, None


def _emptyDetection(originalSize, reason):
    return {'points': np.full((68, 2), np.nan),
            'variants': np.full((3, 68, 2), np.nan),
            'originalVariants': np.full((3, 68, 2), np.nan),
            'peaks': np.zeros((3, 68), np.float32),
            'valid': np.zeros(68, bool), 'spread': np.full(68, np.inf),
            'seconds': 0., 'sourceSize': list(originalSize),
            'status': 'unavailable', 'reason': reason,
            'gates': {key: np.zeros(68, bool) for key in
                      ('peak', 'variantPeak', 'stability', 'insideOriginal', 'insideWorking')}}


def _detect(detector, image, box, mapping, workingSize, faceScale, *, photograph=False):
    """Exact native-frame FAN crop variants and working-frame gates."""
    center, half = (box[:2]+box[2:])/2, (box[2:]-box[:2])/2
    boxes = np.array([np.r_[center-half*s, center+half*s] for s in SCALES])
    tick = perf_counter()
    # Only invalid observations are converted to a rejected view. Backend,
    # dependency, allocation and image-integrity failures must remain errors.
    with _DETECTOR_LOCK:
        inferred = detector(np.asarray(image), boxes, rgb=True)
    if not isinstance(inferred, (tuple, list)) or len(inferred) != 2:
        result = _emptyDetection(image.size, 'unusableFanOutput')
        result['seconds'] = perf_counter()-tick
        return result
    points, peaks = map(np.asarray, inferred)
    if (points.shape != (3, 68, 2) or peaks.shape != (3, 68)
            or points.dtype.kind not in 'fiu' or peaks.dtype.kind not in 'fiu'):
        result = _emptyDetection(image.size, 'unusableFanOutputShape')
        result['seconds'] = perf_counter()-tick
        return result
    working = np.array([_transform(p, mapping) for p in points])
    spread = np.linalg.norm(working-working[:1], axis=-1).max(0)
    gates = {'peak': peaks[0] >= .25, 'variantPeak': peaks.min(0) >= .20,
             'stability': spread/faceScale <= .015 if photograph else spread <= .015*faceScale,
             'insideOriginal': (points[0] >= 0).all(1) & (points[0] < image.size).all(1),
             'insideWorking': (working[0] >= 0).all(1) & (working[0] < workingSize).all(1)}
    valid = np.isfinite(working).all(axis=(0, 2)) & np.isfinite(peaks).all(0)
    for gate in gates.values():
        valid &= gate
    return {'points': working[0], 'variants': working, 'originalVariants': points,
            'peaks': peaks, 'valid': valid, 'spread': spread,
            'seconds': perf_counter()-tick, 'sourceSize': list(image.size),
            'status': 'detected' if valid.any() else 'unavailable',
            'reason': None if valid.any() else 'noValidFanLandmarks', 'gates': gates}


def observePhotos(projectRoot, photoPaths, inputs):
    """Infer current original photographs and return batch-compatible arrays.

    ``diagnostics`` describes per-view rejections, input identities and the
    local model. NaNs/false masks mark absent observations; identity errors are
    never converted to a no-op. Original photo paths must be supplied in the
    exact order of the reconstruction ``inputs`` metadata.
    """
    if len(photoPaths) != len(inputs) or not inputs:
        raise ValueError('Photo paths and current inputs require equal nonzero counts')
    started = perf_counter()
    detector = getDetector(projectRoot)
    workingAll, originalAll, peaksAll, validAll, spreads, scaleAll = [], [], [], [], [], []
    gates = {key: [] for key in ('peak', 'variantPeak', 'stability', 'insideOriginal', 'insideWorking')}
    records = []
    for index, (pathValue, record) in enumerate(zip(photoPaths, inputs)):
        originalSize, workingSize, mapping = _inputCoordinates(record)
        path = Path(pathValue)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != record['sha256']:
            raise ValueError(f'Original photograph hash differs from current input: view {index}')
        original, colorManagement = readSrgbImage(io.BytesIO(payload))
        if original.size != originalSize:
            raise ValueError(f'EXIF-oriented dimensions differ from current input: view {index}')
        box, rejected = _savedBox(record, originalSize)
        scale = np.nan
        if box is not None:
            boxWorking = _transform(box.reshape(2, 2), mapping)
            scale = float(np.sqrt(np.prod(np.abs(boxWorking[1]-boxWorking[0]))))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('Invalid current face-box scale')
            # Reproduce the exact ingestion crop. Detection or metadata drift
            # cannot silently change the correspondence coordinate system.
            _, replay = readImage(path, imageSize=workingSize[0],
                cropPolicy=record['requestedCropPolicy'], cropMargin=record.get('cropMargin', 1.35),
                faceDetector=lambda _, record=record: record['faceDetection'])
            for key in ('sha256', 'orientedSize', 'workingSize', 'inputToWorking',
                        'cropBoxXyxy', 'paddingLeftTopRightBottom'):
                if replay[key] != record[key]:
                    raise ValueError(f'Working crop replay mismatch: view {index}, {key}')
            prediction = _detect(detector, original, box, mapping, workingSize, scale, photograph=True)
        else:
            prediction = _emptyDetection(originalSize, rejected)
        workingAll.append(prediction['variants']); originalAll.append(prediction['originalVariants'])
        peaksAll.append(prediction['peaks']); validAll.append(prediction['valid'])
        spreads.append(prediction['spread']); scaleAll.append(scale)
        for key in gates:
            gates[key].append(prediction['gates'][key])
        info = {'viewIndex': index, 'photoPath': str(path), 'photoSha256': digest,
                'colorManagement': colorManagement,
                'orientedSize': list(originalSize), 'workingSize': list(workingSize),
                'inputToWorking': record['inputToWorking'], 'status': prediction['status'],
                'reason': prediction['reason'], 'faceScaleWorkingPixels': scale,
                'validLandmarkCount': int(prediction['valid'].sum()),
                'inferenceSeconds': prediction['seconds'], 'bboxScaleVariants': SCALES.tolist()}
        if box is not None:
            center, half = (box[:2]+box[2:])/2, (box[2:]-box[:2])/2
            boxes = np.array([np.r_[center-half*s, center+half*s] for s in SCALES])
            info.update(savedRetinaFaceBoxOriginal=box.tolist(),
                fanCropOriginal=_expandedCrop(box, detector.config.crop_ratio).tolist(),
                variantCropsOriginal=[_expandedCrop(b, detector.config.crop_ratio).tolist() for b in boxes])
        records.append(info)
    workingAll, originalAll, peaksAll, validAll, spreads, scaleAll = map(np.asarray,
        (workingAll, originalAll, peaksAll, validAll, spreads, scaleAll))
    return {'faceLandmarksWorking': workingAll[:, 0], 'faceLandmarksOriginal': originalAll[:, 0],
            'faceRawPeakScores': peaksAll[:, 0], 'landmarkCandidateValidMask': validAll,
            'cropVariantLandmarksWorking': workingAll, 'cropVariantLandmarksOriginal': originalAll,
            'cropVariantRawPeakScores': peaksAll, 'bboxScaleVariants': SCALES.copy(),
            'maximumCropVariationPixels': spreads, 'faceScaleWorkingPixels': scaleAll,
            'maximumCropVariationPerFaceScale': spreads/scaleAll[:, None],
            'acceptedViewMask': validAll[:, 7:10].all(1),
            'acceptedWideJawViewMask': validAll[:, 5:12].all(1),
            'workingImageShape': np.asarray([r['workingSize'][::-1] for r in inputs]),
            **{key+'GateMask': np.asarray(value) for key, value in gates.items()},
            'diagnostics': {'method': 'Current EXIF-oriented original photographs, CPU FAN 2dfan2_alt',
                'modelPath': detector.faceAppearanceWeightPath,
                'modelSha256': detector.faceAppearanceWeightSha256,
                'fanCropRatio': detector.config.crop_ratio, 'fanInputSize': detector.config.input_size,
                'backend': 'CPU PyTorch, one thread, no JIT; cached model',
                'candidateGate': {'rawPeakMinimum': .25, 'minimumAcrossVariantPeaks': .20,
                    'maximumCropVariationPerFaceScale': .015, 'insideOriginalAndWorkingImage': True},
                'views': records, 'seconds': perf_counter()-started}}


def buildScene(assets, vertices, templateToUff):
    """Freeze original accessories/materials once against baseline skin.

    ``templateToUff`` is the proper row-vector similarity used by
    AssemblyAssets: canonical = (clientCm @ rotation)*scale + translation.
    The returned scene can be reused for every proposal in this request.
    """
    arrays = assets.arrays
    vertices = np.asarray(vertices, np.float64)
    rotation = np.asarray(templateToUff['rotation'], np.float64)
    translation = np.asarray(templateToUff['translation'], np.float64)
    scale = float(templateToUff['scale'])
    if (vertices.shape != (len(arrays['nicpToFullIds']), 3) or not np.isfinite(vertices).all()
            or rotation.shape != (3, 3) or translation.shape != (3,)
            or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
            or not np.isfinite(scale) or scale <= 0
            or not np.allclose(rotation@rotation.T, np.eye(3), atol=1e-6)
            or np.linalg.det(rotation) <= 0):
        raise ValueError('Invalid baseline skin or template-to-UFF similarity')
    clientCm = ((vertices-translation) @ rotation.T)/scale
    assembled, components = assets._assembleSkin(clientCm, eyePlacementMode='assetDefault')
    assembled['vertices'] = (assembled['vertices'] @ rotation)*scale+translation
    assembled['vertices'][arrays['nicpToFullIds']] = vertices
    fullLookup = {tuple(face): i for i, face in enumerate(arrays['faces'])}
    skinIds = np.array([fullLookup[tuple(face)]
        for face in arrays['nicpToFullIds'][arrays['nicpFaces']]], np.int64)
    fullToSkin = np.full(len(arrays['faces']), -1, np.int32)
    fullToSkin[skinIds] = np.arange(len(skinIds))
    materials, hairPolicy, hiddenBrows, hiddenLashes = assets._materialsFromHead(
        assets.materials[0]['baseColor'].copy(), [])
    maps = [material['baseColor'] for material in materials]
    for material, image in zip(materials, maps):
        if material['alphaMode'] == 'OPAQUE':
            image[..., 3] = 1
    return {**assembled, 'skinFaceIds': skinIds, 'fullToSkinFaceIndex': fullToSkin,
            'nicpToFullIds': arrays['nicpToFullIds'].copy(), 'materials': materials, 'maps': maps,
            'components': components, 'provenance': {
                'templatePath': str(assets.assetRoot), 'templateManifestSha256': assets.manifestSha256,
                'templateArraySha256': assets.manifest['files']['template.npz'],
                'templateMaterialFiles': {m['texture']: assets.manifest['files'][m['texture']]
                                          for m in assets.materials},
                'accessoryPlacement': 'AssemblyAssets._assembleSkin(assetDefault); frozen against baseline skin',
                'templateTransform': templateToUff, 'hairPolicy': hairPolicy,
                'hiddenBrowMaterials': hiddenBrows, 'hiddenEyelashMaterials': hiddenLashes,
                'photoDerivedMaterials': False}}


def _linearToSrgb(value):
    return np.where(value <= .0031308, value*12.92, 1.055*np.maximum(value,0)**(1/2.4)-.055)


def _sample(image, uv):
    """Original v-down UVs use pixel edges and bilinear border clamp."""
    coordinates = np.stack((uv[:,1]*image.shape[0]-.5, uv[:,0]*image.shape[1]-.5))
    return np.stack([map_coordinates(image[...,c], coordinates, order=1,
        mode='nearest', prefilter=False) for c in range(image.shape[2])],axis=-1)


def _raster(scene, camera, shape):
    height,width=shape
    vertices,faces=scene['vertices'],scene['faces']
    camera_vertices=vertices@np.asarray(camera['rotation']).T+camera['translation']
    projected=camera_vertices@np.asarray(camera['intrinsics']).T
    pixels=projected[:,:2]/projected[:,2:]
    triangles=camera_vertices[faces]
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    front=(cross*(-triangles.mean(1))).sum(1)>0
    normals=np.zeros_like(camera_vertices)
    for i in range(3): np.add.at(normals,faces[:,i],cross)
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)
    depth=np.full((height,width),np.inf,np.float32)
    face_index=np.full((height,width),-1,np.int32)
    barycentric=np.zeros((height,width,3),np.float32)
    for face_id in np.flatnonzero(front):
        face=faces[face_id]
        z=camera_vertices[face,2]
        if (z<=0).any(): continue
        xy=pixels[face]
        lo=np.maximum(np.floor(xy.min(0)).astype(int),0)
        hi=np.minimum(np.ceil(xy.max(0)).astype(int),[width-1,height-1])
        if (lo>hi).any(): continue
        a,b,c=xy
        denominator=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(denominator)<1e-12: continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        u=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/denominator
        v=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/denominator
        w=1-u-v
        invz=u/z[0]+v/z[1]+w/z[2]
        distance=1/np.maximum(invz,1e-20)
        window=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        keep=(u>=-1e-10)&(v>=-1e-10)&(w>=-1e-10)&(distance<window)
        if not keep.any(): continue
        row,col=np.where(keep)
        weights=np.stack((u[keep]/z[0],v[keep]/z[1],w[keep]/z[2]),axis=-1)*distance[keep,None]
        material_id=scene['materialIds'][face_id]
        material=scene['materials'][material_id]
        if material['alphaMode']=='MASK':
            uv=weights@scene['uvs'][scene['uvFaces'][face_id]]
            valid=_sample(scene['maps'][material_id],uv)[:,3]>=material.get('alphaCutoff',.5)
            row,col,weights=row[valid],col[valid],weights[valid]
        row,col=row+lo[1],col+lo[0]
        depth[row,col]=(weights*z).sum(-1)
        face_index[row,col]=face_id
        barycentric[row,col]=weights
    visible=face_index>=0
    ids=face_index[visible]
    bary=barycentric[visible]
    material_ids=scene['materialIds'][ids]
    uv=np.einsum('ni,nij->nj',bary,scene['uvs'][scene['uvFaces'][ids]])
    normals=np.einsum('ni,nij->nj',bary,normals[faces[ids]])
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)
    colors=np.zeros((len(ids),3),np.float32)
    for i,image in enumerate(scene['maps']):
        mask=material_ids==i
        if mask.any(): colors[mask]=_sample(image,uv[mask])[:,:3]
    skin=scene['fullToSkinFaceIndex'][ids]>=0
    gray=colors.copy(); gray[skin]=.42
    variants={}
    for name,base,direction in [('gray',gray,[-.5,-.5,-1]),('grayAlternate',gray,[.5,-.5,-1]),
                               ('template',colors,[-.5,-.5,-1])]:
        light=np.asarray(direction,float); light/=np.linalg.norm(light)
        shade=.35+.65*np.maximum(normals@light,0)
        values=np.clip(_linearToSrgb(base*shade[:,None]),0,1)
        rgb=np.full((height,width,3),[27,33,40],np.uint8)
        rgb[visible]=np.rint(values*255).astype(np.uint8)
        variants[name]=Image.fromarray(rgb)
    material_map=np.full((height,width),-1,np.int16); material_map[visible]=material_ids
    skin_map=np.zeros((height,width),bool); skin_map[visible]=skin
    return variants,{'faceIndex':face_index,'barycentric':barycentric,'depth':depth,
        'materialId':material_map,'skinMask':skin_map}, {'visiblePixels':int(visible.sum()),
        'skinPixels':int(skin.sum()),'frontFacingFaceCount':int(front.sum())}


def renderDetect(baseScene, vertices, cameras, inputs, observations, outputDirectory,
                 *, projectRoot, viewIndices=None):
    """Render and detect template/gray/opposite-light gray from current mesh.

    All camera transforms remain fixed. Native RGB enters FAN using the same
    saved photo bbox, while a separate working-resolution raster is only saved
    for diagnostics. No photo colors are sampled. Returned views retain their
    original indices even when ``viewIndices`` selects only the frontal view.
    Accessory vertices and UV/material payloads are never refitted here.
    """
    if len(cameras) != len(inputs):
        raise ValueError('Cameras and current inputs must have equal counts')
    if len(observations['faceScaleWorkingPixels']) != len(inputs):
        raise ValueError('Photo observation and input counts differ')
    indices = list(range(len(inputs))) if viewIndices is None else list(viewIndices)
    if (any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
            or not 0 <= i < len(inputs) for i in indices) or len(indices) != len(set(indices))):
        raise ValueError('viewIndices must be distinct valid input indices')
    skin = np.asarray(vertices, np.float64)
    if skin.shape != (len(baseScene['nicpToFullIds']), 3) or not np.isfinite(skin).all():
        raise ValueError('Invalid candidate skin vertices')
    if 'photoHead' in baseScene:
        raise ValueError('Photo-derived maps are forbidden in semantic runtime rendering')
    current = {**baseScene, 'vertices': baseScene['vertices'].copy()}
    current['vertices'][current['nicpToFullIds']] = skin
    directory = None if outputDirectory is None else Path(outputDirectory)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    detector = getDetector(projectRoot)
    views = []
    renderSeconds = detectSeconds = 0.
    for index in indices:
        camera, record = cameras[index], inputs[index]
        originalSize, workingSize, mapping = _inputCoordinates(record)
        box, rejected = _savedBox(record, originalSize)
        faceScale = float(observations['faceScaleWorkingPixels'][index])
        if box is None or not np.isfinite(faceScale) or faceScale <= 0:
            views.append({'viewIndex': index, 'variants': {
                name: _emptyDetection(originalSize, rejected or 'invalidPhotoFaceScale')
                for name in APPEARANCES}})
            continue
        if camera is None:
            raise ValueError(f'Missing current render camera: view {index}')
        rotation = np.asarray(camera['rotation'], np.float64)
        translation = np.asarray(camera['translation'], np.float64)
        intrinsic = np.asarray(camera['intrinsics'], np.float64)
        if (rotation.shape != (3, 3) or translation.shape != (3,) or intrinsic.shape != (3, 3)
                or not np.isfinite(rotation).all() or not np.isfinite(translation).all()
                or not np.isfinite(intrinsic).all() or abs(np.linalg.det(intrinsic)) < 1e-12):
            raise ValueError(f'Invalid current render camera: view {index}')
        nativeCamera = {**camera, 'intrinsics': np.linalg.inv(mapping)@intrinsic}
        tick = perf_counter()
        native, _, _ = _raster(current, nativeCamera, originalSize[::-1])
        working, _, _ = _raster(current, camera, workingSize[::-1])
        renderSeconds += perf_counter()-tick
        row = {'viewIndex': index, 'variants': {}}
        for variant in APPEARANCES:
            prediction = _detect(detector, native[variant], box, mapping, workingSize, faceScale)
            row['variants'][variant] = prediction
            detectSeconds += prediction['seconds']
            if directory is not None:
                native[variant].save(directory/f'view{index}-{variant}-original.png')
                working[variant].save(directory/f'view{index}-{variant}.png')
                np.savez_compressed(directory/f'view{index}-{variant}.npz',
                    **{k: v for k, v in prediction.items() if isinstance(v, np.ndarray)})
        views.append(row)
    return views, {'renderSeconds': renderSeconds, 'detectionSeconds': detectSeconds,
                   'viewIndices': [int(i) for i in indices]}
