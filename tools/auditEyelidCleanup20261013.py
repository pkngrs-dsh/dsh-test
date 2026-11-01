"""Independent ZIP/UV/source audit for the opt-in eyelid direct-projection trial.

Reads saved CPU experiment artifacts only. No UFF, GPU or texture resampling is
run. Checks use the exported model and original source pixels rather than the
sampler's declared safe/visible masks.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tools')]
from faceAppearance.artifacts import writeJson
from faceAppearance.colmapTexture import _destinationUvLookup
from faceAppearance.legacyPreviewAsset import _load
from faceAppearance.surfaceVisibility import ProjectedSurfaceVisibility
from auditFastFrontEyeArtifacts import photoOutside

CASES = ('currentMale', 'highPitch4', 'cyanWall4', 'outdoorSideLight4')


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image(asset):
    return np.asarray(Image.open(io.BytesIO(asset.payloads['head_color.png']))).copy()


def linear(pixels):
    color = np.asarray(pixels, np.float64)/255.
    return np.where(color <= .04045, color/12.92, ((color+.055)/1.055)**2.4)


def encoded(color):
    value = np.clip(color, 0, 1)
    return np.rint(np.where(value <= .0031308, value*12.92, 1.055*value**(1/2.4)-.055)*255).astype(np.uint8)


def sample(source, query):
    floor = np.floor(query).astype(np.int64)
    x, y = floor.T
    dx, dy = (query-floor).T
    return (source[y, x]*(1-dx[:, None])*(1-dy[:, None])
            + source[y, x+1]*dx[:, None]*(1-dy[:, None])
            + source[y+1, x]*(1-dx[:, None])*dy[:, None]
            + source[y+1, x+1]*dx[:, None]*dy[:, None])


def nearest_semantics(query, photo, model):
    starts, ends, pstarts, pends, widths = [], [], [], [], []
    for name, eye in zip(('imageLeft', 'imageRight'), photo):
        width = float(np.linalg.norm(eye[1, -1]-eye[1, 0]))
        for part, photo_index in (('upper', 1), ('lower', 0)):
            points = np.asarray(model['eyes'][name][part]['sourcePixels'])
            target = eye[photo_index]
            require(points.shape == target.shape, 'Model and photo semantic indices differ')
            starts.extend(points[:-1]); ends.extend(points[1:])
            pstarts.extend(target[:-1]); pends.extend(target[1:])
            widths.extend([width]*(len(points)-1))
    starts, ends, pstarts, pends, widths = map(np.asarray, (starts, ends, pstarts, pends, widths))
    edge = ends-starts
    fraction = np.clip(np.sum((query[:, None]-starts)*edge, axis=2)/np.maximum(np.sum(edge*edge, axis=1), 1e-16), 0, 1)
    closest = starts+fraction[..., None]*edge
    index = np.sum((query[:, None]-closest)**2, axis=2).argmin(axis=1)
    position = np.arange(len(query)); f = fraction[position, index]
    residual = (pstarts[index]+f[:, None]*(pends[index]-pstarts[index]))-closest[position, index]
    return widths[index], residual


def audit(directory):
    directory = Path(directory)
    report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
    require(report['status'] == 'completeLocalTextureExperiment', 'Incomplete experiment')
    require(report['geometryChanged'] is False and report['fullGenerationMeasured'] is False,
            'Incorrect experiment scope')
    source_case = Path(report['sourceCase'])
    paths = {key: Path(value) for key, value in report['paths'].items()}
    observation_manifest = json.loads((paths['observations']/'manifest.json').read_text(encoding='utf-8'))
    require(observation_manifest['inputsSha256'] == digest(paths['source']/'inputs.json')
            and observation_manifest['camerasSha256'] == digest(paths['source']/'cameras.json'),
            'Photo observations use a different input or camera frame')
    inputs = json.loads((paths['source']/'inputs.json').read_text(encoding='utf-8'))
    source_inputs = json.loads((paths['delight']/'inputs.json').read_text(encoding='utf-8'))
    require(len(inputs) == len(source_inputs) and all(all(a[key] == b[key]
            for key in ('sha256', 'inputToWorking', 'workingSize')) for a, b in zip(inputs, source_inputs)),
            'Source albedo uses different photos or crops')
    baseline_zip, candidate_zip = (directory/name/'legacyAsset.zip' for name in ('baseline', 'candidate'))
    require(baseline_zip.read_bytes() == (source_case/'combined/legacyAsset.zip').read_bytes(),
            'Baseline is not the exact preceding archive')
    require(digest(baseline_zip) == report['sourceArchiveSha256'], 'Baseline archive hash mismatch')
    baseline, candidate = _load(baseline_zip), _load(candidate_zip)
    require(baseline.manifest['materials'] == candidate.manifest['materials'], 'Material metadata changed')
    require(baseline.manifest['assemblyDiagnostics'] == candidate.manifest['assemblyDiagnostics'],
            'Assembly/camera frame metadata changed')
    require(all(np.array_equal(baseline.mesh[key], candidate.mesh[key]) for key in baseline.mesh),
            'Exported geometry/UV/topology/material arrays changed')
    unchanged = [name for name in baseline.payloads if name not in ('head_color.png', 'face_nicp.png')]
    require(all(baseline.payloads[name] == candidate.payloads[name] for name in unchanged),
            'Non-head payload or OBJ/MTL bytes changed')
    require(candidate.payloads['head_color.png'] == candidate.payloads['face_nicp.png'], 'Head aliases differ')
    before, after = image(baseline), image(candidate)
    require(before.shape == after.shape, 'Atlas dimensions changed')
    prior_bake = np.asarray(Image.open(source_case/'geometryTexture.png')).copy()
    p, control = npz(directory/'textureProvenance.npz'), npz(directory/'controlProvenance.npz')
    d = report['texture']; alpha = p['alpha']; shape = alpha.shape
    require(shape == before.shape[:2], 'Provenance shape differs from atlas')
    for name in ('alpha', 'linearRGB', 'filterAlpha'):
        value = p[name]
        require(np.isfinite(value).all() and (value >= 0).all() and (value <= 1).all(), 'Invalid '+name)
    bool_names = ('observedMask', 'estimatedMask', 'warpMask', 'extensionMask', 'filterPadding',
                  'projectionMask', 'aliasMask', 'writableMask', 'directProjectionFallbackMask',
                  'directProjectionDomainMask', 'originalBakeCoverage')
    for name in bool_names:
        require(p[name].dtype == np.bool_ and p[name].shape == shape, 'Invalid '+name)
    modified = (alpha > 0)|p['filterPadding']
    estimated = p['warpMask']|p['extensionMask']|p['filterPadding']
    observed = (alpha > 0)&~estimated
    fallback = p['directProjectionFallbackMask']
    bake_coverage = npz(source_case/'geometryBake/clientTexture.npz')['coverageMask']
    require(np.array_equal(p['originalBakeCoverage'], bake_coverage), 'Recorded original bake coverage differs')
    expected_domain = binary_dilation(~bake_coverage, iterations=1)&p['writableMask']
    require(np.array_equal(p['directProjectionDomainMask'], expected_domain), 'Direct fallback domain differs from original hole+one-texel policy')
    require(not (fallback&~expected_domain).any(), 'Direct fallback escaped original hole+one-texel domain')
    require(np.array_equal(p['estimatedMask'], estimated) and np.array_equal(p['observedMask'], observed),
            'Observed/estimated provenance definition differs')
    require(not (observed&estimated).any() and np.array_equal(modified, observed|estimated),
            'Observed/estimated masks overlap or miss edits')
    require(not (fallback&~observed).any(), 'Direct fallback marked warped/extended/padded or unobserved')
    require(not (modified&~p['writableMask']).any() and not (modified&p['aliasMask']).any(),
            'Protected or aliased UV edited')
    require(np.array_equal(after[~modified], before[~modified]), 'Other head UV changed')
    require(np.array_equal(before[..., 3:], after[..., 3:]), 'Head alpha changed')
    control_modified = (control['alpha'] > 0)|control['filterPadding']
    require(np.array_equal(p['warpMask'], control['warpMask'])
            and np.array_equal(p['extensionMask'], control['extensionMask']),
            'Fallback changed an existing warp/extension classification')
    require(np.array_equal(alpha[~fallback], control['alpha'][~fallback])
            and np.array_equal(alpha > 0, (control['alpha'] > 0)|fallback),
            'New center support/weight escaped the direct fallback branch')
    require(not (p['filterPadding']&(p['surfaceFaceIds'] >= 0)).any()
            and not (p['filterPadding']&~binary_dilation(alpha > 0, iterations=1)).any(),
            'Filter padding owns a surface or exceeds one texel')
    for py, px in zip(*np.nonzero(p['filterPadding'])):
        donor_found = False
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            y, x = py+dy, px+dx
            if 0 <= y < shape[0] and 0 <= x < shape[1] and alpha[y, x] > 0:
                donor_found |= (np.array_equal(p['linearRGB'][py, px], p['linearRGB'][y, x])
                                and np.array_equal(p['sourceXY'][py, px], p['sourceXY'][y, x])
                                and p['filterAlpha'][py, px] == alpha[y, x])
        require(donor_found, 'Padding color/source/alpha does not come from one adjacent accepted center')
    expected_before = prior_bake.copy()
    expected_before[control_modified, :3] = encoded(control['linearRGB'][control_modified])
    require(np.array_equal(expected_before, before), 'Disabled result does not reproduce preceding PNG')
    expected_after = prior_bake.copy()
    expected_after[modified, :3] = encoded(p['linearRGB'][modified])
    require(np.array_equal(expected_after, after), 'Export does not match candidate linear result')
    require(np.array_equal(p['linearRGB'][control['alpha'] > 0], control['linearRGB'][control['alpha'] > 0]),
            'Fallback changed an existing accepted center sample')
    require(not (fallback&(control['alpha'] > 0)).any(), 'Fallback replaced existing accepted center')
    # Recompute chart protection from all non-skin head faces, including corners.
    saved = npz(source_case/'geometry.npz'); skin = saved['skinVertexIds']
    sim = candidate.manifest['assemblyDiagnostics']['templateToUff']
    full = candidate.mesh['vertices']*100@np.asarray(sim['rotation'])*float(sim['scale'])+sim['translation']
    member = np.zeros(len(full), bool); member[skin] = True
    non_skin = (candidate.mesh['materialIds'] == 0)&~member[candidate.mesh['faces']].all(1)
    protected_geometry = dict(vertices=full, faces=candidate.mesh['faces'][non_skin],
        uvs=candidate.mesh['uvs'], uvFaces=candidate.mesh['uvFaces'][non_skin], uvConvention='topLeftPixelEdges')
    require(shape[0] == shape[1], 'Audit currently requires a square client atlas')
    protected = binary_dilation(_destinationUvLookup(protected_geometry, shape[0])[0]['validMask'], iterations=2)
    require(not (modified&protected).any() and np.array_equal(before[protected], after[protected]),
            'Non-skin head UV protection violated')
    source = npz(paths['delight']/'delight.npz'); valid = source['validMasks'][0]
    observations = npz(paths['observations']/'view00.npz')['eyelidCurvesWorking']
    model = json.loads((source_case/'textureBinding.json').read_text(encoding='utf-8'))
    query = p['sourceXY'][modified]
    require(np.isfinite(query).all(), 'Accepted source coordinates are nonfinite')
    floor = np.floor(query).astype(int); height, width = valid.shape
    require((floor >= 0).all() and (floor < [width-1, height-1]).all(), 'Source support out of bounds')
    neighbors = np.unique(np.concatenate([floor+[dx, dy] for dy in (0, 1) for dx in (0, 1)]), axis=0)
    require((valid[neighbors[:, 1], neighbors[:, 0]] >= .999).all(), 'Invalid bilinear source support')
    for eye in observations:
        polygon = np.concatenate([eye[1], eye[0][::-1]])
        require(photoOutside(neighbors, polygon, d['sourceSafetyWorkingPixels']).all(),
                'Bilinear support intersects photographic eye or safety margin')
    direct = npz(paths['source']/'directClient/geometry.npz')
    lookup = {tuple(face): index for index, face in enumerate(candidate.mesh['faces'])}
    full_ids = np.asarray([lookup[tuple(face)] for face in skin[direct['faces']]])
    require(np.array_equal(direct['uvs'], candidate.mesh['uvs'])
            and np.array_equal(direct['uvFaces'], candidate.mesh['uvFaces'][full_ids])
            and np.all(candidate.mesh['materialIds'][full_ids] == 0), 'Skin UV/topology binding differs')
    yy, xx = np.nonzero(alpha > 0); ids = p['surfaceFaceIds'][yy, xx]
    require((ids >= 0).all() and (ids < len(direct['faces'])).all(), 'Invalid sampled face ID')
    uv = direct['uvs']*shape[::-1]-.5
    tri = uv[direct['uvFaces'][ids]]; first, second, third = tri[:, 0], tri[:, 1], tri[:, 2]
    denominator = (second[:, 1]-third[:, 1])*(first[:, 0]-third[:, 0])+(third[:, 0]-second[:, 0])*(first[:, 1]-third[:, 1])
    require((abs(denominator) > 1e-12).all(), 'Degenerate owning UV triangle')
    b0 = ((second[:, 1]-third[:, 1])*(xx-third[:, 0])+(third[:, 0]-second[:, 0])*(yy-third[:, 1]))/denominator
    b1 = ((third[:, 1]-first[:, 1])*(xx-third[:, 0])+(first[:, 0]-third[:, 0])*(yy-third[:, 1]))/denominator
    bary = np.c_[b0, b1, 1-b0-b1]
    require((bary >= -1e-7).all(), 'Sample lies outside owning UV triangle')
    camera = json.loads((paths['source']/'cameras.json').read_text(encoding='utf-8'))[0]
    R, t, K = (np.asarray(camera[key]) for key in ('rotation', 'translation', 'intrinsics'))
    camera_triangles = full[skin][direct['faces'][ids]]@R.T+t
    points = np.sum(camera_triangles*bary[:, :, None], axis=1)
    h = points@K.T; xy = h[:, :2]/h[:, 2:]
    require(np.allclose(xy, p['projectedXY'][yy, xx], rtol=0, atol=7e-5),
            'Saved projection differs from actual exported model')
    normal = np.cross(camera_triangles[:, 1]-camera_triangles[:, 0], camera_triangles[:, 2]-camera_triangles[:, 0])
    require((np.sum(normal*-points, axis=1) > 0).all(), 'Accepted backfacing skin texel')
    opaque = candidate.mesh['faces'][candidate.mesh['materialIds'] != 3]
    visibility = ProjectedSurfaceVisibility(full@R.T+t, opaque, K, valid.shape)
    depth = visibility.closestDepth(points)['depth']; extent = np.ptp(full[skin], axis=0).max()
    require(np.isfinite(depth).all() and (abs(depth-points[:, 2]) <= extent*1e-6).all(),
            'Accepted center is occluded by the complete opaque assembly')
    require(np.array_equal(p['sourceXY'][fallback], p['projectedXY'][fallback]),
            'Direct projection fallback moved source coordinates')
    fallback_error = float(np.max(abs(p['sourceXY'][fallback]-p['projectedXY'][fallback]), initial=0))
    require(d['maximumWarpEyeWidths'] <= .10 and d['maximumExtensionEyeWidths'] <= .035,
            'Original warp or extension limit was relaxed')
    widths, semantic_residual = nearest_semantics(xy, observations, model)
    regular = ~fallback[yy, xx]
    residual = np.linalg.norm(semantic_residual, axis=1)
    require((residual[regular] <= widths[regular]*.10+1e-4).all(), 'Regular sample bypasses original semantic warp gate')
    require((residual[~regular] >= widths[~regular]*.10-1e-4).all(),
            'Direct fallback was used outside the rejected semantic-warp branch')
    displacement = np.linalg.norm(p['sourceXY'][yy, xx]-p['projectedXY'][yy, xx], axis=1)
    limit = widths*(.10+.035*p['extensionMask'][yy, xx])
    require((displacement[regular] <= limit[regular]+1e-4).all(), 'Actual query exceeds original bounded displacement')
    # Direct centers must actually use the unwarped original source color.
    source_linear = np.clip(source['albedoImages'][0], 0, 1).astype(np.float32)**2.2
    fyy, fxx = np.nonzero(fallback)
    amount = alpha[fyy, fxx, None]
    expected_direct = linear(prior_bake[..., :3])[fyy, fxx]*(1-amount)+sample(source_linear, p['sourceXY'][fallback])*amount
    color_error = float(np.max(abs(expected_direct-p['linearRGB'][fallback]), initial=0))
    require(color_error <= 8e-6, 'Direct fallback RGB does not match original source interpolation')
    changed = np.any(before != after, axis=2)
    return {'status': 'passedIndependentArtifactAudit', 'caseId': report['caseId'],
        'geometryUvMaterialArraysExact': True, 'objMtlAndNonHeadPayloadBytesExact': True,
        'unchangedPayloads': unchanged, 'baselineArchiveExact': True, 'headAliasesExact': True,
        'outsideModifiedHeadUvExact': True, 'protectedNonSkinHeadUvExact': True,
        'controlReproducesPreviousPngExactly': True, 'existingAcceptedCentersExact': True,
        'originalWarpExtensionAndCenterWeightsExact': True, 'filterPaddingAdjacentDonorVerified': True,
        'sourcePhotoCropCameraIdentityChecked': True,
        'sourceProvenanceDisjointComplete': True, 'uniqueSafeSourcePixels': len(neighbors),
        'directFallbackDomainRecomputedAndRespected': True,
        'observedMaskMeaning': 'Unwarped valid source contribution; alpha below one still mixes the preceding baked texture and does not establish wholly observed final RGB',
        'safeBilinearSupportOutsideBothPhotoEyes': True, 'centerSampleCount': len(xx),
        'directProjectionFallbackTexels': int(fallback.sum()), 'fallbackSourceMaximumDisplacementPixels': fallback_error,
        'fallbackSourceColorMaximumLinearError': color_error,
        'originalWarpAndExtensionLimitsPreserved': True,
        'projectionMaximumErrorWorkingPixels': float(np.max(abs(xy-p['projectedXY'][yy, xx]), initial=0)),
        'completeOpaqueSceneVisibilityPassed': True,
        'visibilityLimitation': 'MASK brow/lash cards excluded; this is not alpha-aware complete-scene validation',
        'filterPaddingTexels': int(p['filterPadding'].sum()), 'changedTexels': int(changed.sum()),
        'candidateArchiveSha256': digest(candidate_zip), 'fullGenerationMeasured': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT/'runs/eyelidCleanup20261013/directProjectionV1')
    parser.add_argument('--case', action='append', choices=CASES)
    args = parser.parse_args(); rows = []
    for case in args.case or CASES:
        result = audit(args.root/case)
        writeJson(args.root/case/'artifactAudit.json', result)
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    writeJson(args.root/'artifactAudit.json', {'status': 'passed', 'cases': rows})


if __name__ == '__main__':
    main()
