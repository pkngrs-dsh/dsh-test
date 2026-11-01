"""Shared reference identity, conservative fallback, and stage integration."""
from copy import deepcopy
import hashlib
from itertools import permutations
import json
from types import SimpleNamespace

import numpy as np
import pytest

from faceAppearance.clientReferenceView import (chooseReferenceReport, selectClientReference,
                                               validateReferenceObservationIdentity)


def photoBatch():
    points = np.full((68, 2), 256., dtype=float)
    eye = np.array([[-12, 0], [-6, -4], [6, -4], [12, 0], [6, 4], [-6, 4]])
    points[36:42] = eye + [210, 180]
    points[42:48] = eye + [302, 180]
    points[6:11] = [[190, 345], [220, 367], [256, 380], [292, 367], [322, 345]]
    points[[48, 54, 57, 62, 66]] = [[220, 280], [292, 280], [256, 300], [256, 286], [256, 290]]
    side = points.copy()
    side[[48, 54, 57, 8], 0] += 55
    observations = {'faceLandmarksWorking': np.stack([side, points, points]),
                    'faceRawPeakScores': np.stack([np.full(68, .9), np.full(68, .7), np.full(68, .8)]),
                    'landmarkCandidateValidMask': np.ones((3, 68), bool),
                    'maximumCropVariationPixels': np.zeros((3, 68)),
                    'workingImageShape': np.tile([512, 512], (3, 1))}
    inputs = [{'sha256': hashlib.sha256(name.encode()).hexdigest(), 'workingSize': [512, 512]}
              for name in ('side', 'frontA', 'frontB')]
    observations['diagnostics'] = {'views': [{'viewIndex': i, 'photoSha256': item['sha256']}
                                           for i, item in enumerate(inputs)]}
    cameras = [{'valid': True, 'intrinsics': [[500, 0, 256], [0, 500, 256], [0, 0, 1]],
                'rotation': np.eye(3).tolist(), 'translation': [0, 0, 3]} for _ in inputs]
    return cameras, inputs, observations


def reorder(cameras, inputs, observations, order):
    result = {key: value[list(order)].copy() if isinstance(value, np.ndarray) else deepcopy(value)
              for key, value in observations.items()}
    result['diagnostics']['views'] = [dict(observations['diagnostics']['views'][old], viewIndex=new)
                                      for new, old in enumerate(order)]
    return [cameras[i] for i in order], [inputs[i] for i in order], result


def testFrontalGateAndAllInputPermutationsSelectSamePhotograph():
    cameras, inputs, observations = photoBatch()
    for order in permutations(range(3)):
        result = selectClientReference(*reorder(cameras, inputs, observations, order))
        assert result['referencePhotoSha256'] == inputs[2]['sha256']
        assert result['referenceViewIndex'] == order.index(2)
        assert result['selectedPhotoPassedAutomaticGate'] is True


def testCallerOverrideWinsEvenWhenAnotherPhotoScoresHigher():
    cameras, inputs, observations = photoBatch()
    result = selectClientReference(cameras, inputs, observations, referenceViewIndex=0)
    assert result['status'] == 'userOverride'
    assert result['referenceViewIndex'] == 0
    assert result['automaticReferenceViewIndex'] == 2
    assert result['selectedPhotoPassedAutomaticGate'] is False


def testUnreliableObservationsHaveNoImplicitFirstPhotoFallback():
    cameras, inputs, observations = photoBatch()
    observations['landmarkCandidateValidMask'][:] = False
    result = selectClientReference(cameras, inputs, observations)
    assert result['status'] == 'noReliableReference'
    assert result['referenceViewIndex'] is None and result['referencePhotoSha256'] is None


def testRejectedCameraCannotBeSelectedAutomaticallyOrByOverride():
    cameras, inputs, observations = photoBatch()
    cameras[2] = {'valid': False}
    assert selectClientReference(cameras, inputs, observations)['referenceViewIndex'] == 1
    with pytest.raises(ValueError, match='accepted camera'):
        selectClientReference(cameras, inputs, observations, referenceViewIndex=2)


@pytest.mark.parametrize('value', [True, -1, 3, 1.2, [], '0'])
def testInvalidOverrideFailsBeforeDownstreamStages(value):
    with pytest.raises(ValueError, match='referenceViewIndex'):
        selectClientReference(*photoBatch(), referenceViewIndex=value)


def testCacheIdentityMismatchOrUnpairedReorderingIsRejected():
    cameras, inputs, observations = photoBatch()
    with pytest.raises(ValueError, match='identities'):
        selectClientReference(cameras, inputs[::-1], observations)
    observations['diagnostics']['views'][0]['photoSha256'] = '0' * 64
    with pytest.raises(ValueError, match='identities'):
        validateReferenceObservationIdentity(observations, inputs)


def testEqualQualityTieUsesPhotoIdentityNotInputOrder():
    identities = ['b' * 64, 'a' * 64]
    reports = [{'viewIndex': i, 'referenceEligible': True, 'referenceSelectionScore': .8} for i in range(2)]
    assert chooseReferenceReport(reports, identities)['referencePhotoSha256'] == 'a' * 64
    assert chooseReferenceReport(reports, identities[::-1])['referencePhotoSha256'] == 'a' * 64


def testDuplicatePhotoIdentityIsAcceptedAndReported():
    cameras, inputs, observations = photoBatch()
    inputs[1]['sha256'] = inputs[2]['sha256']
    observations['diagnostics']['views'][1]['photoSha256'] = inputs[2]['sha256']
    result = selectClientReference(cameras, inputs, observations)
    assert result['referenceViewIndex'] == 2
    assert result['referencePhotoSha256'] == inputs[2]['sha256']
    assert result['duplicatePhotoIdentityCount'] == 1


def testSharedReferenceDetectorIsPreloadedWhenJawIsDisabled(tmp_path, monkeypatch):
    import faceAppearance.directClientService as service
    import faceAppearance.renderedJawRuntime as runtime
    import faceAppearance.clientLandmarks as landmarks
    import faceAppearance.adapters.eyelidLandmarks as eyelid
    calls = []
    monkeypatch.setattr(service.subprocess, 'run', lambda *args, **kwargs:
        SimpleNamespace(stdout='MeshTextureMapping', stderr=''))
    adapter = SimpleNamespace(modelMetadata={'weightSha256': 'stfr'})
    monkeypatch.setattr(eyelid, 'EyelidLandmarksAdapter', lambda *args:
        SimpleNamespace(loadModels=lambda: adapter))
    monkeypatch.setattr(landmarks, 'loadClientLandmarks', lambda *args: {})

    def getDetector(root):
        calls.append(root)
        return SimpleNamespace(faceAppearanceWeightSha256='fan')

    monkeypatch.setattr(runtime, 'getDetector', getDetector)
    config = SimpleNamespace(projectRoot=tmp_path, colmapExecutable='unused',
                             enableRenderedJawRefinement=False)
    assets = SimpleNamespace(geometryMethod='directSurface',
        geometryTransfer={'restWeight': 0, 'staticConstraint': 'hard'}, arrays={'nicpFaces': []})
    result = service.checkDirectRuntime(config, assets)
    assert calls == [tmp_path]
    assert result['referenceWeightSha256'] == 'fan'
    assert 'renderedJawMethod' not in result


def testPinnedJawReferenceCannotFallBackToEligibleOtherView(monkeypatch):
    import faceAppearance.renderedJawRefinement as jaw
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    semantic = {'index': 0, 'ids': np.array([7, 8, 9])}
    monkeypatch.setattr(jaw, '_prepare_semantic_views', lambda *args: ([semantic], []))
    monkeypatch.setattr(jaw, '_prepare_views', lambda *args: (
        [{'index': 0, 'report': {'referenceEligible': True, 'referenceSelectionScore': .9}}], []))
    cameras, _, observations = photoBatch()
    result = jaw.proposeRenderedJawCandidates(vertices, np.array([[0, 1, 2]]), cameras,
        observations, {'views': []}, frontPrimary=True, referenceViewIndex=1)
    assert result['diagnostics']['status'] == 'unchanged_insufficient_reliable_semantics'
    assert result['diagnostics']['requestedReferenceViewIndex'] == 1
    assert 'no other photograph' in result['diagnostics']['reason']
    assert len(result['candidates']) == 1
    np.testing.assert_array_equal(result['candidates'][0]['vertices'], vertices)


def testNoReferenceRetainsDirectVerticesAndWritesSafeUnavailableState(tmp_path, monkeypatch):
    import faceAppearance.clientGeometryRefinement as geometry
    vertices = np.zeros((2795, 3), float)
    vertices[:3] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    faces = np.array([[0, 1, 2]])
    fixed = np.zeros(2795, bool)
    fixed[:3] = True
    transform = {'rotation': np.eye(3), 'translation': [0., 0., 0.], 'scale': 1.}
    assets = SimpleNamespace(geometryMethod='directSurface', directArrays={},
        geometryTransfer={'restWeight': 0, 'staticConstraint': 'hard'},
        arrays={'nicpToFullIds': np.arange(2795), 'nicpFaces': faces, 'staticMask': fixed},
        _deform=lambda _: ({'vertices': vertices.copy()}, None, {'templateToUff': transform}))
    binding = {'ibugIds': np.array([48]), 'vertexIndices': np.array([[0, 1, 2]]),
               'barycentricWeights': np.array([[1., 0., 0.]]), 'validMask': np.array([True])}
    monkeypatch.setattr(geometry, 'loadClientLandmarks', lambda *args, **kwargs: {'isOriginalBinding': True})
    monkeypatch.setattr(geometry, 'ibugClientLandmarks', lambda *args, **kwargs: binding)
    monkeypatch.setattr(geometry, '_observations', lambda *args: ([None], [{'viewIndex': 0, 'status': 'unavailable'}]))
    monkeypatch.setattr(geometry, 'intersectionPairs', lambda *args: set())

    def unexpected(*args, **kwargs):
        pytest.fail('A missing reference must not enter any reference-driven correction')

    monkeypatch.setattr(geometry, '_stagedLandmarks', unexpected)
    monkeypatch.setattr(geometry, '_refineCrease', unexpected)
    cameras, inputs, _ = photoBatch()
    result = geometry.refineClientGeometry(assets, {}, cameras[:1], inputs[:1],
        photoPaths=[None], delight={}, projectRoot=tmp_path, outputDirectory=tmp_path / 'result',
        referenceViewIndex=None)
    np.testing.assert_array_equal(result['vertices'], vertices)
    np.testing.assert_array_equal(result['templateVertices'], vertices)
    assert result['diagnostics']['referenceViewIndex'] is None
    assert result['mouthState']['mouthObservationReliable'] is False
    assert not result['mouthState']['frontEvaluationMask'].any()
    with np.load(tmp_path / 'result/mouthState.npz', allow_pickle=False) as saved:
        assert int(saved['referenceViewIndex']) == -1
        assert not bool(saved['referenceViewAvailable'])
        assert all(saved[key].dtype.kind != 'O' for key in saved.files)


@pytest.mark.parametrize('available,override,expected', [(True, None, 2), (True, 1, 1), (False, None, None)])
def testDirectClientPassesOneReferenceToGeometryJawAndTexture(tmp_path, monkeypatch,
                                                           available, override, expected):
    from faceAppearance.directClientService import assembleDirectClient
    import faceAppearance.clientGeometryRefinement as geometryModule
    import faceAppearance.clientTextureBaking as bakingModule
    import faceAppearance.clientRenderedJaw as jawModule
    import faceAppearance.renderedJawRuntime as runtime
    cameras, inputs, observations = photoBatch()
    if not available:
        observations['landmarkCandidateValidMask'][:] = False
    (tmp_path / 'cameras.json').write_text(json.dumps(cameras))
    (tmp_path / 'inputs.json').write_text(json.dumps(inputs))
    np.savez(tmp_path / 'delight.npz', albedoImages=np.ones((3, 4, 4, 3)), validMasks=np.ones((3, 4, 4), bool))
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    faces = np.array([[0, 1, 2]])
    transform = {'rotation': np.eye(3), 'translation': np.zeros(3), 'scale': 1.}
    events = []

    def observe(*args):
        events.append(('observations', None))
        return observations

    def refine(*args, **kwargs):
        events.append(('geometry', kwargs['referenceViewIndex']))
        return {'vertices': vertices.copy(), 'faces': faces, 'templateToUff': transform,
                'templateVertices': vertices.copy(), 'mouthState': None,
                'diagnostics': {'referenceViewIndex': kwargs['referenceViewIndex']}}

    def jaw(*args, **kwargs):
        events.append(('jaw', kwargs['referenceViewIndex']))
        assert kwargs['photoObservations'] is observations
        return {'vertices': vertices.copy(), 'diagnostics': {'referenceViewIndex': kwargs['referenceViewIndex']}}

    def bake(*args, **kwargs):
        events.append(('baking', kwargs['referenceView']))
        return {'linearRGB': np.full((4, 4, 3), .2), 'coverageMask': np.ones((4, 4), bool),
                'diagnostics': {'referenceViewIndex': kwargs['referenceView']}}

    monkeypatch.setattr(runtime, 'observePhotos', observe)
    monkeypatch.setattr(geometryModule, 'refineClientGeometry', refine)
    monkeypatch.setattr(jawModule, 'refineRenderedClientJaw', jaw)
    monkeypatch.setattr(bakingModule, 'bakeClientTexture', bake)
    assets = SimpleNamespace(arrays={'nicpFaces': faces, 'faces': faces, 'nicpToFullIds': np.arange(3),
        'materialIds': np.array([0]), 'uvs': np.array([[0., 0.], [1., 0.], [0., 1.]]), 'uvFaces': faces},
        materials=[{'baseColor': np.ones((4, 4, 3))}],
        _transferTexture=lambda *args: (None, None, {'templateLinearColorGain': np.ones(3)}),
        assembleFinalClientSurface=lambda *args, **kwargs: {'status': 'testAssembled'})
    config = SimpleNamespace(projectRoot=tmp_path, assemblyColorEncoding='gamma2.2',
                             enableRenderedJawRefinement=True, colmapExecutable='not-executed')
    _, report = assembleDirectClient(assets, {}, {}, tmp_path, ['a', 'b', 'c'], config,
                                     referenceViewIndex=override)
    wanted = [('observations', None), ('geometry', expected)]
    if expected is not None:
        wanted.append(('jaw', expected))
    wanted.append(('baking', expected))
    assert events == wanted
    assert report['referenceSelection']['referenceViewIndex'] == expected
    if expected is None:
        assert report['renderedJaw']['status'] == 'unchanged_no_shared_reference'
