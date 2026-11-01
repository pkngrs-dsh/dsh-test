"""Sampling provenance must retain the exact validated bilinear cell.

The synthetic plane gives one accepted atlas center a source coordinate just
below an integer. No subject, template index, inferred curve or real-case data
is used. The existing float32 sourceXY storage fails the boundary cases.
"""
import numpy as np
import pytest

import faceAppearance.eyelidTextureSampling as sampling
from faceAppearance.clientEyelidAppearance import validateEyelidSourceFootprints


def syntheticInputs(sourcePoint):
    sourcePoint = np.asarray(sourcePoint, np.float64)
    origin = sourcePoint - [4.5, 4.5]
    corners = origin + np.array([[0., 0.], [8., 0.], [8., 8.], [0., 8.]])
    vertices = np.column_stack([corners, np.ones(4)])
    faces = np.array([[0, 2, 1], [0, 3, 2]], np.int64)
    geometry = dict(vertices=vertices, faces=faces,
        uvs=np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
        uvFaces=faces.copy(), uvConvention='topLeftPixelEdges')
    camera = dict(intrinsics=np.eye(3), rotation=np.eye(3), translation=np.zeros(3))
    yy, xx = np.indices((128, 128))
    source = np.stack([xx/127., yy/127., (xx+yy)/254.], axis=-1).astype(np.float32)
    valid = np.zeros((128, 128), bool)
    x, y = np.floor(sourcePoint).astype(int)
    valid[y:y+2, x:x+2] = True
    eyes = {}
    for name, center in (('imageLeft', 32.), ('imageRight', 88.)):
        eyes[name] = dict(
            upper=np.array([[center-6, 104.], [center, 102.], [center+6, 104.]]),
            lower=np.array([[center-6, 104.], [center, 108.], [center+6, 104.]]))
    return geometry, camera, source, valid, np.zeros((8, 8, 3), np.float32), eyes


@pytest.mark.parametrize('sourcePoint', [
    (31.9999998, 100.2),
    (32.2, 99.9999995),
    (31.9999998, 99.9999995),
    (31.25, 100.25),
], ids=['nearXInteger', 'nearYInteger', 'nearBothIntegers', 'binaryExactControl'])
def testSavedQueryKeepsActualValidatedSamplingCell(monkeypatch, sourcePoint):
    geometry, camera, source, valid, baseline, eyes = syntheticInputs(sourcePoint)
    realSamplePixels = sampling.samplePixels
    captured = []

    def recordActualQuery(image, query):
        captured.append(np.asarray(query).copy())
        return realSamplePixels(image, query)

    monkeypatch.setattr(sampling, 'samplePixels', recordActualQuery)
    result = sampling.reprojectEyelidTexture(geometry, camera, source, valid,
        baseline, eyes, eyes, corePixels=12., outerPixels=24.,
        sourceSafetyPixels=.75, maxExtensionPixels=0., filterFootprint=False)
    selected = result['alpha'] > 0
    assert len(captured) == 1
    assert selected.sum() == 1
    assert selected[4, 4]
    actual = captured[0]
    stored = result['sourceXY'][selected]
    exactMap = np.full(result['sourceXY'].shape, np.nan, np.float64)
    exactMap[selected] = actual
    actualChecks = validateEyelidSourceFootprints(exactMap, selected, valid, eyes, eyes)
    assert actualChecks['invalidSourceSupportQueryCount'] == 0
    assert np.array_equal(np.floor(stored), np.floor(actual)), 'Stored query changed the validated bilinear cell'
    savedChecks = validateEyelidSourceFootprints(result['sourceXY'], selected, valid, eyes, eyes)
    assert savedChecks['invalidSourceSupportQueryCount'] == 0
    np.testing.assert_array_equal(stored, actual)


def testIntegerBoundaryCounterexampleIsRejectedWithoutRelaxingTheGuard():
    actual = np.array([[[31.9999998, 100.2]]], np.float64)
    stored = actual.astype(np.float32)
    _, _, _, valid, _, eyes = syntheticInputs(actual[0, 0])
    selected = np.ones((1, 1), bool)
    assert validateEyelidSourceFootprints(actual, selected, valid, eyes, eyes)['invalidSourceSupportQueryCount'] == 0
    rejected = validateEyelidSourceFootprints(stored, selected, valid, eyes, eyes)
    assert rejected['invalidSourceSupportQueryCount'] == 1
    assert not rejected['allFourSourcePixelsValid']
