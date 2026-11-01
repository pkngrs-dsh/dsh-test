"""Prevent historical OD caches from silently changing input color processing."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from faceAppearance.data import readImage


toolPath = Path(__file__).resolve().parents[1] / 'tools/runClientShComparison.py'
specification = importlib.util.spec_from_file_location('clientShComparisonReplayTestTool', toolPath)
experiment = importlib.util.module_from_spec(specification)
specification.loader.exec_module(experiment)


def photo(tmp_path, tagged):
    path = tmp_path / 'input.png'
    kwargs = {}
    if tagged:
        kwargs['icc_profile'] = (Path(__file__).parent / 'fixtures/displayP3.icc').read_bytes()
    Image.new('RGB', (80, 120), (184, 145, 122)).save(path, **kwargs)
    face = {'boxXyxy': [10, 20, 70, 100], 'landmarks': [[25, 40], [55, 40]], 'score': .99}
    working, record = readImage(path, imageSize=64, cropPolicy='faceSquare', faceDetector=lambda _: face)
    return path, working, record


def testLegacyUntaggedReplayRetainsPixelsAndCoordinates(tmp_path):
    path, expected, record = photo(tmp_path, False)
    record.pop('colorManagement')
    actual, _ = experiment.replayComparisonPhoto(path, record)
    np.testing.assert_array_equal(actual, expected)


def testLegacyTaggedReplayFailsInsteadOfMixingIccCorrectedPhotosWithOldOd(tmp_path):
    path, _, record = photo(tmp_path, True)
    record.pop('colorManagement')
    before = path.read_bytes()
    with pytest.raises(ValueError, match='predates ICC color provenance'):
        experiment.replayComparisonPhoto(path, record)
    assert path.read_bytes() == before


def testManagedReplayAcceptsSameColorProvenanceAndRejectsChangedPolicy(tmp_path):
    path, expected, record = photo(tmp_path, True)
    actual, _ = experiment.replayComparisonPhoto(path, record)
    np.testing.assert_array_equal(actual, expected)
    changed = copy.deepcopy(record)
    changed['colorManagement']['conversion'] = 'assumeSrgb'
    with pytest.raises(ValueError, match='Color preprocessing replay differs'):
        experiment.replayComparisonPhoto(path, changed)


def testReplayStillRejectsSourceOrCoordinateChanges(tmp_path):
    path, _, record = photo(tmp_path, False)
    changed = copy.deepcopy(record)
    changed['cropBoxXyxy'][0] += 1
    with pytest.raises(ValueError, match='Preprocessing replay differs'):
        experiment.replayComparisonPhoto(path, changed)
    Image.new('RGB', (80, 120), (20, 30, 40)).save(path)
    with pytest.raises(ValueError, match='Original photo hash differs'):
        experiment.replayComparisonPhoto(path, record)
