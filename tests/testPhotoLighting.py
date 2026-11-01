import json

import numpy as np
import pytest

from faceAppearance.photoLighting import fitPhotoShLighting


def syntheticPhoto(size=64):
    """Independent polynomial illumination, with high-frequency albedo detail."""
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    z = .85 * y
    phi = np.pi * x
    radial = np.sqrt(1 - z * z)
    normals = np.stack([radial * np.cos(phi), radial * np.sin(phi), z], axis=-1)
    nx, ny, nz = np.moveaxis(normals, -1, 0)
    shading = (1 + .16 * nx - .12 * ny + .10 * nz + .08 * nx * ny
               + .05 * (3 * nz ** 2 - 1))[..., None] * np.array([1.1, .95, .8])
    detail = 1 + .2 * (2 * (np.indices((size, size)).sum(axis=0) % 2) - 1)
    albedo = detail[..., None] * np.array([.35, .24, .18])
    return albedo, albedo * shading, normals, np.ones((size, size), bool), shading


def testKnownLightingRecoversWithoutAlbedoChangesOrHiddenGamma():
    albedo, observed, normals, mask, shading = syntheticPhoto()
    before = [value.copy() for value in (albedo, observed, normals, mask)]
    result = fitPhotoShLighting(albedo, observed, normals, mask, ridge=0)
    np.testing.assert_allclose(result["shading"], shading, atol=1e-11)
    np.testing.assert_allclose(result["renderedLinear"], observed, atol=1e-11)
    np.testing.assert_array_equal(result["renderedLinear"], albedo * result["shading"])
    for original, value in zip(before, (albedo, observed, normals, mask)):
        np.testing.assert_array_equal(original, value)
    assert result["diagnostics"]["metricScope"] == "inSampleLightingFit"
    assert result["diagnostics"]["photometricParameterCount"] == 27
    json.dumps(result["diagnostics"], allow_nan=False)


def testDefaultRegularizationStillRecoversKnownShading():
    albedo, observed, normals, mask, shading = syntheticPhoto()
    result = fitPhotoShLighting(albedo, observed, normals, mask)
    assert np.sqrt(np.mean((result["shading"] - shading) ** 2)) < .006
    assert result["diagnostics"]["evaluationRmseLinear"] < .002
    assert result["validMask"].all()


def testFixedAlbedoScaleIsAbsorbedByLightingNotAnImplicitColorTransform():
    albedo, observed, normals, mask, _ = syntheticPhoto()
    first = fitPhotoShLighting(albedo, observed, normals, mask, ridge=0)
    second = fitPhotoShLighting(albedo * .5, observed, normals, mask, ridge=0)
    np.testing.assert_allclose(second["coefficients"], first["coefficients"] * 2, atol=1e-10)
    np.testing.assert_allclose(second["renderedLinear"], first["renderedLinear"], atol=1e-10)


def testOutsideMaskPixelsCannotInfluenceFitAndMissingNormalsAreExcluded():
    albedo, observed, normals, mask, _ = syntheticPhoto()
    mask[:16] = False
    normals[-2:] = 0
    first = fitPhotoShLighting(albedo, observed, normals, mask)
    observed[:16] = np.random.default_rng(2).uniform(size=observed[:16].shape)
    albedo[:16] = .01
    second = fitPhotoShLighting(albedo, observed, normals, mask)
    np.testing.assert_array_equal(first["coefficients"], second["coefficients"])
    assert not first["validMask"][-2:].any()
    assert first["diagnostics"]["invalidNormalPixelCount"] == 128


def testFitExclusionsDoNotSilentlyRemoveEvaluationPixels():
    albedo, observed, normals, mask, _ = syntheticPhoto()
    observed[:3] = 1
    observed[-3:] = .001
    albedo[20:22] = 0
    result = fitPhotoShLighting(albedo, observed, normals, mask)
    assert not result["fitMask"][:3].any()
    assert not result["fitMask"][-3:].any()
    assert not result["fitMask"][20:22].any()
    assert result["validMask"].all()
    assert result["diagnostics"]["evaluationPixelCount"] == 4096
    assert result["diagnostics"]["saturatedExcludedFitPixelCount"] == 192
    assert result["diagnostics"]["darkExcludedFitPixelCount"] == 192


def testLocalizedUnclippedColorOutlierDoesNotSetGlobalLighting():
    albedo, observed, normals, mask, shading = syntheticPhoto()
    observed[20:27, 30:37] = [.85, .08, .7]
    result = fitPhotoShLighting(albedo, observed, normals, mask)
    cleanMask = mask.copy()
    cleanMask[17:30, 27:40] = False
    error = np.sqrt(np.mean((result["renderedLinear"][cleanMask] - (albedo * shading)[cleanMask]) ** 2))
    assert error < .002
    assert max(result["diagnostics"]["heavilyDownweightedSampleFractionPerChannel"]) > 0


def testExtremeIlluminationUsesCoefficientBlendAndRecordsFailureToFit():
    albedo = np.full((24, 24, 3), .01)
    observed = np.full_like(albedo, .6)
    normals = np.zeros_like(albedo)
    normals[..., 2] = 1
    result = fitPhotoShLighting(albedo, observed, normals, np.ones((24, 24), bool))
    diagnostics = result["diagnostics"]
    assert diagnostics["unsafeFittedShadingPixelCountBeforeBlend"] == 576
    assert 0 < diagnostics["coefficientBlendWithNeutral"] < 1
    assert diagnostics["unsafeFinalShadingPixelCount"] == 0
    assert result["shading"].max() <= 8
    assert diagnostics["evaluationRmseLinear"] > .5


def testRenderIsUnclippedEvenWhenSomePhotoPixelsAreSaturated():
    albedo = np.full((24, 24, 3), .2)
    observed = np.full_like(albedo, .4)
    normals = np.zeros_like(albedo)
    normals[..., 2] = 1
    albedo[10:13, 10:13] = 1
    observed[10:13, 10:13] = 1
    result = fitPhotoShLighting(albedo, observed, normals, np.ones((24, 24), bool))
    assert result["renderedLinear"][11, 11].min() > 1.9
    assert result["diagnostics"]["renderedOutOfUnitRangePixelCount"] == 9
    assert result["validMask"][11, 11]
    assert not result["fitMask"][11, 11]


def testUnsafeInitializationIsMadeSafeAndRecordedBeforeUseAsPrior():
    albedo, observed, normals, mask, _ = syntheticPhoto()
    initial = np.zeros((9, 3))
    initial[0] = -100
    result = fitPhotoShLighting(albedo, observed, normals, mask, initialCoefficients=initial)
    assert result["diagnostics"]["initialUnsafeShadingPixelCount"] == mask.sum()
    assert result["diagnostics"]["initialCoefficientBlendWithNeutral"] < 1
    assert result["diagnostics"]["initialCoefficientsProvided"]
    assert np.isfinite(result["coefficients"]).all()


def testBoundedSamplingIsDeterministic():
    albedo, observed, normals, mask, _ = syntheticPhoto()
    first = fitPhotoShLighting(albedo, observed, normals, mask, maxSamples=83)
    second = fitPhotoShLighting(albedo, observed, normals, mask, maxSamples=83)
    np.testing.assert_array_equal(first["coefficients"], second["coefficients"])
    assert first["diagnostics"]["sampleCount"] == 83
    assert first["diagnostics"]["evaluationRmseLinear"] < .002


@pytest.mark.parametrize("argument,value", [("maxSamples", 11), ("maxSamples", True),
                                           ("ridge", -1), ("ridge", float("nan")),
                                           ("initialCoefficients", np.zeros((3, 9)))])
def testInvalidOptionsFailExplicitly(argument, value):
    albedo, observed, normals, mask, _ = syntheticPhoto(size=16)
    with pytest.raises(ValueError):
        fitPhotoShLighting(albedo, observed, normals, mask, **{argument: value})


def testInvalidInputOrInsufficientEvidenceFailsExplicitly():
    albedo, observed, normals, mask, _ = syntheticPhoto(size=16)
    with pytest.raises(ValueError, match="12 usable"):
        fitPhotoShLighting(albedo, observed, normals, mask & False)
    with pytest.raises(ValueError, match="finite"):
        fitPhotoShLighting(albedo * np.nan, observed, normals, mask)
    with pytest.raises(ValueError, match="linear RGB"):
        fitPhotoShLighting(albedo, observed * 4, normals, mask)
    with pytest.raises(ValueError, match="zero/one"):
        fitPhotoShLighting(albedo, observed, normals, mask.astype(float) * .5)
