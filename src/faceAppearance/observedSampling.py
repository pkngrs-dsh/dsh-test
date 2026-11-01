"""Coverage-normalized sampling of already observed atlas colors.

This removes the numerical darkening caused by interpolating valid texels with
zero-filled invalid neighbors. It neither reconstructs missing colors nor moves
UVs. Consumers must keep geometry visibility and existing evaluation masks
separate; accepting a partial footprint is a diagnostic sampling policy.
"""

from __future__ import annotations

import numpy as np


def normalizeObservedSamples(sampledPremultipliedColors: np.ndarray,
                             sampledCoverage: np.ndarray, *,
                             minimumCoverage: float = .5) -> dict:
    """Divide sampled premultiplied linear RGB by the exact sampled coverage.

    Inputs must use the same nonnegative interpolation weights. Because RGB is
    in [0,1], each premultiplied channel must be in [0, coverage]. The threshold
    is inclusive, but zero coverage is always invalid. Invalid output RGB is
    zero; ``partialMask`` identifies *accepted* footprints with coverage < 1.
    ``sampleCoverage`` retains every input weight, including rejected samples.

    No epsilon is added to the divisor and no color is synthesized where the
    observation support is absent. This changes neither the input arrays nor
    any caller-owned evaluation mask.
    """
    colors = np.asarray(sampledPremultipliedColors)
    coverage = np.asarray(sampledCoverage)
    if (colors.dtype.kind not in "fiu" or coverage.dtype.kind not in "fiu"
            or colors.ndim != 3 or colors.shape[-1] != 3
            or min(colors.shape[:2], default=0) < 1 or coverage.shape != colors.shape[:2]
            or not np.isfinite(colors).all() or not np.isfinite(coverage).all()
            or np.any(coverage < 0) or np.any(coverage > 1)
            or np.any(colors < 0) or np.any(colors > coverage[..., None])):
        raise ValueError("Expected finite HWC premultiplied RGB in [0,coverage] and HW coverage in [0,1]")
    if (isinstance(minimumCoverage, (bool, np.bool_))
            or not isinstance(minimumCoverage, (int, float, np.integer, np.floating))
            or not np.isfinite(minimumCoverage) or not 0 < minimumCoverage <= 1):
        raise ValueError("minimumCoverage must be finite in (0,1]")
    dtype = np.result_type(colors.dtype, coverage.dtype, np.float32)
    coverage = coverage.astype(dtype, copy=True)
    colors = colors.astype(dtype, copy=False)
    valid = (coverage > 0) & (coverage >= minimumCoverage)
    normalized = np.zeros(colors.shape, dtype=dtype)
    np.divide(colors, coverage[..., None], out=normalized, where=valid[..., None])
    return {"linearColors": normalized, "validMask": valid,
            "partialMask": valid & (coverage < 1), "sampleCoverage": coverage}


def _toNumpy(values) -> np.ndarray:
    # Duck typing keeps this CPU helper independent of Torch/PyTorch3D imports.
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values)


def sampleObservedAtlas(renderer, linearColors: np.ndarray,
                        coverageMask: np.ndarray, *, minimumCoverage: float = .5) -> dict:
    """Sample color*support and support through one renderer or callable sampler.

    ``renderer.sampleAtlas`` must use identical nonnegative interpolation for
    both calls. A bare callable with that contract is also accepted for CPU
    testing or other backends. Outputs may be NumPy arrays or tensor objects
    providing detach/cpu/numpy. Coverage samples may have shape HW or HW1.
    The returned masks describe atlas support, not mesh visibility.
    """
    colors = np.asarray(linearColors)
    coverage = np.asarray(coverageMask)
    if (colors.dtype.kind not in "fiu" or colors.ndim != 3 or colors.shape[-1] != 3
            or min(colors.shape[:2], default=0) < 1 or coverage.shape != colors.shape[:2]
            or coverage.dtype != np.bool_ or not np.isfinite(colors).all()
            or np.any(colors < 0) or np.any(colors > 1)):
        raise ValueError("Expected finite HWC linear RGB in [0,1] and matching boolean HW coverageMask")
    if (isinstance(minimumCoverage, (bool, np.bool_))
            or not isinstance(minimumCoverage, (int, float, np.integer, np.floating))
            or not np.isfinite(minimumCoverage) or not 0 < minimumCoverage <= 1):
        raise ValueError("minimumCoverage must be finite in (0,1]")
    sampler = getattr(renderer, "sampleAtlas", renderer)
    if not callable(sampler):
        raise ValueError("renderer must provide callable sampleAtlas or be a callable sampler")
    # Copy before passing to a foreign backend, so even an in-place sampler
    # cannot mutate the user's material or support arrays.
    dtype = np.result_type(colors.dtype, np.float32)
    premultiplied = colors.astype(dtype) * coverage[..., None]
    sampledColors = _toNumpy(sampler(premultiplied))
    sampledCoverage = _toNumpy(sampler(coverage.astype(dtype)))
    if sampledCoverage.ndim == 3 and sampledCoverage.shape[-1] == 1:
        sampledCoverage = sampledCoverage[..., 0]
    return normalizeObservedSamples(sampledColors, sampledCoverage, minimumCoverage=minimumCoverage)
