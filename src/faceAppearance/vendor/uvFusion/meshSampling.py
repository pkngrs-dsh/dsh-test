"""Vendored UVFaceFusion computation; upstream naming is intentionally retained.

Source: https://github.com/grignarder/UVFaceFusion (local commit 3de1c9a).
License: GPL-3.0, included in this directory. See SOURCES.md for extraction notes.
"""

import torch


import torch

def sample_uv_pointmap_masked(
    point_map_uv,   # (B,H,W,3) or (H,W,3)
    mask_uv,        # (B,H,W)   or (H,W), bool/float
    vertex_uv,      # (B,N,2)   or (N,2), in [0,1]
    eps=1e-8,
    align_corners=True,
    flip_v=False,
    valid_thresh=1e-4,
):
    """
    Masked bilinear sampling from UV-space point map to mesh vertices.

    Args:
        point_map_uv: UV-space point map, shape (B,H,W,3) or (H,W,3)
        mask_uv: UV-space valid mask, shape (B,H,W) or (H,W)
        vertex_uv: vertex UV coordinates in [0,1], shape (B,N,2) or (N,2)
        eps: numerical epsilon
        align_corners: keep True to match uv*(S-1) convention
        flip_v: whether to flip v -> 1-v before sampling
        valid_thresh: threshold on accumulated valid weight to mark a vertex valid

    Returns:
        vertices:     (B,N,3)
        vertex_valid: (B,N) float in [0,1]
    """
    squeeze_batch = False

    if point_map_uv.ndim == 3:
        point_map_uv = point_map_uv.unsqueeze(0)   # (1,H,W,3)
        squeeze_batch = True
    if mask_uv.ndim == 2:
        mask_uv = mask_uv.unsqueeze(0)             # (1,H,W)
    if vertex_uv.ndim == 2:
        vertex_uv = vertex_uv.unsqueeze(0)         # (1,N,2)

    B, H, W, C = point_map_uv.shape
    assert C == 3, f"point_map_uv last dim must be 3, got {C}"
    assert mask_uv.shape == (B, H, W), f"mask_uv shape mismatch: {mask_uv.shape}"
    assert vertex_uv.shape[0] == B, f"vertex_uv batch mismatch: {vertex_uv.shape}"

    device = point_map_uv.device
    dtype = point_map_uv.dtype
    N = vertex_uv.shape[1]

    uv = vertex_uv.to(device=device, dtype=dtype).clone()
    if flip_v:
        uv[..., 1] = 1.0 - uv[..., 1]

    uv = uv.clamp(0.0, 1.0)

    if align_corners:
        x = uv[..., 0] * (W - 1)
        y = uv[..., 1] * (H - 1)
    else:
        # Usually you do not want this branch for your pipeline,
        # but included for completeness.
        x = uv[..., 0] * W - 0.5
        y = uv[..., 1] * H - 0.5

    x0 = torch.floor(x).long().clamp(0, W - 1)
    y0 = torch.floor(y).long().clamp(0, H - 1)
    x1 = (x0 + 1).clamp(0, W - 1)
    y1 = (y0 + 1).clamp(0, H - 1)

    wx1 = x - x0.to(dtype)
    wy1 = y - y0.to(dtype)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    w00 = wx0 * wy0
    w01 = wx0 * wy1
    w10 = wx1 * wy0
    w11 = wx1 * wy1

    # flatten UV map for gather
    point_flat = point_map_uv.view(B, H * W, 3)          # (B,HW,3)
    mask_flat = mask_uv.to(dtype).view(B, H * W)         # (B,HW)

    idx00 = y0 * W + x0   # (B,N)
    idx01 = y1 * W + x0
    idx10 = y0 * W + x1
    idx11 = y1 * W + x1

    def gather_points_and_mask(idx):
        idx_exp_p = idx.unsqueeze(-1).expand(-1, -1, 3)  # (B,N,3)
        p = torch.gather(point_flat, 1, idx_exp_p)       # (B,N,3)
        m = torch.gather(mask_flat, 1, idx)              # (B,N)
        return p, m

    p00, m00 = gather_points_and_mask(idx00)
    p01, m01 = gather_points_and_mask(idx01)
    p10, m10 = gather_points_and_mask(idx10)
    p11, m11 = gather_points_and_mask(idx11)

    w00m = w00 * m00
    w01m = w01 * m01
    w10m = w10 * m10
    w11m = w11 * m11

    num = (
        w00m.unsqueeze(-1) * p00 +
        w01m.unsqueeze(-1) * p01 +
        w10m.unsqueeze(-1) * p10 +
        w11m.unsqueeze(-1) * p11
    )                                                     # (B,N,3)

    den = (w00m + w01m + w10m + w11m).unsqueeze(-1)       # (B,N,1)

    vertices = num / (den + eps)
    vertex_valid = den.squeeze(-1)                        # (B,N), soft confidence

    # For numerically bad cases with no valid support, set output to zero.
    vertices = torch.where(
        (vertex_valid > valid_thresh).unsqueeze(-1),
        vertices,
        torch.zeros_like(vertices)
    )

    if squeeze_batch:
        vertices = vertices[0]
        vertex_valid = vertex_valid[0]

    return vertices, vertex_valid

import torch


def sample_uv_pointmap_local_masked(
    point_map_uv,   # (B,H,W,3) or (H,W,3)
    mask_uv,        # (B,H,W)   or (H,W)
    vertex_uv,      # (B,N,2)   or (N,2), in [0,1]
    radius=2,       # local window radius; 2 => 5x5
    sigma=1.0,      # gaussian distance weight in pixel space
    eps=1e-8,
    align_corners=True,
    flip_v=False,
    valid_thresh=1e-4,
    min_valid_neighbors=1,
):
    """
    Robust local masked sampling from UV-space point map to vertices.

    Compared with 2x2 bilinear sampling, this method searches a local window
    around each vertex UV, uses only valid texels, and performs distance-weighted
    averaging. This is much more stable for partial visibility, seams, and holes.

    Args:
        point_map_uv: (B,H,W,3) or (H,W,3)
        mask_uv: (B,H,W) or (H,W), bool or float
        vertex_uv: (B,N,2) or (N,2), UV in [0,1]
        radius: local window radius in pixels
        sigma: Gaussian weighting sigma in pixel units
        eps: numerical stability
        align_corners: keep True to match uv*(S-1) convention
        flip_v: whether to use v -> 1-v before sampling
        valid_thresh: final threshold for vertex_valid
        min_valid_neighbors: minimum number of valid texels required

    Returns:
        vertices:     (B,N,3) or (N,3)
        vertex_valid: (B,N)   or (N,), soft confidence in [0,+)
        valid_count:  (B,N)   or (N,), number of valid texels used
    """
    squeeze_batch = False

    if point_map_uv.ndim == 3:
        point_map_uv = point_map_uv.unsqueeze(0)
        squeeze_batch = True
    if mask_uv.ndim == 2:
        mask_uv = mask_uv.unsqueeze(0)
    if vertex_uv.ndim == 2:
        vertex_uv = vertex_uv.unsqueeze(0)

    B, H, W, C = point_map_uv.shape
    assert C == 3, f"point_map_uv last dim must be 3, got {C}"
    assert mask_uv.shape == (B, H, W), f"mask_uv shape mismatch: {mask_uv.shape}"
    assert vertex_uv.shape[0] == B, f"vertex_uv batch mismatch: {vertex_uv.shape}"

    device = point_map_uv.device
    dtype = point_map_uv.dtype
    N = vertex_uv.shape[1]

    uv = vertex_uv.to(device=device, dtype=dtype).clone()
    if flip_v:
        uv[..., 1] = 1.0 - uv[..., 1]
    uv = uv.clamp(0.0, 1.0)

    if align_corners:
        x = uv[..., 0] * (W - 1)   # (B,N)
        y = uv[..., 1] * (H - 1)   # (B,N)
    else:
        x = uv[..., 0] * W - 0.5
        y = uv[..., 1] * H - 0.5

    # nearest integer pixel center as local search center
    xc = torch.round(x).long()   # (B,N)
    yc = torch.round(y).long()   # (B,N)

    point_flat = point_map_uv.view(B, H * W, 3)              # (B,HW,3)
    mask_flat = mask_uv.to(dtype).view(B, H * W)             # (B,HW)

    # accumulators
    num = torch.zeros((B, N, 3), device=device, dtype=dtype)
    den = torch.zeros((B, N), device=device, dtype=dtype)
    valid_count = torch.zeros((B, N), device=device, dtype=dtype)

    # iterate over local window offsets
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xi = (xc + dx).clamp(0, W - 1)   # (B,N)
            yi = (yc + dy).clamp(0, H - 1)   # (B,N)
            idx = yi * W + xi                # (B,N)

            idx_exp_p = idx.unsqueeze(-1).expand(-1, -1, 3)   # (B,N,3)
            p = torch.gather(point_flat, 1, idx_exp_p)        # (B,N,3)
            m = torch.gather(mask_flat, 1, idx)               # (B,N)

            # true distance to the continuous UV location, in pixel space
            ddx = x - xi.to(dtype)
            ddy = y - yi.to(dtype)
            dist2 = ddx * ddx + ddy * ddy

            w = torch.exp(-dist2 / (2.0 * sigma * sigma))     # (B,N)
            wm = w * m

            num = num + wm.unsqueeze(-1) * p
            den = den + wm
            valid_count = valid_count + (m > 0.5).to(dtype)

    vertices = num / (den.unsqueeze(-1) + eps)

    # only keep vertices with enough support
    good = (den > valid_thresh) & (valid_count >= float(min_valid_neighbors))
    vertices = torch.where(good.unsqueeze(-1), vertices, torch.zeros_like(vertices))
    vertex_valid = den

    if squeeze_batch:
        vertices = vertices[0]
        vertex_valid = vertex_valid[0]
        valid_count = valid_count[0]

    return vertices, vertex_valid, valid_count