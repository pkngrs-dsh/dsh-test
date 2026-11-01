"""Vendored UVFaceFusion computation; upstream naming is intentionally retained.

Source: https://github.com/grignarder/UVFaceFusion (local commit 3de1c9a).
License: GPL-3.0, included in this directory. See SOURCES.md for extraction notes.
"""

import numpy as np
import torch
import cv2

def umeyama(P, Q):
    """
    Estimate a scale, rotation, and translation between two sets of points P and Q which have correspondances.
    Args:
        P (np.ndarray): Source points of shape (n, dim).
        Q (np.ndarray): Target points of shape (n, dim).
    Returns:
        c (float): Scale factor.
        R (np.ndarray): Rotation matrix of shape (dim, dim).
        t (np.ndarray): Translation vector of shape (dim,).
    """
    assert P.shape == Q.shape
    n, dim = P.shape

    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)

    C = np.dot(np.transpose(centeredP), centeredQ) / n

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    R = np.dot(V, W)

    varP = np.var(P, axis=0).sum()
    c = 1/varP * np.sum(S) # scale factor

    t = Q.mean(axis=0) - P.mean(axis=0).dot(c*R)

    return c, R, t

@torch.no_grad()
def bake_pointmap_to_uv_nvdiffrast(
    V,
    Fidx,
    UV,
    S=1024,
    device=None,
    ctx = None
):
    import nvdiffrast.torch as dr  # Lazy import permits CPU-only adapter tests.
    if device is None:
        device = V.device

    V = V.to(device=device, dtype=torch.float32)
    UV = UV.to(device=device, dtype=torch.float32)
    Fidx = Fidx.to(device=device)

    M = Fidx.shape[0]
    corners = Fidx.reshape(-1).long()
    uv_c = UV[corners]
    pos_c = V[corners]
    F_new = torch.arange(3 * M, device=device).reshape(M, 3).int()

    pos_clip = torch.zeros((3 * M, 4), device=device, dtype=torch.float32)
    pos_clip[:, 0:2] = uv_c * 2.0 - 1.0
    pos_clip[:, 2] = 0.0
    pos_clip[:, 3] = 1.0

    rast, _ = dr.rasterize(ctx, pos_clip[None, ...], F_new, (S, S))

    tri_id = rast[0, ..., 3].to(torch.int64) - 1
    mask_uv = tri_id >= 0

    P_uv, _ = dr.interpolate(pos_c[None, ...], rast, F_new)
    P_uv = P_uv[0]

    P_uv = torch.where(mask_uv.unsqueeze(-1), P_uv, torch.zeros_like(P_uv))
    return P_uv, mask_uv, tri_id

def masked_gaussian_blur(img, mask, ksize=3, sigma=0, keep_outside=False, eps=1e-8):
    """
    只在 mask 区域内做 Gaussian blur，不让 mask 外的垃圾像素参与进来。

    Args:
        img:   (H, W) or (H, W, C), numpy array
        mask:  (H, W), bool / 0-1 / 0-255 都可以
        ksize: Gaussian kernel size, int or (kH, kW)
        sigma: Gaussian sigma, same as cv2.GaussianBlur
        keep_outside: 
            - False: 输出在 mask 外设为 0
            - True:  输出在 mask 外保留原图
        eps: 数值稳定项

    Returns:
        out: same shape as img
    """
    img = img.astype(np.float32)
    mask = mask.astype(np.float32)
    mask = (mask > 0).astype(np.float32)

    if isinstance(ksize, int):
        ksize = (ksize, ksize)

    if img.ndim == 2:
        weighted = img * mask
        num = cv2.GaussianBlur(weighted, ksize, sigma)
        den = cv2.GaussianBlur(mask,     ksize, sigma)
        out = num / (den + eps)

        if keep_outside:
            out = np.where(mask > 0, out, img)
        else:
            out = np.where(mask > 0, out, 0.0)

    elif img.ndim == 3:
        mask_3 = mask[..., None]  # (H, W, 1)
        weighted = img * mask_3

        num = cv2.GaussianBlur(weighted, ksize, sigma)
        den = cv2.GaussianBlur(mask,     ksize, sigma)[..., None]
        out = num / (den + eps)

        if keep_outside:
            out = np.where(mask_3 > 0, out, img)
        else:
            out = np.where(mask_3 > 0, out, 0.0)
    else:
        raise ValueError(f"Unsupported img shape: {img.shape}")

    return out
