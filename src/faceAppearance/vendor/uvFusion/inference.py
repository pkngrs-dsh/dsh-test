"""Vendored UVFaceFusion computation; upstream naming is intentionally retained.

Source: https://github.com/grignarder/UVFaceFusion (local commit 3de1c9a).
License: GPL-3.0, included in this directory. See SOURCES.md for extraction notes.
"""

import numpy as np
import torch
from .meshSampling import sample_uv_pointmap_local_masked
from .utilities import bake_pointmap_to_uv_nvdiffrast, umeyama

@torch.no_grad()
def fuse_uv_single_view(
    uv_map,          # (H,W,2), in [0,1]
    point_map,       # (H,W,3)
    mask=None,       # (H,W), optional
    weight=None,     # (H,W), optional
    S=512,
    eps=1e-8,
    chunk=2_000_000,
):
    """
    Fuse one image-space uv/point_map pair into UV-space point map.

    Returns:
        uv_point_map: (S,S,3)
        wacc:         (S,S)
    """
    device = uv_map.device
    H, W, _ = uv_map.shape

    uv = uv_map.unsqueeze(0)          # (1,H,W,2)
    attr = point_map.unsqueeze(0)     # (1,H,W,3)

    w = torch.ones((1, H, W), device=device, dtype=torch.float32)
    if mask is not None:
        w = w * mask.unsqueeze(0).to(torch.float32)
    if weight is not None:
        w = w * weight.unsqueeze(0).to(torch.float32)

    uv = uv.clamp(1e-7, 1.0 - 1e-7)
    x = uv[..., 0] * (S - 1)
    y = uv[..., 1] * (S - 1)

    x0 = x.floor().long()
    y0 = y.floor().long()
    x1 = (x0 + 1).clamp(max=S - 1)
    y1 = (y0 + 1).clamp(max=S - 1)

    wx1 = x - x0.to(x.dtype)
    wy1 = y - y0.to(y.dtype)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1

    w00 = (wx0 * wy0) * w
    w01 = (wx0 * wy1) * w
    w10 = (wx1 * wy0) * w
    w11 = (wx1 * wy1) * w

    A = torch.zeros((S * S, 3), device=device, dtype=torch.float32)
    Wacc = torch.zeros((S * S,), device=device, dtype=torch.float32)

    def splat(idx, ww, vec):
        A.index_add_(0, idx, ww[:, None] * vec)
        Wacc.index_add_(0, idx, ww)

    attr_flat = attr.reshape(-1, 3).to(torch.float32)
    x0f, x1f = x0.reshape(-1), x1.reshape(-1)
    y0f, y1f = y0.reshape(-1), y1.reshape(-1)
    w00f, w01f = w00.reshape(-1), w01.reshape(-1)
    w10f, w11f = w10.reshape(-1), w11.reshape(-1)

    N = attr_flat.shape[0]
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        vec = attr_flat[s:e]

        i00 = y0f[s:e] * S + x0f[s:e]
        i01 = y1f[s:e] * S + x0f[s:e]
        i10 = y0f[s:e] * S + x1f[s:e]
        i11 = y1f[s:e] * S + x1f[s:e]

        splat(i00, w00f[s:e].float(), vec)
        splat(i01, w01f[s:e].float(), vec)
        splat(i10, w10f[s:e].float(), vec)
        splat(i11, w11f[s:e].float(), vec)

    uv_point_map = A / (Wacc[:, None] + eps)
    return uv_point_map.view(S, S, 3), Wacc.view(S, S)

@torch.no_grad()
def build_template_uv_valid_mask(
    canonical_vertices,   # (N,3), np or torch
    canonical_faces,      # (F,3), np or torch
    uvs_array,            # (N,2), np or torch
    uv_size=512,
    device="cuda",
    ctx=None,
):
    if not torch.is_tensor(canonical_vertices):
        canonical_vertices = torch.from_numpy(np.asarray(canonical_vertices)).float().to(device)
    else:
        canonical_vertices = canonical_vertices.float().to(device)

    if not torch.is_tensor(canonical_faces):
        canonical_faces = torch.from_numpy(np.asarray(canonical_faces)).long().to(device)
    else:
        canonical_faces = canonical_faces.long().to(device)

    if not torch.is_tensor(uvs_array):
        uvs_array = torch.from_numpy(np.asarray(uvs_array)).float().to(device)
    else:
        uvs_array = uvs_array.float().to(device)

    _, mask_uv, _ = bake_pointmap_to_uv_nvdiffrast(
        canonical_vertices,
        canonical_faces,
        uvs_array,
        S=uv_size,
        ctx=ctx,
    )
    return mask_uv

@torch.no_grad()
def canonicalize_partial_uv_point_map(
    pred_uv_point_map,         # (S,S,3), torch
    valid_mask,                # (S,S), bool/float, torch
    uvs_array,                 # (N,2), np or torch
    canonical_vertices,        # (N,3), np or torch
    radius=1,
    sigma=0.8,
    min_valid_neighbors=1,
    partial_mesh_valid_thresh=1.0,
    template_valid_mask=None,  # (S,S), np or torch, optional
):
    """
    Estimate a similarity transform from one partial UV point map to canonical space,
    using sampled partial mesh vertices, then transform the UV point map.

    Returns:
        pred_uv_point_map_canon: (S,S,3) torch
        valid_mask_canon:        (S,S) torch bool
        transform: dict with c,R,t,vertices_mask
    """
    device = pred_uv_point_map.device

    if not torch.is_tensor(uvs_array):
        uvs_array_t = torch.from_numpy(np.asarray(uvs_array)).float().to(device)
    else:
        uvs_array_t = uvs_array.float().to(device)

    if not torch.is_tensor(canonical_vertices):
        canonical_vertices_np = np.asarray(canonical_vertices).astype(np.float32)
    else:
        canonical_vertices_np = canonical_vertices.detach().cpu().numpy().astype(np.float32)

    zero_mask = torch.all(pred_uv_point_map == 0, dim=-1)
    valid_mask = torch.logical_and(valid_mask.bool(), ~zero_mask)

    sampled_vertices_pred, vertices_valid, _ = sample_uv_pointmap_local_masked(
        pred_uv_point_map.float(),
        valid_mask,
        uvs_array_t,
        radius=radius,
        sigma=sigma,
        min_valid_neighbors=min_valid_neighbors,
    )

    vertices_mask = (vertices_valid > partial_mesh_valid_thresh).detach().cpu().numpy()

    if vertices_mask.sum() < 10:
        c = 1.0
        R = np.eye(3, dtype=np.float32)
        t = np.zeros((3,), dtype=np.float32)
    else:
        c, R, t = umeyama(
            sampled_vertices_pred.detach().cpu().numpy()[vertices_mask],
            canonical_vertices_np[vertices_mask],
        )
        R = R.astype(np.float32)
        t = t.astype(np.float32)

    pred_uv_np = pred_uv_point_map.detach().cpu().numpy()
    pred_uv_np = c * (pred_uv_np @ R) + t

    valid_np = valid_mask.detach().cpu().numpy().astype(np.float32)
    pred_uv_np = pred_uv_np * valid_np[..., None]

    if template_valid_mask is not None:
        if torch.is_tensor(template_valid_mask):
            template_mask_np = template_valid_mask.detach().cpu().numpy().astype(np.float32)
        else:
            template_mask_np = np.asarray(template_valid_mask).astype(np.float32)

        pred_uv_np = pred_uv_np * template_mask_np[..., None]
        valid_np = valid_np * template_mask_np

    pred_uv_np = np.clip(pred_uv_np, -1, 1)

    pred_uv_point_map_canon = torch.from_numpy(pred_uv_np).float().to(device)
    valid_mask_canon = torch.from_numpy(valid_np > 0.5).to(device)

    transform = {
        "c": c,
        "R": R,
        "t": t,
        "vertices_mask": vertices_mask,
    }
    return pred_uv_point_map_canon, valid_mask_canon, transform

@torch.no_grad()
def infer_uv_point_map_with_model(
    model,
    uvs,                  # (V,H,W,2), np or torch
    masks,                # (V,H,W),   np or torch
    point_maps,           # (V,H,W,3), np or torch
    uvs_array,            # (N,2),     np or torch
    canonical_vertices,   # (N,3),     np or torch
    template_valid_mask=None,   # (S,S), np or torch, optional
    device="cuda",
    uv_size=512,
    num_views=None,
    view_indices=None,
    random_sample_views=False,
    partial_mesh_radius=1,
    partial_mesh_sigma=0.8,
    partial_mesh_min_valid_neighbors=1,
    partial_mesh_valid_thresh=1.0,
    mask_threshold=0.5,
    capture_input_fusion=False,
):
    """
    Use a trained model to infer one full UV-space point map.
    Compatible with MultiViewFusionNet / MultiViewAttentionUNetLite.

    Returns:
        result: dict with
            pred_uv_point_map:    (S,S,3) numpy
            pred_valid_mask:      (S,S)   numpy
            input_support_mask:   (S,S)   numpy
            input_uv_point_maps:  (K,S,S,3) numpy
            input_valid_masks:    (K,S,S) numpy
            selected_view_indices: list[int]
            transforms: list[dict]
            model_input_mean / model_input_coverage: optional, evaluated with
                the model's own compute_input_fusion and forward mask rule.
                No inference weights or network outputs are changed.
    """
    model.eval()

    # to tensor
    if not torch.is_tensor(uvs):
        uvs = torch.from_numpy(np.asarray(uvs)).float()
    if not torch.is_tensor(masks):
        masks = torch.from_numpy(np.asarray(masks)).float()
    if not torch.is_tensor(point_maps):
        point_maps = torch.from_numpy(np.asarray(point_maps)).float()

    total_views = uvs.shape[0]

    # decide which views to use
    if view_indices is not None:
        selected = list(view_indices)
    else:
        if num_views is None:
            selected = list(range(total_views))
        else:
            k = min(int(num_views), total_views)
            if random_sample_views:
                selected = np.random.choice(total_views, size=k, replace=False).tolist()
            else:
                selected = list(range(k))

    pred_uv_list = []
    valid_mask_list = []
    transforms = []

    for vid in selected:
        uv_map = uvs[vid].to(device)             # (H,W,2)
        mask_map = masks[vid].to(device)         # (H,W)
        point_map = point_maps[vid].to(device)   # (H,W,3)

        # 和训练保持一致：mask 先二值化
        mask_map_bin = (mask_map > mask_threshold).float()

        # 1) image-space -> UV-space
        pred_uv_point_map, pred_w = fuse_uv_single_view(
            uv_map=uv_map,
            point_map=point_map,
            mask=mask_map_bin,
            weight=None,
            S=uv_size,
        )
        valid_mask = pred_w > 1e-1

        # 2) canonicalize this partial UV point map
        pred_uv_point_map_canon, valid_mask_canon, transform = canonicalize_partial_uv_point_map(
            pred_uv_point_map=pred_uv_point_map,
            valid_mask=valid_mask,
            uvs_array=uvs_array,
            canonical_vertices=canonical_vertices,
            radius=partial_mesh_radius,
            sigma=partial_mesh_sigma,
            min_valid_neighbors=partial_mesh_min_valid_neighbors,
            partial_mesh_valid_thresh=partial_mesh_valid_thresh,
            template_valid_mask=template_valid_mask,
        )

        pred_uv_list.append(pred_uv_point_map_canon)
        valid_mask_list.append(valid_mask_canon)
        transforms.append(transform)

    # stack and run model
    pred_uv_tensor = torch.stack(pred_uv_list, dim=0)             # (K,S,S,3)
    valid_mask_tensor = torch.stack(valid_mask_list, dim=0)       # (K,S,S)

    pred_t = pred_uv_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(device)          # (1,K,3,S,S)
    mask_t = valid_mask_tensor.unsqueeze(1).unsqueeze(0).float().to(device)      # (1,K,1,S,S)

    out_t = model(pred_t, mask_t)   # (1,3,S,S)
    pred_uv_point_map_full = out_t[0].permute(1, 2, 0).detach().cpu().numpy()

    diagnostic_inputs = {}
    if capture_input_fusion:
        if not hasattr(model, "compute_input_fusion"):
            raise ValueError("Diagnostic input mean requires model.compute_input_fusion")
        diagnostic_mask = mask_t.to(dtype=pred_t.dtype).clamp(0.0, 1.0)
        diagnostic_mean, diagnostic_coverage = model.compute_input_fusion(
            pred_t * diagnostic_mask, diagnostic_mask,
        )
        diagnostic_inputs = {
            "model_input_mean": diagnostic_mean[0].permute(1, 2, 0).detach().cpu().numpy(),
            "model_input_coverage": diagnostic_coverage[0, 0].detach().cpu().numpy(),
        }

    # 输入覆盖区域
    input_support_mask = valid_mask_tensor.any(dim=0).detach().cpu().numpy()

    # 输出 valid 区域：如果有 template_valid_mask，就优先用它
    if template_valid_mask is not None:
        if torch.is_tensor(template_valid_mask):
            pred_valid_mask = template_valid_mask.detach().cpu().numpy().astype(bool)
        else:
            pred_valid_mask = np.asarray(template_valid_mask).astype(bool)
    else:
        pred_valid_mask = input_support_mask

    return {
        "pred_uv_point_map": pred_uv_point_map_full,                       # (S,S,3)
        "pred_valid_mask": pred_valid_mask,                                # (S,S)
        "input_support_mask": input_support_mask,                          # (S,S)
        "input_uv_point_maps": pred_uv_tensor.detach().cpu().numpy(),      # (K,S,S,3)
        "input_valid_masks": valid_mask_tensor.detach().cpu().numpy(),     # (K,S,S)
        "selected_view_indices": selected,
        "transforms": transforms,
        **diagnostic_inputs,
    }
