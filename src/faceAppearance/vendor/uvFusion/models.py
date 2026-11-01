"""Vendored UVFaceFusion computation; upstream naming is intentionally retained.

Source: https://github.com/grignarder/UVFaceFusion (local commit 3de1c9a).
License: GPL-3.0, included in this directory. See SOURCES.md for extraction notes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic blocks
# ============================================================

def make_norm(norm_type, num_channels, num_groups=8):
    if norm_type == "bn":
        return nn.BatchNorm2d(num_channels)

    if norm_type == "gn":
        num_groups = min(num_groups, num_channels)
        while num_groups > 1 and num_channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups, num_channels)

    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ResidualConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        norm_type="gn",
        num_groups=8,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = make_norm(norm_type, out_channels, num_groups)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = make_norm(norm_type, out_channels, num_groups)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False,
                ),
                make_norm(norm_type, out_channels, num_groups),
            )
        else:
            self.shortcut = nn.Identity()

        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + identity
        x = self.act2(x)
        return x


class DownsampleBlock(nn.Module):
    """
    H,W -> H/2,W/2
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        norm_type="gn",
        num_groups=8,
    ):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            make_norm(norm_type, out_channels, num_groups),
            nn.SiLU(inplace=True),
        )

        self.block = nn.Sequential(
            ResidualConvBlock(
                out_channels,
                out_channels,
                norm_type,
                num_groups,
            ),
            ResidualConvBlock(
                out_channels,
                out_channels,
                norm_type,
                num_groups,
            ),
        )

    def forward(self, x):
        x = self.down(x)
        x = self.block(x)
        return x


class UpsampleBlock(nn.Module):
    """
    使用 bilinear upsample，避免 ConvTranspose2d 的棋盘伪影。
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_type="gn",
        num_groups=8,
    ):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_norm(norm_type, out_channels, num_groups),
            nn.SiLU(inplace=True),
        )

        self.block = nn.Sequential(
            ResidualConvBlock(
                out_channels + skip_channels,
                out_channels,
                norm_type,
                num_groups,
            ),
            ResidualConvBlock(
                out_channels,
                out_channels,
                norm_type,
                num_groups,
            ),
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = self.up(x)

        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


# ============================================================
# Per-view encoder
# ============================================================

class PerViewEncoder(nn.Module):
    def __init__(
        self,
        in_channels=4,
        base_channels=32,
        norm_type="gn",
        num_groups=8,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                c1,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            make_norm(norm_type, c1, num_groups),
            nn.SiLU(inplace=True),
            ResidualConvBlock(c1, c1, norm_type, num_groups),
            ResidualConvBlock(c1, c1, norm_type, num_groups),
        )

        self.down1 = DownsampleBlock(
            c1, c2, norm_type, num_groups
        )
        self.down2 = DownsampleBlock(
            c2, c3, norm_type, num_groups
        )
        self.down3 = DownsampleBlock(
            c3, c4, norm_type, num_groups
        )
        self.down4 = DownsampleBlock(
            c4, c5, norm_type, num_groups
        )

    def forward(self, x):
        """
        x: (B*V, C, H, W)

        returns:
            f1: H
            f2: H/2
            f3: H/4
            f4: H/8
            f5: H/16
        """
        f1 = self.stem(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f5 = self.down4(f4)

        return f1, f2, f3, f4, f5


# ============================================================
# Completion-aware multi-view fusion
# ============================================================

class CompletionAwareViewFusion(nn.Module):
    """
    每个尺度分别计算：

    1. observed feature:
       只融合当前尺度下被观测到的多视角特征；

    2. prior feature:
       不乘 mask，保留 encoder 已经传播到缺失区域的上下文特征；

    3. feature uncertainty:
       多视角之间的特征方差；

    4. coverage:
       当前位置被多少个视角覆盖。

    在所有视角均缺失的位置，网络主要使用 prior feature，
    而不是将 feature 强制清零。
    """

    def __init__(
        self,
        channels,
        norm_type="gn",
        num_groups=8,
        eps=1e-6,
    ):
        super().__init__()

        self.channels = channels
        self.eps = eps

        # 为每个视角、每个位置预测一个融合分数
        self.score_net = nn.Sequential(
            nn.Conv2d(
                channels + 1,
                channels,
                kernel_size=1,
                bias=False,
            ),
            make_norm(norm_type, channels, num_groups),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                channels,
                1,
                kernel_size=1,
                bias=True,
            ),
        )

        context_channels = channels * 3 + 1

        # 决定当前区域更依赖 observation 还是 prior
        self.gate_net = nn.Sequential(
            nn.Conv2d(
                context_channels,
                channels,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.refine = nn.Sequential(
            nn.Conv2d(
                context_channels,
                channels,
                kernel_size=1,
                bias=False,
            ),
            make_norm(norm_type, channels, num_groups),
            nn.SiLU(inplace=True),
            ResidualConvBlock(
                channels,
                channels,
                norm_type,
                num_groups,
            ),
        )

    def forward(self, feat, mask, B, V):
        """
        feat:
            (B*V, C, Hf, Wf)

        mask:
            (B, V, 1, H, W)

        return:
            (B, C, Hf, Wf)
        """
        _, C, Hf, Wf = feat.shape

        feat = feat.reshape(B, V, C, Hf, Wf)

        # area interpolation 会保留局部 coverage 比例，
        # 比 nearest interpolation 更适合 feature-level mask。
        mask_ds = mask.reshape(
            B * V,
            1,
            mask.shape[-2],
            mask.shape[-1],
        ).float()

        mask_ds = F.interpolate(
            mask_ds,
            size=(Hf, Wf),
            mode="area",
        )

        mask_ds = mask_ds.reshape(
            B,
            V,
            1,
            Hf,
            Wf,
        )

        # ----------------------------------------------------
        # Learned observed-feature fusion
        # ----------------------------------------------------

        feat_flat = feat.reshape(B * V, C, Hf, Wf)
        mask_flat = mask_ds.reshape(B * V, 1, Hf, Wf)

        score_input = torch.cat(
            [feat_flat, mask_flat],
            dim=1,
        )

        scores = self.score_net(score_input)
        scores = scores.reshape(B, V, 1, Hf, Wf)

        # 使用 log(mask) 将 soft coverage 加到 attention score 中
        scores = scores + torch.log(mask_ds.clamp_min(self.eps))

        weights = torch.softmax(scores, dim=1)

        weight_sum = mask_ds.sum(dim=1)
        valid_any = (weight_sum > self.eps).float()

        observed_feat = (weights * feat).sum(dim=1)
        observed_feat = observed_feat * valid_any

        # ----------------------------------------------------
        # Completion prior
        # ----------------------------------------------------

        # 这里不能乘 mask。
        # encoder 在缺失区域生成的上下文 feature 会被保留下来。
        prior_feat = feat.mean(dim=1)

        # ----------------------------------------------------
        # Multi-view uncertainty
        # ----------------------------------------------------

        diff = feat - observed_feat.unsqueeze(1)

        variance = (
            weights * diff.square()
        ).sum(dim=1)

        variance = variance * valid_any
        uncertainty = torch.sqrt(
            variance + self.eps
        )

        coverage = (
            weight_sum / float(V)
        ).clamp(0.0, 1.0)

        context = torch.cat(
            [
                observed_feat,
                prior_feat,
                uncertainty,
                coverage,
            ],
            dim=1,
        )

        gate = self.gate_net(context)

        # 完全缺失位置强制使用 prior branch
        gate = gate * valid_any

        fused = (
            gate * observed_feat
            + (1.0 - gate) * prior_feat
        )

        fused = fused + self.refine(context)
        return fused


# ============================================================
# Strong bottleneck
# ============================================================

class MultiScaleBottleneck(nn.Module):
    def __init__(
        self,
        channels,
        norm_type="gn",
        num_groups=8,
    ):
        super().__init__()

        branch_channels = channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(
                channels,
                branch_channels,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            make_norm(
                norm_type,
                branch_channels,
                num_groups,
            ),
            nn.SiLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                channels,
                branch_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False,
            ),
            make_norm(
                norm_type,
                branch_channels,
                num_groups,
            ),
            nn.SiLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(
                channels,
                branch_channels,
                kernel_size=3,
                padding=4,
                dilation=4,
                bias=False,
            ),
            make_norm(
                norm_type,
                branch_channels,
                num_groups,
            ),
            nn.SiLU(inplace=True),
        )

        self.branch8 = nn.Sequential(
            nn.Conv2d(
                channels,
                branch_channels,
                kernel_size=3,
                padding=8,
                dilation=8,
                bias=False,
            ),
            make_norm(
                norm_type,
                branch_channels,
                num_groups,
            ),
            nn.SiLU(inplace=True),
        )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                channels,
                branch_channels,
                kernel_size=1,
                bias=True,
            ),
            nn.SiLU(inplace=True),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(
                branch_channels * 5,
                channels,
                kernel_size=1,
                bias=False,
            ),
            make_norm(norm_type, channels, num_groups),
            nn.SiLU(inplace=True),
        )

        self.refine = nn.Sequential(
            ResidualConvBlock(
                channels,
                channels,
                norm_type,
                num_groups,
            ),
            ResidualConvBlock(
                channels,
                channels,
                norm_type,
                num_groups,
            ),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b4 = self.branch4(x)
        b8 = self.branch8(x)

        bg = self.global_pool(x)
        bg = F.interpolate(
            bg,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        out = torch.cat(
            [b1, b2, b4, b8, bg],
            dim=1,
        )

        out = self.fuse(out)
        out = out + x
        out = self.refine(out)
        return out


# ============================================================
# Main network
# ============================================================

class MultiViewFusionCompletionNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_channels=32,
        use_mask=True,
        norm_type="gn",
        num_groups=8,
        eps=1e-6,
        use_output_residual=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_mask = use_mask
        self.eps = eps

        self.use_output_residual = (
            use_output_residual
            and in_channels == out_channels
        )

        encoder_in_channels = (
            in_channels + 1
            if use_mask
            else in_channels
        )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        # ----------------------------------------------------
        # Per-view encoder
        # ----------------------------------------------------

        self.encoder = PerViewEncoder(
            in_channels=encoder_in_channels,
            base_channels=base_channels,
            norm_type=norm_type,
            num_groups=num_groups,
        )

        # ----------------------------------------------------
        # Multi-scale completion-aware fusion
        # ----------------------------------------------------

        self.fuse1 = CompletionAwareViewFusion(
            c1, norm_type, num_groups, eps
        )
        self.fuse2 = CompletionAwareViewFusion(
            c2, norm_type, num_groups, eps
        )
        self.fuse3 = CompletionAwareViewFusion(
            c3, norm_type, num_groups, eps
        )
        self.fuse4 = CompletionAwareViewFusion(
            c4, norm_type, num_groups, eps
        )
        self.fuse5 = CompletionAwareViewFusion(
            c5, norm_type, num_groups, eps
        )

        # ----------------------------------------------------
        # Strong global completion bottleneck
        # ----------------------------------------------------

        self.bottleneck = nn.Sequential(
            MultiScaleBottleneck(
                c5,
                norm_type,
                num_groups,
            ),
            MultiScaleBottleneck(
                c5,
                norm_type,
                num_groups,
            ),
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.up4 = UpsampleBlock(
            c5, c4, c4, norm_type, num_groups
        )
        self.up3 = UpsampleBlock(
            c4, c3, c3, norm_type, num_groups
        )
        self.up2 = UpsampleBlock(
            c3, c2, c2, norm_type, num_groups
        )
        self.up1 = UpsampleBlock(
            c2, c1, c1, norm_type, num_groups
        )

        # 最后再次输入原始 fused XYZ 和 coverage，
        # 让网络区分 observed / missing region。
        final_input_channels = c1 + in_channels + 1

        self.final_refine = nn.Sequential(
            ResidualConvBlock(
                final_input_channels,
                c1,
                norm_type,
                num_groups,
            ),
            ResidualConvBlock(
                c1,
                c1,
                norm_type,
                num_groups,
            ),
        )

        self.head = nn.Conv2d(
            c1,
            out_channels,
            kernel_size=1,
        )

    def compute_input_fusion(self, x, mask):
        """
        x:
            (B, V, C, H, W)

        mask:
            (B, V, 1, H, W)

        returns:
            fused_input: (B, C, H, W)
            coverage:    (B, 1, H, W)
        """
        weight_sum = mask.sum(dim=1)

        fused_input = (
            x * mask
        ).sum(dim=1) / weight_sum.clamp_min(self.eps)

        coverage = (
            weight_sum / float(x.shape[1])
        ).clamp(0.0, 1.0)

        return fused_input, coverage

    def forward(self, x, mask=None):
        """
        Parameters
        ----------
        x:
            (B, V, in_channels, H, W)

        mask:
            (B, V, 1, H, W)

        Returns
        -------
        out:
            (B, out_channels, H, W)
        """
        if x.dim() != 5:
            raise ValueError(
                "x must have shape (B, V, C, H, W), "
                f"but got {tuple(x.shape)}"
            )

        B, V, C, H, W = x.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected input channels={self.in_channels}, "
                f"but got C={C}"
            )

        if mask is None:
            if self.use_mask:
                raise ValueError(
                    "mask must be provided when use_mask=True"
                )

            mask = torch.ones(
                B,
                V,
                1,
                H,
                W,
                device=x.device,
                dtype=x.dtype,
            )

        if mask.dim() != 5:
            raise ValueError(
                "mask must have shape (B, V, 1, H, W), "
                f"but got {tuple(mask.shape)}"
            )

        mask = mask.to(
            device=x.device,
            dtype=x.dtype,
        )
        mask = mask.clamp(0.0, 1.0)

        # 非法区域输入必须先清零，防止错误 XYZ 扩散到有效区域。
        x_masked = x * mask

        fused_input, coverage = self.compute_input_fusion(
            x_masked,
            mask,
        )

        # ----------------------------------------------------
        # Per-view encoding
        # ----------------------------------------------------

        if self.use_mask:
            x_encoder = torch.cat(
                [x_masked, mask],
                dim=2,
            )
        else:
            x_encoder = x_masked

        x_encoder = x_encoder.reshape(
            B * V,
            x_encoder.shape[2],
            H,
            W,
        )

        f1, f2, f3, f4, f5 = self.encoder(x_encoder)

        # ----------------------------------------------------
        # Completion-aware multi-view fusion
        # ----------------------------------------------------

        f1 = self.fuse1(f1, mask, B, V)
        f2 = self.fuse2(f2, mask, B, V)
        f3 = self.fuse3(f3, mask, B, V)
        f4 = self.fuse4(f4, mask, B, V)
        f5 = self.fuse5(f5, mask, B, V)

        # ----------------------------------------------------
        # Global completion
        # ----------------------------------------------------

        z = self.bottleneck(f5)

        z = self.up4(z, f4)
        z = self.up3(z, f3)
        z = self.up2(z, f2)
        z = self.up1(z, f1)

        # 防止奇数分辨率导致最终尺寸不一致
        if z.shape[-2:] != (H, W):
            z = F.interpolate(
                z,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )

        z = torch.cat(
            [
                z,
                fused_input,
                coverage,
            ],
            dim=1,
        )

        z = self.final_refine(z)
        prediction = self.head(z)

        if self.use_output_residual:
            prediction = fused_input + prediction

        return prediction