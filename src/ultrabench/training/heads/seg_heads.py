import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """A standard double-conv residual block to add depth to the decoder."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.GELU()  # GELU is often better for ViT-based models

        # Shortcut if dimensions change
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class SimpleUpsamplerHead(nn.Module):
    """
    Lightweight probe analog for segmentation, modeled off ProgressiveDecoderHead.

    Uses the same progressive ConvTranspose2d + ResidualConvBlock upsampling
    structure as ProgressiveDecoderHead, with the same input handling for both
    [B, N, D] token sequences and [B, C, H, W] spatial grids.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        image_size: int,
        use_residual: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.proj = nn.Conv2d(feature_dim, 256, kernel_size=1)

        channels = [256, 128, 64, 32, 16]
        stages = []
        for in_ch, out_ch in zip(channels, channels[1:]):
            stages.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            if use_residual:
                stages.append(ResidualConvBlock(out_ch, out_ch))
        self.decoder = nn.Sequential(*stages)

        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            B, N, C = x.shape
            H = W = int(N**0.5)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)

        x = self.proj(x)
        x = self.decoder(x)
        x = self.classifier(x)
        return F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )


# class MultiScaleFPNDecoder(nn.Module):
#     """
#     FPN-style decoder for multi-scale CNN encoders (e.g. EchoCare Swin Transformer).

#     Accepts a list of spatial feature maps at different resolutions, applies lateral
#     1x1 convolutions to project to a common channel dimension, then performs a
#     top-down feature fusion pass. Spatial sizes are read from actual feature map
#     dims at runtime — no hardcoded scale ratios — so the decoder handles any
#     number of scales and any spatial resolution.

#     Architecture follows segmentation_models.pytorch FPNDecoder:
#     - Per-scale SegmentationBlocks with bilinear upsampling to a common resolution
#     - Configurable merge policy ("add" or "cat") across all scales
#     - Dropout2d before output
#     """

#     def __init__(
#         self,
#         scale_dims: List[int],
#         num_classes: int,
#         image_size: int,
#         decoder_channels: int = 256,
#         segmentation_channels: int = 128,
#         merge_policy: Literal["add", "cat"] = "add",
#         dropout: float = 0.2,
#     ):
#         """
#         Args:
#             scale_dims:            Channel counts at each scale, ordered fine-to-coarse
#                                    (e.g. [128, 256, 512, 1024, 2048] for EchoCare).
#             num_classes:           Number of output segmentation classes.
#             image_size:            Output spatial resolution (assumed square).
#             decoder_channels:      Common channel width after lateral projections.
#             segmentation_channels: Channel width after per-scale SegmentationBlocks.
#             merge_policy:          How to combine scales — "add" or "cat".
#             dropout:               Dropout2d probability before the prediction head.
#         """
#         super().__init__()
#         if len(scale_dims) < 2:
#             raise ValueError(
#                 f"MultiScaleFPNDecoder requires at least 2 scales, got {len(scale_dims)}."
#             )

#         self.image_size = image_size
#         self.merge_policy = merge_policy
#         num_scales = len(scale_dims)

#         # Lateral 1x1 projections — one per scale, fine-to-coarse
#         self.lateral_convs = nn.ModuleList([
#             nn.Conv2d(c, decoder_channels, kernel_size=1)
#             for c in scale_dims
#         ])

#         # Per-scale segmentation blocks.
#         # The coarsest scale needs (num_scales - 1) upsamples to reach the finest
#         # resolution; each finer scale needs one fewer. The finest needs 0.
#         self.seg_blocks = nn.ModuleList([
#             _SegmentationBlock(
#                 decoder_channels,
#                 segmentation_channels,
#                 n_upsamples=i,           # scale 0 (finest) → 0 upsamples
#             )
#             for i in range(num_scales)   # scale i (coarser) → i upsamples
#         ])

#         merged_channels = (
#             segmentation_channels
#             if merge_policy == "add"
#             else segmentation_channels * num_scales
#         )
#         self.dropout = nn.Dropout2d(p=dropout, inplace=True)
#         self.pred = nn.Conv2d(merged_channels, num_classes, kernel_size=1)

#     def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
#         """
#         Args:
#             features_list: Spatial feature maps ordered fine-to-coarse,
#                            each [B, C_i, H_i, W_i].
#         Returns:
#             Logits [B, num_classes, image_size, image_size].
#         """
#         # 1. Lateral projections
#         laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features_list)]

#         # 2. Top-down fusion — coarsest to finest, reading spatial dims at runtime
#         out = laterals[-1]
#         for i in range(len(laterals) - 2, -1, -1):
#             out = F.interpolate(out, size=laterals[i].shape[2:], mode="bilinear", align_corners=False)
#             out = out + laterals[i]
#             laterals[i] = out  # write fused result back so seg_blocks see fused features

#         # 3. Per-scale segmentation blocks — bring all scales to finest resolution
#         # laterals is now fine-to-coarse with top-down fusion applied.
#         # seg_blocks[0] handles finest (0 upsamples), seg_blocks[-1] handles coarsest.
#         seg_outputs = [
#             seg_block(lateral)
#             for seg_block, lateral in zip(self.seg_blocks, laterals)
#         ]

#         # 4. Merge across scales
#         if self.merge_policy == "add":
#             x = torch.stack(seg_outputs).sum(dim=0)
#         else:
#             x = torch.cat(seg_outputs, dim=1)

#         # 5. Dropout → prediction head → upsample to image_size
#         x = self.dropout(x)
#         x = self.pred(x)
#         return F.interpolate(
#             x,
#             size=(self.image_size, self.image_size),
#             mode="bilinear",
#             align_corners=False,
#         )

# class _Conv3x3BNReLU(nn.Module):
#     """3x3 Conv → BatchNorm2d → ReLU, with optional 2x bilinear upsample."""

#     def __init__(self, in_channels: int, out_channels: int, upsample: bool = False):
#         super().__init__()
#         self.upsample = upsample
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.block(x)
#         if self.upsample:
#             x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
#         return x


# class _SegmentationBlock(nn.Module):
#     """Stack of Conv3x3GNReLU blocks that bring a feature map up by 2^n_upsamples."""

#     def __init__(self, in_channels: int, out_channels: int, n_upsamples: int = 0):
#         super().__init__()
#         blocks = [_Conv3x3BNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]
#         for _ in range(1, n_upsamples):
#             blocks.append(_Conv3x3BNReLU(out_channels, out_channels, upsample=True))
#         self.block = nn.Sequential(*blocks)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.block(x)


# Maps each decoder to the encoder output_type(s) it supports.
# New users: pick your decoder based on what your encoder outputs.
#   'upsampler'  → any single-feature encoder (patch tokens or spatial maps)
#   'setr_mla'   → isotropic ViTs only; taps intermediate blocks via forward hooks
#   'segformer'  → hierarchical multi-scale encoders only (e.g. EchoCare)
DECODER_COMPATIBILITY = {
    "upsampler": {"patch_tokens", "spatial_map"},
    "setr_mla": {"patch_tokens"},
    "segformer": {"multi_scale"},
}


def get_seg_head(
    decoder_type: str,
    output_type: str,
    feature_dim: int,
    num_classes: int,
    image_size: int,
    block_indices: Optional[List[int]] = None,
    use_residual: bool = True,
    scale_dims: Optional[List[int]] = None,
) -> nn.Module:  # ty:ignore[invalid-return-type]
    """
    Factory for segmentation decoder heads.

    Decoder compatibility by encoder output_type:
        'upsampler'  (SimpleUpsamplerHead)         — patch_tokens, spatial_map
        'setr_mla'   (SETRMLADecoder)              — patch_tokens (isotropic ViTs only)
        'segformer'  (SegFormerMultiScaleDecoder)  — multi_scale (hierarchical encoders only)

    Args:
        decoder_type: One of 'upsampler', 'setr_mla', or 'segformer'
        output_type:  Encoder output type from encoder_metadata ('patch_tokens',
                      'spatial_map', or 'multi_scale')
        feature_dim:  Encoder token/channel embedding dimension
        num_classes:  Number of output segmentation classes
        image_size:   Output spatial resolution (assumed square)
        block_indices: ViT block indices tapped by SETRMLADecoder; required when
                       decoder_type == 'setr_mla'
        scale_dims:   Per-scale channel counts; required when decoder_type == 'segformer'
        use_residual: Whether to include residual refinement blocks (upsampler only)

    Returns:
        An nn.Module that maps encoder features -> [B, num_classes, H, W]
    """
    compatible = DECODER_COMPATIBILITY.get(decoder_type)
    if compatible is None:
        raise ValueError(
            f"Unknown decoder_type '{decoder_type}'. "
            f"Valid options: {sorted(DECODER_COMPATIBILITY)}. "
            f"See DECODER_COMPATIBILITY in seg_heads.py for the full compatibility table."
        )
    if output_type not in compatible:
        raise ValueError(
            f"Decoder '{decoder_type}' does not support encoder output_type='{output_type}'. "
            f"Compatible output types: {sorted(compatible)}. "
            f"See DECODER_COMPATIBILITY in seg_heads.py for the full compatibility table."
        )

    if decoder_type == "upsampler":
        return SimpleUpsamplerHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
            image_size=image_size,
            use_residual=use_residual,
        )
    elif decoder_type == "setr_mla":
        if block_indices is None:
            raise ValueError("block_indices required for setr_mla decoder")
        return SETRMLADecoder(
            feature_dim=feature_dim,
            num_classes=num_classes,
            image_size=image_size,
            block_indices=block_indices,
        )
    elif decoder_type == "segformer":
        if scale_dims is None:
            raise ValueError(
                "scale_dims required for segformer decoder (multi_scale encoder)"
            )
        return SegFormerMultiScaleDecoder(
            scale_dims=scale_dims,
            num_classes=num_classes,
            image_size=image_size,
        )


class MLP(nn.Module):
    """
    Per-scale linear projection, faithful to the SegFormer reference.
    Accepts either [B, C, H, W] spatial maps or [B, N, D] token sequences.
    """

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            # [B, C, H, W] → [B, H*W, C]
            x = x.flatten(2).transpose(1, 2)
        # [B, N, D] — strip CLS if needed
        B, N, _ = x.shape
        h = math.isqrt(N)
        if h * h != N:
            x = x[:, 1:, :]
            N -= 1
            h = math.isqrt(N)
        x = self.proj(x)  # [B, N, embed_dim]
        x = x.transpose(1, 2).reshape(B, -1, h, h)  # [B, embed_dim, H, W]
        return x


class SegFormerMultiScaleDecoder(nn.Module):
    """
    SegFormer MLP decoder for hierarchical encoders (e.g. EchoCare/Swin).

    Expects features_list of [B, C_i, H_i, W_i] spatial maps, fine-to-coarse,
    as returned directly by the encoder's forward pass.

    Based on: https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py
    """

    def __init__(
        self,
        scale_dims: List[int],
        num_classes: int,
        image_size: int,
        embed_dim: int = 256,
        dropout: float = 0.1,
        upsample_to_image_size: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.upsample_to_image_size = upsample_to_image_size
        num_scales = len(scale_dims)

        self.linear_projections = nn.ModuleList(
            [MLP(input_dim=d, embed_dim=embed_dim) for d in scale_dims]
        )
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * num_scales, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(p=dropout)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        projected = [proj(f) for proj, f in zip(self.linear_projections, features_list)]
        target_size = projected[0].shape[2:]
        aligned = [
            F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            if p.shape[2:] != target_size
            else p
            for p in projected
        ]
        x = self.linear_fuse(torch.cat(aligned, dim=1))
        x = self.dropout(x)
        x = self.linear_pred(x)
        if self.upsample_to_image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return x


def _tokens_to_spatial(x: torch.Tensor) -> torch.Tensor:
    """Reshape [B, N, D] patch token sequence to [B, D, H, W] spatial grid."""
    B, N, D = x.shape
    H = W = int(N**0.5)
    return x.permute(0, 2, 1).reshape(B, D, H, W)


class SETRMLADecoder(nn.Module):
    """
    SETR-MLA decoder, faithful to the official fudan-zvg implementation.
    Source
        - https://github.com/fudan-zvg/SETR/blob/main/configs/_base_/models/setr_mla.py
        - https://github.com/fudan-zvg/SETR/blob/main/mmseg/models/backbones/vit_mla.py#L19
        - https://github.com/fudan-zvg/SETR/blob/main/mmseg/models/decode_heads/vit_mla_head.py

    Unifies Conv_MLA (backbone-side in official code) and VIT_MLAHead
    into a single standalone module that accepts raw [B, N, D] token sequences.

    Key detail from backbone code: each tapped block output receives a
    dedicated LayerNorm before entering the aggregation block. This is
    absent from the paper but present in the official implementation and
    likely important for training stability on intermediate (unnormalized)
    block outputs.

    Full per-stream channel schedule (ViT-B, feature_dim=768,
    mla_channels=256, mlahead_channels=128, 4 streams, 224px input):

      [B, 196, 768]  raw tokens (CLS stripped if present)
      LayerNorm:     [B, 196, 768]
      reshape:       [B, 768, 14, 14]
      1×1 conv:      [B, 256, 14, 14]   ← Conv_MLA
      top-down add across streams
      3×3 fuse:      [B, 256, 14, 14]   ← Conv_MLA  (p5 skips addition)
      3×3 conv:      [B, 128, 14, 14]   ← MLAHead
      3×3 conv:      [B, 128, 14, 14]   ← MLAHead
      4× upsample:   [B, 128, 56, 56]   ← H/4
    concat(4):       [B, 512, 56, 56]
    3×3 pred conv:   [B, num_classes, 56, 56]
    upsample:        [B, num_classes, 224, 224]
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        image_size: int,
        block_indices: List[int],
        mla_channels: int = 256,
        mlahead_channels: int = 128,
    ):
        super().__init__()
        self.image_size = image_size
        self.block_indices = block_indices
        num_scales = len(block_indices)

        # ── Per-scale LayerNorm (norm_0 … norm_N in official backbone) ─────────
        # Applied to raw token sequences before reshape and projection.
        self.norms = nn.ModuleList(
            [nn.LayerNorm(feature_dim) for _ in range(num_scales)]
        )

        # ── Conv_MLA (backbone-side in official code) ───────────────────────────

        # 1×1 channel reduction: feature_dim → mla_channels, per stream
        self.channel_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(feature_dim, mla_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(mla_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_scales)
            ]
        )

        # 3×3 refinement after top-down addition, per stream
        self.channel_fuse = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        mla_channels, mla_channels, kernel_size=3, padding=1, bias=False
                    ),
                    nn.BatchNorm2d(mla_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_scales)
            ]
        )

        # ── MLAHead (decoder-side in official code) ─────────────────────────────

        # Two 3×3 convs per stream: mla_channels → mlahead_channels → mlahead_channels
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        mla_channels,
                        mlahead_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(mlahead_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        mlahead_channels,
                        mlahead_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(mlahead_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_scales)
            ]
        )

        # 3×3 prediction conv on concatenated streams at H/4
        self.pred = nn.Conv2d(
            mlahead_channels * num_scales, num_classes, kernel_size=3, padding=1
        )

    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features_list: [B, N, D] token tensors, one per block in block_indices,
                           ordered shallow-to-deep.
        Returns:
            Logits [B, num_classes, image_size, image_size].
        """
        # 1. LayerNorm per stream (on token sequences, before reshape)
        normed = [norm(f) for norm, f in zip(self.norms, features_list)]

        # 2. Reshape token sequences → spatial grids [B, D, H_p, W_p]
        grids = [_tokens_to_spatial(f) for f in normed]

        # 3. Per-stream 1×1 channel reduction: feature_dim → mla_channels
        reduced = [proj(g) for proj, g in zip(self.channel_proj, grids)]

        # 4. Top-down addition deep → shallow.
        #    Deepest stream (index -1) is never added to — it only donates.
        #    Each shallower stream accumulates all deeper streams via the chain.
        fused = list(reduced)
        for i in range(len(fused) - 2, -1, -1):
            fused[i] = fused[i] + fused[i + 1]

        # 5. Per-stream 3×3 refinement after addition
        fused = [conv(f) for conv, f in zip(self.channel_fuse, fused)]

        # 6. Per-stream two 3×3 convs + 4× upsample to H/4
        ups = []
        for head, f in zip(self.heads, fused):
            x = head(f)
            x = F.interpolate(
                x,
                size=(x.shape[-2] * 4, x.shape[-1] * 4),
                mode="bilinear",
                align_corners=True,
            )
            ups.append(x)

        # 7. Concatenate all streams at H/4
        x = torch.cat(ups, dim=1)  # [B, mlahead_channels * N, H/4, W/4]

        # 8. 3×3 prediction conv at H/4
        x = self.pred(x)  # [B, num_classes, H/4, W/4]

        # 9. Final upsample to image_size
        return F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=True,
        )
