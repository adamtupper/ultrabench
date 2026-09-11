"""
Standalone PyTorch implementation of ViTSAM.

Replicates mmpretrain.models.backbones.ViTSAM with no mmpretrain/mmcv/mmengine
dependency. Parameter names are intentionally kept identical to the mmpretrain
source so that checkpoints saved under that framework load without any key
remapping.

Key naming conventions that must match the checkpoint:
    - `patch_embed.projection.* `: mmcv PatchEmbed uses 'projection', not
      'proj'
    - `layers.{i}.*`: mmpretrain uses 'layers', SAM-original uses 'blocks'
    - `layers.{i}.ln1/.ln2 `: mmpretrain uses 'ln1/ln2', SAM-original uses
      'norm1/norm2'
    - `layers.{i}.ffn.layers.0.0.*`: mmcv FFN nested Sequential structure, see
      _FFN below
    - `layers.{i}.ffn.layers.1.*`:
    - `channel_reduction.*`: replaced with nn.Identity() in ultrasam.py

Reference: https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/models/backbones/vit_sam.py
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ── tiny utilities ───────────────────────────────────────────────────────────


def _to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


class DropPath(nn.Module):
    """Stochastic depth (drop path) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = (
            torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep_prob
        )
        return x.div(keep_prob) * random_tensor


class LayerNorm2d(nn.Module):
    """Channel-wise layer norm for (B, C, H, W) tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def _resize_pos_embed(
    pos_embed: torch.Tensor,
    src_shape: Tuple[int, int],
    dst_shape: Tuple[int, int],
    mode: str = "bicubic",
    num_extra_tokens: int = 0,
) -> torch.Tensor:
    """Resize position embeddings of shape (1, L, C)."""
    if src_shape == dst_shape:
        return pos_embed
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    extra = pos_embed[:, :num_extra_tokens] if num_extra_tokens > 0 else None
    pe = pos_embed[:, num_extra_tokens:]  # (1, src_h*src_w, C)
    B, L, C = pe.shape
    pe = pe.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)  # (1, C, src_h, src_w)
    pe = F.interpolate(pe, size=(dst_h, dst_w), mode=mode, align_corners=False)
    pe = pe.permute(0, 2, 3, 1).reshape(1, dst_h * dst_w, C)  # (1, dst_h*dst_w, C)
    if extra is not None:
        pe = torch.cat([extra, pe], dim=1)
    return pe


# ── window attention helpers (borrowed from Meta's SAM) ──────────────────────


def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    return attn


# ── sub-modules ──────────────────────────────────────────────────────────────


class _PatchEmbed(nn.Module):
    """Patch embedding whose conv layer is named 'projection' to match mmcv."""

    def __init__(
        self,
        in_channels: int = 3,
        input_size: int = 224,
        embed_dims: int = 768,
        kernel_size: int = 16,
        stride: int = 16,
        **_ignored,
    ):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size=kernel_size, stride=stride
        )
        h = input_size // stride
        self.init_out_size = (h, h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        x = self.projection(x)  # (B, C, H, W)
        out_h, out_w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return x, (out_h, out_w)


class Attention(nn.Module):
    """Multi-head attention with optional decomposed relative position bias."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dims // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.use_rel_pos = use_rel_pos
        if use_rel_pos:
            assert input_size is not None
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = (
            self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )
        attn = attn.softmax(dim=-1)
        x = (
            (attn @ v)
            .view(B, self.num_heads, H, W, -1)
            .permute(0, 2, 3, 1, 4)
            .reshape(B, H, W, -1)
        )
        return self.proj(x)


class _FFN(nn.Module):
    """
    Feed-forward network whose internal structure mirrors mmcv's FFN exactly:

        self.layers = Sequential(
            Sequential(Linear(embed, ff), GELU, Dropout),   # layers.0.0/.1/.2
            Linear(ff, embed),                               # layers.1
            Dropout,                                         # layers.2
        )
        self.dropout_layer = DropPath(...)

    This ensures checkpoint keys like 'ffn.layers.0.0.weight' load correctly.
    """

    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: int,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(
                nn.Linear(embed_dims, feedforward_channels),
                nn.GELU(),
                nn.Dropout(drop_rate),
            ),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(drop_rate),
        )
        self.dropout_layer = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, identity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        out = self.layers(x)
        out = self.dropout_layer(out)
        return (x if identity is None else identity) + out


class TransformerEncoderLayer(nn.Module):
    """SAM ViT block with optional window attention."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        feedforward_channels: int,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        norm_cfg: dict = dict(type="LN", eps=1e-6),
        **_ignored,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size
        eps = norm_cfg.get("eps", 1e-6)
        self.ln1 = nn.LayerNorm(embed_dims, eps=eps)
        self.attn = Attention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.ln2 = nn.LayerNorm(embed_dims, eps=eps)
        self.ffn = _FFN(embed_dims, feedforward_channels, drop_rate, drop_path_rate)

    # mmpretrain exposes these as norm1/norm2 properties
    # Kept for compatibility
    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.ln1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        x = self.ffn(self.ln2(x), identity=x)
        return x


# ── main backbone ────────────────────────────────────────────────────────────


class ViTSAM(nn.Module):
    """
    Standalone drop-in replacement for mmpretrain.models.backbones.ViTSAM.

    Default parameters are identical to the mmpretrain version. The forward
    method returns a tuple of feature maps: ``(feat,)`` where feat has shape
    ``(B, out_channels, H', W')`` — or ``(B, embed_dims, H', W')`` when
    ``channel_reduction`` has been replaced with ``nn.Identity()``.
    """

    arch_zoo = {
        **dict.fromkeys(
            ["b", "base"],
            {
                "embed_dims": 768,
                "num_layers": 12,
                "num_heads": 12,
                "feedforward_channels": 3072,
                "global_attn_indexes": [2, 5, 8, 11],
            },
        ),
        **dict.fromkeys(
            ["l", "large"],
            {
                "embed_dims": 1024,
                "num_layers": 24,
                "num_heads": 16,
                "feedforward_channels": 4096,
                "global_attn_indexes": [5, 11, 17, 23],
            },
        ),
        **dict.fromkeys(
            ["h", "huge"],
            {
                "embed_dims": 1280,
                "num_layers": 32,
                "num_heads": 16,
                "feedforward_channels": 5120,
                "global_attn_indexes": [7, 15, 23, 31],
            },
        ),
    }
    OUT_TYPES = {"raw", "featmap", "avg_featmap"}

    def __init__(
        self,
        arch: str = "base",
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        out_channels: int = 256,
        out_indices: list | int = -1,
        out_type: str = "raw",
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        qkv_bias: bool = True,
        use_abs_pos: bool = True,
        use_rel_pos: bool = True,
        window_size: int = 14,
        norm_cfg: dict = dict(type="LN", eps=1e-6),
        frozen_stages: int = -1,
        interpolate_mode: str = "bicubic",
        patch_cfg: dict = dict(),
        layer_cfgs=dict(),
        init_cfg=None,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in self.arch_zoo, f'Unknown arch "{arch}"'
            self.arch_settings = self.arch_zoo[arch]
        else:
            self.arch_settings = arch

        self.embed_dims = self.arch_settings["embed_dims"]
        self.num_layers = self.arch_settings["num_layers"]
        self.global_attn_indexes = self.arch_settings["global_attn_indexes"]
        self.img_size = _to_2tuple(img_size)
        self.interpolate_mode = interpolate_mode
        self.out_type = out_type
        assert out_type in self.OUT_TYPES

        # Patch embedding ('projection' key matches mmcv PatchEmbed)
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = _PatchEmbed(**_patch_cfg)  # ty:ignore[invalid-argument-type]
        self.patch_resolution = self.patch_embed.init_out_size

        # Absolute position embedding
        self.use_abs_pos = use_abs_pos
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, *self.patch_resolution, self.embed_dims)  # ty:ignore[invalid-argument-type]
            )
            self.drop_after_pos = nn.Dropout(p=drop_rate)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # out_indices normalisation
        if isinstance(out_indices, int):
            out_indices = [out_indices]
        self.out_indices = [i if i >= 0 else self.num_layers + i for i in out_indices]

        # Transformer layers
        dpr = np.linspace(0, drop_path_rate, self.num_layers).tolist()  # ty:ignore[no-matching-overload]
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers  # ty:ignore[unsupported-operator]

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):  # ty:ignore[invalid-argument-type]
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings["num_heads"],
                feedforward_channels=self.arch_settings["feedforward_channels"],
                drop_rate=drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=qkv_bias,
                use_rel_pos=use_rel_pos,
                window_size=window_size if i not in self.global_attn_indexes else 0,  # ty:ignore[unsupported-operator]
                input_size=self.patch_resolution,
                norm_cfg=norm_cfg,
            )
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))  # ty:ignore[invalid-argument-type]

        # Channel reduction neck
        self.out_channels = out_channels
        if out_channels > 0:
            self.channel_reduction = nn.Sequential(
                nn.Conv2d(self.embed_dims, out_channels, kernel_size=1, bias=False),  # ty:ignore[invalid-argument-type]
                LayerNorm2d(out_channels, eps=1e-6),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                LayerNorm2d(out_channels, eps=1e-6),
            )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.frozen_stages = frozen_stages
        if frozen_stages > 0:
            self._freeze_stages()

    # ── weight loading with automatic pos/rel-pos resizing ───────────────────

    def load_state_dict(self, state_dict, strict: bool = True):  # ty:ignore[invalid-method-override]
        """Load module state dict.

        Overridden to resize pos_embed and rel_pos tensors when shapes differ.
        """
        # Resize pos_embed if needed
        if self.use_abs_pos and "pos_embed" in state_dict:
            ckpt_pe = state_dict["pos_embed"]  # (1, H', W', C)
            if ckpt_pe.shape != self.pos_embed.shape:
                ckpt_h, ckpt_w = ckpt_pe.shape[1], ckpt_pe.shape[2]
                dst_h, dst_w = self.patch_resolution
                flat = ckpt_pe.flatten(1, 2)  # (1, H'*W', C)
                flat = _resize_pos_embed(
                    flat, (ckpt_h, ckpt_w), (dst_h, dst_w), self.interpolate_mode, 0
                )
                state_dict["pos_embed"] = flat.view(1, dst_h, dst_w, self.embed_dims)  # ty:ignore[invalid-argument-type]

        # Resize relative position embeddings if needed
        own_sd = self.state_dict()
        for key, own_val in own_sd.items():
            if "rel_pos_" not in key or key not in state_dict:
                continue
            ckpt_val = state_dict[key]
            if ckpt_val.shape[0] != own_val.shape[0]:
                L_src = ckpt_val.shape[0]
                L_dst = own_val.shape[0]
                resized = F.interpolate(
                    ckpt_val.reshape(1, L_src, -1).permute(0, 2, 1),
                    size=L_dst,
                    mode="linear",
                )
                state_dict[key] = resized.reshape(-1, L_dst).permute(1, 0)

        return super().load_state_dict(state_dict, strict=strict)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)
        x = x.view(B, patch_resolution[0], patch_resolution[1], self.embed_dims)

        if self.use_abs_pos:
            resized_pe = _resize_pos_embed(
                self.pos_embed.flatten(1, 2),
                self.patch_resolution,
                patch_resolution,
                mode=self.interpolate_mode,
                num_extra_tokens=0,
            )
            x = x + resized_pe.view(1, *patch_resolution, self.embed_dims)  # ty:ignore[invalid-argument-type]
            x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            if self.use_gradient_checkpointing and i in self.global_attn_indexes:  # ty:ignore[unsupported-operator]
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
            if i in self.out_indices:
                x_out = x.permute(0, 3, 1, 2)  # (B, C, H, W)
                if self.out_channels > 0:
                    x_out = self.channel_reduction(x_out)
                outs.append(self._format_output(x_out))

        return tuple(outs)

    def _format_output(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_type in ("raw", "featmap"):
            return x
        # avg_featmap: (B, C, H, W) -> (B, C)
        return x.flatten(2).mean(dim=2)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _freeze_stages(self):
        if self.use_abs_pos:
            self.pos_embed.requires_grad = False
            self.drop_after_pos.eval()
        self.patch_embed.eval()
        for p in self.patch_embed.parameters():
            p.requires_grad = False
        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.eval()
            for p in m.parameters():
                p.requires_grad = False
        if self.frozen_stages == self.num_layers and self.out_channels > 0:
            self.channel_reduction.eval()
            for p in self.channel_reduction.parameters():
                p.requires_grad = False
