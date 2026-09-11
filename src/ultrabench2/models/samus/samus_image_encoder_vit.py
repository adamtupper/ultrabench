# Self-contained extraction of ImageEncoderViT and JointTransform2D from the SAMUS project.
# Source: https://github.com/xianlin7/SAMUS
# Original code is reproduced verbatim; no modifications have been made.

# ============================================================
# From: models/segment_anything_samus/modeling/common.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# ============================================================

import math
from typing import Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as Fv


class Adapter(nn.Module):
    def __init__(
        self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True
    ):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class AugAdapter(nn.Module):
    def __init__(
        self,
        D_features,
        mlp_ratio=0.25,
        num_heads=12,
        act_layer=nn.GELU,
        skip_connect=True,
    ):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        self.aug_fc = nn.Linear(num_heads, D_hidden_features)

    def forward(self, x, important_key):
        xs = self.D_fc1(x)
        aug = self.aug_fc(important_key)
        xs = self.act(xs * aug)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# ============================================================
# From: models/segment_anything_samus/modeling/image_encoder.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# ============================================================


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 8,
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        use_gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.global_attn_indexes = set(global_attn_indexes)
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.cnn_embed = SingleCNNEmbed(
            patchsize=patch_size, in_chans=3, embed_dim=embed_dim
        )
        self.patch_embed = PatchEmbed0(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=3,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, 1024 // 16, 1024 // 16, embed_dim)
            )
            self.post_pos_embed = PostPosEmbed(
                embed_dim=embed_dim,
                ori_feature_size=1024 // 16,
                new_feature_size=img_size // patch_size,
            )

        self.blocks = nn.ModuleList()

        for i in range(depth):
            block = ParaBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                depth=i,
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
        self.input_Adapter = Adapter(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        cnnx = self.cnn_embed(x)
        x = self.patch_embed(x)
        x = self.input_Adapter(x)

        if self.pos_embed is not None:
            pos_embed = self.post_pos_embed(self.pos_embed)
            x = x + pos_embed.repeat(x.shape[0], 1, 1, 1)

        for i, blk in enumerate(self.blocks):
            if self.use_gradient_checkpointing and i in self.global_attn_indexes:
                x, cnnx = checkpoint(blk, x, cnnx, use_reentrant=False)
            else:
                x, cnnx = blk(x, cnnx)

        x = x + 0.5 * cnnx

        x = self.neck(x.permute(0, 3, 1, 2))

        return x


class ParaBlock(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        depth: int = 0,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(
            embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer
        )

        self.window_size = window_size

        if self.window_size == 0:
            self.MLP_Adapter = Adapter(dim, skip_connect=False)
            self.Space_Adapter = qkvAttention(dim=dim, num_heads=num_heads)
            self.refine_Adapter = SingleConv(in_channels=dim, out_channels=dim)
            self.scale = 0.5

        self.dim = dim
        self.depth = depth

    def forward(
        self, x: torch.Tensor, cnnx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shortcut = x
        x = self.norm1(x)

        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        if self.window_size == 0:
            sax = self.Space_Adapter(x, cnnx, cnnx)
            x = x + sax
            cnnx = self.refine_Adapter(cnnx.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        x = self.attn(x)

        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x

        xn = self.norm2(x)
        x = x + self.mlp(xn)

        if self.window_size == 0:
            x = x + self.scale * self.MLP_Adapter(xn)

        return x, cnnx


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, (
                "Input size must be provided if using relative positional encoding."
            )
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv0 = self.qkv(x)
        qkv = qkv0.reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
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
        x = self.proj(x)

        return x


class qkvAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, (
                "Input size must be provided if using relative positional encoding."
            )
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        B, H, W, _ = q.shape
        q = (
            self.q(q)
            .reshape(B, H * W, self.num_heads, -1)
            .permute(0, 2, 1, 3)
            .reshape(B * self.num_heads, H * W, -1)
        )
        k = (
            self.k(k)
            .reshape(B, H * W, self.num_heads, -1)
            .permute(0, 2, 1, 3)
            .reshape(B * self.num_heads, H * W, -1)
        )
        v = (
            self.v(v)
            .reshape(B, H * W, self.num_heads, -1)
            .permute(0, 2, 1, 3)
            .reshape(B * self.num_heads, H * W, -1)
        )

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
        x = self.proj(x)

        return x


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


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class SingleDown(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class SingleConv(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)


class CNNEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        patchsize: int = 8,
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        downtimes = int(math.log2(patchsize))
        mid_channel = 64
        self.inc = DoubleConv(in_chans, mid_channel)
        self.downs = nn.ModuleList()
        for i in range(downtimes):
            if i == downtimes - 1:
                down = Down(mid_channel, embed_dim)
            else:
                down = Down(mid_channel, mid_channel * 2)
            mid_channel = mid_channel * 2
            self.downs.append(down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)
        for down in self.downs:
            x = down(x)
        x = x.permute(0, 2, 3, 1)
        return x


class SingleCNNEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        patchsize: int = 8,
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        downtimes = int(math.log2(patchsize))
        mid_channel = 64
        self.inc = SingleConv(in_chans, mid_channel)
        self.downs = nn.ModuleList()
        for i in range(downtimes):
            if i == downtimes - 1:
                down = SingleDown(mid_channel, embed_dim)
            else:
                down = SingleDown(mid_channel, mid_channel * 2)
            mid_channel = mid_channel * 2
            self.downs.append(down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)
        for down in self.downs:
            x = down(x)
        x = x.permute(0, 2, 3, 1)
        return x


class PostPosEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        embed_dim: int = 768,
        ori_feature_size: int = 64,
        new_feature_size: int = 32,
    ) -> None:
        super().__init__()
        downtimes = int(math.log2(ori_feature_size // new_feature_size))
        self.downs = nn.ModuleList()
        for i in range(downtimes):
            down = SingleDown(embed_dim, embed_dim)
            self.downs.append(down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)
        for down in self.downs:
            x = down(x)
        x = x.permute(0, 2, 3, 1)
        return x


class PatchEmbed0(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=16, stride=(8, 8), padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, (256 + 8, 256 + 8), mode="bilinear", align_corners=False)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


# ============================================================
# From: utils/data_us.py  (JointTransform2D and its dependency)
# ============================================================


def to_long_tensor(pic):
    img = torch.from_numpy(np.array(pic, np.uint8))
    return img.long()


class JointTransform2D:
    """
    Performs augmentation on image and mask when called. Due to the randomness of augmentation transforms,
    it is not enough to simply apply the same Transform from torchvision on the image and mask separetely.
    Doing this will result in messing up the ground truth mask. To circumvent this problem, this class can
    be used, which will take care of the problems above.

    Args:
        crop: tuple describing the size of the random crop. If bool(crop) evaluates to False, no crop will
            be taken.
        p_flip: float, the probability of performing a random horizontal flip.
        color_jitter_params: tuple describing the parameters of torchvision.transforms.ColorJitter.
            If bool(color_jitter_params) evaluates to false, no color jitter transformation will be used.
        p_random_affine: float, the probability of performing a random affine transform using
            torchvision.transforms.RandomAffine.
        long_mask: bool, if True, returns the mask as LongTensor in label-encoded format.
    """

    def __init__(
        self,
        img_size=256,
        low_img_size=256,
        ori_size=256,
        crop=(32, 32),
        p_flip=0.0,
        p_rota=0.0,
        p_scale=0.0,
        p_gaussn=0.0,
        p_contr=0.0,
        p_gama=0.0,
        p_distor=0.0,
        color_jitter_params=(0.1, 0.1, 0.1, 0.1),
        p_random_affine=0,
        long_mask=False,
    ):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_gama = p_gama
        self.p_contr = p_contr
        self.p_distortion = p_distor
        self.img_size = img_size
        self.color_jitter_params = color_jitter_params
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)
        self.p_random_affine = p_random_affine
        self.long_mask = long_mask
        self.low_img_size = low_img_size
        self.ori_size = ori_size

    def __call__(self, image, mask):
        if np.random.rand() < self.p_gama:
            c = 1
            g = np.random.randint(10, 25) / 10.0
            image = (np.power(image / 255, 1.0 / g) / c) * 255
            image = image.astype(np.uint8)
        image, mask = Fv.to_pil_image(image), Fv.to_pil_image(mask)
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image, self.crop)
            image, mask = Fv.crop(image, i, j, h, w), Fv.crop(mask, i, j, h, w)
        if np.random.rand() < self.p_flip:
            image, mask = Fv.hflip(image), Fv.hflip(mask)
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))  # ty:ignore[invalid-argument-type]
            image, mask = Fv.rotate(image, angle), Fv.rotate(mask, angle)
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
            image = Fv.resize(image, (new_h, new_w), InterpolationMode.BILINEAR)  # ty:ignore[invalid-argument-type]
            mask = Fv.resize(mask, (new_h, new_w), InterpolationMode.NEAREST)  # ty:ignore[invalid-argument-type]
            i, j, h, w = T.RandomCrop.get_params(image, (self.img_size, self.img_size))
            image, mask = Fv.crop(image, i, j, h, w), Fv.crop(mask, i, j, h, w)
        if np.random.rand() < self.p_gaussn:
            ns = np.random.randint(3, 15)
            noise = (
                np.random.normal(loc=0, scale=1, size=(self.img_size, self.img_size))
                * ns
            )
            noise = noise.astype(int)
            image = np.array(image) + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = Fv.to_pil_image(image.astype("uint8"))
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image = contr_tf(image)
        if np.random.rand() < self.p_distortion:
            distortion = T.RandomAffine(0, None, None, (5, 30))
            image = distortion(image)
        if self.color_jitter_params:
            image = self.color_tf(image)
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params(
                (-90, 90),
                (1, 1),
                (2, 2),
                (-45, 45),
                self.crop,  # ty:ignore[invalid-argument-type]
            )
            image, mask = (
                Fv.affine(image, *affine_params),  # ty:ignore[invalid-argument-type]
                Fv.affine(mask, *affine_params),  # ty:ignore[invalid-argument-type]
            )
        image = Fv.resize(
            image,
            (self.img_size, self.img_size),
            InterpolationMode.BILINEAR,  # ty:ignore[invalid-argument-type]
        )
        mask = Fv.resize(
            mask,
            (self.ori_size, self.ori_size),
            InterpolationMode.NEAREST,  # ty:ignore[invalid-argument-type]
        )
        low_mask = Fv.resize(
            mask,
            (self.low_img_size, self.low_img_size),
            InterpolationMode.NEAREST,  # ty:ignore[invalid-argument-type]
        )
        image = Fv.to_tensor(image)  # ty:ignore[invalid-argument-type]

        if not self.long_mask:
            mask = Fv.to_tensor(mask)  # ty:ignore[invalid-argument-type]
            low_mask = Fv.to_tensor(low_mask)  # ty:ignore[invalid-argument-type]
        else:
            mask = to_long_tensor(mask)
            low_mask = to_long_tensor(low_mask)
        return image, mask, low_mask
