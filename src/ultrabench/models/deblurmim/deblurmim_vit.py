"""DeblurMIM encoder architectures: standard MAE ViT and ConvViT variants.

DeblurMIM (Deblurring Masked Image Modelling) pretraining supports two
backbone architectures:

  - Deblurring MAE  (``deblurmim_mae_base_patch16``): standard ViT-Base/16,
    identical structure to a vanilla MAE checkpoint.
  - Deblurring ConvMAE (``deblurmim_convmae_base``): three-stage ConvViT
    (CBlock × 2 + CBlock × 2 + transformer Block × 11). Adapted from the
    DeblurringMIM repository.

Both classes retain full decoder weights so pre-trained checkpoints load
cleanly with strict=False; only forward_features() is used at inference.
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import (
    Block as _VitBlock,
    PatchEmbed as _VitPatchEmbed,
)

try:
    from timm.models.layers import DropPath, to_2tuple
except ImportError:
    from timm.layers import DropPath, to_2tuple


# ---------------------------------------------------------------------------
# Positional embedding utilities
# ---------------------------------------------------------------------------


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Build a 2-D sinusoidal positional embedding."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ---------------------------------------------------------------------------
# Deblurring MAE encoder (standard ViT-Base/16)
# ---------------------------------------------------------------------------


class DeblurMIMMAEEncoder(nn.Module):
    """Encoder-only ViT matching the Deblurring MAE checkpoint structure.

    Only the encoder-side parameters (patch_embed, cls_token, pos_embed,
    blocks, norm) are declared. Loading a full MAE checkpoint with
    strict=False places all decoder checkpoint keys in unexpected_keys, which
    are silently ignored.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        norm_layer=nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.patch_embed = _VitPatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                _VitBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.tensor(pos_embed, dtype=torch.float32).unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Full encoder pass without masking.

        Args:
            x: Float32 input tensor [B, C, H, W].

        Returns:
            Token sequence with CLS prepended [B, 1+N, embed_dim].
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_tokens = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def deblurmim_mae_base_patch16(**kwargs) -> DeblurMIMMAEEncoder:
    """Standard ViT-Base/16 encoder for the Deblurring MAE checkpoint."""
    return DeblurMIMMAEEncoder(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Building blocks (ConvViT)
# ---------------------------------------------------------------------------


class CMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBlock(nn.Module):
    """Convolutional block used in stages 1 and 2 of the ConvViT encoder."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = CMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mask=None):
        if mask is not None:
            x = x + self.drop_path(
                self.conv2(
                    self.attn(
                        mask
                        * self.conv1(
                            self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                        )
                    )
                )
            )
        else:
            x = x + self.drop_path(
                self.conv2(
                    self.attn(
                        self.conv1(
                            self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                        )
                    )
                )
            )
        x = x + self.drop_path(
            self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        )
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image-to-patch embedding with conv projection + LayerNorm + GELU."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1]
        )
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}x{W}) doesn't match model "
            f"({self.img_size[0]}x{self.img_size[1]})."
        )
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.act(x)


# ---------------------------------------------------------------------------
# DeblurringMIM ConvViT MAE
# ---------------------------------------------------------------------------


class DMaskedAutoencoderConvViT(nn.Module):
    """Deblurring Masked Autoencoder with ConvViT backbone.

    Three-stage encoder:
      - Stage 1: CBlock layers on [B, embed_dim[0], 56, 56] feature maps.
      - Stage 2: CBlock layers on [B, embed_dim[1], 28, 28] feature maps.
      - Stage 3: standard attention Block layers on [B, N, embed_dim[2]] tokens.

    Decoder weights are retained so checkpoints can be loaded intact and
    missing decoder keys discarded with strict=False. Only forward_features()
    is used at inference time.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        super().__init__()

        # Encoder
        self.patch_embed1 = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size[0],
            in_chans=in_chans,
            embed_dim=embed_dim[0],
        )
        self.patch_embed2 = PatchEmbed(
            img_size=img_size[1],
            patch_size=patch_size[1],
            in_chans=embed_dim[0],
            embed_dim=embed_dim[1],
        )
        self.patch_embed3 = PatchEmbed(
            img_size=img_size[2],
            patch_size=patch_size[2],
            in_chans=embed_dim[1],
            embed_dim=embed_dim[2],
        )
        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])
        self.stage1_output_decode = nn.Conv2d(embed_dim[0], embed_dim[2], 4, stride=4)
        self.stage2_output_decode = nn.Conv2d(embed_dim[1], embed_dim[2], 2, stride=2)

        num_patches = self.patch_embed3.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim[2]), requires_grad=False
        )

        self.blocks1 = nn.ModuleList(
            [
                CBlock(
                    dim=embed_dim[0],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio[0],
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(depth[0])
            ]
        )
        self.blocks2 = nn.ModuleList(
            [
                CBlock(
                    dim=embed_dim[1],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio[1],
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(depth[1])
            ]
        )
        self.blocks3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio[2],
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(depth[2])
            ]
        )
        self.norm = norm_layer(embed_dim[-1])

        # Decoder (retained so checkpoint loads cleanly with strict=False)
        self.decoder_embed = nn.Linear(embed_dim[-1], decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False
        )
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio[0],
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            (patch_size[0] * patch_size[1] * patch_size[2]) ** 2 * in_chans,
            bias=True,
        )

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed3.num_patches**0.5),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.tensor(pos_embed, dtype=torch.float32).unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed3.num_patches**0.5),
            cls_token=False,
        )
        self.decoder_pos_embed.data.copy_(
            torch.tensor(decoder_pos_embed, dtype=torch.float32).unsqueeze(0)
        )

        w = self.patch_embed3.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder forward pass without masking for inference.

        Args:
            x: Float32 input tensor [B, 3, 224, 224].

        Returns:
            Patch token tensor [B, N, embed_dim[-1]] after norm. No CLS token.
        """
        # Stage 1: convolutional blocks at 56×56
        x = self.patch_embed1(x)
        for blk in self.blocks1:
            x = blk(x)
        stage1_embed = self.stage1_output_decode(x).flatten(2).permute(0, 2, 1)

        # Stage 2: convolutional blocks at 28×28
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        stage2_embed = self.stage2_output_decode(x).flatten(2).permute(0, 2, 1)

        # Stage 3: transformer blocks on patch tokens
        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
        x = x + self.pos_embed

        for blk in self.blocks3:
            x = blk(x)
        x = x + stage1_embed + stage2_embed
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def dconvmae_convvit_base_patch16(**kwargs) -> DMaskedAutoencoderConvViT:
    """Build the base ConvViT variant: embed_dim=[256,384,768], depth=[2,2,11]."""
    return DMaskedAutoencoderConvViT(
        img_size=[224, 56, 28],
        patch_size=[4, 2, 2],
        embed_dim=[256, 384, 768],
        depth=[2, 2, 11],
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=[4, 4, 4],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
