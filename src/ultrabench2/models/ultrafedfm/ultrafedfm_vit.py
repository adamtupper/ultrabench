"""UltraFedFMEncoder: Encoder-only MAE ViT for UltraFedFM checkpoint loading.

The decoder is intentionally excluded. When a full MAE checkpoint (which
includes decoder weights) is loaded with strict=False, all decoder keys land
in unexpected_keys and are silently discarded, leaving only encoder weights.
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed


# ---------------------------------------------------------------------------
# Positional embedding utilities
# ---------------------------------------------------------------------------


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False):
    """Build a 2-D sinusoidal positional embedding."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(
        2, 1, grid_size, grid_size
    )
    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid):
    assert embed_dim % 2 == 0
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class UltraFedFMEncoder(nn.Module):
    """Encoder-only ViT matching the UltraFedFM MAE encoder structure.

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
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
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
        self.pos_embed.data.copy_(
            torch.tensor(pos_embed, dtype=torch.float32).unsqueeze(0)
        )

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
            Patch tokens with CLS prepended [B, 1+N, embed_dim].
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_tokens = (self.cls_token + self.pos_embed[:, :1, :]).expand(
            x.shape[0], -1, -1
        )
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def ultrafedfm_vit_base_patch16(**kwargs) -> UltraFedFMEncoder:
    """ViT-Base encoder: embed_dim=768, depth=12."""
    return UltraFedFMEncoder(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def ultrafedfm_vit_large_patch16(**kwargs) -> UltraFedFMEncoder:
    """ViT-Large encoder: embed_dim=1024, depth=24."""
    return UltraFedFMEncoder(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def ultrafedfm_vit_huge_patch14(**kwargs) -> UltraFedFMEncoder:
    """ViT-Huge encoder: embed_dim=1280, depth=32, patch_size=14."""
    return UltraFedFMEncoder(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
