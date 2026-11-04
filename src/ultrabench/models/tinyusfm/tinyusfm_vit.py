"""ViT-Tiny encoder for TinyUSFM.

Modified from https://github.com/MacDunno/TinyUSFM. Published alongside
"TinyUSFM: A Tiny and Efficient Universal Student Model for Vision"
(https://arxiv.org/abs/2510.19239).
"""

from functools import partial

import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


class TinyUSFMEncoder(VisionTransformer):
    """ViT-Tiny encoder matching the TinyUSFM student architecture.

    The architecture is fixed to the values hard-coded in the official
    implementation:
        - patch size: 16
        - embedding dimension: 192
        - depth: 12
        - number of attention heads: 12
        - MLP ratio: 4
        - QKV bias: True
        - ``LayerNorm(eps=1e-6)``

    The model has 5.52M total trainable parameters.
    """

    def __init__(self) -> None:
        """Initialize TinyUSFMEncoder with TinyUSFM's fixed hyperparameters."""
        super().__init__(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=0,
            global_pool="token",
            embed_dim=192,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
