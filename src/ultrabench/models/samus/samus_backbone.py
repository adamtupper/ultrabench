"""SAMUSBackbone: Backbone wrapper for the SAMUS ImageEncoderViT."""

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms

from ultrabench.models.backbone import Backbone, BackboneOutput
from ultrabench.models.samus.samus_image_encoder_vit import ImageEncoderViT


class SAMUSBackbone(Backbone):
    """Backbone wrapper for the SAMUS CNN-ViT image encoder.

    Accepts H x W sized uint8 tensors that are resized to 256 x 256 pixels and
    converted to float32 before being passed to the encoder. No per-channel
    normalisation is applied beyond the scale=True conversion to [0, 1].

    The encoder applies a SAM-style neck (two Conv2d + LayerNorm2d) that
    projects internal 768-dim features to 256 channels and outputs a spatial
    map of shape (B, 256, 32, 32) for 256 x 256 inputs.

    When pool=False the raw spatial map (B, 256, 32, 32) is returned. When
    pool=True, avg_pool2d with kernel/stride 16 is applied, giving
    (B, 256, 2, 2), which is flattened to (B, 1024).

    Attributes:
        feature_dim: 1024 (spatial positions after pooling).
        output_type: Always "spatial_map".
        has_cls_token: Always False.
        block_indices: Always None.
        input_size: 256.
    """

    feature_dim: int = 1024
    output_type: str = "spatial_map"
    has_cls_token: bool = False
    block_indices: list[int] | None = None
    input_size: int = 256

    def __init__(
        self, checkpoint_path: str, output_intermediate_features: bool
    ) -> None:
        """Initialise SAMUSBackbone.

        Args:
            checkpoint_path: Path to the SAMUS checkpoint.
            output_intermediate_features: Accepted for interface uniformity but
                ignored; SAMUS does not expose intermediate layer outputs.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
        """
        super().__init__(checkpoint_path, output_intermediate_features)

        self._encoder = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=256,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),  # ty:ignore[invalid-argument-type]
            num_heads=12,
            patch_size=8,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=(2, 5, 8, 11),
            window_size=14,
            out_chans=256,
            use_gradient_checkpointing=False,
        )

        self._load_checkpoint()

        self._transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.input_size, self.input_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToDtype(torch.float32, scale=True),
            ]
        )

    def _load_checkpoint(self) -> None:
        """Load encoder weights from checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
        """
        path = Path(self._checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(f"Could not find SAMUS weights at: {path}")

        state_dict = torch.load(path, map_location="cpu")
        clean = {k.replace("image_encoder.", ""): v for k, v in state_dict.items()}
        missing, unexpected = self._encoder.load_state_dict(clean, strict=False)
        non_encoder = [
            k
            for k in unexpected
            if not any(s in k for s in ["mask_decoder", "prompt_encoder"])
        ]
        assert not non_encoder, f"Unexpected non-encoder keys: {non_encoder}"

    def _resize_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Resize and convert a batch of uint8 images to float32 in [0, 1].

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Resized float32 tensor [B, C, 256, 256].
        """
        return self._transform(x)

    def forward(self, x: torch.Tensor, pool: bool = False) -> BackboneOutput:
        """Extract spatial features from pre-processed images.

        Args:
            x: uint8 input image tensor [B, C, H, W]. Resized to
                ``self.input_size`` and converted to float32 before encoding.
            pool: If True, apply avg_pool2d (kernel=16, stride=16) then
                flatten to [B, 1024]. If False, return the raw spatial map
                [B, 256, 32, 32].

        Returns:
            BackboneOutput with features [B, 256, 32, 32] when pool=False or
            [B, 1024] when pool=True. intermediate_features is always None.
        """
        x = self._resize_and_normalize(x)
        features: torch.Tensor = self._encoder(x)
        if pool:
            features = F.avg_pool2d(features, kernel_size=16, stride=16).flatten(
                start_dim=1
            )
        return BackboneOutput(features=features)
