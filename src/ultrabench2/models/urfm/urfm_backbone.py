"""Backbone wrapper for the URFM encoder."""

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms

from ultrabench2.models.backbone import Backbone, BackboneOutput
from ultrabench2.models.urfm.urfm import MaskedAutoencoderViT

_VARIANT_CONFIGS: dict[str, dict] = {
    "urfm_mae_vit_base_patch16": {
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "intermediate_features": [2, 5, 8, 11],
    },
    "urfm_mae_vit_large_patch16": {
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "intermediate_features": [5, 11, 17, 23],
    },
    "urfm_mae_vit_huge_patch16": {
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "intermediate_features": [7, 15, 23, 31],
    },
}


class URFMBackbone(Backbone):
    """Backbone wrapper for the URFM masked autoencoder encoder.

    Accepts H x W sized uint8 tensors that are resized to 224 x 224 pixels and
    then normalized using ImageNet normalization before being passed to the
    encoder.

    Attributes:
        feature_dim: Encoder embedding dimension (768, 1024, or 1280).
        output_type: Always "patch_tokens".
        has_cls_token: Always True.
        block_indices: ViT block indices collected when
            return_intermediate=True.
        input_size: Native spatial resolution expected by the backbone
            (e.g. 224). Inputs at a different resolution are resized inside
            forward() before being passed to the encoder.
        input_mean: Per-channel mean for input normalization (ImageNet values).
        input_std: Per-channel std for input normalization (ImageNet values).
    """

    output_type: str = "patch_tokens"
    has_cls_token: bool = True
    input_size = 224
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_variant: str,
        checkpoint_path: str,
        output_intermediate_features: bool,
    ) -> None:
        """Initialise URFMBackbone.

        Args:
            model_variant: One of "urfm_mae_vit_base_patch16",
                "urfm_mae_vit_large_patch16", or "urfm_mae_vit_huge_patch16".
            checkpoint_path: Path to the model checkpoint. Resolved relative to
                the repo root if not absolute.
            block_indices: ViT block indices to collect when
                return_intermediate=True. Defaults to 4 evenly-spaced blocks
                spanning the full depth.

        Raises:
            ValueError: If model_variant is not supported.
            FileNotFoundError: If the checkpoint file cannot be found.
        """
        super().__init__(checkpoint_path, output_intermediate_features)

        if model_variant not in _VARIANT_CONFIGS:
            raise ValueError(
                f"Unsupported URFM variant: {model_variant!r}. "
                f"Choose from: {list(_VARIANT_CONFIGS)}"
            )

        cfg = _VARIANT_CONFIGS[model_variant]
        self.feature_dim: int = cfg["embed_dim"]
        self.block_indices: list[int] = cfg["intermediate_features"]

        self._encoder = MaskedAutoencoderViT(
            patch_size=16,
            embed_dim=cfg["embed_dim"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            decoder_embed_dim=768,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        self._load_checkpoint()

        self._transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=self.input_mean, std=self.input_std),
            ]
        )

        self._intermediate_features: list[torch.Tensor] | None = None
        if output_intermediate_features:
            self._register_forward_hooks()

    def _load_checkpoint(self) -> None:
        """Load encoder weights from checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
        """

        path = Path(self._checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(f"Could not find URFM weights at: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        missing, _ = self._encoder.load_state_dict(clean_state_dict, strict=False)
        assert all("decoder" in k for k in missing), (
            f"Unexpected missing non-decoder keys: "
            f"{[k for k in missing if 'decoder' not in k]}"
        )

    def _resize_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Resize and normalize a batch of uint8 images.

        Args:
            x: Input tensor [B, C, H, W] at any spatial size and value range.

        Returns:
            Resized and normalized tensor [B, C, H', W'] where H' and W' are
            determined by self.input_size (usually 224).
        """
        return self._transform(x)

    def _register_forward_hooks(self) -> None:
        """Register forward hooks to capture intermediate features.

        This is used when return_intermediate=True in forward() to populate
        BackboneOutput.intermediate features from encoder blocks listed in
        self.block_indices.
        """
        assert self.block_indices is not None

        buf: list[torch.Tensor] = []
        self._intermediate_features = buf

        def hook(_module, _input, output):
            buf.append(output[:, 1:, :])  # strip CLS

        for idx in self.block_indices:
            self._encoder.blocks[idx].register_forward_hook(hook)

    def forward(self, x: torch.Tensor, pool: bool = False) -> BackboneOutput:
        """Extract patch token features from pre-processed images.

        Args:
            x: uint8 input image tensor [B, C, H, W]. The images are resized to
                ``self.input_size`` and normalized using ImageNet
                normalization before being passed to the encoder.
            pool: If True, mean-pool over patch tokens to return [B, D].
                If False, return the full sequence [B, N, D].

        Returns:
            BackboneOutput with features [B, N, D] when pool=False or [B, D]
            when pool=True, and optionally intermediate features (list of 4
            tensors of shape [B, N, D]).
        """
        if self._intermediate_features is not None:
            self._intermediate_features.clear()

        x = self._resize_and_normalize(x)
        out = self._encoder.forward_encoder(x, mask=None)
        features = out[:, 1:, :]
        if pool:
            features = features.mean(dim=1)

        return BackboneOutput(
            features=features,
            intermediate_features=self._intermediate_features,
        )
