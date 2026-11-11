"""UltraFedFMBackbone: Backbone wrapper for the UltraFedFM MAE ViT encoder."""

from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms

from ultrabench.models.backbone import Backbone, BackboneOutput
from ultrabench.models.ultrafedfm.ultrafedfm_vit import (
    UltraFedFMEncoder,
    ultrafedfm_vit_base_patch16,
    ultrafedfm_vit_huge_patch14,
    ultrafedfm_vit_large_patch16,
)

_VARIANT_CONFIGS: dict[str, dict] = {
    "ultrafedfm_vit_base_patch16": {
        "factory": ultrafedfm_vit_base_patch16,
        "embed_dim": 768,
        "block_indices": [2, 5, 8, 11],
    },
    "ultrafedfm_vit_large_patch16": {
        "factory": ultrafedfm_vit_large_patch16,
        "embed_dim": 1024,
        "block_indices": [5, 11, 17, 23],
    },
    "ultrafedfm_vit_huge_patch14": {
        "factory": ultrafedfm_vit_huge_patch14,
        "embed_dim": 1280,
        "block_indices": [7, 15, 23, 31],
    },
}


class UltraFedFMBackbone(Backbone):
    """Backbone wrapper for the UltraFedFM federated MAE ViT encoder.

    Accepts H x W sized uint8 tensors that are resized to 224 x 224 pixels
    and normalized with ImageNet statistics before being passed to the encoder.

    The encoder is a ViT pre-trained with federated, attention-guided masked
    image modelling on multi-center ultrasound data. Three variants are
    supported: base (ViT-B/16), large (ViT-L/16), and huge (ViT-H/14). The
    released UltraFedFM checkpoint is the base (ViT-B/16) variant.

    Checkpoint format: ``{"model": state_dict}`` containing the full MAE
    encoder+decoder state. Decoder keys are unexpected when loading into the
    encoder-only model and are silently skipped by strict=False.

    Attributes:
        feature_dim: Encoder embedding dimension (768, 1024, or 1280).
        output_type: Always "patch_tokens".
        has_cls_token: Always True (CLS prepended but stripped from output).
        block_indices: Four evenly-spaced ViT block depths for intermediate
            feature collection.
        intermediate_feature_dims: [embed_dim] × 4.
        input_size: 224.
        input_mean: Per-channel mean for ImageNet normalization.
        input_std: Per-channel std for ImageNet normalization.
    """

    output_type: str = "patch_tokens"
    has_cls_token: bool = True
    input_size: int = 224
    input_mean: list[float] = [0.485, 0.456, 0.406]
    input_std: list[float] = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_variant: str,
        checkpoint_path: str,
        output_intermediate_features: bool,
    ) -> None:
        """Initialise UltraFedFMBackbone.

        Args:
            model_variant: Architecture variant — one of
                "ultrafedfm_vit_base_patch16", "ultrafedfm_vit_large_patch16",
                or "ultrafedfm_vit_huge_patch14".
            checkpoint_path: Path to the UltraFedFM MAE checkpoint (.pth).
            output_intermediate_features: Whether to collect patch token
                outputs at self.block_indices in BackboneOutput.

        Raises:
            ValueError: If model_variant is not one of the supported values.
            FileNotFoundError: If the checkpoint file cannot be found.
            RuntimeError: If core encoder keys are missing after loading.
        """
        super().__init__(checkpoint_path, output_intermediate_features)

        if model_variant not in _VARIANT_CONFIGS:
            raise ValueError(
                f"Unsupported UltraFedFM variant: {model_variant!r}. "
                f"Choose from: {list(_VARIANT_CONFIGS)}"
            )

        cfg = _VARIANT_CONFIGS[model_variant]
        self.feature_dim: int = cfg["embed_dim"]
        self.block_indices: list[int] = cfg["block_indices"]
        self.intermediate_feature_dims: list[int] = [self.feature_dim] * 4

        self._encoder: UltraFedFMEncoder = cfg["factory"]()

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
        """Load encoder weights from a UltraFedFM MAE checkpoint.

        The checkpoint must contain a "model" key with the full MAE state dict
        (encoder + decoder). Decoder keys appear as unexpected_keys and are
        silently ignored. A RuntimeError is raised if any encoder keys are
        absent after loading.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
            RuntimeError: If core patch_embed or block keys are missing after
                loading.
        """
        path = Path(self._checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find UltraFedFM weights at: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = (
            ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        )

        msg = self._encoder.load_state_dict(state_dict, strict=False)

        bad = [
            k
            for k in msg.missing_keys
            if k.startswith(("patch_embed.", "blocks.", "norm."))
        ]
        if bad:
            raise RuntimeError(
                f"Core encoder keys missing after loading UltraFedFM checkpoint. "
                f"First 5 missing: {bad[:5]}"
            )

    def _resize_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Resize and normalize a batch of uint8 images.

        Args:
            x: Input tensor [B, C, H, W] at any spatial size.

        Returns:
            Resized and normalized float32 tensor [B, C, 224, 224].
        """
        return self._transform(x)

    def _register_forward_hooks(self) -> None:
        """Register forward hooks on ViT blocks for intermediate features.

        Populates BackboneOutput.intermediate_features with patch tokens (CLS
        stripped) from encoder blocks listed in self.block_indices.
        """
        buf: list[torch.Tensor] = []
        self._intermediate_features = buf

        def hook(_module, _input, output):
            buf.append(output[:, 1:, :])  # strip CLS

        for idx in self.block_indices:
            self._encoder.blocks[idx].register_forward_hook(hook)

    def forward(self, x: torch.Tensor, pool: bool = False) -> BackboneOutput:
        """Extract patch token features from input images.

        Args:
            x: uint8 input image tensor [B, C, H, W]. Resized to
                ``self.input_size`` and normalized before encoding.
            pool: If True, mean-pool over patch tokens to return [B, D].
                If False, return the full sequence [B, N, D].

        Returns:
            BackboneOutput with features [B, N, D] when pool=False or [B, D]
            when pool=True, and optionally intermediate_features (list of 4
            tensors [B, N, D] from the block depths in self.block_indices).
        """
        if self._intermediate_features is not None:
            self._intermediate_features.clear()

        x = self._resize_and_normalize(x)
        out = self._encoder.forward_features(x)
        features = out[:, 1:, :]  # strip CLS
        if pool:
            features = features.mean(dim=1)

        return BackboneOutput(
            features=features,
            intermediate_features=self._intermediate_features,
        )
