"""Backbone wrapper for the TinyUSFM ViT-Tiny encoder."""

from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms

from ultrabench2.models.backbone import Backbone, BackboneOutput
from ultrabench2.models.tinyusfm.tinyusfm_vit import (
    TinyUSFMEncoder,
)

_CORE_KEY_PREFIXES = (
    "patch_embed.",
    "blocks.",
    "norm.",
    "cls_token",
    "pos_embed",
)


class TinyUSFMBackbone(Backbone):
    """Backbone wrapper for the TinyUSFM ViT-Tiny encoder.

    Accepts H x W sized uint8 tensors that are resized to 224 x 224 pixels,
    converted to float32, and normalized with ImageNet statistics before being
    passed to the encoder.

    The encoder is a 5.52M parameter ViT-Tiny distilled from USFM. The features
    are the patch tokens (CLS prepended but stripped from output) from the
    final encoder block. If ``pool=True`` is passed to ``forward()``, the patch
    tokens are mean-pooled to a single 192-dimensional vector per image.

    Attributes:
        feature_dim: 192
        output_type: Always "patch_tokens".
        has_cls_token: Always True (CLS prepended but stripped from output).
        block_indices: [3, 5, 7, 11], the out_indices used for the official
            TinyUSFM segmentation evaluations.
        intermediate_feature_dims: [192] x 4.
        input_size: 224.
        input_mean: Per-channel mean for ImageNet normalization.
        input_std: Per-channel std for ImageNet normalization.
    """

    feature_dim: int = 192
    output_type: str = "patch_tokens"
    has_cls_token: bool = True
    block_indices: list[int] = [3, 5, 7, 11]
    intermediate_feature_dims: list[int] = [192] * 4
    input_size: int = 224
    input_mean: list[float] = [0.485, 0.456, 0.406]
    input_std: list[float] = [0.229, 0.224, 0.225]

    def __init__(
        self, checkpoint_path: str, output_intermediate_features: bool
    ) -> None:
        """Initialize TinyUSFMBackbone.

        Args:
            checkpoint_path: Path to the TinyUSFM checkpoint.
            output_intermediate_features: Whether to collect patch token
                outputs at self.block_indices in BackboneOutput.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
            RuntimeError: If core encoder keys are missing after loading.
        """
        super().__init__(checkpoint_path, output_intermediate_features)

        self._encoder: TinyUSFMEncoder = TinyUSFMEncoder()
        self._load_checkpoint()

        self._transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.input_size, self.input_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=self.input_mean, std=self.input_std),
            ]
        )

        self._intermediate_features: list[torch.Tensor] | None = None
        if output_intermediate_features:
            self._register_forward_hooks()

    def _load_checkpoint(self) -> None:
        """Load encoder weights from a TinyUSFM checkpoint.

        Every key in the released checkpoint carries a single ``model.``
        prefix, from the ``TinyUSFM`` module that wraps the ViT in the official
        implementation. Our encoder removes this wrapper, so this prefix is
        removed.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
            RuntimeError: If core encoder keys are missing after loading.
        """
        path = Path(self._checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find TinyUSFM weights at: {path}")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ("model", "state_dict"):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break

        state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}

        msg = self._encoder.load_state_dict(state_dict, strict=False)

        bad = [k for k in msg.missing_keys if k.startswith(_CORE_KEY_PREFIXES)]
        if bad:
            raise RuntimeError(
                f"Core encoder keys missing after loading TinyUSFM checkpoint. "
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
            self._encoder.blocks[idx].register_forward_hook(hook)  # type: ignore

    def forward(self, x: torch.Tensor, pool: bool = False) -> BackboneOutput:
        """Extract patch token features from input images.

        Args:
            x: uint8 input image tensor [B, C, H, W]. Resized to
                ``self.input_size`` and normalized before encoding.
            pool: If True, mean-pool over patch tokens to return [B, 192]. If
                False, return the full sequence [B, N, 192].

        Returns:
            BackboneOutput with features [B, N, 192] when pool=False or
            [B, 192] when pool=True, and optionally intermediate_features (list
            of 4 tensors [B, N, 192] from the block depths in
            self.block_indices).
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
