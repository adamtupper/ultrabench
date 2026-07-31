"""DeblurMIMBackbone: Backbone wrapper for DeblurMIM pre-trained encoders.

Supports two architecture variants:
  - ``deblurmim_mae_base_patch16``: standard ViT-Base/16 trained with the
    Deblurring MAE objective. Checkpoint format: {"model": state_dict} with
    encoder + decoder weights; decoder keys are dropped via strict=False.
  - ``deblurmim_convmae_base``: three-stage ConvViT trained with the
    Deblurring ConvMAE objective. Same checkpoint format.
"""

from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms

from ultrabench.models.backbone import Backbone, BackboneOutput
from ultrabench.models.deblurmim.deblurmim_vit import (
    deblurmim_mae_base_patch16,
    dconvmae_convvit_base_patch16,
)

_VARIANT_CONFIGS: dict[str, dict] = {
    "deblurmim_mae_base_patch16": {
        "factory": deblurmim_mae_base_patch16,
        "embed_dim": 768,
        "block_indices": [2, 5, 8, 11],
        "has_cls_token": True,
        "blocks_attr": "blocks",
        "validation_prefixes": ("patch_embed.", "blocks.", "norm."),
    },
    "deblurmim_convmae_base": {
        "factory": dconvmae_convvit_base_patch16,
        "embed_dim": 768,
        "block_indices": [2, 5, 8, 10],
        "has_cls_token": False,
        "blocks_attr": "blocks3",
        "validation_prefixes": ("patch_embed1.", "patch_embed2.", "patch_embed3.", "blocks3."),
    },
}


class DeblurMIMBackbone(Backbone):
    """Backbone wrapper for DeblurMIM pre-trained encoders.

    Accepts H x W sized uint8 tensors that are resized to 224 x 224 pixels
    and normalized with ImageNet statistics before being passed to the encoder.

    Two architecture variants are supported (see ``model_variant``). Both
    produce patch token features [B, N, 768]; the MAE variant additionally has
    a CLS token that is stripped before returning.

    Attributes:
        feature_dim: Always 768.
        output_type: Always "patch_tokens".
        has_cls_token: True for the MAE variant, False for ConvMAE.
        block_indices: Four block depths used for intermediate features.
        intermediate_feature_dims: [768, 768, 768, 768].
        input_size: 224.
        input_mean: Per-channel mean for ImageNet normalization.
        input_std: Per-channel std for ImageNet normalization.
    """

    output_type: str = "patch_tokens"
    input_size: int = 224
    input_mean: list[float] = [0.485, 0.456, 0.406]
    input_std: list[float] = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_variant: str,
        checkpoint_path: str,
        output_intermediate_features: bool,
    ) -> None:
        """Initialise DeblurMIMBackbone.

        Args:
            model_variant: Architecture variant — one of
                "deblurmim_mae_base_patch16" or "deblurmim_convmae_base".
            checkpoint_path: Path to the DeblurMIM checkpoint (.pth).
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
                f"Unsupported DeblurMIM variant: {model_variant!r}. "
                f"Choose from: {list(_VARIANT_CONFIGS)}"
            )

        cfg = _VARIANT_CONFIGS[model_variant]
        self.feature_dim: int = cfg["embed_dim"]
        self.has_cls_token: bool = cfg["has_cls_token"]
        self.block_indices: list[int] = cfg["block_indices"]
        self.intermediate_feature_dims: list[int] = [self.feature_dim] * 4
        self._blocks_attr: str = cfg["blocks_attr"]
        self._validation_prefixes: tuple[str, ...] = cfg["validation_prefixes"]

        self._encoder = cfg["factory"]()

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
        """Load encoder weights from a DeblurMIM checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
            RuntimeError: If core encoder keys are missing after loading.
        """
        path = Path(self._checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find DeblurMIM weights at: {path}"
            )

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        msg = self._encoder.load_state_dict(state_dict, strict=False)

        bad = [
            k
            for k in msg.missing_keys
            if k.startswith(self._validation_prefixes)
        ]
        if bad:
            raise RuntimeError(
                f"Core encoder keys missing after loading DeblurMIM checkpoint. "
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
        """Register forward hooks on encoder blocks for intermediate features.

        For the MAE variant the CLS token (position 0) is stripped inside
        each hook. For the ConvMAE variant no stripping is needed.
        """
        buf: list[torch.Tensor] = []
        self._intermediate_features = buf
        strip_cls = self.has_cls_token

        def hook(_module, _input, output):
            buf.append(output[:, 1:, :] if strip_cls else output)

        block_list = getattr(self._encoder, self._blocks_attr)
        for idx in self.block_indices:
            block_list[idx].register_forward_hook(hook)

    def forward(self, x: torch.Tensor, pool: bool = False) -> BackboneOutput:
        """Extract patch token features from input images.

        Args:
            x: uint8 input image tensor [B, C, H, W]. Resized to
                ``self.input_size`` and normalized before encoding.
            pool: If True, mean-pool over patch tokens to return [B, 768].
                If False, return the full sequence [B, N, 768].

        Returns:
            BackboneOutput with features [B, N, 768] or [B, 768] when
            pool=True, and optionally intermediate_features (list of 4
            patch-token tensors [B, N, 768]).
        """
        if self._intermediate_features is not None:
            self._intermediate_features.clear()

        x = self._resize_and_normalize(x)
        out = self._encoder.forward_features(x)
        features = out[:, 1:, :] if self.has_cls_token else out
        if pool:
            features = features.mean(dim=1)

        return BackboneOutput(
            features=features,
            intermediate_features=self._intermediate_features,
        )
