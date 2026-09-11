"""Backbone wrappers for DINOv3 encoders loaded via PyTorch Hub.

DINOv3 models are loaded from a local clone of the ``facebookresearch/dinov3``
repository through ``torch.hub.load(..., source="local")``; the gated
pretrained weights are supplied as a local file via the ``weights`` argument.
This avoids network access at runtime on gated/offline compute clusters. Two
wrappers are provided:

  - ``DINOv3ViTBackbone``: ViT-B/16 and ViT-L/16 encoders producing patch token
    features (``output_type="patch_tokens"``).
  - ``DINOv3ConvNeXtBackbone``: ConvNeXt-Base and ConvNeXt-Large encoders
    producing multi-scale feature maps (``output_type="multi_scale"``).
"""

from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms

from ultrabench2.models.backbone import Backbone, BackboneOutput

_VIT_VARIANT_CONFIGS: dict[str, dict] = {
    "dinov3_vitb16": {
        "feature_dim": 768,
        "block_indices": [2, 5, 8, 11],
    },
    "dinov3_vitl16": {
        "feature_dim": 1024,
        "block_indices": [5, 11, 17, 23],
    },
}

_CONVNEXT_VARIANT_CONFIGS: dict[str, dict] = {
    "dinov3_convnext_base": {
        "intermediate_feature_dims": [128, 256, 512, 1024],
    },
    "dinov3_convnext_large": {
        "intermediate_feature_dims": [192, 384, 768, 1536],
    },
}


class DINOv3ViTBackbone(Backbone):
    """Backbone wrapper for DINOv3 ViT encoders loaded via PyTorch Hub.

    Accepts H x W sized uint8 tensors that are resized to 224 x 224 pixels and
    normalized with ImageNet statistics before being passed to the encoder.

    Two variants are supported (see ``model_variant``). Both produce patch
    token features [B, N, D]; a CLS token is present internally but stripped
    before returning.

    Attributes:
        feature_dim: Encoder embedding dimension (768 or 1024).
        output_type: Always "patch_tokens".
        has_cls_token: Always True.
        block_indices: Four ViT block depths used for intermediate features.
        intermediate_feature_dims: [feature_dim] * 4.
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
        repo_dir: str,
        checkpoint_path: str,
        output_intermediate_features: bool,
    ) -> None:
        """Initialise DINOv3ViTBackbone.

        Args:
            model_variant: One of "dinov3_vitb16" or "dinov3_vitl16".
            repo_dir: Path to a local clone of the facebookresearch/dinov3
                repository, used as the source for torch.hub.load.
            checkpoint_path: Path to the gated DINOv3 weights file (.pth).
            output_intermediate_features: Whether to collect patch token
                outputs at self.block_indices in BackboneOutput.

        Raises:
            ValueError: If model_variant is not one of the supported values.
            FileNotFoundError: If the checkpoint file or the local repository
                cannot be found.
        """
        super().__init__(checkpoint_path, output_intermediate_features)

        self._repo_dir: str = repo_dir

        if model_variant not in _VIT_VARIANT_CONFIGS:
            raise ValueError(
                f"Unsupported DINOv3 ViT variant: {model_variant!r}. "
                f"Choose from: {list(_VIT_VARIANT_CONFIGS)}"
            )

        cfg = _VIT_VARIANT_CONFIGS[model_variant]
        self._hub_entrypoint: str = model_variant
        self.feature_dim: int = cfg["feature_dim"]
        self.block_indices: list[int] = cfg["block_indices"]
        self.intermediate_feature_dims: list[int] = [self.feature_dim] * 4

        self._load_checkpoint()

        self._transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=self.input_mean, std=self.input_std),
            ]
        )

    def _load_checkpoint(self) -> None:
        """Load the DINOv3 ViT encoder and its weights via PyTorch Hub.

        Raises:
            FileNotFoundError: If the checkpoint file or the local repository
                cannot be found.
        """
        path = Path(self._checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find DINOv3 weights at: {path}")

        repo_dir = Path(self._repo_dir)
        if not repo_dir.is_dir():
            raise FileNotFoundError(
                f"Could not find local DINOv3 repository at: {repo_dir}"
            )

        self._encoder = torch.hub.load(
            str(repo_dir), self._hub_entrypoint, source="local", weights=str(path)
        )
        self._encoder.eval()

    def _resize_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Resize and normalize a batch of uint8 images.

        Args:
            x: Input tensor [B, C, H, W] at any spatial size.

        Returns:
            Resized and normalized float32 tensor [B, C, 224, 224].
        """
        return self._transform(x)

    def forward(self, x: torch.Tensor, pool: bool = False) -> BackboneOutput:
        """Extract patch token features from input images.

        Args:
            x: uint8 input image tensor [B, C, H, W]. Resized to
                ``self.input_size`` and normalized before encoding.
            pool: If True, mean-pool over patch tokens to return
                [B, feature_dim]. If False, return the full sequence
                [B, N, feature_dim].

        Returns:
            BackboneOutput with features [B, N, feature_dim] or
            [B, feature_dim] when pool=True, and optionally
            intermediate_features (list of 4 patch-token tensors
            [B, N, feature_dim]).
        """
        x = self._resize_and_normalize(x)

        features = self._encoder.forward_features(x)["x_norm_patchtokens"]

        intermediate: list[torch.Tensor] | None = None
        if self._output_intermediate_features:
            intermediate = list(
                self._encoder.get_intermediate_layers(
                    x,
                    n=self.block_indices,
                    reshape=False,
                    return_class_token=False,
                    norm=True,
                )
            )

        if pool:
            features = features.mean(dim=1)

        return BackboneOutput(features=features, intermediate_features=intermediate)


class DINOv3ConvNeXtBackbone(Backbone):
    """Backbone wrapper for DINOv3 ConvNeXt encoders loaded via PyTorch Hub.

    Accepts H x W sized uint8 tensors that are resized to 224 x 224 pixels and
    normalized with ImageNet statistics before being passed to the encoder.

    Two variants are supported (see ``model_variant``). The encoder produces
    four hierarchical stage feature maps [B, C_i, H_i, W_i] (fine to coarse).
    They are included in BackboneOutput.intermediate_features only when
    output_intermediate_features=True.

    Attributes:
        feature_dim: Channel dim of the last (coarsest) stage (1024 or 1536).
        output_type: Always "multi_scale".
        has_cls_token: Always False.
        block_indices: Always None (ConvNeXt, not a ViT).
        intermediate_feature_dims: Channel dims per stage, fine to coarse.
        input_size: 224.
        input_mean: Per-channel mean for ImageNet normalization.
        input_std: Per-channel std for ImageNet normalization.
    """

    output_type: str = "multi_scale"
    has_cls_token: bool = False
    block_indices: list[int] | None = None
    input_size: int = 224
    input_mean: list[float] = [0.485, 0.456, 0.406]
    input_std: list[float] = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_variant: str,
        repo_dir: str,
        checkpoint_path: str,
        output_intermediate_features: bool,
    ) -> None:
        """Initialise DINOv3ConvNeXtBackbone.

        Args:
            model_variant: One of "dinov3_convnext_base" or
                "dinov3_convnext_large".
            repo_dir: Path to a local clone of the facebookresearch/dinov3
                repository, used as the source for torch.hub.load.
            checkpoint_path: Path to the gated DINOv3 weights file (.pth).
            output_intermediate_features: Whether to include all 4 ConvNeXt
                stage outputs in BackboneOutput.intermediate_features.

        Raises:
            ValueError: If model_variant is not one of the supported values.
            FileNotFoundError: If the checkpoint file or the local repository
                cannot be found.
        """
        super().__init__(checkpoint_path, output_intermediate_features)

        self._repo_dir: str = repo_dir

        if model_variant not in _CONVNEXT_VARIANT_CONFIGS:
            raise ValueError(
                f"Unsupported DINOv3 ConvNeXt variant: {model_variant!r}. "
                f"Choose from: {list(_CONVNEXT_VARIANT_CONFIGS)}"
            )

        cfg = _CONVNEXT_VARIANT_CONFIGS[model_variant]
        self._hub_entrypoint: str = model_variant
        self.intermediate_feature_dims: list[int] = cfg["intermediate_feature_dims"]
        self.feature_dim: int = self.intermediate_feature_dims[-1]
        self._num_stages: int = len(self.intermediate_feature_dims)

        self._load_checkpoint()

        self._transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=self.input_mean, std=self.input_std),
            ]
        )

    def _load_checkpoint(self) -> None:
        """Load the DINOv3 ConvNeXt encoder and its weights via PyTorch Hub.

        Raises:
            FileNotFoundError: If the checkpoint file or the local repository
                cannot be found.
        """
        path = Path(self._checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find DINOv3 weights at: {path}")

        repo_dir = Path(self._repo_dir)
        if not repo_dir.is_dir():
            raise FileNotFoundError(
                f"Could not find local DINOv3 repository at: {repo_dir}"
            )

        self._encoder = torch.hub.load(
            str(repo_dir), self._hub_entrypoint, source="local", weights=str(path)
        )
        self._encoder.eval()

    def _resize_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Resize and normalize a batch of uint8 images.

        Args:
            x: Input tensor [B, C, H, W] at any spatial size.

        Returns:
            Resized and normalized float32 tensor [B, C, 224, 224].
        """
        return self._transform(x)

    def forward(self, x: torch.Tensor, pool: bool = False) -> BackboneOutput:
        """Extract multi-scale features from input images.

        Args:
            x: uint8 input image tensor [B, C, H, W]. Resized to
                ``self.input_size`` and normalized before encoding.
            pool: If True, apply adaptive average pooling then flatten the last
                stage output to [B, feature_dim]. If False, return the spatial
                map [B, feature_dim, H', W'].

        Returns:
            BackboneOutput with features [B, feature_dim, H', W'] when
            pool=False or [B, feature_dim] when pool=True, and optionally
            intermediate_features (all 4 ConvNeXt stage outputs, fine to
            coarse).
        """
        x = self._resize_and_normalize(x)

        stage_outputs = list(
            self._encoder.get_intermediate_layers(
                x,
                n=list(range(self._num_stages)),
                reshape=True,
                return_class_token=False,
                norm=True,
            )
        )
        features = stage_outputs[-1]
        if pool:
            features = F.adaptive_avg_pool2d(features, 1).flatten(start_dim=1)

        intermediate = stage_outputs if self._output_intermediate_features else None
        return BackboneOutput(features=features, intermediate_features=intermediate)
