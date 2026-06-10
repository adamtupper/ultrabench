"""EchoCareBackbone: Backbone wrapper for the EchoCare Swin Transformer."""

from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms

from ultrabench.models.backbone import Backbone, BackboneOutput
from ultrabench.models.echocare.echocare_swin_transformer_arch import SwinTransformer


class EchoCareBackbone(Backbone):
    """Backbone wrapper for the EchoCare Swin Transformer encoder.

    Accepts H x W sized uint8 tensors that are resized to 256 x 256 pixels and
    converted to float32 before being passed to the encoder.

    The Swin Transformer forward pass always computes all 5 stage outputs
    (channel dims [128, 256, 512, 1024, 2048]). They are included in
    BackboneOutput.intermediate only when return_intermediate=True.

    Attributes:
        feature_dim: 2048 (last Swin stage output dimension).
        output_type: Always "multi_scale".
        has_cls_token: Always False.
        block_indices: Always None (not a ViT backbone).
        input_size: Native spatial resolution expected by the backbone (256).
        input_mean: Per-channel mean for input normalization (all zeros).
        input_std: Per-channel std for input normalization (all ones).
        intermediate_feature_dims: Channel dims per Swin stage, fine→coarse:
            [128, 256, 512, 1024, 2048].
    """

    feature_dim: int = 2048
    output_type: str = "multi_scale"
    has_cls_token: bool = False
    block_indices: list[int] | None = None
    input_size: int = 256
    input_mean: list[float] = [0.0, 0.0, 0.0]
    input_std: list[float] = [1.0, 1.0, 1.0]
    intermediate_feature_dims: list[int] = [128, 256, 512, 1024, 2048]

    def __init__(
        self, checkpoint_path: str, output_intermediate_features: bool
    ) -> None:
        """Initialise EchoCareBackbone.

        Args:
            checkpoint_path: Path to the model checkpoint.
            output_intermediate_features: Whether to include all 5 Swin stage
                outputs in BackboneOutput.intermediate_features.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
        """
        super().__init__(checkpoint_path, output_intermediate_features)

        self._encoder = SwinTransformer(
            in_chans=3,
            embed_dim=128,
            window_size=[8, 8],
            patch_size=[2, 2],
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            mlp_ratio=4.0,
            qkv_bias=True,
            use_checkpoint=False,
            spatial_dims=2,
            use_v2=True,
        )

        self._load_checkpoint()

        self._transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=self.input_mean, std=self.input_std),
            ]
        )

    def _load_checkpoint(self) -> None:
        """Load encoder weights from checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
        """
        path = Path(self._checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(f"Could not find EchoCare weights at: {path}")

        state_dict = torch.load(path, map_location="cpu")
        state_dict.pop("mask_token", None)
        self._encoder.load_state_dict(state_dict, strict=False)

    def _resize_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Resize and normalize a batch of uint8 images.

        Args:
            x: Input tensor [B, C, H, W] at any spatial size and value range.

        Returns:
            Resized and normalized tensor [B, C, H', W'] where H' and W' are
            determined by self.input_size (256).
        """
        return self._transform(x)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """Extract multi-scale features from pre-processed images.

        Args:
            x: uint8 input image tensor [B, C, H, W]. The images are resized to
                ``self.input_size`` and converted to float32 before being
                passed to the encoder.

        Returns:
            BackboneOutput with features [B, 2048, H, W] (last stage) and
            optionally intermediate features (all 5 Swin stage outputs, channel
            dims [128, 256, 512, 1024, 2048], fine→coarse).
        """
        x = self._resize_and_normalize(x)
        stage_outputs: list[torch.Tensor] = self._encoder(x)
        features = stage_outputs[-1]
        intermediate = stage_outputs if self._output_intermediate_features else None
        return BackboneOutput(features=features, intermediate_features=intermediate)
