"""UltraSAMBackbone: Backbone wrapper for the UltraSAM ViT-SAM encoder."""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms

from ultrabench2.models.backbone import Backbone, BackboneOutput
from ultrabench2.models.ultrasam.ultrasam_vit_sam_arch import ViTSAM


class UltraSAMBackbone(Backbone):
    """Backbone wrapper for the UltraSAM ViT-SAM encoder.

    Accepts H x W sized uint8 tensors that are resized to 256 x 256 pixels,
    converted to float32, and normalised with ImageNet statistics before being
    passed to the encoder.

    The channel-reduction neck is replaced with nn.Identity so that the encoder
    outputs the full 768-dim spatial feature map (B, 768, 16, 16) for 256 x 256
    inputs. The backbone flattens this to patch-token form (B, 256, 768).

    Attributes:
        feature_dim: 768 (embed_dims with identity channel reduction).
        output_type: Always "patch_tokens".
        has_cls_token: Always False.
        block_indices: Always None.
        input_size: 256.
        input_mean: ImageNet per-channel mean.
        input_std: ImageNet per-channel std.
    """

    feature_dim: int = 768
    output_type: str = "patch_tokens"
    has_cls_token: bool = False
    block_indices: list[int] = [2, 5, 8, 11]
    input_size: int = 256
    input_mean: list[float] = [0.485, 0.456, 0.406]
    input_std: list[float] = [0.229, 0.224, 0.225]

    def __init__(
        self, checkpoint_path: str, output_intermediate_features: bool
    ) -> None:
        """Initialise UltraSAMBackbone.

        Args:
            checkpoint_path: Path to the UltraSAM checkpoint.
            output_intermediate_features: Whether to collect patch token
                outputs at self.block_indices in BackboneOutput.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
        """
        super().__init__(checkpoint_path, output_intermediate_features)

        if output_intermediate_features:
            self._encoder = ViTSAM(
                img_size=self.input_size,
                out_indices=self.block_indices,
            )
        else:
            self._encoder = ViTSAM(img_size=self.input_size, out_indices=-1)

        # Disable channel reduction bottleneck
        self._encoder.channel_reduction = nn.Identity()  # ty:ignore[invalid-assignment]

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

    def _load_checkpoint(self) -> None:
        """Load encoder weights from checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
        """
        path = Path(self._checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(f"Could not find UltraSAM weights at: {path}")

        state_dict = torch.load(path, map_location="cpu", weights_only=False)
        clean = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        missing, unexpected = self._encoder.load_state_dict(clean, strict=False)
        non_backbone = [
            k
            for k in unexpected
            if not any(
                s in k
                for s in ["decoder", "prompt_encoder", "bbox_head", "channel_reduction"]
            )
        ]
        assert not non_backbone, f"Unexpected non-backbone keys: {non_backbone}"

    def _resize_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Resize, convert to float32, and apply ImageNet normalisation.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Processed float32 tensor [B, C, 256, 256].
        """
        return self._transform(x)

    def forward(self, x: torch.Tensor, pool: bool = False) -> BackboneOutput:
        """Extract patch token features from pre-processed images.

        The spatial output (B, 768, H, W) is flattened to (B, H*W, 768)
        patch-token form.

        Args:
            x: uint8 input image tensor [B, C, H, W]. Resized to
                ``self.input_size`` and normalised before encoding.
            pool: If True, mean-pool over patch tokens to return [B, 768].
                If False, return the full sequence [B, 256, 768].

        Returns:
            BackboneOutput with features [B, 256, 768] when pool=False or
            [B, 768] when pool=True, and optionally intermediate features
            (list of 4 tensors of shape [B, 256, 768]).
        """

        x = self._resize_and_normalize(x)
        block_outputs = [out.flatten(2).permute(0, 2, 1) for out in self._encoder(x)]

        features = block_outputs[-1]
        if pool:
            features = features.mean(dim=1)

        if self._output_intermediate_features:
            return BackboneOutput(
                features=features, intermediate_features=block_outputs
            )
        else:
            return BackboneOutput(features=features)
