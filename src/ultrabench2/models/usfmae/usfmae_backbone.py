"""USFMAEBackbone: Backbone wrapper for the USF-MAE masked autoencoder."""

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms

from ultrabench2.models.backbone import Backbone, BackboneOutput
from ultrabench2.models.usfmae.usfmae import MaskedAutoencoderViT


class USFMAEBackbone(Backbone):
    """Backbone wrapper for the USF-MAE masked autoencoder encoder.

    Accepts H x W sized uint8 tensors that are resized to 224 x 224 pixels,
    converted to float32, and normalised with ImageNet statistics before being
    passed to the encoder.

    Decoder components are deleted after checkpoint loading to reduce memory
    usage. Inference uses mask_ratio=0.0 so all patches are retained in their
    original raster order.

    Unlike the other patch token backbones in this package, which mean-pool
    the patch tokens, pooled classification features are taken from the CLS
    token. This is deliberate: The original authors of USF-MAE use the CLS
    token in their evaluations.

    Attributes:
        feature_dim: 768 (embed_dim of ViT-Base).
        output_type: Always "patch_tokens".
        has_cls_token: Always True (CLS returned as the pooled feature and
            stripped from the patch token sequence).
        block_indices: [2, 5, 8, 11] (four evenly-spaced depths in 12 blocks).
        input_size: 224.
        input_mean: ImageNet per-channel mean.
        input_std: ImageNet per-channel std.
    """

    feature_dim: int = 768
    output_type: str = "patch_tokens"
    has_cls_token: bool = True
    block_indices: list[int] = [2, 5, 8, 11]
    input_size: int = 224
    input_mean: list[float] = [0.485, 0.456, 0.406]
    input_std: list[float] = [0.229, 0.224, 0.225]

    def __init__(
        self, checkpoint_path: str, output_intermediate_features: bool
    ) -> None:
        """Initialise USFMAEBackbone.

        Args:
            checkpoint_path: Path to the USF-MAE checkpoint.
            output_intermediate_features: Whether to collect patch token
                outputs at self.block_indices in BackboneOutput.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
        """
        super().__init__(checkpoint_path, output_intermediate_features)

        self._encoder = MaskedAutoencoderViT(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            decoder_embed_dim=512,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        self._load_checkpoint()

        for attr in (
            "decoder_embed",
            "decoder_blocks",
            "decoder_norm",
            "decoder_pred",
            "mask_token",
            "decoder_pos_embed",
        ):
            delattr(self._encoder, attr)

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
        """Load encoder weights from checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
        """
        path = Path(self._checkpoint_path)

        if not path.exists():
            raise FileNotFoundError(f"Could not find USF-MAE weights at: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        self._encoder.load_state_dict(checkpoint, strict=False)

    def _resize_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Resize, convert to float32, and apply ImageNet normalisation.

        Args:
            x: Input tensor [B, C, H, W].

        Returns:
            Processed float32 tensor [B, C, 224, 224].
        """
        return self._transform(x)

    def _register_forward_hooks(self) -> None:
        """Register forward hooks to capture intermediate features.

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
        """Extract features from pre-processed images.

        Args:
            x: uint8 input image tensor [B, C, H, W]. Resized to
                ``self.input_size`` and normalised before encoding.
            pool: If True, return the CLS token [B, 768]. If False,
                return the full patch token sequence [B, N, 768].

        Returns:
            BackboneOutput with features [B, N, 768] when pool=False or
            [B, 768] when pool=True, and optionally intermediate features
            (list of 4 tensors of shape [B, N, 768]).
        """
        if self._intermediate_features is not None:
            self._intermediate_features.clear()

        x = self._resize_and_normalize(x)
        latent, _, _ = self._encoder.forward_encoder(x, mask_ratio=0.0)
        # latent[:, 0] is the CLS token, already normalised by the encoder's
        # final LayerNorm; latent[:, 1:] are the patch tokens.
        features = latent[:, 0, :] if pool else latent[:, 1:, :]

        return BackboneOutput(
            features=features,
            intermediate_features=self._intermediate_features,
        )
