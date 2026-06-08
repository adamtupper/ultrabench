from typing import Any, List, Optional

import torch
import torch.nn as nn

from ultrabench.training.heads.cls_heads import get_cls_head
from ultrabench.training.heads.seg_heads import get_seg_head


# Downstream Wrappers for different training
class DownstreamWrapper(nn.Module):
    """
    Base class for downstream evaluation wrappers.

    Wraps encoder with task-specific heads and defines which parameters to train.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_transforms: Optional[Any],
        encoder_metadata: dict,
        num_classes: int,
        task_type: str = "classification",
    ):
        """
        Args:
            encoder: Pretrained encoder model
            num_classes: Number of output classes
            task_type: 'classification' or 'segmentation'
        """
        super().__init__()
        self.encoder = encoder
        self.encoder_transforms = encoder_transforms
        self.encoder_metadata = encoder_metadata
        self.num_classes = num_classes
        self.task_type = task_type
        self._encoder_frozen = False

    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()  # keep BN/dropout stable for probing
        self._encoder_frozen = True

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        self.encoder.train()
        self._encoder_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._encoder_frozen:
            self.encoder.eval()
        return self

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Return list of trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]

    def _register_intermediate_block_hooks(self, strip_cls: bool = False):
        """
        Register forward hooks on the specified ViT blocks to collect intermediate
        patch token outputs for the SETR-MLA decoder.
        """
        self._hook_outputs: List[torch.Tensor] = []

        def make_hook(strip):
            def hook(module, input, output):
                self._hook_outputs.append(output[:, 1:, :] if strip else output)

            return hook

        self._hooks = [
            self.encoder.blocks[i].register_forward_hook(make_hook(strip_cls))  # ty:ignore[not-subscriptable, unresolved-attribute]
            for i in self.block_indices  # ty:ignore[not-iterable]
        ]

    def extract_features(self, x: torch.Tensor):
        """Forward pass. Returns [B, N, D], [B, C, H, W], or List[[B, C_i, H_i, W_i]]
        depending on the baseline and task_type."""
        baseline_name = self.encoder_metadata.baseline  # ty:ignore[unresolved-attribute]
        is_seg = self.task_type == "segmentation"

        if baseline_name == "urfm":
            x = self.encoder_transforms(x)[0]  # ty:ignore[call-non-callable]
            out = self.encoder.forward_encoder(x, mask=None)  # ty:ignore[call-non-callable]
            return out[:, 1:, :]  # drop CLS, keep patch tokens [B, N, D]
        elif baseline_name == "ultrasam":
            x = self.encoder_transforms(x)  # ty:ignore[call-non-callable]
            out = self.encoder(x)
            if is_seg:
                return out[0]  # [B, C, H, W]
            out = out[0].mean(dim=(2, 3))
        elif baseline_name == "echocare":
            x = self.encoder_transforms(x)  # ty:ignore[call-non-callable]
            out = self.encoder(x)
            if is_seg:
                return out  # List[[B, C_i, H_i, W_i]], fine-to-coarse
            out = out[-1].mean(dim=(2, 3))  # [B, 2048] from last scale
        elif baseline_name == "usfmae":
            x = self.encoder_transforms(x)  # ty:ignore[call-non-callable]
            out = self.encoder.forward_encoder(x, mask_ratio=0.0)[0]  # ty:ignore[call-non-callable]
            if is_seg:
                return out[:, 1:, :]  # patch tokens [B, N, D]
            out = out[:, 0, :]  # CLS token [B, D]
        elif baseline_name == "samus":
            x = self.encoder_transforms(x)  # ty:ignore[call-non-callable]
            out = self.encoder(x)
            if is_seg:
                return out  # [B, C, H, W]
            out = out.mean(dim=(2, 3))
        elif baseline_name == "usfm":
            x = self.encoder_transforms(x)  # ty:ignore[call-non-callable]
            out = self.encoder(x)
            return out  # patch tokens [B, N, D]
        else:
            x = self.encoder_transforms(x) if self.encoder_transforms else x
            out = self.encoder(x)

        # classification path: add token dimension for consistency
        if out.dim() == 2:
            out = out.unsqueeze(1)

        return out


class ZeroShotWrapper(DownstreamWrapper):
    """
    Zero-shot evaluation: Frozen encoder, no trainable head.
    """

    def __init__(
        self, encoder: nn.Module, num_classes: int, task_type: str = "classification"
    ):
        super().__init__(encoder, num_classes, task_type)  # ty:ignore[invalid-argument-type, missing-argument]


class ProbeWrapper(DownstreamWrapper):
    """
    Probe: Frozen encoder + trainable head.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_transforms: Optional[Any],
        encoder_metadata: dict,
        num_classes: int,
        task_type: str = "classification",
        probe_type: str = "linear",
    ):
        """
        Args:
            encoder: Pretrained encoder
            encoder_metadata: Metadata for the encoder
            num_classes: Number of classes
            task_type: 'classification' or 'segmentation'
            probe_type: 'linear' or 'attentive'
        """
        super().__init__(
            encoder, encoder_transforms, encoder_metadata, num_classes, task_type
        )
        self.probe_type = probe_type

        self.freeze_encoder()  # Freeze encoder for probing
        self.build_probe()

    def build_probe(self):
        """Build the probe head based on probe_type."""
        self.head = get_cls_head(
            probe_type=self.probe_type,
            feature_dim=self.encoder_metadata["feature_dim"],
            num_classes=self.num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and probe head."""
        with torch.no_grad():
            features = self.extract_features(x)

        # Assuming shape of features is [batch_size, tokens, feature_dim]
        features = features.mean(
            dim=1
        )  # Global average pooling over tokens -> [batch_size, feature_dim]

        logits = self.head(features)
        return logits


class SegmentationWrapper(DownstreamWrapper):
    """
    Frozen encoder + trainable segmentation head.

    Decoder is selected via decoder_type. See DECODER_COMPATIBILITY in seg_heads.py
    for which decoders are valid for each encoder output_type.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_transforms: Optional[Any],
        encoder_metadata: dict,
        num_classes: int,
        image_size: int,
        task_type: str = "segmentation",
        decoder_type: str = "upsampler",
        block_indices: Optional[List[int]] = None,
        upsampler_use_residual: bool = True,
    ):
        """
        Args:
            encoder: Pretrained encoder
            encoder_transforms: Transforms associated with the encoder (if any)
            encoder_metadata: Metadata for the encoder (must include feature_dim, output_type)
            num_classes: Number of output classes
            image_size: Input/output spatial resolution (assumed square)
            task_type: 'segmentation'
            decoder_type: 'upsampler', 'setr_mla', or 'segformer' — see DECODER_COMPATIBILITY
                          in seg_heads.py for which encoders each supports
            block_indices: ViT block indices to hook for setr_mla (e.g. [2, 5, 8, 11]);
                           defaults to evenly-spaced last 4 blocks if not specified
            upsampler_use_residual: Whether to include residual refinement blocks in the upsampler head
        """
        super().__init__(
            encoder, encoder_transforms, encoder_metadata, num_classes, task_type
        )
        self.decoder_type = decoder_type
        self.image_size = image_size
        self.block_indices = block_indices
        self.upsampler_use_residual = upsampler_use_residual

        self.freeze_encoder()
        self.build_seg_head()

    def build_seg_head(self):
        feature_dim = self.encoder_metadata["feature_dim"]
        scale_dims = self.encoder_metadata.get("scale_dims", None)
        output_type = self.encoder_metadata.get("output_type", "patch_tokens")

        if self.decoder_type == "setr_mla":
            if self.block_indices is None:
                n = len(self.encoder.blocks)  # ty:ignore[invalid-argument-type]
                self.block_indices = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
            self._register_intermediate_block_hooks(
                strip_cls=self.encoder_metadata.get("has_cls_token")  # ty:ignore[invalid-argument-type]
            )

        self.head = get_seg_head(
            decoder_type=self.decoder_type,
            output_type=output_type,
            feature_dim=feature_dim,
            scale_dims=scale_dims,
            num_classes=self.num_classes,
            image_size=self.image_size,
            block_indices=self.block_indices,
            use_residual=self.upsampler_use_residual,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            logits of shape [B, num_classes, H, W]
        """
        if self.decoder_type == "setr_mla":
            self._hook_outputs.clear()
            with torch.no_grad():
                self.extract_features(x)  # hooks populate _hook_outputs
            return self.head(self._hook_outputs)
        else:
            with torch.no_grad():
                features = self.extract_features(x)
            return self.head(features)


class FullFinetuningWrapper(DownstreamWrapper):
    """
    Full finetuning: All encoder parameters + head trainable.

    Starts with the encoder frozen (phase 1 — head warmup). Trainer calls
    unfreeze_encoder() at epoch == probe_warmup_epochs to begin phase 2.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_transforms: Optional[Any],
        encoder_metadata: dict,
        num_classes: int,
        task_type: str = "classification",
        probe_type: str = "linear",
        seg_decoder_type: str = "upsampler",
        image_size: Optional[int] = None,
        block_indices: Optional[List[int]] = None,
        upsampler_use_residual: bool = True,
    ):
        """
        Args:
            encoder: Pretrained encoder
            encoder_transforms: Transforms associated with the encoder (if any)
            encoder_metadata: Metadata for the encoder (must include feature_dim)
            num_classes: Number of output classes
            task_type: 'classification' or 'segmentation'
            probe_type: Classification head type ('linear')
            seg_decoder_type: Segmentation decoder type — 'upsampler', 'setr_mla', or 'segformer';
                              see DECODER_COMPATIBILITY in seg_heads.py for which encoders each supports
            image_size: Input/output spatial resolution (required for segmentation)
            block_indices: ViT block indices to hook for setr_mla;
                           defaults to evenly-spaced last 4 blocks if not specified
            upsampler_use_residual: Whether to include residual refinement blocks
        """
        super().__init__(
            encoder, encoder_transforms, encoder_metadata, num_classes, task_type
        )
        self.probe_type = probe_type
        self.seg_decoder_type = seg_decoder_type
        self.image_size = image_size
        self.block_indices = block_indices
        self.upsampler_use_residual = upsampler_use_residual

        self.freeze_encoder()  # phase 1 starts frozen; Trainer unfreezes at phase boundary

        if task_type == "classification":
            self.head = get_cls_head(
                probe_type=self.probe_type,
                feature_dim=self.encoder_metadata["feature_dim"],
                num_classes=self.num_classes,
            )
        else:
            output_type = self.encoder_metadata.get("output_type", "patch_tokens")
            if self.seg_decoder_type == "setr_mla":
                if self.block_indices is None:
                    n = len(self.encoder.blocks)  # ty:ignore[invalid-argument-type]
                    self.block_indices = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
                self._register_intermediate_block_hooks(
                    strip_cls=self.encoder_metadata.get("has_cls_token")  # ty:ignore[invalid-argument-type]
                )
            self.head = get_seg_head(
                decoder_type=self.seg_decoder_type,
                output_type=output_type,
                feature_dim=self.encoder_metadata["feature_dim"],
                scale_dims=self.encoder_metadata.get("scale_dims", None),
                num_classes=self.num_classes,
                image_size=self.image_size,  # ty:ignore[invalid-argument-type]
                block_indices=self.block_indices,
                use_residual=self.upsampler_use_residual,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder and task head.

        The encoder may be frozen (phase 1) or unfrozen (phase 2); the
        freeze/unfreeze state is managed externally by Trainer, so no
        special casing is needed here.
        """
        if self.task_type == "classification":
            features = self.extract_features(x)
            features = features.mean(dim=1)  # global average pooling -> [B, D]
            return self.head(features)
        else:
            if self.seg_decoder_type == "setr_mla":
                self._hook_outputs.clear()
                self.extract_features(x)  # hooks populate _hook_outputs
                return self.head(self._hook_outputs)
            else:
                features = self.extract_features(x)
                return self.head(features)
