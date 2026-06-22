"""Abstract base class for ultrasound foundation model backbones."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class BackboneOutput:
    """Output container for Backbone forward passes.

    Attributes:
        features: Primary encoder output. Shape depends on output_type:
            - "patch_tokens": [B, N, D]
            - "multi_scale": [B, C_last, H, W] (last/coarsest stage)
        intermediate_features: Auxiliary outputs. Interpretation depends on
        output_type:
            - "patch_tokens": list of [B, N, D] at block depths
            - "multi_scale": list of [B, C_i, H_i, W_i] per stage (fine to
              coarse features). None when return_intermediate=False.
    """

    features: torch.Tensor
    intermediate_features: list[torch.Tensor] | None = field(default=None)


class Backbone(ABC, nn.Module):
    """Abstract base class for ultrasound foundation model backbones.

    Accepts uint8 tensors that are resized to the appropriate shape (e.g.,
    224 x 224 pixels) and normalized on-the-fly inside forward() before being
    passed to the model.

    The shape of intermediate features are determined by output_type:
        - "patch_tokens": list of [B, N, D] collected at block_indices depths,
          CLS token stripped.
        - "multi_scale": list of [B, C_i, H_i, W_i] from each encoder stage.

    Intermediate features are only extracted when
    output_intermediate_features=True.

    Attributes:
        input_size: Spatial size to which input images are resized (e.g., 224).
        feature_dim: Dimensionality of the final feature vectors.
        output_type: Shape convention ("patch_tokens" or "multi_scale").
        intermediate_feature_dims: Per-stage channel dimensions for multi_scale
            backbones. None for non-multi-scale backbones.
    """

    input_size: int
    feature_dim: int
    output_type: str
    intermediate_feature_dims: list[int] | None = None

    def __init__(
        self, checkpoint_path: str, output_intermediate_features: bool
    ) -> None:
        """Initialize Backbone.

        Args:
            checkpoint_path: Path to the backbone checkpoint.
            output_intermediate_features: Whether to populate
                BackboneOutput.intermediate with features from block_indices.
        """
        super().__init__()

        self._checkpoint_path = checkpoint_path
        self._output_intermediate_features = output_intermediate_features

    @abstractmethod
    def _load_checkpoint(self) -> None:
        """Load backbone weights from checkpoint_path.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
        """
        ...

    @abstractmethod
    def _resize_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Resize and normalize raw input images.

        Args:
            x: Raw input tensor, shape [B, C, H, W], dtype uint8.

        Returns:
            Pre-processed tensor, shape [B, C, input_size, input_size], dtype
            float32.
        """
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor, pool: bool = False) -> BackboneOutput:
        """Extract features from pre-processed input images.

        Args:
            x: Raw uint8 input tensor with shape [B, C, H, W].
            pool: If True, reduce features to [B, feature_dim] for
                classification. If False, return the full spatial or
                sequence representation.

        Returns:
            A BackboneOutput with features and optionally intermediate outputs.
        """
        ...
