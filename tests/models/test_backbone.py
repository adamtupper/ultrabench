"""Unit tests for BackboneOutput and Backbone base class."""

import pytest
import torch

from ultrabench2.models.backbone import Backbone, BackboneOutput


class TestBackboneOutput:
    """Tests for BackboneOutput dataclass."""

    def test_features_only(self) -> None:
        features = torch.zeros(2, 196, 768)
        out = BackboneOutput(features=features)
        assert out.features is features
        assert out.intermediate_features is None

    def test_with_intermediate(self) -> None:
        features = torch.zeros(2, 196, 768)
        intermediate = [torch.zeros(2, 196, 768) for _ in range(4)]
        out = BackboneOutput(features=features, intermediate_features=intermediate)
        assert out.features is features
        assert out.intermediate_features is intermediate
        assert len(out.intermediate_features) == 4


class TestBackboneABC:
    """Tests for Backbone abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            Backbone(  # type: ignore[abstract]
                checkpoint_path="", output_intermediate_features=False
            )

    def test_concrete_subclass_can_be_instantiated(self) -> None:
        class ConcreteBackbone(Backbone):
            feature_dim = 128
            output_type = "patch_tokens"

            def _load_checkpoint(self) -> None:
                pass

            def _resize_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
                return x

            def forward(self, x: torch.Tensor, pool: bool = False) -> BackboneOutput:
                return BackboneOutput(features=x)

        model = ConcreteBackbone(checkpoint_path="", output_intermediate_features=False)
        assert model.feature_dim == 128
        assert model.output_type == "patch_tokens"

    def test_subclass_without_forward_cannot_instantiate(self) -> None:
        class IncompleteBackbone(Backbone):
            feature_dim = 128
            output_type = "patch_tokens"

        with pytest.raises(TypeError):
            IncompleteBackbone(  # type: ignore[abstract]
                checkpoint_path="", output_intermediate_features=False
            )
