"""Unit tests for SAMUSBackbone."""

from unittest.mock import patch

import torch

from ultrabench.models.samus.samus_backbone import SAMUSBackbone

_BATCH = 2
_IMG_SIZE = 256
_PATCH_SIZE = 8
_NUM_PATCHES = (_IMG_SIZE // _PATCH_SIZE) ** 2  # 1024
_FEATURE_DIM = 256


def _make_backbone(output_intermediate_features: bool = False) -> SAMUSBackbone:
    """Return a SAMUSBackbone with checkpoint loading bypassed."""
    with patch.object(SAMUSBackbone, "_load_checkpoint"):
        return SAMUSBackbone(
            checkpoint_path="",
            output_intermediate_features=output_intermediate_features,
        )


class TestSAMUSBackboneAttributes:
    """Tests for backbone metadata attributes."""

    def test_attributes(self) -> None:
        model = _make_backbone()
        assert model.feature_dim == _FEATURE_DIM
        assert model.output_type == "patch_tokens"
        assert model.has_cls_token is False
        assert model.block_indices is None


class TestSAMUSBackboneForward:
    """Tests for forward pass output shapes."""

    def test_forward_features_shape(self) -> None:
        model = _make_backbone()
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, _FEATURE_DIM)
        assert out.intermediate_features is None

    def test_forward_intermediate_not_exposed(self) -> None:
        model = _make_backbone(output_intermediate_features=True)
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, _FEATURE_DIM)
        assert out.intermediate_features is None

    def test_forward_accepts_non_square_input(self) -> None:
        model = _make_backbone()
        x = torch.randint(0, 256, (_BATCH, 3, 128, 192), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, _FEATURE_DIM)
