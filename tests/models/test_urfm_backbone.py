"""Unit tests for URFMBackbone."""

from unittest.mock import patch

import pytest
import torch

from ultrabench.models.urfm.urfm_backbone import URFMBackbone

_BATCH = 2
_IMG_SIZE = 224
_PATCH_SIZE = 16
_NUM_PATCHES = (_IMG_SIZE // _PATCH_SIZE) ** 2  # 196


def _make_backbone(
    variant: str = "urfm_mae_vit_base_patch16",
    output_intermediate_features: bool = False,
) -> URFMBackbone:
    """Return an URFMBackbone with checkpoint loading bypassed."""
    with patch.object(URFMBackbone, "_load_checkpoint"):
        return URFMBackbone(
            variant,
            checkpoint_path="",
            output_intermediate_features=output_intermediate_features,
        )


class TestURFMBackboneAttributes:
    """Tests for backbone metadata attributes."""

    def test_base_variant_attributes(self) -> None:
        model = _make_backbone("urfm_mae_vit_base_patch16")
        assert model.feature_dim == 768
        assert model.output_type == "patch_tokens"
        assert model.has_cls_token is True
        assert model.block_indices == [2, 5, 8, 11]

    def test_large_variant_feature_dim(self) -> None:
        model = _make_backbone("urfm_mae_vit_large_patch16")
        assert model.feature_dim == 1024
        assert model.block_indices == [5, 11, 17, 23]

    def test_huge_variant_feature_dim(self) -> None:
        model = _make_backbone("urfm_mae_vit_huge_patch16")
        assert model.feature_dim == 1280
        assert model.block_indices == [7, 15, 23, 31]

    def test_unsupported_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported URFM variant"):
            URFMBackbone(
                "urfm_mae_vit_unknown",
                checkpoint_path="",
                output_intermediate_features=False,
            )


class TestURFMBackboneForward:
    """Tests for forward pass output shapes."""

    def test_forward_features_shape(self) -> None:
        model = _make_backbone()
        x = torch.randn(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, 768)
        assert out.intermediate_features is None

    def test_forward_no_intermediate_when_disabled(self) -> None:
        model = _make_backbone(output_intermediate_features=False)
        x = torch.randn(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)
        out = model(x)
        assert out.intermediate_features is None

    def test_forward_intermediate_shape(self) -> None:
        model = _make_backbone(output_intermediate_features=True)
        x = torch.randn(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, 768)
        assert out.intermediate_features is not None
        assert len(out.intermediate_features) == 4
        for tensor in out.intermediate_features:
            assert tensor.shape == (_BATCH, _NUM_PATCHES, 768)
