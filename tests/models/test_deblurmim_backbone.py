"""Unit tests for DeblurMIMBackbone."""

from unittest.mock import patch

import pytest
import torch

from ultrabench.models.deblurmim.deblurmim_backbone import DeblurMIMBackbone

_BATCH = 2
_IMG_SIZE = 224
_PATCH_SIZE = 16
_NUM_PATCHES = (_IMG_SIZE // _PATCH_SIZE) ** 2  # 196
_FEATURE_DIM = 768


def _make_backbone(
    variant: str = "deblurmim_mae_base_patch16",
    output_intermediate_features: bool = False,
) -> DeblurMIMBackbone:
    """Return a DeblurMIMBackbone with checkpoint loading bypassed."""
    with patch.object(DeblurMIMBackbone, "_load_checkpoint"):
        return DeblurMIMBackbone(
            model_variant=variant,
            checkpoint_path="",
            output_intermediate_features=output_intermediate_features,
        )


class TestDeblurMIMBackboneAttributes:
    """Tests for backbone metadata attributes."""

    def test_mae_variant_attributes(self) -> None:
        model = _make_backbone("deblurmim_mae_base_patch16")
        assert model.feature_dim == _FEATURE_DIM
        assert model.output_type == "patch_tokens"
        assert model.has_cls_token is True
        assert model.block_indices == [2, 5, 8, 11]
        assert model.intermediate_feature_dims == [_FEATURE_DIM] * 4

    def test_convmae_variant_attributes(self) -> None:
        model = _make_backbone("deblurmim_convmae_base")
        assert model.feature_dim == _FEATURE_DIM
        assert model.output_type == "patch_tokens"
        assert model.has_cls_token is False
        assert model.block_indices == [2, 5, 8, 10]
        assert model.intermediate_feature_dims == [_FEATURE_DIM] * 4

    def test_unsupported_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported DeblurMIM variant"):
            DeblurMIMBackbone(
                model_variant="deblurmim_unknown",
                checkpoint_path="",
                output_intermediate_features=False,
            )


class TestDeblurMIMMAEBackboneForward:
    """Forward-pass tests for the Deblurring MAE variant."""

    def test_forward_features_shape(self) -> None:
        model = _make_backbone("deblurmim_mae_base_patch16")
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, _FEATURE_DIM)
        assert out.intermediate_features is None

    def test_forward_no_intermediate_when_disabled(self) -> None:
        model = _make_backbone("deblurmim_mae_base_patch16", output_intermediate_features=False)
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.intermediate_features is None

    def test_forward_intermediate_shape(self) -> None:
        model = _make_backbone("deblurmim_mae_base_patch16", output_intermediate_features=True)
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, _FEATURE_DIM)
        assert out.intermediate_features is not None
        assert len(out.intermediate_features) == 4
        for tensor in out.intermediate_features:
            assert tensor.shape == (_BATCH, _NUM_PATCHES, _FEATURE_DIM)

    def test_forward_accepts_non_square_input(self) -> None:
        model = _make_backbone("deblurmim_mae_base_patch16")
        x = torch.randint(0, 256, (_BATCH, 3, 128, 192), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, _FEATURE_DIM)

    def test_forward_pool_shape(self) -> None:
        model = _make_backbone("deblurmim_mae_base_patch16")
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x, pool=True)
        assert out.features.shape == (_BATCH, _FEATURE_DIM)


class TestDeblurMIMConvMAEBackboneForward:
    """Forward-pass tests for the Deblurring ConvMAE variant."""

    def test_forward_features_shape(self) -> None:
        model = _make_backbone("deblurmim_convmae_base")
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, _FEATURE_DIM)
        assert out.intermediate_features is None

    def test_forward_intermediate_shape(self) -> None:
        model = _make_backbone("deblurmim_convmae_base", output_intermediate_features=True)
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, _FEATURE_DIM)
        assert out.intermediate_features is not None
        assert len(out.intermediate_features) == 4
        for tensor in out.intermediate_features:
            assert tensor.shape == (_BATCH, _NUM_PATCHES, _FEATURE_DIM)

    def test_forward_pool_shape(self) -> None:
        model = _make_backbone("deblurmim_convmae_base")
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x, pool=True)
        assert out.features.shape == (_BATCH, _FEATURE_DIM)
