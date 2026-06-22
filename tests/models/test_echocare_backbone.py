"""Unit tests for EchoCareBackbone."""

from unittest.mock import patch

import torch

from ultrabench.models.echocare.echocare_backbone import EchoCareBackbone

_BATCH = 2
_IMG_SIZE = 256
_SCALE_DIMS = [128, 256, 512, 1024, 2048]
_NUM_STAGES = len(_SCALE_DIMS)


def _make_backbone(output_intermediate_features: bool = False) -> EchoCareBackbone:
    """Return an EchoCareBackbone with checkpoint loading bypassed."""
    with patch.object(EchoCareBackbone, "_load_checkpoint"):
        return EchoCareBackbone(
            checkpoint_path="",
            output_intermediate_features=output_intermediate_features,
        )


class TestEchoCareBackboneAttributes:
    """Tests for backbone metadata attributes."""

    def test_attributes(self) -> None:
        model = _make_backbone()
        assert model.feature_dim == 2048
        assert model.output_type == "multi_scale"
        assert model.has_cls_token is False
        assert model.block_indices is None
        assert model.intermediate_feature_dims == _SCALE_DIMS


class TestEchoCareBackboneForward:
    """Tests for forward pass output shapes."""

    def test_forward_features_shape(self) -> None:
        model = _make_backbone()
        x = torch.randn(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)
        out = model(x)
        assert out.features.shape[0] == _BATCH
        assert out.features.shape[1] == 2048

    def test_forward_no_intermediate_when_disabled(self) -> None:
        model = _make_backbone(output_intermediate_features=False)
        x = torch.randn(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)
        out = model(x)
        assert out.intermediate_features is None

    def test_forward_intermediate_length(self) -> None:
        model = _make_backbone(output_intermediate_features=True)
        x = torch.randn(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)
        out = model(x)
        assert out.intermediate_features is not None
        assert len(out.intermediate_features) == _NUM_STAGES

    def test_forward_intermediate_channel_dims(self) -> None:
        model = _make_backbone(output_intermediate_features=True)
        x = torch.randn(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)
        out = model(x)
        assert out.intermediate_features is not None
        for stage_out, expected_channels in zip(out.intermediate_features, _SCALE_DIMS):
            assert stage_out.shape[0] == _BATCH
            assert stage_out.shape[1] == expected_channels

    def test_features_matches_last_intermediate(self) -> None:
        model = _make_backbone(output_intermediate_features=True)
        x = torch.randn(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)
        out = model(x)
        assert out.intermediate_features is not None
        assert torch.equal(out.features, out.intermediate_features[-1])

    def test_forward_pool_shape(self) -> None:
        model = _make_backbone()
        x = torch.randn(_BATCH, 3, _IMG_SIZE, _IMG_SIZE)
        out = model(x, pool=True)
        assert out.features.shape == (_BATCH, model.feature_dim)
