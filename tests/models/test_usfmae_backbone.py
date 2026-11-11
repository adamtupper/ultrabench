"""Unit tests for USFMAEBackbone."""

from unittest.mock import patch

import torch

from ultrabench.models.usfmae.usfmae_backbone import USFMAEBackbone

_BATCH = 2
_IMG_SIZE = 224
_PATCH_SIZE = 16
_NUM_PATCHES = (_IMG_SIZE // _PATCH_SIZE) ** 2  # 196
_FEATURE_DIM = 768


def _make_backbone(output_intermediate_features: bool = False) -> USFMAEBackbone:
    """Return a USFMAEBackbone with checkpoint loading bypassed."""
    with patch.object(USFMAEBackbone, "_load_checkpoint"):
        return USFMAEBackbone(
            checkpoint_path="",
            output_intermediate_features=output_intermediate_features,
        )


class TestUSFMAEBackboneAttributes:
    """Tests for backbone metadata attributes."""

    def test_attributes(self) -> None:
        model = _make_backbone()
        assert model.feature_dim == _FEATURE_DIM
        assert model.output_type == "patch_tokens"
        assert model.has_cls_token is True
        assert model.block_indices == [2, 5, 8, 11]


class TestUSFMAEBackboneForward:
    """Tests for forward pass output shapes."""

    def test_forward_features_shape(self) -> None:
        model = _make_backbone()
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, _FEATURE_DIM)
        assert out.intermediate_features is None

    def test_forward_no_intermediate_when_disabled(self) -> None:
        model = _make_backbone(output_intermediate_features=False)
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.intermediate_features is None

    def test_forward_intermediate_shape(self) -> None:
        model = _make_backbone(output_intermediate_features=True)
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, _FEATURE_DIM)
        assert out.intermediate_features is not None
        assert len(out.intermediate_features) == 4
        for tensor in out.intermediate_features:
            assert tensor.shape == (_BATCH, _NUM_PATCHES, _FEATURE_DIM)

    def test_forward_accepts_non_square_input(self) -> None:
        model = _make_backbone()
        x = torch.randint(0, 256, (_BATCH, 3, 128, 192), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, _FEATURE_DIM)

    def test_forward_pool_shape(self) -> None:
        model = _make_backbone()
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x, pool=True)
        assert out.features.shape == (_BATCH, _FEATURE_DIM)

    def test_pool_returns_cls_token(self) -> None:
        """Pooled features are the CLS token, not the patch token mean."""
        model = _make_backbone()
        model.eval()
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        with torch.no_grad():
            latent, _, _ = model._encoder.forward_encoder(
                model._resize_and_normalize(x), mask_ratio=0.0
            )
            tokens = model(x).features
            pooled = model(x, pool=True).features

        torch.testing.assert_close(pooled, latent[:, 0, :])
        assert not torch.allclose(pooled, tokens.mean(dim=1))
