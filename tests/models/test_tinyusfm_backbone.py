"""Unit tests for TinyUSFMBackbone."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from ultrabench2.models.tinyusfm.tinyusfm_backbone import TinyUSFMBackbone
from ultrabench2.models.tinyusfm.tinyusfm_vit import TinyUSFMEncoder

_BATCH = 2
_IMG_SIZE = 224
_PATCH_SIZE = 16
_NUM_PATCHES = (_IMG_SIZE // _PATCH_SIZE) ** 2  # 196
_EMBED_DIM = 192


def _make_backbone(
    output_intermediate_features: bool = False,
) -> TinyUSFMBackbone:
    """Return a TinyUSFMBackbone with checkpoint loading bypassed."""
    with patch.object(TinyUSFMBackbone, "_load_checkpoint"):
        return TinyUSFMBackbone(
            checkpoint_path="",
            output_intermediate_features=output_intermediate_features,
        )


def _save_checkpoint(path: Path, prefix: str = "", drop_blocks: bool = False) -> None:
    """Write a synthetic TinyUSFM checkpoint to path.

    Args:
        path: Destination file path.
        prefix: Prefix to prepend to every key (e.g. "model.").
        drop_blocks: If True, omit all transformer block weights.
    """
    state_dict = TinyUSFMEncoder().state_dict()
    if drop_blocks:
        state_dict = {
            k: v for k, v in state_dict.items() if not k.startswith("blocks.")
        }

    checkpoint = {f"{prefix}{k}": v for k, v in state_dict.items()}
    # The released checkpoint retains distillation classification heads.
    checkpoint[f"{prefix}head.weight"] = torch.zeros(2, _EMBED_DIM)
    checkpoint[f"{prefix}head.bias"] = torch.zeros(2)

    torch.save(checkpoint, path)


class TestTinyUSFMBackboneAttributes:
    """Tests for backbone metadata attributes."""

    def test_attributes(self) -> None:
        """Metadata attributes match the TinyUSFM architecture."""
        model = _make_backbone()
        assert model.feature_dim == _EMBED_DIM
        assert model.output_type == "patch_tokens"
        assert model.has_cls_token is True
        assert model.block_indices == [3, 5, 7, 11]
        assert model.intermediate_feature_dims == [_EMBED_DIM] * 4
        assert model.input_size == _IMG_SIZE

    def test_parameter_count_matches_paper(self) -> None:
        """Encoder size matches the 5.5M reported in the paper."""
        model = _make_backbone()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == pytest.approx(5.5e6, abs=0.05e6)

    def test_encoder_key_set_matches_released_checkpoint(self) -> None:
        """Encoder declares the 150 keys the released checkpoint holds."""
        keys = _make_backbone()._encoder.state_dict()
        assert len(keys) == 150
        assert "norm.weight" in keys and "norm.bias" in keys
        assert not any(k.startswith("fc_norm.") for k in keys)
        assert not any(k.startswith("head.") for k in keys)


class TestTinyUSFMBackboneForward:
    """Tests for forward pass output shapes."""

    def test_forward_features_shape(self) -> None:
        """An unpooled forward pass returns one token per patch."""
        model = _make_backbone()
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, _EMBED_DIM)
        assert out.intermediate_features is None

    def test_forward_no_intermediate_when_disabled(self) -> None:
        """Intermediate features stay None when collection is disabled."""
        model = _make_backbone(output_intermediate_features=False)
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.intermediate_features is None

    def test_forward_intermediate_shape(self) -> None:
        """Enabling collection yields one token map per block index."""
        model = _make_backbone(output_intermediate_features=True)
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, _EMBED_DIM)
        assert out.intermediate_features is not None
        assert len(out.intermediate_features) == 4
        for tensor in out.intermediate_features:
            assert tensor.shape == (_BATCH, _NUM_PATCHES, _EMBED_DIM)

    def test_forward_accepts_non_square_input(self) -> None:
        """Non-square inputs are resized to the encoder's input size."""
        model = _make_backbone()
        x = torch.randint(0, 256, (_BATCH, 3, 128, 192), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, _EMBED_DIM)

    def test_forward_pool_shape(self) -> None:
        """Pooling reduces patch tokens to a single feature vector."""
        model = _make_backbone()
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x, pool=True)
        assert out.features.shape == (_BATCH, model.feature_dim)

    def test_pool_is_mean_over_patch_tokens(self) -> None:
        """Pooled features are the mean of the returned patch tokens."""
        model = _make_backbone()
        model.eval()

        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        with torch.no_grad():
            tokens = model(x).features
            pooled = model(x, pool=True).features

        torch.testing.assert_close(pooled, tokens.mean(dim=1))


class TestTinyUSFMBackboneCheckpointLoading:
    """Tests for checkpoint discovery and state dict key remapping."""

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        """An absent checkpoint file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Could not find TinyUSFM weights"):
            TinyUSFMBackbone(
                checkpoint_path=str(tmp_path / "absent.pth"),
                output_intermediate_features=False,
            )

    @pytest.mark.parametrize("prefix", ["model.", ""])
    def test_load_checkpoint_strips_prefix(self, tmp_path: Path, prefix: str) -> None:
        """Keys load whether or not they carry the `model.` prefix."""
        path = tmp_path / "TinyUSFM.pth"
        _save_checkpoint(path, prefix=prefix)

        model = TinyUSFMBackbone(
            checkpoint_path=str(path), output_intermediate_features=False
        )

        assert model.feature_dim == _EMBED_DIM

    def test_load_checkpoint_accepts_wrapped_state_dict(self, tmp_path: Path) -> None:
        """A state dict nested under a `model` key is unwrapped."""
        path = tmp_path / "TinyUSFM.pth"
        torch.save(
            {"model": TinyUSFMEncoder().state_dict()},
            path,
        )

        model = TinyUSFMBackbone(
            checkpoint_path=str(path), output_intermediate_features=False
        )

        assert model.feature_dim == _EMBED_DIM

    def test_load_checkpoint_missing_core_keys_raises(self, tmp_path: Path) -> None:
        """Absent core encoder keys raise RuntimeError."""
        path = tmp_path / "TinyUSFM.pth"
        _save_checkpoint(path, drop_blocks=True)

        with pytest.raises(RuntimeError, match="Core encoder keys missing"):
            TinyUSFMBackbone(
                checkpoint_path=str(path), output_intermediate_features=False
            )
