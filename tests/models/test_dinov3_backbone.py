"""Unit tests for the DINOv3 backbones.

``torch.hub.load`` both builds the model and loads gated weights (requiring
network access and downloaded checkpoints), so it is mocked here with a
lightweight fake module that mimics the DINOv3 ViT / ConvNeXt
feature-extraction interface with the expected output shapes.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from ultrabench2.models.dinov3.dinov3_backbone import (
    DINOv3ConvNeXtBackbone,
    DINOv3ViTBackbone,
)

_BATCH = 2
_IMG_SIZE = 224
_PATCH_SIZE = 16
_NUM_PATCHES = (_IMG_SIZE // _PATCH_SIZE) ** 2  # 196


class _FakeViT(nn.Module):
    """Minimal stand-in for a DINOv3 ViT hub model."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim

    def _num_tokens(self, x: torch.Tensor) -> int:
        return (x.shape[-2] // _PATCH_SIZE) * (x.shape[-1] // _PATCH_SIZE)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        n = self._num_tokens(x)
        return {
            "x_norm_patchtokens": torch.zeros(x.shape[0], n, self.feature_dim),
            "x_norm_clstoken": torch.zeros(x.shape[0], self.feature_dim),
        }

    def get_intermediate_layers(
        self, x: torch.Tensor, n, reshape=False, return_class_token=False, norm=True
    ):
        num_tokens = self._num_tokens(x)
        return tuple(torch.zeros(x.shape[0], num_tokens, self.feature_dim) for _ in n)


class _FakeConvNeXt(nn.Module):
    """Minimal stand-in for a DINOv3 ConvNeXt hub model."""

    def __init__(self, stage_dims: list[int]) -> None:
        super().__init__()
        self.stage_dims = stage_dims

    def get_intermediate_layers(
        self, x: torch.Tensor, n, reshape=True, return_class_token=False, norm=True
    ):
        b, _, h, w = x.shape
        maps = []
        for i, idx in enumerate(n):
            stride = 4 * (2**i)  # 4, 8, 16, 32
            maps.append(torch.zeros(b, self.stage_dims[idx], h // stride, w // stride))
        return tuple(maps)


def _make_vit(
    variant: str = "dinov3_vitb16",
    output_intermediate_features: bool = False,
) -> DINOv3ViTBackbone:
    """Return a DINOv3ViTBackbone with torch.hub.load mocked."""
    feature_dim = 768 if variant == "dinov3_vitb16" else 1024
    with (
        patch("torch.hub.load", return_value=_FakeViT(feature_dim)),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_dir", return_value=True),
    ):
        return DINOv3ViTBackbone(
            model_variant=variant,
            repo_dir="dummy-repo",
            checkpoint_path="dummy.pth",
            output_intermediate_features=output_intermediate_features,
        )


def _make_convnext(
    variant: str = "dinov3_convnext_base",
    output_intermediate_features: bool = False,
) -> DINOv3ConvNeXtBackbone:
    """Return a DINOv3ConvNeXtBackbone with torch.hub.load mocked."""
    dims = (
        [128, 256, 512, 1024]
        if variant == "dinov3_convnext_base"
        else [192, 384, 768, 1536]
    )
    with (
        patch("torch.hub.load", return_value=_FakeConvNeXt(dims)),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.is_dir", return_value=True),
    ):
        return DINOv3ConvNeXtBackbone(
            model_variant=variant,
            repo_dir="dummy-repo",
            checkpoint_path="dummy.pth",
            output_intermediate_features=output_intermediate_features,
        )


class TestDINOv3ViTBackboneAttributes:
    """Tests for ViT backbone metadata attributes."""

    def test_vitb16_attributes(self) -> None:
        model = _make_vit("dinov3_vitb16")
        assert model.feature_dim == 768
        assert model.output_type == "patch_tokens"
        assert model.has_cls_token is True
        assert model.block_indices == [2, 5, 8, 11]
        assert model.intermediate_feature_dims == [768] * 4

    def test_vitl16_attributes(self) -> None:
        model = _make_vit("dinov3_vitl16")
        assert model.feature_dim == 1024
        assert model.output_type == "patch_tokens"
        assert model.has_cls_token is True
        assert model.block_indices == [5, 11, 17, 23]
        assert model.intermediate_feature_dims == [1024] * 4

    def test_unsupported_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported DINOv3 ViT variant"):
            _make_vit("dinov3_vits16")

    def test_missing_checkpoint_raises(self) -> None:
        with patch("torch.hub.load", return_value=_FakeViT(768)):
            with pytest.raises(
                FileNotFoundError, match="Could not find DINOv3 weights"
            ):
                DINOv3ViTBackbone(
                    model_variant="dinov3_vitb16",
                    repo_dir="dummy-repo",
                    checkpoint_path="/nonexistent/does_not_exist.pth",
                    output_intermediate_features=False,
                )

    def test_missing_repo_raises(self) -> None:
        with (
            patch("torch.hub.load", return_value=_FakeViT(768)),
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=False),
        ):
            with pytest.raises(
                FileNotFoundError, match="Could not find local DINOv3 repository"
            ):
                DINOv3ViTBackbone(
                    model_variant="dinov3_vitb16",
                    repo_dir="/nonexistent/dinov3-repo",
                    checkpoint_path="dummy.pth",
                    output_intermediate_features=False,
                )


class TestDINOv3LocalRepoLoading:
    """Tests that the encoder is loaded from a local repo clone."""

    def test_hub_load_uses_local_source(self) -> None:
        with (
            patch("torch.hub.load", return_value=_FakeViT(768)) as mock_load,
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=True),
        ):
            DINOv3ViTBackbone(
                model_variant="dinov3_vitb16",
                repo_dir="/local/dinov3-repo",
                checkpoint_path="dummy.pth",
                output_intermediate_features=False,
            )

        args, kwargs = mock_load.call_args
        assert args[0] == "/local/dinov3-repo"
        assert args[1] == "dinov3_vitb16"
        assert kwargs["source"] == "local"
        assert kwargs["weights"] == "dummy.pth"


class TestDINOv3ViTBackboneForward:
    """Forward-pass tests for the ViT variants."""

    @pytest.mark.parametrize(
        ("variant", "feature_dim"),
        [("dinov3_vitb16", 768), ("dinov3_vitl16", 1024)],
    )
    def test_forward_features_shape(self, variant: str, feature_dim: int) -> None:
        model = _make_vit(variant)
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, feature_dim)
        assert out.intermediate_features is None

    def test_forward_pool_shape(self) -> None:
        model = _make_vit("dinov3_vitb16")
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x, pool=True)
        assert out.features.shape == (_BATCH, 768)

    def test_forward_intermediate_shape(self) -> None:
        model = _make_vit("dinov3_vitb16", output_intermediate_features=True)
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.intermediate_features is not None
        assert len(out.intermediate_features) == 4
        for tensor in out.intermediate_features:
            assert tensor.shape == (_BATCH, _NUM_PATCHES, 768)

    def test_forward_accepts_non_square_input(self) -> None:
        model = _make_vit("dinov3_vitb16")
        x = torch.randint(0, 256, (_BATCH, 3, 128, 192), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, _NUM_PATCHES, 768)


class TestDINOv3ConvNeXtBackboneAttributes:
    """Tests for ConvNeXt backbone metadata attributes."""

    def test_convnext_base_attributes(self) -> None:
        model = _make_convnext("dinov3_convnext_base")
        assert model.feature_dim == 1024
        assert model.output_type == "multi_scale"
        assert model.has_cls_token is False
        assert model.block_indices is None
        assert model.intermediate_feature_dims == [128, 256, 512, 1024]

    def test_convnext_large_attributes(self) -> None:
        model = _make_convnext("dinov3_convnext_large")
        assert model.feature_dim == 1536
        assert model.output_type == "multi_scale"
        assert model.intermediate_feature_dims == [192, 384, 768, 1536]

    def test_unsupported_variant_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported DINOv3 ConvNeXt variant"):
            _make_convnext("dinov3_convnext_tiny")


class TestDINOv3ConvNeXtBackboneForward:
    """Forward-pass tests for the ConvNeXt variants."""

    def test_forward_features_shape(self) -> None:
        model = _make_convnext("dinov3_convnext_base")
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape == (_BATCH, 1024, 7, 7)
        assert out.intermediate_features is None

    def test_forward_pool_shape(self) -> None:
        model = _make_convnext("dinov3_convnext_base")
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x, pool=True)
        assert out.features.shape == (_BATCH, 1024)

    def test_forward_intermediate_shape(self) -> None:
        model = _make_convnext(
            "dinov3_convnext_base", output_intermediate_features=True
        )
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.intermediate_features is not None
        assert len(out.intermediate_features) == 4
        expected = [
            (_BATCH, 128, 56, 56),
            (_BATCH, 256, 28, 28),
            (_BATCH, 512, 14, 14),
            (_BATCH, 1024, 7, 7),
        ]
        for tensor, shape in zip(out.intermediate_features, expected):
            assert tensor.shape == shape
