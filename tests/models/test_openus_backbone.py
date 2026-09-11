"""Unit tests for OpenUSBackbone."""

from collections.abc import Iterator
from unittest.mock import patch

import pytest
import torch

from ultrabench2.models.openus import openus_backbone, openus_vmamba
from ultrabench2.models.openus.openus_backbone import OpenUSBackbone

_BATCH = 2
_IMG_SIZE = 224
_FEATURE_DIM = 768
_INTERMEDIATE_DIMS = [96, 192, 384, 768]
_NUM_STAGES = len(_INTERMEDIATE_DIMS)


@pytest.fixture(autouse=True)
def use_pytorch_selective_scan() -> Iterator[None]:
    """Force the pure-PyTorch selective scan for the whole module.

    Marking resolution as already done with no kernel available stops the
    forward pass from lazily reaching for a CUDA-only kernel it cannot use.
    """
    openus_vmamba._selective_scan_kernel = None
    openus_vmamba._selective_scan_kernel_key = (None, "", None)
    yield
    openus_vmamba._selective_scan_kernel_key = None


def _make_backbone(output_intermediate_features: bool = False) -> OpenUSBackbone:
    """Return an OpenUSBackbone with checkpoint and kernel loading bypassed.

    Kernel resolution is patched out so the tests neither reach the Hugging
    Face Hub nor depend on a pre-compiled kernel being present. The forward
    pass therefore runs on the pure-PyTorch selective scan.
    """
    with (
        patch.object(OpenUSBackbone, "_load_checkpoint"),
        patch.object(openus_backbone, "load_selective_scan_kernel"),
    ):
        return OpenUSBackbone(
            checkpoint_path="",
            output_intermediate_features=output_intermediate_features,
        )


class TestOpenUSBackboneAttributes:
    """Tests for backbone metadata attributes."""

    def test_attributes(self) -> None:
        model = _make_backbone()
        assert model.feature_dim == _FEATURE_DIM
        assert model.output_type == "multi_scale"
        assert model.has_cls_token is False
        assert model.block_indices is None
        assert model.intermediate_feature_dims == _INTERMEDIATE_DIMS


class TestOpenUSBackboneKernelConfig:
    """Tests for the selective scan kernel configuration."""

    def test_kernel_config_is_forwarded(self) -> None:
        with (
            patch.object(OpenUSBackbone, "_load_checkpoint"),
            patch.object(openus_backbone, "load_selective_scan_kernel") as load_kernel,
        ):
            OpenUSBackbone(
                checkpoint_path="",
                output_intermediate_features=False,
                selective_scan_kernel_path="/kernels/mamba-ssm",
                selective_scan_kernel_repo_id="org/kernel",
                selective_scan_kernel_version=3,
            )

        load_kernel.assert_called_once_with(
            local_kernel_path="/kernels/mamba-ssm",
            repo_id="org/kernel",
            version=3,
        )


class TestOpenUSBackboneForward:
    """Tests for forward pass output shapes."""

    def test_forward_features_shape(self) -> None:
        model = _make_backbone()
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape[0] == _BATCH
        assert out.features.shape[1] == _FEATURE_DIM
        assert out.intermediate_features is None

    def test_forward_no_intermediate_when_disabled(self) -> None:
        model = _make_backbone(output_intermediate_features=False)
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.intermediate_features is None

    def test_forward_intermediate_length(self) -> None:
        model = _make_backbone(output_intermediate_features=True)
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.intermediate_features is not None
        assert len(out.intermediate_features) == _NUM_STAGES

    def test_forward_intermediate_channel_dims(self) -> None:
        model = _make_backbone(output_intermediate_features=True)
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.intermediate_features is not None
        for stage_out, expected_channels in zip(
            out.intermediate_features, _INTERMEDIATE_DIMS
        ):
            assert stage_out.shape[0] == _BATCH
            assert stage_out.shape[1] == expected_channels

    def test_features_matches_last_intermediate(self) -> None:
        model = _make_backbone(output_intermediate_features=True)
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x)
        assert out.intermediate_features is not None
        assert torch.equal(out.features, out.intermediate_features[-1])

    def test_forward_accepts_non_square_input(self) -> None:
        model = _make_backbone()
        x = torch.randint(0, 256, (_BATCH, 3, 128, 192), dtype=torch.uint8)
        out = model(x)
        assert out.features.shape[0] == _BATCH
        assert out.features.shape[1] == _FEATURE_DIM

    def test_forward_pool_shape(self) -> None:
        model = _make_backbone()
        x = torch.randint(0, 256, (_BATCH, 3, _IMG_SIZE, _IMG_SIZE), dtype=torch.uint8)
        out = model(x, pool=True)
        assert out.features.shape == (_BATCH, _FEATURE_DIM)
