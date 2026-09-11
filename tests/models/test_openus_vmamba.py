"""Unit tests for the OpenUS VMamba scan ops and kernel resolution.

The cross-scan/cross-merge ops have a triton-accelerated implementation and
a pure-PyTorch fallback. The fallback is what runs wherever triton is
unavailable (e.g. compute clusters whose torch wheels do not bundle triton),
so it must produce gradients with the same shapes and values as the triton
path.

The selective scan resolves a pre-compiled HF Kernels build when it can and
falls back to a pure-PyTorch implementation otherwise, so both the
resolution order and the fallback dispatch are covered here.
"""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from ultrabench2.models.openus import openus_vmamba
from ultrabench2.models.openus.openus_vmamba import (
    _selective_scan_torch,
    cross_merge_fn,
    cross_scan_fn,
    load_selective_scan_kernel,
    selective_scan_fn,
)

_BATCH = 2
_GROUPS = 4
_CHANNELS = 8
_HEIGHT = 7
_WIDTH = 5

_STATES = 3
_INNER = 6
_LENGTH = 5


@pytest.fixture(autouse=True)
def reset_kernel_cache() -> Iterator[None]:
    """Clear the cached selective scan kernel around every test."""
    openus_vmamba._selective_scan_kernel = None
    openus_vmamba._selective_scan_kernel_key = None
    yield
    openus_vmamba._selective_scan_kernel = None
    openus_vmamba._selective_scan_kernel_key = None


def _scan_inputs() -> tuple[torch.Tensor, ...]:
    """Build CPU selective scan inputs shaped like the real call site.

    Returns:
        The (u, delta, A, B, C, D, delta_bias) tensors expected by
        selective_scan_fn.
    """
    k_dim = _GROUPS * _INNER
    return (
        torch.randn(_BATCH, k_dim, _LENGTH),
        torch.randn(_BATCH, k_dim, _LENGTH),
        -torch.rand(k_dim, _STATES),
        torch.randn(_BATCH, _GROUPS, _STATES, _LENGTH),
        torch.randn(_BATCH, _GROUPS, _STATES, _LENGTH),
        torch.rand(k_dim),
        torch.rand(k_dim),
    )


class TestCrossScanBackward:
    """Tests for the pure-PyTorch cross-scan backward pass."""

    def test_gradient_shape_channel_first(self) -> None:
        x = torch.randn(_BATCH, _CHANNELS, _HEIGHT, _WIDTH, requires_grad=True)
        cross_scan_fn(x, force_torch=True).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_shape_channel_last(self) -> None:
        x = torch.randn(_BATCH, _HEIGHT, _WIDTH, _CHANNELS, requires_grad=True)
        cross_scan_fn(
            x, in_channel_first=False, out_channel_first=False, force_torch=True
        ).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestCrossMergeBackward:
    """Tests for the pure-PyTorch cross-merge backward pass."""

    def test_gradient_shape_channel_first(self) -> None:
        ys = torch.randn(
            _BATCH, _GROUPS, _CHANNELS, _HEIGHT, _WIDTH, requires_grad=True
        )
        cross_merge_fn(ys, force_torch=True).sum().backward()
        assert ys.grad is not None
        assert ys.grad.shape == ys.shape

    def test_gradient_shape_channel_last(self) -> None:
        ys = torch.randn(
            _BATCH, _HEIGHT, _WIDTH, _GROUPS, _CHANNELS, requires_grad=True
        )
        cross_merge_fn(
            ys, in_channel_first=False, out_channel_first=False, force_torch=True
        ).sum().backward()
        assert ys.grad is not None
        assert ys.grad.shape == ys.shape

    def test_gradient_values_match_cross_scan(self) -> None:
        """Cross-merge is linear, so its backward is the cross-scan forward."""
        ys = torch.randn(
            _BATCH,
            _GROUPS,
            _CHANNELS,
            _HEIGHT,
            _WIDTH,
            dtype=torch.double,
            requires_grad=True,
        )
        grad_out = torch.randn(_BATCH, _CHANNELS, _HEIGHT * _WIDTH, dtype=torch.double)
        cross_merge_fn(ys, force_torch=True).backward(grad_out)

        expected = cross_scan_fn(
            grad_out.view(_BATCH, _CHANNELS, _HEIGHT, _WIDTH), force_torch=True
        ).view(_BATCH, _GROUPS, _CHANNELS, _HEIGHT, _WIDTH)
        assert ys.grad is not None
        assert torch.allclose(ys.grad, expected)

    def test_backward_through_scan_merge_roundtrip(self) -> None:
        """Full-finetuning backprops through a scan followed by a merge."""
        x = torch.randn(_BATCH, _CHANNELS, _HEIGHT, _WIDTH, requires_grad=True)
        xs = cross_scan_fn(x, force_torch=True)
        ys = xs.view(_BATCH, _GROUPS, _CHANNELS, _HEIGHT, _WIDTH)
        cross_merge_fn(ys, force_torch=True).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestLoadSelectiveScanKernel:
    """Tests for the selective scan kernel resolution order."""

    def test_local_kernel_preferred_over_hub(self) -> None:
        local = MagicMock()
        with (
            patch.object(openus_vmamba, "get_local_kernel", return_value=local) as gl,
            patch.object(openus_vmamba, "get_kernel") as gk,
        ):
            resolved = load_selective_scan_kernel("/kernels/mamba-ssm")

        assert resolved is local.selective_scan_fn
        gl.assert_called_once_with(Path("/kernels/mamba-ssm"))
        gk.assert_not_called()

    def test_no_path_skips_local_kernel(self) -> None:
        hub = MagicMock()
        with (
            patch.object(openus_vmamba, "get_local_kernel") as gl,
            patch.object(openus_vmamba, "get_kernel", return_value=hub) as gk,
        ):
            resolved = load_selective_scan_kernel(None, "org/kernel", 3)

        assert resolved is hub.selective_scan_fn
        gl.assert_not_called()
        gk.assert_called_once_with("org/kernel", version=3)

    def test_falls_back_to_hub_when_local_kernel_missing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        hub = MagicMock()
        with (
            patch.object(
                openus_vmamba,
                "get_local_kernel",
                side_effect=FileNotFoundError("no kernel"),
            ),
            patch.object(openus_vmamba, "get_kernel", return_value=hub),
        ):
            resolved = load_selective_scan_kernel("/missing")

        assert resolved is hub.selective_scan_fn
        assert "Could not load the local selective scan kernel" in caplog.text

    def test_falls_back_to_pytorch_when_both_unavailable(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            patch.object(
                openus_vmamba, "get_local_kernel", side_effect=FileNotFoundError
            ),
            patch.object(openus_vmamba, "get_kernel", side_effect=OSError("offline")),
        ):
            resolved = load_selective_scan_kernel("/missing")

        assert resolved is None
        assert "pure-PyTorch selective scan" in caplog.text

    def test_resolution_is_cached_per_configuration(self) -> None:
        with patch.object(
            openus_vmamba, "get_local_kernel", return_value=MagicMock()
        ) as gl:
            first = load_selective_scan_kernel("/kernels/mamba-ssm")
            second = load_selective_scan_kernel("/kernels/mamba-ssm")
            load_selective_scan_kernel("/kernels/other")

        assert first is second
        assert gl.call_count == 2


class TestSelectiveScanDispatch:
    """Tests for the per-call selective scan backend dispatch."""

    def test_cpu_inputs_bypass_the_kernel(self) -> None:
        """The pre-compiled kernels are CUDA-only, so CPU must use torch."""
        kernel = MagicMock()
        openus_vmamba._selective_scan_kernel = kernel
        openus_vmamba._selective_scan_kernel_key = (None, "org/kernel", 1)

        inputs = _scan_inputs()
        out = selective_scan_fn(*inputs)

        kernel.assert_not_called()
        assert torch.equal(out, _selective_scan_torch(*inputs))

    def test_kernel_is_resolved_on_first_call(self) -> None:
        """Callers that never configure a kernel still get a working scan."""
        with patch.object(openus_vmamba, "load_selective_scan_kernel") as load_kernel:
            selective_scan_fn(*_scan_inputs())

        load_kernel.assert_called_once_with()

    def test_output_shape_matches_input_sequence(self) -> None:
        # Mark resolution as already done with no kernel available, so the
        # test neither reaches the Hugging Face Hub nor needs a kernel.
        openus_vmamba._selective_scan_kernel_key = (None, "org/kernel", 1)
        out = selective_scan_fn(*_scan_inputs())
        assert out.shape == (_BATCH, _GROUPS * _INNER, _LENGTH)
