"""Unit tests for the OpenUS VMamba cross-scan/cross-merge ops.

These ops have a triton-accelerated implementation and a pure-PyTorch
fallback. The fallback is what runs wherever triton is unavailable (e.g.
compute clusters whose torch wheels do not bundle triton), so it must
produce gradients with the same shapes and values as the triton path.
"""

import torch

from ultrabench.models.openus.openus_vmamba import cross_merge_fn, cross_scan_fn

_BATCH = 2
_GROUPS = 4
_CHANNELS = 8
_HEIGHT = 7
_WIDTH = 5


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
