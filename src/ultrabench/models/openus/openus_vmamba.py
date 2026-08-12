"""Standalone OpenUS model — no repo imports required.

All architecture code (VMamba backbone, scan ops) is inlined here.
External pip dependencies: torch, numpy, timm==0.4.12, kernels

The selective scan is provided by the pre-compiled `mamba-ssm` HF Kernels
build when one can be resolved, and by a pure-PyTorch implementation
otherwise. See `load_selective_scan_kernel`.

Usage:
    model = build_openus_model("openus_cpt0150.pth")
    model.eval()
    out = model(imgs)   # [B, L+1, C] — cls token + spatial tokens
    # or with seg_head=True:  list of 4 feature maps [B, C_i, H_i, W_i]
"""

import contextlib
import logging
import math
import warnings
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from kernels import get_kernel, get_local_kernel
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

logger = logging.getLogger(__name__)

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

DEFAULT_KERNEL_REPO_ID = "kernels-community/mamba-ssm"
DEFAULT_KERNEL_VERSION = 1

# ============================================================
# Optional triton extension
# ============================================================
_WITH_TRITON = True
try:
    import triton
    import triton.language as tl
except Exception:
    _WITH_TRITON = False
    warnings.warn("triton not installed — falling back to pure-PyTorch scan ops.")

try:
    from analyze.attnmap import AttnMamba as _AttnMamba
except Exception:
    _AttnMamba = None


# ============================================================
# Cross-scan helpers  (from vmamba_models/csm_triton.py)
# ============================================================


def _cross_scan_fwd(
    x: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0
):
    if in_channel_first:
        B, C, H, W = x.shape
        if scans == 0:
            y = x.new_empty((B, 4, C, H * W))
            y[:, 0] = x.flatten(2, 3)
            y[:, 1] = x.transpose(2, 3).flatten(2, 3)
            y[:, 2:4] = torch.flip(y[:, 0:2], dims=[-1])
        elif scans == 1:
            y = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        elif scans == 2:
            y = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
            y = torch.cat([y, y.flip(dims=[-1])], dim=1)
    else:
        B, H, W, C = x.shape
        if scans == 0:
            y = x.new_empty((B, H * W, 4, C))
            y[:, :, 0] = x.flatten(1, 2)
            y[:, :, 1] = x.transpose(1, 2).flatten(1, 2)
            y[:, :, 2:4] = torch.flip(y[:, :, 0:2], dims=[1])
        elif scans == 1:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 4, 1)
        elif scans == 2:
            y = x.view(B, H * W, 1, C).repeat(1, 1, 2, 1)
            y = torch.cat([y, y.flip(dims=[1])], dim=2)
    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 3, 1, 2).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 3, 1).contiguous()
    return y


def _cross_merge_fwd(
    y: torch.Tensor, in_channel_first=True, out_channel_first=True, scans=0
):
    if out_channel_first:
        B, K, D, H, W = y.shape
        y = y.view(B, K, D, -1)
        if scans == 0:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y[:, 0] + y[:, 1].view(B, -1, W, H).transpose(2, 3).contiguous().view(
                B, D, -1
            )
        elif scans == 1:
            y = y.sum(1)
        elif scans == 2:
            y = y[:, 0:2] + y[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = y.sum(1)
    else:
        B, H, W, K, D = y.shape
        y = y.view(B, -1, K, D)
        if scans == 0:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y[:, :, 0] + y[:, :, 1].view(B, W, H, -1).transpose(
                1, 2
            ).contiguous().view(B, -1, D)
        elif scans == 1:
            y = y.sum(2)
        elif scans == 2:
            y = y[:, :, 0:2] + y[:, :, 2:4].flip(dims=[1]).view(B, -1, 2, D)
            y = y.sum(2)
    if in_channel_first and (not out_channel_first):
        y = y.permute(0, 2, 1).contiguous()
    elif (not in_channel_first) and out_channel_first:
        y = y.permute(0, 2, 1).contiguous()
    return y


class _CrossScanF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, in_channel_first=True, out_channel_first=True, one_by_one=False, scans=0
    ):
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        if one_by_one:
            B, K, C, H, W = (
                x.shape
                if in_channel_first
                else (*x.shape[:1], *x.shape[-2:], *x.shape[1:3])
            )
        else:
            B, C, H, W = (
                x.shape
                if in_channel_first
                else (x.shape[0], x.shape[3], x.shape[1], x.shape[2])
            )
        ctx.shape = (B, C, H, W)
        return _cross_scan_fwd(x, in_channel_first, out_channel_first, scans)

    @staticmethod
    def backward(ctx, ys):
        in_cf = ctx.in_channel_first
        out_cf = ctx.out_channel_first
        scans = ctx.scans
        B, C, H, W = ctx.shape
        ys = ys.view(B, -1, C, H, W) if out_cf else ys.view(B, H, W, -1, C)
        y = _cross_merge_fwd(ys, in_cf, out_cf, scans)
        y = y.view(B, -1, H, W) if in_cf else y.view(B, H, W, -1)
        return y, None, None, None, None


class _CrossMergeF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ys,
        in_channel_first=True,
        out_channel_first=True,
        one_by_one=False,
        scans=0,
    ):
        ctx.in_channel_first = in_channel_first
        ctx.out_channel_first = out_channel_first
        ctx.one_by_one = one_by_one
        ctx.scans = scans
        B, K, C, H, W = (
            ys.shape
            if out_channel_first
            else (ys.shape[0], ys.shape[3], ys.shape[4], ys.shape[1], ys.shape[2])
        )
        ctx.shape = (B, C, H, W)
        return _cross_merge_fwd(ys, in_channel_first, out_channel_first, scans)

    @staticmethod
    def backward(ctx, x):
        in_cf = ctx.in_channel_first
        out_cf = ctx.out_channel_first
        scans = ctx.scans
        B, C, H, W = ctx.shape
        y = _cross_scan_fwd(
            x.view(B, -1, H, W) if in_cf else x.view(B, H, W, -1),
            in_cf,
            out_cf,
            scans,
        )
        # _cross_scan_fwd flattens the spatial dims; restore the (H, W) layout
        # of the forward input so autograd sees a matching gradient shape.
        y = y.view(B, -1, C, H, W) if out_cf else y.view(B, H, W, -1, C)
        return y, None, None, None, None


# Triton-accelerated versions (only if triton is available + CUDA tensor)
if _WITH_TRITON:

    @triton.jit
    def _triton_cross_scan_flex(
        x: tl.tensor,
        y: tl.tensor,
        x_layout: tl.constexpr,
        y_layout: tl.constexpr,
        operation: tl.constexpr,
        onebyone: tl.constexpr,
        scans: tl.constexpr,
        BC: tl.constexpr,
        BH: tl.constexpr,
        BW: tl.constexpr,
        DC: tl.constexpr,
        DH: tl.constexpr,
        DW: tl.constexpr,
        NH: tl.constexpr,
        NW: tl.constexpr,
    ):
        i_hw, i_c, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        i_h, i_w = (i_hw // NW), (i_hw % NW)
        _mask_h = (i_h * BH + tl.arange(0, BH)) < DH
        _mask_w = (i_w * BW + tl.arange(0, BW)) < DW
        _mask_hw = _mask_h[:, None] & _mask_w[None, :]
        _for_C = min(DC - i_c * BC, BC)
        pos_h = i_h * BH + tl.arange(0, BH)[:, None]
        pos_w = i_w * BW + tl.arange(0, BW)[None, :]
        neg_h = DH - i_h * BH - 1 - tl.arange(0, BH)[:, None]
        neg_w = DW - i_w * BW - 1 - tl.arange(0, BW)[None, :]
        if scans == 0:
            HWRoute0 = pos_h * DW + pos_w
            HWRoute1 = pos_w * DH + pos_h
            HWRoute2 = neg_h * DW + neg_w
            HWRoute3 = neg_w * DH + neg_h
        elif scans == 1:
            HWRoute0 = pos_h * DW + pos_w
            HWRoute1 = HWRoute0
            HWRoute2 = HWRoute0
            HWRoute3 = HWRoute0
        elif scans == 2:
            HWRoute0 = pos_h * DW + pos_w
            HWRoute1 = HWRoute0
            HWRoute2 = neg_h * DW + neg_w
            HWRoute3 = HWRoute2
        _tmp1 = DC * DH * DW
        y_ptr_base = (
            y + i_b * 4 * _tmp1 + (i_c * BC * DH * DW if y_layout == 0 else i_c * BC)
        )
        if y_layout == 0:
            p_y1 = y_ptr_base + HWRoute0
            p_y2 = y_ptr_base + _tmp1 + HWRoute1
            p_y3 = y_ptr_base + 2 * _tmp1 + HWRoute2
            p_y4 = y_ptr_base + 3 * _tmp1 + HWRoute3
        else:
            p_y1 = y_ptr_base + HWRoute0 * 4 * DC
            p_y2 = y_ptr_base + DC + HWRoute1 * 4 * DC
            p_y3 = y_ptr_base + 2 * DC + HWRoute2 * 4 * DC
            p_y4 = y_ptr_base + 3 * DC + HWRoute3 * 4 * DC
        if onebyone == 0:
            x_ptr_base = (
                x + i_b * _tmp1 + (i_c * BC * DH * DW if x_layout == 0 else i_c * BC)
            )
            p_x = x_ptr_base + (HWRoute0 if x_layout == 0 else HWRoute0 * DC)
            if operation == 0:
                for idxc in range(_for_C):
                    _ix = idxc * DH * DW if x_layout == 0 else idxc
                    _iy = idxc * DH * DW if y_layout == 0 else idxc
                    _x = tl.load(p_x + _ix, mask=_mask_hw)
                    tl.store(p_y1 + _iy, _x, mask=_mask_hw)
                    tl.store(p_y2 + _iy, _x, mask=_mask_hw)
                    tl.store(p_y3 + _iy, _x, mask=_mask_hw)
                    tl.store(p_y4 + _iy, _x, mask=_mask_hw)
            else:
                for idxc in range(_for_C):
                    _ix = idxc * DH * DW if x_layout == 0 else idxc
                    _iy = idxc * DH * DW if y_layout == 0 else idxc
                    tl.store(
                        p_x + _ix,
                        tl.load(p_y1 + _iy, mask=_mask_hw)
                        + tl.load(p_y2 + _iy, mask=_mask_hw)
                        + tl.load(p_y3 + _iy, mask=_mask_hw)
                        + tl.load(p_y4 + _iy, mask=_mask_hw),
                        mask=_mask_hw,
                    )

    class _CrossScanTritonF(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            x,
            in_channel_first=True,
            out_channel_first=True,
            one_by_one=False,
            scans=0,
        ):
            B, C, H, W = (
                x.shape
                if in_channel_first
                else (x.shape[0], x.shape[3], x.shape[1], x.shape[2])
            )
            B, C, H, W = int(B), int(C), int(H), int(W)
            BC, BH, BW = 1, 32, 32
            NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
            ctx.in_channel_first = in_channel_first
            ctx.out_channel_first = out_channel_first
            ctx.one_by_one = one_by_one
            ctx.scans = scans
            ctx.shape = (B, C, H, W)
            ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
            y = (
                x.new_empty((B, 4, C, H * W))
                if out_channel_first
                else x.new_empty((B, H * W, 4, C))
            )
            _triton_cross_scan_flex[(NH * NW, NC, B)](
                x.contiguous(),
                y,
                (0 if in_channel_first else 1),
                (0 if out_channel_first else 1),
                0,
                0,
                scans,
                BC,
                BH,
                BW,
                C,
                H,
                W,
                NH,
                NW,
            )
            return y

        @staticmethod
        def backward(ctx, y):
            in_cf = ctx.in_channel_first
            out_cf = ctx.out_channel_first
            scans = ctx.scans
            B, C, H, W = ctx.shape
            BC, BH, BW, NC, NH, NW = ctx.triton_shape
            x = y.new_empty((B, C, H, W)) if in_cf else y.new_empty((B, H, W, C))
            _triton_cross_scan_flex[(NH * NW, NC, B)](
                x,
                y.contiguous(),
                (0 if in_cf else 1),
                (0 if out_cf else 1),
                1,
                0,
                scans,
                BC,
                BH,
                BW,
                C,
                H,
                W,
                NH,
                NW,
            )
            return x, None, None, None, None

    class _CrossMergeTritonF(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            y,
            in_channel_first=True,
            out_channel_first=True,
            one_by_one=False,
            scans=0,
        ):
            B, K, C, H, W = (
                y.shape
                if out_channel_first
                else (y.shape[0], y.shape[3], y.shape[4], y.shape[1], y.shape[2])
            )
            B, C, H, W = int(B), int(C), int(H), int(W)
            BC, BH, BW = 1, 32, 32
            NH, NW, NC = triton.cdiv(H, BH), triton.cdiv(W, BW), triton.cdiv(C, BC)
            ctx.in_channel_first = in_channel_first
            ctx.out_channel_first = out_channel_first
            ctx.one_by_one = one_by_one
            ctx.scans = scans
            ctx.shape = (B, C, H, W)
            ctx.triton_shape = (BC, BH, BW, NC, NH, NW)
            x = (
                y.new_empty((B, C, H * W))
                if in_channel_first
                else y.new_empty((B, H * W, C))
            )
            _triton_cross_scan_flex[(NH * NW, NC, B)](
                x,
                y.contiguous(),
                (0 if in_channel_first else 1),
                (0 if out_channel_first else 1),
                1,
                0,
                scans,
                BC,
                BH,
                BW,
                C,
                H,
                W,
                NH,
                NW,
            )
            return x

        @staticmethod
        def backward(ctx, x):
            in_cf = ctx.in_channel_first
            out_cf = ctx.out_channel_first
            scans = ctx.scans
            B, C, H, W = ctx.shape
            BC, BH, BW, NC, NH, NW = ctx.triton_shape
            y = x.new_empty((B, 4, C, H, W)) if out_cf else x.new_empty((B, H, W, 4, C))
            _triton_cross_scan_flex[(NH * NW, NC, B)](
                x.contiguous(),
                y,
                (0 if in_cf else 1),
                (0 if out_cf else 1),
                0,
                0,
                scans,
                BC,
                BH,
                BW,
                C,
                H,
                W,
                NH,
                NW,
            )
            return y, None, None, None, None


def cross_scan_fn(
    x: torch.Tensor,
    in_channel_first=True,
    out_channel_first=True,
    one_by_one=False,
    scans=0,
    force_torch=False,
):
    if _WITH_TRITON and x.is_cuda and not force_torch:
        CSF = _CrossScanTritonF
    else:
        CSF = _CrossScanF
    ctx = torch.cuda.device(x.device) if x.is_cuda else contextlib.nullcontext()
    with ctx:
        return CSF.apply(x, in_channel_first, out_channel_first, one_by_one, scans)


def cross_merge_fn(
    y: torch.Tensor,
    in_channel_first=True,
    out_channel_first=True,
    one_by_one=False,
    scans=0,
    force_torch=False,
):
    if _WITH_TRITON and y.is_cuda and not force_torch:
        CMF = _CrossMergeTritonF
    else:
        CMF = _CrossMergeF
    ctx = torch.cuda.device(y.device) if y.is_cuda else contextlib.nullcontext()
    with ctx:
        return CMF.apply(y, in_channel_first, out_channel_first, one_by_one, scans)


# ============================================================
# Selective scan  (from vmamba_models/csms6s.py)
# ============================================================


def _selective_scan_torch(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    delta_bias=None,
    delta_softplus=True,
    oflex=True,
    *args,
    **kwargs,
):
    dtype_in = u.dtype
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]
    Cdim = int(KCdim / K)
    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    C = C.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)
    x = A.new_zeros((Batch, KCdim, N))
    ys = []
    for i in range(L):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D.unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)


_selective_scan_kernel: Callable | None = None
_selective_scan_kernel_key: tuple[str | None, str, int | None] | None = None


def load_selective_scan_kernel(
    local_kernel_path: str | Path | None = None,
    repo_id: str = DEFAULT_KERNEL_REPO_ID,
    version: int | None = DEFAULT_KERNEL_VERSION,
) -> Callable | None:
    """Resolve the selective scan kernel and cache it for the module.

    Backends are tried in order of preference:

    1. A pre-compiled HF Kernels build in a local directory, loaded with
       ``kernels.get_local_kernel``. Skipped when no path is given.
    2. The ``repo_id`` kernel from the Hugging Face Hub, loaded with
       ``kernels.get_kernel``. This resolves against the local Hugging
       Face cache when ``HF_HUB_OFFLINE=1`` is set.
    3. The pure-PyTorch scan, which is much slower but runs anywhere.

    Nothing is raised when a backend cannot be loaded: the failure is
    logged and the next backend is tried. Resolution is cached per
    argument combination, so repeated calls with the same arguments are
    free.

    Args:
        local_kernel_path: Directory holding a pre-compiled kernel build.
            May be a kernel repository root, its "build" subdirectory, or
            a single build variant directory. None skips this backend.
        repo_id: Hugging Face Hub repository holding the kernel.
        version: Kernel version to load from the Hub. None loads the
            latest revision.

    Returns:
        The ``selective_scan_fn`` of the resolved kernel, or None when no
        kernel could be loaded and the pure-PyTorch scan should be used.
    """
    global _selective_scan_kernel, _selective_scan_kernel_key

    key = (
        str(local_kernel_path) if local_kernel_path is not None else None,
        repo_id,
        version,
    )
    if key == _selective_scan_kernel_key:
        return _selective_scan_kernel

    kernel_fn = None
    # FileNotFoundError is an OSError; AttributeError covers a kernel
    # build that does not expose selective_scan_fn.
    if local_kernel_path is None:
        logger.debug("No local selective scan kernel path configured.")
    else:
        try:
            kernel_fn = get_local_kernel(Path(local_kernel_path)).selective_scan_fn
            logger.info(
                "Loaded the local selective scan kernel from %s.", local_kernel_path
            )
        except (OSError, ImportError, ValueError, AttributeError) as error:
            logger.warning(
                "Could not load the local selective scan kernel from %s (%s). "
                "Falling back to the %s kernel from the Hugging Face Hub.",
                local_kernel_path,
                error,
                repo_id,
            )

    if kernel_fn is None:
        try:
            kernel_fn = get_kernel(repo_id, version=version).selective_scan_fn
            logger.info(
                "Loaded the %s selective scan kernel (version %s).", repo_id, version
            )
        except (OSError, ImportError, ValueError, AttributeError) as error:
            logger.warning(
                "Could not load the %s selective scan kernel from the Hugging "
                "Face Hub (%s). Falling back to the pure-PyTorch selective "
                "scan, which is substantially slower.",
                repo_id,
                error,
            )

    _selective_scan_kernel = kernel_fn
    _selective_scan_kernel_key = key
    return kernel_fn


def selective_scan_fn(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    delta_bias=None,
    delta_softplus=True,
    oflex=True,
):
    """Run the selective scan on the best available backend.

    The kernel resolved by `load_selective_scan_kernel` is used for CUDA
    inputs. CPU inputs always use the pure-PyTorch scan, because the
    pre-compiled kernels are CUDA-only. When the kernel has not been
    configured yet it is resolved here with the default arguments.

    Args:
        u: Input sequence [B, K * D, L].
        delta: Timestep tensor [B, K * D, L].
        A: State transition matrix [K * D, N].
        B: Input projection [B, K, N, L].
        C: Output projection [B, K, N, L].
        D: Optional skip connection weights [K * D].
        delta_bias: Optional bias added to delta [K * D].
        delta_softplus: Whether to apply softplus to delta.
        oflex: Whether the pure-PyTorch scan may return float32 output
            instead of casting back to the input dtype. Ignored by the
            pre-compiled kernels.

    Returns:
        The scan output [B, K * D, L].
    """
    if _selective_scan_kernel_key is None:
        load_selective_scan_kernel()

    if _selective_scan_kernel is None or not u.is_cuda:
        return _selective_scan_torch(
            u, delta, A, B, C, D, delta_bias, delta_softplus, oflex
        )

    # The kernel takes a gating tensor `z` between `D` and `delta_bias`,
    # so the trailing arguments must be passed by keyword.
    return _selective_scan_kernel(
        u,
        delta,
        A,
        B,
        C,
        D,
        z=None,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
    )


# ============================================================
# VMamba model architecture  (from vmamba_models/dino_vmamba.py)
# ============================================================


class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(
            self.weight.shape
        )
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps
        )
        return x.permute(0, 3, 1, 2)


class PatchEmbed(nn.Module):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x.transpose(1, 2).reshape(B, C, H, W)


class PatchEmbed2D(nn.Module):
    def __init__(
        self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm, channel_first=False):
        super().__init__()
        self.dim = dim
        Linear = Linear2d if channel_first else nn.Linear
        self._pad = self._pad_cf if channel_first else self._pad_cl
        self.reduction = Linear(
            4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False
        )
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _pad_cl(x):
        H, W, _ = x.shape[-3:]
        if W % 2 != 0 or H % 2 != 0:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]
        x1 = x[..., 1::2, 0::2, :]
        x2 = x[..., 0::2, 1::2, :]
        x3 = x[..., 1::2, 1::2, :]
        return torch.cat([x0, x1, x2, x3], -1)

    @staticmethod
    def _pad_cf(x):
        H, W = x.shape[-2:]
        if W % 2 != 0 or H % 2 != 0:
            x = F.pad(x, (0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2]
        x1 = x[..., 1::2, 0::2]
        x2 = x[..., 0::2, 1::2]
        x3 = x[..., 1::2, 1::2]
        return torch.cat([x0, x1, x2, x3], 1)

    def forward(self, x):
        return self.reduction(self.norm(self._pad(x)))


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        channels_first=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class gMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        channels_first=False,
    ):
        super().__init__()
        self.channel_first = channels_first
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        Linear = Linear2d if channels_first else nn.Linear
        self.fc1 = Linear(in_features, 2 * hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
        return self.drop(self.fc2(x * self.act(z)))


class SoftmaxSpatial(nn.Softmax):
    def forward(self, x: torch.Tensor):
        if self.dim == -1:
            B, C, H, W = x.shape
            return super().forward(x.view(B, C, -1)).view(B, C, H, W)
        elif self.dim == 1:
            B, H, W, C = x.shape
            return super().forward(x.view(B, -1, C)).view(B, H, W, C)
        else:
            raise NotImplementedError


class _mamba_init:
    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        A = (
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
            .view(1, -1)
            .repeat(d_inner, 1)
            .contiguous()
        )
        A_log = torch.log(A)
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(
        cls,
        d_state,
        dt_rank,
        d_inner,
        dt_scale,
        dt_init,
        dt_min,
        dt_max,
        dt_init_floor,
        k_group=4,
    ):
        dt_projs = [
            cls.dt_init(
                dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor
            )
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))
        del dt_projs
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias


class SS2Dv2(nn.Module):
    def __init__(
        self,
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        d_conv=3,
        conv_bias=True,
        dropout=0.0,
        bias=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        forward_type="v2",
        channel_first=False,
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.k_group = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = int(
            math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        )
        self.channel_first = channel_first
        self.with_dconv = d_conv > 1
        Linear = Linear2d if channel_first else nn.Linear

        checkpostfix = self.checkpostfix
        self.disable_force32, forward_type = checkpostfix("_no32", forward_type)
        self.oact, forward_type = checkpostfix("_oact", forward_type)
        self.disable_z, forward_type = checkpostfix("_noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("_nozact", forward_type)
        self.out_norm, forward_type = self.get_outnorm(
            forward_type, self.d_inner, channel_first
        )

        FORWARD_TYPES = dict(
            v01=partial(
                self.forward_corev2,
                force_fp32=(not self.disable_force32),
                scan_force_torch=True,
            ),
            v02=partial(
                self.forward_corev2,
                force_fp32=(not self.disable_force32),
            ),
            v03=partial(
                self.forward_corev2,
                force_fp32=(not self.disable_force32),
            ),
            v04=partial(self.forward_corev2, force_fp32=False),
            v05=partial(self.forward_corev2, force_fp32=False, no_einsum=True),
            v051d=partial(
                self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="unidi"
            ),
            v052d=partial(
                self.forward_corev2, force_fp32=False, no_einsum=True, scan_mode="bidi"
            ),
            v052dc=partial(
                self.forward_corev2,
                force_fp32=False,
                no_einsum=True,
                scan_mode="cascade2d",
            ),
            v2=partial(
                self.forward_corev2,
                force_fp32=(not self.disable_force32),
            ),
            v3=partial(self.forward_corev2, force_fp32=False),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)

        d_proj = self.d_inner if self.disable_z else (self.d_inner * 2)
        self.in_proj = Linear(self.d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )
        del self.x_proj

        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if initialize == "v0":
            self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = (
                _mamba_init.init_dt_A_D(
                    self.d_state,
                    self.dt_rank,
                    self.d_inner,
                    dt_scale,
                    dt_init,
                    dt_min,
                    dt_max,
                    dt_init_floor,
                    k_group=self.k_group,
                )
            )
        elif initialize == "v1":
            self.Ds = nn.Parameter(torch.ones(self.k_group * self.d_inner))
            self.A_logs = nn.Parameter(
                torch.randn(self.k_group * self.d_inner, self.d_state)
            )
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.randn(self.k_group, self.d_inner, self.dt_rank)
            )
            self.dt_projs_bias = nn.Parameter(
                0.1 * torch.randn(self.k_group, self.d_inner)
            )
        elif initialize == "v2":
            self.Ds = nn.Parameter(torch.ones(self.k_group * self.d_inner))
            self.A_logs = nn.Parameter(
                torch.zeros(self.k_group * self.d_inner, self.d_state)
            )
            self.dt_projs_weight = nn.Parameter(
                0.1 * torch.rand(self.k_group, self.d_inner, self.dt_rank)
            )
            self.dt_projs_bias = nn.Parameter(
                0.1 * torch.rand(self.k_group, self.d_inner)
            )

    def forward_corev2(
        self,
        x: torch.Tensor = None,
        force_fp32=False,
        ssoflex=True,
        no_einsum=False,
        scan_mode="cross2d",
        scan_force_torch=False,
        **kwargs,
    ):
        _scan_mode = (
            dict(cross2d=0, unidi=1, bidi=2, cascade2d=-1).get(scan_mode, None)
            if isinstance(scan_mode, str)
            else scan_mode
        )
        assert isinstance(_scan_mode, int)
        delta_softplus = True
        out_norm = self.out_norm
        channel_first = self.channel_first

        def to_fp32(*args):
            return (_a.to(torch.float32) for _a in args)

        B, D, H, W = x.shape
        N = self.d_state
        K, D, R = self.k_group, self.d_inner, self.dt_rank
        L = H * W

        def selective_scan(
            u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True
        ):
            return selective_scan_fn(
                u, delta, A, B, C, D, delta_bias, delta_softplus, ssoflex
            )

        if _scan_mode == -1:
            x_proj_bias = getattr(self, "x_proj_bias", None)

            def scan_rowcol(
                x, proj_weight, proj_bias, dt_weight, dt_bias, _As, _Ds, width=True
            ):
                XB, XD, XH, XW = x.shape
                if width:
                    _B, _D, _L = XB * XH, XD, XW
                    xs = x.permute(0, 2, 1, 3).contiguous()
                else:
                    _B, _D, _L = XB * XW, XD, XH
                    xs = x.permute(0, 3, 1, 2).contiguous()
                xs = torch.stack([xs, xs.flip(dims=[-1])], dim=2)
                if no_einsum:
                    x_dbl = F.conv1d(
                        xs.view(_B, -1, _L),
                        proj_weight.view(-1, _D, 1),
                        bias=(proj_bias.view(-1) if proj_bias is not None else None),
                        groups=2,
                    )
                    dts, Bs, Cs = torch.split(
                        x_dbl.view(_B, 2, -1, _L), [R, N, N], dim=2
                    )
                    dts = F.conv1d(
                        dts.contiguous().view(_B, -1, _L),
                        dt_weight.view(2 * _D, -1, 1),
                        groups=2,
                    )
                else:
                    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, proj_weight)
                    if x_proj_bias is not None:
                        x_dbl = x_dbl + x_proj_bias.view(1, 2, -1, 1)
                    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_weight)
                xs = xs.view(_B, -1, _L)
                dts = dts.contiguous().view(_B, -1, _L)
                As = _As.view(-1, N).to(torch.float)
                Bs = Bs.contiguous().view(_B, 2, N, _L)
                Cs = Cs.contiguous().view(_B, 2, N, _L)
                Ds = _Ds.view(-1)
                delta_bias = dt_bias.view(-1).to(torch.float)
                if force_fp32:
                    xs = xs.to(torch.float)
                dts = dts.to(xs.dtype)
                Bs = Bs.to(xs.dtype)
                Cs = Cs.to(xs.dtype)
                ys = selective_scan(
                    xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
                ).view(_B, 2, -1, _L)
                return ys

            As = -self.A_logs.to(torch.float).exp().view(4, -1, N)
            x = (
                F.layer_norm(x.permute(0, 2, 3, 1), normalized_shape=(int(x.shape[1]),))
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            y_row = (
                scan_rowcol(
                    x,
                    proj_weight=self.x_proj_weight.view(4, -1, D)[:2].contiguous(),
                    proj_bias=(
                        x_proj_bias.view(4, -1)[:2].contiguous()
                        if x_proj_bias is not None
                        else None
                    ),
                    dt_weight=self.dt_projs_weight.view(4, D, -1)[:2].contiguous(),
                    dt_bias=(
                        self.dt_projs_bias.view(4, -1)[:2].contiguous()
                        if self.dt_projs_bias is not None
                        else None
                    ),
                    _As=As[:2].contiguous().view(-1, N),
                    _Ds=self.Ds.view(4, -1)[:2].contiguous().view(-1),
                    width=True,
                )
                .view(B, H, 2, -1, W)
                .sum(dim=2)
                .permute(0, 2, 1, 3)
            )
            y_row = (
                F.layer_norm(
                    y_row.permute(0, 2, 3, 1), normalized_shape=(int(y_row.shape[1]),)
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            y_col = (
                scan_rowcol(
                    y_row,
                    proj_weight=self.x_proj_weight.view(4, -1, D)[2:]
                    .contiguous()
                    .to(y_row.dtype),
                    proj_bias=(
                        x_proj_bias.view(4, -1)[2:].contiguous().to(y_row.dtype)
                        if x_proj_bias is not None
                        else None
                    ),
                    dt_weight=self.dt_projs_weight.view(4, D, -1)[2:]
                    .contiguous()
                    .to(y_row.dtype),
                    dt_bias=(
                        self.dt_projs_bias.view(4, -1)[2:].contiguous().to(y_row.dtype)
                        if self.dt_projs_bias is not None
                        else None
                    ),
                    _As=As[2:].contiguous().view(-1, N),
                    _Ds=self.Ds.view(4, -1)[2:].contiguous().view(-1),
                    width=False,
                )
                .view(B, W, 2, -1, H)
                .sum(dim=2)
                .permute(0, 2, 3, 1)
            )
            y = y_col
        else:
            x_proj_bias = getattr(self, "x_proj_bias", None)
            xs = cross_scan_fn(
                x,
                in_channel_first=True,
                out_channel_first=True,
                scans=_scan_mode,
                force_torch=scan_force_torch,
            )
            if no_einsum:
                x_dbl = F.conv1d(
                    xs.view(B, -1, L),
                    self.x_proj_weight.view(-1, D, 1),
                    bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None),
                    groups=K,
                )
                dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = F.conv1d(
                        dts.contiguous().view(B, -1, L),
                        self.dt_projs_weight.view(K * D, -1, 1),
                        groups=K,
                    )
            else:
                x_dbl = torch.einsum(
                    "b k d l, k c d -> b k c l", xs, self.x_proj_weight
                )
                if x_proj_bias is not None:
                    x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
                dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
                if hasattr(self, "dt_projs_weight"):
                    dts = torch.einsum(
                        "b k r l, k d r -> b k d l", dts, self.dt_projs_weight
                    )

            xs = xs.view(B, -1, L)
            dts = dts.contiguous().view(B, -1, L)
            As = -self.A_logs.to(torch.float).exp()
            Ds = self.Ds.to(torch.float)
            Bs = Bs.contiguous().view(B, K, N, L)
            Cs = Cs.contiguous().view(B, K, N, L)
            delta_bias = self.dt_projs_bias.view(-1).to(torch.float)

            if force_fp32:
                xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

            ys = selective_scan(
                xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
            ).view(B, K, -1, H, W)
            y = cross_merge_fn(
                ys,
                in_channel_first=True,
                out_channel_first=True,
                scans=_scan_mode,
                force_torch=scan_force_torch,
            )

            if getattr(self, "__DEBUG__", True):
                setattr(
                    self,
                    "__data__",
                    dict(
                        A_logs=self.A_logs,
                        Bs=Bs,
                        Cs=Cs,
                        Ds=Ds,
                        us=xs,
                        dts=dts,
                        delta_bias=delta_bias,
                        ys=ys,
                        y=y,
                        H=H,
                        W=W,
                    ),
                )

        y = y.view(B, -1, H, W)
        if not channel_first:
            y = y.view(B, -1, H * W).transpose(1, 2).contiguous().view(B, H, W, -1)
        y = out_norm(y)
        return y.to(x.dtype)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1))
            if not self.disable_z_act:
                z = self.act(z)
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.with_dconv:
            x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        return self.dropout(self.out_proj(y))

    @staticmethod
    def get_outnorm(forward_type="", d_inner=192, channel_first=True):
        def checkpostfix(tag, value):
            ret = value[-len(tag) :] == tag
            if ret:
                value = value[: -len(tag)]
            return ret, value

        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm
        out_norm_none, forward_type = checkpostfix("_onnone", forward_type)
        out_norm_dwconv3, forward_type = checkpostfix("_ondwconv3", forward_type)
        out_norm_cnorm, forward_type = checkpostfix("_oncnorm", forward_type)
        out_norm_softmax, forward_type = checkpostfix("_onsoftmax", forward_type)
        out_norm_sigmoid, forward_type = checkpostfix("_onsigmoid", forward_type)

        out_norm = nn.Identity()
        if out_norm_cnorm:
            out_norm = nn.Sequential(
                LayerNorm(d_inner),
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(
                    d_inner,
                    d_inner,
                    kernel_size=3,
                    padding=1,
                    groups=d_inner,
                    bias=False,
                ),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_dwconv3:
            out_norm = nn.Sequential(
                (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
                nn.Conv2d(
                    d_inner,
                    d_inner,
                    kernel_size=3,
                    padding=1,
                    groups=d_inner,
                    bias=False,
                ),
                (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            )
        elif out_norm_softmax:
            out_norm = SoftmaxSpatial(dim=(-1 if channel_first else 1))
        elif out_norm_sigmoid:
            out_norm = nn.Sigmoid()
        elif not out_norm_none:
            out_norm = LayerNorm(d_inner)
        return out_norm, forward_type

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag) :] == tag
        if ret:
            value = value[: -len(tag)]
        return ret, value


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v0",
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        use_checkpoint: bool = False,
        post_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2Dv2(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )
        self.drop_path = DropPath(drop_path)
        if self.mlp_branch:
            _MLP = Mlp if not gmlp else gMlp
            self.norm2 = norm_layer(hidden_dim)
            self.mlp = _MLP(
                in_features=hidden_dim,
                hidden_features=int(hidden_dim * mlp_ratio),
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channels_first=channel_first,
            )

    def _forward(self, input: torch.Tensor):
        x = input
        if self.ssm_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm(self.op(x)))
            else:
                x = x + self.drop_path(self.op(self.norm(x)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        return self._forward(input)


class VSSM(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=2,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        ssm_d_state=1,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v0",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        drop_path_rate=0.2,
        patch_norm=True,
        norm_layer="LN",
        downsample_version: str = "v3",
        patchembed_version: str = "v2",
        use_checkpoint=False,
        posembed=False,
        imgsize=224,
        **kwargs,
    ):
        super().__init__()
        self.channel_first = norm_layer.lower() in ["bn", "ln2d"]
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2**i) for i in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        _NORMLAYERS = dict(ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        _ACTLAYERS = dict(silu=nn.SiLU, gelu=nn.GELU, relu=nn.ReLU, sigmoid=nn.Sigmoid)

        norm_layer_cls: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)
        ssm_act_layer_cls: nn.Module = _ACTLAYERS.get(ssm_act_layer.lower(), None)
        mlp_act_layer_cls: nn.Module = _ACTLAYERS.get(mlp_act_layer.lower(), None)

        self.pos_embed = (
            self._pos_embed(dims[0], patch_size, imgsize) if posembed else None
        )

        _make_patch_embed = dict(
            v1=self._make_patch_embed, v2=self._make_patch_embed_v2
        ).get(patchembed_version, None)
        self.patch_embed = _make_patch_embed(
            in_chans,
            dims[0],
            patch_size,
            patch_norm,
            norm_layer_cls,
            channel_first=self.channel_first,
        )

        _make_downsample = dict(
            v1=PatchMerging2D,
            v2=self._make_downsample,
            v3=self._make_downsample_v3,
            none=(lambda *_, **_k: None),
        ).get(downsample_version, None)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            downsample = (
                _make_downsample(
                    self.dims[i_layer],
                    self.dims[i_layer + 1],
                    norm_layer=norm_layer_cls,
                    channel_first=self.channel_first,
                )
                if i_layer < self.num_layers - 1
                else nn.Identity()
            )

            self.layers.append(
                self._make_layer(
                    dim=self.dims[i_layer],
                    drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                    use_checkpoint=use_checkpoint,
                    norm_layer=norm_layer_cls,
                    downsample=downsample,
                    channel_first=self.channel_first,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_act_layer=ssm_act_layer_cls,
                    ssm_conv=ssm_conv,
                    ssm_conv_bias=ssm_conv_bias,
                    ssm_drop_rate=ssm_drop_rate,
                    ssm_init=ssm_init,
                    forward_type=forward_type,
                    mlp_ratio=mlp_ratio,
                    mlp_act_layer=mlp_act_layer_cls,
                    mlp_drop_rate=mlp_drop_rate,
                    gmlp=gmlp,
                )
            )

        self.classifier = nn.Sequential(
            OrderedDict(
                norm=norm_layer_cls(self.num_features),
                permute=(
                    Permute(0, 3, 1, 2) if not self.channel_first else nn.Identity()
                ),
                avgpool=nn.AdaptiveAvgPool2d(1),
                flatten=nn.Flatten(1),
                head=nn.Linear(self.num_features, num_classes),
            )
        )

        self.apply(self._init_weights)

    @staticmethod
    def _pos_embed(embed_dims, patch_size, img_size):
        ph, pw = img_size // patch_size, img_size // patch_size
        pos_embed = nn.Parameter(torch.zeros(1, embed_dims, ph, pw))
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    @staticmethod
    def _make_patch_embed(
        in_chans=3,
        embed_dim=96,
        patch_size=4,
        patch_norm=True,
        norm_layer=nn.LayerNorm,
        channel_first=False,
    ):
        return nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias=True,
            ),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(
        in_chans=3,
        embed_dim=96,
        patch_size=4,
        patch_norm=True,
        norm_layer=nn.LayerNorm,
        channel_first=False,
    ):
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(
                in_chans,
                embed_dim // 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            (
                nn.Identity()
                if (channel_first or not patch_norm)
                else Permute(0, 2, 3, 1)
            ),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (
                nn.Identity()
                if (channel_first or not patch_norm)
                else Permute(0, 3, 1, 2)
            ),
            nn.GELU(),
            nn.Conv2d(
                embed_dim // 2,
                embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_downsample(
        dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False
    ):
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(
        dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False
    ):
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
        dim=96,
        drop_path=[0.1, 0.1],
        use_checkpoint=False,
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        channel_first=False,
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v2",
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        gmlp=False,
        **kwargs,
    ):
        depth = len(drop_path)
        blocks = [
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
            )
            for d in range(depth)
        ]
        return nn.Sequential(
            OrderedDict(blocks=nn.Sequential(*blocks), downsample=downsample)
        )

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = (
                self.pos_embed.permute(0, 2, 3, 1)
                if not self.channel_first
                else self.pos_embed
            )
            x = x + pos_embed
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        def check_name(src, strict=False):
            key = prefix + src
            return (
                any(k.startswith(key) for k in state_dict)
                if not strict
                else (key in state_dict)
            )

        def change_name(src, dst):
            key = prefix + src
            for k in list(state_dict.keys()):
                if k.startswith(key):
                    state_dict[prefix + dst + k[len(key) :]] = state_dict.pop(k)

        if check_name("pos_embed", strict=True):
            srcEmb: torch.Tensor = state_dict[prefix + "pos_embed"]
            state_dict[prefix + "pos_embed"] = F.interpolate(
                srcEmb.float(),
                size=self.pos_embed.shape[2:4],
                align_corners=False,
                mode="bicubic",
            ).to(srcEmb.device)

        change_name("patch_embed.proj", "patch_embed.0")
        change_name("patch_embed.norm", "patch_embed.2")
        for i in range(100):
            for j in range(100):
                change_name(
                    f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm"
                )
                change_name(
                    f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op"
                )
        change_name("norm", "classifier.norm")
        change_name("head", "classifier.head")

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class Backbone_DINOv2_VSSM_2(VSSM):
    """VMamba-Small backbone used for OpenUS DINO pre-training and downstream
    tasks.

    forward() returns:
      - default:          [B, L+1, C]  (cls token concat with spatial tokens)
      - seg_head=True:    list of 4 feature tensors [B, C_i, H_i, W_i]
      - return_attn=True: ([B, L+1, C], attn_map)  (requires analyze.attnmap)
    """

    def __init__(
        self,
        depths=[2, 2, 15, 2],
        dims=96,
        drop_path_rate=0.3,
        ssm_ratio=2.0,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v05_noz",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        patch_norm=True,
        downsample_version="v3",
        patchembed_version="v2",
        use_checkpoint=False,
        posembed=False,
        imgsize=224,
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        norm_layer="ln2d",
        masked_im_modeling=False,
        ema_momentum=0.7,
        use_ema_for_loss_map=True,
        training=True,
        seg_head=False,
        **kwargs,
    ):
        super().__init__(
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            ssm_init=ssm_init,
            forward_type=forward_type,
            mlp_ratio=mlp_ratio,
            mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=mlp_drop_rate,
            gmlp=gmlp,
            patch_norm=patch_norm,
            downsample_version=downsample_version,
            patchembed_version=patchembed_version,
            use_checkpoint=use_checkpoint,
            posembed=posembed,
            imgsize=imgsize,
            norm_layer=norm_layer,
            **kwargs,
        )
        self.channel_first = norm_layer.lower() in ["bn", "ln2d"]
        _NORMLAYERS = dict(ln=nn.LayerNorm, ln2d=LayerNorm2d, bn=nn.BatchNorm2d)
        norm_layer_cls: nn.Module = _NORMLAYERS.get(norm_layer.lower(), None)

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer_cls(self.dims[i])
            self.add_module(f"outnorm{i}", layer)

        del self.classifier

        # optional: load a VMamba-pretrained checkpoint as starting point
        self._load_pretrained(pretrained)

        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, self.dims[0]))

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.encoder_stride = 32
        self.patch_size = patch_size * 8
        self.decoder = nn.Sequential(
            nn.Conv2d(self.dims[3], self.encoder_stride**2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(self.dims[2], 16**2 * 3, kernel_size=1),
            nn.PixelShuffle(16),
        )

        self.ema_momentum = ema_momentum
        self.use_ema_for_loss_map = use_ema_for_loss_map
        self.register_buffer("ema_loss_map", None)
        self.register_buffer("ema_initialized", torch.tensor(False))
        self.training = training
        self.seg_head = seg_head

    def _load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return
        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location="cpu")
            print(f"Loaded VMamba init weights: {ckpt}")
            self.load_state_dict(_ckpt[key], strict=False)
        except Exception as e:
            print(f"Failed to load VMamba init weights from {ckpt}: {e}")

    def update_ema_loss_map(self, current_loss_map):
        if not self.use_ema_for_loss_map:
            return current_loss_map
        B, C, L = current_loss_map.shape
        if not self.ema_initialized or self.ema_loss_map is None:
            self.ema_loss_map = current_loss_map.mean(dim=0).detach().clone()
            self.ema_initialized.fill_(True)
            return current_loss_map
        if self.training:
            batch_mean = current_loss_map.mean(dim=0).detach()
            if self.ema_loss_map.device != batch_mean.device:
                self.ema_loss_map = self.ema_loss_map.to(batch_mean.device)
            if self.ema_loss_map.dtype != batch_mean.dtype:
                self.ema_loss_map = self.ema_loss_map.to(batch_mean.dtype)
            with torch.no_grad():
                self.ema_loss_map.mul_(self.ema_momentum).add_(
                    batch_mean, alpha=(1 - self.ema_momentum)
                )
            ema_baseline = self.ema_loss_map.unsqueeze(0)
            return (
                self.ema_momentum * current_loss_map
                + (1 - self.ema_momentum) * ema_baseline
            )
        return current_loss_map

    def forward(
        self,
        x,
        mask=None,
        return_attn=False,
        return_rec_loss=False,
        return_recon_importance=True,
    ):
        def layer_forward(layer, x):
            x = layer.blocks(x)
            y = layer.downsample(x)
            return x, y

        x_orig = x
        x = self.patch_embed(x)
        if mask is not None:
            x = self._mask_model(x, mask)

        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x)
            if i in self.out_indices:
                norm_layer = getattr(self, f"outnorm{i}")
                out = norm_layer(o)
                if not self.channel_first:
                    out = out.permute(0, 3, 1, 2)
                outs.append(out.contiguous())

        x_region = outs[-1]
        B, C, H, W = x_region.shape
        x_region = x_region.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_cls = self.avgpool(x_region.transpose(1, 2))
        x_cls = torch.flatten(x_cls, 1)
        x_out = torch.cat([x_cls.unsqueeze(1), x_region], dim=1)

        if return_attn:
            if _AttnMamba is None:
                raise ImportError(
                    "analyze.attnmap not available; cannot use return_attn=True"
                )
            stage, block_id = 3, 1
            ss2ds = self.layers[stage].blocks[block_id].op
            setattr(ss2ds, "__DEBUG__", True)
            attnmap = _AttnMamba.get_attnmap_mamba(
                ss2ds, stage, "all_norm_CwBw", raw_attn=True, block_id=block_id
            )
            return x_out, attnmap

        if mask is not None and return_rec_loss:
            x_rec = self.decoder(outs[-1])
            mask1 = (
                mask.repeat_interleave(self.patch_size, 1)
                .repeat_interleave(self.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            rec_loss = F.mse_loss(x_orig, x_rec, reduction="none")
            recon_loss = (rec_loss * mask1).sum() / (mask1.sum() + 1e-5) / 3
            if return_recon_importance:
                patch_loss = self._compute_patch_recon_importance(
                    x_orig, x_rec, mask, self.patch_size
                )
                recon_importance = self.update_ema_loss_map(patch_loss)
            else:
                recon_importance = None
            return x_out, recon_loss, recon_importance

        if len(self.out_indices) == 0:
            return x

        if self.seg_head:
            return outs

        return x_out

    def _compute_patch_recon_importance(self, x_orig, x_rec, mask, patch_size):
        B, C, H, W = x_orig.shape
        pixel_loss = ((x_rec - x_orig) ** 2).mean(dim=1)
        mask = mask.unsqueeze(1)
        mask_resized = F.interpolate(mask.float(), size=(H, W), mode="nearest")
        masked_pixel_loss = pixel_loss * mask_resized.squeeze(1)
        patch_loss = F.avg_pool2d(
            masked_pixel_loss.unsqueeze(1), kernel_size=patch_size, stride=patch_size
        )
        patch_loss = patch_loss.squeeze(1).view(B, -1)
        patch_loss = patch_loss.unsqueeze(1).repeat(1, self.dims[-1], 1)
        return patch_loss

    def _mask_model(self, x, mask):
        if x.shape[-2:] != mask.shape[-2:]:
            htimes, wtimes = np.array(x.shape[-2:]) // np.array(mask.shape[-2:])
            mask = mask.repeat_interleave(htimes, -2).repeat_interleave(wtimes, -1)
        x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        return x


# ============================================================
# Weight loading
# ============================================================


def _load_openus_weights(
    backbone: nn.Module,
    ckpt_path: str,
    ckpt_key: str = "teacher",
    device: str = "cpu",
) -> None:
    """Strip DINO wrapper prefix and load into backbone (strict=False)."""
    print(f"Loading OpenUS weights: {ckpt_path}  (key={ckpt_key!r})")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if ckpt_key not in ckpt:
        raise RuntimeError(
            f"Checkpoint missing key {ckpt_key!r}. Available: {list(ckpt.keys())}"
        )

    raw = ckpt[ckpt_key]
    sd = {}
    for k, v in raw.items():
        if k.startswith("backbone."):
            sd[k[len("backbone.") :]] = v
        elif k.startswith("module.backbone."):
            sd[k[len("module.backbone.") :]] = v

    if not sd:
        raise RuntimeError(
            f"No 'backbone.*' keys found after prefix-strip. "
            f"First 3 raw keys: {list(raw.keys())[:3]}"
        )

    msg = backbone.load_state_dict(sd, strict=False)

    bad = [
        k
        for k in msg.missing_keys
        if k.startswith("patch_embed.") or k.startswith("layers.")
    ]
    if bad:
        raise RuntimeError(
            f"Core encoder keys still missing after load — prefix strip may be wrong. "
            f"First 5 missing: {bad[:5]}"
        )

    print(
        f"  raw={len(raw)}  stripped={len(sd)}  "
        f"missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}"
    )
    if msg.missing_keys:
        print(f"  missing   (first 5): {msg.missing_keys[:5]}")
    if msg.unexpected_keys:
        print(f"  unexpected (first 5): {msg.unexpected_keys[:5]}")


# ============================================================
# Public factory
# ============================================================


def build_openus_model(
    ckpt_path: str,
    *,
    ckpt_key: str = "teacher",
    seg_head: bool = False,
    device: str = "cpu",
) -> Backbone_DINOv2_VSSM_2:
    """Build a weight-loaded OpenUS-S backbone ready for inference.

    Args:
        ckpt_path: Path to the .pth checkpoint (e.g. openus_cpt0150.pth).
        ckpt_key:  Sub-dict key to read from the checkpoint ("teacher" or
                   "student").
        seg_head:  If True, forward() returns multi-scale feature maps
                   (list of 4 tensors [B, C_i, H_i, W_i]) instead of
                   the default token tensor [B, L+1, C].
        device:    map_location device string used when loading weights.

    Returns:
        Backbone_DINOv2_VSSM_2 in eval mode with OpenUS weights loaded.
    """
    model = Backbone_DINOv2_VSSM_2(
        depths=[2, 2, 15, 2],
        dims=96,
        drop_path_rate=0.3,
        ssm_ratio=2.0,
        patch_size=4,
        in_chans=3,
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",
        ssm_conv=3,
        ssm_conv_bias=False,
        ssm_drop_rate=0.0,
        ssm_init="v0",
        forward_type="v05_noz",
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        gmlp=False,
        patch_norm=True,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        use_checkpoint=False,
        posembed=False,
        imgsize=224,
        out_indices=(0, 1, 2, 3),
        masked_im_modeling=False,
        seg_head=seg_head,
    )
    _load_openus_weights(model, ckpt_path, ckpt_key=ckpt_key, device=device)
    model.eval()
    return model
