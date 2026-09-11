"""OpenUSBackbone: Backbone wrapper for the OpenUS VMamba encoder."""

from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as transforms

from ultrabench2.models.backbone import Backbone, BackboneOutput
from ultrabench2.models.openus.openus_vmamba import (
    DEFAULT_KERNEL_REPO_ID,
    DEFAULT_KERNEL_VERSION,
    Backbone_DINOv2_VSSM_2,
    load_selective_scan_kernel,
)


class OpenUSBackbone(Backbone):
    """Backbone wrapper for the OpenUS VMamba (VSSM-Small) encoder.

    Accepts H x W sized uint8 tensors that are resized to 224 x 224 pixels
    and normalized with ImageNet statistics before being passed to the encoder.

    The encoder is a VMamba-Small model pre-trained with DINO on the US308k
    dataset. With seg_head=True the encoder forward pass always computes all 4
    stage outputs (channel dims [96, 192, 384, 768]). They are included in
    BackboneOutput.intermediate_features only when
    output_intermediate_features=True.

    Attributes:
        feature_dim: 768 (last VMamba stage output dimension).
        output_type: Always "multi_scale".
        has_cls_token: Always False.
        block_indices: Always None (VMamba, not ViT).
        intermediate_feature_dims: Channel dims per VMamba stage, fine→coarse:
            [96, 192, 384, 768].
        input_size: 224.
        input_mean: Per-channel mean for ImageNet normalization.
        input_std: Per-channel std for ImageNet normalization.
    """

    feature_dim: int = 768
    output_type: str = "multi_scale"
    has_cls_token: bool = False
    block_indices: list[int] | None = None
    intermediate_feature_dims: list[int] = [96, 192, 384, 768]
    input_size: int = 224
    input_mean: list[float] = [0.485, 0.456, 0.406]
    input_std: list[float] = [0.229, 0.224, 0.225]

    def __init__(
        self,
        checkpoint_path: str,
        output_intermediate_features: bool,
        selective_scan_kernel_path: str | None = None,
        selective_scan_kernel_repo_id: str = DEFAULT_KERNEL_REPO_ID,
        selective_scan_kernel_version: int | None = DEFAULT_KERNEL_VERSION,
    ) -> None:
        """Initialise OpenUSBackbone.

        The selective scan kernel is resolved once per process from a local
        HF Kernels build, then from the Hugging Face Hub, then from the
        pure-PyTorch implementation. Resolution never raises: a kernel that
        cannot be loaded is logged and the next option is tried.

        Args:
            checkpoint_path: Path to the OpenUS DINO checkpoint (.pth).
            output_intermediate_features: Whether to include all 4 VMamba
                stage outputs in BackboneOutput.intermediate_features.
            selective_scan_kernel_path: Directory holding a pre-compiled
                selective scan kernel build, tried before the Hub. None
                skips straight to the Hub.
            selective_scan_kernel_repo_id: Hugging Face Hub repository to
                load the selective scan kernel from.
            selective_scan_kernel_version: Kernel version to load from the
                Hub. None loads the latest revision.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
            ValueError: If the checkpoint is missing the expected "teacher" key
                or if no backbone keys are found after prefix stripping.
            RuntimeError: If core encoder keys are missing after loading.
        """
        super().__init__(checkpoint_path, output_intermediate_features)

        load_selective_scan_kernel(
            local_kernel_path=selective_scan_kernel_path,
            repo_id=selective_scan_kernel_repo_id,
            version=selective_scan_kernel_version,
        )

        self._encoder = Backbone_DINOv2_VSSM_2(
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
            seg_head=True,
        )

        self._load_checkpoint()

        self._transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=self.input_mean, std=self.input_std),
            ]
        )

    def _load_checkpoint(self) -> None:
        """Load encoder weights from the OpenUS DINO checkpoint.

        The checkpoint is expected to contain a "teacher" sub-dict whose keys
        are prefixed with "backbone." or "module.backbone.". The prefix is
        stripped before calling load_state_dict.

        Raises:
            FileNotFoundError: If the checkpoint file cannot be found.
            ValueError: If the "teacher" key is absent or no backbone keys
                are found after prefix stripping.
            RuntimeError: If core patch_embed or layer keys are missing after
                loading, indicating a broken prefix strip.
        """
        path = Path(self._checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find OpenUS weights at: {path}")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        ckpt_key = "teacher"
        if ckpt_key not in ckpt:
            raise ValueError(
                f"Checkpoint missing key {ckpt_key!r}. "
                f"Available keys: {list(ckpt.keys())}"
            )

        raw = ckpt[ckpt_key]
        state_dict = {}
        for k, v in raw.items():
            if k.startswith("backbone."):
                state_dict[k[len("backbone.") :]] = v
            elif k.startswith("module.backbone."):
                state_dict[k[len("module.backbone.") :]] = v

        if not state_dict:
            raise ValueError(
                f"No 'backbone.*' keys found in checkpoint after prefix strip. "
                f"First 3 raw keys: {list(raw.keys())[:3]}"
            )

        msg = self._encoder.load_state_dict(state_dict, strict=False)

        bad = [k for k in msg.missing_keys if k.startswith(("patch_embed.", "layers."))]
        if bad:
            raise RuntimeError(
                f"Core encoder keys missing after loading OpenUS checkpoint — "
                f"prefix strip may be wrong. First 5 missing: {bad[:5]}"
            )

    def _resize_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Resize and normalize a batch of uint8 images.

        Args:
            x: Input tensor [B, C, H, W] at any spatial size.

        Returns:
            Resized and normalized float32 tensor [B, C, 224, 224].
        """
        return self._transform(x)

    def forward(self, x: torch.Tensor, pool: bool = False) -> BackboneOutput:
        """Extract multi-scale features from input images.

        Args:
            x: uint8 input image tensor [B, C, H, W]. Resized to
                ``self.input_size`` and normalized before encoding.
            pool: If True, apply adaptive average pooling then flatten the
                last stage output to [B, 768]. If False, return the spatial
                map [B, 768, H', W'].

        Returns:
            BackboneOutput with features [B, 768, H', W'] when pool=False or
            [B, 768] when pool=True, and optionally intermediate_features
            (all 4 VMamba stage outputs, channel dims [96, 192, 384, 768]).
        """
        x = self._resize_and_normalize(x)
        stage_outputs: list[torch.Tensor] = self._encoder(x)
        features = stage_outputs[-1]
        if pool:
            features = F.adaptive_avg_pool2d(features, 1).flatten(start_dim=1)
        intermediate = stage_outputs if self._output_intermediate_features else None
        return BackboneOutput(features=features, intermediate_features=intermediate)
