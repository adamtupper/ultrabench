import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as tv_transforms

# from mmpretrain.models.backbones import ViTSAM
from ultrabench.models.ultrasam.ultrasam_vit_sam_arch import ViTSAM

# Use UltraSam conda environment to use this as downstream


class ResizeLongestSide:
    """Scale so the longest side equals max_size, preserving aspect ratio."""

    def __init__(self, max_size=1024):
        self.max_size = max_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2], img.shape[-1]
        scale = self.max_size / max(h, w)
        if scale == 1.0:
            return img
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        batched = img.dim() == 4
        x = img if batched else img.unsqueeze(0)
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        return x if batched else x.squeeze(0)


class PadToSquare:
    """Zero-pad on the right and bottom to reach target square size."""

    def __init__(self, size=1024):
        self.size = size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2], img.shape[-1]
        # F.pad order: (left, right, top, bottom)
        return F.pad(img, (0, self.size - w, 0, self.size - h), value=0.0)


def get_ultrasam_extractor(device="cuda", checkpoint_path=None):
    """
    Load UltraSAM model and return encoder and transforms.
    """

    default_path = "baselines/ultrasam/UltraSam.pt"
    if checkpoint_path is None:
        checkpoint_path = default_path
    print(f"Looking for UltraSam weights at: {checkpoint_path}")

    if os.path.exists(checkpoint_path):
        print(f"Found UltraSam weights at: {checkpoint_path}")
        state_dict = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    else:
        raise FileNotFoundError(
            f"Could not find UltraSAM weights at: {checkpoint_path}"
        )

    encoder = ViTSAM(
        use_gradient_checkpointing=True  # hard setting to True because we know this is necessary to avoid using accum_grad
    )

    clean_state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    missing, unexpected = encoder.load_state_dict(clean_state_dict, strict=False)
    # print("Missing keys:", missing)
    # print("Unexpected keys:", unexpected)
    encoder.channel_reduction = nn.Identity()  # ty:ignore[invalid-assignment]

    encoder.eval()
    encoder.to(device)

    # We want to make sure the backbone is being loaded properly, the remaining decoder, prompt_encoder and bbox_head don't need to be loaded
    assert (
        sum(
            [
                not any(sub in x for sub in ["decoder", "prompt_encoder", "bbox_head"])
                for x in unexpected
            ]
        )
        == 0
    ), "Unexpected keys found in state_dict!"

    image_size = 256  # 1024

    transforms = tv_transforms.Compose(
        [
            # PIL → uint8 tensor (C, H, W)
            # tv_transforms.ToImage(),
            # uint8 → float32 in [0, 1]
            # tv_transforms.ToDtype(torch.float32, scale=True),
            # Resize longest side to 1024, preserving aspect ratio
            ResizeLongestSide(image_size),
            # Pad shorter side to get 1024×1024
            PadToSquare(image_size),
            # ImageNet normalisation (matches mmdet config mean/std ÷ 255)
            tv_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    return encoder, transforms
