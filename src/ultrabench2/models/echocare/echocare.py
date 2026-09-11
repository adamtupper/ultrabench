import os

import torch
import torchvision.transforms.v2 as tv_transforms

# from monai.networks.nets.swin_unetr import SwinTransformer
from ultrabench2.models.echocare.echocare_swin_transformer_arch import SwinTransformer


def get_echocare_extractor(device="cuda", checkpoint_path=None):
    extractor = SwinTransformer(
        in_chans=3,
        embed_dim=128,
        window_size=[8] * 2,
        patch_size=[2] * 2,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        mlp_ratio=4.0,
        qkv_bias=True,
        use_checkpoint=False,  # This was set to True, however, we set to False to address overlap with torch no grad
        spatial_dims=2,
        use_v2=True,
    )

    extractor.embed_dim = 2048

    default_path = "baselines/echocare/echocare_encoder.pth"
    if checkpoint_path is None:
        checkpoint_path = default_path
    print(f"Looking for EchoCare weights at: {checkpoint_path}")

    if os.path.exists(checkpoint_path):
        print(f"Found EchoCare weights at: {checkpoint_path}")
        model_dict = torch.load(checkpoint_path, map_location=device)
        state_dict = model_dict
        state_dict.pop("mask_token")
    else:
        raise FileNotFoundError(
            f"Could not find EchoCare weights at: {checkpoint_path}"
        )

    missing, unexpected = extractor.load_state_dict(state_dict, strict=False)
    # print("Missing keys:", missing)
    # print("Unexpected keys:", unexpected)

    for param in extractor.parameters():
        param.requires_grad = False
    extractor.eval()
    extractor.to(device)

    transforms = tv_transforms.Compose(
        [
            # T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            tv_transforms.Resize(
                size=(256, 256), interpolation=tv_transforms.InterpolationMode.BILINEAR
            ),
            tv_transforms.ToTensor(),
        ]
    )

    return extractor, transforms
