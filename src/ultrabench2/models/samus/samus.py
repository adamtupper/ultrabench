import os
from functools import partial

import torch
import torchvision.transforms.v2 as tv_transforms

# import sys
# sys.path.append('/raid/aradhachandran/thyroid/code/foundation_model/downstreams/SAMUS/models/segment_anything_samus/')
# from modeling import ImageEncoderViT
# sys.path.append('/raid/aradhachandran/thyroid/code/foundation_model/downstreams/SAMUS/utils/')
# from data_us import JointTransform2D
from ultrabench2.models.samus.samus_image_encoder_vit import ImageEncoderViT


def get_samus_extractor(device="cuda", checkpoint_path=None):
    """
    Load SAMUS model and return encoder and transforms.
    """

    default_path = "baselines/samus/SAMUS.pth"
    if checkpoint_path is None:
        checkpoint_path = default_path
    print(f"Looking for SAMUS weights at: {checkpoint_path}")

    if os.path.exists(checkpoint_path):
        print(f"Found SAMUS weights at: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
    else:
        raise FileNotFoundError(f"Could not find SAMUS weights at: {checkpoint_path}")

    encoder_embed_dim = 768
    encoder_depth = 12
    encoder_num_heads = 12
    encoder_global_attn_indexes = [2, 5, 8, 11]

    image_size = 256  # I think they expect 256
    patch_size = image_size // 32
    prompt_embed_dim = 256

    encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),  # ty:ignore[invalid-argument-type]
        num_heads=encoder_num_heads,
        patch_size=patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,  # ty:ignore[invalid-argument-type]
        window_size=14,
        out_chans=prompt_embed_dim,
        use_gradient_checkpointing=True,  # hard setting to True because we know this is necessary to avoid using accum_grad
    )
    encoder.embed_dim = 256  # ty:ignore[unresolved-attribute]

    clean_state_dict = {
        k.replace("image_encoder.", ""): v for k, v in state_dict.items()
    }

    missing, unexpected = encoder.load_state_dict(clean_state_dict, strict=False)
    # print("Missing keys:", missing)
    # print("Unexpected keys:", unexpected)

    encoder.eval()
    encoder.to(device)

    assert (
        sum(
            [
                not any(sub in x for sub in ["mask_decoder", "prompt_encoder"])
                for x in unexpected
            ]
        )
        == 0
    )

    # # Resize to 256x256
    # encoder_input_size = 256
    # low_image_size = 128
    # # opt.
    # crop = None
    # transforms = JointTransform2D(img_size=encoder_input_size, low_img_size=low_image_size,
    #                         ori_size=image_size, crop=crop, p_flip=0, color_jitter_params=None, long_mask=True)
    transforms = tv_transforms.Compose(
        [
            tv_transforms.Resize(
                (256, 256), interpolation=tv_transforms.InterpolationMode.BILINEAR
            ),
            tv_transforms.ToTensor(),
        ]
    )

    return encoder, transforms
