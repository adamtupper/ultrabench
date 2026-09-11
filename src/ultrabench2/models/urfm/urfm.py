import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Block, PatchEmbed

# import torchvision.transforms as transforms
from torchvision.transforms import v2 as transforms


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        asymmetric_decoder=False,
        mask_ratio=0.75,
        vis_mask_ratio=0.0,
        learning_loss=True,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.vis_mask_ratio = vis_mask_ratio
        if vis_mask_ratio > 0:
            self.vis_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.learning_loss = learning_loss

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                # Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics (projector, predictor, and loss predictor)
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        # reconstructor (e.g., projector at feature-level)
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True)          # dino/ema
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, decoder_embed_dim, bias=True
        )  # clip

        # loss predictor
        if self.learning_loss:
            self.decoder_blocks_losspred = nn.ModuleList(
                [
                    Block(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )
                    for i in range(decoder_depth)
                ]
            )
            self.decoder_norm_losspred = norm_layer(decoder_embed_dim)
            self.decoder_pred_losspred = nn.Linear(
                decoder_embed_dim, patch_size**2 * in_chans, bias=True
            )
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        if hasattr(self, "vis_mask_token"):
            torch.nn.init.normal_(self.vis_mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        # x = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_encoder(self, x, mask):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        N, L, D = x.shape
        if mask is None:
            mask = torch.zeros((N, L), dtype=torch.bool, device=x.device)
        x = x[~mask].reshape(N, -1, D)

        if self.vis_mask_ratio > 0:
            vis_mask_token = self.vis_mask_token + self.pos_embed[:, 1:, :]
            vis_mask_token = vis_mask_token.expand(N, -1, -1)
            vis_mask_token = vis_mask_token[~mask].reshape(N, -1, D)
            L = x.size(1)
            noise = torch.rand(N, L, device=x.device)
            ids_restore = torch.argsort(noise, dim=1)

            len_keep = int(L * (1 - self.vis_mask_ratio))
            vis_mask = torch.ones([N, L], device=x.device)
            vis_mask[:, :len_keep] = 0
            vis_mask = torch.gather(vis_mask, dim=1, index=ids_restore).unsqueeze(-1)

            x = x * (1.0 - vis_mask) + vis_mask_token * vis_mask

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_decoder(self, x, mask):
        # embed tokens
        x = self.decoder_embed(x)
        x_vis = x[:, 1:, :]
        N, _, D = x_vis.shape

        # append mask tokens to sequence
        expand_pos_embed = self.decoder_pos_embed[:, 1:, :].expand(N, -1, -1)
        pos_vis = expand_pos_embed[~mask].reshape(N, -1, D)
        pos_mask = expand_pos_embed[mask].reshape(N, -1, D)

        x_ = torch.cat([x_vis + pos_vis, self.mask_token + pos_mask], dim=1)

        # add cls_token + decoder_pos_embed
        x = torch.cat([x[:, :1, :] + self.decoder_pos_embed[:, :1, :], x_], dim=1)
        loss_pred = x.clone()

        # apply reconstructor
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]

        if self.learning_loss:
            # apply loss predictor
            for blk in self.decoder_blocks_losspred:
                loss_pred = blk(loss_pred)
            loss_pred = self.decoder_norm_losspred(loss_pred)
            loss_pred = self.decoder_pred_losspred(loss_pred)
            loss_pred = loss_pred[:, 1:, :]  # (N, L, 1)

            return x, pos_mask.shape[1], loss_pred.mean(dim=-1)

        return x, pos_mask.shape[1]

    def forward_loss(self, pred, target, mask):
        """
        pred: [N, mask, D]
        target: [N, L, D]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        N, _, D = target.shape
        target = target[mask].reshape(N, -1, D)

        pred = torch.nn.functional.normalize(pred, p=2, dim=-1)
        target = torch.nn.functional.normalize(target, p=2, dim=-1)
        # print ("pred.shape:", pred.shape)
        # print ("target.shape:", target.shape)
        loss = ((pred - target) ** 2).sum(dim=-1)

        return {"mean": loss.mean(), "matrix": loss}

    def forward(self, imgs, mask):
        latent = self.forward_encoder(imgs, mask)  # returned mask may change

        if self.learning_loss:
            pred, mask_num, loss_pred = self.forward_decoder(
                latent, mask
            )  # [N, L, p*p*3]
        else:
            pred, mask_num = self.forward_decoder(latent, mask)
        # loss = self.forward_loss(imgs, pred[:, -mask_num:], mask)
        # return loss, pred, mask
        out = {
            "pix_pred": pred,
            "mask": mask,
            "mask_num": mask_num,
            "features": latent,
        }

        if self.learning_loss:
            out["loss_pred"] = loss_pred

        return out

    def generate_mask(
        self,
        loss_pred,
        mask_ratio=0.75,
        images=None,
        guide=True,
        epoch=0,
        total_epoch=200,
    ):
        loss_pred = loss_pred.squeeze()
        N, L = loss_pred.shape
        len_keep = int(L * (1 - mask_ratio))

        ids_shuffle_loss = torch.argsort(loss_pred, dim=1)  # (N, L)

        # keep `keep_ratio` loss and `1 - keep_ratio` random
        keep_ratio = 0.2
        ids_shuffle = torch.zeros_like(ids_shuffle_loss, device=loss_pred.device).int()

        if guide:
            keep_ratio = float((epoch + 1) / total_epoch) * 0.5

        ## top 0 -> 0.5
        if int((L - len_keep) * keep_ratio) <= 0:
            # random
            noise = torch.randn(N, L, device=loss_pred.device)
            ids_shuffle = torch.argsort(noise, dim=1)
        else:
            for i in range(N):
                ## mask top `keep_ratio` loss and `1 - keep_ratio` random
                len_loss = int((L - len_keep) * keep_ratio)
                ids_shuffle[i, -len_loss:] = ids_shuffle_loss[i, -len_loss:]

                temp = torch.arange(L, device=loss_pred.device)
                deleted = np.delete(
                    temp.cpu().numpy(), ids_shuffle[i, -len_loss:].cpu().numpy()
                )
                np.random.shuffle(deleted)
                ids_shuffle[i, : (L - len_loss)] = torch.LongTensor(deleted).to(
                    loss_pred.device
                )

        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=loss_pred.device)
        mask[:, :len_keep] = 0
        # unshuffle to get final mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask

    def forward_learning_loss(self, loss_pred, mask, loss_target, relative=False):
        """
        loss_pred: [N, L, 1]
        mask: [N, L], 0 is keep, 1 is remove,
        loss_target: [N, L]
        """
        # N, L = loss_target.shape
        # loss_pred = loss_pred[mask].reshape(N, L)
        assert self.learning_loss

        if relative:
            # binary classification for LxL
            labels_positive = loss_target.unsqueeze(1) > loss_target.unsqueeze(2)
            labels_negative = loss_target.unsqueeze(1) < loss_target.unsqueeze(2)
            labels_valid = labels_positive + labels_negative

            loss_matrix = loss_pred.unsqueeze(1) - loss_pred.unsqueeze(2)
            loss = -labels_positive.int() * torch.log(
                torch.sigmoid(loss_matrix) + 1e-6
            ) - labels_negative.int() * torch.log(1 - torch.sigmoid(loss_matrix) + 1e-6)

            return loss.sum() / labels_valid.sum()

        else:
            # normalize by each image
            mean = loss_target.mean(dim=1, keepdim=True)
            var = loss_target.var(dim=1, keepdim=True)
            loss_target = (loss_target - mean) / (var + 1.0e-6) ** 0.5  # [N, L, 1]

            loss = (loss_pred - loss_target) ** 2
            loss = loss.mean()
            return loss


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio, regular=False, block=False):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)
        self.regular = regular
        self.block = block

        if block:
            assert mask_ratio == 0.75

        if regular:
            assert mask_ratio == 0.75

            candidate_list = []
            while True:  # add more
                for j in range(4):
                    candidate = np.ones(4)
                    candidate[j] = 0
                    candidate_list.append(candidate)
                if len(candidate_list) * 4 >= self.num_patches * 2:
                    break
            self.mask_candidate = np.vstack(candidate_list)
            print("using regular, mask_candidate shape = ", self.mask_candidate.shape)

    def __repr__(self):
        repr_str = (
            "Mask: total patches {}, mask patches {}, regular {}, block {}".format(
                self.num_patches, self.num_mask, self.regular, self.block
            )
        )
        return repr_str

    def __call__(self):
        # returns a (num_patches,) binary mask
        if self.regular:
            mask = self.mask_candidate.copy()
            np.random.shuffle(mask)
            mask = rearrange(
                mask[: self.num_patches // 4],
                "(h w) (p1 p2) -> (h p1) (w p2)",
                h=self.height // 2,
                w=self.width // 2,
                p1=2,
                p2=2,
            )
            mask = mask.flatten()

        elif self.block:
            visible_width = int(np.sqrt(self.num_patches - self.num_mask))
            total_width = int(np.sqrt(self.num_patches))
            mask = np.ones((total_width, total_width))

            left_x = np.random.randint(low=0, high=total_width // 2)
            up_y = np.random.randint(low=0, high=total_width // 2)

            mask[up_y : up_y + visible_width, left_x : left_x + visible_width] = 0
            mask = mask.flatten()

        else:
            mask = np.hstack(
                [
                    np.zeros(self.num_patches - self.num_mask),  # 0 is visible
                    np.ones(self.num_mask),  # 1 is masked
                ]
            )
            np.random.shuffle(mask)

        return mask


class MaskTransform(object):
    def __init__(self, input_size, token_size, mask_ratio, is_train=True):
        model_mean = [0.485, 0.456, 0.406]
        model_std = [0.229, 0.224, 0.225]
        interpolation = transforms.InterpolationMode.BICUBIC
        # -----------------------------------------------------------------------

        if is_train:
            self.transform = transforms.Compose(
                [
                    # Handles Tensors directly on GPU
                    transforms.RandomResizedCrop(
                        input_size,
                        scale=(0.2, 1.0),
                        interpolation=interpolation,
                        antialias=True,
                    ),
                    transforms.RandomHorizontalFlip(),
                    # Converts to float and scales to [0, 1] if input is uint8
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(mean=model_mean, std=model_std),
                ]
            )
        else:
            # Standard validation/eval transform
            # size = int(input_size / (224 / 256)) if input_size <= 224 else input_size
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        input_size, interpolation=interpolation, antialias=True
                    ),
                    # transforms.Resize(size, interpolation=interpolation, antialias=True),
                    # transforms.CenterCrop(input_size),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(mean=model_mean, std=model_std),
                ]
            )

        # Your original mask generator logic
        self.masked_position_generator = RandomMaskingGenerator(
            token_size,
            mask_ratio,
            regular=False,
            block=False,
        )

    def __call__(self, image):
        # 'image' is a Tensor [C, H, W]
        return self.transform(image), self.masked_position_generator()


def get_urfm_extractor(model_variant, device, checkpoint_path=None):
    patch_size = 16
    input_size = 224
    token_size = 224 // patch_size
    mask_ratio = 0.0

    # vit_base_patch16_dec768d8b
    if model_variant == "urfm_mae_vit_base_patch16":
        encoder = MaskedAutoencoderViT(
            patch_size=patch_size,
            embed_dim=768,
            depth=12,
            num_heads=12,
            decoder_embed_dim=768,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
    elif model_variant == "urfm_mae_vit_large_patch16":
        encoder = MaskedAutoencoderViT(
            patch_size=patch_size,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            decoder_embed_dim=768,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
    elif model_variant == "urfm_mae_vit_huge_patch16":
        encoder = MaskedAutoencoderViT(
            patch_size=patch_size,
            embed_dim=1280,
            depth=32,
            num_heads=16,
            decoder_embed_dim=768,
            decoder_depth=8,
            decoder_num_heads=16,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
    else:
        raise ValueError(f"Unsupported URFM variant: {model_variant}")

    default_path = f"baselines/urfm/{model_variant}_dec768d8b_all_biomedclip_1199.pth"
    if checkpoint_path is None:
        checkpoint_path = default_path
    print(f"Looking for URFM weights at: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        print(f"Found URFM weights at: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    else:
        raise FileNotFoundError(f"Could not find URFM weights at: {checkpoint_path}")

    state_dict = checkpoint["state_dict"]
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = encoder.load_state_dict(clean_state_dict, strict=False)

    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    encoder.to(device)
    # print("Missing keys:", missing)
    # print("Unexpected keys:", unexpected)
    assert sum(["decoder" not in x for x in missing]) == 0

    transforms = MaskTransform(input_size, token_size, mask_ratio, is_train=False)

    return encoder, transforms
