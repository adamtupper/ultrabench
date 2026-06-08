from pathlib import Path
from typing import Any

from ultrabench.models.echocare.echocare import get_echocare_extractor
from ultrabench.models.samus.samus import get_samus_extractor
from ultrabench.models.ultrasam.ultrasam import get_ultrasam_extractor
from ultrabench.models.urfm.urfm import get_urfm_extractor
from ultrabench.models.usfm.usfm import get_usfm_extractor
from ultrabench.models.usfmae.usfmae import get_usfmae_extractor

from .adapter_types import EncoderBundle


def load_ijepa(model_cfg: Any, device: str) -> EncoderBundle:
    raise NotImplementedError(
        "IJEPA loader is not implemented yet. Add a concrete loader in "
        "models/adapters/ijepa.py and keep model.loader._target_ pointing here."
    )


def load_urfm(model_cfg: Any, device: str) -> EncoderBundle:
    variant = (
        getattr(model_cfg, "variant", None)
        or getattr(model_cfg, "checkpoint_name", None)
        or getattr(model_cfg, "checkpoint_path", None)
    )
    checkpoint_path = getattr(model_cfg, "checkpoint_path", None) or None

    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[1]
            checkpoint_path = repo_root / checkpoint_path
        checkpoint_path = str(checkpoint_path)

    if not variant:
        raise ValueError("URFM config requires model.variant or model.checkpoint_name")

    encoder, transforms = get_urfm_extractor(
        variant,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    return EncoderBundle(
        encoder=encoder,
        transforms=transforms,
        metadata={
            "baseline": "urfm",
            "variant": variant,
            "checkpoint_path": checkpoint_path,
            "feature_dim": getattr(encoder, "embed_dim", None),
            "output_type": model_cfg.output_type,
            "has_cls_token": model_cfg.has_cls_token,
        },
    )


def load_usfm(model_cfg: Any, device: str) -> EncoderBundle:
    variant = (
        getattr(model_cfg, "variant", None)
        or getattr(model_cfg, "checkpoint_name", None)
        or getattr(model_cfg, "checkpoint_path", None)
    )
    checkpoint_path = getattr(model_cfg, "checkpoint_path", None) or None

    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[1]
            checkpoint_path = repo_root / checkpoint_path
        checkpoint_path = str(checkpoint_path)

    if not variant:
        raise ValueError("USFM config requires model.variant or model.checkpoint_name")

    encoder, transforms = get_usfm_extractor(
        device=device, checkpoint_path=checkpoint_path, return_patch_embeddings=True
    )

    return EncoderBundle(
        encoder=encoder,
        transforms=transforms,
        metadata={
            "baseline": "usfm",
            "variant": variant,
            "checkpoint_path": checkpoint_path,
            "feature_dim": getattr(encoder, "embed_dim", None),
            "output_type": model_cfg.output_type,
            "has_cls_token": model_cfg.has_cls_token,
        },
    )


def load_ultrasam(model_cfg: Any, device: str) -> EncoderBundle:
    checkpoint_path = getattr(model_cfg, "checkpoint_path", None) or None

    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[1]
            checkpoint_path = repo_root / checkpoint_path
        checkpoint_path = str(checkpoint_path)

    encoder, transforms = get_ultrasam_extractor(
        device=device,
        checkpoint_path=checkpoint_path,
    )

    return EncoderBundle(
        encoder=encoder,
        transforms=transforms,
        metadata={
            "baseline": "ultrasam",
            "variant": model_cfg.variant,
            "checkpoint_path": checkpoint_path,
            "feature_dim": encoder.embed_dims,
            "output_type": model_cfg.output_type,
            "has_cls_token": model_cfg.has_cls_token,
        },
    )


def load_usfmae(model_cfg: Any, device: str) -> EncoderBundle:
    checkpoint_path = getattr(model_cfg, "checkpoint_path", None) or None

    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[1]
            checkpoint_path = repo_root / checkpoint_path
        checkpoint_path = str(checkpoint_path)

    encoder, transforms = get_usfmae_extractor(
        device=device,
        checkpoint_path=checkpoint_path,
    )

    return EncoderBundle(
        encoder=encoder,
        transforms=transforms,
        metadata={
            "baseline": "usfmae",
            "variant": getattr(model_cfg, "variant", "usfmae"),
            "checkpoint_path": checkpoint_path,
            "feature_dim": encoder.embed_dim,
            "output_type": model_cfg.output_type,
            "has_cls_token": model_cfg.has_cls_token,
        },
    )


def load_echocare(model_cfg: Any, device: str) -> EncoderBundle:
    checkpoint_path = getattr(model_cfg, "checkpoint_path", None) or None

    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[1]
            checkpoint_path = repo_root / checkpoint_path
        checkpoint_path = str(checkpoint_path)

    encoder, transforms = get_echocare_extractor(
        device=device,
        checkpoint_path=checkpoint_path,
    )

    return EncoderBundle(
        encoder=encoder,
        transforms=transforms,
        metadata={
            "baseline": "echocare",
            "variant": model_cfg.variant,
            "checkpoint_path": checkpoint_path,
            "feature_dim": encoder.embed_dim,
            "scale_dims": [128, 256, 512, 1024, 2048],
            "output_type": model_cfg.output_type,
            "has_cls_token": model_cfg.has_cls_token,
        },
    )


def load_samus(model_cfg: Any, device: str) -> EncoderBundle:
    checkpoint_path = getattr(model_cfg, "checkpoint_path", None) or None

    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            repo_root = Path(__file__).resolve().parents[1]
            checkpoint_path = repo_root / checkpoint_path
        checkpoint_path = str(checkpoint_path)

    encoder, transforms = get_samus_extractor(
        device=device,
        checkpoint_path=checkpoint_path,
    )

    return EncoderBundle(
        encoder=encoder,
        transforms=transforms,
        metadata={
            "baseline": "samus",
            "variant": getattr(model_cfg, "variant", "samus"),
            "checkpoint_path": checkpoint_path,
            "feature_dim": encoder.embed_dim,
            "output_type": model_cfg.output_type,
            "has_cls_token": model_cfg.has_cls_token,
        },
    )
