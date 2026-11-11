"""Unit tests for the classification and segmentation heads."""

import pytest
import torch
import torch.nn as nn

from ultrabench.models.heads import (
    DECODER_COMPATIBILITY,
    SegFormerMultiScaleDecoder,
    SETRMLADecoder,
    SimpleUpsamplerHead,
    get_cls_head,
    get_seg_head,
)

BATCH_SIZE = 2
FEATURE_DIM = 32
NUM_CLASSES = 3
IMAGE_SIZE = 64


class TestGetClsHead:
    """Tests for the classification head factory."""

    def test_linear_head_maps_features_to_logits(self) -> None:
        head = get_cls_head("linear", FEATURE_DIM, NUM_CLASSES)
        assert isinstance(head, nn.Linear)
        logits = head(torch.randn(BATCH_SIZE, FEATURE_DIM))
        assert logits.shape == (BATCH_SIZE, NUM_CLASSES)

    def test_attentive_head_is_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            get_cls_head("attentive", FEATURE_DIM, NUM_CLASSES)

    def test_unknown_probe_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown probe_type"):
            get_cls_head("mlp", FEATURE_DIM, NUM_CLASSES)


class TestGetSegHead:
    """Tests for the segmentation head factory and the heads it builds."""

    @pytest.mark.parametrize("use_residual", [True, False])
    def test_upsampler_on_patch_tokens(self, use_residual: bool) -> None:
        head = get_seg_head(
            decoder_type="upsampler",
            output_type="patch_tokens",
            feature_dim=FEATURE_DIM,
            num_classes=NUM_CLASSES,
            image_size=IMAGE_SIZE,
            use_residual=use_residual,
        )
        assert isinstance(head, SimpleUpsamplerHead)
        out = head(torch.randn(BATCH_SIZE, 16, FEATURE_DIM))
        assert out.shape == (BATCH_SIZE, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)

    def test_upsampler_on_spatial_map(self) -> None:
        head = get_seg_head(
            decoder_type="upsampler",
            output_type="spatial_map",
            feature_dim=FEATURE_DIM,
            num_classes=NUM_CLASSES,
            image_size=IMAGE_SIZE,
        )
        out = head(torch.randn(BATCH_SIZE, FEATURE_DIM, 4, 4))
        assert out.shape == (BATCH_SIZE, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)

    def test_setr_mla_on_intermediate_tokens(self) -> None:
        block_indices = [2, 5, 8, 11]
        head = get_seg_head(
            decoder_type="setr_mla",
            output_type="patch_tokens",
            feature_dim=FEATURE_DIM,
            num_classes=NUM_CLASSES,
            image_size=IMAGE_SIZE,
            block_indices=block_indices,
        )
        assert isinstance(head, SETRMLADecoder)
        features = [torch.randn(BATCH_SIZE, 16, FEATURE_DIM) for _ in block_indices]
        out = head(features)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)

    def test_segformer_on_multi_scale_features(self) -> None:
        scale_dims = [8, 16, 32, 64]
        head = get_seg_head(
            decoder_type="segformer",
            output_type="multi_scale",
            feature_dim=FEATURE_DIM,
            num_classes=NUM_CLASSES,
            image_size=IMAGE_SIZE,
            scale_dims=scale_dims,
        )
        assert isinstance(head, SegFormerMultiScaleDecoder)
        sizes = [16, 8, 4, 2]
        features = [
            torch.randn(BATCH_SIZE, dim, size, size)
            for dim, size in zip(scale_dims, sizes)
        ]
        out = head(features)
        assert out.shape == (BATCH_SIZE, NUM_CLASSES, IMAGE_SIZE, IMAGE_SIZE)

    def test_unknown_decoder_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown decoder_type"):
            get_seg_head(
                decoder_type="unet",
                output_type="patch_tokens",
                feature_dim=FEATURE_DIM,
                num_classes=NUM_CLASSES,
                image_size=IMAGE_SIZE,
            )

    @pytest.mark.parametrize(
        ("decoder_type", "output_type"),
        [
            (decoder, output_type)
            for decoder, supported in DECODER_COMPATIBILITY.items()
            for output_type in ("patch_tokens", "spatial_map", "multi_scale")
            if output_type not in supported
        ],
    )
    def test_incompatible_output_type_raises(
        self, decoder_type: str, output_type: str
    ) -> None:
        with pytest.raises(ValueError, match="does not support"):
            get_seg_head(
                decoder_type=decoder_type,
                output_type=output_type,
                feature_dim=FEATURE_DIM,
                num_classes=NUM_CLASSES,
                image_size=IMAGE_SIZE,
            )

    def test_setr_mla_requires_block_indices(self) -> None:
        with pytest.raises(ValueError, match="block_indices"):
            get_seg_head(
                decoder_type="setr_mla",
                output_type="patch_tokens",
                feature_dim=FEATURE_DIM,
                num_classes=NUM_CLASSES,
                image_size=IMAGE_SIZE,
            )

    def test_segformer_requires_scale_dims(self) -> None:
        with pytest.raises(ValueError, match="scale_dims"):
            get_seg_head(
                decoder_type="segformer",
                output_type="multi_scale",
                feature_dim=FEATURE_DIM,
                num_classes=NUM_CLASSES,
                image_size=IMAGE_SIZE,
            )
