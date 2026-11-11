"""Task heads that map backbone features to class or segmentation logits."""

from .classification import get_cls_head
from .segmentation import (
    DECODER_COMPATIBILITY,
    SegFormerMultiScaleDecoder,
    SETRMLADecoder,
    SimpleUpsamplerHead,
    get_seg_head,
)

__all__ = [
    "DECODER_COMPATIBILITY",
    "SETRMLADecoder",
    "SegFormerMultiScaleDecoder",
    "SimpleUpsamplerHead",
    "get_cls_head",
    "get_seg_head",
]
