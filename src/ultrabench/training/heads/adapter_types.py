from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch.nn as nn

TransformFn = Callable[..., Any]


@dataclass
class EncoderBundle:
    encoder: nn.Module
    transforms: Optional[TransformFn] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
