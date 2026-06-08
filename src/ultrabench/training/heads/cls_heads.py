import torch.nn as nn


def get_cls_head(probe_type: str, feature_dim: int, num_classes: int) -> nn.Module:
    """
    Factory for classification probe heads.

    Args:
        probe_type: 'linear' or 'attentive'
        feature_dim: Encoder output feature dimension
        num_classes: Number of output classes

    Returns:
        An nn.Module that maps [B, feature_dim] -> [B, num_classes]
    """
    if probe_type == "linear":
        return nn.Linear(feature_dim, num_classes)
    elif probe_type == "attentive":
        raise NotImplementedError("Attentive probe head not implemented yet.")
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")
