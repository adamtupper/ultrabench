import logging
import os
import random
from typing import Dict

import numpy as np
import torch
import yaml
from omegaconf import DictConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_model_label(cfg: DictConfig) -> str:
    baseline = cfg.model.get("baseline") if "baseline" in cfg.model else None
    variant = cfg.model.get("variant") if "variant" in cfg.model else None
    checkpoint_name = (
        cfg.model.get("checkpoint_name") if "checkpoint_name" in cfg.model else None
    )

    model_name = variant or checkpoint_name or "unnamed_model"
    return f"{baseline}:{model_name}" if baseline else model_name


def set_global_seed(seed: int = 42):
    """
    Set random seeds for all libraries to ensure reproducible results

    Args:
        seed: Random seed value (default: 42)
    """

    # Python's built-in random module
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    logger.info(f"Global random seed set to: {seed}")

    return seed
