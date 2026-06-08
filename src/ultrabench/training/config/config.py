from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ModelConfig:
    baseline: Optional[str] = None
    variant: Optional[str] = None
    checkpoint_path: str = MISSING
    checkpoint_config: str = MISSING
    checkpoint_name: str = MISSING
    pretrained: bool = True
    output_type: str = "patch_tokens"  # patch_tokens | spatial_map | multi_scale | cls
    has_cls_token: bool = False


@dataclass
class DataConfig:
    setup: str = "ultrabench"
    version: str = "v2.0.0"
    image_size: int = 224
    preprocessing_fn: Optional[str] = (
        None  # Optional: name of preprocessing function from user_preprocessing.py
    )
    shot_pct: float = 1.0
    train_val_split: float = 0.8  # Train/validation split ratio


@dataclass
class OptimizationConfig:
    batch_size: int = 32
    num_epochs: int = 150
    # learning_rate: float = 1.0e-3
    # weight_decay: float = 1.0e-4
    patience: Optional[int] = 15
    seed: int = 42
    accum_grad: int = 1


@dataclass
class SystemConfig:
    max_parallel: int = 1
    data_dir: str = MISSING
    output_dir: str = MISSING
    data_loader_workers: int = 0
    log_every_n_steps: int = 50
    enable_progress_bar: bool = True


@dataclass
class UltraBenchConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cls_mode: Optional[Any] = None
    seg_mode: Optional[Any] = None
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="config", node=UltraBenchConfig)
