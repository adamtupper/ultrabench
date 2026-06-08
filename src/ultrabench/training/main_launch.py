"""
Downstream Evaluation Pipeline with Hydra Config
"""

import logging
import os

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Import your config
from ultrabench.training.config.config import register_configs
from ultrabench.training.setup_ptl import DownstreamPipeline
from ultrabench.utils.common import get_model_label, set_global_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for downstream evaluation pipeline using Hydra.
    Hydra's multi-run handles sweeps over seeds, shot_pcts, datasets, etc.

    Args:
        cfg: Hydra configuration object
    """
    # Print config
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Set seed for this run
    set_global_seed(cfg.optimization.seed)

    # Initialize pipeline
    pipeline = DownstreamPipeline(data_dir=cfg.system.data_dir)

    task = HydraConfig.get().runtime.choices.task  # ty:ignore[unresolved-attribute]
    task_type = cfg.task.type

    # Guard: catch mismatched task type / mode before anything runs
    def _mode_is_none(mc):
        return mc is None or getattr(mc, "name", "none").lower() == "none"

    if task_type == "classification" and _mode_is_none(cfg.cls_mode):
        raise ValueError(
            f"'{task}' is a classification task but cls_mode is the 'none' sentinel. "
            "Override with: 'mode@cls_mode=linear_probe' on the CLI."
        )
    if task_type == "segmentation" and _mode_is_none(cfg.seg_mode):
        raise ValueError(
            f"'{task}' is a segmentation task but seg_mode is the 'none' sentinel. "
            "Override with: 'mode@seg_mode=upsampler_probe' on the CLI."
        )

    mode_name = (
        cfg.cls_mode.name if task_type == "classification" else cfg.seg_mode.name
    )

    logger.info("=" * 80)
    logger.info(
        "Running: %s | dataset_task=%s | mode=%s | seed=%s | shot=%s",
        get_model_label(cfg),
        task,
        mode_name,
        cfg.optimization.seed,
        cfg.data.shot_pct,
    )
    logger.info("=" * 80)

    mode_output_dir = os.path.join(HydraConfig.get().runtime.output_dir, mode_name)
    os.makedirs(mode_output_dir, exist_ok=True)

    result = pipeline.run_downstream(cfg=cfg, output_dir=mode_output_dir)
    test_metrics = result["test_metrics"]
    pd.DataFrame(test_metrics, index=[0]).to_csv(  # ty:ignore[invalid-argument-type]
        os.path.join(mode_output_dir, "test_metrics.csv"), index=False
    )

    logger.info(f"Results:\n{OmegaConf.to_yaml(result)}")
    logger.info("Run completed!")


if __name__ == "__main__":
    # Register configs
    register_configs()
    main()
