"""
Downstream Evaluation Pipeline with Hydra Config
"""

import logging
import os

import hydra
import lightning.pytorch as pl
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Import your config
from ultrabench.training.lightning import DownstreamPipeline
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

    if cfg.task.type != cfg.head.type:
        raise ValueError(
            f"Mismatch between task type '{cfg.task.type}' and head type "
            f"'{cfg.head.type}'. Select an appropriate head for the task."
        )

    mode_name = "{}_{}".format(
        HydraConfig.get().runtime.choices.head,  # ty:ignore[unresolved-attribute]
        HydraConfig.get().runtime.choices.mode,  # ty:ignore[unresolved-attribute]
    )

    logger.info("=" * 80)
    logger.info(
        "Running: %s | dataset_task=%s | mode=%s | seed=%s | shot=%s",
        HydraConfig.get().runtime.choices.model,  # ty:ignore[unresolved-attribute]
        HydraConfig.get().runtime.choices.task,  # ty:ignore[unresolved-attribute]
        mode_name,
        cfg.optimization.seed,
        cfg.data.shot_pct,
    )
    logger.info("=" * 80)

    mode_output_dir = os.path.join(HydraConfig.get().runtime.output_dir, mode_name)
    os.makedirs(mode_output_dir, exist_ok=True)

    result = pipeline.run_downstream(cfg=cfg, output_dir=mode_output_dir)
    test_metrics = result["test_metrics"]
    pd.DataFrame(test_metrics, index=[0]).to_csv(
        os.path.join(mode_output_dir, "test_metrics.csv"), index=False
    )

    logger.info(f"Results:\n{OmegaConf.to_yaml(result)}")
    logger.info("Run completed!")


if __name__ == "__main__":
    main()
