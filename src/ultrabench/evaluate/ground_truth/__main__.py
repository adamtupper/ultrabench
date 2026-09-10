"""Entry point for building the benchmark's ground-truth CSVs.

Usage::

    python -m ultrabench.evaluate.ground_truth output_dir=<dir>
    python -m ultrabench.evaluate.ground_truth output_dir=<dir> \\
        tasks='[tn3k_segmentation]' overwrite=true

The ground truth is shared by every model, so build it once and point
the evaluation at it.
"""

import hydra
from omegaconf import DictConfig

from ultrabench.evaluate.ground_truth.builder import build_all_ground_truth


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build the ground truth for the configured tasks.

    Args:
        cfg: Hydra configuration.
    """
    build_all_ground_truth(
        output_dir=cfg.output_dir,
        tasks=cfg.tasks,
        overwrite=cfg.overwrite,
    )


if __name__ == "__main__":
    main()
