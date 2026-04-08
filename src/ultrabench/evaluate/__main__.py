"""Entry point for the evaluation layer.

Usage:

    python -m ultrabench.evaluate task=segmentation ground_truth=... predictions=...
"""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Build the configured evaluator and run evaluation.

    Args:
        cfg: Hydra configuration.
    """
    evaluator = hydra.utils.instantiate(cfg.task)
    metrics = evaluator.evaluate()
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
