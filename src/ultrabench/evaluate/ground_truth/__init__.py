"""Generation of the ground-truth CSVs the evaluation layer scores against.

See `docs/evaluation_protocol.md` for how the resulting per-task scores
are reduced to a score per category.
"""

from .builder import build_all_ground_truth, build_ground_truth, ground_truth_path

__all__ = ["build_all_ground_truth", "build_ground_truth", "ground_truth_path"]
