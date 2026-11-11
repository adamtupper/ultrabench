# Examples

These scripts show how to train a model on an UltraBench task and score it. They are not part of the `ultrabench` package. Copy and adapt them to your own framework.

| Script | What it trains |
|--------|----------------|
| `classification.py` | A linear probe on a frozen backbone, for any classification task. |
| `segmentation.py` | A segmentation decoder (`upsampler`, `setr_mla` or `segformer`) on a frozen backbone, for any segmentation task. |
| `common.py` | Helpers both scripts share: loading a task's splits, building a backbone, and scoring the predictions. |

Each script loads the task's `train` split and holds out part of it for validation. It keeps the head with the best validation score, writes the predictions on the `test.256` split to `<output-dir>/predictions.csv`, and scores them against the published ground truth with `ultrabench.evaluate`.

## Running the examples

From a clone of the repository, `uv sync` installs the `examples` dependency group along with the package. Then run a script:

```bash
uv run examples/segmentation.py \
  --task tn3k_segmentation \
  --backbone ultrabench.models.dinov3.dinov3_backbone.DINOv3ViTBackbone \
  --backbone-kwargs '{"model_variant": "dinov3_vitb16",
    "repo_dir": "path/to/dinov3-repo",
    "checkpoint_path": "path/to/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"}' \
  --output-dir outputs/tn3k_segmentation
```

- `--task` names a task from `src/ultrabench/evaluate/task_configs/`, such as `tn3k_classification` or `busbra_segmentation`. The task config supplies the dataset, revision, columns and number of classes.
- `--backbone` is the import path of any `Backbone` subclass in `ultrabench.models`. Pass its constructor arguments as a JSON object in `--backbone-kwargs`. Each backbone loads its own pretrained weights, so you download those first.
- `--path` gives the raw data directory of the local datasets (GBCU and Open Kidney).
- `--max-steps 5` runs a quick smoke test.

Run a script with `--help` for the full list of options.

If you installed UltraBench from PyPI rather than from a clone, install the `models` extra and the packages listed in the `examples` group of [`pyproject.toml`](../pyproject.toml), then run the scripts with `python`.
