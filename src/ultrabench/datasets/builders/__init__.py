"""Build the datasets that are processed on the user's machine.

`ultrabench.load_dataset` builds the raw and local datasets (GBCU,
MMOTU, Open Kidney and COVID-BLUES) on first use. Each module exposes a
`build_*` function that returns a `BuildResult` and a
`write_ground_truth` function that writes the test splits'
ground-truth CSVs.
"""
