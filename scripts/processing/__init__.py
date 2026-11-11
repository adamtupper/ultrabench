"""Process the raw datasets into the UltraBench format.

The processors for the datasets published ready to use live here. The
datasets the package builds on the user's machine are processed by
`local.process_local`, which wraps the builders in
`ultrabench.datasets.builders`.

Each processor is imported only when it is run, because several of them
need optional dependencies (nibabel, SimpleITK, openpyxl) that the
others do not.
"""

PUBLISHED_PROCESSORS = {
    "aul": ("aul", "process_aul"),
    "busbra": ("busbra", "process_busbra"),
    "camus": ("camus", "process_camus"),
    "fetal-planes": ("fetal_planes", "process_fetal_planes"),
    "masld": ("masld", "process_masld"),
    "psfhs": ("psfhs", "process_psfhs"),
    "stmus": ("stmus", "process_stmus"),
    "tn3k": ("tn3k", "process_tn3k"),
}
"""The module and function processing each published dataset."""
