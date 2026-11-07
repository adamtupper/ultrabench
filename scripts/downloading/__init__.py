"""Download the raw datasets that are publicly available for download."""

from collections.abc import Callable

from .aul import download_aul
from .busbra import download_busbra
from .camus import download_camus
from .covid_blues import download_covid_blues
from .fetal_planes import download_fetal_planes
from .masld import download_masld
from .psfhs import download_psfhs
from .stmus import download_stmus

DOWNLOADERS: dict[str, Callable[[str], None]] = {
    "aul": download_aul,
    "busbra": download_busbra,
    "camus": download_camus,
    "covid-blues": download_covid_blues,
    "fetal-planes": download_fetal_planes,
    "masld": download_masld,
    "psfhs": download_psfhs,
    "stmus": download_stmus,
}
"""The download function of each dataset, keyed by its CLI name."""
