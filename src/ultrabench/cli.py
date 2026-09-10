import warnings

import typer

from .download import (
    download_aul,
    download_busbra,
    download_camus,
    download_covid_blues,
    download_fetal_planes,
    download_masld,
    download_psfhs,
    download_stmus,
)
from .process import (
    process_aul,
    process_busbra,
    process_camus,
    process_covid_blues,
    process_fetal_planes,
    process_gbcu,
    process_masld,
    process_mmotu,
    process_open_kidney,
    process_psfhs,
    process_stmus,
    process_tn3k,
)

# Filter scikit-image deprecation warnings. Must be migrated before 0.28.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*(`min_size`|`area_threshold`|`binary_erosion`|`binary_dilation`).*",
)

app = typer.Typer(no_args_is_help=True)
download_app = typer.Typer(no_args_is_help=True, help="Download a dataset.")
process_app = typer.Typer(
    no_args_is_help=True,
    help="Process a raw dataset into the standardized format.",
)

app.add_typer(download_app, name="download")
app.add_typer(process_app, name="process")

_DOWNLOADABLE = [
    ("aul", download_aul),
    ("busbra", download_busbra),
    ("camus", download_camus),
    ("covid-blues", download_covid_blues),
    ("fetal-planes", download_fetal_planes),
    ("masld", download_masld),
    ("psfhs", download_psfhs),
    ("stmus", download_stmus),
]
for _name, _dl_fn in _DOWNLOADABLE:
    download_app.command(_name)(_dl_fn)

_ALL = [
    ("aul", process_aul),
    ("covid-blues", process_covid_blues),
    ("fetal-planes", process_fetal_planes),
    ("busbra", process_busbra),
    ("camus", process_camus),
    ("masld", process_masld),
    ("gbcu", process_gbcu),
    ("mmotu", process_mmotu),
    ("open-kidney", process_open_kidney),
    ("psfhs", process_psfhs),
    ("stmus", process_stmus),
    ("tn3k", process_tn3k),
]
for _name, _process_fn in _ALL:
    process_app.command(_name)(_process_fn)


if __name__ == "__main__":
    app()
