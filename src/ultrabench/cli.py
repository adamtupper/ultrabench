import typer

from .download import (
    download_aul,
    download_busbra,
    download_butterfly,
    download_camus,
    download_covid_blues,
    download_fetal_planes,
    download_masld,
    download_pocus,
    download_psfhs,
    download_stmus,
)
from .process import (
    process_aul,
    process_busbra,
    process_butterfly,
    process_camus,
    process_covid_blues,
    process_fetal_planes,
    process_gbcu,
    process_masld,
    process_mmotu,
    process_open_kidney,
    process_pocus,
    process_psfhs,
    process_stanford_thyroid,
    process_stmus,
    process_tn3k,
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
    ("butterfly", download_butterfly),
    ("camus", download_camus),
    ("covid-blues", download_covid_blues),
    ("fetal-planes", download_fetal_planes),
    ("masld", download_masld),
    ("pocus", download_pocus),
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
    ("butterfly", process_butterfly),
    ("camus", process_camus),
    ("masld", process_masld),
    ("gbcu", process_gbcu),
    ("mmotu", process_mmotu),
    ("open-kidney", process_open_kidney),
    ("pocus", process_pocus),
    ("psfhs", process_psfhs),
    ("stanford-thyroid", process_stanford_thyroid),
    ("stmus", process_stmus),
    ("tn3k", process_tn3k),
]
for _name, _process_fn in _ALL:
    process_app.command(_name)(_process_fn)


if __name__ == "__main__":
    app()
