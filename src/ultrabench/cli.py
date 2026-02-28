import typer

from .download import (
    download_aul,
    download_butterfly,
    download_camus,
    download_fatty_liver,
    download_pocus,
    download_psfhs,
)
from .process import (
    process_aul,
    process_butterfly,
    process_camus,
    process_fatty_liver,
    process_gbcu,
    process_mmotu,
    process_open_kidney,
    process_pocus,
    process_psfhs,
    process_stanford_thyroid,
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
    ("butterfly", download_butterfly),
    ("camus", download_camus),
    ("fatty-liver", download_fatty_liver),
    ("pocus", download_pocus),
    ("psfhs", download_psfhs),
]
for _name, _dl_fn in _DOWNLOADABLE:
    download_app.command(_name)(_dl_fn)

_ALL = [
    ("aul", process_aul),
    ("butterfly", process_butterfly),
    ("camus", process_camus),
    ("fatty-liver", process_fatty_liver),
    ("gbcu", process_gbcu),
    ("mmotu", process_mmotu),
    ("open-kidney", process_open_kidney),
    ("pocus", process_pocus),
    ("psfhs", process_psfhs),
    ("stanford-thyroid", process_stanford_thyroid),
]
for _name, _process_fn in _ALL:
    process_app.command(_name)(_process_fn)


if __name__ == "__main__":
    app()
