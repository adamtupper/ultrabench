import typer

from .datasets import (
    aul,
    butterfly,
    camus,
    fatty_liver,
    gbcu,
    mmotu,
    open_kidney,
    pocus,
    psfhs,
    stanford_thyroid,
)
from .download import (
    download_aul,
    download_butterfly,
    download_camus,
    download_fatty_liver,
    download_pocus,
    download_psfhs,
)

app = typer.Typer(no_args_is_help=True)

# Commands for downloadable datasets
_DOWNLOADABLE = [
    ("aul", download_aul, aul),
    ("butterfly", download_butterfly, butterfly),
    ("camus", download_camus, camus),
    ("fatty-liver", download_fatty_liver, fatty_liver),
    ("pocus", download_pocus, pocus),
    ("psfhs", download_psfhs, psfhs),
]
for _name, _dl_fn, _process_fn in _DOWNLOADABLE:
    _sub = typer.Typer(no_args_is_help=True)
    _sub.command("dl")(_dl_fn)
    _sub.command("process")(_process_fn)
    app.add_typer(_sub, name=_name)

# Commands for remaining datasets
app.command()(gbcu)
app.command()(mmotu)
app.command()(open_kidney)
app.command()(stanford_thyroid)


if __name__ == "__main__":
    app()
