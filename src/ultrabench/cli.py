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

app = typer.Typer(no_args_is_help=True)
app.command()(aul)
app.command()(butterfly)
app.command()(camus)
app.command()(fatty_liver)
app.command()(gbcu)
app.command()(mmotu)
app.command()(open_kidney)
app.command()(pocus)
app.command()(psfhs)
app.command()(stanford_thyroid)


if __name__ == "__main__":
    app()
