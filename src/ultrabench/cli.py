import typer

from .datasets import aul, butterfly

app = typer.Typer()
app.command()(aul)
app.command()(butterfly)


if __name__ == "__main__":
    app()
