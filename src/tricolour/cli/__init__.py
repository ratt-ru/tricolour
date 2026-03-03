"""CLI for tricolour."""

import typer

app = typer.Typer(
  name="tricolour",
  help="A Radio Astronomy Flagging Software Suite",
  no_args_is_help=True,
)


@app.callback()
def callback() -> None:
  """A Radio Astronomy Flagging Software Suite"""
  pass


# Register subcommands below. Imports go here (bottom) to avoid circular imports.
from tricolour.cli.onboard import onboard  # noqa: E402

app.command(name="onboard")(onboard)

__all__ = ["app"]
