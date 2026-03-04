"""CLI for tricolour."""

import os
from enum import Enum
from importlib.resources import files as resource_files
from os.path import join as pjoin
from typing import Annotated, List, Optional

import typer

app = typer.Typer(
  name="tricolour", help="A Radio Astronomy Flagging Software Suite", no_args_is_help=True, invoke_without_command=True
)


DEFAULT_CONFIG = pjoin(resource_files("tricolour"), "conf", "default.yaml")


class FlaggingStrategy(str, Enum):
  standard = "standard"
  polarisation = "polarisation"
  total_power = "total_power"


class WindowBackend(str, Enum):
  numpy = "numpy"
  zarr_disk = "zarr-disk"


@app.callback()
def callback(
  ms: str,
  config: Annotated[
    str,
    typer.Option(
      "--config",
      "-c",
      help=("YAML config file containing parameters for the flagger in the 'sum_threshold' key"),
    ),
  ] = DEFAULT_CONFIG,
  ignore_flags: Annotated[
    bool,
    typer.Option("--ignore-flags", "-if", help="Ignore existing flags in the Measurement Set"),
  ] = False,
  flagging_strategy: Annotated[
    FlaggingStrategy,
    typer.Option(
      "--flagging-strategy",
      "-fs",
      help=(
        "Flagging Strategy. "
        "If 'standard' all correlations in the visibility are flagged independently. "
        "If 'polarisation' the polarised intensity sqrt(Q^2 + U^2 + V^2) is calculated "
        "and used to flag all correlations in the visibility. "
        "If 'total_power' the available quadrature power sqrt(I^2 + Q^2 + U^2 + V^2) "
        "or a subset is computed and used to flag all correlations in the visibility."
      ),
    ),
  ] = FlaggingStrategy.standard,
  row_chunks: Annotated[
    int,
    typer.Option(
      "--row-chunks",
      "-rc",
      help=(
        "Hint indicating the number of Measurement Set rows to read in a single chunk. "
        "Smaller and larger numbers will tend to respectively decrease or increase both "
        "memory usage and computational efficiency."
      ),
    ),
  ] = 10000,
  baseline_chunks: Annotated[
    int,
    typer.Option("--baseline-chunks", "-bc", help="Number of baselines in a window chunk"),
  ] = 16,
  nworkers: Annotated[
    int,
    typer.Option(
      "--nworkers",
      "-nw",
      help=(
        "Number of workers (threads) to use. By default, set to the number of logical CPUs "
        "on the system. Many workers can also affect memory usage on systems with many cores."
      ),
    ),
  ] = os.cpu_count(),
  dilate_masks: Annotated[
    Optional[str],
    typer.Option("--dilate-masks", "-dm", help="Number of channels to dilate as int or string with units"),
  ] = None,
  data_column: Annotated[
    str,
    typer.Option("--data-column", "-dc", help="Name of visibility data column to flag"),
  ] = "DATA",
  field_names: Annotated[
    Optional[List[str]],
    typer.Option("--field-names", "-fn", help="Name(s) of fields to flag. Defaults to flagging all."),
  ] = None,
  scan_numbers: Annotated[
    Optional[str],
    typer.Option("--scan-numbers", "-sn", help="Scan numbers to flag (casa style range like 5~9)"),
  ] = None,
  disable_post_mortem: Annotated[
    bool,
    typer.Option(
      "--disable-post-mortem",
      "-dpm",
      help=(
        "Disable the default behaviour of starting the Interactive Python Debugger upon an "
        "unhandled exception. This may be necessary for batch pipelining."
      ),
    ),
  ] = False,
  window_backend: Annotated[
    WindowBackend,
    typer.Option(
      "--window-backend",
      "-wb",
      help=(
        "Visibility and flag data is re-ordered from a MS row ordering into time-frequency "
        "windows ordered by baseline. For smaller problems, it may be possible to pack a "
        "couple of scans worth of visibility data into memory, but for larger problem sizes, "
        "it is necessary to reorder the data on disk."
      ),
    ),
  ] = WindowBackend.numpy,
  temporary_directory: Annotated[
    Optional[str],
    typer.Option("--temporary-directory", "-td", help="Directory location of temporary data"),
  ] = None,
  subtract_model_column: Annotated[
    Optional[str],
    typer.Option(
      "--subtract-model-column",
      "-smc",
      help=("Subtracts specified column from data column specified. Flagging will proceed on residual data."),
    ),
  ] = None,
) -> None:
  """A Radio Astronomy Flagging Software Suite"""
  print(f"Flagging {ms}")


# Register subcommands below. Imports go here (bottom) to avoid circular imports.
from tricolour.cli.onboard import onboard  # noqa: E402

app.command(name="onboard")(onboard)

__all__ = ["app"]
