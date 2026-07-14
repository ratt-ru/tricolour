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


DEFAULT_CONFIG = pjoin(resource_files("tricolour"), "conf", "default_config.yaml")


class FlaggingStrategy(str, Enum):
  standard = "standard"
  polarisation = "polarisation"
  total_power = "total_power"


@app.callback()
def callback(
  ctx: typer.Context,
  ms: str,
  ray_scheduler_address: Annotated[
    Optional[str],
    typer.Option(
      "--ray-scheduler-address",
      "-rsa",
      help="ray scheduler address",
    ),
  ] = None,
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
  time_chunks: Annotated[
    int,
    typer.Option(
      "--time-chunks",
      "-tc",
      help=(
        "Hint indicating the number of Measurement Set timestamps to read in a single chunk. "
        "Smaller and larger numbers will tend to respectively decrease or increase both "
        "memory usage and computational efficiency."
      ),
    ),
  ] = 100,
  baseline_chunks: Annotated[
    int,
    typer.Option("--baseline-chunks", "-bc", help="Number of baselines flagged in a single thread"),
  ] = 16,
  frequency_chunks: Annotated[
    Optional[int], typer.Option("--frequency-chunks", "-fc", help="Number of frequencies in a channel chunk")
  ] = None,
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
  data_variable: Annotated[
    str,
    typer.Option("--data-variable", "-dv", help="Name of visibility data variable to flag"),
  ] = "VISIBILITY",
  field_names: Annotated[
    Optional[List[str]],
    typer.Option("--field-names", "-fn", help="Name(s) of fields to flag. Defaults to flagging all."),
  ] = None,
  scan_names: Annotated[
    Optional[str],
    typer.Option("--scan-names", "-sn", help="Scan names to flag. Defaults to flagging all."),
  ] = None,
  subtract_model_variable: Annotated[
    Optional[str],
    typer.Option(
      "--subtract-model-variable",
      "-smv",
      help=("Subtracts specified variable from the data variable. Flagging will proceed on residual data."),
    ),
  ] = None,
) -> None:
  """A Radio Astronomy Flagging Software Suite"""
  from rarg_python_patterns import Multiton
  from ray import serve
  from ray.serve.handle import DeploymentHandle

  from tricolour.core.application.backend import open_datatree
  from tricolour.core.application.config import load_config, log_configuration
  from tricolour.core.application.supervisor2 import DataLoader, DataWriter, Flagger, Tricolour
  from tricolour.core.kernels.mask import load_masks

  datatree = Multiton(open_datatree, ms)
  config = Multiton(load_config, config).with_serialise_instance()
  masks = Multiton(load_masks, dilate_masks).with_serialise_instance()
  log_configuration(flagging_strategy, config.instance)

  autoscaling_config = {
    "upscale_delay_s": 1.0,
    "min_replicas": 1,
    "initial_replicas": 1,
    "max_ongoing_requests": 1,
    "max_replicas": nworkers,
  }
  common_options = {"num_replicas": "auto", "autoscaling_config": autoscaling_config}

  data_loader = DataLoader.options(**common_options, ray_actor_options={"num_cpus": 0}).bind(
    datatree=datatree, variables="ALL"
  )

  flagger = Flagger.options(**common_options).bind(
    masks=masks,
    baseline_chunks=baseline_chunks,
    flagging_strategy=flagging_strategy,
    flagging_config=config,
    ignore_flags=ignore_flags,
    data_variable=data_variable,
    model_variable=subtract_model_variable,
  )
  writer = DataWriter.options(**common_options).bind()

  app = Tricolour.bind(
    datatree=datatree,
    time_chunks=time_chunks,
    freq_chunks=frequency_chunks,
    field_names=field_names,
    scan_names=scan_names,
    data_loader=data_loader,
    flagger=flagger,
    data_writer=writer,
  )

  handle: DeploymentHandle = serve.run(app, name="tricolour")
  for line in handle.remote().result():
    print(line)


# Register subcommands below. Imports go here (bottom) to avoid circular imports.
from tricolour.cli.onboard import onboard  # noqa: E402

app.command(name="onboard")(onboard)

__all__ = ["app"]

if __name__ == "__main__":
  app()
