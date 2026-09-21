"""CLI for tricolour."""

import os
import warnings
from enum import Enum
from importlib.resources import files as resource_files
from os.path import join as pjoin
from typing import Annotated, List, Optional

import typer
from numpy.exceptions import VisibleDeprecationWarning

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
  ray_cluster_address: Annotated[
    Optional[str],
    typer.Option(
      "--ray-cluster-address",
      "-rca",
      help="ray cluster address",
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
  # deprecated, but still supported for now
  scan_numbers: Annotated[
    Optional[str],
    typer.Option(
      "--scan-numbers", help="[Deprecated, use -sn instead] Scan numbers to flag. Defaults to flagging all."
    ),
  ] = None,
  subtract_model_variable: Annotated[
    Optional[str],
    typer.Option(
      "--subtract-model-variable",
      "-smv",
      help=("Subtracts specified variable from the data variable. Flagging will proceed on residual data."),
    ),
  ] = None,
  subtract_model_column: Annotated[
    Optional[str],
    typer.Option(
      "--subtract-model-column",
      "-smc",
      help=(
        "[Deprecated, use -smv instead] Subtracts specified column from the data column. "
        "Flagging will proceed on residual data."
      ),
    ),
  ] = None,
) -> None:
  """A Radio Astronomy Flagging Software Suite"""
  import logging

  import ray
  import xarray
  from rarg_python_patterns import Multiton
  from ray import serve
  from ray.serve.handle import DeploymentHandle

  from tricolour.core.application.backend import infer_and_import_backend
  from tricolour.core.application.config import load_config, log_configuration
  from tricolour.core.application.implementation import DataLoader, DataWriter, Flagger, Tricolour
  from tricolour.core.kernels.mask import load_masks

  backend, open_kwargs = infer_and_import_backend(ms)

  # If supplied, connect to the ray cluster
  if ray_cluster_address is not None:
    ray.init(address=ray_cluster_address, logging_level=logging.ERROR)
  else:
    ray.init(
      logging_level=logging.ERROR,
      log_to_driver=False,
      logging_config=ray.LoggingConfig(encoding="JSON", log_level="INFO"),
    )
  datatree = Multiton(xarray.open_datatree, ms, **open_kwargs)
  config = Multiton(load_config, config).with_serialise_instance()
  masks = Multiton(load_masks, dilate_masks).with_serialise_instance()
  log_configuration(flagging_strategy, config.instance)

  # NOTE: `max_ongoing_requests` is a deployment-level option, not an
  # AutoscalingConfig field -- placing it in the autoscaling config silently
  # discards it. One request per replica is expressed by the pair below.
  autoscaling_config = {
    "upscale_delay_s": 0.0,
    "min_replicas": 1,
    "initial_replicas": 1,
    "target_ongoing_requests": 1,
    # The defaults (10s/30s) are far too sluggish for a batch job that only
    # runs for a few minutes; the deployment would never reach max_replicas.
    "metrics_interval_s": 1.0,
    "look_back_period_s": 5.0,
    "max_replicas": nworkers,
  }
  common_options = {
    "num_replicas": "auto",
    "autoscaling_config": autoscaling_config,
    "max_ongoing_requests": 1,
  }

  # Only the variables the Flagger actually touches are worth moving through
  # the object store. WEIGHT alone is a third of an MSv4 partition's bytes.
  load_variables = [data_variable, "FLAG", "UVW"]
  if subtract_model_column is not None:
    warnings.warn(
      "Switch subtract-model-column is deprecated. Use subtract-model-variable instead.",
      category=VisibleDeprecationWarning,
      stacklevel=2,
    )
    if subtract_model_variable is not None:
      raise ValueError("Cannot simultaneously specify subtract-model-column and subtract-model-variable.")
    load_variables.append(subtract_model_column)

  if subtract_model_variable is not None:
    load_variables.append(subtract_model_variable)

  if scan_numbers is not None:
    warnings.warn("Switch scan-numbers is deprecated. Use scan-names instead.", category=VisibleDeprecationWarning)
    if scan_names is not None:
      raise ValueError("Cannot simultaneously specify scan-numbers and scan-names.")

  data_loader = DataLoader.options(**common_options, ray_actor_options={"num_cpus": 0}).bind(
    datatree=datatree, variables=load_variables
  )

  flagger_options = {
    **common_options,
    "autoscaling_config": {**autoscaling_config, "max_replicas": max(1, nworkers - 2)},
  }

  flagger = Flagger.options(**flagger_options).bind(
    masks=masks,
    baseline_chunks=baseline_chunks,
    flagging_strategy=flagging_strategy,
    flagging_config=config,
    ignore_flags=ignore_flags,
    data_variable=data_variable,
    model_variable=subtract_model_variable,
    # multithreaded=True,
  )
  # A single writer: concurrent writers block on the CASA table lock and one
  # will eventually wedge, fail its Serve health check and get force-killed,
  # taking the job with it. Writing flags back is cheap (~160MB for this MS)
  # so serialising it costs little, and the queue here is useful backpressure.
  # A single writer, since writes to one MS serialise anyway.
  #
  # NOTE: xarray-ms's to_msv2() reopens the MS on every call and leaks OS
  # threads and file descriptors as it goes (reads do not leak). After a few
  # dozen writes the call deadlocks with every thread parked on a futex.
  # Serve's default health check is currently the only thing that recovers
  # from this -- it kills the wedged replica and retries the request -- so do
  # NOT raise health_check_timeout_s here: doing so turns an intermittent
  # failure into an indefinite hang. The real fix belongs upstream.
  writer = DataWriter.options(num_replicas=1, max_ongoing_requests=1).bind(path=ms, backend=backend)

  app = Tricolour.bind(
    datatree=datatree,
    time_chunks=time_chunks,
    freq_chunks=frequency_chunks,
    baseline_chunks=baseline_chunks,
    field_names=field_names,
    scan_names=scan_numbers if scan_numbers is not None else scan_names,
    data_loader=data_loader,
    flagger=flagger,
    data_writer=writer,
    # Enough outstanding work to keep every replica of all three stages busy
    max_in_flight=4 * nworkers,
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
