from __future__ import annotations

import asyncio
import concurrent.futures as cf
import logging
import multiprocessing as mp
import time
from collections import deque
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Generator, Iterable, List, Literal, Tuple, get_args

import numpy as np
import numpy.typing as npt
import xarray
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from msv4_utils import MSv4Backend
from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES
from rarg_python_patterns.multiton import Multiton
from ray import serve
from ray.serve.handle import DeploymentHandle

from tricolour.core.application.dashboard import dashboard_app, index_html, snapshot_stream
from tricolour.core.application.progress import ChunkOccupancy, ChunkReport, RunState
from tricolour.core.kernels.flag_statistics import WindowStatistics, window_stats
from tricolour.core.kernels.flagging import (
  apply_static_mask,
  flag_autos,
  flag_nans_and_zeros,
  sum_threshold_flagger,
  uvcontsub_flagger,
)
from tricolour.core.kernels.stokes import STOKES_TYPES, polarised_intensity, stokes_corr_map
from tricolour.core.types import FlagStrategy

STATISTICS_CHAN_BINS = 10
MISSING_SENTINEL = object()
# Canonical MSv4 ordering
FLAG_DIM_ORDER = ("time", "baseline_id", "frequency", "polarization")
# SumTreshold Flagger ordering
CP_FLAG_DIM_ORDER = ("baseline_id", "polarization", "time", "frequency")


@dataclass(slots=True)
class WorkItem:
  path: str
  region: Dict[str, slice]

  def __hash__(self):
    return hash((self.path, frozenset(self.region.items())))


def chunk_window_stats(
  flags: npt.NDArray,
  dataset: xarray.Dataset,
  ubl: npt.NDArray,
  antenna_names: npt.NDArray,
  nchanbins: int,
  bin_edges: npt.NDArray,
) -> WindowStatistics:
  """Accumulate window statistics for a dataset chunk.

  Statistics are attributed per (field, scan) pair by selecting
  along the time axis, mirroring the per-partition accumulation
  of the original tricolour.

  Parameters
  ----------
  flags:
    Flags in canonical (time, baseline_id, frequency, polarization) order.
  dataset:
    The dataset chunk, supplying scan/field/frequency metadata.
  ubl:
    Unique baselines of shape (baseline, 3).
  antenna_names:
    Antenna names indexed by columns 1 and 2 of ``ubl``.
  nchanbins:
    Number of frequency bins for per-spectral-window histograms.
  bin_edges:
    Frequency bin edges spanning the node's full spectral window.
  """
  ntime = flags.shape[0]
  scan_names = np.broadcast_to(np.atleast_1d(dataset.scan_name.values), (ntime,))
  field_names = np.broadcast_to(np.atleast_1d(dataset.field_name.values), (ntime,))
  chan_freqs = dataset.frequency.values
  ddid = dataset.frequency.spectral_window_name
  stats = WindowStatistics(nchanbins)

  for field, scan in np.unique(np.stack([field_names, scan_names], axis=1), axis=0):
    time_sel = np.logical_and(field_names == field, scan_names == scan)
    stats.update(
      window_stats(
        flags[time_sel],
        ubl,
        chan_freqs,
        antenna_names,
        str(scan),
        str(field),
        ddid,
        nchanbins,
        bin_edges=bin_edges,
      )
    )

  return stats


@serve.deployment(max_replicas_per_node=1)
class DataLoader:
  def __init__(self, datatree: Multiton[xarray.DataTree], variables: Literal["ALL"] | Iterable[str] = "ALL"):
    self._datatree = datatree.instance
    self._variables = variables

  def load(self, item: WorkItem) -> xarray.Dataset:
    """Load dataset specified by item"""
    # Select out the dataset
    dataset = self._datatree[item.path].ds

    if self._variables == "ALL":
      pass
    elif isinstance(self._variables, (tuple, list)):
      # Drop any variables that aren't required
      drop_vars = set(dataset.data_vars) - set(self._variables)
      dataset = dataset.drop_vars(drop_vars)
    else:
      raise TypeError(f'{self._variables} should be "ALL" or a list of data variables')

    return dataset.isel(**item.region).load()


@serve.deployment
class Flagger:
  def __init__(
    self,
    masks: Multiton[Dict[str, npt.NDArray]],
    baseline_chunks: int,
    flagging_strategy: FlagStrategy = "standard",
    flagging_config: Multiton[List[str, Any]] | None = None,
    ignore_flags: bool = False,
    data_variable: str = "VISIBILITY",
    model_variable: str | None = None,
    multithreaded: bool = False,
  ):
    self._pool = cf.ThreadPoolExecutor(max_workers=mp.cpu_count()) if multithreaded else None
    self._masks = masks
    self._ignore_flags = ignore_flags
    self._baseline_chunks = baseline_chunks
    self._flagging_strategy = flagging_strategy
    self._flagging_config = flagging_config or []
    self._data_variable = data_variable
    self._model_variable = model_variable

  def flag(
    self, item: WorkItem, dataset: xarray.Dataset, bin_edges: npt.NDArray
  ) -> tuple[xarray.Dataset, WindowStatistics, WindowStatistics, ChunkReport]:
    concatenated_antennas = np.concat((dataset.baseline_antenna1_name, dataset.baseline_antenna2_name))
    antenna_names, ant_inv = np.unique(concatenated_antennas, return_inverse=True)
    ubl = np.column_stack(
      [dataset.baseline_id.values, ant_inv[: dataset.sizes["baseline_id"]], ant_inv[dataset.sizes["baseline_id"] :]]
    )

    logger = logging.getLogger()
    print(f"Flag {item} begins")
    logger.info("Flag %s begins", item)
    flag_start = time.perf_counter()

    if self._data_variable not in dataset:
      raise ValueError(f"Visibility variable {self._data_variable} not in dataset")

    # Statistics of the flags as they arrived, for comparison against the final flags
    incoming_flags = dataset["FLAG"].transpose(*FLAG_DIM_ORDER).values
    original_stats = chunk_window_stats(incoming_flags, dataset, ubl, antenna_names, STATISTICS_CHAN_BINS, bin_edges)
    # Reduce over baseline and polarisation for the dashboard's (time, frequency)
    # waterfall. Work items tile that plane, so the driver reassembles the full
    # picture by dropping each of these into place.
    original_occupancy = np.count_nonzero(incoming_flags, axis=(1, 3)).astype(np.uint32)

    bp_vis = dataset[self._data_variable].transpose(*CP_FLAG_DIM_ORDER).values
    bp_uvw = dataset["UVW"].transpose("baseline_id", "time", ...).values

    # MSv4 stores FLAG as uint8, where MSv2 (and the original dask tricolour)
    # used bool. Kernels such as uvcontsub_flagger index with the flag array
    # (`vis[flags] = np.nan`); with a uint8 array numpy reads that as integer
    # fancy-indexing on the first axis rather than as a boolean mask, which is
    # both wrong and catastrophically slow. Work in bool and cast back on write.
    flag_dtype = dataset["FLAG"].dtype

    if not self._ignore_flags:
      bp_flags = dataset["FLAG"].transpose(*CP_FLAG_DIM_ORDER).values.astype(bool)
    else:
      bp_flags = np.zeros_like(bp_vis, dtype=bool)

    # Flag the difference of the visibilities and the model
    if self._model_variable:
      if self._model_variable not in dataset:
        raise ValueError(f"Model visibility variable {self._model_variable} not in dataset")

      bp_vis -= dataset[self._model_variable].transpose(*CP_FLAG_DIM_ORDER).values

    # Apply any modifications to the visibilities
    if self._flagging_strategy == "standard":
      pass
    elif self._flagging_strategy == "polarisation":
      stokes_map = stokes_corr_map([STOKES_TYPES[p] for p in dataset.polarization.values])
      stokes_pol = tuple(v for k, v in stokes_map.items() if k != "I")
      bp_vis = polarised_intensity(bp_vis, stokes_pol)
      bp_flags = np.any(bp_flags, axis=1, keepdims=True)
    elif self._flagging_strategy == "total_power":
      stokes_map = stokes_corr_map([STOKES_TYPES[c] for c in dataset.polarization.values])
      stokes_pol = tuple(stokes_map.values())
      bp_vis = polarised_intensity(bp_vis, stokes_pol)
      bp_flags = np.any(bp_flags, axis=1, keepdims=True)
    else:
      raise ValueError(
        f"Invalid flagging strategy {self._flagging_strategy}. Should be one of {list(get_args(FlagStrategy))}"
      )

    # Impose some constraints on our inputs
    bp_flags = np.require(bp_flags, None, ["C", "W"])
    bp_vis = np.require(bp_vis, None, "C")
    original = bp_flags.copy()

    # Apply each strategy in the flagging configuation
    strategy_stats: List[Tuple[str, str, int, int, float]] = []

    for s, strategy in enumerate(self._flagging_config.instance.get("strategies", [])):
      if (task := strategy.get("task", MISSING_SENTINEL)) is MISSING_SENTINEL:
        raise ValueError(f"Strategy '{strategy}' has no task")

      strategy_start = time.perf_counter()

      if task == "sum_threshold":
        if self._pool is None:
          new_flags = sum_threshold_flagger(bp_vis, bp_flags, **strategy["kwargs"])
          np.logical_or(bp_flags, new_flags, out=bp_flags)
        else:
          futures = {}
          nbl = bp_flags.shape[0]

          # Submit sumthresholding work to the threadpool in baseline chunks
          for bl in range(0, nbl, self._baseline_chunks):
            bl_slice = slice(bl, min(nbl, bl + self._baseline_chunks))
            futures[
              self._pool.submit(
                sum_threshold_flagger, bp_vis[bl_slice, ...], bp_flags[bl_slice, ...], **strategy["kwargs"]
              )
            ] = bl_slice

          # Gather the results
          for f in cf.as_completed(futures.keys()):
            assert not (bl_flag_view := bp_flags[futures[f], ...]).flags.owndata
            np.logical_or(bl_flag_view, f.result(), out=bl_flag_view)
      elif task == "uvcontsub_flagger":
        # this task discards previous flags by default during its
        # second iteration. The original flags from MS should be or'd
        # back in afterwards. Flags from steps prior to this one serves
        # only as a "initial guess"
        bp_flags = uvcontsub_flagger(bp_vis, bp_flags, **strategy["kwargs"])
      elif task == "flag_autos":
        np.logical_or(bp_flags, flag_autos(bp_flags, ubl), out=bp_flags)
      elif task == "combine_with_input_flags":
        np.logical_or(bp_flags, original, out=bp_flags)
      elif task == "unflag":
        bp_flags[...] = 0
      elif task == "flag_nans_zeros":
        bp_flags = flag_nans_and_zeros(bp_vis, bp_flags)
      elif task == "apply_static_mask":
        masks = list(self._masks.instance.values())
        chan_freq = dataset.frequency.values
        chan_width = np.broadcast_to(dataset.frequency.channel_width["data"], chan_freq.shape)
        new_flags = apply_static_mask(bp_flags, bp_uvw, masks, chan_freq, chan_width, **strategy["kwargs"])
        accumulation_mode = strategy["kwargs"].get("accumulation_mode", "<none>").strip()
        if accumulation_mode == "or":
          np.logical_or(bp_flags, new_flags, out=bp_flags)
        elif accumulation_mode == "override":
          # override option will override any flags computed previously
          # this may not be desirable so use with care or in combination
          # with combine_with_input_flags option!
          bp_flags = new_flags
        else:
          raise ValueError(f"apply_static.mask.accumulation_mode {accumulation_mode} not in {{'or', 'override'}}")
      else:
        raise ValueError(f"Invalid task {task}")

      # Cumulative occupancy and cost after this strategy. The reduction is a
      # rounding error next to sum_threshold, and it is the only way to tell
      # which of a dozen configured strategies is doing the work.
      strategy_stats.append(
        (
          strategy.get("name", f"strategy {s}"),
          task,
          int(np.count_nonzero(bp_flags)),
          int(bp_flags.size),
          time.perf_counter() - strategy_start,
        )
      )

    # Flag the whole visibility if any correlation is flagged: a correlation
    # affected by RFI is unlikely to leave the others clean, as they mix through
    # the Stokes parameters. This also reintroduces the polarization axis for
    # the 'polarisation' and 'total_power' strategies, which collapsed it above.
    bp_flags = np.any(bp_flags, axis=1, keepdims=True)
    full_shape = tuple(dataset.sizes[d] for d in CP_FLAG_DIM_ORDER)
    bp_flags = np.broadcast_to(bp_flags, full_shape).astype(flag_dtype)

    print(f"Flag {item} stops")
    logger.info("Flag %s stops", item)

    flag_array = xarray.DataArray(bp_flags, dims=CP_FLAG_DIM_ORDER).transpose(*FLAG_DIM_ORDER)

    final_flags = flag_array.values
    final_stats = chunk_window_stats(final_flags, dataset, ubl, antenna_names, STATISTICS_CHAN_BINS, bin_edges)

    report = ChunkReport(
      occupancy=ChunkOccupancy(
        original=original_occupancy,
        final=np.count_nonzero(final_flags, axis=(1, 3)).astype(np.uint32),
        samples_per_cell=dataset.sizes["baseline_id"] * dataset.sizes["polarization"],
      ),
      strategies=strategy_stats,
      flag_seconds=time.perf_counter() - flag_start,
      visibilities=int(np.prod([dataset.sizes[d] for d in FLAG_DIM_ORDER])),
    )

    dataset = dataset.assign(FLAG=flag_array)
    dataset = dataset.drop_vars(set(dataset.data_vars) - {"FLAG"})

    return dataset, original_stats, final_stats, report


@serve.deployment
class DataWriter:
  def __init__(self, path: str, backend: MSv4Backend):
    self._path = path
    self._backend = backend

  def write(
    self, item: WorkItem, flag_result: tuple[xarray.Dataset, WindowStatistics, WindowStatistics, ChunkReport]
  ) -> tuple[WindowStatistics, WindowStatistics, ChunkReport]:
    dataset, original_stats, final_stats, report = flag_result
    write_start = time.perf_counter()

    if self._backend == MSv4Backend.CASA_TABLE:
      # TODO(sjperkins)
      # This is a bit of a hack
      import xarray_ms  # noqa: F401

      dataset.to_msv2(compute=True, region=item.region)
    elif self._backend == MSv4Backend.ZARR:
      dataset.to_zarr(f"{self._path}/{item.path}", compute=True, region=item.region)
    else:
      raise NotImplementedError(f"Backend writeback unsupported for {self._backend}")

    report.write_seconds = time.perf_counter() - write_start
    return original_stats, final_stats, report


# `max_ongoing_requests` has to accommodate the long-running `run` request
# *plus* any dashboard requests, since the ingress routes are served by this
# same replica. Server-sent event streams are held open for the life of the
# page, so leave generous headroom for open browser tabs.
@serve.deployment(num_replicas=1, max_replicas_per_node=1, max_ongoing_requests=64)
@serve.ingress(dashboard_app)
class Tricolour:
  """Distributes chunked work over an autoscaled deployment.

  Also serves the progress dashboard. This deployment already observes every
  work item as it completes and is pinned to a single replica, which makes it
  the only place in the pipeline with a coherent view of the whole run.
  """

  def __init__(
    self,
    datatree: Multiton[xarray.DataTree],
    time_chunks: int,
    freq_chunks: int | None,
    baseline_chunks: int | None,
    field_names: List[str] | None,
    scan_names: List[str] | None,
    data_loader: DeploymentHandle,
    flagger: DeploymentHandle,
    data_writer: DeploymentHandle,
    max_in_flight: int = 64,
    ms: str = "",
    backend: str = "",
    run_metadata: Dict[str, Any] | None = None,
  ):
    self._datatree = datatree.instance
    self._time_chunks = time_chunks
    self._freq_chunks = freq_chunks
    self._baseline_chunks = baseline_chunks
    self._field_names = field_names
    self._scan_names = scan_names
    self._data_loader = data_loader
    self._flagger = flagger
    self._data_writer = data_writer
    self._max_in_flight = max_in_flight
    self._ms = ms
    self._backend = backend
    self._run_metadata = run_metadata or {}
    self._state = RunState(STATISTICS_CHAN_BINS)

  @dashboard_app.get("/", response_class=HTMLResponse)
  def dashboard(self) -> HTMLResponse:
    return HTMLResponse(index_html())

  def snapshot(self) -> Dict[str, Any]:
    """The run's current state.

    Reachable over HTTP through :meth:`api_state` and directly through a
    deployment handle, which is how the CLI drives its terminal progress bar
    without depending on the dashboard being served at all.
    """
    return self._state.snapshot()

  @dashboard_app.get("/api/state")
  def api_state(self) -> Dict[str, Any]:
    return self.snapshot()

  @dashboard_app.get("/api/events")
  def api_events(self) -> StreamingResponse:
    return StreamingResponse(
      snapshot_stream(self._state),
      media_type="text/event-stream",
      # Without this a reverse proxy will happily sit on the stream until it
      # has "enough" to forward, which for a 1Hz trickle is never.
      headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

  @dashboard_app.get("/api/waterfall")
  def api_waterfall(self, path: str, kind: str = "final") -> Dict[str, Any]:
    try:
      return self._state.waterfall(path, kind)
    except KeyError:
      raise HTTPException(status_code=404, detail=f"No occupancy recorded for {path!r}")
    except ValueError as e:
      raise HTTPException(status_code=400, detail=str(e))

  def work_generator(self) -> Generator[WorkItem]:
    """Yields MSv4 DataTree node paths and time/frequency regions to flag.

    Iterates over the visibility nodes of the given MSv4 datatree,
    subdividing the dataset found at that path into
    regions of time and frequency.

    Yields
    ------
      work_item: An item of flagging work.
    """

    for path, node in self._datatree.children.items():
      if node.attrs.get("type") not in VISIBILITY_XDS_TYPES:
        continue

      # Skip partitions whose scan names don't match a supplied list
      if self._scan_names is not None:
        if len(scan_names := np.unique(node.scan_name.data)) > 1:
          raise NotImplementedError(f"Flagging partitions containing multiple scan_names. {path}: {scan_names}")

        if (scan := next(iter(scan_names))) not in self._scan_names:
          print(f"Skipping scan {scan}")
          continue

      # Skip partitions whose field names don't match a supplied list
      if self._field_names is not None:
        if len(field_names := np.unique(node.field_name.data)) > 1:
          raise NotImplementedError(f"Flagging partitions containing multiple field_names. {path}: {field_names}")

        if (field := next(iter(field_names))) not in self._field_names:
          print(f"Skipping field {field}")
          continue

      ntime = node.sizes["time"]
      nbaseline = node.sizes["baseline_id"]
      nfreq = node.sizes["frequency"]
      time_chunks = self._time_chunks
      # Take all frequencies if no frequency chunks are specified
      freq_chunks = self._freq_chunks if isinstance(self._freq_chunks, int) else nfreq
      baseline_chunks = self._baseline_chunks
      time_range = range(0, ntime, time_chunks)
      baseline_range = range(0, nbaseline, baseline_chunks)
      freq_range = range(0, nfreq, freq_chunks)

      for t, bl, f in product(time_range, baseline_range, freq_range):
        region = {
          "time": slice(t, min(t + time_chunks, ntime)),
          "baseline_id": slice(bl, min(bl + baseline_chunks, nbaseline)),
          "frequency": slice(f, min(f + freq_chunks, nfreq)),
        }

        yield WorkItem(path, region)

  def plan(self) -> Tuple[int, int]:
    """Count the work before doing any of it.

    ``work_generator`` derives items from dataset sizes and the configured
    chunking, touching only metadata, so an extra pass over it is cheap and
    buys the dashboard a real denominator rather than a spinner.

    Returns
    -------
      A tuple (work items, visibilities) covered by this run.
    """
    items = 0
    visibilities = 0

    for item in self.work_generator():
      npol = self._datatree[item.path].sizes["polarization"]
      extent = np.prod([s.stop - s.start for s in item.region.values()])
      items += 1
      visibilities += int(extent) * npol

    return items, visibilities

  async def run(self) -> List[str]:
    work_queue = deque()
    # Start with fresh state on each invocation
    self._state = state = RunState(STATISTICS_CHAN_BINS)
    state.begin(self._ms, self._backend, self._run_metadata)
    state.plan(*self.plan())
    stats_bin_edges: Dict[str, npt.NDArray] = {}

    async def maybe_drain_queue(max_queue_size=0):
      if len(work_queue) > max_queue_size:
        to_drain = len(work_queue) - max_queue_size
        items = [work_queue.pop() for _ in range(to_drain)]

        try:
          results = await asyncio.gather(*(response for _, response in items))
        except Exception as e:
          # Record before re-raising: the driver is about to unwind and this
          # is the last chance to get the reason in front of the dashboard.
          state.fail(f"{type(e).__name__}: {e}")
          raise

        for (item, _), (original_stats, final_stats, report) in zip(items, results):
          state.record(
            item.path,
            item.region["time"].start,
            item.region["frequency"].start,
            original_stats,
            final_stats,
            report,
          )

    for work_item in self.work_generator():
      await maybe_drain_queue(self._max_in_flight)
      if (bin_edges := stats_bin_edges.get(work_item.path)) is None:
        # Bin edges spanning the node's full spectral window, so that
        # frequency-chunked histograms accumulate into aligned bins
        node = self._datatree[work_item.path]
        freqs = node.frequency.values
        bin_edges = np.linspace(freqs.min(), freqs.max(), STATISTICS_CHAN_BINS)
        stats_bin_edges[work_item.path] = bin_edges
        state.grid(work_item.path, node.sizes["time"], node.sizes["frequency"], freqs)
      load_response = self._data_loader.load.remote(work_item)
      flag_response = self._flagger.flag.remote(work_item, load_response, bin_edges)
      write_response = self._data_writer.write.remote(work_item, flag_response)
      work_queue.appendleft((work_item, write_response))
      state.dispatch()

    await maybe_drain_queue()
    summary = WindowStatistics.summarise_stats(state.statistics, state.original_statistics)
    state.finish(summary)
    return summary
