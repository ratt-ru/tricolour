from __future__ import annotations

import asyncio
import concurrent.futures as cf
import multiprocessing as mp
from collections import deque
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Generator, Iterable, List, Literal, get_args

import numpy as np
import numpy.typing as npt
import xarray
from msv4_utils import MSv4Backend
from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES
from rarg_python_patterns.multiton import Multiton
from ray import serve
from ray.serve.handle import DeploymentHandle

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
  ) -> tuple[xarray.Dataset, WindowStatistics, WindowStatistics]:
    concatenated_antennas = np.concat((dataset.baseline_antenna1_name, dataset.baseline_antenna2_name))
    antenna_names, ant_inv = np.unique(concatenated_antennas, return_inverse=True)
    ubl = np.column_stack(
      [dataset.baseline_id.values, ant_inv[: dataset.sizes["baseline_id"]], ant_inv[dataset.sizes["baseline_id"] :]]
    )

    if self._data_variable not in dataset:
      raise ValueError(f"Visibility variable {self._data_variable} not in dataset")

    # Statistics of the flags as they arrived, for comparison against the final flags
    original_stats = chunk_window_stats(
      dataset["FLAG"].transpose(*FLAG_DIM_ORDER).values, dataset, ubl, antenna_names, STATISTICS_CHAN_BINS, bin_edges
    )

    bp_vis = dataset[self._data_variable].transpose(*CP_FLAG_DIM_ORDER).values
    bp_uvw = dataset["UVW"].transpose("baseline_id", "time", ...).values

    if not self._ignore_flags:
      bp_flags = dataset["FLAG"].transpose(*CP_FLAG_DIM_ORDER).values
    else:
      bp_flags = np.ones_like(bp_vis, dtype=np.uint8)

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
    for strategy in self._flagging_config.instance.get("strategies", []):
      if (task := strategy.get("task", MISSING_SENTINEL)) is MISSING_SENTINEL:
        raise ValueError(f"Strategy '{strategy}' has no task")

      if task == "sum_threshold":
        if self._pool is None:
          new_flags = sum_threshold_flagger(bp_vis, bp_flags, **strategy["kwargs"])
          np.logical_or(bp_flags, new_flags, out=bp_flags)
        else:
          futures = {}
          nbl = bp_flags.shape[0]

          # Submit sumthresholding work to the threadpool in baseline chunks
          for bl in range(0, nbl, self._baseline_chunks):
            bl_slice = slice(0, min(nbl, bl + self._baseline_chunks))
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
        np.logical_or(bp_flags, flag_autos(bp_vis, ubl), out=bp_flags)
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

    # Reintroduce the missing polarizations if necessary
    if self._flagging_strategy in ("polarisation", "total_power"):
      full_shape = tuple(dataset.sizes[d] for d in CP_FLAG_DIM_ORDER)
      bp_flags = np.broadcast_to(bp_flags, full_shape)

    flag_array = xarray.DataArray(bp_flags, dims=CP_FLAG_DIM_ORDER).transpose(*FLAG_DIM_ORDER)

    final_stats = chunk_window_stats(flag_array.values, dataset, ubl, antenna_names, STATISTICS_CHAN_BINS, bin_edges)

    dataset = dataset.assign(FLAG=flag_array)
    dataset = dataset.drop_vars(set(dataset.data_vars) - {"FLAG"})

    return dataset, original_stats, final_stats


@serve.deployment
class DataWriter:
  def __init__(self, path: str, backend: MSv4Backend):
    self._path = path
    self._backend = backend

  def write(
    self, item: WorkItem, flag_result: tuple[xarray.Dataset, WindowStatistics, WindowStatistics]
  ) -> tuple[WindowStatistics, WindowStatistics]:
    dataset, original_stats, final_stats = flag_result

    if self._backend == MSv4Backend.CASA_TABLE:
      # TODO(sjperkins)
      # This is a bit of a hack
      import xarray_ms  # noqa: F401

      dataset.to_msv2(compute=True, region=item.region)
    elif self._backend == MSv4Backend.ZARR:
      dataset.to_zarr(f"{self._path}/{item.path}", compute=True, region=item.region)
    else:
      raise NotImplementedError(f"Backend writeback unsupported for {self._backend}")

    return original_stats, final_stats


@serve.deployment(num_replicas=1, max_replicas_per_node=1)
class Tricolour:
  """Distributes chunked work over an autoscaled deployment."""

  def __init__(
    self,
    datatree: Multiton[xarray.DataTree],
    time_chunks: int,
    freq_chunks: int | None,
    field_names: List[str] | None,
    scan_names: List[str] | None,
    data_loader: DeploymentHandle,
    flagger: DeploymentHandle,
    data_writer: DeploymentHandle,
  ):
    self._datatree = datatree.instance
    self._time_chunks = time_chunks
    self._freq_chunks = freq_chunks
    self._field_names = field_names
    self._scan_names = scan_names
    self._data_loader = data_loader
    self._flagger = flagger
    self._data_writer = data_writer
    self._statistics = WindowStatistics(STATISTICS_CHAN_BINS)
    self._original_statistics = WindowStatistics(STATISTICS_CHAN_BINS)

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
      nfreq = node.sizes["frequency"]
      time_chunks = self._time_chunks
      # Take all frequencies if no frequency chunks are specified
      freq_chunks = self._freq_chunks if isinstance(self._freq_chunks, int) else nfreq
      time_range = range(0, ntime, time_chunks)
      freq_range = range(0, nfreq, freq_chunks)

      for t, f in product(time_range, freq_range):
        region = {
          "time": slice(t, min(t + time_chunks, ntime)),
          "frequency": slice(f, min(f + freq_chunks, nfreq)),
        }

        yield WorkItem(path, region)

  async def __call__(self) -> List[str]:
    work_queue = deque()
    # Start with fresh statistics on each invocation
    self._statistics = WindowStatistics(STATISTICS_CHAN_BINS)
    self._original_statistics = WindowStatistics(STATISTICS_CHAN_BINS)
    stats_bin_edges: Dict[str, npt.NDArray] = {}

    async def maybe_drain_queue(max_queue_size=0):
      if len(work_queue) > max_queue_size:
        to_drain = len(work_queue) - max_queue_size
        waiters = [work_queue.pop() for _ in range(to_drain)]
        for original_stats, final_stats in await asyncio.gather(*waiters):
          self._original_statistics.update(original_stats)
          self._statistics.update(final_stats)

    for work_item in self.work_generator():
      await maybe_drain_queue(20)
      if (bin_edges := stats_bin_edges.get(work_item.path)) is None:
        # Bin edges spanning the node's full spectral window, so that
        # frequency-chunked histograms accumulate into aligned bins
        freqs = self._datatree[work_item.path].frequency.values
        bin_edges = np.linspace(freqs.min(), freqs.max(), STATISTICS_CHAN_BINS)
        stats_bin_edges[work_item.path] = bin_edges
      load_response = self._data_loader.load.remote(work_item)
      flag_response = self._flagger.flag.remote(work_item, load_response, bin_edges)
      write_response = self._data_writer.write.remote(work_item, flag_response)
      work_queue.appendleft(write_response)

    await maybe_drain_queue()
    return WindowStatistics.summarise_stats(self._statistics, self._original_statistics)
