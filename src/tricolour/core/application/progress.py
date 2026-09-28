"""Run state backing the progress dashboard.

The :class:`Tricolour` driver is a single Serve replica that already observes
every work item as it completes, so it is the natural place to accumulate a
view of the run. This module holds that view -- :class:`RunState` -- along with
the reductions that make it cheap to maintain.

Nothing here touches Ray. The dashboard routes delegate to :class:`RunState`
methods so that both can be exercised without standing up a cluster.
"""

from __future__ import annotations

import base64
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from tricolour.core.kernels.flag_statistics import WindowStatistics

# Occupancy grids are accumulated at a fixed display resolution rather than at
# the resolution of the data. A long observation can run to tens of thousands
# of integrations against thousands of channels; binning on the way in keeps
# the driver's memory flat and keeps the payload small enough to push to a
# browser every few seconds.
MAX_TIME_BINS = 512
MAX_FREQ_BINS = 1024

# Number of entries returned per ranked axis in a snapshot.
TOP_N = 16

# Shortest trailing window that is allowed to define a completion rate. The
# driver dispatches up to `max_in_flight` items before awaiting any of them, so
# on a short run every item completes in one burst; a rate measured across that
# burst is thousands of chunks per second and an ETA derived from it is
# meaningless. Fall back to the run average until the window spans real time.
MIN_RATE_SPAN_S = 1.0

# Sentinel written into a quantised occupancy image for cells no work item has
# covered yet, so that the page can leave them blank instead of drawing them
# as "0% flagged".
NO_DATA = 255
MAX_LEVEL = NO_DATA - 1

OccupancyKind = str  # one of "original", "final", "added"


def bin_stride(n: int, max_bins: int) -> int:
  """Smallest stride binning ``n`` samples into at most ``max_bins`` bins."""
  if n <= 0:
    raise ValueError(f"Cannot bin a non-positive number of samples ({n})")

  return max(1, math.ceil(n / max_bins))


def _reduce_axis(array: npt.NDArray, axis: int, offset: int, stride: int) -> Tuple[npt.NDArray, int, int]:
  """Sum ``array`` along ``axis`` into bins of width ``stride``.

  ``offset`` is the position of the array's first sample within the *global*
  axis, so that a chunk starting part-way through a bin contributes to that
  bin rather than starting a new one.

  Returns the reduced array along with the half-open range of destination
  bins it covers.
  """
  n = array.shape[axis]

  if stride == 1:
    return array, offset, offset + n

  dest = np.arange(offset, offset + n) // stride
  # Sum between the positions at which the destination bin changes. `dest` is
  # non-decreasing and covers a contiguous run of bins, so the reduced array
  # lines up with `dest[0]:dest[-1] + 1` without further indexing.
  starts = np.flatnonzero(np.concatenate(([True], dest[1:] != dest[:-1])))
  return np.add.reduceat(array, starts, axis=axis), int(dest[0]), int(dest[-1]) + 1


def _axis_bin_counts(n: int, offset: int, stride: int) -> npt.NDArray:
  """Number of source samples falling in each destination bin."""
  if stride == 1:
    return np.ones(n, dtype=np.int64)

  dest = np.arange(offset, offset + n) // stride
  return np.bincount(dest - dest[0])


@dataclass(slots=True)
class ChunkOccupancy:
  """A work item's flag counts reduced over baseline and polarisation.

  Attributes
  ----------
  original:
    Flagged counts of shape ``(time, frequency)`` before flagging.
  final:
    Flagged counts of shape ``(time, frequency)`` after flagging.
  samples_per_cell:
    Number of flag values summed into each ``(time, frequency)`` count, i.e.
    the chunk's baseline count times its polarisation count. Chunks tile the
    baseline axis, so this is how a partially-covered grid cell knows what
    fraction of its data has arrived.
  """

  original: npt.NDArray
  final: npt.NDArray
  samples_per_cell: int


@dataclass
class OccupancyGrid:
  """Binned ``(time, frequency)`` flag occupancy for one datatree node."""

  ntime: int
  nfreq: int
  time_stride: int
  freq_stride: int
  freq_hz: npt.NDArray
  original: npt.NDArray
  final: npt.NDArray
  weights: npt.NDArray

  @classmethod
  def empty(cls, ntime: int, nfreq: int, freq_hz: npt.NDArray) -> OccupancyGrid:
    time_stride = bin_stride(ntime, MAX_TIME_BINS)
    freq_stride = bin_stride(nfreq, MAX_FREQ_BINS)
    shape = (math.ceil(ntime / time_stride), math.ceil(nfreq / freq_stride))

    return cls(
      ntime=ntime,
      nfreq=nfreq,
      time_stride=time_stride,
      freq_stride=freq_stride,
      freq_hz=np.asarray(freq_hz, dtype=np.float64),
      original=np.zeros(shape, dtype=np.float64),
      final=np.zeros(shape, dtype=np.float64),
      weights=np.zeros(shape, dtype=np.float64),
    )

  def accumulate(self, occupancy: ChunkOccupancy, time_start: int, freq_start: int) -> None:
    """Bin a chunk's counts down to the grid resolution and add them in."""
    for source, target in ((occupancy.original, self.original), (occupancy.final, self.final)):
      binned, t0, t1 = _reduce_axis(np.asarray(source, dtype=np.float64), 0, time_start, self.time_stride)
      binned, f0, f1 = _reduce_axis(binned, 1, freq_start, self.freq_stride)
      target[t0:t1, f0:f1] += binned

    ntime, nfreq = occupancy.final.shape
    time_counts = _axis_bin_counts(ntime, time_start, self.time_stride)
    freq_counts = _axis_bin_counts(nfreq, freq_start, self.freq_stride)
    t0 = time_start // self.time_stride
    f0 = freq_start // self.freq_stride
    self.weights[t0 : t0 + time_counts.size, f0 : f0 + freq_counts.size] += (
      np.outer(time_counts, freq_counts) * occupancy.samples_per_cell
    )

  def fraction(self, kind: OccupancyKind) -> npt.NDArray:
    """Flagged fraction in ``[0, 1]``, NaN where no data has arrived."""
    if kind == "original":
      numerator = self.original
    elif kind == "final":
      numerator = self.final
    elif kind == "added":
      # Strategies such as `unflag` and `apply_static_mask` in `override` mode
      # can clear flags, so this difference is signed. Only the flags tricolour
      # put there are interesting here, so clamp the negative side away.
      numerator = np.clip(self.final - self.original, 0.0, None)
    else:
      raise ValueError(f"Unknown occupancy kind {kind!r}")

    with np.errstate(invalid="ignore", divide="ignore"):
      return np.where(self.weights > 0, numerator / self.weights, np.nan)

  def display_max(self, kind: OccupancyKind, fraction: npt.NDArray) -> float:
    """Upper end of the colour range for an occupancy image.

    ``original`` and ``final`` are occupancies and are worth comparing against
    each other and between runs, so they are drawn against an absolute
    0-100% range. ``added`` is a difference, and is routinely a couple of
    percent against data that arrived half flagged already -- drawn absolutely
    it is a black rectangle. Scale it to its own peak instead, and report the
    peak so that the colour bar can say what it means.
    """
    if kind != "added" or np.all(np.isnan(fraction)):
      return 1.0

    return peak if (peak := float(np.nanmax(fraction))) > 0.0 else 1.0

  def image(self, kind: OccupancyKind) -> Dict[str, Any]:
    """Quantise an occupancy fraction into a base64 image payload."""
    fraction = self.fraction(kind)
    vmax = self.display_max(kind, fraction)
    levels = np.where(
      np.isnan(fraction),
      NO_DATA,
      np.rint(np.clip(fraction / vmax, 0.0, 1.0) * MAX_LEVEL),
    ).astype(np.uint8)

    freq_hz = self.freq_hz
    return {
      "kind": kind,
      "vmax": vmax,
      "shape": list(levels.shape),
      "time_stride": self.time_stride,
      "freq_stride": self.freq_stride,
      "ntime": self.ntime,
      "nfreq": self.nfreq,
      "freq_start_mhz": float(freq_hz.min()) / 1e6 if freq_hz.size else None,
      "freq_end_mhz": float(freq_hz.max()) / 1e6 if freq_hz.size else None,
      "no_data": NO_DATA,
      "max_level": MAX_LEVEL,
      "data": base64.b64encode(levels.tobytes()).decode("ascii"),
    }


@dataclass
class StrategyStat:
  """Cumulative effect of one configured strategy across every chunk."""

  name: str
  task: str
  flagged: int = 0
  total: int = 0
  seconds: float = 0.0
  chunks: int = 0

  def as_dict(self) -> Dict[str, Any]:
    return {
      "name": self.name,
      "task": self.task,
      "flagged": self.flagged,
      "total": self.total,
      "fraction": self.flagged / self.total if self.total else None,
      "seconds": self.seconds,
      "chunks": self.chunks,
    }


@dataclass(slots=True)
class ChunkReport:
  """Everything a completed work item contributes to the run's view of itself.

  Travels back from the :class:`Flagger` through the :class:`DataWriter` to the
  driver alongside the window statistics.
  """

  occupancy: ChunkOccupancy
  strategies: List[Tuple[str, str, int, int, float]]
  flag_seconds: float
  visibilities: int
  write_seconds: float = 0.0


def _rank(
  final: Dict[str, Tuple[int, int]],
  original: Dict[str, Tuple[int, int]],
  limit: int,
) -> List[Dict[str, Any]]:
  """Rank an axis by how much flagging tricolour added to it."""
  rows = []

  for label, (flagged, size) in final.items():
    if size == 0:
      continue

    was_flagged, was_size = original.get(label, (0, 0))
    before = was_flagged / was_size if was_size else 0.0
    after = flagged / size
    rows.append({"label": label, "original": before, "final": after, "added": after - before})

  rows.sort(key=lambda row: (row["added"], row["final"]), reverse=True)
  return rows[:limit]


class RunState:
  """Mutable view of a flagging run, maintained by the driver.

  Every mutator is called from the driver's event loop and every accessor from
  a dashboard request on that same loop, so no locking is required.
  """

  def __init__(self, nchanbins: int, rate_window: int = 64):
    self.ms: str | None = None
    self.backend: str | None = None
    self.config: Dict[str, Any] = {}
    self.started_at: float | None = None
    self.finished_at: float | None = None
    self.total_items = 0
    self.total_visibilities = 0
    self.dispatched = 0
    self.completed = 0
    self.completed_visibilities = 0
    self.errors: List[str] = []
    self.flag_seconds = 0.0
    self.write_seconds = 0.0
    self.summary: List[str] = []

    self._nchanbins = nchanbins
    self.statistics = WindowStatistics(nchanbins)
    self.original_statistics = WindowStatistics(nchanbins)
    self.occupancy: Dict[str, OccupancyGrid] = {}
    self.strategies: List[StrategyStat] = []
    self._strategy_index: Dict[Tuple[str, str], int] = {}
    self._completions: Deque[float] = deque(maxlen=rate_window)

  # -- mutation ------------------------------------------------------------

  def begin(self, ms: str, backend: str, config: Dict[str, Any]) -> None:
    self.ms = ms
    self.backend = backend
    self.config = config
    self.started_at = time.time()
    self.finished_at = None

  def plan(self, total_items: int, total_visibilities: int) -> None:
    self.total_items = total_items
    self.total_visibilities = total_visibilities

  def grid(self, path: str, ntime: int, nfreq: int, freq_hz: npt.NDArray) -> OccupancyGrid:
    if (grid := self.occupancy.get(path)) is None:
      grid = self.occupancy[path] = OccupancyGrid.empty(ntime, nfreq, freq_hz)

    return grid

  def dispatch(self) -> None:
    self.dispatched += 1

  def record(
    self,
    path: str,
    time_start: int,
    freq_start: int,
    original_stats: WindowStatistics,
    final_stats: WindowStatistics,
    report: ChunkReport,
  ) -> None:
    """Fold a completed work item into the run's accumulators."""
    self.original_statistics.update(original_stats)
    self.statistics.update(final_stats)

    if (grid := self.occupancy.get(path)) is not None:
      grid.accumulate(report.occupancy, time_start, freq_start)

    for name, task, flagged, total, seconds in report.strategies:
      key = (name, task)

      if (index := self._strategy_index.get(key)) is None:
        index = self._strategy_index[key] = len(self.strategies)
        self.strategies.append(StrategyStat(name=name, task=task))

      stat = self.strategies[index]
      stat.flagged += flagged
      stat.total += total
      stat.seconds += seconds
      stat.chunks += 1

    self.flag_seconds += report.flag_seconds
    self.write_seconds += report.write_seconds
    self.completed_visibilities += report.visibilities
    self.completed += 1
    self._completions.append(time.time())

  def fail(self, message: str) -> None:
    self.errors.append(message)

  def finish(self, summary: List[str]) -> None:
    self.summary = summary
    self.finished_at = time.time()

  # -- accessors -----------------------------------------------------------

  @property
  def elapsed(self) -> float:
    if self.started_at is None:
      return 0.0

    return (self.finished_at or time.time()) - self.started_at

  @property
  def rate(self) -> float | None:
    """Work item completion rate, in items per second.

    Measured over a trailing window rather than the whole run where that
    window covers enough time to mean anything: Serve autoscales replicas up
    over the first few seconds, so a run-long average understates throughput
    for most of the run. See :data:`MIN_RATE_SPAN_S` for the other direction.
    """
    if self.completed == 0:
      return None

    if len(self._completions) >= 2:
      span = self._completions[-1] - self._completions[0]

      if span >= MIN_RATE_SPAN_S:
        return (len(self._completions) - 1) / span

    elapsed = self.elapsed
    return self.completed / elapsed if elapsed > 0 else None

  @property
  def eta(self) -> float | None:
    if self.finished_at is not None:
      return 0.0

    remaining = self.total_items - self.completed

    if remaining <= 0 or not (rate := self.rate):
      return None

    return remaining / rate

  def snapshot(self) -> Dict[str, Any]:
    """A light, JSON-serialisable view of the run.

    Deliberately excludes the occupancy grids, which are an order of magnitude
    larger than everything else here and are fetched separately.
    """
    final = self.statistics.totals()
    original = self.original_statistics.totals()

    return {
      "ms": self.ms,
      "backend": self.backend,
      "config": self.config,
      "running": self.started_at is not None and self.finished_at is None,
      "started_at": self.started_at,
      "finished_at": self.finished_at,
      "elapsed": self.elapsed,
      "eta": self.eta,
      "rate": self.rate,
      "progress": {
        "total_items": self.total_items,
        "dispatched": self.dispatched,
        "completed": self.completed,
        "in_flight": max(0, self.dispatched - self.completed),
        "total_visibilities": self.total_visibilities,
        "completed_visibilities": self.completed_visibilities,
        "fraction": self.completed / self.total_items if self.total_items else 0.0,
      },
      "timings": {"flag_seconds": self.flag_seconds, "write_seconds": self.write_seconds},
      "strategies": [stat.as_dict() for stat in self.strategies],
      "ranked": {axis: _rank(final.get(axis, {}), original.get(axis, {}), TOP_N) for axis in final},
      "occupancy_paths": sorted(self.occupancy),
      "errors": self.errors[-32:],
      "summary": self.summary,
    }

  def waterfall(self, path: str, kind: OccupancyKind = "added") -> Dict[str, Any]:
    if (grid := self.occupancy.get(path)) is None:
      raise KeyError(path)

    return {"path": path, **grid.image(kind)}
