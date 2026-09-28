"""Tests for the run state backing the progress dashboard."""

import base64
import time

import numpy as np
import pytest

from tricolour.core.application.progress import (
  MAX_FREQ_BINS,
  MAX_TIME_BINS,
  NO_DATA,
  ChunkOccupancy,
  ChunkReport,
  OccupancyGrid,
  RunState,
  bin_stride,
)
from tricolour.core.kernels.flag_statistics import window_stats

pytestmark = pytest.mark.unit

NCHANBINS = 10


def chunk_occupancy(flags):
  """Reduce ``(time, baseline, freq, pol)`` flags the way the Flagger does."""
  return ChunkOccupancy(
    original=np.zeros(flags.shape[::2], dtype=np.uint32),
    final=np.count_nonzero(flags, axis=(1, 3)).astype(np.uint32),
    samples_per_cell=flags.shape[1] * flags.shape[3],
  )


@pytest.mark.parametrize(
  "n, max_bins, expected",
  [(10, 100, 1), (100, 100, 1), (101, 100, 2), (1000, 100, 10), (1001, 100, 11)],
)
def test_bin_stride(n, max_bins, expected):
  assert bin_stride(n, max_bins) == expected


def test_bin_stride_rejects_empty():
  with pytest.raises(ValueError, match="non-positive"):
    bin_stride(0, 100)


@pytest.mark.parametrize("ntime, nfreq", [(1, 1), (100_000, 32_768), (513, 1025), (7, 4096)])
def test_grid_is_bounded_by_the_display_resolution(ntime, nfreq):
  """A grid never exceeds the display resolution, however large the data."""
  grid = OccupancyGrid.empty(ntime, nfreq, np.linspace(856e6, 1712e6, nfreq))
  rows, cols = grid.original.shape

  assert rows <= MAX_TIME_BINS and cols <= MAX_FREQ_BINS
  assert grid.final.shape == grid.weights.shape == grid.original.shape
  # Every sample lands in a bin
  assert rows * grid.time_stride >= ntime
  assert cols * grid.freq_stride >= nfreq


@pytest.mark.parametrize("time_chunk, freq_chunk", [(7, 11), (16, 64), (100, 1), (1, 100)])
def test_tiled_chunks_reproduce_the_full_occupancy(time_chunk, freq_chunk):
  """Chunks tiling the plane must accumulate to the same answer as one pass.

  The driver never sees a whole node at once, so the grid is only correct if
  arbitrary chunkings of time and frequency land in the right bins.
  """
  rng = np.random.default_rng(42)
  ntime, nbl, nfreq, npol = 100, 6, 100, 4
  flags = rng.random((ntime, nbl, nfreq, npol)) < 0.3
  freqs = np.linspace(856e6, 1712e6, nfreq)

  whole = OccupancyGrid.empty(ntime, nfreq, freqs)
  whole.accumulate(chunk_occupancy(flags), 0, 0)

  tiled = OccupancyGrid.empty(ntime, nfreq, freqs)
  for t in range(0, ntime, time_chunk):
    for f in range(0, nfreq, freq_chunk):
      window = flags[t : t + time_chunk, :, f : f + freq_chunk, :]
      tiled.accumulate(chunk_occupancy(window), t, f)

  np.testing.assert_allclose(tiled.final, whole.final)
  np.testing.assert_allclose(tiled.weights, whole.weights)
  # And the whole thing agrees with the flagged fraction of the raw array
  assert tiled.final.sum() == np.count_nonzero(flags)


def test_baseline_chunks_accumulate_into_the_same_cells():
  """Baseline chunks overlap in (time, frequency) and must sum, not overwrite."""
  ntime, nfreq = 8, 8
  freqs = np.linspace(856e6, 1712e6, nfreq)
  grid = OccupancyGrid.empty(ntime, nfreq, freqs)

  for _ in range(4):
    flags = np.ones((ntime, 3, nfreq, 2), dtype=bool)
    grid.accumulate(chunk_occupancy(flags), 0, 0)

  # Four chunks of 3 baselines x 2 polarisations, all flagged
  np.testing.assert_allclose(grid.weights, 24)
  np.testing.assert_allclose(grid.fraction("final"), 1.0)


def test_fraction_marks_uncovered_cells_as_nan():
  grid = OccupancyGrid.empty(4, 4, np.linspace(856e6, 1712e6, 4))
  grid.accumulate(chunk_occupancy(np.ones((2, 1, 4, 1), dtype=bool)), 0, 0)

  fraction = grid.fraction("final")
  assert np.all(fraction[:2] == 1.0)
  assert np.all(np.isnan(fraction[2:]))


def test_added_occupancy_ignores_flags_that_were_cleared():
  """`unflag` and mask overrides can clear flags; `added` only shows gains."""
  grid = OccupancyGrid.empty(2, 2, np.linspace(856e6, 1712e6, 2))
  grid.original = np.array([[4.0, 0.0], [0.0, 0.0]])
  grid.final = np.array([[0.0, 4.0], [0.0, 0.0]])
  grid.weights = np.full((2, 2), 4.0)

  np.testing.assert_allclose(grid.fraction("added"), [[0.0, 1.0], [0.0, 0.0]])


def test_image_quantises_and_flags_missing_data():
  grid = OccupancyGrid.empty(2, 2, np.linspace(856e6, 1712e6, 2))
  grid.accumulate(chunk_occupancy(np.ones((1, 1, 2, 1), dtype=bool)), 0, 0)

  image = grid.image("final")
  levels = np.frombuffer(base64.b64decode(image["data"]), dtype=np.uint8).reshape(image["shape"])

  assert image["shape"] == [2, 2]
  assert np.all(levels[0] == 254)  # fully flagged
  assert np.all(levels[1] == NO_DATA)  # nothing written there yet
  assert image["freq_start_mhz"] == pytest.approx(856.0)
  assert image["freq_end_mhz"] == pytest.approx(1712.0)


def test_absolute_kinds_are_drawn_against_a_full_scale():
  grid = OccupancyGrid.empty(1, 2, np.linspace(856e6, 1712e6, 2))
  grid.original = np.array([[1.0, 0.0]])
  grid.final = np.array([[2.0, 0.0]])
  grid.weights = np.full((1, 2), 4.0)

  for kind in ("original", "final"):
    assert grid.image(kind)["vmax"] == 1.0


def test_added_is_scaled_to_its_own_peak():
  """A few percent of added flags must not render as a black rectangle."""
  grid = OccupancyGrid.empty(1, 2, np.linspace(856e6, 1712e6, 2))
  grid.original = np.array([[0.0, 0.0]])
  grid.final = np.array([[0.2, 0.1]])
  grid.weights = np.full((1, 2), 4.0)

  image = grid.image("added")
  levels = np.frombuffer(base64.b64decode(image["data"]), dtype=np.uint8).reshape(image["shape"])

  assert image["vmax"] == pytest.approx(0.05)
  # The peak saturates the ramp and half the peak lands mid-ramp
  assert levels[0, 0] == 254
  assert levels[0, 1] == pytest.approx(127, abs=1)


def test_added_scale_falls_back_when_nothing_was_added():
  grid = OccupancyGrid.empty(1, 2, np.linspace(856e6, 1712e6, 2))
  grid.original = np.array([[4.0, 4.0]])
  grid.final = np.array([[4.0, 4.0]])
  grid.weights = np.full((1, 2), 4.0)

  image = grid.image("added")
  assert image["vmax"] == 1.0
  levels = np.frombuffer(base64.b64decode(image["data"]), dtype=np.uint8)
  assert np.all(levels == 0)


def test_added_scale_falls_back_on_an_empty_grid():
  grid = OccupancyGrid.empty(2, 2, np.linspace(856e6, 1712e6, 2))
  assert grid.image("added")["vmax"] == 1.0


def test_image_rejects_an_unknown_kind():
  grid = OccupancyGrid.empty(2, 2, np.linspace(856e6, 1712e6, 2))
  with pytest.raises(ValueError, match="Unknown occupancy kind"):
    grid.image("sideways")


def make_stats(flagged_fraction, antenna_names=("m000", "m001")):
  """Build a WindowStatistics over a small array with a known flagged fraction."""
  ntime, nfreq, npol = 4, 8, 2
  names = np.asarray(antenna_names)
  nant = names.size
  pairs = [(a1, a2) for a1 in range(nant) for a2 in range(a1, nant)]
  ubls = np.array([[i, a1, a2] for i, (a1, a2) in enumerate(pairs)])

  flags = np.zeros((ntime, len(pairs), nfreq, npol), dtype=bool)
  flat = flags.reshape(-1)
  flat[: int(round(flagged_fraction * flat.size))] = True

  return window_stats(flags, ubls, np.linspace(856e6, 1712e6, nfreq), names, "1", "PKS1934-63", "spw0", NCHANBINS)


def make_report(nstrategies=2, seconds=1.0):
  return ChunkReport(
    occupancy=chunk_occupancy(np.ones((2, 1, 2, 1), dtype=bool)),
    strategies=[(f"strategy_{i}", "sum_threshold", 10 * (i + 1), 100, seconds) for i in range(nstrategies)],
    flag_seconds=seconds,
    visibilities=16,
    write_seconds=seconds / 2,
  )


def test_snapshot_is_serialisable_before_any_work_lands():
  import json

  state = RunState(NCHANBINS)
  state.begin("/data/test.ms", "CASA_TABLE", {"nworkers": 4})
  state.plan(10, 1000)

  snapshot = state.snapshot()
  json.dumps(snapshot)  # must not raise

  assert snapshot["ms"] == "/data/test.ms"
  assert snapshot["running"] is True
  assert snapshot["progress"] == {
    "total_items": 10,
    "dispatched": 0,
    "completed": 0,
    "in_flight": 0,
    "total_visibilities": 1000,
    "completed_visibilities": 0,
    "fraction": 0.0,
  }
  assert snapshot["eta"] is None


def test_recording_advances_progress_and_accumulates_strategies():
  state = RunState(NCHANBINS)
  state.begin("/data/test.ms", "ZARR", {})
  state.plan(4, 64)
  state.grid("node", 2, 2, np.linspace(856e6, 1712e6, 2))

  for _ in range(2):
    state.dispatch()
    state.record("node", 0, 0, make_stats(0.25), make_stats(0.75), make_report())

  snapshot = state.snapshot()
  assert snapshot["progress"]["completed"] == 2
  assert snapshot["progress"]["fraction"] == 0.5
  assert snapshot["progress"]["completed_visibilities"] == 32
  assert snapshot["timings"] == {"flag_seconds": 2.0, "write_seconds": 1.0}

  # Two strategies, each seen by both chunks
  assert [s["name"] for s in snapshot["strategies"]] == ["strategy_0", "strategy_1"]
  assert [s["chunks"] for s in snapshot["strategies"]] == [2, 2]
  assert snapshot["strategies"][0]["fraction"] == pytest.approx(0.1)
  assert snapshot["strategies"][1]["fraction"] == pytest.approx(0.2)


def test_in_flight_tracks_dispatched_less_completed():
  state = RunState(NCHANBINS)
  state.begin("/data/test.ms", "ZARR", {})
  state.plan(8, 128)

  for _ in range(5):
    state.dispatch()
  assert state.snapshot()["progress"]["in_flight"] == 5

  state.record("node", 0, 0, make_stats(0.0), make_stats(0.5), make_report())
  assert state.snapshot()["progress"]["in_flight"] == 4


def test_ranked_axes_are_sorted_by_flags_added():
  state = RunState(NCHANBINS)
  state.begin("/data/test.ms", "ZARR", {})
  state.plan(1, 16)
  state.dispatch()
  state.record("node", 0, 0, make_stats(0.25), make_stats(0.75), make_report())

  ranked = state.snapshot()["ranked"]
  assert set(ranked) == {"antenna", "baseline", "scan", "field"}

  field = ranked["field"][0]
  assert field["label"] == "PKS1934-63"
  assert field["original"] == pytest.approx(0.25)
  assert field["final"] == pytest.approx(0.75)
  assert field["added"] == pytest.approx(0.5)

  added = [row["added"] for row in ranked["baseline"]]
  assert added == sorted(added, reverse=True)


def test_finishing_stops_the_clock_and_keeps_the_summary():
  state = RunState(NCHANBINS)
  state.begin("/data/test.ms", "ZARR", {})
  state.plan(1, 16)
  state.dispatch()
  state.record("node", 0, 0, make_stats(0.0), make_stats(1.0), make_report())
  state.finish(["FLAG SUMMARY", "all done"])

  snapshot = state.snapshot()
  assert snapshot["running"] is False
  assert snapshot["eta"] == 0.0
  assert snapshot["summary"] == ["FLAG SUMMARY", "all done"]
  assert snapshot["elapsed"] == pytest.approx(state.elapsed)


def test_failures_are_recorded_for_the_dashboard():
  state = RunState(NCHANBINS)
  state.begin("/data/test.ms", "ZARR", {})
  state.fail("RuntimeError: writer wedged")

  assert state.snapshot()["errors"] == ["RuntimeError: writer wedged"]


def test_waterfall_reports_an_unknown_path():
  state = RunState(NCHANBINS)
  with pytest.raises(KeyError):
    state.waterfall("nope")


def test_rate_ignores_a_burst_of_completions():
  """A queue drained all at once must not produce an absurd rate or ETA.

  The driver dispatches up to `max_in_flight` items before awaiting any of
  them, so a short run completes every item within milliseconds of the last.
  Measured naively across that burst the rate runs to thousands of chunks per
  second and the ETA collapses to nothing.
  """
  state = RunState(NCHANBINS)
  state.begin("/data/test.ms", "ZARR", {})
  state.plan(24, 384)
  state.started_at = time.time() - 60.0

  for _ in range(12):
    state.dispatch()
    state.record("node", 0, 0, make_stats(0.0), make_stats(0.5), make_report())

  # 12 items over the 60s the run has actually been going, not 12 items over
  # the handful of milliseconds the burst occupied.
  assert state.rate == pytest.approx(12 / 60.0, rel=0.05)
  assert state.eta == pytest.approx(60.0, rel=0.05)


def test_rate_prefers_the_trailing_window_once_it_spans_real_time():
  state = RunState(NCHANBINS)
  state.begin("/data/test.ms", "ZARR", {})
  state.plan(100, 1600)
  # A long-idle start followed by steady completions: the trailing window
  # should win, so that a slow ramp-up does not depress the estimate forever.
  state.started_at = time.time() - 1000.0
  state.completed = 4
  state._completions.extend(time.time() - offset for offset in (4.0, 3.0, 2.0, 1.0))

  assert state.rate == pytest.approx(1.0, rel=0.05)
