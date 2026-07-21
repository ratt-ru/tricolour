import numpy as np
import pytest
import xarray

from tricolour.core.application.implementation import STATISTICS_CHAN_BINS, chunk_window_stats
from tricolour.core.kernels.flag_statistics import WindowStatistics, window_stats

NTIME = 4
NBL = 3
NCHAN = 8
NPOL = 2

# Baselines A&B, A&C, B&C
ANTENNA_NAMES = np.array(["A", "B", "C"])
UBL = np.array([[0, 0, 1], [1, 0, 2], [2, 1, 2]])
FREQS = 1e9 + np.arange(NCHAN) * 1e6
BIN_EDGES = np.linspace(FREQS.min(), FREQS.max(), STATISTICS_CHAN_BINS)


def make_dataset(flags: np.ndarray) -> xarray.Dataset:
  return xarray.Dataset(
    {"FLAG": (("time", "baseline_id", "frequency", "polarization"), flags)},
    coords={
      "time": np.arange(flags.shape[0]),
      "baseline_id": np.arange(flags.shape[1]),
      "frequency": ("frequency", FREQS[: flags.shape[2]], {"spectral_window_name": "spw0"}),
      "polarization": ["XX", "YY"],
      "scan_name": ("time", np.array(["s1", "s1", "s2", "s2"])[: flags.shape[0]]),
      "field_name": ("time", np.array(["f1", "f1", "f1", "f2"])[: flags.shape[0]]),
    },
  )


@pytest.fixture
def flags() -> np.ndarray:
  """Everything flagged at the first timestep, nothing else"""
  flags = np.zeros((NTIME, NBL, NCHAN, NPOL), dtype=np.uint8)
  flags[0] = 1
  return flags


def test_chunk_window_stats_counts(flags):
  dataset = make_dataset(flags)
  stats = chunk_window_stats(flags, dataset, UBL, ANTENNA_NAMES, STATISTICS_CHAN_BINS, BIN_EDGES)

  # Antenna A participates in baselines 0 and 1, B in 0 and 2, C in 1 and 2
  for ant in ANTENNA_NAMES:
    assert stats._counts_per_ant[ant] == 2 * NCHAN * NPOL
    assert stats._size_per_ant[ant] == NTIME * 2 * NCHAN * NPOL

  for bl in ["A&B", "A&C", "B&C"]:
    assert stats._counts_per_bl[bl] == NCHAN * NPOL
    assert stats._size_per_bl[bl] == NTIME * NCHAN * NPOL

  # Fields f1 (times 0-2) and f2 (time 3)
  assert stats._counts_per_field["f1"] == NBL * NCHAN * NPOL
  assert stats._size_per_field["f1"] == 3 * NBL * NCHAN * NPOL
  assert stats._counts_per_field["f2"] == 0
  assert stats._size_per_field["f2"] == NBL * NCHAN * NPOL

  # Scans s1 (times 0-1) and s2 (times 2-3)
  assert stats._counts_per_scan["s1"] == NBL * NCHAN * NPOL
  assert stats._size_per_scan["s1"] == 2 * NBL * NCHAN * NPOL
  assert stats._counts_per_scan["s2"] == 0
  assert stats._size_per_scan["s2"] == 2 * NBL * NCHAN * NPOL

  # The topmost channel is excluded by the exclusive upper bin edge
  assert stats._counts_per_ddid["spw0"].sum() == (NCHAN - 1) * NBL * NPOL
  assert stats._size_per_ddid["spw0"] == flags.size
  np.testing.assert_array_equal(stats._bins_per_ddid["spw0"], BIN_EDGES)


def assert_stats_equal(a: WindowStatistics, b: WindowStatistics):
  for attr in (
    "_counts_per_ant",
    "_counts_per_field",
    "_counts_per_scan",
    "_counts_per_bl",
    "_size_per_ant",
    "_size_per_field",
    "_size_per_scan",
    "_size_per_bl",
    "_size_per_ddid",
  ):
    assert dict(getattr(a, attr)) == dict(getattr(b, attr)), attr

  assert set(a._counts_per_ddid) == set(b._counts_per_ddid)

  for ddid, counts in a._counts_per_ddid.items():
    np.testing.assert_array_equal(counts, b._counts_per_ddid[ddid])


def test_chunked_stats_merge_to_whole(flags):
  """Stats over time/frequency sub-chunks merge into whole-window stats"""
  rng = np.random.default_rng(42)
  flags = rng.integers(0, 2, size=flags.shape).astype(np.uint8)
  dataset = make_dataset(flags)
  whole = chunk_window_stats(flags, dataset, UBL, ANTENNA_NAMES, STATISTICS_CHAN_BINS, BIN_EDGES)

  merged = WindowStatistics(STATISTICS_CHAN_BINS)

  for time_slice in (slice(0, 2), slice(2, 4)):
    for freq_slice in (slice(0, 4), slice(4, 8)):
      sub_dataset = dataset.isel(time=time_slice, frequency=freq_slice)
      sub_flags = flags[time_slice, :, freq_slice, :]
      # Shared bin edges align each sub-chunk's histogram
      merged.update(chunk_window_stats(sub_flags, sub_dataset, UBL, ANTENNA_NAMES, STATISTICS_CHAN_BINS, BIN_EDGES))

  assert_stats_equal(merged, whole)


def test_window_stats_rejects_mismatched_bin_edges(flags):
  with pytest.raises(ValueError, match="bin_edges"):
    window_stats(flags, UBL, FREQS, ANTENNA_NAMES, "s1", "f1", "spw0", STATISTICS_CHAN_BINS, bin_edges=BIN_EDGES[:-1])
