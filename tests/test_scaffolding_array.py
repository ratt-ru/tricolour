import numpy as np
import pytest
import xarray

from tricolour.core.scaffolding_array import ScaffoldingArray


@pytest.fixture
def scaffolding_datset():
  na = 7
  nbl = na * (na - 1) // 2

  time = np.linspace(1.0, 10.0, 10)
  frequency = np.linspace(0.856e9, 2 * 0.856e9, 16)
  pol = ["XX", "XY", "YX", "YY"]

  ntime = len(time)
  nchan = len(frequency)
  npol = len(pol)

  rng = np.random.default_rng(seed=42)
  vis = rng.random((ntime, nbl, nchan, npol)) + rng.random((ntime, nbl, nchan, npol)) * 1j
  weight = rng.random((ntime, nbl, nchan, npol))

  return xarray.Dataset(
    {
      "VISIBILITY": (("time", "baseline_id", "frequency", "polarization"), vis),
      "WEIGHT": (("time", "baseline_id", "frequency", "polarization"), weight),
    },
    coords={
      "time": ("time", time),
      "frequency": ("frequency", frequency),
    },
  )


def test_scaffolding_array():
  shape = (10, 20, 30)
  data = np.empty(shape)
  array = ScaffoldingArray(data)
  assert array.chunks == tuple((d,) for d in shape)
  assert array.ndim == len(shape)
  assert array.dtype == data.dtype
  assert array.shape == shape


def test_scaffolding_array_getitem():
  array = ScaffoldingArray(np.empty((10, 20, 30), dtype=np.float32), chunks=(5, 4, 30))
  assert array.chunks == ((5, 5), (4, 4, 4, 4, 4), (30,))

  # Contiguous slice preserves chunk boundaries (overlap of [2, 5) with two size-5 chunks).
  sliced = array[2:5, :, :]
  assert sliced.shape == (3, 20, 30)
  assert sliced.chunks == ((3,), (4, 4, 4, 4, 4), (30,))
  assert sliced.dtype == np.float32

  # Partial leading chunk + full trailing chunk.
  assert array[3:10].chunks[0] == (2, 5)
  # Clean cross-boundary slice keeps the boundary.
  assert array[0:5].chunks[0] == (5,)

  # Integer index drops the dimension.
  dropped = array[0]
  assert dropped.shape == (20, 30)
  assert dropped.ndim == 2

  # Fancy indexing collapses the dimension to a single chunk.
  fancy = array[:, np.array([1, 3, 5]), :]
  assert fancy.shape == (10, 3, 30)
  assert fancy.chunks == ((5, 5), (3,), (30,))


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning", reason="Consolidated Metadata Warning")
def test_write_scaffolding_dataset(tmp_path, scaffolding_datset):
  ds = scaffolding_datset
  time_chunks = 5
  chan_chunks = 4
  chunked_ds = ds.chunk(
    chunks={"time": time_chunks, "frequency": chan_chunks}, chunked_array_type="tricolour:scaffolding"
  )
  out_store = tmp_path / "out.zarr"

  # Scaffold the store: lays out structure/metadata (and coords),
  # but writes no chunk data for the scaffolded data variables.
  chunked_ds.to_zarr(out_store)
  written = xarray.open_dataset(out_store, engine="zarr")
  assert not ds.identical(written)

  # Fill the scaffold by writing regions of the original (unchunked)
  # dataset block-by-block, matching the scaffold chunking.
  ntime = ds.sizes["time"]
  nchan = ds.sizes["frequency"]
  for tstart in range(0, ntime, time_chunks):
    for fstart in range(0, nchan, chan_chunks):
      region = {"time": slice(tstart, tstart + time_chunks), "frequency": slice(fstart, fstart + chan_chunks)}
      # Select region and avoid rewriting coordinates
      block = ds.isel(region).drop_vars(["time", "frequency"])
      block.to_zarr(out_store, region=region)

  # Confirm the regions were accurately written back.
  written = xarray.open_dataset(out_store, engine="zarr")
  xarray.testing.assert_identical(written, ds)
