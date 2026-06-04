import numpy as np
import pytest
import xarray

from tricolour.core.scaffold_array import ScaffoldArray


@pytest.fixture
def small_dataset():
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


def test_scaffold_array():
  shape = (10, 20, 30)
  data = np.empty(shape)
  array = ScaffoldArray(data)
  assert array.chunks == tuple((d,) for d in shape)
  assert array.ndim == len(shape)
  assert array.dtype == data.dtype
  assert array.shape == shape


@pytest.mark.filterwarnings("ignore::zarr.errors.ZarrUserWarning", reason="Consolidated Metadata Warning")
def test_write_scaffold_dataset(tmp_path, small_dataset):
  pytest.importorskip("zarr")
  ds = small_dataset
  time_chunks = 5
  chan_chunks = 4
  chunks = {"time": time_chunks, "frequency": chan_chunks}
  chunked_ds = ds.chunk(chunks=chunks, chunked_array_type="tricolour:scaffold")

  # Variables should be substituted with a ScaffoldArray
  for var in chunked_ds.data_vars.values():
    assert isinstance(var.data, ScaffoldArray)

  out_store = tmp_path / "out.zarr"

  # Scaffold the store: lays out structure/metadata (and coords),
  # but writes no chunk data for the scaffolded data variables.
  chunked_ds.to_zarr(out_store)
  written = xarray.open_dataset(out_store, engine="zarr")
  assert not ds.identical(written)

  # The zarr store must be chunked exactly as the scaffold dataset asked.
  # The scaffold carries its chunking in ``.chunks`` (a per-dimension tuple
  # of chunk-size tuples); zarr uses a single chunk shape per dimension --
  # the first (leading) chunk size along each axis. After a round-trip that
  # layout surfaces in each variable's ``encoding`` as ``chunks`` (the zarr
  # chunk shape) and ``preferred_chunks`` (a dim -> chunk-size mapping).
  for name in ds.data_vars:
    var = chunked_ds[name]
    expected_chunks = tuple(sizes[0] for sizes in var.chunks)
    expected_preferred = dict(zip(var.dims, expected_chunks))
    assert written[name].encoding["chunks"] == expected_chunks
    assert written[name].encoding["preferred_chunks"] == expected_preferred
    # Sanity check the explicit values requested via ``chunk`` survived.
    assert written[name].encoding["preferred_chunks"]["time"] == time_chunks
    assert written[name].encoding["preferred_chunks"]["frequency"] == chan_chunks

  # Coordinates stay in-memory (not scaffolded) through ``chunk``, so
  # ``to_zarr`` writes their real data during the metadata write. They must
  # come back identical even though no data-variable chunks were written yet.
  assert set(written.coords) == set(ds.coords)
  for name in ds.coords:
    xarray.testing.assert_identical(written[name], ds[name])

  # The scaffolded data variables, by contrast, carry no data yet: their
  # values must differ from the source (they sit at the zarr fill value).
  for name in ds.data_vars:
    assert not np.allclose(np.nan_to_num(written[name].values), np.nan_to_num(ds[name].values))

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

  # Confirm the regions were accurately written back, and that filling the
  # scaffold region-by-region left the on-disk chunking untouched.
  written = xarray.open_dataset(out_store, engine="zarr")
  xarray.testing.assert_identical(written, ds)
  for name in ds.data_vars:
    expected_chunks = tuple(sizes[0] for sizes in chunked_ds[name].chunks)
    assert written[name].encoding["chunks"] == expected_chunks
    assert written[name].encoding["preferred_chunks"] == dict(zip(chunked_ds[name].dims, expected_chunks))
