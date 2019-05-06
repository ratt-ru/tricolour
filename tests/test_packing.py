"""Tests for :mod:`tricolour.packing`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask.array as da
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
import zarr

from tricolour.packing import (unique_baselines,
                               create_vis_windows,
                               create_flag_windows,
                               pack_data,
                               unpack_data)


def flag_data(vis_windows, flag_windows):
    """ Returns flag_windows untouched """

    def _flag_data(vis_windows, flag_windows):
        return flag_windows

    dims = ("time", "chan", "bl", "corr")

    return da.blockwise(_flag_data, dims,
                        vis_windows, dims,
                        flag_windows, dims,
                        dtype=vis_windows.dtype)


@pytest.mark.parametrize("backend", ["numpy", "zarr-disk"])
def test_vis_and_flag_packing(tmpdir, backend):
    na = 7
    ntime = 10
    nchan = 16
    ncorr = 4
    tmpdir = str(tmpdir)

    time = np.linspace(0.1, 0.9, ntime)
    antenna1, antenna2 = (a.astype(np.int32) for a in np.triu_indices(na, 1))
    nbl = antenna1.size

    antenna1 = np.tile(antenna1, ntime)
    antenna2 = np.tile(antenna2, ntime)
    time = np.repeat(time, nbl)

    nrow = time.size

    vis = (np.random.random((nrow, nchan, ncorr)) +
           np.random.random((nrow, nchan, ncorr))*1j)

    flag = np.random.randint(0, 2, (nrow, nchan, ncorr))

    bl_chunks = nbl // 4
    row_chunks = 10

    antenna1 = da.from_array(antenna1, chunks=row_chunks)
    antenna2 = da.from_array(antenna2, chunks=row_chunks)
    time = da.from_array(time, chunks=row_chunks)
    vis = da.from_array(vis, chunks=(row_chunks, nchan, ncorr))
    flag = da.from_array(flag, chunks=(row_chunks, nchan, ncorr))

    ubl = unique_baselines(antenna1, antenna2)
    ubl = ubl.compute().view(np.int32).reshape(-1, 2)
    # Stack the baseline index with the unique baselines
    bl_range = np.arange(ubl.shape[0], dtype=ubl.dtype)[:, None]
    ubl = np.concatenate([bl_range, ubl], axis=1)
    ubl = da.from_array(ubl, chunks=(bl_chunks, 3))

    _, time_inv = da.unique(time, return_inverse=True)

    result = pack_data(time_inv, ubl, antenna1, antenna2,
                       vis, flag, ntime,
                       backend=backend, path=tmpdir,
                       return_objs=True)

    vis_windows, flag_windows, vis_win_obj, flag_win_obj = result

    flag_windows = flag_data(vis_windows, flag_windows)

    unpacked_flags = unpack_data(antenna1, antenna2, time_inv,
                                 ubl, flag_windows)

    unpacked_vis = unpack_data(antenna1, antenna2, time_inv,
                               ubl, vis_windows)

    result = da.compute(vis, flag, vis_win_obj, flag_win_obj,
                        unpacked_vis, unpacked_flags)
    (vis, flag, vis_win_obj, flag_win_obj,
     unpacked_vis, unpacked_flags) = result

    # Check that we've created the correct type of backend object
    if backend == "numpy":
        assert isinstance(vis_win_obj, np.ndarray)
        assert isinstance(flag_win_obj, np.ndarray)
    elif backend == "zarr-disk":
        assert isinstance(vis_win_obj, zarr.Array)
        assert isinstance(flag_win_obj, zarr.Array)
    else:
        raise ValueError("Unhandled backend '%s'" % backend)

    assert_array_almost_equal(flag, unpacked_flags)
    assert_array_almost_equal(vis, unpacked_vis)
