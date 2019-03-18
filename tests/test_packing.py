"""Tests for :mod:`tricolour.packing`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask.array as da
import numpy as np
from numpy.testing import assert_array_almost_equal

from tricolour.packing import (unique_baselines,
                               create_vis_windows,
                               create_flag_windows,
                               pack_data,
                               unpack_data)


def flag_data(vis_windows, flag_windows):
    """ Returns flag_windows untouched """

    def _flag_data(vis_windows, flag_windows):
        return flag_windows

    return da.blockwise(_flag_data, ("bl", "time", "chan", "corr"),
                        vis_windows, ("bl", "time", "chan", "corr"),
                        flag_windows, ("bl", "time", "chan", "corr"),
                        dtype=vis_windows.dtype)


def test_vis_and_flag_packing(tmpdir):
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

    antenna1 = da.from_array(antenna1, chunks=10)
    antenna2 = da.from_array(antenna2, chunks=10)
    time = da.from_array(time, chunks=10)
    vis = da.from_array(vis, chunks=(10, nchan, ncorr))
    flag = da.from_array(flag, chunks=(10, nchan, ncorr))

    ubl = unique_baselines(antenna1, antenna2)
    ubl = ubl.compute().view(np.int32).reshape(-1, 2)

    vis_windows = create_vis_windows(ubl, ntime, nchan, ncorr,
                                     np.complex64, path=tmpdir)

    flag_windows = create_flag_windows(ubl, ntime, nchan, ncorr,
                                       np.bool, path=tmpdir)

    _, time_inv = da.unique(time, return_inverse=True)

    vis_windows, flag_windows = pack_data(time_inv, ubl, antenna1, antenna2,
                                          vis, flag,
                                          vis_windows, flag_windows)

    flag_windows = flag_data(vis_windows, flag_windows)

    unpacked_flags = unpack_data(antenna1, antenna2, time_inv,
                                 ubl, flag_windows)

    unpacked_vis = unpack_data(antenna1, antenna2, time_inv,
                               ubl, vis_windows)

    result = da.compute(vis, flag, unpacked_vis, unpacked_flags)
    vis, flag, unpacked_vis, unpacked_flags = result

    assert_array_almost_equal(flag, unpacked_flags)
    assert_array_almost_equal(vis, unpacked_vis)
