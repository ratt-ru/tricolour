from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join as pjoin
import random
from tempfile import mkdtemp

import dask
import dask.array as da
from dask.highlevelgraph import HighLevelGraph
import numpy as np
import zarr


def unique_baselines(ant1, ant2):
    """
    Returns unique baseline pairs across all dask chunks as 64 bit ints

    The resulting computed numpy array should be recast and shaped
    as follows:

    .. code-block:: python

        ubl_dask = unique_baselines(ant1, ant2)

        ubl = dask.compute(ubl_dask)[0].view(np.int32).reshape(-1, 2)
    """
    if not (ant1.dtype == np.int32 and ant2.dtype == np.int32):
        raise TypeError("antenna1 '%s' and antenna2 '%s' dtypes "
                        "must both be np.int32" % (ant1.dtype, ant2.dtype))

    # Stack, create a 64 bit baseline values
    bl = da.stack([ant1, ant2], axis=1)
    bl = bl.rechunk(-1, 2).view(np.int64)
    return da.unique(bl)


def _create_vis_windows(ubl, ntime, nchan, ncorr, dtype, path):
    vis = zarr.creation.create(shape=(ubl.shape[0], ntime, nchan, ncorr),
                               chunks=(1, ntime, nchan, ncorr),
                               dtype=dtype,
                               synchronizer=zarr.ThreadSynchronizer(),
                               overwrite=True,
                               # https://github.com/zarr-developers/zarr/issues/244
                               # Set values post-creation
                               fill_value=None,
                               read_only=False,
                               store=pjoin(path, "vis"))

    vis[:] = 0

    return vis


def _create_flag_windows(ubl, ntime, nchan, ncorr, dtype, path):
    return zarr.creation.create(shape=(ubl.shape[0], ntime, nchan, ncorr),
                                chunks=(1, ntime, nchan, ncorr),
                                dtype=dtype,
                                synchronizer=zarr.ThreadSynchronizer(),
                                overwrite=True,
                                # Flagged by default
                                fill_value=0,
                                read_only=False,
                                store=pjoin(path, "flag"))


def create_vis_windows(ubl, ntime, nchan, ncorr, dtype, path=None):
    if path is None:
        path = mkdtemp(prefix='tricolour-vis-windows-')

    token = dask.base.tokenize(ubl, ntime, nchan, ncorr, dtype, path)

    name = "create-vis-windows-" + token
    layers = {(name, 0, 0, 0, 0): (_create_vis_windows,
                                   ubl, ntime, nchan, ncorr,
                                   dtype, path)}

    graph = HighLevelGraph.from_collections(name, layers, ())
    chunks = ((ubl.shape[0],), (ntime,), (nchan,), (ncorr,))
    windows = da.Array(graph, name, chunks, dtype=dtype)

    return windows


def create_flag_windows(ubl, ntime, nchan, ncorr, dtype, path=None):
    if path is None:
        path = mkdtemp(prefix='tricolour-flag-windows-')

    token = dask.base.tokenize(ubl, ntime, nchan, ncorr, dtype, path)

    name = "create-flag-windows-" + token
    layers = {(name, 0, 0, 0, 0): (_create_flag_windows,
                                   ubl, ntime, nchan, ncorr,
                                   dtype, path)}

    graph = HighLevelGraph.from_collections(name, layers, ())
    chunks = ((ubl.shape[0],), (ntime,), (nchan,), (ncorr,))
    windows = da.Array(graph, name, chunks, dtype=dtype)

    return windows


def _pack_data(time_inv, ubl,
               ant1, ant2, data, flag,
               vis_windows, flag_windows):

    time_inv = time_inv
    ant1 = ant1
    ant2 = ant2
    data = data
    flag = flag

    vis_windows = vis_windows[0][0]
    flag_windows = flag_windows[0][0]

    assert vis_windows.shape[0] == flag_windows.shape[0] == ubl.shape[0]

    for bl, (a1, a2) in sorted(enumerate(ubl), key=lambda k: random.random()):
        valid = (a1 == ant1) & (a2 == ant2)
        time_idx = time_inv[valid]

        # Ignore if we have nothing to pack
        if time_idx.size == 0:
            continue

        # Assign contiguous time range of values to the windows
        if np.all(np.diff(time_idx) == 1):
            start, end = time_idx[0], time_idx[-1] + 1
            vis_windows.oindex[bl, start:end, :, :] = data[valid, :, :]
            flag_windows.oindex[bl, start:end, :, :] = flag[valid, :, :]
        # Non-contiguous case
        else:
            vis_windows.oindex[bl, time_idx, :, :] = data[valid, :, :]
            flag_windows.oindex[bl, time_idx, :, :] = flag[valid, :, :]

    return np.array([[[True]]])


def _packed_windows(dummy_result, window):
    return window


def pack_data(time_inv, ubl,
              antenna1, antenna2,
              data, flags,
              vis_windows, flag_windows):

    window_shape = ("bl", "time", "chan", "corr")

    packing = da.blockwise(_pack_data, ("row", "chan", "corr"),
                           time_inv, ("row", ),
                           ubl, None,
                           antenna1, ("row",),
                           antenna2, ("row",),
                           data, ("row", "chan", "corr"),
                           flags, ("row", "chan", "corr"),
                           vis_windows, window_shape,
                           flag_windows, window_shape,
                           dtype=np.bool)

    vis_windows = da.blockwise(_packed_windows, window_shape,
                               packing, ("row", "chan", "corr"),
                               vis_windows, window_shape,
                               dtype=vis_windows.dtype)

    flag_windows = da.blockwise(_packed_windows, window_shape,
                                packing, ("row", "chan", "corr"),
                                flag_windows, window_shape,
                                dtype=flag_windows.dtype)

    return vis_windows, flag_windows


def _unpack_data(antenna1, antenna2, time_inv, ubl, windows):
    windows = windows[0][0]

    assert windows.shape[0] == ubl.shape[0]

    # (row, chan, corr)
    data_shape = (antenna1.shape[0],) + windows.shape[2:]
    data = np.zeros(data_shape, dtype=windows.dtype)

    for bl, (a1, a2) in sorted(enumerate(ubl), key=lambda k: random.random()):
        valid = (a1 == antenna1) & (a2 == antenna2)
        time_idx = time_inv[valid]

        # Ignore if we have nothing to pack
        if time_idx.size == 0:
            continue

        # Assign contiguous time range of values from the windows
        if np.all(np.diff(time_idx) == 1):
            start, end = time_idx[0], time_idx[-1] + 1
            data[valid, :, :] = windows.oindex[bl, start:end, :, :]
        # Non-contiguous case
        else:
            data[valid, :, :] = windows.oindex[bl, time_idx, :, :]

    return data


def unpack_data(antenna1, antenna2, time_inv, ubl, flag_windows):
    return da.blockwise(_unpack_data, ("row", "chan", "corr"),
                        antenna1, ("row",),
                        antenna2, ("row",),
                        time_inv, ("row",),
                        ubl, None,
                        flag_windows, ("bl", "time", "chan", "corr"),
                        dtype=flag_windows.dtype)
