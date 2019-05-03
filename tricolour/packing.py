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
from numcodecs import Blosc
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


def _create_vis_windows(ntime, nchan, nbl, ncorr, bl_chunks,
                        dtype, backend, path):
    if backend == "zarr":
        if path is None:
            path = mkdtemp(prefix='tricolour-vis-windows-')

        compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)

        bl_chunks = 1
        vis = zarr.creation.create(shape=(ntime, nchan, nbl, ncorr),
                                   chunks=(ntime, nchan, bl_chunks, ncorr),
                                   compressor=compressor,
                                   dtype=dtype,
                                   synchronizer=zarr.ThreadSynchronizer(),
                                   overwrite=True,
                                   fill_value=0 + 0j,
                                   read_only=False,
                                   store=pjoin(path, "vis"))
    elif backend == "numpy":
        return np.zeros((ntime, nchan, nbl, ncorr), dtype=dtype)
    else:
        raise ValueError("Invalid backend '%s'" % backend)

    return vis


def _create_flag_windows(ntime, nchan, nbl, ncorr, bl_chunks,
                         dtype, backend, path):

    if backend == "zarr":
        if path is None:
            path = mkdtemp(prefix='tricolour-flag-windows-')

        compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.SHUFFLE)

        bl_chunks = 1

        return zarr.creation.create(shape=(ntime, nchan, nbl, ncorr),
                                    chunks=(ntime, nchan, bl_chunks, ncorr),
                                    compressor=compressor,
                                    dtype=dtype,
                                    synchronizer=zarr.ThreadSynchronizer(),
                                    overwrite=True,
                                    # Unflagged by default
                                    fill_value=0,
                                    read_only=False,
                                    store=pjoin(path, "flag"))
    elif backend == "numpy":
        return np.zeros((ntime, nchan, nbl, ncorr), dtype=dtype)
    else:
        raise ValueError("Invalid backend '%s'" % backend)


def create_vis_windows(ntime, nchan, nbl, ncorr, bl_chunks,
                       dtype, backend="numpy", path=None):

    token = dask.base.tokenize(ntime, nchan, nbl, ncorr,
                               bl_chunks, dtype, path)

    name = "create-vis-windows-" + token
    layers = {(name, 0): (_create_vis_windows,
                          ntime, nchan, nbl, ncorr,
                          bl_chunks, dtype, backend, path)}

    graph = HighLevelGraph.from_collections(name, layers, ())
    chunks = ((0,),)  # One chunk containing single zarr array object
    windows = da.Array(graph, name, chunks, dtype=dtype)

    return windows


def create_flag_windows(ntime, nchan, nbl, ncorr, bl_chunks,
                        dtype, backend="numpy", path=None):
    token = dask.base.tokenize(ntime, nchan, nbl, ncorr,
                               bl_chunks, dtype, path)

    name = "create-flag-windows-" + token
    layers = {(name, 0): (_create_flag_windows,
                          ntime, nchan, nbl, ncorr,
                          bl_chunks, dtype, backend, path)}

    graph = HighLevelGraph.from_collections(name, layers, ())
    chunks = ((0,),)  # One chunk containing single zarr array object
    windows = da.Array(graph, name, chunks, dtype=dtype)

    return windows


def _rand_sort(key):
    return random.random()


def _pack_data(time_inv, ubl, bl_index,
               ant1, ant2, data, flag,
               vis_windows, flag_windows):

    time_inv = time_inv
    ant1 = ant1
    ant2 = ant2
    data = data
    flag = flag

    vis_windows = vis_windows[0]
    flag_windows = flag_windows[0]

    assert vis_windows.shape[2] == flag_windows.shape[2]

    if isinstance(vis_windows, zarr.Array) and (flag_windows, zarr.Array):
        zarr_case = True
    elif (isinstance(vis_windows, np.ndarray) and
            isinstance(flag_windows, np.ndarray)):
        zarr_case = False
    else:
        raise TypeError("visibility '%s' and flag '%s' types must both "
                        "be zarr or numpy arrays")

    # This double for loop is strange, mostly because ubl and bl_index
    # are lists (or lists of lists) of ndarrays. As the "bl" and "bl-comp"
    # dimensions are reduced, all chunks are supplied to this function as
    # elements of a list.
    # The outer loop is a loop over each chunk, while the inner loop
    # a loop over the baselines in each chunk
    for bl_idx_list, bl_list in zip(bl_index, ubl):
        for bl, (a1, a2) in zip(bl_idx_list, bl_list[0]):
            valid = (a1 == ant1) & (a2 == ant2)
            time_idx = time_inv[valid]

            # Ignore if we have nothing to pack
            if time_idx.size == 0:
                continue

            # Slice if we have a contiguous time range of values
            if np.all(np.diff(time_idx) == 1):
                time_idx = slice(time_idx[0], time_idx[-1] + 1)

            if zarr_case:
                vis_windows.oindex[time_idx, :, bl, :] = data[valid, :, :]
                flag_windows.oindex[time_idx, :, bl, :] = flag[valid, :, :]
            else:
                vis_windows[time_idx, :, bl, :] = data[valid, :, :]
                flag_windows[time_idx, :, bl, :] = flag[valid, :, :]

    return np.array([[[True]]])


def _packed_windows(dummy_result, ubl, bl_index, window):
    window = window[0]

    if np.all(np.diff(bl_index) == 1):
        bl_index = slice(bl_index[0], bl_index[-1] + 1)

    return window[:, :, bl_index, :]


def pack_data(time_inv, ubl,
              antenna1, antenna2,
              data, flags,
              vis_windows, flag_windows,
              ntime):

    window_shape = ("time", "chan", "bl", "corr")

    bl_index = da.arange(ubl.shape[0], chunks=ubl.chunks[0])

    # Pack data into our window objects
    packing = da.blockwise(_pack_data, ("row", "chan", "corr"),
                           time_inv, ("row", ),
                           ubl, ("bl", "bl-comp"),
                           bl_index, ("bl",),
                           antenna1, ("row",),
                           antenna2, ("row",),
                           data, ("row", "chan", "corr"),
                           flags, ("row", "chan", "corr"),
                           vis_windows, ("windim",),
                           flag_windows, ("windim",),
                           dtype=np.bool)

    # Expose visibility data at it's full resolution
    vis_windows = da.blockwise(_packed_windows, window_shape,
                               packing, ("row", "chan", "corr"),
                               ubl, ("bl", "bl-comp"),
                               bl_index, ("bl",),
                               vis_windows, ("windim",),
                               new_axes={"time": ntime},
                               dtype=vis_windows.dtype)

    flag_windows = da.blockwise(_packed_windows, window_shape,
                                packing, ("row", "chan", "corr"),
                                ubl, ("bl", "bl-comp"),
                                bl_index, ("bl",),
                                flag_windows, ("windim",),
                                new_axes={"time": ntime},
                                dtype=flag_windows.dtype)

    return vis_windows, flag_windows


def _unpack_data(antenna1, antenna2, time_inv, ubl, windows):
    exemplar = windows[0][0]

    # (row, chan, corr)
    data_shape = (antenna1.shape[0], exemplar.shape[1], exemplar.shape[3])
    data = np.zeros(data_shape, dtype=exemplar.dtype)

    for baselines, window in zip(ubl, windows[0]):
        for bl, (a1, a2) in enumerate(baselines[0]):
            valid = (a1 == antenna1) & (a2 == antenna2)
            time_idx = time_inv[valid]

            # Ignore if we have nothing to pack
            if time_idx.size == 0:
                continue

            # Slice if we have a contiguous time range of values
            if np.all(np.diff(time_idx) == 1):
                time_idx = slice(time_idx[0], time_idx[-1] + 1)

            data[valid, :, :] = window[time_idx, :, bl, :]

    return data


def unpack_data(antenna1, antenna2, time_inv, ubl, flag_windows):
    return da.blockwise(_unpack_data, ("row", "chan", "corr"),
                        antenna1, ("row",),
                        antenna2, ("row",),
                        time_inv, ("row",),
                        ubl, ("bl", "bl-comp"),
                        flag_windows, ("time", "chan", "bl", "corr"),
                        dtype=flag_windows.dtype)
