# -*- coding: utf-8 -*-

from os.path import join as pjoin
import random
from tempfile import mkdtemp

import dask
import dask.array as da
from dask.highlevelgraph import HighLevelGraph
import numba
import numpy as np
import zarr


_WINDOW_SCHEMA = ("bl", "corr", "time", "chan")


def _debug_inputs(data):
    if isinstance(data, np.ndarray):
        return (data.shape, data.dtype)
    elif isinstance(data, list):
        return [_debug_inputs(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(_debug_inputs(d) for d in data)
    elif isinstance(data, dict):
        return {k: _debug_inputs(v) for k, v in data.items()}
    else:
        return type(data)


def _print(data):
    from pprint import pprint
    pprint(_debug_inputs(data))


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


def _create_window(name, ntime, nchan, nbl, ncorr,
                   dtype, default, token, backend="numpy", path=None):
    if backend == "zarr-disk":
        return zarr.creation.create(shape=(nbl, ncorr, ntime, nchan),
                                    chunks=(1, ncorr, ntime, nchan),
                                    compressor=None,
                                    dtype=dtype,
                                    synchronizer=zarr.ThreadSynchronizer(),
                                    overwrite=True,
                                    fill_value=default,
                                    read_only=False,
                                    store=pjoin(path, "-".join((name, token))))
    elif backend == "numpy":
        return np.full((nbl, ncorr, ntime, nchan), default, dtype=dtype)
    else:
        raise ValueError("Invalid backend '%s'" % backend)


def _create_window_dask(name, ntime, nchan, nbl, ncorr, token,
                        dtype, default=0, backend="numpy", path=None):
    if backend == "zarr-disk" and path is None:
        path = mkdtemp(prefix='-'.join(('tricolour', name, 'windows', '')))

    # Include name and token in new token
    token = dask.base.tokenize(name, ntime, nchan, nbl, ncorr, token,
                               dtype, default, backend, path)

    collection_name = '-'.join(("create", name, "windows", token))
    layers = {(collection_name, 0): (_create_window, name,
                                     ntime, nchan, nbl, ncorr,
                                     dtype, default, token, backend, path)}

    graph = HighLevelGraph.from_collections(collection_name, layers, ())
    chunks = ((0,),)  # One chunk containing single zarr array object
    return da.Array(graph, collection_name, chunks, dtype=np.object)


def create_vis_windows(ntime, nchan, nbl, ncorr, token,
                       dtype, default=np.nan + np.nan * 1j,
                       backend="numpy", path=None):
    """
    Returns
    -------
    vis_window : :class:`dask.array.Array`
        dask array containing either

        1. A zarr array
        2. A list of per-baseline numpy arrays

        Compute should never directly be called on this array,
        but it should be passed to other functions
    """

    return _create_window_dask("vis", ntime, nchan, nbl, ncorr, token,
                               dtype, default, backend, path)


def create_flag_windows(ntime, nchan, nbl, ncorr, token,
                        dtype, default=1, backend="numpy", path=None):
    """
    Returns
    -------
    vis_window : :class:`dask.array.Array`
        dask array containing either

        1. A zarr array
        2. A list of per-baseline numpy arrays

        Compute should never directly be called on this array,
        but it should be passed to other functions
    """
    return _create_window_dask("flag", ntime, nchan, nbl, ncorr, token,
                               dtype, default, backend, path)


def _rand_sort(key):
    return random.random()


@numba.njit(nogil=True, cache=True)
def _zarr_pack_transpose(data, valid):
    rows, chans, corrs = data.shape

    if rows != valid.shape[0]:
        raise ValueError("data rows don't match valid rows")

    valid_rows = []

    for r in range(rows):
        if valid[r]:
            valid_rows.append(r)

    result = np.empty((corrs, len(valid_rows), chans), dtype=data.dtype)

    # We're selecting all times for one baseline
    # So we order by (corr, row (time), chan) for
    # the assignments below
    for out_row, in_row in enumerate(valid_rows):
        for f in range(chans):
            for c in range(corrs):
                result[c, out_row, f] = data[in_row, f, c]

    return result


@numba.njit(nogil=True, cache=True)
def _numpy_pack_transpose(window, data, valid, row_idx):
    rows, chans, corrs = data.shape

    if rows != valid.shape[0]:
        raise ValueError("data rows don't match valid rows")

    valid_rows = []

    for r in range(rows):
        if valid[r]:
            valid_rows.append(r)

    if len(valid_rows) != row_idx.shape[0]:
        raise ValueError("len(valid_rows) != row_idx.shape[0]")

    # We're selecting all times for one baseline
    # So we order by (corr, row (time), chan) for
    # the assignments below
    for out_row, in_row in zip(row_idx, valid_rows):
        for f in range(chans):
            for c in range(corrs):
                window[c, out_row, f] = data[in_row, f, c]


def _slow_pack_data(time_inv, ubl,
                    ant1, ant2, data, flag,
                    vis_windows, flag_windows):
    vis_windows = vis_windows[0]
    flag_windows = flag_windows[0]

    assert vis_windows.shape == flag_windows.shape

    if (isinstance(vis_windows, zarr.Array) and
            isinstance(flag_windows, zarr.Array)):
        zarr_case = True
    elif (isinstance(vis_windows, np.ndarray) and
            isinstance(flag_windows, np.ndarray)):
        zarr_case = False
    else:
        raise TypeError("visibility '%s' and flag '%s' types must both "
                        "be numpy or zarr arrays")

    # We're dealing with all baselines at once
    assert sum(len(bl_list[0]) for bl_list in ubl) == vis_windows.shape[0]

    # This double for loop is strange, mostly because ubl and bl_index
    # are lists (or lists of lists) of ndarrays. As the "bl" and "bl-comp"
    # dimensions are reduced, all chunks are supplied to this function as
    # elements of a list.
    # The outer loop is a loop over each chunk, while the inner loop
    # a loop over the baselines in each chunk
    for bl_list in ubl:
        for bl, a1, a2 in bl_list[0]:
            valid = (a1 == ant1) & (a2 == ant2)
            time_idx = time_inv[valid]

            # Ignore if we have nothing to pack
            if time_idx.size == 0:
                continue

            if zarr_case:
                # Slice if we have a contiguous time range of values
                if np.all(np.diff(time_idx) == 1):
                    time_idx = slice(time_idx[0], time_idx[-1] + 1)

                data_t = _zarr_pack_transpose(data, valid)
                flag_t = _zarr_pack_transpose(flag, valid)

                vis_windows.oindex[bl, :, time_idx, :] = data_t
                flag_windows.oindex[bl, :, time_idx, :] = flag_t
            else:
                # Faster path
                _numpy_pack_transpose(vis_windows[bl], data, valid, time_idx)
                _numpy_pack_transpose(flag_windows[bl], flag, valid, time_idx)

    return np.array([[[True]]])


@numba.njit(nogil=True, cache=True)
def _numba_pack_data(time_inv, ubl,
                     ant1, ant2, data, flag,
                     vis_windows, flag_windows):
    rows, chans, corrs = data.shape

    if vis_windows.shape[3] != chans:
        raise ValueError("channels mismatch")

    if vis_windows.shape[1] != corrs:
        raise ValueError("correlations mismatch")

    if vis_windows.shape != flag_windows.shape:
        raise ValueError("vis_windows.shape != flag_windows.shape")

    # We're dealing with all baselines at once
    assert ubl.shape == (vis_windows.shape[0], 3)

    # Pack each baseline
    for b in range(ubl.shape[0]):
        bl, a1, a2 = ubl[b]

        for r in range(rows):
            # Only handle rows for this baseline
            if ant1[r] != a1 or ant2[r] != a2:
                continue

            # lookup time index
            t = time_inv[r]

            for f in range(chans):
                for c in range(corrs):
                    vis_windows[bl, c, t, f] = data[r, f, c]
                    flag_windows[bl, c, t, f] = flag[r, f, c]

    return np.array([[[True]]])


def _fast_pack_data(time_inv, ubl,
                    ant1, ant2, data, flag,
                    vis_windows, flag_windows):

    # Flatten the baseline lists
    ubl = np.concatenate([bl for bl_list in ubl for bl in bl_list])

    return _numba_pack_data(time_inv, ubl,
                            ant1, ant2,
                            data, flag,
                            vis_windows[0],
                            flag_windows[0])


def _packed_windows(dummy_result, ubl, window):
    window = window[0]
    bl_index = ubl[0][:, 0]

    # Slice if possible
    if np.all(np.diff(bl_index) == 1):
        bl_index = slice(bl_index[0], bl_index[-1] + 1)

    return window[bl_index, :, :, :]


def pack_data(time_inv, ubl,
              antenna1, antenna2,
              data, flags, ntime,
              backend="numpy", path=None,
              return_objs=False):

    nchan, ncorr = data.shape[1:3]
    nbl = ubl.shape[0]

    token = dask.base.tokenize(time_inv, ubl, antenna1, antenna2,
                               data, flags, ntime, backend, path,
                               return_objs)

    vis_win_obj = create_vis_windows(ntime, nchan, nbl, ncorr, token,
                                     dtype=data.dtype,
                                     backend=backend,
                                     path=path)

    flag_win_obj = create_flag_windows(ntime, nchan, nbl, ncorr, token,
                                       dtype=flags.dtype,
                                       backend=backend,
                                       path=path)

    if backend == "numpy":
        pack_fn = _fast_pack_data
    elif backend == "zarr-disk":
        pack_fn = _slow_pack_data
    else:
        raise ValueError("Invalid backend '%s'" % backend)

    # Pack data into our window objects
    packing = da.blockwise(pack_fn, ("row", "chan", "corr"),
                           time_inv, ("row", ),
                           ubl, ("bl", "bl-comp"),
                           antenna1, ("row",),
                           antenna2, ("row",),
                           data, ("row", "chan", "corr"),
                           flags, ("row", "chan", "corr"),
                           vis_win_obj, ("windim",),
                           flag_win_obj, ("windim",),
                           dtype=np.bool)

    # Expose visibility data at it's full resolution
    vis_windows = da.blockwise(_packed_windows, _WINDOW_SCHEMA,
                               packing, ("row", "chan", "corr"),
                               ubl, ("bl", "bl-comp"),
                               vis_win_obj, ("windim",),
                               new_axes={"time": ntime},
                               dtype=data.dtype)

    flag_windows = da.blockwise(_packed_windows, _WINDOW_SCHEMA,
                                packing, ("row", "chan", "corr"),
                                ubl, ("bl", "bl-comp"),
                                flag_win_obj, ("windim",),
                                new_axes={"time": ntime},
                                dtype=flags.dtype)

    if return_objs:
        return vis_windows, flag_windows, vis_win_obj, flag_win_obj

    return vis_windows, flag_windows


@numba.njit(nogil=True, cache=True)
def _numpy_unpack_transpose(data, window, valid, row_idx):
    rows, chans, corrs = data.shape

    valid_rows = []

    for r in range(rows):
        if valid[r]:
            valid_rows.append(r)

    if len(valid_rows) != row_idx.shape[0]:
        raise ValueError("len(valid_rows) != row_idx.shape[0]")

    # We're selecting all times for one baseline
    # So we order by (corr, row (time), chan) for
    # the assignments below
    for out_row, in_row in zip(valid_rows, row_idx):
        for f in range(chans):
            for c in range(corrs):
                data[out_row, f, c] = window[c, in_row, f]


def _unpack_data(antenna1, antenna2, time_inv, ubl, windows):
    exemplar = windows[0][0]

    # (row, chan, corr)
    data_shape = (antenna1.shape[0], exemplar.shape[3], exemplar.shape[1])
    data = np.zeros(data_shape, dtype=exemplar.dtype)

    for baselines, window in zip(ubl, windows):
        baselines = baselines[0]
        window = window[0]
        bl_min = baselines[:, 0].min()

        for bl, a1, a2 in baselines:
            # Normalise the baseline index within this baseline chunk
            bl = bl - bl_min
            valid = (a1 == antenna1) & (a2 == antenna2)
            time_idx = time_inv[valid]

            # Ignore if we have nothing to pack
            if time_idx.size == 0:
                continue

            _numpy_unpack_transpose(data, window[bl], valid, time_idx)

    return data


def unpack_data(antenna1, antenna2, time_inv, ubl, flag_windows):
    return da.blockwise(_unpack_data, ("row", "chan", "corr"),
                        antenna1, ("row",),
                        antenna2, ("row",),
                        time_inv, ("row",),
                        ubl, ("bl", "bl-comp"),
                        flag_windows, _WINDOW_SCHEMA,
                        dtype=flag_windows.dtype)
