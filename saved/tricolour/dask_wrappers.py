# -*- coding: utf-8 -*-

from functools import wraps

from dask.highlevelgraph import HighLevelGraph
import dask.array as da
import dask.blockwise as db

from tricolour.flagging import (
    flag_nans_and_zeros as np_flag_nans_and_zeros,
    sum_threshold_flagger as np_sum_threshold_flagger,
    uvcontsub_flagger as np_uvcontsub_flagger,
    apply_static_mask as np_apply_static_mask,
    flag_autos as np_flag_autos)

from tricolour.stokes import (
    polarised_intensity as np_polarised_intensity,
    unpolarised_intensity as np_unpolarised_intensity)

from tricolour.packing import _WINDOW_SCHEMA


def sum_threshold_flagger(vis, flag, **kwargs):
    """
    Dask wrapper for :func:`~tricolour.flagging.sum_threshold_flagger`
    """

    # We use dask.blockwise.blockwise rather than dask.array.blockwise because,
    # while
    # ant1, ant2 and chunks will have the same number of chunks,
    # the size of each chunk is different
    token = da.core.tokenize(vis, flag, kwargs)
    name = 'sum-threshold-flagger-' + token

    layers = db.blockwise(np_sum_threshold_flagger, name, _WINDOW_SCHEMA,
                          vis.name, _WINDOW_SCHEMA,
                          flag.name, _WINDOW_SCHEMA,
                          numblocks={
                              vis.name: vis.numblocks,
                              flag.name: flag.numblocks,
                          },
                          **kwargs)

    # Add input graphs to the graph
    graph = HighLevelGraph.from_collections(name, layers, (vis, flag))
    return da.Array(graph, name, vis.chunks, dtype=flag.dtype)


def uvcontsub_flagger(vis, flag, **kwargs):
    """
    Dask wrapper for :func:`~tricolour.uvcontsub_flagger`
    """
    name = 'uvcontsub-flagger-' + da.core.tokenize(vis, flag, **kwargs)

    layers = db.blockwise(np_uvcontsub_flagger, name, _WINDOW_SCHEMA,
                          vis.name, _WINDOW_SCHEMA,
                          flag.name, _WINDOW_SCHEMA,
                          numblocks={
                              vis.name: vis.numblocks,
                              flag.name: flag.numblocks,
                          },
                          **kwargs)

    # Add input graphs to the graph
    graph = HighLevelGraph.from_collections(name, layers, (vis, flag))
    return da.Array(graph, name, vis.chunks, dtype=flag.dtype)


def _apply_static_mask_wrapper(flag, ubl, antspos, masks,
                               spw_chanlabels, spw_chanwidths,
                               **kwargs):

    return np_apply_static_mask(flag, ubl[0], antspos, masks,
                                spw_chanlabels, spw_chanwidths,
                                **kwargs)


def flag_nans_and_zeros(vis_windows, flag_windows):
    return da.blockwise(np_flag_nans_and_zeros, _WINDOW_SCHEMA,
                        vis_windows, _WINDOW_SCHEMA,
                        flag_windows, _WINDOW_SCHEMA,
                        dtype=flag_windows.dtype)


def apply_static_mask(flag, ubl, antspos, masks,
                      spw_chanlabels, spw_chanwidths,
                      **kwargs):
    """
    Dask wrapper for :func:`~tricolour.apply_static_mask`
    """

    return da.blockwise(_apply_static_mask_wrapper, _WINDOW_SCHEMA,
                        flag, _WINDOW_SCHEMA,
                        ubl, ("bl", "bl-comp"),
                        antspos, None,
                        masks, None,
                        spw_chanlabels, None,
                        spw_chanwidths, None,
                        dtype=flag.dtype,
                        **kwargs)


def _flag_autos_wrapper(flag, ubl):
    return np_flag_autos(flag, ubl[0])


def flag_autos(flag, ubl, **kwargs):
    """
    Dask wrapper for :func:`~tricolour.flag_autos`
    """
    return da.blockwise(np_flag_autos, _WINDOW_SCHEMA,
                        flag, _WINDOW_SCHEMA,
                        ubl, ("bl", "bl-comp"),
                        dtype=flag.dtype)


def unpolarised_intensity(vis, stokes_unpol, stokes_pol):
    """
    Dask wrapper for :func:`~tricolour.stokes.unpolarised_intensity`
    """
    @wraps(np_unpolarised_intensity)
    def _wrapper(vis, stokes_unpol=None, stokes_pol=None):
        return np_unpolarised_intensity(vis, stokes_unpol, stokes_pol)

    return da.blockwise(_wrapper, ("row", "chan", "corr"),
                        vis, ("row", "chan", "corr"),
                        stokes_unpol=stokes_unpol,
                        stokes_pol=stokes_pol,
                        adjust_chunks={"corr": 1},
                        dtype=vis.dtype)


def polarised_intensity(vis, stokes_pol):
    """
    Dask wrapper for :func:`~tricolour.stokes.polarised_intensity`
    """
    @wraps(np_polarised_intensity)
    def _wrapper(vis, stokes_pol=None):
        return np_polarised_intensity(vis, stokes_pol)

    return da.blockwise(_wrapper, ("row", "chan", "corr"),
                        vis, ("row", "chan", "corr"),
                        stokes_pol=stokes_pol,
                        adjust_chunks={"corr": 1},
                        dtype=vis.dtype)
