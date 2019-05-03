# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

from dask.highlevelgraph import HighLevelGraph
import dask.array as da
import dask.blockwise as db
import numpy as np

from .flagging import sum_threshold_flagger as np_sum_threshold_flagger
from .flagging import uvcontsub_flagger as np_uvcontsub_flagger
from .flagging import apply_static_mask as np_apply_static_mask
from .flagging import flag_autos as np_flag_autos

from .stokes import (polarised_intensity as np_polarised_intensity,
                     unpolarised_intensity as np_unpolarised_intensity)
from .util import check_baseline_ordering as np_check_baseline_ordering


def check_baseline_ordering(ant1, ant2, chunks, g):
    """
    Dask wrapper for :func:`~tricolour.util.check_baseline_ordering`
    """

    # We use top rather than atop because, while
    # ant1, ant2 and chunks will have the same number of chunks,
    # the size of each chunk is different
    token = da.core.tokenize(ant1, ant2, chunks, g)
    name = '-'.join(("check-baseline-ordering", token))
    dims = ("row",)

    layers = db.blockwise(np_check_baseline_ordering, name, dims,
                          ant1.name, dims,
                          ant2.name, dims,
                          chunks.name, dims,
                          numblocks={
                              ant1.name: ant1.numblocks,
                              ant2.name: ant2.numblocks,
                              chunks.name: chunks.numblocks,
                          },
                          g=g)

    graph = HighLevelGraph.from_collections(name, layers, (ant1, ant2, chunks))
    return da.Array(graph, name, chunks.chunks, dtype=np.bool)


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
    dims = ("time", "chan", "bl", "corr")

    layers = db.blockwise(np_sum_threshold_flagger, name, dims,
                          vis.name, dims,
                          flag.name, dims,
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
    dims = ("time", "chan", "bl", "corr")

    layers = db.blockwise(np_uvcontsub_flagger, name, dims,
                          vis.name, dims,
                          flag.name, dims,
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


def apply_static_mask(flag, ubl, antspos, masks,
                      spw_chanlabels, spw_chanwidths,
                      **kwargs):
    """
    Dask wrapper for :func:`~tricolour.apply_static_mask`
    """
    dims = ("time", "chan", "bl", "corr")  # corrprod = ncorr * nbl

    return da.blockwise(_apply_static_mask_wrapper, dims,
                        flag, dims,
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
    dims = ("time", "chan", "bl", "corr")

    return da.blockwise(np_flag_autos, dims,
                        flag, dims,
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
