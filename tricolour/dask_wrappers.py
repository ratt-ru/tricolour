# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

from dask import sharedict
import dask.array as da
import numpy as np

from .flagging import sum_threshold_flagger as np_sum_threshold_flagger
from .flagging import uvcontsub_flagger as np_uvcontsub_flagger
from .flagging import apply_static_mask as np_apply_static_mask
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

    dsk = da.core.top(np_check_baseline_ordering, name, dims,
                      ant1.name, dims,
                      ant2.name, dims,
                      chunks.name, dims,
                      numblocks={
                          ant1.name: ant1.numblocks,
                          ant2.name: ant2.numblocks,
                          chunks.name: chunks.numblocks,
                      },
                      g=g)

    dsk = sharedict.merge(dsk, ant1.__dask_graph__(),
                          ant2.__dask_graph__(),
                          chunks.__dask_graph__())

    return da.Array(dsk, name, chunks.chunks, dtype=np.bool)

def sum_threshold_flagger(vis, flag, chunks, **kwargs):
    """
    Dask wrapper for :func:`~tricolour.flagging.sum_threshold_flagger`
    """

    # We use top rather than atop because, while
    # ant1, ant2 and chunks will have the same number of chunks,
    # the size of each chunk is different
    token = da.core.tokenize(vis, flag, chunks, kwargs)
    name = '-'.join(('flagger', token))
    dims = ("row", "chan", "corr")

    dsk = da.core.top(np_sum_threshold_flagger, name, dims,
                      vis.name, dims,
                      flag.name, dims,
                      chunks.name, ("row",),
                      numblocks={
                          vis.name: vis.numblocks,
                          flag.name: flag.numblocks,
                          chunks.name: chunks.numblocks
                      },
                      **kwargs)

    # Add input graphs to the graph
    dsk = sharedict.merge(dsk, vis.__dask_graph__(),
                          flag.__dask_graph__(),
                          chunks.__dask_graph__())

    return da.Array(dsk, name, vis.chunks, dtype=flag.dtype)

def uvcontsub_flagger(vis, flag, **kwargs):
    """
    Dask wrapper for :func:`~tricolour.uvcontsub_flagger`
    """

    # We use top rather than atop because, while
    # ant1, ant2 and chunks will have the same number of chunks,
    # the size of each chunk is different
    token = da.core.tokenize(vis, flag, kwargs)
    name = '-'.join(('flagger', token))
    dims = ("row", "chan", "corr")

    dsk = da.core.top(np_uvcontsub_flagger, name, dims,
                      vis.name, dims,
                      flag.name, dims,
                      numblocks={
                          vis.name: vis.numblocks,
                          flag.name: flag.numblocks,
                      },
                      **kwargs)

    # Add input graphs to the graph
    dsk = sharedict.merge(dsk, vis.__dask_graph__(),
                          flag.__dask_graph__())

    return da.Array(dsk, name, vis.chunks, dtype=flag.dtype)

def apply_static_mask(vis, flag, a1, a2, antspos, masks, spw_chanlabels, spw_chanwidths, ncorrs, **kwargs):
    """
    Dask wrapper for :func:`~tricolour.apply_static_mask`
    """

    # We use top rather than atop because, while
    # ant1, ant2 and chunks will have the same number of chunks,
    # the size of each chunk is different
    token = da.core.tokenize(vis, flag, kwargs)
    name = '-'.join(('flagger', token))
    dims = ("row", "chan", "corr") # corr = ncorr * nbl
    kwargs["antspos"] = antspos
    kwargs["masks"] = masks
    kwargs["spw_chanlabels"] = spw_chanlabels
    kwargs["spw_chanwidths"] = spw_chanwidths
    kwargs["ncorr"] = ncorrs
    dsk = da.core.top(np_apply_static_mask, name, dims,
                      vis.name, dims,
                      flag.name, dims,
                      a1.name, ("row", "corr"),
                      a2.name, ("row", "corr"),
                      numblocks={
                          vis.name: vis.numblocks,
                          flag.name: flag.numblocks,
                          a1.name: a1.numblocks,
                          a2.name: a2.numblocks,
                      },
                      **kwargs)

    # Add input graphs to the graph
    dsk = sharedict.merge(dsk, vis.__dask_graph__(),
                          flag.__dask_graph__(),
                          a1.__dask_graph__(),
                          a2.__dask_graph__())

    return da.Array(dsk, name, vis.chunks, dtype=flag.dtype)


def unpolarised_intensity(vis, stokes_unpol, stokes_pol):
    """
    Dask wrapper for :func:`~tricolour.stokes.unpolarised_intensity`
    """
    @wraps(np_unpolarised_intensity)
    def _wrapper(vis, stokes_unpol=None, stokes_pol=None):
        return np_unpolarised_intensity(vis, stokes_unpol, stokes_pol)

    return da.core.atop(_wrapper, ("time", "bl", "chan", "corr"),
                        vis, ("time", "bl", "chan", "corr"),
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

    return da.core.atop(_wrapper, ("time", "bl", "chan", "corr"),
                        vis, ("time", "bl", "chan", "corr"),
                        stokes_pol=stokes_pol,
                        adjust_chunks={"corr": 1},
                        dtype=vis.dtype)
