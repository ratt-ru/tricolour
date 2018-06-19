# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

import dask.array as da
import numpy as np

from .flagging import sum_threshold_flagger as np_sum_threshold_flagger
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

    dsk.update(ant1.__dask_graph__())
    dsk.update(ant2.__dask_graph__())
    dsk.update(chunks.__dask_graph__())

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
    dsk.update(vis.__dask_graph__())
    dsk.update(flag.__dask_graph__())
    dsk.update(chunks.__dask_graph__())

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
