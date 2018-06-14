from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

import dask.array as da

from .flagging import sum_threshold_flagger
from .stokes import unpolarised_intensity as np_unpolarised_intensity


def flagger(vis, flag, chunks, **kwargs):
    """
    Dask wrapper for :func:`~tricolour.flagging.sum_threshold`
    """
    token = da.core.tokenize(vis, flag, chunks, kwargs)
    name = '-'.join(('flagger', token))
    dims = ("row", "chan", "corr")

    dsk = da.core.top(sum_threshold_flagger, name, dims,
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

    return da.core.atop(_wrapper, ("row", "chan", "corr"),
                        vis, ("row", "chan", "corr"),
                        stokes_unpol=stokes_unpol,
                        stokes_pol=stokes_pol,
                        adjust_chunks={"corr": 1},
                        dtype=vis.dtype)
