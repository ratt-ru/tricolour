# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np


def check_baseline_ordering(ant1, ant2, chunks, g=None):
    """
    Checks that the baseline ordering for each timestep
    is the same

    Parameters
    ----------
    ant1: :class:`numpy.ndarray`
        antenna1 values
    ant2: :class:`numpy.ndarray`
        antenna2 values
    chunks: :class: `numpy.ndarray`
        Number of baselines per timestep.
    g: int
        Group number

    Returns
    -------
    :class:`numpy.ndarray`
        Array of True equal with shape :code:`(chunks.size,)`
    """
    if not ant1.size == ant2.size == np.sum(chunks):
        raise ValueError("Number of antenna values do not equal chunk sum")

    start = 0
    chunk1_len = chunks[0]

    for c, chunk in enumerate(chunks):
        end = start + chunk
        ant1_ok = np.all(ant1[start:end] == ant1[0:chunk1_len])
        ant2_ok = np.all(ant2[start:end] == ant2[0:chunk1_len])

        if not ant1_ok or not ant2_ok:
            raise ValueError("Baseline ordering for chunk %d in group %d "
                             "is inconsistent with other chunks. "
                             "Fully general Measurement Sets are not "
                             "yet supported\n"
                             "%s != %s\n"
                             "%s != %s" % (c, g,
                                           ant1[start:end], ant1[0:chunk1_len],
                                           ant2[start:end], ant2[0:chunk1_len]))

    return np.full(chunks.shape, True, np.bool)


def aggregate_chunks(chunks, max_chunks, return_groups=False):
    """
    Aggregate dask ``chunks`` together into chunks no larger than
    ``max_chunks``.

    .. code-block:: python

        chunks, max_c = ((3,4,6,3,6,7),(1,1,1,1,1,1)), (10,3)
        expected = ((7,9,6,7), (2,2,1,1))
        assert aggregate_chunks(chunks, max_c) == expected


    Parameters
    ----------
    chunks : sequence of tuples or tuple
    max_chunks : sequence of ints or int
    return_groups : bool

    Returns
    -------
    sequence of tuples or tuple

    """

    if isinstance(max_chunks, int):
        chunks = (chunks,)
        max_chunks = (max_chunks,)

    singleton = True if len(max_chunks) == 1 else False

    if len(chunks) != len(max_chunks):
        raise ValueError("len(chunks) != len(max_chunks)")

    if not all(len(chunks[0]) == len(c) for c in chunks):
        raise ValueError("Number of chunks do not match")

    agg_chunks = [[] for _ in max_chunks]
    agg_chunk_counts = [0] * len(max_chunks)
    chunk_scratch = [0] * len(max_chunks)
    ndim = len(chunks[0])

    # For each chunk dimension
    for di in range(ndim):
        # For each chunk
        aggregate = False

        for ci, chunk in enumerate(chunks):
            chunk_scratch[ci] = agg_chunk_counts[ci] + chunk[di]
            if chunk_scratch[ci] > max_chunks[ci]:
                aggregate = True

        if aggregate:
            for ci, chunk in enumerate(chunks):
                agg_chunks[ci].append(agg_chunk_counts[ci])
                agg_chunk_counts[ci] = chunk[di]
        else:
            for ci, chunk in enumerate(chunks):
                agg_chunk_counts[ci] = chunk_scratch[ci]

    # Do the final aggregation
    for ci, chunk in enumerate(chunks):
        agg_chunks[ci].append(agg_chunk_counts[ci])
        agg_chunk_counts[ci] = chunk[di]

    agg_chunks = tuple(tuple(ac) for ac in agg_chunks)

    return agg_chunks[0] if singleton else agg_chunks


def casa_style_range(val):
    """ returns list of ints """
    if not isinstance(val, str):
        raise argparse.ArgumentTypeError("Value must be of type string")
    if val == "":
        return (0, 1e9)
    elif re.match(r"^(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?~"
                  r"(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?[\s]*[m]?$", val):

        return map(float, val.replace(" ", "")
                             .replace("\t", "")
                             .replace("m", "")
                             .split("~"))
    else:
        raise ValueError("Value must be range or blank")
