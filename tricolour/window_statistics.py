# -*- coding: utf-8 -*-

from collections import defaultdict
from functools import partial

import dask.array as da
import numpy as np

from tricolour.packing import _WINDOW_SCHEMA


def _window_stats(flag_window, ubls, chan_freqs,
                  antenna_names, scan_no, field_name, ddid, nchanbins):
    """
    Calculate stats for a **chunk** of a flag window.
    These stats should be accumulated to form a final
    stat object for the window. See _combine_baseline_window_stats
    """
    ubls = ubls[0]
    flag_window = flag_window[0][0][0]
    chan_freqs = chan_freqs[0]

    stats = WindowStatistics(nchanbins)

    # per antenna
    for ai, a in enumerate(antenna_names):
        sel = np.logical_or(ubls[:, 1] == ai, ubls[:, 2] == ai)
        cnt = np.sum(flag_window[sel, :, :, :], dtype=np.uint64)
        sz = flag_window[sel, :, :, :].size
        stats._counts_per_ant[a] += cnt
        stats._size_per_ant[a] += sz

    # per baseline
    for bi, b in enumerate(np.unique(ubls[:, 0])):
        sel = ubls[:, 0] == b
        sela1 = antenna_names[ubls[sel, 1][0]]
        sela2 = antenna_names[ubls[sel, 2][0]]
        blname = "{0:s}&{1:s}".format(sela1, sela2)
        cnt = np.sum(flag_window[sel, :, :, :], dtype=np.uint64)
        sz = flag_window[sel, :, :, :].size
        stats._counts_per_bl[blname] += cnt
        stats._size_per_bl[blname] += sz

    # per scan and field
    cnt = np.sum(flag_window, dtype=np.uint64)
    sz = flag_window.size
    stats._counts_per_field[field_name] += cnt
    stats._size_per_field[field_name] += sz
    stats._counts_per_scan[scan_no] += cnt
    stats._size_per_scan[scan_no] += sz

    # binned per channel
    bins_edges = np.linspace(np.min(chan_freqs), np.max(chan_freqs),
                             nchanbins)
    bins = np.zeros(nchanbins, dtype=np.uint32)

    for ch_i, ch in enumerate(bins_edges[:-1]):
        sel = np.logical_and(chan_freqs >= bins_edges[ch_i],
                             chan_freqs < bins_edges[ch_i + 1])
        bins[ch_i] = np.sum(flag_window[:, :, :, sel], dtype=np.uint64)

    stats._counts_per_ddid[ddid] += bins
    stats._bins_per_ddid[ddid] = bins_edges  # frequency labels
    stats._size_per_ddid[ddid] += flag_window.size

    return stats


def _combine_baseline_window_stats(baseline_stat_list, prev_stats):
    """
    Combine per baseline window stats into a greater window stat object
    """
    result = prev_stats.copy()

    for stats in baseline_stat_list:
        result.update(stats)

    return result


def window_stats(flag_window, ubls, chan_freqs,
                 antenna_names, scan_no, field_name, ddid,
                 nchanbins=10, prev_stats=None):
    """
    Calculates stats for a chunk of a `flag_window`.
    Should be combined with the stats from other chunks.

    Parameters
    ----------
    flag_window : :class:`dask.Array`
        Flag window of shape :code:`(bl, corr, time, chan)`
    ubls : :class:`dask.Array`
        Unique baselines of shape :code:`(bl, 3)`
    chan_freqs : :class:`dask.Array`
        Channel frequencies of shape :code:`(chan,)`
    antenna_names : list or :class:`numpy.ndarray`
        Antenna names of shape :code:`(ant,)
    scan_no : int
        Scan number
    field_name : str
        Field name
    ddid : int
        Data descriptor id
    nchanbins : int, optional
        Number of bins in a channel
    prev_stats : :class:`WindowStatistics`, optional
        Previous stats

    Returns
    -------
    stats : :class:`dask.Array`
        Dask array containing a single :class:`WindowStatistics` object.
        `prev_stats` is merged into this result, if present.
    """

    # Construct as array of per-baseline stats objects
    stats = da.blockwise(_window_stats, ("bl",),
                         flag_window, _WINDOW_SCHEMA,
                         ubls, ("bl", "bl-comp"),
                         chan_freqs, ("chan",),
                         antenna_names, None,
                         scan_no, None,
                         field_name, None,
                         ddid, None,
                         nchanbins, None,
                         dtype=np.object)

    # Create an empty stats object if the user hasn't supplied one
    if prev_stats is None:
        def _window_stat_creator():
            return WindowStatistics(nchanbins)

        prev_stats = da.blockwise(_window_stat_creator, (),
                                  dtype=np.object)

    # Combine per-baseline stats into a single stats object
    return da.blockwise(_combine_baseline_window_stats, (),
                        stats, ("bl",),
                        prev_stats, (),
                        dtype=np.object)


def _combine_window_stats(*args):
    result = args[0].copy()

    for arg in args[1:]:
        result.update(arg)

    return result


def combine_window_stats(window_stats):
    """
    Combines a list of window_stats in one final stat object

    Parameters
    ----------
    window_stats : list of :class:`dask.Array`
        Each entry of the list should be a dask array containing
        a single :class:`WindowStatistics` object.

    Returns
    -------
    final_stats : :class:`dask.Array`
        Dask array containing a single :class:`WindowStatistics` object
    """
    args = (v for ws in window_stats for v in (ws, ()))

    return da.blockwise(_combine_window_stats, (),
                        *args, dtype=np.object)


class WindowStatistics(object):
    def __init__(self, nchanbins):
        self._nchanbins = nchanbins
        self._counts_per_ant = defaultdict(lambda: 0)
        self._counts_per_field = defaultdict(lambda: 0)
        self._counts_per_scan = defaultdict(lambda: 0)
        self._counts_per_bl = defaultdict(lambda: 0)
        self._size_per_ant = defaultdict(lambda: 0)
        self._size_per_field = defaultdict(lambda: 0)
        self._size_per_scan = defaultdict(lambda: 0)
        self._size_per_bl = defaultdict(lambda: 0)

        bin_factory = partial(np.zeros, nchanbins, dtype=np.uint64)
        self._counts_per_ddid = defaultdict(bin_factory)
        self._bins_per_ddid = defaultdict(lambda: 0)
        self._size_per_ddid = defaultdict(lambda: 0)

    def update(self, other):
        # Counts
        for a, count in other._counts_per_ant.items():
            self._counts_per_ant[a] += count

        for f, count in other._counts_per_field.items():
            self._counts_per_field[f] += count

        for s, count in other._counts_per_scan.items():
            self._counts_per_scan[s] += count

        for d, count in other._counts_per_ddid.items():
            self._counts_per_ddid[d] += count

        for b, count in other._counts_per_bl.items():
            self._counts_per_bl[b] += count

        # Sizes
        for a, size in other._size_per_ant.items():
            self._size_per_ant[a] += size

        for f, size in other._size_per_field.items():
            self._size_per_field[f] += size

        for s, size in other._size_per_scan.items():
            self._size_per_scan[s] += size

        for s, size in other._size_per_ddid.items():
            self._size_per_ddid[s] += size

        for b, size in other._size_per_bl.items():
            self._size_per_bl[b] += size

        # ddid
        for d, bins in other._bins_per_ddid.items():
            self._bins_per_ddid[d] = bins  # this is the frequency labels

    def copy(self):
        """ Creates a copy of the current WindowStatistics"""
        result = WindowStatistics(self._nchanbins)
        result.update(self)
        return result


def summarise_stats(final, original):
    """
    Returns
    -------
    summary : list of str
        A list of strings summarising both final and original flags
    """
    l = []  # noqa

    l.append("********************************")
    l.append("   BEGINNING OF FLAG SUMMARY    ")
    l.append("********************************")

    l.append("Per antenna:")
    for a in final._counts_per_ant:
        l.append("\t {0:s}: {1:.3f}%, original {2:.3f}%".format(
            a, final._counts_per_ant[a] * 100.0 /
            final._size_per_ant[a],
            original._counts_per_ant[a] * 100.0 /
            original._size_per_ant[a]))

    l.append("Per scan:")
    for s in final._counts_per_scan:
        l.append("\t {0:d}: {1:.3f}%, original {2:.3f}%".format(
            s, final._counts_per_scan[s] * 100.0 /
            final._size_per_scan[s],
            original._counts_per_scan[s] * 100.0 /
            original._size_per_scan[s]))

    l.append("Per field:")
    for f in final._counts_per_field:
        l.append("\t {0:s}: {1:.3f}%, original {2:.3f}%".format(
            f, final._counts_per_field[f] * 100.0 /
            final._size_per_field[f],
            original._counts_per_field[f] * 100.0 /
            original._size_per_field[f]))

    l.append("Per baseline:")
    for b in final._counts_per_bl:
        l.append("\t {0:s}: {1:.3f}%, original {2:.3f}%".format(
            b, final._counts_per_bl[b] * 100.0 /
            final._size_per_bl[b],
            original._counts_per_bl[b] * 100.0 /
            original._size_per_bl[b]))

    l.append("Per data descriptor id:")

    for d in final._counts_per_ddid:
        ratios = final._counts_per_ddid[d] * 100.0 / final._size_per_ddid[d]
        ratio_str = '\t'.join(["{0:<7.2f}".format(r) for r in ratios])
        l.append("\t {0:d}: {1:s}%".format(d, ratio_str))

        ddid_freqs = final._bins_per_ddid[d] / 1e6
        ddid_freqs_str = '\t'.join(["{0:<7.1f}".format(f) for f in ddid_freqs])
        l.append("\t    {0:s} MHz".format(ddid_freqs_str))

    l.append("********************************")
    l.append("       END OF FLAG SUMMARY      ")
    l.append("********************************")

    return l
