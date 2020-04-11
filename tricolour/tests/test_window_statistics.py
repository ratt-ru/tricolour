"""Tests for :mod:`tricolour.flagging`."""

import dask.array as da
import numpy as np
import pytest

from tricolour.window_statistics import (WindowStatistics, window_stats,
                                         combine_window_stats,
                                         summarise_stats)

ntime = 10
nchan = 16
ncorr = 4


@pytest.fixture
def antenna_names():
    return ["A1", "A2", "A3", "A4"]


@pytest.fixture
def unique_baselines(antenna_names):
    ant1, ant2 = np.triu_indices(len(antenna_names), 0)
    ubl = np.unique(np.stack([ant1, ant2], axis=1), axis=1)
    bl_range = np.arange(ubl.shape[0])[:, None]
    return np.concatenate([bl_range, ubl], axis=1)


@pytest.fixture
def channels():
    return np.linspace(.856e9, 2 * .856e9, nchan)


@pytest.fixture
def flag_windows(unique_baselines):
    nbl = unique_baselines.shape[0]
    return np.random.randint(0, 2, (nbl, ncorr, ntime, nchan))


@pytest.mark.parametrize("scan_nrs", [[0, 1, 2]])
@pytest.mark.parametrize("field_names", [["M87", "Sag A*"]])
@pytest.mark.parametrize("ddids", [[0, 1, 2]])
def test_window_statistics(antenna_names, unique_baselines,
                           channels, flag_windows,
                           scan_nrs, field_names, ddids):

    fw = flag_windows
    # Chunk into groups of 2 baselines
    ubl = da.from_array(unique_baselines, chunks=(2, 3))
    chunks = fw.shape[:2] + (ubl.chunks[0],) + (fw.shape[3],)
    flag_windows = da.from_array(flag_windows, chunks=chunks)
    # Channels are full resolution
    channels = da.from_array(channels, chunks=channels.shape[0])

    prev_stats = None
    stats_list = []

    for field_name in field_names:
        for scan_nr in scan_nrs:
            for ddid in ddids:
                prev_stats = window_stats(flag_windows, ubl, channels,
                                          antenna_names, scan_nr,
                                          field_name, ddid,
                                          prev_stats=prev_stats)
                stats_list.append(prev_stats)

    # Test sequential accumulation of stats
    stats = prev_stats.compute()
    assert set(field_names) == set(stats._counts_per_field.keys())
    assert set(scan_nrs) == set(stats._counts_per_scan.keys())
    assert set(ddids) == set(stats._counts_per_ddid.keys())

    # Test combination of lists of window stats
    assert len(stats_list) > 0
    combined_stats = combine_window_stats(stats_list)
    combined_stats = combined_stats.compute()
    assert isinstance(combined_stats, WindowStatistics)

    summary = summarise_stats(stats, combined_stats)
    summary_str = '\n'.join(summary)  # noqa
