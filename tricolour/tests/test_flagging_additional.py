
# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from tricolour.flagging import (flag_autos,
                                apply_static_mask,
                                flag_nans_and_zeros)


@pytest.fixture
def wsrt_ants():
    """ Westerbork antenna positions """
    return np.array([
        [3828763.10544699, 442449.10566454, 5064923.00777],
        [3828746.54957258, 442592.13950824, 5064923.00792],
        [3828729.99081359, 442735.17696417, 5064923.00829],
        [3828713.43109885, 442878.2118934, 5064923.00436],
        [3828696.86994428, 443021.24917264, 5064923.00397],
        [3828680.31391933, 443164.28596862, 5064923.00035],
        [3828663.75159173, 443307.32138056, 5064923.00204],
        [3828647.19342757, 443450.35604638, 5064923.0023],
        [3828630.63486201, 443593.39226634, 5064922.99755],
        [3828614.07606798, 443736.42941621, 5064923.],
        [3828609.94224429, 443772.19450029, 5064922.99868],
        [3828601.66208572, 443843.71178407, 5064922.99963],
        [3828460.92418735, 445059.52053929, 5064922.99071],
        [3828452.64716351, 445131.03744105, 5064922.98793]],
        dtype=np.float64)


@pytest.fixture
def baselines(wsrt_ants):
    return np.triu_indices(wsrt_ants.shape[0], 0)


@pytest.fixture
def squared_baseline_lengths(wsrt_ants, unique_baselines):
    _, ant1, ant2 = unique_baselines.T
    diff = (wsrt_ants[ant1] - wsrt_ants[ant2])
    return (diff**2).sum(axis=1)


@pytest.fixture
def unique_baselines(baselines):
    ubl = np.unique(np.stack(baselines, axis=1), axis=1)
    bl_range = np.arange(ubl.shape[0])[:, None]
    return np.concatenate([bl_range, ubl], axis=1)


def test_flag_nans_and_zeros(unique_baselines):
    ntime = 10
    nchan = 16
    ncorr = 4

    shape = (unique_baselines.shape[0], ncorr, ntime, nchan)

    zero_flags = np.zeros(shape, dtype=np.uint8)
    vis = np.random.random(shape) + 1j * np.random.random(shape)

    vis[4, 2, 4, 5] = 0
    vis[0, 1, 2, 7] = np.nan + np.nan * 1j

    out_flags = flag_nans_and_zeros(vis, zero_flags)

    zero_sel = vis == 0
    nan_sel = np.isnan(vis)

    # Flagged at zero and nan locations
    assert out_flags[4, 2, 4, 5] == 1
    assert out_flags[0, 1, 2, 7] == 1
    # Unflagged everywhere else
    assert np.all(out_flags[np.where(~(zero_sel | nan_sel))] == 0)

    flags = np.random.randint(0, 2, shape, dtype=np.uint8)

    # Test that zero, nan and original flags are or'd together
    out_flags = flag_nans_and_zeros(vis, flags)
    assert_array_equal(out_flags, flags | zero_sel | nan_sel)


def test_flag_autos(unique_baselines):
    ntime = 10
    nchan = 16
    ncorr = 4

    ubl = unique_baselines
    flags = np.ones((ubl.shape[0], ncorr, ntime, nchan), dtype=np.uint8)

    # Unflag auto-correlations
    ant1, ant2 = ubl[:, 1], ubl[:, 2]
    sel = ant1 == ant2
    flags[sel, :, :, :] = 0

    new_flags = flag_autos(flags, [ubl])

    # Auto-correlations should now all be flagged
    assert np.all(new_flags[sel, :, :, :] == 1)


def test_apply_static_mask(wsrt_ants, unique_baselines,
                           squared_baseline_lengths):
    ntime = 10
    nchan = 16
    ncorr = 4

    first_freq = .856e9
    last_freq = 2 * .856e9

    chan_freqs = np.linspace(first_freq, last_freq, nchan, dtype=np.float64)
    chan_widths = np.zeros_like(chan_freqs)
    chan_widths[0:-1] = np.diff(chan_freqs)
    chan_widths[-1] = chan_widths[0]

    mask_one = np.asarray([chan_freqs[2] + 128.,
                           chan_freqs[10]])[:, None]

    mask_two = np.asarray([chan_freqs[4] - 64,
                           chan_freqs[11] + 64,
                           chan_freqs[5] - 128])[:, None]

    ubl = unique_baselines
    flags = np.zeros((ubl.shape[0], ncorr, ntime, nchan), dtype=np.uint8)

    #  Logical or mode
    new_flags = apply_static_mask(flags, ubl, wsrt_ants,
                                  [mask_one],
                                  chan_freqs, chan_widths,
                                  accumulation_mode="or")

    # Check that first mask's flags are applied
    chan_sel = np.zeros(chan_freqs.shape[0], dtype=bool)
    chan_sel[[2, 10]] = True

    assert np.all(new_flags[:, :, :, chan_sel] == 1)
    assert np.all(new_flags[:, :, :, ~chan_sel] == 0)

    # Logical Or Mode
    new_flags = apply_static_mask(flags, ubl, wsrt_ants,
                                  [mask_one, mask_two],
                                  chan_freqs, chan_widths,
                                  accumulation_mode="or")

    # Check that both mask's flags have been applied
    chan_sel = np.zeros(chan_freqs.shape[0], dtype=bool)
    chan_sel[[2, 10, 4, 11, 5]] = True

    assert np.all(new_flags[:, :, :, chan_sel] == 1)
    assert np.all(new_flags[:, :, :, ~chan_sel] == 0)

    # Override mode
    new_flags = apply_static_mask(flags, ubl, wsrt_ants,
                                  [mask_one, mask_two],
                                  chan_freqs, chan_widths,
                                  accumulation_mode="override")

    # Check that only last mask's flags applied
    chan_sel = np.zeros(chan_freqs.shape[0], dtype=bool)
    chan_sel[[4, 11, 5]] = True

    assert np.all(new_flags[:, :, :, chan_sel] == 1)
    assert np.all(new_flags[:, :, :, ~chan_sel] == 0)

    # Test Baseline range selection
    min_range = 1e3
    max_range = 2e4
    uvrange = "%f~%f" % (min_range, max_range)

    # Flag using UV range
    new_flags = apply_static_mask(flags, ubl, wsrt_ants,
                                  [mask_one, mask_two],
                                  chan_freqs, chan_widths,
                                  accumulation_mode="or",
                                  uvrange=uvrange)

    # Check that both mask's flags have been applied
    chan_sel = np.zeros(chan_freqs.shape[0], dtype=bool)
    chan_sel[[2, 10, 4, 11, 5]] = True

    # Select baselines based on the uvrange
    sqrd_bl_len = 0.5 * squared_baseline_lengths
    bl_sel = np.logical_and(sqrd_bl_len > min_range**2,
                            sqrd_bl_len < max_range**2)

    # Everything inside the selection is flagged
    idx = np.ix_(bl_sel, np.arange(ncorr), np.arange(ntime), chan_sel)
    assert np.all(new_flags[idx] == 1)
    # Everything outside the selection is unflagged
    idx = np.ix_(~bl_sel, np.arange(ncorr), np.arange(ntime), ~chan_sel)
    assert np.all(new_flags[idx] == 0)
