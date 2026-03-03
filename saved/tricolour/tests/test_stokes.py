# -*- coding: utf-8 -*-

import numpy as np
import pytest

from tricolour.stokes import (stokes_corr_map,
                              STOKES_TYPES,
                              polarised_intensity,
                              unpolarised_intensity)


@pytest.mark.parametrize('stokes', [
    ['YX', 'XX', 'XY', 'YY'],
    ['XX', 'XY', 'YX', 'YY'],
    ['RR', 'RL', 'LR', 'LL'],
    ['RL', 'RR', 'LL', 'LR']])
def test_unpolarised_intensity(stokes):
    # Set up our stokes parameters in an interesting order
    stokes = list(map(STOKES_TYPES.__getitem__, stokes))
    vis = np.asarray([[[1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j]]], np.complex128)

    stokes_map = stokes_corr_map(stokes)

    # Unpolarised stokes mappings
    stokes_unpol = tuple(v for k, v in stokes_map.items() if k == 'I')
    unpol = 0

    for c1, c2, a, s1, s2 in stokes_unpol:
        v = a * (s1 * vis[0, 0, c1] + s2 * vis[0, 0, c2])
        unpol += np.abs(v)

    # Polarised stokes mappings
    stokes_pol = tuple(v for k, v in stokes_map.items() if k != 'I')
    pol = 0

    for c1, c2, a, s1, s2 in stokes_pol:
        v = a * (s1 * vis[0, 0, c1] + s2 * vis[0, 0, c2])
        pol += np.abs(v)**2

    upi = unpol - np.sqrt(pol)
    val = unpolarised_intensity(vis, stokes_unpol, stokes_pol)
    assert np.allclose(val, upi)


@pytest.mark.parametrize('stokes', [
    ['YX', 'XX', 'XY', 'YY'],
    ['XX', 'XY', 'YX', 'YY'],
    ['RR', 'RL', 'LR', 'LL'],
    ['RL', 'RR', 'LL', 'LR']])
def test_polarised_intensity(stokes):
    # Set up our stokes parameters in an interesting order
    stokes = list(map(STOKES_TYPES.__getitem__, stokes))
    vis = np.asarray([[[1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j]]], np.complex128)

    stokes_map = stokes_corr_map(stokes)

    # Polarised stokes mappings
    stokes_pol = tuple(v for k, v in stokes_map.items() if k != 'I')
    pol = 0

    for c1, c2, a, s1, s2 in stokes_pol:
        v = a * (s1 * vis[0, 0, c1] + s2 * vis[0, 0, c2])
        pol += np.abs(v)**2  # imaginary contains only noise

    pi = np.sqrt(pol)
    val = polarised_intensity(vis, stokes_pol)
    assert np.allclose(val, pi)
