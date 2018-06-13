from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint

import numpy as np
import pytest

from tricolour.stokes import (stokes_corr_map, STOKES_TYPES,
                                unpolarised_intensity)


@pytest.mark.parametrize('stokes', [
    ['YX', 'XX', 'XY', 'YY'],
    ['XX', 'XY', 'YX', 'YY'],
    ['RR', 'RL', 'LR', 'LL'],
    ['RL', 'RR', 'LL', 'LR']])
def test_stokes_corr_map(stokes):
    # Set up our stokes parameters in an interesting order
    stokes = map(STOKES_TYPES.__getitem__, stokes)
    vis = np.asarray([[[1+1j, 2+2j, 3+3j, 4+4j]]], np.complex128)

    stokes_map = stokes_corr_map(stokes)
    stokes_map = tuple(v for k, v in stokes_map.items() if k != 'I')

    upi = 0

    for c1, c2, a, s1, s2 in stokes_map:
        v = a*(s1*vis[0,0,c1] + s2*vis[0,0,c2])
        upi += v**2

    assert np.allclose(unpolarised_intensity(vis, stokes_map), np.sqrt(upi))
