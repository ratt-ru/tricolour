from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numba
import numpy as np

"""
Enumeration of stokes, linear and circular correlations used in
Measurement Set 2.0 as per Stokes.h in casacore:
https://casacore.github.io/casacore/classcasacore_1_1Stokes.html
the rest are left unimplemented.
"""
STOKES_TYPES = {
    'I': 1,     # stokes unpolarised
    'Q': 2,     # stokes linear
    'U': 3,     # stokes linear
    'V': 4,     # stokes circular
    'RR': 5,    # right-right circular
    'RL': 6,    # right-left cross-circular
    'LR': 7,    # left-right cross-circular
    'LL': 8,    # left-left circular
    'XX': 9,    # parallel linear
    'XY': 10,   # XY cross linear
    'YX': 11,   # YX cross linear
    'YY': 12    # parallel linear
}

INV_STOKES_TYPES = {v: k for k, v in STOKES_TYPES.items()}

# Correlation dependencies required for reconstructing stokes values
# (corr1, corr2, a, s1, s2). stokes = a*(s1*corr1 + s2*corr2)
stokes_deps = {
    'I': [('XX', 'YY', 0.5 + 0.0j, 1, 1),    ('RR', 'LL', 0.5 + 0.0j, 1, 1)],
    'Q': [('XX', 'YY', 0.5 + 0.0j, 1, -1),   ('RL', 'LR', 0.5 + 0.0j, 1, 1)],
    'U': [('XY', 'YX', 0.0 + 0.5j, 1, 1),    ('RL', 'LR', 0.0 - 0.5j, 1, -1)],
    'V': [('XY', 'YX', 0.0 - 0.5j, 1, -1),   ('RR', 'LL', 0.0 + 0.5j, 1, -1)]
}

# Convert to numeric stokes types
stokes_deps = { k: [(STOKES_TYPES[c1], STOKES_TYPES[c2], a, s1, s2)
                    for (c1, c2, a, s1, s2) in deps]
                    for k, deps in stokes_deps.items()}


def stokes_corr_map(corr_types):
    """
    Produces a map describing how to combine visibility correlations
    in order to form a stokes parameter.

    Parameters
    ----------
    corr_ids : list of integers
        List of correlation types as defined in `casacore <stokes>_`_

    .. _stokes: https://casacore.github.io/casacore/classcasacore_1_1Stokes.html


    Returns
    -------
    dict
        Correlation map with schema :code:`{ stokes: (c1, c2, a, s1, s2)}`

        .. code-block:: python

            stokes = a*(s1*vis[:,:,c1] + s2*vis[:,:,c2])
    """
    corr_type_set = set(corr_types)
    corr_maps = {}

    for stokes, deps in stokes_deps.items():
        for (corr1, corr2, alpha, sign1, sign2) in deps:
            # If both correlations are available as dependencies
            # we can generate this stokes parameter
            if len(corr_type_set.intersection((corr1, corr2))) == 2:
                c1 = corr_types.index(corr1)
                c2 = corr_types.index(corr2)
                corr_maps[stokes] = (c1, c2, alpha, sign1, sign2)

    return corr_maps


@numba.jit(nopython=True, nogil=True, cache=True)
def unpolarised_intensity(vis, corr_map):
    """
    Generate unpolarised intensity from visibilities
    """
    out_vis = np.zeros(vis.shape[:2] + (1,), vis.dtype)

    for r in range(vis.shape[0]):
        for f in range(vis.shape[1]):
            for (c1, c2, a, s1, s2) in corr_map:
                value = a*(s1*vis[r,f,c1] + s2*vis[r,f,c2])
                out_vis[r,f,0] += value**2

            out_vis[r,f,0] = np.sqrt(out_vis[r,f,0])

    return out_vis


