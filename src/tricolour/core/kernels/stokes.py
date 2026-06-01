# -*- coding: utf-8 -*-

import numba
import numpy as np

"""
Enumeration of stokes, linear and circular correlations used in
Measurement Set 2.0 as per Stokes.h in casacore:
https://casacore.github.io/casacore/classcasacore_1_1Stokes.html
the rest are left unimplemented.
"""
STOKES_TYPES = {
  "I": 1,  # stokes unpolarised
  "Q": 2,  # stokes linear
  "U": 3,  # stokes linear
  "V": 4,  # stokes circular
  "RR": 5,  # right-right circular
  "RL": 6,  # right-left cross-circular
  "LR": 7,  # left-right cross-circular
  "LL": 8,  # left-left circular
  "XX": 9,  # parallel linear
  "XY": 10,  # XY cross linear
  "YX": 11,  # YX cross linear
  "YY": 12,  # parallel linear
}

# Correlation dependencies required for reconstructing stokes values
# (corr1, corr2, a, s1, s2). stokes = a*(s1*corr1 + s2*corr2)
stokes_deps = {
  "I": [("XX", "YY", 0.5 + 0.0j, 1, 1), ("RR", "LL", 0.5 + 0.0j, 1, 1)],
  "Q": [("XX", "YY", 0.5 + 0.0j, 1, -1), ("RL", "LR", 0.5 + 0.0j, 1, 1)],
  "U": [("XY", "YX", 0.5 + 0.0j, 1, 1), ("RL", "LR", 0.0 - 0.5j, 1, -1)],
  "V": [("XY", "YX", 0.0 - 0.5j, 1, -1), ("RR", "LL", 0.5 + 0.0j, 1, -1)],
}

# Convert to numeric stokes types
stokes_deps = {
  k: [(STOKES_TYPES[c1], STOKES_TYPES[c2], a, s1, s2) for (c1, c2, a, s1, s2) in deps]
  for k, deps in stokes_deps.items()
}


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
  """  # noqa
  corr_type_set = set(corr_types)
  corr_maps = {}

  for stokes, deps in stokes_deps.items():
    for corr1, corr2, alpha, sign1, sign2 in deps:
      # If both correlations are available as dependencies
      # we can generate this stokes parameter
      if len(corr_type_set.intersection((corr1, corr2))) == 2:
        c1 = corr_types.index(corr1)
        c2 = corr_types.index(corr2)
        corr_maps[stokes] = (c1, c2, alpha, sign1, sign2)

  return corr_maps


@numba.njit(nogil=False, cache=True)
def polarised_intensity(vis, stokes_pol):
  r"""
  Derives the polarised intensity from visibilities
  and tuples describing how to derive stokes parameters
  from visibility correlations.

  .. math::

      \sqrt(Q^2 + U^2 + V^2)

  ``stokes_pol`` can be derived from :func:`stokes_corr_map`.

  Parameters
  ----------
  vis: :class:`numpy.ndarray`
      Visibilities of shape :code:`(baseline, corr, time, freq)`
  stokes_pol: tuple
      Tuple with schema :code:`(c1,c2,a,s1,s2)` describing
      how to derive polarised stokes parameters (Q,U,V):

          1. ``c1`` -- First correlation index
          2. ``c2`` -- Second correlation index
          3. ``a``  -- alpha, multiplier
          4. ``s1`` -- First correlation sign
          5. ``s2`` -- Second correlation sign

  Returns
  -------
  :class:`numpy.ndarray`
      Polarised intensities of shape :code:`(baseline, 1, time, freq)`.
  """

  # Only one output correlation -- polarised intensity
  out_vis = np.empty((vis.shape[0], 1, vis.shape[2], vis.shape[3]), vis.dtype)

  for bl in range(vis.shape[0]):
    for t in range(vis.shape[2]):
      for f in range(vis.shape[3]):
        # Polarised intensity (Q,U,V)
        pol = 0

        for c1, c2, a, s1, s2 in stokes_pol:
          value = a * (s1 * vis[bl, c1, t, f] + s2 * vis[bl, c2, t, f])
          # uncalibrated data may have a substantial amount of power in
          # the imaginary
          pol += np.abs(value) ** 2
          # use absolute to be certain

        # sqrt(Q^2 + U^2 + V^2)
        out_vis[bl, 0, t, f] = np.sqrt(pol)

  return out_vis
