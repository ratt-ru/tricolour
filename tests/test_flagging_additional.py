
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

from tricolour.flagging import flag_autos


def test_flag_autos():
    ntime = 10
    nchan = 16
    ncorr = 4

    ant1, ant2 = np.triu_indices(7, 0)
    ubl = np.unique(np.stack([ant1, ant2], axis=1), axis=1)
    bl_range = np.arange(ubl.shape[0])[:, None]
    ubl = np.concatenate([bl_range, ubl], axis=1)

    flags = np.random.randint(0, 2, (ntime, nchan, ubl.shape[0], ncorr))

    new_flags = flag_autos(flags, [ubl])

    ant1, ant2 = ubl[:, 1], ubl[:, 2]
    assert np.any(new_flags != flags)
    assert np.all(new_flags[:, :, ant1 == ant2, :] == 1)
