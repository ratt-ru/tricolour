"""Tests for :mod:`tricolour.config`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tempfile import NamedTemporaryFile

from ruamel.yaml import YAML
import pytest

from tricolour.config import collect


def test_config_load():
    yaml = YAML(typ='safe')

    config = {
        'sum_threshold': {
            'outlier_nsigma': 4.5,
            'windows_time': [1, 2, 4, 8],
            'windows_freq': [1, 2, 4, 8],
            'background_reject': 2.0,
            'background_iterations': 1,
            'spike_width_time': 12.5,
            'spike_width_freq': 10.0,
            'time_extend': 3,
            'freq_extend': 3,
            'freq_chunks': 10,
            'average_freq': 1,
            'flag_all_time_frac': 0.6,
            'flag_all_freq_frac': 0.8,
            'rho': 1.3,
        },
    }

    with NamedTemporaryFile(mode='w') as f:
        yaml.dump(config, f)
        f.flush()

        assert config == collect([f.name])
