from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from ruamel.yaml import YAML
from toolz import merge
import tricolour
paths = [
    '/etc/tricolour',
    os.path.join(sys.prefix, 'etc', 'tricolour'),
    os.path.join(os.path.split(tricolour.__file__)[0], "conf"),
    os.path.join(os.path.expanduser('~'), '.config', 'tricolour'),
    os.path.join(os.path.expanduser('~'), '.tricolour')
]


if 'TRICOLOUR_CONFIG' in os.environ:
    paths.append(os.environ['TRICOLOUR_CONFIG'])


def collect(filename="", paths=paths):
    """
    Collect yaml configuration from paths

    Parameters
    ----------
    paths : list of str
        A list of paths to search for yaml configuration files

    Returns
    -------
    config: dict
        Dictionary containing configuration
    """
    yaml = YAML(typ='safe')
    configs = []

    if filename == "":

        file_paths = []
        file_exts = ('.json', '.yaml', '.yml')
        for path in paths:
            tricolour.log.info("Searching {0:s}".format(path))
            if os.path.exists(path):
                if os.path.isdir(path):
                    file_paths.extend(sorted([
                        os.path.join(path, p)
                        for p in os.listdir(path)
                        if os.path.splitext(p)[1].lower() in file_exts
                    ]))
                else:
                    file_paths.append(path)
        for fp in file_paths:
            tricolour.log.info("Found configuration file {0:s}".format(fp))
    else:
        file_paths = [filename]
    # Parse yaml files
    for path in file_paths:
        with open(path) as f:
            data = yaml.load(f.read()) or {}
            configs.append(data)

    return merge(*configs)
