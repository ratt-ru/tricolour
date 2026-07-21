"""A Radio Astronomy Flagging Software Suite"""

import os

# Ray's uv-run runtime-env hook (on by default) re-installs the project in worker
# processes via `uv` *without* the optional extras (msv2/meerkat/zarr), which drops
# transitive deps such as pyarrow and breaks partition deserialisation in workers.
# tricolour always runs a local Ray instance, so workers should simply inherit the
# driver's environment. This must be set before `ray` is first imported; setdefault
# leaves any explicit user/cluster override intact.
os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

from importlib.metadata import version

from donfig import Config

__version__ = version("tricolour")
__author__ = """Simon Perkins"""
__email__ = "sperkins@ska.ac.za"
__all__ = ["__version__", "__author__", "__email__", "config"]

config = Config("tricolour")
