# -*- coding: utf-8 -*-

"""Top-level package for Tricolour."""

from donfig import Config

import pkg_resources
try:
    __version__ = pkg_resources.require("tricolour")[0].version
except pkg_resources.DistributionNotFound:
    __version__ = "dev"

__author__ = """Simon Perkins"""
__email__ = 'sperkins@ska.ac.za'

config = Config("tricolour")
