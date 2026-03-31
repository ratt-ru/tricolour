# -*- coding: utf-8 -*-

"""Top-level package for Tricolour."""

from donfig import Config
from importlib.metadata import version
__version__ = version('tricolour')
__author__ = """Simon Perkins"""
__email__ = 'sperkins@ska.ac.za'

config = Config("tricolour")
