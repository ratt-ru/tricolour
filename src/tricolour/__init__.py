"""A Radio Astronomy Flagging Software Suite"""

from donfig import Config
from importlib.metadata import version
__version__ = version('tricolour')
__author__ = """Simon Perkins"""
__email__ = 'sperkins@ska.ac.za'
config = Config("tricolour")
__all__ = ["__version__","__author__","__email__","config"]
