# -*- coding: utf-8 -*-

"""Top-level package for Tricolour."""

import logging
import logging.handlers

from flagging import sum_threshold_flagger

__author__ = """Simon Perkins"""
__email__ = 'sperkins@ska.ac.za'
__version__ = '0.1.0'


def _create_logger():
    """ Setup logging configuration """

    # Console formatter, mention name
    cfmt = logging.Formatter(('%(name)s - %(levelname)s - %(message)s'))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(cfmt)

    # Create the logger, adding the console handler
    log = logging.getLogger(__name__)
    log.addHandler(ch)


_create_logger()
