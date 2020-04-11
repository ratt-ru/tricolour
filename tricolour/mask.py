# -*- coding: utf-8 -*-

import logging
import os
import re

from pkg_resources import resource_filename
import numpy as np

from scipy.ndimage import binary_dilation

from tricolour import config

log = logging.getLogger(__name__)

# Look in default configuration paths, as well as the data directory
_DEFAULT_PATHS = config.paths + [resource_filename('tricolour', 'data')]


def dilate_mask(mask_chans, mask_flags, dilate):
    """
    Dilates mask array by number of channels indicated in dilate

    Parameters
    ----------
    dilate: int
        Dilation channel width in either Hz or number of channels
    mask_chans: :class:`numpy.ndarray`
        centre frequencies of mask
    mask_flags: :class:`numpy.ndarray`
        boolean array of mask

    Returns
    -------
        dilated_mask: boolean array of shape mask_flags
    """
    try:
        dilate_width = int(dilate)
    except ValueError:
        value, units = re.match(r"([\d.]+)([a-zA-Z]+)", dilate, re.I).groups()
        if units == 'GHz':
            value = float(value) * 1e9
        elif units == 'MHz':
            value = float(value) * 1e6
        elif units == 'kHz':
            value = float(value) * 1e3
        elif units == 'Hz':
            value = float(value)
        else:
            raise ValueError(
                'Unrecognised units for --dilate value::  %s' % units)

        chan_width = mask_chans[1] - mask_chans[0]
        dilate_width = int(value / chan_width) + 1
    dstruct = np.array([True, True, True])
    return binary_dilation(mask_flags, dstruct, iterations=dilate_width)


def load_mask(filename, dilate):
    # Load mask
    mask = np.load(filename)

    if mask.dtype[0] != np.bool or mask.dtype[1] != np.float64:
        raise ValueError("Mask %s is not a valid static mask "
                         "with labelled channel axis "
                         "[dtype == (bool, float64)]" % filename)

    mask_chans = mask["chans"][1]
    mask_flags = mask["mask"][0]

    # Dilate mask
    if dilate:
        mask_flags = dilate_mask(mask_chans, mask_flags, dilate)

    masked_channels = mask_chans[np.argwhere(mask_flags)]

    log.info("Loaded mask {0:s} {1:s} with {2:.2f}% "
             "flagged bandwidth between {3:.3f} "
             "and {4:.3f} GHz".format(
                 filename,
                 "(dilated)" if dilate else "(non-dilated)",
                 100.0 * masked_channels.size / mask_chans.size,
                 np.min(mask_chans) / 1.0e9,
                 np.max(mask_chans) / 1.0e9))

    return masked_channels


def collect_masks(filename="", paths=_DEFAULT_PATHS):
    """
    Collect masks from paths

    Parameters
    ----------
    paths : list of str
        A list of paths to search for masks

    Returns
    -------
    masks: list of str
        Discovered masks
    """
    if filename == "":

        file_paths = []
        file_exts = ('.staticmask', '.npy')
        log.info("Looking for static masks...")
        for path in paths:
            log.info("Searching {0:s}".format(path))
            if os.path.exists(path):
                if os.path.isdir(path):
                    file_paths.extend(sorted([
                        os.path.join(path, p) for p in os.listdir(path)
                        if os.path.splitext(p)[1].lower() in file_exts
                    ]))
                else:
                    file_paths.append(path)

        for fp in file_paths:
            log.info("Found static mask file {0:s}".format(fp))
    else:
        file_paths = [filename]

    return file_paths
