# -*- coding: utf-8 -*-

"""Top-level package for Tricolour."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import contextlib
from collections import OrderedDict
from functools import partial
import logging
import logging.handlers
from multiprocessing.pool import ThreadPool, cpu_count
import os
import time

import dask
import dask.array as da
from dask.diagnostics import (ProgressBar, Profiler,
                              ResourceProfiler,
                              CacheProfiler, visualize)
import numpy as np
import xarray as xr
from xarrayms import xds_from_ms, xds_from_table, xds_to_table


from tricolour.mask import collect_masks, load_mask
from tricolour.stokes import stokes_corr_map
from tricolour.dask_wrappers import (sum_threshold_flagger,
                                     polarised_intensity,
                                     uvcontsub_flagger,
                                     flag_autos,
                                     apply_static_mask)

from tricolour.packing import (unique_baselines,
                               pack_data,
                               unpack_data)

from tricolour.config import collect
from tricolour.util import casa_style_range
import tricolour.post_mortem_handler as post_mortem_handler

__author__ = """Simon Perkins"""
__email__ = 'sperkins@ska.ac.za'
__version__ = '0.2.0'

post_mortem_handler.enable_pdb_on_error()

try:
    import bokeh  # noqa
    can_profile = True
except ImportError:
    can_profile = False


def create_logger():
    """ Create a console logger """
    log = logging.getLogger("tricolour")
    cfmt = logging.Formatter(
        ('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)
    filehandler = logging.FileHandler("tricolour.log")
    filehandler.setFormatter(cfmt)
    log.addHandler(filehandler)
    log.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)

    log.addHandler(console)

    return log, filehandler, console, cfmt


def remove_log_handler(hndl):
    log.removeHandler(hndl)


def add_log_handler(hndl):
    log.addHandler(hndl)


# Create the log object
log, log_filehandler, log_console_handler, log_formatter = create_logger()

DEFAULT_CONFIG = os.path.join(os.path.split(
    __file__)[0], "conf", "default.yaml")


def print_info():
    RED = '\033[0;31m'
    WHITE = '\033[0;37m'
    BLUE = '\033[0;34m'
    RESET = '\033[0m'
    log.info("""
***********************************************************************************************************************************************
{0:s} ▄▄▄▄▄▄▄▄▄▄▄ {1:s} ▄▄▄▄▄▄▄▄▄▄▄{2:s}  ▄▄▄▄▄▄▄▄▄▄▄{3:s}  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄         ▄  ▄▄▄▄▄▄▄▄▄▄▄
{0:s}▐░░░░░░░░░░░▌{1:s}▐░░░░░░░░░░░▌{2:s}▐░░░░░░░░░░░▌{3:s}▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌          ▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌
{0:s} ▀▀▀▀█░█▀▀▀▀{1:s} ▐░█▀▀▀▀▀▀▀█░▌{2:s} ▀▀▀▀█░█▀▀▀▀{3:s} ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀█░▌
{0:s}     ▐░▌    {1:s} ▐░▌       ▐░▌{2:s}     ▐░▌    {3:s} ▐░▌          ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌
{0:s}     ▐░▌    {1:s} ▐░█▄▄▄▄▄▄▄█░▌{2:s}     ▐░▌    {3:s} ▐░▌          ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌
{0:s}     ▐░▌    {1:s} ▐░░░░░░░░░░░▌{2:s}     ▐░▌    {3:s} ▐░▌          ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌
{0:s}     ▐░▌    {1:s} ▐░█▀▀▀▀█░█▀▀ {2:s}     ▐░▌    {3:s} ▐░▌          ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░█▀▀▀▀█░█▀▀
{0:s}     ▐░▌    {1:s} ▐░▌     ▐░▌  {2:s}     ▐░▌    {3:s} ▐░▌          ▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌     ▐░▌
{0:s}     ▐░▌    {1:s} ▐░▌      ▐░▌ {2:s} ▄▄▄▄█░█▄▄▄▄{3:s} ▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░▌      ▐░▌
{0:s}     ▐░▌    {1:s} ▐░▌       ▐░▌{2:s}▐░░░░░░░░░░░▌{3:s}▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌
{0:s}      ▀     {1:s}  ▀         ▀ {2:s} ▀▀▀▀▀▀▀▀▀▀▀ {3:s} ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀

Viva la révolution!

A DASK distributed RFI flagger by Science Data Processing and Radio Astronomy Research Group
Copyright 2019 South African Radio Astronomy Observatory (SARAO, SKA-SA)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

***********************************************************************************************************************************************
""".format(BLUE, WHITE, RED, RESET))  # noqa make it Frenchy


def load_config(config_file):
    """
    Parameters
    ----------
    config_file : str

    Returns
    -------
    str
      Configuration file name
    dict
      Configuration
    """
    if config_file == DEFAULT_CONFIG:
        log.warn("User strategy not provided. Will now attempt to "
                 "find some defaults in the install paths")
    else:
        log.info("Loading custom user strategy {0:s}".format(config_file))

    config = collect(config_file)

    # Load configuration from file if present
    GD = dict([(k, config[k]) for k in config])
    GD = OrderedDict([(k, GD[k]) for k in sorted(
        GD.keys(), key=lambda x: GD[x].get("order", 99))])

    def _print_tree(tree, indent=0):
        for k in tree:
            if isinstance(tree[k], dict) or isinstance(tree[k], OrderedDict):
                log.info(('\t' * indent) + "Step {0:s} (type '{1:s}')".format(
                    k, tree[k].get("task", "Task is nameless")))
                _print_tree(tree[k], indent + 1)
            elif k == "order" or k == "task":
                continue
            else:
                log.info(('\t' * indent) +
                         "{0:s}:{1:s}".format(k.ljust(30), str(tree[k])))

    log.info("********************************")
    log.info("   BEGINNING OF CONFIGURATION   ")
    log.info("********************************")
    _print_tree(GD)
    log.info("********************************")
    log.info("      END OF CONFIGURATION      ")
    log.info("********************************")

    return GD


def create_parser():
    formatter = argparse.ArgumentDefaultsHelpFormatter
    p = argparse.ArgumentParser(formatter_class=formatter)
    p.add_argument("ms", help="Measurement Set")
    p.add_argument("-c", "--config", default=DEFAULT_CONFIG,
                   required=False, type=load_config,
                   help="YAML config file containing parameters for "
                   "the flagger in the 'sum_threshold' key.")
    p.add_argument("-if", "--ignore-flags", action="store_true")
    p.add_argument("-fs", "--flagging-strategy", default="standard",
                   choices=["standard", "polarisation"],
                   help="Flagging Strategy. "
                        "If 'standard' all correlations in the visibility "
                          "are flagged independently. "
                        "If 'polarisation' the polarised intensity "
                          "sqrt(Q^2 + U^2 + V^2) is "
                          "calculated and used to flag all correlations "
                          "in the visibility.")
    p.add_argument("-rc", "--row-chunks", type=int, default=10000,
                   help="Hint indicating the number of Measurement Set rows "
                   "to read in a single chunk. "
                   "Smaller and larger numbers will tend to "
                   "respectively decrease or increase both memory usage "
                   "and computational efficiency")
    p.add_argument("-bc", "--baseline-chunks", type=int, default=16,
                   help="Number of baselines in a window chunk")
    p.add_argument("-nw", "--nworkers", type=int, default=cpu_count() * 2,
                   help="Number of workers (threads) to use. "
                   "By default, set to twice the "
                   "number of logical CPUs on the system. "
                   "Many workers can also affect memory usage "
                   "on systems with many cores.")
    p.add_argument("-dm", "--dilate-masks", type=str, default=None,
                   help="Number of channels to dilate as int "
                        "or string with units")
    p.add_argument("-dc", "--data-column", type=str, default="DATA",
                   help="Name of visibility data column to flag")
    p.add_argument("-fn", "--field-names", type=str, action='append',
                   default=[],
                   help="Name(s) of fields to flag. Defaults to flagging all")
    p.add_argument("-sn", "--scan-numbers",
                   type=partial(casa_style_range, argparse=True),
                   default=[],
                   help="Scan numbers to flag (casa style range like 5~9)")
    p.add_argument("-dpm", "--disable-post-mortem", action="store_true",
                   help="Disable the default behaviour of starting "
                        "the Interactive Python Debugger upon an "
                        "unhandled exception. "
                        "This may be necessary for batch pipelining")
    p.add_argument("-wb", "--window-backend", choices=["numpy", "zarr-disk"],
                   default="numpy",
                   help="Visibility and flag data is re-ordered from a "
                        "MS row ordering into time-frequency windows "
                        "ordered by baseline. "
                        "For smaller problems, it may be possible to pack "
                        "a couple of scans worth of visibility data into "
                        "memory, but for larger problem sizes, it is "
                        "necessary to reorder the data on disk.")
    p.add_argument("-td", "--temporary-directory", default=None,
                   help="Directory Location of Temporary data")

    return p


def main():
    tic = time.time()

    print_info()

    args = create_parser().parse_args()

    if args.scan_numbers != []:
        log.info("Only considering scans '{0:s}' as "
                 "per user selection criterion"
                 .format(",".join(map(str, args.scan_numbers))))

    if args.flagging_strategy == "polarisation":
        log.info("Flagging based on quadrature polarized power")
    else:
        log.info("Flagging per correlation ('standard' mode)")

    if args.disable_post_mortem:
        log.warn("Disabling crash debugging with the "
                 "Interactive Python Debugger, as per user request")
        post_mortem_handler.disable_pdb_on_error()

    log.info("Flagging on the {0:s} column".format(args.data_column))
    data_column = args.data_column
    masked_channels = [load_mask(fn, dilate=args.dilate_masks)
                       for fn in collect_masks()]
    GD = args.config

    # Group datasets by these columns
    group_cols = ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]
    # Index datasets by these columns
    index_cols = ['TIME']

    # Reopen the datasets using the aggregated row ordering
    xds = list(xds_from_ms(args.ms,
                           columns=(data_column, "FLAG",
                                    "TIME", "ANTENNA1", "ANTENNA2"),
                           group_cols=group_cols,
                           index_cols=index_cols,
                           chunks={"row": args.row_chunks}))

    # Get datasets for DATA_DESCRIPTION and POLARIZATION,
    # partitioned by row
    data_desc_tab = "::".join((args.ms, "DATA_DESCRIPTION"))
    ddid_ds = list(xds_from_table(data_desc_tab, group_cols="__row__"))
    pol_tab = "::".join((args.ms, "POLARIZATION"))
    pol_ds = list(xds_from_table(pol_tab, group_cols="__row__"))
    ant_tab = "::".join((args.ms, "ANTENNA"))
    ads = list(xds_from_table(ant_tab))
    spw_tab = "::".join((args.ms, "SPECTRAL_WINDOW"))
    spw_ds = list(xds_from_table(spw_tab, group_cols="__row__"))
    antspos = ads[0].POSITION.values
    fld_tab = "::".join((args.ms, "FIELD"))
    field_ds = list(xds_from_table(fld_tab))
    fieldnames = field_ds[0].NAME.values

    if args.field_names != []:
        if not set(args.field_names) <= set(fieldnames):
            raise ValueError("One or more fields cannot be "
                             "found in dataset '{0:s}' "
                             "You specified {1:s}, but "
                             "only {2:s} are available".format(
                                args.ms,
                                ",".join(args.field_names),
                                ",".join(fieldnames)))

        field_dict = dict([(np.where(fieldnames == fn)[0][0], fn)
                           for fn in args.field_names])
    else:
        field_dict = dict([(findx, fn) for findx, fn in enumerate(fieldnames)])

    write_computes = []

    # Iterate through each dataset
    for ds in xds:
        if ds.FIELD_ID not in field_dict:
            continue

        if args.scan_numbers != [] and ds.SCAN_NUMBER not in args.scan_numbers:
            continue

        log.info("Adding field '{0:s}' scan {1:d} to "
                 "compute graph for processing"
                 .format(field_dict[ds.FIELD_ID], ds.SCAN_NUMBER))

        ddid = ddid_ds[ds.attrs['DATA_DESC_ID']]
        spw_info = spw_ds[ddid.SPECTRAL_WINDOW_ID.values]
        pol_info = pol_ds[ddid.POLARIZATION_ID.values]

        nrow, nchan, ncorr = getattr(ds, data_column).data.shape

        # Visibilities from the dataset
        vis = getattr(ds, data_column).data
        antenna1 = ds.ANTENNA1.data
        antenna2 = ds.ANTENNA2.data
        chan_freq = spw_info.CHAN_FREQ.values
        chan_width = spw_info.CHAN_WIDTH.values

        # Generate unflagged defaults if we should ignore existing flags
        # otherwise take flags from the dataset
        if args.ignore_flags is True:
            flags = da.full_like(vis, False, dtype=np.bool)
        else:
            flags = ds.FLAG.data

        # If we're flagging on polarised intensity,
        # we convert visibilities to polarised intensity
        # and any flagged correlation will flag the entire visibility
        if args.flagging_strategy == "polarisation":
            corr_type = pol_info.CORR_TYPE.data.compute().tolist()
            stokes_map = stokes_corr_map(corr_type)
            stokes_pol = tuple(v for k, v in stokes_map.items() if k != 'I')
            vis = polarised_intensity(vis, stokes_pol)
            flags = da.any(flags, axis=2, keepdims=True)
            xncorr = 1
        elif args.flagging_strategy == "standard":
            xncorr = ncorr
        else:
            raise ValueError("Invalid flagging strategy '%s'" %
                             args.flagging_strategy)

        ubl = unique_baselines(antenna1, antenna2)
        utime, time_inv = da.unique(ds.TIME.data, return_inverse=True)
        utime, ubl = dask.compute(utime, ubl)
        ubl = ubl.view(np.int32).reshape(-1, 2)
        ntime = utime.shape[0]
        ubl = da.from_array(ubl, chunks=(args.baseline_chunks, 2))
        nbl = ubl.shape[0]

        vis_windows, flag_windows = pack_data(time_inv, ubl,
                                              antenna1, antenna2,
                                              vis, flags, ntime,
                                              backend=args.window_backend,
                                              path=args.temporary_directory)

        original = flag_windows

        # Run the flagger
        for k in GD:
            if GD[k].get("task", "unnamed") == "sum_threshold":
                task_kwargs = GD[k].copy()
                task_kwargs.pop("task", None)
                task_kwargs.pop("order", None)
                flag_windows = sum_threshold_flagger(vis_windows, flag_windows,
                                                     **task_kwargs)
            elif GD[k].get("task", "unnamed") == "uvcontsub_flagger":
                task_kwargs = GD[k].copy()
                task_kwargs.pop("task", None)
                task_kwargs.pop("order", None)
                flag_windows = uvcontsub_flagger(vis_windows, flag_windows,
                                                 **task_kwargs)
            elif GD[k].get("task", "unnamed") == "flag_autos":
                task_kwargs = GD[k].copy()
                task_kwargs.pop("task", None)
                task_kwargs.pop("order", None)
                flag_windows = flag_autos(flag_windows, ubl,
                                          **task_kwargs)
            elif GD[k].get("task", "unnamed") == "combine_with_input_flags":
                flag_windows = da.logical_or(flag_windows, original)
            elif GD[k].get("task", "unnamed") == "unflag":
                flag_windows = da.zeros_like(flag_windows)
            elif GD[k].get("task", "unnamed") == "apply_static_mask":
                task_kwargs = GD[k].copy()
                task_kwargs.pop("task", None)
                task_kwargs.pop("order", None)
                flag_windows = apply_static_mask(flag_windows,
                                                 ubl,
                                                 antspos,
                                                 masked_channels,
                                                 chan_freq,
                                                 chan_width,
                                                 **task_kwargs)

            else:
                raise ValueError("Task '{0:s}' does not name a valid task"
                                 .format(GD[k].get("task", "unnamed")))

        unpacked_flags = unpack_data(antenna1, antenna2, time_inv,
                                     ubl, flag_windows)

        # Polarised flagging, broadcast the single correlation
        # back to the full correlation range (all flagged)
        if args.flagging_strategy == "polarisation":
            unpacked_flags = da.broadcast_to(unpacked_flags,
                                             (nrow, nchan, ncorr))

        # Create new dataset containing new flags
        xarray_flags = xr.DataArray(unpacked_flags, dims=ds.FLAG.dims)
        new_ds = ds.assign(FLAG=xarray_flags)

        # Write back to original dataset
        writes = xds_to_table(new_ds, args.ms, "FLAG")
        write_computes.append(writes)

    # Create dask contexts
    profilers = ([Profiler(), CacheProfiler(), ResourceProfiler()]
                 if can_profile else [])
    contexts = [ProgressBar()] + profilers

    pool = ThreadPool(args.nworkers)

    with contextlib.nested(*contexts), dask.config.set(pool=pool):
        dask.compute(write_computes)

    if can_profile:
        visualize(profilers)

    toc = time.time()

    elapsed = toc - tic

    log.info("Data flagged successfully in "
             "{0:02.0f}h{1:02.0f}m{2:02.0f}s"
             .format((elapsed // 60) // 60,
                     (elapsed // 60) % 60,
                     elapsed % 60))
