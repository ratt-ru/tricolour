# -*- coding: utf-8 -*-

"""Top-level package for Tricolour."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import contextlib
from collections import OrderedDict
from functools import wraps
import logging
import logging.handlers
from multiprocessing.pool import ThreadPool, cpu_count
import re
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
                                     unpolarised_intensity,
                                     check_baseline_ordering,
                                     uvcontsub_flagger,
                                     flag_autos,
                                     apply_static_mask)
from tricolour.config import collect
from tricolour.util import aggregate_chunks, casa_style_range
import tricolour.post_mortem_handler

__author__ = """Simon Perkins"""
__email__ = 'sperkins@ska.ac.za'
__version__ = '0.2.0'

post_mortem_handler.enable_pdb_on_error()

try:
    import bokeh
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
    p.add_argument("-sn", "--scan-numbers", type=casa_style_range, default=[],
                   help="Scan numbers to flag (casa style range like 5~9)")
    p.add_argument("-dpm", "--disable-post-mortem", action="store_true",
                   help="Disable the default behaviour of starting "
                        "the Interactive Python Debugger upon an "
                        "unhandled exception. "
                        "This may be necessary for batch pipelining")
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

    log.info("Will process {0:s} column".format(args.data_column))
    data_column = args.data_column
    masked_channels = [load_mask(fn, dilate=args.dilate_masks)
                       for fn in collect_masks()]
    GD = args.config

    # Group datasets by these columns
    group_cols = ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]
    # Index datasets by these columns
    index_cols = ['TIME']

    xds = list(xds_from_ms(args.ms,
                           columns=("TIME", "ANTENNA1", "ANTENNA2"),
                           group_cols=group_cols,
                           index_cols=index_cols,
                           chunks={"row": 1e9}))

    # Find the unique times and their row counts
    utime_counts = [da.unique(ds.TIME.data, return_counts=True)
                    for i, ds in enumerate(xds)]
    utime_counts = dask.compute(utime_counts)[0]
    scan_rows = tuple(counts for _, counts in utime_counts)
    scan_chunks = [da.from_array(rc, chunks=ut.size)
                   for ut, rc in utime_counts]
    assert len(scan_rows) == len(xds)

    # Ensure that baseline ordering is consistent per timestep
    dask.compute([check_baseline_ordering(ds.ANTENNA1.data,
                                          ds.ANTENNA2.data,
                                          chunks,
                                          g=g)
                  for g, (ds, chunks)
                  in enumerate(zip(xds, scan_chunks))])

    # Determine how many rows we should handle at once.
    # The row chunk sizes in scan rows already correspond to
    # single timesteps, so choose the maximum supplied either by the user
    # or discovered intrinsically in the data itself
    row_chunks = max(args.row_chunks, max(c.max() for c in scan_rows))

    # Aggregate time and rows together into chunks that
    # are at least len(counts) in time or counts in rows,
    # whichever gets reached first
    agg_time, agg_row = zip(*[aggregate_chunks(((1,) * len(counts), counts),
                                               (len(counts), row_chunks))
                              for utime, counts in utime_counts])

    # Reopen the datasets using the aggregated row ordering
    xds = list(xds_from_ms(args.ms,
                           columns=(data_column, "FLAG",
                                    "ANTENNA1", "ANTENNA2"),
                           group_cols=group_cols,
                           index_cols=index_cols,
                           chunks=[{"row": r} for r in agg_row]))

    # Get datasets for DATA_DESCRIPTION and POLARIZATION,
    # partitioned by row
    data_desc_tab = "::".join((args.ms, "DATA_DESCRIPTION"))
    ddid_ds = list(xds_from_table(data_desc_tab, group_cols="__row__"))
    pol_tab = "::".join((args.ms, "POLARIZATION"))
    pds = list(xds_from_table(pol_tab, group_cols="__row__"))
    ant_tab = "::".join((args.ms, "ANTENNA"))
    ads = list(xds_from_table(ant_tab))
    spw_tab = "::".join((args.ms, "SPECTRAL_WINDOW"))
    sds = list(xds_from_table(spw_tab, group_cols="__row__"))
    antspos = ads[0].POSITION.values
    fld_tab = "::".join((args.ms, "FIELD"))
    fds = list(xds_from_table(fld_tab))
    fieldnames = fds[0].NAME.values

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

    ddid = ddid_ds[ds.attrs['DATA_DESC_ID']].drop('table_row')
    spw_info = sds[ddid.SPECTRAL_WINDOW_ID.values].drop('table_row')

    # Add data from the POLARIZATION table into the dataset
    def _add_pol_data(ds):
        ddid = ddid_ds[ds.attrs['DATA_DESC_ID']].drop('table_row')
        pol = pds[ddid.POLARIZATION_ID.values].drop('table_row')
        return ds.assign(CORR_TYPE=pol.CORR_TYPE,
                         CORR_PRODUCT=pol.CORR_PRODUCT)

    xds = [_add_pol_data(ds) for ds in xds]

    write_computes = []

    # Iterate through each dataset
    for ds, agg_time_counts, row_counts in zip(xds, agg_time, scan_rows):
        if ds.FIELD_ID not in field_dict:
            continue

        if args.scan_numbers != []:
            if ds.SCAN_NUMBER not in args.scan_numbers:
                continue

        log.info("Adding field '{0:s}' scan {1:d} to "
                 "compute graph for processing"
                 .format(field_dict[ds.FIELD_ID], ds.SCAN_NUMBER))

        row_counts = np.asarray(row_counts)
        ntime, nbl = row_counts.size, row_counts[0]
        nrow, nchan, ncorr = getattr(ds, data_column).data.shape
        chunks = da.from_array(row_counts, chunks=(agg_time_counts,))

        # Visibilities from the dataset
        vis = getattr(ds, data_column).data
        a1 = ds.ANTENNA1.data
        a2 = ds.ANTENNA2.data
        chan_freq = spw_info.CHAN_FREQ.values
        chan_width = spw_info.CHAN_WIDTH.values

        # Generate unflagged defaults if we should ignore existing flags
        # otherwise take flags from the dataset
        if args.ignore_flags == True:  # noqa
            flags = da.full_like(vis, False, dtype=np.bool)
        else:
            flags = ds.FLAG.data

        # Reorder vis and flags into katdal-like format
        # (ntime, nchan, ncorrprod). Chunk the corrprod
        # dimension into groups of 64 baselines
        vis = vis.reshape(ntime, nbl, nchan, ncorr)
        flags = flags.reshape(ntime, nbl, nchan, ncorr)

        # Rechunk on baseline dimension
        vis = vis.rechunk({1: 64})
        flags = flags.rechunk({1: 64})

        # If we're flagging on polarised intensity,
        # we convert visibilities to polarised intensity
        # and any flagged correlation will flag the entire visibility
        if args.flagging_strategy == "polarisation":
            corr_type = ds.CORR_TYPE.data.compute().tolist()
            stokes_map = stokes_corr_map(corr_type)
            stokes_pol = tuple(v for k, v in stokes_map.items() if k != 'I')
            vis = polarised_intensity(vis, stokes_pol)
            flags = da.any(flags, axis=3, keepdims=True)
            xncorr = 1
        elif args.flagging_strategy == "standard":
            xncorr = ncorr
        else:
            raise ValueError("Invalid flagging Strategy %s" %
                             args.flagging_strategy)

        a1 = a1.repeat(xncorr).reshape(ntime, nbl * xncorr)
        a2 = a2.repeat(xncorr).reshape(ntime, nbl * xncorr)
        a1 = a1.rechunk({1: 64})
        a2 = a2.rechunk({1: 64})

        vis = vis.transpose(0, 2, 1, 3)
        vis = vis.reshape((ntime, nchan, nbl * xncorr))
        flags = flags.transpose(0, 2, 1, 3)
        flags = flags.reshape((ntime, nchan, nbl * xncorr))
        # Run the flagger
        original = flags.copy()
        new_flags = flags

        for k in GD:
            if GD[k].get("task", "unnamed") == "sum_threshold":
                task_kwargs = GD[k].copy()
                task_kwargs.pop("task", None)
                task_kwargs.pop("order", None)
                new_flags = sum_threshold_flagger(vis, new_flags,
                                                  **task_kwargs)
            elif GD[k].get("task", "unnamed") == "uvcontsub_flagger":
                task_kwargs = GD[k].copy()
                task_kwargs.pop("task", None)
                task_kwargs.pop("order", None)
                new_flags = uvcontsub_flagger(vis, new_flags, **task_kwargs)
            elif GD[k].get("task", "unnamed") == "flag_autos":
                task_kwargs = GD[k].copy()
                task_kwargs.pop("task", None)
                task_kwargs.pop("order", None)

                new_flags = flag_autos(new_flags, a1, a2, **task_kwargs)
            elif GD[k].get("task", "unnamed") == "combine_with_input_flags":
                new_flags = da.logical_or(new_flags, original)
            elif GD[k].get("task", "unnamed") == "unflag":
                new_flags = da.zeros_like(new_flags)
            elif GD[k].get("task", "unnamed") == "apply_static_mask":
                task_kwargs = GD[k].copy()
                task_kwargs.pop("task", None)
                task_kwargs.pop("order", None)
                new_flags = apply_static_mask(new_flags,
                                              a1,
                                              a2,
                                              antspos,
                                              masked_channels,
                                              chan_freq,
                                              chan_width,
                                              xncorr,
                                              **task_kwargs)

            else:
                raise ValueError("Task '{0:s}' does not name a valid task"
                                 .format(GD[k].get("task", "unnamed")))

        # Reorder flags from katdal-like format back to the MS ordering
        # (ntime*nbl, nchan, ncorr)
        new_flags = new_flags.reshape((ntime, nchan, nbl, xncorr))
        new_flags = new_flags.transpose(0, 2, 1, 3)
        new_flags = new_flags.reshape((-1, nchan, xncorr))

        # Polarised flagging, broadcast the single correlation
        # back to the full correlation range (all flagged)
        if args.flagging_strategy == "polarisation":
            new_flags = da.broadcast_to(new_flags, (ntime * nbl, nchan, ncorr))

        # Make a single chunk for the write back to disk
        new_flags = new_flags.rechunk(new_flags.shape)
        new_ms = ds.assign(FLAG=xr.DataArray(new_flags, dims=ds.FLAG.dims))

        writes = xds_to_table(new_ms, args.ms, "FLAG")
        write_computes.append(writes)

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
