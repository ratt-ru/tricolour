# -*- coding: utf-8 -*-

"""Top-level package for Tricolour."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import logging
import logging.handlers

__author__ = """Simon Perkins"""
__email__ = 'sperkins@ska.ac.za'
__version__ = '0.2.0'
import tricolour.post_mortem_handler
post_mortem_handler.enable_pdb_on_error()
import argparse
import contextlib
import os
from functools import wraps
import logging
from multiprocessing.pool import ThreadPool, cpu_count
from collections import OrderedDict
import dask
import dask.array as da
from dask.diagnostics import (ProgressBar, Profiler,
                              ResourceProfiler,
                              CacheProfiler, visualize)

import numpy as np
import xarray as xr

from xarrayms import xds_from_ms, xds_from_table, xds_to_table

try:
    import bokeh
    can_profile = True
except ImportError:
    can_profile = False


from tricolour.util import aggregate_chunks
from tricolour.config import collect
from tricolour.dask_wrappers import (sum_threshold_flagger,
                                     polarised_intensity,
                                     unpolarised_intensity,
                                     check_baseline_ordering,
                                     uvcontsub_flagger,
                                     apply_static_mask)
from tricolour.stokes import stokes_corr_map
from tricolour.mask import collect_masks, load_mask

TRICOLOUR_LOG = "tricolor.log"
def create_logger():
    """ Create a console logger """
    log = logging.getLogger("tricolour")
    cfmt = logging.Formatter(('%(name)s - %(asctime)s %(levelname)s - %(message)s'))
    log.setLevel(logging.DEBUG)
    filehandler = logging.FileHandler(TRICOLOUR_LOG)
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

global GD
GD = {}
DEFAULT_CONFIG = os.path.join(os.path.split(__file__)[0], "conf", "default.yaml")

def print_info():
    RED='\033[0;31m'
    WHITE='\033[0;37m'
    BLUE='\033[0;34m'
    RESET='\033[0m'
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
""".format(BLUE, WHITE, RED, RESET)) # make it Frenchy

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
    if config_file == "":
        log.warn("User strategy not provided. Will now attempt to find some defaults in the install paths")
    else:
        log.info("Loading in user customized strategy {0:s}".format(config_file))
    config = collect(config_file)
    # Load configuration from file if present
    global GD
    GD = dict([(k, config[k]) for k in config])
    GD = OrderedDict([(k, GD[k]) for k in sorted(GD.keys(), key=lambda x: GD[x].get("order", 99))])
    def _print_tree(tree, indent=0):
        for k in tree:
            if isinstance(tree[k], dict) or isinstance(tree[k], OrderedDict):
                log.info(('\t' * indent) + "Step {0:s} (type '{1:s}')".format(k, tree[k].get("task", "Task is nameless")))
                _print_tree(tree[k], indent + 1)
            elif k == "order" or k == "task":
                continue
            else:
                log.info(('\t' * indent) + "{0:s}:{1:s}".format(k.ljust(30), str(tree[k])))
    log.info("********************************")
    log.info("   BEGINNING OF CONFIGURATION   ")
    log.info("********************************")
    try:
        _print_tree(GD)
    except Exception, e:
        import ipdb
        ipdb.set_trace()
    log.info("********************************")
    log.info("      END OF CONFIGURATION      ")
    log.info("********************************")

    return config_file, GD


def create_parser():
    formatter = argparse.ArgumentDefaultsHelpFormatter
    p = argparse.ArgumentParser(formatter_class=formatter)
    p.add_argument("ms", help="Measurement Set")
    p.add_argument("-c", "--config", default="",
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
                   help="Number of channels to dilate as int or string with units")
    p.add_argument("-dc", "--data-column", type=str, default="DATA",
                   help="Name of visibility data column to flag")
    return p


def main():
    tic = time.time()
    print_info()
    args = create_parser().parse_args()
    log.info("Will process {0:s} column".format(args.data_column))
    data_column = args.data_column
    masked_channels = [load_mask(fn, dilate=args.dilate_masks) for fn in collect_masks()]
    cfg_file, flagger_kwargs = args.config

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
                           columns=(data_column, "FLAG", "ANTENNA1", "ANTENNA2"),
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
        row_counts = np.asarray(row_counts)
        ntime, nbl = row_counts.size, row_counts[0]
        nrow, nchan, ncorr = ds.DATA.data.shape
        chunks = da.from_array(row_counts, chunks=(agg_time_counts,))

        # Visibilities from the dataset
        vis = getattr(ds, data_column).data
        a1 = ds.ANTENNA1.data
        a2 = ds.ANTENNA2.data
        chan_freq = spw_info.CHAN_FREQ.values
        chan_width = spw_info.CHAN_WIDTH.values

        # Generate unflagged defaults if we should ignore existing flags
        # otherwise take flags from the dataset
        if args.ignore_flags == True:
            flags = da.full_like(vis, False, dtype=np.bool)
        else:
            flags = ds.FLAG.data

        # Reorder vis and flags into katdal-like format
        # (ntime, nchan, ncorrprod). Chunk the corrprod
        # dimension into groups of 64 baselines
        vis = vis.reshape(ntime, nbl, nchan, ncorr)
        flags = flags.reshape(ntime, nbl, nchan, ncorr)
        a1 = a1.reshape(ntime, nbl)
        a2 = a2.reshape(ntime, nbl)
        # Rechunk on baseline dimension
        vis = vis.rechunk({1: 64})
        flags = flags.rechunk({1: 64})
        a1 = a1.rechunk({1: 64})
        a2 = a2.rechunk({1: 64})

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

        vis = vis.transpose(0, 2, 1, 3)
        vis = vis.reshape((ntime, nchan, nbl * xncorr))
        flags = flags.transpose(0, 2, 1, 3)
        flags = flags.reshape((ntime, nchan, nbl * xncorr))
        # a1 = a1.repeat(xncorr, axis=0).reshape((ntime, nbl * xncorr))
        # a2 = a2.repeat(xncorr, axis=0).reshape((ntime, nbl * xncorr))

        def _explode_ants(ants, xncorr=None):
            return np.tile(ants, (ants.shape[0], ants.shape[1]*xncorr))
            # ntime, nbl = ants.shape
            # return ants.repeat(xncorr, axis=0).reshape(ntime, nbl*xncorr)

        a1 = da.blockwise(_explode_ants, ("time", "corrprod"),
                          a1, ("time", "corrprod"),
                          adjust_chunks={"corrprod": lambda x: x*xncorr},
                          xncorr=xncorr,
                          dtype=a1.dtype)

        a2 = da.blockwise(_explode_ants, ("time", "corrprod"),
                          a2, ("time", "corrprod"),
                          adjust_chunks={"corrprod": lambda x: x*xncorr},
                          xncorr=xncorr,
                          dtype=a1.dtype)


        # Run the flagger
        original = flags.copy()
        new_flags = flags
        for k in GD:
            if GD[k].get("task", "unnamed") == "sum_threshold":
                ("task" in GD[k]) and GD[k].pop("task")
                ("order" in GD[k]) and GD[k].pop("order")
                new_flags = sum_threshold_flagger(vis,
                                                  new_flags,
                                                  chunks,
                                                  **GD[k])
            elif GD[k].get("task", "unnamed") == "uvcontsub_flagger":
                ("task" in GD[k]) and GD[k].pop("task")
                ("order" in GD[k]) and GD[k].pop("order")
                new_flags = uvcontsub_flagger(vis,
                                              new_flags,
                                              **GD[k])
            elif GD[k].get("task", "unnamed") == "combine_with_input_flags":
                new_flags = da.logical_or(new_flags,
                                          original)
            elif GD[k].get("task", "unnamed") == "apply_static_mask":
                ("task" in GD[k]) and GD[k].pop("task")
                ("order" in GD[k]) and GD[k].pop("order")
                new_flags = apply_static_mask(vis,
                                              new_flags,
                                              a1,
                                              a2,
                                              antspos,
                                              masked_channels,
                                              chan_freq,
                                              chan_width,
                                              xncorr,
                                              **GD[k])

            else:
                raise TypeError("Task {0:s} does not name a valid task".format(GD[k].get("task", "unnamed")))

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
        log.info("Data flagged successfully in {0:02.0f}:{1:02.0f}:{2:02.0f} hours".format((elapsed // 60) // 60,
                                                                                           (elapsed // 60) % 60,
                                                                                           elapsed % 60))
