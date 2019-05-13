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
import pkg_resources
import os
from os.path import join as pjoin
import time

import dask
import dask.array as da
from dask.diagnostics import (ProgressBar, Profiler,
                              ResourceProfiler,
                              CacheProfiler, visualize)
import numpy as np
import xarray as xr
from xarrayms import xds_from_ms, xds_from_table, xds_to_table

from tricolour.banner import banner
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
from tricolour.window_statistics import (window_stats,
                                         combine_window_stats,
                                         summarise_stats)

##############################################################
# Initialize Post Mortem debugger
##############################################################
import tricolour.post_mortem_handler as post_mortem_handler

post_mortem_handler.enable_pdb_on_error()

try:
    import bokeh  # noqa
    can_profile = True
except ImportError:
    can_profile = False

##############################################################
# Initialize Global logging
##############################################################


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

DEFAULT_CONFIG = pkg_resources.resource_filename('tricolour',
                                                 pjoin("conf", "default.yaml"))


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

    log.info(banner())

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
    table_kwargs = {'ack': False}
    xds = list(xds_from_ms(args.ms,
                           columns=(data_column, "FLAG",
                                    "TIME", "ANTENNA1", "ANTENNA2"),
                           group_cols=group_cols,
                           index_cols=index_cols,
                           chunks={"row": args.row_chunks},
                           table_kwargs=table_kwargs))

    # Get datasets for DATA_DESCRIPTION and POLARIZATION,
    # partitioned by row
    data_desc_tab = "::".join((args.ms, "DATA_DESCRIPTION"))
    ddid_ds = list(xds_from_table(data_desc_tab, group_cols="__row__",
                                  table_kwargs=table_kwargs))
    pol_tab = "::".join((args.ms, "POLARIZATION"))
    pol_ds = list(xds_from_table(pol_tab, group_cols="__row__",
                                 table_kwargs=table_kwargs))
    ant_tab = "::".join((args.ms, "ANTENNA"))
    ads = list(xds_from_table(ant_tab))
    spw_tab = "::".join((args.ms, "SPECTRAL_WINDOW"))
    spw_ds = list(xds_from_table(spw_tab, group_cols="__row__",
                                 table_kwargs=table_kwargs))
    antspos = ads[0].POSITION.values
    antsnames = ads[0].NAME.values
    fld_tab = "::".join((args.ms, "FIELD"))
    field_ds = list(xds_from_table(fld_tab, table_kwargs=table_kwargs))
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

    # List which hold our dask compute graphs for each dataset
    write_computes = []
    original_stats = []
    final_stats = []

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
            log.warn("!!!NOTE: COMPLETELY IGNORING MEASUREMENT SET FLAGS "
                     " AS PER -IF FLAG REQUEST!!!")
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
        elif args.flagging_strategy == "standard":
            pass
        else:
            raise ValueError("Invalid flagging strategy '%s'" %
                             args.flagging_strategy)

        ubl = unique_baselines(antenna1, antenna2)
        utime, time_inv = da.unique(ds.TIME.data, return_inverse=True)
        utime, ubl = dask.compute(utime, ubl)
        ubl = ubl.view(np.int32).reshape(-1, 2)
        # Stack the baseline index with the unique baselines
        bl_range = np.arange(ubl.shape[0], dtype=ubl.dtype)[:, None]
        ubl = np.concatenate([bl_range, ubl], axis=1)
        ubl = da.from_array(ubl, chunks=(args.baseline_chunks, 3))

        vis_windows, flag_windows = pack_data(time_inv, ubl,
                                              antenna1, antenna2,
                                              vis, flags, utime.shape[0],
                                              backend=args.window_backend,
                                              path=args.temporary_directory)

        original = flag_windows.copy()
        original_stats.append(window_stats(original, ubl, chan_freq,
                                           antsnames, ds.SCAN_NUMBER,
                                           field_dict[ds.FIELD_ID],
                                           ds.attrs['DATA_DESC_ID']))

        # Run the flagger
        for k in GD:
            if GD[k].get("task", "unnamed") == "sum_threshold":
                task_kwargs = GD[k].copy()
                task_kwargs.pop("task", None)
                task_kwargs.pop("order", None)
                new_flags = sum_threshold_flagger(vis_windows, flag_windows,
                                                  **task_kwargs)
                # sum threshold builds upon any flags that came previous
                flag_windows = da.logical_or(new_flags, flag_windows)
            elif GD[k].get("task", "unnamed") == "uvcontsub_flagger":
                task_kwargs = GD[k].copy()
                task_kwargs.pop("task", None)
                task_kwargs.pop("order", None)
                new_flags = uvcontsub_flagger(vis_windows, flag_windows,
                                              **task_kwargs)
                # this task discards previous flags by default during its
                # second iteration. The original flags from MS should be or'd
                # back in afterwards. Flags from steps prior to this one serves
                # only as a "initial guess"
                flag_windows = new_flags
            elif GD[k].get("task", "unnamed") == "flag_autos":
                task_kwargs = GD[k].copy()
                task_kwargs.pop("task", None)
                task_kwargs.pop("order", None)
                new_flags = flag_autos(flag_windows, ubl, **task_kwargs)
                flag_windows = da.logical_or(new_flags, flag_windows)
            elif GD[k].get("task", "unnamed") == "combine_with_input_flags":
                # or's in original flags from the measurement set
                # (if -if option has not been specified,
                # in which case this option will do nothing)
                flag_windows = da.logical_or(flag_windows, original)
            elif GD[k].get("task", "unnamed") == "unflag":
                flag_windows = da.zeros_like(flag_windows)
            elif GD[k].get("task", "unnamed") == "flag_nans_zeros":
                new_flags = flag_windows.copy()
                sel = da.logical_or(vis_windows == 0 + 0j,
                                    da.isnan(vis_windows))
                new_flags[sel] = True
                flag_windows = da.logical_or(flag_windows,
                                             new_flags)
            elif GD[k].get("task", "unnamed") == "apply_static_mask":
                task_kwargs = GD[k].copy()
                task_kwargs.pop("task", None)
                task_kwargs.pop("order", None)
                new_flags = apply_static_mask(flag_windows,
                                              ubl,
                                              antspos,
                                              masked_channels,
                                              chan_freq,
                                              chan_width,
                                              **task_kwargs)
                # override option will override any flags computed previously
                # this may not be desirable so use with care or in combination
                # with combine_with_input_flags option!
                if task_kwargs["accumulation_mode"].strip() == "or":
                    flag_windows = da.logical_or(new_flags, flag_windows)
                else:
                    flag_windows = new_flags

            else:
                raise ValueError("Task '{0:s}' does not name a valid task"
                                 .format(GD[k].get("task", "unnamed")))

        final_stats.append(window_stats(flag_windows, ubl, chan_freq,
                                        antsnames, ds.SCAN_NUMBER,
                                        field_dict[ds.FIELD_ID],
                                        ds.attrs['DATA_DESC_ID']))

        # finally unpack back for writing
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
        # original should also have .compute called because we need stats
        write_computes.append(writes)

    # Create dask contexts
    profilers = ([Profiler(), CacheProfiler(), ResourceProfiler()]
                 if can_profile else [])
    contexts = [ProgressBar()] + profilers

    pool = ThreadPool(args.nworkers)

    # Combine stats from all datasets
    original_stats = combine_window_stats(original_stats)
    final_stats = combine_window_stats(final_stats)

    with contextlib.nested(*contexts), dask.config.set(pool=pool):
        _, original_stats, final_stats = dask.compute(write_computes,
                                                      original_stats,
                                                      final_stats)

    if can_profile:
        visualize(profilers)

    toc = time.time()

    # Log each summary line
    for line in summarise_stats(final_stats, original_stats):
        log.info(line)

    elapsed = toc - tic
    log.info("Data flagged successfully in "
             "{0:02.0f}h{1:02.0f}m{2:02.0f}s"
             .format((elapsed // 60) // 60,
                     (elapsed // 60) % 60,
                     elapsed % 60))
