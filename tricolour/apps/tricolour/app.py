# -*- coding: utf-8 -*-

""" Main tricolour application """

import re
import argparse
import contextlib
from functools import partial
import logging
import logging.handlers
from multiprocessing.pool import ThreadPool
import pkg_resources
import os
from os.path import join as pjoin
import sys
import time

import dask
import dask.array as da
from dask.diagnostics import (ProgressBar, Profiler,
                              ResourceProfiler,
                              CacheProfiler, visualize)
import numpy as np
from daskms import xds_from_ms, xds_from_table, xds_to_table
from threadpoolctl import threadpool_limits

from tricolour.apps.tricolour.strat_executor import StrategyExecutor
from tricolour.banner import banner
from tricolour.mask import collect_masks, load_mask
from tricolour.stokes import stokes_corr_map
from tricolour.dask_wrappers import polarised_intensity

from tricolour.packing import (unique_baselines,
                               pack_data,
                               unpack_data)
from tricolour.util import (casa_style_int_list)
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
# Initialize Application Logger
##############################################################


def create_logger():
    """ Create a console logger """
    log = logging.getLogger("tricolour")
    cfmt = logging.Formatter(u'%(name)s - %(asctime)s '
                             '%(levelname)s - %(message)s')
    log.setLevel(logging.DEBUG)
    filehandler = logging.FileHandler("tricolour.log")
    filehandler.setFormatter(cfmt)
    log.addHandler(filehandler)
    log.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)

    log.addHandler(console)

    return log


# Create the log object
log = create_logger()

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
    from tricolour import config
    import yaml

    with open(config_file) as cf:
        config.update_defaults(yaml.full_load(cf))

    return config


def log_configuration(args):
    cfg = args.config.to_dict()
    empty_dict = {}

    try:
        strategies = cfg['strategies']
    except KeyError:
        log.warn("Configuration has no strategies")
        return

    if len(strategies) > 0:
        log.info("*****************************************")
        log.info("The following strategies will be applied:")
        log.info("*****************************************")

        for s, strategy in enumerate(strategies):
            name = strategy.get("name", "<nameless>")

            try:
                task = strategy["task"]
            except KeyError:
                log.warn("Strategy '%s' has no associate task", name)

            log.info("%d: %s (%s)", s, task, name)

            for key, value in strategy.get("kwargs", empty_dict).items():
                log.info("\t%s: %s", key, value)
        log.info("***************** END ********************")

    if args.flagging_strategy == "polarisation":
        log.info("Flagging based on quadrature polarized power")
    elif args.flagging_strategy == "total_power":
        log.info("Flagging on total quadrature power")
    else:
        log.info("Flagging per correlation ('standard' mode)")


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
                   choices=["standard", "polarisation", "total_power"],
                   help="Flagging Strategy. "
                        "If 'standard' all correlations in the visibility "
                          "are flagged independently. "
                        "If 'polarisation' the polarised intensity "
                          "sqrt(Q^2 + U^2 + V^2) is "
                          "calculated and used to flag all correlations "
                          "in the visibility."
                        "If 'total_power' the available quadrature power "
                        "is computed "
                        "sqrt(I^2 + Q^2 + U^2 + V^2) or a subset, and used "
                        "to flag "
                        "all correlations in the visibility")
    p.add_argument("-rc", "--row-chunks", type=int, default=10000,
                   help="Hint indicating the number of Measurement Set rows "
                   "to read in a single chunk. "
                   "Smaller and larger numbers will tend to "
                   "respectively decrease or increase both memory usage "
                   "and computational efficiency")
    p.add_argument("-bc", "--baseline-chunks", type=int, default=16,
                   help="Number of baselines in a window chunk")
    p.add_argument("-nw", "--nworkers", type=int, default=os.cpu_count(),
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
                   type=partial(casa_style_int_list,
                                argparse=True, opt_unit=" "),
                   default=None,
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
    p.add_argument("-smc", "--subtract-model-column", default=None, type=str,
                   help="Subtracts specified column from data column "
                        "specified. "
                        "Flagging will proceed on residual "
                        "data.")
    return p


def support_tables(ms):
    """
    Parameters
    ----------
    ms : str
        base measurement set
    Returns
    -------
    table_map : dict of Dataset
        {name: dataset}
    """

    # Get datasets for sub-tables partitioned by row when variably shaped
    support = {t: xds_from_table("::".join((ms, t)), group_cols="__row__")
               for t in ["FIELD", "POLARIZATION", "SPECTRAL_WINDOW"]}
    # These columns have fixed shapes
    support.update({t: xds_from_table("::".join((ms, t)))
                    for t in ["ANTENNA", "DATA_DESCRIPTION"]})

    # Reify all values upfront
    return dask.compute(support)[0]


def main():
    with contextlib.ExitStack() as stack:
        # Limit numpy/blas etc threads to 1, as we obtain
        # our parallelism with dask threads
        stack.enter_context(threadpool_limits(limits=1))

        args = create_parser().parse_args()

        # Configure dask pool
        if args.nworkers <= 1:
            log.warn("Entering single threaded mode per user request!")
            dask.config.set(scheduler='single-threaded')
        else:
            stack.enter_context(dask.config.set(
                pool=ThreadPool(args.nworkers)))

        _main(args)


def _main(args):
    tic = time.time()

    log.info(banner())

    if args.disable_post_mortem:
        log.warn("Disabling crash debugging with the "
                 "Interactive Python Debugger, as per user request")
        post_mortem_handler.disable_pdb_on_error()

    log.info("Flagging on the {0:s} column".format(args.data_column))
    data_column = args.data_column
    masked_channels = [load_mask(fn, dilate=args.dilate_masks)
                       for fn in collect_masks()]
    GD = args.config

    log_configuration(args)

    # Group datasets by these columns
    group_cols = ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]
    # Index datasets by these columns
    index_cols = ['TIME']

    # Reopen the datasets using the aggregated row ordering
    columns = [data_column,
               "FLAG",
               "TIME",
               "ANTENNA1",
               "ANTENNA2"]

    if args.subtract_model_column is not None:
        columns.append(args.subtract_model_column)

    xds = list(xds_from_ms(args.ms,
                           columns=tuple(columns),
                           group_cols=group_cols,
                           index_cols=index_cols,
                           chunks={"row": args.row_chunks}))

    # Get support tables
    st = support_tables(args.ms)
    ddid_ds = st["DATA_DESCRIPTION"]
    field_ds = st["FIELD"]
    pol_ds = st["POLARIZATION"]
    spw_ds = st["SPECTRAL_WINDOW"]
    ant_ds = st["ANTENNA"]

    assert len(ant_ds) == 1
    assert len(ddid_ds) == 1

    antspos = ant_ds[0].POSITION.data
    antsnames = ant_ds[0].NAME.data
    fieldnames = [fds.NAME.data[0] for fds in field_ds]

    avail_scans = [ds.SCAN_NUMBER for ds in xds]
    args.scan_numbers = list(set(avail_scans).intersection(
        args.scan_numbers if args.scan_numbers is not None else avail_scans))

    if args.scan_numbers != []:
        log.info("Only considering scans '{0:s}' as "
                 "per user selection criterion"
                 .format(", ".join(map(str, map(int, args.scan_numbers)))))

    if args.field_names != []:
        flatten_field_names = []
        for f in args.field_names:
            # accept comma lists per specification
            flatten_field_names += [x.strip() for x in f.split(",")]
        for f in flatten_field_names:
            if re.match(r"^\d+$", f) and int(f) < len(fieldnames):
                flatten_field_names.append(fieldnames[int(f)])
        flatten_field_names = list(
            set(filter(lambda x: not re.match(r"^\d+$", x),
                       flatten_field_names)))
        log.info("Only considering fields '{0:s}' for flagging per "
                 "user "
                 "selection criterion.".format(
                    ", ".join(flatten_field_names)))
        if not set(flatten_field_names) <= set(fieldnames):
            raise ValueError("One or more fields cannot be "
                             "found in dataset '{0:s}' "
                             "You specified {1:s}, but "
                             "only {2:s} are available".format(
                                 args.ms,
                                 ",".join(flatten_field_names),
                                 ",".join(fieldnames)))

        field_dict = {fieldnames.index(fn): fn for fn in flatten_field_names}
    else:
        field_dict = {i: fn for i, fn in enumerate(fieldnames)}

    # List which hold our dask compute graphs for each dataset
    write_computes = []
    original_stats = []
    final_stats = []

    # Iterate through each dataset
    for ds in xds:
        if ds.FIELD_ID not in field_dict:
            continue

        if (args.scan_numbers is not None and
                ds.SCAN_NUMBER not in args.scan_numbers):
            continue

        log.info("Adding field '{0:s}' scan {1:d} to "
                 "compute graph for processing"
                 .format(field_dict[ds.FIELD_ID], ds.SCAN_NUMBER))

        ddid = ddid_ds[ds.attrs['DATA_DESC_ID']]
        spw_info = spw_ds[ddid.SPECTRAL_WINDOW_ID.data[0]]
        pol_info = pol_ds[ddid.POLARIZATION_ID.data[0]]

        nrow, nchan, ncorr = getattr(ds, data_column).data.shape

        # Visibilities from the dataset
        vis = getattr(ds, data_column).data
        if args.subtract_model_column is not None:
            log.info("Forming residual data between '{0:s}' and "
                     "'{1:s}' for flagging.".format(
                        data_column, args.subtract_model_column))
            vismod = getattr(ds, args.subtract_model_column).data
            vis = vis - vismod

        antenna1 = ds.ANTENNA1.data
        antenna2 = ds.ANTENNA2.data
        chan_freq = spw_info.CHAN_FREQ.data[0]
        chan_width = spw_info.CHAN_WIDTH.data[0]

        # Generate unflagged defaults if we should ignore existing flags
        # otherwise take flags from the dataset
        if args.ignore_flags is True:
            flags = da.full_like(vis, False, dtype=np.bool)
            log.critical("Completely ignoring measurement set "
                         "flags as per '-if' request. "
                         "Strategy WILL NOT or with original flags, even if "
                         "specified!")
        else:
            flags = ds.FLAG.data

        # If we're flagging on polarised intensity,
        # we convert visibilities to polarised intensity
        # and any flagged correlation will flag the entire visibility
        if args.flagging_strategy == "polarisation":
            corr_type = pol_info.CORR_TYPE.data[0].tolist()
            stokes_map = stokes_corr_map(corr_type)
            stokes_pol = tuple(v for k, v in stokes_map.items() if k != "I")
            vis = polarised_intensity(vis, stokes_pol)
            flags = da.any(flags, axis=2, keepdims=True)
        elif args.flagging_strategy == "total_power":
            if args.subtract_model_column is None:
                log.critical("You requested to flag total quadrature "
                             "power, but not on residuals. "
                             "This is not advisable and the flagger "
                             "may mistake fringes of "
                             "off-axis sources for broadband RFI.")
            corr_type = pol_info.CORR_TYPE.data[0].tolist()
            stokes_map = stokes_corr_map(corr_type)
            stokes_pol = tuple(v for k, v in stokes_map.items())
            vis = polarised_intensity(vis, stokes_pol)
            flags = da.any(flags, axis=2, keepdims=True)
        elif args.flagging_strategy == "standard":
            if args.subtract_model_column is None:
                log.critical("You requested to flag per correlation, "
                             "but not on residuals. "
                             "This is not advisable and the flagger "
                             "may mistake fringes of off-axis sources "
                             "for broadband RFI.")
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

        original_stats.append(window_stats(flag_windows, ubl, chan_freq,
                                           antsnames, ds.SCAN_NUMBER,
                                           field_dict[ds.FIELD_ID],
                                           ds.attrs['DATA_DESC_ID']))

        with StrategyExecutor(antspos, ubl, chan_freq, chan_width,
                              masked_channels, GD['strategies']) as se:

            flag_windows = se.apply_strategies(flag_windows, vis_windows)

        final_stats.append(window_stats(flag_windows, ubl, chan_freq,
                                        antsnames, ds.SCAN_NUMBER,
                                        field_dict[ds.FIELD_ID],
                                        ds.attrs['DATA_DESC_ID']))

        # Unpack window data for writing back to the MS
        unpacked_flags = unpack_data(antenna1, antenna2, time_inv,
                                     ubl, flag_windows)

        # Flag entire visibility if any correlations are flagged
        equalized_flags = da.sum(unpacked_flags, axis=2, keepdims=True) > 0
        corr_flags = da.broadcast_to(equalized_flags, (nrow, nchan, ncorr))

        if corr_flags.chunks != ds.FLAG.data.chunks:
            raise ValueError("Output flag chunking does not "
                             "match input flag chunking")

        # Create new dataset containing new flags
        new_ds = ds.assign(FLAG=(("row", "chan", "corr"), corr_flags))

        # Write back to original dataset
        writes = xds_to_table(new_ds, args.ms, "FLAG")
        # original should also have .compute called because we need stats
        write_computes.append(writes)

    if len(write_computes) > 0:
        # Combine stats from all datasets
        original_stats = combine_window_stats(original_stats)
        final_stats = combine_window_stats(final_stats)

        with contextlib.ExitStack() as stack:
            # Create dask profiling contexts
            profilers = []

            if can_profile:
                profilers.append(stack.enter_context(Profiler()))
                profilers.append(stack.enter_context(CacheProfiler()))
                profilers.append(stack.enter_context(ResourceProfiler()))

            if sys.stdout.isatty():
                # Interactive terminal, default ProgressBar
                stack.enter_context(ProgressBar())
            else:
                # Non-interactive, emit a bar every 5 minutes so
                # as not to spam the log
                stack.enter_context(ProgressBar(minimum=1, dt=5*60))

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
    else:
        log.info(
            "User data selection criteria resulted in empty dataset. "
            "Nothing to be done. Bye!")


if __name__ == "__main__":
    main()
