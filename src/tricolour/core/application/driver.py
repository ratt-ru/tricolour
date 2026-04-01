import importlib
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict
from msv4_utils import MSv4Backend, infer_backend
import xarray
from tricolour.core.util import casa_style_int_list, casa_style_range
import numpy as np
from tricolour.core.application.worker import WorkQueue, FlaggingWorker
import ray
import logging
import os
from datetime import datetime
import tricolour.core.application.post_mortem_handler as post_mortem_handler
from tricolour.core.application.banner import banner
import time
from tricolour import config
from tricolour.core.kernels.mask import collect_masks, load_mask
from tricolour.core.kernels.flag_statistics import (
    combine_window_stats,
    WindowStatistics
)

def create_logger():
    """ Create a console logger """
    log = logging.getLogger("tricolour")
    cfmt = logging.Formatter(u'%(name)s - %(asctime)s '
                             '%(levelname)s - %(message)s')
    log.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(cfmt)
    log.addHandler(console)

    # add an optional file handler
    logger_path = os.environ.get("TRICOLOUR_LOGPATH", os.getcwd())
    nowT = int(np.ceil(datetime.timestamp(datetime.now())))
    logfile = os.path.join(logger_path,
                           f"tricolour.{nowT}.log")
    try:
        with open(logfile, "w") as f:
            f.write("")
        filehandler = logging.FileHandler(logfile)
        filehandler.setFormatter(cfmt)
        log.addHandler(filehandler)
        if logger_path != os.getcwd():
            log.info(f"A copy of this log is available at {logfile}")
    except PermissionError:
        log.warning(f"Failed to initialize logfile for this run. "
                    f"Check your permissions and available space on "
                    f"'{logger_path}'. Proceeding without writing "
                    f"a logfile.")
    return log


# Create the log object
log = create_logger()

@dataclass
class FlagItem:
  region: Dict[str, int]


@dataclass
class BackendImport:
  package: str
  install_option: str


BACKEND_MAP = {
  MSv4Backend.CASA_TABLE: BackendImport("xarray_ms", "msv2"),
  MSv4Backend.MEERKAT: BackendImport("xarray_kat", "meerkat"),
  MSv4Backend.ZARR: BackendImport("zarr", "zarr"),
}

SUPPORTS_WRITEBACK = {MSv4Backend.CASA_TABLE: True, MSv4Backend.MEERKAT: False, MSv4Backend.ZARR: True}


def infer_and_import_backend(uri: str) -> MSv4Backend:
  uri_backend = infer_backend(uri, strict=False)

  try:
    backend_import = BACKEND_MAP[uri_backend]
  except KeyError as e:
    raise ValueError(f"Unsupported MSv4 backend {uri} {uri_backend}") from e

  try:
    importlib.import_module(backend_import.package)
  except ImportError:
    raise ImportError(
      f"The {uri_backend.name} backend is not installed.\npip install tricolour[{backend_import.install_option}]"
    )

  return uri_backend

def load_partitions(cfg):
  source_backend = infer_and_import_backend(cfg.ms)
  if source_backend == MSv4Backend.CASA_TABLE:
    from xarray_ms.backend.msv2.structure import DEFAULT_PARTITION_COLUMNS
  else: 
    # TODO
    DEFAULT_PARTITION_COLUMNS = []
  dt = xarray.open_datatree(
    cfg.ms,
    partition_schema=["FIELD_ID", "SCAN_NUMBER"] + DEFAULT_PARTITION_COLUMNS,
    auto_corrs=True
  )
  partitions = list(map(lambda partition: dt[partition], dt.children))
  if cfg.field_names:
    # we don't use sel here because scan is not a single coordinate here
    partitions = list(filter(lambda partition: set(list(np.unique(partition.field_name.data))).issubset(set(cfg.field_names)), 
                             partitions))
  if cfg.scan_numbers:
    scans = []
    for scr in cfg.scan_numbers.split(","):
      scans += casa_style_int_list(scr, opt_unit=" ")
    # we don't use sel here because scan is not a single coordinate here
    partitions = list(filter(lambda partition: set(map(lambda x: int(x), 
                                                       list(np.unique(partition.scan_name.data)))).issubset(set(scans)), 
                             partitions))
  return partitions

def chunk_partitions(partitions, num_bl, num_time):
  # chunks by time group
  chunked_partitions = []
  regions = []
  for pi in partitions:
    nrows = pi.time.size * pi.baseline_id.size
    nchunk_t = pi.time.size // num_time + (pi.time.size % num_time > 0)
    nchunk_bl = pi.baseline_id.size // num_bl + (pi.baseline_id.size % num_bl > 0)
    vels_sel = 0
    for icht in range(nchunk_t):
      tlb = icht * num_time
      tub = min((icht + 1) * num_time, pi.time.size)
      for ichb in range(nchunk_bl):
        blb = ichb * num_bl
        bub = min((ichb + 1) * num_bl, pi.baseline_id.size)
        region = dict(
           time = slice(tlb, tub),
           baseline_id = slice(blb, bub)
        )
        chunked_partitions.append(pi.isel(**region))
        regions.append(region)
        vels_sel += (tub - tlb) * (bub - blb)
    assert vels_sel == nrows
  return chunked_partitions, regions

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
    cfg = config.to_dict()
    empty_dict = {}

    try:
        strategies = cfg['strategies']
    except KeyError:
        log.warning("Configuration has no strategies")
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
                log.warning("Strategy '%s' has no associate task", name)

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

def driver(cfg: Namespace):

  if cfg.nworkers == 1:
    context = ray.init(num_cpus=1, local_mode=True)
  else:
    context = ray.init(num_cpus=cfg.nworkers)
  if not cfg.disable_post_mortem:
    post_mortem_handler.enable_pdb_on_error()
  else:
    log.warning("Disabling crash debugging with the "
                "Interactive Python Debugger, as per user request")
  print(banner())

  config_file = load_config(cfg.config)
  log_configuration(cfg)

  masks = {}
  for mask in collect_masks():
     masks[mask] = load_mask(mask, dilate=cfg.dilate_masks)
  log.info(f"Partitioning database {cfg.ms}")
  partitions, regions = chunk_partitions(load_partitions(cfg),
                                         cfg.baseline_chunks,
                                         cfg.time_chunks)
  
  log.info(f"Enquing partitions for processing...")
  wq = WorkQueue.remote()
  for pi, ri in zip(partitions, regions):
    wq.enqueue_partition.remote(pi, ri)
  log.info(f"Starting flagging operations")
  tic = time.time()
  fw = []
  for icpu in range(cfg.nworkers):
    fw.append(FlaggingWorker.remote(
      wq,
      masks = masks,
      data_column = cfg.data_column,
      subtract_model_column = cfg.subtract_model_column,
      flagging_strategy = cfg.flagging_strategy,
      flagging_config = config_file["strategies"]
    ))
  ray.get([fwi.run.remote(wq) for fwi in fw])
  final_stats = combine_window_stats(ray.get([fwi.report_statistics.remote() for fwi in fw]))
  original_stats = combine_window_stats(ray.get([fwi.report_original_statistics.remote() for fwi in fw]))
  toc = time.time()

  # finally print flagging statistics
  for line in WindowStatistics.summarise_stats(final_stats, original_stats):
    log.info(line)

  elapsed = toc - tic
  log.info("Data flagged successfully in "
            "{0:02.0f}h{1:02.0f}m{2:02.0f}s"
            .format((elapsed // 60) // 60,
                    (elapsed // 60) % 60,
                    elapsed % 60))