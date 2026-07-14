import time
from argparse import Namespace

import numpy as np
import ray
import xarray
from msv4_utils import MSv4Backend
from rarg_python_patterns.multiton import Multiton

from tricolour import config
from tricolour.core.application.backend import infer_and_import_backend
from tricolour.core.application.banner import banner
from tricolour.core.application.worker import FlaggingWorker, WorkQueue
from tricolour.core.kernels.flag_statistics import WindowStatistics, combine_window_stats
from tricolour.core.kernels.mask import collect_masks, load_mask
from tricolour.core.util import casa_style_int_list


def open_datatree(uri: str) -> xarray.DataTree:
  _, open_kwargs = infer_and_import_backend(uri)
  return xarray.open_datatree(uri, **open_kwargs)


def load_partitions(cfg):
  source_backend = infer_and_import_backend(cfg.ms)
  if source_backend == MSv4Backend.CASA_TABLE:
    from xarray_ms.backend.msv2.structure import DEFAULT_PARTITION_COLUMNS

    kwargs = {"partition_schema": ["FIELD_ID", "SCAN_NUMBER"] + DEFAULT_PARTITION_COLUMNS, "auto_corrs": True}
  elif source_backend == MSv4Backend.ZARR:
    kwargs = {}
  else:
    raise NotImplementedError("Currently only MSv2/v4 and Zarr datatrees are supported")
  dt = xarray.open_datatree(cfg.ms, **kwargs)
  partitions = list(map(lambda partition: dt[partition], dt.children))

  if source_backend == MSv4Backend.ZARR:
    for p in partitions:
      scan_numbers = np.unique(p.scan_name.data)
      if len(scan_numbers) > 1:
        raise RuntimeError(
          "Zarr dataset has to be prepartitioned by scan. Re-dump with partition_schema including SCAN_NUMBER"
        )
      nant = len(p.antenna_xds.antenna_name.data)
      nbl = len(p.baseline_id)
      if nbl != nant * (nant - 1) // 2 + nant:
        raise RuntimeError("Zarr dataset should have autocorrelations in the datatree. Re-dump with auto_corrs=True")

  if cfg.field_names:
    # we don't use sel here because scan is not a single coordinate here
    partitions = list(
      filter(
        lambda partition: set(list(np.unique(partition.field_name.data))).issubset(set(cfg.field_names)), partitions
      )
    )
  if cfg.scan_numbers:
    scans = []
    for scr in cfg.scan_numbers.split(","):
      scans += casa_style_int_list(scr, opt_unit=" ")
    # we don't use sel here because scan is not a single coordinate here
    partitions = list(
      filter(
        lambda partition: set(map(lambda x: int(x), list(np.unique(partition.scan_name.data)))).issubset(set(scans)),
        partitions,
      )
    )
  return partitions


def chunk_partitions(partitions, num_bl, num_time):
  # chunks by time group
  chunked_partitions = []
  regions = []
  data_trees = []
  for pi in partitions:
    data_trees.append(pi.name)
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
          time=slice(tlb, tub),
          baseline_id=slice(blb, bub),
          frequency=slice(None),
          polarization=slice(None),
          uvw_label=slice(None),
        )
        chunked_partitions.append(pi.isel(**region))
        regions.append(region)
        vels_sel += (tub - tlb) * (bub - blb)
    assert vels_sel == nrows
  return chunked_partitions, regions, data_trees


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
  import yaml

  from tricolour import config

  with open(config_file) as cf:
    config.update_defaults(yaml.full_load(cf))

  return config


def log_configuration(args):
  cfg = config.to_dict()
  empty_dict = {}

  try:
    strategies = cfg["strategies"]
  except KeyError:
    print("Configuration has no strategies")
    return

  if len(strategies) > 0:
    print("*****************************************")
    print("The following strategies will be applied:")
    print("*****************************************")

    for s, strategy in enumerate(strategies):
      name = strategy.get("name", "<nameless>")

      try:
        task = strategy["task"]
      except KeyError:
        print(f"Strategy '{name}' has no associated task")

      print(f"{s}: {task} ({name})")

      for key, value in strategy.get("kwargs", empty_dict).items():
        print(f"\t{key}: {value}", key, value)
    print("***************** END ********************")

  if args.flagging_strategy == "polarisation":
    print("Flagging based on quadrature polarized power")
  elif args.flagging_strategy == "total_power":
    print("Flagging on total quadrature power")
  else:
    print("Flagging per correlation ('standard' mode)")


def driver(cfg: Namespace):
  config_file = load_config(cfg.config)
  log_configuration(cfg)

  masks = {m: load_mask(m, dilate=cfg.dilate_masks) for m in collect_masks()}
  datatree = Multiton(open_datatree, cfg.ms).with_infinite_ttl()

  if cfg.ray_scheduler_address is None:
    ray_ctx = ray.init(num_cpus=cfg.nworkers)
  else:
    ray_ctx = ray.init(address=cfg.ray_scheduler_address)

  return

  if cfg.nworkers == 1:
    ray.init(num_cpus=1, local_mode=True)
  else:
    ray.init(num_cpus=cfg.nworkers)

  print(banner())

  config_file = load_config(cfg.config)
  log_configuration(cfg)

  masks = {}
  for mask in collect_masks():
    masks[mask] = load_mask(mask, dilate=cfg.dilate_masks)
  print(f"Partitioning database {cfg.ms}")
  partitions, regions, parent_data_trees = chunk_partitions(load_partitions(cfg), cfg.baseline_chunks, cfg.time_chunks)

  print("Enquing partitions for processing...")
  wq = WorkQueue.remote()
  for pi, ri, dt in zip(partitions, regions, parent_data_trees):
    wq.enqueue_partition.remote(pi, ri, dt)
  print("Starting flagging operations")
  tic = time.time()
  fw = []
  # Never spawn more workers than there are partitions to process; idle
  # workers accumulate no statistics and only add scheduling overhead.
  nworkers = max(1, min(cfg.nworkers, len(partitions)))
  for icpu in range(nworkers):
    fw.append(
      FlaggingWorker.remote(
        wq,
        masks=masks,
        data_column=cfg.data_column,
        subtract_model_column=cfg.subtract_model_column,
        flagging_strategy=cfg.flagging_strategy,
        flagging_config=config_file["strategies"],
        source_backend=infer_and_import_backend(cfg.ms),
        dataset_path=cfg.ms,
      )
    )
  ray.get([fwi.run.remote(wq) for fwi in fw])
  # A worker that processed no partitions reports None; drop those before combining.
  final_stats = combine_window_stats(
    [s for s in ray.get([fwi.report_statistics.remote() for fwi in fw]) if s is not None]
  )
  original_stats = combine_window_stats(
    [s for s in ray.get([fwi.report_original_statistics.remote() for fwi in fw]) if s is not None]
  )
  toc = time.time()

  # finally print flagging statistics
  for line in WindowStatistics.summarise_stats(final_stats, original_stats):
    print(line)

  elapsed = toc - tic
  print(
    "Data flagged successfully in {0:02.0f}h{1:02.0f}m{2:02.0f}s".format(
      (elapsed // 60) // 60, (elapsed // 60) % 60, elapsed % 60
    )
  )
