import importlib
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict
from msv4_utils import MSv4Backend, infer_backend
import xarray
from tricolour.core.util import casa_style_int_list, casa_style_range
import numpy as np

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
    partition_schema=["FIELD_ID", "SCAN_NUMBER"] + DEFAULT_PARTITION_COLUMNS
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



def driver(cfg: Namespace):
  partitions = load_partitions(cfg)  