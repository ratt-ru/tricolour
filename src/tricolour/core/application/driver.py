import importlib
from argparse import Namespace
from dataclasses import dataclass
from typing import Dict

from msv4_utils import MSv4Backend, infer_backend


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


def driver(cfg: Namespace):
  source_backend = infer_and_import_backend(cfg.ms)

  print(f"Complete me: Flag {cfg.ms} of backend type {source_backend}")
