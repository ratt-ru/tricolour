from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from msv4_utils import MSv4Backend, infer_backend


@dataclass
class BackendImport:
  package: str
  install_option: str
  open_kwargs: Dict[str, Any]


BACKEND_MAP = {
  MSv4Backend.CASA_TABLE: BackendImport(
    "xarray_ms",
    "msv2",
    {
      "engine": "xarray-ms:msv2",
      "partition_schema": ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"],
    },
  ),
  MSv4Backend.MEERKAT: BackendImport(
    "xarray_kat",
    "meerkat",
    {
      "engine": "xarray-kat",
      "applycal": "all",
      "chunked_array_type": "xarray-kat",
      "chunks": {},
      "uvw_sign_convention": "casa",
    },
  ),
  MSv4Backend.ZARR: BackendImport(
    "zarr",
    "zarr",
    {
      "engine": "zarr",
      "chunks": None,
    },
  ),
}

SUPPORTS_WRITEBACK = {MSv4Backend.CASA_TABLE: True, MSv4Backend.MEERKAT: False, MSv4Backend.ZARR: True}


def infer_and_import_backend(uri: str) -> Tuple[MSv4Backend, Dict[str, Any]]:
  """Infers the xarray backend from `uri`, imports the required backend module.

  Args
  ----
    uri: Uniform Resource Indicator

  Returns
  -------
    A tuple (backend, open_kwargs) where `backend` is a backend enumeration
    and `open_kwargs` are the kwargs that should be passed through to
    `xarray.open_datatree`.
  """
  uri_backend = infer_backend(uri, strict=False)

  try:
    backend_import = BACKEND_MAP[uri_backend]
  except KeyError:
    raise ValueError(f"Unsupported MSv4 backend {uri} {uri_backend}")

  try:
    importlib.import_module(backend_import.package)
  except ImportError:
    raise ImportError(
      f"The {uri_backend.name} backend is not installed.\npip install tricolour[{backend_import.install_option}]"
    )

  return uri_backend, backend_import.open_kwargs
