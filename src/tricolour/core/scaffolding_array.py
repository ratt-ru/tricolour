from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.stride_tricks import as_strided
from xarray.core.types import T_Chunks, T_DuckArray, T_NormalizedChunks
from xarray.namedarray._typing import _Chunks
from xarray.namedarray.parallelcompat import ChunkManagerEntrypoint

from tricolour.core.util import normalize_chunks

if TYPE_CHECKING:
  import numpy.typing as npt


class ScaffoldingArray:
  """A duck array that doesn't store any data.

  It exists as scaffolding used to represent chunked array metadata
  for the purpose of writing metadata to disk (i.e. during to_zarr calls).

  Actually trying to use it as a duck array will almost certainly result
  in failure: It should be used for writing metadata only.

  Example
  -------
    ds = xarray.Dataset({
      "A": (("x", "y", "z"), np.ones((10, 20, 30)),
      "B": (("x", "y"), np.zeros((10, 20))),
    })

    chunked_ds = ds.chunk(
      chunks={"x": 5, "y": 10, "z": 15},
      chunked_array_type=""tricolour:scaffolding"
    )

    chunked_ds.to_zarr("/tmp/output.zarr")
  """

  chunks: T_NormalizedChunks
  dtype: npt.DTypeLike

  def __init__(self, array: T_DuckArray, chunks: T_NormalizedChunks | None = None):
    self.chunks = normalize_chunks(chunks or array.shape, array.shape)
    self.dtype = array.dtype

  def __getitem__(self, key):
    raise NotImplementedError("Accessing ScaffoldingArray data")

  def __array_namespace__(self, *, api_version: str | None = None):
    # Presence makes xarray's is_duck_array recognise this as a duck array
    raise NotImplementedError("Calling array API methods on ScaffoldingArrays")

  @property
  def ndim(self) -> int:
    return len(self.chunks)

  @property
  def shape(self) -> tuple[int, ...]:
    return tuple(sum(c) for c in self.chunks)

  def rechunk(self, chunks):
    dummy = np.empty(1, dtype=self.dtype)
    view = as_strided(dummy, shape=self.shape, strides=(0,) * len(self.chunks))
    return ScaffoldingArray(view, chunks=chunks)


class ScaffoldingChunkManager(ChunkManagerEntrypoint):
  """"""

  def __init__(self):
    self.array_cls = ScaffoldingArray

  def is_chunked_array(self, data) -> bool:
    return isinstance(data, ScaffoldingArray)

  def chunks(self, data: ScaffoldingArray) -> T_NormalizedChunks:
    return data.chunks

  def normalize_chunks(
    self,
    chunks: T_Chunks | T_NormalizedChunks,
    shape: tuple[int, ...],
    limit: int | None = None,
    dtype: np.dtype | None = None,
    previous_chunks: T_NormalizedChunks | None = None,
  ) -> T_NormalizedChunks:
    if limit is not None:
      warnings.warn(f"limit {limit} ignored in normalize_chunks", UserWarning)

    return normalize_chunks(chunks, shape)

  def from_array(self, data: T_DuckArray | npt.ArrayLike, chunks: _Chunks, **kw) -> ScaffoldingArray:
    return ScaffoldingArray(data, chunks)

  def rechunk(self, data: ScaffoldingArray, chunks, **kwargs) -> ScaffoldingArray:
    return data.rechunk(chunks, **kwargs)

  def compute(self, *data: ScaffoldingArray, **kwargs) -> tuple[np.ndarray, ...]:
    raise NotImplementedError("Computing ScaffoldingArrays")

  def store(self, sources, targets, **kwargs):
    # Scaffolding only lays out the zarr structure (shape/chunks/dtype);
    # chunk data is never materialised or written, so storing is a no-op.
    return None

  def apply_gufunc(
    self,
    func,
    signature,
    *args,
    axes=None,
    axis=None,
    keepdims=False,
    output_dtypes=None,
    output_sizes=None,
    vectorize=None,
    allow_rechunk=False,
    meta=None,
    **kwargs,
  ):
    raise NotImplementedError("GUFuncs on ScaffoldingArrays")
