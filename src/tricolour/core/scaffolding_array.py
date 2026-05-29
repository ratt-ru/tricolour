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
  """A "duck array" that carries shape, dtype and chunk metadata but no data.

  Its sole purpose is to stand in for a real array while writing the
  *structure* of a Zarr store to disk -- i.e. during ``Dataset.to_zarr``
  calls. Because it only describes metadata (shape, dtype and chunking),
  no chunk data is ever materialised or written when it is stored.

  Attempting to use it as a genuine array (indexing it, running array API
  operations on it, computing it) raises :class:`NotImplementedError` by
  design -- it must only be used for laying out metadata.

  The intended workflow is two-phase (see
  ``tests/test_scaffolding_array.py`` for a full example):

  1. **Scaffold.** Chunk a dataset with this backend and call ``to_zarr``.
     This writes the store's shape/chunks/dtype and coordinates, but
     writes no chunk data for the scaffolded data variables.
  2. **Fill.** Write the real data back into the empty store one region at
     a time using ``to_zarr(..., region=...)``, where each region matches
     the scaffold chunking.

  The backend is registered as the ``"tricolour:scaffolding"`` xarray chunk
  manager entry point (see ``pyproject.toml``), so it is selected via the
  ``chunked_array_type`` argument to :meth:`xarray.Dataset.chunk`.

  Example
  -------
  .. code-block:: python

      ds = xarray.Dataset({
        "A": (("x", "y", "z"), np.ones((10, 20, 30))),
        "B": (("x", "y"), np.zeros((10, 20))),
      })

      chunked_ds = ds.chunk(
        chunks={"x": 5, "y": 10, "z": 15},
        chunked_array_type="tricolour:scaffolding",
      )

      chunked_ds.to_zarr("/tmp/output.zarr")

  Attributes
  ----------
  chunks : T_NormalizedChunks
    Normalised chunking as a tuple of per-dimension chunk-size tuples.
  dtype : npt.DTypeLike
    The dtype of the array being scaffolded.
  """

  chunks: T_NormalizedChunks
  dtype: npt.DTypeLike

  def __init__(self, array: T_DuckArray, chunks: _Chunks | None = None):
    """Construct a scaffold from a source array's shape and dtype.

    Parameters
    ----------
    array
      The source (duck) array. Only its ``shape`` and ``dtype`` are read;
      its data is never accessed.
    chunks
      Desired chunking, in any form accepted by
      :func:`tricolour.core.util.normalize_chunks`. When ``None``, the
      array's shape is used, yielding a single chunk per dimension.
    """
    self.chunks = normalize_chunks(array.shape if chunks is None else chunks, array.shape)
    self.dtype = array.dtype

  def __getitem__(self, key):
    """Indexing is unsupported -- a scaffold holds no data."""
    raise NotImplementedError("Accessing ScaffoldingArray data")

  def __array_namespace__(self, *, api_version: str | None = None):
    """Mark this object as an array-API duck array for xarray.

    xarray's ``is_duck_array`` check only tests that this method *exists*,
    so its mere presence is enough; actually invoking it (i.e. trying to
    run array operations) is unsupported and raises.
    """
    raise NotImplementedError("Calling array API methods on ScaffoldingArrays")

  @property
  def ndim(self) -> int:
    """Number of dimensions, inferred from the chunk specification."""
    return len(self.chunks)

  @property
  def shape(self) -> tuple[int, ...]:
    """Full array shape, the sum of chunk sizes along each dimension."""
    return tuple(sum(c) for c in self.chunks)

  def rechunk(self, chunks):
    """Return a new scaffold with the same shape/dtype but new chunking.

    A zero-strided view over a one-element dummy buffer is used to present
    the correct shape and dtype to the new scaffold without allocating any
    real backing storage.
    """
    dummy = np.empty(1, dtype=self.dtype)
    view = as_strided(dummy, shape=self.shape, strides=(0,) * len(self.chunks))
    return ScaffoldingArray(view, chunks=chunks)


class ScaffoldingChunkManager(ChunkManagerEntrypoint):
  """xarray chunk manager backend for :class:`ScaffoldingArray`.

  Implements the :class:`~xarray.namedarray.parallelcompat.ChunkManagerEntrypoint`
  interface so xarray can treat :class:`ScaffoldingArray` as a chunked array
  type. Registered under the ``"tricolour:scaffolding"`` entry point in the
  ``xarray.chunkmanagers`` group, and selected via the ``chunked_array_type``
  argument to :meth:`xarray.Dataset.chunk`.

  Only the methods needed to *describe* and *store the structure of* a chunked
  array are meaningfully implemented; operations that would require real data
  (``compute``, ``apply_gufunc``) raise :class:`NotImplementedError`.
  """

  def __init__(self):
    self.array_cls = ScaffoldingArray

  def is_chunked_array(self, data) -> bool:
    """Return whether ``data`` is a :class:`ScaffoldingArray`."""
    return isinstance(data, ScaffoldingArray)

  def chunks(self, data: ScaffoldingArray) -> T_NormalizedChunks:
    """Return the normalised chunking of a scaffold."""
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
    """Wrap ``data`` in a :class:`ScaffoldingArray` with the given chunking."""
    return ScaffoldingArray(data, chunks)

  def rechunk(self, data: ScaffoldingArray, chunks, **kwargs) -> ScaffoldingArray:
    """Return a re-chunked scaffold (see :meth:`ScaffoldingArray.rechunk`)."""
    return data.rechunk(chunks, **kwargs)

  def compute(self, *data: ScaffoldingArray, **kwargs) -> tuple[np.ndarray, ...]:
    """Unsupported: a scaffold has no data to compute."""
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
    """Unsupported: scaffolds carry no data to apply functions over."""
    raise NotImplementedError("GUFuncs on ScaffoldingArrays")
