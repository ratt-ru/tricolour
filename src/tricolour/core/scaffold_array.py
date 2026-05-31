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


class ScaffoldArray:
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

  @classmethod
  def _empty(cls, chunks: T_NormalizedChunks, dtype: npt.DTypeLike) -> ScaffoldArray:
    """Build a scaffold of the given chunking/dtype without allocating data.

    A zero-strided view over a one-element dummy buffer presents the correct
    shape and dtype while backing storage stays a single element.
    """
    shape = tuple(sum(c) for c in chunks)
    dummy = np.empty(1, dtype=dtype)
    view = as_strided(dummy, shape=shape, strides=(0,) * len(shape))
    return cls(view, chunks=chunks)

  def __getitem__(self, key) -> ScaffoldArray:
    """This method shouldn't be called for ScaffoldArray's use case.

    Raises
    ------
      NotImplementedError
    """
    raise NotImplementedError("ScaffoldArray.__getitem__")

  def __array_namespace__(self, *, api_version: str | None = None):
    """Mark this object as an array-API duck array for xarray.

    xarray's ``is_duck_array`` check only tests that this method *exists*,
    so its mere presence is enough; actually invoking it (i.e. trying to
    run array operations) is unsupported and raises.
    """
    raise NotImplementedError("Calling array API methods on ScaffoldArrays")

  @property
  def ndim(self) -> int:
    """Number of dimensions, inferred from the chunk specification."""
    return len(self.chunks)

  @property
  def shape(self) -> tuple[int, ...]:
    """Full array shape, the sum of chunk sizes along each dimension."""
    return tuple(sum(c) for c in self.chunks)

  def rechunk(self, chunks):
    """Return a new scaffold with the same shape/dtype but new chunking."""
    return self._empty(normalize_chunks(chunks, self.shape), self.dtype)


class ScaffoldChunkManager(ChunkManagerEntrypoint):
  """xarray chunk manager backend for :class:`ScaffoldArray`.

  Implements the :class:`~xarray.namedarray.parallelcompat.ChunkManagerEntrypoint`
  interface so xarray can treat :class:`ScaffoldArray` as a chunked array
  type. Registered under the ``"tricolour:scaffolding"`` entry point in the
  ``xarray.chunkmanagers`` group, and selected via the ``chunked_array_type``
  argument to :meth:`xarray.Dataset.chunk`.

  Only the methods needed to *describe* and *store the structure of* a chunked
  array are meaningfully implemented; operations that would require real data
  (``compute``, ``apply_gufunc``) raise :class:`NotImplementedError`.
  """

  def __init__(self):
    self.array_cls = ScaffoldArray

  def is_chunked_array(self, data) -> bool:
    """Return whether ``data`` is a :class:`ScaffoldArray`."""
    return isinstance(data, ScaffoldArray)

  def chunks(self, data: ScaffoldArray) -> T_NormalizedChunks:
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

  def from_array(self, data: T_DuckArray | npt.ArrayLike, chunks: _Chunks, **kw) -> ScaffoldArray:
    """Wrap ``data`` in a :class:`ScaffoldArray` with the given chunking."""
    return ScaffoldArray(data, chunks)

  def rechunk(self, data: ScaffoldArray, chunks, **kwargs) -> ScaffoldArray:
    """Return a re-chunked scaffold (see :meth:`ScaffoldArray.rechunk`)."""
    return data.rechunk(chunks, **kwargs)

  def compute(self, *data: ScaffoldArray, **kwargs) -> tuple[np.ndarray, ...]:
    """Unsupported: a scaffold has no data to compute."""
    raise NotImplementedError("Computing ScaffoldArrays")

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
    raise NotImplementedError("GUFuncs on ScaffoldArrays")
