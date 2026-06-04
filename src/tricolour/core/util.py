# -*- coding: utf-8 -*-

import numbers
import re
from argparse import ArgumentTypeError
from typing import Iterable

import numpy as np
from xarray.core.types import T_NormalizedChunks


def normalize_chunks(chunks: Iterable[int | Iterable[int]], shape: Iterable[int]) -> T_NormalizedChunks:
  """
  Normalizes chunk sizes for an N-dimensional array based on a target shape.

  For each dimension:
  1. If an integer is provided, the dimension is split into uniform chunks of that
     size (the last chunk may be smaller to fit the shape).
  2. If an iterable is provided:
     - If the chunks sum to less than the shape, the last chunk value is repeated
       until the shape is covered.
     - If the chunks sum to more than the shape, the chunks are truncated at the
       point where the shape limit is reached.

  Args:
      chunks: An iterable of length L, where L is the rank of the shape.
              Each element specifies chunking for that dimension.
      shape: An iterable of integers representing the full size of each dimension.

  Returns:
      A tuple of tuples, where each inner tuple contains the explicit
      integer sizes for chunks in that dimension.
  """
  shape = tuple(int(d) if isinstance(d, numbers.Integral) else d for d in shape)
  if not all(isinstance(d, int) for d in shape):
    raise TypeError(f"shape {shape} must be an Iterable[int]")

  chunks_list = list(chunks)
  if len(chunks_list) != len(shape):
    raise TypeError(f"chunks length {len(chunks_list)} must match shape length {len(shape)}")

  normalized = []

  for c, s in zip(chunks_list, shape):
    # Case 1: Uniform integer chunk size
    if isinstance(c, numbers.Integral):
      c = int(c)
      if c <= 0:
        raise ValueError(f"Chunk size must be greater than 0, got {c}")

      if s == 0:
        dim_chunks = []
      else:
        n, rem = divmod(s, c)
        dim_chunks = [c] * n
        if rem > 0:
          dim_chunks.append(rem)
      normalized.append(tuple(dim_chunks))

    # Case 2: Explicit iterable of chunks
    elif isinstance(c, Iterable):
      c_list = [int(x) for x in c]
      if any(x <= 0 for x in c_list):
        raise ValueError(f"Explicit chunk sizes must be > 0: {c_list}")

      dim_chunks = []
      current_sum = 0

      if s == 0:
        normalized.append(())
        continue

      # Handle empty iterable for a positive shape (default to one big chunk)
      if not c_list:
        normalized.append((s,))
        continue

      # Iterate through provided chunks
      for chunk_val in c_list:
        if current_sum + chunk_val >= s:
          # Overfill/Exact case: Truncate and finish
          remaining = s - current_sum
          if remaining > 0:
            dim_chunks.append(remaining)
          current_sum = s
          break
        else:
          dim_chunks.append(chunk_val)
          current_sum += chunk_val

      # Underfill case: Use the last given chunk value to finish covering the shape
      if current_sum < s:
        fill_val = c_list[-1]
        while current_sum < s:
          next_chunk = min(fill_val, s - current_sum)
          dim_chunks.append(next_chunk)
          current_sum += next_chunk

      normalized.append(tuple(dim_chunks))

    else:
      raise TypeError(f"Invalid chunk type: {type(c)}. Expected int or Iterable.")

  return tuple(normalized)


def aggregate_chunks(chunks, max_chunks, return_groups=False):
  """
  Aggregate dask ``chunks`` together into chunks no larger than
  ``max_chunks``.

  .. code-block:: python

      chunks, max_c = ((3,4,6,3,6,7),(1,1,1,1,1,1)), (10,3)
      expected = ((7,9,6,7), (2,2,1,1))
      assert aggregate_chunks(chunks, max_c) == expected


  Parameters
  ----------
  chunks : sequence of tuples or tuple
  max_chunks : sequence of ints or int
  return_groups : bool

  Returns
  -------
  sequence of tuples or tuple

  """

  if isinstance(max_chunks, int):
    chunks = (chunks,)
    max_chunks = (max_chunks,)

  singleton = True if len(max_chunks) == 1 else False

  if len(chunks) != len(max_chunks):
    raise ValueError("len(chunks) != len(max_chunks)")

  if not all(len(chunks[0]) == len(c) for c in chunks):
    raise ValueError("Number of chunks do not match")

  agg_chunks = [[] for _ in max_chunks]
  agg_chunk_counts = [0] * len(max_chunks)
  chunk_scratch = [0] * len(max_chunks)
  ndim = len(chunks[0])

  # For each chunk dimension
  for di in range(ndim):
    # For each chunk
    aggregate = False

    for ci, chunk in enumerate(chunks):
      chunk_scratch[ci] = agg_chunk_counts[ci] + chunk[di]
      if chunk_scratch[ci] > max_chunks[ci]:
        aggregate = True

    if aggregate:
      for ci, chunk in enumerate(chunks):
        agg_chunks[ci].append(agg_chunk_counts[ci])
        agg_chunk_counts[ci] = chunk[di]
    else:
      for ci, chunk in enumerate(chunks):
        agg_chunk_counts[ci] = chunk_scratch[ci]

  # Do the final aggregation
  for ci, chunk in enumerate(chunks):
    agg_chunks[ci].append(agg_chunk_counts[ci])
    agg_chunk_counts[ci] = chunk[di]

  agg_chunks = tuple(tuple(ac) for ac in agg_chunks)

  return agg_chunks[0] if singleton else agg_chunks


def casa_style_range(val, argparse=False, opt_unit="m"):
  """returns list of floats"""
  RangeException = ArgumentTypeError if argparse else ValueError  # noqa: N806

  if not isinstance(val, str):
    raise RangeException("Value must be a string")
  if val.strip() == "" or val.strip() == "*":
    return (0, np.inf)
  elif re.match(
    r"^(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?~"
    r"(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?[\s]*[" + opt_unit + "]?$",
    val,
  ):
    val = val.replace(" ", "").replace("\t", "")
    for u in opt_unit:
      val = val.replace(opt_unit, "")
    vals = list(map(float, val.split("~")))
    return vals
  else:
    raise RangeException("Value must be range or blank")


def casa_style_int_list(val, argparse=False, opt_unit="m"):
  """returns list of ints"""
  RangeException = ArgumentTypeError if argparse else ValueError  # noqa: N806
  if val.strip() == "" or val.strip() == "*":
    return None
  elif re.match(r"^(\d+)(~\d+[" + opt_unit + r"]?)?(,(\d+)(~\d+[" + opt_unit + r"]?)?)*$", val):
    val = val.replace(" ", "").replace("\t", "")
    for u in opt_unit:
      val = val.replace(opt_unit, "")
    vals = val.split(",")
    range_vals = list(filter(lambda x: "~" in x, vals))
    list_vals = filter(lambda x: "~" not in x, vals)
    list_vals = list(map(int, list_vals))
    range_vals = [tuple(map(int, v.split("~"))) for v in range_vals]
    range_vals = [np.arange(rmin, rmax + 1) for rmin, rmax in range_vals]
    range_vals_flat = []
    for r in range_vals:
      range_vals_flat += list(r)
    vals = list(set(range_vals_flat + list_vals))
    return vals
  else:
    raise RangeException("Value must be range, comma list or blank")
