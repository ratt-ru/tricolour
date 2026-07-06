from dataclasses import dataclass

import ray
import xarray
from rarg_python_patterns.multiton import Multiton


@dataclass(slots=True)
class LoadResult:
  """Outcome of Flagger.load_data: exactly one of value/error is set."""
  value: xarray.DataTree | None = None
  error: Exception | None = None

  @property
  def ok(self) -> bool:
    return self.error is None


@ray.remote(concurrency_groups={"io-read": 1})
class Flagger:
  def __init__(self, dt: Multiton[xarray.DataTree]):
    self._dt = dt

  @ray.method(concurrency_group="io-read")
  def load_data(self, path, region) -> LoadResult:
    try:
      value = self._dt.instance[path].isel(**region).load()
    except Exception as exc:
      return LoadResult(error=exc)
    return LoadResult(value=value)