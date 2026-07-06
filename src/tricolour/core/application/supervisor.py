from __future__ import annotations

import asyncio
from collections import Counter, deque
from dataclasses import dataclass
from itertools import product, batched
from typing import AsyncGenerator, Deque, Dict, Generator, List, Tuple

from msv4_utils.msv4_types import VISIBILITY_XDS_TYPES
import ray
from rarg_python_patterns.multiton import Multiton
from rarg_ray_patterns.autoscaling import ActorAutoscaler, ActorSpec
from rarg_ray_patterns.utils import wrap_future
import xarray

from tricolour.core.application.flagger import Flagger, LoadResult

# Maximum futures in flight per worker: one executing plus one queued
# on the actor keeps it busy without hoarding work that a newly
# scaled-up worker could otherwise absorb.
OCCUPANCY = 2


@dataclass(slots=True)
class WorkItem:
  path: str
  region: Dict[str, slice]


@ray.remote
class Supervisor:
  """Distributes chunked work over an autoscaled deployment of Flaggers.

  Occupancy per worker is derived from the in-flight future map rather
  than tracked separately, so it cannot drift when workers scale up or
  down. Work whose worker died mid-flight is requeued and reassigned,
  making scale-down loss-free.
  """

  def __init__(
      self,
      datatree: Multiton[xarray.DataTree],
      time_chunks: int,
      frequency_chunks: int,
      nworkers: int,
      batch_size: int = 5
    ):

    self._datatree = datatree.instance
    self._time_chunks = time_chunks
    self._frequency_chunks = frequency_chunks
    self._batch_size = batch_size
    self._autoscaler = ActorAutoscaler(
      nworkers,
      actor_specs=[
        # The unresolved Multiton, not .instance: the spec is installed
        # from this actor's process and must stay Ray-serializable.
        ActorSpec(Flagger, args=(datatree,), options={"max_restarts": -1}),
      ],
    )
    # future -> (assigned node id, its work item). The single source of
    # truth: occupancy is counted from it and failed work recovered from it.
    self._futures: Dict[asyncio.Future, Tuple[str, WorkItem]] = {}
    self._retry_queue: Deque[WorkItem] = deque()
    # Deterministic load failures accumulated by the current run(), surfaced
    # to its caller instead of being silently dropped or retried forever.
    self._failures: List[Tuple[WorkItem, LoadResult]] = []

  def work_generator(self) -> Generator[WorkItem]:
    for path, node in self._datatree.children.items():
      if node.attrs.get("type") not in VISIBILITY_XDS_TYPES:
        continue

      ntime = node.sizes["time"]
      nfreq = node.sizes["frequency"]
      time_range = range(0, ntime, self._time_chunks)
      freq_range = range(0, nfreq, self._frequency_chunks)

      for t, f in product(time_range, freq_range):
        region = {
          "time": slice(t, min(t + self._time_chunks, ntime)),
          "frequency": slice(f, min(f + self._frequency_chunks, nfreq))
        }

        yield WorkItem(path, region)

  def _work_items(self) -> Generator[WorkItem]:
    """Yields requeued work ahead of fresh work from the generator."""
    generator = self.work_generator()
    while True:
      while self._retry_queue:
        yield self._retry_queue.popleft()
      try:
        yield next(generator)
      except StopIteration:
        return

  async def drain_futures(self, timeout: float | None = 0.2) -> AsyncGenerator:
    """Yields (item, result) for completed futures as they resolve.

    Three outcomes:
      - Load succeeded: ``result.ok`` is True, yielded.
      - Load failed deterministically: Flagger already caught the error
        into a LoadResult; ``result.ok`` is False, yielded but not
        requeued, since retrying a deterministic failure just repeats it.
      - Worker died mid-flight: the future itself raises RayActorError.
        Nothing is yielded; the item is requeued. Covers reaping by the
        autoscaler and Flagger restarts under max_restarts.

    With a timeout, returns once no future completes within it; with
    ``timeout=None``, runs until nothing remains in flight.
    """
    while self._futures:
      ready, _ = await asyncio.wait(
        self._futures.keys(),
        timeout=timeout,
        return_when=asyncio.FIRST_COMPLETED,
      )
      if not ready:
        return
      for future in ready:
        _, item = self._futures.pop(future)
        try:
          result = await future
        except ray.exceptions.RayActorError:
          self._retry_queue.append(item)
          continue
        yield item, result

  async def _acquire_worker(self) -> Tuple[str, ray.actor.ActorHandle]:
    """Returns a (node id, Flagger) with occupancy below OCCUPANCY.

    Blocks until such a worker exists: reads the live deployment view
    each attempt so scale-up is picked up immediately, and autoscales
    then drains while waiting so in-flight work keeps completing.
    """
    while True:
      if len(flaggers := self._autoscaler.deployments(Flagger)) > 0:
        occupancy = Counter(nid for nid, _ in self._futures.values())
        node_id = min(flaggers, key=lambda nid: occupancy[nid])
        if occupancy[node_id] < OCCUPANCY:
          return node_id, flaggers[node_id]
      await self._autoscaler.autoscale()
      async for item, result in self.drain_futures(timeout=0.5):
        if not result.ok:
          self._failures.append((item, result))

  async def _submit(self, item: WorkItem) -> None:
    node_id, flagger = await self._acquire_worker()
    ref = flagger.load_data.remote(item.path, item.region)
    self._futures[wrap_future(ref)] = (node_id, item)

  async def run(self) -> List[Tuple[WorkItem, LoadResult]]:
    """Runs to completion, returning deterministic load failures.

    Worker-crash retries refill the retry queue and are resubmitted
    transparently; deterministic load failures are collected instead and
    returned once nothing remains in flight or queued for retry.
    """
    self._failures = []

    for item in self._work_items():
      await self._submit(item)

    # Final drain: failures refill the retry queue, so alternate between
    # draining and resubmitting until both are empty.
    while self._futures or self._retry_queue:
      async for item, result in self.drain_futures(timeout=None):
        if not result.ok:
          self._failures.append((item, result))
      while self._retry_queue:
        await self._submit(self._retry_queue.popleft())

    return list(self._failures)

  async def close(self) -> None:
    await self._autoscaler.close()
