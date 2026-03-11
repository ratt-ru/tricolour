import ray
import xarray
import os

@ray.remote
class WorkQueue:
  def __init__(self):
    self._queue = []
  
  def enqueue_partition(self, partition):
    if not isinstance(partition, xarray.DataTree):
      raise TypeError("Expected an xarray.DataTree type")
    self._queue.append(partition)
  
  def dequeue(self):
    if self._queue:
      return self._queue.pop(0)
    else:
      return None

@ray.remote
class FlaggingWorker:
  def __init__(self,
               dilate_masks,
               data_column,
               subtract_model_column,
               flagging_strategy,
               flagging_config):
    self._dilate_masks = dilate_masks
    self._data_column = data_column
    self._subtract_model_column = subtract_model_column
    self._flagging_strategy = flagging_strategy
    self._config = flagging_config
  
  def _exec_strategy(self):
    pass
  def run(self, workqueue):
    while True:
      partition = ray.get(workqueue.dequeue.remote())
      if not partition:
        break
      
      data = getattr(partition, self._data_column).load()
      flag = partition.FLAG.load()

      