import ray
import xarray
import os
import numpy as np
from tricolour.core.kernels.flagging import (
   sum_threshold_flagger,
   uvcontsub_flagger,
   flag_autos,
   flag_nans_and_zeros,
   apply_static_mask
)
from tricolour.core.kernels.stokes import (
    stokes_corr_map,
    polarised_intensity,
)
@ray.remote
class WorkQueue:
  def __init__(self):
    self._queue = []
  
  def enqueue_partition(self, partition):
    if not isinstance(partition, xarray.DataTree):
      raise TypeError("Expected an xarray.DataTree type")
    self._queue.append(partition)

  def dequeue(self, data_column):
    if self._queue:
      partition = self._queue.pop(0)
      vis_windows = getattr(partition, data_column).load()
      vis_windows = vis_windows.transpose("baseline_id","polarization","time","frequency").data
      flag_windows = partition.FLAG.load()
      flag_windows = flag_windows.transpose("baseline_id","polarization","time","frequency").data
      return vis_windows, flag_windows, partition
    else:
      return None, None, None

@ray.remote
class FlaggingWorker:
  def __init__(self,
               workqueue,
               masks,
               data_column,
               subtract_model_column,
               flagging_strategy,
               flagging_config):
    self.workqueue = workqueue
    self._masks = masks
    self._data_column = data_column
    self._subtract_model_column = subtract_model_column
    self._flagging_strategy = flagging_strategy
    self._config = flagging_config
    self._data = None
    self._flag = None
    self._partition = None

  def set_metadata(self):
    if self._partition:
      bl = self._partition.baseline_id.data
      self._ant_pos = self._partition["antenna_xds"].ANTENNA_POSITION.load().data
      mata1, mata2 = np.triu_indices(self._ant_pos.shape[0])
      a1 = mata1[bl]; a2 = mata2[bl]
      self._ubl = np.vstack([bl,a1,a2]).T
      self._chan_freq = self._partition.frequency.data
      self._chan_width = self._partition.frequency.channel_width['data'] * np.ones_like(self._chan_freq)
      self._data_loaded = True
      self._correlation_types = self._partition.polarization.data

  def exec_strategy(self):
    if self._data_loaded:
      from tricolour.core.kernels.stokes import STOKES_TYPES
      if self._flagging_strategy == "polarisation":
        stokes_map = stokes_corr_map([STOKES_TYPES[c] for c in self._correlation_types])
        stokes_pol = tuple(v for k, v in stokes_map.items() if k != "I")
        self._vis_windows = polarised_intensity(self._vis_windows, stokes_pol)
        self._flag_windows = np.any(self._flag_windows, axis=1, keepdims=True)        
      elif self._flagging_strategy == "total_power":
        stokes_map = stokes_corr_map([STOKES_TYPES[c] for c in self._correlation_types])
        stokes_pol = tuple(v for k, v in stokes_map.items())
        self._vis_windows = polarised_intensity(self._vis_windows, stokes_pol)
        self._flag_windows = np.any(self._flag_windows, axis=1, keepdims=True)
      elif self._flagging_strategy == "standard":
         pass
      else:
         raise ValueError("Expected one of 'polarisation', 'total_power' or 'standard' as strategy")
      original = self._flag_windows.copy()
      for strategy in self._config:
        try:
          task = strategy['task']
        except KeyError:
          raise ValueError("strategy has no 'task': %s" % strategy)
        
        if task == "sum_threshold":
            new_flags = sum_threshold_flagger(self._vis_windows, self._flag_windows,
                                              **strategy['kwargs'])
            self._flag_windows = np.logical_or(new_flags, self._flag_windows)
        elif task == "uvcontsub_flagger":
            new_flags = uvcontsub_flagger(self._vis_windows, self._flag_windows,
                                          **strategy['kwargs'])
            # this task discards previous flags by default during its
            # second iteration. The original flags from MS should be or'd
            # back in afterwards. Flags from steps prior to this one serves
            # only as a "initial guess"
            self._flag_windows = new_flags
        elif task == "flag_autos":
            new_flags = flag_autos(self._flag_windows, self._ubl)
            self._flag_windows = np.logical_or(new_flags, self._flag_windows)
        elif task == "combine_with_input_flags":
            self._flag_windows = np.logical_or(self._flag_windows, original)
        elif task == "unflag":
            self._flag_windows = np.zeros_like(self._flag_windows)
        elif task == "flag_nans_zeros":
            self._flag_windows = flag_nans_and_zeros(self._vis_windows, self._flag_windows)
        elif task == "apply_static_mask":
            new_flags = apply_static_mask(self._flag_windows,
                                          self._ubl,
                                          self._ant_pos,
                                          [self._masks[k] for k in self._masks.keys()],
                                          self._chan_freq,
                                          self._chan_width,
                                          **strategy['kwargs'])
            # override option will override any flags computed previously
            # this may not be desirable so use with care or in combination
            # with combine_with_input_flags option!
            if strategy['kwargs']["accumulation_mode"].strip() == "or":
                self._flag_windows = np.logical_or(new_flags, self._flag_windows)
            elif strategy['kwargs']["accumulation_mode"].strip() == "and":
                self._flag_windows = new_flags
            else:
               raise ValueError("Expected 'or' or 'and' in apply_static_mask.accumulation_mode")

        else:
            raise ValueError("Task '%s' does not name a valid task", task)
  
  def writeback(self):
    if self._flag_windows is not None and self._partition:
      # b, c, t, f -> t, b, f, c
      flT = np.transpose(self._flag_windows, axes=(2, 0, 3, 1))
      if self._flagging_strategy == "polarisation" or self._flagging_strategy == "total_power":
        flTbcast = np.zeros_like(self._partition.FLAG.data)
        for ci in range(self._partition.FLAG.data.shape[3]):
           flTbcast[:,:,:,ci] = flT[:,:,:,0]
      else:
        flTbcast = flT
      self._partition.FLAG.data = flTbcast
      ds = self._partition.dataset.drop_vars(list(filter(lambda k: k != "FLAG", 
                                                         self._partition.dataset.data_vars.keys())))
      ds.to_msv2(compute=True)

  def run(self, workqueue):
    self.work_item_ref = workqueue.dequeue.remote(self._data_column)
    while True:
      self._vis_windows, self._flag_windows, self._partition = ray.get(self.work_item_ref)
      if self._partition is None:
            break
      self.work_item_ref = workqueue.dequeue.remote(self._data_column)

      # process current chunk of data
      self.set_metadata()
      self.exec_strategy()
      self.writeback()

      