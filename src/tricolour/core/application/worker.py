import numpy as np
import ray
import xarray

from tricolour.core.kernels.flag_statistics import WindowStatistics, combine_window_stats, window_stats
from tricolour.core.kernels.flagging import (
  apply_static_mask,
  flag_autos,
  flag_nans_and_zeros,
  sum_threshold_flagger,
  uvcontsub_flagger,
)
from tricolour.core.kernels.stokes import (
  polarised_intensity,
  stokes_corr_map,
)

STATISTICS_CHAN_BINS = 10


@ray.remote
class WorkQueue:
  def __init__(self):
    self._queue = []

  def enqueue_partition(self, partition, region):
    if not isinstance(partition, xarray.DataTree):
      raise TypeError("Expected an xarray.DataTree type")
    self._queue.append((partition, region))

  def dequeue(self, data_column, model_column=None):
    if self._queue:
      partition, region = self._queue.pop(0)
      vis_windows = getattr(partition, data_column).load()
      vis_windows = vis_windows.transpose("baseline_id", "polarization", "time", "frequency").data
      flag_windows = partition.FLAG.load()
      flag_windows = flag_windows.transpose("baseline_id", "polarization", "time", "frequency").data
      if model_column is not None:
        model_windows = getattr(partition, model_column).load()
        model_windows = model_windows.transpose("baseline_id", "polarization", "time", "frequency").data
      else:
        model_windows = None
      return vis_windows, flag_windows, model_windows, partition, region
    else:
      return None, None, None, None, None


@ray.remote
class FlaggingWorker:
  def __init__(self, workqueue, masks, data_column, subtract_model_column, flagging_strategy, flagging_config):
    self.workqueue = workqueue
    self._masks = masks
    self._data_column = data_column
    self._subtract_model_column = subtract_model_column
    self._flagging_strategy = flagging_strategy
    self._config = flagging_config
    self._data_windows = None
    self._model_windows = None
    self._flag_windows = None
    self._partition = None
    self._region = None
    self._statistics = dict()
    self._original_statistics = dict()

  def report_statistics(self):
    obs = []
    for fi in self._statistics:
      for ddid in self._statistics[fi]:
        for si in self._statistics[fi][ddid]:
          obs.append(self._statistics[fi][ddid][si])
    return combine_window_stats(obs)

  def report_original_statistics(self):
    obs = []
    for fi in self._original_statistics:
      for ddid in self._original_statistics[fi]:
        for si in self._original_statistics[fi][ddid]:
          obs.append(self._original_statistics[fi][ddid][si])
    return combine_window_stats(obs)

  def set_metadata(self):
    if self._partition:
      bl = self._partition.baseline_id.data
      self._ant_pos = self._partition["antenna_xds"].ANTENNA_POSITION.load().data
      mata1, mata2 = np.triu_indices(self._ant_pos.shape[0])
      a1 = mata1[bl]
      a2 = mata2[bl]
      self._ubl = np.vstack([bl, a1, a2]).T
      self._chan_freq = self._partition.frequency.data
      self._chan_width = self._partition.frequency.channel_width["data"] * np.ones_like(self._chan_freq)
      self._data_loaded = True
      self._correlation_types = self._partition.polarization.data
      self._antenna_names = list(self._partition["antenna_xds"].antenna_name.data)
      self._scan_numbers = list(np.unique(self._partition.scan_name.data))
      self._field_names = list(np.unique(self._partition.field_name))
      self._ddid_name = self._partition.frequency.spectral_window_name
      for stats in [self._original_statistics, self._statistics]:
        for fi in self._field_names:
          stats.setdefault(fi, dict())
          stats[fi].setdefault(self._ddid_name, dict())
          for si in self._scan_numbers:
            stats[fi][self._ddid_name].setdefault(si, WindowStatistics(STATISTICS_CHAN_BINS))

  def exec_strategy(self):
    def __update_stats(dico_stats):
      for fi in self._field_names:
        ds = self._partition.sel(field_name=fi)
        for si in self._scan_numbers:
          flag_window = ds.FLAG.where(ds.scan_name == si)
          stats = window_stats(
            flag_window.data,
            ubls=self._ubl,
            chan_freqs=self._chan_freq,
            antenna_names=self._antenna_names,
            scan_no=si,
            field_name=fi,
            ddid=self._ddid_name,
            nchanbins=STATISTICS_CHAN_BINS,
          )
          dico_stats[fi][self._ddid_name][si].update(stats)

    if self._data_loaded:
      # update original flag statistics per field, scan and spwid
      __update_stats(self._original_statistics)

      from tricolour.core.kernels.stokes import STOKES_TYPES

      if self._model_windows is not None:
        self._vis_windows = self._vis_windows - self._model_windows
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
          task = strategy["task"]
        except KeyError:
          raise ValueError("strategy has no 'task': %s" % strategy)

        if task == "sum_threshold":
          new_flags = sum_threshold_flagger(self._vis_windows, self._flag_windows, **strategy["kwargs"])
          self._flag_windows = np.logical_or(new_flags, self._flag_windows)
        elif task == "uvcontsub_flagger":
          new_flags = uvcontsub_flagger(self._vis_windows, self._flag_windows, **strategy["kwargs"])
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
          new_flags = apply_static_mask(
            self._flag_windows,
            self._ubl,
            self._ant_pos,
            [self._masks[k] for k in self._masks.keys()],
            self._chan_freq,
            self._chan_width,
            **strategy["kwargs"],
          )
          # override option will override any flags computed previously
          # this may not be desirable so use with care or in combination
          # with combine_with_input_flags option!
          if strategy["kwargs"]["accumulation_mode"].strip() == "or":
            self._flag_windows = np.logical_or(new_flags, self._flag_windows)
          elif strategy["kwargs"]["accumulation_mode"].strip() == "and":
            self._flag_windows = new_flags
          else:
            raise ValueError("Expected 'or' or 'and' in apply_static_mask.accumulation_mode")

        else:
          raise ValueError("Task '%s' does not name a valid task", task)

      # transpose back: b, c, t, f -> t, b, f, c
      flT = np.transpose(self._flag_windows, axes=(2, 0, 3, 1))
      if self._flagging_strategy == "polarisation" or self._flagging_strategy == "total_power":
        flTbcast = np.zeros_like(self._partition.FLAG.data)
        for ci in range(self._partition.FLAG.data.shape[3]):
          flTbcast[:, :, :, ci] = flT[:, :, :, 0]
      else:
        flTbcast = flT
      self._partition.FLAG.data = flTbcast

      # update flagging stats per field, scan and spwid
      __update_stats(self._statistics)

  def writeback(self):
    if self._flag_windows is not None and self._partition:
      ds = self._partition.dataset.drop_vars(filter(lambda k: k != "FLAG", self._partition.dataset.data_vars.keys()))

      ds.to_msv2(compute=True, region=self._region)

  def run(self, workqueue):
    self.work_item_ref = workqueue.dequeue.remote(self._data_column, model_column=self._subtract_model_column)
    while True:
      self._vis_windows, self._flag_windows, self._model_windows, self._partition, self._region = ray.get(
        self.work_item_ref
      )
      if self._partition is None:
        break
      self.work_item_ref = workqueue.dequeue.remote(self._data_column, model_column=self._subtract_model_column)

      # process current chunk of data
      self.set_metadata()
      self.exec_strategy()
      self.writeback()
