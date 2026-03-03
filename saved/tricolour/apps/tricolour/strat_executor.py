# -*- coding: utf-8 -*-

import dask.array as da

from tricolour.dask_wrappers import (sum_threshold_flagger,
                                     uvcontsub_flagger,
                                     flag_autos,
                                     flag_nans_and_zeros,
                                     apply_static_mask)


class StrategyExecutor(object):
    def __init__(self, antenna_positions, unique_baselines,
                 chan_freq, chan_width, masked_channels, strategies):

        self.ant_pos = antenna_positions
        self.ubl = unique_baselines
        self.chan_freq = chan_freq
        self.chan_width = chan_width
        self.masked_channels = masked_channels
        self.strategies = strategies

    def __enter__(self):
        return self

    def __exit__(self, etype, evalue, etraceback):
        pass

    def apply_strategies(self, flag_windows, vis_windows):
        original = flag_windows.copy()

        # Run flagger strategies
        for strategy in self.strategies:
            try:
                task = strategy['task']
            except KeyError:
                raise ValueError("strategy has no 'task': %s" % strategy)

            if task == "sum_threshold":
                new_flags = sum_threshold_flagger(vis_windows, flag_windows,
                                                  **strategy['kwargs'])
                # sum threshold builds upon any flags that came previous
                flag_windows = da.logical_or(new_flags, flag_windows)
            elif task == "uvcontsub_flagger":
                new_flags = uvcontsub_flagger(vis_windows, flag_windows,
                                              **strategy['kwargs'])
                # this task discards previous flags by default during its
                # second iteration. The original flags from MS should be or'd
                # back in afterwards. Flags from steps prior to this one serves
                # only as a "initial guess"
                flag_windows = new_flags
            elif task == "flag_autos":
                new_flags = flag_autos(flag_windows, self.ubl)
                flag_windows = da.logical_or(new_flags, flag_windows)
            elif task == "combine_with_input_flags":
                # or's in original flags from the measurement set
                # (if -if option has not been specified,
                # in which case this option will do nothing)
                flag_windows = da.logical_or(flag_windows, original)
            elif task == "unflag":
                flag_windows = da.zeros_like(flag_windows)
            elif task == "flag_nans_zeros":
                flag_windows = flag_nans_and_zeros(vis_windows, flag_windows)
            elif task == "apply_static_mask":
                new_flags = apply_static_mask(flag_windows,
                                              self.ubl,
                                              self.ant_pos,
                                              self.masked_channels,
                                              self.chan_freq,
                                              self.chan_width,
                                              **strategy['kwargs'])
                # override option will override any flags computed previously
                # this may not be desirable so use with care or in combination
                # with combine_with_input_flags option!
                if strategy['kwargs']["accumulation_mode"].strip() == "or":
                    flag_windows = da.logical_or(new_flags, flag_windows)
                else:
                    flag_windows = new_flags

            else:
                raise ValueError("Task '%s' does not name a valid task", task)

        return flag_windows
