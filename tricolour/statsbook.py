from tricolour import log
import numpy as np
import dask
import dask.array as da

class statsbook:
    def __init__(self):
        self._counts_per_ant = {}
        self._counts_per_field = {}
        self._counts_per_scan = {}
        self._size_per_ant = {}
        self._size_per_field = {}
        self._size_per_scan = {}

    def update(self, antname_map, flag_window, ubls, scan_no, field_name):
        dims = ("time", "chan", "bl", "corr")  # corrprod = ncorr * nbl
        def __update(antname_map, flag_window, ubls, scan_no, field_name):
            for ai, a in enumerate(antname_map):
                # per antenna
                sel = np.logical_or(ubls[0][:, 1] == ai,
                                    ubls[0][:, 2] == ai)
                cnt = np.sum(flag_window[:, :, sel, :])
                sz = flag_window[:, :, sel, :].size
                self._counts_per_ant[a] = self._counts_per_ant.get(a, 0) + cnt
                self._size_per_ant[a] = self._size_per_ant.get(a, 0) + sz

                # per scan and field
                cnt = np.sum(flag_window)
                sz = flag_window.size
                self._counts_per_field[field_name] = self._counts_per_field.get(field_name, 0) + cnt
                self._size_per_field[field_name] = self._size_per_field.get(field_name, 0) + sz
                self._counts_per_scan[scan_no] = self._counts_per_scan.get(scan_no, 0) + cnt
                self._size_per_scan[scan_no] = self._size_per_scan.get(scan_no, 0) + sz

            return flag_window
        return da.blockwise(__update, dims,
                            antname_map, None,
                            flag_window, dims,
                            ubls, ("bl", "bl-comp"),
                            scan_no, None,
                            field_name, None,
                            dtype=flag_window.dtype)


    def summarize(self, original):
        log.info("********************************")
        log.info("   BEGINNING OF FLAG SUMMARY    ")
        log.info("********************************")
        log.info("Per antenna:")
        for a in self._counts_per_ant:
            log.info("\t {0:s}: {1:.3f}%, original {2:.3f}%".format(a,
                    self._counts_per_ant[a] * 100.0 / self._size_per_ant[a],
                    original._counts_per_ant[a] * 100.0 / original._size_per_ant[a]))
        log.info("Per scan:")
        for s in self._counts_per_scan:
            log.info("\t {0:d}: {1:.3f}%, original {2:.3f}%".format(s,
                    self._counts_per_scan[s] * 100.0 / self._size_per_scan[s],
                    original._counts_per_scan[s] * 100.0 / original._size_per_scan[s]))
        log.info("Per field:")
        for f in self._counts_per_field:
            log.info("\t {0:s}: {1:.3f}%, original {2:.3f}%".format(f,
                    self._counts_per_field[f] * 100.0 / self._size_per_field[f],
                    original._counts_per_field[f] * 100.0 / original._size_per_field[f]))
        log.info("********************************")
        log.info("       END OF FLAG SUMMARY      ")
        log.info("********************************")

