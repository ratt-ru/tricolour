from __future__ import print_function
import os
import sys
import subprocess
import unittest
import time
from pyrap.tables import table as tbl
import numpy as np

TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", ".")

class tricolour_acceptance_test(unittest.TestCase):
    """
        Automated qualification test for the tricolour flagger
        Expects environment variable TEST_DATA_DIR set to folder containing
        MeerKAT acceptance_test_data.tar.gz prior to running
    """
    @classmethod
    def timeoutsecs(cls):
        return 250

    @classmethod
    def setUpClass(cls):
        unittest.TestCase.setUpClass()
        if TEST_DATA_DIR == ".":
            raise RuntimeError("Expected TEST_DATA_DIR to be set to directory containing acceptance_test_data.tar.gz")
        # remove if exists and extract data
        args = ['rm -rf 1519747221.subset.ms']
        subprocess.check_call(args, shell=True)
        args = ['tar -zxvf ' + os.path.join(TEST_DATA_DIR, 'acceptance_test_data.tar.gz')]
        subprocess.check_call(args, shell=True)
        # now run tricolour on a timer, stop if timeout
        this_test_path = os.path.dirname(__file__)
        args = ['tricolour',
                '-fs', 'polarisation',
                '-c', os.path.join(this_test_path, 'custom.yaml'),
                '1519747221.subset.ms']
        p = subprocess.Popen(args, env=os.environ.copy())
        x = cls.timeoutsecs()
        delay = 1.0
        timeout = int(x / delay)
        while p.poll() is None and timeout > 0:
            time.sleep(delay)
            timeout -= delay
        #timeout reached, kill process if it is still rolling
        ret = p.poll()
        if ret is None:
            p.kill()
            ret = 99
        if ret == 99:
            raise RuntimeError("Test timeout reached. Killed flagger")
        elif ret != 0:
            raise RuntimeError("Tricolour exited with non-zero return code")

    @classmethod
    def tearDownClass(cls):
        unittest.TestCase.tearDownClass()
        args = ['rm -rf 1519747221.subset.ms']
        subprocess.check_call(args, shell=True)

    def test_mean_chisq(cls, tol_unflagged_flagged_mean=1e3):
        """ Tests for improvement in mean chisq per correlation """
        with tbl("1519747221.subset.ms::FIELD") as t:
            fnames = t.getcol("NAME")
        with tbl("1519747221.subset.ms") as t:
            fid = t.getcol("FIELD_ID")
            flag = t.getcol("FLAG")
            data = t.getcol("DATA")
        sel = fid == fnames.index("3C286")
        chisq_unflagged_3c286 = np.nanmean(np.nanmean((np.abs(data[sel]) - np.nanmean(np.abs(data[sel]), 
                                                                          axis=0))**2,
                                                     axis=0), axis=0)
        print("Chi^2 unflagged 3C286:", ",".join(["{0:.3f}".format(f) for f in chisq_unflagged_3c286]))

        sel = fid == fnames.index("PKS1934-63")
        chisq_unflagged_1934 = np.nanmean(np.nanmean((np.abs(data[sel]) - np.nanmean(np.abs(data[sel]), 
                                                                                   axis=0))**2,
                                                     axis=0), axis=0)

        print("Chi^2 unflagged PKS B1934-638:", ",".join(["{0:.3f}".format(f) for f in chisq_unflagged_1934]))

        # flag data
        data[flag] = np.nan
        chisq_flagged_3c286 = np.nanmean(np.nanmean((np.abs(data[sel]) - np.nanmean(np.abs(data[sel]), 
                                                                          axis=0))**2,
                                                     axis=0), axis=0)
        print("Chi^2 flagged 3C286:", ",".join(["{0:.3f}".format(f) for f in chisq_flagged_3c286]))

        sel = fid == fnames.index("PKS1934-63")
        chisq_flagged_1934 = np.nanmean(np.nanmean((np.abs(data[sel]) - np.nanmean(np.abs(data[sel]), 
                                                                                   axis=0))**2,
                                                     axis=0), axis=0)

        print("Chi^2 flagged PKS B1934-638:", ",".join(["{0:.3f}".format(f) for f in chisq_flagged_1934]))
        assert(all(chisq_unflagged_3c286 > chisq_flagged_3c286 * tol_unflagged_flagged_mean))
        assert(all(chisq_unflagged_1934 > chisq_flagged_1934 * tol_unflagged_flagged_mean))


    def test_max_chisq(cls, tol_unflagged_flagged_mean=1e4):
        """ Tests for improvement in max chisq per correlation """
        with tbl("1519747221.subset.ms::FIELD") as t:
            fnames = t.getcol("NAME")
        with tbl("1519747221.subset.ms") as t:
            fid = t.getcol("FIELD_ID")
            flag = t.getcol("FLAG")
            data = t.getcol("DATA")
        sel = fid == fnames.index("3C286")
        chisq_unflagged_3c286 = np.nanmax(np.nanmax((np.abs(data[sel]) - np.nanmean(np.abs(data[sel]), 
                                                                          axis=0))**2,
                                                     axis=0), axis=0)
        print("Max Chi^2 unflagged 3C286:", ",".join(["{0:.3f}".format(f) for f in chisq_unflagged_3c286]))

        sel = fid == fnames.index("PKS1934-63")
        chisq_unflagged_1934 = np.nanmax(np.nanmax((np.abs(data[sel]) - np.nanmean(np.abs(data[sel]), 
                                                                                   axis=0))**2,
                                                     axis=0), axis=0)

        print("Max Chi^2 unflagged PKS B1934-638:", ",".join(["{0:.3f}".format(f) for f in chisq_unflagged_1934]))

        # flag data
        data[flag] = np.nan
        chisq_flagged_3c286 = np.nanmax(np.nanmax((np.abs(data[sel]) - np.nanmean(np.abs(data[sel]), 
                                                                          axis=0))**2,
                                                     axis=0), axis=0)
        print("Max Chi^2 flagged 3C286:", ",".join(["{0:.3f}".format(f) for f in chisq_flagged_3c286]))

        sel = fid == fnames.index("PKS1934-63")
        chisq_flagged_1934 = np.nanmax(np.nanmax((np.abs(data[sel]) - np.nanmean(np.abs(data[sel]), 
                                                                                   axis=0))**2,
                                                     axis=0), axis=0)

        print("Max Chi^2 flagged PKS B1934-638:", ",".join(["{0:.3f}".format(f) for f in chisq_flagged_1934]))
        assert(all(chisq_unflagged_3c286 > chisq_flagged_3c286 * tol_unflagged_flagged_mean))
        assert(all(chisq_unflagged_1934 > chisq_flagged_1934 * tol_unflagged_flagged_mean))

    def test_flag_count(cls, tol_amount_flagged=0.65):
        """ Tests for flag count less than tolerance """
        with tbl("1519747221.subset.ms::FIELD") as t:
            fnames = t.getcol("NAME")
        with tbl("1519747221.subset.ms") as t:
            fid = t.getcol("FIELD_ID")
            flag = t.getcol("FLAG")
            data = t.getcol("DATA")
        sel = fid == fnames.index("3C286")
        count_flagged_3c286 = np.nansum(np.nansum(flag[sel], axis=0), axis=0)
        print("Tot flagged for 3C286:", np.nansum(count_flagged_3c286) / float(flag[sel].size) * 100.0, "%")
        assert(np.nansum(count_flagged_3c286) / float(flag[sel].size) < tol_amount_flagged)
        sel = fid == fnames.index("PKS1934-63")
        count_flagged_1934 = np.nansum(np.nansum(flag[sel], axis=0), axis=0)
        print("Tot flagged for PKS B1934-63:", np.nansum(count_flagged_1934) / float(flag[sel].size) * 100.0, "%")
        assert(np.nansum(count_flagged_1934) / float(flag[sel].size) < tol_amount_flagged)

    def test_bandwidth_flagged(cls, tol_amount_flagged=0.40):
        """ Tests for total bandwidth flagged less than tolerance """
        with tbl("1519747221.subset.ms::FIELD") as t:
            fnames = t.getcol("NAME")
        with tbl("1519747221.subset.ms") as t:
            fid = t.getcol("FIELD_ID")
            flag = t.getcol("FLAG")
            data = t.getcol("DATA")
        sel = fid == fnames.index("3C286")
        count_flagged_3c286 = np.nansum(data[sel][:, :, 0], axis=0) > 0
        print("Tot bandwidth flagged for 3C286:", np.nansum(count_flagged_3c286) / float(data[sel].shape[1]) * 100.0, "%")
        assert(np.nansum(count_flagged_3c286) / float(data[sel].shape[1]) < tol_amount_flagged)
        sel = fid == fnames.index("PKS1934-63")
        count_flagged_1934 = np.nansum(data[sel][:, :, 0], axis=0) > 0
        print("Tot bandwidth flagged for PKS B1934-63:", np.nansum(count_flagged_1934) / float(flag[sel].shape[1]) * 100.0, "%")
        assert(np.nansum(count_flagged_1934) / float(data[sel].shape[1]) < tol_amount_flagged)

if __name__ == '__main__':
    unittest.main()
