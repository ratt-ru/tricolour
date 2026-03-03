# -*- coding: utf-8 -*-
"""
Automated qualification test for the tricolour flagger
Expects environment variable TRICOLOUR_TEST_MS set to
MeerKAT acceptance_test_data.tar.gz prior to running
"""

import os
from os.path import join as pjoin
import shutil
import subprocess
import tarfile
import time

from pyrap.tables import table as tbl
import numpy as np
import gdown
import pytest

_GOOGLE_FILE_ID = "1yxDIXUo3Xun9WXxA0x_hvX9Fmxo9Igpr"
_MS_FILENAME = '1519747221.subset.ms'


def _download_file_from_google_drive(id, destination):
    URL = f"https://drive.google.com/uc?id={_GOOGLE_FILE_ID}"
    gdown.download(URL, destination, quiet=False)


# Set timeout to 6 minutes
@pytest.fixture(params=[360], scope="module")
def flagged_ms(request, tmp_path_factory):
    """
    fixture yielding an MS flagged by tricolour
    """
    tmp_path = str(tmp_path_factory.mktemp('data'))
    test_directory = os.path.dirname(__file__)
    try:
        ms_filename = os.environ["TRICOLOUR_TEST_MS"]
    except KeyError:
        tar_dir = tmp_path_factory.mktemp("tar-download")
        tarred_ms_filename = os.path.join(tar_dir, "test_data.tar.gz")
        _download_file_from_google_drive(_GOOGLE_FILE_ID, tarred_ms_filename)
        # Open and extract tarred ms
        tarred_ms = tarfile.open(tarred_ms_filename)
        tarred_ms.extractall(tmp_path)
        ms_filename = pjoin(tmp_path, _MS_FILENAME)

    args = ['tricolour',
            '-fs', 'total_power',
            '-c', os.path.join(test_directory, 'custom.yaml'),
            ms_filename]

    # Flag the MS, waiting for timeout period to expre
    p = subprocess.Popen(args, env=os.environ.copy())
    delay = 1.0
    timeout = int(request.param / delay)

    while p.poll() is None and timeout > 0:
        time.sleep(delay)
        timeout -= delay

    # timeout reached, kill process if it is still rolling
    ret = p.poll()

    if ret is None:
        p.kill()
        ret = 99

    if ret == 99:
        raise RuntimeError("Test timeout reached. Killed flagger")
    elif ret != 0:
        raise RuntimeError("Tricolour exited with non-zero return code")

    yield ms_filename

    # Remove MS
    shutil.rmtree(ms_filename)


@pytest.mark.parametrize("tol", [1e3])
def test_mean_chisq(flagged_ms, tol):
    """ Tests for improvement in mean chisq per correlation """
    with tbl("::".join((flagged_ms, "FIELD"))) as t:
        fnames = t.getcol("NAME")

    with tbl(flagged_ms) as t:
        fid = t.getcol("FIELD_ID")
        flag = t.getcol("FLAG")
        data = t.getcol("DATA")

    abs_data_sel = np.abs(data[fid == fnames.index("3C286")])
    diff = (abs_data_sel - np.nanmean(abs_data_sel, axis=0))**2
    chisq_unflagged_3c286 = np.nanmean(diff, axis=(0, 1))
    corrs_str = ",".join(["{0:.3f}".format(f) for f in chisq_unflagged_3c286])
    print("Chi^2 unflagged 3C286: [%s]" % corrs_str)

    abs_data_sel = np.abs(data[fid == fnames.index("PKS1934-63")])
    diff = (abs_data_sel - np.nanmean(abs_data_sel, axis=0))**2
    chisq_unflagged_1934 = np.nanmean(diff, axis=(0, 1))
    corrs_str = ",".join(["{0:.3f}".format(f) for f in chisq_unflagged_1934])
    print("Chi^2 unflagged PKS B1934-638: [%s]" % corrs_str)

    # flag data
    data[flag] = np.nan

    abs_data_sel = np.abs(data[fid == fnames.index("3C286")])
    diff = (abs_data_sel - np.nanmean(abs_data_sel, axis=0))**2
    chisq_flagged_3c286 = np.nanmean(diff, axis=(0, 1))
    corrs_str = ",".join(["{0:.3f}".format(f) for f in chisq_flagged_3c286])
    print("Chi^2 flagged 3C286: [%s]" % corrs_str)

    abs_data_sel = np.abs(data[fid == fnames.index("PKS1934-63")])
    diff = (abs_data_sel - np.nanmean(abs_data_sel, axis=0))**2
    chisq_flagged_1934 = np.nanmean(diff, axis=(0, 1))
    corrs_str = ",".join(["{0:.3f}".format(f) for f in chisq_flagged_1934])
    print("Chi^2 flagged PKS B1934-638: [%s]" % corrs_str)

    assert all(chisq_unflagged_3c286 > chisq_flagged_3c286 * tol)
    assert all(chisq_unflagged_1934 > chisq_flagged_1934 * tol)


@pytest.mark.parametrize("tol", [1e4])
def test_max_chisq(flagged_ms, tol):
    """ Tests for improvement in max chisq per correlation """
    with tbl("::".join((flagged_ms, "FIELD"))) as t:
        fnames = t.getcol("NAME")

    with tbl(flagged_ms) as t:
        fid = t.getcol("FIELD_ID")
        flag = t.getcol("FLAG")
        data = t.getcol("DATA")

    abs_data_sel = np.abs(data[fid == fnames.index("3C286")])
    diff = (abs_data_sel - np.nanmean(abs_data_sel, axis=0))**2
    chisq_unflagged_3c286 = np.nanmax(diff, axis=(0, 1))
    corrs_str = ",".join(["{0:.3f}".format(f) for f in chisq_unflagged_3c286])
    print("Max Chi^2 unflagged 3C286: [%s]" % corrs_str)

    abs_data_sel = np.abs(data[fid == fnames.index("PKS1934-63")])
    diff = (abs_data_sel - np.nanmean(abs_data_sel, axis=0))**2
    chisq_unflagged_1934 = np.nanmax(diff, axis=(0, 1))
    corrs_str = ",".join(["{0:.3f}".format(f) for f in chisq_unflagged_1934])
    print("Max Chi^2 unflagged PKS B1934-638: [%s]" % corrs_str)

    # flag data
    data[flag] = np.nan

    abs_data_sel = np.abs(data[fid == fnames.index("3C286")])
    diff = (abs_data_sel - np.nanmean(abs_data_sel, axis=0))**2
    chisq_flagged_3c286 = np.nanmax(diff, axis=(0, 1))
    corrs_str = ",".join(["{0:.3f}".format(f) for f in chisq_flagged_3c286])
    print("Max Chi^2 flagged 3C286: [%s]" % corrs_str)

    abs_data_sel = np.abs(data[fid == fnames.index("PKS1934-63")])
    diff = (abs_data_sel - np.nanmean(abs_data_sel, axis=0))**2
    chisq_flagged_1934 = np.nanmax(diff, axis=(0, 1))
    corrs_str = ",".join(["{0:.3f}".format(f) for f in chisq_flagged_1934])
    print("Max Chi^2 flagged PKS B1934-638: [%s]" % corrs_str)

    assert all(chisq_unflagged_3c286 > chisq_flagged_3c286 * tol)
    assert all(chisq_unflagged_1934 > chisq_flagged_1934 * tol)


@pytest.mark.parametrize("tol", [0.65])
def test_flag_count(flagged_ms, tol):
    """ Tests for flag count less than tolerance """
    with tbl("::".join((flagged_ms, "FIELD"))) as t:
        fnames = t.getcol("NAME")

    with tbl(flagged_ms) as t:
        fid = t.getcol("FIELD_ID")
        flag = t.getcol("FLAG")

    flag_sel = flag[fid == fnames.index("3C286")]
    count_flagged_3c286 = np.nansum(flag_sel, axis=(0, 1, 2))
    flagged_ratio = count_flagged_3c286 / flag_sel.size
    print("Percent flagged for 3C286: %.3f%%" % (100. * flagged_ratio))
    assert flagged_ratio < tol

    flag_sel = flag[fid == fnames.index("PKS1934-63")]
    count_flagged_1934 = np.nansum(flag_sel, axis=(0, 1, 2))
    flagged_ratio = count_flagged_1934 / flag_sel.size
    print("Percent flagged for PKS1934-63: %.3f%%" % (100. * flagged_ratio))
    assert flagged_ratio < tol


@pytest.mark.parametrize("tol", [0.40])
def test_bandwidth_flagged(flagged_ms, tol):
    """ Tests for total bandwidth flagged less than tolerance """
    with tbl("::".join((flagged_ms, "FIELD"))) as t:
        fnames = t.getcol("NAME")

    with tbl(flagged_ms) as t:
        fid = t.getcol("FIELD_ID")
        data = t.getcol("DATA")

    data_sel = data[fid == fnames.index("3C286"), :, 0]
    count_flagged_3c286 = np.nansum(data_sel, axis=0) > 0
    flagged_ratio = count_flagged_3c286.sum() / data_sel.shape[1]
    print("Percent bandwidth flagged for 3C286: %.3f%%"
          % (100. * flagged_ratio))
    assert flagged_ratio < tol

    data_sel = data[fid == fnames.index("PKS1934-63"), :, 0]
    count_flagged_1934 = np.nansum(data_sel, axis=0) > 0
    flagged_ratio = count_flagged_1934.sum() / data_sel.shape[1]
    print("Percent bandwidth flagged for PKS1934-63: %.3f%%"
          % (100. * flagged_ratio))
    assert flagged_ratio < tol
