# -*- coding: utf-8 -*-
"""
Automated qualification test for the tricolour flagger
Expects environment variable TRICOLOUR_TEST_MS set to
MeerKAT acceptance_test_data.tar.gz prior to running
"""

import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from pyrap.tables import table as tbl
import numpy as np
import requests
import pytest

import dask
from daskms import xds_from_ms, xds_to_table

_GOOGLE_FILE_ID = "1yxDIXUo3Xun9WXxA0x_hvX9Fmxo9Igpr"
_MS_FILENAME = '1519747221.subset.ms'


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def _download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, destination)


@pytest.fixture(scope="session")
def ms_tarfile(tmp_path_factory):
    try:
        tarred_ms_filename = Path(os.environ["TRICOLOUR_TEST_MS"])
    except KeyError:
        tar_dir = tmp_path_factory.mktemp("acceptance-download-")
        tarred_ms_filename = tar_dir / "test_data.tar.gz"

        _download_file_from_google_drive(_GOOGLE_FILE_ID, tarred_ms_filename)

    yield tarred_ms_filename


@pytest.fixture(scope="function")
def ms_filename(ms_tarfile, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("acceptance-data-")

    with tarfile.open(ms_tarfile) as tarred_ms:
        tarred_ms.extractall(tmp_path)

    yield str(Path(tmp_path / _MS_FILENAME))

# Set timeout to 6 minutes
@pytest.fixture(params=[360], scope="function")
def flagged_ms(request, ms_filename):
    """
    fixture yielding an MS flagged by tricolour
    """
    test_directory = os.path.dirname(__file__)

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

    try:
        yield ms_filename
    finally:
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


@pytest.fixture(params=[360], scope="function")
def multi_model_ms(request, ms_filename):
    """
    Multi-model 'DATA' column
    """
    test_directory = os.path.dirname(__file__)

    # Open ms
    xds = xds_from_ms(ms_filename)
    # Create 'MODEL_DATA' column
    for i, ds in enumerate(xds):
        dims = ds.DATA.dims
        xds[i] = ds.assign(MODEL_DATA=(dims, ds.DATA.data / 2))

    # Write 'MODEL_DATA column - delayed operation
    writes = xds_to_table(xds, ms_filename, "MODEL_DATA")
    dask.compute(writes)

    # pass the expression to Tricolour
    args = ['tricolour',
            '-fs', 'total_power',
            '-c', os.path.join(test_directory, 'custom.yaml'),
            '-dc', 'DATA - MODEL_DATA',
            ms_filename]

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

    try:
        yield ms_filename
    finally:
        shutil.rmtree(ms_filename)


def test_multi_model(multi_model_ms):
    """
    Test Multi-model 'DATA' column
    """
    pass
