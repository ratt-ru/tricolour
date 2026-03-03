# -*- coding: utf-8 -*-
"""Tests for :mod:`tricolour.flagging`."""

import numpy as np
import scipy.interpolate
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import pytest
from tricolour import flagging
import unittest


class TestAsbool(unittest.TestCase):
    def _test(self, dtype, expect_view):
        a = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype)
        expected = a.astype(np.bool_)
        out = flagging._asbool(a)

        assert np.bool_ == out.dtype
        np.testing.assert_array_equal(expected, out)

        if expect_view:
            # Change a, out must change because it is a view
            a[0] = not a[0]
            assert bool(a[0]) == out[0]

    def test_uint8(self):
        self._test(np.uint8, True)

    def test_uint16(self):
        self._test(np.uint16, False)

    def test_bool(self):
        self._test(np.bool_, True)


class TestAverageFreq(unittest.TestCase):
    def setUp(self):
        self.small_data = np.arange(30, dtype=np.float32).reshape(1, 5, 6)
        self.small_data = self.small_data.repeat(2, axis=0)
        self.small_flags = np.zeros(self.small_data.shape, np.bool_)
        self.small_flags[0, 3, :] = 1
        self.small_flags[0, :, 4] = 1
        self.small_flags[:, 2, 0] = 1
        self.small_flags[:, 2, 5] = 1

    def test_one(self):
        """
        _average_freq with 1 channel must have no effect on unflagged data
        """
        avg_data, avg_flags = flagging._average_freq(self.small_data,
                                                     self.small_flags,
                                                     flagging._as_min_dtype(1))
        expected = self.small_data.copy()
        expected[self.small_flags] = 0
        assert np.float32 == avg_data.dtype
        assert np.bool_ == avg_flags.dtype
        np.testing.assert_array_equal(expected, avg_data)
        np.testing.assert_array_equal(self.small_flags, avg_flags)

    def test_divides(self):
        """Test _average_freq when averaging factor divides in exactly"""
        expected_data = np.array([
            [
                [0.5, 2.5, 5.0],
                [6.5, 8.5, 11.0],
                [13.0, 14.5, 0.0],
                [0.0, 0.0, 0.0],
                [24.5, 26.5, 29.0]
            ],
            [
                [0.5, 2.5, 4.5],
                [6.5, 8.5, 10.5],
                [13.0, 14.5, 16.0],
                [18.5, 20.5, 22.5],
                [24.5, 26.5, 28.5]
            ]], np.float32)
        expected_flags = np.array([
            [
                [False, False, False],
                [False, False, False],
                [False, False, True],
                [True, True, True],
                [False, False, False]
            ],
            [[False, False, False]] * 5])

        avg_data, avg_flags = flagging._average_freq(self.small_data,
                                                     self.small_flags,
                                                     flagging._as_min_dtype(2))

        assert np.float32 == avg_data.dtype
        assert np.bool_ == avg_flags.dtype
        np.testing.assert_array_equal(expected_data, avg_data)
        np.testing.assert_array_equal(expected_flags, avg_flags)

    def test_uneven(self):
        """
        Test _average_freq when averaging factor
        does not divide number of channels
        """
        expected_data = np.array([
            [
                [1.5, 5.0],
                [7.5, 11.0],
                [14.0, 0.0],
                [0.0, 0.0],
                [25.5, 29.0],
            ],
            [
                [1.5, 4.5],
                [7.5, 10.5],
                [14.0, 16.0],
                [19.5, 22.5],
                [25.5, 28.5]
            ]], np.float32)
        expected_flags = np.array([
            [
                [False, False],
                [False, False],
                [False, True],
                [True, True],
                [False, False]
            ], [[False, False]] * 5], np.bool_)
        avg_data, avg_flags = flagging._average_freq(self.small_data,
                                                     self.small_flags,
                                                     flagging._as_min_dtype(4))
        assert np.float32 == avg_data.dtype
        assert np.bool_ == avg_flags.dtype
        np.testing.assert_array_equal(expected_data, avg_data)
        np.testing.assert_array_equal(expected_flags, avg_flags)


def test_time_median():
    """Test for :func:`katsdpsigproc.rfi.flagging._time_median`."""
    data = np.array([
        [2.0, 1.0, 2.0, 5.0],
        [3.0, 1.0, 8.0, 6.0],
        [4.0, 1.0, 4.0, 7.0],
        [5.0, 1.0, 5.0, 6.5],
        [1.5, 1.0, 1.5, 5.5]], np.float32)
    flags = np.array([
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1]], np.bool_)
    out_data, out_flags = flagging._time_median(data, flags)
    expected_data = np.array([[3.0, 0.0, 3.0, 6.0]], np.float32)
    expected_flags = np.array([[0, 1, 0, 0]], np.bool_)
    np.testing.assert_array_equal(expected_data, out_data)
    np.testing.assert_array_equal(expected_flags, out_flags)


class TestMedianAbs(unittest.TestCase):
    """Tests for :func:`katsdpsigproc.rfi.flagging._median_abs` and
    :func:`katsdpsigproc.rfi.flagging._median_abs_axis0`."""

    def setUp(self):
        self.data = np.array([[-2.0, -6.0, 4.5], [1.5, 3.3, 0.5]], np.float32)
        self.flags = np.array([[0, 0, 0], [0, 1, 0]], np.uint8)

    def test(self):
        out = flagging._median_abs(self.data, self.flags)
        assert 2.0 == out

    def test_all_flagged(self):
        out = flagging._median_abs(self.data, np.ones_like(self.flags))
        assert np.isnan(out)

    def test_axis0(self):
        out = flagging._median_abs_axis0(self.data, self.flags)
        expected = np.array([[1.75, 6.0, 2.5]])
        np.testing.assert_array_equal(expected, out)

    def test_axis0_all_flagged(self):
        self.flags[:, 1] = True
        out = flagging._median_abs_axis0(self.data, self.flags)
        expected = np.array([[1.75, np.nan, 2.5]])
        np.testing.assert_array_equal(expected, out)


class TestLinearlyInterpolateNans(unittest.TestCase):
    """
    Tests for :func:`katsdpsigproc.rfi.flagging._linearly_interpolate_nans`.
    """

    def setUp(self):
        self.y = np.array([np.nan, np.nan, 4.0, np.nan, np.nan,
                           10.0, np.nan, -2.0, np.nan, np.nan])
        self.expected = np.array([4.0, 4.0, 4.0, 6.0, 8.0,
                                  10.0, 4.0, -2.0, -2.0, -2.0])

    def test_basic(self):
        flagging._linearly_interpolate_nans1d(self.y)
        np.testing.assert_allclose(self.expected, self.y)

    def test_no_nans(self):
        y = self.expected[:]
        flagging._linearly_interpolate_nans1d(y)
        np.testing.assert_allclose(self.expected, y)

    def test_all_nans(self):
        self.y[:] = np.nan
        self.expected[:] = 0
        flagging._linearly_interpolate_nans1d(self.y)
        np.testing.assert_array_equal(self.expected, self.y)

    def test_float32(self):
        expected = self.expected.astype(np.float32)
        y = self.y.astype(np.float32)
        flagging._linearly_interpolate_nans1d(y)
        np.testing.assert_allclose(expected, y, rtol=1e-6)

    def test_2d(self):
        y = np.zeros((3, self.y.size))
        y[0, :] = self.y
        y[1, :] = self.expected
        y[2, :] = np.nan
        expected = np.zeros_like(y)
        expected[0, :] = self.expected
        expected[1, :] = self.expected
        expected[2, :] = 0
        flagging._linearly_interpolate_nans(y)
        np.testing.assert_allclose(expected, y)


class TestBoxGaussianFilter(unittest.TestCase):
    def test_one_pass(self):
        """Test that _box_gaussian_filter1d places the box correctly"""
        a = np.array([50.0, 10.0, 60.0, -70.0, 30.0, 20.0, -15.0], np.float32)
        b = np.empty_like(a)
        flagging._box_gaussian_filter1d(a, 2, b, 1)
        np.testing.assert_equal(
            np.array([24.0, 10.0, 16.0, 10.0, 5.0, -7.0, 7.0], np.float32), b)

    def test_width(self):
        """Impulse response must have approximately correct standard deviation,
        and must be symmetric with sum 1."""
        a = np.zeros((1, 200), np.float32)
        a[:, a.size // 2] = 1.0
        sigma = np.array([0.0, 10.0])
        b = np.empty_like(a)
        flagging._box_gaussian_filter(a, sigma, b)
        x = np.arange(a.size) - a.size // 2
        total = np.sum(b)
        np.testing.assert_allclose(1.0, total, rtol=1e-5)
        mean = np.sum(x * b)
        np.testing.assert_allclose(0.0, mean, atol=1e-5)
        std = np.sqrt(np.sum(x * x * b))
        # Very loose test, because box_gaussian_filter1d quantises
        np.testing.assert_allclose(std, sigma[1], atol=1)

    def test_bad_sigma_dim(self):
        a = np.zeros((50, 50), np.float32)
        with pytest.raises(ValueError):
            flagging._box_gaussian_filter(a, np.array([3.0]), a)

    def test_2d(self):
        rs = np.random.RandomState(seed=1)
        shape = (77, 53)
        sigma = np.array([8, 2.3])
        data = rs.uniform(size=shape).astype(np.float32)
        expected = gaussian_filter(data, sigma, mode='constant')
        actual = np.zeros_like(data)
        flagging._box_gaussian_filter(data, sigma, actual)
        np.testing.assert_allclose(expected, actual, rtol=1e-1)

    def test_axes(self):
        """Test that the axes are handled consistently"""
        rs = np.random.RandomState(seed=1)
        shape = (77, 53)
        data = rs.uniform(size=shape).astype(np.float32)
        out0 = np.zeros_like(data)
        out1 = np.zeros_like(data)
        flagging._box_gaussian_filter(data, np.array([8.0, 0.0]), out0)
        flagging._box_gaussian_filter(data.T, np.array([0.0, 8.0]), out1.T)
        np.testing.assert_array_equal(out0, out1)

    def test_edge(self):
        """Test that values outside the boundary are handled like zeros."""
        rs = np.random.RandomState(seed=1)
        data = np.zeros((1, 200), np.float32)
        core = data[:, 80:120]
        core[:] = rs.uniform(size=core.shape)
        fdata = np.ones_like(data)
        fcore = np.ones_like(core)
        flagging._box_gaussian_filter(data, np.array([0.0, 3.0]), fdata)
        flagging._box_gaussian_filter(core, np.array([0.0, 3.0]), fcore)
        np.testing.assert_allclose(fdata[:, 80:120], fcore, rtol=1e-5)


class TestMaskedGaussianFilter(unittest.TestCase):
    def setUp(self):
        self.rs = np.random.RandomState(seed=1)
        shape = (77, 53)
        self.data = self.rs.uniform(size=shape).astype(np.float32)
        self.flags = self.rs.uniform(size=shape) >= 0.5

    def _get_expected(self, sigma, truncate):
        weight = 1.0 - self.flags
        data = self.data * weight
        for i, (s, t) in enumerate(zip(sigma, truncate)):
            weight = gaussian_filter1d(weight, s, axis=i,
                                       mode='constant', truncate=t)
            data = gaussian_filter1d(data, s, axis=i,
                                     mode='constant', truncate=t)
        with np.errstate(invalid='ignore'):
            data /= weight
        return data

    def test_basic(self):
        sigma = np.array([5, 2.3])
        expected = self._get_expected(sigma, (4.0, 4.0))
        actual = np.ones_like(expected)
        flagging.masked_gaussian_filter(self.data, self.flags, sigma, actual)
        np.testing.assert_allclose(expected, actual, rtol=1e-1)

    def test_nan(self):
        # Set a big block of zeros to get NaNs in the result
        self.flags[:] = False
        self.flags[30:70, 10:40] = True
        # To match NaN positions, we need to match the footprint of the kernels
        sigma = np.array([3, 3.3])
        passes = 4
        radius = [int(0.5 * np.sqrt(12.0 * s**2 / passes + 1)) for s in sigma]
        truncate = [passes * r / s for (r, s) in zip(radius, sigma)]
        expected = self._get_expected(sigma, truncate)
        actual = np.ones_like(self.data)
        flagging.masked_gaussian_filter(self.data, self.flags, sigma, actual)
        np.testing.assert_allclose(expected, actual, rtol=1e-1)
        # Check that some NaNs were generated
        assert 0 < np.sum(np.isnan(expected))


class TestGetBackground2D(unittest.TestCase):
    """Tests for :func:`katsdpsigproc.rfi.flagging._get_background2d`.

    This is a difficult function to test, because it's not really practical to
    determine expected results by hand. The tests mainly check corner cases
    where large regions are flagged.
    """

    def setUp(self):
        self.shape = (95, 86)
        self.data = np.ones(self.shape, np.float32) * 7.5
        self.flags = np.zeros(self.shape, np.uint8)

    def _get_background2d(self, data, flags=None, iterations=1,
                          spike_width=(10.0, 10.0), reject_threshold=2.0,
                          freq_chunks=None):
        if flags is None:
            flags = np.zeros(data.shape, np.uint8)
        if freq_chunks is None:
            freq_chunks = np.array([0, data.shape[1]])
        spike_width = np.array(spike_width, np.float32)
        return flagging._get_background2d(data, flags, iterations,
                                          spike_width, reject_threshold,
                                          freq_chunks)

    def test_no_flags(self):
        background = self._get_background2d(self.data)
        assert np.float32 == background.dtype
        # It's all constant, so background and output should match.
        # It won't be exact though, because the Gaussian filter accumulates
        # errors as it sums.
        np.testing.assert_allclose(self.data, background, rtol=1e-5)

    def test_all_flagged(self):
        self.flags[:] = True
        background = self._get_background2d(self.data, self.flags)
        assert np.float32 == background.dtype
        np.testing.assert_array_equal(np.zeros(self.shape, np.float32),
                                      background)

    def test_in_flags(self):
        # This needs to be done carefully, because getbackground_2d does
        # internal masking on outliers too. We give every 3rd time a higher
        # power and flag it.
        self.data[::3] = 20.0
        self.flags[::3] = True
        background = self._get_background2d(self.data, self.flags)
        expected = np.ones_like(self.data) * 7.5
        np.testing.assert_allclose(expected, background, rtol=1e-5)

    def test_interpolate(self):
        """Linear interpolation across completely flagged data"""
        # Block of channels is 7.5, then a block is flagged (still 7.5), then
        # a block is 3.0.
        self.data[:, 70:] = 3.0
        self.flags[:, 30:70] = True
        # The setup above has no deviation from the background, which makes the
        # outlier rejection unstable, so we add noise to half the timesteps,
        # and test them at lower precision.
        # We use uniform noise to guarantee no outliers.
        rs = np.random.RandomState(seed=1)
        random_shape = self.data[0:50].shape
        self.data[:50, :] += rs.uniform(-0.001, 0.001, random_shape)

        # The rejection threshold is adjusted, because the default doesn't do
        # well when only about half the data is noisy.
        background = self._get_background2d(self.data, self.flags,
                                            spike_width=(2.5, 2.5),
                                            reject_threshold=5.0)
        expected = np.zeros_like(self.data)
        expected[:, :37] = 7.5
        expected[:, 63:] = 3.0
        expected[:, 37:63] = np.linspace(7.5, 3.0, 26)
        np.testing.assert_allclose(expected[56:], background[56:], rtol=1e-4)
        np.testing.assert_allclose(expected[:56], background[:56], rtol=1e-2)

    def test_iterations(self):
        expected = self.data.copy()
        # Add some noise
        rs = np.random.RandomState(seed=1)
        self.data += rs.standard_normal(self.data.shape) * 0.1
        # Add a "spike" that's larger than the initial spike_width to check
        # that it gets masked out.
        self.data[20:50, 30:80] += 15

        background = self._get_background2d(self.data, iterations=3)
        np.testing.assert_allclose(expected, background, rtol=1e-2)


class TestSumThreshold(unittest.TestCase):
    def setUp(self):
        self.small_data = np.arange(30, dtype=np.float32).reshape(5, 6)
        self.small_flags = np.zeros(self.small_data.shape, np.bool_)
        self.small_flags[3, :] = 1
        self.small_flags[:, 4] = 1
        self.small_flags[2, 0] = 1
        self.small_flags[2, 5] = 1
        self.outlier_nsigma = 4.5
        self.rho = 1.3
        self.windows = np.array([1, 2, 4, 8])

    def test_sum_threshold_all_flagged(self):
        self.small_flags[:] = True
        out_flags = flagging._sum_threshold(self.small_data, self.small_flags,
                                            0, np.array([1, 2, 4]),
                                            self.outlier_nsigma, self.rho)
        np.testing.assert_array_equal(np.zeros_like(self.small_flags),
                                      out_flags)

    def _test_sum_threshold_basic(self, axis):
        rs = np.random.RandomState(seed=1)
        data = rs.standard_normal((100, 90)).astype(np.float32) * 3.0
        rfi = np.zeros_like(data)
        # Add some individual spikes and some bad channels
        rfi[10, 20] = 100.0
        rfi[80, 80] = -100.0
        rfi[:, 40] = rs.uniform(80.0, 120.0, size=(100,))
        rfi[:, 2] = -rfi[:, 40]
        # Smaller but wider spike
        rfi[:, 60:67] = rs.uniform(15.0, 20.0, size=(100, 7))
        rfi[:, 10:17] = -rfi[:, 60:67]
        in_flags = np.zeros(data.shape, np.bool_)
        expected_flags = rfi != 0
        data += rfi
        if axis == 0:
            # Swap axes around so that we're doing essentially the same test
            rfi = rfi.T.copy()
            data = data.T.copy()
            in_flags = in_flags.T.copy()
        out_flags = flagging._sum_threshold(data, in_flags, axis,
                                            self.windows,
                                            self.outlier_nsigma,
                                            self.rho)
        if axis == 0:
            out_flags = out_flags.T
        # Due to random data, won't get perfect agreement, but should get close
        errors = np.sum(expected_flags != out_flags)
        assert errors / data.size < 0.01
        # Check for exact match on the individual spikes
        for region in (np.s_[8:13, 18:23], np.s_[78:83, 78:83]):
            np.testing.assert_equal(expected_flags[region], out_flags[region])

    def test_sum_threshold_time(self):
        self._test_sum_threshold_basic(axis=0)

    def test_sum_threshold_frequency(self):
        self._test_sum_threshold_basic(axis=1)

    def test_sum_threshold_existing(self):
        rs = np.random.RandomState(seed=1)
        data = rs.standard_normal((100, 90)).astype(np.float32) * 3.0
        in_flags = np.zeros(data.shape, np.bool_)
        # Corrupt but pre-flag just under half the data, which will skew the
        # noise estimate if not taken into account.
        data[:48] += 1000.0
        in_flags[:48] = True
        # Add some spikes that should be just under the detection limit.
        data[70, 0] = 12.5
        data[70, 1] = -12.5
        # Add some spikes that should still be detected.
        data[70, 2] = 20.0
        data[70, 3] = -20.0
        # Test it
        out_flags = flagging._sum_threshold(data, in_flags, 0,
                                            self.windows, 5, self.rho)
        np.testing.assert_array_equal([False, False, True, True],
                                      out_flags[70, :4])


class TestSumThresholdFlagger(unittest.TestCase):
    """Tests for :class:`katsdpsigproc.rfi.flagging.SumThresholdFlagger`."""

    def setUp(self):
        self.flagger = flagging.SumThresholdFlagger()

    def _make_background(self, shape, rs):
        """Simulate a bandpass with some smooth variation."""
        ncp, ntime, nfreq = shape
        nx = 10
        x = np.linspace(0.0, nfreq, nx)
        y = np.ones((ncp, ntime, nx)) * 2.34
        y[:, :, 0] = 0.1
        y[:, :, -1] = 0.1
        y[:] += rs.uniform(0.0, 0.1, y.shape)
        f = scipy.interpolate.interp1d(x, y, axis=2,
                                       kind='cubic', assume_sorted=True)
        return f(np.arange(nfreq))

    def _make_data(self, flagger, rs, shape=(1, 234, 345)):
        background = self._make_background(shape, rs).astype(np.float32)
        data = background + (rs.standard_normal(shape)
                             * 0.1).astype(np.float32)
        rfi = np.zeros(shape, np.float32)
        # Some completely bad channels and bad times
        rfi[:, 12, :] = 1
        rfi[:, 20:25, :] = 1
        rfi[:, :, 17] = 1
        rfi[:, :, 200:220] = 1
        # Some mostly bad channels and times
        rfi[:, 30, :300] = 1
        rfi[:, 50:, 80] = 1
        # Some smaller blocks of RFI
        rfi[:, 60:65, 100:170] = 1
        rfi[:, 150:200, 150:153] = 1
        expected = rfi.astype(np.bool_)
        # The mostly-bad channels and times must be fully flagged
        expected[:, 30, :] = True
        expected[:, :, 80] = True
        data += rfi * rs.standard_normal(shape) * 3.0
        # Channel that is slightly biased, but
        # wouldn't be picked up in a single dump
        data[:, :, 260] += 0.2 * flagger.average_freq
        expected[:, :, 260] = True
        # Test input NaN value flagged on output
        data[:, 225, 225] = np.nan
        expected[:, 225, 225] = True
        in_flags = np.zeros(shape, np.bool_)
        # Pre-flag some channels, and make those values NaN (because cal
        # currently does this - but should be fixed).
        in_flags[:, :, 185:190] = True
        data[:, :, 185:190] = np.nan
        return np.abs(data), in_flags, expected

    def _test_get_flags(self, flagger):
        rs = np.random.RandomState(seed=1)
        data, in_flags, expected = self._make_data(flagger, rs)

        orig_data = data.copy()
        orig_in_flags = in_flags.copy()
        out_flags = flagger.get_flags(data, in_flags)
        # Check that the original values aren't disturbed
        np.testing.assert_equal(orig_data, data)
        np.testing.assert_equal(orig_in_flags, in_flags)

        # Check the results. Everything that's expected must be flagged,
        # everything within time_extend and freq_extend kernels may be
        # flagged. A small number of values outside this may be flagged.
        # The backgrounding doesn't currently handle the edges of the band
        # well, so for now we also allow the edges to be flagged too.
        # TODO: improve _get_background2d so that it better fits a slope
        # at the edges of the passband.
        allowed = expected | in_flags
        allowed[:, :-1, :] |= allowed[:, 1:, :]
        allowed[:, 1:, :] |= allowed[:, :-1, :]
        allowed[:, :, :-1] |= allowed[:, :, 1:]
        allowed[:, :, 1:] |= allowed[:, :, :-1]
        allowed[:, :, :40] = True
        allowed[:, :, -40:] = True
        missing = expected & ~out_flags
        extra = out_flags & ~allowed
        # Uncomment for debugging failures
        # print(np.where(missing > 0))
        # import matplotlib.pyplot as plt
        # plt.imshow(expected[0, ...] +
        #            2 * out_flags[0, ...] +
        #            4 * allowed[0, ...])
        # plt.show()
        assert 0 == missing.sum()
        assert extra.sum() / data.size < 0.03

    def test_get_flags(self):
        self._test_get_flags(self.flagger)

    def test_get_flags_single_chunk(self):
        flagger = flagging.SumThresholdFlagger(freq_chunks=1)
        self._test_get_flags(flagger)

    def test_get_flags_many_chunks(self):
        # Number of chunks can't be too high, otherwise the block of channels
        # affected by RFI isn't detected as it mostly falls into one chunk.
        flagger = flagging.SumThresholdFlagger(freq_chunks=15)
        self._test_get_flags(flagger)

    def test_get_flags_average_freq(self):
        flagger = flagging.SumThresholdFlagger(average_freq=2)
        self._test_get_flags(flagger)

    def test_get_flags_iterations(self):
        # TODO: fix up the overflagging of the background in the flagger,
        # which currently causes this to fail.
        pytest.skip('Backgrounder overflags edges of the slope')
        # flagger = flagging.SumThresholdFlagger(background_iterations=3)
        # self._test_get_flags(flagger)

    def _test_get_flags_all_flagged(self, flagger):
        data = np.zeros((4, 100, 80), np.float32)
        in_flags = np.ones(data.shape, np.bool_)
        out_flags = flagger.get_flags(data, in_flags)
        np.testing.assert_array_equal(np.zeros_like(in_flags), out_flags)

    def test_get_flags_all_flagged(self):
        self._test_get_flags_all_flagged(self.flagger)

    def test_get_flags_all_flagged_average_freq(self):
        flagger = flagging.SumThresholdFlagger(average_freq=4)
        self._test_get_flags_all_flagged(flagger)

    def test_variable_noise(self):
        """Noise level that varies across the band."""
        rs = np.random.RandomState(seed=1)
        shape = (1, 234, 345)
        # For this test we use a flat background, to avoid the issues with
        # bandpass estimation in sloped regions.
        background = np.ones(shape, np.float32) * 11
        # Noise level that varies from 0 to 1 across the band
        noise = rs.standard_normal(shape)
        noise *= np.arange(shape[2])[np.newaxis, np.newaxis, :] / shape[2]
        noise = noise.astype(np.float32)
        noise[:, 100, 17] = 1.0    # About 20 sigma - must be detected
        noise[:, 200, 170] = 1.0   # About 2 sigma -  must not be detected
        data = np.abs(background + noise)
        in_flags = np.zeros(shape, np.bool_)
        out_flags = self.flagger.get_flags(data, in_flags)
        assert out_flags[0, 100, 17] == True  # noqa
        assert out_flags[0, 200, 170] == False  # noqa

    def _test_parallel(self, pool):
        """Test that parallel execution gets same results as serial"""
        rs = np.random.RandomState(seed=1)
        data, in_flags, expected = self._make_data(self.flagger, rs,
                                                   shape=(32, 234, 512))
        out_serial = self.flagger.get_flags(data, in_flags)
        out_parallel = self.flagger.get_flags(data, in_flags, pool=pool)
        np.testing.assert_array_equal(out_serial, out_parallel)
