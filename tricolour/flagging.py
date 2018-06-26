"""Library to contain 2d RFI flagging routines and other RFI related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numba

MAD_NORMAL = 1.4826
"""Ratio between `median absolute deviation`_ and standard deviation of a Gaussian distribution.
.. _median absolute deviation: https://en.wikipedia.org/wiki/Median_absolute_deviation
"""


def _as_min_dtype(value):
    """Convert a non-negative integer into a numpy scalar of the narrowest
    type will hold it.

    This is used because in some cases an array must be allocated of the
    same type later, and using the narrowest type saves memory in that array.
    """
    if value >= 0 and value < 2**8:
        dtype = np.uint8
    elif value >= 0 and value < 2**16:
        dtype = np.uint16
    elif value >= 0 and value < 2**32:
        dtype = np.uint32
    else:
        dtype = np.int64
    return np.array(value, dtype)


@numba.generated_jit(nopython=True, nogil=True, cache=True)
def _asbool(data):
    """Create a boolean array with the same values as `data`.

    The `data` contain only 0's and 1's. If possible, a view is returned,
    otherwise a copy.
    """
    if isinstance(data, np.ndarray):
        # We're being called as a regular function due to NUMBA_DISABLE_JIT.
        if data.dtype.itemsize == 1:
            return data.view(np.bool_)
        else:
            return data.astype(np.bool_)
    elif (isinstance(data.dtype, numba.types.Boolean)
            or (isinstance(data.dtype, numba.types.Integer) and data.dtype.bitwidth == 8)):
        return lambda data: data.view(np.bool_)
    else:
        return lambda data: data.astype(np.bool_)


@numba.jit(nopython=True, nogil=True, cache=True)
def _time_median(data, flags):
    """Independently for each channel, compute the median of the unflagged
    values. If all values for a channel are flagged, 0 is used instead, and
    the result is flagged.

    The time dimension is kept in the result as a length-1 dimension.

    Parameters
    ----------
    data : ndarray, real
        Visibilities, with shape (time, frequency)
    flags : ndarray, bool
        Flags corresponding to `data`

    Returns
    -------
    out_data : ndarray, real
        Median of `data` for each frequency
    out_flags : ndarray, bool
        Flags corresponding to `out_data`
    """
    n_time, n_freq = data.shape
    values = np.empty((n_freq, n_time), data.dtype)
    counts = np.zeros(n_freq, np.uint32)
    out_data = np.empty((1, n_freq), data.dtype)
    out_flags = np.zeros((1, n_freq), np.bool_)
    for t in range(n_time):
        for f in range(n_freq):
            if not flags[t, f]:
                values[f, counts[f]] = data[t, f]
                counts[f] += 1
    for f in range(n_freq):
        if counts[f] == 0:
            out_data[0, f] = 0     # No data
            out_flags[0, f] = True
        else:
            out_data[0, f] = np.median(values[f, :counts[f]])
    return out_data, out_flags


@numba.jit(nopython=True, nogil=True, cache=True)
def _median_abs(data, flags):
    """Compute median of absolute values of non-flagged values in `data`."""
    values = np.empty(data.size, data.dtype)
    n = np.int64(0)
    for idx in np.ndindex(data.shape):
        if not flags[idx]:
            values[n] = np.abs(data[idx])
            n += 1
    if n == 0:
        return np.nan
    else:
        return np.median(values[:n])


@numba.jit(nopython=True, nogil=True, cache=True)
def _median_abs_axis0(data, flags):
    """Compute median of absolute values of non-flagged values in `data`, along axis 0.

    The first dimension is kept in the output as a dimension of size 1 (to
    avoid issues with numba converting 0d arrays to scalars.
    """
    values = np.empty(data.shape[1:] + data.shape[:1], data.dtype)
    counts = np.zeros(data.shape[1:], np.uint32)
    out_data = np.empty((1,) + data.shape[1:], data.dtype)
    for i in range(data.shape[0]):
        for j in np.ndindex(data.shape[1:]):
            if not flags[(i,) + j]:
                values[j + (counts[j],)] = np.abs(data[(i,) + j])
                counts[j] += 1
    for j in np.ndindex(data.shape[1:]):
        if counts[j] == 0:
            out_data[(0,) + j] = np.nan
        else:
            out_data[(0,) + j] = np.median(values[j + (slice(0, counts[j]),)])
    return out_data


@numba.jit(nopython=True, nogil=True, cache=True)
def _linearly_interpolate_nans1d(data):
    """Replace NaNs in `data` by linear interpolation in-place.

    Extrapolation is done by repeating the first/last valid element.  If all
    input data are NaNs, they are all replaced by zeros.

    Parameters
    ----------
    data : ndarray, real
        Data to interpolate, 1D. It is modified in-place.
    """
    n = data.size
    # Find first valid value
    p = 0
    while p < n and np.isnan(data[p]):
        p += 1
    if p == n:
        data[:] = 0
        return
    data[:p] = data[p]     # Extrapolate backwards
    p += 1
    while p < n:
        if np.isnan(data[p]):
            # Find next valid value
            q = p + 1
            while q < n and np.isnan(data[q]):
                q += 1
            if q == n:
                data[p:] = data[p - 1]   # Extrapolate forwards
            else:
                start = data[p - 1]
                grad = (data[q] - start) / (q - (p - 1))
                for i in range(p, q):
                    data[i] = start + (i - (p - 1)) * grad
            p = q
        else:
            p += 1


@numba.jit(nopython=True, nogil=True, cache=True)
def _linearly_interpolate_nans(data):
    """Replace nans in `data` by linear interpolation across frequencies.

    Extrapolation is done by repeating the first/last valid element.

    Parameters
    ----------
    data : ndarray, real
        Data to interpolate, with shape (time, frequency).
    """
    for i in range(data.shape[0]):
        _linearly_interpolate_nans1d(data[i])


@numba.jit(nopython=True, nogil=True, cache=True)
def _box_gaussian_filter1d(data, r, out, passes):
    """Implementation of :func:`_box_gaussian_filter` along the first axis of an array.

    It is safe to use this function in-place i.e. with `out` equal to `data`.

    Parameters
    ----------
    data : ndarray, real
        Input data, with at least 1 dimension.
    r : int
        Radius of the box filter
    out : ndarray, real
        Output data, with same shape as `data`.
    passes : int
        Number of boxcar filters to apply
    """
    K = passes
    if data.shape[0] == 0 or K == 0:
        out[:] = data[:]
        return
    d = 2 * r + 1
    # Pad on left with zeros.
    padding = r * K
    # TODO: hoist memory allocations into caller
    padded = np.empty((data.shape[0] + padding,) + data.shape[1:], data.dtype)
    padded[:padding] = 0
    padded[padding:] = data
    prev_start = padding   # First element with valid data
    s = np.zeros(data.shape[1:], np.float64)
    for p in range(1, K + 1):
        # On each pass, padded[i] is replaced by the sum of padded[i : i + d]
        # from the previous pass. The accumulator is kept in double precision
        # to avoid excessive accumulation of errors.
        s[()] = 0
        start = padding - 2 * r * p
        stop = start + data.shape[0] + 2 * padding
        start = max(start, 0)
        stop = min(stop, padded.shape[0])
        tail = min(stop, padded.shape[0] - 2 * r)
        for i in range(prev_start, min(start + 2 * r, padded.shape[0])):
            s += padded[i]
        for i in range(start, tail):
            for j in np.ndindex(data.shape[1:]):
                s[j] += padded[(i + 2 * r,) + j]
                prev = padded[(i,) + j]
                padded[(i,) + j] = s[j]
                s[j] -= prev
        for i in range(tail, stop):
            for j in np.ndindex(data.shape[1:]):
                prev = padded[(i,) + j]
                padded[(i,) + j] = s[j]
                s[j] -= prev
        prev_start = start
    for idx in np.ndindex(out.shape):
        out[idx] = padded[idx] / data.dtype.type(d)**K


@numba.jit(nopython=True, nogil=True, cache=True)
def _box_gaussian_filter(data, sigma, out, passes=4):
    """Filter `data` with an approximate Gaussian filter.

    The filter is based on repeated filtering with a boxcar function. See
    [Get13]_ for details. It has finite support. Values outside the boundary
    are taken as zero.

    This function is not suitable when the input contains non-finite values,
    or very large variations in magnitude, as it internally computes a rolling
    sum. It also quantizes the requested sigma.

    .. [Get13] Pascal Getreuer, A Survey of Gaussian Convolution Algorithms,
       Image Processing On Line, 3 (2013), pp. 286-310.

    Parameters
    ----------
    data : ndarray
        Input data to filter (2D)
    sigma : ndarray
        Standard deviation of the Gaussian filter, per axis
    out : ndarray
        Output data, with the same shape as the input
    passes : int
        Number of boxcar filters to apply
    """
    if len(sigma) != data.ndim:
        raise ValueError('sigma has wrong number of elements')
    assert data.ndim == 2
    r = (0.5 * np.sqrt(12.0 * sigma**2 / passes + 1)).astype(np.int_)
    need_copy = True
    if r[0] > 0:
        # Process chunks of columns. See _sum_threshold for explanation.
        step = 256
        for i in range(0, data.shape[1], step):
            sub = slice(i, min(i + step, data.shape[1]))
            _box_gaussian_filter1d(data[:, sub], r[0], out[:, sub], passes)
        data = out       # Use out in next step
        need_copy = False
    if r[1] > 0:
        for i in range(data.shape[0]):
            _box_gaussian_filter1d(data[i], r[1], out[i], passes)
        need_copy = False
    if need_copy:
        out[:] = data[:]


@numba.jit(nopython=True, nogil=True, cache=True)
def masked_gaussian_filter(data, flags, sigma, out, passes=4):
    """Filter an image using an approximate Gaussian filter.

    Some values may be flagged and are ignored. Values outside the grid are
    also treated as if flagged.

    See :func:`box_gaussian_filter` for a number of caveats. The result may
    contain non-finite values where the finite support of the Gaussian
    approximation contains no values without flags.

    Parameters
    ----------
    data : ndarray, 2D
        Input data to filter
    flags : ndarray, bool
        True values correspond to elements of `data` to be ignored
    sigma : float or sequence of floats
        Standard deviation of the Gaussian filter, per axis
    passes : int
        Number of boxcar filters to apply

    Returns
    -------
    ndarray
        Output data, with the same shape as the input
    """
    if data.shape != flags.shape:
        raise ValueError('shape mismatch between data and flags')
    if data.shape != out.shape:
        raise ValueError('shape mismatch between data and out')
    weight = np.empty_like(data)
    for idx in np.ndindex(data.shape):
        weight[idx] = not flags[idx]
        out[idx] = 0 if flags[idx] else data[idx]
    _box_gaussian_filter(weight, sigma, weight, passes=passes)
    _box_gaussian_filter(out, sigma, out, passes=passes)
    for idx in np.ndindex(out.shape):
        # Numeric instability can make out non-zero (but tiny) even
        # where filtered_weight is zero, which would make the ratio +/-inf
        # rather than NaN. Set to NaN explicitly in this case.
        if weight[idx] == 0:
            out[idx] = np.nan
        else:
            out[idx] /= weight[idx]


@numba.jit(nopython=True, nogil=True, cache=True)
def _get_background2d(data, flags, iterations, spike_width, reject_threshold, freq_chunk_ends):
    """Determine a smooth background over a 2D array by iteratively convolving
    the data with elliptical Gaussians with linearly decreasing width from
    `iterations`*`spike_width` down to `spike width`. Outliers greater than
    `reject_threshold`*sigma from the background are masked on each
    iteration.

    Initial weights are set to zero at positions specified in `in_flags` if given.
    After the final iteration a final Gaussian smoothed background is computed
    and any stray NaNs in the background are interpolated in frequency (axis 1)
    for each timestamp (axis 0). The NaNs can appear when the the convolving
    Gaussian is completely covering masked data as the sum of convolved weights
    will be zero.

    Parameters
    ----------
    data : 2D ndarray, float
        The input data array to be smoothed, with shape (time, frequency).
    flags : 2D ndarray, boolean
        Flags corresponding to `data`
    spike_width : ndarray, float
        Two-element array containing the 1-sigma radius of the Gaussian filter
        in each axis.
    reject_threshold : float
        Number of standard deviations above which to flag data.
    freq_chunk_ends : ndarray, float
        Endpoints of intervals in which to compute noise estimates
        independently. This array must start with 0 and end of the number of
        channels, and be strictly increasing.
    """

    n_time, n_freq = data.shape
    flags = flags.copy()   # Gets modified
    background = np.empty_like(data)
    for extend_factor in range(iterations, 0, -1):
        sigma = extend_factor * spike_width
        masked_gaussian_filter(data, flags, sigma, background)
        for c in range(freq_chunk_ends.size - 1):
            sub = (slice(None, None), slice(
                freq_chunk_ends[c], freq_chunk_ends[c + 1]))
            sub_data = data[sub]
            sub_flags = flags[sub]
            # Convert background to an absolute value residual, in-place
            sub_residual = background[sub]
            for t in range(n_time):
                for f in range(sub_data.shape[1]):
                    sub_residual[t, f] = np.abs(
                        sub_data[t, f] - sub_residual[t, f])
            threshold = _median_abs(sub_residual, sub_flags)
            threshold *= MAD_NORMAL * reject_threshold
            for t in range(n_time):
                for f in range(sub_data.shape[1]):
                    # sub_residual can contain NaNs, but only where the flags
                    # already apply
                    if sub_residual[t, f] > threshold:
                        sub_flags[t, f] = True
    # Compute final background
    masked_gaussian_filter(data, flags, spike_width, background)
    # Remove NaNs via linear interpolation
    _linearly_interpolate_nans(background)
    return background


@numba.jit(nopython=True, nogil=True, cache=True)
def _convolve_flags(in_values, scale, threshold, out_flags, window):
    """Flag values with a threshold, and smear the flags.

    This is rolled into a single function for efficient implementation, but
    there are logically several steps:
    - For each value v in `in_values`, flag it if ``v * scale > threshold``.
    - Convolve the flags by a box filter of size `window`, expanding the width.
    - Logical OR these new flags into `out_flags`.
    """
    cum_size = in_values.shape[0] + 2 * window - 1
    # TODO: could preallocate this externally
    # Cumulative flagged values
    cum = np.empty((cum_size,) + (in_values.shape[1:]), np.uint32)
    cum[:window] = 0
    for i in range(in_values.shape[0]):
        for j in np.ndindex(in_values.shape[1:]):
            flag = in_values[(i,) + j] * scale > threshold[(0,) + j]
            cum[(window + i,) + j] = cum[(window + i - 1,) + j] + flag
    # numba doesn't seem to fully support negative indices, hence
    # the addition of cum_size.
    cum[cum_size - (window - 1):] = cum[cum_size - window]
    for i in range(out_flags.shape[0]):
        for j in np.ndindex(out_flags.shape[1:]):
            out_flags[(i,) + j] |= (cum[(i + window,) + j] -
                                    cum[(i,) + j] != 0)


@numba.jit(nopython=True, nogil=True, cache=True)
def _sum_threshold1d(input_data, input_flags, output_flags, windows, outlier_nsigma, rho, chunks):
    """Implementation of :func:`_sum_threshold`. It operates along the first axis."""
    for ci in range(chunks.size - 1):
        chunk_slice = slice(chunks[ci], chunks[ci + 1])
        chunk_data = input_data[chunk_slice]
        chunk_flags = input_flags[chunk_slice]

        # Get standard deviation using MAD and set up initial threshold
        threshold = _median_abs_axis0(chunk_data, chunk_flags)
        threshold_scale = outlier_nsigma * MAD_NORMAL
        for idx in np.ndindex(threshold.shape):
            if np.isnan(threshold[idx]):
                threshold[idx] = np.inf
            else:
                threshold[idx] *= threshold_scale

        padded_slice = slice(max(chunks[ci] - np.max(windows) + 1, 0),
                             min(chunks[ci + 1] + np.max(windows) - 1, input_data.size))
        padded_data = input_data[padded_slice]
        # TODO: can pre-allocate these outside the loop (but will need
        # resizing)
        output_flags_pos = np.zeros(padded_data.shape, np.bool_)
        output_flags_neg = np.zeros(padded_data.shape, np.bool_)
        for window in windows:
            # The threshold for this iteration is calculated from the initial threshold
            # using the equation from Offringa (2010).
            tf = pow(rho, np.log2(window))
            # Get the thresholds
            thisthreshold = threshold / tf
            # Set already flagged values to be the +/- value of the
            # threshold if they are outside the threshold, and take
            # a cumulative sum.
            cum_data = np.empty(
                (padded_data.shape[0] + 1,) + padded_data.shape[1:], np.float64)
            cum_data[0] = 0
            for i in range(padded_data.shape[0]):
                for j in np.ndindex(padded_data.shape[1:]):
                    idx = (i,) + j
                    clamped = padded_data[idx]
                    limit = thisthreshold[(0,) + j]
                    if output_flags_pos[idx] and clamped > limit:
                        clamped = limit
                    elif output_flags_neg[idx] and clamped < -limit:
                        clamped = -limit
                    cum_data[(i + 1,) + j] = cum_data[idx] + clamped
            # Calculate a rolling sum array from the data with the window for this iteration,
            # which is later scaled by rolliing_scale to give the rolling
            # average.
            avgarray = cum_data[window:] - cum_data[:-window]
            rolling_scale = np.float32(1.0 / window)

            # Work out the flags from the average data above the current threshold,
            # convolve them, and combine with current flags.
            _convolve_flags(avgarray, rolling_scale,
                            thisthreshold, output_flags_pos, window)

            # Work out the flags from the average data below the current threshold,
            # convolve them, and OR with current flags.
            _convolve_flags(avgarray, -rolling_scale,
                            thisthreshold, output_flags_neg, window)

        # Extract just the portion of output_flags_pos/neg corresponding to the
        # chunk itself, without the padding
        rel_slice = slice(chunk_slice.start - padded_slice.start,
                          chunk_slice.stop - padded_slice.start)
        output_flags[chunk_slice] = output_flags_pos[rel_slice] | output_flags_neg[rel_slice]


@numba.jit(nopython=True, nogil=True, cache=True)
def _sum_threshold(input_data, input_flags, axis, windows, outlier_nsigma, rho, chunks=None):
    """Apply the SumThreshold method along the given axis of
    `input_data`.

    Parameters
    ----------
    input_data : ndarray, real
        Deviations from the background. The implementation is optimised for
        2D (and does not currently work in 1D), but higher dimensions are
        supported.
    input_flags : ndarray, bool
        Input flags. Used as a mask when computing the initial
        standard deviations of the input data.
    axis : int
        The axis on which to apply the SumThreshold operation. In the current
        implementation, must be 0 or 1.
    windows : ndarray, int
        Window sizes to average data in each SumThreshold step
    outlier_nsigma : float
        Number of standard deviations at which to flag
    rho : float
        Parameter controlling the relationship between threshold and window size
    chunks : ndarray, int
        Boundaries between chunks in which each chunk has a separate noise
        estimation. This array must start with 0, be strictly increasing, and
        end with ``input_data.shape[axis]``.

    Returns
    -------
    output_flags : ndarray, bool
        The derived flags
    """
    if chunks is None:
        chunks = np.array([0, input_data.shape[axis]])
    output_flags = np.empty(input_data.shape, np.bool_)
    if axis < 0 or axis >= input_data.ndim:
        raise ValueError('axis is out of range')
    elif axis == 1:
        for i in range(input_data.shape[0]):
            _sum_threshold1d(input_data[i], input_flags[i], output_flags[i],
                             windows, outlier_nsigma, rho, chunks)
    elif axis == 0:
        # The operation is independent of the other dimensions, but we process
        # them in chunks to be cache friendly. The step size should be big
        # enough that whole cache lines are used (even if the alignment is
        # poor), but small enough that multiple rows fit in L1. This heuristic
        # value assumes a 2D input.
        step = 256
        for i in range(0, input_data.shape[1], step):
            sub = slice(i, min(i + step, input_data.shape[1]))
            _sum_threshold1d(input_data[:, sub], input_flags[:, sub], output_flags[:, sub],
                             windows, outlier_nsigma, rho, chunks)
    else:
        raise ValueError('axis must be 0 or 1')
    return output_flags


@numba.jit(nopython=True, nogil=True, cache=True)
def _get_flags_impl(
        in_data, in_flags, out_flags,
        outlier_nsigma, windows_time, windows_freq,
        background_reject, background_iterations,
        spike_width_time, spike_width_freq, time_extend, freq_extend,
        freq_chunk_ends, average_freq, flag_all_time_frac, flag_all_freq_frac,
        rho):
    n_time, n_freq, n_bl = in_data.shape
    # Average `in_data` in frequency. This is done unconditionally, because it
    # also does other useful steps (see the documentation).
    data, flags = _average_freq(in_data, in_flags, average_freq)

    # Output flags, in baseline-major order
    tmp_flags = np.empty((n_bl, n_time, n_freq), np.bool_)
    # Do operations independently per baseline.
    for bl in range(data.shape[0]):
        _get_baseline_flags(
            data[bl], flags[bl], tmp_flags[bl],
            outlier_nsigma, windows_time, windows_freq,
            background_reject, background_iterations,
            spike_width_time, spike_width_freq,
            time_extend, freq_extend,
            freq_chunk_ends, average_freq,
            flag_all_time_frac, flag_all_freq_frac,
            rho)

    # Transpose the output flags and explicitly flag nans from input
    for t in range(n_time):
        for f in range(n_freq):
            for bl in range(n_bl):
                out_flags[t, f, bl] = tmp_flags[bl, t,
                                                f] or np.isnan(in_data[t, f, bl])


@numba.jit(nopython=True, nogil=True, cache=True)
def _combine_flags(spec_flags, time_flags, freq_flags, time_extend, out):
    """Combine several sources of flags and smear them in time.

    Parameters
    ----------
    spec_flags : 1D ndarray, bool
        Flags with shape (frequency)
    time_flags, freq_flags : 2D ndarray, bool
        Flags with shape (time, frequency)
    time_extend : int
        Width of the convolution kernel for time smearing (should be odd)
    out : 2D ndarray, bool
        Output flags
    """
    n_time, n_freq = time_flags.shape
    # Combine spec_flags, time_flags and freq_flags, and take a cumulative sum
    # along the time axis for the purposes of convolution.
    flag_sum = np.empty((n_time + 1, n_freq), time_extend.dtype)
    flag_sum[0] = 0
    for t in range(n_time):
        for f in range(n_freq):
            flag = spec_flags[0, f] or time_flags[t, f] or freq_flags[t, f]
            flag_sum[t + 1, f] = flag_sum[t, f] + flag
    # Difference the cumulative sums to get time-smeared flags.
    time_delta_lo = -(time_extend // 2)
    time_delta_hi = time_delta_lo + time_extend
    for t in range(n_time):
        # Rows to difference, clamping to the data limits
        t0 = max(t + time_delta_lo, 0)
        t1 = min(t + time_delta_hi, n_time)
        for f in range(n_freq):
            out[t, f] = (flag_sum[t0, f] != flag_sum[t1, f])


@numba.jit(nopython=True, nogil=True, cache=True)
def _average_freq(in_data, in_flags, factor):
    """Does several preconditioning steps:

    1. Converts complex data to real.
    2. Flags data with non-finite values.
    3. Sets the value of flagged elements to zero.
    4. Does frequency averaging by a factor of `factor`.
    5. Transposes the data ordering so that baseline is the first,
       slowest-varying axis.

    Parameters
    ----------
    in_data : ndarray, real or complex
        Visibilities or their magnitudes, with shape (time, frequency, baseline).
    in_flags : ndarray, bool
        Flags corresponding to the visibilities. This can safely be a type
        other than bool, where non-zero values indicate flagged data.
    factor : int
        Amount by which to decimate in frequency. This must be a numpy 0-d
        array (so that the dtype can be extracted).
    """
    if in_data.shape != in_flags.shape:
        raise ValueError('shape mismatch')
    n_time, n_freq, n_bl = in_data.shape
    a_freq = (n_freq + factor - 1) // factor
    out_shape = (n_bl, n_time, a_freq)
    avg_data = np.zeros(out_shape, np.float32)
    avg_weight = np.zeros(out_shape, factor.dtype)
    # TODO: might need to do this through a temporary buffer to avoid cache
    # aliasing problems.
    for i in range(n_time):
        for j in range(n_freq):
            jout = j // factor
            for k in range(n_bl):
                data = np.abs(in_data[i, j, k])
                if not in_flags[i, j, k] and not np.isnan(data):
                    avg_data[k, i, jout] += data
                    avg_weight[k, i, jout] += 1
    for i in range(n_bl):
        for j in range(n_time):
            for k in range(a_freq):
                flag = avg_weight[i, j, k] == 0
                if flag:
                    avg_data[i, j, k] = 0   # Avoid divide by zero and a NaN
                else:
                    avg_data[i, j, k] /= avg_weight[i, j, k]
                # Replace weight with flag (in-place) to save memory
                avg_weight[i, j, k] = flag
    return avg_data, _asbool(avg_weight)


@numba.jit(nopython=True, nogil=True, cache=True)
def _unaverage_freq(flags, freq_extend, average_freq,
                    flag_all_time_frac,
                    flag_all_freq_frac, out):
    """Final processing for a single baseline:
    1. Flags are replicated to undo the effect of frequency averaging.
    2. Flags are smeared, using a kernel of width `flag_all_freq_frac`.
    3. Times and frequencies where more than `flag_all_freq_frac` or
       `flag_all_time_frac` of the values are already flagged become fully
       flagged.
    """
    # Frequency replication and smearing
    n_time, n_freq = flags.shape
    orig_freq = out.shape[-1]
    flag_sum = np.empty(orig_freq + 1, np.int32)
    flag_sum_time = np.zeros(orig_freq, np.int32)
    freq_delta_lo = -(freq_extend // 2)
    freq_delta_hi = freq_delta_lo + freq_extend
    for t in range(n_time):
        flag_sum[0] = 0
        for f in range(orig_freq):
            flag_sum[f + 1] = flag_sum[f] + flags[t, f // average_freq]
        # Take differences of the cumulative sums to get smearing
        tot = 0
        for f in range(orig_freq):
            f0 = max(f + freq_delta_lo, 0)
            f1 = min(f + freq_delta_hi, orig_freq)
            flag = (flag_sum[f1] != flag_sum[f0])
            out[t, f] = flag
            tot += flag
            flag_sum_time[f] += flag
        # If too much is flagged, flag the entire time
        if tot > flag_all_freq_frac * orig_freq:
            out[t, :] = True

    # Flag all times if too much is flagged. This should be rare, so we
    # write in columns even though that's normally an unfriendly access
    # pattern.
    for f in range(orig_freq):
        if flag_sum_time[f] > n_time * flag_all_time_frac:
            out[:, f] = True


@numba.jit(nopython=True, nogil=True, cache=True)
def _get_baseline_flags(
        data, flags, out_flags,
        outlier_nsigma, windows_time, windows_freq,
        background_reject, background_iterations,
        spike_width_time, spike_width_freq, time_extend, freq_extend,
        freq_chunk_ends, average_freq, flag_all_time_frac, flag_all_freq_frac,
        rho):
    """Compute flags for a single baseline. It is called after frequency
    averaging, but writes back un-averaged results.

    Parameters
    ----------
    data : ndarray, real
        Visibility magnitudes, with shape (time, frequency)
    flags : ndarray, bool
        User-input flags corresponding to `data`
    out_flags : ndarray, bool
        Returned flags (which will have more channels than `data` if
        `average_freq` is greater than 1).
    """
    n_time, n_freq = data.shape
    # Generate median spectrum, background it, and flag it
    spec_data, spec_flags = _time_median(data, flags)
    spec_background = _get_background2d(spec_data, spec_flags, background_iterations,
                                        np.array((0.0, spike_width_freq)),
                                        background_reject,
                                        freq_chunk_ends)
    spec_data -= spec_background
    spec_flags = _sum_threshold(spec_data, spec_flags, 1, windows_freq,
                                outlier_nsigma, rho, freq_chunk_ends)
    # Broadcast spectral flags to per-timestamp
    flags |= spec_flags

    # Get and subtract 2D background
    background = _get_background2d(data, flags, background_iterations,
                                   np.array(
                                       (spike_width_time, spike_width_freq)),
                                   background_reject,
                                   freq_chunk_ends)
    data -= background
    # SumThreshold along time axis
    time_flags = _sum_threshold(data, flags, 0, windows_time,
                                outlier_nsigma, rho)
    # SumThreshold along frequency axis - with time flags in the input flags
    flags |= time_flags
    freq_flags = _sum_threshold(data, flags, 1, windows_freq,
                                outlier_nsigma, rho, freq_chunk_ends)

    # Combine flag sources and do time smearing. We overwrite 'flags' since the
    # previous result is no longer needed.
    _combine_flags(spec_flags, time_flags, freq_flags, time_extend, flags)

    _unaverage_freq(flags, freq_extend, average_freq,
                    flag_all_time_frac, flag_all_freq_frac, out_flags)


def _get_flags_mp(in_data, in_flags, flagger):
    """
    Callback function for ProcessPoolExecutor.
    It allocates its own storage for the output.
    """
    out_flags = np.empty_like(in_flags)
    flagger._get_flags(in_data, in_flags, out_flags)
    return out_flags


def sum_threshold_flagger(vis, flags, chunks=None, outlier_nsigma=4.5,
                          windows_time=[1, 2, 4, 8], windows_freq=[1, 2, 4, 8],
                          background_reject=2.0, background_iterations=1,
                          spike_width_time=12.5, spike_width_freq=10.0,
                          time_extend=3, freq_extend=3,
                          freq_chunks=10, average_freq=1,
                          flag_all_time_frac=0.6, flag_all_freq_frac=0.8,
                          rho=1.3):
    """
    Flagger that uses the SumThreshold method (Offringa, A., MNRAS, 405, 155-167, 2010)
    to detect spikes in both frequency and time axes.
    The full algorithm does the following:

        1. Average the data in the frequency dimension (axis 1) into bins of
           size `self.average_freq`
        2. Divide the data into overlapping sub-chunks in frequency which are
           backgrounded and thresholded independently
        3. Flag a 1d spectrum median filtered in time to get fainter contaminated
           channels.
        4. Derive a smooth 2d background through each chunk
        5. SumThreshold the background subtracted chunks in time and frequency
        6. Extend derived flags in time and frequency, via self.freq_extend and
           self.time_extend
        7. Extend flags to all times and frequencies in cases when more than
           a given fraction of samples are flagged (via `self.flag_all_time_frac` and
           `self.flag_all_freq_frac`)

    Parameters
    ----------
    vis : :class:`numpy.ndarray`
        input visibilities
    flags : :class:`numpy.ndarray`
        input flags
    chunks : :class:`numpy.ndarray`
        chunking
    outlier_nsigma : float
        Number of sigma to reject outliers when thresholding
    windows_time : array, int
        Size of averaging windows to use in the SumThreshold method in time
    windows_freq : array, int
        Size of averaging windows to use in the SumThreshold method in frequency
    background_reject : float
        Number of sigma to reject outliers when backgrounding
    background_iterations : int
        Number of iterations to use when determining a smooth background, after each
        iteration data in excess of `background_reject`*`sigma` are masked
    spike_width_time : float
        Characteristic width in dumps to smooth over when backgrounding. This is
        the one-sigma width of the convolving Gaussian in axis 0.
    spike_width_freq : float
        Characteristic width in channels to smooth over when backgrounding. This is
        the one-sigma width of the convolving Gaussian in axis 1.
    time_extend : int
        Size of kernel in time to convolve with flags after detection
    freq_extend : int
        Size of kernel in frequency to convolve with flags after detection
    freq_chunks : int
        Number of equal-sized chunks to independently flag in frequency. Smaller
        chunks will be less affected by variations in the band in the frequency domain.
    average_freq : int
        Number of channels to average frequency before flagging. Flags will be extended
        to the frequency shape of the input data before being returned
    flag_all_time_frac : float
        Fraction of data flagged above which to extend flags to all data in time axis.
    flag_all_freq_frac : float
        Fraction of data flagged above which to extend flags to all data in frequency axis.
    rho : float
        Falloff exponent for SumThreshold
    """

    windows_freq = np.asarray(windows_freq, dtype=np.float32)
    windows_freq = np.ceil(windows_freq) / average_freq
    windows_freq = np.unique(windows_freq.astype(np.int_))

    time_extend = _as_min_dtype(time_extend)
    freq_extend = _as_min_dtype(freq_extend)
    freq_chunks = freq_chunks
    average_freq = _as_min_dtype(average_freq)

    averaged_channels = (vis.shape[1] + average_freq - 1) // average_freq

    # Set up frequency chunks
    freq_chunk_ends = np.linspace(
        0, averaged_channels, freq_chunks + 1).astype(np.int_)

    # Clip the windows to the available time and frequency range
    windows_time = np.array(
        [w for w in windows_time if w <= vis.shape[0]], np.int_)
    windows_freq = np.array(
        [w for w in windows_freq if w <= averaged_channels], np.int_)

    out_flags = np.empty_like(flags)

    _get_flags_impl(
        vis, flags, out_flags,
        outlier_nsigma, windows_time, windows_freq,
        background_reject, background_iterations,
        spike_width_time, spike_width_freq,
        time_extend, freq_extend,
        freq_chunk_ends, average_freq,
        flag_all_time_frac, flag_all_freq_frac,
        rho)

    return out_flags


class SumThresholdFlagger(object):
    """Flagger that uses the SumThreshold method (Offringa, A., MNRAS, 405, 155-167, 2010)
    to detect spikes in both frequency and time axes.
    The full algorithm does the following:

        1. Average the data in the frequency dimension (axis 1) into bins of
           size `self.average_freq`
        2. Divide the data into overlapping sub-chunks in frequency which are
           backgrounded and thresholded independently
        3. Flag a 1d spectrum median filtered in time to get fainter contaminated
           channels.
        4. Derive a smooth 2d background through each chunk
        5. SumThreshold the background subtracted chunks in time and frequency
        6. Extend derived flags in time and frequency, via self.freq_extend and
           self.time_extend
        7. Extend flags to all times and frequencies in cases when more than
           a given fraction of samples are flagged (via `self.flag_all_time_frac` and
           `self.flag_all_freq_frac`)

    Parameters
    ----------

    outlier_nsigma : float
        Number of sigma to reject outliers when thresholding
    windows_time : array, int
        Size of averaging windows to use in the SumThreshold method in time
    windows_freq : array, int
        Size of averaging windows to use in the SumThreshold method in frequency
    background_reject : float
        Number of sigma to reject outliers when backgrounding
    background_iterations : int
        Number of iterations to use when determining a smooth background, after each
        iteration data in excess of `background_reject`*`sigma` are masked
    spike_width_time : float
        Characteristic width in dumps to smooth over when backgrounding. This is
        the one-sigma width of the convolving Gaussian in axis 0.
    spike_width_freq : float
        Characteristic width in channels to smooth over when backgrounding. This is
        the one-sigma width of the convolving Gaussian in axis 1.
    time_extend : int
        Size of kernel in time to convolve with flags after detection
    freq_extend : int
        Size of kernel in frequency to convolve with flags after detection
    freq_chunks : int
        Number of equal-sized chunks to independently flag in frequency. Smaller
        chunks will be less affected by variations in the band in the frequency domain.
    average_freq : int
        Number of channels to average frequency before flagging. Flags will be extended
        to the frequency shape of the input data before being returned
    flag_all_time_frac : float
        Fraction of data flagged above which to extend flags to all data in time axis.
    flag_all_freq_frac : float
        Fraction of data flagged above which to extend flags to all data in frequency axis.
    rho : float
        Falloff exponent for SumThreshold
    """

    def __init__(self, outlier_nsigma=4.5, windows_time=[1, 2, 4, 8],
                 windows_freq=[1, 2, 4, 8], background_reject=2.0, background_iterations=1,
                 spike_width_time=12.5, spike_width_freq=10.0, time_extend=3, freq_extend=3,
                 freq_chunks=10, average_freq=1, flag_all_time_frac=0.6, flag_all_freq_frac=0.8,
                 rho=1.3):
        self.outlier_nsigma = outlier_nsigma
        self.windows_time = windows_time
        # Scale the frequency windows, and remove possible duplicates
        windows_freq = np.ceil(
            np.array(windows_freq, dtype=np.float32) / average_freq)
        self.windows_freq = np.unique(windows_freq.astype(np.int_))
        self.background_reject = background_reject
        self.background_iterations = background_iterations
        self.spike_width_time = spike_width_time
        # Scale spike_width by average_freq
        self.spike_width_freq = spike_width_freq / average_freq
        self.time_extend = _as_min_dtype(time_extend)
        self.freq_extend = _as_min_dtype(freq_extend)
        self.freq_chunks = freq_chunks
        self.average_freq = _as_min_dtype(average_freq)
        self.flag_all_time_frac = flag_all_time_frac
        self.flag_all_freq_frac = flag_all_freq_frac
        self.rho = rho

    def _get_flags(self, in_data, in_flags, out_flags):
        """Flag a batch of baselines.

        The batches are doled out by :meth:`get_flags`, either to an executor
        pool or directly. The batching is important because it affects memory
        access patterns. The batch size should not be too large, as otherwise
        it will overload the cache.

        This function is the interface between Python code and numba code, and
        takes care of conditioning the parameters into a form that the numba
        code can consume. All the actual work is done in
        :func:`_get_flags_impl`.
        """
        averaged_channels = (
            in_data.shape[1] + self.average_freq - 1) // self.average_freq

        # Set up frequency chunks
        freq_chunk_ends = np.linspace(
            0, averaged_channels, self.freq_chunks + 1).astype(np.int_)

        # Clip the windows to the available time and frequency range
        windows_time = np.array(
            [w for w in self.windows_time if w <= in_data.shape[1]], np.int_)
        windows_freq = np.array(
            [w for w in self.windows_freq if w <= averaged_channels], np.int_)

        _get_flags_impl(
            in_data, in_flags, out_flags,
            self.outlier_nsigma, windows_time, windows_freq,
            self.background_reject, self.background_iterations,
            self.spike_width_time, self.spike_width_freq,
            self.time_extend, self.freq_extend,
            freq_chunk_ends, self.average_freq,
            self.flag_all_time_frac, self.flag_all_freq_frac,
            self.rho)

    def get_flags(self, data, flags, pool=None, chunk_size=None, is_multiprocess=None):
        """Get flags in data array, with optional input flags of same shape
        that denote samples in data to ignore when backgrounding and deriving
        thresholds.

        This can run in parallel if given a
        :class:`concurrent.futures.Executor`. Performance is generally better
        with a :class:`~current.futures.ThreadPoolExecutor`. While a
        :class:`~concurrent.futures.ProcessPoolExecutor` is supported, it is
        usually limited by the speed at which the data can be pickled and
        transferred to the other processes.

        Parameters
        ----------
        data : 3D array
            The input visibility data, in (time, frequency, baseline) order. It may
            also contain just the magnitudes.
        flags : 3D array, boolean
            Input flags.
        pool : :class:`concurrent.futures.Executor`, optional
            Worker pool for parallel computation. If not specified,
            computation will be done serially.
        chunk_size : int, optional
            Number of baselines to process at a time. If not specified,
            heuristics are used to pick a reasonable value. Values above 16
            give diminishing returns and much larger values may actually reduce
            performance. Power-of-two sizes are likely to perform best.
        is_multiprocess : bool, optional
            If `pool` behaves like
            :class:`concurrent.futures.ProcessPoolExecutor` (in particular, if
            it makes copies of its arguments) then this must be set to
            ``True`` to invoke a slower path that ensures that results are
            returned and reassembled. If unspecified, it defaults to true for
            :class:`concurrent.futures.ProcessPoolExecutor` and false for all
            other types. Thus, it only needs to be specified when using an
            object that isn't a :class:`concurrent.futures.ProcessPoolExecutor`
            but behaves like one.

        Returns
        -------
        out_flags : 3D array, boolean, same shape as `data`
            Derived flags (True=flagged)

        """
        if data.shape != flags.shape:
            raise ValueError('Shape mismatch')
        if data.ndim != 3:
            raise ValueError('data has wrong number of dimensions')
        out_flags = np.empty(flags.shape, np.bool_)

        n_bl = data.shape[-1]
        if not chunk_size:
            chunk_size = 16
            if pool is not None:
                # Make sure there is enough parallelism. There is no way to
                # query the number of workers in a pool, so we'll just assume
                # it is equal to cpu_count. We want at least 4 tasks per CPU
                # to avoid load imbalances.
                workers = multiprocessing.cpu_count()
                while chunk_size > 1 and chunk_size * workers * 4 > n_bl:
                    chunk_size //= 2
        if pool is not None and is_multiprocess is None:
            is_multiprocess = isinstance(
                pool, concurrent.futures.ProcessPoolExecutor)
        futures = []
        outputs = {}
        try:
            for i in range(0, n_bl, chunk_size):
                chunk_data = data[..., i: i + chunk_size]
                chunk_flags = flags[..., i: i + chunk_size]
                chunk_out = out_flags[..., i: i + chunk_size]
                self._get_flags(chunk_data, chunk_flags, chunk_out)
                # if pool is not None and is_multiprocess:
                #     future = pool.submit(_get_flags_mp, chunk_data, chunk_flags, self)
                #     outputs[future] = chunk_out
                #     futures.append(future)
                # elif pool is not None:
                #     futures.append(pool.submit(self._get_flags, chunk_data, chunk_flags, chunk_out))
                # else:
                #    self._get_flags(chunk_data, chunk_flags, chunk_out)
            # Wait for all the futures to complete, and raise any exception.
            # In multiprocessing mode, copy results back.
            # for future in concurrent.futures.as_completed(futures):
            #     result = future.result()
            #     if is_multiprocess:
            #         outputs[future][:] = result
            return out_flags
        finally:
            pass
            # If there's an exception, stop any work we can
            # for future in futures:
            #     future.cancel()
