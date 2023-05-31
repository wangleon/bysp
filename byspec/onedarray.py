import numpy as np
import scipy.interpolate as intp
from scipy.signal import savgol_filter

def iterative_savgol_filter(y, winlen=5, order=3, maxiter=10,
        upper_clip=None, lower_clip=None):
    """Smooth the input array with Savitzky-Golay filter with lower and/or
    upper clippings.

    Args:
        y (:class:`numpy.ndarray`): Input array.
        winlen (int): Window length of Savitzky-Golay filter.
        order (int): Order of Savitzky-Gaoly filter.
        maxiter (int): Maximum number of iterations.
        lower_clip (float): Lower sigma-clipping value.
        upper_clip (float): Upper sigma-clipping value.

    Returns:
        tuple: A tuple containing:

            * **ysmooth** (:class:`numpy.ndarray`) -- Smoothed y values.
            * **yres** (:class:`numpy.ndarray`) -- Residuals of y values.
            * **mask** (:class:`numpy.ndarray`) – Mask of y values.
            * **std** (float) – Standard deviation.
    """
    x = np.arange(y.size)
    mask = np.ones_like(y, dtype=np.bool)

    for ite in range(maxiter):

        # fill masked values in y using interpolation
        f = intp.InterpolatedUnivariateSpline(x[mask], y[mask], k=3)
        ysmooth = savgol_filter(f(x),
                    window_length=winlen, polyorder=order, mode='mirror')
        yres = y - ysmooth
        std = yres[mask].std()

        # generate new mask
        # make a copy of existing mask
        new_mask = mask * np.ones_like(mask, dtype=np.bool)
        # give new mask with lower and upper clipping value
        if lower_clip is not None:
            new_mask *= (yres > -lower_clip * std)
        if upper_clip is not None:
            new_mask *= (yres < upper_clip * std)

        if new_mask.sum() == mask.sum():
            break
        mask = new_mask

    return ysmooth, yres, mask, std

