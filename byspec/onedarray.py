import math
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

def gengaussian(A, alpha, beta, c, x):
    return A*np.exp(-(np.abs(x-c)/alpha)**beta)

def gaussian(A, fwhm, c, x):
    s = fwhm/2.35482
    return A*np.exp(-(x-c)**2/2./s**2)

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) > stepsize)[0]+1)

def get_simple_ccf(flux1, flux2, shift_lst):
    """Get cross-correlation function of two fluxes with the given relative
    shift.
    Args:
        flux1 (:class:`numpy.ndarray`): Input flux array.
        flux2 (:class:`numpy.ndarray`): Input flux array.
        shift_lst (:class:`numpy.ndarray`): List of pixel shifts.
    Returns:
        :class:`numpy.ndarray`: Cross-correlation function
    """

    n = flux1.size
    ccf_lst = []
    for shift in shift_lst:
        segment1 = flux1[max(0,shift):min(n,n+shift)]
        segment2 = flux2[max(0,-shift):min(n,n-shift)]
        c1 = math.sqrt((segment1**2).sum())
        c2 = math.sqrt((segment2**2).sum())
        corr = np.correlate(segment1, segment2)/c1/c2
        ccf_lst.append(corr)
    return np.array(ccf_lst)

