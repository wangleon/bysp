import math
import numpy as np
import scipy.interpolate as intp
import scipy.optimize as opt
from scipy.signal import savgol_filter

def iterative_savgol_filter(y, winlen=5, order=3, mode='interp', maxiter=10,
        upper_clip=None, lower_clip=None):
    """Smooth the input array with Savitzky-Golay filter with lower and/or
    upper clippings.

    Args:
        y (:class:`numpy.ndarray`): Input array.
        winlen (int): Window length of Savitzky-Golay filter.
        mode (str):
        order (int): Order of Savitzky-Gaoly filter.
        maxiter (int): Maximum number of iterations.
        lower_clip (float): Lower sigma-clipping value.
        upper_clip (float): Upper sigma-clipping value.

    Returns:
        tuple: A tuple containing:

            * **ysmooth** (:class:`numpy.ndarray`) -- Smoothed y values.
            * **yres** (:class:`numpy.ndarray`) -- Residuals of y values.
            * **mask** (:class:`numpy.ndarray`) -- Mask of y values.
            * **std** (float) -- Standard deviation.
    """
    x = np.arange(y.size)
    mask = np.ones_like(y, dtype=bool)

    for ite in range(maxiter):

        # fill masked values in y using interpolation
        f = intp.InterpolatedUnivariateSpline(x[mask], y[mask], k=3)
        ysmooth = savgol_filter(f(x),
                    window_length=winlen, polyorder=order, mode=mode)
        yres = y - ysmooth
        std = yres[mask].std()

        # generate new mask
        # make a copy of existing mask
        new_mask = mask * np.ones_like(mask, dtype=bool)
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
    """Generalized Gaussian function.

    Args:
        A (float): Amplitude.
        alpha (float): alpha value.
        beta (float): beta value.
        c (float): Central posistion of the function.
        x (:class:`numpy.ndarray`): Input X values.
    Returns:
        :class:`numpy.ndarray`: Function values.

    """
    return A*np.exp(-(np.abs(x-c)/alpha)**beta)

def gaussian(A, fwhm, c, x):
    """Gaussian function.

    Args:
        A (float): Amplitude.
        fwhm (float): Full-width half maximum.
        c (float): Central posistion of the function.
        x (:class:`numpy.ndarray`): Input X values.
    Returns:
        :class:`numpy.ndarray`: Function values.

    """
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

def find_shift_ccf(f1, f2, shift0=0.0):
    x = np.arange(f1.size)
    interf = intp.InterpolatedUnivariateSpline(x, f1, k=3, ext=1)
    func = lambda shift: -(interf(x - shift)*f2).sum(dtype=np.float64)
    res = opt.minimize(func, shift0, method='Powell')
    return res['x']

def get_clip_mean(x, err=None, mask=None, high=3, low=3, maxiter=5):
    """Get the mean value of an input array using the sigma-clipping method

    Args:
        x (:class:`numpy.ndarray`): The input array.
        err (:class:`numpy.ndarray`): Errors of the input array.
        mask (:class:`numpy.ndarray`): Initial mask of the input array.
        high (float): Upper rejection threshold.
        low (float): Loweer rejection threshold.
        maxiter (int): Maximum number of iterations.

    Returns:
        tuple: A tuple containing:

            * **mean** (*float*) – Mean value after the sigma-clipping.
            * **std** (*float*) – Standard deviation after the sigma-clipping.
            * **mask** (:class:`numpy.ndarray`) – Mask of accepted values in the
              input array.
    """
    x = np.array(x)
    if mask is None:
        mask = np.zeros_like(x)<1

    niter = 0
    while(True):

        niter += 1
        if err is None:
            mean = x[mask].mean()
            std  = x[mask].std()
        else:
            mean = (x/err*mask).sum()/((1./err*mask).sum())
            std = math.sqrt(((x - mean)**2/err*mask).sum()/((1./err*mask).sum()))

        if maxiter==0 or niter>maxiter:
            # return without new mask
            break

        # calculate new mask
        new_mask = mask * (x < mean + high*std) * (x > mean - low*std)

        if mask.sum() == new_mask.sum():
            break
        else:
            mask = new_mask

    return mean, std, new_mask

def get_local_minima(x, window=None):
    """Get the local minima of a 1d array in a window.

    Args:
        x (:class:`numpy.ndarray`): A list or Numpy 1d array.
        window (*int* or :class:`numpy.ndarray`): An odd integer or a list of
            odd integers as the lengthes of searching window.
    Returns:
        tuple: A tuple containing:

            * **index** (:class:`numpy.ndarray`): A numpy 1d array containing 
              indices of all local minima.
            * **x[index]** (:class:`numpy.ndarray`): A numpy 1d array containing
              values of all local minima.

    """
    x = np.array(x)
    dif = np.diff(x)
    ind = dif > 0
    tmp = np.logical_xor(ind, np.roll(ind,1))
    idx = np.logical_and(tmp,ind)
    index = np.where(idx)[0]
    if window is None:
        # window is not given
        return index, x[index]
    else:
        # window is given
        if isinstance(window, int):
            # window is an integer
            window = np.repeat(window, len(x))
        elif isinstance(window, np.ndarray):
            # window is a numpy array
            #if np.issubdtype(window.dtype, int):
            if window.dtype.type in [np.int16, np.int32, np.int64]:
                pass
            else:
                # window are not integers
                print('window array are not integers')
                raise ValueError
        else:
            raise ValueError

        if 0 in window%2:
            # not all of the windows are odd
            raise ValueError

        halfwin_lst = (window-1)//2
        index_lst = []
        for i in index:
            halfwin = halfwin_lst[i]
            i1 = max(0, i-halfwin)
            i2 = min(i+halfwin+1, len(x))
            if i == x[i1:i2].argmin() + i1:
                index_lst.append(i)
        if len(index_lst)>0:
            index_lst = np.array(index_lst)
            return index_lst, x[index_lst]
        else:
            return np.array([]), np.array([])

