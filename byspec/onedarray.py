import math
import numpy as np
import scipy.interpolate as intp
import scipy.optimize as opt
from scipy.signal import savgol_filter
import itertools


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
            * **mask** (:class:`numpy.ndarray`) -- Mask of y values.
            * **std** (float) -- Standard deviation.
    """
    x = np.arange(y.size)
    mask = np.ones_like(y, dtype=bool)

    for ite in range(maxiter):

        # fill masked values in y using interpolation
        f = intp.InterpolatedUnivariateSpline(x[mask], y[mask], k=2)
        ysmooth = savgol_filter(f(x),
                                window_length=winlen, polyorder=order, mode='mirror')

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


def polyfit2d(x, y, z, xorder=3, yorder=3, linear=False):
    """Two-dimensional polynomial fit.

    Args:
        x (:class:`numpy.ndarray`): Input X array.
        y (:class:`numpy.ndarray`): Input Y array.
        z (:class:`numpy.ndarray`): Input Z array.
        xorder (int): X order.
        yorder (int): Y order.
        linear (bool): Return linear solution if `True`.
    Returns:
        :class:`numpy.ndarray`: Coefficient array.

    Examples:

        .. code-block:: python

           import numpy as np
           numdata = 100
           x = np.random.random(numdata)
           y = np.random.random(numdata)
           z = 6*y**2+8*y-x-9*x*y+10*x*y**2+7+np.random.random(numdata)
           m = polyfit2d(x, y, z, xorder=1, yorder=3)
           # evaluate it on a grid
           nx, ny = 20, 20
           xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx),
                                np.linspace(y.min(), y.max(), ny))
           zz = polyval2d(xx, yy, m)

           fig1 = plt.figure(figsize=(10,5))
           ax1 = fig1.add_subplot(121,projection='3d')
           ax2 = fig1.add_subplot(122,projection='3d')
           ax1.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap='jet',
               linewidth=0, antialiased=True, alpha=0.3)
           ax1.set_xlabel('X (pixel)')
           ax1.set_ylabel('Y (pixel)')
           ax1.scatter(x, y, z, linewidth=0)
           ax2.scatter(x, y, z-polyval2d(x,y,m),linewidth=0)
           plt.show()

        if `linear = True`, the fitting only consider linear solutions such as

        .. math::

            z = a(x-x_0)^2 + b(y-y_0)^2 + c

        the returned coefficients are organized as an *m* x *n* array, where *m*
        is the order along the y-axis, and *n* is the order along the x-axis::

            1   + x     + x^2     + ... + x^n     +
            y   + xy    + x^2*y   + ... + x^n*y   +
            y^2 + x*y^2 + x^2*y^2 + ... + x^n*y^2 +
            ... + ...   + ...     + ... + ...     +
            y^m + x*y^m + x^2*y^m + ... + x^n*y^m

    """
    ncols = (xorder + 1) * (yorder + 1)
    G = np.zeros((x.size, ncols))
    ji = itertools.product(range(yorder + 1), range(xorder + 1))
    for k, (j, i) in enumerate(ji):
        G[:, k] = x ** i * y ** j
        if linear & (i != 0) & (j != 0):
            G[:, k] = 0
    coeff, residuals, _, _ = np.linalg.lstsq(G, z, rcond=None)
    coeff = coeff.reshape(yorder + 1, xorder + 1)
    return coeff


def polyval2d(x, y, m):
    """Get values for the 2-D polynomial values

    Args:
        x (:class:`numpy.ndarray`): Input X array.
        y (:class:`numpy.ndarray`): Input Y array.
        m (:class:`numpy.ndarray`): Coefficients of the 2-D polynomial.
    Returns:
        z (:class:`numpy.ndarray`): Values of the 2-D polynomial.
    """
    yorder = m.shape[0] - 1
    xorder = m.shape[1] - 1
    z = np.zeros_like(x)
    for j, i in itertools.product(range(yorder + 1), range(xorder + 1)):
        z += m[j, i] * x ** i * y ** j
    return z


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
    return A * np.exp(-(np.abs(x - c) / alpha) ** beta)


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
    s = fwhm / 2.35482
    return A * np.exp(-(x - c) ** 2 / 2. / s ** 2)


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) > stepsize)[0] + 1)


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
        segment1 = flux1[max(0, shift):min(n, n + shift)]
        segment2 = flux2[max(0, -shift):min(n, n - shift)]
        c1 = math.sqrt((segment1 ** 2).sum())
        c2 = math.sqrt((segment2 ** 2).sum())
        corr = np.correlate(segment1, segment2) / c1 / c2
        ccf_lst.append(corr)
    return np.array(ccf_lst)


def find_shift_ccf(f1, f2, shift0=0.0):
    x = np.arange(f1.size)
    interf = intp.InterpolatedUnivariateSpline(x, f1, k=3, ext=1)
    func = lambda shift: -(interf(x - shift) * f2).sum(dtype=np.float64)
    res = opt.minimize(func, shift0, method='Powell')
    return res['x']


def distortion_fitting(f1, f2):
    x = np.arange(f1.size)
    interf1 = intp.InterpolatedUnivariateSpline(x, f1, k=3, ext=1)
    interf2 = intp.InterpolatedUnivariateSpline(x, f2, k=3, ext=1)

    def func(p, x):
        return p[0] * x ** 2 + p[1] * x + p[2]

    def errfunc(p, x, func):
        return interf1(x) - interf2(x - func(p, x))

    p0 = [0, 0, 0]
    fitres = opt.least_squares(errfunc, p0,
                               bounds=([-100, -100, -100],
                                       [100, 100, 100]),
                               args=(x, func))

    if fitres.success:
        param = fitres.x
        return param[0], param[1], param[2]

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
        mask = np.zeros_like(x) < 1

    niter = 0
    while (True):

        niter += 1
        if err is None:
            mean = x[mask].mean()
            std = x[mask].std()
        else:
            mean = (x / err * mask).sum() / ((1. / err * mask).sum())
            std = math.sqrt(
                ((x - mean) ** 2 / err * mask).sum() / ((1. / err * mask).sum()))

        if maxiter == 0 or niter > maxiter:
            # return without new mask
            break

        # calculate new mask
        new_mask = mask * (x < mean + high * std) * (x > mean - low * std)

        if mask.sum() == new_mask.sum():
            break
        else:
            mask = new_mask

    return mean, std, new_mask

# def plane_model(p, x, y):
#     a, b, c = p
#     return a * x + b * y + c
#
# def errfunc(p, x, y, plane_model, z):
#     model_z = plane_model(p, x, y)
#     residuals = z - model_z
#     return residuals.ravel()
#
# p0 = [1, 1, 1]
# xdata, ydata = np.meshgrid(np.arange(nx), np.arange(ny))
# z = flat_sens
#
# result = opt.least_squares(errfunc, p0,
#                            bounds=([-np.inf, -np.inf, -np.inf],
#                                    [np.inf, np.inf, np.inf]),
#                            args=(xdata, ydata, plane_model, z)
#                            )
#
# a, b, c = result.x
# flat_sens = a * xdata + b * ydata + c
