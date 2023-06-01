import os
import re
import numpy as np
import scipy.optimize as opt
from astropy.table import Table
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from .imageproc import combine_images
from .onedarray import iterative_savgol_filter, get_simple_ccf, gengaussian
from .visual import plot_image_with_hist

def print_wrapper(string, item):
    """A wrapper for obslog printing

    """
    datatype = item['datatype']
    objname  = item['object']

    if datatype=='BIAS':
        # bias, use dim (2)
        return '\033[2m'+string.replace('\033[0m', '')+'\033[0m'

    elif datatype in ['SLITTARGET', 'SPECLTARGET']:
        # science targets, use nighlights (1)
        return '\033[1m'+string.replace('\033[0m', '')+'\033[0m'

    elif datatype=='SPECLLAMP':
        # lamp, use light yellow (93)
        return '\033[93m'+string.replace('\033[0m', '')+'\033[0m'

    else:
        return string

def make_obslog(path, display=True):
    obstable = Table(dtype=[
            ('frameid',     int),
            ('fileid',      int),
            ('datatype',    str),
            ('object',      str),
            ('mode',        str),
            ('config',      str),
            ('exptime',     float),
            ('dateobs',     str),
            ('RAJ2000',     str),
            ('DEJ2000',     str),
            ('note',        str),
        ], masked=True)


    fmt_str = ('  - {:7s} {:12s} {:>12s} {:>16s} {:8s} {:6s} {:>7} {:23s}'
               '{:10s}')
    head_str = fmt_str.format('frameid', 'fileid', 'datatype', 'object',
                              'mode', 'config', 'exptime', 'dateobs',
                              'filter')
    if display:
        print(head_str)

    for fname in sorted(os.listdir(path)):
        if not fname.endswith('.fit'):
            continue
        mobj = re.match('(\d{12})_([A-Z]+)_\S*\.fit', fname)
        if not mobj:
            continue
        fileid = int(mobj.group(1))
        frameid = int(str(fileid)[8:])
        datatype = mobj.group(2)

        filename = os.path.join(path, fname)
        header = fits.getheader(filename)
        objname    = header['OBJECT']
        exptime    = header['EXPTIME']
        dateobs    = header['DATE-OBS']
        ra         = header['RA']
        dec        = header['DEC']
        filtername = header['FILTER']
        if datatype not in ['BIAS']:
            mode = 'longslit'
        else:
            mode = ''
        config     = ''

        if datatype in ['BIAS', 'SPECLFLAT', 'SPECLLAMP']:
            ra, dec = '', ''

        obstable.add_row((frameid, fileid, datatype, objname, mode, config,
                          exptime, dateobs, ra, dec, filtername))

        if display:
            logitem = obstable[-1]

            string = fmt_str.format(
                    '[{:d}]'.format(frameid), str(fileid),
                    '{:3s}'.format(datatype),
                    objname, mode, config, exptime, dateobs[0:23],
                    filtername
                    )
            print(print_wrapper(string, logitem))

    obstable.sort('fileid')

    return obstable



class _BFOSC(object):
    def __init__(self, rawdata_path=None, mode=None):
        self.mode = mode

        if rawdata_path is not None and os.path.exists(rawdata_path):
            self.rawdata_path = rawdata_path

    def set_mode(self, mode):
        pass

    def set_path(self, **kwargs):
        pass

    def set_rawdata_path(self, rawdata_path):
        if os.path.exists(rawdata_path):
            self.rawdata_path = rawdata_path

    def set_reduction_path(self, reduction_path):
        if not os.path.exists(reduction_path):
            os.mkdir(reduction_path)
        self.reduction_path = reduction_path

        self.figpath = os.path.join(self.reduction_path, 'figures')
        if not os.path.exists(self.figpath):
            os.mkdir(self.figpath)

        self.bias_file = os.path.join(self.reduction_path, 'bias.fits')
        self.flat_file = os.path.join(self.reduction_path, 'flat.fits')
        self.sens_file = os.path.join(self.reduction_path, 'sens.fits')

    def make_obslog(self):
        obstable = make_obslog(self.rawdata_path, display=True)

        # find obsdate
        obsdate = obstable[0]['dateobs'][0:10]

        tablename = 'log-{}.txt'.format(obsdate)

        filename = os.path.join(self.reduction_path, tablename)
        obstable.write(filename, format='ascii.fixed_width_two_line',
                        overwrite=True)
        ###
        self.logtable = obstable

    def fileid_to_filename(self, fileid):
        for fname in os.listdir(self.rawdata_path):
            if fname.startswith(str(fileid)):
                return os.path.join(self.rawdata_path, fname)
        raise ValueError

    def get_bias(self):

        if os.path.exists(self.bias_file):
            self.bias_data = fits.getdata(self.bias_file)

        else:
            print('Combine bias')
            data_lst = []

            bias_item_lst = filter(lambda item: item['datatype']=='BIAS',
                                   self.logtable)

            for logitem in bias_item_lst:
                filename = self.fileid_to_filename(logitem['fileid'])
                data = fits.getdata(filename)
                data_lst.append(data)
            data_lst = np.array(data_lst)

            bias_data = combine_images(data_lst, mode='mean',
                            upper_clip=5, maxiter=10, maskmode='max')
            fits.writeto(self.bias_file, bias_data, overwrite=True)
            self.bias_data = bias_data

    def plot_bias(self, show=True):
        figfilename = os.path.join(self.figpath, 'bias.png')
        title = 'Bias ({})'.format(os.path.basename(self.bias_file))
        plot_image_with_hist(self.bias_data,
                        show        = show,
                        figfilename = figfilename,
                        title       = title,
                        )

    def plot_flat(self, show=True):
        figfilename = os.path.join(self.figpath, 'flat.png')
        title = 'Flat ({})'.format(os.path.basename(self.flat_file))
        plot_image_with_hist(self.flat_data,
                        show        = show,
                        figfilename = figfilename,
                        title       = title,
                        )

    def plot_sens(self, show=True):
        figfilename = os.path.join(self.figpath, 'sens.png')
        title = 'Sensitivity ({})'.format(os.path.basename(self.sens_file))
        plot_image_with_hist(self.sens_data,
                        show        = show,
                        figfilename = figfilename,
                        title       = title,
                        )

    def combine_flat(self):
        if os.path.exists(self.flat_file):
            self.flat_data = fits.getdata(self.flat_file)
        else:
            print('Combine Flat')
            data_lst = []

            flat_item_lst = filter(lambda item: item['datatype']=='SPECLFLAT',
                                   self.logtable)
            for logitem in flat_item_lst:
                filename = self.fileid_to_filename(logitem['fileid'])
                data = fits.getdata(filename)
                # correct bias
                data = data - self.bias_data
                data_lst.append(data)
            data_lst = np.array(data_lst)
            flat_data = combine_images(data_lst, mode='mean',
                                    upper_clip=5, maxiter=10, maskmode='max')
            fits.writeto(self.flat_file, flat_data, overwrite=True)
            self.flat_data = flat_data

    def get_sens(self):
        ny, nx = self.flat_data.shape
        allx = np.arange(nx)
        flat_sens = np.ones_like(self.flat_data, dtype=np.float64)

        for y in np.arange(ny):
            #flat1d = self.flat_data[y, 20:int(nx) - 20]
            flat1d = self.flat_data[y, :]

            flat1d_sm, _, mask, std = iterative_savgol_filter(flat1d,
                                        winlen=51, order=3,
                                        upper_clip=6, lower_clip=6, maxiter=10)
            #flat_sens[y, 20:int(nx) - 20] = flat1d / flat1d_sm
            flat_sens[y, :] = flat1d / flat1d_sm

        fits.writeto(self.sens_file, flat_sens, overwrite=True)

        self.sens_data = flat_sens

    def extract_lamp(self):
        lamp_item_lst = filter(lambda item: item['datatype']=='SPECLLAMP',
                               self.logtable)

        hwidth = 5

        spec_lst = {} # use to save the extracted 1d spectra of calib lamp

        for logitem in lamp_item_lst:
            fileid = logitem['fileid']
            filename = self.fileid_to_filename(fileid)
            data = fits.getdata(filename)
            data = data - self.bias_data
            data = data / self.sens_data

            ny, nx = data.shape

            # extract 1d spectra of wavelength calibration lamp
            spec = data[ny//2-hwidth: ny//2+hwidth+1, :].sum(axis=0)
            spec_lst[fileid] = {'wavelength': None, 'flux': spec}

        self.lamp_spec_lst = spec_lst

    def identify_wavelength(self):

        filename = os.path.dirname(__file__) + '/data/linelist/FeAr.dat'
        linelist = Table.read(filename, format='ascii.fixed_width_two_line')
        wavebound = 6900 # wavelength boundary to separate the red and blue

        # determine blue and red calib lamp
        q95_lst = {}

        lamp_item_lst = filter(lambda item: item['datatype']=='SPECLLAMP',
                               self.logtable)

        for logitem in lamp_item_lst:
            filename = self.fileid_to_filename(logitem['fileid'])
            data = fits.getdata(filename)
            q95 = np.percentile(data, 95)
            q95_lst[logitem['fileid']] = q95
        sorted_q95_lst = sorted(q95_lst.items(), key=lambda item: item[1])
        blue_fileid = sorted_q95_lst[0][0]
        red_fileid  = sorted_q95_lst[-1][0]
        print('BLUE:', blue_fileid)
        print('RED:',  red_fileid)

        def errfunc(p, x, y, fitfunc):
            return y - fitfunc(p, x)
        def fitline(p, x):
            return gengaussian(p[0], p[1], p[2], p[3], x) + p[4]

        fig = plt.figure(figsize=(15, 8))
        nrow, ncol = 4, 6

        linelist=np.loadtxt('wl_guess.txt2') 
        wlred = np.loadtxt('wlcalib_red.dat')
        wlblue = np.loadtxt('wlcalib_blue.dat')

        count_line = 0
        center_lst = []
        wave_lst = []

        shift_lst = np.arange(-100, 100)
        for fileid in [red_fileid, blue_fileid]:
            flux = self.lamp_spec_lst[fileid]['flux']
            if fileid == red_fileid:
                m = linelist[:, 1] > wavebound
                ccf_lst = get_simple_ccf(flux, wlred[:, 1], shift_lst)
                band = 'red'
            elif fileid == blue_fileid:
                m = linelist[:, 1] < wavebound
                ccf_lst = get_simple_ccf(flux, wlblue[:,1], shift_lst)
                band = 'blue'
            else:
                raise ValueError

            '''
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(shift_lst, ccf_lst)

            fig2 = plt.figure()
            ax2 = fig2.gca()
            ax2.plot(flux)
            ax2.plot(wlred[:,1])
            ax2.plot(wlblue[:,1])
            '''

            pixel_correct = shift_lst[ccf_lst.argmax()]
            cutlinelist = linelist[m]

            init_pixel_lst = cutlinelist[:, 0] + pixel_correct
            init_wave_lst = cutlinelist[:, 1]

            for x, wave in zip(init_pixel_lst, init_wave_lst):
                nx = flux.size
                allx = np.arange(nx)
                i = np.searchsorted(allx, x)
                i1, i2 = i - 9, i + 10
                xdata = allx[i1:i2]
                ydata = flux[i1:i2]

                p0 = [ydata.max()-ydata.min(), 3.6, 3.5, i, ydata.min()]
                fitres = opt.least_squares(errfunc, p0,
                            bounds=([-np.inf, 0.5,  0.1, i1, -np.inf],
                                    [np.inf, 20.0, 20.0, i2, ydata.max()]),
                            args=(xdata, ydata, fitline),
                        )

                p = fitres['x']
                A, alpha, beta, center, bkg = p
                center_lst.append(center)
                wave_lst.append(wave)

                ix = count_line % ncol
                iy = nrow - 1 - count_line // ncol
                axs = fig.add_axes([0.07+ix*0.16, 0.08+iy*0.23, 0.12, 0.20])
                if band == 'red':
                    color = 'C3'
                elif band == 'blue':
                    color = 'C0'
                else:
                    color = 'k'
                axs.scatter(xdata, ydata, s=10, alpha=0.6, color=color)
                newx = np.arange(i1, i2, 0.1)
                newy = fitline(p, newx)
                axs.plot(newx, newy, ls='-', color='C1', lw=1, alpha=0.7)
                axs.axvline(x=center, color='k', ls='--', lw=0.7)
                axs.set_xlim(newx[0], newx[-1])
                x1, x2 = axs.get_xlim()
                y1, y2 = axs.get_ylim()
                axs.text(0.95*x1+0.05*x2, 0.2*y1+0.8*y2, '{:9.4f}'.format(wave),
                         fontsize=9)
                axs.xaxis.set_major_locator(tck.MultipleLocator(5))
                axs.xaxis.set_minor_locator(tck.MultipleLocator(1))
                for tick in axs.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(9)
                
                count_line += 1

        figname = 'fitlines.png'
        figfilename = os.path.join(self.figpath, figname)
        fig.savefig(figfilename)
        plt.show()
        plt.close(fig)


        wave_lst = np.array(wave_lst)
        center_lst = np.array(center_lst)

        idx = center_lst.argsort()
        center_lst = center_lst[idx]
        wave_lst = wave_lst[idx]

        coeff_wave = np.polyfit(center_lst, wave_lst, deg=5)
        allwave = np.polyval(coeff_wave, allx)
        fitwave = np.polyval(coeff_wave, center_lst)
        reswave = wave_lst - fitwave
        stdwave = reswave.std()


        # plot wavelength solution
        figt = plt.figure(figsize=(12, 6), dpi=300)
        axt1 = figt.add_axes([0.07, 0.40, 0.44, 0.52])
        axt2 = figt.add_axes([0.07, 0.10, 0.44, 0.26])
        #axt3 = figt.add_axes([0.07, 0.10, 0.44, 0.26])
        axt4 = figt.add_axes([0.58, 0.54, 0.37, 0.38])
        axt5 = figt.add_axes([0.58, 0.10, 0.37, 0.38])
        axt1.scatter(center_lst, wave_lst, s=20)
        axt1.plot(allx, allwave)
        axt2.scatter(center_lst, wave_lst - np.polyval(coeff_wave, center_lst), s=20)
        axt2.axhline(y=0, color='k', ls='--')
        axt2.axhline(y=stdwave, color='k', ls='--', alpha=0.4)
        axt2.axhline(y=-stdwave, color='k', ls='--', alpha=0.4)
        for ax in figt.get_axes():
            ax.grid(True, color='k', alpha=0.4, ls='--', lw=0.5)
            ax.set_axisbelow(True)
            ax.set_xlim(0, nx - 1)
        y1, y2 = axt2.get_ylim()
        axt2.text(0.03 * nx, 0.2 * y1 + 0.8 * y2, u'RMS = {:5.3f} \xc5'.format(stdwave))
        axt2.set_ylim(y1, y2)
        axt2.set_xlabel('Pixel')
        axt1.set_ylabel(u'\u03bb (\xc5)')
        axt2.set_ylabel(u'\u0394\u03bb (\xc5)')
        axt1.set_xticklabels([])
        axt4.plot(allx[0:-1], -np.diff(allwave))
        axt5.plot(allx[0:-1], -np.diff(allwave) / (allwave[0:-1]) * 299792.458)
        axt4.set_ylabel(u'd\u03bb/dx (\xc5)')
        axt5.set_xlabel('Pixel')
        axt5.set_ylabel(u'dv/dx (km/s)')
        title = 'Wavelength Solution'
        figt.suptitle(title)
        figname = 'wavelength.png'
        figfilename = os.path.join(self.figpath, figname)
        figt.savefig(figfilename)
        plt.close(figt)


BFOSC = _BFOSC()
