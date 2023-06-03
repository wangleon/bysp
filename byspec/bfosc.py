import os
import re
import datetime
import dateutil.parser

import numpy as np
import scipy.optimize as opt
from astropy.table import Table
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from .utils import get_file
from .imageproc import combine_images
from .onedarray import (iterative_savgol_filter, get_simple_ccf, gengaussian,
                        consecutive)
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
            ('exptime',     float),
            ('dateobs',     str),
            ('RAJ2000',     str),
            ('DEJ2000',     str),
            ('mode',        str),
            ('config',      str),
            ('slit',        str),
            ('filter',      str),
            ('binning',     str),
            ('gain',        float),
            ('rdnoise',     float),
            ('q99',         int),
            ('observer',    str),
        ], masked=True)


    fmt_str = ('  - {:7s} {:12s} {:>12s} {:>16s} {:>7} {:23s}'
               '{:8s} {:6s} {:8s} {:8s} {:7s} {:7s} {:7s} {:5s} {:15s}')
    head_str = fmt_str.format('frameid', 'fileid', 'datatype', 'object',
                    'exptime', 'dateobs', 'mode', 'config',
                    'slit', 'filter', 'binning', 'gain', 'rdnoise', 'q99',
                    'observer')
    if display:
        print(head_str)

    for fname in sorted(os.listdir(path)):
        if not fname.endswith('.fit'):
            continue
        mobj = re.match('(\d{12})_([A-Z]+)_(\S*)_(\S*)_(\S*)_(\S*)\.fit', fname)
        if not mobj:
            continue
        fileid = int(mobj.group(1))
        frameid = int(str(fileid)[8:])
        datatype = mobj.group(2)
        objname2 = mobj.group(3)
        key1     = mobj.group(4)
        key2     = mobj.group(5)
        key3     = mobj.group(6)

        filename = os.path.join(path, fname)
        data, header = fits.getdata(filename, header=True)
        objname    = header['OBJECT']
        exptime    = header['EXPTIME']
        dateobs    = header['DATE-OBS']
        ra         = header['RA']
        dec        = header['DEC']
        _filter    = header['FILTER']
        xbinning   = header['XBINNING']
        ybinning   = header['YBINNING']
        gain       = header['GAIN']
        rdnoise    = header['RDNOISE']
        observer   = header['OBSERVER']
        q99        = int(np.percentile(data, 99))

        binning = '{}x{}'.format(xbinning, ybinning)

        if re.match('G\d+', key3):
            mode       = 'longslit'
            config     = key3
            filtername = key2
        elif re.match('G\d+', key2) and re.match('E\d+', key3):
            mode       = 'echelle'
            config     = '{}+{}'.format(key3, key2)
            filtername = ''
        else:
            mode       = ''
            config     = ''
            filtername = key2

        if filtername == 'Free':
            filtername = ''

        # find slit width
        mobj = re.match('slit(\d+)s?', key1)
        if mobj:
            slit = str(int(mobj.group(1))/10) + '"'
        else:
            slit = ''

        if datatype in ['BIAS', 'SPECLFLAT', 'SPECLLAMP']:
            ra, dec = '', ''

        obstable.add_row((frameid, fileid, datatype, objname, exptime, dateobs,
                        ra, dec, mode, config, slit, filtername,
                        binning, float(gain), float(rdnoise), q99, observer))

        if display:
            logitem = obstable[-1]

            string = fmt_str.format(
                    '[{:d}]'.format(frameid), str(fileid),
                    '{:3s}'.format(datatype),
                    objname, exptime, dateobs[0:23], mode, config, 
                    slit, filtername, binning, gain, rdnoise,
                    '{:5d}'.format(q99), observer,
                    )
            print(print_wrapper(string, logitem))

    obstable.sort('fileid')

    return obstable

def get_mosaic_fileid(obsdate, dateobs):
    date = dateutil.parser.parse(obsdate)
    t0 = datetime.datetime.combine(date, datetime.time(0, 0, 0))
    t1 = dateutil.parser.parse(dateobs)
    delta_t = t1 - t0
    delta_minutes = int(delta_t.total_seconds()/60)
    newid = '{:4d}{:02d}{:02d}c{:4d}'.format(date.year, date.month, date.day,
                                           delta_minutes)
    return newid

def select_calib_from_database(index_file, dateobs):
    calibtable = Table.read(index_file, format='ascii.fixed_width_two_line')

    input_date = dateutil.parser.parse(dateobs)

    # select the closest ThAr
    timediff = [(dateutil.parser.parse(t)-input_date).total_seconds()
                for t in calibtable['obsdate']]
    irow = np.abs(timediff).argmin()
    row = calibtable[irow]
    fileid = row['fileid']  # selected fileid
    md5    = row['md5']

    message = 'Select {} from database index as ThAr reference'.format(fileid)
    #logger.info(message)
    print(message)

    filepath = os.path.join('calib/bfosc/', 'wlcalib_{}.fits'.format(fileid))
    filename = get_file(filepath, md5)

    # load spec, calib, and aperset from selected FITS file
    hdu_lst = fits.open(filename)
    head = hdu_lst[0].header
    spec = hdu_lst[1].data
    hdu_lst.close()

    return spec


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
        self.obsdate = obstable[0]['dateobs'][0:10]

        tablename = 'log-{}.txt'.format(self.obsdate)

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
        # x axis is the main-dispersion axis
        self.ndisp = nx
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
        pixel_lst = np.arange(self.ndisp)

        filename = os.path.dirname(__file__) + '/data/linelist/FeAr.dat'
        linelist = Table.read(filename, format='ascii.fixed_width_two_line')
        wavebound = 6907 # wavelength boundary to separate the red and blue

        index_file = os.path.join(os.path.dirname(__file__),
                                  'data/calib/wlcalib_bfosc.dat')
        ref_spec = select_calib_from_database(index_file, self.obsdate)
        ref_wave  = ref_spec['wavelength']
        ref_flux  = ref_spec['flux']
        shift_lst = np.arange(-100, 100)

        def errfunc(p, x, y, fitfunc):
            return y - fitfunc(p, x)
        def fitline(p, x):
            return gengaussian(p[0], p[1], p[2], p[3], x) + p[4]

        func = lambda item: item['datatype']=='SPECLLAMP' and \
                item['object']=='FeAr'
        lamp_item_lst = list(filter(func, self.logtable))
        logitem_groups = group_caliblamps(lamp_item_lst)

        for logitem_lst in logitem_groups:
            # determine blue and red calib lamp
            q95_lst = {}

            for logitem in logitem_lst:
                filename = self.fileid_to_filename(logitem['fileid'])
                data = fits.getdata(filename)
                q95 = np.percentile(data, 95)
                q95_lst[logitem['fileid']] = q95
            # sort FeAr lamps with intensity from the smallest to the largest
            sorted_q95_lst = sorted(q95_lst.items(), key=lambda item: item[1])
            bandselect_fileids = {
                    'R': sorted_q95_lst[0][0], # choose the smallest as red
                    'B': sorted_q95_lst[-1][0], # choose the largets as blue
                    }
            print('BLUE:', bandselect_fileids['B'])
            print('RED:',  bandselect_fileids['R'])

            # use red flux to compute ccf
            red_fileid = bandselect_fileids['R']
            flux = self.lamp_spec_lst[red_fileid]['flux']
            ccf_lst = get_simple_ccf(flux, ref_flux, shift_lst)

            fig = plt.figure()
            ax = fig.gca()
            ax.plot(shift_lst, ccf_lst, c='C3')

            pixel_corr = shift_lst[ccf_lst.argmax()]
            print(pixel_corr)

            # mosaic red and blue
            fileid_R = bandselect_fileids['R']
            fileid_B = bandselect_fileids['B']
            flux_R = self.lamp_spec_lst[fileid_R]['flux']
            flux_B = self.lamp_spec_lst[fileid_B]['flux']
            norm_R = np.percentile(flux_R, 10)
            norm_B = np.percentile(flux_B, 10)
            flux_mosaic = np.zeros(self.ndisp, dtype=np.float32)
            if ref_wave[0] > ref_wave[-1]:
                # reverse wavelength order
                idx = ref_wave.size - np.searchsorted(ref_wave[::-1], wavebound)
                idx = idx + pixel_corr
                mask = pixel_lst < idx
            else:
                idx = np.searchsorted(ref_wave, wavebound)
                idx = idx + pixel_corr
                mask = pixel_lst > idx
            flux_mosaic[mask]  = flux_R[mask]/norm_R
            flux_mosaic[~mask] = flux_B[~mask]/norm_B

            # get new fileid
            for _logitem in self.logtable:
                if _logitem['fileid'] in [fileid_R, fileid_B]:
                    dateobs = _logitem['dateobs']
            newfileid = get_mosaic_fileid(self.obsdate, dateobs)

            fig2 = plt.figure()
            ax2 = fig2.gca()
            ax2.plot(flux_mosaic)
            plt.show()

            wave_lst = []
            center_lst = []

            fig_lbl = plt.figure(figsize=(15, 8), dpi=150)
            nrow, ncol = 5, 6
            count_line = 0
            for row in linelist:
                linewave = row['wavelength']
                element  = row['element']
                ion      = row['ion']
                if ref_wave[0]>ref_wave[-1]:
                    # reverse wavelength order
                    ic = self.ndisp - np.searchsorted(ref_wave[::-1], linewave)
                else:
                    ic = np.searchsorted(ref_wave, linewave)
                ic = ic + pixel_corr
                i1, i2 = ic - 9, ic + 10
                xdata = np.arange(i1, i2)
                ydata = flux_mosaic[xdata]

                p0 = [ydata.max()-ydata.min(), 3.6, 3.5, ic, ydata.min()]
                fitres = opt.least_squares(errfunc, p0,
                            bounds=([-np.inf, 0.5,  0.1, i1, -np.inf],
                                   [np.inf, 20.0, 20.0, i2, ydata.max()]),
                            args=(xdata, ydata, fitline),
                            )

                param = fitres['x']
                A, alpha, beta, center, bkg = param
                center_lst.append(center)
                wave_lst.append(linewave)

                # plot
                ix = count_line % ncol
                iy = nrow - 1 - count_line // ncol
                axs = fig_lbl.add_axes([0.07+ix*0.16, 0.05+iy*0.18, 0.12, 0.15])
                color = 'C0' if linewave < wavebound else 'C3'
                axs.scatter(xdata, ydata, s=10, alpha=0.6, color=color)
                newx = np.linspace(i1, i2, 100)
                newy = fitline(param, newx)
                axs.plot(newx, newy, ls='-', color='C1', lw=1, alpha=0.7)
                axs.axvline(x=center, color='k', ls='--', lw=0.7)
                axs.set_xlim(newx[0], newx[-1])
                _x1, _x2 = axs.get_xlim()
                _y1, _y2 = axs.get_ylim()
                _y2 = _y2 + 0.2*(_y2 - _y1)
                _text = '{} {} {:9.4f}'.format(element, ion, linewave)
                axs.text(0.95*_x1+0.05*_x2, 0.15*_y1+0.85*_y2, _text,
                         fontsize=9)
                axs.set_ylim(_y1, _y2)
                axs.xaxis.set_major_locator(tck.MultipleLocator(5))
                axs.xaxis.set_minor_locator(tck.MultipleLocator(1))
                for tick in axs.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(9)

            
                count_line += 1
            title = 'Line-by-line fitting of {}'.format(newfileid)
            fig_lbl.suptitle(title)
            figname = 'wavelength_ident_{}.png'.format(newfileid)
            figfilename = os.path.join(self.figpath, figname)
            fig_lbl.savefig(figfilename)
            plt.close(fig_lbl)

            ######
            # fit wavelength solution
            wave_lst = np.array(wave_lst)
            center_lst = np.array(center_lst)

            idx = center_lst.argsort()
            center_lst = center_lst[idx]
            wave_lst = wave_lst[idx]

            coeff_wave = np.polyfit(center_lst, wave_lst, deg=5)
            allwave = np.polyval(coeff_wave, pixel_lst)
            fitwave = np.polyval(coeff_wave, center_lst)
            reswave = wave_lst - fitwave
            stdwave = reswave.std()

            # plot wavelength solution
            fig_sol = plt.figure(figsize=(12, 6), dpi=300)
            axt1 = fig_sol.add_axes([0.07, 0.40, 0.44, 0.52])
            axt2 = fig_sol.add_axes([0.07, 0.10, 0.44, 0.26])
            axt4 = fig_sol.add_axes([0.58, 0.54, 0.37, 0.38])
            axt5 = fig_sol.add_axes([0.58, 0.10, 0.37, 0.38])
            axt1.plot(center_lst, wave_lst, 'o', ms=4)
            axt1.plot(pixel_lst, allwave)
            axt2.plot(center_lst, reswave, 'o', ms=4)
            axt2.axhline(y=0, color='k', ls='--')
            axt2.axhline(y=stdwave, color='k', ls='--', alpha=0.4)
            axt2.axhline(y=-stdwave, color='k', ls='--', alpha=0.4)
            for ax in fig_sol.get_axes():
                ax.grid(True, color='k', alpha=0.4, ls='--', lw=0.5)
                ax.set_axisbelow(True)
                ax.set_xlim(0, self.ndisp - 1)
            y1, y2 = axt2.get_ylim()
            axt2.text(0.03 * self.ndisp, 0.2 * y1 + 0.8 * y2,
                      u'RMS = {:5.3f} \xc5'.format(stdwave))
            axt2.set_ylim(y1, y2)
            axt2.set_xlabel('Pixel')
            axt1.set_ylabel(u'\u03bb (\xc5)')
            axt2.set_ylabel(u'\u0394\u03bb (\xc5)')
            axt1.set_xticklabels([])
            axt4.plot(pixel_lst[0:-1], -np.diff(allwave))
            axt5.plot(pixel_lst[0:-1], -np.diff(allwave) / (allwave[0:-1]) * 299792.458)
            axt4.set_ylabel(u'd\u03bb/dx (\xc5)')
            axt5.set_xlabel('Pixel')
            axt5.set_ylabel(u'dv/dx (km/s)')
            title = 'Wavelength Solution for {}'.format(newfileid)
            fig_sol.suptitle(title)
            figname = 'wavelength_solution_{}.png'.format(newfileid)
            figfilename = os.path.join(self.figpath, figname)
            fig_sol.savefig(figfilename)
            plt.close(fig_sol)

            fig_imap = plt.figure(dpi=200)
            ax_imap1 = fig_imap.add_subplot(211)
            ax_imap2 = fig_imap.add_subplot(212)
            ax_imap1.plot(allwave, flux_mosaic, lw=0.5)
            _y1, _y2 = ax_imap1.get_ylim()
            ax_imap1.vlines(wave_lst, _y1, _y2, color='k', lw=0.5, ls='--')
            ax_imap1.set_ylim(_y1, _y2)
            ax_imap1.set_xlim(allwave[0], allwave[-1])
            figname = 'wavelength_identmap_{}.png'.format(newfileid)
            figfilename = os.path.join(self.figpath, figname)
            fig_imap.savefig(figfilename)
            plt.show()
            plt.close(fig_imap)


        # combine blue and red calib lamp
        flux_blue = self.lamp_spec_lst[blue_fileid]['flux']
        flux_red  = self.lamp_spec_lst[red_fileid]['flux']

        flux_combined = combine_blue_red(allwave,
                            flux_blue = flux_blue,
                            flux_red  = flux_red,
                            wavebound = wavebound,
                            baseline  = 10,
                            )
        for logitem in self.logtable:
            if logitem['fileid'] in [red_fileid, blue_fileid]:
                dateobs = logitem['dateobs']
        newfileid = get_combined_fileid(self.obsdate, dateobs)
        data = Table(dtype=[
            ('wavelength',  np.float64),
            ('flux',        np.float32),
            ])
        for _w, _f in zip(allwave, flux):
            data.add_row((_w, _f))
        head = fits.Header()
        head['OBJECT']   = 'FeAr'
        head['TELESCOP'] = 'Xinglong 2.16m'
        head['INSTRUME'] = 'BFOSC'
        head['FILEID']   = newfileid
        head['COMBINED'] = True
        head['NCOMBINE'] = 2
        head['WAVEBD']   = wavebound,
        head['FILEID1']  = blue_fileid
        head['FILEID2']  = red_fileid
        head['INSTMODE'] = logitem['mode']
        head['CONFIG']   = logitem['config']
        head['NDISP']    = self.ndisp
        head['SLIT']     = logitem['slit']
        head['FILTER']   = logitem['filter']
        head['BINNING']  = logitem['binning']
        head['GAIN']     = logitem['gain']
        head['RDNOISE']  = logitem['rdnoise']
        head['OBSERVER'] = logitem['observer']
        hdulst = fits.HDUList([
                fits.PrimaryHDU(header=head),
                fits.BinTableHDU(data=data),
            ])
        fname = 'wlcalib_{}.fits'.format(newfileid)
        filename = os.path.join(self.reduction_path, fname)
        hdulst.writeto(filename, overwrite=True)


        ident_list = Table(dtype=[
            ('wavelength',  np.float64),
            ('element',     str),
            ('ion',         str),
            ('pixel',       np.float32),
            ('residual',    np.float64),
            ('use',         int),
            ])


        self.wavelength = allwave
        self.calibflux  = flux_combined



    def plot_wlcalib(self):
        fig = plt.figure(figsize=(12,6), dpi=150)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.plot(self.wavelength, self.calibflux, lw=0.5)
        ax1.set_xlabel(u'Wavelength (\xc5)')
        plt.show()

def group_caliblamps(lamp_item_lst):
    frameid_lst = [_logitem['frameid'] for _logitem in lamp_item_lst]

    logitem_groups = []
    for group in consecutive(frameid_lst):
        logitem_lst = []
        for frameid in group:
            for _logitem in lamp_item_lst:
                if _logitem['frameid']==frameid:
                    logitem_lst.append(_logitem)
                    break
        logitem_groups.append(logitem_lst)
    return logitem_groups


BFOSC = _BFOSC()
