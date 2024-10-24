import os
import re
import datetime
import dateutil.parser

import numpy as np
import scipy.signal as sg
import scipy.interpolate as intp
import scipy.optimize as opt
from astropy.table import Table
import astropy.io.fits as fits

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
#from mpl_toolkits import mplot3d

from .utils import get_file
from .imageproc import combine_images
from .onedarray import (iterative_savgol_filter, get_simple_ccf,
                        gaussian, gengaussian,
                        find_shift_ccf)
from .common import (FOSCReducer, find_distortion, correct_distortion,
                     find_longslit_wavelength)

def print_wrapper(string, item):
    """A wrapper for obslog printing

    """
    datatype = item['datatype']
    objname  = item['object']

    if datatype=='BIAS':
        # bias, use dim (2)
        return '\033[2m'+string.replace('\033[0m', '')+'\033[0m'

    elif datatype in ['EXPOSE']:
        # science targets, use nighlights (1)
        return '\033[1m'+string.replace('\033[0m', '')+'\033[0m'

    elif datatype in ['LAMPFLAT', 'LAMP']:
        # lamp, use light yellow (93)
        return '\033[93m'+string.replace('\033[0m', '')+'\033[0m'

    else:
        return string

def make_obslog(path, display=True):
    obstable = Table(dtype=[
            ('frameid',     int),
            ('fileid',      str),
            ('datatype',    str),
            ('object',      str),
            ('exptime',     float),
            ('dateobs',     str),
            ('RAJ2000',     str),
            ('DEJ2000',     str),
            ('mode',        str),
            ('config',      str),
            ('slit',        str),
            #('filter',      str),
            ('binning',     str),
            ('gain',        float),
            ('rdspeed',     str),
            ('q99',         int),
            #('observer',    str),
        ], masked=True)

    fmt_str = ('  - {:7s} {:17s} {:>12s} {:>16s} {:>8} {:23s}'
               ' {:8s} {:8s} {:8s} {:7s} {:5s} {:7s} {:5s}')
    head_str = fmt_str.format('frameid', 'fileid', 'datatype', 'object',
                    'exptime', 'dateobs', 'mode', 'config',
                    'slit', 'binning', 'gain', 'rdspeed', 'q99')
    if display:
        print(head_str)

    fn_pattern = 'ljg2m401\-yf01\-(\d{8}\-\d{4}\-[a-z]\d{2})\.fits\.fz'

    for fname in sorted(os.listdir(path)):
        if not (fname.endswith('.fits') or fname.endswith('.fits.fz')):
            continue
        filename = os.path.join(path, fname)
        hdulst = fits.open(filename)
        head = hdulst[1].header
        data = hdulst[1].data
        hdulst.close()

        # get fileid
        mobj = re.match(fn_pattern, fname)
        if not mobj:
            continue
        fileid      = mobj.group(1)

        # get header keywords
        frameid  = head['FRAMENUM']
        datatype = head['OBSTYPE']
        if datatype in ['BIAS', 'LAMPFLAT', 'LAMP']:
            ra, dec = '', ''
        else:
            ra  = head['RA']
            dec = head['DEC']
        objname = head['OBJECT']
        exptime = head['EXPTIME']
        dateobs = head['DATE-OBS']
        filter1 = head['FILTER1']
        filter2 = head['FILTER2']
        filter3 = head['FILTER3']
        filter4 = head['FILTER4']
        filter5 = head['FILTER5']
        filter6 = head['FILTER6']
        filter7 = head['FILTER7']

        #print(frameid, fileid, datatype, exptime, dateobs, filter1, filter2,
        #  filter3, filter4, filter5, filter6, filter7)

        # determine instrument mode and config
        if datatype=='BIAS' and exptime < 0.1:
            # BIAS is shared for all modes and configs
            objname = 'BIAS'
            mode, config, slit = '', '', ''
        elif filter1[0:5]=='lslit' and filter3[0:7]!='echelle':
            mode = 'longslit'
            config = filter3
            slit = filter1[5:].replace('_', '.') + '"'
        elif filter1[0:5]=='sslit' and filter3[0:7]=='echelle':
            mode = 'echelle'
            config = filter3
            slit = filter1[5:].replace('_', '.') + '"'
        else:
            mode = 'unknown'
            config = filter3
            slit = 'unknown'

        # determine the datatype and objname if this frame is not BIAS
        if datatype != 'BIAS' and filter6=='mirror_in':
            if datatype=='LAMPFLAT' and filter7=='lamp_halogen':
                objname = 'FLAT'
            elif datatype=='EXPOSE':
                if filter7=='lamp_neon_helium':
                    datatype = 'LAMP'
                    objname = 'HeNe'
                elif filter7=='lamp_fe_argon':
                    datatype = 'LAMP'
                    objname = 'FeAr'
                else:
                    objname = 'unknown'
            else:
                pass


        # determine CCD parameters (binning, gain, readout noise)
        ccdsum = head['CCDSUM']
        binning = ccdsum.replace(' ','x')
        gain    = head['GAIN']
        rdspeed = str(int(head['RDSPEED']))+'kps'
        rdnoise = head['RDNOISE']

        # get 99 percentile
        q99 = np.percentile(data, 99)
    
        obstable.add_row((frameid, fileid, datatype, objname, exptime, dateobs,
                          ra, dec, mode, config, slit, binning, gain, rdspeed,
                          q99))

        if display:
            logitem = obstable[-1]

            string = fmt_str.format(
                    '[{:d}]'.format(frameid), str(fileid),
                    datatype,
                    objname,
                    '{:8.3f}'.format(exptime),
                    dateobs[0:23], mode, config, 
                    slit, binning, str(gain), rdspeed,
                    '{:5d}'.format(int(q99)),
                    )
            print(print_wrapper(string, logitem))

    obstable['exptime'].info.format='%7.3f'
    return obstable

def trim_rawdata(data):
    """Trim image"""
    ny, nx = data.shape
    if (ny, nx)==(4612, 2148):
        #y1, y2 = 1900, 4300
        y1, y2 = 1650, 4250
        return data[y1:y2, :]
    else:
        raise ValueError

def trim_longslit(data):
    """Trim"""
    ny, nx = data.shape
    if nx == 2048 and ny==2600:
        y1 = 500
        return data[y1:, :]
    else:
        print(data.shape)
        raise ValueError

def correct_overscan(data):
    ny, nx = data.shape
    if nx == 2148:
        overscan1 = data[:, 0:48] # first 50 columns are overscan regions
        overscan2 = data[:, -48:] # last 50 columns are overscan regions
        # take the mean of two prescan and overscan regions
        overdata = (overscan1.mean(axis=1) + overscan2.mean(axis=1))/2
        n = overdata.size
        ally = np.arange(ny)
        coeff = np.polyfit(ally, overdata, deg=1)
        overmean = np.polyval(coeff, ally)

        scidata = data[:, 50:-50]
        ncol = scidata.shape[1]
        overdata = np.repeat(overmean, ncol).reshape(-1, ncol)
        return scidata - overdata
    else:
        raise ValueError

def get_longslit_sensmap(data):
    ny, nx = data.shape
    allx = np.arange(nx)
    ally = np.arange(ny)
    sensmap = np.ones_like(data, dtype=np.float64)

    for x in allx:
        # take columns
        flux1d = data[:, x]
        flux1d_sm, _, mask, std = iterative_savgol_filter(
                                    flux1d,
                                    winlen=51, order=3, mode='interp',
                                    upper_clip=6, lower_clip=6, maxiter=10)
        sensmap[:, x] = flux1d/flux1d_sm

    return sensmap

def select_calib_from_database(index_file, lamp, mode, config, dateobs):
    calibtable = Table.read(index_file, format='ascii.fixed_width_two_line')

    # select the corresponding arclamp, mode, and config
    mask = (calibtable['object'] == lamp) * \
           (calibtable['mode']   == mode) * \
           (calibtable['config'] == config)

    calibtable = calibtable[mask]
    
    input_date = dateutil.parser.parse(dateobs)

    # select the closest ThAr
    timediff = [(dateutil.parser.parse(t)-input_date).total_seconds()
                for t in calibtable['dateobs']]
    irow = np.abs(timediff).argmin()
    row = calibtable[irow]
    fileid = row['fileid']  # selected fileid
    md5    = row['md5']

    message = 'Select {} from database index as {} reference'.format(fileid, lamp)
    print(message)

    filepath = os.path.join('calib/yfosc/', 'wlcalib_{}.fits'.format(fileid))
    filename = get_file(filepath, md5)

    # load spec, calib, and aperset from selected FITS file
    hdu_lst = fits.open(filename)
    head = hdu_lst[0].header
    spec = hdu_lst[1].data
    linelist = Table(hdu_lst[2].data)
    hdu_lst.close()

    return spec, linelist


class YFOSC(FOSCReducer):
    def __init__(self, **kwargs):
        super(YFOSC, self).__init__(**kwargs)

    def make_obslog(self, filename=None):
        logtable = make_obslog(self.rawdata_path)

        # find obsdate
        self.obsdate = logtable[0]['dateobs'][0:10]

        if filename is None:
            filename = 'YFOSC.{}.txt'.format(self.obsdate)

        filename = os.path.join(self.reduction_path, filename)

        logtable.write(filename, format='ascii.fixed_width_two_line',
                       overwrite=True)

        self.logtable = logtable


    def get_bias(self):

        self.get_all_ccdconf()
        self.bias = {}

        for ccdconf in self.ccdconf_lst:
            ccdconf_string = self.get_ccdconf_string(ccdconf)
            bias_file = 'bias_{}.fits'.format(ccdconf_string)
            bias_filename = os.path.join(self.reduction_path, bias_file)
            if os.path.exists(bias_filename):
                hdulst = fits.open(bias_filename)
                bias_data = hdulst[1].data
                bias_img  = hdulst[2].data
                hdulst.close()
            else:
                print('Combine Bias')
                selectfunc = lambda item: item['datatype']=='BIAS' and \
                                self.get_ccdconf(item)==ccdconf

                item_lst = list(filter(selectfunc, self.logtable))
                if len(item_lst)==0:
                    continue

                data_lst = []
                for logitem in item_lst:
                    filename = self.fileid_to_filename(logitem['fileid'])
                    data = fits.getdata(filename)
                    data = trim_rawdata(data)
                    data = correct_overscan(data)
                    data_lst.append(data)
                data_lst = np.array(data_lst)

                bias_data = combine_images(data_lst, mode='mean',
                                upper_clip=5, maxiter=10, maskmode='max')

                # get the mean cross-section of coadded bias image
                section = bias_data.mean(axis=0)

                # fix the hot columns
                if section.size == 2048:
                    mask = np.ones(section.size, dtype=bool)
                    mask[1664:1668] = False
                    allx = np.arange(section.size)
                    fintp = intp.InterpolatedUnivariateSpline(
                            allx[mask], section[mask], k=3)
                    section = fintp(allx)

                # get the smoothed cross-section
                smsection = sg.savgol_filter(section,
                            window_length=151, polyorder=3)
                # broadcast the smoothed section to the entire image
                ny, nx = bias_data.shape
                #bias_img = np.repeat(smsection, nx).reshape(-1, nx)
                bias_img = np.tile(smsection, ny).reshape(ny, -1)

                # save to fits file
                hdulst = fits.HDUList([fits.PrimaryHDU(),
                                       fits.ImageHDU(data=bias_data),
                                       fits.ImageHDU(data=bias_img),
                                       ])
                hdulst.writeto(bias_filename, overwrite=True)

            self.bias[ccdconf] = bias_img

    def combine_flat(self):

        self.get_all_conf()
        self.flat = {}

        for conf in self.conf_lst:
            conf_string = self.get_conf_string(conf)
            flat_file = 'flat_{}.fits'.format(conf_string)
            flat_filename = os.path.join(self.reduction_path, flat_file)
            if os.path.exists(flat_filename):
                flat_data = fits.getdata(flat_filename)
            else:
                selectfunc = lambda item: item['datatype']=='LAMPFLAT' and \
                                self.get_conf(item)==conf

                item_lst = list(filter(selectfunc, self.logtable))

                if len(item_lst)==0:
                    continue

                print('Combine Flat for {}'.format(conf_string))

                data_lst = []
                for logitem in item_lst:
                    filename = self.fileid_to_filename(logitem['fileid'])
                    data = fits.getdata(filename)
                    data = trim_rawdata(data)
                    data = correct_overscan(data)
                    ccdconf = self.get_ccdconf(logitem)
                    data = data - self.bias[ccdconf]
                    data_lst.append(data)
                data_lst = np.array(data_lst)
                
                flat_data = combine_images(data_lst, mode='mean',
                                    upper_clip=5, maxiter=10, maskmode='max')
                fits.writeto(flat_filename, flat_data, overwrite=True)

            self.flat[conf] = flat_data

    def get_longslit_sens(self):
        self.sensmap = {}
        for conf, flat_data in self.flat.items():
            mode = conf[0]
            conf_string = self.get_conf_string(conf)
            if mode == 'longslit':
                sensmap = get_longslit_sensmap(flat_data)
                sens_file = 'sens_{}.fits'.format(conf_string)
                sens_filename = os.path.join(self.reduction_path, sens_file)
                fits.writeto(sens_filename, sensmap, overwrite=True)
                self.sensmap[conf] = sensmap

    def extract_longslit_arclamp(self):
        self.arclamp = {}
        for logitem in self.logtable:

            # select longslit mode
            if logitem['mode'] != 'longslit':
                continue

            if logitem['datatype'] != 'LAMP':
                continue

            ccdconf = self.get_ccdconf(logitem)
            conf = self.get_conf(logitem)

            filename = self.fileid_to_filename(logitem['fileid'])
            data = fits.getdata(filename)
            data = trim_rawdata(data)
            data = correct_overscan(data)
            data = data - self.bias[ccdconf]
            data = data / self.sensmap[conf]
            data = trim_longslit(data)

            ny, nx = data.shape
            halfwidth = 5
            spec = data[:, nx//2-halfwidth:nx//2+halfwidth].mean(axis=1)

            self.arclamp[logitem['fileid']] = spec

    def ident_longslit_wavelength(self):
        self.wave = {}
        self.ident = {}
        index_file = os.path.join(os.path.dirname(__file__),
                              'data/calib/wlcalib_yfosc.dat')

        for logitem in self.logtable:
            if logitem['mode']!='longslit' or logitem['datatype']!='LAMP':
                continue
            if logitem['fileid'] not in self.arclamp:
                continue

            fileid = logitem['fileid']
            lamp   = logitem['object']

            spec = self.arclamp[fileid]

            ref_data, linelist = select_calib_from_database(index_file,
                                lamp = lamp,
                                mode = logitem['mode'],
                                config = logitem['config'],
                                dateobs = logitem['dateobs'],
                                )
            ref_wave = ref_data['wavelength']
            ref_flux = ref_data['flux']

            #filename = os.path.join(os.path.dirname(__file__),
            #                        'data/linelist/{}.dat'.format(lamp))
            #linelist = Table.read(filename, format='ascii.fixed_width_two_line')
            linelist = linelist['wave_air', 'element', 'ion', 'source']

            window = 17
            deg = 4
            clipping = 3
            q_threshold = 5

            result = find_longslit_wavelength(
                    spec, ref_wave, ref_flux, (-50, 50),
                    linelist    = linelist,
                    window      = window,
                    deg         = deg,
                    q_threshold = q_threshold,
                    clipping    = clipping,
                    )

            allwave  = result['wavelength']
            linelist = result['linelist']
            stdwave  = result['std']
            fig_sol  = result['fig_solution']

            # save line-by-line fit figure
            fig_lbl_lst = result['fig_fitlbl']
            if len(fig_lbl_lst)==1:
                fig_lbl = fig_lbl_lst[0]
                title = 'Line-by-line fit of {} ({})'.format(fileid, lamp)
                fig_lbl.suptitle(title)
                figname = 'linefit_lbl_{}.png'.format(fileid)
                figfilename = os.path.join(self.reduction_path, figname)
                fig_lbl.savefig(figfilename)
                plt.close(fig_lbl)
            else:
                for ifig, fig_lbl in enumerate(fig_lbl_lst):
                    # add figure title
                    title = 'Line-by-line fit of {} ({}, {} of {})'.format(
                            fileid, lamp, ifig+1, len(fig_lbl_lst))
                    fig_lbl.suptitle(title)
                    figname = 'linefit_lbl_{}_{:02d}.png'.format(
                            fileid, ifig+1)
                    figfilename = os.path.join(self.reduction_path, figname)
                    fig_lbl.savefig(figfilename)
                    plt.close(fig_lbl)

            # save wavelength solution figure
            # add title
            title = 'Wavelength Solution for {} ({})'.format(fileid, lamp)
            fig_sol.suptitle(title)
            figname = 'wlcalib_{}.png'.format(fileid)
            figfilename = os.path.join(self.reduction_path, figname)
            fig_sol.savefig(figfilename)
            plt.close(fig_sol)

            # prepare the wavelength calibrated arc lamp spectra
            newtable = Table([allwave, spec], names=['wavelength', 'flux'])

            ntotal = len(linelist)
            nused = sum(list(linelist['use']))

            # prepare the FITS header
            head = fits.Header()
            head['OBSERVAT'] = 'YNAO'
            head['TELESCOP'] = 'Lijiang 2.4m'
            head['INSTRUME'] = 'YFOSC'
            head['FILEID']   = fileid
            head['OBJECT']   = lamp
            head['EXPTIME']  = logitem['exptime']
            head['DATEOBS']  = logitem['dateobs']
            head['MODE']     = logitem['mode']
            head['CONFIG']   = logitem['config']
            head['SLIT']     = logitem['slit']
            head['BINNING']  = logitem['binning']
            head['GAIN']     = logitem['gain']
            head['RDSPEED']  = logitem['rdspeed']
            head['FITFUNC']  = 'GENGAUSSIAN'
            head['WINDOW']   = window
            head['FITDEG']   = deg
            head['CLIPPING'] = clipping
            head['QTHOLD']   = q_threshold
            head['WAVERMS']  = stdwave
            head['NTOTAL']   = ntotal
            head['NUSED']    = nused
            hdulst = fits.HDUList([
                        fits.PrimaryHDU(header=head),
                        fits.BinTableHDU(data=newtable),
                        fits.BinTableHDU(data=linelist),
                ])
            hdulst.writeto('wlcalib_{}.fits'.format(fileid),
                           overwrite=True)

            self.wave[logitem['fileid']] = allwave
            self.ident[logitem['fileid']] = linelist



    def ident_longslit_wavelength1(self):

        self.wave = {}
        self.ident = {}
        index_file = os.path.join(os.path.dirname(__file__),
                              'data/calib/wlcalib_yfosc.dat')

        def errfunc(p, x, y, fitfunc):
            return y - fitfunc(p, x)
        def fitline(p, x):
            nline = int((len(p)-1)/4)
            y = np.ones_like(x, dtype=np.float64) + p[0]
            for i in range(nline):
                A, alpha, beta, center = p[i*4+1:i*4+5]
                y = y + gengaussian(A, alpha, beta, center, x)
            return y

        deg = 4

        for logitem in self.logtable:
            if logitem['mode']!='longslit' or logitem['datatype']!='LAMP':
                continue
            if logitem['fileid'] not in self.arclamp:
                continue

            spec = self.arclamp[logitem['fileid']]
            n = spec.size

            ref_data = select_calib_from_database(index_file,
                                lamp = logitem['object'],
                                mode = logitem['mode'],
                                config = logitem['config'],
                                dateobs = logitem['dateobs'],
                                )
            ref_wave = ref_data['wavelength']
            ref_flux = ref_data['flux']
            ref_pixel = np.arange(ref_wave.size)

            shift_lst = np.arange(-50, 50)
            ccf_lst = get_simple_ccf(spec, ref_flux, shift_lst)
            shift = shift_lst[ccf_lst.argmax()]

            #fig = plt.figure()
            #ax = fig.gca()
            ##ax.plot(spec)
            ##ax.plot(ref_spec['flux'])
            #ax.plot(shift_lst, ccf_lst)
            #plt.show()

            if ref_wave[0] > ref_wave[-1]:
                ref_wave = ref_wave[::-1]
                ref_flux = ref_flux[::-1]
                ref_pixel = ref_pixel[::-1]

            f_wave_to_pix = intp.InterpolatedUnivariateSpline(
                    ref_wave, ref_pixel, k=3)

            lamp = logitem['object']
            filename = os.path.dirname(__file__) + '/data/linelist/{}.dat'.format(lamp)
            linelist = Table.read(filename, format='ascii.fixed_width_two_line')
            linelist.add_column([-1]*len(linelist), index=-1, name='pixel')
            linelist.add_column([-1]*len(linelist), index=-1, name='i1')
            linelist.add_column([-1]*len(linelist), index=-1, name='i2')
            linelist.add_column([-1]*len(linelist), index=-1, name='fitid')

            n = spec.size
            pixel_lst = np.arange(n)

            for iline, line in enumerate(linelist):
                pix1 = f_wave_to_pix(line['wave_air']) + shift
                cint1 = int(round(pix1))
                i1, i2 = cint1 - 8, cint1 + 9
                i1 = max(i1, 0)
                i2 = min(i2, n-1)
                linelist[iline]['pixel'] = cint1
                linelist[iline]['i1'] = i1
                linelist[iline]['i2'] = i2

            m = (linelist['pixel']>0)*(linelist['pixel']<n-1)
            linelist = linelist[m]

            wave_lst, species_lst = [], []
            # fitting parameter results
            alpha_lst, beta_lst, center_lst = [], [], []
            A_lst, bkg_lst, std_lst, fwhm_lst = [], [], [], []
            # initialize line-by-line figure
            fig_lbl = plt.figure(figsize=(16, 9), dpi=150, tight_layout=True)
            nrow, ncol = 6, 7

            count_line = 0  # fitting counter
            for iline, line in enumerate(linelist):
                if line['fitid'] >=0:
                    continue

                i1 = line['i1']
                i2 = line['i2']
                # add background level
                ydata = spec[i1:i2]
                p0 = [ydata.min()]
                p0.append(spec[line['pixel']]-ydata.min())
                p0.append(3.6)
                p0.append(3.5)
                p0.append(line['pixel'])

                lower_bounds = [-np.inf, 0,      0.5, 0.1, i1]
                upper_bounds = [np.inf,  np.inf, 20,   20, i2]

                inextline = iline + 1
                idx_lst = [iline]
                while(True):
                    if inextline >= len(linelist):
                        break
                    nextline = linelist[inextline]
                    next_i1 = nextline['i1']
                    next_i2 = nextline['i2']
                    if next_i1 < i2 - 3:
                        i2 = next_i2
                        ydata = spec[i1:i2]
                        # update backgroud
                        p0[0] = ydata.min()
                        p0.append(spec[nextline['pixel']] - ydata.min())
                        p0.append(3.6)
                        p0.append(3.5)
                        p0.append(nextline['pixel'])
                        lower_bounds.append(0)
                        lower_bounds.append(0.5)
                        lower_bounds.append(0.1)
                        lower_bounds.append(next_i1)
                        upper_bounds.append(np.inf)
                        upper_bounds.append(20)
                        upper_bounds.append(20)
                        upper_bounds.append(next_i2)
                        idx_lst.append(inextline)
                    else:
                        break

                    inextline += 1

                xdata = np.arange(i1, i2)

                fitres = opt.least_squares(errfunc, p0,
                            bounds=(lower_bounds, upper_bounds),
                            args=(xdata, ydata, fitline),
                )

                param = fitres['x']

                fmt_str = '{:10.4f} {:4d} {:4d} {:6.3f} {:6.3f} {:8.3f} {:3d}'

                nline = int((len(param)-1)/4)
                for i in range(nline):
                    A, alpha, beta, center = param[i*4+1:i*4+5]
                    line = linelist[iline+i]
                    print(fmt_str.format(line['wave_air'], i1, i2,
                            alpha, beta, center, count_line))
                    wave_lst.append(line['wave_air'])
                    species_lst.append('{} {}'.format(line['element'], line['ion']))
                    # append fitting results
                    center_lst.append(center)
                    A_lst.append(A)
                    alpha_lst.append(alpha)
                    beta_lst.append(beta)
                    fwhm = 2*alpha*np.power(np.log(2), 1/beta)
                    fwhm_lst.append(fwhm)
                    bkg_lst.append(param[0])
                    std = np.sqrt(fitres['cost']*2/xdata.size)
                    std_lst.append(std)

                    linelist['i1'][idx_lst[i]] = i1
                    linelist['i2'][idx_lst[i]] = i2
                    linelist['fitid'][iline+i] = count_line

                count_line += 1

                # plot line-by-line
                if count_line > nrow*ncol:
                    continue
                # create small axes
                axs = fig_lbl.add_subplot(nrow, ncol, count_line)
                axs.plot(xdata, ydata*1e-3, 'o', c='C0', ms=3, alpha=0.6)
                # plot fitted curve
                newx = np.linspace(xdata[0], xdata[-1], 100)
                newy = fitline(param, newx)
                axs.plot(newx, newy*1e-3, ls='-', color='C1', lw=1, alpha=0.7)

                # draw text and plot line center as vertical lines
                text_lst = []
                for i in range(nline):
                    A, alpha, beta, center = param[i*4+1:i*4+5]
                    line = linelist[iline+i]
                    axs.axvline(x=center, color='k', ls='--', lw=0.7)
                    text = '{} {} {:9.4f}'.format(line['element'], line['ion'], line['wave_air'])
                    text_lst.append(text)
                _text = '\n'.join(text_lst)
                axs.set_xlim(newx[0], newx[-1])
                _x1, _x2 = axs.get_xlim()
                _y1, _y2 = axs.get_ylim()
                _y2 = _y2 + 0.2*(_y2 - _y1)
                axs.text(0.95*_x1+0.05*_x2, 0.05*_y1+0.95*_y2, _text,
                         ha='left', va='top', fontsize=8)
                axs.set_ylim(_y1, _y2)
                axs.xaxis.set_major_locator(tck.MultipleLocator(5))
                axs.xaxis.set_minor_locator(tck.MultipleLocator(1))
                for tick in axs.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
                for tick in axs.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
            
            figfilename = 'linefit_lbl.png'
            fig_lbl.savefig(figfilename)
            plt.close(fig_lbl)

            # fit wavelength solution
            wave_lst = np.array(wave_lst)
            center_lst = np.array(center_lst)
            idx = center_lst.argsort()
            center_lst = center_lst[idx]
            wave_lst = wave_lst[idx]

            mask = np.ones_like(wave_lst, dtype=bool)
            for i in range(3):
                coeff_wave = np.polyfit(center_lst[mask], wave_lst[mask],
                                        deg=deg)
                allwave = np.polyval(coeff_wave, pixel_lst)
                fitwave = np.polyval(coeff_wave, center_lst)
                reswave = wave_lst - fitwave
                stdwave = reswave[mask].std()
                newmask = np.abs(reswave) < 3*stdwave
                if newmask.sum()==mask.sum():
                    break
                mask = newmask

            # prepare the wavelength calibrated arc lamp spectra
            newtable = Table([allwave, spec], names=['wavelength', 'flux'])

            # prepare the identified line list
            linelist = linelist['wave_air', 'element', 'ion', 'source',
                                'i1', 'i2', 'fitid']
            linelist = linelist[linelist['fitid']>=0]
            linelist.add_column(center_lst,     index=-1, name='pixel')
            linelist.add_column(reswave,        index=-1, name='residual')
            linelist.add_column(np.int16(mask), index=-1, name='use')
            linelist.add_column(A_lst,          index=-1, name='A')
            linelist.add_column(alpha_lst,      index=-1, name='alpha')
            linelist.add_column(beta_lst,       index=-1, name='beta')
            linelist.add_column(bkg_lst,        index=-1, name='bkg')
            linelist.add_column(fwhm_lst,       index=-1, name='fwhm')
            linelist.add_column(std_lst,        index=-1, name='std')

            # prepare the FITS header
            head = fits.Header()
            head['OBSERVAT'] = 'YNAO'
            head['TELESCOP'] = 'Lijiang 2.4m'
            head['INSTRUME'] = 'YFOSC'
            head['FILEID']   = logitem['fileid']
            head['OBJECT']   = logitem['object']
            head['EXPTIME']  = logitem['exptime']
            head['DATEOBS']  = logitem['dateobs']
            head['MODE']     = logitem['mode']
            head['CONFIG']   = logitem['config']
            head['SLIT']     = logitem['slit']
            head['BINNING']  = logitem['binning']
            head['GAIN']     = logitem['gain']
            head['RDSPEED']  = logitem['rdspeed']
            head['FITFUNC']  = 'GENGAUSSIAN'
            head['FITDEG']   = deg
            head['WAVERMS']  = stdwave
            head['NTOTAL']   = len(mask)
            head['NUSED']    = mask.sum()
            hdulst = fits.HDUList([
                        fits.PrimaryHDU(header=head),
                        fits.BinTableHDU(data=newtable),
                        fits.BinTableHDU(data=linelist),
                ])
            hdulst.writeto('wlcalib_{}.fits'.format(logitem['fileid']),
                           overwrite=True)

            # plot wavelength solution
            fig_sol = plt.figure(figsize=(12, 6), dpi=300)
            axt1 = fig_sol.add_axes([0.07, 0.40, 0.44, 0.52])
            axt2 = fig_sol.add_axes([0.07, 0.10, 0.44, 0.26])
            axt4 = fig_sol.add_axes([0.58, 0.54, 0.37, 0.38])
            axt5 = fig_sol.add_axes([0.58, 0.10, 0.37, 0.38])

            axt1.plot(pixel_lst, allwave, c='k', lw=0.6, zorder=10)
            species = np.unique(species_lst)
            for ispecies, _species in enumerate(sorted(species)):
                m = np.array([v==_species for v in species_lst])
                c = 'C{}'.format(ispecies)
                axt1.plot(center_lst[mask*m], wave_lst[mask*m], 'o',
                          ms=4, c=c, label=_species, alpha=0.7)
                axt1.plot(center_lst[(~mask)*m], wave_lst[(~mask)*m], 'o',
                          ms=5, c='none', mec=c, alpha=0.7)
                axt2.plot(center_lst[mask*m], reswave[mask*m], 'o',
                          ms=4, c=c, alpha=0.7)
                axt2.plot(center_lst[(~mask)*m], reswave[(~mask)*m], 'o',
                          ms=5, c='none', mec=c, alpha=0.7)
            axt1.legend(loc='upper center', ncols=len(species))

            axt2.axhline(y=0, color='k', ls='--', lw=0.6, zorder=-1)
            axt2.axhline(y=stdwave, color='k', ls='--', lw=0.6, zorder=-1)
            axt2.axhline(y=-stdwave, color='k', ls='--', lw=0.6, zorder=-1)
            for ax in fig_sol.get_axes():
                ax.grid(True, color='k', alpha=0.4, ls='--', lw=0.5)
                ax.set_axisbelow(True)
                ax.set_xlim(0, n-1)
            y1, y2 = axt2.get_ylim()
            axt2.text(0.03*n, 0.1*y1 + 0.9*y2,
                      u'RMS = {:5.3f} \xc5'.format(stdwave),
                      ha='left', va='top')
            z = max(abs(y1), abs(y2))
            axt2.set_ylim(-z, z)
            axt2.set_xlabel('Pixel')
            axt1.set_ylabel(u'\u03bb (\xc5)')
            axt2.set_ylabel(u'\u0394\u03bb (\xc5)')
            axt1.set_xticklabels([])
            axt4.plot(pixel_lst[0:-1], -np.diff(allwave))
            axt5.plot(pixel_lst[0:-1], -np.diff(allwave) / (allwave[0:-1]) * 299792.458)
            axt4.set_ylabel(u'd\u03bb/dx (\xc5)')
            axt5.set_xlabel('Pixel')
            axt5.set_ylabel(u'dv/dx (km/s)')
            title = 'Wavelength Solution for {}'.format(logitem['fileid'])
            fig_sol.suptitle(title)
            figname = 'wavelength_solution2_{}.png'.format(logitem['fileid'])
            figfilename = os.path.join(self.reduction_path, figname)
            fig_sol.savefig(figfilename)
            plt.close(fig_sol)

            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            ax1.plot(spec, lw=0.5)
            ax2.plot(allwave, spec, lw=0.5)
            for c, w in zip(center_lst, wave_lst):
                ax1.axvline(c, c='k', ls='--', lw=0.5)
                ax2.axvline(w, c='k', ls='--', lw=0.5)

            _y1, _y2 = ax2.get_ylim()
            for w in linelist['wave_air']:
                ax2.axvline(w, ymin=0.9, ymax=1,
                            c='k', ls='-', lw=0.5, alpha=0.2)

            self.wave[logitem['fileid']] = allwave
            self.ident[logitem['fileid']] = linelist
            plt.show()

    def ident_longslit_wavelength2(self):

        for logitem in self.logtable:
            if logitem['mode']!='longslit' or logitem['datatype']!='LAMP':
                continue
            if logitem['fileid'] not in self.arclamp:
                continue

            spec = self.arclamp[logitem['fileid']]
            n = spec.size
            
            def errfunc(p, x, y, fitfunc):
                return y - fitfunc(p, x)
            def fitline(p, x):
                nline = int((len(p)-1)/4)
                y = np.ones_like(x, dtype=np.float64) + p[0]
                for i in range(nline):
                    A, alpha, beta, center = p[i*4+1:i*4+5]
                    y = y + gengaussian(A, alpha, beta, center, x)
                return y
            
            c_lst, w_lst = [], []
            file1 = open('input.dat')
            for row in file1:
                row = row.strip()
                if len(row)==0 or row[0]=='#':
                    continue
                g = row.split()
                c_lst.append(float(g[0])-500)
                w_lst.append(float(g[1]))
            file1.close()

            wave_lst, center_lst = [], []
            
            fig_lbl = plt.figure(figsize=(15, 8), dpi=150, tight_layout=True)
            nrow, ncol = 4, 5

            count_line = 0
            for c, w in zip(c_lst, w_lst):
                cint = int(round(c))
                i1, i2 = cint - 8, cint + 9
                xdata = np.arange(i1, i2)
                ydata = spec[i1:i2]
            
                p0 = [ydata.max() - ydata.min(), 3.6, 3.5, c, ydata.min()]
                fitres = opt.least_squares(errfunc, p0,
                            bounds=([-np.inf, 0.5,  0.1, i1, -np.inf],
                                   [np.inf, 20.0, 20.0, i2, ydata.max()]),
                            args=(xdata, ydata, fitline),
                            )
            
                param = fitres['x']
                A, alpha, beta, center, bkg = param
                center_lst.append(center)
                wave_lst.append(w)
            
                # plot line-by-line
                axs = fig_lbl.add_subplot(nrow, ncol, count_line+1)
                #axs = fig_lbl.add_axes([0.07+ix*0.16, 0.05+iy*0.18, 0.12, 0.15])
                axs.plot(xdata, ydata, 'o', ms=3, alpha=0.6)
                newx = np.linspace(i1, i2, 100)
                newy = fitline(param, newx)
                axs.plot(newx, newy, ls='-', color='C1', lw=1, alpha=0.7)
                axs.axvline(x=center, color='k', ls='--', lw=0.7)
                axs.set_xlim(newx[0], newx[-1])
                _x1, _x2 = axs.get_xlim()
                _y1, _y2 = axs.get_ylim()
                _y2 = _y2 + 0.2*(_y2 - _y1)
                _text = '{:9.4f}'.format(w)
                axs.text(0.95*_x1+0.05*_x2, 0.15*_y1+0.85*_y2, _text,
                         fontsize=9)
                axs.set_ylim(_y1, _y2)
                axs.xaxis.set_major_locator(tck.MultipleLocator(5))
                axs.xaxis.set_minor_locator(tck.MultipleLocator(1))
                for tick in axs.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(9)
            
                count_line += 1
           
            figfilename = 'linefit_lbl.png'
            fig_lbl.savefig(figfilename)
            plt.close(fig_lbl)

            # fit wavelength solution
            wave_lst = np.array(wave_lst)
            center_lst = np.array(center_lst)

            idx = center_lst.argsort()
            center_lst = center_lst[idx]
            wave_lst = wave_lst[idx]

            pixel_lst = np.arange(n)
            coeff_wave = np.polyfit(center_lst, wave_lst, deg=4)
            allwave = np.polyval(coeff_wave, pixel_lst)
            fitwave = np.polyval(coeff_wave, center_lst)
            reswave = wave_lst - fitwave
            stdwave = reswave.std()

            filename = os.path.dirname(__file__) + '/data/linelist/HeNe.dat'
            linelist = Table.read(filename, format='ascii.fixed_width_two_line')
            linelist.add_column([-1]*len(linelist), index=-1, name='pixel')
            linelist.add_column([-1]*len(linelist), index=-1, name='i1')
            linelist.add_column([-1]*len(linelist), index=-1, name='i2')
            linelist.add_column([-1]*len(linelist), index=-1, name='fitid')

            idx = wave_lst.argsort()
            center_lst = center_lst[idx]
            wave_lst = wave_lst[idx]
            coeff_wave_to_pix = np.polyfit(wave_lst, center_lst, deg=4)

            for iline, line in enumerate(linelist):
                pix1 = np.polyval(coeff_wave_to_pix, line['wave_air'])
                cint1 = int(round(pix1))
                i1, i2 = cint1 - 8, cint1 + 9
                i1 = max(i1, 0)
                i2 = min(i2, n-1)
                linelist[iline]['pixel'] = cint1
                linelist[iline]['i1'] = i1
                linelist[iline]['i2'] = i2

            wave_lst, center_lst = [], []
            species_lst = []
            fwhm_lst = []
            fig_lbl2 = plt.figure(figsize=(16, 9), dpi=150, tight_layout=True)
            nrow, ncol = 6, 7
            count_line = 0

            for iline, line in enumerate(linelist):
                if line['fitid'] >=0:
                    continue

                i1 = line['i1']
                i2 = line['i2']
                # add background level

                ydata = spec[i1:i2]
                p0 = [ydata.min()]
                p0.append(spec[line['pixel']]-ydata.min())
                p0.append(3.6)
                p0.append(3.5)
                p0.append(line['pixel'])

                lower_bounds = [-np.inf, 0,      0.5, 0.1, i1]
                upper_bounds = [np.inf,  np.inf, 20,   20, i2]

                inextline = iline + 1
                idx_lst = [iline]
                while(True):
                    if inextline >= len(linelist):
                        break
                    nextline = linelist[inextline]
                    next_i1 = nextline['i1']
                    next_i2 = nextline['i2']
                    if next_i1 < i2 - 3:
                        i2 = next_i2
                        ydata = spec[i1:i2]
                        # update backgroud
                        p0[0] = ydata.min()
                        p0.append(spec[nextline['pixel']] - ydata.min())
                        p0.append(3.6)
                        p0.append(3.5)
                        p0.append(nextline['pixel'])
                        lower_bounds.append(0)
                        lower_bounds.append(0.5)
                        lower_bounds.append(0.1)
                        lower_bounds.append(next_i1)
                        upper_bounds.append(np.inf)
                        upper_bounds.append(20)
                        upper_bounds.append(20)
                        upper_bounds.append(next_i2)
                        idx_lst.append(inextline)
                    else:
                        break

                    inextline += 1

                xdata = np.arange(i1, i2)

                fitres = opt.least_squares(errfunc, p0,
                            bounds=(lower_bounds, upper_bounds),
                            args=(xdata, ydata, fitline),
                            )

                param = fitres['x']

                fmt_str = '{:10.4f} {:4d} {:4d} {:6.3f} {:6.3f} {:8.3f} {:3d}'

                nline = int((len(param)-1)/4)
                for i in range(nline):
                    A, alpha, beta, center = param[i*4+1:i*4+5]
                    fwhm = 2*alpha*np.power(np.log(2), 1/beta)
                    line = linelist[iline+i]
                    print(fmt_str.format(line['wave_air'], i1, i2,
                            alpha, beta, center, count_line))
                    center_lst.append(center)
                    wave_lst.append(line['wave_air'])
                    fwhm_lst.append(fwhm)
                    species_lst.append('{} {}'.format(line['element'], line['ion']))
                    linelist['i1'][idx_lst[i]] = i1
                    linelist['i2'][idx_lst[i]] = i2
                    linelist['fitid'][iline+i] = count_line

                count_line += 1

                # plot line-by-line
                if count_line > nrow*ncol:
                    continue
                axs = fig_lbl2.add_subplot(nrow, ncol, count_line)
                #axs = fig_lbl.add_axes([0.07+ix*0.16, 0.05+iy*0.18, 0.12, 0.15])
                axs.plot(xdata, ydata*1e-3, 'o', c='k', ms=3, alpha=0.6)
                newx = np.linspace(xdata[0], xdata[-1], 100)
                newy = fitline(param, newx)
                axs.plot(newx, newy*1e-3, ls='-', color='k', lw=1, alpha=0.7)

                text_lst = []
                for i in range(nline):
                    A, alpha, beta, center = param[i*4+1:i*4+5]
                    line = linelist[iline+i]
                    axs.axvline(x=center, color='k', ls='--', lw=0.7)
                    text = '{} {} {:9.4f}'.format(line['element'], line['ion'], line['wave_air'])
                    text_lst.append(text)
                _text = '\n'.join(text_lst)
                axs.set_xlim(newx[0], newx[-1])
                _x1, _x2 = axs.get_xlim()
                _y1, _y2 = axs.get_ylim()
                _y2 = _y2 + 0.2*(_y2 - _y1)
                axs.text(0.95*_x1+0.05*_x2, 0.05*_y1+0.95*_y2, _text,
                         ha='left', va='top', fontsize=8)
                axs.set_ylim(_y1, _y2)
                axs.xaxis.set_major_locator(tck.MultipleLocator(5))
                axs.xaxis.set_minor_locator(tck.MultipleLocator(1))
                for tick in axs.xaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
                for tick in axs.yaxis.get_major_ticks():
                    tick.label1.set_fontsize(7)
            
            figfilename = 'linefit_lbl2.png'
            fig_lbl2.savefig(figfilename)
            plt.close(fig_lbl2)

            # fit wavelength solution
            wave_lst = np.array(wave_lst)
            center_lst = np.array(center_lst)
            idx = center_lst.argsort()
            center_lst = center_lst[idx]
            wave_lst = wave_lst[idx]

            mask = np.ones_like(wave_lst, dtype=bool)
            for i in range(3):
                coeff_wave = np.polyfit(center_lst[mask], wave_lst[mask], deg=4)
                allwave = np.polyval(coeff_wave, pixel_lst)
                fitwave = np.polyval(coeff_wave, center_lst)
                reswave = wave_lst - fitwave
                stdwave = reswave[mask].std()
                newmask = np.abs(reswave) < 3*stdwave
                if newmask.sum()==mask.sum():
                    break
                mask = newmask

            linelist = linelist['wave_air', 'element', 'ion', 'source',
                                'i1', 'i2', 'fitid']
            linelist.add_column(center_lst,     index=-1, name='pixel')
            linelist.add_column(np.int16(mask), index=-1, name='mask')
            linelist.add_column(fwhm_lst,       index=-1, name='fwhm')

            newtable = Table([allwave, spec], names=['wavelength', 'flux'])

            head = fits.Header()
            head['OBSERVAT'] = 'YNAO'
            head['TELESCOP'] = 'Lijiang 2.4m'
            head['INSTRUME'] = 'YFOSC'
            head['FILEID']   = logitem['fileid']
            head['OBJECT']   = logitem['object']
            head['EXPTIME']  = logitem['exptime']
            head['DATEOBS']  = logitem['dateobs']
            head['MODE']     = logitem['mode']
            head['CONFIG']   = logitem['config']
            head['SLIT']     = logitem['slit']
            head['BINNING']  = logitem['binning']
            head['GAIN']     = logitem['gain']
            head['RDSPEED']  = logitem['rdspeed']
            head['FITFUNC']  = 'GENGAUSSIAN'
            head['WAVERMS']  = stdwave
            head['NTOTAL']   = len(mask)
            head['NUSED']    = mask.sum()
            hdulst = fits.HDUList([
                        fits.PrimaryHDU(header=head),
                        fits.BinTableHDU(data=newtable),
                        fits.BinTableHDU(data=linelist),
                ])
            hdulst.writeto('wlcalib_{}.fits'.format(logitem['fileid']),
                           overwrite=True)

            # plot wavelength solution
            fig_sol = plt.figure(figsize=(12, 6), dpi=300)
            axt1 = fig_sol.add_axes([0.07, 0.40, 0.44, 0.52])
            axt2 = fig_sol.add_axes([0.07, 0.10, 0.44, 0.26])
            axt4 = fig_sol.add_axes([0.58, 0.54, 0.37, 0.38])
            axt5 = fig_sol.add_axes([0.58, 0.10, 0.37, 0.38])

            axt1.plot(pixel_lst, allwave, c='k', lw=0.6, zorder=10)
            species = np.unique(species_lst)
            for ispecies, _species in enumerate(sorted(species)):
                m = np.array([v==_species for v in species_lst])
                c = 'C{}'.format(ispecies)
                axt1.plot(center_lst[mask*m], wave_lst[mask*m], 'o',
                          ms=4, c=c, label=_species, alpha=0.7)
                axt1.plot(center_lst[(~mask)*m], wave_lst[(~mask)*m], 'o',
                          ms=5, c='none', mec=c, alpha=0.7)
                axt2.plot(center_lst[mask*m], reswave[mask*m], 'o',
                          ms=4, c=c, alpha=0.7)
                axt2.plot(center_lst[(~mask)*m], reswave[(~mask)*m], 'o',
                          ms=5, c='none', mec=c, alpha=0.7)
            axt1.legend(loc='upper center', ncols=len(species))

            axt2.axhline(y=0, color='k', ls='--', lw=0.6, zorder=-1)
            axt2.axhline(y=stdwave, color='k', ls='--', lw=0.6, zorder=-1)
            axt2.axhline(y=-stdwave, color='k', ls='--', lw=0.6, zorder=-1)
            for ax in fig_sol.get_axes():
                ax.grid(True, color='k', alpha=0.4, ls='--', lw=0.5)
                ax.set_axisbelow(True)
                ax.set_xlim(0, n-1)
            y1, y2 = axt2.get_ylim()
            axt2.text(0.03*n, 0.1*y1 + 0.9*y2,
                      u'RMS = {:5.3f} \xc5'.format(stdwave),
                      ha='left', va='top')
            z = max(abs(y1), abs(y2))
            axt2.set_ylim(-z, z)
            axt2.set_xlabel('Pixel')
            axt1.set_ylabel(u'\u03bb (\xc5)')
            axt2.set_ylabel(u'\u0394\u03bb (\xc5)')
            axt1.set_xticklabels([])
            axt4.plot(pixel_lst[0:-1], -np.diff(allwave))
            axt5.plot(pixel_lst[0:-1], -np.diff(allwave) / (allwave[0:-1]) * 299792.458)
            axt4.set_ylabel(u'd\u03bb/dx (\xc5)')
            axt5.set_xlabel('Pixel')
            axt5.set_ylabel(u'dv/dx (km/s)')
            title = 'Wavelength Solution for {}'.format(logitem['fileid'])
            fig_sol.suptitle(title)
            figname = 'wavelength_solution2_{}.png'.format(logitem['fileid'])
            figfilename = os.path.join(self.reduction_path, figname)
            fig_sol.savefig(figfilename)
            plt.close(fig_sol)

            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            ax1.plot(spec, lw=0.5)
            ax2.plot(allwave, spec, lw=0.5)
            for c, w in zip(center_lst, wave_lst):
                ax1.axvline(c, c='k', ls='--', lw=0.5)
                ax2.axvline(w, c='k', ls='--', lw=0.5)

            _y1, _y2 = ax2.get_ylim()
            for w in linelist['wave_air']:
                ax2.axvline(w, ymin=0.9, ymax=1,
                            c='k', ls='-', lw=0.5, alpha=0.2)

            plt.show()

    def find_longslit_distortion(self):

        for logitem in self.logtable:
            if logitem['mode']!='longslit' or logitem['datatype']!='LAMP':
                continue

            ccdconf = self.get_ccdconf(logitem)
            conf = self.get_conf(logitem)

            filename = self.fileid_to_filename(logitem['fileid'])
            data = fits.getdata(filename)
            data = trim_rawdata(data)
            data = correct_overscan(data)
            data = data - self.bias[ccdconf]
            data = data / self.sensmap[conf]
            data = trim_longslit(data)

            allwave = self.wave[logitem['fileid']]
            linelist = self.ident[logitem['fileid']]

            distortion = find_distortion(data, hwidth=5, disp_axis='y',
                                         wave=allwave, linelist=linelist,
                                         deg=4, xorder=4, yorder=4)

            newdata = correct_distortion(data, distortion)
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.imshow(data)
            ax2.imshow(newdata)
            plt.show()

    def get_conf(self, logitem):
        return (logitem['mode'], logitem['config'],
                logitem['binning'], logitem['gain'], logitem['rdspeed'])

    def get_conf_string(self, conf):
        mode, config, binning, gain, rdspeed = conf
        return '{}_{}_{}_gain{:3.1f}_{}'.format(mode, config,
                                                binning, gain, rdspeed)

    def get_ccdconf(self, logitem):
        return (logitem['binning'], logitem['gain'], logitem['rdspeed'])

    def get_ccdconf_string(self, ccdconf):
        binning, gain, rdspeed = ccdconf
        return '{}_gain{:3.1f}_{}'.format(binning, gain, rdspeed)

    def fileid_to_filename(self, fileid):
        prefix = 'ljg2m401-yf01-'
        for fname in os.listdir(self.rawdata_path):
            if fname.startswith(prefix + str(fileid)):
                return os.path.join(self.rawdata_path, fname)
        raise ValueError

    def plot_bias(self, show=True, figname=None):
        for ccdconf, ccdimg in self.bias.items():
            fig = plt.figure(dpi=200, figsize=(8,8))
            ax1 = fig.gca()
            ax2 = ax1.twinx()
            vmin = np.percentile(ccdimg, 1)
            vmax = np.percentile(ccdimg, 99)
            ax1.imshow(ccdimg, vmin=vmin, vmax=vmax, cm='gray')
        if show:
            plt.show(fig)
