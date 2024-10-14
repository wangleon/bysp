import os
import re
import datetime

import numpy as np
import scipy.signal as sg
import scipy.interpolate as intp
import scipy.optimize as opt
from astropy.table import Table
import astropy.io.fits as fits

import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from .imageproc import combine_images
from .onedarray import (iterative_savgol_filter,
                        gaussian, gengaussian,
                        find_shift_ccf)
from .common import FOSCReducer


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
        flux1d_sm, _, mask, std = iterative_savgol_filter(flux1d,
                                    winlen=51, order=3,
                                    upper_clip=6, lower_clip=6, maxiter=10)
        sensmap[:, x] = flux1d/flux1d_sm

    return sensmap

def find_distortion(data):
    ny, nx = data.shape
    allx = np.arange(nx)
    ally = np.arange(ny)
    hwidth = 5
    ref_spec = data[:, nx//2-hwidth:nx//2+hwidth].sum(axis=1)
    
    fig = plt.figure(dpi=200, figsize=(10,5))
    ax1 = fig.add_axes([0.05, 0.53, 0.4, 0.42])
    ax2 = fig.add_axes([0.05, 0.10, 0.4, 0.42])
    ax3 = fig.add_axes([0.5, 0.1, 0.45, 0.85])

    xcoord_lst = np.arange(200, nx-200, 200)
    yshift_lst = []
    for i, x in enumerate(xcoord_lst):
        spec = data[:, x-hwidth:x+hwidth].sum(axis=1)
        shift = find_shift_ccf(ref_spec, spec)
        yshift_lst.append(shift)
        if i==0:
            ax1.plot(spec, color='w', lw=0)
            y1, y2 = ax1.get_ylim()
            offset = (y2-y1)/20
        ycoord = np.arange(ny)
        ax1.plot(ycoord, spec+offset*i, '-', lw=0.5)
        ax2.plot(ycoord-shift, spec+offset*i, '-', lw=0.5)

    ax1.set_xlim(0, ny-1)
    ax2.set_xlim(0, ny-1)
    coeff = np.polyfit(xcoord_lst, yshift_lst, deg=2)
    ax3.plot(xcoord_lst, yshift_lst, 'o')
    ax3.plot(allx, np.polyval(coeff, allx))
    ax3.set_xlim(0, nx-1)

    plt.show()


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

    def find_longslit_distortion(self):
        for logitem in self.logtable:
            if logitem['mode']=='longslit' and logitem['datatype']=='LAMP':
                ccdconf = self.get_ccdconf(logitem)
                conf = self.get_conf(logitem)
                filename = self.fileid_to_filename(logitem['fileid'])
                data = fits.getdata(filename)
                data = trim_rawdata(data)
                data = correct_overscan(data)
                data = data - self.bias[ccdconf]
                data = data / self.sensmap[conf]
                
                find_distortion(data)

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
            spec = data[:, nx//2-halfwidth: nx//2+halfwidth].mean(axis=1)

            self.arclamp[logitem['fileid']] = spec

    def ident_longslit_wavelength(self):

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
            fig_lbl2 = plt.figure(figsize=(16, 9), dpi=150, tight_layout=True)
            nrow, ncol = 6, 7
            count_line = 0

            for iline, line in enumerate(linelist):
                if line['fitid'] >=0:
                    continue

                i1 = line['i1']
                i2 = line['i2']
                # add background level
                p0 = [ydata.min()]

                ydata = spec[i1:i2]
                p0.append(spec[line['pixel']]-ydata.min())
                p0.append(3.6)
                p0.append(3.5)
                p0.append(line['pixel'])

                lower_bounds = [-np.inf, 0,      0.5, 0.1, i1]
                upper_bounds = [np.inf,  np.inf, 20,   20, i2]

                inextline = iline + 1
                while(True):
                    if inextline >= len(linelist):
                        break
                    nextline = linelist[inextline]
                    next_i1 = nextline['i1']
                    next_i2 = nextline['i2']
                    if next_i1 < i2-3:
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
                    center_lst.append(center)
                    wave_lst.append(line['wave_air'])
                    species_lst.append('{} {}'.format(line['element'], line['ion']))
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
