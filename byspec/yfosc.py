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
from .common import (FOSCReducer, find_distortion,
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

            self.wave[fileid] = allwave
            self.ident[fileid] = linelist


    def find_longslit_distortion(self):

        self.distortion = {}

        for logitem in self.logtable:
            if logitem['mode']!='longslit' or logitem['datatype']!='LAMP':
                continue

            ccdconf = self.get_ccdconf(logitem)
            conf = self.get_conf(logitem)

            fileid = logitem['fileid']
            filename = self.fileid_to_filename(fileid)
            data = fits.getdata(filename)
            data = trim_rawdata(data)
            data = correct_overscan(data)
            data = data - self.bias[ccdconf]
            data = data / self.sensmap[conf]
            data = trim_longslit(data)

            allwave = self.wave[fileid]
            linelist = self.ident[fileid]

            distortion, fig1, fig3d = find_distortion(data,
                                        hwidth=5, disp_axis='y',
                                        linelist=linelist,
                                        deg=4, xorder=4, yorder=4)
            fig1.suptitle('Distortion of {}'.format(fileid))
            fig1.savefig('distortion_{}.pdf'.format(fileid))
            fig1.savefig('distortion_{}.png'.format(fileid))
            plt.close(fig1)
            #fig3d.suptitle('Distortion of {}'.format(fileid))
            fig3d.suptitle('YFOSC')
            fig3d.savefig('distortion3d_{}.pdf'.format(fileid))
            fig3d.savefig('distortion3d_{}.png'.format(fileid))
            plt.close(fig3d)

            # plot distortion map
            fig = distortion.plot(times=10)
            fig.suptitle('YFOSC', fontsize=13)
            fig.savefig('distortion_yfosc.pdf')
            fig.savefig('distortion_yfosc.png')
            plt.close(fig)
            newdata = distortion.correct_image(data)

            #fig = plt.figure()
            #ax1 = fig.add_subplot(121)
            #ax2 = fig.add_subplot(122)
            #ax1.imshow(data)
            #ax2.imshow(newdata)
            #plt.show()
            self.distortion[conf] = distortion


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
