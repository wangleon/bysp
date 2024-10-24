import os
import re
import datetime
import dateutil.parser

import numpy as np
import scipy.interpolate as intp
import scipy.optimize as opt
import scipy.signal as sg
from astropy.table import Table, Row
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from .utils import get_file
from .imageproc import combine_images
from .onedarray import (iterative_savgol_filter, get_simple_ccf,
                        gaussian, gengaussian,
                        consecutive, find_shift_ccf, get_clip_mean,
                        get_local_minima)
from .visual import plot_image_with_hist

from .common import FOSCReducer, find_longslit_wavelength

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

        if datatype in ['BIAS', 'SPECLFLAT', 'SPECLLAMP',
                                'SPECSFLAT', 'SPECSLAMP',]:
            ra, dec = '', ''

        obstable.add_row((frameid, fileid, datatype, objname, exptime, dateobs,
                        ra, dec, mode, config, slit, filtername,
                        binning, float(gain), float(rdnoise), q99, observer))

        if display:
            logitem = obstable[-1]

            string = fmt_str.format(
                    '[{:d}]'.format(frameid), str(fileid),
                    '{:12s}'.format(datatype),
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

    filepath = os.path.join('calib/bfosc/', 'wlcalib_{}.fits'.format(fileid))
    filename = get_file(filepath, md5)

    # load spec, calib, and aperset from selected FITS file
    hdu_lst = fits.open(filename)
    head = hdu_lst[0].header
    spec = hdu_lst[1].data
    linelist = Table(hdu_lst[2].data)
    hdu_lst.close()

    return spec, linelist

def find_echelle_apertures(data, align_deg, scan_step):
    ny, nx = data.shape
    allx = np.arange(nx)
    ally = np.arange(ny)

    logdata = np.log10(np.maximum(data, 1))

    x0 = nx//2
    x_lst = {-1:[], 1:[]}
    x1 = x0
    direction = -1
    icol = 0

    csec_i1 = -ny//2
    csec_i2 = ny + ny//2
    csec_lst  = np.zeros(csec_i2 - csec_i1)
    csec_nlst = np.zeros(csec_i2 - csec_i1, dtype=np.int32)

    param_lst = {-1:[], 1:[]}
    nodes_lst = {}

    def forward(x, p):
        deg = len(p) - 1 # determine the polynomial degree
        res = p[0]
        for i in range(deg):
            res = res*x + p[i+1]
        return res
    def forward_der(x, p):
        deg = len(p)-1  # determine the polynomial degree
        p_der = [(deg-i)*p[i] for i in range(deg)]
        return forward(x, p_der)
    def backward(y, p):
        x = y
        for ite in range(20):
            dy    = forward(x, p) - y
            y_der = forward_der(x, p)
            dx = dy/y_der
            x = x - dx
            if (abs(dx) < 1e-7).all():
                break
        return x
    def fitfunc(p, interfunc, n):
        return interfunc(forward(np.arange(n), p[0:-1])) + p[-1]
    def resfunc(p, interfunc, flux0, mask=None):
        res_lst = flux0 - fitfunc(p, interfunc, flux0.size)
        if mask is None:
            mask = np.ones_like(flux0, dtype=bool)
        return res_lst[mask]
    def find_shift(flux0, flux1, deg):
        #p0 = [1.0, 0.0, 0.0]
        #p0 = [0.0, 1.0, 0.0, 0.0]
        #p0 = [0.0, 0.0, 1.0, 0.0, 0.0]

        p0 = [0.0 for i in range(deg+1)]
        p0[-3] = 1.0

        interfunc = intp.InterpolatedUnivariateSpline(
                    np.arange(flux1.size), flux1, k=3, ext=3)
        mask = np.ones_like(flux0, dtype=bool)
        clipping = 5.
        for i in range(10):
            p, _ = opt.leastsq(resfunc, p0, args=(interfunc, flux0, mask))
            res_lst = resfunc(p, interfunc, flux0)
            std  = res_lst.std()
            mask1 = res_lst <  clipping*std
            mask2 = res_lst > -clipping*std
            new_mask = mask1*mask2
            if new_mask.sum() == mask.sum():
                break
            mask = new_mask
        return p, mask

    while(True):
        nodes_lst[x1] = []

        flux1 = logdata[:,x1]
        linflux1 = np.median(data[:,x1-2:x1+3], axis=1)

        if icol == 0:
            flux1_center = flux1

            # the middle column
            i1 = 0 - csec_i1
            i2 = ny - csec_i1
            # stack the linear flux to the stacked cross-section
            csec_lst[i1:i2] += linflux1
            csec_nlst[i1:i2] += 1

        else:
            param, _ = find_shift(flux0, flux1, deg=align_deg)
            param_lst[direction].append(param[0:-1])
            ysta, yend = 0., ny-1.
            for param in param_lst[direction][::-1]:
                ysta = backward(ysta, param)
                yend = backward(yend, param)
            # interpolate the new crosssection
            ynew = np.linspace(ysta, yend, ny)
            interfunc = intp.InterpolatedUnivariateSpline(ynew, linflux1, k=3)
            #
            ysta_int = int(round(ysta))
            yend_int = int(round(yend))
            fnew = interfunc(np.arange(ysta_int, yend_int+1))
            i1 = ysta_int - csec_i1
            i2 = yend_int + 1 - csec_i1
            csec_lst[i1:i2] += fnew
            csec_nlst[i1:i2] += 1

        x1 += direction*scan_step
        if x1 <= 10:
            direction = +1
            x1 = x0 + direction*scan_step
            x_lst[direction].append(x1)
            flux0 = flux1_center
            icol += 1
            continue
        elif x1 >= nx - 10:
            # scan ends
            break
        else:
            x_lst[direction].append(x1)
            flux0 = flux1
            icol += 1
            continue
    #
    i_nonzero = np.nonzero(csec_nlst)[0]
    istart, iend = i_nonzero[0], i_nonzero[-1]
    csec_ylst = np.arange(csec_lst.size) + csec_i1

    x = csec_ylst[istart:iend]
    y = csec_lst[istart:iend]
    x = x[100:-30]
    y = y[100:-30]
    n = y.size

    #########################
    # cross-section stacking
    #fig = plt.figure()
    #ax1 = fig.add_subplot(211)
    #ax2 = fig.add_subplot(212)
    #for x0 in np.arange(nx//2, nx, 100):
    #    ax1.plot(data[:,x0], lw=0.5)
    #ax1.set_yscale('log')
    #ax2.plot(x, y, lw=0.5)
    #ax2.set_yscale('log')

    ############################

    winmask = np.zeros_like(y, dtype=bool)
    xnodes = [100, 1100]
    wnodes = [30, 240]
    snodes = [20, 220]
    c1 = np.polyfit(xnodes, wnodes, deg=len(xnodes)-1)
    c2 = np.polyfit(xnodes, snodes, deg=len(xnodes)-1) 
    get_winlen = lambda x: np.polyval(c1, x)
    get_gaplen = lambda x: np.polyval(c2, x)
    for i1 in np.arange(0, n):
        winlen = get_winlen(i1)
        gaplen = get_gaplen(i1)
        gaplen = max(gaplen, 5)
        percent = gaplen/winlen*100
        i2 = i1 + int(winlen)
        if i2 >= n-1:
            break
        v = np.percentile(y[i1:i2], percent)
        pick = y[i1:i2]>v
        if (~pick).sum()==0:
            pick[pick.argmin()] = False
        idx = np.nonzero(pick)[0]
        winmask[idx+i1] = True

    bkgmask = ~winmask
    maxiter = 10
    for ite in range(maxiter):
        c = np.polyfit(x[bkgmask], np.log(y[bkgmask]), deg=15)
        newy = np.polyval(c, x)
        resy = np.log(y) - newy
        std = resy[bkgmask].std()
        newbkgmask = resy < 2*std
        if newbkgmask.sum() == bkgmask.sum():
            break
        bkgmask = newbkgmask

    aper_mask = y > np.exp(newy + 3*std)
    aper_idx = np.nonzero(aper_mask)[0]

    gap_mask = ~aper_mask
    gap_idx = np.nonzero(gap_mask)[0]

    max_order_width = 120
    min_order_width = 3

    order_index_lst = []
    for group in np.split(aper_idx, np.where(np.diff(aper_idx)>3)[0]+1):
        i1 = group[0]
        i2 = group[-1]
        if i2-i1 > max_order_width or i2-i1<min_order_width:
            continue
        chunk = y[i1:i2]
        m = chunk > (chunk.max()*0.3 + chunk.min()*0.7)
        i11 = np.nonzero(m)[0][0] + i1
        i22 = np.nonzero(m)[0][-1] + i1
        order_index_lst.append((i11, i22))

    norder = len(order_index_lst)
    order_lst = np.arange(norder)
    order_cen_lst = np.array([(i1+i2)/2 for i1, i2 in order_index_lst])
    goodmask = np.zeros(norder, dtype=bool)
    goodmask[0:10] = True

    ### find good orders
    #fig3 = plt.figure()
    #ax31 = fig3.add_subplot(211)
    #ax32 = fig3.add_subplot(212)
    #ax31.plot(order_lst[goodmask], order_cen_lst[goodmask], 'o', c='C0')
    #ax31.plot(order_lst[~goodmask], order_cen_lst[~goodmask], 'o', c='none', mec='C0')
    for i in range((~goodmask).sum()):
        fintp = intp.InterpolatedUnivariateSpline(
                np.arange(goodmask.sum()), order_cen_lst[goodmask], k=3)
        newcen = fintp(goodmask.sum())
        #ax31.axhline(newcen, color='k', ls='--')
        min_order = None
        min_dist = 9999
        for iorder, cen in enumerate(order_cen_lst):
            if abs(cen - newcen) < min_dist:
                min_dist = abs(cen - newcen)
                min_order = iorder
        goodmask[min_order] = True
        if newcen > order_cen_lst[-1]:
            break
    #ax31.plot(order_lst[goodmask], order_cen_lst[goodmask], 'o', c='C0', ms=2)


    fig2 = plt.figure()
    ax2 = fig2.gca()
    ax2.plot(x, y, lw=0.5)
    #ax2.plot(x[aper_idx], y[aper_idx], 'o', ms=1)
    ax2.plot(x, np.exp(newy), '-')
    _y1, _y2 = ax2.get_ylim()
    for iorder, (i1, i2) in enumerate(order_index_lst):
        if goodmask[iorder]:
            color = 'C0'
        else:
            color = 'C1'
        ax2.fill_betweenx([_y1, _y2], x[i1], x[i2], color=color, alpha=0.1, lw=0)
    ax2.plot(x, np.exp(newy+3*std), '--')
    ax2.set_yscale('log')
    ax2.set_ylim(_y1, _y2)



    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(np.log10(data))

    coeff_lst = []
    for iorder, (i1, i2) in enumerate(order_index_lst):
        cen = (x[i1] + x[i2])/2
        xnode_lst = [x0]
        ynode_lst = [cen]
        for direction in [-1, 1]:
            cen1 = cen
            for icol, param in enumerate(param_lst[direction]):
                cen1 = forward(cen1, param)
                xcol = x_lst[direction][icol]
                xnode_lst.append(xcol)
                ynode_lst.append(cen1)
        xnode_lst = np.array(xnode_lst)
        ynode_lst = np.array(ynode_lst)
        # resort
        args = xnode_lst.argsort()
        xnode_lst = xnode_lst[args]
        ynode_lst = ynode_lst[args]
        # fit polynomial with 3rd degree
        c = np.polyfit(xnode_lst, ynode_lst, deg=3)
        coeff_lst.append(c)

        if goodmask[iorder]:
            color = 'r'
            ls = '-'
        else:
            color = 'orange'
            ls = '--'
        # plot node
        #ax.plot(xnode_lst, ynode_lst, ls=ls, c=color, lw=0.5)
        # plot positions
        ax.plot(allx, np.polyval(c, allx), ls=ls, c=color, lw=0.5)


        #fig0 = plt.figure()
        #ax01 = fig0.add_subplot(211)
        #ax02 = fig0.add_subplot(212)
        #ax01.plot(xnode_lst, ynode_lst, 'o', ms=3)
        #ax01.plot(allx, np.polyval(c, allx), '-')
        #ax02.plot(xnode_lst, ynode_lst - np.polyval(c, xnode_lst), 'o', ms=3)
        #fig0.savefig('order_{}.png'.format(iorder))
        #plt.close(fig0)
        
    ax.set_xlim(0, nx-1)
    ax.set_ylim(0, ny-1)
    plt.show()

    return coeff_lst, goodmask


def get_longslit_sensmap(data):
    ny, nx = data.shape
    allx = np.arange(nx)
    ally = np.arange(ny)
    sensmap = np.ones_like(data, dtype=np.float64)

    for y in np.arange(ny):
        #fluxt1d = self.flat_data[y, 20:int(nx) - 20]
        flux1d = data[y, :]
        flux1d_sm, _, mask, std = iterative_savgol_filter(
                                    flux1d,
                                    winlen=51, order=3, mode='interp',
                                    upper_clip=6, lower_clip=6, maxiter=10)
        sensmap[y, :] = flux1d/flux1d_sm

    return sensmap


class BFOSC(FOSCReducer):
    def __init__(self, **kwargs):
        super(BFOSC, self).__init__(**kwargs)

    def make_obslog(self, filename=None):
        """Scan the raw data path and generate an observing log.
        """
        logtable = make_obslog(self.rawdata_path, display=True)

        # find obsdate
        self.obsdate = logtable[0]['dateobs'][0:10]

        if filename is None:
            filename = 'BFOSC.{}.txt'.format(self.obsdate)
        filename = os.path.join(self.reduction_path, filename)

        logtable.write(filename, format='ascii.fixed_width_two_line',
                        overwrite=True)
        self.logtable = logtable

    def fileid_to_filename(self, fileid):
        for fname in os.listdir(self.rawdata_path):
            if fname.startswith(str(fileid)):
                return os.path.join(self.rawdata_path, fname)
        raise ValueError

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
                print('Combine bias')
                selectfunc = lambda item: item['datatype']=='BIAS' and \
                                self.get_ccdconf(item)==ccdconf

                item_lst = list(filter(selectfunc, self.logtable))
                if len(item_lst)==0:
                    continue
                
                data_lst = []
                for logitem in item_lst:
                    filename = self.fileid_to_filename(logitem['fileid'])
                    data = fits.getdata(filename)
                    data_lst.append(data)
                data_lst = np.array(data_lst)
                
                bias_data = combine_images(data_lst, mode='mean',
                                upper_clip=5, maxiter=10, maskmode='max')

                # get the mean cross-section of coadded bias image
                section = bias_data.mean(axis=0)
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
        """Combine flat images
        """

        self.get_all_conf()
        self.flat = {}

        for conf in self.conf_lst:
            conf_string = self.get_conf_string(conf)
            flat_file = 'flat_{}.fits'.format(conf_string)
            flat_filename = os.path.join(self.reduction_path, flat_file)
            if os.path.exists(flat_filename):
                flat_data = fits.getdata(flat_filename)
            else:
                selectfunc = lambda item: item['object']=='FLAT' and \
                                self.get_conf(item)==conf

                item_lst = list(filter(selectfunc, self.logtable))

                if len(item_lst)==0:
                    continue

                print('Combine Flat for {}'.format(conf_string))

                data_lst = []
                for logitem in item_lst:
                    filename = self.fileid_to_filename(logitem['fileid'])
                    data = fits.getdata(filename)
                    ccdconf = self.get_ccdconf(logitem)
                    data = data - self.bias[ccdconf]
                    data_lst.append(data)
                data_lst = np.array(data_lst)

                flat_data = combine_images(data_lst, mode='mean',
                                    upper_clip=5, maxiter=10, maskmode='max')
                fits.writeto(flat_filename, flat_data, overwrite=True)
            self.flat[conf] = flat_data

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

    def get_conf(self, logitem):
        return (logitem['mode'], logitem['config'],
                logitem['binning'], logitem['gain'], logitem['rdnoise'])
    
    def get_conf_string(self, conf):
        mode, config, binning, gain, rdspeed = conf
        return '{}_{}_{}_gian{:3.1f}.ron{:.1f}'.format(mode, config, binning,
                                                       gain, rdspeed)

    def get_ccdconf(self, logitem):
        return (logitem['binning'], logitem['gain'], logitem['rdnoise'])

    def get_ccdconf_string(self, ccdconf):
        binning, gain, rdnoise = ccdconf
        return '{}_gain{:.1f}_ron{}'.format(binning, gain, rdnoise)

    def trace(self):
        if self.mode != 'echelle':
            raise ValueError

        # crop
        data = self.flat_data[800:, :]

        coeff_lst, goodmask = find_echelle_apertures(data,
                                scan_step=50, align_deg=3)
        self.coeff_lst = coeff_lst
        self.goodmask = goodmask

    def get_echelle_sens(self):
        data = self.flat_data[800:, :]
        ny, nx = data.shape
        allx = np.arange(nx)
        ally = np.arange(ny)

        # fix flat data
        #m = np.ones(ny, dtype=bool)
        #m[1537-800:1538+1-800] = False
        #fixdata = data.copy()
        #fig = plt.figure()
        #ax = fig.gca()
        #for x in np.arange(nx):
        #    flux = np.log(data[:, x])
        #    fixfunc = intp.InterpolatedUnivariateSpline(ally[m], flux[m], k=2)
        #    fixdata[:, x][~m] = np.exp(fixfunc(ally[~m]))
        #    if 320 < x < 330:
        #        ax.plot(ally, flux, '-')
        #        ax.plot(ally[m], flux[m], '--')
        #        ax.plot(ally[~m], fixdata[:,x][~m], 'o')
        #plt.show()
        #data = fixdata

        yy, xx = np.mgrid[:ny:, :nx:]
        sensmap = np.ones_like(data, dtype=np.float32)

        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(np.log10(data))
        fig2 = plt.figure()
        ax2 = fig2.gca()

        allmask = np.zeros_like(data, dtype=bool)
        for iorder, coeff in enumerate(self.coeff_lst):
            win = 5

            cen_lst = np.polyval(coeff, allx)
            ax.plot(allx, cen_lst, c='r', ls='-', lw=0.5)

            mask = (yy<cen_lst+win)*(yy>cen_lst-win)
            allmask += (yy<cen_lst+win+5)*(yy>cen_lst-win-5)

            # skip ghost orders
            if not self.goodmask[iorder]:
                continue

            spec = (data*mask).sum(axis=0)/10
            ax2.plot(spec, lw=0.5)
           
            fig0 =plt.figure(dpi=200)
            ax0 = fig0.gca()
            ax0.plot(spec, lw=0.5)
            ## smooth

            #spec_sm, _, m, std = iterative_savgol_filter(spec,
            #                    winlen=21, order=3,
            #                    upper_clip=5, lower_clip=5, maxiter=5)
           

            #core = sg.gaussian(15, std=3)
            #core /= core.sum()
            #spec_sm = np.convolve(spec, core, mode='same')

            m = np.ones_like(spec, dtype=bool)
            for i in range(3):
                coeff = np.polyfit(allx[m], spec[m], deg=7)
                res_lst = spec - np.polyval(coeff, allx)
                std = res_lst[m].std()
                newm = (res_lst > -3*std) * (res_lst < 3*std)
                if newm.sum()==m.sum():
                    break
                m = newm
            spec_sm = np.polyval(coeff, allx)
        
            ax0.plot(spec_sm, lw=0.5)
            minidx, minvalue = get_local_minima(spec_sm, window=31)
            ax0.plot(minidx, minvalue, 'o', ms=2, alpha=0.6)
            fig0.savefig('smooth_{:02d}.png'.format(iorder))
            plt.close(fig0)

            smooth_2d = np.tile(spec_sm, ny).reshape(ny, nx)
            sensmap[mask] = (data/smooth_2d)[mask]

    

        hdulst = fits.HDUList([fits.PrimaryHDU(data=data),
                               fits.ImageHDU(data=np.int16(~allmask)),
                               ])
        hdulst.writeto('flat_background.fits', overwrite=True)

        fig3 = plt.figure()
        ax3 = fig3.gca()
        ax3.imshow(np.log10(data*(~allmask)))

        fig4 = plt.figure()
        ax4 = fig4.gca()
        ax4.imshow(sensmap)

        fits.writeto('sens.fits', sensmap, overwrite=True)

        plt.show()

        self.sens_data = sensmap

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


    def extract_echelle_lamp(self):
        lamp_item_lst = filter(lambda item: item['datatype']=='SPECSLAMP',
                               self.logtable)

        nx = self.flat_data.shape[1]
        # define dtype of 1-d spectra
        types = [
                ('aperture',   np.int16),
                ('order',      np.int16),
                ('points',     np.int16),
                ('wavelength', (np.float64, nx)),
                ('flux',       (np.float32, nx)),
                ('mask',       (np.int32,   nx)),
                ]
        names, formats = list(zip(*types))
        wlcalib_spectype = np.dtype({'names': names, 'formats': formats})


        for logitem in lamp_item_lst:
            fileid = logitem['fileid']
            filename = self.fileid_to_filename(fileid)
            data, header = fits.getdata(filename, header=True)
            data = data - self.bias_data
            data = data[800:, :]
            data = data / self.sens_data
            ny, nx = data.shape
            allx = np.arange(nx)
            yy, xx = np.mgrid[:ny:, :nx:]

            spec_lst = []
            aper = 0
            for iorder, coeff in enumerate(self.coeff_lst):
                win = 5
                if not self.goodmask[iorder]:
                    continue
                cen_lst = np.polyval(coeff, allx)
                mask = (yy<cen_lst+win)*(yy>cen_lst-win)
                spec = (data*mask).sum(axis=0)

                # pack to table
                row = (aper, 0, spec.size,
                        np.zeros(nx, dtype=np.float64), # wavelength
                        spec,                           # flux
                        np.zeros(nx, dtype=np.int16),   # mask
                       )
                spec_lst.append(row)
                aper += 1
            spec_lst = np.array(spec_lst, dtype=wlcalib_spectype)
            
            hdulst = fits.HDUList([fits.PrimaryHDU(header=header),
                                   fits.BinTableHDU(data=spec_lst),
                                   ])
            hdulst.writeto('spec_{}.fits'.format(fileid), overwrite=True)


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

    def ident_longslit_wavelength(self):

        index_file = os.path.join(os.path.dirname(__file__),
                                  'data/calib/wlcalib_bfosc.dat')

        for logitem in self.logtable:
            if logitem['mode']!='longslit' or logitem['datatype']!='SPECLLAMP':
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
            #                        'data/linelist/{}_l.dat'.format(lamp))
            #linelist = Table.read(filename, format='ascii.fixed_width_two_line')
            #linelist = linelist[np.where(linelist['intlev']<=3)]

            linelist = linelist['wave_air', 'element', 'ion', 'source']

            window = 17
            deg = 5
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
            fig_lbl_lst  = result['fig_fitlbl']
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
            head['OBSERVAT'] = 'Xinglong'
            head['TELESCOP'] = 'Xinglong 2.16m'
            head['INSTRUME'] = 'BFOSC'
            head['FILEID']   = fileid
            head['OBJECT']   = lamp
            head['EXPTIME']  = logitem['exptime']
            head['DATEOBS']  = logitem['dateobs']
            head['MODE']     = logitem['mode']
            head['CONFIG']   = logitem['config']
            head['SLIT']     = logitem['slit']
            head['BINNING']  = logitem['binning']
            head['GAIN']     = logitem['gain']
            head['RDNOISE']  = logitem['rdnoise']
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

        self.wave_solutions = {}
        coeff_wave_lst = []

        for logitem_lst in self.calib_groups:
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
            _fileid = bandselect_fileids['R']
            flux = self.lamp_spec_lst[_fileid]['flux']
            ccf_lst = get_simple_ccf(flux, ref_flux, shift_lst)

            #fig = plt.figure()
            #ax = fig.gca()
            #ax.plot(shift_lst, ccf_lst, c='C3')

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

            #fig2 = plt.figure()
            #ax2 = fig2.gca()
            #ax2.plot(flux_mosaic)
            #plt.show()

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
            # append coeff
            coeff_wave_lst.append(coeff_wave)

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
            ax_imap2.plot(allwave, pixel_lst, lw=0.5)
            for ax in fig_imap.get_axes():
                ax.grid(True, ls='--', lw=0.5)
                ax.set_axisbelow(True)
                ax.set_xlim(allwave[0], allwave[-1])
            ax_imap2.set_xlabel(u'Wavelength (\xc5)')
            figname = 'wavelength_identmap_{}.png'.format(newfileid)
            figfilename = os.path.join(self.figpath, figname)
            fig_imap.savefig(figfilename)
            plt.show()
            plt.close(fig_imap)

            # save wavelength calibration data
            data = Table(dtype=[
                ('wavelength',  np.float64),
                ('flux',        np.float32),
                ])
            for _w, _f in zip(allwave, flux_mosaic):
                data.add_row((_w, _f))
            head = fits.Header()
            head['OBJECT']   = 'FeAr'
            head['TELESCOP'] = 'Xinglong 2.16m'
            head['INSTRUME'] = 'BFOSC'
            head['FILEID']   = newfileid
            head['MOSAICED'] = True
            head['NMOSAIC']  = 2
            head['WAVEBD']   = wavebound,
            head['FILEID1']  = fileid_B,
            head['FILEID2']  = fileid_R,
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

            self.wave_solutions[newfileid] = allwave

        # determine the global wavelength coeff
        self.wave_coeff = np.array(coeff_wave_lst).mean(axis=0)


    def plot_wlcalib(self):
        fig = plt.figure(figsize=(12,6), dpi=150)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.plot(self.wavelength, self.calibflux, lw=0.5)
        ax1.set_xlabel(u'Wavelength (\xc5)')
        plt.show()
    
    def find_distortion(self):

        wavebound = 6907
        coeff_lst = {}

        fig_dist = plt.figure(figsize=(8,6), dpi=100)
        ax_dist = fig_dist.add_axes([0.1, 0.1, 0.85, 0.8])

        coeff_lst = []

        for logitem_lst in self.calib_groups:
            q95_lst = {}
            for logitem in logitem_lst:
                filename = self.fileid_to_filename(logitem['fileid'])
                data = fits.getdata(filename)
                q95 = np.percentile(data, 95)
                q95_lst[logitem['fileid']] = q95
            sorted_q95_lst = sorted(q95_lst.items(), key=lambda item: item[1])
            bandselect_fileids = {
                    'R': sorted_q95_lst[0][0], # choose the smallest as red
                    'B': sorted_q95_lst[-1][0], # choose the largets as blue
                    }

            allwave = list(self.wave_solutions.values())[0]
            for band in ['B', 'R']:
                fileid = bandselect_fileids[band]
                filename = self.fileid_to_filename(fileid)
                data = fits.getdata(filename)
                data = data - self.bias_data
                data = data / self.sens_data
                ny, nx = data.shape
                allx = np.arange(nx)
                ally = np.arange(ny)
                hwidth = 5
                ref_spec = data[ny//2-hwidth:ny//2+hwidth, :].sum(axis=0)
                if band == 'R':
                    mask = allwave > wavebound
                else:
                    mask = allwave < wavebound
                ref_spec = ref_spec[mask]
                xcoord = allx[mask]
                ycoord_lst = []
                xshift_lst = []

                fig_distortion = plt.figure(dpi=100, figsize=(8, 6))
                ax01 = fig_distortion.add_axes([0.1, 0.55, 0.85, 0.36])
                ax02 = fig_distortion.add_axes([0.1, 0.10, 0.85, 0.36])
                for i, y in enumerate(np.arange(100, ny-100, 200)):
                    spec = data[y-hwidth:y+hwidth,:].sum(axis=0)
                    spec = spec[mask]
                    shift = find_shift_ccf(ref_spec, spec)
                    ycoord_lst.append(y)
                    xshift_lst.append(shift)
                    if i == 0:
                        ax01.plot(xcoord, spec, color='w', lw=0)
                        y1, y2 = ax01.get_ylim()
                        offset = (y2 - y1)/20
                    ax01.plot(xcoord-shift, spec+offset*i, lw=0.5)
                    ax02.plot(xcoord, spec+offset*i, lw=0.5)
                ax01.set_xlim(xcoord[0], xcoord[-1])
                ax02.set_xlim(xcoord[0], xcoord[-1])
                ax02.set_xlabel('Pixel')
                #fig.suptitle('{}'.format(fileid))
                figname = 'distortion_{}.png'.format(band)
                figfilename = os.path.join(self.figpath, figname)
                fig_distortion.savefig(figfilename)
                plt.close(fig_distortion)

                coeff = np.polyfit(ycoord_lst, xshift_lst, deg=2)
                # append to results
                coeff_lst.append(coeff)

                color = {'B': 'C0', 'R': 'C3'}[band]
                sign =  {'B': '<',  'R': '>'}[band]
                label = u'(\u03bb {} {} \xc5)'.format(sign, wavebound)
                ax_dist.plot(xshift_lst, ycoord_lst, 'o', c=color,
                             alpha=0.7, label=label)
                ax_dist.plot(np.polyval(coeff, ally), ally, color=color,
                             alpha=0.7)
        ax_dist.axhline(y=ny//2, ls='-', color='k', lw=0.7)
        ax_dist.set_ylim(0, ny - 1)
        ax_dist.xaxis.set_major_locator(tck.MultipleLocator(1))
        ax_dist.set_xlabel('Shift (pixel)')
        ax_dist.set_ylabel('Y (pixel)')
        ax_dist.grid(True, ls='--')
        ax_dist.set_axisbelow(True)
        ax_dist.legend(loc='upper left')
        figname = 'distortion_fitting.png'
        figfilename = os.path.join(self.figpath, figname)
        fig_dist.savefig(figfilename)
        plt.close(fig_dist)

        # take the average of coeff_lst as final curve_coeff
        self.curve_coeff = np.array(coeff_lst).mean(axis=0)

    def extract_longslit_arclamp(self):
        self.arclamp = {}
        for logitem in self.logtable:
            if logitem['mode'] != 'longslit':
                continue

            if logitem['datatype'] != 'SPECLLAMP':
                continue

            ccdconf = self.get_ccdconf(logitem)
            conf = self.get_conf(logitem)

            filename = self.fileid_to_filename(logitem['fileid'])
            data = fits.getdata(filename)
            data = data - self.bias[ccdconf]
            data = data / self.sensmap[conf]

            ny, nx = data.shape
            halfwidth = 5
            spec = data[ny//2-halfwidth:ny//2+halfwidth, :].mean(axis=0)

            self.arclamp[logitem['fileid']] = spec

            fig = plt.figure()
            ax = fig.gca()
            ax.plot(spec, lw=0.5)
            plt.show()
        

    def extract_all_science(self):
        func = lambda item: item['datatype']=='SPECLTARGET'
        logitem_lst = list(filter(func, self.logtable))
        for logitem in logitem_lst:
            self.extract(logitem)


    def extract(self, arg):

        if isinstance(arg, int):
            if arg > 19000000:
                func = lambda item: item['datatype']=='SPECLTARGET' \
                            and item['fileid']==arg
                logitem_lst = list(filter(func, self.logtable))
            else:
                func = lambda item: item['datatype']=='SPECLTARGET' \
                            and item['frameid']==arg
                logitem_lst = list(filter(func, self.logtable))

            if len(logitem_lst)==0:
                print('unknown fileid: {}'.format(arg))
                return None
            elif len(logitem_lst)>1:
                print('Multiple items found for {}'.format(arg))
                return None
            else:
                logitem = logitem_lst[0]
        elif isinstance(arg, str):

            func = lambda item: item['datatype']=='SPECLTARGET' \
                        and item['object'].lower()==arg.strip().lower()
            logitem_lst = list(filter(func, self.logtable))

            if len(logitem_lst)==0:
                print('unknown object name: {}'.format(arg))
                return None
            elif len(logitem_lst)>1:
                print('Multiple items found for {}'.format(arg))
                return None
            else:
                logitem = logitem_lst[0]

        elif isinstance(arg, Row):
            logitem = arg
        else:
            print('Unkown target: {}'.format(arg_))
            return None

        fileid  = logitem['fileid']
        objname = logitem['object']

        print('* FileID: {} - 1d spectra extraction'.format(fileid))
        filename = self.fileid_to_filename(fileid)
        data = fits.getdata(filename)

        data = data - self.bias_data
        data = data / self.sens_data

        ny, nx = data.shape
        allx = np.arange(nx)
        ally = np.arange(ny)
        xdata = ally

        figname = 'trace_{}.png'.format(fileid)
        figfilename = os.path.join(self.figpath, figname)
        title = 'Trace for {} ({})'.format(fileid, objname)
        coeff_loc, fwhm_mean, profile_func = find_order_location(
                                                data, figfilename, title)

        # generate order location array
        ycen = np.polyval(coeff_loc, allx)
     
        # generate wavelength list considering horizontal shift
        xshift_lst = np.polyval(self.curve_coeff, ycen)
        wave_lst = np.polyval(self.wave_coeff, allx - xshift_lst)
     
        # extract 1d sepectra
     
        # summ extraction
        yy, xx = np.mgrid[:ny, :nx]
        upper_line = ycen + fwhm_mean
        lower_line = ycen - fwhm_mean
        upper_ints = np.int32(np.round(upper_line))
        lower_ints = np.int32(np.round(lower_line))
        extmask = (yy > lower_ints) * (yy < upper_ints)
        mask = np.float32(extmask)
        # determine the weights in the boundary
        mask[upper_ints, allx] = (upper_line + 0.5) % 1
        mask[lower_ints, allx] = 1 - (lower_line + 0.5) % 1

        # extract
        spec_sum = (data * mask).sum(axis=0)
        nslit = mask.sum(axis=0)

        # correct image distortion
        ycen0 = ycen[0:-200].mean()
        cdata = np.zeros_like(data, dtype=data.dtype)
        for y in np.arange(ny):
            row = data[y, :]
            shift = np.polyval(self.curve_coeff, y) - np.polyval(self.curve_coeff, ycen0)
            f = intp.InterpolatedUnivariateSpline(allx, row, k=3, ext=3)
            cdata[y, :] = f(allx + shift)

        # initialize background mask
        bkgmask = np.zeros_like(data, dtype=bool)
        # index of rows to exctract background
        background_rows = [(520, 800), (1000, 1250)]
        for r1, r2 in background_rows:
            bkgmask[r1:r2, :] = True

        # remove cosmic rays in the background region
        # ori_bkgspec= (cdata*bkgmask).sum(axis=0)/(bkgmask.sum(axis=0))
     
        # method 1
        # for r1, r2 in background_rows:
        #    cutdata = cdata[r1:r2, :]
        #    fildata = median_filter(cutdata, (1, 5), mode='nearest')
        #    resdata = cutdata - fildata
        #    std = resdata.std()
        #    mask = (resdata < 3*std)*(resdata > -3*std)
        #    bkgmask[r1:r2, :] = mask
     
        # method 2
        for r1, r2 in background_rows:
            cutdata = cdata[r1:r2, :]
            mean = cutdata.mean(axis=0)
            std = cutdata.std()
            mask = (cutdata < mean + 3 * std) * (cutdata > mean - 3 * std)
            bkgmask[r1:r2, :] = mask

        # plot the bkg and bkg mask
        # fig0 = plt.figure()
        # ax01 = fig0.add_subplot(121)
        # ax02 = fig0.add_subplot(122)
        # for r1, r2 in background_rows:
        #    for y in np.arange(r1, r2):
        #        ax01.plot(cdata[y, :]+y*15, lw=0.5)
        #        m = ~bkgmask[y, :]
        #        ax01.plot(allx[m], cdata[y, :][m]+y*15, 'o', color='C0')
        # bkgspec = (cdata*bkgmask).sum(axis=0)/(bkgmask.sum(axis=0))
        # ax02.plot(ori_bkgspec)
        # ax02.plot(bkgspec)
        # plt.show()
     
        # remove the peaks in the spatial direction
        # sum of background mask along y
        bkgmasksum = bkgmask.sum(axis=1)
        # find positive positions
        posmask = np.nonzero(bkgmasksum)[0]
        # initialize crossspec
        crossspec = np.zeros(ny)
        crossspec[posmask] = (cdata * bkgmask).sum(axis=1)[posmask] / bkgmasksum[posmask]
        fitx = ally[posmask]
        fity = crossspec[posmask]
        fitmask = np.ones_like(posmask, dtype=bool)
        maxiter = 3
        for i in range(maxiter):
            c = np.polyfit(fitx[fitmask], fity[fitmask], deg=2)
            res_lst = fity - np.polyval(c, fitx)
            std = res_lst[fitmask].std()
            new_fitmask = (res_lst > -2 * std) * (res_lst < 2 * std)
            if new_fitmask.sum() == fitmask.sum():
                break
            fitmask = new_fitmask
     
        # block these pixels in bkgmask
        for y in ally[posmask][~fitmask]:
            bkgmask[y, :] = False
     
        # plot the cross-section of background regions
        fig100 = plt.figure(figsize=(9, 6), dpi=100)
        ax1 = fig100.add_axes([0.07, 0.54, 0.87, 0.36])
        ax2 = fig100.add_axes([0.07, 0.12, 0.87, 0.36])
        newy = np.polyval(c, ally)
        for ax in fig100.get_axes():
            ax.plot(ally, cdata.mean(axis=1), alpha=0.3, color='C0', lw=0.7)
        y1, y2 = ax1.get_ylim()
     
        ylst = ally[posmask][fitmask]
        for idxlst in np.split(ylst, np.where(np.diff(ylst) != 1)[0] + 1):
            for ax in fig100.get_axes():
                ax.plot(ally[idxlst], crossspec[idxlst], color='C0', lw=0.7)
                ax.fill_betweenx([y1, y2], idxlst[0], idxlst[-1],
                                 facecolor='C2', alpha=0.15)
     
        for ax in fig100.get_axes():
            ax.plot(ally, newy, color='C1', ls='-', lw=0.5)
     
        ax2.plot(ally, newy + std, color='C1', ls='--', lw=0.5)
        ax2.plot(ally, newy - std, color='C1', ls='--', lw=0.5)
        for ax in fig100.get_axes():
            ax.set_xlim(0, ny - 1)
            ax.grid(True, ls='--', lw=0.5)
            ax.set_axisbelow(True)
        ax1.set_ylim(y1, y2)
        ax2.set_ylim(newy.min() - 6 * std, newy.max() + 6 * std)
        ax2.set_xlabel('Y (pixel)')
        title = '{} ({})'.format(fileid, objname)
        fig100.suptitle(title)
        figname = 'bkg_cross_{}.png'.format(fileid)
        figfilename = os.path.join(self.figpath, figname)
        fig100.savefig(figfilename)
        plt.close(fig100)
     
        # plot a 2d image of distortion corrected image
        # and background region
        fig3 = plt.figure(dpi=100, figsize=(12, 6))
        ax31 = fig3.add_axes([0.07, 0.1, 0.4, 0.8])
        ax32 = fig3.add_axes([0.55, 0.1, 0.4, 0.8])
        vmin = np.percentile(cdata, 10)
        vmax = np.percentile(cdata, 99)
        ax31.imshow(cdata, origin='lower', vmin=vmin, vmax=vmax)
        bkgdata = np.zeros_like(cdata, dtype=cdata.dtype)
        bkgdata[bkgmask] = cdata[bkgmask]
        bkgdata[~bkgmask] = (vmin + vmax) / 2
        ax32.imshow(bkgdata, origin='lower', vmin=vmin, vmax=vmax)
        for ax in fig3.get_axes():
            ax.set_xlim(0, nx - 1)
            ax.set_ylim(0, ny - 1)
            ax.set_xlabel('X (pixel)')
            ax.set_ylabel('Y (pixel)')
        title = '{} ({})'.format(fileid, objname)
        fig3.suptitle(title)
        figname = 'bkg_region_{}.png'.format(fileid)
        figfilename = os.path.join(self.figpath, figname)
        fig3.savefig(figfilename)
        plt.close(fig3)
     
        # background spectra per pixel along spatial direction
        bkgspec = (cdata * bkgmask).sum(axis=0) / (bkgmask.sum(axis=0))
        # background spectra in the spectrum aperture
        background_sum = bkgspec * nslit
     
        spec_sum_dbkg = spec_sum - background_sum


def find_order_location(data, figfilename, title):

    def errfunc(p, x, y, fitfunc):
        return y - fitfunc(p, x)

    def fitfunc(p, x):
        return gaussian(p[0], p[1], p[2], x)+p[3]


    ny, nx = data.shape
    allx = np.arange(nx)
    ally = np.arange(ny)
    ymax = data[:,30:250].mean(axis=1).argmax()
    #print(ymax)
    xscan_lst = []
    ycen_lst = []
    fwhm_lst = []

    xnode_lst = []
    ynode_lst = []

    # make a plot
    fig1 = plt.figure(figsize=(12, 8), dpi=100)
    w1 = 0.39
    ax1 = fig1.add_axes([0.07, 0.43, 0.39, 0.50])
    ax2 = fig1.add_axes([0.56, 0.07, w1, 0.22])
    ax3 = fig1.add_axes([0.07, 0.07, 0.39, 0.30])
    ax4 = fig1.add_axes([0.56, 0.35, w1, w1 / 2 * 3])
    offset_mark = 0
    yc = ymax
    for ix, x in enumerate(np.arange(30, nx-200, 50)):
        xdata = ally
        # ydata = data[:,x]
        ydata = data[:, x - 20:x + 21].mean(axis=1)
        y1 = yc - 20
        y2 = yc + 20
        yc = ydata[y1:y2].argmax() + y1
        succ = True


        for i in range(2):
            y1 = yc - 20
            y2 = yc + 20
            xfitdata = ally[y1:y2]
            yfitdata = ydata[y1:y2]
            p0 = [yfitdata.max() - yfitdata.min(), 6.0, yc, yfitdata.min()]

            mask = np.ones_like(xfitdata, dtype=bool)
            for j in range(2):
                fitres = opt.least_squares(errfunc, p0,
                            bounds=([0,      3,  -np.inf, -np.inf],
                                    [np.inf, 50, np.inf,  np.inf]),
                            args=(xfitdata[mask], yfitdata[mask],
                                                 fitfunc))
                p = fitres['x']
                res_lst = errfunc(p, xfitdata, yfitdata, fitfunc)
                std = res_lst[mask].std()
                new_mask = (res_lst > -3 * std) * (res_lst < 3 * std)
                mask = new_mask

            A, fwhm, center, bkg = p
            if A < 0 or fwhm > 100:
                succ = False
                break
            yc = int(round(center))

        if not succ:
            continue

        # pack results
        xscan_lst.append(x)
        ycen_lst.append(center)
        fwhm_lst.append(fwhm)

        newx = np.arange(xfitdata[0], xfitdata[-1], 0.1)
        newy = fitfunc(p, newx)

        # plot fitting in ax1
        # determine the color

        if offset_mark == 0:
            offset_rate = A * 0.002
            offset_mark = 1
        offset = x * offset_rate

        color = 'C{:d}'.format(ix % 10)
        ax1.scatter(xfitdata, yfitdata - bkg + offset,
                    alpha=0.5, s=4, c=color)
        ax1.plot(newx, newy - bkg + offset,
                 alpha=0.5, lw=0.5, color=color)
        ax1.vlines(center, offset, A + offset,
                   color=color, lw=0.5, alpha=0.5)
        ax1.hlines(offset, center - fwhm, center + fwhm,
                   color=color, lw=0.5, alpha=0.5)

        # plot stacked profiles in ax3
        xprofile = xfitdata - center
        yprofile = (yfitdata - bkg) / A
        ax3.plot(xprofile, yprofile, color=color, alpha=0.5, lw=0.5)

        for vx, vy in zip(xprofile, yprofile):
            xnode_lst.append(vx)
            ynode_lst.append(vy)

    xscan_lst = np.array(xscan_lst)
    ycen_lst = np.array(ycen_lst)
    fwhm_lst = np.array(fwhm_lst)

    # fit the order position with iterative polynomial
    mask = np.ones_like(xscan_lst, dtype=bool)
    for i in range(2):
        coeff_loc = np.polyfit(xscan_lst[mask], ycen_lst[mask], deg=3)
        res_lst = ycen_lst - np.polyval(coeff_loc, xscan_lst)
        std = res_lst[mask].std()
        new_mask = (res_lst < 3 * std) * (res_lst > -3 * std)
        mask = new_mask

    # final position
    ycen = np.polyval(coeff_loc, allx)

    # calculate averaged FWHM
    fwhm_mean, _, _ = get_clip_mean(fwhm_lst)

    # adjust fitting in ax1
    ax1.set_xlim(ycen_lst.min() - 20, ycen_lst.max() + 20)
    ax1.set_ylim(-offset_rate * 200, offset_rate * 2500)
    ax1.set_xlabel('Y (pixel)')
    ax1.set_ylabel('Flux (with offset)')

    # plot fitted positon in ax2
    ax2.scatter(xscan_lst, ycen_lst,
                alpha=0.8, s=20, edgecolor='C0', c='w')
    ax2.scatter(xscan_lst[mask], ycen_lst[mask],
                alpha=0.8, s=20, c='C0')
    ax2.errorbar(xscan_lst, ycen_lst, yerr=fwhm_lst,
                 fmt='o', capsize=0, ms=0, zorder=-1)
    ax2.plot(allx, ycen, ls='-', color='C3', lw=1, alpha=0.6)
    ax2.plot(allx, ycen + std, ls='--', color='C3', lw=1, alpha=0.6)
    ax2.plot(allx, ycen - std, ls='--', color='C3', lw=1, alpha=0.6)
    ax2.grid(True, ls='--')
    ax2.set_axisbelow(True)
    x1, x2 = 0, nx - 1
    y1 = (ycen - fwhm_mean * 3).min()
    y2 = (ycen + fwhm_mean * 3).max()
    ax2.text(0.95 * x1 + 0.05 * x2, 0.15 * y1 + 0.85 * y2,
             'Mean FWHM = {:4.2f}'.format(fwhm_mean))
    ax2.set_xlim(x1, x2)
    ax2.set_ylim(y1, y2)
    ax2.set_xlabel('X (pixel)')
    ax2.set_ylabel('Y (pixel)')

    # calculate average profile
    xnode_lst = np.array(xnode_lst)
    ynode_lst = np.array(ynode_lst)

    step = 1
    xprofile_node_lst = []
    yprofile_node_lst = []
    for x in np.arange(-18, 18, step):
        x1 = x - step / 2
        x2 = x + step / 2
        m = (xnode_lst > x1) * (xnode_lst < x2)
        ymean, _, mask = get_clip_mean(ynode_lst[m])
        xprofile_node_lst.append(xnode_lst[m][mask].mean())
        yprofile_node_lst.append(ymean)
    xprofile_node_lst = np.array(xprofile_node_lst)
    yprofile_node_lst = np.array(yprofile_node_lst)

    yprofile_node_lst = np.maximum(yprofile_node_lst, 0)
    interf = intp.InterpolatedUnivariateSpline(
        xprofile_node_lst, yprofile_node_lst, k=3, ext=1)

    # adjust ax3
    ax3.plot(xprofile_node_lst, yprofile_node_lst, 'o-',
             ms=3, color='k', lw=0.5)
    ax3.set_ylim(-0.5, 1.5)
    ax3.grid(True, ls='--', lw=0.5)
    ax3.axvline(0, ls='--', lw=1, color='k')
    ax3.axhline(0, ls='--', lw=1, color='k')
    ax3.set_axisbelow(True)
    ax3.set_xlabel('X (pixel)')

    # plot image data in ax4
    vmin = np.percentile(data, 10)
    vmax = np.percentile(data, 99)
    ax4.imshow(data, origin='lower', vmin=vmin, vmax=vmax)
    ax4.set_xlim(0, nx - 1)
    ax4.set_ylim(0, ny - 1)
    ax4.set_xlabel('X (pixel)')
    ax4.set_ylabel('Y (pixel)')
    ax4.plot(allx, ycen, ls='-', color='C3', lw=0.5, alpha=1)
    ax4.plot(allx, ycen + fwhm_mean, ls='--', color='C3', lw=0.5, alpha=1)
    ax4.plot(allx, ycen - fwhm_mean, ls='--', color='C3', lw=0.5, alpha=1)
    # set title
    fig1.suptitle(title)
    fig1.savefig(figfilename)
    plt.close(fig1)

    return coeff_loc, fwhm_mean, interf


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

