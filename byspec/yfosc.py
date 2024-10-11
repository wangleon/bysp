import os
import re
import datetime

import numpy as np
import scipy.signal as sg
import scipy.interpolate as intp
from astropy.table import Table
import astropy.io.fits as fits

from .imageproc import combine_images
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

    fmt_str = ('  - {:7s} {:17s} {:>10s} {:>16s} {:>8} {:23s}'
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
                    '{:10s}'.format(datatype),
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
        y1, y2 = 1900, 4300
        return data[y1:y2, :]
    else:
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
            bias_file = 'bias_{}.fits'.format(ccdconf)
            bias_filename = os.path.join(self.reduction_path, bias_file)
            if os.path.exists(bias_filename):
                hdulst = fits.open(bias_filename)
                bias_data = hdulst[1].data
                bias_img  = hdulst[2].data
                hdulst.close()
            else:
                print('Combine Bias')
                data_lst = []
                selectfunc = lambda item: item['datatype']=='BIAS' and \
                                self.get_ccdconf_string(item)==ccdconf

                bias_item_lst = filter(selectfunc, self.logtable)

                for logitem in bias_item_lst:
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

    def get_ccdconf(self, logitem):
        return (logitem['binning'], logitem['gain'], logitem['rdspeed'])

    def get_ccdconf_string(self, logitem):
        return '{binning}_gain{gain:3.1f}_{rdspeed}'.format(**logitem)

    def fileid_to_filename(self, fileid):
        prefix = 'ljg2m401-yf01-'
        for fname in os.listdir(self.rawdata_path):
            if fname.startswith(prefix + str(fileid)):
                return os.path.join(self.rawdata_path, fname)
        raise ValueError

