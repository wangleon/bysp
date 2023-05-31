import os
import re
import numpy as np
from astropy.table import Table
import astropy.io.fits as fits
import matplotlib.pyplot as plt

from .imageproc import combine_images

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

        self.bias_file = os.path.join(self.reduction_path, 'bias.fits')
        self.flat_file = os.path.join(self.reduction_path, 'flat.fits')

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

        if False:
            pass

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



BFOSC = _BFOSC()
