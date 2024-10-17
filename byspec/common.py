import os
import numpy as np
from astropy.table import Table

from .imageproc import combine_images

class FOSCReducer(object):
    
    def __init__(self, **kwargs):

        rawdata_path = kwargs.pop('rawdata_path', None)
        if rawdata_path is not None and os.path.exists(rawdata_path):
            self.set_rawdata_path(rawdata_path)

        reduction_path = kwargs.pop('reduction_path', './')
        if reduction_path is not None:
            self.set_reduction_path(reduction_path)

        obslogfile = kwargs.pop('obslog', None)
        if obslogfile is not None:
            self.read_obslog(obslogfile)

    def set_rawdata_path(self, rawdata_path):
        """Set rawdata path.
        """
        if os.path.exists(rawdata_path):
            self.rawdata_path = rawdata_path

    def set_reduction_path(self, reduction_path):
        """Set data reduction path

        """
        if not os.path.exists(reduction_path):
            os.mkdir(reduction_path)
        self.reduction_path = reduction_path

        self.figpath = os.path.join(self.reduction_path, 'figures')
        if not os.path.exists(self.figpath):
            os.mkdir(self.figpath)

        #self.bias_file = os.path.join(self.reduction_path, 'bias.fits')
        #self.flat_file = os.path.join(self.reduction_path, 'flat.fits')
        #self.sens_file = os.path.join(self.reduction_path, 'sens.fits')

    def read_obslog(self, filename):
        self.logtable = Table.read(filename,
                            format='ascii.fixed_width_two_line')
        print(self.logtable)

    def get_all_conf(self):
        conf_lst = []
        for logitem in self.logtable:
            conf = self.get_conf(logitem)
            if conf not in conf_lst:
                conf_lst.append(conf)
        self.conf_lst = conf_lst

    def get_all_ccdconf(self):
        ccdconf_lst = []
        for logitem in self.logtable:
            ccdconf = self.get_ccdconf(logitem)
            if ccdconf not in ccdconf_lst:
                ccdconf_lst.append(ccdconf)
        self.ccdconf_lst = ccdconf_lst

    def combine_flat(self):
        """Combine flat images
        """

        conf_lst = []
        # scan the entire logtable
        for logitem in self.logtable:
            if logitem['object']=='FLAT':
                conf = get_conf_string(logitem)
                if conf not in conf_lst:
                    conf_lst.append(conf)

        for conf in conf_lst:
            flat_file = 'flat.{}.fits'.format(conf)

            flat_filename = os.path.join(self.reduction_path, flat_file)

            if os.path.exists(flat_filename):
                self.flat_data[_conf] = fits.getdata(flat_filename)
            else:
                print('Combine Flat')
                data_lst = []
                for logitem in logtable:
                    _conf = self.get_conf_string(logitem)
                    if _conf != conf:
                        continue
                    filename = self.fileid_to_filename(logitem['fileid'])
                    data = fits.getdata(filename)
                    # correct bias
                    data = data - self.bias_data
                    data_lst.append(data)
                data_lst = np.array(data_lst)
                flat_data = combine_images(data_lst, mode='mean',
                                    upper_clip=5, maxiter=10, maskmode='max')
                fits.writeto(flat_filename, flat_data, ovewrite=True)
                self.flat_data[_conf] = flat_data
