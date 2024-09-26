import os
import re
import datetime

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
            #('observer',    str),
        ], masked=True)

    fn_pattern = 'ljg2m401\-yf01\-(\d{8}\-\d{4}\-[a-z]\d{2})\.fits\.fz'

    for fname in sorted(os.listdir(rawpath)):
        if not (fname.endswith('.fits') or fname.endswith('.fits.fz')):
            continue
        filename = os.path.join(rawpath, fname)
        hdulst = fits.open(filename)
        head = hdulst[1].header
        hdulst.close()

        # get fileid
        mobj = re.match(fn_pattern, fname)
        if not mobj:
            continue
        fileid      = mobj.group(1)

        # get header keywords
        frameid     = head['FRAMENUM']
        datatype    = head['OBSTYPE']
        ra          = head['RA']
        dec         = head['DEC']
        objname     = head['OBJECT']
        exptime     = head['EXPTIME']
        dateobs     = head['DATE-OBS']
        filter1     = head['FILTER1']
        filter2     = head['FILTER2']
        filter3     = head['FILTER3']
        filter4     = head['FILTER4']
        filter5     = head['FILTER5']
        filter6     = head['FILTER6']
        filter7     = head['FILTER7']

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
        rdnoise = head['RDNOISE']
    
        logtable.add_row((frameid, fileid, datatype, objname, exptime, dateobs,
                          ra, dec, mode, config, slit, binning, gain, rdnoise))

    logtable['exptime'].info.format='%7.3f'
    return logtable

class _YFOSC(object):
    def __init__(self, rawdata_path=None):

        if rawdata_path is not None and os.path.exists(rawdata_path):
            self.rawdata_path = rawdata_path

    def make_obslog(self, filename=None):
        logtable = make_obslog(self.rawdata_path)

        # find obsdate
        self.obsdate = obstable[0]['dateobs'][0:10]

        if filename is None:
            filename = 'YFOSC.{}.txt'.format(self.obsdate)

        filename = os.path.join(self.reduction_path, filename)

        logtable.write(filename, format='ascii.fixed_width_two_line',
                       overwrite=True)

        self.logtable = logtable

    def fileid_to_filename(self, fileid):
        pass

YFOSC = _YFOSC()
