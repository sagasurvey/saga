import os
import logging
import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from astropy.io import fits
from astropy.io.ascii.cparser import CParserError # pylint: disable=E0611
import astropy.constants
from easyquery import Query

from .core import FitsTable, FileObject
from ..utils import get_empty_str_array, add_skycoord


__all__ = ['read_gama', 'read_mmt', 'read_aat', 'read_aat_mz', 'read_imacs',
           'read_wiyn', 'read_deimos', 'read_palomar', 'read_2df', 'read_6df',
           'read_ozdes', 'read_2dflens',
           'extract_sdss_spectra', 'extract_nsa_spectra', 'SpectraData',
           'ensure_specs_dtype', 'SPECS_COLUMNS']


_SPEED_OF_LIGHT = astropy.constants.c.to_value('km/s') # pylint: disable=E1101

SPECS_COLUMNS = {
    'RA': '<f8',
    'DEC': '<f8',
    'SPEC_Z': '<f4',
    'SPEC_Z_ERR': '<f4',
    'ZQUALITY': '<i2',
    'SPECOBJID': '<U48',
    'MASKNAME': '<U48',
    'TELNAME': '<U6',
    'EM_ABS': '<i2',
    'HELIO_CORR': '|b1',
}

def ensure_specs_dtype(spectra, cols_definition=SPECS_COLUMNS):
    for c, t in cols_definition.items():
        if c not in spectra.colnames:
            if t[1] == 'f':
                spectra[c] = np.nan
            elif t[1] == 'i':
                spectra[c] = -1
            elif t[1] == 'U':
                spectra[c] = ''
            elif t[1] == 'b':
                spectra[c] = False
            else:
                raise ValueError('unknown spec type!')
        if spectra[c].dtype.str != t:
            spectra[c] = spectra[c].astype(t)

    return spectra


def heliocentric_correction(fits_filepath, site_name,
                            ra_name='RA', dec_name='DEC', time_name='MJD',
                            ra_unit='hourangle', dec_unit='deg', time_format='mjd'):
    hdr = fits.getheader(fits_filepath)
    sc = SkyCoord(hdr[ra_name], hdr[dec_name], unit=(ra_unit, dec_unit))
    obstime = Time(hdr[time_name], format=time_format)
    helio_corr = sc.radial_velocity_correction('heliocentric', obstime=obstime, location=EarthLocation.of_site(site_name))
    return helio_corr.to_value(astropy.constants.c), obstime # pylint: disable=E1101


def read_gama(file_path):
    if not hasattr(file_path, 'read'):
        file_path = FitsTable(file_path)
    specs = file_path.read()['RA', 'DEC', 'SPECID', 'CATAID', 'NQ', 'Z', 'SURVEY']
    specs = Query('NQ >= 3', (lambda x: x != 'SDSS', 'SURVEY')).filter(specs)
    del specs['SURVEY']
    specs.rename_column('SPECID', 'SPECOBJID')
    specs.rename_column('CATAID', 'MASKNAME')
    specs.rename_column('NQ', 'ZQUALITY')
    specs.rename_column('Z', 'SPEC_Z')
    specs['SPEC_Z_ERR'] = 60 / _SPEED_OF_LIGHT
    specs['TELNAME'] = get_empty_str_array(len(specs), 6, 'GAMA')
    specs['ZQUALITY'][Query('ZQUALITY > 4').mask(specs)] = 4
    specs['HELIO_CORR'] = True

    return ensure_specs_dtype(specs)


def read_generic_spectra(dir_path, extension, telname, usecols, n_cols_total,
                         cuts=None, postprocess=None, midprocess=None, **kwargs):

    names = [usecols.get(i+1, '_{}'.format(i)) for i in range(n_cols_total)]
    exclude_names = [n for n in names if n.startswith('_')]

    output = []

    for f in os.listdir(dir_path):
        if not f.endswith(extension):
            continue
        try:
            this = Table.read(os.path.join(dir_path, f), format='ascii.fast_no_header',
                              guess=False, names=names, exclude_names=exclude_names, **kwargs)
        except (IOError, CParserError) as e:
            logging.warning('SKIPPING spectra file {}/{} - could not read or parse\n{}'.format(dir_path, f, e))
            continue
        this = Query(cuts).filter(this)
        if 'MASKNAME' not in this.colnames:
            this['MASKNAME'] = f
        if midprocess:
            this = midprocess(this)
        if this is not None:
            output.append(this)

    output = vstack(output, 'exact', 'error')
    output['TELNAME'] = telname
    if postprocess:
        output = postprocess(output)

    return ensure_specs_dtype(output)


def read_mmt(dir_path, before_time=None):

    usecols = {2:'RA', 3:'DEC', 4:'mag', 5:'SPEC_Z', 6:'SPEC_Z_ERR',
               7:'ZQUALITY', 8:'SPECOBJID'}
    cuts = Query('mag != 0', 'ZQUALITY >= 1')

    def midprocess(t):
        fits_filepath = os.path.join(dir_path, t['MASKNAME'][0].replace('.zlog', '.fits.gz'))
        try:
            corr, obstime = heliocentric_correction(fits_filepath, 'mmt', 'RA', 'DEC', 'MJD')
        except IOError:
            t['HELIO_CORR'] = False
        else:
            if before_time is not None and obstime > before_time:
                return
            t['SPEC_Z'] += corr
            t['HELIO_CORR'] = True
        return t

    def postprocess(t):
        del t['mag']
        t['RA'] *= 15.0
        return t

    return read_generic_spectra(dir_path, '.zlog', 'MMT', usecols, 11, cuts, postprocess, midprocess)


def read_aat(dir_path, before_time=None):

    usecols = {2:'RA', 3:'DEC', 5:'SPEC_Z', 7:'ZQUALITY', 8:'SPECOBJID'}
    cuts = Query('ZQUALITY >= 1')

    def midprocess(t):
        fits_filepath = os.path.join(dir_path, t['MASKNAME'][0].replace('.zlog', '.fits.gz'))
        try:
            corr, obstime = heliocentric_correction(fits_filepath, 'sso', 'MEANRA', 'MEANDEC', 'UTMJD')
        except IOError:
            t['HELIO_CORR'] = False
        else:
            if before_time is not None and obstime > before_time:
                return
            t['SPEC_Z'] += corr
            t['HELIO_CORR'] = True
        return t

    def postprocess(t):
        t['SPEC_Z_ERR'] = 10 / _SPEED_OF_LIGHT
        return t

    return read_generic_spectra(dir_path, '.zlog', 'AAT', usecols, 11, cuts, postprocess, midprocess)


def read_aat_mz(dir_path, before_time=None):

    usecols = {3:'RA', 4:'DEC', 13:'SPEC_Z', 14:'ZQUALITY', 1:'SPECOBJID'}
    cuts = Query('ZQUALITY >= 1')

    def midprocess(t):
        fits_filepath = os.path.join(dir_path, t['MASKNAME'][0].replace('.mz', '.fits.gz'))
        try:
            corr, obstime = heliocentric_correction(fits_filepath, 'sso', 'MEANRA', 'MEANDEC', 'UTMJD')
        except IOError:
            t['HELIO_CORR'] = False
        else:
            if before_time is not None and obstime > before_time:
                return
            t['SPEC_Z'] += corr
            t['HELIO_CORR'] = True
        return t

    def postprocess(t):
        t['RA'] *= 180.0/np.pi
        t['DEC'] *= 180.0/np.pi
        t['SPEC_Z_ERR'] = 10 / _SPEED_OF_LIGHT
        return t

    return read_generic_spectra(dir_path, '.mz', 'AAT', usecols, 15, cuts, postprocess, midprocess, delimiter=',')


def read_imacs(dir_path):

    usecols = {2:'RA', 3:'DEC', 5:'SPEC_Z', 6:'SPEC_Z_ERR', 7:'ZQUALITY', 1:'SPECOBJID', 11:'MASKNAME'}
    cuts = Query('ZQUALITY >= 1')
    def midprocess(t):
        t['SPECOBJID'] = t['SPECOBJID'].astype('<U48')
        return t
    return read_generic_spectra(dir_path, '.zlog', 'IMACS', usecols, 12, cuts, midprocess=midprocess)


def read_wiyn(dir_path):

    output = []

    for f in os.listdir(dir_path):
        if not f.endswith('.fits.gz'):
            continue
        this = FitsTable(os.path.join(dir_path, f)).read()[['RA', 'DEC', 'ZQUALITY', 'FID', 'Z', 'Z_ERR']]
        this = Query('ZQUALITY >= 1').filter(this)
        this['MASKNAME'] = f
        output.append(this)

    output = vstack(output, 'exact', 'error')

    output.rename_column('FID', 'SPECOBJID')
    output.rename_column('Z', 'SPEC_Z')
    output.rename_column('Z_ERR', 'SPEC_Z_ERR')
    output['TELNAME'] = 'WIYN'

    sc = SkyCoord(output['RA'], output['DEC'], unit=("hourangle", "deg"))
    output['RA'] = sc.ra.deg
    output['DEC'] = sc.dec.deg
    del sc

    return ensure_specs_dtype(output)


def read_deimos():
    data = {
        'RA'         : [247.825839103498, 221.86742, 150.12470],
        'DEC'        : [20.210825313885, -0.28144459, 32.561687],
        'MASKNAME'   : ['deimos2014', 'deimos2016-DN1', 'deimos2016-MD1'],
        'SPECOBJID'  : [0, 0, 0],
        'SPEC_Z'     : [2375/_SPEED_OF_LIGHT, 0.056, 1.08],
        'SPEC_Z_ERR' : [0.001, 0.001, 0.001],
        'ZQUALITY'   : [4, 4, 4],
        'TELNAME'    : ['DEIMOS', 'DEIMOS', 'DEIMOS'],
    }
    return ensure_specs_dtype(Table(data))


def read_palomar():
    data = {
        'RA'         : [248.048926969, 335.696461603],
        'DEC'        : [19.902625348, -3.311516291],
        'MASKNAME'   : ['PAL', 'PAL'],
        'SPECOBJID'  : ['', ''],
        'SPEC_Z'     : [0.0907, 0.0524],
        'SPEC_Z_ERR' : [0.0001, 0.0001],
        'ZQUALITY'   : [4, 4],
        'TELNAME'    : ['MMT', 'MMT'],
    }
    return ensure_specs_dtype(Table(data))


def read_6df(file_path):
    if not hasattr(file_path, 'read'):
        file_path = FitsTable(file_path)
    specs = file_path.read()['RAJ2000', 'DEJ2000', '_6dFGS', 'cz', 'e_cz', 'q_cz']

    # 3 = probably galaxy, 4 = definite galaxy, 6 = confirmed star
    specs = (Query('q_cz == 3') | Query('q_cz == 4') | Query('q_cz == 6')).filter(specs)
    specs['SPEC_Z'] = specs['cz'] / _SPEED_OF_LIGHT
    specs['SPEC_Z_ERR'] = specs['e_cz'] / _SPEED_OF_LIGHT
    del specs['cz'], specs['e_cz']
    specs.rename_column('_6dFGS', 'SPECOBJID')
    specs.rename_column('RAJ2000', 'RA')
    specs.rename_column('DEJ2000', 'DEC')
    specs.rename_column('q_cz', 'ZQUALITY')
    specs['TELNAME'] = '6dF'
    specs['MASKNAME'] = '6dF'
    specs['HELIO_CORR'] = True

    return ensure_specs_dtype(specs)


def read_2df(file_path):
    if not hasattr(file_path, 'read'):
        file_path = FitsTable(file_path)
    specs = file_path.read()['RAJ2000', 'DEJ2000', 'Name', 'z', 'q_z', 'n_z']

    # 3 = probably galaxy, 4 = definite galaxy, 6 = confirmed star
    specs = Query('q_z >= 3').filter(specs)
    specs.rename_column('Name', 'SPECOBJID')
    specs.rename_column('RAJ2000', 'RA')
    specs.rename_column('DEJ2000', 'DEC')
    specs.rename_column('q_z', 'ZQUALITY')
    specs.rename_column('z', 'SPEC_Z')
    specs.rename_column('n_z', 'EM_ABS')
    specs['SPEC_Z_ERR'] = 60 / _SPEED_OF_LIGHT
    specs['TELNAME'] = '2dF'
    specs['MASKNAME'] = '2dF'
    specs['HELIO_CORR'] = True

    return ensure_specs_dtype(specs)


def read_2dflens(file_path):
    if not hasattr(file_path, 'read'):
        file_path = FileObject(file_path, format='ascii.fast_commented_header')
    specs = file_path.read()['R.A.', 'Dec.', 'z', 'qual']

    specs.rename_column('R.A.', 'RA')
    specs.rename_column('Dec.', 'DEC')
    specs.rename_column('qual', 'ZQUALITY')
    specs.rename_column('z', 'SPEC_Z')
    specs['SPECOBJID'] = ['2dFLenS_{}'.format(i) for i in range(len(specs))]
    specs['SPEC_Z_ERR'] = 60 / _SPEED_OF_LIGHT
    specs['TELNAME'] = '2dFLen' # max # of char is 6
    specs['MASKNAME'] = '2dFLenS'
    specs['HELIO_CORR'] = True

    return ensure_specs_dtype(specs)


def read_ozdes(file_path):
    if not hasattr(file_path, 'read'):
        file_path = FitsTable(file_path)
    specs = file_path.read()['OzDES_ID', 'RA', 'DEC', 'z', 'flag']
    specs.rename_column('OzDES_ID', 'SPECOBJID')
    specs.rename_column('z', 'SPEC_Z')
    # flag 3 = probably galaxy, 4 = definite galaxy, 6 = confirmed star
    specs.rename_column('flag', 'ZQUALITY')
    specs['SPEC_Z_ERR'] = 60 / _SPEED_OF_LIGHT
    specs['TELNAME'] = 'OzDES'
    specs['MASKNAME'] = 'OzDES'
    specs['HELIO_CORR'] = True

    return ensure_specs_dtype(specs)


def read_misc(file_path):
    if not hasattr(file_path, 'read'):
        file_path = FitsTable(file_path)
    specs = file_path.read()
    sepcs = specs[np.isfinite(specs['SPEC_Z'])]
    return ensure_specs_dtype(specs)


def extract_sdss_spectra(sdss):
    specs = Query('SPEC_Z > -1.0').filter(sdss['RA', 'DEC', 'SPEC_Z', 'SPEC_Z_ERR', 'SPEC_Z_WARN', 'OBJID'])
    specs['ZQUALITY'] = np.where(specs['SPEC_Z_WARN'] == 0, 4, 1)
    del specs['SPEC_Z_WARN']
    specs.rename_column('OBJID', 'SPECOBJID')
    specs['TELNAME'] = 'SDSS'
    specs['MASKNAME'] = 'SDSS'
    specs['HELIO_CORR'] = True
    return ensure_specs_dtype(specs)


def extract_nsa_spectra(nsa):
    specs = nsa['RA', 'DEC', 'Z', 'ZSRC', 'NSAID']
    specs['TELNAME'] = 'NSA'
    specs['SPEC_Z_ERR'] = 20 / _SPEED_OF_LIGHT
    specs['ZQUALITY'] = 4
    specs['HELIO_CORR'] = True
    specs.rename_column('Z', 'SPEC_Z')
    specs.rename_column('ZSRC', 'MASKNAME')
    specs.rename_column('NSAID', 'SPECOBJID')
    return ensure_specs_dtype(specs)


class SpectraData(object):
    def __init__(self, spectra_dir_path=None, external_specs_dict=None):
        self.spectra_dir_path = spectra_dir_path
        self._external_specs_dict = external_specs_dict or {}

    def read(self, add_coord=True, before_time=None):
        all_specs = []

        all_specs.extend((globals()['read_{}'.format(k)](v) \
                for k, v in self._external_specs_dict.items()))

        if self.spectra_dir_path is not None:
            if before_time is not None and not isinstance(before_time, Time):
                before_time = Time(before_time)
            all_specs.extend([
                read_mmt(os.path.join(self.spectra_dir_path, 'MMT'), before_time=before_time),
                read_aat(os.path.join(self.spectra_dir_path, 'AAT'), before_time=before_time),
                read_aat_mz(os.path.join(self.spectra_dir_path, 'AAT'), before_time=before_time),
                read_wiyn(os.path.join(self.spectra_dir_path, 'WIYN')),
                read_imacs(os.path.join(self.spectra_dir_path, 'IMACS')),
            ])

        all_specs.extend([
            read_deimos(),
            read_palomar(),
        ])

        all_specs = vstack(all_specs, 'exact', 'error')

        return add_skycoord(all_specs) if add_coord else all_specs
