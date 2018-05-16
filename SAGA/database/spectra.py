import os
import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from easyquery import Query
from .core import FitsTable
from ..utils import get_empty_str_array, add_skycoord


__all__ = ['read_gama', 'read_mmt', 'read_aat', 'read_aat_mz', 'read_imacs',
           'read_wiyn', 'read_deimos', 'read_palomar', 'read_2dF', 'read_6dF',
           'extract_sdss_spectra', 'extract_nsa_spectra', 'SpectraData']


def ensure_dtype(spectra):

    dtype = {'RA':'<f8', 'DEC':'<f8', 'SPEC_Z':'<f4', 'SPEC_Z_ERR': '<f4',
             'ZQUALITY': '<i2', 'SPECOBJID':'<U48', 'MASKNAME': '<U48', 'TELNAME': '<U6'}

    for c, t in dtype.items():
        if spectra[c].dtype.str != t:
            spectra[c] = spectra[c].astype(t)

    return spectra


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
    specs['SPEC_Z_ERR'] = np.float32(60)
    specs['TELNAME'] = get_empty_str_array(len(specs), 6, 'GAMA')
    specs['ZQUALITY'][Query('ZQUALITY > 4').mask(specs)] = 4

    return ensure_dtype(specs)


def read_generic_spectra(dir_path, extension, telname, usecols, n_cols_total,
                         cuts=None, postprocess=None, midprocess=None, **kwargs):

    names = [usecols.get(i+1, '_{}'.format(i)) for i in range(n_cols_total)]
    exclude_names = [n for n in names if n.startswith('_')]

    output = []

    for f in os.listdir(dir_path):
        if not f.endswith(extension):
            continue
        this = Table.read(os.path.join(dir_path, f), format='ascii.fast_no_header',
                          guess=False, names=names, exclude_names=exclude_names, **kwargs)
        this = Query(cuts).filter(this)
        if 'MASKNAME' not in this.colnames:
            this['MASKNAME'] = f
        if midprocess:
            this = midprocess(this)
        output.append(this)

    output = vstack(output, 'exact', 'error')
    output['TELNAME'] = telname
    if postprocess:
        output = postprocess(output)

    return ensure_dtype(output)


def read_mmt(dir_path):

    usecols = {2:'RA', 3:'DEC', 4:'mag', 5:'SPEC_Z', 6:'SPEC_Z_ERR',
               7:'ZQUALITY', 8:'SPECOBJID'}
    cuts = Query('mag != 0', 'ZQUALITY >= 1')
    def postprocess(t):
        del t['mag']
        t['RA'] *= 15.0
        return t

    return read_generic_spectra(dir_path, '.zlog', 'MMT', usecols, 11, cuts, postprocess)


def read_aat(dir_path):

    usecols = {2:'RA', 3:'DEC', 5:'SPEC_Z', 7:'ZQUALITY', 8:'SPECOBJID'}
    cuts = Query('ZQUALITY >= 1')
    def postprocess(t):
        t['SPEC_Z_ERR'] = 10.0
        return t

    return read_generic_spectra(dir_path, '.zlog', 'AAT', usecols, 11, cuts, postprocess)


def read_aat_mz(dir_path):

    usecols = {3:'RA', 4:'DEC', 13:'SPEC_Z', 14:'ZQUALITY', 1:'SPECOBJID'}
    cuts = Query('ZQUALITY >= 1')
    def postprocess(t):
        t['RA'] *= 180.0/np.pi
        t['DEC'] *= 180.0/np.pi
        t['SPEC_Z_ERR'] = 10.0
        return t
    return read_generic_spectra(dir_path, '.mz', 'AAT', usecols, 15, cuts, postprocess, delimiter=',')


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

    return ensure_dtype(output)


def read_deimos():
    data = {
        'RA'         : [247.825839103498, 221.86742, 150.12470],
        'DEC'        : [20.210825313885, -0.28144459, 32.561687],
        'MASKNAME'   : ['deimos2014', 'deimos2016-DN1', 'deimos2016-MD1'],
        'SPECOBJID'  : [0, 0, 0],
        'SPEC_Z'     : [2375.0/3.0e5, 0.056, 1.08],
        'SPEC_Z_ERR' : [0.001, 0.001, 0.001],
        'ZQUALITY'   : [4, 4, 4],
        'TELNAME'    : ['DEIMOS', 'DEIMOS', 'DEIMOS'],
    }
    return ensure_dtype(Table(data))


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
    return ensure_dtype(Table(data))


def read_6dF(file_path):
    if not hasattr(file_path, 'read'):
        file_path = FitsTable(file_path)
    specs = file_path.read()['RAJ2000', 'DEJ2000', '_6dFGS', 'cz', 'e_cz', 'q_cz']

    # 3 = probably galaxy, 4 = definite galaxy, 6 = confirmed star
    specs = (Query('q_cz == 3') | Query('q_cz == 4') | Query('q_cz == 6')).filter(specs)
    specs.rename_column('_6dFGS', 'SPECOBJID')
    specs.rename_column('RAJ2000', 'RA')
    specs.rename_column('DEJ2000', 'DEC')
    specs.rename_column('q_cz', 'ZQUALITY')
    specs.rename_column('cz', 'SPEC_Z')
    specs.rename_column('e_cz','SPEC_Z_ERR')
    specs['TELNAME'] = '6dF'
    specs['MASKNAME'] = '6dF'

    return ensure_dtype(specs)


def read_2dF(file_path):
    if not hasattr(file_path, 'read'):
        file_path = FitsTable(file_path)
    specs = file_path.read()['RAJ2000', 'DEJ2000', 'Name', 'z', 'q_z']

    # 3 = probably galaxy, 4 = definite galaxy, 6 = confirmed star
    specs = Query('q_z >= 3').filter(specs)
    specs.rename_column('Name', 'SPECOBJID')
    specs.rename_column('RAJ2000', 'RA')
    specs.rename_column('DEJ2000', 'DEC')
    specs.rename_column('q_z', 'ZQUALITY')
    specs.rename_column('z', 'SPEC_Z')
    specs['SPEC_Z_ERR'] = 60
    specs['TELNAME'] = '2dF'
    specs['MASKNAME'] = '2dF'

    return ensure_dtype(specs)


def extract_sdss_spectra(sdss):
    specs = Query('SPEC_Z > -1.0').filter(sdss['RA', 'DEC', 'SPEC_Z', 'SPEC_Z_ERR', 'SPEC_Z_WARN'])
    specs['ZQUALITY'] = np.where(specs['SPEC_Z_WARN'] == 0, 4, 1)
    specs['TELNAME'] = 'SDSS'
    specs['MASKNAME'] = 'SDSS'
    specs['SPECOBJID'] = ''
    del specs['SPEC_Z_WARN']
    return ensure_dtype(specs)


def extract_nsa_spectra(nsa):
    specs = nsa['RA', 'DEC', 'Z', 'ZSRC', 'NSAID']
    specs['TELNAME'] = 'NSA'
    specs['SPEC_Z_ERR'] = 0
    specs['ZQUALITY'] = 4
    specs.rename_column('Z', 'SPEC_Z')
    specs.rename_column('ZSRC', 'MASKNAME')
    specs.rename_column('NSAID', 'SPECOBJID')
    return ensure_dtype(specs)


class SpectraData(object):
    def __init__(self, spectra_dir_path=None, gama_file=None, twodf_file=None, sixdf_file=None):
        self.spectra_dir_path = spectra_dir_path
        self.gama_file = gama_file
        self.twodf_file = twodf_file
        self.sixdf_file = sixdf_file

    def read(self, add_coord=True):
        all_specs = []
        if self.gama_file is not None:
            all_specs.append(read_gama(self.gama_file))

        if self.twodf_file is not None:
            all_specs.append(read_2dF(self.twodf_file))

        if self.sixdf_file is not None:
            all_specs.append(read_6dF(self.sixdf_file))

        if self.spectra_dir_path is not None:
            all_specs.extend([
                read_mmt(os.path.join(self.spectra_dir_path, 'MMT')),
                read_aat(os.path.join(self.spectra_dir_path, 'AAT')),
                read_aat_mz(os.path.join(self.spectra_dir_path, 'AAT')),
                read_wiyn(os.path.join(self.spectra_dir_path, 'WIYN')),
                read_imacs(os.path.join(self.spectra_dir_path, 'IMACS')),
            ])

        all_specs.extend([
            read_deimos(),
            read_palomar(),
        ])

        all_specs = vstack(all_specs, 'exact', 'error')

        return add_skycoord(all_specs) if add_coord else all_specs
