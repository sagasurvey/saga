import numpy as np
from easyquery import Query

from .common import SPEED_OF_LIGHT, ensure_specs_dtype
from ..database import FitsTable, FileObject

__all__ = ['read_gama', 'read_2df', 'read_2dflens', 'read_6df', 'read_ozdes',
           'read_ukst', 'read_lcrs', 'read_slackers']


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
    specs['SPEC_Z_ERR'] = 60 / SPEED_OF_LIGHT
    specs['TELNAME'] = 'GAMA'
    specs['ZQUALITY'][Query('ZQUALITY > 4').mask(specs)] = 4
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
    specs['SPEC_Z_ERR'] = 60 / SPEED_OF_LIGHT
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
    specs['SPEC_Z_ERR'] = 60 / SPEED_OF_LIGHT
    specs['TELNAME'] = '2dFLen' # max # of char is 6
    specs['MASKNAME'] = '2dFLenS'
    specs['HELIO_CORR'] = True

    return ensure_specs_dtype(specs)


def read_6df(file_path):
    if not hasattr(file_path, 'read'):
        file_path = FitsTable(file_path)
    specs = file_path.read()['RAJ2000', 'DEJ2000', '_6dFGS', 'cz', 'e_cz', 'q_cz']

    # 3 = probably galaxy, 4 = definite galaxy, 6 = confirmed star
    specs = (Query('q_cz == 3') | Query('q_cz == 4') | Query('q_cz == 6')).filter(specs)
    specs['SPEC_Z'] = specs['cz'] / SPEED_OF_LIGHT
    specs['SPEC_Z_ERR'] = specs['e_cz'] / SPEED_OF_LIGHT
    del specs['cz'], specs['e_cz']
    specs.rename_column('_6dFGS', 'SPECOBJID')
    specs.rename_column('RAJ2000', 'RA')
    specs.rename_column('DEJ2000', 'DEC')
    specs.rename_column('q_cz', 'ZQUALITY')
    specs['TELNAME'] = '6dF'
    specs['MASKNAME'] = '6dF'
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
    specs['SPEC_Z_ERR'] = 60 / SPEED_OF_LIGHT
    specs['TELNAME'] = 'OzDES'
    specs['MASKNAME'] = 'OzDES'
    specs['HELIO_CORR'] = True

    return ensure_specs_dtype(specs)


def read_ukst(file_path):
    """REFERENCE:  DURHAM UKST
    http://adsabs.harvard.edu/abs/1998MNRAS.300..417R
    """
    if not hasattr(file_path, 'read'):
        file_path = FitsTable(file_path)
    specs = file_path.read()['_RAJ2000', '_DEJ2000', 'DUGRS', 'RV']
    specs = specs[specs['RV'] != np.iinfo(np.int32).min]

    specs.rename_column('_RAJ2000', 'RA')
    specs.rename_column('_DEJ2000', 'DEC')
    specs.rename_column('DUGRS', 'SPECOBJID')

    specs['SPEC_Z'] = specs['RV'].astype(np.float64) / SPEED_OF_LIGHT
    del specs['RV']
    specs['SPEC_Z_ERR'] = 100 / SPEED_OF_LIGHT
    specs['ZQUALITY'] = 4
    specs['TELNAME'] = 'UKST'
    specs['HELIO_CORR'] = True

    return ensure_specs_dtype(specs)


def read_lcrs(file_path):
    """REFERENCE:  LCRS
    http://vizier.cfa.harvard.edu/viz-bin/VizieR-2
    http://adsabs.harvard.edu/abs/1996ApJ...470..172S
    """
    if not hasattr(file_path, 'read'):
        file_path = FitsTable(file_path)
    specs = file_path.read()['_RAJ2000', '_DEJ2000', 'p', 'cz', 'e_cz']
    specs = specs[specs['cz'] != np.iinfo(np.int32).min]

    specs.rename_column('_RAJ2000', 'RA')
    specs.rename_column('_DEJ2000', 'DEC')
    specs.rename_column('p', 'SPECOBJID')

    specs['SPEC_Z'] = specs['cz'].astype(np.float64) / SPEED_OF_LIGHT
    specs['SPEC_Z_ERR'] = specs['e_cz'].astype(np.float64) / SPEED_OF_LIGHT
    del specs['cz'], specs['e_cz']

    specs['ZQUALITY'] = 4
    specs['TELNAME'] = 'LCRS'
    specs['HELIO_CORR'] = True
    return ensure_specs_dtype(specs)


def read_slackers(file_path):
    """REFERENCE:  SLACKERS
    FILE WAS FOUND IN MG's EMAIL.   UNPUBLISHED SURVEY FROM LAS CAMPANAS 2.5M
    """
    if not hasattr(file_path, 'read'):
        file_path = FitsTable(file_path)
    specs = file_path.read()['OBJ_RA', 'OBJ_DEC', 'ID', 'WFCCD_ZQUAL', 'WFCCD_Z', 'WFCCD_ZERR']
    specs = Query('WFCCD_ZQUAL > 2').filter(specs)

    specs.rename_column('OBJ_RA', 'RA')
    specs.rename_column('OBJ_DEC', 'DEC')
    specs.rename_column('ID', 'SPECOBJID')
    specs.rename_column('WFCCD_ZQUAL', 'ZQUALITY')
    specs.rename_column('WFCCD_Z', 'SPEC_Z')
    specs.rename_column('WFCCD_ZERR', 'SPEC_Z_ERR')

    specs['TELNAME'] = 'slack'
    specs['HELIO_CORR'] = True
    return ensure_specs_dtype(specs)
