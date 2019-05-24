import numpy as np
from easyquery import Query
from ..utils import fill_values_by_query
from .common import SPEED_OF_LIGHT, ensure_specs_dtype
from .manual_fixes import fixes_sdss_spec_by_objid

__all__ = ['extract_sdss_spectra', 'extract_nsa_spectra']


def extract_sdss_spectra(sdss):
    specs = Query('SPEC_Z > -1.0').filter(sdss['RA', 'DEC', 'SPEC_Z', 'SPEC_Z_ERR', 'SPEC_Z_WARN', 'OBJID'])
    if not len(specs):
        return
    specs['ZQUALITY'] = np.where(specs['SPEC_Z_WARN'] == 0, 4, 1)
    del specs['SPEC_Z_WARN']

    for objid, fixes in fixes_sdss_spec_by_objid.items():
        fill_values_by_query(specs, 'OBJID == {}'.format(objid), fixes)

    specs.rename_column('OBJID', 'SPECOBJID')
    specs['TELNAME'] = 'SDSS'
    specs['MASKNAME'] = 'SDSS'
    specs['HELIO_CORR'] = True
    return ensure_specs_dtype(specs)


def extract_nsa_spectra(nsa):
    specs = nsa['RA', 'DEC', 'Z', 'ZSRC', 'NSAID']
    specs['TELNAME'] = 'NSA'
    specs['SPEC_Z_ERR'] = 20 / SPEED_OF_LIGHT
    specs['ZQUALITY'] = 4
    specs['HELIO_CORR'] = True
    specs.rename_column('Z', 'SPEC_Z')
    specs.rename_column('ZSRC', 'MASKNAME')
    specs.rename_column('NSAID', 'SPECOBJID')
    return ensure_specs_dtype(specs)
