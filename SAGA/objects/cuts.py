"""
This file collects a set of common cuts we apply to objects.
See https://github.com/yymao/easyquery for further documentation.

In Jupyter, use ?? to see the definitions of the cuts

Examples
--------
>>> is_sat.filter(base_table)
"""

from easyquery import Query
import numpy as np

COLUMNS_USED = ['ZQUALITY', 'REMOVE', 'PHOTPTYPE', 'FIBERMAG_R', 'SPEC_Z',
                'RHOST_KPC', 'HOST_VHOST', 'SATS', 'OBJ_NSAID', 'HOST_NSAID',
                'SPEC_REPEAT', 'r_mag', 'ug', 'ug_err', 'gr', 'gr_err', 'ri', 'ri_err']

COLUMNS_USED2 = ['ZQUALITY', 'REMOVE', 'is_galaxy', 'SPEC_Z',
                'RHOST_KPC', 'HOST_VHOST', 'SATS', 'SPEC_REPEAT',
                'r_mag', 'ug', 'ug_err', 'gr', 'gr_err', 'ri', 'ri_err']

has_spec = Query('ZQUALITY >= 3')
is_clean = Query('REMOVE == -1')
is_clean2 = Query('REMOVE == 0')
is_galaxy = Query('PHOTPTYPE == 3')
is_galaxy2 = Query('is_galaxy')
fibermag_r_cut = Query('FIBERMAG_R <= 23.0')

faint_end_limit = Query('r_mag < 20.75')
sdss_limit = Query('r_mag < 17.77')

sat_vcut = Query('abs(SPEC_Z * 2.99792458e5 - HOST_VHOST) < 250.0')
sat_rcut = Query('RHOST_KPC < 300.0')

gr_cut = Query('gr-gr_err*2.0 < 0.85')
ri_cut = Query('ri-ri_err*2.0 < 0.55')
ug_cut = Query('(ug+ug_err*2.0) > (gr-gr_err*2.0)*1.5')
gri_cut = (gr_cut & ri_cut)
ugri_cut = (gri_cut & ug_cut)

is_sat = Query('SATS == 1')

is_high_z = Query('SPEC_Z >= 0.03')
is_low_z = Query('SPEC_Z >= 0.0038', 'SPEC_Z <= 0.015')

obj_is_host = Query('OBJ_NSAID==HOST_NSAID')

has_sdss_spec = Query((lambda c: np.fromiter(('SDSS' in i for i in c), np.bool, len(c)), 'SPEC_REPEAT'))
has_sdss_nsa_spec = Query((lambda c: np.fromiter((('SDSS' in i) or ('NSA' in i) for i in c), np.bool, len(c)), 'SPEC_REPEAT'))

basic_cut = is_clean & is_galaxy & fibermag_r_cut & faint_end_limit & sat_rcut
basic_cut2 = is_clean2 & is_galaxy2 & faint_end_limit & sat_rcut
