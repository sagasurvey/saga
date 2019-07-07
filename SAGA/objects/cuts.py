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

def _vectorize(mask_func):
    return lambda c: np.fromiter(map(mask_func, c), np.bool, len(c))

COLUMNS_USED = ['ZQUALITY', 'REMOVE', 'PHOTPTYPE', 'FIBERMAG_R', 'SPEC_Z',
                'RHOST_KPC', 'HOST_VHOST', 'SATS', 'OBJ_NSAID', 'HOST_NSAID',
                'SPEC_REPEAT', 'r_mag', 'ug', 'ug_err', 'gr', 'gr_err', 'ri', 'ri_err']

COLUMNS_USED2 = ['ZQUALITY', 'REMOVE', 'is_galaxy', 'SPEC_Z',
                 'RHOST_KPC', 'HOST_VHOST', 'SATS', 'SPEC_REPEAT',
                 'r_mag', 'i_mag',
                 'ug', 'ug_err', 'gr', 'gr_err', 'ri', 'ri_err', 'rz', 'rz_err', 'sb_r']

has_spec = Query('ZQUALITY >= 3')
is_clean = Query('REMOVE == -1')
is_clean2 = Query('REMOVE == 0')
is_galaxy = Query('PHOTPTYPE == 3')
is_galaxy2 = Query('is_galaxy')
fibermag_r_cut = Query('FIBERMAG_R <= 23.0')

faint_end_limit = Query('r_mag < 20.75')
sdss_limit = Query('r_mag < 17.77')
lowz_mag_cut = Query('r_mag > 18', 'r_mag < 20')

sat_vcut = Query('abs(SPEC_Z * 2.99792458e5 - HOST_VHOST) < 250.0')
sat_rcut = Query('RHOST_KPC < 300.0')

gr_cut = Query('gr-abs(gr_err)*2.0 < 0.85')
ri_cut = Query('ri-abs(ri_err)*2.0 < 0.55')
rz_cut = Query('rz-abs(rz_err)*2.0 < 1.0')
ug_cut = Query('(ug+abs(ug_err)*2.0) > (gr-abs(gr_err)*2.0)*1.5')
gri_cut = (gr_cut & ri_cut)
ugri_cut = (gri_cut & ug_cut)
valid_i_mag = Query('i_mag > 0', 'i_mag < 30')
grz_cut = (gr_cut & rz_cut)
gri_or_grz_cut = (gr_cut & ((valid_i_mag & ri_cut) | (~valid_i_mag & rz_cut)))

high_priority_cuts = Query(
    'gr - abs(gr_err) < (1.55 - 0.05*r_mag)',
    'sb_r > 0.6 * (r_mag - abs(r_err)) + 10.1',
)

is_sat = Query('SATS == 1')

is_high_z = Query('SPEC_Z >= 0.03')
is_low_z = Query('SPEC_Z >= 0.0038', 'SPEC_Z <= 0.015')

obj_is_host = Query('OBJ_NSAID == HOST_NSAID')

basic_cut = is_clean & is_galaxy & fibermag_r_cut & faint_end_limit & sat_rcut
basic_cut2 = is_clean2 & is_galaxy2 & faint_end_limit & sat_rcut
basic_cut_lowz = is_clean2 & is_galaxy2 & lowz_mag_cut & gri_cut

has_sdss_spec = Query((_vectorize(lambda x: 'SDSS' in x), 'SPEC_REPEAT'))
has_nsa_spec = Query((_vectorize(lambda x: 'NSA' in x), 'SPEC_REPEAT'))
has_sdss_nsa_spec = Query((_vectorize(lambda x: 'SDSS' in x or 'NSA' in x), 'SPEC_REPEAT'))

has_aat_spec = Query((_vectorize(lambda x: 'AAT' in x), 'SPEC_REPEAT'))
has_mmt_spec = Query((_vectorize(lambda x: 'MMT' in x), 'SPEC_REPEAT'))

_known_telnames = {'2dF', '6dF', 'SDSS', 'NSA', 'GAMA', 'OzDES', '2dFLen', 'WiggleZ' ,'UKST', 'LCRS', 'slack'}
has_our_specs_only = Query((_vectorize(lambda x: x and set(x.split('+')).isdisjoint(_known_telnames)), 'SPEC_REPEAT'))
has_our_specs = Query((_vectorize(lambda x: x and not set(x.split('+')).issubset(_known_telnames)), 'SPEC_REPEAT'))
