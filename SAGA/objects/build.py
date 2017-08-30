import numpy as np
import numexpr as ne
from astropy.coordinates import SkyCoord
from astropy.table import Table, join
from easyquery import Query
from . import cuts as C
from .manual_fixes import fixes_by_sdss_objid
from ..utils import join_table_by_coordinates, fill_values_by_query, get_empty_str_array, get_sdss_bands


def add_host_info(base, host, saga_names=None, overwrite_if_different_host=False):
    """
    Add host information to the base catalog (for a single host).
    `base` is modified in-place.

    Parameters
    ----------
    base : astropy.table.Table
    host : astropy.table.Row

    Returns
    -------
    base : astropy.table.Table
    """

    if ('HOST_NSAID' in base.columns and base['HOST_NSAID'][0] != host['NSAID']) and not overwrite_if_different_host:
        raise ValueError('Host info exists and differs from input host info.')

    base['HOST_NSAID'] = host['NSAID']
    base['HOST_RA'] = host['RA']
    base['HOST_DEC'] = host['Dec']
    base['HOST_DIST'] = host['distance']
    base['HOST_VHOST'] = host['vhelio']
    base['HOST_MK'] = host['M_K']
    base['HOST_MR'] = host['M_r']
    base['HOST_MG'] = host['M_g']

    host_sc = SkyCoord(host['RA'], host['Dec'], unit='deg')
    sep = SkyCoord(base['RA'], base['DEC'], unit="deg").separation(host_sc)
    base['RHOST_ARCM'] = sep.arcmin
    base['RHOST_KPC'] = np.sin(sep.radian) * (1000.0 * host['distance'])

    cols = ('HOST_SAGA_NAME', 'HOST_NGC_NAME')
    for col in cols:
        if col not in base.columns:
            base[col] = get_empty_str_array(len(base))

    if saga_names:
        idx = np.where(saga_names['NSA'] == host['NSAID'])[0]
        if len(idx) == 1:
            base['HOST_SAGA_NAME'] = saga_names['SAGA'][idx]
            base['HOST_NGC_NAME'] = saga_names['NGC'][idx]

    return base


def add_wise(base, wise, missing_value=-9999.0):
    """
    Add more photometric data to the base catalog (for a single host).
    `base` is modified in-place.

    Parameters
    ----------
    base : astropy.table.Table
    wise : astropy.table.Table

    Returns
    -------
    base : astropy.table.Table

    Notes
    -----
    Currently this function only adds wise data, but in the future this function
    may add more photometric data
    """
    cols_rename = {'w1_mag':'W1', 'w1_mag_err':'W1ERR', 'w2_mag':'W2', 'w2_mag_err':'W2ERR'}

    wise = wise[['has_wise_phot', 'objid'] + list(cols_rename.keys())]
    wise = wise[wise['has_wise_phot']]
    wise['OBJID'] = wise['objid'].astype(np.int64)
    del wise['objid']
    del wise['has_wise_phot']

    t = Table({'OBJID':base['OBJID'], 'index':np.arange(len(base))})
    t = join(t, wise, keys='OBJID')

    for k, v in cols_rename.items():
        if v not in base.columns:
            base[v] = missing_value
        t[k][np.isnan(t[k])] = missing_value
        base[v][t['index']] = t[k]

    return base


def set_remove_flag(base, objects_to_remove, objects_to_add):
    """
    Set remove flag in the base catalog (for a single host),
    using the remove list and other info existing in the base catalog.
    `base` is modified in-place.

    Parameters
    ----------
    base : astropy.table.Table
    objects_to_remove : astropy.table.Table
    objects_to_add : astropy.table.Table

    Returns
    -------
    base : astropy.table.Table
    """
    if 'REMOVE' not in base.columns:
        base['REMOVE'] = -1

    ids_to_remove = np.unique(objects_to_remove['SDSS ID'].data.compressed())
    fill_values_by_query(base, Query((lambda x: np.in1d(x, ids_to_remove), 'OBJID')), {'REMOVE': 1})
    del ids_to_remove

    fill_values_by_query(base, C.too_close_to_host, {'REMOVE': 1})

    q  = Query('BINNED1 == 0')
    q |= Query('SATURATED != 0')
    q |= Query('BAD_COUNTS_ERROR != 0')
    fill_values_by_query(base, q, {'REMOVE': 3})

    q  = Query('abs(PETRORAD_R - PETRORAD_G) > 40', 'r < 18')
    q |= Query('abs(PETRORAD_R - PETRORAD_I) > 40', 'r < 18')
    fill_values_by_query(base, q, {'REMOVE': 4})

    q  = Query('SB_EXP_R > 24', '(PETRORADERR_G + PETRORADERR_R + PETRORADERR_I)/3.0 == -1000.0')
    q |= Query('SB_EXP_R > 24', (lambda *x: np.median(x, axis=0) == -1000.0, 'PETRORADERR_G', 'PETRORADERR_R', 'PETRORADERR_I'), 'r < 18')
    fill_values_by_query(base, q, {'REMOVE': 5})

    q = Query((lambda *x: np.abs(np.median(x, axis=0)) > 0.5, 'g_err', 'r_err', 'i_err'))
    fill_values_by_query(base, q, {'REMOVE': 3})

    ids_to_add = np.unique(objects_to_add['SDSS ID'].data.compressed())
    fill_values_by_query(base, Query((lambda x: np.in1d(x, ids_to_add), 'OBJID')), {'REMOVE': -1})

    return base



def fix_photometry_with_nsa(base, nsa):
    """
    Use NSA catalog to remove shereded object.

    Parameters
    ----------
    base : astropy.table.Table
    nsa : astropy.table.Table

    Returns
    -------
    sdss : astropy.table.Table
    """

    if 'TELNAME' not in base.columns:
        base['TELNAME'] = get_empty_str_array(len(base), 6)
    if 'MASKNAME' not in base.columns:
        base['MASKNAME'] = get_empty_str_array(len(base))
    if 'ZQUALITY' not in base.columns:
        base['ZQUALITY'] = -1
    if 'SPEC_REPEAT' not in base.columns:
        base['SPEC_REPEAT'] = get_empty_str_array(len(base))
    if 'SPECOBJID' not in base.columns:
        base['SPECOBJID'] = get_empty_str_array(len(base), 48)
    if 'SPEC_HA_EW' not in base.columns:
        base['SPEC_HA_EW'] = -9999.0
    if 'SPEC_HA_EWERR' not in base.columns:
        base['SPEC_HA_EWERR'] = -9999.0
    if 'OBJ_NSAID' not in base.columns:
        base['OBJ_NSAID'] = -1

    host_sc = SkyCoord(base['HOST_RA'][0], base['HOST_DEC'][0], unit='deg')
    nsa = nsa[SkyCoord(nsa['RA'], nsa['DEC'], unit="deg").separation(host_sc).deg < 1.0]

    if len(nsa) == 0:
        return base

    for nsa_obj in nsa:

        values_for_ellipse_calculation = {
            'a': nsa_obj['PETROTH90'] * 2.0 / 3600.0,
            'b': nsa_obj['SERSIC_BA'] * nsa_obj['PETROTH90'] * 2.0 / 3600.0,
            's': np.sin(np.deg2rad(nsa_obj['SERSIC_PHI'] + 270.0)),
            'c': np.cos(np.deg2rad(nsa_obj['SERSIC_PHI'] + 270.0)),
            'nra': nsa_obj['RA'],
            'ndec': nsa_obj['DEC'],
            'RA': base['RA'],
            'DEC': base['DEC'],
        }

        r2_ellipse = ne.evaluate('(((RA-nra)*c - (DEC-ndec)*s)/a)**2.0 + (((RA-nra)*s + (DEC-ndec)*c)/b)**2.0',
                    local_dict=values_for_ellipse_calculation, global_dict={})

        closest_base_obj_index = r2_ellipse.argmin()
        assert r2_ellipse[closest_base_obj_index] < 1.0
        base['REMOVE'][r2_ellipse < 1.0] = 2

        del r2_ellipse, values_for_ellipse_calculation

        values_to_rewrite = {
            'REMOVE': -1,
            'ZQUALITY': 4,
            'TELNAME': 'NSA',
            'PHOTPTYPE': 3,
            'PHOT_SG': 'GALAXY',
            'RA': nsa_obj['RA'],
            'DEC': nsa_obj['DEC'],
            'SPEC_Z': nsa_obj['Z'],
            'SPEC_HA_EW': nsa_obj['HAEW'],
            'SPEC_HA_EWERR': nsa_obj['HAEWERR'],
            'SPEC_REPEAT': 'SDSS+NSA',
            'MASKNAME': nsa_obj['ZSRC'],
            'OBJ_NSAID': nsa_obj['NSAID'],
            'PETROR90_R': nsa_obj['PETROTH90'],
            'PETROR50_R': nsa_obj['PETROTH50'],
        }

        mag = 22.5 - 2.5*np.log10(nsa_obj['SERSICFLUX'])
        var = 1.0 / nsa_obj['SERSICFLUX_IVAR']
        tmp1 = 2.5 / (nsa_obj['SERSICFLUX'] * np.log(10.0))
        mag_err = np.sqrt(var * tmp1**2)     # IS THIS RIGHT??
        mag[np.isnan(mag)] = -9999.0
        mag_err[np.isnan(mag_err)] = -9999.0

        for i, b in enumerate(get_sdss_bands()):
            values_to_rewrite[b] = mag[i+2]
            values_to_rewrite['{}_err'.format(b)] = mag_err[i+2]
        values_to_rewrite['SB_EXP_R'] = mag[4] + 2.5 * np.log10(2.0*np.pi*nsa_obj['PETROTH50']**2.0 + 1.0e-20)

        for k, v in values_to_rewrite.items():
            base[k][closest_base_obj_index] = v

    return base



def add_spectra(base, spectra):
    """
    Add spectra to base catalog.
    `base` is modified in-place.

    Parameters
    ----------
    base : astropy.table.Table
    spectra : astropy.table.Table

    Returns
    -------
    base : astropy.table.Table
    """

    cols_to_copy = ('TELNAME', 'MASKNAME', 'ZQUALITY', 'SPEC_Z', 'SPEC_Z_ERR', 'SPECOBJID')

    if 'TELNAME' not in base.columns:
        base['TELNAME'] = get_empty_str_array(len(base), 6)
    if 'MASKNAME' not in base.columns:
        base['MASKNAME'] = get_empty_str_array(len(base))
    if 'ZQUALITY' not in base.columns:
        base['ZQUALITY'] = -1
    if 'SPEC_REPEAT' not in base.columns:
        base['SPEC_REPEAT'] = get_empty_str_array(len(base))
    if 'SPECOBJID' not in base.columns:
        base['SPECOBJID'] = get_empty_str_array(len(base), 48)


    host_sc = SkyCoord(base['HOST_RA'][0], base['HOST_DEC'][0], unit='deg')
    spectra = spectra[spectra['coord'].separation(host_sc).deg < 1.0]

    if len(spectra) == 0:
        return base

    for spec in spectra:
        sep = base['coord'].separation(spec['coord']).arcsec

        nearby_obj_indices = np.where((sep < 5.0) & (base['REMOVE']==-1))[0]
        assert len(nearby_obj_indices) > 0

        nearby_nsa_indices = nearby_obj_indices[base['OBJ_NSAID'][nearby_obj_indices] > -1]
        if len(nearby_nsa_indices) == 0:
            closest_obj_index = sep.argmin()
        else:
            assert len(nearby_nsa_indices) == 1
            closest_obj_index = nearby_nsa_indices[0]

        if spec['ZQUALITY'] > base['ZQUALITY'][closest_obj_index]:
            for col in cols_to_copy:
                base[col][closest_obj_index] = spec[col]
        else:
            base['SPEC_REPEAT'][closest_obj_index] = '+'.join(set(spec['SPEC_REPEAT'].split('+') + base['SPEC_REPEAT'][closest_obj_index].split('+')))

        base['REMOVE'][nearby_obj_indices] = 0
        base['REMOVE'][closest_obj_index] = -1

    return base


def find_satelites(base):
    """
    Add `SATS` column to the base catalog.
    `base` is modified in-place.

    Parameters
    ----------
    base : astropy.table.Table

    Returns
    -------
    base : astropy.table.Table
    """

    if 'SATS' not in base.columns:
        base['SATS'] = -1

    fill_values_by_query(base, C.is_galaxy & C.is_high_z, {'SATS':0})
    fill_values_by_query(base, C.is_galaxy & ~C.is_high_z, {'SATS':2})
    fill_values_by_query(base, C.is_galaxy & C.sat_rcut & C.sat_vcut, {'SATS':1})
    fill_values_by_query(base, C.obj_is_host, {'SATS':3})

    return base


def apply_manual_fixes(base):
    """
    Apply manual fixes to base catalog.
    `base` is modified in-place.

    Parameters
    ----------
    base : astropy.table.Table

    Returns
    -------
    base : astropy.table.Table
    """
    for objid, fixes in fixes_by_sdss_objid.items():
        fill_values_by_query(base, 'OBJID == {}'.format(objid), fixes)

    return base



def add_stellar_mass(base):
    """
    Calculate stellar mass based only on gi colors and redshift
    Based on GAMA data using Taylor et al (2011)
    Add to `base`, modified in-place.

    Parameters
    ----------
    base : astropy.table.Table

    Returns
    -------
    base : astropy.table.Table
    """
    #TODO: implement this
    return base
