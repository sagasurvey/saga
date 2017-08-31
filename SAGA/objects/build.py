import numpy as np
import numexpr as ne
import warnings
from astropy.coordinates import SkyCoord
from astropy.table import Table, join, vstack
from astropy.constants import c
from easyquery import Query
from . import cuts as C
from .manual_fixes import fixes_by_sdss_objid
from ..utils import (fill_values_by_query, get_empty_str_array, get_sdss_bands, add_skycoord)


__all__ = ['initialize_base_catalog', 'add_host_info', 'add_wise', 'set_remove_flag',
           'remove_shreds_with_nsa', 'remove_shreds_with_sdss', 'clean_repeat_spectra',
           'add_spectra', 'find_satellites', 'apply_manual_fixes', 'add_stellar_mass',
           'build_full_stack', 'wise_cols_used', 'nsa_cols_used']

wise_cols_used = ['has_wise_phot', 'objid', 'w1_mag', 'w1_mag_err', 'w2_mag', 'w2_mag_err']

nsa_cols_used = ['RA', 'DEC', 'PETROTH90', 'PETROTH50', 'SERSIC_BA', 'SERSIC_PHI',
                 'Z', 'HAEW', 'HAEWERR', 'ZSRC', 'NSAID', 'SERSICFLUX', 'SERSICFLUX_IVAR']


def initialize_base_catalog(base):

    base['coord'] = SkyCoord(base['RA'], base['DEC'], unit="deg")

    base['REMOVE'] = np.int16(-1)
    base['ZQUALITY'] = np.int16(-1)
    base['SATS'] = np.int16(-1)

    base['SPEC_HA_EW'] = np.float32(-9999.0)
    base['SPEC_HA_EWERR'] = np.float32(-9999.0)
    base['OBJ_NSAID'] = np.int32(-1)

    empty_str_arr = get_empty_str_array(len(base), 48)
    base['HOST_SAGA_NAME'] = empty_str_arr
    base['HOST_NGC_NAME'] = empty_str_arr
    base['MASKNAME'] = empty_str_arr
    base['SPECOBJID'] = empty_str_arr

    base['SPEC_REPEAT'] = get_empty_str_array(len(base), 96)
    base['TELNAME'] = get_empty_str_array(len(base), 6)

    fill_values_by_query(base, Query('SPEC_Z > -1.0'), {'TELNAME':'SDSS', 'MASKNAME':'SDSS', 'SPEC_REPEAT':'SDSS', 'ZQUALITY':4})
    fill_values_by_query(base, Query('SPEC_Z > -1.0', 'SPEC_Z_WARN != 0'), {'ZQUALITY':1})

    return base


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

    if ('HOST_NSAID' in base.colnames and base['HOST_NSAID'][0] != host['NSAID']) and not overwrite_if_different_host:
        raise ValueError('Host info exists and differs from input host info.')

    base['HOST_NSAID'] = np.int32(host['NSAID'])
    base['HOST_RA'] = np.float32(host['RA'])
    base['HOST_DEC'] = np.float32(host['Dec'])
    base['HOST_DIST'] = np.float32(host['distance'])
    base['HOST_VHOST'] = np.float32(host['vhelio'])
    base['HOST_MK'] = np.float32(host['M_K'])
    base['HOST_MR'] = np.float32(host['M_r'])
    base['HOST_MG'] = np.float32(host['M_g'])

    host_sc = SkyCoord(host['RA'], host['Dec'], unit='deg')
    sep = base['coord'].separation(host_sc)
    base['RHOST_ARCM'] = sep.arcmin.astype(np.float32)
    base['RHOST_KPC'] = (np.sin(sep.radian) * (1000.0 * host['distance'])).astype(np.float32)
    del sep

    if saga_names:
        idx = np.where(saga_names['NSA'] == host['NSAID'])[0]
        if len(idx) == 1:
            base['HOST_SAGA_NAME'] = saga_names['SAGA'][idx]
            base['HOST_NGC_NAME'] = saga_names['NGC'][idx]

    return base


def add_wise(base, wise, missing_value=9999.0):
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

    if set(wise.colnames).issuperset(set(wise_cols_used)):
        if len(wise.colnames) > len(wise_cols_used):
            wise = wise[wise_cols_used]
    else:
        raise KeyError('`wise` does not have all needed columns')

    wise = wise[wise['has_wise_phot']]
    wise['OBJID'] = wise['objid'].astype(np.int64)
    del wise['objid']
    del wise['has_wise_phot']

    t = Table({'OBJID':base['OBJID'], 'index':np.arange(len(base))})
    t = join(t, wise, keys='OBJID')
    del wise

    for k, v in cols_rename.items():
        if v not in base.colnames:
            base[v] = np.float32(missing_value)
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

    if objects_to_remove is not None:
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

    if objects_to_add is not None:
        ids_to_add = np.unique(objects_to_add['SDSS ID'].data.compressed())
        fill_values_by_query(base, Query((lambda x: np.in1d(x, ids_to_add), 'OBJID')), {'REMOVE': -1})

    return base



def remove_shreds_with_nsa(base, nsa):
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

    nsa_cols_used_this = set(nsa_cols_used)
    if 'coord' in nsa.colnames:
        nsa_cols_used_this.add('coord')

    if set(nsa.colnames).issuperset(nsa_cols_used_this):
        if len(nsa.colnames) > len(nsa_cols_used_this):
            nsa = nsa[list(nsa_cols_used_this)]
    else:
        raise KeyError('`nsa` does not have all needed columns')

    host_sc = SkyCoord(base['HOST_RA'][0], base['HOST_DEC'][0], unit='deg')
    nsa_sc = nsa['coord'] if 'coord' in nsa.colnames else SkyCoord(nsa['RA'], nsa['DEC'], unit="deg")
    nsa = nsa[nsa_sc.separation(host_sc).deg < 1.0]
    del nsa_sc

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
        if r2_ellipse[closest_base_obj_index] > 1.0:
            warnings.warn('No object around NSA {} ({}, {})'.format(nsa_obj['NSAID'], nsa_obj['RA'], nsa_obj['DEC']))
            continue
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


def remove_shreds_with_sdss(base):
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

    sdss_specs = Query('SPEC_Z > 0.05', 'PETRORADERR_R > 0', 'PETRORAD_R > 2.0*PETRORADERR_R', 'REMOVE==-1').filter(base)

    for i, spec in enumerate(sdss_specs):

        nearby_obj_indices = np.where(base['coord'].separation(spec['coord']).arcsec < 1.25 * spec['PETRORAD_R'])[0]
        nearby_obj_indices = nearby_obj_indices[nearby_obj_indices != i]

        if len(nearby_obj_indices) == 0:
            continue

        base['REMOVE'][nearby_obj_indices] = 4

        if len(nearby_obj_indices) > 25:
            warnings.warn('Too many shreds around SDSS ({}, {})'.format(spec['RA'], spec['DEC']))

    return base


def clean_repeat_spectra(spectra):
    """
    Clean all spectra to remove repeats.
    `spectra` is modified in-place.

    Parameters
    ----------
    spectra : astropy.table.Table

    Returns
    -------
    spectra : astropy.table.Table
    """
    spectra = add_skycoord(spectra)
    spec_repeat = get_empty_str_array(len(spectra), 96)
    not_done = np.ones(len(spectra), np.bool)

    for i, spec in enumerate(spectra):
        if not not_done[i]:
            continue

        # search nearby spectra in 3D
        nearby_mask = (np.abs(spectra['SPEC_Z'] - spec['SPEC_Z']) < 50.0/c.to('km/s').value)
        nearby_mask &= (spectra['coord'].separation(spec['coord']).arcsec < 20.0)
        nearby_mask &= not_done
        nearby_mask = np.where(nearby_mask)[0]
        assert len(nearby_mask) >= 1

        not_done[nearby_mask] = False

        best_spec_idx = nearby_mask[spectra['ZQUALITY'][nearby_mask].argmax()]
        spec_repeat[best_spec_idx] = '+'.join(set(spectra['TELNAME'][nearby_mask]))

    del not_done
    spectra['SPEC_REPEAT'] = spec_repeat
    spectra = spectra[spectra['SPEC_REPEAT'] != '']

    return spectra


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

    to_remove_coord = False
    if 'coord' not in spectra.colnames:
        spectra = add_skycoord(spectra)
        to_remove_coord = True # because we cannot modify spectra

    host_sc = SkyCoord(base['HOST_RA'][0], base['HOST_DEC'][0], unit='deg')
    spectra = spectra[spectra['coord'].separation(host_sc).deg < 1.0]

    if len(spectra) == 0:
        if to_remove_coord:
            del spectra['coord']
        return base

    spectra = clean_repeat_spectra(spectra)

    for spec in spectra:
        sep = base['coord'].separation(spec['coord']).arcsec

        nearby_obj_indices = np.where(sep < 20.0)[0]
        if len(nearby_obj_indices) == 0:
            warnings.warn('No object within 20 arcsec of {} spec ({}, {})'.format(spec['TELNAME'], spec['RA'], spec['DEC']))
            continue

        nearby_obj = base[nearby_obj_indices]

        nearby_mask = (nearby_obj['REMOVE'] == -1)
        nearby_clean_indices = nearby_obj_indices[nearby_mask]

        nearby_mask &= (nearby_obj['ZQUALITY'] == 4)
        nearby_mask &= (np.abs(nearby_obj['SPEC_Z'] - spec['SPEC_Z']) < 50.0/c.to('km/s').value)
        nearby_has_spec_indices = nearby_obj_indices[nearby_mask]

        nearby_mask &= (nearby_obj['OBJ_NSAID'] > -1)
        nearby_nsa_indices = nearby_obj_indices[nearby_mask]

        del nearby_obj, nearby_mask, nearby_obj_indices

        if len(nearby_nsa_indices) > 0:
            closest_obj_index = nearby_nsa_indices[sep[nearby_nsa_indices].argmin()]
        elif len(nearby_has_spec_indices) > 0:
            closest_obj_index = nearby_has_spec_indices[sep[nearby_has_spec_indices].argmin()]
        elif len(nearby_clean_indices) > 0:
            closest_obj_index = nearby_clean_indices[sep[nearby_clean_indices].argmin()]
        else:
            closest_obj_index = sep.argmin()

        if spec['ZQUALITY'] > base['ZQUALITY'][closest_obj_index] or \
                (spec['ZQUALITY'] == base['ZQUALITY'][closest_obj_index] and spec['TELNAME'] == 'MMT'):
            for col in ('TELNAME', 'MASKNAME', 'ZQUALITY', 'SPEC_Z', 'SPEC_Z_ERR', 'SPECOBJID'):
                base[col][closest_obj_index] = spec[col]

        spec_repeat = set(spec['SPEC_REPEAT'].split('+'))
        for i in nearby_has_spec_indices:
            if base['SPEC_REPEAT'][i]:
                spec_repeat.update(base['SPEC_REPEAT'][i].split('+'))
        base['SPEC_REPEAT'][closest_obj_index] = '+'.join(spec_repeat)

        if len(nearby_clean_indices) > 0:
            base['REMOVE'][nearby_clean_indices] = 0
            base['REMOVE'][closest_obj_index] = -1

    if to_remove_coord:
        del spectra['coord']

    return base


def find_satellites(base):
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
    basic_cuts = C.is_galaxy & C.has_spec & C.is_clean
    fill_values_by_query(base, basic_cuts & C.is_high_z, {'SATS':0})
    fill_values_by_query(base, basic_cuts & ~C.is_high_z, {'SATS':2})
    fill_values_by_query(base, basic_cuts & C.sat_rcut & C.sat_vcut, {'SATS':1})
    fill_values_by_query(base, basic_cuts & C.obj_is_host, {'SATS':3})

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


def build_full_stack(base, host, saga_names=None, wise=None, nsa=None,
                     objects_to_remove=None, objects_to_add=None, spectra=None):

    base = initialize_base_catalog(base)
    base = add_host_info(base, host, saga_names)
    if wise is not None:
        base = add_wise(base, wise)
    base = set_remove_flag(base, objects_to_remove, objects_to_add)
    if nsa is not None:
        base = remove_shreds_with_nsa(base, nsa)
    base = remove_shreds_with_sdss(base)
    if spectra:
        base = add_spectra(base, spectra)
    base = apply_manual_fixes(base)
    base = find_satellites(base)
    base = add_stellar_mass(base)

    return base
