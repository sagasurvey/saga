import logging
import numpy as np
import numexpr as ne
from astropy.coordinates import SkyCoord
from astropy.table import Table, join
import astropy.constants
from easyquery import Query
from . import cuts as C
from .manual_fixes import fixes_by_sdss_objid
from ..utils import (fill_values_by_query, get_empty_str_array, get_sdss_bands, add_skycoord)


__all__ = ['initialize_base_catalog', 'add_host_info', 'add_wise',
           'remove_human_inspected', 'remove_too_close_to_host',
           'remove_shreds_with_nsa', 'remove_shreds_with_highz',
           'remove_bad_photometry', 'recover_whitelisted_objects',
           'apply_manual_fixes', 'clean_repeat_spectra', 'add_cleaned_spectra',
           'add_spectra', 'clean_sdss_spectra', 'find_satellites',
           'add_stellar_mass', 'build_full_stack',
           'WISE_COLS_USED', 'NSA_COLS_USED']


_WISE_COLS_USED = ['has_wise_phot', 'objid', 'w1_mag', 'w1_mag_err', 'w2_mag', 'w2_mag_err']
WISE_COLS_USED = list(_WISE_COLS_USED)

_NSA_COLS_USED = ['RA', 'DEC', 'PETROTH90', 'PETROTH50', 'SERSIC_BA', 'SERSIC_PHI',
                 'Z', 'HAEW', 'HAEWERR', 'ZSRC', 'NSAID', 'SERSICFLUX', 'SERSICFLUX_IVAR']
NSA_COLS_USED = list(_NSA_COLS_USED)


def _join_spec_repeat(*repeats):
    out = set()
    for repeat in repeats:
        if repeat:
            out.update(repeat.split('+'))
    return '+'.join(out)


def _get_unique_objids(objid_col):
    try:
        objid_col = objid_col.compressed()
    except AttributeError:
        pass
    return np.unique(np.asarray(objid_col, dtype=np.int64))


def _get_spec_search_radius(spec_z):
    return 20.0 if spec_z < 0.2 else 10.0


_spec_search_dz = 50.0/astropy.constants.c.to('km/s').value # pylint: disable=E1101


def initialize_base_catalog(base):
    """
    Initialize the base catalog with empty columns.
    Also fill in some columns for objects that already have SDSS specs.

    `base` is modified in-place.

    Parameters
    ----------
    base : astropy.table.Table

    Returns
    -------
    base : astropy.table.Table
    """
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
    base['SPEC_REPEAT'] = empty_str_arr

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
    saga_names : astropy.table.Table or None, optional
    overwrite_if_different_host : bool, optional

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
    Add wise photometric data to the base catalog.

    `base` is modified in-place.

    Parameters
    ----------
    base : astropy.table.Table
    wise : astropy.table.Table
    missing_value : float

    Returns
    -------
    base : astropy.table.Table

    Notes
    -----
    Currently this function only adds wise data, but in the future this function
    may add more photometric data
    """
    cols_rename = {'w1_mag':'W1', 'w1_mag_err':'W1ERR', 'w2_mag':'W2', 'w2_mag_err':'W2ERR'}

    if set(wise.colnames).issuperset(set(_WISE_COLS_USED)):
        if len(wise.colnames) > len(_WISE_COLS_USED):
            wise = wise[_WISE_COLS_USED]
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


def remove_human_inspected(base, objects_to_remove):
    """
    Use the "remove list" to set REMOVE to 1 in the base catalog.
    This is step (1) of object removal.

    `base` is modified in-place.

    Parameters
    ----------
    base : astropy.table.Table
    objects_to_remove : astropy.table.Table

    Returns
    -------
    base : astropy.table.Table
    """
    ids_to_remove = _get_unique_objids(objects_to_remove['SDSS ID'])
    fill_values_by_query(base, Query((lambda x: np.in1d(x, ids_to_remove), 'OBJID')), {'REMOVE': 1})
    return base


def remove_too_close_to_host(base):
    """
    Remove objects that are too close to the host. Set REMOVE to 1.
    This is step (2) of object removal.

    This will unfortunately also remove the host itself!
    But the host will be added back during `remove_shreds_with_nsa`

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
    fill_values_by_query(base, Query('RHOST_KPC < 10.0'), {'REMOVE': 1})
    return base


def remove_shreds_with_nsa(base, nsa):
    """
    Use NSA catalog to remove shereded objects. Set REMOVE to 2.
    This is step (3) of object removal.

    For each NSA galaxy, find in the base catalog all objects within the ellipse,
    mark them as removed except for the closest one.

    `base` is modified in-place.

    Parameters
    ----------
    base : astropy.table.Table
    nsa : astropy.table.Table

    Returns
    -------
    sdss : astropy.table.Table
    """
    nsa_cols_used_this = set(_NSA_COLS_USED)
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

        not_star_indices = np.where(base['PHOTPTYPE'] != 6)[0]

        ellipse_calculation = dict()
        ellipse_calculation['a'] = nsa_obj['PETROTH90'] * 2.0 / 3600.0
        ellipse_calculation['b'] = ellipse_calculation['a'] * nsa_obj['SERSIC_BA']
        ellipse_calculation['th'] = np.deg2rad(nsa_obj['SERSIC_PHI'] + 270.0)
        ellipse_calculation['s'] = np.sin(ellipse_calculation['th'])
        ellipse_calculation['c'] = np.cos(ellipse_calculation['th'])
        ellipse_calculation['x'] = base['RA'][not_star_indices] - nsa_obj['RA']
        ellipse_calculation['y'] = base['DEC'][not_star_indices] - nsa_obj['DEC']

        r2_ellipse = ne.evaluate('((x*c - y*s)/a)**2.0 + ((x*s + y*c)/b)**2.0',
                                 local_dict=ellipse_calculation, global_dict={})

        closest_base_obj_index = r2_ellipse.argmin() if len(r2_ellipse) else None
        if closest_base_obj_index is None or r2_ellipse[closest_base_obj_index] > 1.0:
            logging.warning('In SAGA.objects.build.remove_shreds_with_nsa()\n No object within the radius of NSA {} ({}, {})'.format(nsa_obj['NSAID'], nsa_obj['RA'], nsa_obj['DEC']))
            continue
        closest_base_obj_index = not_star_indices[closest_base_obj_index]
        base['REMOVE'][not_star_indices[r2_ellipse < 1.0]] = 2

        del r2_ellipse, not_star_indices, ellipse_calculation

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

        invalid_mag = (nsa_obj['SERSICFLUX'] <= 0)
        nsa_sersic_flux = np.array(nsa_obj['SERSICFLUX'])
        nsa_sersic_flux[invalid_mag] = 1.0

        mag = 22.5 - 2.5 * np.log10(nsa_sersic_flux)
        mag_err = np.fabs((2.5/np.log(10.0))/nsa_sersic_flux/np.sqrt(nsa_obj['SERSICFLUX_IVAR']))
        mag[invalid_mag] = -9999.0
        mag_err[invalid_mag] = -9999.0

        for i, b in enumerate(get_sdss_bands()):
            values_to_rewrite[b] = mag[i+2]
            values_to_rewrite['{}_err'.format(b)] = mag_err[i+2]
        values_to_rewrite['SB_EXP_R'] = mag[4] + 2.5 * np.log10(2.0*np.pi*nsa_obj['PETROTH50']**2.0 + 1.0e-20)

        for k, v in values_to_rewrite.items():
            base[k][closest_base_obj_index] = v

    return base


def remove_bad_photometry(base):
    """
    Remove objects that have bad SDSS photometry. Set REMOVE to 3, 4, or 5.
    This is step (4) of object removal.

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
    has_nsa = Query('OBJ_NSAID > -1')

    q  = Query('BINNED1 == 0')
    q |= Query('SATURATED != 0')
    q |= Query('BAD_COUNTS_ERROR != 0')
    fill_values_by_query(base, q & (~has_nsa), {'REMOVE': 3})

    q = Query((lambda *x: np.abs(np.median(x, axis=0)) > 0.5, 'g_err', 'r_err', 'i_err'))
    fill_values_by_query(base, q & (~has_nsa), {'REMOVE': 4})

    return base


def recover_whitelisted_objects(base, objects_to_add):
    """
    Use the "add list" to set REMOVE back to -1 for whitelisted objects.
    This is mainly to deal with objects that are removed by `remove_bad_photometry`
    This is step (5) of object removal.

    `base` is modified in-place.

    Parameters
    ----------
    base : astropy.table.Table
    objects_to_add : astropy.table.Table

    Returns
    -------
    base : astropy.table.Table
    """
    ids_to_add = _get_unique_objids(objects_to_add['SDSS ID'])
    fill_values_by_query(base, Query((lambda x: np.in1d(x, ids_to_add), 'OBJID')), {'REMOVE': -1})
    return base


def remove_shreds_with_highz(base):
    """
    Remove shereded objects that are within 1.25 R of each high-z (beyond NSA
    redshift cutoff) object in the base catalog. Set REMOVE to 4.

    Because this function uses objects with specs, it should be applied to
    base catalog after the specs are cleaned (i.e., after `clean_sdss_spectra`)

    `base` is modified in-place.

    Parameters
    ----------
    base : astropy.table.Table

    Returns
    -------
    base : astropy.table.Table
    """
    highz_spec_cut = Query('SPEC_Z > 0.05', 'ZQUALITY >= 3', 'PETRORADERR_R > 0', 'PETRORAD_R > 2.0*PETRORADERR_R', 'REMOVE == -1')
    highz_spec_indices = np.where(highz_spec_cut.mask(base))[0]

    for idx in highz_spec_indices:

        if base['REMOVE'][idx] != -1:
            continue

        nearby_obj_mask  = (base['coord'].separation(base['coord'][idx]).arcsec < 1.25 * base['PETRORAD_R'][idx])
        nearby_obj_mask &= (base['REMOVE'] == -1)

        assert nearby_obj_mask[idx]
        nearby_obj_mask[idx] = False
        nearby_obj_count = np.count_nonzero(nearby_obj_mask)

        if not nearby_obj_count:
            continue

        if nearby_obj_count > 25:
            logging.warning('In SAGA.objects.build.remove_shreds_with_highz()\n Too many (> 25) shreds around high-z object {} ({}, {})'.format(base['OBJID'][idx], base['RA'][idx], base['DEC'][idx]))

        base['REMOVE'][nearby_obj_mask] = 4

    return base


def apply_manual_fixes(base):
    """
    Apply manual fixes to base catalog using `manual_fixes.fixes_by_sdss_objid`

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


def clean_repeat_spectra(spectra):
    """
    Remove repeated spectra by search nearby spectra in 3D,
    within 20 arcsec (or 10 arcsec for z > 0.2) and dz in +/-50 km/s
    Keep the best one, other are removed (but TELNAME is recorded and added to SPEC_REPEAT)

    `spectra` is modified in-place.

    Parameters
    ----------
    spectra : astropy.table.Table

    Returns
    -------
    spectra : astropy.table.Table

    Notes
    -----
    `add_spectra` and `clean_sdss_spectra` calls this function.
    """
    spectra = add_skycoord(spectra)
    spec_repeat = get_empty_str_array(len(spectra), 48)
    not_done = np.ones(len(spectra), np.bool)

    spec_repeat_col = 'TELNAME' if 'TELNAME' in spectra.colnames else 'SPEC_REPEAT'

    for i, spec in enumerate(spectra):
        if not not_done[i]:
            continue

        # search nearby spectra in 3D
        nearby_mask = (np.fabs(spectra['SPEC_Z'] - spec['SPEC_Z']) < _spec_search_dz)
        nearby_mask &= (spectra['coord'].separation(spec['coord']).arcsec < _get_spec_search_radius(spec['SPEC_Z']))
        nearby_mask &= not_done
        nearby_mask = np.where(nearby_mask)[0]
        assert len(nearby_mask) >= 1

        not_done[nearby_mask] = False

        best_spec_idx = nearby_mask[spectra['ZQUALITY'][nearby_mask].argmax()]
        spec_repeat[best_spec_idx] = _join_spec_repeat(*spectra[spec_repeat_col][nearby_mask])

    del not_done

    mask = (spec_repeat != '')
    spectra = spectra[mask]
    spectra['SPEC_REPEAT'] = spec_repeat[mask]

    return spectra


def add_cleaned_spectra(base, spectra_clean):
    """
    Add cleaned spectra to base catalog.
    For each entry in `spectra_clean`, search nearby objects in base catalog
    to find an object to match the spectrum to.

    The search is done in the following order.
    In each category, if any objects are found, the closest one (on the sky)
    will be the match, and the search ends.

    1. within 20 arcsec, spec within +/- 50 km/s, REMOVE==-1, has NSA
    2. within 20 arcsec, spec within +/- 50 km/s, REMOVE==-1, any ZQUALITY==4
    3. within 3 arcsec, REMOVE==-1
    4. within 3 arcsec

    Once a match is found, any other objects that satisfy the condition (2) above
    are turned off (REMOVE set to 3).

    For high-z objects (z > 0.2), the number "20 arcsec" becomes "10 arcsec"

    `base` is modified in-place.
    `spectra_clean` is expected to be the output of `clean_repeat_spectra`

    Parameters
    ----------
    base : astropy.table.Table
    spectra_clean : astropy.table.Table

    Returns
    -------
    base : astropy.table.Table

    Notes
    -----
    `add_spectra` calls this function.
    """
    for spec in spectra_clean:
        sep = spec['coord'].separation(base['coord']).arcsec
        nearby_obj_indices = np.where(sep < _get_spec_search_radius(spec['SPEC_Z']))[0]

        if len(nearby_obj_indices) == 0:
            if spec['TELNAME'] != 'GAMA':
                logging.warning('In SAGA.objects.build.add_cleaned_spectra()\n No object within 20 arcsec of {} spec ({}, {})'.format(spec['TELNAME'], spec['RA'], spec['DEC']))
            continue

        # for faster access, slice a small set of the base catalog.
        nearby_obj = base[['REMOVE', 'ZQUALITY', 'SPEC_Z', 'OBJ_NSAID']][nearby_obj_indices]

        mask = (sep[nearby_obj_indices] < 3.0)
        very_close_indices = nearby_obj_indices[mask]

        mask &= Query('REMOVE == -1').mask(nearby_obj)
        very_close_clean_indices = nearby_obj_indices[mask]

        mask = Query('REMOVE == -1', 'ZQUALITY == 4', 'abs(SPEC_Z - {}) < {}'.format(spec['SPEC_Z'], _spec_search_dz)).mask(nearby_obj)
        nearby_has_spec_indices = nearby_obj_indices[mask]

        mask &= Query('OBJ_NSAID > -1').mask(nearby_obj)
        nearby_nsa_indices = nearby_obj_indices[mask]

        for indices in (nearby_nsa_indices, nearby_has_spec_indices, very_close_clean_indices, very_close_indices):
            if len(indices) > 0:
                closest_obj_index = indices[sep[indices].argmin()]
                break
        else:
            if spec['TELNAME'] != 'GAMA':
                logging.warning('In SAGA.objects.build.add_cleaned_spectra()\n No object can be matched to {} spec ({}, {})'.format(spec['TELNAME'], spec['RA'], spec['DEC']))
            continue

        del sep, mask, nearby_obj, nearby_obj_indices, very_close_indices, very_close_clean_indices, nearby_nsa_indices

        if spec['ZQUALITY'] > base['ZQUALITY'][closest_obj_index] or \
                (spec['ZQUALITY'] == base['ZQUALITY'][closest_obj_index] and spec['TELNAME'] == 'MMT'):
            for col in ('TELNAME', 'MASKNAME', 'ZQUALITY', 'SPEC_Z', 'SPEC_Z_ERR', 'SPECOBJID'):
                base[col][closest_obj_index] = spec[col]

        base['SPEC_REPEAT'][closest_obj_index] = _join_spec_repeat(spec['SPEC_REPEAT'], base['SPEC_REPEAT'][closest_obj_index], *base['SPEC_REPEAT'][nearby_has_spec_indices])

        if len(nearby_has_spec_indices) > 0:
            base['REMOVE'][nearby_has_spec_indices] = 3
            base['REMOVE'][closest_obj_index] = -1

    return base


def add_spectra(base, spectra):
    """
    Add spectra to base catalog.
    This function calls `clean_repeat_spectra` to clean input spectra,
    and then calls `add_cleaned_spectra` to add cleaned spectra to base catalog.

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
        to_remove_coord = True # because we should NOT modify spectra in place!

    host_sc = SkyCoord(base['HOST_RA'][0], base['HOST_DEC'][0], unit='deg')
    spectra_here = spectra[spectra['coord'].separation(host_sc).deg < 1.0]

    if to_remove_coord:
        del spectra['coord']

    del spectra

    if len(spectra_here) > 0:
        base = add_cleaned_spectra(base, clean_repeat_spectra(spectra_here))

    return base


def clean_sdss_spectra(base):
    """
    Clean up SDSS spectra (i.e., those already in base catalog).
    First, use `clean_repeat_spectra` to clean SDSS spectra, and then keep only unique ones.

    `base` is modified in-place.

    Parameters
    ----------
    base : astropy.table.Table

    Returns
    -------
    base : astropy.table.Table
    """
    find_sdss_only = lambda t: np.fromiter(((x and set(x.split('+')).issubset({'NSA', 'SDSS'})) for x in t['SPEC_REPEAT']), np.bool, len(t))
    sdss_specs_indices = np.where(Query('ZQUALITY == 4', 'REMOVE == -1', find_sdss_only).mask(base))[0]
    if len(sdss_specs_indices) > 0:
        sdss_specs = base[['SPEC_REPEAT', 'SPEC_Z', 'TELNAME', 'ZQUALITY', 'coord']][sdss_specs_indices]
        sdss_specs['indices'] = sdss_specs_indices
        sdss_specs['ZQUALITY'][sdss_specs['TELNAME'] == 'NSA'] = 5 # so that `clean_repeat_spectra` will prefer NSA
        del sdss_specs['TELNAME'] # so that `clean_repeat_spectra` will use SPEC_REPEAT rather than TELNAME
        sdss_specs = clean_repeat_spectra(sdss_specs)
        base['REMOVE'][sdss_specs_indices] = 3
        base['REMOVE'][sdss_specs['indices']] = -1
        base['SPEC_REPEAT'][sdss_specs['indices']] = sdss_specs['SPEC_REPEAT']

    return base


def find_satellites(base):
    """
    Add `SATS` column to the base catalog.

    -1 - default value
     0 - high-z galaxies
     1 - Satellites!!!
     2 - low-z galaxies
     3 - host galaxy itself
    91 - removed objects but otherwise satisfying satellite cut
    92 - removed objects but otherwise satisfying low-z galaxy cut

    `base` is modified in-place.

    Parameters
    ----------
    base : astropy.table.Table

    Returns
    -------
    base : astropy.table.Table
    """
    # clean objects
    clean_obj = C.is_galaxy & C.has_spec & C.is_clean
    fill_values_by_query(base, clean_obj & C.is_high_z, {'SATS':0})
    fill_values_by_query(base, clean_obj & ~C.is_high_z, {'SATS':2})
    fill_values_by_query(base, clean_obj & C.sat_rcut & C.sat_vcut, {'SATS':1})

    # removed objects
    removed_obj = C.is_galaxy & C.has_spec & (~C.is_clean)
    fill_values_by_query(base, removed_obj & ~C.is_high_z, {'SATS':92})
    fill_values_by_query(base, removed_obj & C.sat_rcut & C.sat_vcut, {'SATS':91})

    # host itself!
    fill_values_by_query(base, C.obj_is_host, {'SATS':3, 'REMOVE':-1})

    return base


def add_stellar_mass(base):
    """
    Calculate stellar mass based only on gi colors and redshift and add to base catalog.
    Based on GAMA data using Taylor et al (2011).

    `base` is modified in-place.

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
    """
    This function calls all needed functions to complete the full stack of building
    a base catalog (for a single host), in the following order:

    >>> initialize_base_catalog(base)
    >>> add_host_info(base, host, saga_names)
    >>> add_wise(base, wise)
    >>> remove_human_inspected(base, objects_to_remove)
    >>> remove_too_close_to_host(base)
    >>> remove_shreds_with_nsa(base, nsa)
    >>> remove_bad_photometry(base)
    >>> recover_whitelisted_objects(base, objects_to_add)
    >>> apply_manual_fixes(base)
    >>> add_spectra(base, spectra)
    >>> clean_sdss_spectra(base)
    >>> remove_shreds_with_highz(base)
    >>> find_satellites(base)
    >>> add_stellar_mass(base) # not yet implemented

    Among these function, `add_wise` can be applied at any time. All other
    functions should be applied in this particular order.
    See docstring of each function to learn more.

    `base` is always modified in-place.

    Parameters
    ----------
    base : astropy.table.Table
    host : astropy.table.Row
    saga_names : astropy.table.Table
    wise : astropy.table.Table
    nsa : astropy.table.Table
    objects_to_remove : astropy.table.Table
    objects_to_add : astropy.table.Table
    spectra : astropy.table.Table

    Returns
    -------
    base : astropy.table.Table
    """
    base = initialize_base_catalog(base)
    base = add_host_info(base, host, saga_names)
    if wise is not None:
        base = add_wise(base, wise)
    if objects_to_remove is not None:
        base = remove_human_inspected(base, objects_to_remove)
    base = remove_too_close_to_host(base)
    if nsa is not None:
        base = remove_shreds_with_nsa(base, nsa)
    base = remove_bad_photometry(base)
    if objects_to_add is not None:
        base = recover_whitelisted_objects(base, objects_to_add)
    base = apply_manual_fixes(base)
    if spectra:
        base = add_spectra(base, spectra)
    base = clean_sdss_spectra(base)
    base = remove_shreds_with_highz(base)
    base = find_satellites(base)
    #base = add_stellar_mass(base)

    return base
