import numpy as np
import numexpr as ne
from astropy.coordinates import SkyCoord
from easyquery import Query
from . import cuts as C
from .manual_fixes import fixes_by_sdss_objid
from ..utils import join_table_by_coordinates, fill_values_by_query, get_empty_str_array


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
        raise ValueError('Existing host info and differs from input host info.')

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


def add_more_photometric_data(base, wise, **kwargs):
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
    if kwargs:
        raise NotImplementedError('Photometric data other than wise has not been implemented.')

    cols_rename = {'W1_MAG':'W1', 'W1_MAG_ERR':'W1ERR', 'W2_MAG':'W2', 'W2_MAG_ERR':'W2ERR'}
    join_table_by_coordinates(base, wise, list(cols_rename.keys()), cols_rename)

    # use -1.0 instead np.nan for missing values
    for col in cols_rename.values():
        fill_values_by_query(base, Query((np.isnan, col)), {col:-1.0})

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

    host_sc = SkyCoord(base['HOST_RA'][0], base['HOST_DEC'][0], unit='deg')
    nsa = nsa[SkyCoord(nsa['RA'], nsa['DEC'], unit="deg").separation(host_sc).deg < 1.0]

    if len(nsa) == 0:
        return base

    base_sc = SkyCoord(base['RA'], base['DEC'], unit='deg')

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
        nearby_obj_to_remove = (r2_ellipse < 1.0)

        base['REMOVE'][nearby_obj_to_remove] = 2

        del nearby_obj_to_remove, r2_ellipse, values_for_ellipse_calculation

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
            #TODO: finish this
        }


    return base



def add_spectra(base, spectra, ignore_imacs=False):
    """
    Add spectra to base catalog.
    `base` is modified in-place.

    Parameters
    ----------
    base : astropy.table.Table
    spectra : astropy.table.Table
    ignore_imacs : bool, optional

    Returns
    -------
    base : astropy.table.Table
    """

    cols_to_copy = ('TELNAME', 'MASKNAME', 'ZQUALITY', 'SPEC_Z', 'SPEC_Z_ERR', 'specobjid')

    if 'REMOVE' not in base.columns:
        base['REMOVE'] = -1
    if 'TELNAME' not in base.columns:
        base['TELNAME'] = get_empty_str_array(len(base), 6)
    if 'MASKNAME' not in base.columns:
        base['MASKNAME'] = get_empty_str_array(len(base))
    if 'ZQUALITY' not in base.columns:
        base['ZQUALITY'] = -1
    if 'SPEC_Z' not in base.columns:
        base['SPEC_Z'] = -1.0
    if 'SPEC_Z_ERR' not in base.columns:
        base['SPEC_Z_ERR'] = -1.0
    if 'SPEC_REPEAT' not in base.columns:
        base['SPEC_REPEAT'] = get_empty_str_array(len(base))
    if 'SPECOBJID' not in base.columns:
        base['SPECOBJID'] = get_empty_str_array(len(base), 48)

    host_sc = SkyCoord(base['HOST_RA'][0], base['HOST_DEC'][0], unit='deg')
    spectra_sc = SkyCoord(spectra['RA'], spectra['DEC'], unit="deg")
    near_host_mask = spectra_sc.separation(host_sc).deg < 1.0

    if not near_host_mask.any():
        return base

    spectra = spectra[near_host_mask]
    spectra_sc = spectra_sc[near_host_mask]
    del near_host_mask

    base_sc = SkyCoord(base['RA'], base['DEC'], unit='deg')

    done_spectra_indices = []

    for i, spec in enumerate(spectra):
        if i in done_spectra_indices:
            continue

        spec_sc = SkyCoord(spec['RA'], spec['DEC'], unit='deg')

        # do an initial search of objects within 5 arcsec
        objects_nearby = base[base_sc.separation(spec_sc).arcsec < 5.0]
        if len(objects_nearby) == 0:
            raise ValueError('Marla said there must be an object!!')

        #TODO: change this to 3D match
        mask = (objects_nearby['PETRORAD_R'] < 30.0)
        if mask.any():
            radius = objects_nearby['PETRORAD_R'][mask].max()
        else:
            radius = objects_nearby['PETRORAD_R'][(objects_nearby['r'] - objects_nearby['EXTINCTION_R']).argmin()]

        # now we search within the object radius
        # note that we need the indices here to keep track of specs and to write to base

        objects_nearby_indices = np.where(base_sc.separation(spec_sc).arcsec < radius)[0]
        objects_nearby = base[objects_nearby_indices]

        specs_nearby_indices = np.where(spectra_sc.separation(spec_sc).arcsec < radius)[0]
        specs_nearby = spectra[specs_nearby_indices]

        done_spectra_indices.extend(specs_nearby_indices)

        if ignore_imacs:
            specs_nearby = specs_nearby[specs_nearby['TELNAME'] != 'IMACS']

        # gather SPEC_REPEAT before we identify the best spec
        spec_repeat = set()
        for r in specs_nearby['SPEC_REPEAT']:
            spec_repeat.update(r.split('+'))
        spec_repeat = '+'.join(spec_repeat)

        # should prefer NSA
        best_spec = specs_nearby[specs_nearby['ZQUALITY'].data.argmax()]
        best_spec_sc = SkyCoord(best_spec['RA'], best_spec['DEC'], unit='deg')
        closest_object_index = SkyCoord(objects_nearby['RA'], objects_nearby['DEC'], unit='deg').separation(best_spec_sc).arcsec.argmin()

        original_base_index = objects_nearby_indices[closest_object_index]
        base['SPEC_REPEAT'][original_base_index] = spec_repeat
        for col in cols_to_copy:
            base[col.upper()][original_base_index] = best_spec[col]

        other_objects = objects_nearby_indices[objects_nearby_indices != original_base_index]
        base['REMOVE'][other_objects] = 0


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



def calc_stellar_mass(base):
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