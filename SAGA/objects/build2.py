from itertools import chain
from collections import defaultdict
import numpy as np
from easyquery import Query
from fast3tree import find_friends_of_friends
from astropy.coordinates import SkyCoord
from astropy.table import Table, join, vstack

from . import build
from ..utils import fill_values_by_query, get_empty_str_array, get_remove_flag, get_sdss_bands, add_skycoord
from ..database import spectra

__all__ = ['get_sdss_spectra', 'prepare_sdss_catalog_for_merging',
           'prepare_des_catalog_for_merging', 'prepare_decals_catalog_for_merging',
           'merge_catalogs']

MERGED_CATALOG_COLUMNS = list(chain(
    ('OBJID', 'RA', 'DEC', 'REMOVE', 'is_galaxy', 'radius'),
    (b+'_mag' for b in 'ugrizy'),
    (b+'_err' for b in 'ugrizy'),
))


def arcsec2dist(sep, r=1.0):
    return np.sin(np.deg2rad(sep / 3600.0 / 2.0)) * 2.0 * r


def get_sdss_spectra(catalog):
    specs = Query('SPEC_Z > -1.0').filter(catalog['RA', 'DEC', 'SPEC_Z', 'SPEC_Z_ERR', 'SPEC_Z_WARN'])
    specs['ZQUALITY'] = np.where(specs['SPEC_Z_WARN'] == 0, 4, 1)
    specs['TELNAME'] = get_empty_str_array(len(specs), 6, 'SDSS')
    specs['MASKNAME'] = get_empty_str_array(len(specs), 48, 'SDSS')
    specs['SPECOBJID'] = get_empty_str_array(len(specs), 48)
    specs['SPEC_REPEAT'] = get_empty_str_array(len(specs), 48, 'SDSS')
    del specs['SPEC_Z_WARN']
    return specs


def prepare_sdss_catalog_for_merging(catalog, to_remove=None, to_recover=None):
    for b in get_sdss_bands():
        catalog['{}_mag'.format(b)] = np.where(catalog['PHOTPTYPE'] == 6, catalog['PSFMAG_{}'.format(b.upper())], catalog[b]) - catalog['EXTINCTION_{}'.format(b.upper())]
    catalog['y_mag'] = 99.0
    catalog['y_err'] = 99.0

    catalog['is_galaxy'] = (catalog['PHOTPTYPE'] == 3)
    catalog['radius'] = catalog['PETRORAD_R']

    remove_queries = [
        'BINNED1 == 0',
        'SATURATED != 0',
        'BAD_COUNTS_ERROR != 0',
        (lambda *x: np.abs(np.median(x, axis=0)) > 0.5, 'g_err', 'r_err', 'i_err'),
    ]
    if to_remove:
        ids_to_remove = build._get_unique_objids(to_remove['SDSS ID'])
        remove_queries.append((lambda x: np.in1d(x, ids_to_remove), 'OBJID'))

    catalog['REMOVE'] = get_remove_flag(catalog, remove_queries)

    if to_recover:
        catalog = build.recover_whitelisted_objects(catalog, to_recover)
        fill_values_by_query(catalog, 'REMOVE == -1', {'REMOVE': 0})

    return catalog[MERGED_CATALOG_COLUMNS]


def prepare_des_catalog_for_merging(catalog):

    catalog['radius'] = catalog['radius_r']
    catalog['is_galaxy'] = (catalog['wavg_extended_coadd_i'] >= 1)

    try:
        catalog.rename_column('ra', 'RA')
        catalog.rename_column('dec', 'DEC')
        catalog.rename_column('objid', 'OBJID')
    except KeyError:
        pass

    gi = catalog['g_mag'] - catalog['i_mag']
    catalog['g_mag'] += (-0.0009 + 0.055 * gi)
    catalog['r_mag'] += (-0.0048 + 0.0703 * gi)
    catalog['i_mag'] += (-0.0065 - 0.0036 * gi + 0.02672 * gi * gi)
    catalog['z_mag'] += (-0.0438 + 0.02854 * gi)

    catalog['u_mag'] = 99.0
    catalog['u_err'] = 99.0

    catalog['REMOVE'] = get_remove_flag(catalog, [
        'imaflags_iso_r != 0',
        'flags_r >= 4',
        'r_mag >= 25',
    ])

    return catalog[MERGED_CATALOG_COLUMNS]


def prepare_decals_catalog_for_merging(catalog):
    catalog['OBJID'] = np.array(catalog['BRICKID'], dtype=np.int64) * int(1e13) + np.array(catalog['OBJID'], dtype=np.int64)
    catalog['is_galaxy'] = catalog['TYPE'] != 'PSF'
    catalog['radius'] = catalog['FRACDEV'] * catalog['SHAPEDEV_R'] + (1.0 - catalog['FRACDEV']) * catalog['SHAPEEXP_R']

    for band in 'uiy':
        catalog['{}_mag'.format(band)] = 99.0
        catalog['{}_err'.format(band)] = 99.0

    catalog['REMOVE'] = get_remove_flag(catalog, [
        'SHAPEEXP_R >= 20',
        'SHAPEDEV_R >= 20',
        'FRACMASKED_G >= 0.3',
        'FRACMASKED_R >= 0.3',
        'RCHISQ_G >= 10',
        'RCHISQ_R >= 10',
        'RCHISQ_Z >= 10',
        'ALLMASK_G > 0',
        'ALLMASK_R > 0',
        'g_err >= 0.2',
        'r_err >= 0.2',
        'NOBS_G == 0',
        'NOBS_R == 0',
        'r_mag >= 25',
    ])
    return catalog[MERGED_CATALOG_COLUMNS]


def assign_choice(surveys):
    choice = list()
    done = set()
    for s in surveys:
        if not done:
            choice.append(2)
            done.add(s)
        elif s not in done:
            choice.append(1)
            done.add(s)
        else:
            choice.append(0)
    return choice


def merge_catalogs(**catalog_dict):

    catalog_dict = {k: v for k, v in catalog_dict.items() if v is not None}
    n_catalogs = len(catalog_dict)
    stacked_catalog = vstack(list(catalog_dict.values()), 'exact', 'error')
    stacked_catalog['survey'] = get_empty_str_array(len(stacked_catalog), max(len(s) for s in catalog_dict))
    i = 0
    for name, cat in catalog_dict.items():
        stacked_catalog['survey'][i:i+len(cat)] = name
        i += len(cat)

    points = SkyCoord(stacked_catalog['RA'], stacked_catalog['DEC'], unit='deg').cartesian.xyz.value.T
    group_id = find_friends_of_friends(points, arcsec2dist(3), reassign_group_indices=False)
    _, group_id, counts = np.unique(group_id, return_inverse=True, return_counts=True)
    n_groups = len(counts)
    counts = counts[group_id]

    rerun_mask = (counts > n_catalogs)
    group_id[rerun_mask] = find_friends_of_friends(points[rerun_mask], arcsec2dist(1), reassign_group_indices=False) + n_groups
    _, group_id, counts = np.unique(group_id, return_inverse=True, return_counts=True)

    stacked_catalog['group_id'] = group_id
    stacked_catalog.sort(['group_id', 'REMOVE', 'r_mag'])
    stacked_catalog['choice'] = -1

    group_id_edges = np.append(np.insert(np.where(np.ediff1d(stacked_catalog['group_id']))[0]+1, 0, 0), len(stacked_catalog))
    for i, j in zip(group_id_edges[:-1], group_id_edges[1:]):
        stacked_catalog['choice'][i:j] = 2 if (j-i == 1) else assign_choice(stacked_catalog['survey'][i:j])

    merged_catalog = Query('choice == 2').filter(stacked_catalog)
    for name in catalog_dict:
        merged_catalog = join(merged_catalog,
                              Query('choice > 0', (lambda x: x == name, 'survey')).filter(stacked_catalog)[MERGED_CATALOG_COLUMNS+['group_id']],
                              keys='group_id',
                              join_type='left',
                              uniq_col_name='{col_name}{table_name}',
                              table_names=['', '_'+name])

    del merged_catalog['group_id']
    del merged_catalog['choice']

    return merged_catalog


def build_full_stack(host, saga_names=None, sdss=None, des=None, decals=None, nsa=None,
                     sdss_remove=None, sdss_recover=None, spectra=None):
    """
    This function calls all needed functions to complete the full stack of building
    a base catalog (for a single host), in the following order:

    Returns
    -------
    base : astropy.table.Table
    """
    if sdss:
        specs_sdss = get_sdss_spectra(sdss)
        sdss = prepare_sdss_catalog_for_merging(sdss, sdss_remove, sdss_recover)

    if des:
        des = prepare_des_catalog_for_merging(des)

    if decals:
        decals = prepare_decals_catalog_for_merging(decals)

    base = merge_catalogs(sdss=sdss, des=des, decals=decals)

    base = build.initialize_base_catalog(base)
    base = build.add_host_info(base, host, saga_names)
    base = build.remove_too_close_to_host(base)
    if nsa is not None:
        base = build.remove_shreds_with_nsa(base, nsa)
    base = build.apply_manual_fixes(base)
    if spectra:
        if sdss:
            if 'coord' in spectra:
                del spectra['coord']
            spectra = add_skycoord(spectra.ensure_dtype(vstack((specs_sdss, spectra), 'exact', 'error')))
        base = build.add_spectra(base, spectra)
    base = build.clean_sdss_spectra(base)
    base = build.remove_shreds_with_highz(base)
    base = build.find_satellites(base)
    #base = add_stellar_mass(base)

    return base
