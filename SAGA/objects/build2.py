import logging
from itertools import chain
from collections import defaultdict
import numpy as np
import numexpr as ne
from easyquery import Query
from fast3tree import find_friends_of_friends
from astropy.coordinates import SkyCoord
from astropy.table import Table, join, vstack

from . import build
from ..utils import fill_values_by_query, get_empty_str_array, get_remove_flag, get_sdss_bands, add_skycoord
from ..database.spectra import extract_nsa_spectra, extract_sdss_spectra

__all__ = ['prepare_sdss_catalog_for_merging',
           'prepare_des_catalog_for_merging',
           'prepare_decals_catalog_for_merging',
           'merge_catalogs',
           'merge_spectra',
           'add_spectra',
           'remove_shreds_near_spec_obj',
           'build_full_stack']

MERGED_CATALOG_COLUMNS = list(chain(
    ('OBJID', 'RA', 'DEC', 'REMOVE', 'is_galaxy', 'radius'),
    (b+'_mag' for b in 'ugrizy'),
    (b+'_err' for b in 'ugrizy'),
))


def arcsec2dist(sep, r=1.0):
    return np.sin(np.deg2rad(sep / 3600.0 / 2.0)) * 2.0 * r


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
    group_id_shift = group_id.max() + 1
    rerun_mask = (counts[group_id] > n_catalogs)
    group_id[rerun_mask] = find_friends_of_friends(points[rerun_mask], arcsec2dist(1), reassign_group_indices=False)
    group_id[rerun_mask] += group_id_shift
    stacked_catalog['group_id'] = group_id
    del points, group_id, rerun_mask

    stacked_catalog.sort(['group_id', 'REMOVE', 'r_mag'])
    stacked_catalog['chosen'] = 0

    group_id_edges = np.flatnonzero(np.hstack(([1], np.ediff1d(stacked_catalog['group_id']), [1])))
    for i, j in zip(group_id_edges[:-1], group_id_edges[1:]):
        stacked_catalog['chosen'][i:j] = 2 if (j-i == 1) else assign_choice(stacked_catalog['survey'][i:j])

    merged_catalog = Query('chosen == 2').filter(stacked_catalog)
    for name in catalog_dict:
        merged_catalog = join(merged_catalog,
                              Query('chosen > 0', (lambda x: x == name, 'survey')).filter(stacked_catalog)[MERGED_CATALOG_COLUMNS+['group_id']],
                              keys='group_id',
                              join_type='left',
                              uniq_col_name='{col_name}{table_name}',
                              table_names=['', '_'+name])

    del merged_catalog['group_id']
    del merged_catalog['chosen']

    return merged_catalog


def find_best_spec(specs):
    rank = ((specs['TELNAME'] == 'NSA') * 2 + (specs['TELNAME'] == 'MMT')) * (specs['ZQUALITY'] == 4) + specs['ZQUALITY']
    return rank.argsort()[-1], '+'.join(set(specs['TELNAME']))


def merge_spectra(specs):
    specs = add_skycoord(specs)
    points = specs['coord'].cartesian.xyz.value.T
    low_z_mask = Query('SPEC_Z < 0.2').mask(specs)
    group_id = np.repeat(-1, len(specs))
    group_id[low_z_mask] = find_friends_of_friends(points[low_z_mask], arcsec2dist(20), reassign_group_indices=False)
    group_id_shift = group_id.max() + 1
    group_id[~low_z_mask] = find_friends_of_friends(points[~low_z_mask], arcsec2dist(10), reassign_group_indices=False)
    group_id[~low_z_mask] += group_id_shift
    assert (group_id >= 0).all()
    specs['group_id'] = group_id
    del points, low_z_mask, group_id, specs['coord']
    specs.sort(['group_id', 'SPEC_Z'])
    specs['SPEC_REPEAT'] = get_empty_str_array(len(specs), 48)
    specs['chosen'] = False

    edge_mask = (np.ediff1d(specs['group_id']) > 0) | (np.ediff1d(specs['SPEC_Z']) > build._spec_search_dz)
    group_id_edges = np.flatnonzero(np.hstack(([True], edge_mask, [True])))
    del edge_mask

    for i, j in zip(group_id_edges[:-1], group_id_edges[1:]):
        if j == i + 1:
            specs['chosen'][i] = True
            specs['SPEC_REPEAT'][i] = specs['TELNAME'][i]
        else:
            k, spec_repeat = find_best_spec(specs[i:j])
            specs['chosen'][k] = True
            specs['SPEC_REPEAT'][k] = spec_repeat

    specs = Query('chosen').filter(specs)

    del specs['group_id']
    del specs['chosen']

    return specs


def add_spectra(base, specs):
    specs = add_skycoord(specs)

    matched_idx = np.repeat(-1, len(specs))
    base_idx = np.flatnonzero(Query('is_galaxy', 'REMOVE == 0').mask(base))
    idx, sep, _ = specs['coord'].match_to_catalog_sky(base['coord'][base_idx])
    matched_mask = (sep.arcsec < 3.0)
    matched_idx[matched_mask] = base_idx[idx[matched_mask]]

    not_yet_matched_mask = (matched_idx == -1)
    if not_yet_matched_mask.any():
        base_idx = np.flatnonzero(Query('is_galaxy', 'REMOVE > 0').mask(base))
        idx, sep, _ = specs['coord'][not_yet_matched_mask].match_to_catalog_sky(base['coord'][base_idx])
        matched_mask = (sep.arcsec < 3.0)
        matched_idx[np.flatnonzero(not_yet_matched_mask)[matched_mask]] = base_idx[idx[matched_mask]]

    not_yet_matched_mask = (matched_idx == -1)
    if not_yet_matched_mask.any():
        base_idx = np.flatnonzero((~Query('is_galaxy')).mask(base))
        idx, sep, _ = specs['coord'][not_yet_matched_mask].match_to_catalog_sky(base['coord'][base_idx])
        matched_mask = (sep.arcsec < 1.0)
        matched_idx[np.flatnonzero(not_yet_matched_mask)[matched_mask]] = base_idx[idx[matched_mask]]

    specs['matched_idx'] = matched_idx
    del matched_idx, base_idx, idx, sep, matched_mask, not_yet_matched_mask

    base['SPEC_Z'] = np.float32(-1)
    base['SPEC_Z_ERR'] = np.float32(-1)
    base['ZQUALITY'] = np.int16(-1)
    base['TELNAME'] = get_empty_str_array(len(base), 6)
    base['MASKNAME'] = get_empty_str_array(len(base), 48)
    base['SPECOBJID'] = get_empty_str_array(len(base), 48)
    base['SPEC_REPEAT'] = get_empty_str_array(len(base), 48)

    del specs['coord']
    specs.sort('matched_idx')
    start_idx = np.flatnonzero(specs['matched_idx'] > -1)[0]
    for col in ('SPEC_Z', 'SPEC_Z_ERR', 'ZQUALITY', 'TELNAME', 'MASKNAME', 'SPECOBJID', 'SPEC_REPEAT'):
        base[col][specs['matched_idx'][start_idx:]] = specs[col][start_idx:]

    has_nsa = (base['TELNAME'] == 'NSA')
    base['OBJ_NSAID'] = np.int32(-1)
    base['OBJ_NSAID'][has_nsa] = np.array(base['SPECOBJID'][has_nsa], dtype=np.int32)

    for spec in specs[:start_idx]:
        if spec['SPEC_REPEAT'] != 'GAMA':
            logging.warning('No photo obj matched to {} spec obj ({}, {})'.format(spec['TELNAME'], spec['RA'], spec['DEC']))

    return base


def remove_shreds_near_spec_obj(base, nsa=None):

    spec_obj = Query('SPEC_Z > 0.05', 'ZQUALITY >= 3', 'is_galaxy', 'radius > 0')
    spec_obj |= Query((lambda x: x == 'NSA', 'TELNAME'))
    spec_obj_indices = np.flatnonzero(spec_obj.mask(base))

    for idx in spec_obj_indices:
        obj_this = base[idx]

        if nsa is not None and obj_this['TELNAME'] == 'NSA':
            nsa_obj = Query('NSAID == {}'.format(obj_this['SPECOBJID'])).filter(nsa)[0]
            ellipse_calculation = dict()
            ellipse_calculation['a'] = nsa_obj['PETROTH90'] * 2.0 / 3600.0
            ellipse_calculation['b'] = ellipse_calculation['a'] * nsa_obj['SERSIC_BA']
            ellipse_calculation['th'] = np.deg2rad(nsa_obj['SERSIC_PHI'] + 270.0)
            ellipse_calculation['s'] = np.sin(ellipse_calculation['th'])
            ellipse_calculation['c'] = np.cos(ellipse_calculation['th'])
            ellipse_calculation['x'] = base['RA'] - nsa_obj['RA']
            ellipse_calculation['y'] = base['DEC'] - nsa_obj['DEC']
            r2_ellipse = ne.evaluate('((x*c - y*s)/a)**2.0 + ((x*s + y*c)/b)**2.0',
                                    local_dict=ellipse_calculation, global_dict={})
            nearby_obj_mask = (r2_ellipse < 1.0)
            remove_flag = 21

        else:
            if obj_this['REMOVE'] > 0:
                continue
            nearby_obj_mask = (base['coord'].separation(obj_this['coord']).arcsec < 1.25 * obj_this['radius'])
            remove_flag = 22

        nearby_obj_mask[idx] = False
        nearby_obj_count = np.count_nonzero(nearby_obj_mask)

        if not nearby_obj_count:
            continue

        if nearby_obj_count > 25:
            logging.warning('Too many (> 25) shreds around spec object {} ({}, {})'.format(obj_this['OBJID'], obj_this['RA'], obj_this['DEC']))

        base['REMOVE'][nearby_obj_mask] += (1 << remove_flag)

    return base


def build_full_stack(host, saga_names=None, sdss=None, des=None, decals=None, nsa=None,
                     sdss_remove=None, sdss_recover=None, spectra=None):
    """
    This function calls all needed functions to complete the full stack of building
    a base catalog (for a single host), in the following order:

    Returns
    -------
    base : astropy.table.Table
    """
    all_spectra = []
    if sdss is not None:
        all_spectra.append(extract_sdss_spectra(sdss))
        sdss = prepare_sdss_catalog_for_merging(sdss, sdss_remove, sdss_recover)

    if des is not None:
        des = prepare_des_catalog_for_merging(des)

    if decals is not None:
        decals = prepare_decals_catalog_for_merging(decals)

    if nsa is not None:
        all_spectra.append(extract_nsa_spectra(nsa))

    if spectra is not None:
        if 'coord' in spectra.colnames:
            del spectra['coord']
        all_spectra.append(spectra)

    base = merge_catalogs(sdss=sdss, des=des, decals=decals)
    del sdss, des, decals, spectra

    base = build.add_host_info(base, host, saga_names)
    base['REMOVE'][Query('RHOST_KPC < 10.0').mask(base)] += (1 << 20)

    if all_spectra:
        all_spectra = merge_spectra(vstack(all_spectra, 'exact', 'error'))
        base = add_spectra(base, all_spectra)
        del all_spectra
        base = remove_shreds_near_spec_obj(base)

    base = build.find_satellites(base)

    return base
