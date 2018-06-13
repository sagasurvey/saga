"""
base catalog building pipeline 2.0
"""
import logging
from itertools import chain
import numpy as np
import numexpr as ne
from easyquery import Query
from fast3tree import find_friends_of_friends
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import join, vstack
import astropy.constants
import astropy.units

from . import build
from ..utils import fill_values_by_query, get_empty_str_array, get_remove_flag, get_sdss_bands, add_skycoord
from ..database.spectra import extract_nsa_spectra, extract_sdss_spectra, ensure_specs_dtype, SPECS_COLUMNS

__all__ = ['prepare_sdss_catalog_for_merging',
           'prepare_des_catalog_for_merging',
           'prepare_decals_catalog_for_merging',
           'merge_catalogs',
           'merge_spectra',
           'add_spectra',
           'remove_shreds_near_spec_obj',
           'add_surface_brightness',
           'build_full_stack',
           'NSA_COLS_USED']

MERGED_CATALOG_COLUMNS = list(chain(
    ('OBJID', 'RA', 'DEC', 'REMOVE', 'is_galaxy', 'morphology_info', 'radius', 'radius_err'),
    (b+'_mag' for b in 'ugrizy'),
    (b+'_err' for b in 'ugrizy'),
))

_NSA_COLS_USED = ['RA', 'DEC', 'PETRO_TH50', 'PETRO_TH90', 'PETRO_BA90', 'PETRO_PHI90',
                  'Z', 'ZSRC', 'NSAID', 'SERSIC_FLUX', 'SERSIC_FLUX_IVAR', 'EXTINCTION']
NSA_COLS_USED = list(_NSA_COLS_USED)

_SPEED_OF_LIGHT = astropy.constants.c.to('km/s').value # pylint: disable=E1101


def filter_nearby_object(catalog, host, radius_deg=1.001, remove_coord=True):
    catalog = add_skycoord(catalog)
    catalog = catalog[catalog['coord'].separation(host['coord']).deg < radius_deg]
    if remove_coord:
        del catalog['coord']
    return catalog


def arcsec2dist(sep, r=1.0):
    return np.sin(np.deg2rad(sep / 3600.0 / 2.0)) * 2.0 * r


def get_fof_group_id(catalog, linking_length_arcsec, reassign_group_indices=False):
    if 'coord' in catalog.colnames:
        sc = catalog['coord']
    else:
        sc = SkyCoord(catalog['RA'], catalog['DEC'], unit='deg')
    return find_friends_of_friends(sc.cartesian.xyz.value.T,
                                   arcsec2dist(linking_length_arcsec),
                                   reassign_group_indices=reassign_group_indices)


def prepare_sdss_catalog_for_merging(catalog, to_remove=None, to_recover=None):
    for b in get_sdss_bands():
        catalog['{}_mag'.format(b)] = np.where(catalog['PHOTPTYPE'] == 6, catalog['PSFMAG_{}'.format(b.upper())], catalog[b]) - catalog['EXTINCTION_{}'.format(b.upper())]
    catalog['y_mag'] = 99.0
    catalog['y_err'] = 99.0

    catalog['is_galaxy'] = (catalog['PHOTPTYPE'] == 3)
    catalog['morphology_info'] = catalog['PHOTPTYPE'].astype(np.int32)

    catalog['radius'] = catalog['PETROR50_R']
    catalog['radius_err'] = catalog['PETRORADERR_R'] * catalog['PETROR50_R'] / catalog['PETRORAD_R']
    fill_values_by_query(catalog, Query('radius_err < 0') | (~Query((np.isfinite, 'radius_err'))), {'radius_err': 9999.0})

    remove_queries = [
        'BINNED1 == 0',
        'SATURATED != 0',
        'BAD_COUNTS_ERROR != 0',
        (lambda *x: np.abs(np.median(x, axis=0)) > 0.5, 'g_err', 'r_err', 'i_err'),
        'abs(r_mag - i_mag) > 10',
        'abs(g_mag - r_mag) > 10',
    ]

    if to_remove:
        ids_to_remove = build.get_unique_objids(to_remove['SDSS ID'])
        remove_queries.append((lambda x: np.in1d(x, ids_to_remove), 'OBJID'))

    catalog['REMOVE'] = get_remove_flag(catalog, remove_queries)

    if to_recover:
        ids_to_recover = build.get_unique_objids(to_recover['SDSS ID'])
        fill_values_by_query(catalog, Query((lambda x: np.in1d(x, ids_to_recover), 'OBJID')), {'REMOVE': 0})

    return catalog[MERGED_CATALOG_COLUMNS]


def prepare_des_catalog_for_merging(catalog, to_remove, to_recover):

    catalog['radius'] = catalog['radius_r']
    catalog['radius_err'] = np.float32(0)

    catalog['is_galaxy'] = (catalog['wavg_extended_coadd_i'] >= 3)
    catalog['morphology_info'] = catalog['wavg_extended_coadd_i'].astype(np.int32)

    try:
        catalog.rename_column('ra', 'RA')
        catalog.rename_column('dec', 'DEC')
        catalog.rename_column('objid', 'OBJID')
    except KeyError:
        if not all((col in catalog.colnames for col in ('RA', 'DEC', 'OBJID'))):
            raise RuntimeError('Cannot rename `RA`, `DEC`, and/or `OBJID` in DES catalog')

    gi = catalog['g_mag'] - catalog['i_mag']
    catalog['g_mag'] += (-0.0009 + 0.055 * gi)
    catalog['r_mag'] += (-0.0048 + 0.0703 * gi)
    catalog['i_mag'] += (-0.0065 - 0.0036 * gi + 0.02672 * gi * gi)
    catalog['z_mag'] += (-0.0438 + 0.02854 * gi)

    catalog['u_mag'] = 99.0
    catalog['u_err'] = 99.0

    remove_queries = [
        'imaflags_iso_r != 0',
        'flags_r >= 4',
        'r_mag >= 25',
    ]

    if to_remove:
        ids_to_remove = build.get_unique_objids(to_remove['DES_OBJID'])
        remove_queries.append((lambda x: np.in1d(x, ids_to_remove), 'OBJID'))

    catalog['REMOVE'] = get_remove_flag(catalog, remove_queries)

    if to_recover:
        ids_to_recover = build.get_unique_objids(to_recover['DES_OBJID'])
        fill_values_by_query(catalog, Query((lambda x: np.in1d(x, ids_to_recover), 'OBJID')), {'REMOVE': 0})

    return catalog[MERGED_CATALOG_COLUMNS]


def prepare_decals_catalog_for_merging(catalog):
    catalog['OBJID'] = np.array(catalog['BRICKID'], dtype=np.int64) * int(1e13) + np.array(catalog['OBJID'], dtype=np.int64)
    catalog['is_galaxy'] = (catalog['TYPE'] != 'PSF')
    catalog['morphology_info'] = catalog['TYPE'].getfield('<U1').view(np.int32)
    catalog['radius'] = catalog['FRACDEV'] * catalog['SHAPEDEV_R'] + (1.0 - catalog['FRACDEV']) * catalog['SHAPEEXP_R']
    catalog['radius_err'] = np.hypot(catalog['FRACDEV']**2.0 / catalog['SHAPEDEV_R_IVAR'], (1.0 - catalog['FRACDEV'])**2.0 / catalog['SHAPEEXP_R_IVAR'])

    fill_values_by_query(catalog, Query('radius_err < 0') | (~Query((np.isfinite, 'radius_err'))), {'radius_err': 9999.0})

    for band in 'uiy':
        catalog['{}_mag'.format(band)] = 99.0
        catalog['{}_err'.format(band)] = 99.0

    for band in 'grz':
        catalog['{}_mag'.format(band)] += 0.1

    catalog['REMOVE'] = get_remove_flag(catalog, [
        'radius >= 20',
        'FRACMASKED_G >= 0.35',
        'FRACMASKED_R >= 0.35',
        'RCHISQ_G >= 10',
        'RCHISQ_R >= 10',
        'RCHISQ_Z >= 10',
        'ALLMASK_G > 0',
        'ALLMASK_R > 0',
        'g_err >= 0.2',
        'r_err >= 0.2',
        Query('NOBS_G == 0', 'NOBS_R == 0'),
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

    if n_catalogs == 0:
        raise ValueError('No catalogs to merge!!')

    elif n_catalogs == 1:
        survey, stacked_catalog = next(iter(catalog_dict.items()))
        stacked_catalog['survey'] = get_empty_str_array(len(stacked_catalog), max(6, len(survey)), survey)
        stacked_catalog['group_id'] = get_fof_group_id(stacked_catalog, 1.0, True)

    else:
        stacked_catalog = vstack(list(catalog_dict.values()), 'exact', 'error')
        stacked_catalog['survey'] = get_empty_str_array(len(stacked_catalog), max(6, max(len(s) for s in catalog_dict)))
        i = 0
        for name, cat in catalog_dict.items():
            stacked_catalog['survey'][i:i+len(cat)] = name
            i += len(cat)

        group_id = get_fof_group_id(stacked_catalog, 3.0)
        for sep in (2.0, 1.0):
            _, group_id, counts = np.unique(group_id, return_inverse=True, return_counts=True)
            group_id_shift = group_id.max() + 1
            regroup_mask = (counts[group_id] > n_catalogs)
            if not regroup_mask.any():
                break
            group_id[regroup_mask] = get_fof_group_id(stacked_catalog[regroup_mask], sep)
            group_id[regroup_mask] += group_id_shift
        stacked_catalog['group_id'] = group_id
        del group_id, regroup_mask

    stacked_catalog.sort(['group_id', 'REMOVE', 'r_mag'])
    stacked_catalog['chosen'] = 0

    group_id_edges = np.flatnonzero(np.hstack(([1], np.ediff1d(stacked_catalog['group_id']), [1])))
    for i, j in zip(group_id_edges[:-1], group_id_edges[1:]):
        if j-i == 1 or n_catalogs == 1:
            stacked_catalog['chosen'][i] = 2
        else:
            stacked_catalog['chosen'][i:j] = assign_choice(stacked_catalog['survey'][i:j])

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

    for name in catalog_dict:
        merged_catalog['OBJID_{}'.format(name)].fill_value = -1
        merged_catalog['REMOVE_{}'.format(name)].fill_value = -1
        merged_catalog['is_galaxy_{}'.format(name)].fill_value = False

    return merged_catalog.filled()


def replace_poor_sdss_sky_subtraction(base):

    mask = Query(
        'abs(r_mag_sdss - r_mag_decals) > 2',
        (lambda s: s == 'sdss', 'survey'),
        'OBJID_decals != -1',
        'REMOVE_decals == 0',
    ).mask(base)

    base['survey'][mask] = 'decals'
    for col in base.colnames:
        if col.endswith('_decals'):
            base[col.rpartition('_')[0]][mask] = base[col][mask]

    return base


def find_best_spec(specs):
    d = dict(SDSS=0, GAMA=1, NSA=2, AAT=4, MMT=5)
    rank = np.fromiter((d.get(t, 3) for t in specs['TELNAME']), np.int, len(specs))
    rank += specs['ZQUALITY'] * (rank.max() + 1)

    nsa_specs = specs[specs['TELNAME'] == 'NSA']
    if len(nsa_specs) > 1:
        logging.warning('More than one NSA obj near ({}, {}): {}'.format(nsa_specs['RA'][0], nsa_specs['DEC'][0], ', '.join(nsa_specs['SPECOBJID'])))
    nsa_id = int(nsa_specs['SPECOBJID'][0]) if len(nsa_specs) else -1

    zq_thres = 3 if (specs['ZQUALITY'].max() >= 3) else 0
    spec_repeat = '+'.join(set(specs['TELNAME'][specs['ZQUALITY'] >= zq_thres]))

    return rank.argmax(), spec_repeat, nsa_id


def merge_spectra(specs):
    if 'coord' in specs.colnames:
        del specs['coord']

    specs['group_id'] = get_fof_group_id(specs, 20.0)
    specs.sort(['group_id', 'SPEC_Z'])
    group_id_shift = specs['group_id'][-1] + 1

    edge_mask = (np.ediff1d(specs['group_id']) > 0) | (np.ediff1d(specs['SPEC_Z']) > 150.0/_SPEED_OF_LIGHT)
    group_id_edges = np.flatnonzero(np.hstack(([True], edge_mask, [True])))
    regroup_mask = np.zeros(len(specs), np.bool)
    for i, j in zip(group_id_edges[:-1], group_id_edges[1:]):
        if j - i > 1 and (np.median(specs['SPEC_Z'][i:j]) > 0.2 or np.count_nonzero(specs['TELNAME'][i:j] == 'NSA') > 1):
            regroup_mask[i:j] = True

    if regroup_mask.any():
        specs['group_id'][regroup_mask] = get_fof_group_id(specs[regroup_mask], 10.0) + group_id_shift
        specs.sort(['group_id', 'SPEC_Z'])
        edge_mask = (np.ediff1d(specs['group_id']) > 0) | (np.ediff1d(specs['SPEC_Z']) > 150.0/_SPEED_OF_LIGHT)
        group_id_edges = np.flatnonzero(np.hstack(([True], edge_mask, [True])))

    del regroup_mask, edge_mask

    specs['SPEC_REPEAT'] = get_empty_str_array(len(specs), 48)
    specs['OBJ_NSAID'] = np.int32(-1)
    specs['chosen'] = False
    for i, j in zip(group_id_edges[:-1], group_id_edges[1:]):
        if j - i > 1:
            k, spec_repeat, nsa_id = find_best_spec(specs[i:j])
        else:
            k = 0
            spec_repeat = specs['TELNAME'][i]
            nsa_id = int(specs['SPECOBJID'][i]) if spec_repeat == 'NSA' else -1
        k += i
        specs['chosen'][k] = True
        specs['SPEC_REPEAT'][k] = spec_repeat
        specs['OBJ_NSAID'][k] = nsa_id

    specs = Query('chosen').filter(specs)
    del specs['chosen'], specs['group_id']

    return specs


def add_columns_for_spectra(base):
    base['OBJ_NSAID'] = np.int32(-1)
    base['SPEC_REPEAT'] = get_empty_str_array(len(base), 48)
    cols_definition = SPECS_COLUMNS.copy()
    for col in ('RA', 'DEC'):
        cols_definition[col+'_spec'] = cols_definition[col]
        del cols_definition[col]
    base = ensure_specs_dtype(base, cols_definition)
    return base


def add_spectra(base, specs):
    if 'coord' in specs.colnames:
        del specs['coord']
    specs.sort(['ZQUALITY', 'SPEC_Z'])
    specs.reverse()
    specs = add_skycoord(specs)

    specs_idx, base_idx, sep, _ = search_around_sky(specs['coord'], base['coord'], 20.0*astropy.units.arcsec) # pylint: disable=E1101
    sep = sep.arcsec

    if (np.ediff1d(specs_idx) < 0).any():
        sorter = specs_idx.argsort()
        specs_idx = specs_idx[sorter]
        base_idx = base_idx[sorter]
        sep = sep[sorter]
        del sorter

    base['index'] = np.arange(len(base))
    base['no_spec_yet'] = True
    specs['matched_idx'] = -1
    specs_idx_edges = np.flatnonzero(np.hstack(([1], np.ediff1d(specs_idx), [1])))
    for i, j in zip(specs_idx_edges[:-1], specs_idx_edges[1:]):
        spec_idx_this = specs_idx[i]
        possible_match = base['REMOVE', 'is_galaxy', 'r_mag', 'no_spec_yet', 'index'][base_idx[i:j]]
        possible_match['sep'] = sep[i:j]

        if Query('sep < 1', ~Query('no_spec_yet')).count(possible_match) > 0:
            specs['matched_idx'][spec_idx_this] = -2

        possible_match = possible_match[possible_match['no_spec_yet']]
        if not len(possible_match):
            continue
        possible_match.sort('sep')

        larger_search_r = build._get_spec_search_radius(specs['SPEC_Z'][spec_idx_this])
        for q in (
                Query('REMOVE == 0', 'sep < 1'),
                Query('REMOVE == 0', 'is_galaxy', 'sep < 3'),
                Query('REMOVE == 0', ~Query('is_galaxy'), 'sep < 3'),
                Query('REMOVE == 0', 'is_galaxy', 'r_mag < 17', 'sep < {:g}'.format(larger_search_r)),
                Query('REMOVE > 0', 'sep < 3'),
                Query('sep >= 3', 'sep < 4'),
        ):
            mask = q.mask(possible_match)
            if mask.any():
                matched_base_idx = possible_match['index'][np.flatnonzero(mask)[0]]
                specs['matched_idx'][spec_idx_this] = matched_base_idx
                base['no_spec_yet'][matched_base_idx] = False
                break

    del base['index'], base['no_spec_yet'], specs['coord'], specs_idx, base_idx, sep, specs_idx_edges

    specs.sort('matched_idx')
    start_idx = np.flatnonzero(specs['matched_idx'] > -1)[0]
    for col in tuple(SPECS_COLUMNS) + ('SPEC_REPEAT', 'OBJ_NSAID'):
        col_base = (col + '_spec') if col in ('RA', 'DEC') else col
        base[col_base][specs['matched_idx'][start_idx:]] = specs[col][start_idx:]

    specs_warn = specs[np.flatnonzero(specs['matched_idx'] > -2)[0]:start_idx]
    specs_warn = specs_warn[specs_warn['SPEC_REPEAT'] != 'GAMA']
    if len(specs_warn) > 100:
        logging.warning('More than 100 spec objects have no match... something is very wrong!')
    else:
        for spec in specs_warn:
            logging.warning('No photo obj matched to {} spec obj {} ({}, {})'.format(spec['TELNAME'], spec['SPECOBJID'], spec['RA'], spec['DEC']))

    del specs['matched_idx']

    return base


def remove_shreds_near_spec_obj(base, nsa=None):

    has_nsa = Query('OBJ_NSAID > -1')
    is_high_z = Query('SPEC_Z > 0.05', 'ZQUALITY >= 3', 'is_galaxy', 'radius > abs(radius_err) * 2.0', ~has_nsa)

    has_nsa_indices = np.flatnonzero(has_nsa.mask(base))
    has_nsa_indices = has_nsa_indices[base['r_mag'][has_nsa_indices].argsort()]

    is_high_z_indices = np.flatnonzero(is_high_z.mask(base))
    is_high_z_indices = is_high_z_indices[base['r_mag'][is_high_z_indices].argsort()]

    for idx in chain(has_nsa_indices, is_high_z_indices):
        obj_this = base[idx]

        if nsa is not None and obj_this['OBJ_NSAID'] > -1:
            nsa_obj = Query('NSAID == {}'.format(obj_this['OBJ_NSAID'])).filter(nsa)[0]
            ellipse_calculation = dict()
            ellipse_calculation['a'] = nsa_obj['PETRO_TH90'] * 2.0 / 3600.0
            ellipse_calculation['b'] = ellipse_calculation['a'] * nsa_obj['PETRO_BA90']
            ellipse_calculation['t'] = np.deg2rad(nsa_obj['PETRO_PHI90'] + 270.0)
            ellipse_calculation['s'] = np.sin(ellipse_calculation['t'])
            ellipse_calculation['c'] = np.cos(ellipse_calculation['t'])
            ellipse_calculation['x'] = base['RA'] - nsa_obj['RA']
            ellipse_calculation['y'] = base['DEC'] - nsa_obj['DEC']
            nearby_obj_mask = ne.evaluate('((x*c - y*s)/a)**2.0 + ((x*s + y*c)/b)**2.0 < 1.0',
                                          local_dict=ellipse_calculation, global_dict={})

            no_spec_z_or_close = Query('ZQUALITY < 3')
            no_spec_z_or_close |= Query((lambda z: np.fabs(z - nsa_obj['Z']) < 200.0/_SPEED_OF_LIGHT, 'SPEC_Z'))
            no_spec_z_or_close |= Query((lambda z: np.fabs(z - obj_this['SPEC_Z']) < 200.0/_SPEED_OF_LIGHT, 'SPEC_Z'))
            nearby_obj_mask &= no_spec_z_or_close.mask(base)

            remove_flag = 21

            values_to_rewrite = {
                'OBJID': nsa_obj['NSAID'],
                'REMOVE': 0,
                'is_galaxy': (nsa_obj['PETRO_TH50'] > 1),
                'RA': nsa_obj['RA'],
                'DEC': nsa_obj['DEC'],
                'radius': nsa_obj['PETRO_TH50'],
                'radius_err': 0,
                'survey': 'NSA',
            }

            invalid_mag = (nsa_obj['SERSIC_FLUX'] <= 0)
            nsa_sersic_flux = np.array(nsa_obj['SERSIC_FLUX'])
            nsa_sersic_flux[invalid_mag] = 1.0

            mag = 22.5 - 2.5 * np.log10(nsa_sersic_flux)
            mag_err = np.fabs((2.5/np.log(10.0))/nsa_sersic_flux/np.sqrt(nsa_obj['SERSIC_FLUX_IVAR']))
            mag[invalid_mag] = 99.0
            mag_err[invalid_mag] = 99.0

            for i, b in enumerate(get_sdss_bands()):
                values_to_rewrite['{}_mag'.format(b)] = mag[i+2] - nsa_obj['EXTINCTION'][i+2]
                values_to_rewrite['{}_err'.format(b)] = mag_err[i+2]

            for k, v in values_to_rewrite.items():
                base[k][idx] = v

        elif obj_this['REMOVE'] > 0:
            continue

        else:
            remove_radius = 2.0 * obj_this['radius']
            nearby_obj_mask = (base['coord'].separation(obj_this['coord']).arcsec < remove_radius)
            remove_flag = 22

        nearby_obj_mask[idx] = False
        nearby_obj_count = np.count_nonzero(nearby_obj_mask)

        if not nearby_obj_count:
            continue

        if nearby_obj_count > 25 and remove_flag == 22:
            logging.warning('More than 25 photo obj within ~ {:.3f}" of {} spec obj {} ({}, {})'.format(remove_radius, obj_this['TELNAME'], obj_this['OBJID'], obj_this['RA'], obj_this['DEC']))

        base['REMOVE'][nearby_obj_mask] += (1 << remove_flag)

    return base


def add_surface_brightness(base):
    base['sb_r'] = base['r_mag'] + 2.5 * np.log10(np.maximum(2.0*np.pi*base['radius']**2.0, 1.0e-40))
    return base


def build_full_stack(host, sdss=None, des=None, decals=None, nsa=None,
                     sdss_remove=None, sdss_recover=None,
                     des_remove=None, des_recover=None,
                     spectra=None, debug=None, **kwargs):
    """
    This function calls all needed functions to complete the full stack of building
    a base catalog (for a single host), in the following order:

    Returns
    -------
    base : astropy.table.Table
    """
    if sdss is None and des is None and decals is None:
        raise ValueError('No photometry catalog to build!')

    all_spectra = []

    if sdss is not None:
        all_spectra.append(extract_sdss_spectra(sdss))
        sdss = prepare_sdss_catalog_for_merging(sdss, sdss_remove, sdss_recover)

    if des is not None:
        des = prepare_des_catalog_for_merging(des, des_remove, des_recover)

    if decals is not None:
        decals = prepare_decals_catalog_for_merging(decals)

    if nsa is not None:
        nsa = filter_nearby_object(nsa, host)
        if len(nsa):
            all_spectra.append(extract_nsa_spectra(nsa))
        else:
            nsa = None

    if spectra is not None:
        spectra = filter_nearby_object(spectra, host)
        if len(spectra):
            all_spectra.append(spectra)

    base = merge_catalogs(sdss=sdss, des=des, decals=decals)
    if sdss is not None and decals is not None:
        base = replace_poor_sdss_sky_subtraction(base)

    base = build.add_host_info(base, host)
    del sdss, des, decals, spectra

    base = add_columns_for_spectra(base)
    if all_spectra:
        all_spectra = vstack(all_spectra, 'exact', 'error')
        all_spectra = merge_spectra(all_spectra)
        if debug is not None:
            debug['all_spectra'] = all_spectra.copy()
        base = add_spectra(base, all_spectra)
        del all_spectra
        base = remove_shreds_near_spec_obj(base, nsa)
        del nsa

    base['REMOVE'][Query('RHOST_KPC < 10.0').mask(base)] += (1 << 20)
    base = add_surface_brightness(base)
    base = build.find_satellites(base, version=2)
    base = build.add_stellar_mass(base)

    return base
