"""
module assign_targeting_score
"""
from itertools import chain
import numpy as np
from easyquery import Query
from ..objects import cuts as C
from ..utils import fill_values_by_query, get_sdss_bands, get_sdss_colors
from .gmm import calc_gmm_satellite_probability, calc_log_likelihood, get_input_data, param_labels_nosat

__all__ = ['assign_targeting_score', 'calc_simple_satellite_probability', 'calc_gmm_satellite_probability']


COLUMNS_USED = list(set(chain(C.COLUMNS_USED,
                              map('{}_mag'.format, get_sdss_bands()),
                              map('{}_err'.format, get_sdss_bands()),
                              get_sdss_colors(),
                              map('{}_err'.format, get_sdss_colors()))))


def calc_simple_satellite_probability(base,
                                      model_parameters=(-0.84526783,
                                                        -0.53434289,
                                                        -1.0123662441968917,
                                                        0.18628167890581865,
                                                        9.4021013202593942,
                                                        0.055890285233031099)):
    x = np.asarray(base['gr'])*model_parameters[0] + np.asarray(base['ri'])*model_parameters[1]
    return np.where(x > model_parameters[2], np.minimum(np.exp((x-model_parameters[3])*model_parameters[4]), model_parameters[5]), 0.0)


def ensure_proper_prob(p):
    p[np.isnan(p)] = 0
    p[p > 1] = 1.0
    p[p < 0] = 0.0
    return p


def assign_targeting_score(base, manual_selected_objids=None,
                           gmm_parameters=None):
    """
    Last updated: 05/07/2018
     100 Human selection and Special targets
     150 satellites
     200 within host,  r < 17.77, gri cuts
     300 within host,  r < 20.75, high p_GMM or GMM outliers, gri cuts
     400 within host,  r < 20.75, high-proirity + gri cuts
     500 outwith host, r < 17.77, gri cuts
     600 within host,  r < 20.75, gri cuts, random selection of 50
     700 very high p_GMM, gri cuts
     800 within host,  r < 20.75, gri cuts, everything else
     900 outwith host, r < 20.75, gri cuts
    1000 everything else
    1100 Not in gri/fibermag_r_cut
    1200 Not galaxy
    1300 Not clean
    1400 Has spec but not a satellite
    """

    base['P_simple'] = ensure_proper_prob(calc_simple_satellite_probability(base))
    base['P_GMM_sdss'] = ensure_proper_prob(calc_gmm_satellite_probability(base, gmm_parameters))
    base['P_GMM'] = base['P_GMM_sdss']
    base['log_L_GMM'] = calc_log_likelihood(*get_input_data(base), *(gmm_parameters[n] for n in param_labels_nosat))

    basic_cut = C.gri_cut & C.fibermag_r_cut & C.is_clean & C.is_galaxy & (~C.has_spec)
    within_host =  basic_cut & C.faint_end_limit & C.sat_rcut
    outwith_host = basic_cut & C.faint_end_limit & (~C.sat_rcut)

    veryhigh_p = Query('P_GMM_sdss >= 0.95', 'log_L_GMM >= -7')
    high_p = Query('P_GMM_sdss >= 0.6', 'log_L_GMM >= -7') | Query('log_L_GMM < -7', 'ri-abs(ri_err) < -0.25')
    median_p = Query('-(ug+abs(ug_err))*0.15+(ri-abs(ri_err)) < 0.08',
                     '(gr-abs(gr_err))*0.65+(ri-abs(ri_err)) < 0.6',
                     '-(ug+abs(ug_err))*0.1+(gr-abs(gr_err)) < 0.5')

    base['TARGETING_SCORE'] = 1000
    fill_values_by_query(base, ~basic_cut, {'TARGETING_SCORE': 1100})
    fill_values_by_query(base, ~C.is_galaxy, {'TARGETING_SCORE': 1200})
    fill_values_by_query(base, ~C.is_clean, {'TARGETING_SCORE': 1300})
    fill_values_by_query(base, C.has_spec, {'TARGETING_SCORE': 1400})

    fill_values_by_query(base, outwith_host, {'TARGETING_SCORE': 900})
    fill_values_by_query(base, within_host, {'TARGETING_SCORE': 800})
    fill_values_by_query(base, basic_cut & veryhigh_p & (~C.sdss_limit), {'TARGETING_SCORE': 700})
    fill_values_by_query(base, outwith_host & C.sdss_limit, {'TARGETING_SCORE': 500})
    fill_values_by_query(base, within_host & median_p, {'TARGETING_SCORE': 400})
    fill_values_by_query(base, within_host & high_p, {'TARGETING_SCORE': 300})
    fill_values_by_query(base, within_host & C.sdss_limit, {'TARGETING_SCORE': 200})

    need_random_selection = np.flatnonzero(base['TARGETING_SCORE'] == 800)
    if len(need_random_selection) > 50:
        random_mask = np.zeros(len(need_random_selection), dtype=np.bool)
        random_mask[:50] = True
        np.random.RandomState(123).shuffle(random_mask)
        need_random_selection = need_random_selection[random_mask]
    base['TARGETING_SCORE'][need_random_selection] = 600

    base['TARGETING_SCORE'] += (np.round((1.0 - base['P_GMM_sdss'])*80.0).astype(np.int) + 10)

    fill_values_by_query(base, C.is_sat, {'TARGETING_SCORE': 150})

    if manual_selected_objids:
        fill_values_by_query(base, \
                Query((lambda x: np.in1d(x, manual_selected_objids), 'OBJID')), \
                {'TARGETING_SCORE': 100})

    base.sort('TARGETING_SCORE')
    return base
