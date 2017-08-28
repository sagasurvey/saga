import numpy as np
from easyquery import Query
from ..objects import cuts as C
from ..utils import fill_values_by_query, get_empty_str_array
from .gmm import calc_gmm_satellite_probability

__all__ = ['assign_targeting_score', 'calc_simple_satellite_probability', 'calc_gmm_satellite_probability']


def calc_simple_satellite_probability(base,
        model_parameters=(-0.84526783, -0.53434289, -1.0123662441968917, 0.18628167890581865, 9.4021013202593942, 0.055890285233031099)):
    x = np.asarray(base['gr'])*model_parameters[0] + np.asarray(base['ri'])*model_parameters[1]
    return np.where(x > model_parameters[2], np.minimum(np.exp((x-model_parameters[3])*model_parameters[4]), model_parameters[5]), 0.0)


def assign_targeting_score(base, manual_selected_objids=None,
                           gmm_parameters=None):

    base['TARGETING_SCORE'] = 900.0

    base['P_SIMPLE'] = calc_simple_satellite_probability(base)

    if gmm_parameters:
        base['P_GMM'] = calc_gmm_satellite_probability(base, gmm_parameters)
    else:
        base['P_GMM'] = 0.0

    cut = C.gri_cut & C.faint_end_limit & C.sat_rcut
    high_p = Query('P_SIMPLE > 0.02') | Query('P_GMM > 0.5')

    fill_values_by_query(base, ~C.gri_cut, {'TARGETING_SCORE': 970.0})
    fill_values_by_query(base, ~C.faint_end_limit, {'TARGETING_SCORE': 960.0})
    fill_values_by_query(base, ~C.sat_rcut, {'TARGETING_SCORE': 950.0})

    fill_values_by_query(base, (~C.gri_cut) & high_p, {'TARGETING_SCORE': 700.0})
    fill_values_by_query(base, (~C.faint_end_limit) & high_p, {'TARGETING_SCORE': 600.0})
    fill_values_by_query(base, (~C.sat_rcut) & high_p, {'TARGETING_SCORE': 500.0})

    fill_values_by_query(base, cut & Query('P_SIMPLE > 0.02'), {'TARGETING_SCORE': 400.0})
    fill_values_by_query(base, cut & Query('P_GMM > 0.5'), {'TARGETING_SCORE': 300.0})

    if manual_selected_objids:
        fill_values_by_query(base, \
                Query((lambda x: np.in1d(x, manual_selected_objids), 'OBJID')), \
                {'TARGETING_SCORE': 200.0})

    fill_values_by_query(base, C.sdss_limit & cut, {'TARGETING_SCORE': 100.0})

    fill_values_by_query(base, ~C.is_clean, {'TARGETING_SCORE': 990.0})
    fill_values_by_query(base, C.has_spec, {'TARGETING_SCORE': 999.0})

    mask = Query('TARGETING_SCORE < 900').mask(base)
    base['TARGETING_SCORE'][mask] += (1.0 - base['P_GMM'][mask])

    base.sort('TARGETING_SCORE')
    return base
