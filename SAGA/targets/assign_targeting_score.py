import numpy as np
from easyquery import Query
from ..objects import cuts as C
from ..utils import fill_values_by_query, get_empty_str_array
from .gmm import calc_satellite_probability

def assign_targeting_score(base, manual_selected_objids=None,
                           gmm_parameters=None, weight_func_parameters=None):

    if 'TARGETING_LABEL' in base.columns:
        base['TARGETING_LABEL'] = ''
    else:
        base['TARGETING_LABEL'] = get_empty_str_array(len(base))
    base['TARGETING_SCORE'] = 200.0

    fill_values_by_query(base, C.gri_cut, {'TARGETING_LABEL':'WITHIN_GRI_CUT', 'TARGETING_SCORE': 100.0})

    if weight_func_parameters:
        pass

    if gmm_parameters:
        p = calc_satellite_probability(base, gmm_parameters)
        p_mask = (p > 0.5)
        base['TARGETING_LABEL'][p_mask] = 'HIGH_P_GMM'
        base['TARGETING_SCORE'][p_mask] = 3.0 - p[p_mask]

    if manual_selected_objids:
        fill_values_by_query(base, \
                Query((lambda x: np.in1d(x, manual_selected_objids), 'OBJID')), \
                {'TARGETING_LABEL':'MANUAL', 'TARGETING_SCORE': 1.0})

    fill_values_by_query(base, C.sdss_limit, \
            {'TARGETING_LABEL':'BRIGHT', 'TARGETING_SCORE': 0.0})

    fill_values_by_query(base, ~C.basic_cut, \
            {'TARGETING_LABEL':'NOT_WITHIN_BASIC_CUT', 'TARGETING_SCORE': 300.0})

    fill_values_by_query(base, C.has_spec, \
            {'TARGETING_LABEL':'HAS_SPEC', 'TARGETING_SCORE': 400.0})

    base.sort('TARGETING_SCORE')
    return base