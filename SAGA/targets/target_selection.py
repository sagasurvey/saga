from easyquery import Query
from ..objects import ObjectCatalog
from ..objects import cuts as C
from ..utils import fill_values_by_query, get_empty_str_array
from .gmm import calc_satellite_probability

_colors = ['ug', 'gr', 'ri', 'iz']

class TargetSelection(object):
    def __init__(self, database):
        self._database = database
        self._objects = ObjectCatalog(self._database)

    def select_preliminary_targets(self, host_id, return_columns=['OBJID', 'RA', 'DEC']):
        columns = list(set(return_columns + _colors + ['{}_err'.format(c) for c in _colors] + ['OBJID']))

        base = self._objects.load(hosts=host_id, has_spec=False, cuts=C.basic_cut, columns=columns)

        base['TARGETING_LABEL'] = get_empty_str_array(len(base))
        base['TARGETING_SCORE'] = 9999.0

        n_bright = fill_values_by_query(base, C.sdss_limit, {'TARGETING_LABEL':'BRIGHT', 'TARGETING_SCORE': 0.0})

        risa_objid = self._database._table['risa_objects'].read()['OBJID']
        n_risa = fill_values_by_query(base,  Query((lambda x: np.in1d(x, risa_objid), 'OBJID')),
                                      {'TARGETING_LABEL':'RISA', 'TARGETING_SCORE': 1.0})

        p = calc_satellite_probability(base, self._database._table['gmm_model_para'].read())
        p_mask = (p > 0.5)
        base['TARGETING_LABEL'][p_mask] = 'HIGH_P_GMM'
        base['TARGETING_SCORE'][p_mask] = 3.0 - p[p_mask]

        #TODO: finish this

        return base
