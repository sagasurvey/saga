from astropy.table import vstack
from ..objects import ObjectCatalog
from ..objects import cuts as C
from ..utils import get_sdss_colors

from .assign_targeting_score import assign_targeting_score

_colors = ['ug', 'gr', 'ri', 'iz']

class TargetSelection(object):
    def __init__(self, database):
        self._database = database
        self._objects = ObjectCatalog(self._database)

    def compile_preliminary_targets(self, hosts=None, return_as='list', return_columns=('OBJID', 'RA', 'DEC')):

        return_as = return_as.lower()
        if return_as[0] not in 'sli':
            raise ValueError('`return_as` should be "list", "stacked", or "iter"')

        columns_in = list(set(list(return_columns) + get_sdss_colors() \
                + ['{}_err'.format(c) for c in get_sdss_colors()] \
                + ['OBJID', 'RA', 'DEC'] + C.COLUMNS_USED))

        columns_out = list(return_columns) + ['TARGETING_LABEL', 'TARGETING_SCORE']

        manual_selected_objids = None #np.unique(self._database['manual_targets'].read()['OBJID'])
        gmm_parameters = self._database['gmm_parameters'].read()
        weight_func_parameters = None #TODO

        output_iter = (assign_targeting_score(base, manual_selected_objids, gmm_parameters, weight_func_parameters)[columns_out]\
                for base in self._objects.load(hosts=hosts, columns=columns_in, return_as='iter'))

        if return_as[0] == 'i':
            return output_iter
        elif return_as[0] == 'l':
            return list(output_iter)
        elif return_as[0] == 's':
            return vstack(list(output_iter))
