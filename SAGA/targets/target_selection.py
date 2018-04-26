from astropy.table import vstack
from ..objects import ObjectCatalog
from ..objects import cuts as C
from ..utils import get_sdss_bands, get_sdss_colors

from .assign_targeting_score import assign_targeting_score

__all__ = ['TargetSelection']

class TargetSelection(object):
    def __init__(self, database):
        self._database = database
        self._objects = ObjectCatalog(self._database)

    def compile_preliminary_targets(self, hosts=None, cuts=None, return_as='list', columns=None):

        return_as = return_as.lower()
        if return_as[0] not in 'sli':
            raise ValueError('`return_as` should be "list", "stacked", or "iter"')

        columns_in = list(set(list(columns) + get_sdss_colors() \
                + ['{}_err'.format(c) for c in get_sdss_colors()] \
                + ['{}_err'.format(b) for b in get_sdss_bands()] \
                + ['OBJID', 'RA', 'DEC'] + C.COLUMNS_USED))

        if columns is None:
            columns_out = None
        else:
            columns_out = list(columns) + ['TARGETING_SCORE', 'P_SIMPLE', 'P_GMM']

        manual_selected_objids = None #np.unique(self._database['manual_targets'].read()['OBJID'])
        gmm_parameters = self._database['gmm_parameters'].read()

        output_iter = (assign_targeting_score(base, manual_selected_objids, gmm_parameters)[columns_out]\
                for base in self._objects.load(hosts=hosts, cuts=cuts, columns=columns_in, return_as='iter'))

        if return_as[0] == 'i':
            return output_iter
        elif return_as[0] == 'l':
            return list(output_iter)
        elif return_as[0] == 's':
            return vstack(list(output_iter))
