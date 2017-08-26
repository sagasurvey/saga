from ..objects import ObjectCatalog
from ..objects import cuts as C
from ..utils import fill_values_by_query

class TargetSelection(object):
    def __init__(self, database):
        self._database = database
        self._objects = ObjectCatalog(self._database)

    def select_preliminary_targets(self, host_id):
        base = self._objects.load(hosts=host_id, query=(Q.basic_cut & ~Q.has_spec))

        base['TARGETING_LABEL'] = get_empty_str_array(len(base))
        base['TARGETING_SCORE'] = 9999.0

        n_bright = fill_values_by_query(base, C.sdss_limit, {'TARGETING_LABEL':'BRIGHT', 'TARGETING_SCORE': 0.0})

        #n_risa = fill_values_by_query(base,  C ,{'TARGETING_LABEL':'RISA', 'TARGETING_SCORE': 1.0})
        #n_high_p =


        return
