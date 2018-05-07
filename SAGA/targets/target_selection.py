"""
TargetSelection class
"""
from itertools import chain
from astropy.table import vstack
from ..objects import ObjectCatalog
from ..objects import cuts as C
from ..utils import get_sdss_bands, get_sdss_colors
from .assign_targeting_score import assign_targeting_score

__all__ = ['TargetSelection']

class TargetSelection(object):
    """
    Parameters
    ----------
    database: SAGA.Database object

    Returns
    -------
    target_selection : SAGA.TargetSelection object

    Examples
    --------
    >>> import SAGA
    >>> from SAGA import ObjectCuts as C
    >>> saga_database = SAGA.Database('/path/to/SAGA/Dropbox')
    >>> saga_targets = SAGA.TargetSelection(saga_database, gmm_parameters='gmm_parameters_no_outlier')
    >>> hosts = [161174, 52773, 163956, 69028, 144953, 165082, 165707, 145729, 165980, 147606]
    >>> saga_targets.load_object_catalogs(hosts, (C.gri_cut & C.fibermag_r_cut & C.is_galaxy & C.is_clean))
    >>> score_bins = [150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    >>> d = np.array([np.searchsorted(base['TARGETING_SCORE'], score_bins) for base in saga_targets.compile_target_list('iter')])
    """
    def __init__(self, database, assign_targeting_score_func=None,
                 manual_selected_objids=None, gmm_parameters=None,
                 additional_columns=None):
        self._database = database
        self._objects = ObjectCatalog(self._database)
        self.object_catalog_cache = None

        self._assign_targeting_score = assign_targeting_score_func or assign_targeting_score
        if not callable(self._assign_targeting_score):
            raise TypeError('*assign_targeting_score_func* must be callable')

        try:
            self._manual_selected_objids = self._database[manual_selected_objids or 'manual_targets'].read()
        except (TypeError, KeyError):
            self._manual_selected_objids = manual_selected_objids

        try:
            self._gmm_parameters = self._database[gmm_parameters or 'gmm_parameters'].read()
        except (TypeError, KeyError):
            self._gmm_parameters = gmm_parameters

        self._additional_columns = list(additional_columns or [])
        self.columns = list(set(chain(('OBJID', 'RA', 'DEC'),
                                      C.COLUMNS_USED,
                                      map('{}_mag'.format, get_sdss_bands()),
                                      map('{}_err'.format, get_sdss_bands()),
                                      get_sdss_colors(),
                                      map('{}_err'.format, get_sdss_colors()),
                                      self._additional_columns)))


    def load_object_catalogs(self, hosts=None, cuts=None):
        """
        load object catalogs (aka "base catalogs")

        Parameters
        ----------
        hosts : int, str, list, None, optional
            host names/IDs or a list of host names/IDs or short-hand names like
            "paper1" or "paper1_complete"

        cuts : easyquery.Query, str, tuple, optional
            To apply to the objects when loaded
        """
        self.object_catalog_cache = self._objects.load(hosts=hosts, cuts=cuts, columns=self.columns, return_as='list')


    def clear_object_catalogs(self):
        """
        clear object catalog cache
        """
        self.object_catalog_cache = None


    def compile_target_list(self, return_as='list'):
        """
        Compile target list. Must run `load_base_catalogs` first.

        Parameters
        ----------
        return_as : str, optional
            If set to 'list' (default), return a list that contains all tables
            If set to 'stacked', return a stacked table
            If set to 'iter', return an iterator for looping over hosts
        """
        if self.object_catalog_cache is None:
            raise RuntimeError('Must call `load_base_catalogs` first')

        return_as = return_as.lower()
        if return_as[0] not in 'sli':
            raise ValueError('`return_as` should be "list", "stacked", or "iter"')

        columns = ['OBJID', 'RA', 'DEC', 'r_mag', 'TARGETING_SCORE']
        columns.extend((c for c in self._additional_columns if c not in columns))

        output_iter = (self._assign_targeting_score(base, self._manual_selected_objids, self._gmm_parameters)[columns] \
                for base in self.object_catalog_cache)

        if return_as[0] == 'i':
            return output_iter
        elif return_as[0] == 'l':
            return list(output_iter)
        elif return_as[0] == 's':
            return vstack(list(output_iter))
