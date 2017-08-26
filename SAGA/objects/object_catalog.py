import numpy as np
from astropy.table import vstack
from easyquery import Query
from . import cuts as C
from ..hosts import HostCatalog


def _slice_columns(table, columns):
    return table[columns] if columns is not None else table


class ObjectCatalog(object):
    """
    This class provides a high-level interface to access object catalogs
    (also known as "base catalogs")

    Parameters
    ----------
    database : SAGA.Database object

    Returns
    -------
    object_catalog : SAGA.ObjectCatalog object

    Examples
    --------
    >>> import SAGA
    >>> saga_database = SAGA.Database('/path/to/SAGA/Dropbox')
    >>> saga_objects = SAGA.ObjectCatalog(saga_database)
    >>> specs = saga_objects.load(has_spec=True)
    >>> base_anak = saga_objects.load(hosts='AnaK')

    Here specs and base_anak are both astropy tables.
    """
    def __init__(self, database):
        self._database = database
        self._hosts = HostCatalog(self._database)


    @staticmethod
    def _add_colors(table):
        sdss_bands = 'ugriz'
        for b in sdss_bands:
            table['{}_mag'.format(b)] = table[b] - table['EXTINCTION_{}'.format(b.upper())]

        for color in map(''.join, zip(sdss_bands[:-1], sdss_bands[1:])):
            table[color] = table['{}_mag'.format(color[0])] - table['{}_mag'.format(color[1])]
            table['{}_err'.format(color)] = np.sqrt(table['{}_err'.format(color[0])]**2.0 + table['{}_err'.format(color[1])]**2.0)

        return table


    def load(self, hosts=None, has_spec=None, cuts=None, iter_hosts=False, columns=None):
        """
        load object catalogs (aka "base catalogs")

        Parameters
        ----------
        hosts : int, str, list, None, optional
            host names/IDs or a list of host names/IDs or short-hand names like
            "paper1" or "paper1_complete"

        has_spec : bool, optional
            If set to True, load only objects that have spectra

        cuts : easyquery.Query, str, tuple, optional
            To apply to the objects when loaded

        iter_hosts : bool, optional
            If set to True, return an iterator for looping over hosts

        columns : list, optional
            If set, only load a subset of columns

        Returns
        -------
        objects : astropy.table.Table

        Examples
        --------
        >>> import SAGA
        >>> from SAGA import ObjectCuts as C
        >>> saga_database = SAGA.Database('/path/to/SAGA/Dropbox')
        >>> saga_objects = SAGA.ObjectCatalog(saga_database)

        To load all spectra, with some basic cuts applied:
        >>> specs = saga_objects.load(has_spec=True, cuts=C.basic_cut)

        Load the base catalog for a certain host, with some basic cuts applied:
        >>> specs = saga_objects.load(hosts='AnaK', cuts=C.basic_cut)

        Load base catalog for all paper1 hosts, with some basic cuts applied,
        and stored as a list:
        >>> base_tables = list(saga_objects.load(hosts='paper1', cuts=C.basic_cut, iter_hosts=True))

        Load base catalog for all paper1 hosts, with some basic cuts applied,
        and stored as one single big table:
        >>> bases_table = saga_objects.load(hosts='paper1', cuts=C.basic_cut)
        """
        if has_spec:
            t = self._database['spectra_clean'].read()

            if hosts is not None:
                host_ids = self._hosts.resolve_id(hosts)
                t = Query((lambda x: np.in1d(x, host_ids), 'HOST_NSAID')).filter(t)

            t = self._add_colors(t)

            if cuts is not None:
                t = Query(cuts).filter(t)

            if iter_hosts:
                if hosts is None:
                    host_ids = np.unique(t['HOST_NSAID'])
                return (Query('HOST_NSAID == {}'.format(i)).filter(t) for i in host_ids)
            else:
                return _slice_columns(t, columns)

        else:
            q = Query(cuts)
            if has_spec is not None:
                q = q & (~C.has_spec)

            hosts = self._hosts.resolve_id('all') if hosts is None else self._hosts.resolve_id(hosts)

            output_iterator = (_slice_columns(q.filter(self._add_colors(self._database['base', host].read())), columns) for host in hosts)

            return output_iterator if iter_hosts else vstack(list(output_iterator))


    def build(self, hosts=None, rebuild=False):
        raise NotImplementedError #TODO: implement this
