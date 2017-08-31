import numpy as np
from astropy.table import vstack
from astropy.coordinates import SkyCoord
from easyquery import Query
from . import cuts as C
from ..hosts import HostCatalog
from ..utils import get_sdss_bands, get_sdss_colors, add_skycoord

__all__ = ['ObjectCatalog']


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
    >>> saga_object_catalog = SAGA.ObjectCatalog(saga_database)
    >>> specs = saga_object_catalog.load(has_spec=True)
    >>> base_anak = saga_object_catalog.load(hosts='AnaK')

    Here specs and base_anak are both astropy tables.
    """

    def __init__(self, database):
        self._database = database
        self._hosts = HostCatalog(self._database)


    @staticmethod
    def _annotate_catalog(table, to_add_skycoord=True):
        for b in get_sdss_bands():
            table['{}_mag'.format(b)] = table[b] - table['EXTINCTION_{}'.format(b.upper())]

        for color in get_sdss_colors():
            table[color] = table['{}_mag'.format(color[0])] - table['{}_mag'.format(color[1])]
            table['{}_err'.format(color)] = np.sqrt(table['{}_err'.format(color[0])]**2.0 + table['{}_err'.format(color[1])]**2.0)

        if to_add_skycoord:
            table = add_skycoord(table)

        return table


    @staticmethod
    def _slice_columns(table, columns, get_coord_later=False):
        if columns is None:
            return table

        if get_coord_later:
            columns_this = list(columns)
            try:
               columns_this.remove('coord')
            except ValueError:
                pass
            if 'RA' not in columns_this:
                columns_this.append('RA')
            if 'DEC' not in columns_this:
                columns_this.append('DEC')

            return table[columns_this]

        return table[columns]


    def load(self, hosts=None, has_spec=None, cuts=None, return_as=None, columns=None):
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

        return_as : str, optional
            If set to 'list' (default when `has_spec` is None), return a list that contains all tables
            If set to 'stacked' (default when `has_spec` is True), return a stacked table
            If set to 'iter', return an iterator for looping over hosts

        columns : list, optional
            If set, only load a subset of columns

        Returns
        -------
        objects : astropy.table.Table, list, or iterator
            (depending on `return_as`)

        Examples
        --------
        >>> import SAGA
        >>> from SAGA import ObjectCuts as C
        >>> saga_database = SAGA.Database('/path/to/SAGA/Dropbox')
        >>> saga_object_catalog = SAGA.ObjectCatalog(saga_database)

        To load all spectra, with some basic cuts applied:
        >>> specs = saga_object_catalog.load(has_spec=True, cuts=C.basic_cut)

        Load the base catalog for a certain host, with some basic cuts applied:
        >>> specs = saga_object_catalog.load(hosts='AnaK', cuts=C.basic_cut)

        Load base catalog for all paper1 hosts, with some basic cuts applied,
        and stored as a list:
        >>> base_tables = saga_object_catalog.load(hosts='paper1', cuts=C.basic_cut, return_as='list')

        Load base catalog for all paper1 hosts, with some basic cuts applied,
        and stored as one single big table:
        >>> bases_table = saga_object_catalog.load(hosts='paper1', cuts=C.basic_cut, return_as='stacked')
        """

        if return_as is None:
            return_as = 'stacked' if has_spec else 'list'
        return_as = return_as.lower()
        if return_as[0] not in 'sli':
            raise ValueError('`return_as` should be "list", "stacked", or "iter"')

        if has_spec:
            t = self._database['spectra_clean'].read()

            if hosts is not None:
                host_ids = self._hosts.resolve_id(hosts)
                t = Query((lambda x: np.in1d(x, host_ids), 'HOST_NSAID')).filter(t)

            t = self._annotate_catalog(t)

            if cuts is not None:
                t = Query(cuts).filter(t)

            if return_as[0] != 's':
                if hosts is None:
                    host_ids = np.unique(t['HOST_NSAID'])
                output_iterator = (self._slice_columns(Query('HOST_NSAID == {}'.format(i)).filter(t), columns) for i in host_ids)
                return output_iterator if return_as[0] == 'i' else list(output_iterator)

            return self._slice_columns(t, columns)

        else:
            q = Query(cuts)
            if has_spec is not None:
                q = q & (~C.has_spec)

            hosts = self._hosts.resolve_id('all') if hosts is None else self._hosts.resolve_id(hosts)

            need_coord = (columns is None or 'coord' in columns)
            to_add_skycoord = (need_coord and return_as[0] != 's') # because skycoord cannot be stacked

            output_iterator = (self._slice_columns(q.filter(self._annotate_catalog(self._database['base', host].read(), to_add_skycoord)), columns, (need_coord and not to_add_skycoord)) for host in hosts)

            if return_as[0] == 'i':
                return output_iterator
            elif return_as[0] == 's':
                out = vstack(list(output_iterator), 'exact', 'error')
                if need_coord:
                    out = self._slice_columns(add_skycoord(out), columns)
                return out
            else:
                return list(output_iterator)


    def build(self, hosts=None, rebuild=False):
        """
        This function is not yet implemented.
        You can do `from SAGA.objects.build import *` to get
        the functions for building object catalogs.
        """
        raise NotImplementedError #TODO: implement this
