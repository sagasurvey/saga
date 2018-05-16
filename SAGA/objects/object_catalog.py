import time
import numpy as np
from astropy.table import vstack
from easyquery import Query
from . import cuts as C
from .build import build_full_stack, WISE_COLS_USED, NSA_COLS_USED
from .manual_fixes import fixes_to_nsa_v012
from ..database import FitsTable, Database
from ..hosts import HostCatalog
from ..utils import get_sdss_bands, get_sdss_colors, add_skycoord, fill_values_by_query


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

    def __init__(self, database=None):
        self._database = database or Database()
        self._host_catalog = HostCatalog(self._database)


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
                host_ids = self._host_catalog.resolve_id(hosts, 'NSA')
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

            hosts = self._host_catalog.resolve_id(hosts or 'all', 'string')

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


    def load_nsa(self, version='0.1.2'):
        nsa = self._database['nsa_v{}'.format(version)].read()[NSA_COLS_USED]
        if version == '0.1.2':
            for nsaid, fixes in fixes_to_nsa_v012.items():
                fill_values_by_query(nsa, 'NSAID == {}'.format(nsaid), fixes)
            # NSA 64408 (127.324917502, 25.75292055) is wrong! For v0.1.2 ONLY!!
            nsa = Query('NSAID != 64408').filter(nsa)
        nsa = add_skycoord(nsa)
        return nsa


    def build_and_write_to_database(self, hosts=None, overwrite=False, base_file_path_pattern=None):
        """
        This function build base catalog and write to the database.

        !! IMPORTANT !!
        If you want to write the base catalog to an alternative location (not the database)
        Make sure you set the base_file_path_pattern option!!

        Parameters
        ----------
        hosts : int, str, list, None, optional
            host names/IDs or a list of host names/IDs or short-hand names like
            "paper1" or "paper1_complete"

        overwrite : bool, optional
            If set to True, overwrite existing base catalog

        base_file_path_pattern : str, optional

        Examples
        --------
        >>> saga_database = SAGA.Database('/path/to/SAGA/Dropbox')

        You need to set some local paths first
        >>> saga_database.sdss_file_path_pattern = '/path/to/sdss/nsa{}.fits.gz'
        >>> saga_database.wise_file_path_pattern = '/path/to/wise/nsa{}.fits.gz'
        >>> saga_database['spectra_gama'].local = '/path/to/gama/SpecObj.fits'
        >>> saga_database['nsa_v0.1.2'].local = '/path/to/nsa_v0_1_2.fits'

        >>> saga_object_catalog = SAGA.ObjectCatalog(saga_database)

        Overwrite the database (Danger!!)
        >>> saga_object_catalog.build_and_write_to_database('paper1', overwrite=True)

        You can also do
        >>> saga_object_catalog.build_and_write_to_database('paper1', base_file_path_pattern='/other/base/catalog/dir/nsa{}.fits.gz')

        """

        nsa = self.load_nsa('0.1.2')
        spectra_raw_all = self._database['spectra_raw_all'].read()
        host_ids = self._host_catalog.resolve_id(hosts or 'all', 'string')

        for i, host_id in enumerate(host_ids):

            if base_file_path_pattern is None:
                data_obj = self._database['base', host_id].remote
            else:
                data_obj = FitsTable(base_file_path_pattern.format(host_id))

            if data_obj.isfile() and not overwrite:
                print(time.strftime('[%m/%d %H:%M:%S]'), 'Base catalog of {} already exists.'.format(host_id), '({}/{})'.format(i+1, len(host_ids)))
                continue

            print(time.strftime('[%m/%d %H:%M:%S]'), 'Building base catalog for {}'.format(host_id), '({}/{})'.format(i+1, len(host_ids)))
            try:
                wise = self._database['wise', host_id].read()[WISE_COLS_USED]
            except OSError:
                wise = None

            base = build_full_stack(self._database['sdss', host_id].read(),
                                    self._host_catalog.load_single(host_id),
                                    self._database['hosts_named'].read(), wise, nsa,
                                    self._database['objects_to_remove'].read(),
                                    self._database['objects_to_add'].read(),
                                    spectra_raw_all)

            print(time.strftime('[%m/%d %H:%M:%S]'), 'Writing base catalog to {}'.format(data_obj.path))
            data_obj.write(base)

        #TODO: extract all cleaned specs!
