"""
SAGA.host.host_catalog

This file defines the HostCatalog class
"""
from collections import defaultdict, Iterable
import numpy as np
from easyquery import Query

from ..database import Database
from ..utils import add_skycoord, find_near_ra_dec

__all__ = ['HostCatalog', 'FieldCatalog']

_paper1_complete_nsa = (166313, 147100, 165536, 61945, 132339, 149781, 33446, 150887)
_paper1_incomplete_nsa = (161174, 85746, 145729, 140594, 126115, 13927, 137625, 129237)
_mmt_2018a_nsa = (161174, 52773, 163956, 69028, 144953, 165082, 165707, 145729, 165980, 147606, 165980, 61945)
_mmt_2018b_nsa = (126115, 129237, 129387, 132339, 149781, 149977, 150307, 150578, 150887, 61945, 169439, 153017)
_aat_2018a_nsa = (3469, 141465, 165082, 145398, 145729, 145879)
_aat_2018a_pgc = (64427, 66318, 66934, 67146, 67663, 67817, 68128, 69521, 70094, 71548, 71729, 2052, 3089)

def _is_string_like(obj):
    """
    Check whether obj behaves like a string.
    """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True


class HostCatalog(object):
    """
    This class provides a high-level interface to access host catalogs
    (also known as "host list" and "master list", while the latter is not supported yet)

    Parameters
    ----------
    database : SAGA.Database object

    Returns
    -------
    host_catalog : SAGA.HostCatalog object

    Examples
    --------
    >>> import SAGA
    >>> saga_database = SAGA.Database('/path/to/SAGA/Dropbox')
    >>> saga_host_catalog = SAGA.HostCatalog(saga_database)
    >>> hosts_no_flag = saga_host_catalog.load('no_flags')
    >>> hosts_no_sdss_flag = saga_host_catalog.load('no_sdss_flags')

    Here hosts_no_flag and hosts_no_sdss_flag are astropy tables.

    >>> saga_host_catalog.resolve_id('AnaK')
    [61945]

    >>> saga_host_catalog.id_to_name(61945)
    'AnaK'
    """

    _predefined_queries = {
        'all': Query(),
        'paper1': Query((lambda x: np.in1d(x, _paper1_complete_nsa + _paper1_incomplete_nsa), 'NSAID')),
        'paper1_complete': Query((lambda x: np.in1d(x, _paper1_complete_nsa), 'NSAID')),
        'paper1_incomplete': Query((lambda x: np.in1d(x, _paper1_incomplete_nsa), 'NSAID')),
        'mmt_2018a': Query((lambda x: np.in1d(x, _mmt_2018a_nsa), 'NSAID')),
        'mmt_2018b': Query((lambda x: np.in1d(x, _mmt_2018b_nsa), 'NSAID')),
        'aat_2018a_has_sdss': Query((lambda x: np.in1d(x, _aat_2018a_nsa), 'NSAID')),
        'aat_2018a_des_only': Query((lambda x: np.in1d(x, _aat_2018a_pgc), 'PGC')),
        'aat_2018a': Query((lambda x, y: np.in1d(x, _aat_2018a_nsa) | np.in1d(y, _aat_2018a_pgc), 'NSAID', 'PGC')),
        'no_flags': Query('flag == 0'),
        'flag0': Query('flag == 0'),
        'has_sdss': Query('flag == 0'),
        'has_nsa': Query('NSAID != -1'),
        'has_deeper_imaging': (Query('decals_dr7 >= 0.95') | Query('decals_dr6 >= 0.95') | Query('des_dr1 >= 0.95')),
        'has_decam': (Query('decals_dr7 >= 0.95') | Query('des_dr1 >= 0.95')),
        'has_des': Query('des_dr1 >= 0.95'),
        'has_des_dr1': Query('des_dr1 >= 0.95'),
        'has_decals': (Query('decals_dr6 >= 0.95') | Query('decals_dr7 >= 0.95')),
        'has_decals_dr5': Query('decals_dr5 >= 0.95'),
        'has_decals_dr6': Query('decals_dr6 >= 0.95'),
        'has_decals_dr7': Query('decals_dr7 >= 0.95'),
        'good': Query(Query('NSAID != -1', (Query('flag == 0') | Query('decals_dr6 >= 0.95') | Query('decals_dr7 >= 0.95'))) | Query('des_dr1 >= 0.95'), 'distance > 22'),
    }


    def __init__(self, database=None):
        self._database = database or Database()
        self._hosts = None
        self._host_index = dict()
        self._index_hosts()


    def _index_hosts(self):
        self._hosts = self._database['hosts'].read()
        index = defaultdict(list)
        for i, n in enumerate(self._hosts['SAGA_name']):
            n = str(n or '')
            if n.strip() and n != '--':
                index[n.strip().replace(' ', '').lower()].append(i)
        for col in ['NSAID', 'NSA1ID', 'PGC', 'NGC', 'UGC']:
            for i, n in enumerate(self._hosts[col]):
                n = int(n)
                if n > -1:
                    index['{}{}'.format(col[:3].lower(), n)].append(i)
                    index[n].append(i)
        self._host_index = {k: tuple(v) for k, v in index.items()}


    def resolve_id(self, hosts, id_to_return='string'):
        """
        Get a list of host IDs from SAGA names or some short-hand names (e.g. 'paper1')

        Currently supports SAGA names and "all", "paper1", paper1_complete",
        and "paper1_incomplete"

        Parameters
        ----------
        hosts : int, str, list
            host names/IDs or a list of host names/IDs
        id_to_return : string

        Returns
        -------
        host_ids : list
            a list of host IDs. The returned value is always a list

        Examples
        --------
        >>> saga_host_catalog.resolve_id('paper1_complete')
        [166313, 147100, 165536, 61945, 132339, 149781, 33446, 150887]

        >>> saga_host_catalog.resolve_id('AnaK')
        [61945]

        """
        indices = []

        if hosts is None:
            hosts = 'all'

        if _is_string_like(hosts):
            hosts = hosts.replace(' ', '').lower()

            if hosts in self._predefined_queries:
                indices = np.flatnonzero(self._predefined_queries[hosts].mask(self._hosts)).tolist()

            else:
                try:
                    hosts = int(hosts)
                except ValueError:
                    pass

                if hosts in self._host_index:
                    indices = list(self._host_index[hosts])

        elif isinstance(hosts, int) and hosts in self._host_index:
            indices = list(self._host_index[hosts])

        elif isinstance(hosts, Iterable):
            for host in hosts:
                indices.extend(self.resolve_id(host, id_to_return='internal'))

        if not indices:
            raise KeyError('Can not find {}'.format(hosts))

        id_to_return = id_to_return.upper()
        if id_to_return[:3] == 'INT':
            return indices
        if id_to_return[:3] in ('STR', 'FIL'):
            return ['nsa{}'.format(self._hosts['NSAID'][i]) if self._hosts['NSAID'][i] != -1 else 'pgc{}'.format(self._hosts['PGC'][i]) for i in indices]
        for start in ('NSA1', 'NSA', 'PGC', 'NGC', 'UGC', 'SAGA'):
            if id_to_return.startswith(start):
                col = start
                if col == 'SAGA':
                    col += '_name'
                if col.startswith('NSA'):
                    col += 'ID'
                return self._hosts[col][indices].tolist()

        raise ValueError('`id_to_return` not known!')


    def id_to_name(self, host_id):
        """
        Get SAGA host name from an ID.

        Parameters
        ----------
        host_id : int

        Returns
        -------
        host_saga_name : str
        """
        names = self.resolve_id(host_id, 'SAGA')
        if len(names) > 1:
            raise ValueError('more than one names found!')
        return names[0] or ''


    def load(self, hosts=None, add_coord=True):
        """
        load a host catalog

        Parameters
        ----------
        host_type : str, optional
            Currently it can be "no_flags", "has_nsa", "has_des", "has_decals",
        reload : bool, optional
            If set to True, do not use tha cached table. Default is False.

        Returns
        -------
        hosts : astropy.table.Table

        Examples
        --------
        >>> hosts_no_flag = saga_host_catalog.load('no_flags')
        >>> hosts_no_sdss_flag = saga_host_catalog.load('no_sdss_flags')
        """
        cat = self._hosts[self.resolve_id('all' if hosts is None else hosts, 'internal')]
        return add_skycoord(cat, dec_label='Dec') if add_coord else cat


    def load_single(self, host, add_coord=True):
        """
        Gets the catalog row corresponding to a specific named host.

        Parameters
        ----------
        host : str, int
            The name or ID of the host, in any form that `resolve_id` understands.

        Returns
        -------
        host_row : astropy.table.Row
            An astropy table row for the requested host.

        Examples
        --------
        >>> anak = saga_host_catalog.load_single('AnaK')
        """
        indices = self.resolve_id(host, 'internal')
        if len(indices) != 1:
            raise ValueError('More than one hosts found!')
        cat = self._hosts[indices]
        cat = add_skycoord(cat, dec_label='Dec') if add_coord else cat
        return cat[0]


    def load_single_near_ra_dec(self, ra, dec):
        """
        ra, dec in degrees
        """
        add_skycoord(self._hosts, dec_label='Dec')
        cat = find_near_ra_dec(self._hosts, ra, dec, 3603.0)
        if len(cat) == 0:
            raise KeyError('No hosts found!')
        if len(cat) != 1:
            raise ValueError('More than one hosts found!')
        return cat[0]


class FieldCatalog(HostCatalog):
    def __init__(self, database=None):
        self._database = database or Database()
        self._fields = self._database['lowz_fields'].read()
        self._hosts = self._fields
        self._field_index = dict(zip(
            (f.replace('_', '').lower() for f in self._fields['field_id']),
            range(len(self._fields))
        ))

    def resolve_id(self, field_ids, id_to_return='string'):
        indices = []

        if field_ids is None or field_ids == 'all':
            indices.extend(range(len(self._fields)))

        elif _is_string_like(field_ids):
            field_ids = field_ids.replace('_', '').lower()
            if field_ids in self._field_index:
                indices.append(self._field_index[field_ids])

        elif isinstance(field_ids, Iterable):
            for field_id in field_ids:
                indices.extend(self.resolve_id(field_id, id_to_return='internal'))

        if not indices:
            raise KeyError('Can not find {}'.format(field_ids))

        id_to_return = id_to_return.upper()
        if id_to_return[:3] == 'INT':
            return indices
        if id_to_return[:3] in ('STR', 'FIL'):
            return self._fields['field_id'][indices].tolist()

        raise ValueError('`id_to_return` not known!')

    def id_to_name(self, host_id):
        raise NotImplementedError
