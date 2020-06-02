"""
SAGA.host.host_catalog

This file defines the HostCatalog class
"""
import logging
import re
from collections import defaultdict

import numpy as np
from astropy.table import join
from easyquery import Query, QueryMaker

from ..database import Database
from ..utils import add_skycoord, find_near_ra_dec
from . import cuts
from .build import build_master_list

__all__ = ["HostCatalog", "FieldCatalog"]


def _is_string_like(obj):
    """
    Check whether obj behaves like a string.
    """
    try:
        obj + ""
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

    _ID_COLNAME = "HOSTID"

    def __init__(self, database=None, version=None, use_master=False):
        self._database = database or Database()
        self._host_table_ = None
        self._host_index_ = None
        self._master_table_ = None
        self._master_index_ = None
        self._version = version
        self.use_master = bool(use_master)

    @property
    def _master_table(self):
        if self._master_table_ is None:
            if self._version == 1:
                self._master_table_ = self._database["master_list_v1"].read()
                if "Dec" in self._master_table_.colnames:
                    self._master_table_.rename_column("Dec", "DEC")
                if "PGC#" in self._master_table_.colnames:
                    self._master_table_.rename_column("PGC#", "PGC")
            else:
                try:
                    self._master_table_ = self._database["master_list"].read()
                except:  # pylint: disable=bare-except # noqa: E722
                    logging.warning(
                        "Cannot load master list; attempt to build from scratch..."
                    )
                    self._master_table_ = self.build_master_list()
                self._master_table_["SAGA_NAME"].fill_value = ""
                self._master_table_ = self._master_table_.filled()
            self._master_index_ = None
        return self._master_table_

    @property
    def _host_table(self):
        if self._host_table_ is None:
            if self._version == 1:
                self._host_table_ = self._database["hosts_v1"].read()
                if "Dec" in self._host_table_.colnames:
                    self._host_table_.rename_column("Dec", "DEC")
                if "SAGA_name" in self._host_table_.colnames:
                    self._host_table_.rename_column("SAGA_name", "SAGA_NAME")
            else:
                try:
                    self._host_table_ = self._database["hosts"].read()
                except:  # pylint: disable=bare-except # noqa: E722
                    logging.warning(
                        "Cannot load host list; attempt to load master list..."
                    )
                    self._host_table_ = cuts.potential_hosts.filter(self._master_table)
                self._host_table_["SAGA_NAME"].fill_value = ""
                self._host_table_ = self._host_table_.filled()
            self._host_index_ = None
        return self._host_table_

    @staticmethod
    def _index_ids(hosts):
        index = defaultdict(set)
        if "SAGA_NAME" in hosts.colnames:
            for i, n in enumerate(hosts["SAGA_NAME"]):
                n = str(n).strip().strip("-").replace(" ", "")
                if n:
                    index[n.lower()].add(i)
        if "COMMON_NAME" in hosts.colnames:
            for i, n in enumerate(hosts["COMMON_NAME"]):
                n = str(n).strip().strip("-").replace(" ", "")
                if n:
                    m = re.match(r"^([A-Za-z]+)0*(\d+)$", n)
                    if m is None:
                        index[n.lower()].add(i)
                    else:
                        prefix, number = m.groups()
                        index["{}{}".format(prefix.lower(), number)].add(i)
        for col in ["NSAID", "PGC", "NGC", "UGC"]:
            if col in hosts.colnames:
                for i, n in enumerate(hosts[col]):
                    n = int(n)
                    if n > -1:
                        index["{}{}".format(col[:3].lower(), n)].add(i)
                        if col == "PGC":
                            index[n].add(i)
        return {k: tuple(v) for k, v in index.items()}

    @property
    def _host_index(self):
        if self._host_index_ is None:
            self._host_index_ = self._index_ids(self._host_table)
        return self._host_index_

    @property
    def _master_index(self):
        if self._master_index_ is None:
            self._master_index_ = self._index_ids(self._master_table)
        return self._master_index_

    def _check_use_master(self, use_master=None):
        return bool(use_master or (use_master is None and self.use_master))

    def _get_table(self, use_master=None):
        if self._check_use_master(use_master):
            return self._master_table
        return self._host_table

    def _get_index(self, use_master=None):
        if self._check_use_master(use_master):
            return self._master_index
        return self._host_index

    def _resolve_indices(self, hosts=None, use_master=None):

        if hosts is None or (_is_string_like(hosts) and hosts.lower() == "all"):
            return list(range(len(self._get_table(use_master))))

        if isinstance(hosts, int):
            hosts = str(hosts)

        if _is_string_like(hosts):
            if hosts.isdigit():
                hosts_key = int(hosts)
            else:
                hosts_key = hosts.strip().replace(" ", "").replace("_", "").lower()

            try:
                return list(self._get_index(use_master)[hosts_key])
            except KeyError:
                pass

            if "," in hosts:
                hosts = hosts.split(",")
            elif not hosts.isdigit():
                hosts = self._resolve_preset_query(hosts)

        if isinstance(hosts, Query):
            return np.flatnonzero(hosts.mask(self._get_table(use_master))).tolist()

        if hasattr(hosts, "__iter__") and not _is_string_like(hosts):
            indices = []
            for host in iter(hosts):
                indices.extend(self._resolve_indices(host, use_master=use_master))
            return indices

        return []

    def resolve_id(self, hosts, id_to_return=None, use_master=None):
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
        indices = self._resolve_indices(hosts, use_master=use_master)

        if not indices:
            raise KeyError("Can not find {}".format(hosts))

        id_to_return = id_to_return or ""
        id_to_return = id_to_return.upper()

        if id_to_return[:2] in ("", "ST", "ID", "HO", "FI"):
            return self._get_table(use_master)[self._ID_COLNAME][indices].tolist()

        if id_to_return[:2] == "IN":
            return indices

        if id_to_return[:3] in ("NSA", "PGC", "SAG", "NGC", "UGC"):
            col = id_to_return[:3]
            if col == "SAG":
                col += "A_NAME"
            if col == "NSA":
                col += "ID"
            if col in self._get_table(use_master).colnames:
                return self._get_table(use_master)[col][indices].tolist()

        raise ValueError("`id_to_return` not known!")

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
        names = self.resolve_id(host_id, "SAGA")
        if len(names) > 1:
            raise ValueError("More than one matched host found!")
        return names[0] or ""

    def _annotate_table(self, d, add_coord=False, include_stats=False):
        if include_stats:
            d = self.add_object_stats(d, use_remote=("remote" in str(include_stats)))
        if add_coord:
            d = add_skycoord(d)
        return d

    def add_object_stats(self, host_table, use_remote=False):
        data_obj = self._database["host_stats"]
        stats = data_obj.remote.read() if use_remote else data_obj.read()
        cols_to_keep = [
            c for c in stats.colnames if c not in host_table.colnames or c == "HOSTID"
        ]
        stats = stats[cols_to_keep]
        d = join(host_table, stats, "HOSTID", "left")
        for col in cols_to_keep:
            if col == "HOSTID":
                continue
            if "need_spec" in col:
                d[col].fill_value = 999999
            else:
                d[col].fill_value = -1
        return d.filled()

    def load(
        self,
        hosts=None,
        add_coord=True,
        use_master=None,
        include_stats=False,
        query=None,
    ):
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
        indices = self.resolve_id(hosts, "internal", use_master)
        d = self._get_table(use_master)[indices]
        d = self._annotate_table(d, add_coord=add_coord, include_stats=include_stats)
        if query is not None:
            d = self._resolve_preset_query(query).filter(d)
        return d

    def load_single(self, host, add_coord=True, use_master=None, include_stats=False):
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
        indices = self.resolve_id(host, "internal", use_master)
        if len(indices) != 1:
            raise ValueError("More than one matched host found! Use `load` instead!")
        d = self._get_table(use_master)[indices]
        d = self._annotate_table(d, add_coord=add_coord, include_stats=include_stats)
        return d[0]

    def load_single_near_ra_dec(
        self, ra, dec, sep=3600, use_master=None, add_coord=True, include_stats=False
    ):
        """
        ra, dec in degrees
        """
        d = add_skycoord(self._get_table(use_master))
        cat = find_near_ra_dec(d, ra, dec, sep)
        del d["coord"]
        if len(cat) == 0:
            raise KeyError("No host near ({:.6f}, {:.6f}) found!".format(ra, dec))
        if len(cat) != 1:
            raise ValueError(
                "More than one hosts near ({:.6f}, {:.6f}) found! Use smaller `sep`!".format(
                    ra, dec
                )
            )
        cat = self._annotate_table(
            cat, add_coord=add_coord, include_stats=include_stats
        )
        return cat[0]

    def build_master_list(self, overwrite=False, overwrite_host_list=False):
        if self._version == 1:
            raise NotImplementedError("cannot build v1 master list!")

        self._master_table_ = build_master_list(
            hyperleda=self._database["hyperleda_kt12"].read(),
            edd_2mrs=self._database["edd_2mrs_slim"].read(),
            edd_lim17=self._database["edd_lim17_slim"].read(),
            nsa=self._database["nsa_v0.1.2"].read(),
            nsa1=self._database["nsa_v1.0.1"].read(),
            remove_list=self._database["host_remove"].read(),
            stars=self._database["hipparcos2"].read(),
            coverage_maps={
                k.partition("_")[-1]: self._database[k].read()
                for k in self._database.keys()
                if k.startswith("footprint_")
            },
        )
        self._master_index_ = None

        self._host_table_ = cuts.potential_hosts.filter(self._master_table_)
        self._host_index_ = None

        if not self._database["master_list"].remote.isfile() or overwrite:
            self._database["master_list"].write(self._master_table_, overwrite=True)

        if self._database["hosts"].local is not None and (
            overwrite_host_list or not self._database["hosts"].local.isfile()
        ):
            self._database["hosts"].local.write(self._host_table_, overwrite=True)

        return self._master_table_

    def load_master_list(self, hosts=None, add_coord=True):
        return self.load(hosts, add_coord, use_master=True)

    def _resolve_preset_query(self, query):
        if isinstance(query, Query):
            return query

        if _is_string_like(query):
            return getattr(cuts, query, None) or Query(query)

        if hasattr(query, "__iter__"):
            return Query(*[self._resolve_preset_query(q) for q in query])

        raise ValueError("Cannot resolve input query")

    def construct_host_query(self, *queries, use_remote=False):
        include_stats = "remote" if use_remote else True
        hosts = self.load(add_coord=False, include_stats=include_stats, query=queries)
        host_ids = hosts["HOSTID"].tolist()
        return QueryMaker.in1d("HOSTID", host_ids)


class FieldCatalog(HostCatalog):

    _ID_COLNAME = "field_id"

    @property
    def _host_table(self):
        if self._host_table_ is None:
            self._host_table_ = self._database["lowz_fields"].read()
            if "Dec" in self._host_table_.colnames:
                self._host_table_.rename_column("Dec", "DEC")
        return self._host_table_

    @property
    def _host_index(self):
        if self._host_index_ is None:
            self._host_index_ = dict(
                zip(
                    (
                        f.replace("_", "").lower()
                        for f in self._host_table[self._ID_COLNAME]
                    ),
                    [(i,) for i in range(len(self._host_table))],
                )
            )
        return self._host_index_

    @property
    def _master_table(self):
        raise AttributeError

    @property
    def _master_index(self):
        raise AttributeError

    def _check_use_master(self, use_master=None):
        return False

    def build(self, overwrite=False):
        raise AttributeError

    def add_object_stats(self, host_table, use_remote=False):
        if use_remote:
            raise NotImplementedError
        return host_table
