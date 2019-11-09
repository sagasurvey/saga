"""
SAGA.host.host_catalog

This file defines the HostCatalog class
"""
import logging
import re
from collections import defaultdict

import numpy as np
from easyquery import Query

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

    def __init__(self, database=NotImplementedError):
        self._database = database or Database()
        self._host_table_ = None
        self._host_index_ = None
        self._master_table_ = None

    @property
    def _master_table(self):
        if self._master_table_ is None:
            try:
                self._master_table_ = self._database["masterlist"].read()
            except:
                logging.warning(
                    "Cannot load master list; attempt to build from scratch..."
                )
                self._master_table_ = self.build()
        return self._master_table_

    @property
    def _host_table(self):
        if self._host_table_ is None:
            try:
                self._host_table_ = self._database["hosts_v2"].read()
            except:
                logging.warning("Cannot load host list; attempt to load master list...")
                self._host_table_ = Query("HOST_SCORE > 0").filter(self._master_table)
        return self._host_table_

    @property
    def _host_index(self):
        if self._host_index_ is None:
            hosts = self._host_table
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
            self._host_index_ = {k: tuple(v) for k, v in index.items()}
        return self._host_index_

    def _resolve_indices(self, hosts=None):

        if hosts is None:
            return list(range(len(self._host_table)))

        if _is_string_like(hosts):
            hosts_key = hosts.strip().replace(" ", "").replace("_", "").lower()

            try:
                hosts_key = int(hosts_key)
            except ValueError:
                pass

            if hosts_key in self._host_index:
                return list(self._host_index[hosts_key])

            hosts = getattr(cuts, hosts_key, None) or Query(hosts)

        if isinstance(hosts, Query):
            return np.flatnonzero(hosts.mask(self._host_table)).tolist()

        if isinstance(hosts, int) and hosts in self._host_index:
            return list(self._host_index[hosts])

        if hasattr(hosts, "__iter__"):
            indices = []
            for host in iter(hosts):
                indices.extend(self._resolve_indices(host))
            return indices

        return []

    def resolve_id(self, hosts, id_to_return=None):
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
        indices = self._resolve_indices(hosts)

        if not indices:
            raise KeyError("Can not find {}".format(hosts))

        id_to_return = id_to_return or ""
        id_to_return = id_to_return.upper()

        if id_to_return[:2] in ("", "ST", "ID", "HO", "FI"):
            return self._host_table[self._ID_COLNAME][indices].tolist()

        if id_to_return[:2] == "IN":
            return indices

        if id_to_return[:3] in ("NSA", "PGC", "SAG", "NGC", "UGC"):
            col = id_to_return[:3]
            if col == "SAG":
                col += "A_NAME"
            if col == "NSA":
                col += "ID"
            if col in self._host_table.colnames:
                return self._host_table[col][indices].tolist()

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
            raise ValueError("more than one names found!")
        return names[0] or ""

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
        d = self._host_table[self.resolve_id(hosts, "internal")]
        return add_skycoord(d) if add_coord else d

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
        indices = self.resolve_id(host, "internal")
        if len(indices) != 1:
            raise ValueError("More than one hosts found!")
        cat = self._host_table[indices]
        cat = add_skycoord(cat) if add_coord else cat
        return cat[0]

    def load_single_near_ra_dec(self, ra, dec):
        """
        ra, dec in degrees
        """
        add_skycoord(self._host_table)
        cat = find_near_ra_dec(self._host_table, ra, dec, 3603.0)
        del self._host_table["coord"]
        if len(cat) == 0:
            raise KeyError("No hosts found!")
        if len(cat) != 1:
            raise ValueError("More than one hosts found!")
        return cat[0]

    def build(self, overwrite=False):

        if self._database["masterlist"].remote.isfile() and not overwrite:
            raise ValueError(
                "masterlist already exist and overwrite is not set to True"
            )

        d = build_master_list(
            hyperleda=self._database["hyperleda_kt12"].read(),
            edd_2mrs=self._database["edd_2mrs_slim"].read(),
            edd_kim17=self._database["edd_kim17_slim"].read(),
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
        self._database["masterlist"].write(d, overwrite=True)
        return d

    def load_master_list(self):
        return self._master_table


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
                    range(len(self._host_table)),
                )
            )
        return self._host_index_

    def build(self, overwrite=False):
        raise NotImplementedError
