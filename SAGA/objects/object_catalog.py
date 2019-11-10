import os
import time
import traceback
from collections import defaultdict

import numpy as np
from astropy.table import Table, join, vstack
from astropy.time import Time
from easyquery import Query

from .. import utils
from ..database import CsvTable, Database, DataObject, FileObject, FitsTable
from ..hosts import HostCatalog
from ..utils import fill_values_by_query, get_all_colors, get_sdss_bands
from . import build, build2
from . import cuts as C
from .manual_fixes import fixes_to_nsa_v012, fixes_to_nsa_v101

__all__ = ["ObjectCatalog", "get_unique_objids"]


def get_unique_objids(objid_col):
    try:
        objid_col = objid_col.compressed()
    except AttributeError:
        pass
    return np.unique(np.asarray(objid_col, dtype=np.int64))


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

    _surveys = ("sdss", "des", "decals")

    def __init__(
        self, database=None, host_catalog_class=HostCatalog, host_catalog_instance=None
    ):
        self._database = database or Database()
        if host_catalog_instance is not None:
            if not isinstance(host_catalog_instance, host_catalog_class):
                raise ValueError(
                    "`host_catalog_instance` must be an instance of `host_catalog_class`."
                )
            self._host_catalog = host_catalog_instance
        else:
            self._host_catalog = host_catalog_class(self._database)

    @classmethod
    def _annotate_catalog(
        cls,
        table,
        add_skycoord=False,
        ensure_all_objid_cols=False,
        fix_2df_lowz_zq=False,
    ):
        if "EXTINCTION_R" in table.colnames:
            for b in get_sdss_bands():
                table["{}_mag".format(b)] = (
                    table[b] - table["EXTINCTION_{}".format(b.upper())]
                )

        for color in get_all_colors():
            if (
                "{}_mag".format(color[0]) not in table.colnames
                or "{}_mag".format(color[1]) not in table.colnames
            ):
                continue
            table[color] = (
                table["{}_mag".format(color[0])] - table["{}_mag".format(color[1])]
            )
            table["{}_err".format(color)] = np.hypot(
                table["{}_err".format(color[0])], table["{}_err".format(color[1])]
            )

        if ensure_all_objid_cols:
            for s in cls._surveys:
                col = "OBJID_{}".format(s)
                if col not in table.colnames:
                    table[col] = -1

        if add_skycoord:
            table = utils.add_skycoord(table)

        # TODO: remove this in the future:
        if fix_2df_lowz_zq:
            fill_values_by_query(
                table,
                Query(
                    (lambda x: ((x == "2dF") | (x == "2dFLen")), "TELNAME"),
                    "ZQUALITY == 3",
                    "SPEC_Z < 0.05",
                ),
                {"ZQUALITY": 2},
            )

        return table

    @staticmethod
    def _slice_table(table, query=None, columns=None, add_skycoord=False):
        if query is not None:
            table = table[Query(query).mask(table)]

        if columns is not None:
            table = table[columns]

        if add_skycoord:
            table = utils.add_skycoord(table)

        return table

    def load(
        self,
        hosts=None,
        has_spec=None,
        cuts=None,
        return_as=None,
        columns=None,
        version=None,
        add_skycoord=True,
        ensure_all_objid_cols=False,
        fix_2df_lowz_zq=True,
    ):
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
            If set to 'dict', return a dictionary with host ids being the keys
            If set to 'items', return an iterator like dict.items()

        columns : list, optional
            If set, only load a subset of columns

        version : int or str, optional
            Set to 'paper1' for paper1 catalogs

        add_skycoord : bool, optional (default: True)
            add `coord` column

        ensure_all_objid_cols : bool, optional (default: False)
            make sure `OBJID_sdss`, `OBJID_des`, `OBJID_decals` all exist

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

        if version is None:
            base_key = "base"
        elif str(version).lower() in ("paper1", "p1", "v0p1", "0", "0.1"):
            base_key = "base_v0p1"
        elif version in (1, 2):
            base_key = "base_v{}".format(version)
        else:
            raise ValueError("`version` must be None, 'paper1', 1 or 2.")

        if return_as is None:
            return_as = "stacked" if (has_spec and base_key == "base_v0p1") else "list"
        return_as = return_as.lower()
        if return_as[0] not in "slid":
            raise ValueError(
                '`return_as` should be "list", "stacked", "iter", or "dict"'
            )

        if has_spec and base_key == "base_v0p1":
            t = self._database["saga_spectra_May2017"].read()

            if hosts is not None:
                host_ids = self._host_catalog.resolve_id(hosts, "NSA")
                t = self._slice_table(t, (lambda x: np.in1d(x, host_ids), "HOST_NSAID"))

            t = self._annotate_catalog(t)

            if return_as[0] == "s":
                return self._slice_table(t, cuts, columns, add_skycoord)

            if hosts is None:
                host_ids = np.unique(t["HOST_NSAID"])
            output_iterator = (
                self._slice_table(
                    t, Query(cuts, "HOST_NSAID == {}".format(i)), columns, add_skycoord
                )
                for i in host_ids
            )
            if return_as[0] == "i":
                return output_iterator
            if return_as[0] == "d":
                return dict(zip(host_ids, output_iterator))
            return list(output_iterator)

        q = Query(cuts)
        if has_spec:
            q = q & C.has_spec
        elif has_spec is not None:
            q = q & (~C.has_spec)

        hosts = self._host_catalog.resolve_id(hosts, "string")

        output_iterator = (
            self._slice_table(
                self._annotate_catalog(
                    self._database[base_key, host].read(),
                    ensure_all_objid_cols=ensure_all_objid_cols,
                    fix_2df_lowz_zq=fix_2df_lowz_zq,
                ),
                q,
                columns,
                (add_skycoord and return_as[0] != "s"),
            )
            for host in hosts
        )

        if return_as.startswith("item"):
            return zip(hosts, output_iterator)
        if return_as[0] == "d":
            return dict(zip(hosts, output_iterator))
        if return_as[0] == "i":
            return output_iterator
        if return_as[0] == "s":
            out = vstack(list(output_iterator), "outer")
            if out.masked:
                for name, (dtype, _) in out.dtype.fields.items():
                    if dtype.kind == "i":
                        out[name].fill_value = -1
                    if dtype.kind == "b":
                        out[name].fill_value = False
            out = out.filled()
            if add_skycoord:
                out = utils.add_skycoord(out)
            return out
        return list(output_iterator)

    def load_nsa(self, version="0.1.2"):
        nsa = self._database["nsa_v{}".format(version)].read()
        remove_mask = np.zeros(len(nsa), np.bool)
        objs_to_remove = []
        fixes_dict = {}
        cols = nsa.colnames
        if version == "0.1.2":
            objs_to_remove = [64408]
            fixes_dict = fixes_to_nsa_v012
            cols = build.NSA_COLS_USED
        elif version == "1.0.1":
            remove_mask |= (nsa["DFLAGS"][:, 3:6] == 24).any(axis=1) & (
                nsa["DFLAGS"][:, 3:6]
            ).all(axis=1)
            objs_to_remove = [614276, 632725]
            fixes_dict = fixes_to_nsa_v101
            cols = build2.NSA_COLS_USED
        remove_mask |= np.in1d(nsa["NSAID"], objs_to_remove, assume_unique=True)
        nsa = nsa[cols][~remove_mask]
        for nsaid, fixes in fixes_dict.items():
            fill_values_by_query(nsa, "NSAID == {}".format(nsaid), fixes)
        nsa = utils.add_skycoord(nsa)
        return nsa

    def build_and_write_to_database(
        self,
        hosts=None,
        overwrite=False,
        base_file_path_pattern=None,
        version=None,
        return_catalogs=False,
        raise_exception=False,
        add_specs_only_before_time=None,
        use_nsa=True,
        convert_to_sdss_filters=True,
        additional_specs=None,
        debug=None,
    ):
        """
        This function builds the base catalog and writes it to the database.

        !! IMPORTANT !!
        If you want to write the base catalog to an alternative location (not the database)
        Make sure you set the `base_file_path_pattern` option!!

        Parameters
        ----------
        hosts : int, str, list, optional
            host names/IDs or a list of host names/IDs or short-hand names like
            "paper1" or "paper1_complete"

        overwrite : bool, optional (default: False)
            If set to True, overwrite existing base catalog
            If set to an astropy.time.Time object, overwrite catalogs that are older than that time.

        base_file_path_pattern : str, optional
        version : int, optional (default: 2)
        return_catalogs : bool, optional (default: False)
        raise_exception : bool, optional (default: False)
        add_specs_only_before_time : astropy.time.Time, optional (default: None)

        Examples
        --------
        >>> saga_database = SAGA.Database('/path/to/SAGA/Dropbox', '/path/to/SAGA/local')
        >>> saga_object_catalog = SAGA.ObjectCatalog(saga_database)

        Overwrite the database (Danger!!)
        >>> saga_object_catalog.build_and_write_to_database('paper1', overwrite=True)

        You can also do
        >>> saga_object_catalog.build_and_write_to_database('paper1', base_file_path_pattern='/other/base/catalog/dir/nsa{}.fits.gz')

        """

        if overwrite not in (True, False, None) and not isinstance(overwrite, Time):
            overwrite = Time(overwrite)

        host_table = self._host_catalog.load(hosts)
        nhosts = len(host_table)
        if not nhosts:
            print(time.strftime("[%m/%d %H:%M:%S]"), "No host to build! Abort!")
            return
        print(
            time.strftime("[%m/%d %H:%M:%S]"),
            "Start to build {} base catalog(s).".format(nhosts),
        )

        if version not in (None, 1, 2):
            raise ValueError("`version` must be None, 1 or 2.")
        build_module = build if version == 1 else build2

        if use_nsa:
            nsa = self.load_nsa("0.1.2" if version == 1 else "1.0.1")
            print(time.strftime("[%m/%d %H:%M:%S]"), "NSA catalog loaded.")
        else:
            nsa = None

        spectra = self._database["spectra_raw_all"].read(
            before_time=add_specs_only_before_time, additional_specs=additional_specs,
        )
        print(time.strftime("[%m/%d %H:%M:%S]"), "All spectra loaded.")

        manual_lists = {}
        for survey, col in (
            ("sdss", "SDSS ID"),
            ("des", "DES_OBJID"),
            ("decals", "decals_objid"),
        ):
            for list_type in ("remove", "recover"):
                key = "{}_{}".format(survey, list_type)
                try:
                    val = get_unique_objids(self._database[key].read()[col])
                except KeyError:
                    pass
                else:
                    if len(val):
                        manual_lists[key] = val
        print(time.strftime("[%m/%d %H:%M:%S]"), "All other manual lists loaded.")

        catalogs_to_return = list()
        for i, host in enumerate(host_table):
            host_id = host["HOSTID"]
            if base_file_path_pattern is None:
                base_key = "base" if version is None else "base_v{}".format(version)
                data_obj = self._database[base_key, host_id].remote
            else:
                data_obj = FitsTable(base_file_path_pattern.format(host_id))

            if data_obj.isfile() and overwrite is not True:
                if overwrite is False or overwrite is None:
                    print(
                        time.strftime("[%m/%d %H:%M:%S]"),
                        "Base catalog v{} for {} already exists ({}).".format(
                            version or 2, host_id, data_obj.path
                        ),
                        "({}/{})".format(i + 1, nhosts),
                    )
                    continue

                file_time = Time(os.path.getmtime(data_obj.path), format="unix")

                if file_time > overwrite:
                    print(
                        time.strftime("[%m/%d %H:%M:%S]"),
                        "Base catalog {} ({}) is newer than {}.".format(
                            host_id, file_time.isot, overwrite.isot
                        ),
                        "({}/{})".format(i + 1, nhosts),
                    )
                    continue

            def _get(col):
                # pylint: disable=cell-var-from-loop
                try:
                    return host["COVERAGE_" + col]
                except KeyError:
                    return 1.0

            if version == 1:
                catalogs = ("sdss", "wise")
            else:
                catalogs = []
                if _get("SDSS") >= 0.95:
                    catalogs.append("sdss")
                if _get("DES_DR1") >= 0.95:
                    catalogs.append("des")
                if max(_get("DECALS_DR6"), _get("DECALS_DR7")) >= 0.95:
                    catalogs.append("decals")
                catalogs = tuple(catalogs)

            def get_catalog_or_none(catalog_name):
                # pylint: disable=cell-var-from-loop
                try:
                    cat = self._database[catalog_name, host_id].read()
                except OSError:
                    print(
                        time.strftime("[%m/%d %H:%M:%S]"),
                        "[WARNING] Not found: {} catalog for {}!!.".format(
                            catalog_name.upper(), host_id
                        ),
                    )
                    return
                return cat[build.WISE_COLS_USED] if catalog_name == "wise" else cat

            catalog_dict = {k: get_catalog_or_none(k) for k in catalogs}

            print(
                time.strftime("[%m/%d %H:%M:%S]"),
                "Use {} to build base catalog v{} for {}".format(
                    ", ".join(
                        (k for k, v in catalog_dict.items() if v is not None)
                    ).upper(),
                    (version or 2),
                    host_id,
                ),
                "({}/{})".format(i + 1, nhosts),
            )

            if debug is None:
                debug_this = None
            else:
                debug[host_id] = dict()
                debug_this = debug[host_id]

            try:
                base = build_module.build_full_stack(
                    host=host,
                    nsa=nsa,
                    spectra=spectra,
                    convert_to_sdss_filters=convert_to_sdss_filters,
                    debug=debug_this,
                    **manual_lists,
                    **catalog_dict,
                )
            except Exception as e:  # pylint: disable=W0703
                print(
                    time.strftime("[%m/%d %H:%M:%S]"),
                    "[ERROR] Fail to build base catalog for {}".format(host_id),
                )
                base = None
                if raise_exception:
                    raise e
                traceback.print_exc()
                continue
            finally:
                del catalog_dict
                if return_catalogs:
                    catalogs_to_return.append(base)

            print(
                time.strftime("[%m/%d %H:%M:%S]"),
                "Write base catalog to {}".format(data_obj.path),
            )
            try:
                data_obj.write(base)
            except (IOError, OSError) as e:
                print(
                    time.strftime("[%m/%d %H:%M:%S]"),
                    "[ERROR] Fail to write base catalog for {}".format(host_id),
                )
                if raise_exception:
                    raise e
                traceback.print_exc()
                continue

        if return_catalogs:
            return catalogs_to_return

    def generate_clean_specs(self, save_to=None, overwrite=False, **kwargs):
        """
        generate clean spectra from all good base catalogs and save to disk
        """
        defaults = dict(
            hosts="good",
            has_spec=True,
            cuts=C.is_clean2,
            return_as="stack",
            add_skycoord=False,
            ensure_all_objid_cols=True,
        )

        defaults.update(kwargs)

        t = self.load(**defaults)

        if save_to is not False:
            if save_to is None:
                save_to = self._database["saga_clean_specs"]
            if not isinstance(save_to, (FileObject, DataObject)):
                save_to = FitsTable(save_to)
            save_to.write(t, overwrite=overwrite)

        return t

    def load_clean_specs(self, generate_if_not_exist=True, **kwargs):
        """
        load clean spectra from all good base catalogs
        """
        try:
            t = self._database["saga_clean_specs"].read()
        except IOError:
            if generate_if_not_exist:
                t = self.generate_clean_specs(**kwargs)
            else:
                raise
        return t

    def _generate_object_stats_single_host(self, host_id, data=None):
        if data is None:
            data = defaultdict(list)

        data["HOSTID"].append(host_id)
        base = self.load(
            host_id, cuts=(C.is_clean2 & C.is_galaxy2), add_skycoord=False
        ).pop()

        simple_cuts = C.faint_end_limit & C.gri_or_grz_cut & C.sat_rcut
        high_p_cuts = Query("r_mag > 12") & C.high_priority_cuts

        simple_mask = simple_cuts.mask(base)
        high_p_mask = high_p_cuts.mask(base)
        high_p_mask &= simple_mask
        has_spec_mask = C.has_spec.mask(base)
        sats_mask = C.is_sat.mask(base)
        low_z_mask = C.is_low_z.mask(base)
        low_z_mask &= ~sats_mask
        our_specs_mask = C.has_our_specs_only.mask(base)
        our_specs_mask &= has_spec_mask
        aat_mask = C.has_aat_spec.mask(base)
        mmt_mask = C.has_mmt_spec.mask(base)

        data["need_spec"].append(np.count_nonzero(simple_mask & (~has_spec_mask)))
        data["really_need_spec"].append(
            np.count_nonzero(high_p_mask & (~has_spec_mask))
        )

        data["specs_total"].append(np.count_nonzero(has_spec_mask))
        data["low_z_total"].append(np.count_nonzero(has_spec_mask & low_z_mask))
        data["sats_total"].append(np.count_nonzero(sats_mask))

        data["specs_ours"].append(np.count_nonzero(our_specs_mask))
        data["low_z_ours"].append(np.count_nonzero(our_specs_mask & low_z_mask))
        data["sats_ours"].append(np.count_nonzero(our_specs_mask & sats_mask))

        data["specs_aat"].append(np.count_nonzero(has_spec_mask & aat_mask))
        data["specs_mmt"].append(np.count_nonzero(has_spec_mask & mmt_mask))

        return data

    def generate_object_stats(
        self, hosts=None, save_to=None, overwrite=False, generate_func=None
    ):
        """
        generate object statistics for *hosts*
        """

        if hosts is None:
            hosts = sorted(self._host_catalog.resolve_id("good"))
        else:
            hosts = self._host_catalog.resolve_id(hosts)

        if generate_func is None:
            generate_func = self._generate_object_stats_single_host

        data = None
        for host in hosts:
            data = generate_func(host, data)

        data = Table(data)

        host_table = self._host_catalog.load(hosts)
        if "distance" in host_table.colnames:
            cols = ["HOSTID", "PGC", "RA", "DEC", "distance", "MK_compiled"]
        else:
            cols = ["HOSTID", "PGC", "RA", "DEC", "DIST", "K_ABS", "HOST_SCORE"]
        host_table = host_table[cols]
        if "distance" in host_table.colnames:
            host_table.rename_column("distance", "DIST")
            host_table.rename_column("MK_compiled", "K_ABS")
        data = join(data, host_table, "HOSTID", "left")

        if save_to is not None:
            if not isinstance(save_to, (FileObject, DataObject)):
                save_to = CsvTable(save_to)
            save_to.write(data, overwrite=overwrite)

        return data
