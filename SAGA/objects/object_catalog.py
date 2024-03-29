import gc
import os
import time
import traceback
from collections import defaultdict

import numpy as np
from astropy.table import Table, unique, vstack
from astropy.time import Time
from easyquery import Query

from .. import utils
from ..database import CsvTable, Database, DataObject, FileObject, FitsTable
from ..hosts import HostCatalog
from ..utils import fill_values_by_query, get_all_colors, get_sdss_bands
from . import build, build2, build3
from . import cuts as C
from .manual_fixes import fixes_to_nsa_v012, fixes_to_nsa_v101

__all__ = ["ObjectCatalog", "get_unique_objids"]


def get_unique_objids(objid_col):
    try:
        objid_col = objid_col.compressed()
    except AttributeError:
        pass
    return np.unique(np.asarray(objid_col, dtype=np.int64))


def _p_sat_model_paper2(base, params=(-1.96, 1.507, -5.498, 0.303, 0.487)):
    r = base["r_mag"]

    gr = np.where(
        C.valid_g_mag.mask(base),
        base["gr"],
        np.where(
            C.valid_i_mag.mask(base),
            base["ri"] + 0.1,
            np.where(C.valid_z_mag.mask(base), base["rz"] + 0.2, 0.92 - 0.03 * r),
        ),
    )

    sb = np.where(
        C.valid_sb.mask(base),
        base["sb_r"],
        20.75 + 0.6 * (r - 14),
    )

    mu = params[0] * r + params[1] * sb + params[2] * gr + params[3]
    mu = np.where(np.isnan(mu), np.inf, mu)
    p = params[-1] / (1 + np.exp(-mu))

    return p


def _p_sat_model_paper3(base, params=(0.65, -3.84, 2.62, -25.6, 1.25, 0.75)):
    gr = np.where(
        Query("gr > -1", "gr < 10").mask(base),
        base["gr"],
        np.where(
            Query("rz > -0.25", "rz < 1.5").mask(base),
            0.1679866 + 1.0276194 * base["r_mag"] - 1.026017 * base["z_mag"],
            0.8387913 - 0.0252538 * base["r_mag"],
        ),
    )

    r_fib = np.where(
        Query("r_fibermag >= r_mag", "r_fibermag < 30").mask(base),
        base["r_fibermag"],
        np.where(
            Query("sb_r > 15", "sb_r < 30").mask(base),
            -1.730775 + 0.22267 * base["r_mag"] + 0.84099 * base["sb_r"],
            10.50927 + 0.60664 * base["r_mag"],
        ),
    )

    r_diff = r_fib - params[0] * base["r_mag"]

    mu = params[1] * gr + params[2] * r_diff + params[3]
    mu = np.where(np.isnan(mu), np.inf, mu)
    p = np.where(gr < params[4], params[-1] / (1 + np.exp(-mu)), 0.0)

    return p


def _correct_fiducial_p_sat(base, fiducial_p_label, bias=None, version=3):
    p = np.array(base[fiducial_p_label])

    if bias and "human_selected" in base.colnames:
        mask = Query(~C.has_spec, "human_selected > 0").mask(base)
        p[mask] = p[mask] * (1 - bias) + bias

    if "SATS" in base.colnames:
        p[Query(C.has_spec, C.is_sat).mask(base)] = 1
        p[Query(C.has_spec, ~C.is_sat).mask(base)] = 0

    good_obj = Query(C.is_galaxy, C.is_clean) if version == 1 else Query(C.is_galaxy2, C.is_clean2)
    p[~good_obj.mask(base)] = 0
    p[p < np.finfo(np.float32).eps] = 0

    return p


def _get_coverage(host, survey):
    try:
        return host["COVERAGE_" + survey.upper()]
    except KeyError:
        return 0.0


def _determine_raw_catalogs_saga_v1(**kwargs):
    return ("sdss", "wise")


def _determine_raw_catalogs_saga_v2(host, using_dr8_by_default=False, **kwargs):
    coverage = {
        s: _get_coverage(host, s)
        for s in (
            "sdss",
            "des_dr1",
            "decals_dr6",
            "decals_dr7",
            "decals_dr8",
        )
    }

    if using_dr8_by_default:
        coverage["decals"] = coverage["decals_dr8"]
    else:
        coverage["decals"] = max(coverage["decals_dr6"], coverage["decals_dr7"])
    coverage["sdss_des"] = max(coverage["sdss"], coverage["des_dr1"])
    coverage["max"] = max(coverage["decals"], coverage["sdss_des"])

    catalogs = []
    if coverage["sdss"] >= 0.85:
        catalogs.append("sdss")
    if coverage["des_dr1"] >= 0.85:
        catalogs.append("des")
    if coverage["decals"] >= 0.95:
        catalogs.append("decals")

    if not using_dr8_by_default:
        if (
            "decals" in catalogs
            and coverage["decals_dr8"] >= 0.99
            and (coverage["sdss_des"] < 0.85 or coverage["max"] < 0.99)
        ):
            catalogs.remove("decals")
            catalogs.append("decals_dr8")
        elif "decals" not in catalogs and coverage["decals_dr8"] >= 0.95 and coverage["des_dr1"] < 0.95:
            catalogs.append("decals_dr8")

    return tuple(set(catalogs))


def _determine_raw_catalogs_saga_v3(host, **kwargs):
    catalogs = ["decals_dr9"]
    if _get_coverage(host, "sdss") > 0:
        catalogs.append("sdss_dr16")
    return tuple(catalogs)


def _determine_raw_catalogs_lowz_v2(host_id, **kwargs):
    field_name = str(host_id)
    if any(field_name.startswith(s) for s in ["GD1", "300S", "Jet", "Styx"]):
        return ("decals_dr67",)
    if any(field_name.startswith(s) for s in ["p13"]):
        return ("decals_dr8",)
    return ("des",)


def _determine_raw_catalogs_lowz_v3(host_id, **kwargs):
    field_name = str(host_id)
    if any(field_name.startswith(s) for s in ["Pal15"]):
        return ("delve_dr1",)
    return ("decals_dr9",)


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

    def __init__(self, database=None, host_catalog_class=HostCatalog, host_catalog_instance=None):
        self._database = database or Database()
        if host_catalog_instance is not None:
            if not isinstance(host_catalog_instance, host_catalog_class):
                raise ValueError("`host_catalog_instance` must be an instance of `host_catalog_class`.")
            self._host_catalog = host_catalog_instance
        else:
            self._host_catalog = host_catalog_class(self._database)

    def _annotate_catalog(
        self,
        table,
        add_skycoord=False,
        ensure_all_objid_cols=False,
    ):
        # determine catalog version
        version = 2
        if "REF_CAT" in table.colnames:
            version = 3
        elif "EXTINCTION_R" in table.colnames:
            version = 1
            for b in get_sdss_bands():
                table["{}_mag".format(b)] = table[b] - table["EXTINCTION_{}".format(b.upper())]

        for color in get_all_colors():
            if "{}_mag".format(color[0]) not in table.colnames or "{}_mag".format(color[1]) not in table.colnames:
                continue
            with np.errstate(invalid="ignore"):
                table[color] = table["{}_mag".format(color[0])] - table["{}_mag".format(color[1])]
                table["{}_err".format(color)] = np.hypot(
                    table["{}_err".format(color[0])], table["{}_err".format(color[1])]
                )

        if "HOST_ID" in table.colnames:
            table.rename_column("HOST_ID", "HOSTID")

        if ensure_all_objid_cols:
            for s in self._surveys:
                col = "OBJID_{}".format(s)
                if col not in table.colnames:
                    table[col] = -1

        if "HOST_DIST" in table.colnames and version >= 2:
            table["human_selected"] = np.isin(table["OBJID"], self._database["human_selected"].read()["OBJID"]).astype(np.int32)

        with np.errstate(over="ignore", invalid="ignore"):
            table["p_sat_model_p2"] = _p_sat_model_paper2(table)  # paper2 model
            if version >= 3:
                table["p_sat_model_p3"] = _p_sat_model_paper3(table)
                table["p_sat_corrected"] = _correct_fiducial_p_sat(table, "p_sat_model_p3")
            else:
                table["p_sat_corrected"] = _correct_fiducial_p_sat(table, "p_sat_model_p2", bias=0.25, version=version)

        # add dr3_sample label
        table["dr3_sample"] = 0
        fill_values_by_query(table, C.is_sat, {"dr3_sample": 3})
        fill_values_by_query(table, Query("p_sat_corrected > 0", "log_sm_phony >= 6.75", C.sat_rcut), {"dr3_sample": 2})
        fill_values_by_query(table, Query("p_sat_corrected > 0", "log_sm_phony >= 7.5", C.sat_rcut), {"dr3_sample": 1})

        if add_skycoord:
            table = utils.add_skycoord(table)

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

        _, version_postfix = self._database.resolve_base_version(version)
        base_key = "base" + version_postfix

        if return_as is None:
            return_as = "stacked" if (has_spec and base_key == "base_v0p1") else "list"
        return_as = return_as.lower()
        if return_as[0] not in "slid":
            raise ValueError('`return_as` should be "list", "stacked", "iter", or "dict"')

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
                self._slice_table(t, Query(cuts, "HOST_NSAID == {}".format(i)), columns, add_skycoord) for i in host_ids
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

    def load_single(self, host, **kwargs):
        if "hosts" in kwargs:
            raise TypeError("load_single() got an unexpected keyword argument 'hosts'")
        if "return_as" in kwargs:
            raise TypeError("load_single() got an unexpected keyword argument 'return_as'")
        return self.load(hosts=host, **kwargs).pop()

    def load_nsa(self, version="0.1.2"):
        nsa = self._database["nsa_v{}".format(version)].read()
        remove_mask = np.zeros(len(nsa), bool)
        objs_to_remove = []
        fixes_dict = {}
        cols = nsa.colnames
        if version == "0.1.2":
            objs_to_remove = [64408]
            fixes_dict = fixes_to_nsa_v012
            cols = build.NSA_COLS_USED
        elif version == "1.0.1":
            remove_mask |= (nsa["DFLAGS"][:, 3:6] == 24).any(axis=1) & (nsa["DFLAGS"][:, 3:6]).all(axis=1)
            objs_to_remove = [614276, 632725, 628283, 694072, 667243, 219164, 632150, 306191]
            fixes_dict = fixes_to_nsa_v101
            cols = build2.NSA_COLS_USED
        remove_mask |= np.in1d(nsa["NSAID"], objs_to_remove, assume_unique=True)
        nsa = nsa[cols][~remove_mask]
        for nsaid, fixes in fixes_dict.items():
            fill_values_by_query(nsa, "NSAID == {}".format(nsaid), fixes)
        nsa = utils.add_skycoord(nsa)
        return nsa

    def load_sga(self):
        try:
            sga = self._database["sga_v3.0"]
        except KeyError:
            return None
        sga = sga.read()[build3.SGA_COLUMNS]
        # remove by SGA ID
        objs_to_remove = [763688]
        sga = sga[np.isin(sga["SGA_ID"], objs_to_remove, invert=True)]
        return sga

    def build_and_write_to_database(
        self,
        hosts="build_default",
        overwrite=False,
        base_file_path_pattern=None,
        version=None,
        return_catalogs=False,
        raise_exception=False,
        add_specs_only_before_time=None,
        use_nsa=True,
        convert_to_sdss_filters=True,
        additional_specs=None,
        exclude_spec_masks=None,
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
        version : int, optional (default: 3)
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
        build_version, version_postfix = self._database.resolve_base_version(version)
        if build_version == 0:
            raise ValueError("Cannot build v0 base catalogs")
        if version_postfix and not version_postfix.startswith("_"):
            version_postfix = "_" + version_postfix

        if overwrite not in (True, False, None) and not isinstance(overwrite, Time):
            overwrite = Time(overwrite)

        host_table = self._host_catalog.load(hosts)
        HOSTID_COLNAME = self._host_catalog._ID_COLNAME

        nhosts = len(host_table)
        if not nhosts:
            print(time.strftime("[%m/%d %H:%M:%S]"), "No host to build! Abort!")
            return
        print(
            time.strftime("[%m/%d %H:%M:%S]"),
            "Start to build {} base catalog(s). Build version = {}".format(nhosts, build_version),
        )

        print(
            time.strftime("[%m/%d %H:%M:%S]"),
            "base_file_path_pattern =",
            self._database._file_path_pattern["base" + version_postfix]
            if base_file_path_pattern is None
            else base_file_path_pattern,
        )

        if build_version < 2:
            build_module = build
            manual_keys = [("sdss", "SDSS ID")]
            catalogs_determining_func = _determine_raw_catalogs_saga_v1
        elif build_version < 3:
            build_module = build2
            manual_keys = [
                ("sdss", "SDSS ID"),
                ("des", "DES_OBJID"),
                ("decals", "decals_objid"),
                ("decals_dr8", "OBJID"),
                ("shreds", "OBJID"),
            ]
            if HOSTID_COLNAME == "field_id":
                catalogs_determining_func = _determine_raw_catalogs_lowz_v2
            else:
                catalogs_determining_func = _determine_raw_catalogs_saga_v2
        else:
            build_module = build3
            if HOSTID_COLNAME == "field_id":
                manual_keys = [
                    ("decals_dr9_lowz", "OBJID"),
                ]
                catalogs_determining_func = _determine_raw_catalogs_lowz_v3
            else:
                manual_keys = [
                    ("decals_dr9", "OBJID"),
                ]
                catalogs_determining_func = _determine_raw_catalogs_saga_v3

        if use_nsa:
            nsa = self.load_nsa("0.1.2" if build_version < 2 else "1.0.1")
            print(time.strftime("[%m/%d %H:%M:%S]"), "NSA catalog loaded.")
        else:
            nsa = None

        sga = self.load_sga()
        if sga is not None:
            print(time.strftime("[%m/%d %H:%M:%S]"), "SGA catalog loaded.")

        spectra = self._database["spectra_raw_all"].read(
            before_time=add_specs_only_before_time,
            additional_specs=additional_specs,
            exclude_spec_masks=exclude_spec_masks,
        )
        print(time.strftime("[%m/%d %H:%M:%S]"), "All spectra loaded.")

        # Loading post-processed catalogs
        halpha = self._database["spectra_halpha"].read() if build_version >= 2 else None
        galex_precalculated_lists = []
        if build_version >= 3:
            # galex_sfr_lowz goes BEFORE galex_sfr_host as the latter overwrites overlapping entries in the former
            for key in ["galex_sfr_lowz", "galex_sfr_host"]:
                try:
                    t = self._database[key]
                except KeyError:
                    continue
                t = t.read()
                t = Query("NUV_SFR_flag >= 0").filter(t)
                t = unique(t, "ID")
                galex_precalculated_lists.append(t)
                del t

        manual_lists = dict()
        for survey, col in manual_keys:
            for list_type in ("remove", "recover", "correction", "morphology"):
                key = "{}_{}".format(survey, list_type)
                try:
                    mlist = self._database[key]
                except KeyError:
                    continue
                val = mlist.read()
                if not len(val):
                    continue
                if list_type in ("remove", "recover"):
                    val = get_unique_objids(val[col])
                if "_" in survey:
                    new_key = "{}_{}".format(survey.partition("_")[0], list_type)
                    if new_key not in manual_lists:
                        key = new_key
                manual_lists[key] = val
        print(time.strftime("[%m/%d %H:%M:%S]"), "All other manual lists loaded.")

        failed_count = 0
        catalogs_to_return = list()
        for i, host in enumerate(host_table):
            host_id = host[HOSTID_COLNAME]
            if base_file_path_pattern is None:
                data_obj = self._database["base" + version_postfix, host_id].remote
            else:
                data_obj = FitsTable(base_file_path_pattern.format(host_id))

            if data_obj.isfile() and overwrite is not True:
                if overwrite is False or overwrite is None:
                    print(
                        time.strftime("[%m/%d %H:%M:%S]"),
                        "Base catalog {} for {} already exists ({}).".format(
                            version_postfix.lstrip("_"), host_id, data_obj.path
                        ),
                        "({}/{})".format(i + 1, nhosts),
                    )
                    continue

                file_time = Time(os.path.getmtime(data_obj.path), format="unix")

                if file_time > overwrite:
                    print(
                        time.strftime("[%m/%d %H:%M:%S]"),
                        "Base catalog {} ({}) is newer than {}.".format(host_id, file_time.isot, overwrite.isot),
                        "({}/{})".format(i + 1, nhosts),
                    )
                    continue

            catalog_dict = dict()
            catalogs = catalogs_determining_func(
                host=host,
                host_id=host_id,
                using_dr8_by_default=("dr8" in self._database.decals_file_path_pattern),
            )
            for catalog_name in catalogs:
                try:
                    cat = self._database[catalog_name, host_id].read()
                except OSError:
                    print(
                        time.strftime("[%m/%d %H:%M:%S]"),
                        "[WARNING] Not found: {} catalog for {}!!.".format(catalog_name.upper(), host_id),
                    )
                    continue
                if catalog_name == "wise":
                    cat = cat[build.WISE_COLS_USED]
                catalog_dict[catalog_name.partition("_")[0]] = cat
                del cat

            print(
                time.strftime("[%m/%d %H:%M:%S]"),
                "Use {} to build base catalog {} for {}".format(
                    ", ".join(catalog_dict).upper(),
                    version_postfix.lstrip("_"),
                    host_id,
                ),
                "({}/{})".format(i + 1, nhosts),
            )

            # add galex
            try:
                cat = self._database["galex", host_id].read()
            except OSError:
                pass
            else:
                catalog_dict["galex"] = cat[build3.GALEX_COLUMNS]
                del cat

            if debug is None:
                debug_this = None
            else:
                debug[host_id] = dict()
                debug_this = debug[host_id]

            try:
                base = build_module.build_full_stack(
                    host=host,
                    nsa=nsa,
                    sga=sga,
                    spectra=spectra,
                    halpha=halpha,
                    galex_precalculated_lists=galex_precalculated_lists,
                    convert_to_sdss_filters=convert_to_sdss_filters,
                    debug=debug_this,
                    **manual_lists,
                    **catalog_dict,
                )
            except Exception as e:
                print(
                    time.strftime("[%m/%d %H:%M:%S]"),
                    "[ERROR] Fail to build base catalog for {}".format(host_id),
                )
                base = None
                if raise_exception:
                    raise e
                traceback.print_exc()
                failed_count += 1
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
                data_obj.write(base, add_meta=True)
            except OSError as e:
                print(
                    time.strftime("[%m/%d %H:%M:%S]"),
                    "[ERROR] Fail to write base catalog for {}".format(host_id),
                )
                if raise_exception:
                    raise e
                traceback.print_exc()
                failed_count += 1
                continue

            del base
            gc.collect()

        print(
            time.strftime("[%m/%d %H:%M:%S]"),
            "All done building base catalogs for {}/{} hosts.".format(nhosts - failed_count, nhosts),
        )

        if return_catalogs:
            return catalogs_to_return

    def generate_clean_specs(self, save_to=None, overwrite=False, **kwargs):
        """
        generate clean spectra from all good base catalogs and save to disk
        """
        defaults = dict(
            hosts="good",
            cuts=Query(C.is_clean2, C.has_spec),
            return_as="stack",
            add_skycoord=False,
        )

        defaults.update(kwargs)

        nhosts = len(self._host_catalog.resolve_id(defaults["hosts"]))
        print(
            time.strftime("[%m/%d %H:%M:%S]"),
            "Generate clean specs for {} hosts".format(nhosts),
        )

        if save_to is not False:
            if save_to is None:
                save_to = self._database["saga_clean_specs"]
            if not isinstance(save_to, (FileObject, DataObject)):
                save_to = FitsTable(save_to)
            print(time.strftime("[%m/%d %H:%M:%S]"), "save to path", save_to.path)

        t = self.load(**defaults)

        if save_to is not False:
            save_to.write(t, overwrite=overwrite, add_meta=True)

        return t

    def load_clean_specs(self, generate_if_not_exist=False, **kwargs):
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
        base = self.load_single(host_id, cuts=Query(C.is_clean2, C.is_galaxy2), add_skycoord=False)

        host_in_base = Query("SATS == 3").filter(base)[0]
        for col in ["r_mag", "gr", "radius", "sb_r", "ba", "phi", "SERSIC", "log_sm", "nuv_sfr"]:
            data[col].append(host_in_base[col])

        basic_targeting_cuts = Query(C.faint_end_limit, C.sat_rcut)
        d = dict()

        d["ptr_total"] = Query(basic_targeting_cuts, C.paper2_targeting_cut)
        d["ptr_has_spec"] = Query(d["ptr_total"], C.has_spec)
        d["ptr_targeted"] = Query(d["ptr_total"], C.has_been_targeted)
        d["ptr_failed"] = Query(d["ptr_targeted"], ~C.has_spec)

        d["nptr_total"] = Query(basic_targeting_cuts, ~C.paper2_targeting_cut)
        d["nptr_has_spec"] = Query(d["nptr_total"], C.has_spec)
        d["nptr_targeted"] = Query(d["nptr_total"], C.has_been_targeted)
        d["nptr_failed"] = Query(d["nptr_targeted"], ~C.has_spec)

        d["specs_total"] = C.has_spec
        d["specs_rvir"] = Query(C.has_spec, C.sat_rcut)
        for k, q in [
            ("ours", C.has_our_specs_only),
            ("aat", C.has_aat_spec),
            ("mmt", C.has_mmt_spec),
        ]:
            d[f"specs_{k}"] = Query(C.has_spec, q)
            d[f"specs_{k}_rvir"] = Query(C.has_spec, q, C.sat_rcut)

        d["sats_total"] = C.is_sat
        d["sats_ours"] = Query(C.is_sat, C.has_our_specs_only)

        for k, q in d.items():
            data[k].append(q.count(base))

        d = dict()
        for k, q in [
            ("gold", C.sample_gold),
            ("gold_silver", (C.sample_gold | C.sample_silver)),
        ]:
            d[f"sats_{k}"] = q
            d[f"sats_{k}_confirmed"] = Query(C.is_sat, q)
            d[f"sats_{k}_confirmed_ours"] = Query(C.is_sat, C.has_our_specs_only, q)
            d[f"sats_{k}_quenched"] = Query(C.is_quenched, q)
            d[f"sats_{k}_confirmed_quenched"] = Query(C.is_sat, C.is_quenched, q)
            d[f"sats_{k}_missed"] = Query(~C.is_sat, q)

        for k, q in d.items():
            data[k].append(q.filter(base, "p_sat_corrected").sum())

        sat_mag = C.is_sat.filter(base, "Mr")
        data["brightest_sat"].append(sat_mag.min() if len(sat_mag) else np.nan)
        sat_mass = C.is_sat.filter(base, "log_sm")
        data["most_massive_sat"].append(sat_mass.max() if len(sat_mass) else np.nan)

        return data

    def generate_object_stats(self, hosts="build_default", save_to=None, overwrite=False, generate_func=None):
        """
        generate object statistics for *hosts*
        """
        hosts = self._host_catalog.resolve_id(hosts)
        print(
            time.strftime("[%m/%d %H:%M:%S]"),
            "Generate object stats for {} hosts".format(len(hosts)),
        )

        if save_to is not False:
            if save_to is None:
                save_to = self._database["host_stats"]
                self._database["host_stats"].clear_cache()
            if not isinstance(save_to, (FileObject, DataObject)):
                save_to = CsvTable(save_to)
            print(time.strftime("[%m/%d %H:%M:%S]"), "Save to path:", save_to.path)

        if generate_func is None:
            generate_func = self._generate_object_stats_single_host

        data = None
        for host in hosts:
            data = generate_func(host, data)

        data = Table(data)
        data.sort("HOSTID")

        if save_to is not False:
            if hosts not in ("build_default", "good") and save_to.isfile() and overwrite:
                data = unique(vstack([data, save_to.read()]), "HOSTID")
                data.sort("HOSTID")
            save_to.write(data, overwrite=overwrite)

        return data

    def load_object_stats(self, generate_if_not_exist=False, **kwargs):
        """
        load object stats from all good base catalogs
        """
        try:
            t = self._database["host_stats"].read()
        except IOError:
            if generate_if_not_exist:
                t = self.generate_object_stats(**kwargs)
            else:
                raise
        return t

    def generate_combined_base_catalog(self, save_to=None, overwrite=False, **kwargs):
        """
        generate combined base catalog from all good base catalogs and save to disk
        """
        defaults = dict(
            hosts="good",
            cuts=Query(C.is_clean2, C.is_galaxy2, (
                Query(C.sat_rcut, C.has_spec | "r_mag < 21" | "dr3_sample > 0")
                | Query(C.has_spec, "SPEC_Z >= 0.002", "SPEC_Z < 0.06")
            )),
            return_as="stack",
            add_skycoord=False,
        )
        defaults.update(kwargs)

        nhosts = len(self._host_catalog.resolve_id(defaults["hosts"]))
        print(
            time.strftime("[%m/%d %H:%M:%S]"),
            "Generate combined base for {} hosts".format(nhosts),
        )

        if save_to is not False:
            if save_to is None:
                save_to = self._database["combined_base"]
            elif not isinstance(save_to, (FileObject, DataObject)):
                save_to = FitsTable(save_to)
            print(time.strftime("[%m/%d %H:%M:%S]"), "save to path", save_to.path)

        t = self.load(**defaults)

        if save_to is not False:
            save_to.write(t, overwrite=overwrite, add_meta=True)

        return t

    def load_combined_base_catalog(self, generate_if_not_exist=False, **kwargs):
        """
        load combined base catalog from all good base catalogs
        """
        try:
            t = self._database["combined_base"].read()
        except IOError:
            if generate_if_not_exist:
                t = self.generate_combined_base_catalog(**kwargs)
            else:
                raise
        return t
