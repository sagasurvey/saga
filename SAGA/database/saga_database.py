import os

from ..spectra import SpectraData
from .core import (CsvTable, DataObject, EcsvTable, FastCsvTable, FileObject,
                   FitsTable, GoogleSheets, NumpyBinary)
from .external_masterlist import EddQuery, HyperledaQuery

__all__ = ["known_google_sheets", "Database", "SpectraData"]

known_google_sheets = {
    "hosts_v1": GoogleSheets("1b3k2eyFjHFDtmHce1xi6JKuj3ATOWYduTBFftx5oPp8", 1471095077),
    "hosts_v2": GoogleSheets("1b3k2eyFjHFDtmHce1xi6JKuj3ATOWYduTBFftx5oPp8", 1765625842),
    "host_stats": GoogleSheets("1b3k2eyFjHFDtmHce1xi6JKuj3ATOWYduTBFftx5oPp8", 1217798377),
    "host_remove": GoogleSheets(
        "1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo",
        1133875164,
        header_start=1,
        include_names=["HOSTID", "flag"],
    ),
    "sdss_remove": GoogleSheets(
        "1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo",
        1379081675,
        header_start=1,
        include_names=["SDSS ID"],
    ),
    "sdss_recover": GoogleSheets(
        "1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo",
        286645731,
        header_start=1,
        include_names=["SDSS ID"],
    ),
    "satellites_named": GoogleSheets("1GJYuhqfKeuJr-IyyGF_NDLb_ezL6zBiX2aeZFHHPr_s", 1),
    "des_remove": GoogleSheets(
        "1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo",
        124397139,
        include_names=["DES_OBJID"],
    ),
    "des_recover": GoogleSheets(
        "1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo",
        797095192,
        include_names=["DES_OBJID"],
    ),
    "decals_remove": GoogleSheets(
        "1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo",
        1574060112,
        include_names=["decals_objid"],
    ),
    "decals_recover": GoogleSheets(
        "1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo",
        1549289503,
        include_names=["decals_objid"],
    ),
    "decals_dr8_remove": GoogleSheets(
        "1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo",
        1157130787,
        include_names=["OBJID"],
    ),
    "shreds_recover": GoogleSheets(
        "1Y3nO7VyU4jDiBPawCs8wJQt2s_PIAKRj-HSrmcWeQZo",
        763845590,
        header_start=1,
        include_names=["OBJID"],
    ),
    "manual_targets_aat2018a": GoogleSheets(
        "1Z8HISgp6ScJ0YZiFK5_TrGDZXY9t2OL3hkCSFJUiC6w", 0, include_names=["OBJID"]
    ),
    "manual_targets_aat2018b": GoogleSheets(
        "1Z8HISgp6ScJ0YZiFK5_TrGDZXY9t2OL3hkCSFJUiC6w",
        478933689,
        include_names=["OBJID"],
    ),
    "manual_targets_mmt2019a": GoogleSheets(
        "1Z8HISgp6ScJ0YZiFK5_TrGDZXY9t2OL3hkCSFJUiC6w",
        85066145,
        include_names=["OBJID"],
    ),
    "manual_targets_mmt2019c": GoogleSheets(
        "1Z8HISgp6ScJ0YZiFK5_TrGDZXY9t2OL3hkCSFJUiC6w",
        1594861456,
        include_names=["OBJID"],
    ),
    "manual_targets_aat2020a_may": GoogleSheets(
        "1Z8HISgp6ScJ0YZiFK5_TrGDZXY9t2OL3hkCSFJUiC6w",
        799699553,
        include_names=["OBJID"],
    ),
    "manual_targets_mmt2020a": GoogleSheets(
        "1Z8HISgp6ScJ0YZiFK5_TrGDZXY9t2OL3hkCSFJUiC6w",
        440730178,
        include_names=["OBJID"],
    ),
    "manual_targets_aat2020a_jul": GoogleSheets(
        "1Z8HISgp6ScJ0YZiFK5_TrGDZXY9t2OL3hkCSFJUiC6w",
        687508250,
        include_names=["OBJID"],
    ),
    "manual_targets_mmt2020b": GoogleSheets(
        "1Z8HISgp6ScJ0YZiFK5_TrGDZXY9t2OL3hkCSFJUiC6w",
        757836001,
        include_names=["OBJID"],
    ),
    "manual_targets_2021_spring": GoogleSheets(
        "1Z8HISgp6ScJ0YZiFK5_TrGDZXY9t2OL3hkCSFJUiC6w",
        1544191200,
        include_names=["OBJID"],
    ),
    "lowz_fields": GoogleSheets("1COd0BjZz0x_9O74Xi0ovVhNZgWeAWNIkVzrZ5UUZr9c", 1883640266),
    "human_selected": GoogleSheets(
        "1WB3aK-DOgtUuSyMQ9Eb-83tJmHsGYWSDI-vHh16Xz8g",
        382210667,
    ),
}

known_google_sheets["manual_targets_mmt2020c"] = known_google_sheets["manual_targets_mmt2020b"]
known_google_sheets["manual_targets_aat2020b"] = known_google_sheets["manual_targets_mmt2020b"]


class Database(object):
    """
    This class provide the interface between the filesystem and other parts of
    the SAGA package.

    Parameters
    ----------
    root_dir : str, optional
        path to the shared SAGA Dropbox root directory
        if you don't have access, set to None.

    Notes
    -----
    Use database[key].read() to access the file.
    Use database[key].path to overwrite the file path.
    For base catalogs, key should be in the form of ('base', host_id).

    Examples
    --------
    >>> import SAGA
    >>> saga_database = SAGA.Database('/path/to/SAGA/Dropbox')
    >>> saga_hosts = SAGA.HostCatalog(saga_database)
    >>> saga_objects = SAGA.ObjectCatalog(saga_database)

    If you don't have access to SAGA Dropbox, you can do:
    >>> saga_database = SAGA.Database()
    >>> saga_database['base', 32].local = 'base_catalogs/base_sql_nsa32.fits.gz'
    >>> saga_database['spectra_clean'].local = 'data/saga_spectra_clean.fits.gz'

    If you don't have internet, you can do things like
    >>> saga_database['hosts_named'].local = 'hosts_named.csv'
    >>> saga_database['hosts_no_flags'].local = 'hosts_no_flags.csv'

    If you have internet now, but want to download the file just in case:
    >>> saga_database['hosts_named'].download('hosts_named.csv')
    >>> saga_database['hosts_no_flags'].download('hosts_no_flags.csv')
    """

    def __init__(self, shared_dir=None, local_dir=None):

        self._shared_dir = shared_dir or os.curdir
        if not os.path.isdir(self._shared_dir):
            raise ValueError("cannot locate {}".format(self._shared_dir))

        self._local_dir = local_dir or os.curdir
        if not os.path.isdir(self._local_dir):
            raise ValueError("cannot locate {}".format(self._local_dir))

        self._tables = {
            "master_list_v1": DataObject(
                FastCsvTable(os.path.join(self._shared_dir, "HostCatalogs", "master_list_v1.csv"))
            ),
            "master_list_v2": DataObject(
                FastCsvTable(os.path.join(self._shared_dir, "HostCatalogs", "master_list_v2.csv"))
            ),
            "saga_spectra_May2017": DataObject(
                FitsTable(
                    os.path.join(self._shared_dir, "Products", "saga_spectra_May2017.fits.gz")
                )
            ),
            "hyperleda_kt12": DataObject(
                HyperledaQuery(
                    "v IS NOT NULL and modbest IS NOT NULL and kt<12 and objtype='G'",
                    "pgc,objname,al2000,de2000,l2,b2,bt,kt,vopt,v,vvir,modz,mod0,modbest,e_modbest",
                ),
                FastCsvTable(
                    os.path.join(self._local_dir, "master_list_v2_sources", "hyperleda_kt12.csv"),
                    comment="#",
                ),
                use_local_first=True,
            ),
            "edd_2mrs_slim": DataObject(
                EddQuery("SELECT pgc, K_tc, H_tc, Vhel from k2m1175"),
                FastCsvTable(
                    os.path.join(self._local_dir, "master_list_v2_sources", "edd_2mrs_slim.csv"),
                    comment="#",
                ),
                use_local_first=True,
            ),
            "edd_lim17_slim": DataObject(
                EddQuery("SELECT PGC, Mhalo from klimgroups"),
                FastCsvTable(
                    os.path.join(self._local_dir, "master_list_v2_sources", "edd_lim17_slim.csv"),
                    comment="#",
                ),
                use_local_first=True,
            ),
            "hipparcos2": DataObject(
                FitsTable("http://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/fits?I/311/hip2.dat.gz"),
                FitsTable(
                    os.path.join(self._local_dir, "master_list_v2_sources", "hipparcos2.fits")
                ),
                use_local_first=True,
            ),
            "nsa_v1.0.1": DataObject(
                FitsTable("https://data.sdss.org/sas/dr14/sdss/atlas/v1/nsa_v1_0_1.fits"),
                FitsTable(
                    os.path.join(self._local_dir, "external_catalogs", "nsa", "nsa_v1_0_1.fits")
                ),
                use_local_first=True,
            ),
            "nsa_v0.1.2": DataObject(
                FitsTable("http://sdss.physics.nyu.edu/mblanton/v0/nsa_v0_1_2.fits"),
                FitsTable(
                    os.path.join(self._local_dir, "external_catalogs", "nsa", "nsa_v0_1_2.fits")
                ),
                use_local_first=True,
            ),
            "spectra_gama_dr2": DataObject(
                FitsTable("http://www.gama-survey.org/dr2/data/cat/SpecCat/v08/SpecObj.fits"),
                FitsTable(
                    os.path.join(
                        self._shared_dir,
                        "Spectra",
                        "Final",
                        "GAMA",
                        "GAMA_SpecObj_dr2.fits",
                    )
                ),
                use_local_first=True,
            ),
            "spectra_gama_dr3": DataObject(
                FitsTable("http://www.gama-survey.org/dr3/data/cat/SpecCat/v27/SpecObj.fits"),
                FitsTable(
                    os.path.join(
                        self._shared_dir,
                        "Spectra",
                        "Final",
                        "GAMA",
                        "GAMA_SpecObj_dr3.fits",
                    )
                ),
                use_local_first=True,
            ),
            "spectra_ozdes_dr1": DataObject(
                FitsTable("http://www.mso.anu.edu.au/ozdes/OzDES-DR1.fits"),
                FitsTable(
                    os.path.join(self._shared_dir, "Spectra", "Final", "OzDES", "OzDES-DR1.fits")
                ),
                use_local_first=True,
            ),
            "spectra_ozdes_dr2": DataObject(
                FitsTable(
                    os.path.join(self._shared_dir, "Spectra", "Final", "OzDES", "OzDES-DR2.fits")
                ),
            ),
            "spectra_2df": DataObject(
                FitsTable(os.path.join(self._shared_dir, "Spectra", "Final", "2dF", "2dF_best.fit"))
            ),
            "spectra_6df": DataObject(
                FitsTable(os.path.join(self._shared_dir, "Spectra", "Final", "6dF", "6dF_DR3.fit"))
            ),
            "spectra_wigglez": DataObject(
                FitsTable(
                    os.path.join(self._shared_dir, "Spectra", "Final", "WiggleZ", "WiggleZ.fits")
                )
            ),
            "spectra_lcrs": DataObject(
                FitsTable(
                    os.path.join(
                        self._shared_dir,
                        "Spectra",
                        "Final",
                        "other",
                        "LCRS_shectman96.fits",
                    )
                )
            ),
            "spectra_ukst": DataObject(
                FitsTable(
                    os.path.join(
                        self._shared_dir,
                        "Spectra",
                        "Final",
                        "other",
                        "UKST_ratcliffe98.fits",
                    )
                )
            ),
            "spectra_palomar": DataObject(
                GoogleSheets("12MM0KdnSb2RdieyUci6DSCnW-h7x0B-bLdxbTkAGhTI", 1276422948),
                CsvTable(
                    os.path.join(
                        self._shared_dir,
                        "Spectra",
                        "Final",
                        "other",
                        "SAGA_Palomar - Final.csv",
                    )
                ),
            ),
            "spectra_slackers": DataObject(
                FitsTable(
                    os.path.join(
                        self._shared_dir,
                        "Spectra",
                        "Final",
                        "other",
                        "Slackers_unpub.fits",
                    )
                )
            ),
            "spectra_2dflens": DataObject(
                FileObject(
                    "http://2dflens.swin.edu.au/2dflens_bestredshifts_goodz_withtypesandmags_final.dat.gz",
                    format="ascii.fast_commented_header",
                ),
                FileObject(
                    os.path.join(self._shared_dir, "Spectra", "Final", "2dF", "2dflens_final.dat"),
                    format="ascii.fast_commented_header",
                ),
                use_local_first=True,
            ),
            "spectra_alfalfa": DataObject(
                FastCsvTable(
                    "http://egg.astro.cornell.edu/alfalfa/data/a100files/a100.code12.table2.190808.csv",
                ),
                FastCsvTable(
                    os.path.join(
                        self._shared_dir,
                        "Spectra",
                        "Final",
                        "ALFALFA",
                        "a100.code12.table2.190808.dat",
                    )
                ),
                use_local_first=True,
            ),
            "spectra_halpha": DataObject(
                FileObject(
                    os.path.join(self._shared_dir, "Spectra", "saga_halpha.dat"),
                    format="ascii.fast_basic",
                )
            ),
        }

        self._tables["spectra_raw_all"] = DataObject(
            SpectraData(
                os.path.join(self._shared_dir, "Spectra", "Final"),
                {
                    "gama": self._tables["spectra_gama_dr3"],
                    "2df": self._tables["spectra_2df"],
                    "2dflens": self._tables["spectra_2dflens"],
                    "6df": self._tables["spectra_6df"],
                    "ozdes": self._tables["spectra_ozdes_dr2"],
                    "wigglez": self._tables["spectra_wigglez"],
                    "lcrs": self._tables["spectra_lcrs"],
                    "ukst": self._tables["spectra_ukst"],
                    "slackers": self._tables["spectra_slackers"],
                    "palomar": self._tables["spectra_palomar"],
                    "alfalfa": self._tables["spectra_alfalfa"],
                },
            )
        )

        gmm_dir = os.path.join(self._shared_dir, "AuxiliaryData", "gmm")
        if os.path.isdir(gmm_dir):
            for fname in os.listdir(gmm_dir):
                if fname.startswith("gmm_parameters") and fname.endswith(".npz"):
                    self._tables[fname[:-4]] = DataObject(NumpyBinary(os.path.join(gmm_dir, fname)))

        footprint_dir = os.path.join(self._shared_dir, "AuxiliaryData", "footprints")
        if os.path.isdir(footprint_dir):
            for fname in os.listdir(footprint_dir):
                if "_footprint_" in fname and fname.endswith(".npy"):
                    name, _, info = fname.partition("_footprint_")
                    nside, order = info[:-4].split("_")
                    obj = NumpyBinary(os.path.join(footprint_dir, fname))
                    obj.nside = int(nside)
                    obj.order = order.upper()
                    self._tables["footprint_" + name] = DataObject(obj)

        for k, v in known_google_sheets.items():
            self._tables[k] = DataObject(v, CsvTable(), cache_in_memory=True)

        self._tables["hosts_v2"].local = CsvTable(
            os.path.join(self._shared_dir, "HostCatalogs", "host_list_v2.csv")
        )

        self._tables["hosts_v1"].local = CsvTable(
            os.path.join(self._shared_dir, "HostCatalogs", "host_list_v1.csv")
        )

        self._tables["lowz_fields"].local = CsvTable(
            os.path.join(self._shared_dir, "HostCatalogs", "lowz_fields.csv")
        )

        self._tables["hosts"] = self._tables["hosts_v2"]
        self._tables["master_list"] = self._tables["master_list_v2"]

        self._file_path_pattern = {
            "base_paper2": os.path.join(
                self._local_dir, "base_catalogs_paper2", "base_v2_{}.fits.gz"
            ),
            "base_v2p1": os.path.join(self._local_dir, "base_catalogs_v2.1", "base_v2_{}.fits.gz"),
            "base_v3": os.path.join(self._local_dir, "base_catalogs_v3", "base_v3_{}.fits.gz"),
            "base_v2": os.path.join(self._local_dir, "base_catalogs", "base_v2_{}.fits.gz"),
            "base_v1": os.path.join(self._local_dir, "base_catalogs", "base_v1_{}.fits.gz"),
            "base_v0p1": os.path.join(
                self._shared_dir, "Paper1", "base_catalogs", "base_sql_{}.fits.gz"
            ),
            "sdss_dr16": os.path.join(
                self._local_dir, "external_catalogs", "sdss_dr16", "{}.fits.gz"
            ),
            "sdss_dr14": os.path.join(
                self._local_dir, "external_catalogs", "sdss_dr14", "{}.fits.gz"
            ),
            "sdss_dr12": os.path.join(
                self._local_dir, "external_catalogs", "sdss_dr12", "{}.fits.gz"
            ),
            "wise": os.path.join(self._local_dir, "external_catalogs", "wise", "{}.fits.gz"),
            "des_dr1": os.path.join(
                self._local_dir, "external_catalogs", "des_dr1", "{}_des_dr1.fits.gz"
            ),
            "decals_dr67": os.path.join(
                self._local_dir, "external_catalogs", "decals", "{}_decals.fits.gz"
            ),
            "decals_dr8": os.path.join(
                self._local_dir,
                "external_catalogs",
                "decals_dr8",
                "{}_decals_dr8.fits.gz",
            ),
            "decals_dr9": os.path.join(
                self._local_dir,
                "external_catalogs",
                "decals_dr9",
                "{}_decals_dr9.fits.gz",
            ),
            "gaia": os.path.join(
                self._local_dir, "external_catalogs", "astrometric", "{}_gaia.ecsv"
            ),
        }

        self._possible_base_versions = tuple(
            k.partition("_")[2] for k in self._file_path_pattern if k.startswith("base_")
        )
        self._file_path_pattern["sdss"] = self._file_path_pattern["sdss_dr14"]
        self._file_path_pattern["des"] = self._file_path_pattern["des_dr1"]
        self._file_path_pattern["decals"] = self._file_path_pattern["decals_dr67"]
        self.set_default_base_version()

    def _add_derived_data(self):
        t = FitsTable(self.base_file_path_pattern.format("saga_clean_specs"))
        if "saga_clean_specs" in self._tables:
            self._tables["saga_clean_specs"].remote = t
        else:
            self._tables["saga_clean_specs"] = DataObject(t)

        stats_path = self.base_file_path_pattern.format("host_stats")
        for ext in (".fits.gz", ".fits"):
            if stats_path.endswith(ext):
                stats_path = stats_path[: -len(ext)]
                break
        stats_path += ".csv"
        t = FastCsvTable(stats_path)

        if "host_stats" in self._tables:
            self._tables["host_stats"].local = t
            self._tables["host_stats"].use_local_first = True
            self._tables["host_stats"].clear_cache()
        else:
            self._tables["host_stats"] = DataObject(
                known_google_sheets["host_stats"],
                t,
                use_local_first=True,
                cache_in_memory=True,
            )

    def _set_file_path_pattern(self, key, value):
        self._file_path_pattern[key] = value
        keys_to_del = [
            k for k in self._tables if isinstance(k, tuple) and len(k) == 2 and k[0] == key
        ]
        for k in keys_to_del:
            del self._tables[k]
        if key == "base":
            self._add_derived_data()

    def get_file_path_pattern(self, key):
        return self._file_path_pattern[key]

    @property
    def base_file_path_pattern(self):
        return self._file_path_pattern["base"]

    @base_file_path_pattern.setter
    def base_file_path_pattern(self, value):
        self._set_file_path_pattern("base", value)

    @property
    def sdss_file_path_pattern(self):
        return self._file_path_pattern["sdss"]

    @sdss_file_path_pattern.setter
    def sdss_file_path_pattern(self, value):
        self._set_file_path_pattern("sdss", value)

    @property
    def wise_file_path_pattern(self):
        return self._file_path_pattern["wise"]

    @wise_file_path_pattern.setter
    def wise_file_path_pattern(self, value):
        self._set_file_path_pattern("wise", value)

    @property
    def des_file_path_pattern(self):
        return self._file_path_pattern["des"]

    @des_file_path_pattern.setter
    def des_file_path_pattern(self, value):
        self._set_file_path_pattern("des", value)

    @property
    def decals_file_path_pattern(self):
        return self._file_path_pattern["decals"]

    @decals_file_path_pattern.setter
    def decals_file_path_pattern(self, value):
        self._set_file_path_pattern("decals", value)

    def __getitem__(self, key):
        if key in self._tables:
            return self._tables[key]

        if isinstance(key, tuple) and len(key) == 2 and key[0] in self._file_path_pattern:
            path = self._file_path_pattern[key[0]].format(key[1])
            data_type = EcsvTable if path.endswith(".ecsv") else FitsTable
            self._tables[key] = DataObject(data_type(path))
            return self._tables[key]

        raise KeyError("cannot find {} in database".format(key))

    def __setitem__(self, key, value):
        self._tables[key] = value

    def __delitem__(self, key):
        del self._tables[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        return sorted((k for k in self._tables if not isinstance(k, tuple)))

    def _ipython_key_completions_(self):
        return list(self.keys())

    def resolve_base_version(self, version=None):
        """
        resolve `version` into `build_version` and `version_postfix`
        """

        if version is None:
            return 3, ""
        version = str(version).lower().strip()
        if version in ("paper1", "p1"):
            return 0, "v0p1"
        if version == "paper2":
            return 2, "paper2"
        if not version.startswith("v"):
            version = "v" + version
        while version.endswith(".0"):
            version = version[:-2]
        version = version.replace(".", "p")
        if version not in self._possible_base_versions:
            raise ValueError("version value unknown!")
        return int(version[1]), version

    def set_default_base_version(self, version=None):
        _, version_postfix = self.resolve_base_version(version)
        version_postfix = version_postfix or "v3"
        self.base_file_path_pattern = self._file_path_pattern["base_" + version_postfix]
