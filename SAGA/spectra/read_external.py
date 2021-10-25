import numpy as np
from easyquery import Query, QueryMaker

from ..database import FastCsvTable, FileObject, FitsTable
from ..utils import fill_values_by_query
from .common import SPEED_OF_LIGHT, ensure_specs_dtype
from .manual_fixes import fixes_2df_spec_by_objid, fixes_ozdes_spec_by_objid

__all__ = [
    "read_gama",
    "read_2df",
    "read_2dflens",
    "read_6df",
    "read_ozdes",
    "read_ukst",
    "read_lcrs",
    "read_slackers",
]


def read_gama(file_path):
    if not hasattr(file_path, "read"):
        file_path = FitsTable(file_path)
    specs = file_path.read()["RA", "DEC", "SPECID", "CATAID", "NQ", "Z", "SURVEY"]

    specs = Query("NQ >= 3", (lambda x: x != "SDSS", "SURVEY")).filter(specs)
    del specs["SURVEY"]

    specs.rename_column("SPECID", "SPECOBJID")
    specs.rename_column("CATAID", "MASKNAME")
    specs.rename_column("NQ", "ZQUALITY")
    specs.rename_column("Z", "SPEC_Z")

    fill_values_by_query(specs, "ZQUALITY > 4", {"ZQUALITY": 4})

    specs["SPEC_Z_ERR"] = 60 / SPEED_OF_LIGHT
    specs["TELNAME"] = "GAMA"
    specs["HELIO_CORR"] = True

    return ensure_specs_dtype(specs)


def read_2df(file_path):
    if not hasattr(file_path, "read"):
        file_path = FitsTable(file_path)
    specs = file_path.read()["RAJ2000", "DEJ2000", "Name", "z", "q_z", "n_z"]

    # 3 = probably galaxy, 4 = definite galaxy, 6 = confirmed star
    specs = Query("q_z >= 3").filter(specs)

    specs.rename_column("Name", "SPECOBJID")
    specs.rename_column("RAJ2000", "RA")
    specs.rename_column("DEJ2000", "DEC")
    specs.rename_column("q_z", "ZQUALITY")
    specs.rename_column("z", "SPEC_Z")
    specs.rename_column("n_z", "EM_ABS")

    # LOWZ REDSHIFTS IN 2dF ARE PROBLEMATIC, SET ZQ = 2
    fill_values_by_query(specs, "ZQUALITY > 4", {"ZQUALITY": 4})
    fill_values_by_query(specs, Query("ZQUALITY == 3", "SPEC_Z < 0.05"), {"ZQUALITY": 2})

    specs["SPEC_Z_ERR"] = 60 / SPEED_OF_LIGHT
    specs["TELNAME"] = "2dF"
    specs["MASKNAME"] = "2dF"
    specs["HELIO_CORR"] = True

    for objid, fixes in fixes_2df_spec_by_objid.items():
        fill_values_by_query(specs, QueryMaker.equals("SPECOBJID", objid), fixes)

    return ensure_specs_dtype(specs)


def read_2dflens(file_path):
    if not hasattr(file_path, "read"):
        file_path = FileObject(file_path, format="ascii.fast_commented_header")
    specs = file_path.read()["R.A.", "Dec.", "z", "qual"]

    # 3 = probably galaxy, 4 = definite galaxy, 6 = confirmed star
    specs = Query("qual >= 3").filter(specs)

    specs.rename_column("R.A.", "RA")
    specs.rename_column("Dec.", "DEC")
    specs.rename_column("qual", "ZQUALITY")
    specs.rename_column("z", "SPEC_Z")

    # LOWZ REDSHIFTS IN 2dFLENS ARE PROBLEMATIC, SET ZQ = 2
    fill_values_by_query(specs, "ZQUALITY > 4", {"ZQUALITY": 4})
    fill_values_by_query(specs, Query("ZQUALITY == 3", "SPEC_Z < 0.05"), {"ZQUALITY": 2})

    specs["SPECOBJID"] = ["2dFLenS_{}".format(i) for i in range(len(specs))]
    specs["SPEC_Z_ERR"] = 60 / SPEED_OF_LIGHT
    specs["TELNAME"] = "2dFLen"  # max # of char is 6
    specs["MASKNAME"] = "2dFLenS"
    specs["HELIO_CORR"] = True

    return ensure_specs_dtype(specs)


def read_6df(file_path):
    if not hasattr(file_path, "read"):
        file_path = FitsTable(file_path)
    specs = file_path.read()["RAJ2000", "DEJ2000", "_6dFGS", "cz", "e_cz", "q_cz"]

    # 3 = probably galaxy, 4 = definite galaxy, 6 = confirmed star
    specs = Query("q_cz >= 3").filter(specs)
    specs["SPEC_Z"] = specs["cz"] / SPEED_OF_LIGHT
    specs["SPEC_Z_ERR"] = specs["e_cz"] / SPEED_OF_LIGHT
    del specs["cz"], specs["e_cz"]

    specs.rename_column("_6dFGS", "SPECOBJID")
    specs.rename_column("RAJ2000", "RA")
    specs.rename_column("DEJ2000", "DEC")
    specs.rename_column("q_cz", "ZQUALITY")

    fill_values_by_query(specs, "ZQUALITY > 4", {"ZQUALITY": 4})

    specs["TELNAME"] = "6dF"
    specs["MASKNAME"] = "6dF"
    specs["HELIO_CORR"] = True

    return ensure_specs_dtype(specs)


def read_ozdes(file_path):
    if not hasattr(file_path, "read"):
        file_path = FitsTable(file_path)

    specs = file_path.read()

    try:
        # DR2
        specs = specs["ozdes_id", "alpha_j2000", "delta_j2000", "z", "qop"]
    except KeyError:
        # DR1
        specs = specs["OzDES_ID", "RA", "DEC", "z", "flag"]
        specs.rename_column("OzDES_ID", "SPECOBJID")
        specs.rename_column("z", "SPEC_Z")
        specs.rename_column("flag", "ZQUALITY")
    else:
        specs.rename_column("ozdes_id", "SPECOBJID")
        specs.rename_column("alpha_j2000", "RA")
        specs.rename_column("delta_j2000", "DEC")
        specs.rename_column("z", "SPEC_Z")
        specs.rename_column("qop", "ZQUALITY")

    # flag 3 = probably galaxy, 4 = definite galaxy, 6 = confirmed star
    specs["ZQUALITY"] = specs["ZQUALITY"].astype(np.int16)
    specs = Query("ZQUALITY >= 3").filter(specs)

    fill_values_by_query(specs, "ZQUALITY > 4", {"ZQUALITY": 4})

    specs["SPEC_Z_ERR"] = 60 / SPEED_OF_LIGHT
    specs["TELNAME"] = "OzDES"
    specs["MASKNAME"] = "OzDES"
    specs["HELIO_CORR"] = True

    for objid, fixes in fixes_ozdes_spec_by_objid.items():
        fill_values_by_query(specs, QueryMaker.equals("SPECOBJID", objid), fixes)

    return ensure_specs_dtype(specs)


def read_wigglez(file_path):
    if not hasattr(file_path, "read"):
        file_path = FitsTable(file_path)

    specs = file_path.read()["RAJ2000", "DEJ2000", "WiggleZ", "z", "q_z"]

    # 3 = probably galaxy, 4 = definite galaxy, 6 = confirmed star
    specs = Query("q_z >= 3").filter(specs)

    specs.rename_column("WiggleZ", "SPECOBJID")
    specs.rename_column("RAJ2000", "RA")
    specs.rename_column("DEJ2000", "DEC")
    specs.rename_column("q_z", "ZQUALITY")
    specs.rename_column("z", "SPEC_Z")

    # LOWZ REDSHIFTS IN WIGGLEZ ARE PROBLEMATIC, SET ZQ = 2
    fill_values_by_query(specs, "ZQUALITY > 4", {"ZQUALITY": 4})
    fill_values_by_query(specs, Query("ZQUALITY == 3", "SPEC_Z < 0.05"), {"ZQUALITY": 2})

    specs["SPEC_Z_ERR"] = 60 / SPEED_OF_LIGHT
    specs["TELNAME"] = "WIGGZ"
    specs["MASKNAME"] = "WIGGZ"
    specs["HELIO_CORR"] = True

    return ensure_specs_dtype(specs)


def read_ukst(file_path):
    """REFERENCE:  DURHAM UKST
    http://adsabs.harvard.edu/abs/1998MNRAS.300..417R
    """
    if not hasattr(file_path, "read"):
        file_path = FitsTable(file_path)
    specs = file_path.read()["_RAJ2000", "_DEJ2000", "DUGRS", "RV"]

    specs = specs[specs["RV"] != np.iinfo(np.int32).min]
    specs["SPEC_Z"] = specs["RV"].astype(np.float64) / SPEED_OF_LIGHT
    del specs["RV"]

    specs.rename_column("_RAJ2000", "RA")
    specs.rename_column("_DEJ2000", "DEC")
    specs.rename_column("DUGRS", "SPECOBJID")

    specs["SPEC_Z_ERR"] = 100 / SPEED_OF_LIGHT
    specs["ZQUALITY"] = 4
    specs["TELNAME"] = "UKST"
    specs["HELIO_CORR"] = True

    return ensure_specs_dtype(specs)


def read_lcrs(file_path):
    """REFERENCE:  LCRS
    http://vizier.cfa.harvard.edu/viz-bin/VizieR-2
    http://adsabs.harvard.edu/abs/1996ApJ...470..172S
    """
    if not hasattr(file_path, "read"):
        file_path = FitsTable(file_path)
    specs = file_path.read()["_RAJ2000", "_DEJ2000", "Field", "p", "cz", "e_cz"]

    specs = specs[specs["cz"] != np.iinfo(np.int32).min]
    specs["SPEC_Z"] = specs["cz"].astype(np.float64) / SPEED_OF_LIGHT
    specs["SPEC_Z_ERR"] = specs["e_cz"].astype(np.float64) / SPEED_OF_LIGHT
    del specs["cz"], specs["e_cz"]

    specs.rename_column("_RAJ2000", "RA")
    specs.rename_column("_DEJ2000", "DEC")
    specs.rename_column("Field", "MASKNAME")
    specs.rename_column("p", "SPECOBJID")

    specs["ZQUALITY"] = 4
    specs["TELNAME"] = "LCRS"
    specs["HELIO_CORR"] = True
    return ensure_specs_dtype(specs)


def read_slackers(file_path):
    """REFERENCE:  SLACKERS
    FILE WAS FOUND IN MG's EMAIL.   UNPUBLISHED SURVEY FROM LAS CAMPANAS 2.5M
    """
    if not hasattr(file_path, "read"):
        file_path = FitsTable(file_path)
    specs = file_path.read()[
        "OBJ_RA",
        "OBJ_DEC",
        "WFCCD_ZQUAL",
        "WFCCD_Z",
        "WFCCD_ZERR",
        "WFCCD_MASKNAME",
        "WFCCD_SLITID",
    ]

    specs = Query("WFCCD_ZQUAL >= 3").filter(specs)

    specs.rename_column("OBJ_RA", "RA")
    specs.rename_column("OBJ_DEC", "DEC")
    specs.rename_column("WFCCD_ZQUAL", "ZQUALITY")
    specs.rename_column("WFCCD_Z", "SPEC_Z")
    specs.rename_column("WFCCD_ZERR", "SPEC_Z_ERR")
    specs.rename_column("WFCCD_MASKNAME", "MASKNAME")
    specs.rename_column("WFCCD_SLITID", "SPECOBJID")

    specs["TELNAME"] = "slack"
    specs["HELIO_CORR"] = True

    return ensure_specs_dtype(specs)


def read_alfalfa(file_path):

    if not hasattr(file_path, "read"):
        file_path = FastCsvTable(file_path)

    specs = file_path.read()[
        "AGCNr",
        "RAdeg_HI",
        "DECdeg_HI",
        "RAdeg_OC",
        "DECdeg_OC",
        "Vhelio",
        "W50",
        "HIcode",
    ]

    specs.rename_column("AGCNr", "SPECOBJID")

    valid_oc_coord = Query(
        "abs(RAdeg_OC) + abs(DECdeg_OC) > 0",
        Query(np.isfinite, "RAdeg_OC"),
        Query(np.isfinite, "DECdeg_OC"),
    )

    specs["RA"] = np.where(valid_oc_coord, specs["RAdeg_OC"], specs["RAdeg_HI"])
    specs["DEC"] = np.where(valid_oc_coord, specs["DECdeg_OC"], specs["DECdeg_HI"])
    del specs["RAdeg_OC"], specs["RAdeg_HI"], specs["DECdeg_OC"], specs["DECdeg_HI"]

    specs["SPEC_Z"] = specs["Vhelio"].astype(np.float64) / SPEED_OF_LIGHT
    specs["SPEC_Z_ERR"] = specs["W50"].astype(np.float64) / SPEED_OF_LIGHT / np.sqrt(2 * np.log(2))
    specs["ZQUALITY"] = np.where(specs["HIcode"] == 1, 4, 3)
    del specs["Vhelio"], specs["W50"], specs["HIcode"]

    specs["TELNAME"] = "ALFALF"
    specs["MASKNAME"] = "ALFALFA"
    specs["HELIO_CORR"] = True

    return ensure_specs_dtype(specs)
