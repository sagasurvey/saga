import numpy as np
from easyquery import Query, QueryMaker

from ..utils import fill_values_by_query
from .common import SPEED_OF_LIGHT, ensure_specs_dtype
from .manual_fixes import fixes_sdss_spec_by_objid, fixes_nsa_v1_spec_by_nsaid

__all__ = ["extract_sdss_spectra", "extract_nsa_spectra"]


def extract_sdss_spectra(sdss):
    if sdss is None or not len(sdss):
        return
    specs = Query("SPEC_Z > -1.0").filter(sdss["RA", "DEC", "SPEC_Z", "SPEC_Z_ERR", "SPEC_Z_WARN", "OBJID"])
    if not len(specs):
        return
    specs["ZQUALITY"] = np.where(specs["SPEC_Z_WARN"] == 0, 4, 1)
    del specs["SPEC_Z_WARN"]

    for objid, fixes in fixes_sdss_spec_by_objid.items():
        fill_values_by_query(specs, QueryMaker.equal("OBJID", objid), fixes)

    specs.rename_column("OBJID", "SPECOBJID")
    specs["TELNAME"] = "SDSS"
    specs["MASKNAME"] = "SDSS"
    specs["HELIO_CORR"] = True
    return ensure_specs_dtype(specs)


def extract_nsa_spectra(nsa):
    if nsa is None:
        return
    specs = nsa["RA", "DEC", "Z", "ZSRC", "NSAID"]
    specs["TELNAME"] = "NSA"
    specs["SPEC_Z_ERR"] = 20 / SPEED_OF_LIGHT
    specs["ZQUALITY"] = 4
    specs["HELIO_CORR"] = True

    # This will break for NSA v0.1.2, but that may be ok
    for nsaid, fixes in fixes_nsa_v1_spec_by_nsaid.items():
        fill_values_by_query(specs, QueryMaker.equal("NSAID", nsaid), fixes)

    specs.rename_column("Z", "SPEC_Z")
    specs.rename_column("ZSRC", "MASKNAME")
    specs.rename_column("NSAID", "SPECOBJID")
    return ensure_specs_dtype(specs)
