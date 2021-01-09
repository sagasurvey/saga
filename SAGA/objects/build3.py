"""
base catalog building pipeline 3.0
"""
from itertools import chain
import numpy as np
from astropy.table import vstack
from easyquery import Query, QueryMaker

from ..spectra import extract_nsa_spectra, extract_sdss_spectra
from ..utils import fill_values_by_query, add_skycoord, get_remove_flag
from . import build, build2


def filter_nearby_object(catalog, host, radius_deg=1.001, remove_coord=True):
    if catalog is not None:
        catalog = build2.filter_nearby_object(catalog, host, radius_deg, remove_coord)
        if len(catalog):
            return catalog


def _fill_not_finite(arr, fill_value=99.0):
    return np.where(np.isfinite(arr), arr, fill_value)


def _ivar2err(ivar):
    with np.errstate(divide="ignore"):
        return 1.0 / np.sqrt(ivar)


def _median_cut(cut):
    def _median_cut_this(*arrays, cut=cut):
        return np.median(np.stack(arrays), axis=0) >= cut
    return _median_cut_this


MERGED_CATALOG_COLUMNS = list(
    chain(
        (
            "OBJID",
            "RA",
            "DEC",
            "REMOVE",
            "is_galaxy",
            "morphology_info",
            "radius",
            "radius_err",
            "ba",
            "phi",
        ),
        (b + "_mag" for b in "ugriz"),
        (b + "_err" for b in "ugriz"),
    )
)


def prepare_decals_catalog_for_merging(catalog, to_remove=None, to_recover=None):
    """
    Refs:
    - https://www.legacysurvey.org/dr9/files/#sweep-catalogs-region-sweep
    - https://www.legacysurvey.org/release/
    - https://www.legacysurvey.org/dr9/description/
    - https://www.legacysurvey.org/dr9/updates/
    - https://www.legacysurvey.org/dr9/catalogs/#ellipticities
    - https://www.legacysurvey.org/dr9/bitmasks/#maskbits
    """

    is_decam = Query("RELEASE == 9010")
    is_bass_mzls = Query("RELEASE == 9011")
    count_bass_mzls = Query("DEC > 32.375", (lambda coord: coord.galactic.b.deg > 0, "coord"))

    assert (is_decam | is_bass_mzls).mask(catalog).all(), "Must use DR9 LS catalogs"

    # Remove duplicated Gaia entries
    catalog = QueryMaker.not_equal("TYPE", "DUP").filter(catalog)

    # Retain only unique entries
    catalog = add_skycoord(catalog)
    catalog = (is_decam | count_bass_mzls).filter(catalog)
    del catalog["coord"]

    # Remove objects fainter than 3 mag of survey limit
    catalog = Query("r_mag < 23.75").filter(catalog)

    # Assign OBJID
    release_short = np.where(is_decam.mask(catalog), 90, 91)
    catalog["OBJID"] = (
        release_short * np.int64(1e16)
        + catalog["BRICKID"] * np.int64(1e10)
        + catalog["OBJID"].astype(np.int64)
    )
    del release_short

    # Do galaxy/star separation
    in_sga = QueryMaker.equal("REF_CAT", "L3")
    catalog["is_galaxy"] = QueryMaker.not_equal("TYPE", "PSF").mask(catalog)
    catalog["is_galaxy"] &= Query("abs(PMRA * sqrt(PMRA_IVAR)) < 2", "abs(PMDEC * sqrt(PMDEC_IVAR)) < 2").mask(catalog)
    catalog["is_galaxy"] |= in_sga.mask(catalog)

    # Rename/add columns
    catalog["morphology_info"] = catalog["TYPE"].getfield("<U1").view(np.int32)
    catalog["radius"] = catalog["SHAPE_R"]
    catalog["radius_err"] = _fill_not_finite(_ivar2err(catalog["SHAPE_R_IVAR"]), 9999.0)
    e_abs = np.hypot(catalog["SHAPE_E1"], catalog["SHAPE_E2"])
    catalog["ba"] = (1 - e_abs) / (1 + e_abs)
    catalog["phi"] = np.rad2deg(np.arctan2(catalog["SHAPE_E2"], catalog["SHAPE_E1"]) * 0.5)
    del e_abs

    for band in "grz":
        catalog[f"{band}_mag"] = _fill_not_finite(catalog[f"{band}_mag"])
        catalog[f"{band}_err"] = _fill_not_finite(catalog[f"{band}_err"])

    for band in "ui":
        catalog["{}_mag".format(band)] = 99.0
        catalog["{}_err".format(band)] = 99.0

    has_band = {band: Query(f"NOBS_{band} > 0", f"FLUX_{band} > 0", f"FLUX_IVAR_{band} > 0") for band in "GRZ"}

    to_remove = [] if to_remove is None else to_remove
    to_recover = [] if to_recover is None else to_recover

    remove_queries = [
        QueryMaker.isin("OBJID", to_remove),
        ~has_band["R"],
        "(MASKBITS >> 1) % 2 > 0",
        "(MASKBITS >> 5) % 2 > 0",
        "(MASKBITS >> 6) % 2 > 0",
        "(MASKBITS >> 7) % 2 > 0",
        "(MASKBITS >> 12) % 2 > 0",
        "(MASKBITS >> 13) % 2 > 0",
        (_median_cut(0.35), "FRACMASKED_G", "FRACMASKED_R", "FRACMASKED_Z"),
        (_median_cut(5), "FRACFLUX_G", "FRACFLUX_R", "FRACFLUX_Z"),
        Query("RCHISQ_W1 >= 50", (_median_cut(50), "RCHISQ_G", "RCHISQ_R", "RCHISQ_Z")),
        Query(has_band["G"], Query("g_mag - r_mag < -1") | Query("g_mag - r_mag > 4")),
        Query(has_band["Z"], Query("r_mag - z_mag < -1") | Query("r_mag - z_mag > 4")),
        Query(
            "NOBS_W1 > 0",
            "FLUX_IVAR_W1 > 0",
            "FLUX_W1 <= 0",
            "NOBS_W2 > 0",
            "FLUX_IVAR_W2 > 0",
            "FLUX_W2 <= 0",
        ),
    ]

    catalog["REMOVE"] = get_remove_flag(catalog, remove_queries)
    fill_values_by_query(
        catalog,
        Query(in_sga, "REMOVE % 2 == 0") | QueryMaker.isin("OBJID", to_recover),
        {"REMOVE": 0}
    )

    catalog = catalog[MERGED_CATALOG_COLUMNS]
    catalog["survey"] = "decals"
    catalog["OBJID_decals"] = catalog["OBJID"]
    catalog["REMOVE_decals"] = catalog["REMOVE"]

    return catalog


def build_full_stack(  # pylint: disable=unused-argument
    host,
    decals=None,
    decals_remove=None,
    decals_recover=None,
    sdss=None,
    nsa=None,
    spectra=None,
    halpha=None,
    shreds_recover=None,
    debug=None,
    **kwargs,
):
    """
    This function calls all needed functions to complete the full stack of building
    a base catalog (for a single host), in the following order:

    Returns
    -------
    base : astropy.table.Table
    """
    if decals is None:
        raise ValueError("No photometry catalog to build!")
    base = prepare_decals_catalog_for_merging(decals, decals_remove, decals_recover)
    del decals, decals_remove, decals_recover

    base = build.add_host_info(base, host)
    base = build2.add_columns_for_spectra(base)

    nsa = filter_nearby_object(nsa, host)
    all_spectra = [
        extract_sdss_spectra(sdss),
        extract_nsa_spectra(nsa),
        filter_nearby_object(spectra, host),
    ]
    del sdss, nsa, spectra

    all_spectra = [s for s in all_spectra if s is not None]
    if all_spectra:
        all_spectra = vstack(all_spectra, "exact")
        if halpha is not None:
            all_spectra = build2.add_halpha_to_spectra(all_spectra, halpha)
        base = build2.add_spectra(base, all_spectra, debug=debug)
    del all_spectra

    base = build2.remove_shreds_near_spec_obj(base, shreds_recover=shreds_recover)

    if "RHOST_KPC" in base.colnames:  # has host info (i.e., not for LOWZ)
        base = build2.remove_too_close_to_host(base)
        base = build.find_satellites(base, version=2)
        base = build2.identify_host(base)

    base = build2.add_surface_brightness(base)
    base = build.add_stellar_mass(base)

    return base
