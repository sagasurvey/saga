"""
base catalog building pipeline 3.0
"""
from itertools import chain

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import vstack
from easyquery import Query, QueryMaker

from ..spectra import extract_nsa_spectra, extract_sdss_spectra
from ..utils import add_skycoord, fill_values_by_query, get_remove_flag
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


def _n_or_more_gt(cols, n, cut):
    def _n_or_more_gt_this(*arrays, n=n, cut=cut):
        return np.count_nonzero((np.stack(arrays) > cut), axis=0) >= n

    return Query((_n_or_more_gt_this,) + tuple(cols))


def _n_or_more_lt(cols, n, cut):
    def _n_or_more_lt_this(*arrays, n=n, cut=cut):
        return np.count_nonzero((np.stack(arrays) < cut), axis=0) >= n

    return Query((_n_or_more_lt_this,) + tuple(cols))


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

    is_decam = Query("RELEASE == 9010") | Query("RELEASE == 9012")
    is_bass_mzls = Query("RELEASE == 9011")
    count_bass_mzls = Query("DEC > 32.375", (lambda coord: coord.galactic.b.deg > 0, "coord"))

    assert (is_decam | is_bass_mzls).mask(catalog).all(), "Must use DR9 LS catalogs"

    # Remove duplicated Gaia entries
    catalog = QueryMaker.not_equal("TYPE", "DUP").filter(catalog)

    # Retain only unique entries
    catalog = add_skycoord(catalog)
    catalog = (is_decam | count_bass_mzls).filter(catalog)
    del catalog["coord"]

    # Remove objects fainter than 3 mag of survey limit (r < 23.75)
    flux_limit = 10 ** ((22.5 - 23.75) / 2.5)
    catalog = Query("FLUX_R >= MW_TRANSMISSION_R * {}".format(flux_limit)).filter(catalog)

    # Assign OBJID
    release_short = (catalog["RELEASE"] // 1000) * 10 + catalog["RELEASE"] % 10
    catalog["OBJID"] = (
        release_short * np.int64(1e16)
        + catalog["BRICKID"] * np.int64(1e10)
        + catalog["OBJID"].astype(np.int64)
    )
    del release_short

    # Do galaxy/star separation
    catalog["is_galaxy"] = QueryMaker.not_equal("TYPE", "PSF").mask(catalog)
    catalog["is_galaxy"] &= Query(
        "abs(PMRA * sqrt(PMRA_IVAR)) < 2", "abs(PMDEC * sqrt(PMDEC_IVAR)) < 2"
    ).mask(catalog)
    catalog["is_galaxy"] |= QueryMaker.equal("REF_CAT", "L3").mask(catalog)  # SGA

    # Rename/add columns
    catalog["morphology_info"] = catalog["TYPE"].getfield("<U1").view(np.int32)
    catalog["radius"] = catalog["SHAPE_R"]
    catalog["radius_err"] = _fill_not_finite(_ivar2err(catalog["SHAPE_R_IVAR"]), 9999.0)
    e_abs = np.hypot(catalog["SHAPE_E1"], catalog["SHAPE_E2"])
    catalog["ba"] = (1 - e_abs) / (1 + e_abs)
    catalog["phi"] = np.rad2deg(np.arctan2(catalog["SHAPE_E2"], catalog["SHAPE_E1"]) * 0.5)
    del e_abs

    for BAND in ("G", "R", "Z", "W1", "W2", "W3", "W4"):
        catalog[f"SIGMA_{BAND}"] = catalog[f"FLUX_{BAND}"] * np.sqrt(catalog[f"FLUX_IVAR_{BAND}"])
        catalog[f"SIGMA_GOOD_{BAND}"] = np.where(catalog[f"RCHISQ_{BAND}"] < 100, catalog[f"SIGMA_{BAND}"], 0.0)

    # Recalculate mag and err to fix old bugs in the raw catalogs
    const = 2.5 / np.log(10)
    for band in "grz":
        BAND = band.upper()
        with np.errstate(divide="ignore", invalid="ignore"):
            catalog[f"{band}_mag"] = _fill_not_finite(
                22.5 - const * np.log(catalog[f"FLUX_{BAND}"] / catalog[f"MW_TRANSMISSION_{BAND}"])
            )
            catalog[f"{band}_err"] = _fill_not_finite(const / np.abs(catalog[f"SIGMA_{BAND}"]))

    for band in "ui":
        catalog[f"{band}_mag"] = 99.0
        catalog[f"{band}_err"] = 99.0

    to_remove = [] if to_remove is None else to_remove
    to_recover = [] if to_recover is None else to_recover

    sigma_grz = [f"SIGMA_GOOD_{b}" for b in "GRZ"]
    sigma_wise = [f"SIGMA_GOOD_W{b}" for b in range(1, 5)]
    fracmasked_grz = [f"FRACMASKED_{b}" for b in "GRZ"]
    fracflux_grz = [f"FRACFLUX_{b}" for b in "GRZ"]

    remove_queries = [
        QueryMaker.isin("OBJID", to_remove),  # 0
        "(MASKBITS >> 1) % 2 > 0",  # 1
        "(MASKBITS >> 5) % 2 > 0",  # 2
        "(MASKBITS >> 6) % 2 > 0",  # 3
        "(MASKBITS >> 7) % 2 > 0",  # 4
        "(MASKBITS >> 12) % 2 > 0",  # 5
        "(MASKBITS >> 13) % 2 > 0",  # 6
        Query(_n_or_more_lt(sigma_grz, 3, 80), _n_or_more_gt(fracflux_grz, 2, 1)),  # 7
        Query(_n_or_more_lt(sigma_grz, 3, 80), _n_or_more_gt(fracmasked_grz, 2, 0.2)),  # 8
        Query(_n_or_more_lt(sigma_grz, 3, 80), _n_or_more_lt(sigma_wise, 2, -5)),  # 9
        Query(_n_or_more_lt(sigma_grz, 3, 30) | _n_or_more_lt(sigma_grz, 2, 20), _n_or_more_lt(sigma_wise, 2, 10)),  # 10
    ]

    catalog["REMOVE"] = get_remove_flag(catalog, remove_queries)
    has_ref = Query((np.char.isalnum, "REF_CAT"))
    manual_recover = QueryMaker.isin("OBJID", to_recover)
    fill_values_by_query(
        catalog,
        Query(has_ref, "REMOVE % 2 == 0") | manual_recover,
        {"REMOVE": 0},
    )

    catalog = catalog[MERGED_CATALOG_COLUMNS]
    catalog["survey"] = "decals"
    catalog["OBJID_decals"] = catalog["OBJID"]
    catalog["REMOVE_decals"] = catalog["REMOVE"]

    return catalog


SPEC_MATCHING_ORDER = (
    (Query("REMOVE == 0", "r_mag < 21", "sep < 1"), "sep"),
    (Query("REMOVE % 2 == 0", "r_mag < 21", "sep < 1"), "sep"),
    (Query("REMOVE == 0", "r_mag < 21", "is_galaxy", "sep_norm < 0.5", "sep < 30"), "r_mag"),
    (Query("REMOVE == 0", "r_mag < 21", "is_galaxy", "sep_norm < 1", "sep < 30"), "r_mag"),
    (Query("REMOVE == 0", "r_mag < 21", "is_galaxy", "sep_norm < 2", "sep < 10"), "r_mag"),
    (Query("REMOVE == 0", "r_mag < 21", "is_galaxy", "sep < 3"), "r_mag"),
    (Query("REMOVE == 0", "is_galaxy", "sep_norm < 0.5", "sep < 10"), "r_mag"),
    (Query("REMOVE == 0", "is_galaxy", "sep_norm < 1", "sep < 10"), "r_mag"),
    (Query("REMOVE == 0", "is_galaxy", "sep < 3"), "r_mag"),
    (Query("REMOVE  > 0", "r_mag < 21", "is_galaxy", "sep_norm < 0.5", "sep < 30"), "r_mag"),
    (Query("REMOVE  > 0", "r_mag < 21", "is_galaxy", "sep_norm < 1", "sep < 30"), "r_mag"),
    (Query("REMOVE  > 0", "r_mag < 21", "is_galaxy", "sep_norm < 2", "sep < 10"), "r_mag"),
    (Query("REMOVE  > 0", "r_mag < 21", "is_galaxy", "sep < 3"), "r_mag"),
    (Query("REMOVE  > 0", "is_galaxy", "sep_norm < 0.5", "sep < 10"), "r_mag"),
    (Query("REMOVE  > 0", "is_galaxy", "sep_norm < 1", "sep < 10"), "r_mag"),
    (Query("REMOVE  > 0", "is_galaxy", "sep < 3"), "r_mag"),
    (Query("REMOVE == 0", "r_mag < 21", "sep < 10"), "sep"),
    (Query("REMOVE  > 0", "r_mag < 21", "sep < 10"), "sep"),
    (Query("REMOVE == 0", "sep < 10"), "sep"),
    (Query("sep < 10"), "sep"),
)


def add_spec_phot_sep(base):
    has_any_spec_mask = base["ZQUALITY"] > -1

    spec_coord = SkyCoord(
        np.where(has_any_spec_mask, base["RA_spec"], 0),
        np.where(has_any_spec_mask, base["DEC_spec"], 0),
        unit="deg",
    )

    base = add_skycoord(base)
    sep = base["coord"].separation(spec_coord).arcsec
    sep = np.where(has_any_spec_mask, sep, -1)
    base["spec_phot_sep"] = sep
    return base


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
        base = build2.add_spectra(
            base, all_spectra, debug=debug, matching_order=SPEC_MATCHING_ORDER
        )
    del all_spectra

    base = build2.remove_shreds_near_spec_obj(base, shreds_recover=shreds_recover)

    if "RHOST_KPC" in base.colnames:  # has host info (i.e., not for LOWZ)
        base = build2.remove_too_close_to_host(base)
        base = build.find_satellites(base, version=2)
        base = build2.identify_host(base)

    base = build2.add_surface_brightness(base)
    base = build.add_stellar_mass(base)
    base = add_spec_phot_sep(base)

    return base
