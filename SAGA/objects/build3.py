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


def _median_cut(cut):
    def _median_cut_this(*arrays, cut=cut):
        return np.median(np.stack(arrays), axis=0) >= cut
    return _median_cut_this


def _flux_sigma_cut(sigma_cut, n_bands=2):
    def _flux_sigma_cut_this(*args, cut=sigma_cut, n=n_bands):
        i = len(args) // 2
        sigma_iter = ((flux * np.sqrt(ivar)) for flux, ivar in zip(args[:i], args[i:]))
        return np.count_nonzero(np.stack(list(sigma_iter)) < sigma_cut, axis=0) >= n
    return _flux_sigma_cut_this


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

    # Remove objects fainter than 3 mag of survey limit (r < 23.75)
    flux_limit = 10 ** ((22.5 - 23.75) / 2.5)
    catalog = Query("FLUX_R >= MW_TRANSMISSION_R * {}".format(flux_limit)).filter(catalog)

    # Assign OBJID
    release_short = np.where(is_decam.mask(catalog), 90, 91)
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

    # Recalculate mag and err to fix old bugs in the raw catalogs
    with np.errstate(divide="ignore", invalid="ignore"):
        for band in "grz":
            BAND = band.upper()
            catalog[f"{band}_mag"] = _fill_not_finite(
                22.5 - 2.5 * np.log10(catalog[f"FLUX_{BAND}"] / catalog[f"MW_TRANSMISSION_{BAND}"])
            )
            catalog[f"{band}_err"] = _fill_not_finite(
                2.5
                / np.log(10)
                / np.abs(catalog[f"FLUX_{BAND}"])
                / np.sqrt(catalog[f"FLUX_IVAR_{BAND}"])
            )

    for band in "ui":
        catalog["{}_mag".format(band)] = 99.0
        catalog["{}_err".format(band)] = 99.0

    to_remove = [] if to_remove is None else to_remove
    to_recover = [] if to_recover is None else to_recover

    remove_queries = [
        QueryMaker.isin("OBJID", to_remove),
        "(MASKBITS >> 1) % 2 > 0",
        "(MASKBITS >> 5) % 2 > 0",
        "(MASKBITS >> 6) % 2 > 0",
        "(MASKBITS >> 7) % 2 > 0",
        "(MASKBITS >> 12) % 2 > 0",
        "(MASKBITS >> 13) % 2 > 0",
        Query(
            (
                _flux_sigma_cut(1),
                *(f"FLUX_{b}" for b in "GRZ"),
                *(f"FLUX_IVAR_{b}" for b in "GRZ"),
            )
        ),
        Query((_median_cut(0.35), "FRACMASKED_G", "FRACMASKED_R", "FRACMASKED_Z")),
        Query((_median_cut(5), "FRACFLUX_G", "FRACFLUX_R", "FRACFLUX_Z")),
        Query("RCHISQ_W1 >= 50", (_median_cut(50), "RCHISQ_G", "RCHISQ_R", "RCHISQ_Z")),
        Query(
            "FLUX_G * sqrt(FLUX_IVAR_G) >= 1",
            Query("g_mag - r_mag < -1") | Query("g_mag - r_mag > 4"),
        ),
        Query(
            "FLUX_Z * sqrt(FLUX_IVAR_Z) >= 1",
            Query("r_mag - z_mag < -1") | Query("r_mag - z_mag > 4"),
        ),
        Query(
            (
                _flux_sigma_cut(-1),
                *(f"FLUX_W{i}" for i in range(1, 5)),
                *(f"FLUX_IVAR_W{i}" for i in range(1, 5)),
            )
        ),
        Query(
            (
                _flux_sigma_cut(1, n_bands=3),
                *(f"FLUX_W{i}" for i in range(1, 5)),
                *(f"FLUX_IVAR_W{i}" for i in range(1, 5)),
            ),
            *(f"NOBS_W{i} > 0" for i in range(1, 5)),
        ),
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
    (Query("REMOVE  > 0", "r_mag < 21", "sep < 1"), "sep"),
    (Query("REMOVE == 0", "is_galaxy", "r_mag < 21", "sep_norm < 1", "sep < 20"), "sep_norm"),
    (Query("REMOVE  > 0", "is_galaxy", "r_mag < 21", "sep_norm < 1", "sep < 20"), "sep_norm"),
    (Query("REMOVE == 0", "is_galaxy", "r_mag < 21", "sep_norm < 2", "sep < 10"), "r_mag"),
    (Query("REMOVE  > 0", "is_galaxy", "r_mag < 21", "sep_norm < 2", "sep < 10"), "r_mag"),
    (Query("REMOVE == 0", "is_galaxy", "sep_norm < 1", "sep < 10"), "sep_norm"),
    (Query("sep < 10", "r_mag < 21"), "sep"),
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
