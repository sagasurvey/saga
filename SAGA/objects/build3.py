"""
base catalog building pipeline 3.0
"""
from itertools import chain

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import Table, vstack, unique
from easyquery import Query, QueryMaker

from ..spectra import extract_nsa_spectra, extract_sdss_spectra
from ..utils import add_skycoord, fill_values_by_query, get_remove_flag, calc_normalized_dist, get_coord
from ..utils.distance import v2z
from . import build, build2
from . import cuts as C


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


SGA_COLUMNS = ["SGA_ID", "PGC", "RA_LEDA", "DEC_LEDA", "Z_LEDA", "D25_LEDA", "PA_LEDA", "BA_LEDA", "REF", "SMA_MOMENT", "D26", "PA", "BA"]
GALEX_COLUMNS = ["ra", "dec", "nuv_exptime", "nuv_mag", "nuv_magerr", "nuv_artifact", "fuv_exptime", "fuv_mag", "fuv_magerr", "fuv_artifact"]

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
            "sma",
            "REF_CAT",
            "SGA_ID",
        ),
        (f"{b}_mag" for b in "ugriz"),
        (f"{b}_err" for b in "ugriz"),
        (f"w{b}_mag" for b in "1234"),
        (f"w{b}_err" for b in "1234"),
        (f"{b}_fibermag" for b in "grz"),
    )
)


def prepare_objid(catalog):
    release_short = (catalog["RELEASE"] // 1000) * 10 + catalog["RELEASE"] % 10
    return release_short * np.int64(1e16) + catalog["BRICKID"] * np.int64(1e10) + catalog["OBJID"].astype(np.int64)


def prepare_decals_catalog_for_merging(catalog, to_remove=None, to_recover=None, trim=True):
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
    catalog = (is_decam ^ count_bass_mzls).filter(catalog)
    del catalog["coord"]

    # Remove objects fainter than r = 23 (unless g or z < 22.5)
    # flux = 10 ** ((22.5 - mag) / 2.5)
    catalog = (
        Query("FLUX_R >= MW_TRANSMISSION_R * 0.63")  # r <= 23
        | Query("FLUX_G >= MW_TRANSMISSION_G")  # g <= 22.5
        | Query("FLUX_Z >= MW_TRANSMISSION_Z")  # z <= 22.5
    ).filter(catalog)

    # Assign OBJID
    catalog["OBJID"] = prepare_objid(catalog)

    # Do galaxy/star separation
    catalog["is_galaxy"] = QueryMaker.not_equal("TYPE", "PSF").mask(catalog)

    # Bright (r < 17) stars that are misclassified as galaxies
    flux_limit = 10 ** ((22.5 - 17) / 2.5)
    bright_stars = Query(
        "SHAPE_R < 1",
        "FLUX_R >= MW_TRANSMISSION_R * {}".format(flux_limit),
        (Query("abs(PMRA * sqrt(PMRA_IVAR)) >= 2") | Query("abs(PMDEC * sqrt(PMDEC_IVAR)) >= 2")),
    ).mask(catalog)

    # Fix galaxy/star separation with bright_stars and SGA masks
    catalog["is_galaxy"] &= ~bright_stars
    catalog["is_galaxy"] |= QueryMaker.equal("REF_CAT", "L3").mask(catalog)

    # Rename/add columns
    catalog["morphology_info"] = catalog["TYPE"].getfield("<U1").view(np.int32)

    e_abs = np.hypot(catalog["SHAPE_E1"], catalog["SHAPE_E2"])
    catalog["ba"] = (1 - e_abs) / (1 + e_abs)
    catalog["phi"] = np.rad2deg(np.arctan2(catalog["SHAPE_E2"], catalog["SHAPE_E1"]) * 0.5)
    del e_abs

    # SHAPE_R is half-light semi-major axis
    d26_to_eff = 3.0  # multiply by 3 to account for the ratio b/w sma and effective radius
    catalog["sma"] = catalog["SHAPE_R"] * d26_to_eff

    # multiply by sqrt(ba) to get effective radius
    sqrt_ba = np.sqrt(catalog["ba"])
    catalog["radius"] = catalog["SHAPE_R"] * sqrt_ba
    catalog["radius_err"] = _fill_not_finite(_ivar2err(catalog["SHAPE_R_IVAR"]) * sqrt_ba, 9999.0)
    del sqrt_ba

    # Recalculate mag and err to fix old bugs in the raw catalogs
    const = 2.5 / np.log(10)
    for BAND in ("U", "G", "R", "I", "Z", "W1", "W2", "W3", "W4"):
        band = BAND.lower()

        if f"FLUX_{BAND}" not in catalog.colnames:
            catalog[f"{band}_mag"] = np.float32(99.0)
            catalog[f"{band}_err"] = np.float32(99.0)
            continue

        catalog[f"SIGMA_{BAND}"] = catalog[f"FLUX_{BAND}"] * np.sqrt(catalog[f"FLUX_IVAR_{BAND}"])
        catalog[f"SIGMA_GOOD_{BAND}"] = np.where(catalog[f"RCHISQ_{BAND}"] < 100, catalog[f"SIGMA_{BAND}"], 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            catalog[f"{band}_mag"] = _fill_not_finite(
                22.5 - const * np.log(catalog[f"FLUX_{BAND}"] / catalog[f"MW_TRANSMISSION_{BAND}"])
            )
            catalog[f"{band}_err"] = _fill_not_finite(const / np.abs(catalog[f"SIGMA_{BAND}"]))
            if f"FIBERFLUX_{BAND}" in catalog.colnames:
                catalog[f"{band}_fibermag"] = _fill_not_finite(
                    22.5 - const * np.log(catalog[f"FIBERFLUX_{BAND}"] / catalog[f"MW_TRANSMISSION_{BAND}"])
                )

    # BASS-DECaLS r mag correction
    catalog["r_mag"] = np.where(
        Query(is_bass_mzls, C.valid_g_mag).mask(catalog),
        -0.0382 * (catalog["g_mag"] - catalog["r_mag"]) + 0.0108 + catalog["r_mag"],
        catalog["r_mag"],
    )

    allmask_grz = [f"ALLMASK_{b}" for b in "GRZ"]
    sigma_grz = [f"SIGMA_GOOD_{b}" for b in "GRZ"]
    sigma_wise = [f"SIGMA_GOOD_W{b}" for b in range(1, 5)]
    fracflux_grz = [f"FRACFLUX_{b}" for b in "GRZ"]

    remove_queries = {
        1: _n_or_more_gt(allmask_grz, 2, 0),
        2: Query("FLUX_R <= 0"),
        3: Query("FLUX_G <= 0", "FLUX_Z <= 0"),
        4: (_n_or_more_lt(sigma_grz, 3, 60) & _n_or_more_gt(fracflux_grz, 1, 2)),
        5: _n_or_more_lt(sigma_grz + sigma_wise, 7, 20),
    }

    catalog["REMOVE"] = np.where(np.char.isalnum(catalog["REF_CAT"]), 0, get_remove_flag(catalog, remove_queries))
    catalog["REF_CAT"] = np.char.strip(catalog["REF_CAT"])
    catalog["SGA_ID"] = np.where(QueryMaker.equal("REF_CAT", "L3").mask(catalog), catalog["REF_ID"], -1)

    if trim:
        catalog = catalog[MERGED_CATALOG_COLUMNS]
    catalog["survey"] = "decals"
    catalog["OBJID_decals"] = catalog["OBJID"]
    catalog["REMOVE_decals"] = catalog["REMOVE"]

    if to_remove is not None:
        catalog["REMOVE"] |= np.isin(catalog["OBJID"], to_remove).astype(np.int)

    if to_recover is not None:
        idx = np.flatnonzero(np.isin(catalog["OBJID"], to_recover))
        catalog["REMOVE"][idx] = 0

    return catalog


def prepare_delve_catalog_for_merging(catalog, to_remove=None, to_recover=None, trim=True):
    cols = ["objid", "ra", "dec", "remove"]
    catalog.rename_columns(cols, [c.upper() for c in cols])
    for b in "griz":
        catalog[f"{b}_mag"] = catalog[f"{b}_mag"].astype(np.float32)
        catalog[f"{b}_err"] = catalog[f"{b}_err"].astype(np.float32)
    catalog["u_mag"] = np.float32(99.0)
    catalog["u_err"] = np.float32(99.0)
    catalog["morphology_info"] = catalog["morphology_info"].astype(np.int32)
    catalog["is_galaxy"] = catalog["is_galaxy"].astype(bool)
    catalog["radius_err"] = catalog["radius"] * 1.0e-4
    catalog["REF_CAT"] = "  "
    catalog["SGA_ID"] = -1

    if trim:
        catalog = catalog[MERGED_CATALOG_COLUMNS]
    catalog["survey"] = "delve"
    catalog["OBJID_delve"] = catalog["OBJID"]
    catalog["REMOVE_delve"] = catalog["REMOVE"]

    if to_remove is not None:
        catalog["REMOVE"] |= np.isin(catalog["OBJID"], to_remove).astype(np.int)

    if to_recover is not None:
        idx = np.flatnonzero(np.isin(catalog["OBJID"], to_recover))
        catalog["REMOVE"][idx] = 0

    return catalog


SPEC_MATCHING_ORDER = (
    (Query("SPEC_Z < 0.002", "REMOVE == 0", "is_galaxy == 0", "sep < 5"), "sep"),
    (Query("SPEC_Z < 0.002", "REMOVE % 2 == 0", "is_galaxy == 0", "sep < 5"), "sep"),
    (Query("SPEC_Z < 0.002", "REMOVE % 2 == 0", "is_galaxy == 0", "r_mag < 17"), "sep"),
    (Query("REMOVE == 0", "sep < 0.5", "r_mag < 21.5", Query("is_galaxy == 0") | "sep_norm < 1"), "r_mag"),
    (Query("REMOVE == 0", QueryMaker.equal("REF_CAT", "L3"), "sep_norm < 0.5"), "r_mag"),
    (Query("REMOVE == 0", "r_mag < 21", "sep_norm < 0.5"), "r_mag"),
    (Query("REMOVE == 0", "r_mag < 21", "sep_norm < 1"), "r_mag"),
    (Query("REMOVE == 0", "r_mag < 21", Query("sep_norm < 1.5") | "sep < 5"), "r_mag"),
    (Query("REMOVE % 2 == 0", Query("sep_norm < 1") | "sep < 3"), "r_mag"),
    (Query("REMOVE % 2 == 0", "sep < 10"), "sep"),
    (Query("sep < 10"), "sep"),
)


def apply_manual_fixes(base):

    fill_values_by_query(
        base,
        QueryMaker.isin(
            "OBJID",
            [
                915147800000001877,
                901039870000003755,
                903341560000000816,
                903442030000003771,
                903233250000000577,
                900906460000008966,
                900906460000009106,
                900896600000003831,
                900809940000005200,
                900800510000009934,
                900809940000005279,
                901029370000006715,
                901039840000008707,
                901039860000005713,
                901018960000005296,
                901039870000003755,
                901050380000000229,
                900896580000007542,
                902577060000004414,
                900452390000002003,
                904733050000000124,
            ],
        ),
        dict(is_galaxy=True),
    )

    # NSA (v1.0.1) 343647 (255.5115, 22.9355)
    fill_values_by_query(
        base,
        QueryMaker.equal("OBJID", 904604600000003107),
        dict(
            is_galaxy=True,
            REMOVE=0,
            sma=28.0,
            radius=11.7,
            radius_err=0.5,
            ba=0.7,
            phi=97.0,
            g_mag=13.86,
            r_mag=13.14,
            z_mag=12.54,
            g_err=0.005,
            r_err=0.002,
            z_err=0.005,
            REF_CAT="N1",
        ),
    )

    # (330.14926, -43.140017)
    fill_values_by_query(
        base,
        QueryMaker.equal("OBJID", 901050380000007338),
        dict(sma=100),
    )

    # (224.594532 -1.090942)
    fill_values_by_query(
        base,
        QueryMaker.equal("OBJID", 903255060000002115),
        dict(sma=270, ba=0.15, phi=81.0),
    )

    # (198.1743458, 22.829911)
    fill_values_by_query(
        base,
        QueryMaker.equal("OBJID", 904589200000000836),
        dict(sma=60, ba=0.55, phi=50.0),
    )

    # (163.0159, 10.1479)
    fill_values_by_query(
        base,
        QueryMaker.equal("OBJID", 903897920000000718),
        dict(sma=135, ba=0.75, phi=40.0),
    )

    # (40.9350, -29.0035)
    fill_values_by_query(
        base,
        QueryMaker.equal("OBJID", 901704030000004690),
        dict(sma=170, phi=83.0),
    )

    # (330.1370, -43.3897)
    fill_values_by_query(
        base,
        QueryMaker.equal("OBJID", 901039870000000760),
        dict(sma=36.0),
    )

    # (35.4017, -5.5212)
    fill_values_by_query(
        base,
        QueryMaker.equal("OBJID", 902988610000000241),
        dict(sma=138.0),
    )

    # (138.5198, 40.1133)
    fill_values_by_query(
        base,
        QueryMaker.equal("OBJID", 915434150000002223),
        dict(sma=150.0),
    )

    # Flipped coordinates for a galaxy/star pair in nsa126115
    fill_values_by_query(
        base,
        QueryMaker.equal("OBJID", 904488130000004194),
        dict(RA=0.4979568838114891, DEC=20.985493504498578),
    )
    fill_values_by_query(
        base,
        QueryMaker.equal("OBJID", 904488130000004168),
        dict(RA=0.4993274319772168, DEC=20.986906437087658),
    )

    return base


def match_sga(base, sga):

    # ensure sga table is sorted by ID
    if (np.ediff1d(sga["SGA_ID"]) < 0).any():
        sga.sort("SGA_ID")

    has_sga_idx = np.flatnonzero(base["SGA_ID"] > -1)

    sga_idx = np.searchsorted(sga["SGA_ID"], base["SGA_ID"][has_sga_idx])
    sga_idx[sga_idx >= len(sga)] = -1
    matched = base["SGA_ID"][has_sga_idx] == sga["SGA_ID"][sga_idx]
    if not matched.all():
        has_sga_idx = has_sga_idx[matched]
        sga_idx = sga_idx[matched]

    sga = sga[sga_idx]

    return has_sga_idx, sga


def add_sga(base, sga):

    matching_idx, sga = match_sga(base, sga)
    base["OBJ_PGC"] = np.int64(-1)
    base["OBJ_PGC"][matching_idx] = sga["PGC"]
    sga["base_idx"] = matching_idx
    del matching_idx

    REF_FLAG = "LG"

    sga_this = Query(
        "SMA_MOMENT > 0",
        "D26 > 0",
        QueryMaker.isin(
            "PGC",
            [4074752, 1681736, 4541156],
            assume_unique=True,
            invert=True,
        ),
    ).filter(sga)

    if len(sga_this):
        idx = sga_this["base_idx"]
        base["ba"][idx] = sga_this["BA"]
        base["phi"][idx] = sga_this["PA"]
        base["sma"][idx] = np.minimum(sga_this["SMA_MOMENT"] * 1.3333, sga_this["D26"] * 30.0)
        base["REMOVE"][idx] = 0
        base["is_galaxy"][idx] = True
        base["REF_CAT"][idx] = REF_FLAG

    sga_this = QueryMaker.isin(
        "PGC",
        [
            10060, 12342, 16065, 16152, 25217, 25999, 26068, 26932, 27734, 29009,
            34325, 36124, 36238, 38115, 44025, 49378, 50191, 50369, 51484, 51913,
            53499, 54120, 58470, 66619, 67818, 68590, 71834, 71926, 101219, 139206,
            1016399, 2045076, 3087653,
        ],
        assume_unique=True,
    ).filter(sga)

    if len(sga_this):
        idx = sga_this["base_idx"]
        base["ba"][idx] = sga_this["BA_LEDA"]
        base["phi"][idx] = sga_this["PA_LEDA"]
        base["sma"][idx] = sga_this["D25_LEDA"] * 1.5 * 30.0
        base["REMOVE"][idx] = 0
        base["is_galaxy"][idx] = True
        base["REF_CAT"][idx] = REF_FLAG

    fill_values_by_query(base, QueryMaker.equal("REF_CAT", "L3"), {"REF_CAT": ""})
    fill_values_by_query(base, QueryMaker.equal("REF_CAT", REF_FLAG), {"REF_CAT": "L3"})

    # later we only need entries with specs
    del sga["base_idx"]
    sga = Query("Z_LEDA > -1").filter(sga)

    return base, sga


def add_sga_specs(base, sga):

    matching_idx, sga = match_sga(base, sga)

    has_spec = base["ZQUALITY"][matching_idx] >= 3
    has_no_spec = base["ZQUALITY"][matching_idx] == -1
    has_poor_spec = ~(has_spec | has_no_spec)
    has_consistent_z = np.abs(base["SPEC_Z"][matching_idx] - sga["Z_LEDA"]) < v2z(500)

    for i in np.flatnonzero(has_spec & has_consistent_z):
        base_idx = matching_idx[i]
        base["SPEC_REPEAT"][base_idx] = build2._join_spec_repeat(base["SPEC_REPEAT"][base_idx], ["SGA"])

    for i in np.flatnonzero(has_poor_spec | has_spec | has_no_spec):
        base_idx = matching_idx[i]
        base["SPEC_REPEAT_ALL"][base_idx] = build2._join_spec_repeat(base["SPEC_REPEAT_ALL"][base_idx], ["SGA"])

    for i in np.flatnonzero(has_no_spec | has_poor_spec):
        base_idx = matching_idx[i]
        sga_this = sga[i]
        base["SPEC_REPEAT"][base_idx] = "SGA"
        base["RA_spec"][base_idx] = sga_this["RA_LEDA"]
        base["DEC_spec"][base_idx] = sga_this["DEC_LEDA"]
        base["SPEC_Z"][base_idx] = sga_this["Z_LEDA"]
        base["SPEC_Z_ERR"][base_idx] = v2z(100)
        base["ZQUALITY"][base_idx] = 4
        base["SPECOBJID"][base_idx] = str(sga_this["SGA_ID"])
        base["MASKNAME"][base_idx] = str(sga_this["REF"])
        base["TELNAME"][base_idx] = "SGA"
        base["HELIO_CORR"][base_idx] = True

    return base


def cap_sma_value(base):
    mask = Query(
        "is_galaxy",
        "r_mag >= 16",
        "r_mag < 30",
        "sma * sqrt(ba) >= exp(-0.3 * r_mag + 7.9)",
    ).mask(base)
    base["sma"] = np.where(mask, np.exp(-0.3 * base["r_mag"] + 7.9) / np.sqrt(base["ba"]), base["sma"])
    return base


def identify_host(base):
    fill_values_by_query(base, C.obj_is_host3, {"SATS": 3, "REMOVE": 0})
    return base


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
    base["spec_phot_sep"] = sep.astype(np.float32)
    return base


def add_galex(base, galex):

    galex_bands = ("nuv", "fuv")

    for band in galex_bands:
        base[f"RA_{band}"] = np.float64(np.nan)
        base[f"DEC_{band}"] = np.float64(np.nan)
        base[f"{band}_mag"] = np.float32(np.nan)
        base[f"{band}_magerr"] = np.float32(np.nan)

    if galex is None:
        return base

    galex_stack = []
    for band in galex_bands:
        cols = ["ra", "dec", f"{band}_mag", f"{band}_magerr"]
        galex_this = Query(f"{band}_exptime > 0", f"{band}_exptime < 1e20").filter(galex, cols)
        galex_this.rename_column(f"{band}_mag", "mag")
        galex_this.rename_column(f"{band}_magerr", "magerr")
        galex_this["band"] = band
        galex_stack.append(galex_this)
    galex = vstack(galex_stack)
    del galex_stack, galex_this

    base = add_skycoord(base)
    galex_coord = get_coord(galex)

    matches = Table(search_around_sky(base["coord"], galex_coord, 40 * u.arcsec), names=["idx_base", "idx_galex", "sep", "sep_3d"])
    del matches["sep_3d"]
    matches["sep"] = matches["sep"].to("arcsec")
    matches["sep_norm"] = calc_normalized_dist(
        galex["ra"][matches["idx_galex"]],
        galex["dec"][matches["idx_galex"]],
        base["RA"][matches["idx_base"]],
        base["DEC"][matches["idx_base"]],
        base["sma"][matches["idx_base"]],
        base["ba"][matches["idx_base"]],
        base["phi"][matches["idx_base"]],
        multiplier=1,
    )

    matches = (Query("sep <= 3") | Query("sep_norm <= 1")).filter(matches)
    matches["invalid_mag"] = 1 - (galex["mag"][matches["idx_galex"]] < 99).astype(np.int32)
    matches["band"] = galex["band"][matches["idx_galex"]]
    matches.sort(["invalid_mag", "sep"])
    matches = unique(matches, keys=["idx_base", "band"])

    for band in galex_bands:
        idx_base, idx_galex = QueryMaker.equal("band", band).filter(matches, ["idx_base", "idx_galex"]).itercols()
        base[f"RA_{band}"][idx_base] = galex["ra"][idx_galex]
        base[f"DEC_{band}"][idx_base] = galex["dec"][idx_galex]
        base[f"{band}_mag"][idx_base] = galex["mag"][idx_galex]
        base[f"{band}_magerr"][idx_base] = galex["magerr"][idx_galex]

    return base


def build_full_stack(  # pylint: disable=unused-argument
    host,
    decals=None,
    decals_remove=None,
    decals_recover=None,
    sdss=None,
    nsa=None,
    sga=None,
    spectra=None,
    halpha=None,
    debug=None,
    delve=None,
    delve_remove=None,
    delve_recover=None,
    galex=None,
    **kwargs,
):
    """
    This function calls all needed functions to complete the full stack of building
    a base catalog (for a single host), in the following order:

    Returns
    -------
    base : astropy.table.Table
    """
    if decals is not None:
        base = prepare_decals_catalog_for_merging(decals, decals_remove, decals_recover)
        del decals, decals_remove, decals_recover
    elif delve is not None:
        base = prepare_delve_catalog_for_merging(delve, delve_remove, delve_recover)
        del delve, delve_remove, delve_recover
    else:
        raise ValueError("No photometry catalog to build!")

    base = build.add_host_info(base, host)
    base = build2.add_columns_for_spectra(base)

    if sga is not None:
        base, sga = add_sga(base, sga)

    base = cap_sma_value(base)
    base = apply_manual_fixes(base)

    all_spectra = [
        filter_nearby_object(spectra, host),
        extract_sdss_spectra(sdss),
        extract_nsa_spectra(filter_nearby_object(nsa, host)),
    ]
    del sdss, nsa, spectra

    all_spectra = [s for s in all_spectra if s is not None]
    if all_spectra:
        all_spectra = vstack(all_spectra, "exact")
        base = build2.add_spectra(base, all_spectra, debug=debug, matching_order=SPEC_MATCHING_ORDER)
    del all_spectra

    if sga is not None:
        base = add_sga_specs(base, sga)
        del sga

    base = build2.remove_shreds_near_spec_obj(base)

    if halpha is not None:
        base = build2.add_halpha(base, halpha, match_by_objid=True)
        del halpha

    base = add_galex(base, galex)
    del galex

    if "RHOST_KPC" in base.colnames:  # has host info (i.e., not for LOWZ)
        base = build.find_satellites(base, version=3)
        base = identify_host(base)

    base = build2.add_surface_brightness(base)
    base = build.add_stellar_mass(base)
    base = add_spec_phot_sep(base)

    return base
