"""
base catalog building pipeline 3.0
"""
import numexpr as ne
import numpy as np
from astropy.table import vstack
from easyquery import Query

from ..spectra import extract_nsa_spectra, extract_sdss_spectra
from ..utils import fill_values_by_query
from . import build, build2


def filter_nearby_object(catalog, host, radius_deg=1.001, remove_coord=True):
    if catalog is not None:
        catalog = build2.filter_nearby_object(catalog, host, radius_deg, remove_coord)
        if len(catalog):
            return catalog


def prepare_decals_catalog_for_merging(catalog, to_remove=None, to_recover=None):
    """
    Ref: https://www.legacysurvey.org/dr9/files/#sweep-catalogs-region-sweep
    """
    assert catalog["RELEASE"][0] >= 9000, "Must use DR9+ LS catalogs"

    catalog["OBJID"] = (
        (catalog["RELEASE"] // 100) * np.int64(1e16)
        + catalog["BRICKID"] * np.int64(1e10)
        + catalog["OBJID"].astype(np.int64)
    )

    catalog["is_galaxy"] = (catalog["TYPE"] != "PSF") & (catalog["TYPE"] != "DUP")
    catalog["morphology_info"] = catalog["TYPE"].getfield("<U1").view(np.int32)
    catalog["radius"] = (
        catalog["FRACDEV"] * catalog["SHAPEDEV_R"]
        + (1.0 - catalog["FRACDEV"]) * catalog["SHAPEEXP_R"]
    )
    catalog["radius_err"] = np.float32(0)
    mask = catalog["TYPE"] == "DEV"
    catalog["radius_err"][mask] = 1.0 / np.sqrt(catalog["SHAPEDEV_R_IVAR"][mask])
    mask = (catalog["TYPE"] == "EXP") | (catalog["TYPE"] == "REX")
    catalog["radius_err"][mask] = 1.0 / np.sqrt(catalog["SHAPEEXP_R_IVAR"][mask])
    mask = catalog["TYPE"] == "COMP"
    catalog["radius_err"][mask] = ne.evaluate(
        "sqrt(f**2.0 / dev_ivar + (1.0-f)**2.0 / exp_ivar + (r_dev - r_exp)**2.0 / f_ivar)",
        {
            "f": catalog["FRACDEV"][mask],
            "r_dev": catalog["SHAPEDEV_R"][mask],
            "r_exp": catalog["SHAPEEXP_R"][mask],
            "f_ivar": catalog["FRACDEV_IVAR"][mask],
            "dev_ivar": catalog["SHAPEDEV_R_IVAR"][mask],
            "exp_ivar": catalog["SHAPEEXP_R_IVAR"][mask],
        },
        {},
    )
    del mask

    e_tot = catalog["FRACDEV"] * np.hypot(catalog["SHAPEDEV_E1"], catalog["SHAPEDEV_E2"]) + (
        1.0 - catalog["FRACDEV"]
    ) * np.hypot(catalog["SHAPEEXP_E1"], catalog["SHAPEEXP_E2"])
    catalog["b_to_a"] = np.sqrt((1 - e_tot) / (1 + e_tot))

    fill_values_by_query(
        catalog,
        Query("radius > 0", ~Query((np.isfinite, "radius_err"))),
        {"radius_err": 9999.0},
    )

    for band in "uiy":
        catalog["{}_mag".format(band)] = 99.0
        catalog["{}_err".format(band)] = 99.0

    remove_queries = [
        "NOBS_R <= 0",
        "FLUX_IVAR_R <= 0",
        (lambda *x: np.stack(x).any(axis=0), "ALLMASK_G", "ALLMASK_R", "ALLMASK_Z"),
        (
            lambda *x: np.median(np.stack(x), axis=0) >= 0.35,
            "FRACMASKED_G",
            "FRACMASKED_R",
            "FRACMASKED_Z",
        ),
        (
            lambda *x: np.median(np.stack(x), axis=0) >= 5,
            "FRACFLUX_G",
            "FRACFLUX_R",
            "FRACFLUX_Z",
        ),
        Query(
            "RCHISQ_W1 >= 50",
            (
                lambda *x: np.median(np.stack(x), axis=0) >= 50,
                "RCHISQ_G",
                "RCHISQ_R",
                "RCHISQ_Z",
            ),
        ),
        Query(
            "NOBS_G > 0",
            "FLUX_IVAR_G > 0",
            Query("g_mag - r_mag < -1") | "g_mag - r_mag > 4",
        ),
        Query(
            "NOBS_Z > 0",
            "FLUX_IVAR_Z > 0",
            Query("r_mag - z_mag < -1") | "r_mag - z_mag > 4",
        ),
        "r_mag >= 25",
        Query(
            "NOBS_W1 > 0",
            "FLUX_IVAR_W1 > 0",
            "FLUX_W1 < 0",
            "NOBS_W2 > 0",
            "FLUX_IVAR_W2 > 0",
            "FLUX_W2 < 0",
        ),
    ]

    catalog = build2.set_remove_flag(catalog, remove_queries, to_remove, to_recover)
    return catalog[build2.MERGED_CATALOG_COLUMNS]


def update_sga_photometry(base, sga):
    # TODO: implement this function
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

    sga = filter_nearby_object(sga, host)
    if sga is not None:
        base = update_sga_photometry(base, sga)
    del sga

    nsa = filter_nearby_object(nsa, host)
    all_spectra = [
        extract_sdss_spectra(sdss),
        extract_nsa_spectra(nsa),
        filter_nearby_object(spectra, host),
    ]
    del sdss, spectra

    all_spectra = [s for s in all_spectra if s is not None]
    if all_spectra:
        all_spectra = vstack(all_spectra, "exact")
        if halpha is not None:
            all_spectra = build2.add_halpha_to_spectra(all_spectra, halpha)
        base = build2.add_spectra(base, all_spectra, debug=debug)
    del all_spectra

    if nsa is not None:
        base = build2.remove_shreds_near_spec_obj(base, nsa, shreds_recover=shreds_recover)
    del nsa

    if "RHOST_KPC" in base.colnames:  # has host info (i.e., not for LOWZ)
        base = build2.remove_too_close_to_host(base)
        base = build.find_satellites(base, version=2)
        base = build2.identify_host(base)

    base = build2.add_surface_brightness(base)
    base = build.add_stellar_mass(base)

    return base
