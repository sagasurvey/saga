"""
base catalog building pipeline 2.0
"""
import logging
from collections import Counter
from itertools import chain, count

import astropy.constants
import astropy.units
import numexpr as ne
import numpy as np
from astropy.coordinates import SkyCoord, search_around_sky
from astropy.table import join, vstack
from easyquery import Query, QueryMaker
from fast3tree import find_friends_of_friends

from ..spectra import (SPECS_COLUMNS, SPEED_OF_LIGHT, ensure_specs_dtype,
                       extract_nsa_spectra, extract_sdss_spectra)
from ..utils import (add_skycoord, calc_normalized_dist, fill_values_by_query,
                     get_empty_str_array, get_remove_flag, get_sdss_bands,
                     group_by)
from ..utils.distance import v2z
from . import build
from . import cuts as C

# pylint: disable=logging-format-interpolation

__all__ = [
    "prepare_sdss_catalog_for_merging",
    "prepare_des_catalog_for_merging",
    "prepare_decals_catalog_for_merging",
    "merge_catalogs",
    "add_spectra",
    "remove_shreds_near_spec_obj",
    "add_surface_brightness",
    "build_full_stack",
    "NSA_COLS_USED",
]

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
            "b_to_a",
        ),
        (b + "_mag" for b in "ugrizy"),
        (b + "_err" for b in "ugrizy"),
    )
)

_NSA_COLS_USED = [
    "RA",
    "DEC",
    "PETRO_TH50",
    "PETRO_TH90",
    "PETRO_BA90",
    "PETRO_PHI90",
    "Z",
    "ZSRC",
    "NSAID",
    "SERSIC_FLUX",
    "SERSIC_FLUX_IVAR",
    "EXTINCTION",
]
NSA_COLS_USED = list(_NSA_COLS_USED)

EXTENDED_SPECS_COLUMNS = dict(
    SPECS_COLUMNS,
    EW_Halpha="<f4",
    EW_Halpha_err="<f4",
    OBJ_NSAID="<i4",
    SPEC_REPEAT="<U48",
    SPEC_REPEAT_ALL="<U48",
)


def filter_nearby_object(catalog, host, radius_deg=1.001, remove_coord=True):
    catalog = add_skycoord(catalog)
    catalog = catalog[catalog["coord"].separation(host["coord"]).deg < radius_deg]
    if remove_coord:
        del catalog["coord"]
    return catalog


def arcsec2dist(sep, r=1.0):
    return np.sin(np.deg2rad(sep / 3600.0 / 2.0)) * 2.0 * r


def get_cartesian_coord(catalog):
    if "coord" in catalog.colnames:
        sc = catalog["coord"]
    else:
        sc = SkyCoord(catalog["RA"], catalog["DEC"], unit="deg")
    return sc.cartesian.xyz.value.T


def set_remove_flag(catalog, remove_queries=None, manual_remove=None, manual_recover=None):

    remove_queries = [] if remove_queries is None else list(remove_queries)

    if manual_remove is None:
        manual_remove = np.array([], dtype=np.int)

    remove_queries.insert(0, QueryMaker.in1d("OBJID", manual_remove))

    catalog["REMOVE"] = get_remove_flag(catalog, remove_queries)

    if manual_recover is not None:
        fill_values_by_query(
            catalog,
            QueryMaker.in1d("OBJID", manual_recover),
            {"REMOVE": 0},
        )

    return catalog


def prepare_sdss_catalog_for_merging(catalog, to_remove=None, to_recover=None):
    for b in get_sdss_bands():
        catalog["{}_mag".format(b)] = (
            np.where(
                catalog["PHOTPTYPE"] == 6,
                catalog["PSFMAG_{}".format(b.upper())],
                catalog[b],
            )
            - catalog["EXTINCTION_{}".format(b.upper())]
        )
    catalog["y_mag"] = 99.0
    catalog["y_err"] = 99.0
    catalog["b_to_a"] = -1.0

    catalog["is_galaxy"] = catalog["PHOTPTYPE"] == 3
    catalog["morphology_info"] = catalog["PHOTPTYPE"].astype(np.int32)

    catalog["radius"] = catalog["PETROR50_R"]
    catalog["radius_err"] = catalog["PETRORADERR_R"] * catalog["PETROR50_R"] / catalog["PETRORAD_R"]
    fill_values_by_query(
        catalog,
        Query("radius_err < 0") | (~Query((np.isfinite, "radius_err"))),
        {"radius_err": 9999.0},
    )

    remove_queries = [
        "BINNED1 == 0",
        "SATURATED != 0",
        "BAD_COUNTS_ERROR != 0",
        (
            lambda *x: np.median(np.abs(np.vstack(x)), axis=0) > 0.5,
            "g_err",
            "r_err",
            "i_err",
        ),
        "abs(r_mag - i_mag) > 10",
        "abs(g_mag - r_mag) > 10",
        "FIBERMAG_R > 23",
        "abs(u_mag - r_mag) > 10",
    ]

    catalog = set_remove_flag(catalog, remove_queries, to_remove, to_recover)
    return catalog[MERGED_CATALOG_COLUMNS]


def prepare_des_catalog_for_merging(catalog, to_remove=None, to_recover=None, convert_to_sdss_filters=True):

    catalog["radius"] = catalog["radius_r"]
    catalog["radius_err"] = np.float32(0)

    if "wavg_extended_coadd_i" in catalog.colnames:
        # only for backward compatibility
        catalog["is_galaxy"] = catalog["wavg_extended_coadd_i"] >= 3
        catalog["morphology_info"] = catalog["wavg_extended_coadd_i"].astype(np.int32)

    else:
        extended_coadd = (catalog["spread_model_r"] + catalog["spreaderr_model_r"] * 3.0 > 0.005).astype(np.int32)
        extended_coadd += (catalog["spread_model_r"] + catalog["spreaderr_model_r"] > 0.003).astype(np.int32)
        extended_coadd += (catalog["spread_model_r"] - catalog["spreaderr_model_r"] > 0.003).astype(np.int32)
        catalog["is_galaxy"] = extended_coadd == 3
        catalog["morphology_info"] = extended_coadd

    try:
        catalog.rename_column("ra", "RA")
        catalog.rename_column("dec", "DEC")
        catalog.rename_column("objid", "OBJID")
    except KeyError:
        if not all((col in catalog.colnames for col in ("RA", "DEC", "OBJID"))):
            raise RuntimeError("Cannot rename `RA`, `DEC`, and/or `OBJID` in DES catalog")

    if convert_to_sdss_filters:
        gi = catalog["g_mag"] - catalog["i_mag"]
        gi = np.where(np.abs(gi) < 5, gi, (catalog["g_mag"] - catalog["r_mag"]) * 1.4)
        gi = np.where(np.abs(gi) < 5, gi, (catalog["r_mag"] - catalog["i_mag"]) * 3.5)
        gi = np.where(np.abs(gi) < 20, gi, 1.5)
        catalog["g_mag"] += -0.0009 + 0.055 * gi
        catalog["r_mag"] += -0.0048 + 0.0703 * gi
        catalog["i_mag"] += -0.0065 - 0.0036 * gi + 0.02672 * gi * gi
        catalog["z_mag"] += -0.0438 + 0.02854 * gi

    # fixing DES bright star misclassification issue
    fill_values_by_query(
        catalog,
        Query(
            "0.7 * (r_mag + 10.2) > r_mag + 2.5 * log10(8 * arctan(1) * radius**2)",
            "g_mag - r_mag < 0.6",
            "r_mag < 17",
        ),
        {"is_galaxy": False},
    )

    catalog["u_mag"] = 99.0
    catalog["u_err"] = 99.0

    try:
        catalog["b_to_a"] = catalog["b_image"] / catalog["a_image"]
    except KeyError:
        catalog["b_to_a"] = -1.0

    remove_queries = [
        "imaflags_iso_r != 0",
        "flags_r >= 4",
        "r_mag >= 25",
        (
            lambda *x: np.median(np.abs(np.vstack(x)), axis=0) > 0.5,
            "g_err",
            "r_err",
            "i_err",
        ),
    ]

    catalog = set_remove_flag(catalog, remove_queries, to_remove, to_recover)
    return catalog[MERGED_CATALOG_COLUMNS]


def prepare_decals_catalog_for_merging(catalog, to_remove=None, to_recover=None, convert_to_sdss_filters=True):
    """
    Ref:
    - DR6/7: https://www.legacysurvey.org/dr7/files/#sweep-7-0-sweep-brickmin-brickmax-fits
    - DR8:   https://www.legacysurvey.org/dr8/files/#sweep-catalogs-region-sweep
    """
    is_dr8plus = catalog["RELEASE"][0] >= 8000

    if is_dr8plus:
        catalog["OBJID"] = (
            (catalog["RELEASE"] // 100) * np.int64(1e16)
            + catalog["BRICKID"] * np.int64(1e10)
            + catalog["OBJID"].astype(np.int64)
        )
    else:
        catalog["OBJID"] = catalog["BRICKID"] * np.int64(1e13) + catalog["OBJID"].astype(np.int64)

    catalog["is_galaxy"] = catalog["TYPE"] != "PSF"
    if is_dr8plus:
        catalog["is_galaxy"] &= catalog["TYPE"] != "DUP"
    catalog["morphology_info"] = catalog["TYPE"].getfield("<U1").view(np.int32)
    catalog["radius"] = catalog["FRACDEV"] * catalog["SHAPEDEV_R"] + (1.0 - catalog["FRACDEV"]) * catalog["SHAPEEXP_R"]
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

    # Recalculate mag and err to fix old bugs in the raw catalogs
    const = 2.5 / np.log(10)
    for band in "grz":
        BAND = band.upper()
        with np.errstate(divide="ignore", invalid="ignore"):
            catalog[f"{band}_mag"] = 22.5 - const * np.log(catalog[f"FLUX_{BAND}"] / catalog[f"MW_TRANSMISSION_{BAND}"])
            catalog[f"{band}_err"] = const / np.abs(catalog[f"FLUX_{BAND}"] * np.sqrt(catalog[f"FLUX_IVAR_{BAND}"]))

    for band in "uiy":
        catalog["{}_mag".format(band)] = 99.0
        catalog["{}_err".format(band)] = 99.0

    if convert_to_sdss_filters:
        catalog["g_mag"] += 0.09
        catalog["r_mag"] += 0.1
        catalog["z_mag"] += 0.02

    remove_queries = [
        "FRACMASKED_G >= 0.35",
        "FRACMASKED_R >= 0.35",
        "FRACMASKED_Z >= 0.35",
        "FRACFLUX_G >= 4",
        "FRACFLUX_R >= 4",
        "FRACFLUX_Z >= 4",
        "RCHISQ_G >= 10",
        "RCHISQ_R >= 10",
        "RCHISQ_Z >= 10",
        Query("RCHISQ_G > 4", "RCHISQ_R > 4", "RCHISQ_Z > 4"),
        Query("FRACIN_G < 0.7", "FRACIN_R < 0.7", "FRACIN_Z < 0.7"),
        "ALLMASK_G > 0",
        "ALLMASK_R > 0",
        "ALLMASK_Z > 0",
        "NOBS_G == 0",
        "NOBS_R == 0",
        "radius >= 15",
        "g_err >= 0.2",
        "r_err >= 0.2",
        "z_err >= 0.2",
        "g_mag - r_mag < -0.5",
        "radius > 10.0**(-0.2 * (r_mag - 23.5))",
        Query("is_galaxy", "radius < 10.0**(-0.2 * (r_mag - 17))"),
        "r_mag >= 25",
    ]

    if is_dr8plus:
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

    catalog = set_remove_flag(catalog, remove_queries, to_remove, to_recover)
    return catalog[MERGED_CATALOG_COLUMNS]


def add_halpha_to_spectra(spectra, halpha):
    halpha = halpha["EW_Halpha", "EW_Halpha_err", "SPECOBJID1", "MASKNAME"]
    halpha.rename_column("SPECOBJID1", "SPECOBJID")
    halpha = ensure_specs_dtype(halpha, cols_definition=EXTENDED_SPECS_COLUMNS, skip_missing_cols=True)
    spectra = join(spectra, halpha, ["SPECOBJID", "MASKNAME"], "left")
    spectra["EW_Halpha"].fill_value = np.nan
    spectra["EW_Halpha_err"].fill_value = np.nan
    spectra.filled()
    return spectra


def assign_photometry_choice(stacked_catalog, indices, is_last):
    if len(indices) == 1:
        return 2  # only one entry, always chosen

    surveys = stacked_catalog["survey"][indices].tolist()
    survey_count = Counter(surveys)
    if min(survey_count.values()) > 1 and not is_last:
        return  # need to redo FoF search with smaller sep

    sorter = np.lexsort(
        tuple((stacked_catalog[col][indices] for col in ("r_mag", "survey_p", "REMOVE")))
    )  # last one is the primary sort key

    choices = [0 for _ in indices]
    done = set()
    not_selected = []
    for s in sorter:
        survey_this = surveys[s]
        if survey_this in done:
            not_selected.append(indices[s])
        else:
            choices[s] = 1 if done else 2
            done.add(survey_this)

    if is_last or not not_selected:
        return choices

    not_selected = np.array(not_selected)
    not_selected_good = not_selected[stacked_catalog["REMOVE"][not_selected] == 0]
    if not not_selected_good.size:
        return choices

    if not_selected_good.size < len(survey_count) and (
        stacked_catalog["r_mag"][not_selected_good].min() > stacked_catalog["r_mag"][indices[sorter[0]]] + 1
    ):
        return choices


def merge_catalogs(debug=None, **catalog_dict):

    survey_priority = ("des", "decals", "sdss")

    catalog_dict = {k: v for k, v in catalog_dict.items() if v is not None}
    n_catalogs = len(catalog_dict)

    if n_catalogs == 0:
        raise ValueError("No catalogs to merge!!")

    elif n_catalogs == 1:
        survey, stacked_catalog = next(iter(catalog_dict.items()))
        stacked_catalog["survey"] = get_empty_str_array(len(stacked_catalog), max(6, len(survey)), survey)
        stacked_catalog["group_id"] = find_friends_of_friends(
            get_cartesian_coord(stacked_catalog),
            arcsec2dist(0.5),
            reassign_group_indices=True,
        )
        stacked_catalog.sort(["group_id", "REMOVE", "r_mag"])
        idx = np.flatnonzero(np.hstack([[1], np.ediff1d(stacked_catalog["group_id"])]))
        stacked_catalog["chosen"] = 0
        stacked_catalog["chosen"][idx] = 2
        del idx

    else:
        stacked_catalog = vstack(list(catalog_dict.values()), "exact")
        stacked_catalog["survey"] = get_empty_str_array(len(stacked_catalog), max(6, max(len(s) for s in catalog_dict)))
        stacked_catalog["survey_p"] = 999
        counter = 0
        for name, cat in catalog_dict.items():
            s = slice(counter, counter + len(cat))
            stacked_catalog["survey"][s] = name
            try:
                p = survey_priority.index(name)
            except ValueError:
                pass
            else:
                stacked_catalog["survey_p"][s] = p
            counter += len(cat)

        coord = get_cartesian_coord(stacked_catalog)
        orig_idx = np.arange(len(stacked_catalog))
        stacked_catalog["group_id"] = 0
        stacked_catalog["chosen"] = 0
        group_id_shift = 1

        for sep in np.arange(3, 0.01, -0.5):
            is_last = sep < 0.9
            group_ids_tmp = find_friends_of_friends(coord, arcsec2dist(sep))

            for group_id, indices in enumerate(group_by(group_ids_tmp)):
                orig_idx_this = orig_idx[indices]
                chosen = assign_photometry_choice(stacked_catalog, orig_idx_this, is_last)
                if chosen is None:
                    continue
                stacked_catalog["group_id"][orig_idx_this] = group_id + group_id_shift
                stacked_catalog["chosen"][orig_idx_this] = chosen
                orig_idx[indices] = -1
                del orig_idx_this

            regroup_mask = orig_idx > -1
            if not regroup_mask.any():
                del regroup_mask
                break
            group_id_shift = stacked_catalog["group_id"].max() + 1
            coord = coord[regroup_mask]
            orig_idx = orig_idx[regroup_mask]
            del regroup_mask

        del coord, orig_idx, stacked_catalog["survey_p"]

    if debug is not None:
        debug["stacked_catalog"] = stacked_catalog.copy()

    merged_catalog = Query("chosen == 2").filter(stacked_catalog)
    for name in catalog_dict:
        # pylint: disable=undefined-loop-variable
        merged_catalog = join(
            merged_catalog,
            Query("chosen > 0", (lambda x: x == name, "survey")).filter(stacked_catalog)[
                MERGED_CATALOG_COLUMNS + ["group_id"]
            ],
            keys="group_id",
            join_type="left",
            uniq_col_name="{col_name}{table_name}",
            table_names=["", "_" + name],
        )

    del stacked_catalog, merged_catalog["group_id"], merged_catalog["chosen"]

    for name in catalog_dict:
        merged_catalog["OBJID_{}".format(name)].fill_value = -1
        merged_catalog["REMOVE_{}".format(name)].fill_value = -1
        merged_catalog["is_galaxy_{}".format(name)].fill_value = False

    return merged_catalog.filled()


def replace_poor_sdss_sky_subtraction(base):

    mask = Query(
        "abs(r_mag_sdss - r_mag_decals) > 2",
        (lambda s: s == "sdss", "survey"),
        "OBJID_decals != -1",
        "REMOVE_decals == 0",
    ).mask(base)

    base["survey"][mask] = "decals"
    for col in base.colnames:
        if col.endswith("_decals"):
            base[col.rpartition("_")[0]][mask] = base[col][mask]

    return base


def add_columns_for_spectra(base):
    cols_definition = EXTENDED_SPECS_COLUMNS.copy()
    for col in ("RA", "DEC"):
        cols_definition[col + "_spec"] = cols_definition[col]
        del cols_definition[col]
    base = ensure_specs_dtype(base, cols_definition)
    return base


SPEC_MATCHING_ORDER = (
    (Query("REMOVE == 0", "is_galaxy == 0", "sep < 1"), "sep"),
    (Query("REMOVE % 2 == 0", "is_galaxy == 0", "sep < 1"), "sep"),
    (Query("REMOVE == 0", "is_galaxy", C.faint_end_limit, "sep_norm < 0.5"), "r_mag"),
    (Query("REMOVE == 0", "is_galaxy", C.faint_end_limit, "sep_norm < 1"), "r_mag"),
    (Query("REMOVE == 0", "is_galaxy", "sep_norm < 0.5"), "r_mag"),
    (Query("REMOVE == 0", "is_galaxy", "sep_norm < 1"), "r_mag"),
    (Query("REMOVE == 0", "sep < 3"), "sep"),
    (Query("REMOVE  > 0", "is_galaxy", C.faint_end_limit, "sep_norm < 0.5"), "r_mag"),
    (Query("REMOVE  > 0", "is_galaxy", C.faint_end_limit, "sep_norm < 1"), "r_mag"),
    (Query("REMOVE  > 0", "is_galaxy", "sep_norm < 0.5"), "r_mag"),
    (Query("REMOVE  > 0", "is_galaxy", "sep_norm < 1"), "r_mag"),
    (Query("REMOVE  > 0", "sep < 3"), "sep"),
    (Query("REMOVE == 0", "sep < 10"), "sep"),
    (Query("REMOVE  > 0", "sep < 10"), "sep"),
)


def match_spectra_to_base_and_merge_duplicates(specs, base, debug=None, matching_order=None):
    """
    This function first match unmerged spectra to base catalog,
    and then merge the spectra that are assigned to the same photo obj.
    """

    if "coord" in specs.colnames:
        del specs["coord"]  # because "coord" breaks "sort"
    specs.sort(["ZQUALITY_sort_key", "SPEC_Z"])

    specs = add_skycoord(specs)
    base = add_skycoord(base)
    specs_idx, base_idx, sep, _ = search_around_sky(
        specs["coord"],
        base["coord"],
        20.0 * astropy.units.arcsec,  # pylint: disable=E1101
    )
    sep = sep.arcsec

    # in case future astropy does not preserve the order of `specs_idx`
    if (np.ediff1d(specs_idx) < 0).any():
        sorter = specs_idx.argsort()
        specs_idx = specs_idx[sorter]
        base_idx = base_idx[sorter]
        sep = sep[sorter]
        del sorter

    # matched_idx will store the index of the matched photo obj.
    specs["matched_idx"] = -1

    matching_order = matching_order or SPEC_MATCHING_ORDER

    if len(specs_idx):
        for group_slice in group_by(specs_idx, True):
            spec_idx_this = specs_idx[group_slice.start]
            possible_match = base[base_idx[group_slice]]
            possible_match["sep"] = sep[group_slice]
            possible_match["sep_norm"] = calc_normalized_dist(
                specs["RA"][spec_idx_this],
                specs["DEC"][spec_idx_this],
                possible_match["RA"],
                possible_match["DEC"],
                possible_match["radius_for_match"],
                possible_match["ba"] if "ba" in possible_match.colnames else None,
                possible_match["phi"] if "phi" in possible_match.colnames else None,
                multiplier=1,
            )
            possible_match["SPEC_Z"] = specs["SPEC_Z"][spec_idx_this]

            # using following criteria one by one to find matching photo obj, stop when found
            for q, sorter in matching_order:
                mask = q.mask(possible_match)
                if mask.any():
                    possible_match_this = possible_match[mask]
                    matched_base_idx = possible_match_this["index"][possible_match_this[sorter].argmin()]
                    specs["matched_idx"][spec_idx_this] = matched_base_idx
                    break

    # now each photo obj can potentially have more than one spec matched to it
    # so for each photo obj that has one or more specs, we will merge the specs

    if "coord" in specs.colnames:
        del specs["coord"]
    specs.sort(["matched_idx", "ZQUALITY_sort_key", "SPEC_Z"])

    specs["index"] = np.arange(len(specs))
    specs["SPEC_REPEAT"] = get_empty_str_array(len(specs), 48)
    specs["SPEC_REPEAT_ALL"] = get_empty_str_array(len(specs), 48)
    specs["OBJ_NSAID"] = np.int32(-1)
    specs["chosen"] = False

    def get_tel_rank(
        tel,
        ranks=("MMT", "AAT", "PAL", "BINO", "NSA", "_OTHERS", "SDSS", "ALFALF", "WIYN"),
    ):
        try:
            return ranks.index(tel)
        except ValueError:
            return ranks.index("_OTHERS")

    for group_slice in group_by(specs["matched_idx"], True):
        # matched_idx < 0 means there is no match, so nothing to do
        if specs["matched_idx"][group_slice.start] < 0:
            continue

        # stop - start == 1 means there is only one match, so it's easy
        if group_slice.stop - group_slice.start == 1:
            i = group_slice.start
            specs["chosen"][i] = True
            specs["SPEC_REPEAT"][i] = specs["TELNAME"][i]
            specs["SPEC_REPEAT_ALL"][i] = specs["TELNAME"][i]
            if specs["TELNAME"][i] == "NSA":
                specs["OBJ_NSAID"][i] = int(specs["SPECOBJID"][i])
            continue

        # now it's the real thing, we have more than one specs
        # we design a rank for each spec, using ZQUALITY, TELNAME, and SPEC_Z_ERR
        specs_to_merge = specs[group_slice]
        rank = np.fromiter(map(get_tel_rank, specs_to_merge["TELNAME"]), np.int, len(specs_to_merge))
        rank += (10 - specs_to_merge["ZQUALITY"]) * (rank.max() + 1)
        rank = rank.astype(np.float) + np.where(
            Query((np.isfinite, "SPEC_Z_ERR"), "SPEC_Z_ERR > 0", "SPEC_Z_ERR < 1").mask(specs_to_merge),
            specs_to_merge["SPEC_Z_ERR"],
            0.99999,
        )
        specs_to_merge = specs_to_merge[rank.argsort()]
        best_spec = specs_to_merge[0]

        # We now check if there is any spec that is not at the same redshift as the best spec (mask_within_dz).
        # If so, and those specs are good or as good as the best spec (mask_same_zq_class),
        #        and those specs are at least 0.5 arcsec away (mask_coord_offset),
        # then, we push them out of this merge process (to_rematch).
        mask_within_dz = np.fabs(specs_to_merge["SPEC_Z"] - best_spec["SPEC_Z"]) < 150.0 / SPEED_OF_LIGHT

        mask_same_zq_class = (specs_to_merge["ZQUALITY_sort_key"] == best_spec["ZQUALITY_sort_key"]) | (
            specs_to_merge["ZQUALITY"] >= 3
        )

        mask_coord_offset = (
            SkyCoord(specs_to_merge["RA"], specs_to_merge["DEC"], unit="deg")
            .separation(SkyCoord(best_spec["RA"], best_spec["DEC"], unit="deg"))
            .arcsec
            > 0.5
        )

        if ((~mask_within_dz) & mask_coord_offset & mask_same_zq_class).any():
            to_rematch = (~mask_within_dz) & mask_coord_offset
            specs["matched_idx"][specs_to_merge["index"][to_rematch]] = -2  # we will deal with these -2 later
            specs_to_merge = specs_to_merge[~to_rematch]
            mask_same_zq_class = mask_same_zq_class[~to_rematch]
            mask_within_dz = mask_within_dz[~to_rematch]

        # so now specs_to_merge has specs that are ok to merge
        # we need to find if there's NSA objects and also get SPEC_REPEAT and put those info on best spec
        best_spec_index = best_spec["index"]
        specs["chosen"][best_spec_index] = True
        specs["SPEC_REPEAT"][best_spec_index] = "+".join(
            set(specs_to_merge["TELNAME"][mask_within_dz & mask_same_zq_class])
        )
        specs["SPEC_REPEAT_ALL"][best_spec_index] = "+".join(set(specs_to_merge["TELNAME"]))

        nsa_specs = specs_to_merge[specs_to_merge["TELNAME"] == "NSA"]
        specs["OBJ_NSAID"][best_spec_index] = int(nsa_specs["SPECOBJID"][0]) if len(nsa_specs) else -1
        if len(nsa_specs) > 1:
            logging.warning(
                "More than one NSA obj near ({}, {}): {}".format(
                    nsa_specs["RA"][0],
                    nsa_specs["DEC"][0],
                    ", ".join(nsa_specs["SPECOBJID"]),
                )
            )

    # print out warnings for unmatched good specs
    for spec in Query("matched_idx == -1", "ZQUALITY >= 3").filter(specs):
        if spec["TELNAME"] in (
            "AAT",
            "MMT",
            "BINO",
            "IMACS",
            "WIYN",
            "NSA",
            "PAL",
            "SALT",
        ):
            logging.warning(
                "No photo obj matched to {0[TELNAME]} spec {0[MASKNAME]} {0[SPECOBJID]} ({0[RA]}, {0[DEC]})".format(
                    spec
                )
            )

    if debug is not None:
        for i in count():
            key = "specs_matching_{}".format(i)
            if key not in debug:
                debug[key] = specs.copy()
                break

    # return both matched specs and specs that need to be rematched (those -2's)
    return Query("chosen").filter(specs), Query("matched_idx == -2").filter(specs)


def add_spectra(base, specs, debug=None, matching_order=None):

    # map ZQ >=4, 3, 2, <=1 to 0, 1, 2, 3
    specs["ZQUALITY_sort_key"] = 4 - specs["ZQUALITY"]
    fill_values_by_query(specs, "ZQUALITY_sort_key < 0", {"ZQUALITY_sort_key": 0})
    fill_values_by_query(specs, "ZQUALITY_sort_key > 3", {"ZQUALITY_sort_key": 3})

    has_sma = "sma" in base.colnames
    has_ba = "ba" in base.colnames
    has_phi = "phi" in base.colnames
    has_ref_cat = "REF_CAT" in base.colnames

    add_skycoord(base)
    cols = ["RA", "DEC", "REMOVE", "is_galaxy", "r_mag", "coord"]
    if has_ba and has_phi:
        cols.extend(["ba", "phi"])
    if has_ref_cat:
        cols.append("REF_CAT")
    base_this = base[cols]
    base_this["index"] = np.arange(len(base))

    if has_sma:
        base_this["radius_for_match"] = base["sma"] * 0.75
    else:
        base_this["radius_for_match"] = base["radius"] * 2.0  # half-light correction
        if has_ba:
            base_this["radius_for_match"] /= np.sqrt(base["ba"])
    fill_values_by_query(base_this, ~Query("radius_for_match > 0"), {"radius_for_match": 0.1})

    needs_rematch_count = 0
    for _ in range(5):
        specs_matched, specs_need_rematch = match_spectra_to_base_and_merge_duplicates(
            specs, base_this, debug=debug, matching_order=matching_order
        )

        # for matched specs, copy their info to base catalog
        for col in EXTENDED_SPECS_COLUMNS:
            col_base = (col + "_spec") if col in ("RA", "DEC") else col
            base[col_base][specs_matched["matched_idx"]] = specs_matched[col]

        # check if there are specs that need to be rematched, prepare specs and base_this for the next iteration
        if len(specs_need_rematch) in (0, needs_rematch_count):
            break
        needs_rematch_count = len(specs_need_rematch)
        specs = specs_need_rematch
        base_this = base_this[np.in1d(base_this["index"], specs_matched["matched_idx"], True, True)]
    else:
        for spec in Query("ZQUALITY >= 3").filter(specs):
            if spec["TELNAME"] in ("AAT", "MMT", "IMACS", "WIYN", "SDSS", "NSA"):
                logging.warning(
                    "Still no photo obj matched to {0[TELNAME]} spec {0[MASKNAME]} {0[SPECOBJID]} ({0[RA]}, {0[DEC]})".format(
                        spec
                    )
                )

    return base


def _join_spec_repeat(current, new):
    return "+".join(set(sum((s.split("+") for s in new), current.split("+"))))


def remove_shreds_near_spec_obj(base, nsa=None, shreds_recover=None):

    has_sma = "sma" in base.colnames
    has_ba = "ba" in base.colnames
    has_phi = "phi" in base.colnames
    has_ref_cat = "REF_CAT" in base.colnames
    has_obj_nsaid = "OBJ_NSAID" in base.colnames

    is_nsa = Query("OBJ_NSAID > -1")
    is_sga = QueryMaker.equal("REF_CAT", "L3")
    is_good_galaxy = Query(C.is_clean2, C.is_galaxy2, "radius > 0", C.has_spec)

    base["spec_rank"] = C.has_spec.mask(base).astype(np.int16) * 2
    if has_ref_cat:
        base["spec_rank"] += is_sga.mask(base).astype(np.int16)
        is_good_galaxy |= is_sga
    elif has_obj_nsaid and nsa is not None:
        base["spec_rank"] += is_nsa.mask(base).astype(np.int16)
        is_good_galaxy |= is_nsa
    base["spec_rank"] *= -1

    cols = ["spec_rank", "r_mag"]
    has_spec_z_indices = np.flatnonzero(is_good_galaxy.mask(base))
    has_spec_z_indices = has_spec_z_indices[base[cols][has_spec_z_indices].argsort(cols)]
    del base["spec_rank"]

    for obj_this_idx in has_spec_z_indices:
        obj_this = base[obj_this_idx]

        if nsa is not None and has_obj_nsaid and obj_this["OBJ_NSAID"] > -1:
            nsa_obj = Query("NSAID == {}".format(obj_this["OBJ_NSAID"])).filter(nsa)[0]
            dist = calc_normalized_dist(
                base["RA"],
                base["DEC"],
                nsa_obj["RA"],
                nsa_obj["DEC"],
                nsa_obj["PETRO_TH90"],
                nsa_obj["PETRO_BA90"],
                nsa_obj["PETRO_PHI90"],
                multiplier=2,
            )

            remove_flag = 28

            values_to_rewrite = {
                "OBJID": nsa_obj["NSAID"],
                "REMOVE": 0,
                "is_galaxy": (nsa_obj["PETRO_TH50"] > 1),
                "RA": nsa_obj["RA"],
                "DEC": nsa_obj["DEC"],
                "radius": nsa_obj["PETRO_TH50"],
                "radius_err": 0,
                "survey": "NSA",
            }

            invalid_mag = nsa_obj["SERSIC_FLUX"] <= 0
            nsa_sersic_flux = np.array(nsa_obj["SERSIC_FLUX"])
            nsa_sersic_flux[invalid_mag] = 1.0

            mag = 22.5 - 2.5 * np.log10(nsa_sersic_flux)
            mag_err = np.fabs((2.5 / np.log(10.0)) / nsa_sersic_flux / np.sqrt(nsa_obj["SERSIC_FLUX_IVAR"]))

            for i, b in enumerate(get_sdss_bands()):
                j = i + 2  # first two bands not needed
                if invalid_mag[j]:
                    continue
                values_to_rewrite["{}_mag".format(b)] = mag[j] - nsa_obj["EXTINCTION"][j]
                values_to_rewrite["{}_err".format(b)] = mag_err[j]

            for k, v in values_to_rewrite.items():
                base[k][obj_this_idx] = v

        elif obj_this["REMOVE"] == 0:  # any other good, non-NSA objects

            # If semi-major axis of the galaxy moment (sma) is available
            # use it as the removal radius
            # If not, use 2.0 * effective radius as the removal radius
            if has_sma:
                remove_radius = obj_this["sma"]
            else:
                remove_radius = obj_this["radius"] * 2.0
                if has_ba:
                    remove_radius /= np.sqrt(obj_this["ba"])

            dist = calc_normalized_dist(
                base["RA"],
                base["DEC"],
                obj_this["RA"],
                obj_this["DEC"],
                remove_radius,
                obj_this["ba"] if has_ba else None,
                obj_this["phi"] if has_phi else None,
                multiplier=1,
            )

            remove_flag = 29

        else:
            continue  # skip anything else

        nearby_obj_mask = dist < 1
        nearby_obj_mask[obj_this_idx] = False
        if not nearby_obj_mask.any():
            continue

        nearby_obj = base[nearby_obj_mask]
        nearby_obj["dist"] = dist[nearby_obj_mask]
        nearby_obj["_idx"] = np.flatnonzero(nearby_obj_mask)
        del nearby_obj_mask, dist

        is_brighter = Query("r_mag < {}".format(obj_this["r_mag"]))
        is_fainter = ~is_brighter
        remove_basic_conditions = Query("is_galaxy", is_fainter, "r_mag > 14")

        if has_ref_cat:
            remove_basic_conditions |= is_nsa
            remove_basic_conditions &= QueryMaker.equal("REF_CAT", "")
        elif has_obj_nsaid:
            remove_basic_conditions &= ~is_nsa

        if shreds_recover is not None:
            remove_basic_conditions &= QueryMaker.isin("OBJID", shreds_recover, invert=True)

        no_spec_z = Query("ZQUALITY < 2")
        if has_ref_cat and obj_this["REF_CAT"] == "L3":  # removed by SGA objects
            no_spec_z &= (Query("dist < 0.75") | Query("r_err >= 0.005"))
        elif nsa is None:  # removed by all other objects
            no_spec_z &= (Query("dist < 0.6") | Query("r_err >= 0.02"))

        if obj_this["ZQUALITY"] >= 3:
            z_limit = v2z([300, 250, 200, 150][np.searchsorted([14, 16, 20], obj_this["r_mag"])])
            close_spec_z = Query("ZQUALITY > -1", (lambda z: np.abs(z - obj_this["SPEC_Z"]) < z_limit, "SPEC_Z"))
            good_close_spec_z = Query(close_spec_z, "ZQUALITY >= 3")
            to_remove = Query(remove_basic_conditions, close_spec_z | no_spec_z)
        else:
            to_remove = Query(remove_basic_conditions, no_spec_z)

        if not to_remove.mask(nearby_obj).any():  # Nothing to remove, carry on
            continue

        new_remove_count = Query(to_remove, "REMOVE == 0").count(nearby_obj)
        if new_remove_count > 25 and "NSA" not in obj_this["SPEC_REPEAT"] and "SGA" not in obj_this["SPEC_REPEAT"]:
            logging.warning(
                '{} photo obj newly removed within ~{:.3f}" of {} spec obj {} ({}, {})'.format(
                    new_remove_count,
                    remove_radius,
                    obj_this["TELNAME"],
                    obj_this["OBJID"],
                    obj_this["RA"],
                    obj_this["DEC"],
                )
            )

        base["REMOVE"][to_remove.filter(nearby_obj, "_idx")] |= 1 << remove_flag

        if obj_this["ZQUALITY"] >= 3:
            close_spec_z = close_spec_z & remove_basic_conditions
            good_close_spec_z = good_close_spec_z & remove_basic_conditions

            if not close_spec_z.mask(nearby_obj).any():
                continue

            base["SPEC_REPEAT"][obj_this_idx] = _join_spec_repeat(
                obj_this["SPEC_REPEAT"], good_close_spec_z.filter(nearby_obj, "SPEC_REPEAT")
            )
            base["SPEC_REPEAT_ALL"][obj_this_idx] = _join_spec_repeat(
                obj_this["SPEC_REPEAT_ALL"],
                close_spec_z.filter(nearby_obj, "SPEC_REPEAT_ALL"),
            )

        del nearby_obj

    return base


def remove_too_close_to_host(base):
    min_rhost = base["RHOST_KPC"].min()
    remove_dist = 10.0

    # hot fix for pgc70094
    if "HOST_PGC" in base.colnames and base["HOST_PGC"][0] == 70094:
        remove_dist = 24.0

    q = Query((lambda r: ((r < remove_dist) & (r > min_rhost)), "RHOST_KPC"))
    base["REMOVE"][q.mask(base)] += 1 << 30
    return base


def add_surface_brightness(base):
    # the factor 2 inside log10 is to account for that the magnitude we are using is total magnitude
    # but radius is half-light radius
    with np.errstate(divide="ignore"):
        sb_r = base["r_mag"] + 2.5 * np.log10(2 * np.pi * base["radius"] * base["radius"])
        sb_r_err = np.hypot(base["r_err"], (5 / np.log(10)) * base["radius_err"] / base["radius"])

    has_radius_mask = base["radius"] > 0
    base["sb_r"] = np.where(has_radius_mask, sb_r, 99.0)
    base["sb_r_err"] = np.where(has_radius_mask, sb_r_err, 99.0)
    return base


def identify_host(base):
    host_search_queries = [
        C.obj_is_host2,
        Query("RHOST_ARCM < 0.5", "r_mag < 14", C.has_spec, C.sat_vcut),
        Query("RHOST_ARCM < 0.5", "r_mag < 14"),
        Query("RHOST_ARCM < 0.3", "r_mag < 16", C.has_spec, C.sat_vcut),
        Query("RHOST_ARCM < 0.3", "r_mag < 16"),
    ]

    # for base v3
    if "REF_CAT" in base.colnames:
        host_search_queries.insert(0, Query(host_search_queries[1], QueryMaker.equal("REF_CAT", "L3")))

    for q in host_search_queries:
        candidate_idx = np.flatnonzero(q.mask(base))
        if len(candidate_idx):
            host_idx = candidate_idx[base["RHOST_ARCM"][candidate_idx].argmin()]
            break
    else:
        host_idx = base["RHOST_ARCM"].argmin()

    base["SATS"][host_idx] = 3
    base["REMOVE"][host_idx] = 0
    base["is_galaxy"][host_idx] = True
    base["RA_spec"][host_idx] = base["HOST_RA"][host_idx]
    base["DEC_spec"][host_idx] = base["HOST_DEC"][host_idx]
    base["SPEC_Z"][host_idx] = v2z(base["HOST_VHOST"][host_idx])
    base["SPEC_Z_ERR"][host_idx] = v2z(60)
    base["SPECOBJID"][host_idx] = str(base["HOST_PGC"][host_idx])
    base["MASKNAME"][host_idx] = "HOST"
    base["TELNAME"][host_idx] = "HOST"
    base["HELIO_CORR"][host_idx] = True

    if base["ZQUALITY"][host_idx] < 3:
        base["SPEC_REPEAT"][host_idx] = "HOST"
    else:
        current = str(base["SPEC_REPEAT"][host_idx])
        base["SPEC_REPEAT"][host_idx] = (current + "+HOST") if current else "HOST"
    current = str(base["SPEC_REPEAT_ALL"][host_idx])
    base["SPEC_REPEAT_ALL"][host_idx] = (current + "+HOST") if current else "HOST"
    base["ZQUALITY"][host_idx] = 4

    return base


def build_full_stack(  # pylint: disable=unused-argument
    host,
    sdss=None,
    des=None,
    decals=None,
    nsa=None,
    sdss_remove=None,
    sdss_recover=None,
    des_remove=None,
    des_recover=None,
    decals_remove=None,
    decals_recover=None,
    spectra=None,
    halpha=None,
    shreds_recover=None,
    convert_to_sdss_filters=True,
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
    if sdss is None and des is None and decals is None:
        raise ValueError("No photometry catalog to build!")

    all_spectra = []

    if sdss is not None:
        sdss_specs = extract_sdss_spectra(sdss)
        if sdss_specs is not None:
            all_spectra.append(sdss_specs)
        del sdss_specs
        sdss = prepare_sdss_catalog_for_merging(sdss, sdss_remove, sdss_recover)

    if des is not None:
        des = prepare_des_catalog_for_merging(des, des_remove, des_recover, convert_to_sdss_filters)

    if decals is not None:
        if "decals_dr8_remove" in kwargs:
            decals_remove = np.concatenate([decals_remove, kwargs["decals_dr8_remove"]])
        decals = prepare_decals_catalog_for_merging(decals, decals_remove, decals_recover, convert_to_sdss_filters)

    if nsa is not None:
        nsa = filter_nearby_object(nsa, host)
        if len(nsa):
            all_spectra.append(extract_nsa_spectra(nsa))
        else:
            nsa = None

    if spectra is not None:
        spectra = filter_nearby_object(spectra, host)
        if len(spectra):
            all_spectra.append(spectra)

    base = merge_catalogs(sdss=sdss, des=des, decals=decals, debug=debug)
    if sdss is not None and decals is not None:
        base = replace_poor_sdss_sky_subtraction(base)

    base = build.add_host_info(base, host)
    del sdss, des, decals, spectra

    base = add_columns_for_spectra(base)
    if all_spectra:
        all_spectra = vstack(all_spectra, "exact")

    if len(all_spectra):
        if halpha is not None:
            all_spectra = add_halpha_to_spectra(all_spectra, halpha)
        base = add_spectra(base, all_spectra, debug=debug)
        del all_spectra
        base = remove_shreds_near_spec_obj(base, nsa, shreds_recover=shreds_recover)
        del nsa

    if "RHOST_KPC" in base.colnames:  # has host info
        base = remove_too_close_to_host(base)
        base = build.find_satellites(base, version=2)
        base = identify_host(base)

    base = add_surface_brightness(base)
    base = build.add_stellar_mass(base)

    return base
