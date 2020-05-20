"""
module assign_targeting_score
"""
from itertools import chain

import numpy as np
from easyquery import Query, QueryMaker

from .. import utils
from ..objects import cuts as C
from ..utils import (fill_values_by_query, get_all_colors, get_sdss_bands,
                     get_sdss_colors)
from .gmm import (calc_gmm_satellite_probability, calc_log_likelihood,
                  get_input_data, param_labels_nosat)

__all__ = [
    "assign_targeting_score_v1",
    "assign_targeting_score_v2",
    "assign_targeting_score_lowz",
    "calc_simple_satellite_probability",
    "calc_gmm_satellite_probability",
    "calc_simple_satellite_probability_gri",
    "calc_simple_satellite_probability_grz",
]

COLUMNS_USED = list(
    set(
        chain(
            C.COLUMNS_USED,
            ["TELNAME"],
            map("{}_mag".format, get_sdss_bands()),
            map("{}_err".format, get_sdss_bands()),
            get_sdss_colors(),
            map("{}_err".format, get_sdss_colors()),
        )
    )
)


def _calc_simple_satellite_probability(base, model_parameters, colors):
    x = (
        np.asarray(base[colors[0]]) * model_parameters[0]
        + np.asarray(base[colors[1]]) * model_parameters[1]
    )
    x = np.where(np.isfinite(x), x, 9999)
    return np.where(
        x > model_parameters[2],
        np.minimum(
            np.exp((x - model_parameters[3]) * model_parameters[4]), model_parameters[5]
        ),
        0.0,
    )


def calc_simple_satellite_probability_gri(base):
    return _calc_simple_satellite_probability(
        base,
        (
            -0.84526783,
            -0.53434289,
            -1.0123662441968917,
            0.18628167890581865,
            9.4021013202593942,
            0.055890285233031099,
        ),
        ("gr", "ri"),
    )


def calc_simple_satellite_probability_grz(base):
    return _calc_simple_satellite_probability(
        base,
        (
            -0.6394620175672536,
            -0.7688226896292912,
            -1.3123654045614568,
            0.30417493992455574,
            5.029875624196831,
            0.04425228193300764,
        ),
        ("gr", "rz"),
    )


calc_simple_satellite_probability = calc_simple_satellite_probability_gri


def ensure_proper_prob(p):
    p[np.isnan(p)] = 0
    p[p > 1] = 1.0
    p[p < 0] = 0.0
    return p


# pylint: disable=unused-argument


def assign_targeting_score_v1(
    base, manual_selected_objids=None, gmm_parameters=None, **kwargs
):
    """
    Last updated: 05/07/2018
     100 Human selection and Special targets
     150 satellites
     200 within host,  r < 17.77, gri cuts
     300 within host,  r < 20.75, high p_GMM or GMM outliers, gri cuts
     400 within host,  r < 20.75, high-proirity + gri cuts
     500 outwith host, r < 17.77, gri cuts
     600 within host,  r < 20.75, gri cuts, random selection of 50
     700 very high p_GMM, gri cuts
     800 within host,  r < 20.75, gri cuts, everything else
     900 outwith host, r < 20.75, gri cuts
    1000 everything else
    1100 Not in gri/fibermag_r_cut
    1200 Not galaxy
    1300 Not clean
    1400 Has spec but not a satellite
    """

    base["P_simple"] = ensure_proper_prob(calc_simple_satellite_probability(base))
    base["P_GMM_sdss"] = ensure_proper_prob(
        calc_gmm_satellite_probability(base, gmm_parameters)
    )
    base["P_GMM"] = base["P_GMM_sdss"]
    base["log_L_GMM"] = calc_log_likelihood(
        *get_input_data(base), *(gmm_parameters[n] for n in param_labels_nosat)
    )

    basic_cut = C.gri_cut & C.fibermag_r_cut & C.is_clean & C.is_galaxy & (~C.has_spec)
    within_host = basic_cut & C.faint_end_limit & C.sat_rcut
    outwith_host = basic_cut & C.faint_end_limit & (~C.sat_rcut)

    veryhigh_p = Query("P_GMM_sdss >= 0.95", "log_L_GMM >= -7")
    high_p = Query("P_GMM_sdss >= 0.6", "log_L_GMM >= -7") | Query(
        "log_L_GMM < -7", "ri-abs(ri_err) < -0.25"
    )
    median_p = Query(
        "-(ug+abs(ug_err))*0.15+(ri-abs(ri_err)) < 0.08",
        "(gr-abs(gr_err))*0.65+(ri-abs(ri_err)) < 0.6",
        "-(ug+abs(ug_err))*0.1+(gr-abs(gr_err)) < 0.5",
    )

    base["TARGETING_SCORE"] = 1000
    fill_values_by_query(base, ~basic_cut, {"TARGETING_SCORE": 1100})
    fill_values_by_query(base, ~C.is_galaxy, {"TARGETING_SCORE": 1200})
    fill_values_by_query(base, ~C.is_clean, {"TARGETING_SCORE": 1300})
    fill_values_by_query(base, C.has_spec, {"TARGETING_SCORE": 1400})

    fill_values_by_query(base, outwith_host, {"TARGETING_SCORE": 900})
    fill_values_by_query(base, within_host, {"TARGETING_SCORE": 800})
    fill_values_by_query(
        base, basic_cut & veryhigh_p & (~C.sdss_limit), {"TARGETING_SCORE": 700}
    )
    fill_values_by_query(base, outwith_host & C.sdss_limit, {"TARGETING_SCORE": 500})
    fill_values_by_query(base, within_host & median_p, {"TARGETING_SCORE": 400})
    fill_values_by_query(base, within_host & high_p, {"TARGETING_SCORE": 300})
    fill_values_by_query(base, within_host & C.sdss_limit, {"TARGETING_SCORE": 200})

    need_random_selection = np.flatnonzero(base["TARGETING_SCORE"] == 800)
    if len(need_random_selection) > 50:
        random_mask = np.zeros(len(need_random_selection), dtype=np.bool)
        random_mask[:50] = True
        np.random.RandomState(123).shuffle(random_mask)  # pylint: disable=no-member
        need_random_selection = need_random_selection[random_mask]
    base["TARGETING_SCORE"][need_random_selection] = 600

    base["TARGETING_SCORE"] += (
        np.round((1.0 - base["P_GMM_sdss"]) * 80.0).astype(np.int) + 10
    )

    fill_values_by_query(base, C.is_sat, {"TARGETING_SCORE": 150})

    if manual_selected_objids is not None:
        fill_values_by_query(
            base,
            Query((lambda x: np.in1d(x, manual_selected_objids), "OBJID")),
            {"TARGETING_SCORE": 100},
        )

    base.sort("TARGETING_SCORE")
    return base


def assign_targeting_score_v2(
    base,
    manual_selected_objids=None,
    gmm_parameters=None,
    ignore_specs=False,
    debug=False,
    n_random=50,
    seed=123,
    remove_lists=None,
    low_priority_objids=None,
    **kwargs
):
    """
    Last updated: 05/19/2020
     100 Human selection and Special targets
     150 sats without AAT/MMT/PAL specs
     180 low-z (z < 0.05) but ZQUALITY = 2
     200 within host,  r < 17.77, gri/grz cuts OR very low SB
     300 within host,  r < 20.75, high p_GMM or GMM outliers or very high priority
     400 within host,  r < 20.75, main targeting cuts
     500 within host,  r < 20.75, gri/grz cuts, low-SB, random selection of 50
     600 outwith host, r < 17.77 OR very high p_GMM, low SB
     700 within host,  r < 20.75, gri/grz cuts, low SB
     800 within host,  r < 20.75, gri/grz cuts, everything else
     900 outwith host, r < 20.75, gri/grz cuts
    1000 everything else
    1100 Not in gri/grz cut
    1200 Not galaxy
    1300 Not clean
    1350 Removed by hand
    1400 Has spec already
    """

    basic_cut = C.gri_or_grz_cut & C.is_clean2 & C.is_galaxy2 & Query("r_mag < 21")
    if not ignore_specs:
        basic_cut &= ~C.has_spec

    base["score_sb_r"] = base["sb_r"] - 0.6 * (base["r_mag"] - np.abs(base["r_err"]))
    base["P_GMM"] = np.float(0)
    base["log_L_GMM"] = np.float(0)
    base["TARGETING_SCORE"] = 1000
    base["index"] = np.arange(len(base))

    surveys = [col[6:] for col in base.colnames if col.startswith("OBJID_")]

    if gmm_parameters is not None:
        for survey in surveys:

            gmm_parameters_this = gmm_parameters.get(survey)
            if gmm_parameters_this is None:
                continue

            postfix = "_" + survey
            base_this = Query(
                basic_cut,
                "OBJID{} != -1".format(postfix),
                "REMOVE{} == 0".format(postfix),
                "is_galaxy{}".format(postfix),
            ).filter(base)

            for color in get_all_colors():
                b1, b2 = color
                n1 = "".join((b1, "_mag", postfix))
                n2 = "".join((b2, "_mag", postfix))
                if n1 not in base_this.colnames or n2 not in base_this.colnames:
                    continue
                with np.errstate(invalid="ignore"):
                    base_this[color] = base_this[n1] - base_this[n2]
                    base_this[color + "_err"] = np.hypot(
                        base_this["".join((b1, "_err", postfix))],
                        base_this["".join((b2, "_err", postfix))],
                    )

            bands = getattr(  # pylint: disable=not-callable
                utils, "get_{}_bands".format(survey)
            )()

            base_this["P_GMM"] = ensure_proper_prob(
                calc_gmm_satellite_probability(
                    base_this,
                    gmm_parameters_this,
                    bands=bands,
                    mag_err_postfix="_err" + postfix,
                )
            )
            base_this["log_L_GMM"] = calc_log_likelihood(
                *get_input_data(
                    base_this, bands=bands, mag_err_postfix="_err" + postfix,
                ),
                *(gmm_parameters_this[n] for n in param_labels_nosat),
            )

            to_update_mask = base_this["P_GMM"] > base["P_GMM"][base_this["index"]]
            if to_update_mask.any():
                to_update_idx = base_this["index"][to_update_mask]
                for col in ("P_GMM", "log_L_GMM"):
                    base[col][to_update_idx] = base_this[col][to_update_mask]

            del base_this, to_update_mask

    del base["index"]

    bright = C.sdss_limit
    exclusion_cuts = Query()

    if low_priority_objids is not None:
        exclusion_cuts = Query(
            exclusion_cuts, QueryMaker.in1d("OBJID", low_priority_objids, invert=True)
        )

    if "sdss" in surveys and ("decals" in surveys or "des" in surveys):
        deep_survey = "des" if "des" in surveys else "decals"
        has_good_deep = Query(
            "OBJID_{} != -1".format(deep_survey), "REMOVE_{} == 0".format(deep_survey),
        )
        over_subtraction = Query(
            QueryMaker.equals("survey", "sdss"),
            Query(has_good_deep, "r_mag_{} > 20.8".format(deep_survey))
            | Query(~has_good_deep, "u_mag > r_mag + 3.5"),
        )
        exclusion_cuts = Query(exclusion_cuts, ~over_subtraction)

    if "des" in surveys:
        des_bright_stars = Query(
            QueryMaker.equals("survey", "des"),
            "0.7 * (r_mag + 10.2) > sb_r",
            "gr < 0.6",
            "r_mag < 17",
            C.valid_g_mag,
            C.valid_sb,
        )
        bright = Query(bright, ~des_bright_stars)
        exclusion_cuts = Query(exclusion_cuts, ~des_bright_stars)

    veryhigh_p_gmm = Query("P_GMM >= 0.95", "log_L_GMM >= -7")
    high_p_gmm = Query("P_GMM >= 0.7") | Query("log_L_GMM < -7")

    low_sb_cut = Query(Query("score_sb_r >= 11.25"), C.valid_sb)
    very_low_sb_cut = Query(
        "r_mag < 20.8",
        (
            Query(C.high_priority_cuts, Query("score_sb_r >= 12.5") | Query("sb_r >= 24.5"))
            | Query(QueryMaker.equals("survey", "des"), Query("score_sb_r >= 12.75") | Query("sb_r >= 24.75"))
        ),
        C.valid_sb,
        exclusion_cuts,
    )

    fill_values_by_query(base, C.faint_end_limit, {"TARGETING_SCORE": 900})
    fill_values_by_query(
        base, Query(C.sat_rcut, C.faint_end_limit), {"TARGETING_SCORE": 800}
    )
    fill_values_by_query(
        base,
        Query(
            C.sat_rcut,
            C.faint_end_limit,
            C.high_priority_sb,
            C.valid_sb,
            exclusion_cuts,
        ),
        {"TARGETING_SCORE": 700},
    )
    fill_values_by_query(
        base,
        (bright | Query(veryhigh_p_gmm, C.high_priority_sb, exclusion_cuts)),
        {"TARGETING_SCORE": 600},
    )
    fill_values_by_query(
        base,
        Query(C.sat_rcut, C.high_priority_cuts, C.faint_end_limit, exclusion_cuts),
        {"TARGETING_SCORE": 400},
    )
    fill_values_by_query(
        base,
        Query("TARGETING_SCORE == 400", (high_p_gmm | low_sb_cut)),
        {"TARGETING_SCORE": 300},
    )
    fill_values_by_query(
        base, Query(C.sat_rcut, (bright | very_low_sb_cut)), {"TARGETING_SCORE": 200}
    )

    need_random_selection = np.flatnonzero(
        Query(basic_cut, "TARGETING_SCORE >= 700", "TARGETING_SCORE < 800").mask(base)
    )
    if len(need_random_selection) > n_random:
        random_mask = np.zeros(len(need_random_selection), dtype=np.bool)
        random_mask[:n_random] = True
        np.random.RandomState(seed).shuffle(random_mask)  # pylint: disable=no-member
        need_random_selection = need_random_selection[random_mask]
    base["TARGETING_SCORE"][need_random_selection] = 500

    base["TARGETING_SCORE"] += (
        8 - np.digitize(base["score_sb_r"], np.linspace(10.5, 13.5, 7))
    ) * 10 + (9 - np.floor(base["P_GMM"] * 10).astype(np.int))

    fill_values_by_query(base, ~basic_cut, {"TARGETING_SCORE": 1100})
    fill_values_by_query(base, ~C.is_galaxy2, {"TARGETING_SCORE": 1200})
    fill_values_by_query(base, ~C.is_clean2, {"TARGETING_SCORE": 1300})

    if not ignore_specs:
        fill_values_by_query(base, C.has_spec, {"TARGETING_SCORE": 1400})

        fill_values_by_query(
            base,
            Query(basic_cut, "ZQUALITY == 2", "SPEC_Z < 0.05"),
            {"TARGETING_SCORE": 180},
        )

        fill_values_by_query(
            base,
            Query(
                C.is_sat,
                (lambda x: (x != "AAT") & (x != "MMT") & (x != "PAL"), "TELNAME"),
            ),
            {"TARGETING_SCORE": 150},
        )

    if remove_lists is not None:
        for survey in surveys:
            if survey not in remove_lists:
                continue
            fill_values_by_query(
                base,
                Query(
                    C.is_clean2,
                    (lambda x: np.in1d(x, remove_lists[survey]), "OBJID"),
                    (lambda x: x == survey, "survey"),
                ),
                {"TARGETING_SCORE": 1350},
            )

    if manual_selected_objids is not None:
        q = Query((lambda x: np.in1d(x, manual_selected_objids), "OBJID"))
        if not ignore_specs:
            q &= ~C.has_spec
        fill_values_by_query(base, q, {"TARGETING_SCORE": 100})

    base.sort("TARGETING_SCORE")
    return base


def assign_targeting_score_lowz(
    base, manual_selected_objids=None, gmm_parameters=None, ignore_specs=False, **kwargs
):
    base["p_GMM"] = calc_gmm_satellite_probability(
        base, gmm_parameters["des_lowz"], bands="grizy"
    )
    base["p_GMM_fsps"] = calc_gmm_satellite_probability(
        base, gmm_parameters["des_lowz_fake"], bands="griz"
    )
    base["pass_r_gr_cut"] = Query("r_mag < (2-gr)*14").mask(base)
    base["pass_gr_ri_cut"] = Query("0.5*gr + 0.05 > ri").mask(base)
    base["pass_r_sb_cut"] = Query("0.9*r_mag + 5.25 < sb_r").mask(base)

    base["TARGETING_SCORE"] = 1000
    basic = Query("r_mag > 18", "pass_r_gr_cut", "pass_gr_ri_cut", "pass_r_sb_cut")
    if not ignore_specs:
        basic &= ~C.has_spec
    fill_values_by_query(base, basic, {"TARGETING_SCORE": 900})
    fill_values_by_query(
        base, Query(basic, "p_GMM + p_GMM_fsps > 0.95"), {"TARGETING_SCORE": 800}
    )
    fill_values_by_query(
        base, Query(basic, "p_GMM > 0.6", "p_GMM_fsps > 0.6"), {"TARGETING_SCORE": 800}
    )

    if manual_selected_objids is not None:
        q = Query((lambda x: np.in1d(x, manual_selected_objids), "OBJID"))
        if not ignore_specs:
            q &= ~C.has_spec
        fill_values_by_query(base, q, {"TARGETING_SCORE": 800})

    base.sort("TARGETING_SCORE")
    return base
