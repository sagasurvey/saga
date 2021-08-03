"""
This file collects a set of common cuts we apply to objects.
See https://github.com/yymao/easyquery for further documentation.

In Jupyter, use ?? to see the definitions of the cuts

Examples
--------
>>> is_sat.filter(base_table)
"""

from easyquery import Query, QueryMaker

COLUMNS_USED = [
    "ZQUALITY",
    "REMOVE",
    "PHOTPTYPE",
    "FIBERMAG_R",
    "SPEC_Z",
    "RHOST_KPC",
    "HOST_VHOST",
    "SATS",
    "OBJ_NSAID",
    "HOST_NSAID",
    "SPEC_REPEAT",
    "r_mag",
    "ug",
    "ug_err",
    "gr",
    "gr_err",
    "ri",
    "ri_err",
]

COLUMNS_USED2 = [
    "ZQUALITY",
    "REMOVE",
    "is_galaxy",
    "SPEC_Z",
    "RHOST_KPC",
    "HOST_VHOST",
    "SATS",
    "SPEC_REPEAT",
    "r_mag",
    "i_mag",
    "ug",
    "ug_err",
    "gr",
    "gr_err",
    "ri",
    "ri_err",
    "rz",
    "rz_err",
    "sb_r",
    "sb_r_err",
]

has_spec = Query("ZQUALITY >= 3")
has_poor_spec = Query("ZQUALITY == 2")
has_failed_spec = Query("ZQUALITY > -1", "ZQUALITY < 2")

is_clean = Query("REMOVE == -1")
is_clean2 = Query("REMOVE == 0")
is_galaxy = Query("PHOTPTYPE == 3")
is_galaxy2 = Query("is_galaxy")
fibermag_r_cut = Query("FIBERMAG_R <= 23.0")

faint_end_limit2 = Query("r_mag < 20.75")
faint_end_limit3 = Query("r_mag < 20.65")
faint_end_limit = faint_end_limit3

sdss_limit = Query("r_mag < 17.77")
lowz_mag_cut = Query("r_mag > 18", "r_mag < 20")
r_abs_limit2 = Query("Mr < -12.295")
r_abs_limit3 = Query("Mr < -12.395")
r_abs_limit = r_abs_limit3

faint_end_limit_strict = Query("r_mag - log10(HOST_DIST)*5 - 25 < -12", faint_end_limit)

sat_vcut = Query("abs(SPEC_Z * 2.99792458e5 - HOST_VHOST) < 275.0")
sat_rcut = Query("RHOST_KPC < 300.0")

valid_u_mag = Query("u_mag > 0", "u_mag < 30")
valid_g_mag = Query("g_mag > 0", "g_mag < 30")
valid_i_mag = Query("i_mag > 0", "i_mag < 30")
valid_z_mag = Query("z_mag > 0", "z_mag < 30")
valid_sb = Query("sb_r > 0", "sb_r < 35")
valid_r_fibermag = Query("r_fibermag > 0", "r_fibermag < 35")

gr_cut = Query("gr - abs(gr_err) * 2 < 0.85") | (~valid_g_mag)
ri_cut = Query("ri - abs(ri_err) * 2 < 0.55") | (~valid_i_mag)
rz_cut = Query("rz - abs(rz_err) * 2 < 1") | (~valid_z_mag)
ug_cut = Query("ug + abs(ug_err) * 2 > (gr - abs(gr_err) * 2) * 1.5") | (~valid_u_mag) | (~valid_g_mag)
gri_cut = gr_cut & ri_cut
ugri_cut = gri_cut & ug_cut
grz_cut = gr_cut & rz_cut
paper1_targeting_cut = griz_cut = gri_or_grz_cut = gr_cut & ri_cut & rz_cut

lsbg_cut = Query(
    "HOST_DIST * 1000 * radius / 3600 / 90 * arccos(0) > 10 ** (-0.1 * (r_mag - log10(HOST_DIST) * 5 - 25 + 18))"
)
high_priority_sb_tight = Query("sb_r + abs(sb_r_err) - 0.7 * (r_mag - 14) > 18.5") | (~valid_sb)
gr_cut_tight = Query("gr - abs(gr_err) < 0.75") | (~valid_g_mag)

high_priority_sb = Query("sb_r + abs(sb_r_err) - 0.6 * (r_mag - 14) > 18.55") | (~valid_sb)
high_priority_gr = Query("gr - abs(gr_err) + 0.06 * (r_mag - 14) < 0.9") | (~valid_g_mag)
high_priority_ri = Query("ri - abs(ri_err) + 0.06 * (r_mag - 14) < 0.65") | (~valid_i_mag)
high_priority_rz = Query("rz - abs(rz_err) + 0.06 * (r_mag - 14) < 0.85") | (~valid_z_mag)

relaxed_cut_sb = Query("sb_r + abs(sb_r_err)*2 - 0.6 * (r_mag - 14) > 18.5") | (~valid_sb)
relaxed_cut_gr = Query("gr - abs(gr_err)*2 + 0.06 * (r_mag - 14) < 1.1") | (~valid_g_mag)
relaxed_cut_ri = Query("ri - abs(ri_err)*2 + 0.06 * (r_mag - 14) < 0.85") | (~valid_i_mag)
relaxed_cut_rz = Query("rz - abs(rz_err)*2 + 0.06 * (r_mag - 14) < 1.05") | (~valid_z_mag)

very_relaxed_cut_sb = Query("sb_r + abs(sb_r_err)*2 - 0.6 * (r_mag - 14) > 17.5") | (~valid_sb)

ba_cut = Query("ba > (r_mag - 20.75) * 0.08 + 0.3")

orig_high_priority_cuts = Query(
    high_priority_sb,
    high_priority_gr,
    high_priority_ri,
    high_priority_rz,
)

relaxed_targeting_cuts = Query(
    relaxed_cut_sb,
    relaxed_cut_gr,
    relaxed_cut_ri,
    relaxed_cut_rz,
)

very_relaxed_targeting_cuts = paper1_targeting_cut | relaxed_targeting_cuts | "p_sat_approx >= 1e-4"

paper2_targeting_cut = high_priority_cuts = main_targeting_cuts = high_priority_sb_tight & high_priority_gr
paper3_targeting_cut = paper2_targeting_cut & ba_cut

is_sat = Query("SATS == 1")
is_host = Query("SATS == 3")

is_high_z = Query("SPEC_Z >= 0.02")
is_low_z = Query("SPEC_Z >= 0.003", ~is_high_z)
is_very_low_z = Query("SPEC_Z >= 0.003", "SPEC_Z < 0.013")

obj_is_host = Query("OBJ_NSAID == HOST_NSAID", "OBJ_NSAID != -1")
obj_is_host2 = Query("OBJ_NSAID == HOST_NSA1ID", "OBJ_NSAID != -1")
obj_is_host3 = Query("OBJ_PGC == HOST_PGC", "OBJ_PGC != -1")

basic_cut = is_clean & is_galaxy & fibermag_r_cut & faint_end_limit & sat_rcut
basic_cut2 = is_clean2 & is_galaxy2 & faint_end_limit & sat_rcut
basic_cut_lowz = is_clean2 & is_galaxy2 & lowz_mag_cut & gri_cut

has_sdss_spec = QueryMaker.contains("SPEC_REPEAT", "SDSS")
has_nsa_spec = QueryMaker.contains("SPEC_REPEAT", "NSA")
has_sdss_nsa_spec = has_sdss_spec | has_nsa_spec

has_aat_spec = QueryMaker.contains("SPEC_REPEAT", "AAT")
has_mmt_spec = QueryMaker.contains("SPEC_REPEAT", "MMT")
has_aat_mmt_spec = has_aat_spec | has_mmt_spec

_known_telnames = {
    "HOST",
    "2dF",
    "6dF",
    "SDSS",
    "NSA",
    "GAMA",
    "OzDES",
    "2dFLen",
    "WIGGZ",
    "UKST",
    "LCRS",
    "slack",
    "ALFALF",
    "SGA",
}
has_our_specs_only = QueryMaker.vectorize(lambda x: x and set(x.split("+")).isdisjoint(_known_telnames), "SPEC_REPEAT")
has_our_specs = QueryMaker.vectorize(lambda x: x and not set(x.split("+")).issubset(_known_telnames), "SPEC_REPEAT")
has_been_targeted = QueryMaker.vectorize(
    lambda x: x and not set(x.split("+")).issubset(_known_telnames), "SPEC_REPEAT_ALL"
)
