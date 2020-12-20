"""
SAGA.host.selection

This file contains the functinos to build the master list and host list
"""

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import Distance, SkyCoord
from astropy.cosmology import WMAP9  # pylint: disable=no-name-in-module
from astropy.table import Table, join, vstack
from astropy.time import Time
from easyquery import Query, QueryMaker

from ..external.calc_kcor import calc_kcor
from ..utils import add_skycoord, fill_values_by_query
from ..utils.distance import d2z, m2d, v2z, z2m

SAGA_NAMES = {
    37845: "Alice",
    71883: "AnaK",
    279: "Bandamanna",
    70795: "Catch22",
    52735: "Dune",
    55588: "Gilgamesh",
    4948: "HarryPotter",
    28805: "MobyDick",
    9988: "Narnia",
    68743: "OBrother",
    58470: "Odyssey",
    53499: "Othello",
    31166: "Oz",
    38802: "ScoobyDoo",
    38031: "Sopranos",
    23028: "StarTrek",
    12626: 'Hiccup',
    13646: 'Genji',
    18880: 'Mulan',
    26246: 'Okonkwo',
    27635: 'Aeneid',
    27723: 'Skywalker',
    35294: 'Chihiro',
    37483: 'Gaukur',
    40284: 'Macondo',
    41083: 'Metamorphoses',
    48815: 'Rand',
    49342: 'Trisolaris',
    50031: 'Essun',
    51340: 'Pippi',
    51471: 'Beloved',
    51620: 'DonQuixote',
    52273: 'SunWukong',
    54119: 'Ynglinga',
    59426: 'Arya',
    64725: 'Moana',
    66318: 'Ozymandias',
    66934: 'Middlemarch',
    67782: 'Bilbo',
    67817: 'Frodo',
    69349: 'PiPatel',
}


def notnull(d):
    try:
        return ~d.mask
    except AttributeError:
        return ~np.isnan(d)


def join_by_pgc(d, to_join, join_type="left", postfix=None):
    d = join(
        d,
        to_join,
        "pgc",
        join_type,
        uniq_col_name="{col_name}{table_name}",
        table_names=["", (("_" + postfix) if postfix else "")],
        metadata_conflicts="silent",
    )
    d.meta = {}
    return d


def add_nsa(d, nsa):

    d["modz_better"] = np.where(d["modz"].mask, d["modbest"], d["modz"])
    d_cols_needed = ["pgc", "modz_better", "bt"]

    nsa_cols_needed = ["NSAID", "RA", "DEC", "ZDIST", "MAG"]
    if "PETRO_TH50" in nsa.colnames:
        nsa_cols_needed.append("PETRO_TH50")
    else:
        nsa_cols_needed.append("PETROTH50")

    nsa = nsa[nsa_cols_needed]
    if "PETRO_TH50" in nsa.colnames:
        nsa.rename_column("PETRO_TH50", "PETROTH50")
        nsa.rename_column("NSAID", "NSA1ID")

    nsa["modz_nsa"] = z2m(nsa["ZDIST"])
    nsa = add_skycoord(nsa)

    q_dmod = Query("abs(modz_nsa - modz_better) < 0.5")
    q_dmag = Query("abs(bt - MAG) < 2")
    q1 = Query("sep < 20", "sep < PETROTH50*0.5", q_dmod, q_dmag)
    q2 = Query("sep < 10", "sep < PETROTH50", q_dmod, q_dmag)
    q3 = Query("sep < 1", q_dmod | q_dmag)
    q_match = q1 | q2 | q3

    idx, sep, _ = nsa["coord"].match_to_catalog_sky(d["coord"])
    for col in d_cols_needed:
        nsa[col] = d[col][idx]
    nsa["sep"] = sep.arcsec

    fill_values_by_query(nsa, ~q_match, {"pgc": -1})
    del nsa["coord"]
    nsa.sort(["pgc", "sep"])
    mask = np.insert(np.ediff1d(nsa["pgc"]), 0, 1) == 0
    nsa["pgc"][mask] = -1

    # give the not matches ones a second chance
    nsa_not_matched = nsa[nsa["pgc"] == -1]
    nsa_not_matched = add_skycoord(nsa_not_matched)
    idx, sep, _ = nsa_not_matched["coord"].match_to_catalog_sky(d["coord"], 2)
    for col in d_cols_needed:
        nsa_not_matched[col] = d[col][idx]
    nsa_not_matched["sep"] = sep.arcsec

    del nsa_not_matched["coord"]
    nsa_matched = vstack([nsa[nsa["pgc"] != -1], q_match.filter(nsa_not_matched)])
    nsa_matched.sort(["pgc", "sep"])
    mask = np.insert(np.ediff1d(nsa_matched["pgc"]), 0, 1) == 0
    nsa_matched = nsa_matched[~mask]
    assert len(np.unique(nsa_matched["pgc"])) == len(nsa_matched)

    if "NSA1ID" in nsa_matched.colnames:
        nsa_matched = nsa_matched["pgc", "NSA1ID", "modz_nsa"]
    else:
        nsa_matched = nsa_matched["pgc", "NSAID"]

    return join_by_pgc(d, nsa_matched)


def add_manual_remove_flag(d, remove):
    if "HOSTID" in remove.colnames:
        remove_id = Query("flag > 1").filter(remove, "HOSTID")
        q = QueryMaker.in1d("HOSTID", remove_id)
    elif "NSAID" in remove.colnames:
        remove_id = Query("flag > 1").filter(remove, "NSAID")
        q = Query((notnull, "NSAID"), QueryMaker.in1d("NSAID", remove_id))
    else:
        raise ValueError("cannot find `HOSTID` or `NSAID` column in `remove`")

    d["REMOVED_BY_HAND"] = q.mask(d).astype(np.int32)

    return d


def add_hostid(d):
    pgc_col = "pgc" if "pgc" in d.colnames else "PGC"
    nsa_col = "NSAID"
    d["HOSTID"] = np.where(
        (d[nsa_col].filled(-1) <= 0),
        np.char.add("pgc", d[pgc_col].astype(np.unicode)),
        np.char.add("nsa", d[nsa_col].astype(np.unicode)),
    )
    return d


def calc_needed_quantities(d):

    d["DISTMOD"] = np.where(
        d["modz_nsa"].mask | ((~d["mod0"].mask) & (d["e_modbest"] < 0.2)),
        d["modbest"],
        d["modz_nsa"],
    )
    d["DIST"] = m2d(d["DISTMOD"])
    d["Z_COSMO"] = d2z(d["DIST"])

    d["kcorrection"] = calc_kcor(
        "Ks2", d["Z_COSMO"].data, "H2 - Ks2", d["H_tc"].data - d["K_tc"].data
    )
    d["K_RAW"] = np.where(d["K_tc"].mask, d["kt"], d["K_tc"])
    d["K_TC"] = np.where(d["K_tc"].mask, d["kt"] - 0.03, d["K_tc"] - d["kcorrection"])
    d["K_ABS"] = d["K_TC"] - d["DISTMOD"]

    return d


def add_skycoord_stars(stars):
    # pylint: disable=no-member
    stars["coord"] = SkyCoord(
        ra=stars["RArad"] * u.rad,
        dec=stars["DErad"] * u.rad,
        distance=Distance(parallax=stars["Plx"] * u.mas, allow_negative=True),
        pm_ra_cosdec=stars["pmRA"] * u.mas / u.yr,
        pm_dec=stars["pmDE"] * u.mas / u.yr,
        obstime=Time(1991.25, format="decimalyear"),
    ).apply_space_motion(Time(2000.0, format="decimalyear"))
    return stars


def find_nearby_brightest(
    d,
    other,
    mag_label,
    output_prefix,
    radii=(0.3, 0.6),
    density_prefix=None,
    density_mag_limit=None,
):
    radii = np.asarray(radii)
    output = []
    for obj in d:
        radii_radian = np.arcsin(radii / obj["DIST"])
        sep = obj["coord"].separation(other["coord"]).radian
        last_r = 0
        for r in radii_radian:
            mask = (sep > last_r) & (sep <= r)
            output.append(other[mag_label][mask].min() if mask.any() else 99.0)
            if density_prefix:
                if density_mag_limit is not None:
                    mask &= other[mag_label] < density_mag_limit
                area = (np.rad2deg(r) ** 2 - np.rad2deg(last_r) ** 2) * np.pi
                output.append(np.count_nonzero(mask) / area)
            last_r = r

    output = np.array(output).reshape(len(d), -1).T
    for i, v in enumerate(output):
        j = (i // len(radii) + 1) if density_prefix else (i + 1)
        if density_prefix and (i % 2):
            d["{}_R{}".format(density_prefix, j)] = v
            continue
        d["{}_R{}".format(output_prefix, j)] = v
    return d


def add_image_coverage(d, coverage_map, name, nest=True):
    frac = []
    nside = hp.npix2nside(len(coverage_map))
    for obj in d:
        ra, dec, dist = (obj["al2000"] * 15.0, obj["de2000"], obj["DIST"])
        if dist <= 0:
            frac.append(np.nan)
            continue
        idx = hp.query_disc(
            nside, hp.ang2vec(ra, dec, lonlat=True), np.arcsin(0.3 / dist), nest=nest
        )
        frac.append(
            np.count_nonzero(coverage_map[idx]) / len(idx) if len(idx) else np.nan
        )
    d["COVERAGE_{}".format(name.upper())] = np.array(frac)
    return d


def add_saga_names(d):
    sn = Table(rows=list(SAGA_NAMES.items()), names=("pgc", "SAGA_NAME"))
    d = join_by_pgc(d, sn)
    d["SAGA_NAME"] = d["SAGA_NAME"].filled("")
    return d


def clean_up_columns(d):

    d.rename_column("pgc", "PGC")
    d.rename_column("objname", "COMMON_NAME")
    d.rename_column("v", "V_HELIO")
    d.rename_column("vvir", "V_VIRGO")
    d.rename_column("de2000", "DEC")
    d.rename_column("l2", "GLON")
    d.rename_column("b2", "GLAT")
    d.rename_column("Mhalo", "M_HALO")

    d["Z_HELIO"] = v2z(d["V_HELIO"])
    d["RA"] = np.remainder(d["al2000"] * 15.0, 360.0)
    d["NSAID"] = d["NSAID"].filled(-1)
    d["NSA1ID"] = d["NSA1ID"].filled(-1)
    d["M_HALO"] = d["M_HALO"].filled(-1.0)

    needed_cols = [
        "HOSTID",
        "PGC",
        "SAGA_NAME",
        "COMMON_NAME",
        "NSAID",
        "NSA1ID",
        "RA",
        "DEC",
        "GLON",
        "GLAT",
        "V_HELIO",
        "V_VIRGO",
        "Z_COSMO",
        "Z_HELIO",
        "DIST",
        "DISTMOD",
        "K_RAW",
        "K_TC",
        "K_ABS",
        "M_HALO",
        "REMOVED_BY_HAND",
    ]

    needed_cols.extend(
        (
            col.upper()
            for col in d.colnames
            if col.startswith("BRIGHTEST_")
            or col.startswith("COVERAGE_")
            or col.startswith("STAR_DENSITY_")
        )
    )

    return d[needed_cols]


def add_selection_flags(d):

    env_allowed = Query(
        "abs(GLAT) >= 25",
        "BRIGHTEST_K_R1 >= K_TC + 1",
        "BRIGHTEST_STAR_R1 >= 5",
        "M_HALO < 13",
        "REMOVED_BY_HAND == 0",
    )

    env_preferred = Query(env_allowed, "BRIGHTEST_K_R1 >= K_TC + 1.6")

    sample_allowed = Query(
        "K_ABS >= -24.7",
        "K_ABS <= -22.9",
        "DIST >= 20",
        "DIST <= 42",
        "V_HELIO >= 1400",
    )

    sample_preferred = Query(
        sample_allowed,
        "K_ABS >= -24.6",
        "K_ABS <= -23.0",
        "DIST >= 25",
        "DIST <= 40.75",
    )

    d["HOST_SCORE"] = (env_allowed & sample_allowed).mask(d).astype(np.int)
    d["HOST_SCORE"] += (env_preferred & sample_allowed).mask(d).astype(np.int)
    d["HOST_SCORE"] += (env_allowed & sample_preferred).mask(d).astype(np.int) * 2
    assert ((env_preferred & sample_preferred).filter(d, "HOST_SCORE") == 4).all()

    image_allowed = Query(
        (
            lambda *cols: np.amax(np.vstack(cols), axis=0) >= 0.99,
            "COVERAGE_DECALS_DR6",
            "COVERAGE_DECALS_DR7",
            "COVERAGE_DES_DR1",
            "COVERAGE_SDSS",
        ),
        (
            lambda c1, c2: np.maximum(c1, c2) >= 0.85,
            "COVERAGE_DES_DR1",
            "COVERAGE_SDSS",
        ),
    )

    image_preferred = Query(
        image_allowed,
        (
            lambda d6, d7, de, s: (
                (de >= 0.99)
                | (np.median(np.vstack([np.maximum(d6, d7), de, s]), axis=0) >= 0.95)
            ),
            "COVERAGE_DECALS_DR6",
            "COVERAGE_DECALS_DR7",
            "COVERAGE_DES_DR1",
            "COVERAGE_SDSS",
        ),
    )

    d["HAS_IMAGE"] = image_allowed.mask(d).astype(np.int)
    d["HAS_IMAGE"] += image_preferred.mask(d).astype(np.int)

    return d


def apply_manual_fixes(d):
    # pgc64427 too many stars, affecting the DES DR1 catalog
    fill_values_by_query(d, QueryMaker.equals("HOSTID", "pgc64427"), {"HOST_SCORE": 3})
    return d


def build_master_list(
    hyperleda,
    edd_2mrs,
    edd_lim17,
    nsa,
    nsa1,
    remove_list=None,
    stars=None,
    coverage_maps=None,
):

    d = hyperleda.copy()
    d.meta = {}
    del hyperleda

    edd_2mrs = edd_2mrs["pgc", "K_tc", "H_tc"]
    d = join_by_pgc(d, edd_2mrs)
    del edd_2mrs

    edd_lim17 = edd_lim17["PGC", "Mhalo"]
    edd_lim17.rename_column("PGC", "pgc")
    edd_lim17["Mhalo"] -= np.log10(WMAP9.h)
    d = join_by_pgc(d, edd_lim17["pgc", "Mhalo"])
    del edd_lim17

    d = add_skycoord(d, "al2000", "de2000", unit=("hourangle", "deg"))

    d = add_nsa(d, nsa)
    del nsa

    d = add_nsa(d, nsa1)
    del nsa1

    d = add_hostid(d)

    if remove_list is not None:
        d = add_manual_remove_flag(d, remove_list)
        del remove_list

    d = calc_needed_quantities(d)

    q_dist = Query("DIST >= 1", "DIST <= 60", (notnull, "DIST"))
    d_bg = (~q_dist).filter(d)
    d = q_dist.filter(d)

    d = find_nearby_brightest(d, d, "K_TC", "BRIGHTEST_K")
    d = find_nearby_brightest(d, d_bg, "K_TC", "BRIGHTEST_K_BG")
    del d_bg

    if stars is not None:
        stars = Query("Plx > 0", (notnull, "Plx"), (notnull, "Hpmag")).filter(stars)
        stars = add_skycoord_stars(stars)
        d = find_nearby_brightest(
            d,
            stars,
            "Hpmag",
            "BRIGHTEST_STAR",
            density_prefix="STAR_DENSITY",
            density_mag_limit=7,
        )
        del stars
    else:
        d["BRIGHTEST_STAR"] = 99.0

    if coverage_maps:
        for name, coverage_map in coverage_maps.items():
            d = add_image_coverage(d, coverage_map, name)

    d = add_saga_names(d)
    d = clean_up_columns(d)

    assert not any(map(np.ma.is_masked, d.itercols()))
    d = d.filled()

    d.sort("PGC")
    d = add_selection_flags(d)
    d = apply_manual_fixes(d)

    return d
