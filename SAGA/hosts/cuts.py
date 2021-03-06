from easyquery import Query, QueryMaker

_list_by_pgc = {
    # fmt: off
    'hostlist_v1': (
        1952, 2052, 2081, 2437, 2778, 3089, 3743, 4948, 5139, 5619, 6110, 6656,
        6993, 8160, 8631, 9057, 9747, 9788, 9843, 9988, 10330, 10965, 771919,
        11819, 12007, 12431, 12626, 13130, 13166, 13602, 13881, 13999, 14084,
        16716, 16906, 16949, 17436, 18394, 18880, 20458, 22270, 23028, 23701,
        24464, 25069, 25248, 25886, 25955, 25999, 26246, 26483, 26649, 26939,
        27004, 27635, 27723, 27982, 28805, 28876, 29009, 30176, 31166, 31428,
        31636, 32533, 32854, 33371, 33623, 33860, 34561, 34731, 34908, 35088,
        35097, 35285, 35294, 35539, 35869, 36138, 36266, 36304, 36737, 36827,
        36914, 37178, 37366, 37483, 37845, 38031, 38049, 38555, 38588, 38802,
        38851, 39414, 40055, 40265, 40284, 41083, 41150, 42139, 44025, 44145,
        44600, 45269, 45795, 46304, 46938, 47394, 47721, 48258, 48307, 48767,
        48815, 48816, 49342, 49563, 49604, 49618, 50031, 50670, 50732, 51055,
        51340, 51471, 51564, 51620, 51787, 51932, 52116, 52266, 52365, 52411,
        52735, 52825, 53037, 53043, 53499, 53630, 53724, 53755, 54119, 54234,
        54348, 54364, 54429, 54761, 54849, 55081, 55281, 55588, 56337, 57345,
        57924, 58470, 59426, 61742, 61833, 61972, 63797, 64427, 64446, 64772,
        65436, 66318, 66566, 66648, 66812, 66934, 67018, 67146, 67546, 67663,
        67817, 67839, 67892, 67932, 68128, 68389, 68469, 68511, 68712, 68743,
        68826, 69094, 69253, 69349, 69521, 69661, 70025, 70094, 70128, 70419,
        70505, 70676, 70795, 71548, 71729, 71883, 72001, 73163,
    ),
    'paper1_complete': (9988, 23028, 52735, 55588, 58470, 68743, 70795, 71883),
    'paper1_incomplete': (279, 4948, 28805, 31166, 37845, 38031, 38802, 53499),
    'paper2_complete': (
        4948, 9988, 12626, 13646, 18880, 23028, 26246, 27635, 27723,
        35294, 37483, 38802, 40284, 41083, 48815, 49342, 50031, 51340,
        51471, 51620, 52273, 52735, 53499, 54119, 55588, 58470, 59426,
        64725, 66318, 66934, 67782, 67817, 68743, 69349, 70795, 71883,
    )
    # fmt: on
}

has_decals_image = Query("COVERAGE_DECALS_DR9 >= 0.95")
has_des_image = Query("COVERAGE_DES_DR1 >= 0.95")
has_sdss_image = Query("COVERAGE_SDSS >= 0.95")
has_decals_dr8_image = Query("COVERAGE_DECALS_DR8 >= 0.95")

has_image = Query("HAS_IMAGE > 0")
has_better_image = Query("HAS_IMAGE > 1")

potential_hosts = Query("HOST_SCORE > 0")
good_exceptions = QueryMaker.in1d("HOSTID", ("nsa163956", "nsa135739", "pgc67817"))
good_hosts = Query("HOST_SCORE >= 4") | good_exceptions
good = good_hosts & has_image
build_default = potential_hosts & has_image

hostlist_v1 = QueryMaker.in1d("PGC", _list_by_pgc["hostlist_v1"])
paper1_complete = QueryMaker.in1d("PGC", _list_by_pgc["paper1_complete"])
paper1_incomplete = QueryMaker.in1d("PGC", _list_by_pgc["paper1_incomplete"])
paper1 = QueryMaker.in1d("PGC", _list_by_pgc["paper1_complete"] + _list_by_pgc["paper1_incomplete"])
paper1_observed = paper1

paper2 = paper2_complete = QueryMaker.in1d("PGC", _list_by_pgc["paper2_complete"])

observed_definition = Query("specs_ours_rvir >= 100", "specs_ours_rvir < 999999")
complete_definition = Query(
    "paper2_need_spec / paper2_total < 0.2",
    "paper2_total > 0",
    "paper2_need_spec < 999999",
)

observed = good_hosts & observed_definition
unobserved = good & (~observed_definition)
complete = observed & complete_definition
incomplete = observed & (~complete_definition)
