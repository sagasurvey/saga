from easyquery import Query, QueryMaker

# fmt: off
_hostlist_v1_pgc = (
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
)

_paper1_complete_hostid = (
    'nsa132339', 'nsa33446', 'nsa165536', 'nsa166313',
    'nsa147100', 'nsa149781', 'nsa150887', 'nsa61945',
)

_paper1_incomplete_hostid = (
    'nsa126115', 'nsa129237', 'nsa85746', 'nsa137625',
    'nsa140594', 'nsa13927', 'nsa161174', 'nsa145729',
)

_paper2_hostid = (
    'nsa129237', 'nsa132339', 'nsa133355', 'nsa133549', 'pgc18880', 'nsa33446',
    'nsa16235', 'nsa157374', 'nsa32', 'nsa139467', 'nsa140458', 'nsa161174',
    'nsa3469', 'nsa141465', 'nsa143856', 'nsa94340', 'nsa69028', 'nsa144953',
    'nsa35340', 'nsa51348', 'nsa145372', 'nsa165536', 'nsa145729', 'nsa145879',
    'nsa166313', 'nsa147100', 'nsa147606', 'pgc64725', 'pgc66318', 'pgc66934',
    'pgc67782', 'pgc67817', 'nsa149781', 'nsa149977', 'nsa150887', 'nsa61945',
)

_paper3_hostid = (
    'nsa126115', 'nsa127226', 'nsa129237', 'nsa129387', 'nsa130133', 'nsa132339',
    'nsa132368', 'nsa133115', 'nsa133141', 'nsa133355', 'nsa133549', 'nsa135296',
    'nsa135591', 'nsa135739', 'nsa13927', 'nsa139467', 'nsa140458', 'nsa141315',
    'nsa141465', 'nsa143856', 'nsa14409', 'nsa144151', 'nsa144953', 'nsa145297',
    'nsa145372', 'nsa145729', 'nsa145879', 'nsa146486', 'nsa147100', 'nsa147606',
    'nsa149781', 'nsa149977', 'nsa150307', 'nsa150578', 'nsa150887', 'nsa153017',
    'nsa155005', 'nsa157300', 'nsa157374', 'nsa158901', 'nsa159488', 'nsa159520',
    'nsa159593', 'nsa160302', 'nsa161174', 'nsa16235', 'nsa163956', 'nsa164865',
    'nsa165162', 'nsa165316', 'nsa165536', 'nsa16559', 'nsa165707', 'nsa165956',
    'nsa165980', 'nsa166313', 'nsa169439', 'nsa32', 'nsa33446', 'nsa3469',
    'nsa35340', 'nsa47117', 'nsa51348', 'nsa52773', 'nsa56434', 'nsa61945',
    'nsa67593', 'nsa69028', 'nsa85746', 'nsa94217', 'nsa94340', 'pgc10330',
    'pgc12342', 'pgc13108', 'pgc13166', 'pgc16044', 'pgc18880', 'pgc1952',
    'pgc26034', 'pgc36304', 'pgc45643', 'pgc50670', 'pgc51055',
    'pgc53630', 'pgc57345', 'pgc64725', 'pgc66318', 'pgc66566', 'pgc66669',
    'pgc66934', 'pgc67146', 'pgc67532', 'pgc67663', 'pgc67782', 'pgc67817',
    'pgc68312', 'pgc69677', 'pgc72009', 'pgc72060', 'pgc8279', 'pgc9747',
)

_cos_saga_hostid = (
    "nsa85746", "nsa85746", "nsa85746", "nsa140594", "nsa13927",
    "nsa144953", "nsa165082", "nsa165153", "nsa165707", "nsa147606",
)
# fmt: on

has_decals_image = Query("COVERAGE_DECALS_DR9 >= 0.95")
has_des_image = Query("COVERAGE_DES_DR1 >= 0.95")
has_sdss_image = Query("COVERAGE_SDSS >= 0.95")
has_decals_dr8_image = Query("COVERAGE_DECALS_DR8 >= 0.95")
has_decals_dr9_image = Query("COVERAGE_DECALS_DR9 >= 0.95")
has_decals_dr9_image_99 = Query("COVERAGE_DECALS_DR9 >= 0.99")

has_image = Query("HAS_IMAGE > 0")
has_better_image = Query("HAS_IMAGE > 1")

potential_hosts = Query("HOST_SCORE > 0")
good_hosts = Query("HOST_SCORE >= 4")
build_default = good = good_hosts & has_image

hostlist_v1 = QueryMaker.isin("PGC", _hostlist_v1_pgc)
paper1_complete = QueryMaker.isin("HOSTID", _paper1_complete_hostid)
paper1_incomplete = QueryMaker.isin("HOSTID", _paper1_incomplete_hostid)
paper1 = paper1_observed = QueryMaker.isin("HOSTID", _paper1_complete_hostid + _paper1_incomplete_hostid)
paper2 = paper2_complete = QueryMaker.isin("HOSTID", _paper2_hostid)
paper3 = paper3_complete = QueryMaker.isin("HOSTID", _paper3_hostid)
cos_saga = QueryMaker.isin("HOSTID", _cos_saga_hostid)

observed_definition = Query("specs_ours_rvir >= 100", "specs_ours_rvir < 999999")

complete_definition2 = Query(
    "paper2_need_spec / paper2_total < 0.2",
    "paper2_total > 0",
    "paper2_need_spec < 999999",
)
complete_definition3 = Query(
    "paper3_need_spec / paper3_total < 0.4",
    "paper3_total > 0",
    "paper3_need_spec < 999999",
)
complete_definition = complete_definition3

observed = good & observed_definition
unobserved = good & (~observed_definition)
complete = observed & complete_definition
incomplete = observed & (~complete_definition)

environment_allowed = Query(
    "abs(GLAT) >= 25",
    "BRIGHTEST_K_R1 >= K_TC + 1",
    "BRIGHTEST_STAR_R1 >= 5",
    "M_HALO < 13",
    "REMOVED_BY_HAND == 0",
)
environment_preferred = Query(environment_allowed, "BRIGHTEST_K_R1 >= K_TC + 1.6")
mass_allowed = Query("K_ABS >= -24.7", "K_ABS <= -22.9")
mass_preferred = Query("K_ABS >= -24.6", "K_ABS <= -23.0")
distance_allowed = Query("DIST >= 20", "DIST <= 42", "V_HELIO >= 1400")
distance_preferred = Query("DIST >= 25", "DIST <= 40.75", "V_HELIO >= 1400")

has_mw_mass_neighbor = QueryMaker.not_equal("NEARBY_MW_COUNT_1.5", 0)
is_local_group_like = QueryMaker.equal("NEARBY_MW_COUNT_1.5", 1)
