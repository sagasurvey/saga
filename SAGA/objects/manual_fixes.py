fixes_by_sdss_objid = {
    # WRONG REDSHIFT IN THE NSA, but good in SDSS
    1237668367995568266: {"SPEC_Z": 0.21068, "TELNAME": "SDSS", "MASKNAME": "SDSS"},
    # DON"T BELIEVE THIS NED REDSHIFT, RESET TO -1
    1237667966962434538: {"SPEC_Z": -1, "ZQUALITY": -1},
    # A SATELLITE WITH A BAD PETRORAD_R
    1237651735757259099: {"PETRORAD_R": 2.97},
    # WRONG REDSHIFT IN NSA, BUT GOOD IN SDSS
    1237678881574551723: {"SPEC_Z": 1.093277, "TELNAME": "SDSS", "MASKNAME": "SDSS"},
    # NSA BELIEVES THE SDSS REDSHIFT, WHICH IS TOO LOW-SN
    1237661356465979704: {"ZQUALITY": -1},
    # ODYSSET SATELLITE SHRED, BUT IS GOOD
    1237662662147638034: {"REMOVE": -1},
    # BRIGHT TARGETS FROM PALOMAR
    1237662698115367389: {
        "SPEC_Z": 0.0907,
        "ZQUALITY": 4,
        "TELNAME": "MMT",
        "MASKNAME": "PAL",
    },
    1237679996084486446: {
        "SPEC_Z": 0.0524,
        "ZQUALITY": 4,
        "TELNAME": "MMT",
        "MASKNAME": "PAL",
    },
    # RISA GALAXY
    1237666408439939196: {
        "RA": 354.28276,
        "DEC": 0.211263,
        "r": 17.8,
        "g": 18.5,
        "i": 17.5,
    },
}

fixes_to_nsa_v012 = {
    69840: {"RA": 255.038042622208, "DEC": 22.9030230453907},
}

fixes_to_nsa_v101 = {
    257573: {"PETRO_TH90": 120.0},
    632316: {"PETRO_TH90": 120.0},
    39145: {"PETRO_TH90": 21.5},
    645571: {"PETRO_TH90": 22.5},
    58266: {"PETRO_TH90": 9.0},
    623347: {"Z": 0.358},
}

fixes_to_decals_dr9 = {
    904604600000003107: dict(  # NSA (v1.0.1) 343647 (255.5115, 22.9355)
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
    901050380000007338: dict(sma=100),  # (330.14926, -43.140017)
    903255060000002115: dict(sma=270, ba=0.15, phi=81.0),  # (224.594532 -1.090942)
    904589200000000836: dict(sma=60, ba=0.55, phi=50.0),  # (198.1743458, 22.829911)
    903897920000000718: dict(sma=135, ba=0.75, phi=40.0),  # (163.0159, 10.1479)
    901704030000004690: dict(sma=170, phi=83.0),  # (40.9350, -29.0035)
    901039870000000760: dict(sma=36.0),  # (330.1370, -43.3897)
    902988610000000241: dict(sma=138.0),  # (35.4017, -5.5212)
    915434150000002223: dict(sma=150.0),  # (138.5198, 40.1133)
}
