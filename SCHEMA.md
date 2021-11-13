# Schema for SAGA Data Release 3 (DR3) Base Catalogs

## Object catalog (v3) schema

Label | Unit | Definition
--- | --- | ---
`OBJID` | - | Unique photometric object identifier
`RA` | degree | Right ascension J2000
`DEC` | degree | Declination J2000
`REMOVE` | - |  0 if good object
`is_galaxy` | - | True if classified as galaxy
`morphology_info` | - | Additional morphology info; users should use `is_galaxy` instead
`radius` | arcsec | Effective (half-light) circularized radius of object
`radius_err` | arcsec | Estimated error on effective radius of object
`ba` | - | galaxy aspect ratio (minor-to-major axis ratio; b/a)
`phi` | - | position angle of the galaxy
`sma` | - | semi-major axis at about ~ mag/arcsec^2 surface brightness isophote
`REF_CAT` | - | whether this object is in the SGA (`L3`) or the Gaia (`G2`) catalogs
`SGA_ID` | - | SGA ID for objects in the SGA catalog
`<band>_mag` | mag | ugrizy magnitude from primary survey, e.g., `r_mag`. [99 or NaN if no measurement]
`<band>_err` | mag | ugrizy magnitude error from primary survey, e.g., `r_err`. [99 or NaN if no measurement]
`r_fibermag` | mag | r-band fiber magnitude (fiber diameter = 1.5 arcsec)
`survey` | - | Primary survey source for photometry
`TELNAME` | - | Primary source for spectroscopy
`SPEC_Z` | - |  Redshift of object (-1 if not measured)
`SPEC_Z_ERR` | - | Estimated error on redshift
`ZQUALITY` | - |  Quality of redshift (use if >=3)
`SPEC_REPEAT` | - |  All good redshift sources
`SPEC_REPEAT_ALL` | - |  All redshift sources, including ones with low quality
`SPECOBJID` | - | spectroscopic object identifier
`MASKNAME` | - | spectroscopic mask name
`RA_spec` | deg | Right ascension J2000 of the spectroscopic object
`DEC_spec`  | deg |  Declination J2000 of the spectroscopic object
`spec_phot_sep` | arcsec | separation between the photometric and the spectroscopic coordinates
`HELIO_CORR` | - | True of heliocentric correction has be applied
`EW_Halpha` | angstrom | Halpha equivalent width
`EW_Halpha_err` | angstrom | Error on Halpha equivalent width
`SATS` | - | 1 for satellites, 2 for non-satellite low-z galaxies, 3 for host
`sb_r` | - | Effective apparent surface brightness in r-band
`log_sm` | - | log (stellar mass / Msun)
`HOST_NSAID` | - |  NSAID of host (v0)
`HOST_NSA1ID` | - |  NSAID of host (v1)
`HOST_PGC` | - |  PGC name of host
`HOST_RA` | deg |  Right ascension J2000 of host
`HOST_DEC` | deg |  Declination J2000 of host
`HOST_DIST` | Mpc | Distance of the host
`HOST_VHOST` | km/s |  Recession (Heliocentric) velocity of host
`RHOST_KPC` | kpc |  Projected radial distance of object to host
`RHOST_ARCM` | arcm |  Projected radial distance of object to host
`p_sat_approx` | - | Raw pproximated probability of this object being a satellite. Do NOT use.
`p_sat_corrected` | - | Corrected approximated probability of this object being a satellite. Use with care.

_*For SAGA DR2, data in these columns are taken from the primary survey source (see `survey`). There are additional columns with postfix to identify the quantities taken from specific surveys: SDSS DR14 (with postfix `_sdss`), DES DR1  (with postfix `_des`), and LS DR6/7  (with postfix `_decals`). For SAGA DR3, these are taken from LS DR9._


## Host list (v2) schema

Label | Unit | Definition
--- | --- | ---
`HOSTID` | - | unique host identifier
`PGC` | - | unique PGC number
`SAGA_NAME` | - | host's SAGA name
`COMMON_NAME` | - | host's common name (e.g., NGC, UGC)
`NSAID` | - | NSAID v0.1.2
`NSA1ID` | - | NSAID v1.0.1
`RA` | deg | Right ascension J2000 of host
`DEC` | deg | Declination J2000 of host
`GLON` | deg | Galactic longitude
`GLAT` | deg | Galactic latitude
`V_HELIO` | km/s | Recession (Heliocentric) velocity of host
`V_VIRGO` | km/s | Recession (Virgocentric) velocity of host
`Z_COSMO` | - | Cosmological redshift
`Z_HELIO` | - | Heliocentric redshift
`DIST` | Mpc | distance
`DISTMOD` | mag | distance modulus
`K_RAW` | mag | Raw Ks-band luminosity from source catalogs
`K_TC` | mag | Total Ks-band luminosity (extinction corrected, K-corrected)
`K_ABS` | Mag | Absolute Ks-band luminosity
`M_HALO` | - | log (halo mass / Msun), for the group it belongs to, if any
`REMOVED_BY_HAND` | - | True if the host is flagged as not usable by hand
`BRIGHTEST_K_R1` | mag | Ks-band mag of brightest galaxy (<60 Mpc) within 1 Rvir
`BRIGHTEST_K_R2` | mag | Ks-band mag of brightest galaxy (<60 Mpc) within 2 Rvir
`BRIGHTEST_K_BG_R1` | mag | Ks-band mag of brightest galaxy (>60 Mpc) within 1 Rvir
`BRIGHTEST_K_BG_R2` | mag | Ks-band mag of brightest galaxy (>60 Mpc) within 2 Rvir
`BRIGHTEST_STAR_R1` | mag | Hp-band mag of brightest star within 1 Rvir
`BRIGHTEST_STAR_R2` | mag | Hp-band mag of brightest star within 2 Rvir
`COVERAGE_DECALS_DR5` | - | Fraction of LS DR5 coverage within 1 Rvir
`COVERAGE_DECALS_DR6` | - | Fraction of LS DR6 coverage within 1 Rvir
`COVERAGE_DECALS_DR7` | - | Fraction of LS DR7 coverage within 1 Rvir
`COVERAGE_DECALS_DR8` | - | Fraction of LS DR8 coverage within 1 Rvir
`COVERAGE_DECALS_DR9` | - | Fraction of LS DR9 coverage within 1 Rvir
`COVERAGE_DES_DR1` | - | Fraction of DES Dr1 coverage within 1 Rvir
`COVERAGE_SDSS` | - | Fraction of SDSS coverage within 1 Rvir
`HOST_SCORE` | - | Usable hosts if >=4
`HAS_IMAGE` | - | Has image coverage if > 0

### Host statistics schema

Label | Unit | Definition
--- | --- | ---
`really_need_spec` | - | number of remaining targets within main targeting cuts (r < 20.75, within Rvir)
`really_need_spec_bright` | - | number of remaining targets within main targeting cuts (r < 17.77, within Rvir)
`specs_ours_rvir` | - | number of SAGA redshifts within Rvir
`sats_total` | - | number of satellites
`sats_r_limit` | - | number of satellites (r < 20.75)
`sats_Mr_limit` | - | number of satellites (Mr < -12.3)
`sats_bright` | - | number of satellites (r < 17.77)
`sats_ours` | - | number of SAGA-discovered satellites
`sats_ours_r_limit` | - | number of SAGA-discovered satellites (r < 20.75)
`sats_ours_Mr_limit` | - | number of SAGA-discovered satellites (Mr < -12.3)
`sats_ours_bright` | - | number of SAGA-discovered satellites (r < 17.77)
`sats_missed_approx` | - | model-predicted number of missed satellites within main targeting cuts (r < 20.75, within Rvir)
