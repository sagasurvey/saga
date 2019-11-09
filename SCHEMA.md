# Schema for SAGA Data Release 2 (DR2) Base Catalogs 

## Host list (v2) schema

Label | Unit | Definition
--- | --- | ---
... | ... | ...

## Object catalog (v2) schema 

_Note: data for SDSS DR14 (with postfix `_sdss`), DES DR1  (with postfix `_des`) and LS DR6/7  (with postfix `_decals`) are included where available._

Label | Unit | Definition
--- | --- | ---
`OBJID` | - | Unique photometric object identifier, taken from SDSS, DES or LS (based on survey flag)
`RA` | degree | Right ascension
`DEC` | degree | Declination
`REMOVE` | - |  0 if good object.  
`is_galaxy` | - | True if classified as galaxy
`radius` | arcsec | Effective radius of object
`<band>_mag` | - | ugriz-mag from primary survey, e.g.,  `r_mag`, [99 if no measurement]
`<band>_err` | - | ugriz magnitude error from primary survey, e.g.,  `r_err`, [99 if no measurement]
`survey` | - | Primary survey source for photometry [SDSS, DES, LS]
`TELNAME` | - | Primary source for spectroscopy
`SPEC_Z` | - |  Redshift of object (-1 if not measured)
`ZQUALITY` | - |  Quality of redshift (use if >=3)
`HOST_NSAID` | - |  NSAID of host (v0)
`HOST_NGC` | - |  NGC name of host
`HOST_PGC` | - |  PGC name of host
`HOST_RA` | deg |  Right Acension of host
`HOST_DEC` | deg |  Declination of host
`HOST_VHOST` | kms |  Recession velocity of host
`RHOST_KPC` | kpc |  Projected radial distance of object to host
`RHOST_ARCM` | arcm |  Projected radial distance of object to host
... | ... | ...
