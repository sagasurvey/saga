import astropy.constants
import numpy as np

__all__ = ["SPECS_COLUMNS", "SPEED_OF_LIGHT", "ensure_specs_dtype"]

SPEED_OF_LIGHT = astropy.constants.c.to_value("km/s")

_SPECS_COLUMNS = (
    ("RA", "<f8"),
    ("DEC", "<f8"),
    ("SPEC_Z", "<f4"),
    ("SPEC_Z_ERR", "<f4"),
    ("ZQUALITY", "<i4"),
    ("SPECOBJID", "<U48"),
    ("MASKNAME", "<U48"),
    ("TELNAME", "<U6"),
    ("HELIO_CORR", "|b1"),
    ("HI_FLUX", "<f4"),
    ("HI_FLUX_ERR", "<f4"),
    ("HI_SOURCE", "<U8"),
)

SPECS_COLUMNS = dict(_SPECS_COLUMNS)


def ensure_specs_dtype(spectra, cols_definition=_SPECS_COLUMNS, skip_missing_cols=False, remove_extra_cols=False):
    cols = []
    cols_iter = cols_definition.items() if isinstance(cols_definition, dict) else cols_definition
    for c, t in cols_iter:
        if c not in spectra.colnames:
            if skip_missing_cols:
                continue
            cols.append(c)
            if t[1] == "f":
                spectra[c] = np.nan
            elif t[1] == "i":
                spectra[c] = -1
            elif t[1] == "U":
                spectra[c] = ""
            elif t[1] == "b":
                spectra[c] = False
            else:
                raise ValueError("unknown spec type!")
        else:
            cols.append(c)
        if spectra[c].dtype.str != t:
            spectra.replace_column(c, spectra[c].astype(t))
        if spectra[c].description:
            spectra[c].description = None

    if remove_extra_cols:
        spectra = spectra[cols]

    return spectra
