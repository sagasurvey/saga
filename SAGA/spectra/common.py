import astropy.constants
import numpy as np

__all__ = ["SPECS_COLUMNS", "SPEED_OF_LIGHT", "ensure_specs_dtype"]

SPEED_OF_LIGHT = astropy.constants.c.to_value("km/s")  # pylint: disable=E1101

_SPECS_COLUMNS = (
    ("RA", "<f8"),
    ("DEC", "<f8"),
    ("SPEC_Z", "<f4"),
    ("SPEC_Z_ERR", "<f4"),
    ("ZQUALITY", "<i2"),
    ("SPECOBJID", "<U48"),
    ("MASKNAME", "<U48"),
    ("TELNAME", "<U6"),
    ("EM_ABS", "<i2"),
    ("HELIO_CORR", "|b1"),
    ("EW_Halpha", "<f8"),
    ("EW_Halpha_err", "<f8"),
)

SPECS_COLUMNS = dict(_SPECS_COLUMNS)


def ensure_specs_dtype(
    spectra, cols_definition=_SPECS_COLUMNS, skip_missing_cols=False
):
    cols_iter = (
        cols_definition.items()
        if isinstance(cols_definition, dict)
        else cols_definition
    )
    for c, t in cols_iter:
        if c not in spectra.colnames:
            if skip_missing_cols:
                continue
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
        if spectra[c].dtype.str != t:
            spectra.replace_column(c, spectra[c].astype(t))

    return spectra
