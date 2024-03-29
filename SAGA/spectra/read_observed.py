import logging
import os

import astropy.constants
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.io.ascii.cparser import CParserError
from astropy.table import Table, vstack
from astropy.time import Time
from easyquery import Query

from ..database import CsvTable, FitsTable
from .common import SPEED_OF_LIGHT, ensure_specs_dtype


__all__ = [
    "read_generic_spectra",
    "read_mmt",
    "read_mmt_bino",
    "read_aat",
    "read_aat_mz",
    "read_imacs",
    "read_wiyn",
    "read_palomar",
    "read_coadd",
]


def get_obs_info_from_fits(
    fits_filepath,
    ra_name="RA",
    dec_name="DEC",
    time_name="MJD",
    ra_unit="hourangle",
    dec_unit="deg",
    time_format="mjd",
):
    hdr = fits.getheader(fits_filepath)
    sc = SkyCoord(hdr[ra_name], hdr[dec_name], unit=(ra_unit, dec_unit))
    obstime = Time(hdr[time_name], format=time_format)
    return sc, obstime


def heliocentric_correction(sc, obstime, site_name):
    helio_corr = sc.radial_velocity_correction(
        "heliocentric", obstime=obstime, location=EarthLocation.of_site(site_name)
    )
    return helio_corr.to_value(astropy.constants.c)


def read_generic_spectra(
    dir_path,
    extension,
    telname,
    usecols,
    n_cols_total,
    cuts=None,
    postprocess=None,
    fits_hdr_kwargs=None,
    helio_corr_site=None,
    before_time=None,
    table_read_kwargs=None,
    exclude_spec_masks=None,
    ignore_fits=False,
    helio_corrected=False,
    **kwargs,
):

    names = [usecols.get(i + 1, "_{}".format(i)) for i in range(n_cols_total)]
    exclude_names = [n for n in names if n.startswith("_")]

    helio_corrected = bool(helio_corrected)
    need_helio_corr = helio_corr_site and not helio_corrected
    need_to_read_fits = fits_hdr_kwargs or need_helio_corr or before_time is not None

    output = []

    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        rootname, extension_this = os.path.splitext(filename)

        if extension_this != extension:
            continue

        if "conflicted copy" in filename:
            logging.warning(
                "SKIPPING spectra file {} - it's a conflicted copy; check what went wrong!".format(filepath)
            )
            continue

        if exclude_spec_masks and (rootname in exclude_spec_masks or filename in exclude_spec_masks):
            continue

        helio_correction = None

        if need_to_read_fits and ignore_fits:
            logging.warning("Fits file skipped for {} because ignore_fits is set to True".format(filepath))

        if need_to_read_fits and not ignore_fits:
            for fits_name in (
                rootname + ".fits.gz",
                rootname + ".fits",
                rootname.rpartition("_")[0] + ".fits.gz",
                rootname.rpartition("_")[0] + ".fits",
            ):
                fits_path = os.path.join(dir_path, fits_name)
                if os.path.isfile(fits_path):
                    break

            try:
                sc, obstime = get_obs_info_from_fits(fits_path, **(fits_hdr_kwargs or {}))
            except OSError:
                logging.warning("Cannot find or read corresponding fits file for {}".format(filepath))
            else:
                if before_time is not None and obstime > before_time:
                    continue
                if need_helio_corr:
                    helio_correction = heliocentric_correction(sc, obstime, helio_corr_site)

        try:
            table_read_kwargs_this = dict(
                format="ascii.fast_no_header",
                guess=False,
                names=names,
                exclude_names=exclude_names,
            )
            if table_read_kwargs:
                table_read_kwargs_this.update(table_read_kwargs)
            this = Table.read(filepath, **table_read_kwargs_this)
        except (IOError, CParserError) as e:
            logging.warning("SKIPPING spectra file {} - could not read or parse\n{}".format(filepath, e))
            continue

        this = ensure_specs_dtype(this, skip_missing_cols=True)
        this = Query(cuts).filter(this)
        if not len(this):
            continue

        if "MASKNAME" not in this.colnames:
            this["MASKNAME"] = filename

        if helio_correction is not None:
            this["SPEC_Z"] = np.expm1(np.log1p(this["SPEC_Z"], dtype=np.float64) - np.log1p(-float(helio_correction)))
            helio_corrected = True

        this["HELIO_CORR"] = helio_corrected

        output.append(this)

    if not output:
        return

    output = vstack(output, "exact")
    output["TELNAME"] = telname
    if postprocess:
        output = postprocess(output)

    return ensure_specs_dtype(output)


def read_mmt(dir_path, before_time=None, exclude_spec_masks=None):
    extension = ".zlog"
    telname = "MMT"
    helio_corr_site = "mmt"

    n_cols_total = 11
    usecols = {
        2: "RA",
        3: "DEC",
        4: "mag",
        5: "SPEC_Z",
        6: "SPEC_Z_ERR",
        7: "ZQUALITY",
        8: "SPECOBJID",
    }

    cuts = Query("mag != 0", "ZQUALITY >= 0", (lambda x: x != "0", "SPECOBJID"))

    def postprocess(t):
        del t["mag"]
        t["RA"] *= 15.0
        return t

    return read_generic_spectra(**locals())


def read_mmt_bino(dir_path, exclude_spec_masks=None):
    extension = ".dat"
    telname = "BINO"

    n_cols_total = 8
    usecols = {
        1: "masknum",
        2: "SPECOBJID",
        3: "RA",
        4: "DEC",
        6: "SPEC_Z",
        7: "ZQUALITY",
    }

    table_read_kwargs = dict(
        format="ascii.fixed_width_no_header",
        col_starts=(0, 5, 11, 25, 38, 58, 66, 71),
    )

    def postprocess(t):
        t["SPEC_Z_ERR"] = 10 / SPEED_OF_LIGHT
        t["MASKNAME"] = np.char.add(np.char.add(t["MASKNAME"], "-"), t["masknum"].astype("<U"))
        del t["masknum"]
        return t

    return read_generic_spectra(**locals())


def read_aat(dir_path, before_time=None, exclude_spec_masks=None):
    extension = ".zlog"
    telname = "AAT"
    helio_corr_site = "sso"

    n_cols_total = 11
    usecols = {2: "RA", 3: "DEC", 5: "SPEC_Z", 7: "ZQUALITY", 8: "SPECOBJID"}
    cuts = Query("ZQUALITY >= 0", (lambda x: x != "0", "SPECOBJID"))

    fits_hdr_kwargs = dict(ra_name="MEANRA", dec_name="MEANDEC", time_name="UTMJD")

    def postprocess(t):
        t["SPEC_Z_ERR"] = 10 / SPEED_OF_LIGHT
        return t

    return read_generic_spectra(**locals())


def read_aat_mz(dir_path, before_time=None, exclude_spec_masks=None):
    extension = ".mz"
    telname = "AAT"
    helio_corr_site = "sso"

    n_cols_total = 15
    usecols = {3: "RA", 4: "DEC", 13: "SPEC_Z", 14: "ZQUALITY", 1: "SPECOBJID"}
    cuts = Query("ZQUALITY >= 0")

    fits_hdr_kwargs = dict(ra_name="MEANRA", dec_name="MEANDEC", time_name="UTMJD")
    table_read_kwargs = dict(delimiter=",")

    def postprocess(t):
        t["RA"] *= 180.0 / np.pi
        t["DEC"] *= 180.0 / np.pi
        t["SPEC_Z_ERR"] = 10 / SPEED_OF_LIGHT
        return t

    return read_generic_spectra(**locals())


def read_imacs(dir_path):
    extension = ".zlog"
    telname = "IMACS"

    n_cols_total = 12
    usecols = {
        2: "RA",
        3: "DEC",
        5: "SPEC_Z",
        6: "SPEC_Z_ERR",
        7: "ZQUALITY",
        8: "SPECOBJID",
        11: "MASKNAME",
    }

    cuts = Query("ZQUALITY >= 1", (lambda x: x != "0", "SPECOBJID"))

    return read_generic_spectra(**locals())


def read_wiyn(dir_path):

    output = []

    for f in os.listdir(dir_path):
        if not f.endswith(".fits.gz"):
            continue
        this = FitsTable(os.path.join(dir_path, f)).read()[["RA", "DEC", "ZQUALITY", "FID", "Z", "Z_ERR"]]
        this = Query("ZQUALITY >= 1").filter(this)
        this["MASKNAME"] = f
        output.append(this)

    output = vstack(output, "exact")

    output.rename_column("FID", "SPECOBJID")
    output.rename_column("Z", "SPEC_Z")
    output.rename_column("Z_ERR", "SPEC_Z_ERR")
    output["TELNAME"] = "WIYN"

    sc = SkyCoord(output["RA"], output["DEC"], unit=("hourangle", "deg"))
    output["RA"] = sc.ra.deg
    output["DEC"] = sc.dec.deg
    del sc

    return ensure_specs_dtype(output)


def read_palomar(file_path):
    if not hasattr(file_path, "read"):
        file_path = CsvTable(file_path)

    cols = [
        "SPECOBJID",
        "RA",
        "DEC",
        "SPEC_Z",
        "SPEC_Z_ERR",
        "ZQUALITY",
        "MASKNAME",
        "TELNAME",
        "HELIO_CORR",
    ]
    specs = file_path.read()[cols]

    return ensure_specs_dtype(specs)


def read_coadd(dir_path, exclude_spec_masks=None):
    extension = ".mz"
    telname = "COADD"
    ignore_fits = True
    helio_corrected = True

    n_cols_total = 15
    usecols = {3: "RA", 4: "DEC", 13: "SPEC_Z", 14: "ZQUALITY", 1: "SPECOBJID"}
    cuts = Query("ZQUALITY >= 0")

    table_read_kwargs = dict(delimiter=",")

    def postprocess(t):
        t["RA"] *= 180.0 / np.pi
        t["DEC"] *= 180.0 / np.pi
        t["SPEC_Z_ERR"] = 10 / SPEED_OF_LIGHT
        return t

    return read_generic_spectra(**locals())
