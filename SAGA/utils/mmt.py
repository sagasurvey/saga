"""
convenience functions for MMT catalogs
"""
import numpy as np
from astropy.coordinates import Angle
from astropy.table import Table


def read_mmt_catalog(filepath, header_line_index=1):
    t = Table.read(
        filepath,
        guess=False,
        format="ascii.fast_tab",
        header_start=header_line_index - 1,
        data_start=header_line_index + 1,
    )
    t["ra"] = Angle(t["ra"], unit="hr").deg
    t["dec"] = Angle(t["dec"], unit="deg").deg
    return t


def read_mmt_config(filepath):
    i = 0
    with open(filepath) as f:
        for line in f:
            if line.startswith("-----"):
                break
            if len(line) > 1:
                i += 1
    t = read_mmt_catalog(filepath, i)
    t["rank"] = t["rank"].astype(np.int32)
    return t


def read_mmt_catalog_plug(filepath):
    t = Table.read(
        filepath,
        guess=False,
        format="ascii.fast_no_header",
        names=["index", "ra", "dec", "mag", "fiber", "status", "object"],
    )
    t["ra"] *= 15.0
    return t
