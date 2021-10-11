import os

import numpy as np
import requests
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import hstack
from easyquery import Query

__all__ = [
    "get_sdss_bands",
    "get_sdss_colors",
    "get_des_bands",
    "get_des_colors",
    "get_decals_bands",
    "get_decals_colors",
    "get_all_bands",
    "get_all_colors",
    "add_skycoord",
    "get_empty_str_array",
    "get_decals_viewer_image",
    "fill_values_by_query",
    "get_remove_flag",
    "view_table_as_2d_array",
    "find_objid",
    "find_near_objid",
    "find_near_coord",
    "find_near_ra_dec",
    "makedirs_if_needed",
    "group_by",
    "decode_flag",
    "join_str_arr",
    "calc_normalized_dist",
    "get_coord",
    "nearest_neighbor_join",
]


_sdss_bands = "ugriz"
_des_bands = "grizy"
_decals_bands = "grz"
_all_bands = "ugrizy"


def get_sdss_bands():
    return list(_sdss_bands)


def get_sdss_colors():
    return list(map("".join, zip(_sdss_bands[:-1], _sdss_bands[1:])))


def get_des_bands():
    return list(_des_bands)


def get_des_colors():
    return list(map("".join, zip(_des_bands[:-1], _des_bands[1:])))


def get_decals_bands():
    return list(_decals_bands)


def get_decals_colors():
    return list(map("".join, zip(_decals_bands[:-1], _decals_bands[1:])))


def get_all_bands():
    return list(_all_bands)


def get_all_colors():
    return list(map("".join, zip(_all_bands[:-1], _all_bands[1:]))) + ["rz"]


def get_empty_str_array(array_length, string_length=48, initialize_with=""):
    a = np.empty(array_length, np.dtype("<U{}".format(string_length)))
    a[:] = initialize_with
    return a


def add_skycoord(table, ra_label="RA", dec_label="DEC", coord_label="coord", unit="deg"):
    if coord_label not in table.colnames:
        table[coord_label] = SkyCoord(table[ra_label], table[dec_label], unit=unit)
    return table


def get_decals_viewer_image(ra, dec, pixscale=0.262, layer="ls-dr9", size=256, out=None, use_dev=False, timeout=120):  # pylint: disable=W0613
    dev = "-dev" if use_dev else ""
    url = f"https://www.legacysurvey.org/viewer{dev}/jpeg-cutout/?ra={ra}&dec={dec}&pixscale={pixscale}&layer={layer}&size={size}"
    content = requests.get(url, timeout=timeout).content
    if out is not None:
        if not out.lower().endswith(".jpg"):
            out += ".jpg"
        with open(out, "wb") as f:
            f.write(content)
    return content


def fill_values_by_query(table, query, values_to_fill):
    """
    Examples
    --------
    fill_values_by_query(table, 'OBJID == 1237668367995568266',
                         {'SPEC_Z': 0.21068, 'TELNAME':'SDSS', 'MASKNAME':'SDSS'})
    """
    mask = Query(query).mask(table)
    n_matched = np.count_nonzero(mask)

    if n_matched:
        for c, v in values_to_fill.items():
            table[c][mask] = v

    return n_matched


def get_remove_flag(catalog, remove_queries):
    """
    get remove flag by remove queries. remove_queries can be a list or dict.
    """

    try:
        iter_queries = iter(remove_queries.items())
    except AttributeError:
        iter_queries = enumerate(remove_queries)

    remove_flag = np.zeros(len(catalog), dtype=np.int)
    for i, remove_query in iter_queries:
        remove_flag[Query(remove_query).mask(catalog)] += 1 << i
    return remove_flag


def find_objid(table, objid):
    """
    Parameters
    ----------
    table : astropy.table.Table
        needs to have an integer column called `OBJID`
    objid : int

    Returns
    -------
    table : astropy.table.Table
    """
    t = Query("OBJID=={}".format(objid)).filter(table)
    if len(t) == 0:
        raise KeyError("Cannot find OBJID {}".format(objid))
    return t[0]


def find_near_coord(table, coord, within_arcsec=3.0):
    """
    Parameters
    ----------
    table : astropy.table.Table
        needs to have a SkyCoord column called `coord`
    coord : astropy.coordinates.SkyCoord
    within_arcsec : float

    Returns
    -------
    table : astropy.table.Table
    """
    table = add_skycoord(table)
    sep = table["coord"].separation(coord).arcsec

    nearby_mask = sep < within_arcsec
    sep = sep[nearby_mask]
    table = table[nearby_mask]

    sorter = sep.argsort()
    table = table[sorter]
    table["sep"] = sep[sorter]
    return table


def find_near_ra_dec(table, ra, dec, within_arcsec=3.0):
    """
    Parameters
    ----------
    table : astropy.table.Table
        needs to have a SkyCoord column called `coord`
    ra : float
        in degree
    dec : float
        in degree
    within_arcsec : float

    Returns
    -------
    table : astropy.table.Table
    """
    return find_near_coord(table, SkyCoord(ra, dec, unit="deg"), within_arcsec)


def find_near_objid(table, objid, within_arcsec=3.0):
    """
    Parameters
    ----------
    table : astropy.table.Table
        needs to have a SkyCoord column called `coord` and an integer column called `OBJID`
    objid : int
    within_arcsec : float

    Returns
    -------
    table : astropy.table.Table
    """
    table = add_skycoord(table)
    return find_near_coord(table, find_objid(table, objid)["coord"], within_arcsec)


def view_table_as_2d_array(table, cols=None, row_mask=None, dtype=np.float64):
    """
    Convert an astropy Table to 2d ndarray with a fixed dtype.
    """
    row_mask = slice(None) if row_mask is None else row_mask
    cols = table.colnames if cols is None else cols

    def _get_data(col_this):
        if hasattr(col_this, "filled"):
            return col_this.filled().data
        return col_this.data

    return np.vstack([_get_data(table[c][row_mask]).astype(dtype, casting="same_kind", copy=False) for c in cols]).T


def makedirs_if_needed(path):
    """
    Makes the directories in the path specified, if they don't exist. If they
    already exist, this returns without doing anything.
    """
    dirs = os.path.dirname(path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def group_by(x, is_sorted=False):
    if not len(x):
        return
    if is_sorted:
        edges = np.flatnonzero(np.hstack([[1], np.ediff1d(x), [1]]))
        for i, j in zip(edges[:-1], edges[1:]):
            yield slice(i, j)
    else:
        sorter = np.argsort(x)
        edges = np.flatnonzero(np.hstack([[1], np.ediff1d(x[sorter]), [1]]))
        for i, j in zip(edges[:-1], edges[1:]):
            yield sorter[i:j]


def decode_flag(flag, offset=0):
    flag = int(flag)
    return [i + offset for i in range(int(np.floor(np.log2(flag))) + 1) if (flag & (1 << i))]


def join_str_arr(*arrays):
    arrays_iter = iter(arrays)
    a = next(arrays_iter)
    for b in arrays_iter:
        a = np.char.add(a, b)
    return a


def calc_normalized_dist(obj_ra, obj_dec, cen_ra, cen_dec, cen_r, cen_ba=None, cen_phi=None, multiplier=2.0):
    """
    obj_ra, obj_dec, cen_ra, cen_dec in degrees
    cen_r is the semi-major axis in arcseconds, set multiplier as needed
    """
    a = cen_r / 3600.0 * multiplier
    cos_dec = np.cos(np.deg2rad((obj_dec + cen_dec) * 0.5))
    dx = np.rad2deg(np.arcsin(np.sin(np.deg2rad(obj_ra - cen_ra)))) * cos_dec
    dy = obj_dec - cen_dec

    if cen_ba is None:
        with np.errstate(divide="ignore"):
            return np.hypot(dx, dy) / a

    b = a * cen_ba
    theta = np.deg2rad(90 - cen_phi)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.hypot((dx * cos_t + dy * sin_t) / a, (-dx * sin_t + dy * cos_t) / b)


def get_coord(table, coord=None):
    if isinstance(coord, SkyCoord):
        return coord
    _cols = table.colnames
    if coord is None:
        if "coord" in _cols and isinstance(table["coord"], SkyCoord):
            return table["coord"]
        if "RA" in _cols and "DEC" in _cols:
            return SkyCoord(table["RA"], table["DEC"], unit="deg")
        if "ra" in _cols and "dec" in _cols:
            return SkyCoord(table["ra"], table["dec"], unit="deg")
    else:
        if isinstance(coord, str) and coord in _cols and isinstance(table[coord], SkyCoord):
            return table[coord]
        try:
            if coord[0] in _cols and coord[1] in _cols:
                return SkyCoord(table[coord[0]], table[coord[1]], unit="deg")
        except (TypeError, IndexError, KeyError):
            pass
    raise ValueError("Cannot identify coord column")


def nearest_neighbor_join(
    left,
    right,
    left_coord=None,
    right_coord=None,
    join_type="left",
    nthneighbor=1,
    sep_label="sep",
    uniq_col_name="{col_name}{table_name}",
    table_names=("_1", "_2"),
):

    left_coord = get_coord(left, left_coord)
    right_coord = get_coord(right, right_coord)

    if join_type == "left":
        idx, sep, _ = match_coordinates_sky(left_coord, right_coord, nthneighbor=nthneighbor)
        right = right[idx]
    elif join_type == "right":
        idx, sep, _ = match_coordinates_sky(right_coord, left_coord, nthneighbor=nthneighbor)
        left = left[idx]
    else:
        raise ValueError("join_type must be 'left' or 'right'")

    d = hstack([left, right], join_type="exact", uniq_col_name=uniq_col_name, table_names=table_names)
    d[sep_label] = sep.arcsec

    return d
