import io
import os
import time

import numpy as np
import requests
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.table import hstack
from easyquery import Query

try:
    from PIL import Image
except ImportError:
    Image = None

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
    "get_decals_cutout_url",
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
    "match_ids",
    "calc_data_quantiles",
    "calc_cdf",
    "percentile_with_weights",
    "binned_percentile",
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


def get_decals_cutout_url(ra, dec, pixscale=0.262, layer="ls-dr9", size=256, use_dev=False, file_type="jpg"):
    file_type = str(file_type).lower()
    if file_type not in ("jpg", "fits"):
        raise ValueError("file_type must be either 'jpg' or 'fits'")
    dev = "-dev" if use_dev else ""
    return f"https://www.legacysurvey.org/viewer{dev}/cutout.{file_type}/?ra={ra}&dec={dec}&pixscale={pixscale}&layer={layer}&size={int(size)}"


def get_decals_viewer_image(ra, dec, pixscale=0.262, layer="ls-dr9", size=256,
                            out=None, use_dev=False, timeout=60, file_type="jpg",
                            convert_to_data=False, cache_dir=None, retry=10):
    url = get_decals_cutout_url(ra, dec, pixscale, layer, size, use_dev, file_type)
    extention = f".{file_type}"

    content = cache_path = None
    if cache_dir:
        cache_path = os.path.join(cache_dir, url.partition("?")[2].replace("&", "_").replace("=", "") + extention)
        try:
            with open(cache_path, "rb") as f:
                content = f.read()
        except OSError:
            pass
        else:
            cache_path = None

    if content is None:
        for i in range(int(retry) + 1):
            try:
                content = requests.get(url, timeout=timeout).content
            except requests.ReadTimeout:
                time.sleep((i + 1) * 5)
            else:
                if cache_path:
                    makedirs_if_needed(cache_path)
                    with open(cache_path, "wb") as f:
                        f.write(content)
                break
        else:
            raise RuntimeError("Cannot obtain image from {}".format(url))

    if out is not None:
        if isinstance(out, str):
            if not out.lower().endswith(extention):
                out += extention
            with open(out, "wb") as f:
                f.write(content)
        else:
            out.write(content)

    if convert_to_data:
        if file_type == "fits":
            return fits.open(io.BytesIO(content))

        if file_type == "jpg":
            if Image is None:
                raise ImportError("PIL is required to convert jpg to data")
            return Image.open(io.BytesIO(content), formats=["JPEG"])

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


def get_remove_flag(catalog, remove_queries, dtype=None):
    """
    get remove flag by remove queries. remove_queries can be a list or dict.
    """

    if dtype is None:
        n = len(remove_queries)
        if n < 32:
            dtype = np.int32
        elif n < 64:
            dtype = np.int64
        else:
            raise ValueError("too many remove queries")

    try:
        iter_queries = iter(remove_queries.items())
    except AttributeError:
        iter_queries = enumerate(remove_queries)

    remove_flag = np.zeros(len(catalog), dtype=dtype)
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


def match_ids(id1, id2, assume_unique=False):
    """
    Find `id1` in `id2`. `id2` should be sorted.
    Ruturns `id1_indices` and `id2_indices` such that
    `id1[id1_indices] == id2[id2_indices]`.
    """
    _, id1_indices, id2_indices = np.intersect1d(id1, id2, assume_unique=assume_unique, return_indices=True)
    return id1_indices, id2_indices


def calc_data_quantiles(a, weights=None, method="median_unbiased"):
    a = np.ravel(a)
    weights = np.ones_like(a) if weights is None else np.ravel(weights)

    mask = np.isfinite(a)
    mask &= np.isfinite(weights)
    mask &= (weights > 0)

    a = a[mask]
    sorter = np.argsort(a)
    a = a[sorter]
    weights = weights[mask][sorter]
    weights = np.cumsum(weights, dtype=np.float64)

    alpha = beta = dict(linear=1.0, hazen=0.5, weibull=0, median_unbiased=(1 / 3), normal_unbiased=(3 / 8), interpolated_inverted_cdf=0)[method]
    if method == "interpolated_inverted_cdf":
        beta = 1.0
    q = (weights - alpha) / (weights[-1] - alpha - beta + 1)
    return a, q


def calc_cdf(a, bins, weights=None, method="median_unbiased"):
    a, q = calc_data_quantiles(a, weights=weights, method=method)
    return np.interp(bins, a, q, left=0, right=1)


def percentile_with_weights(a, percentiles, weights=None, method="median_unbiased"):
    """
    Calculates the percentile of x with weights.
    """
    if weights is None:
        return np.percentile(a, percentiles, method=method)
    a, q = calc_data_quantiles(a, weights=weights, method=method)
    return np.interp(np.atleast_1d(percentiles).ravel() / 100, q, a)


def binned_percentile(x, values, percentiles, bins=10, method="median_unbiased", weights=None):
    """
    Parameters
    ----------
    x : array_like
        The input data to be binned.
    values : array_like
        The input values to calculate percentiles.
    percentiles : array_like
        The percentiles to calculate.
    bins : int or array_like, optional
        If bins is an int, it defines the number of equal-width bins in the
        given range. If bins is a sequence, it defines the bin edges.
    method : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to use,

    Returns
    -------
    bin_edges : ndarray
        The edges of the bins used in the calculation.
    counts : ndarray
        Number of values in each bin.
    percentiles : ndarray
        The requested percentiles, with a shape of (len(percentiles), len(bins)-1).
    """
    x = np.asanyarray(x)
    values = np.asanyarray(values)

    mask = np.isfinite(x) & np.isfinite(values)
    x = x[mask]
    sorter = np.argsort(x)
    x = x[sorter]
    values = values[mask][sorter]

    if isinstance(bins, int):
        bins = np.linspace(x.min(), x.max(), bins)
    else:
        bins = np.atleast_1d(bins).ravel()

    indices = np.searchsorted(x, bins)
    percentiles = np.atleast_1d(percentiles).ravel()

    if weights is None:
        def _calc_percentiles(values, weights):
            return np.quantile(values, percentiles / 100, method=method)
    else:
        weights = weights[mask][sorter]

        def _calc_percentiles(values, weights):
            return percentile_with_weights(values, percentiles, weights, method)

    count = []
    out = []
    for i, j in zip(indices[:-1], indices[1:]):
        count.append(j - i)
        if j == i:
            nan = np.empty(percentiles.size)
            nan.fill(np.nan)
            out.append(nan)
        else:
            out.append(_calc_percentiles(values[i:j], weights=(weights[i:j] if weights is not None else None)))
    return bins, np.array(count), np.stack(out).T
