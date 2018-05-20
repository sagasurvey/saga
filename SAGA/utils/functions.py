import requests
import numpy as np
from easyquery import Query
from astropy.coordinates import search_around_sky, SkyCoord
from astropy import units as u


__all__ = ['get_sdss_bands', 'get_sdss_colors', 'add_skycoord',
           'get_empty_str_array', 'get_decals_viewer_image', 'fill_values_by_query',
           'get_remove_flag', 'view_table_as_2d_array',
           'find_objid', 'find_near_objid', 'find_near_coord', 'find_near_ra_dec']


_sdss_bands = 'ugriz'


def get_sdss_bands():
    return list(_sdss_bands)


def get_sdss_colors():
    return list(map(''.join, zip(_sdss_bands[:-1], _sdss_bands[1:])))


def get_empty_str_array(array_length, string_length=48, initialize_with=''):
    a = np.empty(array_length, np.dtype('<U{}'.format(string_length)))
    a[:] = initialize_with
    return a


def add_skycoord(table, ra_label='RA', dec_label='DEC', coord_label='coord', unit='deg'):
    if coord_label not in table.colnames:
        table[coord_label] = SkyCoord(table[ra_label], table[dec_label], unit=unit)
    return table


def get_decals_viewer_image(ra, dec, pixscale=0.2, layer='sdssco', size=256, out=None):
    url = 'http://legacysurvey.org/viewer-dev/jpeg-cutout/?ra={ra}&dec={dec}&pixscale={pixscale}&layer={layer}&size={size}'.format(**locals())
    content = requests.get(url).content
    if out is not None:
        if not out.lower().endswith('.jpg'):
            out += '.jpg'
        with open(out, 'wb') as f:
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
        remove_flag[Query(remove_query).mask(catalog)] += (1 << i)
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
    t = Query('OBJID=={}'.format(objid)).filter(table)
    if len(t) == 0:
        raise KeyError('Cannot find OBJID {}'.format(objid))
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
    sep = table['coord'].separation(coord).arcsec
    nearby_indices = np.flatnonzero(sep < within_arcsec)
    nearby_indices = nearby_indices[sep[nearby_indices].argsort()]
    return table[nearby_indices]

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
    return find_near_coord(table, SkyCoord(ra, dec, unit='deg'), within_arcsec)

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
    return find_near_coord(table, find_objid(table, objid)['coord'], within_arcsec)


def view_table_as_2d_array(table, cols=None, row_mask=None, dtype=np.float64):
    """
    Convert an astropy Table to 2d ndarray with a fixed dtype.
    """
    row_mask = slice() if row_mask is None else row_mask
    cols = table.colnames if cols is None else cols
    return np.vstack((table[c][row_mask].data.astype(dtype, casting='same_kind', copy=False) for c in cols)).T
