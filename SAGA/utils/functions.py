import requests
import numpy as np
from easyquery import Query
from astropy.coordinates import search_around_sky, SkyCoord
from astropy import units as u


__all__ = ['get_sdss_bands', 'get_sdss_colors', 'add_skycoord',
           'get_empty_str_array', 'get_decals_viewer_image',
           'join_table_by_coordinates', 'fill_values_by_query']


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


def join_table_by_coordinates(table, table_to_join,
                              columns_to_join=None, columns_to_rename=None,
                              max_distance=1.0*u.arcsec, missing_value=np.nan,
                              table_ra_name='RA', table_dec_name='DEC',
                              table_to_join_ra_name='RA',
                              table_to_join_dec_name='DEC', unit='deg',
                              table_coord_name='coord',
                              table_to_join_coord_name='coord'):
    """
    Join two table by matching the sky coordinates.

    The `unit` input controls how the tables' coordinates are interpreted, and
    also the `max_distance`, but *only* if they are not quantity objects.

    Also, note that if the tables have a `'coord'` column, that will be used as
    the SkyCoord *instead* of accessing ``table_ra_name``/``table_dec_name``.

    Examples
    --------
    wise_cols = ('W1_MAG', 'W1_MAG_ERR', 'W2_MAG', 'W2_MAG_ERR')
    cols_rename = {'W1_MAG':'W1', 'W1_MAG_ERR':'W1ERR', 'W2_MAG':'W2', 'W2_MAG_ERR':'W2ERR'}
    join_table_by_coordinates(base, wise, wise_cols, cols_rename)
    """

    t1 = table
    t2 = table_to_join

    ra1 = table_ra_name
    dec1 = table_dec_name
    ra2 = table_to_join_ra_name
    dec2 = table_to_join_dec_name

    if not hasattr(max_distance, 'unit'):
        max_distance = u.Quantity(max_distance, unit=unit)

    # note that if a unit-ful ra/dec are present, the *unit* argument here is
    # ignored
    if table_coord_name in t1.colnames:
        sc1 = t1[table_coord_name]
    else:
        sc1 = SkyCoord(t1[ra1], t1[dec1], unit=unit)
    if table_to_join_coord_name in t2.colnames:
        sc2 = t2[table_to_join_coord_name]
    else:
        sc2 = SkyCoord(t2[ra2], t2[dec2], unit=unit)
    idx1, idx2 = search_around_sky(sc1, sc2, max_distance)[:2]

    n_matched = len(idx1)

    if n_matched:
        if columns_to_join is None:
            columns_to_join = t2.colnames

        if columns_to_rename is None:
            columns_to_rename = dict()

        if isinstance(missing_value, dict):
            missing_value_dict = missing_value
            missing_value = np.nan
        else:
            missing_value_dict = dict()

        for c2 in columns_to_join:
            c1 = columns_to_rename.get(c2, c2)
            if c1 not in t1:
                t1[c1] = missing_value_dict.get(c1, missing_value)
            t1[c1][idx1] = t2[c2][idx2]

    return n_matched


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
