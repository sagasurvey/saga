import numpy as np
from astropy.coordinates import SkyCoord
from astropy.constants import c
from ..utils import get_empty_str_array

def clean_repeats(spectra):
    """
    Clean all spectra to remove repeats.
    `spectra` is modified in-place.

    Parameters
    ----------
    spectra : astropy.table.Table

    Returns
    -------
    spectra : astropy.table.Table
    """

    if 'coord' not in spectra.columns:
        spectra['coord'] = SkyCoord(spectra['RA'], spectra['DEC'], unit="deg")

    if 'SPEC_REPEAT' not in spectra.columns:
        spectra['SPEC_REPEAT'] = get_empty_str_array(len(spectra))
    else:
        spectra['SPEC_REPEAT'] = ''

    spectra['raw'] = True

    for spec in spectra:
        if not spec['raw']:
            continue

        # search nearby spectra in 3D
        nearby_mask = (np.abs(spectra['SPEC_Z'] - spec['SPEC_Z']) < 50.0/c.to('km/s').value)
        nearby_mask &= (spectra['coord'].separation(spec['coord']).arcsec < 30.0)
        nearby_mask &= spectra['raw']
        nearby_mask = np.where(nearby_mask)[0]
        assert len(nearby_mask) >= 1

        spectra['raw'][nearby_mask] = False

        best_spec_idx = nearby_mask[spectra['ZQUALITY'][nearby_mask].argmax()]
        spectra['SPEC_REPEAT'][best_spec_idx] = '+'.join(spectra['TELNAME'][nearby_mask])

    del spectra['raw']
    spectra = spectra[spectra['SPEC_REPEAT'] != '']

    return spectra
